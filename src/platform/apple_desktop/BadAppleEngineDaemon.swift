// BadAppleEngineDaemon — Swift-native SLICKS v1/v2 MLX inference server.
// Replaces badapple_mlx_server.py as the backend behind the Rust gatekeeper.
//
// This is a top-level executable: `import Foundation` and a `main()` function
// below. It compiles with the same `swiftc` invocation used for the menu bar.

import Foundation
import Dispatch
import CryptoKit
import Security
import Darwin
import BadAppleMLX

@main
struct BadAppleEngineDaemon {
    static func main() {
        runDaemonMain()
    }
}

// MARK: - Protocol constants

private let DEFAULT_SOCKET_PATH = "/var/run/badapple/substrate_mlx.sock"
private let SLICKS_VERSION: Int = 1
private let SLICKS_VERSION_2: Int = 2
private let HANDSHAKE_MAX_SKEW_MS: Int64 = 30_000
private let MAX_PROMPT_BYTES = 64 * 1024
private let MAX_NEW_TOKENS = 4096
private let MAX_FRAME_BYTES = 1024 * 1024
private let DEFAULT_FAST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

// MARK: - Logging

private func log(_ message: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    print("[\(ts)] \(message)")
    fflush(stdout)
}

// MARK: - SLICKS helpers

private func timestampNowMS() -> UInt64 {
    UInt64(Date().timeIntervalSince1970 * 1000)
}

private func isTimestampFresh(_ timestampMs: UInt64) -> Bool {
    let now = Int64(timestampNowMS())
    let ts = Int64(timestampMs)
    return abs(now - ts) <= HANDSHAKE_MAX_SKEW_MS
}

private func randomNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 32)
    let status = SecRandomCopyBytes(kSecRandomDefault, 32, &bytes)
    if status != errSecSuccess {
        // Fallback only if SecRandomCopyBytes fails unexpectedly.
        for i in 0..<32 { bytes[i] = UInt8.random(in: 0...255) }
    }
    return bytes.map { String(format: "%02x", $0) }.joined()
}

private func isNonceValid(_ nonce: String) -> Bool {
    nonce.count == 64 && nonce.allSatisfy { $0.isHexDigit }
}

private func hexToData(_ hex: String) -> Data? {
    guard hex.count % 2 == 0 else { return nil }
    var data = Data(capacity: hex.count / 2)
    var index = hex.startIndex
    while index < hex.endIndex {
        let next = hex.index(index, offsetBy: 2)
        guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
        data.append(byte)
        index = next
    }
    return data
}

private func dataToHex(_ data: Data) -> String {
    data.map { String(format: "%02x", $0) }.joined()
}

private func sha256Hex(_ string: String) -> String {
    let digest = SHA256.hash(data: Data(string.utf8))
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func hmacSHA256Hex(_ secret: Data, _ message: String) -> String {
    let key = SymmetricKey(data: secret)
    let mac = HMAC<SHA256>.authenticationCode(for: Data(message.utf8), using: key)
    return Data(mac).map { String(format: "%02x", $0) }.joined()
}

private func constantTimeCompare(_ a: Data, _ b: Data) -> Bool {
    guard a.count == b.count else { return false }
    var result: UInt8 = 0
    for (x, y) in zip(a, b) { result |= x ^ y }
    return result == 0
}

private func loadSlicksSecret() -> Data? {
    let env = ProcessInfo.processInfo.environment
    if let raw = env["BADAPPLE_SLICKS_SECRET"], !raw.isEmpty {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.allSatisfy({ $0.isHexDigit }) && trimmed.count >= 32 {
            return hexToData(trimmed)
        }
        return trimmed.data(using: .utf8)
    }

    let keyPath = env["BADAPPLE_SLICKS_KEY_PATH"] ?? "/var/lib/bad_apple/slicks.key"
    guard FileManager.default.fileExists(atPath: keyPath) else {
        log("SLICKS secret not found at \(keyPath)")
        return nil
    }
    guard let raw = try? String(contentsOfFile: keyPath, encoding: .utf8) else {
        log("SLICKS secret file is unreadable at \(keyPath)")
        return nil
    }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.allSatisfy({ $0.isHexDigit }) && trimmed.count >= 32 {
        return hexToData(trimmed)
    }
    return trimmed.data(using: .utf8)
}

// MARK: - Secure Enclave identity agent client for SLICKS v2

/// Talks to the long-lived `badapple-identity-agent` (or the legacy Python
/// identity agent) over its Unix socket. The agent lives in the user's Aqua
/// / GUI session, so Secure Enclave signing and verification succeed reliably.
private final class IdentityAgentClient {
    static let shared = IdentityAgentClient()

    private let socketPath: String
    private var cachedPublicKey: String?
    private let cacheLock = NSLock()

    private init() {
        let env = ProcessInfo.processInfo.environment
        socketPath = env["BADAPPLE_IDENTITY_AGENT_SOCKET"] ?? "/var/run/badapple/identity.sock"
    }

    var isAvailable: Bool {
        var s = stat()
        return stat(socketPath, &s) == 0 && (s.st_mode & S_IFMT) == S_IFSOCK
    }

    func publicKey() -> String? {
        cacheLock.lock()
        if let cached = cachedPublicKey {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        let pub = call(command: "public_key").value
        if let pub = pub {
            cacheLock.lock()
            cachedPublicKey = pub
            cacheLock.unlock()
        }
        return pub
    }

    func sign(message: Data) -> String? {
        call(command: "sign", messageB64: message.base64EncodedString()).value
    }

    func verify(message: Data, signature: Data, publicKey: Data) -> Bool {
        let resp = call(
            command: "verify",
            extras: [
                "message_b64": message.base64EncodedString(),
                "signature": signature.base64EncodedString(),
                "public_key": publicKey.base64EncodedString()
            ]
        )
        return resp.json?["valid"] as? Bool == true
    }

    private struct AgentResponse {
        let ok: Bool
        let value: String?
        let json: [String: Any]?
    }

    private func call(command: String, messageB64: String? = nil, extras: [String: Any] = [:]) -> AgentResponse {
        var request: [String: Any] = ["command": command]
        if let messageB64 = messageB64 {
            request["message_b64"] = messageB64
        }
        for (k, v) in extras { request[k] = v }

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            log("identity agent socket() failed: \(errno)")
            return AgentResponse(ok: false, value: nil, json: nil)
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let path = socketPath.utf8CString
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path) - 1
        guard path.count <= maxLen else {
            log("identity agent socket path too long")
            return AgentResponse(ok: false, value: nil, json: nil)
        }
        path.withUnsafeBufferPointer { src in
            withUnsafeMutablePointer(to: &addr.sun_path) { dst in
                _ = memcpy(dst, src.baseAddress, path.count)
            }
        }

        let conn = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(fd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard conn == 0 else {
            log("identity agent connect() failed: \(errno)")
            return AgentResponse(ok: false, value: nil, json: nil)
        }

        guard let payload = try? JSONSerialization.data(withJSONObject: request, options: []) else {
            log("identity agent request JSON encode failed")
            return AgentResponse(ok: false, value: nil, json: nil)
        }
        let line = payload + Data([0x0A])
        _ = line.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> Int in
            Darwin.send(fd, buf.baseAddress, buf.count, 0)
        }

        var buffer = Data()
        var byte: UInt8 = 0
        while Darwin.read(fd, &byte, 1) == 1 {
            if byte == 0x0A { break }
            buffer.append(byte)
            if buffer.count > 1_000_000 { break }
        }

        guard let json = try? JSONSerialization.jsonObject(with: buffer) as? [String: Any] else {
            log("identity agent response JSON parse failed")
            return AgentResponse(ok: false, value: nil, json: nil)
        }

        let ok = json["ok"] as? Bool == true
        if !ok {
            if let error = json["error"] as? String, !error.isEmpty {
                log("identity agent error (\(command)): \(error)")
            }
        }
        return AgentResponse(ok: ok, value: json[command == "public_key" ? "public_key" : "signature"] as? String, json: json)
    }
}

private func serverMaterial(version: Int, timestampMs: UInt64, clientNonce: String, serverNonce: String) -> String {
    "BADAPPLE-SLICKS/\(version)|server|\(timestampMs)|\(clientNonce)|\(serverNonce)"
}

private func clientMaterial(version: Int, timestampMs: UInt64, clientNonce: String, serverNonce: String, prompt: String, maxTokens: Int) -> String {
    let promptHash = sha256Hex(prompt)
    return "BADAPPLE-SLICKS/\(version)|client|\(timestampMs)|\(clientNonce)|\(serverNonce)|\(maxTokens)|\(promptHash)"
}

private func serverProof(version: Int, secret: Data, timestampMs: UInt64, clientNonce: String, serverNonce: String) -> String {
    hmacSHA256Hex(secret, serverMaterial(version: version, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce))
}

private func clientProof(version: Int, secret: Data, timestampMs: UInt64, clientNonce: String, serverNonce: String, prompt: String, maxTokens: Int) -> String {
    hmacSHA256Hex(secret, clientMaterial(version: version, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce, prompt: prompt, maxTokens: maxTokens))
}

private func verifyClientProof(version: Int, secret: Data?, timestampMs: UInt64, clientNonce: String, serverNonce: String, prompt: String, maxTokens: Int, proof: String, clientPubkey: String?, helper: IdentityAgentClient?) -> Bool {
    if version == SLICKS_VERSION_2 {
        guard let clientPubkey = clientPubkey,
              let clientPubkeyData = Data(base64Encoded: clientPubkey),
              let signature = Data(base64Encoded: proof),
              let helper = helper, helper.isAvailable else {
            return false
        }
        let material = clientMaterial(version: version, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce, prompt: prompt, maxTokens: maxTokens)
        return helper.verify(message: Data(material.utf8), signature: signature, publicKey: clientPubkeyData)
    }
    guard let secret = secret, !secret.isEmpty else {
        // Placeholder mode: accept any well-formed hex HMAC (64 chars).
        return proof.count == 64 && proof.allSatisfy { $0.isHexDigit }
    }
    let expected = clientProof(version: version, secret: secret, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce, prompt: prompt, maxTokens: maxTokens)
    guard expected == proof.lowercased() else { return false }
    return true
}

// MARK: - Socket helpers

private final class BufferedReader {
    private let fd: Int32
    private var buffer = Data()
    private let maxLineBytes: Int

    init(fd: Int32, maxLineBytes: Int = MAX_FRAME_BYTES) {
        self.fd = fd
        self.maxLineBytes = maxLineBytes
    }

    func readLine() -> Data? {
        while true {
            if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                var line = buffer.subdata(in: 0..<newlineIndex)
                // Trim a trailing CR, if any, so Windows-style line endings also parse.
                if let last = line.last, last == 0x0D {
                    line.removeLast()
                }
                buffer.removeSubrange(0...newlineIndex)
                return line
            }
            if buffer.count > maxLineBytes {
                log("Frame exceeded \(maxLineBytes)-byte limit")
                return nil
            }
            var chunk = [UInt8](repeating: 0, count: 4096)
            let n = chunk.withUnsafeMutableBytes { raw -> Int in
                read(fd, raw.baseAddress, 4096)
            }
            if n > 0 {
                buffer.append(contentsOf: chunk[0..<n])
            } else if n == 0 {
                return nil
            } else {
                let err = errno
                if err == EINTR { continue }
                log("read error \(err)")
                return nil
            }
        }
    }
}

private func writeAll(_ fd: Int32, data: Data) -> Bool {
    var total = 0
    return data.withUnsafeBytes { raw -> Bool in
        guard let base = raw.baseAddress else { return false }
        let count = data.count
        while total < count {
            let n = write(fd, base.advanced(by: total), count - total)
            if n < 0 {
                let err = errno
                if err == EINTR { continue }
                log("write error \(err)")
                return false
            }
            if n == 0 { return false }
            total += n
        }
        return true
    }
}

private func writeJSON(_ fd: Int32, _ object: [String: Any]) -> Bool {
    guard let jsonData = try? JSONSerialization.data(withJSONObject: object, options: []),
          var line = String(data: jsonData, encoding: .utf8) else {
        return false
    }
    line.append("\n")
    guard let data = line.data(using: .utf8) else { return false }
    return writeAll(fd, data: data)
}

private func setSocketOptions(_ fd: Int32) {
    var one: Int32 = 1
    _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))

    // Give clients a generous window to complete the handshake. Streaming
    // generation uses the same connection, but it writes; it does not read.
    var tv = timeval(tv_sec: 30, tv_usec: 0)
    _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
}

private var gListenFd: Int32 = -1

private func bindUnixSocket(path: String) -> Int32 {
    let parent = (path as NSString).deletingLastPathComponent
    let fm = FileManager.default
    if !fm.fileExists(atPath: parent) {
        do {
            try fm.createDirectory(atPath: parent, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o770])
        } catch {
            log("Cannot create socket directory \(parent): \(error)")
            return -1
        }
    }
    parent.withCString { _ = Darwin.chmod($0, 0o770) }

    try? fm.removeItem(atPath: path)

    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        log("socket() failed: \(errno)")
        return -1
    }

    var one: Int32 = 1
    _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(path.utf8)
    let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard pathBytes.count < maxPath else {
        log("Socket path too long: \(path)")
        close(fd)
        return -1
    }
    pathBytes.withUnsafeBufferPointer { src in
        _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
            memcpy(dst, src.baseAddress!, pathBytes.count)
        }
    }
    addr.sun_len = UInt8(2 + pathBytes.count + 1)
    let addrLen = socklen_t(addr.sun_len)

    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            bind(fd, sockaddrPtr, addrLen)
        }
    }
    guard bindResult == 0 else {
        log("bind() failed: \(errno)")
        close(fd)
        return -1
    }

    path.withCString { _ = Darwin.chmod($0, 0o660) }

    let listenResult = listen(fd, 64)
    guard listenResult == 0 else {
        log("listen() failed: \(errno)")
        close(fd)
        return -1
    }

    return fd
}

// MARK: - Prompt preprocessing

private func stripPrefixes(_ prompt: inout String, voice: inout Bool, benchmark: inout Bool) {
    while true {
        if prompt.hasPrefix("__BADAPPLE_PERSONA__") {
            let after = String(prompt.dropFirst("__BADAPPLE_PERSONA__".count))
            if let range = after.range(of: "__") {
                let name = String(after[..<range.lowerBound])
                let rest = String(after[range.upperBound...]).trimmingCharacters(in: .whitespaces)
                _ = BadAppleEngine.shared.switchPersona(name)
                prompt = rest
                continue
            }
        }
        if prompt.hasPrefix("__BADAPPLE_VOICE__ ") {
            prompt = String(prompt.dropFirst("__BADAPPLE_VOICE__ ".count))
            voice = true
            continue
        }
        if prompt.hasPrefix("__BADAPPLE_BENCHMARK__ ") {
            prompt = String(prompt.dropFirst("__BADAPPLE_BENCHMARK__ ".count))
            benchmark = true
            continue
        }
        break
    }
}

// MARK: - Meta/control commands

private func handleMetaRequest(_ prompt: String) async -> String? {
    if let response = BadAppleEngine.shared.handlePersonaCommand(prompt) {
        return response
    }

    let lower = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    if lower.hasPrefix("generate an image of") {
        let imagePrompt = String(prompt.dropFirst("generate an image of".count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return await BadAppleEngine.shared.executeTool(name: "image_generation", args: ["prompt": imagePrompt])
    }
    switch lower {
    case "enable private mode", "private mode on":
        BadAppleEngine.shared.privateMode = true
        return "Private mode enabled. Memory, cache, conversation, and audit persistence are paused."
    case "disable private mode", "private mode off":
        BadAppleEngine.shared.privateMode = false
        return "Private mode disabled. Local persistence is active again."
    case "new chat", "clear conversation":
        BadAppleEngine.shared.resetConversation()
        return "Okay, so... fresh start."
    case "runtime status", "health status", "bad apple status":
        let status = await BadAppleEngine.shared.runtimeStatus()
        guard let data = try? JSONSerialization.data(withJSONObject: status, options: .prettyPrinted),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    case "fast tier on", "enable fast tier":
        BadAppleEngine.shared.fastTierEnabled = true
        return "Fast tier enabled. Simple queries will route to the 0.5B model when possible."
    case "fast tier off", "disable fast tier":
        BadAppleEngine.shared.fastTierEnabled = false
        return "Fast tier disabled. All queries route through the main model."
    case "flush vram", "purge vram", "clear metal cache":
        BadAppleEngine.shared.clearCache()
        return "VRAM cache cleared."
    case "unload all models":
        await BadAppleEngine.shared.unload()
        return "All models unloaded."
    case "autopilot on", "enable autopilot":
        BadAppleEngine.shared.autopilot = true
        return "Autopilot enabled. Destructive tools run without approval."
    case "autopilot off", "disable autopilot":
        BadAppleEngine.shared.autopilot = false
        return "Autopilot disabled. Destructive tools require approval."
    default:
        return nil
    }
}

// MARK: - Metrics

private func makeMetrics() -> [String: Any] {
    let speculativeActive = BadAppleInference.envSpeculativeDraftModel != nil
    var metrics: [String: Any] = [
        "tokens": BadAppleEngine.shared.lastTokenCount,
        "decode_tps": Double(BadAppleEngine.shared.lastTokensPerSecond),
        "total_tps": Double(BadAppleEngine.shared.lastTokensPerSecond),
        "draft_accept_pct": speculativeActive ? 0.0 : 0.0,  // TODO: extract from MLX stream info
        "peak_memory_gb": Double(BadAppleEngine.shared.memoryUsageGB),
        "speculative": speculativeActive
    ]
    if BadAppleEngine.shared.lastCacheHit {
        metrics["tier"] = "cache"
    } else if BadAppleEngine.shared.lastTokenCount == 0 {
        metrics["tier"] = "deterministic"
    } else if BadAppleEngine.shared.fastTierEnabled {
        metrics["tier"] = "fast"
    } else if speculativeActive {
        metrics["tier"] = "speculative"
    } else {
        metrics["tier"] = "main"
    }
    return metrics
}

// MARK: - Connection handling

private func validateRequest(_ prompt: String, _ maxTokens: Int) -> String? {
    let byteCount = prompt.utf8.count
    if byteCount == 0 {
        return "prompt must not be empty"
    }
    if byteCount > MAX_PROMPT_BYTES {
        return "prompt exceeds the \(MAX_PROMPT_BYTES)-byte limit"
    }
    if maxTokens < 1 || maxTokens > MAX_NEW_TOKENS {
        return "max_new_tokens must be between 1 and \(MAX_NEW_TOKENS)"
    }
    return nil
}

private func ensureModelLoaded() async -> Bool {
    if BadAppleEngine.shared.isLoaded {
        return true
    }
    if !BadAppleEngine.shared.isLoading {
        await BadAppleEngine.shared.loadModel()
    }
    let deadline = Date().addingTimeInterval(300)
    while !BadAppleEngine.shared.isLoaded && BadAppleEngine.shared.isLoading && Date() < deadline {
        try? await Task.sleep(nanoseconds: 100_000_000)
    }
    return BadAppleEngine.shared.isLoaded
}

// MARK: - Agent Protocol (LAP)

private func agentRespond(_ fd: Int32, writeQueue: DispatchQueue, reqId: String?, result: Any?, error: String?) {
    var frame: [String: Any] = ["id": reqId ?? NSNull()]
    if let error = error {
        frame["type"] = "error"
        frame["message"] = error
    } else {
        frame["type"] = "response"
        frame["result"] = result ?? NSNull()
    }
    writeQueue.sync {
        _ = writeJSON(fd, frame)
    }
}

private func handleAgentRequest(_ raw: String, fd: Int32, writeQueue: DispatchQueue) async {
    let jsonStr = String(raw.dropFirst("__BADAPPLE_AGENT__ ".count))
    guard let jsonData = jsonStr.data(using: .utf8),
          let req = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
        agentRespond(fd, writeQueue: writeQueue, reqId: nil, result: nil, error: "invalid agent JSON")
        return
    }

    let reqId = req["id"] as? String
    let method = req["method"] as? String ?? ""
    let params = req["params"] as? [String: Any] ?? [:]

    switch method {
    case "list_models":
        let statuses = BadAppleEngine.shared.modelManager.status()
        let current = BadAppleEngine.shared.modelId
        let models: [[String: Any]] = statuses.map { s in
            var m = s
            let mid = s["id"] as? String ?? ""
            m["loaded"] = (mid == current)
            m["active"] = (mid == current)
            return m
        }
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "text": models.isEmpty ? "No models tracked" : "Found \(models.count) model(s)",
            "models": models,
        ], error: nil)

    case "scan_models":
        BadAppleEngine.shared.modelManager.backgroundRefreshAll()
        let statuses = BadAppleEngine.shared.modelManager.status()
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "text": "Scanned HuggingFace cache. Found \(statuses.count) tracked model(s).",
            "models": statuses,
        ], error: nil)

    case "model_info":
        let modelId = params["model_id"] as? String ?? ""
        if modelId.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "model_id is required")
            return
        }
        if let status = BadAppleEngine.shared.modelManager.modelStatus(modelId: modelId) {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: status, error: nil)
        } else {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "unknown or invalid model_id: \(modelId)")
        }

    case "switch_main_model":
        let modelRef = params["model_ref"] as? String ?? ""
        if modelRef.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "model_ref is required")
            return
        }
        // Resolve model_id -> repo_id if a built-in id was passed.
        let repoId: String
        if let profile = BadAppleEngine.shared.modelManager.listProfiles().first(where: { $0.id == modelRef || $0.repoId == modelRef }) {
            repoId = profile.repoId
        } else {
            repoId = modelRef
        }
        await BadAppleEngine.shared.unload()
        BadAppleEngine.shared.configureMainModel(modelId: repoId)
        await BadAppleEngine.shared.loadModel()
        let loaded = BadAppleEngine.shared.isLoaded
        let matched = BadAppleEngine.shared.modelManager.listProfiles().first { $0.repoId == repoId }
        BadAppleEngine.shared.modelManager.markLoaded(modelId: matched?.id ?? repoId, localPath: nil)
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "result": loaded ? "Switched to \(repoId)" : "Failed to load \(repoId)",
            "model": repoId,
            "loaded": loaded,
        ], error: nil)

    case "verify_models":
        var results: [[String: Any]] = []
        for profile in BadAppleEngine.shared.modelManager.listProfiles() {
            var result = BadAppleEngine.shared.modelManager.verifyProvenance(modelId: profile.id)
            result["id"] = profile.id
            result["repo_id"] = profile.repoId
            results.append(result)
        }
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "text": "Verified \(results.count) model(s)",
            "results": results,
        ], error: nil)

    case "add_model":
        let localPath = params["path"] as? String ?? ""
        let modelId = params["model_id"] as? String ?? ""
        let repoId = params["repo_id"] as? String ?? modelId
        if localPath.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "path is required")
            return
        }
        let result = BadAppleEngine.shared.modelManager.addModel(
            modelId: modelId.isEmpty ? repoId : modelId,
            repoId: repoId,
            localPath: localPath
        )
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: result, error: result["error"] as? String)

    case "remove_model":
        let modelId = params["model_id"] as? String ?? ""
        if modelId.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "model_id is required")
            return
        }
        let result = BadAppleEngine.shared.modelManager.removeModel(modelId: modelId)
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: result, error: result["error"] as? String)

    case "recommend_model":
        let query = (params["query"] as? String) ?? ""
        let result = query.isEmpty
            ? BadAppleEngine.shared.modelManager.recommendForMemory()
            : BadAppleEngine.shared.modelManager.recommendForQuery(query)
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: result, error: nil)

    case "runtime_status":
        let status = await BadAppleEngine.shared.runtimeStatus()
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: status, error: nil)

    case "discover_tools":
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "tools": ["get_current_time", "read_file", "list_directory", "search_content",
                       "run_shell", "write_file", "screen_capture", "run_applescript",
                       "list_shortcuts", "run_shortcut", "index_documents", "search_notes",
                       "read_working_memory", "write_working_memory", "clear_working_memory",
                       "runtime_status", "describe_image", "translate_text",
                       "consolidate_memory", "workspace_status", "read_document",
                       "search_local_files", "set_session_seed", "get_session_seed"],
        ], error: nil)

    case "invoke_tool":
        let toolName = params["name"] as? String ?? ""
        let toolArgs = (params["args"] as? [String: Any] ?? [:]).mapValues { "\($0)" }
        let result = await BadAppleEngine.shared.executeTool(name: toolName, args: toolArgs)
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["tool": toolName, "result": result], error: nil)

    case "inference":
        let inferencePrompt = params["prompt"] as? String ?? ""
        let maxTokens = params["max_new_tokens"] as? Int ?? 120
        if inferencePrompt.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "inference requires prompt")
            return
        }
        do {
            let text = try await BadAppleEngine.shared.generate(prompt: inferencePrompt, maxTokens: maxTokens)
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["text": text], error: nil)
        } catch {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "inference failed: \(error.localizedDescription)")
        }

    case "run_agent_task":
        let goal = params["goal"] as? String ?? ""
        let maxSteps = params["max_steps"] as? Int ?? 10
        if goal.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "goal is required")
            return
        }
        do {
            let task = try await BadAppleEngine.shared.submitAgentTask(goal: goal, maxSteps: maxSteps)
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
                "task_id": task.id, "status": task.status.rawValue, "goal": task.goal,
            ], error: nil)
        } catch {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "agent task failed: \(error.localizedDescription)")
        }

    case "list_agent_tasks":
        let tasks = await BadAppleEngine.shared.listAgentTasks()
        let taskList: [[String: Any]] = tasks.map { t in
            ["id": t.id, "status": t.status.rawValue, "goal": t.goal, "steps": t.steps.count, "max_steps": t.maxSteps]
        }
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["tasks": taskList], error: nil)

    case "cancel_agent_task":
        let taskId = params["task_id"] as? String ?? ""
        let ok = (try? await BadAppleEngine.shared.cancelAgentTask(taskId)) ?? false
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["cancelled": ok], error: nil)

    case "pause_agent_task":
        let taskId = params["task_id"] as? String ?? ""
        let ok = (try? await BadAppleEngine.shared.pauseAgentTask(taskId)) ?? false
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["paused": ok], error: nil)

    case "resume_agent_task":
        let taskId = params["task_id"] as? String ?? ""
        let ok = (try? await BadAppleEngine.shared.resumeAgentTask(taskId)) ?? false
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["resumed": ok], error: nil)

    case "set_fast_tier":
        let enabled = params["enabled"] as? Bool ?? true
        BadAppleEngine.shared.fastTierEnabled = enabled
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["fast_tier": enabled], error: nil)

    case "set_autopilot":
        let enabled = params["enabled"] as? Bool ?? false
        BadAppleEngine.shared.autopilot = enabled
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["autopilot": enabled], error: nil)

    case "switch_persona":
        let name = params["name"] as? String ?? ""
        let ok = BadAppleEngine.shared.switchPersona(name)
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId,
                     result: ok ? ["active_persona": BadAppleEngine.shared.activePersona] : nil,
                     error: ok ? nil : "unknown persona '\(name)'")

    case "set_workspace":
        let path = params["path"] as? String ?? ""
        if path.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "set_workspace requires path")
            return
        }
        BadAppleEngine.shared.workspacePath = path
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["status": "workspace set to \(path)"], error: nil)

    case "get_workspace":
        let ws = BadAppleEngine.shared.workspacePath
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["workspace": ws as Any? ?? NSNull()], error: nil)

    case "flush_vram":
        BadAppleEngine.shared.clearCache()
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["result": "VRAM cache cleared"], error: nil)

    case "unload_model":
        await BadAppleEngine.shared.unload()
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["result": "model unloaded"], error: nil)

    case "get_pending_approvals":
        let pending = BadAppleEngine.shared.listPendingApprovals()
        let summary: [[String: Any]] = pending.map { p in
            ["id": p.id, "name": p.name, "arguments": p.args]
        }
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["pending": summary], error: nil)

    case "audit_tail":
        let ledgerPath = "/var/lib/bad_apple/ledger.jsonl"
        var entries: [Any] = []
        if let lines = try? String(contentsOfFile: ledgerPath, encoding: .utf8) {
            let allLines = lines.split(separator: "\n").suffix(20)
            for line in allLines {
                if let data = line.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    entries.append(json)
                }
            }
        }
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["entries": entries], error: nil)

    case "p2p_peers", "p2p_sync", "p2p_models", "p2p_pull_model", "p2p_send_model", "p2p_receive_model":
        let env = ProcessInfo.processInfo.environment
        guard env["BADAPPLE_P2P"] == "1" || env["BADAPPLE_P2P_ENABLED"] == "1" else {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil,
                         error: "P2P is off. Set BADAPPLE_P2P=1 or enable it from the menu bar Mesh > P2P Sync.")
            return
        }
        let p2pArgs = p2pCommandArgs(method: method, params: params)
        let output = runP2PHelper(arguments: p2pArgs)
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let err = json["error"] as? String
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: err == nil ? json : nil, error: err)
        } else if !output.isEmpty {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["text": output], error: nil)
        } else {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil,
                         error: "P2P helper produced no output")
        }

    case "identity_status":
        let helper = IdentityAgentClient.shared
        let available = helper.isAvailable
        let pubKey = helper.publicKey()
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: [
            "status": available ? "online" : "offline",
            "public_key": pubKey as Any? ?? NSNull(),
        ], error: nil)

    case "identity_sign":
        let challenge = params["challenge"] as? String ?? ""
        if challenge.isEmpty || challenge.count > 4096 {
            agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil,
                         error: "identity_sign requires a challenge up to 4096 characters")
            return
        }
        let sig = IdentityAgentClient.shared.sign(message: Data(challenge.utf8))
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId,
                     result: sig.map { ["signature": $0] } ?? nil,
                     error: sig == nil ? "identity agent unavailable" : nil)

    case "kill_switch":
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["runtime": ["mode": "normal"]], error: nil)

    case "private_mode":
        let enabled = params["enabled"] as? Bool ?? true
        BadAppleEngine.shared.privateMode = enabled
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["private_mode": enabled], error: nil)

    case "set_airgap":
        let enabled = params["enabled"] as? Bool ?? false
        BadAppleEngine.shared.airgap = enabled
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["airgap": enabled], error: nil)

    case "airgap_status":
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: ["airgap": BadAppleEngine.shared.airgap], error: nil)

    default:
        agentRespond(fd, writeQueue: writeQueue, reqId: reqId, result: nil, error: "unknown agent method: \(method)")
    }
}

private func handleConnection(_ fd: Int32, secret: Data?) async {
    defer { close(fd) }
    setSocketOptions(fd)

    let reader = BufferedReader(fd: fd)

    // 1) Read Hello
    guard let helloData = reader.readLine(),
          let hello = try? JSONSerialization.jsonObject(with: helloData) as? [String: Any] else {
        _ = writeJSON(fd, ["type": "error", "message": "invalid or empty hello frame"])
        return
    }

    guard let version = hello["version"] as? Int,
          (version == SLICKS_VERSION || version == SLICKS_VERSION_2),
          let timestampMs = hello["timestamp_ms"] as? UInt64 ?? (hello["timestamp_ms"] as? Int).map(UInt64.init),
          isTimestampFresh(timestampMs),
          let clientNonce = hello["client_nonce"] as? String,
          isNonceValid(clientNonce) else {
        _ = writeJSON(fd, ["type": "error", "message": "invalid or stale SLICKS hello"])
        return
    }

    let helloClientPubkey = hello["client_pubkey"] as? String
    let serverNonce = randomNonce()
    let helper = version == SLICKS_VERSION_2 ? IdentityAgentClient.shared : nil

    let serverPubkey: Any
    let serverProofValue: String?
    if version == SLICKS_VERSION_2 {
        guard let helper = helper, helper.isAvailable,
              let serverPubkeyString = helper.publicKey() else {
            _ = writeJSON(fd, ["type": "error", "message": "SLICKS v2 identity helper unavailable"])
            return
        }
        serverPubkey = serverPubkeyString
        serverProofValue = helper.sign(message: Data(serverMaterial(version: version, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce).utf8))
        guard serverProofValue != nil else {
            _ = writeJSON(fd, ["type": "error", "message": "SLICKS v2 server signing failed"])
            return
        }
    } else {
        serverPubkey = NSNull()
        serverProofValue = secret.map { serverProof(version: version, secret: $0, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce) } ?? randomNonce()
    }

    let challenge: [String: Any] = [
        "type": "challenge",
        "version": version,
        "server_nonce": serverNonce,
        "server_pubkey": serverPubkey as Any,
        "proof": serverProofValue as Any
    ]
    guard writeJSON(fd, challenge) else { return }

    // 2) Read Execute
    guard let execData = reader.readLine(),
          let exec = try? JSONSerialization.jsonObject(with: execData) as? [String: Any] else {
        _ = writeJSON(fd, ["type": "error", "message": "invalid execute frame"])
        return
    }

    guard let execVersion = exec["version"] as? Int, execVersion == version,
          let execTimestamp = exec["timestamp_ms"] as? UInt64 ?? (exec["timestamp_ms"] as? Int).map(UInt64.init),
          execTimestamp == timestampMs,
          let execClientNonce = exec["client_nonce"] as? String, execClientNonce == clientNonce,
          let execServerNonce = exec["server_nonce"] as? String, execServerNonce == serverNonce,
          let prompt = exec["prompt"] as? String,
          let maxTokens = exec["max_new_tokens"] as? Int,
          let proof = exec["proof"] as? String else {
        _ = writeJSON(fd, ["type": "error", "message": "invalid SLICKS execute frame"])
        return
    }

    if let execClientPubkey = exec["client_pubkey"] as? String,
       let helloClientPubkey = helloClientPubkey,
       execClientPubkey != helloClientPubkey {
        _ = writeJSON(fd, ["type": "error", "message": "SLICKS client_pubkey mismatch"])
        return
    }

    if let validationError = validateRequest(prompt, maxTokens) {
        _ = writeJSON(fd, ["type": "error", "message": validationError])
        return
    }

    let execClientPubkey = exec["client_pubkey"] as? String ?? helloClientPubkey

    if !verifyClientProof(version: version, secret: secret, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce, prompt: prompt, maxTokens: maxTokens, proof: proof, clientPubkey: execClientPubkey, helper: helper) {
        _ = writeJSON(fd, ["type": "error", "message": "SLICKS client authentication failed"])
        return
    }

    // 3) Agent protocol: JSON-RPC style request over the SLICKS channel.
    if prompt.hasPrefix("__BADAPPLE_AGENT__ ") {
        let writeQueue = DispatchQueue(label: "badapple-engine.write.\(fd)")
        writeQueue.sync {
            _ = writeJSON(fd, ["type": "accepted"])
        }
        await handleAgentRequest(prompt, fd: fd, writeQueue: writeQueue)
        return
    }

    // 4) Accepted
    let writeQueue = DispatchQueue(label: "badapple-engine.write.\(fd)")
    writeQueue.sync {
        _ = writeJSON(fd, ["type": "accepted"])
    }

    // 5) Preprocess voice / persona / benchmark prefixes
    var userPrompt = prompt
    var voiceMode = false
    var benchmarkMode = false
    stripPrefixes(&userPrompt, voice: &voiceMode, benchmark: &benchmarkMode)
    _ = benchmarkMode  // reserved for future deterministic-benchmark path

    // 6) Meta / control commands
    if let metaResponse = await handleMetaRequest(userPrompt) {
        writeQueue.sync {
            _ = writeJSON(fd, ["type": "done", "text": metaResponse, "metrics": makeMetrics()])
        }
        return
    }

    // 7) Load model if needed, then generate.
    let loaded = await ensureModelLoaded()
    guard loaded else {
        _ = writeJSON(fd, ["type": "error", "message": "The AI model is not loaded yet."])
        return
    }

    await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
        var completed = false

        BadAppleEngine.shared.generateStreaming(
            prompt: userPrompt,
            voiceMode: voiceMode,
            maxTokens: maxTokens,
            onToken: { token in
                writeQueue.async {
                    guard !completed, !token.isEmpty else { return }
                    _ = writeJSON(fd, ["type": "token", "text": token])
                }
            },
            onComplete: { fullText in
                writeQueue.async {
                    guard !completed else { return }
                    completed = true
                    _ = writeJSON(fd, ["type": "done", "text": fullText, "metrics": makeMetrics()])
                    continuation.resume()
                }
            },
            onError: { error in
                writeQueue.async {
                    guard !completed else { return }
                    completed = true
                    _ = writeJSON(fd, ["type": "error", "message": error])
                    continuation.resume()
                }
            }
        )
    }
}

// MARK: - Accept loop

private func acceptLoop(fd: Int32, secret: Data?) {
    while !shouldStop {
        var addr = sockaddr_un()
        var len = socklen_t(MemoryLayout<sockaddr_un>.size)
        let client = withUnsafeMutablePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                withUnsafeMutablePointer(to: &len) { lenPtr in
                    accept(fd, sockaddrPtr, lenPtr)
                }
            }
        }
        if client < 0 {
            let err = errno
            if err == EINTR { continue }
            if err == EBADF || err == EINVAL || shouldStop { break }
            log("accept() failed: \(err)")
            continue
        }
        Task {
            await handleConnection(client, secret: secret)
        }
    }
    log("Accept loop stopped")
}

private var shouldStop = false

private func installSignalHandlers() {
    let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)

    let stop: () -> Void = {
        shouldStop = true
        if gListenFd >= 0 {
            close(gListenFd)
            gListenFd = -1
        }
        exit(0)
    }

    sigint.setEventHandler(handler: stop)
    sigterm.setEventHandler(handler: stop)
    sigint.resume()
    sigterm.resume()

    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
}

// MARK: - Main

private func printUsage() {
    print("""
    Bad Apple native MLX inference daemon (SLICKS v1/v2)

    Usage:
      badapple-engine [--help]

    Environment:
      BADAPPLE_SOCKET_PATH          Unix socket path (default: \(DEFAULT_SOCKET_PATH))
      BADAPPLE_MODEL                Main model id (default: \(DEFAULT_FAST_MODEL))
      BADAPPLE_MAIN_MODEL           Alias for BADAPPLE_MODEL if the former is unset
      BADAPPLE_MODEL_REVISION       Model revision or branch (default: main)
      BADAPPLE_LAZY_MAIN_MODEL      1 to load on first request (default: 1)
      BADAPPLE_SLICKS_KEY_PATH      Path to SLICKS HMAC key (default: /var/lib/bad_apple/slicks.key)
      BADAPPLE_WORKSPACE_DIR        Optional workspace for RAG context
      BADAPPLE_FAST_TIER            1 to enable 0.5B fast tier for simple queries
      BADAPPLE_FAST_MODEL           Fast tier model id (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit)
      BADAPPLE_SPECULATIVE_DRAFT    Draft model id for speculative decoding (empty = disabled)
      BADAPPLE_NUM_DRAFT_TOKENS     Draft tokens per step (default: 2)
      BADAPPLE_MAX_KV_SIZE          Max KV cache size (default: 4096)
      BADAPPLE_PREFILL_STEP_SIZE    Prefill step size (default: 4096)
      BADAPPLE_VRAM_BUDGET_GB       VRAM budget in GB (default: 80% of physical memory)
    """)
}

private func p2pCommandArgs(method: String, params: [String: Any]) -> [String] {
    switch method {
    case "p2p_peers": return ["peers"]
    case "p2p_sync": return ["sync"]
    case "p2p_models": return ["models"]
    case "p2p_pull_model":
        return ["pull", params["peer_id"] as? String ?? "", params["model_id"] as? String ?? ""]
    case "p2p_send_model":
        return ["send", params["peer_id"] as? String ?? "", params["model_id"] as? String ?? ""]
    case "p2p_receive_model":
        var args = ["receive"]
        if let peerId = params["peer_id"] as? String, !peerId.isEmpty { args.append(peerId) }
        if let modelId = params["model_id"] as? String, !modelId.isEmpty { args.append(modelId) }
        return args
    default: return []
    }
}

private func findP2PHelper() -> String? {
    let fm = FileManager.default
    // Try the running executable's directory first (release build).
    if let exe = Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("badapple-p2p").path,
       fm.fileExists(atPath: exe) { return exe }
    let candidates = [
        "target/release/badapple-p2p",
        "../target/release/badapple-p2p",
        "../../target/release/badapple-p2p",
    ]
    let env = ProcessInfo.processInfo.environment
    if let custom = env["BADAPPLE_P2P_HELPER"], fm.fileExists(atPath: custom) { return custom }
    for c in candidates {
        if fm.fileExists(atPath: c) { return c }
        if let home = NSHomeDirectory() as String? {
            let abs = (home as NSString).appendingPathComponent(c)
            if fm.fileExists(atPath: abs) { return abs }
        }
    }
    return nil
}

private func runP2PHelper(arguments: [String]) -> String {
    guard let helper = findP2PHelper() else {
        return "{\"error\": \"badapple-p2p helper not found; build with cargo build --release --bin badapple-p2p\"}"
    }
    let task = Process()
    task.executableURL = URL(fileURLWithPath: helper)
    task.arguments = arguments
    var env = ProcessInfo.processInfo.environment
    env["BADAPPLE_ORIGIN_INSTANCE"] = "badapple-engine"
    env["BADAPPLE_P2P_SECRET"] = env["BADAPPLE_P2P_SECRET"] ?? env["BADAPPLE_SLICKS_SECRET"]
    task.environment = env
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    do {
        try task.run()
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    } catch {
        return "{\"error\": \"failed to run badapple-p2p: \(error.localizedDescription)\"}"
    }
}

private func resolveMainModel() -> String {
    let env = ProcessInfo.processInfo.environment
    if let m = env["BADAPPLE_MODEL"]?.trimmingCharacters(in: .whitespacesAndNewlines), !m.isEmpty {
        return m
    }
    if let m = env["BADAPPLE_MAIN_MODEL"]?.trimmingCharacters(in: .whitespacesAndNewlines), !m.isEmpty {
        return m
    }
    return DEFAULT_FAST_MODEL
}

private func resolveRevision() -> String {
    let env = ProcessInfo.processInfo.environment
    if let r = env["BADAPPLE_MODEL_REVISION"]?.trimmingCharacters(in: .whitespacesAndNewlines), !r.isEmpty {
        return r
    }
    return "main"
}

func runDaemonMain() {
    if CommandLine.arguments.contains("--help") || CommandLine.arguments.contains("-h") {
        printUsage()
        return
    }

    // Set workspace if provided; this is used by the RAG module.
    if let workspace = ProcessInfo.processInfo.environment["BADAPPLE_WORKSPACE_DIR"],
       !workspace.isEmpty {
        BadAppleEngine.shared.workspacePath = workspace
    }

    // Configure the main model before any lazy or eager load.
    let modelId = resolveMainModel()
    let revision = resolveRevision()
    BadAppleEngine.shared.configureMainModel(modelId: modelId, revision: revision)
    log("Configured main model: \(modelId)@\(revision)")

    // Enable fast tier if requested by the environment.
    if let fastTier = ProcessInfo.processInfo.environment["BADAPPLE_FAST_TIER"],
       (fastTier == "1" || fastTier.lowercased() == "true" || fastTier.lowercased() == "on") {
        BadAppleEngine.shared.fastTierEnabled = true
        log("Fast tier enabled (BADAPPLE_FAST_TIER=1). Simple queries will route to \(BadAppleInference.envFastModelId).")
    }

    // Override VRAM budget if provided (in GB).
    if let budgetGB = ProcessInfo.processInfo.environment["BADAPPLE_VRAM_BUDGET_GB"],
       let gb = UInt64(budgetGB), gb > 0 {
        Task { await BadAppleEngine.shared.setVRAMBudgetGB(gb) }
        log("VRAM budget set to \(gb) GB (BADAPPLE_VRAM_BUDGET_GB)")
    }

    let lazy = ProcessInfo.processInfo.environment["BADAPPLE_LAZY_MAIN_MODEL"]?.trimmingCharacters(in: .whitespacesAndNewlines) != "0"
    if !lazy {
        Task(priority: .userInitiated) {
            await BadAppleEngine.shared.loadModel()
            log("Eager model load complete: \(BadAppleEngine.shared.isLoaded ? "ready" : "failed")")
        }
    } else {
        log("Lazy model load enabled; model will load on first request")
    }

    let socketPath = ProcessInfo.processInfo.environment["BADAPPLE_SOCKET_PATH"] ?? DEFAULT_SOCKET_PATH
    let fd = bindUnixSocket(path: socketPath)
    guard fd >= 0 else {
        log("Cannot bind socket at \(socketPath)")
        exit(1)
    }
    gListenFd = fd
    log("Listening on \(socketPath)")

    let secret = loadSlicksSecret()

    installSignalHandlers()

    DispatchQueue(label: "badapple-engine.accept").async {
        acceptLoop(fd: fd, secret: secret)
    }

    dispatchMain()
}
