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

private func verifyClientProof(version: Int, secret: Data?, timestampMs: UInt64, clientNonce: String, serverNonce: String, prompt: String, maxTokens: Int, proof: String) -> Bool {
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
    default:
        return nil
    }
}

// MARK: - Metrics

private func makeMetrics() -> [String: Any] {
    var metrics: [String: Any] = [
        "tokens": BadAppleEngine.shared.lastTokenCount,
        "decode_tps": Double(BadAppleEngine.shared.lastTokensPerSecond),
        "total_tps": Double(BadAppleEngine.shared.lastTokensPerSecond),
        "draft_accept_pct": 0.0,
        "peak_memory_gb": Double(BadAppleEngine.shared.memoryUsageGB)
    ]
    if BadAppleEngine.shared.lastCacheHit {
        metrics["tier"] = "cache"
    } else if BadAppleEngine.shared.lastTokenCount == 0 {
        metrics["tier"] = "deterministic"
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

    let challenge: [String: Any] = [
        "type": "challenge",
        "version": version,
        "server_nonce": serverNonce,
        "server_pubkey": NSNull(),
        "proof": secret.map { serverProof(version: version, secret: $0, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce) } ?? randomNonce()
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

    if !verifyClientProof(version: version, secret: secret, timestampMs: timestampMs, clientNonce: clientNonce, serverNonce: serverNonce, prompt: prompt, maxTokens: maxTokens, proof: proof) {
        _ = writeJSON(fd, ["type": "error", "message": "SLICKS client authentication failed"])
        return
    }

    // 3) Agent protocol is not yet implemented in the native daemon.
    if prompt.hasPrefix("__BADAPPLE_AGENT__ ") {
        _ = writeJSON(fd, ["type": "error", "message": "Agent protocol is not supported by badapple-engine"])
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
      BADAPPLE_SOCKET_PATH        Unix socket path (default: \(DEFAULT_SOCKET_PATH))
      BADAPPLE_MODEL              Main model id (default: \(DEFAULT_FAST_MODEL))
      BADAPPLE_MAIN_MODEL         Alias for BADAPPLE_MODEL if the former is unset
      BADAPPLE_MODEL_REVISION     Model revision or branch (default: main)
      BADAPPLE_LAZY_MAIN_MODEL    1 to load on first request (default: 1)
      BADAPPLE_SLICKS_KEY_PATH    Path to SLICKS HMAC key (default: /var/lib/bad_apple/slicks.key)
      BADAPPLE_WORKSPACE_DIR      Optional workspace for RAG context
    """)
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
