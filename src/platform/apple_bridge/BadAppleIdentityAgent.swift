import CryptoKit
import Foundation
import LocalAuthentication
import Security

// Long-lived identity agent that keeps Secure Enclave signing in the user
// Aqua / GUI session.
//
// The Bad Apple MLX daemon and CLI connect to this agent over a Unix socket
// instead of calling the one-shot `badapple-identity` helper directly. Because
// this agent runs in the user's Aqua session, CryptoTokenKit / Secure Enclave
// key operations succeed reliably.
//
// The agent serializes signing requests because the one-shot helper is not
// safe for immediate back-to-back invocations from the same process.
//
// Protocol (JSON line per connection):
//   {"command": "public_key"}
//     -> {"ok": true, "public_key": "<base64>"}
//   {"command": "sign", "message_b64": "<base64>"}
//     -> {"ok": true, "signature": "<base64>"}
//   {"command": "verify", "message_b64": "<base64>", "signature": "<base64>",
//      "public_key": "<base64>"}
//     -> {"ok": true, "valid": <bool>}
//   {"command": "status"}
//     -> {"ok": true, "status": "secure-enclave:<fingerprint>|missing|unavailable"}
//   {"command": "biometric_gate", "reason": "..."}
//     -> {"ok": true, "result": "approved"|"denied"} or {"ok": false, "error": "..."}

private let defaultSocketPath = "/var/run/badapple/identity.sock"
private let signCooldown: TimeInterval = 0.25

// MARK: - JSON helpers

private struct AnyJSON: Codable {
    let value: Any

    init(_ value: Any) { self.value = value }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            value = NSNull()
        } else if let v = try? container.decode(Bool.self) {
            value = v
        } else if let v = try? container.decode(Int.self) {
            value = v
        } else if let v = try? container.decode(Double.self) {
            value = v
        } else if let v = try? container.decode(String.self) {
            value = v
        } else if let v = try? container.decode([AnyJSON].self) {
            value = v.map { $0.value }
        } else if let v = try? container.decode([String: AnyJSON].self) {
            value = v.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull: try container.encodeNil()
        case let v as Bool: try container.encode(v)
        case let v as Int: try container.encode(v)
        case let v as Double: try container.encode(v)
        case let v as String: try container.encode(v)
        case let v as [Any]: try container.encode(v.map { AnyJSON($0) })
        case let v as [String: Any]: try container.encode(v.mapValues { AnyJSON($0) })
        default:
            try container.encodeNil()
        }
    }
}

private func jsonDecode(_ data: Data) -> [String: Any]? {
    guard let parsed = try? JSONDecoder().decode([String: AnyJSON].self, from: data) else {
        return nil
    }
    return parsed.mapValues { $0.value }
}

private func jsonEncode(_ object: [String: Any]) -> String {
    let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
    return String(data: data, encoding: .utf8) ?? "{}"
}

// MARK: - Agent

final class IdentityAgent {
    let socketPath: String
    let helperPath: String
    private var serverFD: Int32 = -1
    private var running = false
    private let signQueue = DispatchQueue(label: "badapple.identity.sign")
    private var lastSignTime: Date = .distantPast
    private var cachedPublicKey: String?
    private let cacheLock = NSLock()

    init(socketPath: String, helperPath: String) {
        self.socketPath = socketPath
        self.helperPath = helperPath
    }

    func start() throws {
        let dir = (socketPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        unlink(socketPath)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(domain: "IdentityAgent", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
        }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        let pathLen = min(pathBytes.count,
                          MemoryLayout.size(ofValue: addr.sun_path))
        withUnsafeMutablePointer(to: &addr) { addrPtr in
            let raw = UnsafeMutableRawPointer(addrPtr)
            let sunPathOffset = MemoryLayout<sockaddr_un>.offset(of: \.sun_path) ?? 0
            let dst = raw.advanced(by: sunPathOffset)
                .assumingMemoryBound(to: CChar.self)
            _ = pathBytes.withUnsafeBufferPointer { src in
                memcpy(dst, src.baseAddress, pathLen)
            }
        }
        let bindResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Foundation.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            close(fd)
            throw NSError(domain: "IdentityAgent", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "bind() failed"])
        }
        guard Foundation.listen(fd, 64) == 0 else {
            close(fd)
            throw NSError(domain: "IdentityAgent", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "listen() failed"])
        }
        guard chmod(socketPath, 0o660) == 0 else {
            close(fd)
            throw NSError(domain: "IdentityAgent", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "chmod() failed"])
        }
        serverFD = fd
        running = true
        acceptLoop()
    }

    func stop() {
        running = false
        if serverFD >= 0 {
            // Close and unlink to unblock accept().
            let fd = serverFD
            serverFD = -1
            close(fd)
        }
        unlink(socketPath)
    }

    private func acceptLoop() {
        while running {
            var clientAddr = sockaddr()
            var clientLen = socklen_t(MemoryLayout<sockaddr>.size)
            let clientFD = accept(serverFD, &clientAddr, &clientLen)
            if clientFD < 0 {
                if !running { return }
                if errno == EINTR { continue }
                return
            }
            DispatchQueue.global().async { [weak self] in
                self?.handleClient(fd: clientFD)
            }
        }
    }

    private func handleClient(fd: Int32) {
        defer { close(fd) }
        // Read one line (up to newline).
        var buffer = Data()
        var byte: UInt8 = 0
        while read(fd, &byte, 1) == 1 {
            if byte == 0x0A { break }
            buffer.append(byte)
            if buffer.count > 1_000_000 { break } // sanity cap
        }
        guard !buffer.isEmpty else { return }
        guard let request = jsonDecode(buffer) else {
            send(fd, ["ok": false, "error": "invalid json"])
            return
        }
        let response = dispatch(request: request)
        send(fd, response)
    }

    private func send(_ fd: Int32, _ payload: [String: Any]) {
        var line = jsonEncode(payload)
        line += "\n"
        let data = Data(line.utf8)
        _ = data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> Int in
            Foundation.send(fd, buf.baseAddress, buf.count, 0)
        }
    }

    // MARK: - Dispatch

    private func dispatch(request: [String: Any]) -> [String: Any] {
        guard let command = request["command"] as? String else {
            return ["ok": false, "error": "missing command"]
        }
        switch command {
        case "public_key": return publicKey()
        case "sign": return sign(request)
        case "verify": return verify(request)
        case "status": return status()
        case "biometric_gate": return biometricGate(request)
        default:
            return ["ok": false, "error": "unknown command: \(command)"]
        }
    }

    // MARK: - Helper invocation

    @discardableResult
    private func runHelper(_ args: [String]) -> (stdout: String, stderr: String, status: Int32) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: helperPath)
        process.arguments = args
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return ("", "\(error)", -1)
        }
        process.waitUntilExit()
        let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
        return (
            String(data: outData, encoding: .utf8) ?? "",
            String(data: errData, encoding: .utf8) ?? "",
            process.terminationStatus
        )
    }

    // MARK: - Commands

    private func publicKey() -> [String: Any] {
        cacheLock.lock()
        let cached = cachedPublicKey
        cacheLock.unlock()
        if let cached { return ["ok": true, "public_key": cached] }
        let result = runHelper(["public-key"])
        guard result.status == 0 else {
            return ["ok": false, "error": result.stderr.isEmpty ? "public-key failed" : result.stderr.trimmed]
        }
        let pub = result.stdout.trimmed
        cacheLock.lock()
        cachedPublicKey = pub
        cacheLock.unlock()
        return ["ok": true, "public_key": pub]
    }

    private func sign(_ request: [String: Any]) -> [String: Any] {
        guard let messageB64 = request["message_b64"] as? String else {
            return ["ok": false, "error": "missing message_b64"]
        }
        guard Data(base64Encoded: messageB64) != nil else {
            return ["ok": false, "error": "invalid message_b64"]
        }
        var result: [String: Any] = [:]
        signQueue.sync {
            // Cooldown: enforce at least signCooldown since the last sign.
            let now = Date()
            let elapsed = now.timeIntervalSince(lastSignTime)
            if elapsed < signCooldown {
                Thread.sleep(forTimeInterval: signCooldown - elapsed)
            }
            let res = runHelper(["sign", messageB64])
            lastSignTime = Date()
            if res.status == 0 {
                result = ["ok": true, "signature": res.stdout.trimmed]
            } else {
                result = ["ok": false, "error": res.stderr.isEmpty ? "sign failed" : res.stderr.trimmed]
            }
        }
        return result
    }

    private func verify(_ request: [String: Any]) -> [String: Any] {
        guard let messageB64 = request["message_b64"] as? String,
              let signatureB64 = request["signature"] as? String,
              let publicKeyB64 = request["public_key"] as? String,
              let message = Data(base64Encoded: messageB64),
              let signatureData = Data(base64Encoded: signatureB64),
              let publicKeyData = Data(base64Encoded: publicKeyB64)
        else {
            return ["ok": false, "error": "invalid base64 inputs"]
        }
        do {
            let pubKey = try P256.Signing.PublicKey(x963Representation: publicKeyData)
            let signature = try P256.Signing.ECDSASignature(derRepresentation: signatureData)
            if pubKey.isValidSignature(signature, for: message) {
                return ["ok": true, "valid": true]
            } else {
                return ["ok": true, "valid": false]
            }
        } catch {
            return ["ok": true, "valid": false]
        }
    }

    private func status() -> [String: Any] {
        let result = runHelper(["status"])
        if result.status == 0 {
            return ["ok": true, "status": result.stdout.trimmed]
        } else {
            return ["ok": false, "status": "unavailable: \(result.stderr.trimmed)"]
        }
    }

    private func biometricGate(_ request: [String: Any]) -> [String: Any] {
        let reason = (request["reason"] as? String)
            ?? "Approve a sensitive Bad Apple action"
        let context = LAContext()
        var authError: NSError?
        guard context.canEvaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics, error: &authError)
        else {
            return ["ok": false, "error": "biometric authentication is unavailable"]
        }
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        context.evaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics,
            localizedReason: reason
        ) { success, _ in
            granted = success
            semaphore.signal()
        }
        semaphore.wait()
        if granted {
            return ["ok": true, "result": "approved"]
        } else {
            return ["ok": false, "error": "biometric authentication was denied"]
        }
    }
}

// MARK: - String helpers

private extension String {
    var trimmed: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Executable path

private func executableDirectory() -> String {
    // Prefer the explicit helper override; otherwise resolve beside this agent.
    if let helper = ProcessInfo.processInfo.environment["BADAPPLE_IDENTITY_HELPER"],
       !helper.isEmpty {
        return (helper as NSString).deletingLastPathComponent
    }
    // Derive the directory containing this executable.
    var buf = [CChar](repeating: 0, count: 4096)
    let count = readlink("/proc/self/exe", &buf, buf.count)
    if count > 0 { return "" } // macOS has no /proc; fall through.
    if let argv0 = CommandLine.arguments.first {
        // Resolve argv[0] relative to cwd if needed.
        if (argv0 as NSString).isAbsolutePath {
            return (argv0 as NSString).deletingLastPathComponent
        }
        let cwd = FileManager.default.currentDirectoryPath
        let resolved = (cwd as NSString)
            .appendingPathComponent(argv0)
        return (resolved as NSString).deletingLastPathComponent
    }
    return "."
}

private func defaultHelperPath() -> String {
    if let helper = ProcessInfo.processInfo.environment["BADAPPLE_IDENTITY_HELPER"],
       !helper.isEmpty {
        return helper
    }
    // Default to the binary beside this executable.
    let dir = executableDirectory()
    let candidate = (dir as NSString).appendingPathComponent("badapple-identity")
    return candidate
}

// MARK: - Signal handling

private var shouldStop = false

private func installSignalHandlers() {
    let handler: @convention(c) (Int32) -> Void = { _ in
        shouldStop = true
    }
    signal(SIGTERM, handler)
    signal(SIGINT, handler)
}

// MARK: - Main

private func main() -> Int32 {
    let env = ProcessInfo.processInfo.environment
    let socketPath = env["BADAPPLE_IDENTITY_AGENT_SOCKET"].flatMap { $0.isEmpty ? nil : $0 }
        ?? defaultSocketPath
    // Disable the agent fallback inside this process so the agent talks
    // directly to the one-shot helper and does not recurse into itself.
    setenv("BADAPPLE_IDENTITY_AGENT_SOCKET", "", 1)
    if env["BADAPPLE_IDENTITY_BLOB"] == nil {
        let home = NSHomeDirectory()
        let blob = "\(home)/Library/Application Support/BadApple/identity.sekey"
        setenv("BADAPPLE_IDENTITY_BLOB", blob, 0)
    }

    let helperPath = defaultHelperPath()
    let agent = IdentityAgent(socketPath: socketPath, helperPath: helperPath)
    do {
        try agent.start()
    } catch {
        FileHandle.standardError.write(Data("[identity_agent] failed to start: \(error)\n".utf8))
        return 1
    }
    print("[identity_agent] listening on \(socketPath)", terminator: "\n")
    installSignalHandlers()
    while !shouldStop {
        Thread.sleep(forTimeInterval: 0.5)
    }
    agent.stop()
    return 0
}

exit(main())
