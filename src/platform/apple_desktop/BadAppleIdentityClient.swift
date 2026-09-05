// BadAppleIdentityClient — client for the long-lived `badapple-identity-agent`.
//
// The identity agent runs in the user's Aqua / GUI session and performs
// Secure Enclave P-256 signing and verification. This file is compiled into
// both the engine and the menu bar so any logic-layer code can sign or verify
// data with the enclave-backed key.

import Foundation
import Darwin

/// Talks to the long-lived `badapple-identity-agent` (or the legacy Python
/// identity agent) over its Unix socket. The agent lives in the user's Aqua
/// / GUI session, so Secure Enclave signing and verification succeed reliably.
final class IdentityAgentClient {
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

private func log(_ message: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    print("[\(ts)] \(message)")
    fflush(stdout)
}
