import AppIntents
import CryptoKit
import Darwin
import Foundation
import Security

private let badAppleSlicksVersion = 1
private let badAppleMaxFrameBytes = 1_048_576

private struct BadAppleServerFrame: Decodable {
    let type: String
    let version: Int?
    let serverNonce: String?
    let proof: String?
    let text: String?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case type
        case version
        case serverNonce = "server_nonce"
        case proof
        case text
        case message
    }
}

private final class BadAppleLineTransport {
    private let handle: FileHandle
    private var buffered = Data()

    init(socketPath: String) throws {
        let descriptor = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard descriptor >= 0 else {
            throw BadAppleIntentError("Unable to create the Bad Apple local socket")
        }

        var address = sockaddr_un()
        address.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8CString)
        let pathCapacity = MemoryLayout.size(ofValue: address.sun_path)
        guard pathBytes.count <= pathCapacity else {
            Darwin.close(descriptor)
            throw BadAppleIntentError("Bad Apple socket path is too long")
        }
        socketPath.withCString { source in
            withUnsafeMutablePointer(to: &address.sun_path) { destination in
                destination.withMemoryRebound(to: CChar.self, capacity: pathCapacity) {
                    _ = strncpy($0, source, pathCapacity - 1)
                }
            }
        }
        let addressLength = socklen_t(MemoryLayout<sa_family_t>.size + pathBytes.count)
        let connected = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(descriptor, $0, addressLength)
            }
        }
        guard connected == 0 else {
            let code = errno
            Darwin.close(descriptor)
            throw BadAppleIntentError("Unable to connect to Bad Apple (errno \(code))")
        }
        var readTimeout = timeval(tv_sec: 900, tv_usec: 0)
        var writeTimeout = timeval(tv_sec: 30, tv_usec: 0)
        _ = withUnsafePointer(to: &readTimeout) {
            setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, $0, socklen_t(MemoryLayout<timeval>.size))
        }
        _ = withUnsafePointer(to: &writeTimeout) {
            setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, $0, socklen_t(MemoryLayout<timeval>.size))
        }
        handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    }

    deinit {
        try? handle.close()
    }

    func write<T: Encodable>(_ frame: T) throws {
        var data = try JSONEncoder().encode(frame)
        guard data.count <= badAppleMaxFrameBytes else {
            throw BadAppleIntentError("Bad Apple request exceeds the IPC frame limit")
        }
        data.append(0x0A)
        try handle.write(contentsOf: data)
    }

    func read() throws -> BadAppleServerFrame {
        while true {
            if let newline = buffered.firstIndex(of: 0x0A) {
                let line = buffered[..<newline]
                buffered.removeSubrange(...newline)
                return try JSONDecoder().decode(BadAppleServerFrame.self, from: line)
            }
            guard let chunk = try handle.read(upToCount: 4096), !chunk.isEmpty else {
                throw BadAppleIntentError("Bad Apple closed the IPC connection")
            }
            buffered.append(chunk)
            guard buffered.count <= badAppleMaxFrameBytes else {
                throw BadAppleIntentError("Bad Apple response exceeds the IPC frame limit")
            }
        }
    }
}

private struct BadAppleHello: Encodable {
    let type = "hello"
    let version = badAppleSlicksVersion
    let timestampMs: UInt64
    let clientNonce: String

    enum CodingKeys: String, CodingKey {
        case type
        case version
        case timestampMs = "timestamp_ms"
        case clientNonce = "client_nonce"
    }
}

private struct BadAppleExecute: Encodable {
    let type = "execute"
    let version = badAppleSlicksVersion
    let timestampMs: UInt64
    let clientNonce: String
    let serverNonce: String
    let prompt: String
    let maxNewTokens: Int
    let proof: String

    enum CodingKeys: String, CodingKey {
        case type
        case version
        case timestampMs = "timestamp_ms"
        case clientNonce = "client_nonce"
        case serverNonce = "server_nonce"
        case prompt
        case maxNewTokens = "max_new_tokens"
        case proof
    }
}

private struct BadAppleIntentError: LocalizedError {
    let errorDescription: String?

    init(_ message: String) {
        errorDescription = message
    }
}

private enum BadAppleSlicks {
    static func socketPath() -> String {
        ProcessInfo.processInfo.environment["BADAPPLE_SOCKET_PATH"]
            ?? "/var/run/badapple/substrate.sock"
    }

    static func secret() throws -> Data {
        let raw: String
        if let configured = ProcessInfo.processInfo.environment["BADAPPLE_SLICKS_SECRET"] {
            raw = configured
        } else {
            let path = ProcessInfo.processInfo.environment["BADAPPLE_SLICKS_KEY_PATH"]
                ?? "/var/lib/bad_apple/slicks.key"
            raw = try String(contentsOfFile: path, encoding: .utf8)
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let data = decodeHex(trimmed) ?? Data(trimmed.utf8)
        guard data.count >= 16 else {
            throw BadAppleIntentError("The Bad Apple SLICKS key is too short")
        }
        return data
    }

    static func nonce() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw BadAppleIntentError("Unable to create a SLICKS nonce")
        }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func nonceIsValid(_ nonce: String) -> Bool {
        decodeHex(nonce)?.count == 32
    }

    static func serverProof(
        secret: Data,
        timestampMs: UInt64,
        clientNonce: String,
        serverNonce: String
    ) -> String {
        sign(
            secret: secret,
            material: "BADAPPLE-SLICKS/\(badAppleSlicksVersion)|server|\(timestampMs)|\(clientNonce)|\(serverNonce)"
        )
    }

    static func clientProof(
        secret: Data,
        timestampMs: UInt64,
        clientNonce: String,
        serverNonce: String,
        prompt: String,
        maxNewTokens: Int
    ) -> String {
        let digest = SHA256.hash(data: Data(prompt.utf8))
        let promptHash = digest.map { String(format: "%02x", $0) }.joined()
        return sign(
            secret: secret,
            material: "BADAPPLE-SLICKS/\(badAppleSlicksVersion)|client|\(timestampMs)|\(clientNonce)|\(serverNonce)|\(maxNewTokens)|\(promptHash)"
        )
    }

    static func verify(
        proof: String,
        secret: Data,
        material: String
    ) -> Bool {
        guard let authenticationCode = decodeHex(proof) else { return false }
        return HMAC<SHA256>.isValidAuthenticationCode(
            authenticationCode,
            authenticating: Data(material.utf8),
            using: SymmetricKey(data: secret)
        )
    }

    private static func sign(secret: Data, material: String) -> String {
        let code = HMAC<SHA256>.authenticationCode(
            for: Data(material.utf8),
            using: SymmetricKey(data: secret)
        )
        return code.map { String(format: "%02x", $0) }.joined()
    }

    private static func decodeHex(_ value: String) -> Data? {
        guard !value.isEmpty, value.count.isMultiple(of: 2) else { return nil }
        var data = Data(capacity: value.count / 2)
        var index = value.startIndex
        while index < value.endIndex {
            let next = value.index(index, offsetBy: 2)
            guard let byte = UInt8(value[index..<next], radix: 16) else { return nil }
            data.append(byte)
            index = next
        }
        return data
    }
}

/// Authenticated client for the local SLICKS Unix-domain socket.
///
/// The menu-bar voice host and AppIntent share this implementation so nonce
/// generation, proofs, frame limits, and ordering checks cannot drift.
public struct BadAppleDaemonClient: Sendable {
    public static let shared = BadAppleDaemonClient()

    public init() {}

    public func generate(prompt: String, maxNewTokens: Int = 256) async throws -> String {
        try await generate(prompt: prompt, maxNewTokens: maxNewTokens, onToken: { _ in })
    }

    /// Reports authenticated token deltas off the main actor as they arrive.
    public func generate(
        prompt: String,
        maxNewTokens: Int = 256,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws -> String {
        print("[BadAppleClient] generate called: \(prompt)")
        return try await Task(priority: .userInitiated) {
            try self.generateSynchronously(
                prompt: prompt,
                maxNewTokens: maxNewTokens,
                onToken: onToken
            )
        }.value
    }

    private func generateSynchronously(
        prompt: String,
        maxNewTokens: Int,
        onToken: @escaping @Sendable (String) -> Void
    ) throws -> String {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw BadAppleIntentError("Bad Apple requires spoken text")
        }
        guard prompt.utf8.count <= 65_536 else {
            throw BadAppleIntentError("Bad Apple prompts are limited to 64 KiB")
        }
        guard (1...4_096).contains(maxNewTokens) else {
            throw BadAppleIntentError("Bad Apple token limits must be between 1 and 4096")
        }
        print("[BadAppleClient] start secret")
        let secret = try BadAppleSlicks.secret()
        print("[BadAppleClient] got secret")
        let clientNonce = try BadAppleSlicks.nonce()
        let timestampMs = UInt64(Date().timeIntervalSince1970 * 1_000)
        print("[BadAppleClient] create transport")
        let transport = try BadAppleLineTransport(socketPath: BadAppleSlicks.socketPath())
        print("[BadAppleClient] write hello")
        try transport.write(
            BadAppleHello(timestampMs: timestampMs, clientNonce: clientNonce)
        )

        print("[BadAppleClient] read challenge")
        let challenge = try transport.read()
        print("[BadAppleClient] got challenge \(challenge.type)")
        guard challenge.type == "challenge",
              challenge.version == badAppleSlicksVersion,
              let serverNonce = challenge.serverNonce,
              BadAppleSlicks.nonceIsValid(serverNonce),
              let serverProof = challenge.proof else {
            throw BadAppleIntentError(challenge.message ?? "Bad Apple returned an invalid SLICKS challenge")
        }
        let material = "BADAPPLE-SLICKS/\(badAppleSlicksVersion)|server|\(timestampMs)|\(clientNonce)|\(serverNonce)"
        guard BadAppleSlicks.verify(proof: serverProof, secret: secret, material: material) else {
            throw BadAppleIntentError("Bad Apple failed SLICKS server authentication")
        }
        print("[BadAppleClient] server proof ok")

        let clientProof = BadAppleSlicks.clientProof(
            secret: secret,
            timestampMs: timestampMs,
            clientNonce: clientNonce,
            serverNonce: serverNonce,
            prompt: prompt,
            maxNewTokens: maxNewTokens
        )
        print("[BadAppleClient] write execute")
        try transport.write(
            BadAppleExecute(
                timestampMs: timestampMs,
                clientNonce: clientNonce,
                serverNonce: serverNonce,
                prompt: prompt,
                maxNewTokens: maxNewTokens,
                proof: clientProof
            )
        )
        print("[BadAppleClient] wrote execute")

        var accepted = false
        var streamed = ""
        while true {
            print("[BadAppleClient] read frame")
            let frame = try transport.read()
            print("[BadAppleClient] frame type: \(frame.type)")
            switch frame.type {
            case "accepted":
                accepted = true
            case "token" where accepted:
                let delta = frame.text ?? ""
                streamed += delta
                if !delta.isEmpty {
                    onToken(delta)
                }
            case "done" where accepted:
                print("[BadAppleClient] done")
                return frame.text ?? streamed
            case "error":
                throw BadAppleIntentError(frame.message ?? "Bad Apple rejected the request")
            default:
                throw BadAppleIntentError("Bad Apple returned an out-of-order IPC frame")
            }
        }
    }
}

@available(macOS 26.0, *)
public struct BadAppleIntent: AppIntent {
    public static let title: LocalizedStringResource = "Execute Bad Apple"
    public static let description = IntentDescription(
        "Send spoken text to the local, air-gapped Bad Apple cognitive substrate."
    )
    public static let openAppWhenRun = false

    @Parameter(title: "Spoken Text")
    public var spokenText: [String]

    public init() {}

    public static var parameterSummary: some ParameterSummary {
        Summary("Execute Bad Apple with \(\.$spokenText)")
    }

    public func perform() async throws -> some IntentResult & ProvidesDialog {
        let prompt = spokenText.joined(separator: " ")
        let response = try await BadAppleDaemonClient.shared.generate(prompt: prompt)
        return .result(dialog: IntentDialog(stringLiteral: response))
    }
}

@available(macOS 26.0, *)
public struct BadAppleShortcuts: AppShortcutsProvider {
    public static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: BadAppleIntent(),
            phrases: [
                "Execute \(.applicationName)",
                "Ask \(.applicationName)"
            ],
            shortTitle: "Execute Bad Apple",
            systemImageName: "apple.logo"
        )
    }
}
