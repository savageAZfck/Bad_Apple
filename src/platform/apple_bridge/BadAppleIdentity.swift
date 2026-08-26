import CryptoKit
import Foundation
import LocalAuthentication

private enum IdentityError: Error, CustomStringConvertible {
    case invalidCommand
    case invalidInput
    case secureEnclaveUnavailable
    case biometricUnavailable
    case biometricDenied

    var description: String {
        switch self {
        case .invalidCommand: return "invalid command"
        case .invalidInput: return "invalid input"
        case .secureEnclaveUnavailable: return "Secure Enclave is unavailable"
        case .biometricUnavailable: return "biometric authentication is unavailable"
        case .biometricDenied: return "biometric authentication was denied"
        }
    }
}

private func keyURL() throws -> URL {
    if let configured = ProcessInfo.processInfo.environment["BADAPPLE_IDENTITY_BLOB"] {
        return URL(fileURLWithPath: configured)
    }
    let base = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/BadApple", isDirectory: true)
    try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
    return base.appendingPathComponent("identity.sekey")
}

private func existingKey() throws -> SecureEnclave.P256.Signing.PrivateKey? {
    let url = try keyURL()
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    return try SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: Data(contentsOf: url))
}

private func ensureKey() throws -> SecureEnclave.P256.Signing.PrivateKey {
    guard SecureEnclave.isAvailable else { throw IdentityError.secureEnclaveUnavailable }
    if let key = try existingKey() { return key }
    let key = try SecureEnclave.P256.Signing.PrivateKey()
    let url = try keyURL()
    try key.dataRepresentation.write(to: url, options: [.atomic, .completeFileProtection])
    try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    return key
}

private func publicKeyData(_ privateKey: SecureEnclave.P256.Signing.PrivateKey) -> Data {
    privateKey.publicKey.x963Representation
}

private func biometricGate(reason: String) throws {
    let context = LAContext()
    var authError: NSError?
    guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &authError) else {
        throw IdentityError.biometricUnavailable
    }
    let semaphore = DispatchSemaphore(value: 0)
    var granted = false
    context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { success, _ in
        granted = success
        semaphore.signal()
    }
    semaphore.wait()
    if !granted { throw IdentityError.biometricDenied }
}

private func run() throws {
    guard CommandLine.arguments.count >= 2 else { throw IdentityError.invalidCommand }
    switch CommandLine.arguments[1] {
    case "ensure":
        print(publicKeyData(try ensureKey()).base64EncodedString())
    case "public-key":
        guard let key = try existingKey() else { throw IdentityError.invalidInput }
        print(publicKeyData(key).base64EncodedString())
    case "sign":
        guard CommandLine.arguments.count == 3,
              let message = Data(base64Encoded: CommandLine.arguments[2]) else { throw IdentityError.invalidInput }
        let signature = try ensureKey().signature(for: message)
        print(signature.derRepresentation.base64EncodedString())
    case "biometric-gate":
        let reason = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "Approve a sensitive Bad Apple action"
        try biometricGate(reason: reason)
        print("approved")
    case "status":
        if let key = try existingKey() {
            let fingerprint = SHA256.hash(data: publicKeyData(key)).map { String(format: "%02x", $0) }.joined()
            print("secure-enclave:\(fingerprint)")
        } else {
            print(SecureEnclave.isAvailable ? "missing" : "unavailable")
        }
    default:
        throw IdentityError.invalidCommand
    }
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
