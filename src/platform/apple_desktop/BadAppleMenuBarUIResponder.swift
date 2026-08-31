import Foundation
import AppKit
import CommonCrypto

/// File-based responder that lets the Aqua helper ask the Bad Apple menu bar
/// process (which has Accessibility permission) to perform AXUIElement actions.
/// The menu bar writes responses to /var/run/badapple/ui_response_<id>.json.
///
/// Security: the request directory is created with 0o700 (owner-only) and every
/// request must carry a SLICKS v1 HMAC proof over its JSON body (excluding the
/// ``proof`` field).  The response file name is restricted to a safe character
/// set to prevent path traversal.
final class BadAppleMenuBarUIResponder: @unchecked Sendable {
    static let shared = BadAppleMenuBarUIResponder()

    private let requestDir = URL(fileURLWithPath: "/var/run/badapple")
    private let requestFile: URL
    private var lastSeenSize: Int64 = 0
    private var timer: Timer?
    private var trustPromptRequested = false

    /// Allowed characters for the request `id` — alphanumeric, dash, underscore.
    private let safeIdCharset = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

    init() {
        requestFile = requestDir.appendingPathComponent("ui_request.json")
    }

    func start() {
        stop()
        do {
            // Create with owner-only permissions to prevent unauthenticated
            // local processes from writing UI action requests.
            try FileManager.default.createDirectory(at: requestDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        } catch {
            print("UIResponder: could not create request dir: \(error)")
        }
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.poll()
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    private func showAccessibilityPermissionAlert() {
        let alert = NSAlert()
        alert.messageText = "Bad Apple needs Accessibility permission"
        alert.informativeText = """
        To control other apps — read the UI tree, click, type, and focus controls — Bad Apple needs to be granted Accessibility access.

        1. Open System Settings.
        2. Go to Privacy & Security → Accessibility.
        3. Remove any existing “Bad Apple” or “BadApple” entries.
        4. Click the + button.
        5. Select /Applications/Bad Apple.app.
        6. Toggle the switch on.
        7. Fully quit Bad Apple, then reopen it.

        Once this is done, the UI actor will work across rebuilds.
        """
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Open Accessibility")
        alert.addButton(withTitle: "OK")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
                NSWorkspace.shared.open(url)
            }
        }
    }

    private func writeResponse(id: String, result: [String: Any]) {
        // Validate the id to prevent path traversal (e.g. "../../etc/passwd").
        guard !id.isEmpty, id.unicodeScalars.allSatisfy({ safeIdCharset.contains($0) }), id.count <= 128 else {
            print("UIResponder: rejecting unsafe response id: \(id)")
            return
        }
        let responseFile = requestDir.appendingPathComponent("ui_response_\(id).json")
        do {
            let data = try JSONSerialization.data(withJSONObject: result, options: .sortedKeys)
            try data.write(to: responseFile, options: .atomic)
            // Owner-only response file.
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: responseFile.path)
        } catch {
            print("UIResponder: could not write response: \(error)")
        }
    }

    /// Load the SLICKS v1 shared secret for request authentication.
    /// SECURITY: Only loads from the pinned key file at /var/lib/bad_apple/slicks.key.
    /// Environment variable overrides are rejected to prevent secret injection.
    private func loadSlicksSecret() -> Data? {
        // Do NOT read from BADAPPLE_SLICKS_SECRET env var — an attacker who
        // can set environment variables (e.g. via launchctl setenv) would be
        // able to forge HMAC proofs and drive arbitrary UI actions.
        let keyPath = "/var/lib/bad_apple/slicks.key"
        guard let raw = try? String(contentsOfFile: keyPath, encoding: .utf8) else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.allSatisfy({ $0.isHexDigit }), trimmed.count >= 32 {
            return Data(hexString: trimmed)
        }
        return trimmed.data(using: .utf8)
    }

    /// Verify the SLICKS v1 HMAC proof on an incoming request body.
    /// Uses constant-time comparison to prevent timing attacks.
    private func verifyProof(body: [String: Any], secret: Data) -> Bool {
        guard let proof = body["proof"] as? String else { return false }
        var bodyWithoutProof = body
        bodyWithoutProof.removeValue(forKey: "proof")
        guard let material = try? JSONSerialization.data(withJSONObject: bodyWithoutProof, options: [.sortedKeys, .withoutEscapingSlashes]) else {
            return false
        }
        var hmac = CCHmacContext()
        CCHmacInit(&hmac, CCHmacAlgorithm(kCCHmacAlgSHA256), secret.withUnsafeBytes { $0.baseAddress }, secret.count)
        material.withUnsafeBytes { ptr in
            if let base = ptr.baseAddress {
                CCHmacUpdate(&hmac, base, material.count)
            }
        }
        var mac = [UInt8](repeating: 0, count: 32)
        CCHmacFinal(&hmac, &mac)
        // Constant-time comparison: decode proof hex to bytes and XOR-compare.
        guard let proofBytes = Data(hexString: proof.lowercased()) else {
            return false
        }
        guard proofBytes.count == mac.count else {
            return false
        }
        var diff: UInt8 = 0
        for i in 0..<mac.count {
            diff |= mac[i] ^ proofBytes[i]
        }
        return diff == 0
    }

    private func poll() {
        guard FileManager.default.fileExists(atPath: requestFile.path) else { return }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: requestFile.path),
              let size = attrs[.size] as? Int64,
              size > 0,
              size != lastSeenSize else { return }
        lastSeenSize = size

        guard let data = try? Data(contentsOf: requestFile),
              let req = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = req["id"] as? String,
              let action = req["action"] as? String else { return }

        // Authenticate the request with a SLICKS v1 HMAC proof.
        guard let secret = loadSlicksSecret() else {
            writeResponse(id: id, result: ["ok": false, "error": "UIResponder has no SLICKS secret configured"])
            try? Data().write(to: requestFile, options: .atomic)
            return
        }
        guard verifyProof(body: req, secret: secret) else {
            writeResponse(id: id, result: ["ok": false, "error": "unauthenticated: invalid or missing proof"])
            try? Data().write(to: requestFile, options: .atomic)
            return
        }

        let access = BadAppleUIAccess.shared
        if !access.isTrusted() {
            if !trustPromptRequested {
                trustPromptRequested = true
                access.requestTrustPrompt()
                showAccessibilityPermissionAlert()
            }
            writeResponse(id: id, result: ["ok": false, "error": "Bad Apple is not trusted for Accessibility. Grant it in System Settings > Privacy & Security > Accessibility and try again."])
            // Clear the request so it is not processed twice.
            try? Data().write(to: requestFile, options: .atomic)
            return
        }

        var result: [String: Any]
        switch action {
        case "info":
            let root = access.runInfo()
            do {
                let data = try JSONEncoder().encode(root)
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    result = ["ok": true, "data": obj]
                } else {
                    result = ["ok": false, "error": "could not encode UI info"]
                }
            } catch {
                result = ["ok": false, "error": "encode error: \(error)"]
            }
        case "click":
            result = access.runClick(target: req["target"] as? String ?? "", role: req["role"] as? String ?? "")
        case "type":
            result = access.runType(target: req["target"] as? String ?? "", text: req["text"] as? String ?? "")
        case "focus":
            result = access.runFocus(target: req["target"] as? String ?? "")
        default:
            result = ["ok": false, "error": "unknown action \(action)"]
        }

        writeResponse(id: id, result: result)
        // Clear the request so it is not processed twice.
        do {
            try Data().write(to: requestFile, options: .atomic)
        } catch {
            print("UIResponder: could not clear request: \(error)")
        }
    }
}

// MARK: - Data hex helper

private extension Data {
    init?(hexString: String) {
        let hex = hexString.lowercased()
        guard hex.count % 2 == 0 else { return nil }
        var data = Data(capacity: hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
            data.append(byte)
            index = next
        }
        self = data
    }
}
