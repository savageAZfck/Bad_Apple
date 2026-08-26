import Foundation
import AppKit

/// File-based responder that lets the Aqua helper ask the Bad Apple menu bar
/// process (which has Accessibility permission) to perform AXUIElement actions.
/// The menu bar writes responses to /var/run/badapple/ui_response_<id>.json.
final class BadAppleMenuBarUIResponder: @unchecked Sendable {
    static let shared = BadAppleMenuBarUIResponder()

    private let requestDir = URL(fileURLWithPath: "/var/run/badapple")
    private let requestFile: URL
    private var lastSeenSize: Int64 = 0
    private var timer: Timer?
    private var trustPromptRequested = false

    init() {
        requestFile = requestDir.appendingPathComponent("ui_request.json")
    }

    func start() {
        stop()
        do {
            try FileManager.default.createDirectory(at: requestDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o777])
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
        let responseFile = requestDir.appendingPathComponent("ui_response_\(id).json")
        do {
            let data = try JSONSerialization.data(withJSONObject: result, options: .sortedKeys)
            try data.write(to: responseFile, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: responseFile.path)
        } catch {
            print("UIResponder: could not write response: \(error)")
        }
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
