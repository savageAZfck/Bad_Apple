import AppKit
import ApplicationServices
import Foundation

@main
struct BadAppleAmbient {
    static func main() {
        let workspace = NSWorkspace.shared
        guard let app = workspace.frontmostApplication else {
            print("{\"app\": \"\", \"window\": \"\"}")
            return
        }

        let appName = app.localizedName ?? ""

        // Try to read the focused window title via the Accessibility API.
        var windowTitle = ""
        if let axApp = AXUIElementCreateApplication(app.processIdentifier) as AXUIElement? {
            var focusedWindow: AnyObject?
            if AXUIElementCopyAttributeValue(axApp, kAXFocusedWindowAttribute as CFString, &focusedWindow) == .success,
               let v = focusedWindow, CFGetTypeID(v) == AXUIElementGetTypeID() {
                let window = v as! AXUIElement
                var titleValue: AnyObject?
                if AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &titleValue) == .success {
                    windowTitle = titleValue as? String ?? ""
                }
            }
        }

        let encoder = JSONEncoder()
        let payload: [String: String] = ["app": appName, "window": windowTitle]
        if let data = try? encoder.encode(payload), let json = String(data: data, encoding: .utf8) {
            print(json)
        } else {
            print("{\"app\": \"\", \"window\": \"\"}")
        }
    }
}
