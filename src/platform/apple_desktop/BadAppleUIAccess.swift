import Foundation
import ApplicationServices
import AppKit

struct UIElement: Encodable {
    let role: String
    let title: String
    let description: String
    let value: String
    let position: [Int]
    let size: [Int]
    let enabled: Bool
    let focused: Bool
    let actions: [String]
    let childrenCount: Int
    let children: [UIElement]?
}

struct UIRoot: Encodable {
    let app: String
    let bundle: String
    let pid: Int
    let window: String
    let elements: [UIElement]
    let error: String?
}

final class BadAppleUIAccess {
    static let shared = BadAppleUIAccess()

    func isTrusted() -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        return AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    func requestTrustPrompt() {
        // The prompt only appears for a foreground app; force activation.
        NSApplication.shared.activate(ignoringOtherApps: true)
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    private func getString(_ element: AXUIElement, _ attr: String) -> String {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr as CFString, &value)
        guard err == .success, let v = value else { return "" }
        if let s = v as? String { return s }
        if let s = v as? NSString { return s as String }
        return ""
    }

    private func getBool(_ element: AXUIElement, _ attr: String) -> Bool {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr as CFString, &value)
        guard err == .success, let v = value else { return false }
        if let b = v as? Bool { return b }
        if let n = v as? NSNumber { return n.boolValue }
        return false
    }

    private func getPoint(_ element: AXUIElement) -> CGPoint? {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, kAXPositionAttribute as CFString, &value)
        guard err == .success, let axValue = value else { return nil }
        var point = CGPoint.zero
        if AXValueGetValue(axValue as! AXValue, .cgPoint, &point) {
            return point
        }
        return nil
    }

    private func getSize(_ element: AXUIElement) -> CGSize? {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, kAXSizeAttribute as CFString, &value)
        guard err == .success, let axValue = value else { return nil }
        var size = CGSize.zero
        if AXValueGetValue(axValue as! AXValue, .cgSize, &size) {
            return size
        }
        return nil
    }

    private func getActions(_ element: AXUIElement) -> [String] {
        var names: CFArray?
        let err = AXUIElementCopyActionNames(element, &names)
        guard err == .success, let list = names as? [String] else { return [] }
        return list
    }

    private func getChildren(_ element: AXUIElement) -> [AXUIElement] {
        var value: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &value)
        guard err == .success, let v = value, let array = v as? [AXUIElement] else { return [] }
        return array
    }

    private func dumpElement(_ element: AXUIElement, depth: Int, limit: Int, count: inout Int) -> UIElement? {
        guard count < limit, depth < 8 else { return nil }
        count += 1

        let role = getString(element, kAXRoleAttribute)
        let title = getString(element, kAXTitleAttribute)
        let description = getString(element, kAXDescriptionAttribute)
        let value = getString(element, kAXValueAttribute)
        let position = getPoint(element)
        let size = getSize(element)
        let enabled = getBool(element, kAXEnabledAttribute)
        let focused = getBool(element, kAXFocusedAttribute)
        let actions = getActions(element)

        var children: [UIElement] = []
        if depth < 7 {
            for child in getChildren(element) {
                if let c = dumpElement(child, depth: depth + 1, limit: limit, count: &count) {
                    children.append(c)
                }
            }
        }

        return UIElement(
            role: role,
            title: title,
            description: description,
            value: value,
            position: position.map { [Int($0.x), Int($0.y)] } ?? [],
            size: size.map { [Int($0.width), Int($0.height)] } ?? [],
            enabled: enabled,
            focused: focused,
            actions: actions,
            childrenCount: children.count,
            children: children.isEmpty ? nil : children
        )
    }

    private func frontmostApp() -> (AXUIElement, NSRunningApplication)? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = app.processIdentifier
        let axApp = AXUIElementCreateApplication(pid)
        return (axApp, app)
    }

    private func targetWindow(for app: AXUIElement) -> AXUIElement? {
        var value: CFTypeRef?
        var err = AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute as CFString, &value)
        if err == .success, let v = value, CFGetTypeID(v) == AXUIElementGetTypeID() { return (v as! AXUIElement) }
        err = AXUIElementCopyAttributeValue(app, kAXMainWindowAttribute as CFString, &value)
        if err == .success, let v = value, CFGetTypeID(v) == AXUIElementGetTypeID() { return (v as! AXUIElement) }
        return nil
    }

    private func findElement(_ element: AXUIElement, byTitle title: String, byRole: String, depth: Int) -> AXUIElement? {
        guard depth < 8 else { return nil }
        let elTitle = getString(element, kAXTitleAttribute)
        let elRole = getString(element, kAXRoleAttribute)
        if (title.isEmpty || elTitle == title) && (byRole.isEmpty || elRole == byRole) {
            return element
        }
        for child in getChildren(element) {
            if let found = findElement(child, byTitle: title, byRole: byRole, depth: depth + 1) {
                return found
            }
        }
        return nil
    }

    func runInfo() -> UIRoot {
        guard isTrusted() else {
            return UIRoot(app: "", bundle: "", pid: 0, window: "", elements: [], error: "Bad Apple is not trusted for Accessibility. Grant it in System Settings > Privacy & Security > Accessibility and try again.")
        }
        guard let (app, nsApp) = frontmostApp() else {
            return UIRoot(app: "", bundle: "", pid: 0, window: "", elements: [], error: "no frontmost app")
        }
        guard let window = targetWindow(for: app) else {
            return UIRoot(app: nsApp.localizedName ?? "", bundle: nsApp.bundleIdentifier ?? "", pid: Int(nsApp.processIdentifier), window: "", elements: [], error: "no focused window")
        }

        var count = 0
        var children: [UIElement] = []
        for child in getChildren(window) {
            if let c = dumpElement(child, depth: 0, limit: 300, count: &count) {
                children.append(c)
            }
        }

        return UIRoot(
            app: nsApp.localizedName ?? "",
            bundle: nsApp.bundleIdentifier ?? "",
            pid: Int(nsApp.processIdentifier),
            window: getString(window, kAXTitleAttribute),
            elements: children,
            error: nil
        )
    }

    func runClick(target: String, role: String) -> [String: Any] {
        guard isTrusted() else { return ["ok": false, "error": "Bad Apple is not trusted for Accessibility"] }
        guard let (app, _) = frontmostApp() else { return ["ok": false, "error": "no frontmost app"] }
        guard let window = targetWindow(for: app) else { return ["ok": false, "error": "no focused window"] }
        guard let element = findElement(window, byTitle: target, byRole: role, depth: 0) else {
            return ["ok": false, "error": "element not found"]
        }
        let actions = getActions(element)
        let action = actions.first { ["AXPress", "AXConfirm"].contains($0) } ?? "AXPress"
        let err = AXUIElementPerformAction(element, action as CFString)
        guard err == .success else {
            return ["ok": false, "error": "perform action failed (\(err))"]
        }
        return ["ok": true, "result": "\(action) on \"\(getString(element, kAXTitleAttribute))\""]
    }

    func runType(target: String, text: String) -> [String: Any] {
        guard isTrusted() else { return ["ok": false, "error": "Bad Apple is not trusted for Accessibility"] }
        guard let (app, _) = frontmostApp() else { return ["ok": false, "error": "no frontmost app"] }
        guard let window = targetWindow(for: app) else { return ["ok": false, "error": "no focused window"] }
        guard let element = findElement(window, byTitle: target, byRole: "", depth: 0) else {
            return ["ok": false, "error": "element not found"]
        }
        let err = AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFString)
        guard err == .success else {
            return ["ok": false, "error": "set value failed (\(err))"]
        }
        return ["ok": true, "result": "typed \(text.count) chars"]
    }

    func runFocus(target: String) -> [String: Any] {
        guard isTrusted() else { return ["ok": false, "error": "Bad Apple is not trusted for Accessibility"] }
        guard let (app, _) = frontmostApp() else { return ["ok": false, "error": "no frontmost app"] }
        guard let window = targetWindow(for: app) else { return ["ok": false, "error": "no focused window"] }
        guard let element = findElement(window, byTitle: target, byRole: "", depth: 0) else {
            return ["ok": false, "error": "element not found"]
        }
        let err = AXUIElementSetAttributeValue(element, kAXFocusedAttribute as CFString, kCFBooleanTrue)
        guard err == .success else { return ["ok": false, "error": "focus failed (\(err))"] }
        return ["ok": true, "result": "focused \(target)"]
    }
}
