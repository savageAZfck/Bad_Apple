import AppKit
import BadAppleBridge
import Foundation
import Darwin

// MARK: - C function signatures from bad_apple_core.h

private typealias BadAppleInitFn = @convention(c) (UnsafePointer<CChar>?) -> OpaquePointer?
private typealias BadAppleFreeFn = @convention(c) (OpaquePointer?) -> Void
private typealias BadAppleGetActivePursuitsFn = @convention(c) (OpaquePointer?) -> UnsafeMutablePointer<CChar>?
private typealias BadApplePushPursuitFn = @convention(c) (OpaquePointer?, UnsafePointer<CChar>?) -> Bool
private typealias BadAppleGetAppleLatencyUsFn = @convention(c) () -> UInt64
private typealias BadAppleGenerateTextFn = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
private typealias BadAppleFreeStringFn = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
private typealias InitBadAppleBridgeFn = @convention(c) () -> Void

// MARK: - Dynamic loader for libbad_apple.dylib

final class BadAppleFFI {
    static let shared = BadAppleFFI()

    private var handle: UnsafeMutableRawPointer?
    private(set) var context: OpaquePointer?

    private var bad_apple_init: BadAppleInitFn?
    private var bad_apple_free: BadAppleFreeFn?
    private var bad_apple_get_active_pursuits: BadAppleGetActivePursuitsFn?
    private var bad_apple_push_pursuit: BadApplePushPursuitFn?
    private var bad_apple_get_apple_latency_us: BadAppleGetAppleLatencyUsFn?
    private var bad_apple_generate_text: BadAppleGenerateTextFn?
    private var bad_apple_free_string: BadAppleFreeStringFn?

    private var lastError: String?

    var isLoaded: Bool { handle != nil && context != nil }

    func load() {
        // First load the Apple Intelligence Siri bridge so the lib can answer prompts.
        let bridgeSearchPaths = [
            "libBadAppleBridge.dylib",
            "./libBadAppleBridge.dylib",
            "../libBadAppleBridge.dylib",
            "target/release/libBadAppleBridge.dylib",
            "target/debug/libBadAppleBridge.dylib",
        ]
        var bridgeHandle: UnsafeMutableRawPointer?
        for path in bridgeSearchPaths {
            if let h = dlopen(path, RTLD_LAZY) {
                bridgeHandle = h
                break
            }
        }
        if let h = bridgeHandle, let sym = dlsym(h, "init_bad_apple_bridge") {
            let initBridge = unsafeBitCast(sym, to: InitBadAppleBridgeFn.self)
            initBridge()
        }

        let bundleDir = (Bundle.main.bundlePath as NSString).deletingLastPathComponent
        let searchPaths = [
            bundleDir + "/libbad_apple.dylib",
            bundleDir + "/../libbad_apple.dylib",
            "libbad_apple.dylib",
            "./libbad_apple.dylib",
            "../libbad_apple.dylib",
            "target/release/libbad_apple.dylib",
            "target/debug/libbad_apple.dylib",
            "/usr/local/lib/libbad_apple.dylib",
        ]

        for path in searchPaths {
            if let h = dlopen(path, RTLD_LAZY) {
                handle = h
                break
            }
        }

        guard let h = handle else {
            lastError = "libbad_apple.dylib not found in any search path"
            return
        }

        bad_apple_init = unsafeBitCast(dlsym(h, "bad_apple_init"), to: BadAppleInitFn.self)
        bad_apple_free = unsafeBitCast(dlsym(h, "bad_apple_free"), to: BadAppleFreeFn.self)
        bad_apple_get_active_pursuits = unsafeBitCast(dlsym(h, "bad_apple_get_active_pursuits"), to: BadAppleGetActivePursuitsFn.self)
        bad_apple_push_pursuit = unsafeBitCast(dlsym(h, "bad_apple_push_pursuit"), to: BadApplePushPursuitFn.self)
        bad_apple_get_apple_latency_us = unsafeBitCast(dlsym(h, "bad_apple_get_apple_latency_us"), to: BadAppleGetAppleLatencyUsFn.self)
        bad_apple_generate_text = unsafeBitCast(dlsym(h, "bad_apple_generate_text"), to: BadAppleGenerateTextFn.self)
        bad_apple_free_string = unsafeBitCast(dlsym(h, "bad_apple_free_string"), to: BadAppleFreeStringFn.self)

        context = bad_apple_init?(nil)
        if context == nil {
            lastError = "bad_apple_init() returned nil"
        }
    }

    func freeString(_ ptr: UnsafeMutablePointer<CChar>?) {
        guard let ptr = ptr, let freeFn = bad_apple_free_string else { return }
        freeFn(ptr)
    }

    func activePursuits() -> [String] {
        guard let ctx = context, let fn = bad_apple_get_active_pursuits else { return [] }
        guard let raw = fn(ctx) else { return [] }
        defer { freeString(raw) }
        guard let cstr = String(cString: raw, encoding: .utf8) else { return [] }
        if let data = cstr.data(using: .utf8),
           let arr = try? JSONSerialization.jsonObject(with: data, options: []) as? [String] {
            return arr
        }
        return []
    }

    @discardableResult
    func pushPursuit(_ text: String) -> Bool {
        guard let ctx = context, let fn = bad_apple_push_pursuit else { return false }
        return text.withCString { cstr in
            fn(ctx, cstr)
        }
    }

    func appleLatencyUs() -> UInt64 {
        return bad_apple_get_apple_latency_us?() ?? 0
    }

    func generateText(_ prompt: String) -> String? {
        guard let fn = bad_apple_generate_text else { return nil }
        return prompt.withCString { cstr in
            guard let raw = fn(cstr) else { return nil }
            defer { freeString(raw) }
            return String(cString: raw, encoding: .utf8)
        }
    }

    deinit {
        if let ctx = context, let freeFn = bad_apple_free {
            freeFn(ctx)
        }
        if let h = handle {
            dlclose(h)
        }
    }
}

// MARK: - Menu-bar application

@main
struct BadAppleMenuBarApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var menu: NSMenu?
    var timer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        BadAppleFFI.shared.load()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "🔥"

        menu = NSMenu(title: "Bad Apple")
        statusItem?.menu = menu

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.rebuildMenu()
        }
        rebuildMenu()
    }

    func rebuildMenu() {
        guard let menu = menu else { return }
        menu.removeAllItems()

        let header = NSMenuItem(title: "Bad Apple Menu Bar", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        menu.addItem(NSMenuItem.separator())

        if !BadAppleFFI.shared.isLoaded {
            let err = NSMenuItem(title: "⚠️ libbad_apple not loaded", action: nil, keyEquivalent: "")
            err.isEnabled = false
            menu.addItem(err)
            menu.addItem(NSMenuItem(title: "Quit", action: #selector(terminate), keyEquivalent: "q"))
            return
        }

        let latencyUs = BadAppleFFI.shared.appleLatencyUs()
        let latencyMs = Double(latencyUs) / 1000.0
        let latencyItem = NSMenuItem(
            title: String(format: "Apple Latency: %.2f ms", latencyMs),
            action: nil,
            keyEquivalent: ""
        )
        latencyItem.isEnabled = false
        menu.addItem(latencyItem)

        let pursuits = BadAppleFFI.shared.activePursuits()
        if pursuits.isEmpty {
            let empty = NSMenuItem(title: "No active pursuits", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
        } else {
            let subMenu = NSMenu(title: "Active Pursuits")
            for p in pursuits.suffix(8) {
                let item = NSMenuItem(title: p.truncated(to: 70), action: nil, keyEquivalent: "")
                item.isEnabled = false
                item.toolTip = p
                subMenu.addItem(item)
            }
            let parent = NSMenuItem(title: "Active Pursuits", action: nil, keyEquivalent: "")
            parent.submenu = subMenu
            menu.addItem(parent)
        }

        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Push Pursuit...", action: #selector(pushPursuit), keyEquivalent: "p"))
        menu.addItem(NSMenuItem(title: "Generate Reflection", action: #selector(generateReflection), keyEquivalent: "g"))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(terminate), keyEquivalent: "q"))
    }

    @objc func pushPursuit() {
        guard BadAppleFFI.shared.isLoaded else { return }
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "Push a new active pursuit"
        alert.informativeText = "Enter the pursuit text to inject across the FFI boundary:"
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 340, height: 24))
        alert.accessoryView = textField
        alert.addButton(withTitle: "Push")
        alert.addButton(withTitle: "Cancel")

        if alert.runModal() == .alertFirstButtonReturn {
            let text = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                _ = BadAppleFFI.shared.pushPursuit(text)
                rebuildMenu()
            }
        }
    }

    @objc func generateReflection() {
        guard BadAppleFFI.shared.isLoaded else { return }
        let prompt = "Reflect on the current state of a local-first AGI runtime. Write one concise sentence about what to focus on next."
        _ = BadAppleFFI.shared.generateText(prompt)
        rebuildMenu()
    }

    @objc func terminate() {
        NSApp.terminate(nil)
    }
}

extension String {
    func truncated(to length: Int) -> String {
        if self.count > length {
            return String(self.prefix(length)) + "..."
        }
        return self
    }
}
