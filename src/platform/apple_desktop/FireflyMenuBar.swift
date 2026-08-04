import AppKit
import Foundation
import Darwin

// MARK: - C function signatures from firefly_core.h

private typealias FireflyInitFn = @convention(c) (UnsafePointer<CChar>?) -> OpaquePointer?
private typealias FireflyFreeFn = @convention(c) (OpaquePointer?) -> Void
private typealias FireflyGetActivePursuitsFn = @convention(c) (OpaquePointer?) -> UnsafeMutablePointer<CChar>?
private typealias FireflyPushPursuitFn = @convention(c) (OpaquePointer?, UnsafePointer<CChar>?) -> Bool
private typealias FireflyGetAppleLatencyUsFn = @convention(c) () -> UInt64
private typealias FireflyGenerateTextFn = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
private typealias FireflyFreeStringFn = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void

// MARK: - Dynamic loader for libsapient_soul.dylib

final class FireflyFFI {
    static let shared = FireflyFFI()

    private var handle: UnsafeMutableRawPointer?
    private(set) var context: OpaquePointer?

    private var firefly_init: FireflyInitFn?
    private var firefly_free: FireflyFreeFn?
    private var firefly_get_active_pursuits: FireflyGetActivePursuitsFn?
    private var firefly_push_pursuit: FireflyPushPursuitFn?
    private var firefly_get_apple_latency_us: FireflyGetAppleLatencyUsFn?
    private var firefly_generate_text: FireflyGenerateTextFn?
    private var firefly_free_string: FireflyFreeStringFn?

    private var lastError: String?

    var isLoaded: Bool { handle != nil && context != nil }

    func load() {
        let bundleDir = (Bundle.main.bundlePath as NSString).deletingLastPathComponent
        let searchPaths = [
            bundleDir + "/libsapient_soul.dylib",
            bundleDir + "/../libsapient_soul.dylib",
            "libsapient_soul.dylib",
            "./libsapient_soul.dylib",
            "../libsapient_soul.dylib",
            "target/release/libsapient_soul.dylib",
            "target/debug/libsapient_soul.dylib",
            "/usr/local/lib/libsapient_soul.dylib",
        ]

        for path in searchPaths {
            if let h = dlopen(path, RTLD_LAZY) {
                handle = h
                break
            }
        }

        guard let h = handle else {
            lastError = "libsapient_soul.dylib not found in any search path"
            return
        }

        firefly_init = unsafeBitCast(dlsym(h, "firefly_init"), to: FireflyInitFn.self)
        firefly_free = unsafeBitCast(dlsym(h, "firefly_free"), to: FireflyFreeFn.self)
        firefly_get_active_pursuits = unsafeBitCast(dlsym(h, "firefly_get_active_pursuits"), to: FireflyGetActivePursuitsFn.self)
        firefly_push_pursuit = unsafeBitCast(dlsym(h, "firefly_push_pursuit"), to: FireflyPushPursuitFn.self)
        firefly_get_apple_latency_us = unsafeBitCast(dlsym(h, "firefly_get_apple_latency_us"), to: FireflyGetAppleLatencyUsFn.self)
        firefly_generate_text = unsafeBitCast(dlsym(h, "firefly_generate_text"), to: FireflyGenerateTextFn.self)
        firefly_free_string = unsafeBitCast(dlsym(h, "firefly_free_string"), to: FireflyFreeStringFn.self)

        context = firefly_init?(nil)
        if context == nil {
            lastError = "firefly_init() returned nil"
        }
    }

    func freeString(_ ptr: UnsafeMutablePointer<CChar>?) {
        guard let ptr = ptr, let freeFn = firefly_free_string else { return }
        freeFn(ptr)
    }

    func activePursuits() -> [String] {
        guard let ctx = context, let fn = firefly_get_active_pursuits else { return [] }
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
        guard let ctx = context, let fn = firefly_push_pursuit else { return false }
        return text.withCString { cstr in
            fn(ctx, cstr)
        }
    }

    func appleLatencyUs() -> UInt64 {
        return firefly_get_apple_latency_us?() ?? 0
    }

    func generateText(_ prompt: String) -> String? {
        guard let fn = firefly_generate_text else { return nil }
        return prompt.withCString { cstr in
            guard let raw = fn(cstr) else { return nil }
            defer { freeString(raw) }
            return String(cString: raw, encoding: .utf8)
        }
    }

    deinit {
        if let ctx = context, let freeFn = firefly_free {
            freeFn(ctx)
        }
        if let h = handle {
            dlclose(h)
        }
    }
}

// MARK: - Menu-bar application

@main
struct FireflyMenuBarApp {
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
        FireflyFFI.shared.load()

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "🔥"

        menu = NSMenu(title: "Firefly")
        statusItem?.menu = menu

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.rebuildMenu()
        }
        rebuildMenu()
    }

    func rebuildMenu() {
        guard let menu = menu else { return }
        menu.removeAllItems()

        let header = NSMenuItem(title: "Firefly Menu Bar", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        menu.addItem(NSMenuItem.separator())

        if !FireflyFFI.shared.isLoaded {
            let err = NSMenuItem(title: "⚠️ libsapient_soul not loaded", action: nil, keyEquivalent: "")
            err.isEnabled = false
            menu.addItem(err)
            menu.addItem(NSMenuItem(title: "Quit", action: #selector(terminate), keyEquivalent: "q"))
            return
        }

        let latencyUs = FireflyFFI.shared.appleLatencyUs()
        let latencyMs = Double(latencyUs) / 1000.0
        let latencyItem = NSMenuItem(
            title: String(format: "Apple Latency: %.2f ms", latencyMs),
            action: nil,
            keyEquivalent: ""
        )
        latencyItem.isEnabled = false
        menu.addItem(latencyItem)

        let pursuits = FireflyFFI.shared.activePursuits()
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
        guard FireflyFFI.shared.isLoaded else { return }
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
                _ = FireflyFFI.shared.pushPursuit(text)
                rebuildMenu()
            }
        }
    }

    @objc func generateReflection() {
        guard FireflyFFI.shared.isLoaded else { return }
        let prompt = "Reflect on the current state of a local-first AGI runtime. Write one concise sentence about what to focus on next."
        _ = FireflyFFI.shared.generateText(prompt)
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
