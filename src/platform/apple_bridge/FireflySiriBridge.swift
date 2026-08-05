import Foundation
import UserNotifications

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Imported Rust registration primitive.  This symbol is exposed by the
/// `sapient_soul` library through `firefly_core.h`.
@_silgen_name("register_apple_intelligence_oracle")
func registerAppleIntelligenceOracle(
    _ callback: @convention(c) (UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?
)

/// On Apple Silicon / Apple-Intelligence-capable Macs, generate a response
/// using the on-device `SystemLanguageModel`.  Returns `nil` on unsupported
/// OS versions, missing entitlements, or model failure.
func buildAppleIntelligenceResponse(for prompt: String) async -> String? {
#if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        do {
            let model = SystemLanguageModel.default
            guard model.availability == .available else {
                return nil
            }
            let session = LanguageModelSession(model: model)
            let response = try await session.respond(to: Prompt(prompt))
            return response.content
        } catch {
            return nil
        }
    }
#endif
    return nil
}

/// C-callable entry point that the Rust runtime invokes for each inference
/// request.
///
/// Ownership contract:
/// - `prompt` is a borrowed, null-terminated UTF-8 C string.  The bridge does
///   not take ownership and must not free it.
/// - The returned pointer (if non-nil) is a freshly `strdup`-allocated C string
///   owned by the Rust caller.  The Rust side must free it exactly once, either
///   through `free()` or through the `free_swift_string` companion function.
@_cdecl("firefly_apple_intelligence_callback")
public func fireflyAppleIntelligenceCallback(
    _ prompt: UnsafePointer<CChar>
) -> UnsafeMutablePointer<CChar>? {
    let promptString = String(cString: prompt)
    let semaphore = DispatchSemaphore(value: 0)
    var result: String?

    Task {
        result = await buildAppleIntelligenceResponse(for: promptString)
        semaphore.signal()
    }

    semaphore.wait()

    guard let text = result, !text.isEmpty,
          let cString = text.cString(using: .utf8) else {
        return nil
    }
    return strdup(cString)
}

/// C-callable deallocator for strings returned by the bridge.
///
/// `ptr` must be a pointer previously returned by `firefly_apple_intelligence_callback`,
/// or `nil`.  Calling `free()` directly is also safe because the bridge uses the
/// C library's `strdup`, but this hook guarantees the same allocator is used on
/// both sides of the FFI boundary.
@_cdecl("free_swift_string")
public func freeSwiftString(_ ptr: UnsafeMutablePointer<CChar>?) {
    guard let ptr = ptr else { return }
    free(ptr)
}

/// Returns true only when the process is running inside a proper `.app`
/// bundle.  `UNUserNotificationCenter` requires a bundle identifier and will
/// throw `NSInternalInconsistencyException` if called from a bare executable.
private func isRunningInAppBundle() -> Bool {
    Bundle.main.bundleURL.pathExtension == "app"
}

/// Request authorization to show local user notifications.
private func requestNotificationAuthorization() {
    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, error in
        if let error = error {
            print("🍎 Notification authorization error: \(error)")
        } else {
            print("🍎 Notification authorization granted: \(granted)")
        }
    }
}

/// C-callable initialization entry point.  The Rust loader calls this after
/// `dlopen`ing the bridge dylib, and the bridge in turn registers its
/// callback with the `register_apple_intelligence_oracle` primitive.
@_cdecl("init_firefly_siri_bridge")
public func initFireflySiriBridge() {
    if isRunningInAppBundle() {
        requestNotificationAuthorization()
    }
    registerAppleIntelligenceOracle(fireflyAppleIntelligenceCallback)
}

/// C-callable desktop notification dispatch.  Pushes a native macOS user
/// alert immediately.  Requires prior notification authorization.
@_cdecl("dispatch_desktop_notification")
public func dispatchDesktopNotification(
    _ title: UnsafePointer<CChar>,
    _ body: UnsafePointer<CChar>
) {
    guard isRunningInAppBundle() else {
        print("🍎 Desktop notifications require an app bundle; skipping.")
        return
    }
    let titleString = String(cString: title)
    let bodyString = String(cString: body)
    let content = UNMutableNotificationContent()
    content.title = titleString
    content.body = bodyString
    content.sound = .default
    let request = UNNotificationRequest(
        identifier: ProcessInfo.processInfo.globallyUniqueString,
        content: content,
        trigger: nil
    )
    UNUserNotificationCenter.current().add(request) { error in
        if let error = error {
            print("🍎 Failed to dispatch notification: \(error)")
        }
    }
}
