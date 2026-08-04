import Foundation

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
/// request.  The prompt is a null-terminated UTF-8 C string; the returned
/// C string is allocated with `strdup` and freed by Rust.
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

/// C-callable initialization entry point.  The Rust loader calls this after
/// `dlopen`ing the bridge dylib, and the bridge in turn registers its
/// callback with the Rust `register_apple_intelligence_oracle` primitive.
@_cdecl("init_firefly_siri_bridge")
public func initFireflySiriBridge() {
    registerAppleIntelligenceOracle(fireflyAppleIntelligenceCallback)
}
