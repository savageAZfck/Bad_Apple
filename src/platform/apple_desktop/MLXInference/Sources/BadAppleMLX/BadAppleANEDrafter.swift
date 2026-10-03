import Darwin
import Foundation

/// Cross-silicon speculative drafting — the 0.6B's real job.
///
/// The Qwen3-0.6B artifact runs on the Neural Engine and proposes draft
/// tokens; the 7B GPU model verifies k proposals in a single forward
/// pass and only commits the longest matching prefix. The draft's
/// residency is the point: it lives in the ANE driver's own region,
/// costing the unified-memory budget ~nothing where a GPU-resident
/// draft model could not fit at all.
///
/// State synchronization (both sides must agree after every round):
/// - GPU: `trimPromptCache` rewinds rejected verify positions.
/// - ANE: `bad_apple_ane_rewind` moves the position-indexed KV frontier
///   back over discarded drafts, then the accepted correction/bonus
///   token is fed so the next round starts from the identical sequence.
///
/// Token-selection semantics: this path is GREEDY. The ANE bridge
/// emits argmax only, so the verifier must compare argmax-vs-argmax —
/// sampling here would invalidate the acceptance measurement.
///
/// Enabled by `~/.bad_apple/ane_drafter` (contents: path to the
/// artifact's conversion_manifest.json) or `BADAPPLE_ANE_DRAFT=1`
/// (which defaults the manifest to the 0.6B sentinel artifact).
public actor BadAppleANEDrafter {
    public static let shared = BadAppleANEDrafter()

    public enum State: String, Sendable {
        case disabled, unloaded, loading, ready, failed
    }

    public enum DrafterError: Error, LocalizedError {
        case disabled
        case notReady(State)
        case bridgeUnavailable
        case rewindUnavailable

        public var errorDescription: String? {
            switch self {
            case .disabled:
                return "ANE drafting is disabled (BADAPPLE_ANE_DRAFT=1 or ~/.bad_apple/ane_drafter)."
            case .notReady(let state):
                return "The ANE drafter is not ready (state: \(state.rawValue))."
            case .bridgeUnavailable:
                return "libBadAppleBridge's Neural Engine entry points are not loaded in this process."
            case .rewindUnavailable:
                return "The loaded ANE bridge predates bad_apple_ane_rewind — rebuild it for speculative drafting."
            }
        }
    }

    // MARK: Bridge entry points

    fileprivate typealias CreateFn = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutableRawPointer?
    fileprivate typealias DestroyFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    fileprivate typealias PredictFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int) -> Int32
    fileprivate typealias ResetFn = @convention(c) (UnsafeMutableRawPointer?) -> Bool
    fileprivate typealias RewindFn = @convention(c) (UnsafeMutableRawPointer?, Int32) -> Bool
    fileprivate typealias PositionFn = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    fileprivate typealias PrefillFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int) -> Bool

    fileprivate struct Bridge: Sendable {
        let create: CreateFn
        let destroy: DestroyFn
        let predict: PredictFn
        let reset: ResetFn
        let rewind: RewindFn
        let position: PositionFn
        let prefill: PrefillFn?

        static let resolved: Bridge? = {
            let rtldDefault = UnsafeMutableRawPointer(bitPattern: -2)
            func sym<T>(_ name: String, _: T.Type) -> T? {
                dlsym(rtldDefault, name).map { unsafeBitCast($0, to: T.self) }
            }
            guard let create = sym("bad_apple_ane_create", CreateFn.self),
                  let destroy = sym("bad_apple_ane_destroy", DestroyFn.self),
                  let predict = sym("bad_apple_ane_predict_next", PredictFn.self),
                  let reset = sym("bad_apple_ane_reset", ResetFn.self),
                  let rewind = sym("bad_apple_ane_rewind", RewindFn.self),
                  let position = sym("bad_apple_ane_position", PositionFn.self) else { return nil }
            // prefill is optional — an older bridge falls back to the
            // token-by-token predict loop inside the C export's callers.
            let prefill = sym("bad_apple_ane_prefill", PrefillFn.self)
            return Bridge(create: create, destroy: destroy, predict: predict,
                          reset: reset, rewind: rewind, position: position,
                          prefill: prefill)
        }()
    }

    /// A checked-out drafting session — the raw handle plus its bridge
    /// calls, safe to invoke inside `ModelContainer.perform` closures
    /// (the bridge serializes on its own lock; the inference loop is
    /// the session's only caller while acquired).
    public final class Session: @unchecked Sendable {
        fileprivate let raw: UnsafeMutableRawPointer
        fileprivate let bridge: Bridge
        public let contextLimit: Int

        fileprivate init(raw: UnsafeMutableRawPointer, bridge: Bridge, contextLimit: Int) {
            self.raw = raw
            self.bridge = bridge
            self.contextLimit = contextLimit
        }

        /// Clear all KV state. Call once before priming a new prompt.
        @discardableResult
        public func reset() -> Bool { bridge.reset(raw) }

        /// Feed tokens; returns the model's argmax proposal for the
        /// position immediately after them, or -1 on failure.
        public func predictNext(_ tokens: [Int32]) -> Int32 {
            tokens.withUnsafeBufferPointer { bridge.predict(raw, $0.baseAddress, $0.count) }
        }

        /// Commit a whole prompt in chunked batched passes when the bridge
        /// exposes `bad_apple_ane_prefill` (falls back to the token loop).
        /// No token is proposed — follow with `predictNext` on the anchor.
        @discardableResult
        public func prefill(_ tokens: [Int32]) -> Bool {
            if let prefill = bridge.prefill {
                return tokens.withUnsafeBufferPointer { prefill(raw, $0.baseAddress, $0.count) }
            }
            for chunkStart in stride(from: 0, to: tokens.count, by: 256) {
                let chunk = Array(tokens[chunkStart ..< min(chunkStart + 256, tokens.count)])
                guard predictNext(chunk) >= 0 || chunk.count == 0 else { return false }
            }
            return true
        }

        /// Rewind the KV frontier so the next predict overwrites
        /// rejected drafts. `position` is an absolute frontier index.
        @discardableResult
        public func rewind(to position: Int32) -> Bool { bridge.rewind(raw, position) }

        /// Current KV frontier — the next unwritten position.
        public var position: Int32 { bridge.position(raw) }
    }

    // MARK: State

    public private(set) var state: State = .unloaded
    public private(set) var lastError: String?
    public private(set) var contextLimit = 512
    public private(set) var rounds = 0
    public private(set) var proposedTokens = 0
    public private(set) var acceptedTokens = 0

    private var handle: UnsafeMutableRawPointer?
    private var inUse = false
    private var loadTask: Task<Void, Never>?
    private var pressureSource: DispatchSourceMemoryPressure?

    private let queue = DispatchQueue(label: "com.badapple.ane-drafter", qos: .utility)

    public init() {}

    // MARK: Configuration

    public nonisolated static var controlFile: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ane_drafter")
    }

    public nonisolated static var enabled: Bool {
        let env = ProcessInfo.processInfo.environment
        if env["BADAPPLE_ANE_DRAFT"] == "0" { return false }
        return env["BADAPPLE_ANE_DRAFT"] == "1"
            || !(env["BADAPPLE_DRAFT_MODEL"] ?? "").isEmpty
            || FileManager.default.fileExists(atPath: controlFile.path)
    }

    /// Default to the 0.6B sentinel artifact — drafting is what it was
    /// placed for; the sentinel itself now rides the 4B.
    public nonisolated static var manifestURL: URL {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_DRAFT_MODEL"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let configured = (try? String(contentsOf: controlFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let path = configured.isEmpty
            ? NSHomeDirectory() + "/.bad_apple/ane_sentinel_model/conversion_manifest.json"
            : (configured as NSString).expandingTildeInPath
        return URL(fileURLWithPath: path)
    }

    nonisolated static var reserveBytes: UInt64 {
        UInt64(ProcessInfo.processInfo.environment["BADAPPLE_DRAFT_RESERVE_MB"].flatMap(Int.init) ?? 128) * 1_048_576
    }

    // MARK: Lifecycle

    public func startIfEnabled() {
        installPressureHandler()
        guard Self.enabled else {
            state = .disabled
            return
        }
        guard state == .unloaded || state == .failed || state == .disabled, loadTask == nil else { return }
        loadTask = Task { await self.load() }
    }

    public func unload(reason: String) {
        guard let current = handle else { return }
        handle = nil
        inUse = false
        state = Self.enabled ? .unloaded : .disabled
        lastError = reason
        NSLog("[BadAppleANEDrafter] unloaded: %@", reason)
        if let bridge = Bridge.resolved {
            queue.async { bridge.destroy(current) }
        }
    }

    private func load() async {
        defer { loadTask = nil }
        guard let bridge = Bridge.resolved else {
            fail(Bridge.symbolCheckFailed
                 ? DrafterError.rewindUnavailable.localizedDescription
                 : DrafterError.bridgeUnavailable.localizedDescription)
            return
        }
        let manifest = Self.manifestURL
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            fail("manifest not found at \(manifest.path)")
            return
        }
        if let refusal = BadAppleANEBrain.admissionRefusal(
            artifactDir: manifest.deletingLastPathComponent(), extraReserve: 0) {
            fail(refusal)
            return
        }

        state = .loading
        contextLimit = BadAppleANEBrain.contextLimit(manifest: manifest) ?? 512

        let path = manifest.path
        let raw: UnsafeMutableRawPointer? = await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: path.withCString { bridge.create($0) }) }
        }
        guard let raw else {
            fail("bridge failed to create the drafter core (see log for CoreML error)")
            return
        }
        handle = raw
        lastError = nil
        state = .ready
        NSLog("[BadAppleANEDrafter] ready (context %d)", contextLimit)
    }

    private func fail(_ message: String) {
        state = .failed
        lastError = message
        NSLog("[BadAppleANEDrafter] load refused/failed: %@", message)
    }

    private func installPressureHandler() {
        guard pressureSource == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak self] in
            guard let self else { return }
            Task { await self.unload(reason: "system memory pressure critical") }
        }
        source.resume()
        pressureSource = source
    }

    // MARK: Sessions

    /// Acquire the drafting session for one generation. Nil when the
    /// drafter is unready or another generation already holds it — the
    /// caller falls back to ordinary GPU decoding either way.
    public func acquire() -> Session? {
        guard Self.enabled, state == .ready, !inUse, let handle, let bridge = Bridge.resolved else {
            NSLog("[BadAppleANEDrafter] acquire declined: enabled=%d state=%@ inUse=%d handle=%d bridge=%d",
                  Self.enabled, state.rawValue, inUse, handle != nil, Bridge.resolved != nil)
            if state == .unloaded || state == .failed { startIfEnabled() }
            return nil
        }
        inUse = true
        return Session(raw: handle, bridge: bridge, contextLimit: contextLimit)
    }

    public func release() {
        inUse = false
    }

    public func record(proposed: Int, accepted: Int, rounds newRounds: Int) {
        rounds += newRounds
        proposedTokens += proposed
        acceptedTokens += accepted
    }

    /// Snapshot for runtime_status / dashboard.
    public func status() -> [String: any Sendable] {
        var s: [String: any Sendable] = [
            "enabled": Self.enabled,
            "state": state.rawValue,
            "manifest": Self.manifestURL.path,
            "context": contextLimit,
            "rounds": rounds,
            "proposed": proposedTokens,
            "accepted": acceptedTokens,
        ]
        if proposedTokens > 0 {
            s["acceptance_rate"] = Double(acceptedTokens) / Double(proposedTokens)
        }
        if let lastError { s["last_error"] = lastError }
        return s
    }
}

extension BadAppleANEDrafter.Bridge {
    /// Rewind and position are the newest bridge symbols — a stale
    /// dylib resolves create/predict/reset but lacks them, which is a
    /// distinct, diagnosable failure.
    fileprivate static var symbolCheckFailed: Bool {
        let rtldDefault = UnsafeMutableRawPointer(bitPattern: -2)
        return dlsym(rtldDefault, "bad_apple_ane_rewind") == nil
    }
}
