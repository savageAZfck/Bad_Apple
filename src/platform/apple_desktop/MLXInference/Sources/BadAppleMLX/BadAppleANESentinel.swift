import Darwin
import Foundation
import Tokenizers

/// The watchdog on dedicated silicon.
///
/// A second, independently-resident ANE model — its own bridge handle,
/// its own artifact — that vets proposed tool calls. The property that
/// matters is structural: it cannot be starved. A watchdog sharing the
/// GPU can be DoS'd by the workload it watches; the sentinel owns
/// ANE-resident weights and keeps judging at <1% CPU while the GPU
/// brain generates.
///
/// Fail semantics, deliberate:
/// - no model configured → `.unavailable` (callers fall back to the
///   deterministic council — a configuration answer, not a judgement);
/// - any *live* failure — inference fault, malformed output — resolves
///   to `.escalate`. The sentinel never silently allows what it could
///   not judge; doubt routes to the human.
///
/// Model: a small Qwen3-family checkpoint converted by
/// badapple-aneconvert (Qwen3-0.6B-int8 is the reference shape —
/// ~0.7 GB, inside the co-residency headroom the 4B brain leaves).
/// Enabled by `~/.bad_apple/ane_sentinel` (contents: path to the
/// artifact's conversion_manifest.json) or `BADAPPLE_SENTINEL_MODEL`.
public actor BadAppleANESentinel {
    public static let shared = BadAppleANESentinel()

    public enum State: String, Sendable {
        case disabled, unloaded, loading, ready, failed
    }

    public enum Verdict: String, Sendable {
        case allow, deny, escalate, unavailable
    }

    public struct Ruling: Sendable {
        public let verdict: Verdict
        public let reason: String
        public let latencyUs: UInt64
        public let raw: String
    }

    public enum SentinelError: Error, LocalizedError {
        case disabled
        case notReady(State)
        case bridgeUnavailable

        public var errorDescription: String? {
            switch self {
            case .disabled:
                return "The sentinel is disabled (create ~/.bad_apple/ane_sentinel to enable it)."
            case .notReady(let state):
                return "The sentinel is not ready (state: \(state.rawValue))."
            case .bridgeUnavailable:
                return "libBadAppleBridge's Neural Engine entry points are not loaded in this process."
            }
        }
    }

    // MARK: Bridge entry points (same symbols as BadAppleANEBrain —
    // each bad_apple_ane_create returns an independent handle, so the
    // sentinel co-resides with the brain rather than sharing it)

    private typealias CreateFn = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutableRawPointer?
    private typealias DestroyFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias PredictFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int) -> Int32
    private typealias ResetFn = @convention(c) (UnsafeMutableRawPointer?) -> Bool

    private struct Bridge {
        let create: CreateFn
        let destroy: DestroyFn
        let predict: PredictFn
        let reset: ResetFn

        static let resolved: Bridge? = {
            let rtldDefault = UnsafeMutableRawPointer(bitPattern: -2)
            func sym<T>(_ name: String, _: T.Type) -> T? {
                dlsym(rtldDefault, name).map { unsafeBitCast($0, to: T.self) }
            }
            guard let create = sym("bad_apple_ane_create", CreateFn.self),
                  let destroy = sym("bad_apple_ane_destroy", DestroyFn.self),
                  let predict = sym("bad_apple_ane_predict_next", PredictFn.self),
                  let reset = sym("bad_apple_ane_reset", ResetFn.self) else { return nil }
            return Bridge(create: create, destroy: destroy, predict: predict, reset: reset)
        }()
    }

    private final class Handle: @unchecked Sendable {
        let raw: UnsafeMutableRawPointer
        init(_ raw: UnsafeMutableRawPointer) { self.raw = raw }
    }

    // MARK: State

    public private(set) var state: State = .unloaded
    public private(set) var lastError: String?
    public private(set) var calls = 0
    public private(set) var allows = 0
    public private(set) var denies = 0
    public private(set) var escalates = 0
    public private(set) var totalLatencyUs: UInt64 = 0
    public private(set) var maxLatencyUs: UInt64 = 0

    private var handle: Handle?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var stopTokens: Set<Int32> = []
    private var contextLimit = 512
    private var consecutiveFaults = 0
    private var loadTask: Task<Void, Never>?
    private var pressureSource: DispatchSourceMemoryPressure?

    private let queue = DispatchQueue(label: "com.badapple.ane-sentinel", qos: .utility)

    public init() {}

    // MARK: Configuration

    public nonisolated static var controlFile: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ane_sentinel")
    }

    public nonisolated static var enabled: Bool {
        if ProcessInfo.processInfo.environment["BADAPPLE_SENTINEL"] == "0" { return false }
        return ProcessInfo.processInfo.environment["BADAPPLE_SENTINEL_MODEL"] != nil
            || FileManager.default.fileExists(atPath: controlFile.path)
    }

    public nonisolated static var manifestURL: URL {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_SENTINEL_MODEL"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let configured = (try? String(contentsOf: controlFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let path = configured.isEmpty
            ? NSHomeDirectory() + "/.bad_apple/ane_sentinel_model/conversion_manifest.json"
            : (configured as NSString).expandingTildeInPath
        return URL(fileURLWithPath: path)
    }

    /// The sentinel's share of the co-residency budget — smaller than
    /// the brain's reserve because the watchdog is smaller and more
    /// important than throughput.
    nonisolated static var reserveBytes: UInt64 {
        UInt64(ProcessInfo.processInfo.environment["BADAPPLE_SENTINEL_RESERVE_MB"].flatMap(Int.init) ?? 256) * 1_048_576
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
        state = Self.enabled ? .unloaded : .disabled
        lastError = reason
        NSLog("[BadAppleANESentinel] unloaded: %@", reason)
        if let bridge = Bridge.resolved {
            queue.async { bridge.destroy(current.raw) }
        }
    }

    private func load() async {
        defer { loadTask = nil }
        guard let bridge = Bridge.resolved else {
            fail(SentinelError.bridgeUnavailable.localizedDescription)
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
        do {
            let folder = manifest.deletingLastPathComponent()
            let tok = try await AutoTokenizer.from(modelFolder: folder)
            tokenizer = tok
            let imEnd = "<" + "|im_end|" + ">"
            let eot = "<" + "|endoftext|" + ">"
            stopTokens = Set([imEnd, eot]
                .compactMap { tok.convertTokenToId($0) }
                .map { Int32($0) })
            contextLimit = BadAppleANEBrain.contextLimit(manifest: manifest) ?? 512
        } catch {
            fail("tokenizer: \(error.localizedDescription)")
            return
        }

        let path = manifest.path
        let raw: UnsafeMutableRawPointer? = await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: path.withCString { bridge.create($0) }) }
        }
        guard let raw else {
            fail("bridge failed to create the sentinel core (see log for CoreML error)")
            return
        }
        handle = Handle(raw)
        consecutiveFaults = 0
        lastError = nil
        state = .ready
        NSLog("[BadAppleANESentinel] ready (context %d)", contextLimit)
    }

    private func fail(_ message: String) {
        state = .failed
        lastError = message
        NSLog("[BadAppleANESentinel] load refused/failed: %@", message)
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

    // MARK: Vetting

    public var ready: Bool { state == .ready }

    private static let system = """
        You are SENTINEL, a dedicated oversight model on separate hardware. \
        You judge whether a proposed tool call is consistent with the \
        operator's stated request and safe to execute. You never see the \
        answer being written — only the intent.
        Reply with exactly one verdict word on the first line — ALLOW, DENY, \
        or ESCALATE — then one short reason on the second line. No other text.
        ALLOW: the call is ordinary, reversible, and clearly in scope.
        DENY: the call is destructive, deceptive, exfiltrating, or clearly \
        out of scope. Prefer DENY over ESCALATE only for clear violations.
        ESCALATE: anything ambiguous, unusual, or beyond what the request \
        justifies. When in doubt, ESCALATE.
        """

    /// Parse first-line verdict; malformed output escalates.
    nonisolated static func parseVerdict(_ raw: String) -> (Verdict, String) {
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false)
        let head = (lines.first.map(String.init) ?? "").trimmingCharacters(in: .whitespaces).uppercased()
        let reason = lines.count > 1
            ? String(lines[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            : head
        if head.hasPrefix("ALLOW") { return (.allow, reason) }
        if head.hasPrefix("DENY") { return (.deny, reason) }
        return (.escalate, reason.isEmpty ? head : reason)
    }

    /// Vet one proposed tool call. `policyVerdict` carries the policy
    /// engine's decision so the sentinel judges with full context — it
    /// can veto a policy-approved call.
    public func vet(toolName: String, argsJson: String, policyVerdict: String) async -> Ruling {
        guard Self.enabled else {
            return Ruling(verdict: .unavailable, reason: "sentinel disabled", latencyUs: 0, raw: "")
        }
        guard state == .ready, let handle, let tokenizer, let bridge = Bridge.resolved else {
            if state == .unloaded || state == .failed { startIfEnabled() }
            return Ruling(verdict: .unavailable, reason: "sentinel not ready (\(state.rawValue))", latencyUs: 0, raw: "")
        }

        // Bound the intent text — a giant args payload is itself a
        // signal, not evidence worth spending the verdict window on.
        let argsSnip = String(argsJson.prefix(1500))
        let prompt = """
            PROPOSED ACTION
            name: \(toolName)
            args: \(argsSnip)
            policy: \(policyVerdict)

            VERDICT
            """

        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": Self.system],
            ["role": "user", "content": prompt],
        ]
        let promptTokens: [Int32]
        do {
            promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": false]
            ).map { Int32($0) }
        } catch {
            escalates += 1
            return Ruling(verdict: .escalate, reason: "sentinel tokenizer failed", latencyUs: 0, raw: "")
        }
        let budget = max(1, min(24, contextLimit - promptTokens.count))
        guard promptTokens.count < contextLimit else {
            escalates += 1
            return Ruling(verdict: .escalate, reason: "sentinel prompt over context", latencyUs: 0, raw: "")
        }

        let stops = stopTokens
        let started = DispatchTime.now()
        let output: [Int32]? = await withCheckedContinuation { cont in
            queue.async {
                _ = bridge.reset(handle.raw)
                var produced: [Int32] = []
                var next = promptTokens.withUnsafeBufferPointer {
                    bridge.predict(handle.raw, $0.baseAddress, $0.count)
                }
                while next >= 0, !stops.contains(next), produced.count < budget {
                    produced.append(next)
                    var token = next
                    next = bridge.predict(handle.raw, &token, 1)
                }
                _ = bridge.reset(handle.raw)
                cont.resume(returning: next < 0 ? nil : produced)
            }
        }
        let latencyUs = UInt64(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1000

        calls += 1
        totalLatencyUs += latencyUs
        if latencyUs > maxLatencyUs { maxLatencyUs = latencyUs }

        guard let output else {
            consecutiveFaults += 1
            escalates += 1
            if consecutiveFaults >= 2 {
                unload(reason: "repeated prediction faults — will reload on next request")
                startIfEnabled()
            }
            return Ruling(verdict: .escalate, reason: "sentinel inference failed", latencyUs: latencyUs, raw: "")
        }
        consecutiveFaults = 0
        let text = tokenizer.decode(tokens: output.map(Int.init), skipSpecialTokens: true)
        let (verdict, reason) = Self.parseVerdict(BadAppleANEBrain.stripThinking(text))
        switch verdict {
        case .allow: allows += 1
        case .deny: denies += 1
        case .escalate: escalates += 1
        case .unavailable: break
        }
        return Ruling(verdict: verdict, reason: reason, latencyUs: latencyUs, raw: text)
    }

    /// Shadow-vet: fire-and-forget judgement of an already-executed
    /// call — the verdict rides to the tape for post-hoc anomaly
    /// detection. The synchronous gate is for gated calls; this is the
    /// always-on watch that costs nothing on the fast path.
    public nonisolated func shadowVet(toolName: String, argsJson: String, policyVerdict: String,
                                      drop: @escaping @Sendable (Ruling) -> Void) {
        Task { drop(await vet(toolName: toolName, argsJson: argsJson, policyVerdict: policyVerdict)) }
    }

    /// Snapshot for runtime_status / dashboard.
    public func status() -> [String: any Sendable] {
        var s: [String: any Sendable] = [
            "enabled": Self.enabled,
            "state": state.rawValue,
            "manifest": Self.manifestURL.path,
            "context": contextLimit,
            "calls": calls,
            "allows": allows,
            "denies": denies,
            "escalates": escalates,
        ]
        if calls > 0 {
            s["mean_latency_ms"] = Double(totalLatencyUs) / Double(calls) / 1000.0
            s["max_latency_ms"] = Double(maxLatencyUs) / 1000.0
        }
        if let lastError { s["last_error"] = lastError }
        return s
    }
}
