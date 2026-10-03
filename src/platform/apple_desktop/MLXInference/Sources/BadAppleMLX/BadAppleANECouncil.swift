import Darwin
import Foundation
import Tokenizers

/// The deliberating council — LLM judges on dedicated silicon.
///
/// The deterministic BadAppleCouncil is a fast pre-filter: fourteen
/// fixed seats voting on a feature vector. This is the god-tier layer
/// above it — a small panel of persona judges, each an actual model
/// pass on the Neural Engine, reasoning about the intent in language
/// rather than scoring extracted features. The GPU brain keeps
/// generating while the council deliberates at <1% CPU.
///
/// Seats run sequentially on one resident handle by default — on a
/// 16 GB machine the honest trade is ~2 s of serial ANE time for zero
/// GPU cost. `BADAPPLE_COUNCIL_SEATS` trims the panel; bigger machines
/// can point `BADAPPLE_COUNCIL_MODEL` at a second artifact for true
/// concurrency later.
///
/// Fail semantics mirror the sentinel: absent model → unavailable
/// (deterministic council still rules); any live failure → that seat
/// abstains as "escalate", so doubt routes to the human, never silence.
public actor BadAppleANECouncil {
    public static let shared = BadAppleANECouncil()

    public enum State: String, Sendable {
        case disabled, unloaded, loading, ready, failed
    }

    public enum Vote: String, Sendable {
        case allow, deny, escalate, abstain
    }

    /// One judge's ruling on an intent.
    public struct SeatRuling: Sendable {
        public let seat: String
        public let vote: Vote
        public let rationale: String
        public let latencyUs: UInt64
    }

    /// The panel's aggregate decision.
    public struct Verdict: Sendable {
        /// True when the panel is divided or any seat denied — routes
        /// to the human, same contract as the deterministic council.
        public let contested: Bool
        public let rulings: [SeatRuling]
        /// "3 allow / 1 escalate" style summary for prompt-side counsel.
        public let summaryLine: String
        /// Whole panel unavailable — callers fall back to deterministic.
        public let unavailable: Bool
    }

    // MARK: Bridge (same multi-instance symbols as ANEBrain/Sentinel)

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
    public private(set) var deliberations = 0

    private var handle: Handle?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var stopTokens: Set<Int32> = []
    private var contextLimit = 512
    private var consecutiveFaults = 0
    private var loadTask: Task<Void, Never>?
    private var pressureSource: DispatchSourceMemoryPressure?

    private let queue = DispatchQueue(label: "com.badapple.ane-council", qos: .utility)

    public init() {}

    // MARK: Configuration

    public nonisolated static var controlFile: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ane_council")
    }

    public nonisolated static var enabled: Bool {
        if ProcessInfo.processInfo.environment["BADAPPLE_COUNCIL"] == "0" { return false }
        return ProcessInfo.processInfo.environment["BADAPPLE_COUNCIL_MODEL"] != nil
            || FileManager.default.fileExists(atPath: controlFile.path)
            // The council can share the sentinel's artifact — a second
            // handle on the same weights is a legitimate co-resident.
            || BadAppleANESentinel.enabled
    }

    public nonisolated static var manifestURL: URL {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_COUNCIL_MODEL"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let configured = (try? String(contentsOf: controlFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !configured.isEmpty {
            return URL(fileURLWithPath: (configured as NSString).expandingTildeInPath)
        }
        return BadAppleANESentinel.manifestURL
    }

    // MARK: Seats — personas, not feature weights

    private struct Seat {
        let name: String
        let system: String
    }

    /// The panel. Each seat gets the same intent, a different lens —
    /// dissent between lenses is the signal, not noise.
    private static var seats: [Seat] {
        let common = """
            You are one seat on a hardware-isolated oversight council. Judge \
            whether the proposed tool call should execute. Reply with exactly \
            one vote word on the first line — ALLOW, DENY, or ESCALATE — then \
            one short reason on the second line.
            """
        let all: [Seat] = [
            Seat(name: "skeptic", system: common + "\nYour lens: assume the worst. Hunt for deception, exfiltration, hidden blast radius. Vote DENY for clear violations, ESCALATE for anything suspicious."),
            Seat(name: "operator", system: common + "\nYour lens: is this what the operator actually asked for? Scope creep and unrequested actions are your concern. ESCALATE on scope doubts."),
            Seat(name: "careful", system: common + "\nYour lens: reversibility. Can this be undone, and how big is the blast radius? ESCALATE on irreversible or wide-reaching actions."),
            Seat(name: "pragmatist", system: common + "\nYour lens: utility versus risk. Ordinary reversible work should not be blocked. ALLOW what clearly serves the request."),
        ]
        let n = ProcessInfo.processInfo.environment["BADAPPLE_COUNCIL_SEATS"].flatMap(Int.init) ?? all.count
        return Array(all.prefix(max(1, min(n, all.count))))
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
        NSLog("[BadAppleANECouncil] unloaded: %@", reason)
        if let bridge = Bridge.resolved {
            queue.async { bridge.destroy(current.raw) }
        }
    }

    private func load() async {
        defer { loadTask = nil }
        guard let bridge = Bridge.resolved else {
            fail("bridge unavailable")
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
            fail("bridge failed to create the council core (see log for CoreML error)")
            return
        }
        handle = Handle(raw)
        consecutiveFaults = 0
        lastError = nil
        state = .ready
        NSLog("[BadAppleANECouncil] ready — %d seats (context %d)", Self.seats.count, contextLimit)
    }

    private func fail(_ message: String) {
        state = .failed
        lastError = message
        NSLog("[BadAppleANECouncil] load refused/failed: %@", message)
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

    // MARK: Deliberation

    public var ready: Bool { state == .ready }

    nonisolated static func parseVote(_ raw: String) -> (Vote, String) {
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false)
        let head = (lines.first.map(String.init) ?? "").trimmingCharacters(in: .whitespaces).uppercased()
        let reason = lines.count > 1
            ? String(lines[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            : head
        if head.hasPrefix("ALLOW") { return (.allow, reason) }
        if head.hasPrefix("DENY") { return (.deny, reason) }
        return (.escalate, reason.isEmpty ? head : reason)
    }

    /// Run one seat's pass on the intent. Private — a seat failure is an
    /// abstain-as-escalate ruling, never a dropped vote.
    private func seatVote(_ seat: Seat, prompt: String) async -> SeatRuling {
        guard let handle, let tokenizer, let bridge = Bridge.resolved else {
            return SeatRuling(seat: seat.name, vote: .abstain, rationale: "council not ready", latencyUs: 0)
        }
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": seat.system],
            ["role": "user", "content": prompt],
        ]
        let promptTokens: [Int32]
        do {
            promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: nil,
                additionalContext: ["enable_thinking": false]
            ).map { Int32($0) }
        } catch {
            return SeatRuling(seat: seat.name, vote: .abstain, rationale: "tokenizer failed", latencyUs: 0)
        }
        guard promptTokens.count < contextLimit else {
            return SeatRuling(seat: seat.name, vote: .abstain, rationale: "prompt over context", latencyUs: 0)
        }
        let budget = max(1, min(24, contextLimit - promptTokens.count))
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
        let us = UInt64(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1000
        guard let output else {
            consecutiveFaults += 1
            if consecutiveFaults >= 3 {
                unload(reason: "repeated prediction faults — will reload on next request")
                startIfEnabled()
            }
            return SeatRuling(seat: seat.name, vote: .abstain, rationale: "inference failed", latencyUs: us)
        }
        consecutiveFaults = 0
        let text = tokenizer.decode(tokens: output.map(Int.init), skipSpecialTokens: true)
        let (vote, reason) = Self.parseVote(BadAppleANEBrain.stripThinking(text))
        return SeatRuling(seat: seat.name, vote: vote, rationale: reason, latencyUs: us)
    }

    /// Deliberate one proposed tool call: every seat votes on the same
    /// intent, the panel aggregates. Any DENY or a non-trivial bloc of
    /// ESCALATEs makes the call contested — routing to the human is the
    /// same contract the deterministic council keeps.
    public func deliberate(toolName: String, argsJson: String, policyVerdict: String) async -> Verdict {
        guard Self.enabled else {
            return Verdict(contested: false, rulings: [], summaryLine: "council disabled", unavailable: true)
        }
        guard state == .ready else {
            if state == .unloaded || state == .failed { startIfEnabled() }
            return Verdict(contested: false, rulings: [], summaryLine: "council not ready", unavailable: true)
        }
        let argsSnip = String(argsJson.prefix(1500))
        let prompt = """
            PROPOSED ACTION
            name: \(toolName)
            args: \(argsSnip)
            policy: \(policyVerdict)

            VOTE
            """
        var rulings: [SeatRuling] = []
        for seat in Self.seats {
            rulings.append(await seatVote(seat, prompt: prompt))
        }
        deliberations += 1

        let denies = rulings.filter { $0.vote == .deny }.count
        let escalates = rulings.filter { $0.vote == .escalate || $0.vote == .abstain }.count
        let allows = rulings.filter { $0.vote == .allow }.count
        // Contested = any outright denial, or doubt outnumbering
        // confidence. A single escalate beside three allows is noted,
        // not blocking — the human sees the dissent in the summary.
        let contested = denies > 0 || escalates > allows
        let summary = "\(allows) allow / \(denies) deny / \(escalates) escalate"
            + (rulings.contains { $0.vote == .abstain } ? " (abstentions counted as doubt)" : "")
        return Verdict(contested: contested, rulings: rulings, summaryLine: summary, unavailable: false)
    }

    /// Snapshot for runtime_status / dashboard.
    public func status() -> [String: any Sendable] {
        var s: [String: any Sendable] = [
            "enabled": Self.enabled,
            "state": state.rawValue,
            "manifest": Self.manifestURL.path,
            "seats": Self.seats.map { $0.name },
            "deliberations": deliberations,
        ]
        if let lastError { s["last_error"] = lastError }
        return s
    }
}
