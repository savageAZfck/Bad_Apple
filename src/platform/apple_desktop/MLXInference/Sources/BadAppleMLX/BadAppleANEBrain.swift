import Darwin
import Foundation
import Tokenizers

/// The second brain: a stateful, sharded CoreML model resident on the
/// Neural Engine, driven through libBadAppleBridge's `bad_apple_ane_*` C
/// exports. It runs beside the MLX/GPU brain on separate silicon — slow
/// (~4–5 tok/s for the FP16 4B) but nearly free in power and CPU, so it
/// takes background work while the GPU brain stays on the conversation.
///
/// Opt-in: the control file `~/.bad_apple/ane_brain` enables it (mirrors
/// eyes/ears). Its contents, if non-empty, are the path to the artifact's
/// `conversion_manifest.json`; `BADAPPLE_ANE_MODEL` overrides both.
///
/// Memory discipline, learned the hard way on a 16 GB machine:
/// - load is admitted only when reclaimable memory covers the artifact
///   plus a reserve, and never under system memory pressure;
/// - a critical memory-pressure event unloads the brain immediately;
/// - the bridge keeps exactly one resident copy of the shards.
public actor BadAppleANEBrain {
    public static let shared = BadAppleANEBrain()

    public enum State: String, Sendable {
        case disabled, unloaded, loading, ready, failed
    }

    public enum BrainError: Error, LocalizedError {
        case disabled
        case notReady(State)
        case bridgeUnavailable
        case promptTooLong(tokens: Int, limit: Int)
        case predictionFailed
        case tokenizer(String)

        public var errorDescription: String? {
            switch self {
            case .disabled:
                return "The Neural Engine brain is disabled (create ~/.bad_apple/ane_brain to enable it)."
            case .notReady(let state):
                return "The Neural Engine brain is not ready (state: \(state.rawValue))."
            case .bridgeUnavailable:
                return "libBadAppleBridge's Neural Engine entry points are not loaded in this process."
            case .promptTooLong(let tokens, let limit):
                return "Prompt is \(tokens) tokens; the Neural Engine brain's context is \(limit)."
            case .predictionFailed:
                return "Neural Engine prediction failed."
            case .tokenizer(let detail):
                return "Neural Engine brain tokenizer error: \(detail)"
            }
        }
    }

    // MARK: Bridge entry points (resolved at runtime — no link-time coupling)

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

    /// Opaque bridge handle; only ever touched on `queue`.
    private final class Handle: @unchecked Sendable {
        let raw: UnsafeMutableRawPointer
        init(_ raw: UnsafeMutableRawPointer) { self.raw = raw }
    }

    // MARK: State

    public private(set) var state: State = .unloaded
    public private(set) var lastError: String?
    public private(set) var loadSeconds: Double = 0
    public private(set) var tokensGenerated = 0
    public private(set) var generationSeconds: Double = 0

    private var handle: Handle?
    private var tokenizer: (any Tokenizers.Tokenizer)?
    private var stopTokens: Set<Int32> = []
    private var contextLimit = 2048
    private var consecutiveFaults = 0
    private var loadTask: Task<Void, Never>?
    private var pressureSource: DispatchSourceMemoryPressure?
    /// Room held back for the GPU brain until it is resident: the chat brain
    /// outranks the background brain. The engine clears this once the main
    /// model loads (or fails to, at which point this brain is the fallback).
    private var mainBrainReserveBytes: UInt64 = 4_831_838_208

    public func setMainBrainReserve(bytes: UInt64) {
        mainBrainReserveBytes = bytes
    }

    /// Every bridge call is synchronous and long (seconds for load, ~200 ms
    /// per token), so it runs here instead of on a cooperative pool thread.
    private let queue = DispatchQueue(label: "com.badapple.ane-brain", qos: .utility)

    public init() {}

    // MARK: Configuration

    public nonisolated static var controlFile: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ane_brain")
    }

    public nonisolated static var enabled: Bool {
        ProcessInfo.processInfo.environment["BADAPPLE_ANE_MODEL"] != nil
            || FileManager.default.fileExists(atPath: controlFile.path)
    }

    public nonisolated static var manifestURL: URL {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MODEL"], !env.isEmpty {
            return URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let configured = (try? String(contentsOf: controlFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let path = configured.isEmpty
            ? NSHomeDirectory() + "/.bad_apple/ane/conversion_manifest.json"
            : (configured as NSString).expandingTildeInPath
        return URL(fileURLWithPath: path)
    }

    /// Reserve kept free after the brain is admitted (`BADAPPLE_ANE_RESERVE_MB`).
    /// "Reclaimable" excludes idle anonymous pages macOS can still compress,
    /// so this is a floor, not the whole margin; the critical-pressure
    /// handler is the backstop that unloads the brain if the floor is wrong.
    nonisolated static var reserveBytes: UInt64 {
        UInt64(ProcessInfo.processInfo.environment["BADAPPLE_ANE_RESERVE_MB"].flatMap(Int.init) ?? 512) * 1_048_576
    }

    // MARK: Lifecycle

    /// Start loading in the background if enabled and admitted. Idempotent.
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
        NSLog("[BadAppleANEBrain] unloaded: %@", reason)
        if let bridge = Bridge.resolved {
            queue.async { bridge.destroy(current.raw) }
        }
    }

    private func load() async {
        defer { loadTask = nil }
        guard let bridge = Bridge.resolved else {
            fail(BrainError.bridgeUnavailable.localizedDescription)
            return
        }
        let manifest = Self.manifestURL
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            fail("manifest not found at \(manifest.path)")
            return
        }
        if let refusal = Self.admissionRefusal(
            artifactDir: manifest.deletingLastPathComponent(), extraReserve: mainBrainReserveBytes) {
            fail(refusal)
            return
        }

        state = .loading
        let started = Date()
        do {
            let folder = manifest.deletingLastPathComponent()
            let tok = try await AutoTokenizer.from(modelFolder: folder)
            tokenizer = tok
            stopTokens = Set(["<|im_end|>", "<|endoftext|>"].compactMap { tok.convertTokenToId($0) }.map { Int32($0) })
            contextLimit = Self.contextLimit(manifest: manifest) ?? 2048
        } catch {
            fail("tokenizer: \(error.localizedDescription)")
            return
        }

        let path = manifest.path
        let footprintBefore = Self.physFootprintBytes()
        let raw: UnsafeMutableRawPointer? = await withCheckedContinuation { cont in
            queue.async { cont.resume(returning: path.withCString { bridge.create($0) }) }
        }
        guard let raw else {
            fail("bridge failed to create the Neural Engine core (see log for CoreML error)")
            return
        }
        handle = Handle(raw)
        loadSeconds = Date().timeIntervalSince(started)
        consecutiveFaults = 0
        lastError = nil
        state = .ready
        let footprintDelta = Int64(Self.physFootprintBytes()) - Int64(footprintBefore)
        NSLog("[BadAppleANEBrain] ready in %.1fs (context %d) — real footprint delta %+.2f GiB%s",
              loadSeconds, contextLimit, Double(footprintDelta) / 1_073_741_824,
              Self.bestEffortEnabled ? " [best-effort]" : "")
    }

    private func fail(_ message: String) {
        state = .failed
        lastError = message
        NSLog("[BadAppleANEBrain] load refused/failed: %@", message)
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

    // MARK: Generation

    public var ready: Bool { state == .ready }

    /// Greedy generation on the Neural Engine. Throws `.notReady` (and kicks
    /// a background load) rather than blocking a caller for a cold load —
    /// callers are expected to fall back to the GPU brain.
    public func generate(prompt: String, systemPrompt: String?, maxTokens: Int) async throws -> String {
        guard Self.enabled else { throw BrainError.disabled }
        guard state == .ready, let handle, let tokenizer, let bridge = Bridge.resolved else {
            if state == .unloaded || state == .failed { startIfEnabled() }
            throw BrainError.notReady(state)
        }

        var messages: [[String: any Sendable]] = []
        if let systemPrompt, !systemPrompt.isEmpty {
            messages.append(["role": "system", "content": systemPrompt])
        }
        messages.append(["role": "user", "content": prompt])
        let promptTokens: [Int32]
        do {
            promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: ["enable_thinking": false]
            ).map { Int32($0) }
        } catch {
            throw BrainError.tokenizer(error.localizedDescription)
        }
        let budget = max(1, min(maxTokens, contextLimit - promptTokens.count))
        guard promptTokens.count < contextLimit else {
            throw BrainError.promptTooLong(tokens: promptTokens.count, limit: contextLimit)
        }

        let stops = stopTokens
        let started = Date()
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

        guard let output else {
            consecutiveFaults += 1
            if consecutiveFaults >= 2 {
                unload(reason: "repeated prediction faults — will reload on next request")
                startIfEnabled()
            }
            throw BrainError.predictionFailed
        }
        consecutiveFaults = 0
        tokensGenerated += output.count
        generationSeconds += Date().timeIntervalSince(started)
        let text = tokenizer.decode(tokens: output.map(Int.init), skipSpecialTokens: true)
        return Self.stripThinking(text)
    }

    /// Snapshot for runtime_status / dashboard.
    public func status() -> [String: any Sendable] {
        var s: [String: any Sendable] = [
            "enabled": Self.enabled,
            "state": state.rawValue,
            "manifest": Self.manifestURL.path,
            "context": contextLimit,
            "load_seconds": loadSeconds,
            "tokens_generated": tokensGenerated,
        ]
        if generationSeconds > 0 {
            s["tokens_per_second"] = Double(tokensGenerated) / generationSeconds
        }
        if let lastError { s["last_error"] = lastError }
        return s
    }

    // MARK: Helpers

    nonisolated static func stripThinking(_ text: String) -> String {
        var t = text
        if let open = t.range(of: "<think>"), let close = t.range(of: "</think>", range: open.upperBound..<t.endIndex) {
            t.removeSubrange(open.lowerBound..<close.upperBound)
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func contextLimit(manifest: URL) -> Int? {
        guard let data = try? Data(contentsOf: manifest),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any] else { return nil }
        return (model["seq_len"] as? NSNumber)?.intValue
    }

    /// Resident cost of the artifact: every compiled shard's weights. The
    /// embedding table is mmapped and only touched one row per token, so it
    /// is excluded — the kernel can evict it for free.
    nonisolated static func residentBytes(artifactDir: URL) -> UInt64 {
        guard let walker = FileManager.default.enumerator(
            at: artifactDir, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        var total: UInt64 = 0
        for case let url as URL in walker where url.lastPathComponent == "weight.bin" {
            total += UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    nonisolated static func reclaimableBytes() -> UInt64 {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        let pages = UInt64(stats.free_count) + UInt64(stats.inactive_count)
            + UInt64(stats.speculative_count) + UInt64(stats.purgeable_count)
        return pages * UInt64(getpagesize())
    }

    /// True when the operator has opted into best-effort admission:
    /// `~/.bad_apple/ane_brain_best_effort` exists, or the env override is
    /// set. Best-effort still records what the gate *would* have decided so
    /// the ledger/log stays honest about the margin.
    nonisolated static var bestEffortEnabled: Bool {
        if ProcessInfo.processInfo.environment["BADAPPLE_ANE_ADMISSION"] == "0" { return true }
        return FileManager.default.fileExists(
            atPath: NSHomeDirectory() + "/.bad_apple/ane_brain_best_effort")
    }

    /// Real resident cost of this process, matching the kernel's jetsam
    /// basis — used to measure what an ANE load actually costs rather than
    /// trusting the artifact-size estimate.
    nonisolated static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return UInt64(info.phys_footprint)
    }

    /// nil = admitted; otherwise the reason the load is refused.
    nonisolated static func admissionRefusal(artifactDir: URL, extraReserve: UInt64 = 0) -> String? {
        let need = residentBytes(artifactDir: artifactDir) + reserveBytes + extraReserve
        let have = reclaimableBytes()
        if bestEffortEnabled {
            if have < need {
                NSLog("[BadAppleANEBrain] best-effort override: gate would refuse (needs %.2f GiB, %.2f GiB reclaimable) — attempting load and measuring real footprint",
                      Double(need) / 1_073_741_824, Double(have) / 1_073_741_824)
            }
            return nil
        }
        guard have >= need else {
            return String(format: "memory admission: needs %.2f GiB (shards + reserve%@), %.2f GiB reclaimable",
                          Double(need) / 1_073_741_824,
                          extraReserve > 0 ? " + main-brain headroom" : "",
                          Double(have) / 1_073_741_824)
        }
        return nil
    }
}
