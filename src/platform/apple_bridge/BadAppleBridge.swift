import CoreML
import Darwin
import Foundation
import UserNotifications

/// QoS class for latency-sensitive, user-visible inference work.
/// On macOS, `QOS_CLASS_USER_INTERACTIVE` (0x21) is the highest thread
/// scheduling class. It gives the ANE prediction path the best chance of
/// running on performance cores and keeping memory bus transactions unparked.
private let badAppleInteractiveQos: qos_class_t = qos_class_t(0x21)

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Imported Rust registration primitive.  This symbol is exposed by the
/// `bad_apple` library through `bad_apple_core.h`.
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
@_cdecl("bad_apple_apple_intelligence_callback")
public func badAppleIntelligenceCallback(
    _ prompt: UnsafePointer<CChar>
) -> UnsafeMutablePointer<CChar>? {
    let promptString = String(cString: prompt)
    let semaphore = DispatchSemaphore(value: 0)
    var result: String?

    Task {
        defer { semaphore.signal() }
        result = await buildAppleIntelligenceResponse(for: promptString)
    }

    guard semaphore.wait(timeout: .now() + .seconds(120)) == .success else {
        return nil
    }

    guard let text = result, !text.isEmpty,
          let cString = text.cString(using: .utf8) else {
        return nil
    }
    return strdup(cString)
}

/// C-callable deallocator for strings returned by the bridge.
///
/// `ptr` must be a pointer previously returned by `bad_apple_apple_intelligence_callback`,
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
            NSLog("%@", "🏴‍☠️  BAD APPLE // Notification authorization error: \(error)")
        } else {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Notification authorization granted: \(granted)")
        }
    }
}

/// C-callable initialization entry point.  The Rust loader calls this after
/// `dlopen`ing the bridge dylib, and the bridge in turn registers its
/// callback with the `register_apple_intelligence_oracle` primitive.
@_cdecl("init_bad_apple_bridge")
public func initBadAppleBridge() {
    if isRunningInAppBundle() {
        requestNotificationAuthorization()
    }
    registerAppleIntelligenceOracle(badAppleIntelligenceCallback)
}

/// C-callable desktop notification dispatch.  Pushes a native macOS user
/// alert immediately.  Requires prior notification authorization.
@_cdecl("dispatch_desktop_notification")
public func dispatchDesktopNotification(
    _ title: UnsafePointer<CChar>,
    _ body: UnsafePointer<CChar>
) {
    guard isRunningInAppBundle() else {
        NSLog("%@", "🏴‍☠️  BAD APPLE // Desktop notifications require an app bundle; skipping.")
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
            NSLog("%@", "🏴‍☠️  BAD APPLE // Failed to dispatch notification: \(error)")
        }
    }
}

@available(macOS 15.0, *)
private final class BadAppleANECore {
    let model: MLModel
    let modelURL: URL
    var configuration: MLModelConfiguration
    let inputName: String
    let outputName: String
    let causalMaskName: String?
    let positionIdsName: String?
    let cacheOffsetName: String?
    var state: MLState
    var placementRatio: Double = -1.0
    var tokensProcessed = 0
    var allowedCounts: [Int] = []
    var fixedCausalLength: Int?
    var lastPredictionError: Error?
    var selectedLoadLatency: TimeInterval = 0
    var selectedPrewarmLatency: TimeInterval = 0

    var smallestAllowedCount: Int { allowedCounts.first ?? 1 }

    private struct Candidate {
        let core: BadAppleANECore
        let loadLatency: TimeInterval
        let prewarmLatency: TimeInterval
        let decodeLatency: TimeInterval

        var score: TimeInterval {
            // Optimize for steady-state tok/s; load and prewarm are one-time costs.
            decodeLatency + prewarmLatency * 0.01 + loadLatency * 0.0001
        }
    }

    /// Try each compute unit, benchmark a few decode steps, and pick the one
    /// with the lowest per-token decode latency while respecting load/prewarm budgets.
    static func load(modelURL: URL) throws -> BadAppleANECore {
        let environment = ProcessInfo.processInfo.environment
        let maxLoadLatency = latencyBudget(
            environment["BADAPPLE_ANE_MAX_LOAD_MS"],
            defaultMilliseconds: 45_000
        )
        let maxPrewarmLatency = latencyBudget(
            environment["BADAPPLE_ANE_MAX_PREWARM_MS"],
            defaultMilliseconds: 5_000
        )
        var lastError: Error?
        let candidateGroups: [[MLComputeUnits]] = [
            [.cpuAndNeuralEngine, .all, .cpuAndGPU],
            [.cpuOnly],
        ]

        for computeUnitsGroup in candidateGroups {
            var accepted: [Candidate] = []
            for computeUnits in computeUnitsGroup {
                let configuration = MLModelConfiguration()
                configuration.computeUnits = computeUnits
                do {
                    let loadStart = ProcessInfo.processInfo.systemUptime
                    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
                    let loadLatency = ProcessInfo.processInfo.systemUptime - loadStart
                    guard loadLatency <= maxLoadLatency else {
                        let error = latencyError(
                            computeUnits,
                            phase: "load",
                            latency: loadLatency,
                            budget: maxLoadLatency
                        )
                        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rejected: \(error.localizedDescription)")
                        lastError = error
                        continue
                    }

                    let core = try BadAppleANECore(
                        model: model,
                        modelURL: modelURL,
                        configuration: configuration
                    )
                    let prewarmStart = ProcessInfo.processInfo.systemUptime
                    let prewarmSucceeded = core.prewarm()
                    let prewarmLatency = ProcessInfo.processInfo.systemUptime - prewarmStart
                    if let error = core.lastPredictionError, isCompilerFailure(error) {
                        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rejected after compiler failure with \(computeUnits): \(error)")
                        lastError = error
                        continue
                    }
                    guard prewarmSucceeded else {
                        let error = core.lastPredictionError ?? NSError(
                            domain: "BadAppleANE",
                            code: 4,
                            userInfo: [NSLocalizedDescriptionKey: "prewarm failed for computeUnits rawValue \(computeUnits.rawValue)"]
                        )
                        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rejected: \(error)")
                        lastError = error
                        continue
                    }
                    guard prewarmLatency <= maxPrewarmLatency else {
                        let error = latencyError(
                            computeUnits,
                            phase: "prewarm",
                            latency: prewarmLatency,
                            budget: maxPrewarmLatency
                        )
                        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rejected: \(error.localizedDescription)")
                        lastError = error
                        continue
                    }

                    core.selectedLoadLatency = loadLatency
                    core.selectedPrewarmLatency = prewarmLatency
                    let decodeLatency = benchmarkDecode(core)
                    let candidate = Candidate(
                        core: core,
                        loadLatency: loadLatency,
                        prewarmLatency: prewarmLatency,
                        decodeLatency: decodeLatency
                    )
                    NSLog(
                        "🏴‍☠️  BAD APPLE // ANE candidate rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok, score %.3f",
                        computeUnits.rawValue,
                        loadLatency,
                        prewarmLatency,
                        decodeLatency,
                        candidate.score
                    )
                    accepted.append(candidate)
                } catch {
                    let signature = isCompilerFailure(error) ? " [compiler failure]" : ""
                    NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rawValue \(computeUnits.rawValue) failed\(signature): \(error)")
                    lastError = error
                }
            }

            if let selected = accepted.min(by: { $0.score < $1.score }) {
                selected.core.auditPlacement()
                NSLog(
                    "🏴‍☠️  BAD APPLE // ANE core selected computeUnits rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok (score %.3f)",
                    selected.core.configuration.computeUnits.rawValue,
                    selected.loadLatency,
                    selected.prewarmLatency,
                    selected.decodeLatency,
                    selected.score
                )
                return selected.core
            }
        }

        throw lastError ?? NSError(domain: "BadAppleANE", code: -1, userInfo: nil)
    }

    private static func latencyBudget(
        _ milliseconds: String?,
        defaultMilliseconds: Double
    ) -> TimeInterval {
        milliseconds
            .flatMap(Double.init)
            .map { max($0, 1.0) / 1_000.0 }
            ?? defaultMilliseconds / 1_000.0
    }

    private static func latencyError(
        _ computeUnits: MLComputeUnits,
        phase: String,
        latency: TimeInterval,
        budget: TimeInterval
    ) -> NSError {
        NSError(
            domain: "BadAppleANE",
            code: 5,
            userInfo: [
                NSLocalizedDescriptionKey: String(
                    format: "%@ latency %.3fs exceeded %.3fs budget for computeUnits rawValue %ld",
                    phase,
                    latency,
                    budget,
                    computeUnits.rawValue
                )
            ]
        )
    }

    private static func isCompilerFailure(_ error: Error) -> Bool {
        let nsError = error as NSError
        let details = ([
            nsError.localizedDescription,
            nsError.localizedFailureReason,
            nsError.localizedRecoverySuggestion,
            nsError.userInfo.description,
        ].compactMap { $0 }).joined(separator: " ").lowercased()
        return [
            "bnns graph compile",
            "no space left on device",
            "failed to build the model execution plan",
            "error code: -14",
            "anefmodel",
            "on-device compiled macho",
        ].contains { details.contains($0) }
    }

    private static func benchmarkDecode(
        _ core: BadAppleANECore,
        iterations: Int = 5,
        maxStepLatency: TimeInterval = 0.5
    ) -> TimeInterval {
        core.reset()
        defer { core.reset() }

        // A short, arbitrary prompt to get the stateful KV cache started.
        let prompt: [Int32] = [128, 456]
        _ = prompt.withUnsafeBufferPointer { core.predictNext($0.baseAddress!, count: 2) }

        var next: Int32 = 1
        // The first decode step after prefill can be atypically slow; discard it.
        _ = core.predictNext(&next, count: 1)

        var total: TimeInterval = 0
        var measured = 0
        for _ in 0..<iterations {
            let stepStart = ProcessInfo.processInfo.systemUptime
            guard let result = core.predictNext(&next, count: 1) else { break }
            let stepLatency = ProcessInfo.processInfo.systemUptime - stepStart
            if stepLatency > maxStepLatency {
                // This candidate is too slow for decode; reject it with a huge score.
                return 1_000_000.0
            }
            total += stepLatency
            measured += 1
            next = result
        }
        guard measured > 0 else { return 1_000_000.0 }
        return total / TimeInterval(measured)
    }

    private init(model: MLModel, modelURL: URL, configuration: MLModelConfiguration) throws {
        self.configuration = configuration
        self.modelURL = modelURL
        self.model = model

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let input = inputs["input_ids"] ?? inputs.first(where: {
            $0.value.type == .multiArray && $0.value.multiArrayConstraint?.dataType == .int32
        })?.value else {
            throw NSError(domain: "BadAppleANE", code: 1)
        }
        guard let inputName = inputs.first(where: { $0.value === input })?.key else {
            throw NSError(domain: "BadAppleANE", code: 2)
        }
        guard let output = model.modelDescription.outputDescriptionsByName.first(where: {
            $0.value.type == .multiArray
        }) else {
            throw NSError(domain: "BadAppleANE", code: 3)
        }

        self.inputName = inputName
        self.outputName = output.key
        self.causalMaskName = inputs["causal_mask"] == nil ? nil : "causal_mask"
        self.positionIdsName = ["position_ids", "positions", "position"].first { inputs[$0] != nil }
        self.cacheOffsetName = ["cache_offset", "cache_len", "cache_position"].first { inputs[$0] != nil }

        // Some stateful ANE models (e.g. TokForge Qwen INT8) only accept a
        // fixed set of input shapes; discover them so we can chunk prompts.
        if let constraint = input.multiArrayConstraint {
            if constraint.shapeConstraint.type == .enumerated {
                allowedCounts = constraint.shapeConstraint.enumeratedShapes
                    .compactMap { $0.last?.intValue }
                    .filter { $0 > 0 }
                    .sorted()
            }
            // If the causal mask has a fixed trailing dimension, use it as the
            // attention key length (stateful models keep a full 2048 cache).
            if let maskName = self.causalMaskName,
               let maskConstraint = inputs[maskName]?.multiArrayConstraint,
               maskConstraint.shapeConstraint.type != .range,
               let last = maskConstraint.shape.last?.intValue, last > 0 {
                fixedCausalLength = last
            }
        }

        self.state = model.makeState()
    }

    func reset() {
        state = model.makeState()
        tokensProcessed = 0
    }

    func predictNext(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        guard count > 0, tokensProcessed + count <= 2048 else { return nil }

        if allowedCounts.isEmpty {
            return predictOne(tokens, count: count)
        }

        var processed = 0
        while processed < count {
            let remaining = count - processed
            guard let chunk = allowedCounts.filter({ $0 <= remaining }).last, chunk > 0 else {
                return nil
            }
            let result = predictOne(tokens.advanced(by: processed), count: chunk)
            processed += chunk
            if processed == count {
                return result
            }
            guard result != nil else { return nil }
        }
        return nil
    }

    private func predictOne(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        guard tokensProcessed + count <= 2048 else { return nil }
        do {
            let array = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: tokens),
                shape: [1, NSNumber(value: count)],
                dataType: .int32,
                strides: [NSNumber(value: count), 1],
                deallocator: nil
            )
            var features = [inputName: MLFeatureValue(multiArray: array)]

            if let positionIdsName {
                let positions = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .int32)
                for i in 0..<count {
                    positions[[0, NSNumber(value: i)]] = NSNumber(value: tokensProcessed + i)
                }
                features[positionIdsName] = MLFeatureValue(multiArray: positions)
            }

            if let cacheOffsetName {
                let cache = try MLMultiArray(shape: [1], dataType: .int32)
                cache[[0]] = NSNumber(value: tokensProcessed)
                features[cacheOffsetName] = MLFeatureValue(multiArray: cache)
            }

            if let causalMaskName {
                let keyLength = fixedCausalLength ?? (tokensProcessed + count)
                let mask = try MLMultiArray(
                    shape: [1, 1, NSNumber(value: count), NSNumber(value: keyLength)],
                    dataType: .float16
                )
                for row in 0..<count {
                    for column in 0..<keyLength {
                        let allowed = column <= tokensProcessed + row
                        mask[[0, 0, NSNumber(value: row), NSNumber(value: column)]] =
                            NSNumber(value: allowed ? Float(0) : Float(-65_504))
                    }
                }
                features[causalMaskName] = MLFeatureValue(multiArray: mask)
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            let prediction = try model.prediction(from: provider, using: state)
            guard let logits = prediction.featureValue(for: outputName)?.multiArrayValue else {
                return nil
            }
            tokensProcessed += count
            lastPredictionError = nil
            return argmaxLastDimension(logits)
        } catch {
            lastPredictionError = error
            return nil
        }
    }

    func prewarm() -> Bool {
        let n = smallestAllowedCount
        let zeros = [Int32](repeating: 0, count: n)
        let success = zeros.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return false }
            return predictNext(base, count: n) != nil
        }
        reset()
        return success
    }

    private func argmaxLastDimension(_ array: MLMultiArray) -> Int32? {
        guard let vocabulary = array.shape.last?.intValue, vocabulary > 0 else { return nil }
        let offset = array.count - vocabulary
        var bestIndex = 0
        var bestValue = -Double.infinity
        for index in 0..<vocabulary {
            let value = array[offset + index].doubleValue
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }
        return Int32(exactly: bestIndex)
    }

    fileprivate static func measurePlacement(
        modelURL: URL,
        configuration: MLModelConfiguration
    ) -> Double {
        let semaphore = DispatchSemaphore(value: 0)
        var measuredRatio = -1.0
        Task {
            defer { semaphore.signal() }
            guard let plan = try? await MLComputePlan.load(
                contentsOf: modelURL,
                configuration: configuration
            ), case .program(let program) = plan.modelStructure else {
                return
            }
            var total = 0
            var ane = 0
            func visit(_ block: MLModelStructure.Program.Block) {
                for operation in block.operations {
                    total += 1
                    if let usage = plan.deviceUsage(for: operation),
                       case .neuralEngine = usage.preferred {
                        ane += 1
                    }
                    for child in operation.blocks {
                        visit(child)
                    }
                }
            }
            for function in program.functions.values {
                visit(function.block)
            }
            if total > 0 {
                measuredRatio = Double(ane) / Double(total)
            }
        }
        guard semaphore.wait(timeout: .now() + .seconds(30)) == .success else {
            return -1.0
        }
        return measuredRatio
    }

    private func auditPlacement() {
        placementRatio = Self.measurePlacement(modelURL: modelURL, configuration: configuration)
    }
}

@_cdecl("bad_apple_coreml_placement_ratio")
public func badAppleCoreMLPlacementRatio(
    _ path: UnsafePointer<CChar>?,
    _ computeUnitsRawValue: Int32
) -> Double {
    guard #available(macOS 15.0, *), let path,
          let computeUnits = MLComputeUnits(rawValue: Int(computeUnitsRawValue)) else {
        return -1.0
    }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    return BadAppleANECore.measurePlacement(
        modelURL: URL(fileURLWithPath: String(cString: path)),
        configuration: configuration
    )
}

@_cdecl("bad_apple_ane_probe_shard")
public func badAppleANEProbeShard(
    _ path: UnsafePointer<CChar>?,
    _ computeUnitsRawValue: Int32,
    _ iterations: Int,
    _ averageLatencyUs: UnsafeMutablePointer<UInt64>?
) -> Bool {
    guard #available(macOS 15.0, *), let path, iterations > 0,
          let computeUnits = MLComputeUnits(rawValue: Int(computeUnitsRawValue)) else {
        return false
    }
    do {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try MLModel(
            contentsOf: URL(fileURLWithPath: String(cString: path)),
            configuration: configuration
        )
        let inputs = model.modelDescription.inputDescriptionsByName
        guard inputs["x"] != nil,
              inputs["rope_cos"] != nil,
              inputs["rope_sin"] != nil,
              inputs["attn_mask"] != nil,
              inputs["kv_write_mask"] != nil,
              let outputName = model.modelDescription.outputDescriptionsByName.first(where: {
                  $0.value.type == .multiArray
              })?.key else {
            return false
        }
        let x = try MLMultiArray(shape: [1, 2048, 1, 1], dataType: .float16)
        let ropeCos = try MLMultiArray(shape: [1, 64], dataType: .float16)
        let ropeSin = try MLMultiArray(shape: [1, 64], dataType: .float16)
        let attnMask = try MLMultiArray(shape: [1, 1, 1, 2048], dataType: .float16)
        let kvWriteMask = try MLMultiArray(shape: [1, 1, 2048, 1], dataType: .float16)
        for index in 0..<64 {
            ropeCos[index] = 1
            ropeSin[index] = 0
        }
        for index in 0..<2048 {
            attnMask[index] = NSNumber(value: index == 0 ? Float(0) : Float(-10_000))
            kvWriteMask[index] = NSNumber(value: index == 0 ? Float(1) : Float(0))
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
            "kv_write_mask": MLFeatureValue(multiArray: kvWriteMask),
        ])
        let state = model.makeState()
        let started = ProcessInfo.processInfo.systemUptime
        for _ in 0..<iterations {
            let prediction = try model.prediction(from: provider, using: state)
            guard prediction.featureValue(for: outputName)?.multiArrayValue != nil else {
                return false
            }
        }
        let elapsed = ProcessInfo.processInfo.systemUptime - started
        averageLatencyUs?.pointee = UInt64(elapsed * 1_000_000 / Double(iterations))
        return true
    } catch {
        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE shard probe failed: \(error)")
        return false
    }
}

@available(macOS 15.0, *)
private struct BadAppleANEShardManifest {
    struct Layer {
        let path: URL
        let start: Int
        let end: Int
    }

    struct Head {
        let path: URL
        let vocabStart: Int
        let vocabEnd: Int
    }

    let layers: [Layer]
    let heads: [Head]
    let embeddingPath: URL
    let hiddenSize: Int
    let vocabSize: Int
    let sequenceLength: Int
    let ropeDimension: Int
    let ropeFrequencyBase: Double

    static func load(from url: URL) throws -> Self {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              root["status"] as? String == "complete",
              let model = root["model"] as? [String: Any],
              let totalLayers = integer(model["total_layers"]),
              let hiddenSize = integer(model["hidden_size"]),
              let vocabSize = integer(model["vocab_size"]),
              let sequenceLength = integer(model["seq_len"]),
              let ropeDimension = integer(model["rope_dim"]),
              let ropeFrequencyBase = number(model["rope_freq_base"]),
              let shardValues = root["shards"] as? [[String: Any]],
              let shared = root["shared"] as? [String: Any],
              let embedding = shared["embedding"] as? [String: Any],
              embedding["status"] as? String == "complete",
              let embeddingPath = embedding["path"] as? String,
              let headValues = shared["lm_head_shards"] as? [[String: Any]] else {
            throw NSError(domain: "BadAppleANEShard", code: 1)
        }

        let base = url.deletingLastPathComponent()
        let layers = try shardValues.map { value -> Layer in
            guard value["status"] as? String == "compiled",
                  let path = value["compiled_path"] as? String,
                  let start = integer(value["layer_start"]),
                  let end = integer(value["layer_end"]) else {
                throw NSError(domain: "BadAppleANEShard", code: 2)
            }
            return Layer(path: resolve(path, relativeTo: base), start: start, end: end)
        }.sorted { $0.start < $1.start }
        guard layers.count == totalLayers else {
            throw NSError(domain: "BadAppleANEShard", code: 3)
        }
        for (index, layer) in layers.enumerated() where layer.start != index || layer.end != index + 1 {
            throw NSError(domain: "BadAppleANEShard", code: 4)
        }

        let heads = try headValues.map { value -> Head in
            guard value["status"] as? String == "compiled",
                  let path = value["compiled_path"] as? String,
                  let start = integer(value["vocab_start"]),
                  let end = integer(value["vocab_end"]) else {
                throw NSError(domain: "BadAppleANEShard", code: 5)
            }
            return Head(path: resolve(path, relativeTo: base), vocabStart: start, vocabEnd: end)
        }.sorted { $0.vocabStart < $1.vocabStart }
        var expectedVocabStart = 0
        for head in heads {
            guard head.vocabStart == expectedVocabStart, head.vocabEnd > head.vocabStart else {
                throw NSError(domain: "BadAppleANEShard", code: 6)
            }
            expectedVocabStart = head.vocabEnd
        }
        guard expectedVocabStart == vocabSize else {
            throw NSError(domain: "BadAppleANEShard", code: 7)
        }

        let resolvedEmbedding = resolve(embeddingPath, relativeTo: base)
        let expectedEmbeddingBytes = vocabSize * hiddenSize * MemoryLayout<Float16>.size
        let actualEmbeddingBytes = try resolvedEmbedding.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard actualEmbeddingBytes == expectedEmbeddingBytes else {
            throw NSError(domain: "BadAppleANEShard", code: 8)
        }
        return Self(
            layers: layers,
            heads: heads,
            embeddingPath: resolvedEmbedding,
            hiddenSize: hiddenSize,
            vocabSize: vocabSize,
            sequenceLength: sequenceLength,
            ropeDimension: ropeDimension,
            ropeFrequencyBase: ropeFrequencyBase
        )
    }

    private static func integer(_ value: Any?) -> Int? {
        (value as? NSNumber)?.intValue
    }

    private static func number(_ value: Any?) -> Double? {
        (value as? NSNumber)?.doubleValue
    }

    private static func resolve(_ path: String, relativeTo base: URL) -> URL {
        if path.hasPrefix("/") {
            return URL(fileURLWithPath: path)
        }
        return base.appendingPathComponent(path)
    }
}

@available(macOS 15.0, *)
private final class BadAppleANELayerShard {
    let model: MLModel
    var state: MLState
    let outputName: String

    init(spec: BadAppleANEShardManifest.Layer, configuration: MLModelConfiguration) throws {
        model = try MLModel(contentsOf: spec.path, configuration: configuration)
        let inputs = model.modelDescription.inputDescriptionsByName
        guard inputs["x"] != nil,
              inputs["rope_cos"] != nil,
              inputs["rope_sin"] != nil,
              inputs["attn_mask"] != nil,
              inputs["kv_write_mask"] != nil,
              let output = model.modelDescription.outputDescriptionsByName.first(where: {
                  $0.value.type == .multiArray
              }) else {
            throw NSError(domain: "BadAppleANEShard", code: 9)
        }
        outputName = output.key
        state = model.makeState()
    }

    func reset() {
        state = model.makeState()
    }

    func predict(
        hidden: MLMultiArray,
        ropeCos: MLMultiArray,
        ropeSin: MLMultiArray,
        attentionMask: MLMultiArray,
        writeMask: MLMultiArray
    ) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: hidden),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attentionMask),
            "kv_write_mask": MLFeatureValue(multiArray: writeMask),
        ])
        let prediction = try model.prediction(from: provider, using: state)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "BadAppleANEShard", code: 10)
        }
        return output
    }
}

@available(macOS 15.0, *)
private final class BadAppleANELMHeadShard {
    let model: MLModel
    let inputName: String
    let outputName: String
    let vocabStart: Int
    let vocabEnd: Int

    init(spec: BadAppleANEShardManifest.Head, configuration: MLModelConfiguration) throws {
        model = try MLModel(contentsOf: spec.path, configuration: configuration)
        guard let input = model.modelDescription.inputDescriptionsByName.first(where: {
            $0.value.type == .multiArray
        }), let output = model.modelDescription.outputDescriptionsByName.first(where: {
            $0.value.type == .multiArray
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 11)
        }
        inputName = input.key
        outputName = output.key
        vocabStart = spec.vocabStart
        vocabEnd = spec.vocabEnd
    }

    func logits(hidden: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: hidden),
        ])
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.count == vocabEnd - vocabStart else {
            throw NSError(domain: "BadAppleANEShard", code: 12)
        }
        return output
    }
}

@available(macOS 15.0, *)
private final class BadAppleANEShardCore {
    let manifest: BadAppleANEShardManifest
    let embeddingData: Data
    let layers: [BadAppleANELayerShard]
    let heads: [BadAppleANELMHeadShard]
    let configuration: MLModelConfiguration
    let ropeCos: MLMultiArray
    let ropeSin: MLMultiArray
    let attentionMask: MLMultiArray
    let writeMask: MLMultiArray
    let lock = NSLock()
    var position = 0
    var placementRatio = -1.0
    var selectedLoadLatency: TimeInterval = 0
    var selectedPrewarmLatency: TimeInterval = 0

    private struct Candidate {
        let core: BadAppleANEShardCore
        let loadLatency: TimeInterval
        let prewarmLatency: TimeInterval
        let decodeLatency: TimeInterval

        var score: TimeInterval {
            // Optimize for steady-state tok/s; load and prewarm are one-time costs.
            decodeLatency + prewarmLatency * 0.01 + loadLatency * 0.0001
        }
    }

    static func load(manifestURL: URL) throws -> BadAppleANEShardCore {
        let manifest = try BadAppleANEShardManifest.load(from: manifestURL)
        let maxPrewarmMilliseconds = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MAX_PREWARM_MS"]
            .flatMap(Double.init) ?? 5_000
        let maxLoadMilliseconds = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MAX_LOAD_MS"]
            .flatMap(Double.init) ?? 45_000
        let fastEnoughDecodeMs = ProcessInfo.processInfo.environment["BADAPPLE_ANE_FAST_DECODE_MS"]
            .flatMap(Double.init) ?? 250
        let fastEnoughDecode = fastEnoughDecodeMs / 1_000.0
        var lastError: Error?
        var accepted: [Candidate] = []

        for (index, computeUnits) in [MLComputeUnits.cpuAndNeuralEngine, .all, .cpuAndGPU].enumerated() {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            let loadStarted = ProcessInfo.processInfo.systemUptime
            do {
                let core = try BadAppleANEShardCore(
                    manifest: manifest,
                    configuration: configuration
                )
                let loadLatency = ProcessInfo.processInfo.systemUptime - loadStarted
                guard loadLatency * 1_000 <= maxLoadMilliseconds else {
                    throw NSError(domain: "BadAppleANEShard", code: 18)
                }
                let prewarmStarted = ProcessInfo.processInfo.systemUptime
                let prewarmed = core.prewarm()
                let prewarmLatency = ProcessInfo.processInfo.systemUptime - prewarmStarted
                guard prewarmed, prewarmLatency * 1_000 <= maxPrewarmMilliseconds else {
                    throw NSError(domain: "BadAppleANEShard", code: 13)
                }
                core.selectedLoadLatency = loadLatency
                core.selectedPrewarmLatency = prewarmLatency
                core.placementRatio = BadAppleANECore.measurePlacement(
                    modelURL: manifest.layers[0].path,
                    configuration: configuration
                )
                let decodeLatency = benchmarkDecode(core)
                let candidate = Candidate(
                    core: core,
                    loadLatency: loadLatency,
                    prewarmLatency: prewarmLatency,
                    decodeLatency: decodeLatency
                )
                NSLog(
                    "🏴‍☠️  BAD APPLE // Sharded ANE candidate rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok, score %.3f",
                    computeUnits.rawValue,
                    loadLatency,
                    prewarmLatency,
                    decodeLatency,
                    candidate.score
                )
                // If the first candidate is already fast enough, avoid the long compile
                // times that GPU/ANE-unrestricted configs can incur on some machines.
                if index == 0, decodeLatency <= fastEnoughDecode {
                    NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE primary candidate fast enough; skipping remaining compute units")
                    return core
                }
                accepted.append(candidate)
            } catch {
                NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE candidate rawValue \(computeUnits.rawValue) failed: \(error)")
                lastError = error
            }
        }

        if let selected = accepted.min(by: { $0.score < $1.score }) {
            NSLog(
                "🏴‍☠️  BAD APPLE // Sharded ANE core selected rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok (score %.3f)",
                selected.core.configuration.computeUnits.rawValue,
                selected.loadLatency,
                selected.prewarmLatency,
                selected.decodeLatency,
                selected.score
            )
            return selected.core
        }

        throw lastError ?? NSError(domain: "BadAppleANEShard", code: 14)
    }

    private static func benchmarkDecode(
        _ core: BadAppleANEShardCore,
        iterations: Int = 5,
        maxStepLatency: TimeInterval = 0.5
    ) -> TimeInterval {
        core.reset()
        defer { core.reset() }

        // A short, arbitrary prompt to warm the stateful KV cache.
        let prompt: [Int32] = [128, 456]
        _ = prompt.withUnsafeBufferPointer { core.predictNext($0.baseAddress!, count: 2) }

        var next: Int32 = 1
        // The first decode step after prefill can be atypically slow; discard it.
        _ = core.predictNext(&next, count: 1)

        var total: TimeInterval = 0
        var measured = 0
        for _ in 0..<iterations {
            let stepStart = ProcessInfo.processInfo.systemUptime
            guard let result = core.predictNext(&next, count: 1) else { break }
            let stepLatency = ProcessInfo.processInfo.systemUptime - stepStart
            if stepLatency > maxStepLatency {
                return 1_000_000.0
            }
            total += stepLatency
            measured += 1
            next = result
        }
        guard measured > 0 else { return 1_000_000.0 }
        return total / TimeInterval(measured)
    }

    private init(
        manifest: BadAppleANEShardManifest,
        configuration: MLModelConfiguration
    ) throws {
        self.manifest = manifest
        self.configuration = configuration
        embeddingData = try Data(contentsOf: manifest.embeddingPath, options: .mappedIfSafe)
        layers = try manifest.layers.map {
            try BadAppleANELayerShard(spec: $0, configuration: configuration)
        }
        heads = try manifest.heads.map {
            try BadAppleANELMHeadShard(spec: $0, configuration: configuration)
        }
        let ropeHalf = manifest.ropeDimension / 2
        ropeCos = try MLMultiArray(
            shape: [1, NSNumber(value: ropeHalf)],
            dataType: .float16
        )
        ropeSin = try MLMultiArray(
            shape: [1, NSNumber(value: ropeHalf)],
            dataType: .float16
        )
        attentionMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: manifest.sequenceLength)],
            dataType: .float16
        )
        writeMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: manifest.sequenceLength), 1],
            dataType: .float16
        )
        resetUnlocked()
    }

    func predictNext(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        guard count > 0, position + count <= manifest.sequenceLength else { return nil }

        // Elevate the calling thread to user-interactive QoS once per generation.
        // This influences scheduling and core placement but does not directly
        // "lock" memory bus lines; it is the highest class available to the process.
        if position == 0 {
            let qosResult = pthread_set_qos_class_self_np(badAppleInteractiveQos, 0)
            if qosResult != 0 {
                NSLog("%@", "🏴‍☠️  BAD APPLE // pthread_set_qos_class_self_np failed: \(qosResult)")
            } else {
                let currentQos = qos_class_self()
                NSLog("%@", "🏴‍☠️  BAD APPLE // QoS elevated to userInteractive (\(currentQos)) on \(Thread.current)")
            }
        }

        do {
            var next: Int32?
            for index in 0..<count {
                next = try processToken(tokens[index], project: index == count - 1)
            }
            return next
        } catch {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE prediction failed and reset state: \(error)")
            resetUnlocked()
            return nil
        }
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        resetUnlocked()
    }

    func prewarm() -> Bool {
        lock.lock()
        defer { lock.unlock() }

        // Apply the same interactive QoS to prewarm so the compiler warms at
        // the same priority class as real inference.
        let qosResult = pthread_set_qos_class_self_np(badAppleInteractiveQos, 0)
        if qosResult == 0 {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Prewarm QoS elevated to userInteractive")
        }

        var token: Int32 = 0
        let result = withUnsafePointer(to: &token) { pointer in
            do {
                return try processToken(pointer.pointee, project: true)
            } catch {
                return nil
            }
        }
        resetUnlocked()
        return result != nil
    }

    private func resetUnlocked() {
        position = 0
        for layer in layers {
            layer.reset()
        }
        for index in 0..<manifest.sequenceLength {
            attentionMask[index] = NSNumber(value: Float(-10_000))
            writeMask[index] = 0
        }
    }

    private func processToken(_ token: Int32, project: Bool) throws -> Int32? {
        let tokenID = Int(token)
        guard tokenID >= 0, tokenID < manifest.vocabSize else {
            throw NSError(domain: "BadAppleANEShard", code: 15)
        }
        let byteOffset = tokenID * manifest.hiddenSize * MemoryLayout<Float16>.size
        return try embeddingData.withUnsafeBytes { bytes -> Int32? in
            guard let base = bytes.baseAddress else {
                throw NSError(domain: "BadAppleANEShard", code: 16)
            }
            let embedding = try MLMultiArray(
                dataPointer: UnsafeMutableRawPointer(mutating: base.advanced(by: byteOffset)),
                shape: [1, NSNumber(value: manifest.hiddenSize), 1, 1],
                dataType: .float16,
                strides: [NSNumber(value: manifest.hiddenSize), 1, 1, 1],
                deallocator: nil
            )
            updatePositionInputs()

            // Execute the multi-shard layer graph on the highest-priority global
            // dispatch queue. This is a synchronous block, so the calling thread
            // waits and no concurrent access to MLMultiArray / MLState occurs.
            var next: Int32?
            try DispatchQueue.global(qos: .userInteractive).sync { [self] in
                var hidden = embedding
                for layer in layers {
                    hidden = try layer.predict(
                        hidden: hidden,
                        ropeCos: ropeCos,
                        ropeSin: ropeSin,
                        attentionMask: attentionMask,
                        writeMask: writeMask
                    )
                }
                position += 1
                if project {
                    next = try argmax(hidden: hidden)
                }
            }
            return next
        }
    }

    private func updatePositionInputs() {
        let ropeHalf = manifest.ropeDimension / 2
        for index in 0..<ropeHalf {
            let exponent = Double(index) / Double(ropeHalf)
            let inverseFrequency = 1.0 / pow(manifest.ropeFrequencyBase, exponent)
            let angle = Double(position) * inverseFrequency
            ropeCos[index] = NSNumber(value: cos(angle))
            ropeSin[index] = NSNumber(value: sin(angle))
        }
        if position > 0 {
            writeMask[position - 1] = 0
        }
        attentionMask[position] = 0
        writeMask[position] = 1
    }

    private func argmax(hidden: MLMultiArray) throws -> Int32 {
        var bestToken = -1
        var bestValue = -Float.infinity
        for head in heads {
            let logits = try head.logits(hidden: hidden)
            logits.withUnsafeMutableBytes { rawBuffer, strides in
                guard let base = rawBuffer.baseAddress else { return }
                let count = logits.count
                let lastStride = strides.last ?? 1
                if logits.dataType == .float16 {
                    let typed = base.bindMemory(to: Float16.self, capacity: count * lastStride)
                    var localBest = -Float16.infinity
                    var localToken = -1
                    for i in 0..<count {
                        let v = typed[i * lastStride]
                        if v > localBest {
                            localBest = v
                            localToken = head.vocabStart + i
                        }
                    }
                    if localToken >= 0 {
                        let v = Float(localBest)
                        if v > bestValue {
                            bestValue = v
                            bestToken = localToken
                        }
                    }
                } else if logits.dataType == .float32 {
                    let typed = base.bindMemory(to: Float.self, capacity: count * lastStride)
                    var localBest = -Float.infinity
                    var localToken = -1
                    for i in 0..<count {
                        let v = typed[i * lastStride]
                        if v > localBest {
                            localBest = v
                            localToken = head.vocabStart + i
                        }
                    }
                    if localToken >= 0, localBest > bestValue {
                        bestValue = localBest
                        bestToken = localToken
                    }
                } else {
                    for i in 0..<count {
                        let v = Float(logits[i].doubleValue)
                        if v > bestValue {
                            bestValue = v
                            bestToken = head.vocabStart + i
                        }
                    }
                }
            }
        }
        guard bestToken >= 0 else {
            throw NSError(domain: "BadAppleANEShard", code: 17)
        }
        return Int32(bestToken)
    }
}

@available(macOS 15.0, *)
private enum BadAppleANEBackend {
    case monolithic(BadAppleANECore)
    case sharded(BadAppleANEShardCore)

    func predictNext(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        switch self {
        case .monolithic(let core): core.predictNext(tokens, count: count)
        case .sharded(let core): core.predictNext(tokens, count: count)
        }
    }

    func reset() {
        switch self {
        case .monolithic(let core): core.reset()
        case .sharded(let core): core.reset()
        }
    }

    func prewarm() -> Bool {
        switch self {
        case .monolithic(let core): core.prewarm()
        case .sharded(let core): core.prewarm()
        }
    }

    var placementRatio: Double {
        switch self {
        case .monolithic(let core): core.placementRatio
        case .sharded(let core): core.placementRatio
        }
    }

    var computeUnitsRawValue: Int32 {
        switch self {
        case .monolithic(let core): Int32(core.configuration.computeUnits.rawValue)
        case .sharded(let core): Int32(core.configuration.computeUnits.rawValue)
        }
    }

    var selectionLatencies: (TimeInterval, TimeInterval) {
        switch self {
        case .monolithic(let core): (core.selectedLoadLatency, core.selectedPrewarmLatency)
        case .sharded(let core): (core.selectedLoadLatency, core.selectedPrewarmLatency)
        }
    }
}

@available(macOS 15.0, *)
private final class BadAppleANEHandle {
    let backend: BadAppleANEBackend

    init(backend: BadAppleANEBackend) {
        self.backend = backend
    }
}

@_cdecl("bad_apple_ane_create")
public func badAppleANECreate(_ path: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard #available(macOS 15.0, *), let path else { return nil }
    do {
        let url = URL(fileURLWithPath: String(cString: path))
        let backend: BadAppleANEBackend
        if url.pathExtension.lowercased() == "json" {
            backend = .sharded(try BadAppleANEShardCore.load(manifestURL: url))
        } else {
            backend = .monolithic(try BadAppleANECore.load(modelURL: url))
        }
        return Unmanaged.passRetained(BadAppleANEHandle(backend: backend)).toOpaque()
    } catch {
        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE backend creation failed: \(error)")
        return nil
    }
}

@_cdecl("bad_apple_ane_destroy")
public func badAppleANEDestroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    if #available(macOS 15.0, *) {
        Unmanaged<BadAppleANEHandle>.fromOpaque(handle).release()
    }
}

@_cdecl("bad_apple_ane_predict_next")
public func badAppleANEPredictNext(
    _ handle: UnsafeMutableRawPointer?,
    _ tokens: UnsafePointer<Int32>?,
    _ count: Int
) -> Int32 {
    guard #available(macOS 15.0, *), let handle, let tokens, count > 0 else { return -1 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.predictNext(tokens, count: count) ?? -1
}

@_cdecl("bad_apple_ane_reset")
public func badAppleANEReset(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 15.0, *), let handle else { return false }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    core.backend.reset()
    return true
}

@_cdecl("bad_apple_ane_prewarm")
public func badAppleANEPrewarm(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 15.0, *), let handle else { return false }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.prewarm()
}

@_cdecl("bad_apple_ane_placement_ratio")
public func badAppleANEPlacementRatio(_ handle: UnsafeMutableRawPointer?) -> Double {
    guard #available(macOS 15.0, *), let handle else { return -1.0 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.placementRatio
}

@_cdecl("bad_apple_ane_compute_units_raw_value")
public func badAppleANEComputeUnitsRawValue(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard #available(macOS 15.0, *), let handle else { return -1 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.computeUnitsRawValue
}

@_cdecl("bad_apple_ane_selection_latency_us")
public func badAppleANESelectionLatencyUs(
    _ handle: UnsafeMutableRawPointer?,
    _ loadLatencyUs: UnsafeMutablePointer<UInt64>?,
    _ prewarmLatencyUs: UnsafeMutablePointer<UInt64>?
) -> Bool {
    guard #available(macOS 15.0, *), let handle else { return false }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    let latencies = core.backend.selectionLatencies
    loadLatencyUs?.pointee = UInt64(latencies.0 * 1_000_000)
    prewarmLatencyUs?.pointee = UInt64(latencies.1 * 1_000_000)
    return true
}
