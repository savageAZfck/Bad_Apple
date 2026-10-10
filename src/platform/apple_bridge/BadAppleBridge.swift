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

    /// Try compute units in preference order and take the first that loads and
    /// prewarms within budget. Only one candidate is resident at a time.
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
                    // First working candidate wins; holding several resident
                    // copies to compare them can exhaust a 16 GB machine.
                    core.auditPlacement()
                    NSLog(
                        "🏴‍☠️  BAD APPLE // ANE core selected computeUnits rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok",
                        computeUnits.rawValue,
                        loadLatency,
                        prewarmLatency,
                        decodeLatency
                    )
                    return core
                } catch {
                    let signature = isCompilerFailure(error) ? " [compiler failure]" : ""
                    NSLog("%@", "🏴‍☠️  BAD APPLE // ANE candidate rawValue \(computeUnits.rawValue) failed\(signature): \(error)")
                    lastError = error
                }
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
            // Copy tokens into a self-owned MLMultiArray to avoid a use-after-free
            // if CoreML retains the array beyond the caller's buffer lifetime.
            let array = try MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .int32)
            for i in 0..<count {
                array[[0, NSNumber(value: i)]] = NSNumber(value: tokens[i])
            }
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
            guard let prediction = try withCoreMLCrashGuard({
                try self.model.prediction(from: provider, using: self.state)
            }) else {
                lastPredictionError = NSError(domain: "BadAppleANE", code: 99,
                    userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during monolithic prediction"])
                return nil
            }
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
                    // const/constexpr ops are compile-time declarations with no
                    // device preference; counting them distorts the ratio.
                    let name = operation.operatorName
                    if name == "const" || name.hasSuffix(".const") || name.contains("constexpr") {
                        continue
                    }
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

/// Debug: per-op-type preferred-device census for a compiled artifact.
/// Writes "optype\ttotal\tane\tcpu\tgpu\tnone" lines to `outPath`.
/// Returns the ANE ratio or -1 on failure.
@_cdecl("bad_apple_coreml_op_placement_report")
public func badAppleCoreMLOpPlacementReport(
    _ path: UnsafePointer<CChar>?,
    _ computeUnitsRawValue: Int32,
    _ outPath: UnsafePointer<CChar>?
) -> Double {
    guard #available(macOS 15.0, *), let path, let outPath,
          let computeUnits = MLComputeUnits(rawValue: Int(computeUnitsRawValue)) else {
        return -1.0
    }
    let modelURL = URL(fileURLWithPath: String(cString: path))
    let reportURL = URL(fileURLWithPath: String(cString: outPath))
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let semaphore = DispatchSemaphore(value: 0)
    var result = -1.0
    Task {
        defer { semaphore.signal() }
        guard let plan = try? await MLComputePlan.load(
            contentsOf: modelURL,
            configuration: configuration
        ), case .program(let program) = plan.modelStructure else {
            return
        }
        var census: [String: [Int]] = [:] // total, ane, cpu, gpu, none
        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                var c = census[operation.operatorName, default: [0, 0, 0, 0, 0]]
                c[0] += 1
                if let usage = plan.deviceUsage(for: operation) {
                    switch usage.preferred {
                    case .neuralEngine: c[1] += 1
                    case .cpu: c[2] += 1
                    case .gpu: c[3] += 1
                    default: c[4] += 1
                    }
                } else {
                    c[4] += 1
                }
                census[operation.operatorName] = c
                for child in operation.blocks { visit(child) }
            }
        }
        for function in program.functions.values { visit(function.block) }
        var lines = ""
        var total = 0
        var ane = 0
        var runtime = 0
        var runtimeAne = 0
        for (ty, c) in census.sorted(by: { $0.value[0] > $1.value[0] }) {
            lines += "\(ty)\t\(c[0])\t\(c[1])\t\(c[2])\t\(c[3])\t\(c[4])\n"
            total += c[0]
            ane += c[1]
            let compileTime = ty == "const" || ty.hasSuffix(".const") || ty.contains("constexpr")
            if !compileTime {
                runtime += c[0]
                runtimeAne += c[1]
            }
        }
        lines += "TOTAL\t\(total)\t\(ane)\t\t\t\n"
        lines += "RUNTIME\t\(runtime)\t\(runtimeAne)\t\t\t\n"
        try? lines.write(to: reportURL, atomically: true, encoding: .utf8)
        result = runtime > 0 ? Double(runtimeAne) / Double(runtime) : -1.0
    }
    guard semaphore.wait(timeout: .now() + .seconds(30)) == .success else {
        return -1.0
    }
    return result
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
        let isHeadShard = inputs["x"] != nil && inputs["rope_cos"] == nil
        guard let xShape = inputs["x"]?.multiArrayConstraint?.shape.map({ $0.intValue }),
              isHeadShard || inputs["rope_cos"] != nil,
              isHeadShard || inputs["rope_sin"] != nil,
              isHeadShard || inputs["attn_mask"] != nil,
              isHeadShard || inputs["kv_write_mask"] != nil || inputs["pos"] != nil,
              let outputName = model.modelDescription.outputDescriptionsByName.first(where: {
                  $0.value.type == .multiArray
              })?.key else {
            return false
        }
        if isHeadShard {
            let x = try MLMultiArray(shape: xShape as [NSNumber], dataType: .float16)
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: x),
            ])
            let state = model.makeState()
            let started = ProcessInfo.processInfo.systemUptime
            for _ in 0..<iterations {
                guard let prediction = try withCoreMLCrashGuard({
                    try model.prediction(from: provider, using: state)
                }) else {
                    return false
                }
                guard prediction.featureValue(for: outputName)?.multiArrayValue != nil else {
                    return false
                }
            }
            let elapsed = ProcessInfo.processInfo.systemUptime - started
            averageLatencyUs?.pointee = UInt64(elapsed * 1_000_000 / Double(iterations))
            return true
        }
        let cosShape = inputs["rope_cos"]!.multiArrayConstraint!.shape.map({ $0.intValue })
        let sinShape = inputs["rope_sin"]!.multiArrayConstraint!.shape.map({ $0.intValue })
        let maskShape = inputs["attn_mask"]!.multiArrayConstraint!.shape.map({ $0.intValue })
        let x = try MLMultiArray(shape: xShape as [NSNumber], dataType: .float16)
        let ropeCos = try MLMultiArray(shape: cosShape as [NSNumber], dataType: .float16)
        let ropeSin = try MLMultiArray(shape: sinShape as [NSNumber], dataType: .float16)
        let attnMask = try MLMultiArray(shape: maskShape as [NSNumber], dataType: .float16)
        for index in 0..<ropeCos.count {
            ropeCos[index] = 1
        }
        for index in 0..<ropeSin.count {
            ropeSin[index] = 0
        }
        // Causal-style probe: permit the current position, mask everything else.
        let maskWidth = maskShape.last ?? 0
        for index in 0..<attnMask.count {
            attnMask[index] = NSNumber(value: index < maskWidth - 1 ? Float(-10_000) : Float(0))
        }
        var dictionary: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: x),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
        ]
        if let slideShape = inputs["attn_mask_slide"]?.multiArrayConstraint?.shape {
            let slideMask = try MLMultiArray(shape: slideShape, dataType: .float16)
            for index in 0..<slideMask.count {
                slideMask[index] = NSNumber(value: index < maskWidth - 1 ? Float(-10_000) : Float(0))
            }
            dictionary["attn_mask_slide"] = MLFeatureValue(multiArray: slideMask)
        }
        if let posShape = inputs["pos"]?.multiArrayConstraint?.shape {
            let pos = try MLMultiArray(shape: posShape, dataType: .int32)
            pos[0] = 0
            dictionary["pos"] = MLFeatureValue(multiArray: pos)
        }
        if let wmShape = inputs["kv_write_mask"]?.multiArrayConstraint?.shape {
            let kvWriteMask = try MLMultiArray(shape: wmShape, dataType: .float16)
            let writeDim = wmShape.count >= 3 ? wmShape[2].intValue : 1
            for index in 0..<kvWriteMask.count {
                // Write at the last position of each broadcast group.
                kvWriteMask[index] = NSNumber(value: index % writeDim == writeDim - 1 ? Float(1) : Float(0))
            }
            dictionary["kv_write_mask"] = MLFeatureValue(multiArray: kvWriteMask)
        }
        // Debug-emit shards may declare extra inputs (weight-as-input
        // bisection, attn_mask_slide): zero-fill any unhandled feature.
        for (name, desc) in inputs where dictionary[name] == nil {
            guard desc.type == .multiArray,
                  let shape = desc.multiArrayConstraint?.shape else { continue }
            let fill = try MLMultiArray(shape: shape, dataType: .float16)
            for index in 0..<fill.count { fill[index] = 0 }
            dictionary[name] = MLFeatureValue(multiArray: fill)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dictionary)
        let state = model.makeState()
        let started = ProcessInfo.processInfo.systemUptime
        for _ in 0..<iterations {
            guard let prediction = try withCoreMLCrashGuard({
                try model.prediction(from: provider, using: state)
            }) else {
                return false
            }
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
        /// Split-FFN artifacts pair each layer shard with a stateless
        /// `ffn_s`/`ffn_p` model computing the projection tail on a real
        /// model input — the monolithic norm→conv chain miscompiles inside
        /// stateful plans on multi-token decode.
        let ffnPath: URL?
    }

    struct Head {
        let path: URL
        let vocabStart: Int
        let vocabEnd: Int
    }

    let layers: [Layer]
    let heads: [Head]
    let prefillLayers: [Layer]
    let prefillChunk: Int
    let embeddingPath: URL
    let hiddenSize: Int
    let vocabSize: Int
    let sequenceLength: Int
    let ropeDimension: Int
    let ropeFrequencyBase: Double
    /// Gemma3 sliding-window layers use a second RoPE base
    /// (`rope_local_base_freq`). When present, rope inputs carry two
    /// rows: 0 = global, 1 = local; shards slice their row in-graph.
    let ropeLocalFrequencyBase: Double?
    /// Sliding-window size for Gemma3 local layers (0 = every layer global).
    /// Sliding shards declare `attn_mask_slide`; the band mask fills here so
    /// the graph stays free of dynamic index arithmetic (plan-build poison).
    let slidingWindow: Int

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
            let ffnPath = (value["ffn_compiled_path"] as? String)
                .map { resolve($0, relativeTo: base) }
            return Layer(path: resolve(path, relativeTo: base), start: start, end: end, ffnPath: ffnPath)
        }.sorted { $0.start < $1.start }
        var expectedLayerStart = 0
        for layer in layers {
            guard layer.start == expectedLayerStart, layer.end > layer.start else {
                throw NSError(domain: "BadAppleANEShard", code: 4)
            }
            expectedLayerStart = layer.end
        }
        guard expectedLayerStart == totalLayers else {
            throw NSError(domain: "BadAppleANEShard", code: 3)
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

        // Optional batched-prefill shards: same layer groups, same state
        // names — they share MLState with the decode shards at call time.
        let prefillChunk = integer(model["prefill_chunk"]) ?? 0
        var prefillLayers: [Layer] = []
        if prefillChunk > 0, let prefillValues = root["shards_prefill"] as? [[String: Any]] {
            prefillLayers = (try? prefillValues.map { value -> Layer in
                guard value["status"] as? String == "compiled",
                      let path = value["compiled_path"] as? String,
                      let start = integer(value["layer_start"]),
                      let end = integer(value["layer_end"]) else {
                    throw NSError(domain: "BadAppleANEShard", code: 2)
                }
                let ffnPath = (value["ffn_compiled_path"] as? String)
                    .map { resolve($0, relativeTo: base) }
                return Layer(path: resolve(path, relativeTo: base), start: start, end: end, ffnPath: ffnPath)
            }.sorted { $0.start < $1.start }) ?? []
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
            prefillLayers: prefillChunk > 0 ? prefillLayers : [],
            prefillChunk: prefillChunk,
            embeddingPath: resolvedEmbedding,
            hiddenSize: hiddenSize,
            vocabSize: vocabSize,
            sequenceLength: sequenceLength,
            ropeDimension: ropeDimension,
            ropeFrequencyBase: ropeFrequencyBase,
            ropeLocalFrequencyBase: number(model["rope_freq_base_local"]),
            slidingWindow: integer(model["sliding_window"]) ?? 0
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

// MARK: - SIGSEGV / SIGBUS crash guard for CoreML prediction
//
// CoreML's ANE E5 plan-build path can SIGSEGV or SIGBUS under resource
// pressure. If that happens while an NSLock is held, the process dies with
// no recovery. This guard installs a per-thread sigaction handler that
// siglongjmps back to the call site, allowing the caller to unlock, log,
// and fail gracefully instead of crashing the whole organism.
//
// SAFETY: siglongjmp from a signal handler is POSIX-defined behavior when
// the jump target is on the same thread's stack and the interrupted code
// is async-signal-safe or we don't care about its state (CoreML internals
// are toast after a SIGSEGV anyway — we just want the process alive).

/// Execute `body` with SIGSEGV/SIGBUS protection. Returns the body's
/// result on success, or `nil` if a crash was caught. The caller is
/// responsible for unlocking any held locks when `nil` is returned.
///
/// Uses the C shim `coreml_crash_guard.c` because Swift rejects
/// `sigsetjmp`/`siglongjmp` (returns_twice attribute). The C layer
/// installs per-thread signal handlers and longjmps back on crash.
private final class CoreMLGuardBody {
    let run: () -> Void
    init(_ run: @escaping () -> Void) { self.run = run }
}

/// Returns nil only when a fault was caught; ordinary errors are rethrown.
private func withCoreMLCrashGuard<T>(_ body: () throws -> T) throws -> T? {
    var result: Result<T, Error>?
    let crashed: Int32 = withoutActuallyEscaping(body) { body in
        let box = CoreMLGuardBody { result = Result { try body() } }
        let ctx = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<CoreMLGuardBody>.fromOpaque(ctx).release() }
        return coreml_guard_run({ ctx in
            Unmanaged<CoreMLGuardBody>.fromOpaque(ctx!).takeUnretainedValue().run()
        }, ctx)
    }
    if crashed != 0 {
        NSLog("%@", "🏴‍☠️  BAD APPLE // CoreML prediction SIGSEGV/SIGBUS caught — failing gracefully")
        return nil
    }
    return try result?.get()
}

@available(macOS 15.0, *)
private final class BadAppleANELayerShard {
    let model: MLModel
    var state: MLState
    let outputName: String
    let attentionMaskRows: Int
    /// Positions per call — the x input's last dim ([1, hidden, 1, s]).
    /// s > 1 shards verify speculative drafts in a single dispatch.
    let slotCount: Int
    /// Packed-KV shards declare a `pos` i32 input and drive slice_update
    /// begin/end in-graph; legacy shards take `kv_write_mask` instead.
    let usesPosInput: Bool
    let posArray: MLMultiArray?
    /// Gemma3 shards containing sliding-window layers declare a second mask
    /// input carrying the lower-bound band restriction.
    let usesSlideMask: Bool

    init(spec: BadAppleANEShardManifest.Layer, configuration: MLModelConfiguration) throws {
        model = try MLModel(contentsOf: spec.path, configuration: configuration)
        let inputs = model.modelDescription.inputDescriptionsByName
        usesPosInput = inputs["pos"] != nil
        usesSlideMask = inputs["attn_mask_slide"] != nil
        let xShape = inputs["x"]?.multiArrayConstraint?.shape
        slotCount = (xShape?.count == 4 ? xShape?[3].intValue : nil) ?? 1
        guard inputs["x"] != nil,
              inputs["rope_cos"] != nil,
              inputs["rope_sin"] != nil,
              let maskShape = inputs["attn_mask"]?.multiArrayConstraint?.shape,
              usesPosInput || inputs["kv_write_mask"] != nil,
              let output = model.modelDescription.outputDescriptionsByName.first(where: {
                  $0.value.type == .multiArray
              }) else {
            throw NSError(domain: "BadAppleANEShard", code: 9)
        }
        outputName = output.key
        attentionMaskRows = maskShape.count == 4 ? maskShape[2].intValue : 1
        posArray = usesPosInput ? try MLMultiArray(shape: [1], dataType: .int32) : nil
        state = model.makeState()
        zeroStateBuffers()
    }

    func reset() {
        state = model.makeState()
        zeroStateBuffers()
    }

    /// MLState buffers are uninitialized memory — garbage (sometimes Inf) in
    /// unwritten KV rows poisons masked attention scores (q*Inf stays Inf
    /// even under a -10000 additive mask). Zero every declared buffer.
    private func zeroStateBuffers() {
        for name in model.modelDescription.stateDescriptionsByName.keys {
            state.withMultiArray(for: name) { array in
                memset(array.dataPointer, 0, array.count * MemoryLayout<Float16>.size)
            }
        }
    }

    func predict(
        hidden: MLMultiArray,
        ropeCos: MLMultiArray,
        ropeSin: MLMultiArray,
        attentionMask: MLMultiArray,
        slideMask: MLMultiArray?,
        writeMask: MLMultiArray?,
        pos: Int
    ) throws -> MLMultiArray {
        var dict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: hidden),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attentionMask),
        ]
        if usesPosInput, let posArray {
            posArray[0] = NSNumber(value: pos)
            dict["pos"] = MLFeatureValue(multiArray: posArray)
        } else if let writeMask {
            dict["kv_write_mask"] = MLFeatureValue(multiArray: writeMask)
        }
        if usesSlideMask {
            guard let slideMask else {
                throw NSError(domain: "BadAppleANEShard", code: 21)
            }
            dict["attn_mask_slide"] = MLFeatureValue(multiArray: slideMask)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider, using: self.state)
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during layer prediction"])
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "BadAppleANEShard", code: 10)
        }
        return output
    }
}

/// A batched-prefill shard: identical layer group to a decode shard but
/// processes `prefillChunk` tokens per call. It owns no KV state of its
/// own — the caller passes the matching decode shard's `MLState` (the
/// converter emits identical state names/shapes, and MLState is shareable
/// across MLModel instances of the same shard).
@available(macOS 15.0, *)
private final class BadAppleANEPrefillShard {
    let model: MLModel
    let outputName: String
    let headsPerKV: Int
    let usesPosInput: Bool
    let posArray: MLMultiArray?
    let usesSlideMask: Bool

    init(path: URL, chunk: Int, configuration: MLModelConfiguration) throws {
        model = try MLModel(contentsOf: path, configuration: configuration)
        let inputs = model.modelDescription.inputDescriptionsByName
        usesPosInput = inputs["pos"] != nil
        usesSlideMask = inputs["attn_mask_slide"] != nil
        guard inputs["x"] != nil,
              inputs["rope_cos"] != nil,
              inputs["rope_sin"] != nil,
              usesPosInput || inputs["kv_write_mask"] != nil,
              let maskShape = inputs["attn_mask"]?.multiArrayConstraint?.shape,
              maskShape.count == 4,
              let output = model.modelDescription.outputDescriptionsByName.first(where: {
                  $0.value.type == .multiArray
              }) else {
            throw NSError(domain: "BadAppleANEShard", code: 9)
        }
        outputName = output.key
        headsPerKV = maskShape[2].intValue / chunk
        posArray = usesPosInput ? try MLMultiArray(shape: [1], dataType: .int32) : nil
    }

    func predict(
        hidden: MLMultiArray,
        ropeCos: MLMultiArray,
        ropeSin: MLMultiArray,
        attentionMask: MLMultiArray,
        slideMask: MLMultiArray?,
        writeMask: MLMultiArray?,
        pos: Int,
        state: MLState
    ) throws -> MLMultiArray {
        var dict: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: hidden),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attentionMask),
        ]
        if usesPosInput, let posArray {
            posArray[0] = NSNumber(value: pos)
            dict["pos"] = MLFeatureValue(multiArray: posArray)
        } else if let writeMask {
            dict["kv_write_mask"] = MLFeatureValue(multiArray: writeMask)
        }
        if usesSlideMask {
            guard let slideMask else {
                throw NSError(domain: "BadAppleANEShard", code: 21)
            }
            dict["attn_mask_slide"] = MLFeatureValue(multiArray: slideMask)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider, using: state)
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during prefill prediction"])
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "BadAppleANEShard", code: 10)
        }
        return output
    }
}

/// Stateless FFN tail of a split layer (converter `split_ffn` artifacts).
/// Consumes the attention shard's `hidden` = concat(res1, fn_norm) output
/// [1, 2d, 1, s] verbatim and emits the completed layer output. Owning the
/// projection convs in a separate model is the fix for the stateful-plan
/// miscompile: a model boundary is the only materialization the compiler
/// cannot fuse across, and conv-on-input is the verified-exact config.
@available(macOS 15.0, *)
private final class BadAppleANEFFNShard {
    let model: MLModel
    let outputName: String

    init(path: URL, configuration: MLModelConfiguration) throws {
        model = try MLModel(contentsOf: path, configuration: configuration)
        guard model.modelDescription.inputDescriptionsByName["x"] != nil,
              let output = model.modelDescription.outputDescriptionsByName
                  .first(where: { $0.value.type == .multiArray }) else {
            throw NSError(domain: "BadAppleANEShard", code: 9)
        }
        outputName = output.key
    }

    func predict(hidden: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: hidden)
        ])
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider)
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during ffn prediction"])
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "BadAppleANEShard", code: 10)
        }
        // The FFN plan returns a 32-strided (channel-blocked) buffer —
        // densify so downstream shards/argmax see a packed array.
        // withUnsafeMutableBytes yields ELEMENT strides (not byte strides).
        let s1 = output.strides.count > 1 ? output.strides[1].intValue : 1
        let s3 = output.strides.count > 3 ? output.strides[3].intValue : 1
        if s1 != 1 || s3 != 1 {
            let n1 = output.shape[1].intValue
            let w = output.shape.count > 3 ? output.shape[3].intValue : 1
            let dense = try MLMultiArray(
                shape: [1, NSNumber(value: n1), 1, NSNumber(value: w)],
                dataType: .float16
            )
            let dst = dense.dataPointer.assumingMemoryBound(to: Float16.self)
            output.withUnsafeMutableBytes { buf, strides in
                let e1 = strides.count > 1 ? strides[1] : 1
                let e3 = strides.count > 3 ? strides[3] : 1
                guard let base = buf.baseAddress else { return }
                let src = base.assumingMemoryBound(to: Float16.self)
                for c in 0..<n1 {
                    for j in 0..<w {
                        dst[c * w + j] = src[c * e1 + j * e3]
                    }
                }
            }
            return dense
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
    /// Positions per call — 1 for classic heads, decodeWidth when the head
    /// was emitted with --decode-width >1 (input [1, d, 1, width]).
    let width: Int

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
        let inShape = input.value.multiArrayConstraint?.shape.map { $0.intValue } ?? []
        width = inShape.count >= 4 ? max(1, inShape[3]) : 1
    }

    func logits(hidden: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: hidden),
        ])
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider)
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during head prediction"])
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.count == vocabEnd - vocabStart else {
            throw NSError(domain: "BadAppleANEShard", code: 12)
        }
        return output
    }

    /// Batched variant: hidden [1, d, 1, width] -> logits [1, vsz, 1, width].
    func logitsBatched(hidden: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputName: MLFeatureValue(multiArray: hidden),
        ])
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider)
        }) else {
            throw NSError(domain: "BadAppleANEShard", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during head prediction"])
        }
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue,
              output.count == (vocabEnd - vocabStart) * width else {
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
    let prefillShards: [BadAppleANEPrefillShard]
    /// Split-FFN tails aligned with `layers`/`prefillShards` (nil when the
    /// artifact is monolithic). Runs right after the paired layer shard.
    let ffnShards: [BadAppleANEFFNShard?]
    let prefillFfnShards: [BadAppleANEFFNShard?]
    let configuration: MLModelConfiguration
    let ropeCos: MLMultiArray
    let ropeSin: MLMultiArray
    let attentionMask: MLMultiArray
    /// Lower-bound band mask for sliding-window layers (Gemma3): zeros on
    /// [q-w+1, q], -10000 elsewhere. Added to the causal mask in-graph.
    let attentionMaskSlide: MLMultiArray?
    /// Legacy shards only — packed-KV shards take `pos` and have no mask.
    let writeMask: MLMultiArray?
    var prefillX: MLMultiArray?
    var prefillRopeCos: MLMultiArray?
    var prefillRopeSin: MLMultiArray?
    var prefillAttnMask: MLMultiArray?
    var prefillSlideMask: MLMultiArray?
    var prefillWriteMask: MLMultiArray?
    /// Positions per decode call when the artifacts were emitted with
    /// `--decode-width` > 1. The batched path chunks hidden rows into
    /// width-W calls carrying per-slot write-mask columns, per-slot RoPE,
    /// and per-(kv-head, slot) causal attention rows — one ANE dispatch
    /// set covers W positions, which is what makes speculative verify
    /// (and fast mesh prefill) cheaper than serial decode.
    let decodeWidth: Int
    var batchX: MLMultiArray?
    var batchRopeCos: MLMultiArray?
    var batchRopeSin: MLMultiArray?
    var batchWriteMask: MLMultiArray?
    /// Reusable [1, hidden, 1, 1] slice buffer for per-slot LM head calls.
    var headSlot: MLMultiArray?
    /// RoPE inverse frequencies — pow() per element per call was measurable.
    var invFreq: [Double]
    /// Second RoPE base for dual-row rope inputs (Gemma3 sliding layers).
    /// Row 0 = `invFreq` (global), row 1 = `invFreqLocal` (local).
    var invFreqLocal: [Double]?
    let lock = NSLock()
    var position = 0
    var placementRatio = -1.0
    var selectedLoadLatency: TimeInterval = 0
    var selectedPrewarmLatency: TimeInterval = 0

    /// `BADAPPLE_ANE_COMPUTE_UNITS`: `ane` (cpuAndNeuralEngine only), `all`
    /// (only `.all`), or unset for ANE-first with `.all` as failure fallback.
    static func candidateComputeUnits() -> [MLComputeUnits] {
        switch ProcessInfo.processInfo.environment["BADAPPLE_ANE_COMPUTE_UNITS"]?.lowercased() {
        case "ane": return [.cpuAndNeuralEngine]
        case "all": return [.all]
        default: return [.cpuAndNeuralEngine, .all]
        }
    }

    static func load(manifestURL: URL) throws -> BadAppleANEShardCore {
        let manifest = try BadAppleANEShardManifest.load(from: manifestURL)
        // A 36-layer FP16 4B measured 80–236 s cold load and 6–14 s prewarm
        // on a 16 GB M-series box; budgets reject broken configs, not slow disks.
        let maxPrewarmMilliseconds = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MAX_PREWARM_MS"]
            .flatMap(Double.init) ?? 60_000
        let maxLoadMilliseconds = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MAX_LOAD_MS"]
            .flatMap(Double.init) ?? 600_000
        var lastError: Error?

        // Each candidate is a full resident copy of every shard (~8 GB for the
        // FP16 4B). Never hold two: a working candidate is taken as-is, even
        // when slow, and the next compute-unit config is tried only after the
        // previous one failed and was released.
        for computeUnits in Self.candidateComputeUnits() {
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
                NSLog(
                    "🏴‍☠️  BAD APPLE // Sharded ANE core selected rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok",
                    computeUnits.rawValue,
                    loadLatency,
                    prewarmLatency,
                    decodeLatency
                )
                return core
            } catch {
                NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE candidate rawValue \(computeUnits.rawValue) failed: \(error)")
                lastError = error
            }
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

        // decode-width > 1 artifacts reject the s=1 token path — benchmark
        // the real batched seam instead: `decodeWidth` hidden rows per call,
        // reported as seconds per position.
        if core.decodeWidth > 1 {
            let rowBytes = core.manifest.hiddenSize * MemoryLayout<Float16>.size
            var bytes = [UInt8](repeating: 0, count: rowBytes * core.decodeWidth)
            _ = bytes.withUnsafeMutableBytes {
                core.forwardHidden($0.baseAddress!, count: $0.count)
            }
            var total: TimeInterval = 0
            var measured = 0
            for _ in 0..<iterations {
                let stepStart = ProcessInfo.processInfo.systemUptime
                let ok = bytes.withUnsafeMutableBytes {
                    core.forwardHidden($0.baseAddress!, count: $0.count) != nil
                }
                let stepLatency = ProcessInfo.processInfo.systemUptime - stepStart
                guard ok else { break }
                total += stepLatency
                measured += 1
            }
            guard measured > 0 else { return 1_000_000.0 }
            return total / TimeInterval(measured) / TimeInterval(core.decodeWidth)
        }

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
        // Load each shard inside an autorelease pool so the ANE E5
        // compiler's intermediate allocations are freed between shards.
        // On memory-constrained machines (16 GB) this prevents the
        // plan-build from exhausting working memory partway through.
        var loadedLayers: [BadAppleANELayerShard] = []
        loadedLayers.reserveCapacity(manifest.layers.count)
        for spec in manifest.layers {
            try autoreleasepool {
                let shard = try BadAppleANELayerShard(spec: spec, configuration: configuration)
                loadedLayers.append(shard)
            }
        }
        layers = loadedLayers
        // Split-FFN tails: one stateless model per layer shard, loaded with
        // the same configuration (they hold the projection weights; ANE
        // runs them because the conv/matmul ops are the ANE-friendly kind).
        var loadedFfn: [BadAppleANEFFNShard?] = []
        loadedFfn.reserveCapacity(manifest.layers.count)
        for spec in manifest.layers {
            if let ffnPath = spec.ffnPath {
                loadedFfn.append(try BadAppleANEFFNShard(path: ffnPath, configuration: configuration))
            } else {
                loadedFfn.append(nil)
            }
        }
        ffnShards = loadedFfn
        var loadedHeads: [BadAppleANELMHeadShard] = []
        loadedHeads.reserveCapacity(manifest.heads.count)
        for spec in manifest.heads {
            try autoreleasepool {
                let head = try BadAppleANELMHeadShard(spec: spec, configuration: configuration)
                loadedHeads.append(head)
            }
        }
        heads = loadedHeads
        // Prefill shards mirror the decode layer grouping; a mismatch in
        // count or layer range silently disables the batched path.
        // Prefill runs on GPU: the s>1 slice_update graph plan-builds on
        // the ANE but its program cancels at processRequest (ios19
        // lowering defect — the same math runs correctly on GPU and
        // shares the decode shards' MLState across compute units).
        // A batched 64-token forward is cheap on GPU regardless.
        let prefillConfiguration = MLModelConfiguration()
        prefillConfiguration.computeUnits = .cpuAndGPU
        var loadedPrefill: [BadAppleANEPrefillShard] = []
        if manifest.prefillLayers.count == manifest.layers.count {
            var aligned = true
            for (index, spec) in manifest.prefillLayers.enumerated() {
                let decode = manifest.layers[index]
                if spec.start != decode.start || spec.end != decode.end {
                    aligned = false
                    break
                }
            }
            if aligned {
                for spec in manifest.prefillLayers {
                    try autoreleasepool {
                        try loadedPrefill.append(BadAppleANEPrefillShard(
                            path: spec.path,
                            chunk: manifest.prefillChunk,
                            configuration: prefillConfiguration
                        ))
                    }
                }
            }
        }
        prefillShards = loadedPrefill
        var loadedPrefillFfn: [BadAppleANEFFNShard?] = []
        if !loadedPrefill.isEmpty {
            for (index, spec) in manifest.prefillLayers.enumerated() {
                if index < loadedPrefill.count, let ffnPath = spec.ffnPath {
                    loadedPrefillFfn.append(try BadAppleANEFFNShard(
                        path: ffnPath,
                        configuration: prefillConfiguration
                    ))
                } else {
                    loadedPrefillFfn.append(nil)
                }
            }
        }
        prefillFfnShards = loadedPrefillFfn
        if !prefillShards.isEmpty {
            let chunk = manifest.prefillChunk
            let headsPerKV = prefillShards[0].headsPerKV
            let ropeRows = manifest.ropeLocalFrequencyBase != nil ? 2 : 1
            prefillX = try MLMultiArray(
                shape: [1, NSNumber(value: manifest.hiddenSize), 1, NSNumber(value: chunk)],
                dataType: .float16
            )
            prefillRopeCos = try MLMultiArray(
                shape: [NSNumber(value: ropeRows), 1, NSNumber(value: manifest.ropeDimension / 2), NSNumber(value: chunk)],
                dataType: .float16
            )
            prefillRopeSin = try MLMultiArray(
                shape: [NSNumber(value: ropeRows), 1, NSNumber(value: manifest.ropeDimension / 2), NSNumber(value: chunk)],
                dataType: .float16
            )
            prefillAttnMask = try MLMultiArray(
                shape: [1, 1, NSNumber(value: headsPerKV * chunk), NSNumber(value: manifest.sequenceLength)],
                dataType: .float16
            )
            prefillSlideMask = prefillShards.contains(where: { $0.usesSlideMask }) ? try MLMultiArray(
                shape: [1, 1, NSNumber(value: headsPerKV * chunk), NSNumber(value: manifest.sequenceLength)],
                dataType: .float16
            ) : nil
            prefillWriteMask = prefillShards[0].usesPosInput ? nil : try MLMultiArray(
                shape: [1, 1, NSNumber(value: manifest.sequenceLength), NSNumber(value: chunk)],
                dataType: .float16
            )
        }
        let ropeRows = manifest.ropeLocalFrequencyBase != nil ? 2 : 1
        let ropeHalf = manifest.ropeDimension / 2
        ropeCos = try MLMultiArray(
            shape: [NSNumber(value: ropeRows), NSNumber(value: ropeHalf)],
            dataType: .float16
        )
        ropeSin = try MLMultiArray(
            shape: [NSNumber(value: ropeRows), NSNumber(value: ropeHalf)],
            dataType: .float16
        )
        // The decode shard may declare a padded attn_mask row axis (the
        // s==1 attention pads queries to a compilable width). Every row
        // carries the identical causal mask.
        let maskRows = layers.first?.attentionMaskRows ?? 1
        attentionMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: maskRows), NSNumber(value: manifest.sequenceLength)],
            dataType: .float16
        )
        attentionMaskSlide = layers.contains(where: { $0.usesSlideMask }) ? try MLMultiArray(
            shape: [1, 1, NSNumber(value: maskRows), NSNumber(value: manifest.sequenceLength)],
            dataType: .float16
        ) : nil
        writeMask = layers[0].usesPosInput ? nil : try MLMultiArray(
            shape: [1, 1, NSNumber(value: manifest.sequenceLength), 1],
            dataType: .float16
        )
        let shardWidth = layers.first?.slotCount ?? 1
        decodeWidth = (shardWidth > 1 && !layers[0].usesPosInput) ? shardWidth : 1
        if decodeWidth > 1 {
            let w = NSNumber(value: decodeWidth)
            batchX = try MLMultiArray(
                shape: [1, NSNumber(value: manifest.hiddenSize), 1, w],
                dataType: .float16
            )
            batchRopeCos = try MLMultiArray(
                shape: [NSNumber(value: ropeRows), 1, NSNumber(value: ropeHalf), w],
                dataType: .float16
            )
            batchRopeSin = try MLMultiArray(
                shape: [NSNumber(value: ropeRows), 1, NSNumber(value: ropeHalf), w],
                dataType: .float16
            )
            batchWriteMask = try MLMultiArray(
                shape: [1, 1, NSNumber(value: manifest.sequenceLength), w],
                dataType: .float16
            )
            headSlot = try MLMultiArray(
                shape: [1, NSNumber(value: manifest.hiddenSize), 1, 1],
                dataType: .float16
            )
        }
        let base = manifest.ropeFrequencyBase
        invFreq = (0..<ropeHalf).map { 1.0 / pow(base, Double($0) / Double(ropeHalf)) }
        if let localBase = manifest.ropeLocalFrequencyBase {
            invFreqLocal = (0..<ropeHalf).map { 1.0 / pow(localBase, Double($0) / Double(ropeHalf)) }
        }
        resetUnlocked()
    }

    /// Every row of the (possibly padded) decode attn_mask carries the
    /// identical causal mask — helper writes value at a key index in all rows.
    private func setAttentionMask(at keyIndex: Int, to value: NSNumber) {
        let seq = manifest.sequenceLength
        let rows = attentionMask.count / seq
        for row in 0..<rows {
            attentionMask[row * seq + keyIndex] = value
        }
    }

    /// Mirror of `setAttentionMask` for the sliding-window band mask —
    /// every row carries the identical [q-w+1, q] band.
    private func setSlideMask(at keyIndex: Int, to value: NSNumber) {
        guard let slide = attentionMaskSlide else { return }
        let seq = manifest.sequenceLength
        let rows = slide.count / seq
        for row in 0..<rows {
            slide[row * seq + keyIndex] = value
        }
    }

    /// Zero-based band fill: marks slide-mask zeros at [hi-w+1, hi].
    private func fillSlideBand(upTo hi: Int) {
        guard attentionMaskSlide != nil else { return }
        let seq = manifest.sequenceLength
        let w = manifest.slidingWindow > 0 ? manifest.slidingWindow : seq
        for index in 0..<seq {
            setSlideMask(at: index, to: NSNumber(value: index <= hi && index > hi - w ? 0 : -10_000))
        }
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

    /// Bulk prompt commit: runs `count` tokens through the batched prefill
    /// shards (or the token loop when the manifest has none), leaving KV
    /// state, masks and `position` exactly as token-by-token would.
    func prefill(_ tokens: UnsafePointer<Int32>, count: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard count > 0, position + count <= manifest.sequenceLength else { return false }
        if prefillShards.isEmpty {
            do {
                for index in 0..<count {
                    _ = try processToken(tokens[index], project: false)
                }
                return true
            } catch {
                NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE token-loop prefill failed and reset state: \(error)")
                resetUnlocked()
                return false
            }
        }
        if position == 0 {
            let qosResult = pthread_set_qos_class_self_np(badAppleInteractiveQos, 0)
            if qosResult == 0 {
                NSLog("%@", "🏴‍☠️  BAD APPLE // QoS elevated to userInteractive (\(qos_class_self())) on \(Thread.current)")
            }
        }
        do {
            var consumed = 0
            while consumed < count {
                let real = min(manifest.prefillChunk, count - consumed)
                try runPrefillChunk(tokens: tokens.advanced(by: consumed), realCount: real)
                consumed += real
            }
            return true
        } catch {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Sharded ANE batched prefill failed and reset state: \(error)")
            resetUnlocked()
            return false
        }
    }

    private func runPrefillChunk(tokens: UnsafePointer<Int32>, realCount: Int) throws {
        guard let x = prefillX, let cosArr = prefillRopeCos, let sinArr = prefillRopeSin,
              let pMask = prefillAttnMask else {
            throw NSError(domain: "BadAppleANEShard", code: 19)
        }
        let pWrite = prefillWriteMask
        let chunk = manifest.prefillChunk
        let seq = manifest.sequenceLength
        let d = manifest.hiddenSize
        let ropeHalf = manifest.ropeDimension / 2
        let hpk = prefillShards[0].headsPerKV
        let pos = position

        // x [1,d,1,P]: embedding row of token j at slot j (transposed);
        // pad slots stay zero.
        x.withUnsafeMutableBytes { dest, _ in
            guard let destBase = dest.baseAddress else { return }
            let xp = destBase.bindMemory(to: Float16.self, capacity: d * chunk)
            memset(xp, 0, d * chunk * MemoryLayout<Float16>.size)
            embeddingData.withUnsafeBytes { src in
                guard let srcBase = src.baseAddress else { return }
                for j in 0..<realCount {
                    let tokenID = Int(tokens[j])
                    guard tokenID >= 0, tokenID < manifest.vocabSize else { continue }
                    let row = srcBase.advanced(by: tokenID * d * MemoryLayout<Float16>.size)
                        .bindMemory(to: Float16.self, capacity: d)
                    for c in 0..<d {
                        xp[c * chunk + j] = row[c]
                    }
                }
                // Pad slots get the last real token's embedding: an all-zero
                // vector explodes through rms_norm (rsqrt(eps) ~ 1.4e3 x) into
                // ~52k activations that overflow fp16 when downstream code
                // reads them. Pad outputs are discarded; pad queries' K/V are
                // never written, so any finite value is safe.
                for j in realCount..<chunk {
                    for c in 0..<d {
                        xp[c * chunk + j] = xp[c * chunk + (realCount - 1)]
                    }
                }
            }
        }

        // rope cos/sin [R,1,rope_half,P] for absolute positions pos..pos+P-1.
        // Row 0 = global base, row 1 = local base (Gemma3 sliding layers).
        cosArr.withUnsafeMutableBytes { cbuf, _ in
            sinArr.withUnsafeMutableBytes { sbuf, _ in
                guard let cp = cbuf.baseAddress?.bindMemory(to: Float16.self, capacity: cosArr.count),
                      let sp = sbuf.baseAddress?.bindMemory(to: Float16.self, capacity: sinArr.count) else { return }
                let rowStride = ropeHalf * chunk
                for index in 0..<ropeHalf {
                    let inverseFrequency = invFreq[index]
                    for j in 0..<chunk {
                        let angle = Double(pos + j) * inverseFrequency
                        cp[index * chunk + j] = Float16(cos(angle))
                        sp[index * chunk + j] = Float16(sin(angle))
                    }
                }
                if let local = invFreqLocal {
                    for index in 0..<ropeHalf {
                        let inverseFrequency = local[index]
                        for j in 0..<chunk {
                            let angle = Double(pos + j) * inverseFrequency
                            cp[rowStride + index * chunk + j] = Float16(cos(angle))
                            sp[rowStride + index * chunk + j] = Float16(sin(angle))
                        }
                    }
                }
            }
        }

        // attn_mask [1,1,hpk*P,seq]: row r = slot*hpk + j, 0 where key
        // index <= pos+slot (causal), -10000 elsewhere; pad slots masked.
        pMask.withUnsafeMutableBytes { mbuf, _ in
            guard let mp = mbuf.baseAddress?.bindMemory(to: Float16.self, capacity: hpk * chunk * seq) else { return }
            for slot in 0..<chunk {
                let allowed = slot < realCount ? pos + slot : -1
                for j in 0..<hpk {
                    let row = mp.advanced(by: (slot * hpk + j) * seq)
                    for key in 0..<seq {
                        row[key] = Float16(key <= allowed ? 0 : -10_000)
                    }
                }
            }
        }

        // attn_mask_slide (sliding-window shards only): same row layout,
        // zeros on the [allowed-w+1, allowed] band — the added lower bound.
        if let pSlide = prefillSlideMask {
            let w = manifest.slidingWindow > 0 ? manifest.slidingWindow : seq
            pSlide.withUnsafeMutableBytes { mbuf, _ in
                guard let mp = mbuf.baseAddress?.bindMemory(to: Float16.self, capacity: hpk * chunk * seq) else { return }
                for slot in 0..<chunk {
                    let allowed = slot < realCount ? pos + slot : -1
                    for j in 0..<hpk {
                        let row = mp.advanced(by: (slot * hpk + j) * seq)
                        for key in 0..<seq {
                            row[key] = Float16(key <= allowed && key > allowed - w ? 0 : -10_000)
                        }
                    }
                }
            }
        }

        // kv_write_mask [1,1,seq,P] (legacy shards only): column slot
        // one-hot at pos+slot for real slots; all-zero for pads.
        if let pWrite {
            pWrite.withUnsafeMutableBytes { wbuf, _ in
                guard let wp = wbuf.baseAddress?.bindMemory(to: Float16.self, capacity: seq * chunk) else { return }
                memset(wp, 0, seq * chunk * MemoryLayout<Float16>.size)
                for slot in 0..<realCount {
                    wp[(pos + slot) * chunk + slot] = 1
                }
            }
        }

        try DispatchQueue.global(qos: .userInteractive).sync { [self] in
            var hidden = x
            for index in 0..<prefillShards.count {
                hidden = try prefillShards[index].predict(
                    hidden: hidden,
                    ropeCos: cosArr,
                    ropeSin: sinArr,
                    attentionMask: pMask,
                    slideMask: prefillSlideMask,
                    writeMask: pWrite,
                    pos: pos,
                    state: layers[index].state
                )
                if index < prefillFfnShards.count, let ffn = prefillFfnShards[index] {
                    hidden = try ffn.predict(hidden: hidden)
                }
            }
        }

        position += realCount
        for index in 0..<position {
            setAttentionMask(at: index, to: 0)
        }
        // Slide band for the next query at `position` — keys aged out of
        // the window must stay masked even though causal just opened them.
        fillSlideBand(upTo: position - 1)
        // Decode writeMask must stay all-zero: the next token's
        // updatePositionInputs clears [position-1] (already 0) and sets
        // [position] — bookkeeping identical to the token loop.
        if let writeMask {
            for index in 0..<seq {
                writeMask[index] = 0
            }
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

        var result: Int32? = nil
        if decodeWidth > 1 {
            // s>1 shards reject the [1,d,1,1] token path — warm the real
            // batched seam instead: one real row + masked pad slots through
            // every layer shard and the head.
            let d = manifest.hiddenSize
            var zeros = [UInt8](repeating: 0, count: d * MemoryLayout<Float16>.size)
            result = zeros.withUnsafeMutableBytes { buf in
                guard let base = buf.baseAddress else { return nil }
                do {
                    let out = try runBatchedChunk(src: base, realCount: 1)
                    return try argmaxWide(hidden: out, slot: 0)
                } catch {
                    return nil
                }
            }
        } else {
            var token: Int32 = 0
            result = withUnsafePointer(to: &token) { pointer in
                do {
                    return try processToken(pointer.pointee, project: true)
                } catch {
                    return nil
                }
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
            setAttentionMask(at: index, to: NSNumber(value: Float(-10_000)))
            setSlideMask(at: index, to: NSNumber(value: Float(-10_000)))
            writeMask?[index] = 0
        }
    }

    /// Rewind the KV frontier for speculative decode: draft tokens that
    /// verification rejected are un-written so the next predict overwrites
    /// them with the accepted continuation. Stale KV at masked positions is
    /// invisible to attention and gets overwritten on re-entry, so the only
    /// state to repair is the masks and the position counter.
    func rewind(to newPosition: Int32) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let target = Int(newPosition)
        guard target >= 0, target <= position else { return false }
        if position > 0 { writeMask?[position - 1] = 0 }
        if target > 0 { writeMask?[target - 1] = 1 }
        for index in target..<position {
            setAttentionMask(at: index, to: NSNumber(value: Float(-10_000)))
        }
        // Rebuild the slide band for the rewound frontier: keys that slid
        // out under the old position but re-enter the window must reopen.
        fillSlideBand(upTo: target - 1)
        position = target
        return true
    }

    private func processToken(_ token: Int32, project: Bool) throws -> Int32? {
        let tokenID = Int(token)
        guard tokenID >= 0, tokenID < manifest.vocabSize else {
            throw NSError(domain: "BadAppleANEShard", code: 15)
        }
        let byteOffset = tokenID * manifest.hiddenSize * MemoryLayout<Float16>.size
        // Copy the embedding row into a self-owned MLMultiArray to avoid a
        // use-after-free if CoreML retains the array beyond the
        // withUnsafeBytes scope (e.g. for async ANE dispatch).
        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: manifest.hiddenSize), 1, 1],
            dataType: .float16
        )
        try embeddingData.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else {
                throw NSError(domain: "BadAppleANEShard", code: 16)
            }
            embedding.withUnsafeMutableBytes { dest, _ in
                guard let destBase = dest.baseAddress else { return }
                memcpy(destBase, base.advanced(by: byteOffset), manifest.hiddenSize * MemoryLayout<Float16>.size)
            }
        }
        updatePositionInputs()

        // Execute the multi-shard layer graph on the highest-priority global
        // dispatch queue. This is a synchronous block, so the calling thread
        // waits and no concurrent access to MLMultiArray / MLState occurs.
        var next: Int32?
        try DispatchQueue.global(qos: .userInteractive).sync { [self] in
            var hidden = embedding
            for (index, layer) in layers.enumerated() {
                hidden = try layer.predict(
                    hidden: hidden,
                    ropeCos: ropeCos,
                    ropeSin: ropeSin,
                    attentionMask: attentionMask,
                    slideMask: attentionMaskSlide,
                    writeMask: writeMask,
                    pos: position
                )
                if index < ffnShards.count, let ffn = ffnShards[index] {
                    hidden = try ffn.predict(hidden: hidden)
                }
            }
            position += 1
            if project {
                next = try argmax(hidden: hidden)
            }
        }
        return next
    }

    private func updatePositionInputs() {
        let ropeHalf = manifest.ropeDimension / 2
        for index in 0..<ropeHalf {
            let angle = Double(position) * invFreq[index]
            ropeCos[index] = NSNumber(value: cos(angle))
            ropeSin[index] = NSNumber(value: sin(angle))
        }
        if let local = invFreqLocal {
            for index in 0..<ropeHalf {
                let angle = Double(position) * local[index]
                ropeCos[ropeHalf + index] = NSNumber(value: cos(angle))
                ropeSin[ropeHalf + index] = NSNumber(value: sin(angle))
            }
        }
        if position > 0 {
            writeMask?[position - 1] = 0
        }
        setAttentionMask(at: position, to: 0)
        setSlideMask(at: position, to: 0)
        // Band for the query at `position` is [position-w+1, position] —
        // the key that aged out of the window gets masked.
        let aged = position - manifest.slidingWindow
        if manifest.slidingWindow > 0, aged >= 0 {
            setSlideMask(at: aged, to: NSNumber(value: Float(-10_000)))
        }
        writeMask?[position] = 1
    }

    /// Mesh-brain tail-rank seam: accept hidden-state rows [n, hidden] fp16
    /// from an upstream rank instead of looking up embeddings locally.
    /// Each row advances the KV frontier; the last row is projected to a
    /// token. Returns nil when the payload is malformed or a shard throws.
    func forwardHidden(_ bytes: UnsafeRawPointer, count: Int) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        let d = manifest.hiddenSize
        let rowBytes = d * MemoryLayout<Float16>.size
        guard count > 0, count % rowBytes == 0 else { return nil }
        let rows = count / rowBytes
        // Overrunning the mask/KV horizon throws an NSException inside CoreML,
        // which no Swift catch can intercept — it would unwind across the FFI
        // and abort the host. Reject oversized batches as a clean nil instead.
        guard rows <= manifest.sequenceLength - position else { return nil }
        if decodeWidth > 1 {
            do {
                var consumed = 0
                var hidden: MLMultiArray?
                var lastReal = 0
                while consumed < rows {
                    let r = min(decodeWidth, rows - consumed)
                    hidden = try runBatchedChunk(
                        src: bytes.advanced(by: consumed * rowBytes), realCount: r)
                    lastReal = r
                    consumed += r
                }
                guard let last = hidden else { return nil }
                return try argmaxWide(hidden: last, slot: lastReal - 1)
            } catch {
                NSLog("%@", "🏴‍☠️  BAD APPLE // Batched ANE forward failed: \(error)")
                return nil
            }
        }
        var next: Int32?
        do {
            for row in 0..<rows {
                let hidden = try MLMultiArray(
                    shape: [1, NSNumber(value: d), 1, 1],
                    dataType: .float16
                )
                memcpy(
                    hidden.dataPointer,
                    bytes.advanced(by: row * rowBytes),
                    rowBytes
                )
                updatePositionInputs()
                try DispatchQueue.global(qos: .userInteractive).sync { [self] in
                    var h = hidden
                    for (index, layer) in layers.enumerated() {
                        h = try layer.predict(
                            hidden: h,
                            ropeCos: ropeCos,
                            ropeSin: ropeSin,
                            attentionMask: attentionMask,
                            slideMask: attentionMaskSlide,
                            writeMask: writeMask,
                            pos: position
                        )
                        if index < ffnShards.count, let ffn = ffnShards[index] {
                            h = try ffn.predict(hidden: h)
                        }
                    }
                    position += 1
                    if row == rows - 1 {
                        next = try argmax(hidden: h)
                    }
                }
            }
        } catch {
            return nil
        }
        return next
    }

    /// Speculative-verify seam: like `forwardHidden` but writes the argmax of
    /// EVERY row into `out` — the K per-position predictions the verifier
    /// compares against the draft continuation. Returns the number of rows
    /// processed (== tokens written, capped at `out.count`), or -1 when the
    /// payload is malformed, exceeds the KV horizon, or a shard throws.
    func forwardHiddenTokens(
        _ bytes: UnsafeRawPointer,
        count: Int,
        out: UnsafeMutableBufferPointer<Int32>
    ) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let d = manifest.hiddenSize
        let rowBytes = d * MemoryLayout<Float16>.size
        guard count > 0, count % rowBytes == 0,
              let outBase = out.baseAddress, !out.isEmpty else { return -1 }
        let rows = count / rowBytes
        guard rows <= manifest.sequenceLength - position else { return -1 }
        let writable = min(rows, out.count)
        do {
            if decodeWidth > 1 {
                // When the head shards were emitted at the same decode width
                // the whole chunk's argmax is one head call per vocab slice —
                // no per-slot gather, no 32-dispatch logits pass.
                let batchedHead = heads.allSatisfy { $0.width == decodeWidth }
                var row = 0
                while row < rows {
                    let r = min(decodeWidth, rows - row)
                    let hidden = try runBatchedChunk(
                        src: bytes.advanced(by: row * rowBytes), realCount: r)
                    let limit = min(r, writable - row)
                    if batchedHead {
                        let toks = try argmaxBatched(hidden: hidden, slots: limit)
                        for slot in 0..<limit {
                            outBase[row + slot] = toks[slot]
                        }
                    } else {
                        for slot in 0..<limit {
                            outBase[row + slot] = try argmaxWide(
                                hidden: hidden, slot: slot)
                        }
                    }
                    row += r
                }
                return rows
            }
            for row in 0..<rows {
                let hidden = try MLMultiArray(
                    shape: [1, NSNumber(value: d), 1, 1],
                    dataType: .float16
                )
                memcpy(hidden.dataPointer, bytes.advanced(by: row * rowBytes), rowBytes)
                updatePositionInputs()
                try DispatchQueue.global(qos: .userInteractive).sync { [self] in
                    var h = hidden
                    for (index, layer) in layers.enumerated() {
                        h = try layer.predict(
                            hidden: h,
                            ropeCos: ropeCos,
                            ropeSin: ropeSin,
                            attentionMask: attentionMask,
                            slideMask: attentionMaskSlide,
                            writeMask: writeMask,
                            pos: position
                        )
                        if index < ffnShards.count, let ffn = ffnShards[index] {
                            h = try ffn.predict(hidden: h)
                        }
                    }
                    position += 1
                    if row < writable {
                        outBase[row] = try argmax(hidden: h)
                    }
                }
            }
            return rows
        } catch {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Batched ANE verify failed: \(error)")
            return -1
        }
    }

    /// One batched decode call: `realCount` rows (<= decodeWidth) starting at
    /// `src` run through every layer shard in a single dispatch each. Slot j
    /// writes KV at position+j, sees keys 0...position+j (strictly causal —
    /// slots never attend to later same-call writes), and gets position+j's
    /// RoPE angle. Pad slots (j >= realCount) get zero write columns, a
    /// fully-masked attention row, and are ignored downstream.
    private func runBatchedChunk(src: UnsafeRawPointer, realCount: Int) throws -> MLMultiArray {
        let d = manifest.hiddenSize
        let seq = manifest.sequenceLength
        let width = decodeWidth
        let pos = position
        let ropeHalf = manifest.ropeDimension / 2
        let maskRows = attentionMask.count / seq
        let hpk = max(1, maskRows / width)
        guard let x = batchX, let cosArr = batchRopeCos, let sinArr = batchRopeSin,
              let wm = batchWriteMask else {
            throw NSError(domain: "BadAppleANEShard", code: 20)
        }
        // x [1, hidden, 1, W]: scatter row j into stride-W lanes. Incoming
        // rows are contiguous [r, hidden] fp16; the shard wants channel-major.
        let dst = x.dataPointer.assumingMemoryBound(to: Float16.self)
        memset(dst, 0, d * width * MemoryLayout<Float16>.size)
        let srcF = src.assumingMemoryBound(to: Float16.self)
        for slot in 0..<realCount {
            for c in 0..<d {
                dst[c * width + slot] = srcF[slot * d + c]
            }
        }
        // kv_write_mask [1, 1, seq, W]: column j one-hot at key pos+j.
        let wmPtr = wm.dataPointer.assumingMemoryBound(to: Float16.self)
        memset(wmPtr, 0, seq * width * MemoryLayout<Float16>.size)
        for slot in 0..<realCount {
            wmPtr[(pos + slot) * width + slot] = 1
        }
        // attn_mask [1, 1, maskRows, seq]: row slot*hpk+h allows keys <= pos+j
        // (pad slots see nothing — output ignored, values stay finite).
        let amPtr = attentionMask.dataPointer.assumingMemoryBound(to: Float16.self)
        for slot in 0..<width {
            let limit = slot < realCount ? pos + slot : -1
            for h in 0..<hpk {
                let base = (slot * hpk + h) * seq
                var key = 0
                while key <= limit { amPtr[base + key] = 0; key += 1 }
                while key < seq { amPtr[base + key] = -10_000; key += 1 }
            }
        }
        // attn_mask_slide: same row layout, zeros on [limit-w+1, limit].
        if let slide = attentionMaskSlide {
            let w = manifest.slidingWindow > 0 ? manifest.slidingWindow : seq
            let smPtr = slide.dataPointer.assumingMemoryBound(to: Float16.self)
            for slot in 0..<width {
                let limit = slot < realCount ? pos + slot : -1
                for h in 0..<hpk {
                    let base = (slot * hpk + h) * seq
                    var key = 0
                    while key < seq {
                        smPtr[base + key] = (key <= limit && key > limit - w) ? 0 : -10_000
                        key += 1
                    }
                }
            }
        }
        // rope_cos/sin [R, 1, ropeHalf, W]: slot j angled at pos+j.
        // Row 0 = global base, row 1 = local base (Gemma3 sliding layers).
        let cosPtr = cosArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let sinPtr = sinArr.dataPointer.assumingMemoryBound(to: Float16.self)
        let rowStride = ropeHalf * width
        for slot in 0..<width {
            let p = pos + min(slot, realCount - 1)
            for i in 0..<ropeHalf {
                let angle = Double(p) * invFreq[i]
                cosPtr[i * width + slot] = Float16(cos(angle))
                sinPtr[i * width + slot] = Float16(sin(angle))
            }
            if let local = invFreqLocal {
                for i in 0..<ropeHalf {
                    let angle = Double(p) * local[i]
                    cosPtr[rowStride + i * width + slot] = Float16(cos(angle))
                    sinPtr[rowStride + i * width + slot] = Float16(sin(angle))
                }
            }
        }
        var thrown: Error?
        var output: MLMultiArray?
        DispatchQueue.global(qos: .userInteractive).sync { [self] in
            var h: MLMultiArray = x
            do {
                for (index, layer) in layers.enumerated() {
                    h = try layer.predict(
                        hidden: h,
                        ropeCos: cosArr,
                        ropeSin: sinArr,
                        attentionMask: attentionMask,
                        slideMask: attentionMaskSlide,
                        writeMask: wm,
                        pos: pos
                    )
                    if index < ffnShards.count, let ffn = ffnShards[index] {
                        h = try ffn.predict(hidden: h)
                    }
                }
                output = h
            } catch {
                thrown = error
            }
        }
        if let thrown { throw thrown }
        guard let output else { throw NSError(domain: "BadAppleANEShard", code: 20) }
        position += realCount
        return output
    }

    /// Slot-aware argmax for W>1 artifacts: width-matched heads read the
    /// whole [1, d, 1, W] output in one pass; width-1 heads keep the
    /// gather-and-scan path. `slot` is the position whose token is wanted.
    private func argmaxWide(hidden: MLMultiArray, slot: Int) throws -> Int32 {
        if heads.allSatisfy({ $0.width == decodeWidth }) {
            return try argmaxBatched(hidden: hidden, slots: slot + 1)[slot]
        }
        return try argmax(hidden: gatherSlot(hidden, slot: slot))
    }

    /// Gather slot j of a batched shard output into `headSlot` — the LM head
    /// shards take [1, hidden, 1, 1] so each position's argmax is a separate
    /// head pass. Handles [1, d, 1, W] outputs and the flat [1, d*W, 1, 1]
    /// stop-state shape (row-major c*W + j either way).
    private func gatherSlot(_ output: MLMultiArray, slot j: Int) throws -> MLMultiArray {
        let d = manifest.hiddenSize
        let width = decodeWidth
        guard let dst = headSlot else {
            throw NSError(domain: "BadAppleANEShard", code: 21)
        }
        let dstPtr = dst.dataPointer.assumingMemoryBound(to: Float16.self)
        var badShape = false
        output.withUnsafeMutableBytes { buf, strides in
            guard let base = buf.baseAddress else {
                badShape = true
                return
            }
            let src = base.assumingMemoryBound(to: Float16.self)
            let dims = output.shape.map { $0.intValue }
            if output.count == d {
                let s1 = strides.count > 1 ? strides[1] : 1
                for c in 0..<d { dstPtr[c] = src[c * s1] }
            } else if dims.count == 4, dims[1] == d, dims[3] == width {
                let s1 = strides.count > 1 ? strides[1] : 1
                let s3 = strides.count > 3 ? strides[3] : 1
                for c in 0..<d { dstPtr[c] = src[c * s1 + j * s3] }
            } else if output.count == d * width {
                for c in 0..<d { dstPtr[c] = src[c * width + j] }
            } else {
                badShape = true
            }
        }
        if badShape { throw NSError(domain: "BadAppleANEShard", code: 21) }
        return dst
    }

    private func argmax(hidden: MLMultiArray) throws -> Int32 {
        var bestToken = -1
        var bestValue = -Float.infinity
        for head in heads {
            let logits = try head.logits(hidden: hidden)
            logits.withUnsafeMutableBytes { rawBuffer, strides in
                guard let base = rawBuffer.baseAddress else { return }
                let count = logits.count
                // The strides from withUnsafeMutableBytes are ELEMENT strides
                // (NOT byte strides) — multiply by element size to get the
                // real byte offset. For (1,V,1,1) the vocab dim is dim 1;
                // on ANE the channel stride can be 32 (= 32 fp16 elements
                // = 64 bytes per tile).
                let elemSize: Int
                switch logits.dataType {
                case .float16: elemSize = 2
                case .float32: elemSize = 4
                default: elemSize = 2
                }
                var elemStride = strides.last ?? 1
                let dims = logits.shape
                for d in 0..<dims.count {
                    if dims[d].intValue == count, d < strides.count {
                        elemStride = strides[d]
                    }
                }
                let byteStride = elemStride * elemSize
                if logits.dataType == .float16 {
                    var localBest = -Float16.infinity
                    var localToken = -1
                    for i in 0..<count {
                        let v = base.load(fromByteOffset: i * byteStride, as: Float16.self)
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
                    var localBest = -Float.infinity
                    var localToken = -1
                    for i in 0..<count {
                        let v = base.load(fromByteOffset: i * byteStride, as: Float.self)
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
                    // Safe fallback: use indexed accessor which handles strides
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

    /// Batched argmax over a [1, vsz, 1, W] head output: one head pass per
    /// vocab slice, then a strided scan picks the best column per slot.
    /// `slots` may be less than the compiled width (final partial chunk) —
    /// pad slots are ignored.
    private func argmaxBatched(hidden: MLMultiArray, slots: Int) throws -> [Int32] {
        var bestValue = [Float](repeating: -.infinity, count: slots)
        var bestToken = [Int32](repeating: -1, count: slots)
        for head in heads {
            let logits = try head.logitsBatched(hidden: hidden)
            let vsz = head.vocabEnd - head.vocabStart
            logits.withUnsafeMutableBytes { buf, strides in
                guard let base = buf.baseAddress else { return }
                let elemSize = logits.dataType == .float32 ? 4 : 2
                // dims [1, vsz, 1, W]: vocab stride = strides[1],
                // slot stride = strides[3]; fall back to dense packing.
                let sV = strides.count > 1 ? strides[1] : decodeWidth
                let sW = strides.count > 3 ? strides[3] : 1
                for j in 0..<slots {
                    var localBest = -Float.infinity
                    var localToken = -1
                    if logits.dataType == .float32 {
                        for i in 0..<vsz {
                            let v = base.load(
                                fromByteOffset: (i * sV + j * sW) * elemSize,
                                as: Float.self)
                            if v > localBest {
                                localBest = v
                                localToken = head.vocabStart + i
                            }
                        }
                    } else {
                        for i in 0..<vsz {
                            let v = base.load(
                                fromByteOffset: (i * sV + j * sW) * elemSize,
                                as: Float16.self)
                            if Float(v) > localBest {
                                localBest = Float(v)
                                localToken = head.vocabStart + i
                            }
                        }
                    }
                    if localToken >= 0, localBest > bestValue[j] {
                        bestValue[j] = localBest
                        bestToken[j] = Int32(localToken)
                    }
                }
            }
        }
        for t in bestToken where t < 0 {
            throw NSError(domain: "BadAppleANEShard", code: 17)
        }
        return bestToken
    }
}

/// A single, unsharded fixed-shape model (`build_fixed_model` output) with the
/// LM head built in. KV caches are explicit fixed-shape inputs/outputs; the
/// output arrays are re-fed as the next step's inputs by reference so the host
/// never copies KV bytes itself.
@available(macOS 15.0, *)
private final class BadAppleANEFixedCore {
    struct Manifest {
        let modelPath: URL
        let embeddingPath: URL
        let hiddenSize: Int
        let vocabSize: Int
        let sequenceLength: Int
        let ropeDimension: Int
        let ropeFrequencyBase: Double
        let totalLayers: Int
        let kvHeads: Int
        let headDimension: Int

        static func load(from url: URL) throws -> Manifest {
            let data = try Data(contentsOf: url)
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let fixedModel = root["fixed_model"] as? String,
                  let embedding = root["embedding"] as? String,
                  let model = root["model"] as? [String: Any],
                  let hiddenSize = (model["hidden_size"] as? NSNumber)?.intValue,
                  let vocabSize = (model["vocab_size"] as? NSNumber)?.intValue,
                  let sequenceLength = (model["seq_len"] as? NSNumber)?.intValue,
                  let ropeDimension = (model["rope_dim"] as? NSNumber)?.intValue,
                  let ropeFrequencyBase = (model["rope_freq_base"] as? NSNumber)?.doubleValue,
                  let totalLayers = (model["total_layers"] as? NSNumber)?.intValue,
                  let kvHeads = (model["n_kv_heads"] as? NSNumber)?.intValue,
                  let headDimension = (model["d_head"] as? NSNumber)?.intValue else {
                throw NSError(domain: "BadAppleANEFixed", code: 1)
            }
            let base = url.deletingLastPathComponent()
            func resolve(_ path: String) -> URL {
                path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
            }
            return Manifest(
                modelPath: resolve(fixedModel),
                embeddingPath: resolve(embedding),
                hiddenSize: hiddenSize,
                vocabSize: vocabSize,
                sequenceLength: sequenceLength,
                ropeDimension: ropeDimension,
                ropeFrequencyBase: ropeFrequencyBase,
                totalLayers: totalLayers,
                kvHeads: kvHeads,
                headDimension: headDimension
            )
        }
    }

    let manifest: Manifest
    let model: MLModel
    let configuration: MLModelConfiguration
    let embeddingData: Data
    let ropeCos: MLMultiArray
    let ropeSin: MLMultiArray
    let attentionMask: MLMultiArray
    let writeMask: MLMultiArray
    var keyCaches: [MLMultiArray] = []
    var valueCaches: [MLMultiArray] = []
    let lock = NSLock()
    var position = 0
    var placementRatio = -1.0
    var selectedLoadLatency: TimeInterval = 0
    var selectedPrewarmLatency: TimeInterval = 0

    static func load(manifestURL: URL) throws -> BadAppleANEFixedCore {
        let manifest = try Manifest.load(from: manifestURL)
        let maxPrewarmMilliseconds = ProcessInfo.processInfo.environment["BADAPPLE_ANE_MAX_PREWARM_MS"]
            .flatMap(Double.init) ?? 60_000
        var lastError: Error?

        // One resident copy at a time — see BadAppleANEShardCore.load.
        for computeUnits in BadAppleANEShardCore.candidateComputeUnits() {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            do {
                let loadStarted = ProcessInfo.processInfo.systemUptime
                let core = try BadAppleANEFixedCore(manifest: manifest, configuration: configuration)
                let loadLatency = ProcessInfo.processInfo.systemUptime - loadStarted
                let prewarmStarted = ProcessInfo.processInfo.systemUptime
                guard core.prewarm() else {
                    throw NSError(domain: "BadAppleANEFixed", code: 2)
                }
                let prewarmLatency = ProcessInfo.processInfo.systemUptime - prewarmStarted
                guard prewarmLatency * 1_000 <= maxPrewarmMilliseconds else {
                    throw NSError(domain: "BadAppleANEFixed", code: 3)
                }
                core.selectedLoadLatency = loadLatency
                core.selectedPrewarmLatency = prewarmLatency
                core.placementRatio = BadAppleANECore.measurePlacement(
                    modelURL: manifest.modelPath,
                    configuration: configuration
                )
                let decodeLatency = benchmarkDecode(core)
                NSLog(
                    "🏴‍☠️  BAD APPLE // Fixed-shape ANE candidate rawValue %ld: load %.3fs, prewarm %.3fs, decode %.3fs/tok",
                    computeUnits.rawValue,
                    loadLatency,
                    prewarmLatency,
                    decodeLatency
                )
                return core
            } catch {
                NSLog("%@", "🏴‍☠️  BAD APPLE // Fixed-shape ANE candidate rawValue \(computeUnits.rawValue) failed: \(error)")
                lastError = error
            }
        }

        throw lastError ?? NSError(domain: "BadAppleANEFixed", code: 4)
    }

    private static func benchmarkDecode(
        _ core: BadAppleANEFixedCore,
        iterations: Int = 5
    ) -> TimeInterval {
        core.reset()
        defer { core.reset() }
        let prompt: [Int32] = [128, 456]
        _ = prompt.withUnsafeBufferPointer { core.predictNext($0.baseAddress!, count: 2) }
        var next: Int32 = 1
        _ = core.predictNext(&next, count: 1)
        var total: TimeInterval = 0
        var measured = 0
        for _ in 0..<iterations {
            let stepStart = ProcessInfo.processInfo.systemUptime
            guard let result = core.predictNext(&next, count: 1) else { break }
            total += ProcessInfo.processInfo.systemUptime - stepStart
            measured += 1
            next = result
        }
        guard measured > 0 else { return 1_000_000.0 }
        return total / TimeInterval(measured)
    }

    private init(manifest: Manifest, configuration: MLModelConfiguration) throws {
        self.manifest = manifest
        self.configuration = configuration
        var modelURL = manifest.modelPath
        if modelURL.pathExtension.lowercased() == "mlpackage" {
            modelURL = try MLModel.compileModel(at: modelURL)
        }
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
        embeddingData = try Data(contentsOf: manifest.embeddingPath, options: .mappedIfSafe)
        let expectedEmbeddingBytes = manifest.vocabSize * manifest.hiddenSize * MemoryLayout<Float16>.size
        guard embeddingData.count == expectedEmbeddingBytes else {
            throw NSError(domain: "BadAppleANEFixed", code: 5)
        }
        let ropeHalf = manifest.ropeDimension / 2
        ropeCos = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
        ropeSin = try MLMultiArray(shape: [1, NSNumber(value: ropeHalf)], dataType: .float16)
        attentionMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: manifest.sequenceLength)],
            dataType: .float16
        )
        writeMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: manifest.sequenceLength), 1],
            dataType: .float16
        )
        try resetUnlocked()
    }

    func predictNext(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        guard count > 0, position + count <= manifest.sequenceLength else { return nil }
        if position == 0 {
            _ = pthread_set_qos_class_self_np(badAppleInteractiveQos, 0)
        }
        do {
            var next: Int32?
            for index in 0..<count {
                next = try processToken(tokens[index], project: index == count - 1)
            }
            return next
        } catch {
            NSLog("%@", "🏴‍☠️  BAD APPLE // Fixed-shape ANE prediction failed and reset state: \(error)")
            try? resetUnlocked()
            return nil
        }
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        try? resetUnlocked()
    }

    func prewarm() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        var token: Int32 = 0
        let result = withUnsafePointer(to: &token) { pointer in
            (try? processToken(pointer.pointee, project: true)) ?? nil
        }
        try? resetUnlocked()
        return result != nil
    }

    /// Rewind the KV frontier for speculative decode — see the sharded
    /// core for the contract; stale entries stay masked until overwritten.
    func rewind(to newPosition: Int32) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let target = Int(newPosition)
        guard target >= 0, target <= position else { return false }
        if position > 0 { writeMask[position - 1] = 0 }
        if target > 0 { writeMask[target - 1] = 1 }
        for index in target..<position {
            attentionMask[index] = NSNumber(value: Float(-10_000))
        }
        position = target
        return true
    }

    private func resetUnlocked() throws {
        position = 0
        for index in 0..<manifest.sequenceLength {
            attentionMask[index] = NSNumber(value: Float(-10_000))
            writeMask[index] = 0
        }
        let shape: [NSNumber] = [
            1,
            NSNumber(value: manifest.kvHeads),
            NSNumber(value: manifest.sequenceLength),
            NSNumber(value: manifest.headDimension),
        ]
        keyCaches = []
        valueCaches = []
        for _ in 0..<manifest.totalLayers {
            let key = try MLMultiArray(shape: shape, dataType: .float16)
            let value = try MLMultiArray(shape: shape, dataType: .float16)
            key.withUnsafeMutableBytes { bytes, _ in _ = bytes.baseAddress.map { memset($0, 0, bytes.count) } }
            value.withUnsafeMutableBytes { bytes, _ in _ = bytes.baseAddress.map { memset($0, 0, bytes.count) } }
            keyCaches.append(key)
            valueCaches.append(value)
        }
    }

    private func processToken(_ token: Int32, project: Bool) throws -> Int32? {
        let tokenID = Int(token)
        guard tokenID >= 0, tokenID < manifest.vocabSize else {
            throw NSError(domain: "BadAppleANEFixed", code: 6)
        }
        let byteOffset = tokenID * manifest.hiddenSize * MemoryLayout<Float16>.size
        // Copy the embedding row into a self-owned MLMultiArray to avoid a
        // use-after-free if CoreML retains the array beyond the
        // withUnsafeBytes scope (e.g. for async ANE dispatch).
        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: manifest.hiddenSize), 1, 1],
            dataType: .float16
        )
        try embeddingData.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else {
                throw NSError(domain: "BadAppleANEFixed", code: 7)
            }
            embedding.withUnsafeMutableBytes { dest, _ in
                guard let destBase = dest.baseAddress else { return }
                memcpy(destBase, base.advanced(by: byteOffset), manifest.hiddenSize * MemoryLayout<Float16>.size)
            }
        }
        updatePositionInputs()

        var features: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: embedding),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attentionMask),
            "kv_write_mask": MLFeatureValue(multiArray: writeMask),
        ]
        for index in 0..<manifest.totalLayers {
            features["k_cache_\(index)"] = MLFeatureValue(multiArray: keyCaches[index])
            features["v_cache_\(index)"] = MLFeatureValue(multiArray: valueCaches[index])
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        guard let prediction = try withCoreMLCrashGuard({
            try self.model.prediction(from: provider)
        }) else {
            throw NSError(domain: "BadAppleANEFixed", code: 99,
                          userInfo: [NSLocalizedDescriptionKey: "SIGSEGV/SIGBUS during fixed-core prediction"])
        }
        // The model emits only the new (1, nkv, 1, dh) KV entry per layer;
        // scatter it into the resident fixed cache at the current position.
        for index in 0..<manifest.totalLayers {
            guard let newKey = prediction.featureValue(for: "new_k_\(index)")?.multiArrayValue,
                  let newValue = prediction.featureValue(for: "new_v_\(index)")?.multiArrayValue else {
                throw NSError(domain: "BadAppleANEFixed", code: 8)
            }
            try scatter(entry: newKey, into: keyCaches[index])
            try scatter(entry: newValue, into: valueCaches[index])
        }
        position += 1
        guard project else { return nil }
        guard let logits = prediction.featureValue(for: "logits")?.multiArrayValue else {
            throw NSError(domain: "BadAppleANEFixed", code: 9)
        }
        return try argmax(logits: logits)
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

    /// Copies a (1, nkv, 1, dh) fp16 entry into a (1, nkv, seq, dh) fp16 cache
    /// at the current position.
    private func scatter(entry: MLMultiArray, into cache: MLMultiArray) throws {
        let heads = manifest.kvHeads
        let headDim = manifest.headDimension
        let sequence = manifest.sequenceLength
        guard entry.dataType == .float16, cache.dataType == .float16,
              entry.count == heads * headDim, cache.count == heads * sequence * headDim else {
            throw NSError(domain: "BadAppleANEFixed", code: 11)
        }
        let currentPosition = position
        entry.withUnsafeBytes { sourceBuffer in
            cache.withUnsafeMutableBytes { destinationBuffer, _ in
                guard let source = sourceBuffer.baseAddress,
                      let destination = destinationBuffer.baseAddress else { return }
                let elementSize = MemoryLayout<Float16>.size
                for head in 0..<heads {
                    let sourceOffset = head * headDim * elementSize
                    let destinationOffset = (head * sequence + currentPosition) * headDim * elementSize
                    memcpy(
                        destination.advanced(by: destinationOffset),
                        source.advanced(by: sourceOffset),
                        headDim * elementSize
                    )
                }
            }
        }
    }

    private func argmax(logits: MLMultiArray) throws -> Int32 {
        var bestToken = -1
        var bestValue = -Float.infinity
        logits.withUnsafeMutableBytes { rawBuffer, strides in
            guard let base = rawBuffer.baseAddress else { return }
            let count = logits.count
            // strides from withUnsafeMutableBytes are ELEMENT strides —
            // multiply by element size to get byte offsets.
            let elemSize: Int
            switch logits.dataType {
            case .float16: elemSize = 2
            case .float32: elemSize = 4
            default: elemSize = 2
            }
            let elemStride = strides.last ?? 1
            let byteStride = elemStride * elemSize
            if logits.dataType == .float16 {
                var localBest = -Float16.infinity
                for i in 0..<count {
                    let v = base.load(fromByteOffset: i * byteStride, as: Float16.self)
                    if v > localBest {
                        localBest = v
                        bestToken = i
                    }
                }
                bestValue = Float(localBest)
            } else if logits.dataType == .float32 {
                for i in 0..<count {
                    let v = base.load(fromByteOffset: i * byteStride, as: Float.self)
                    if v > bestValue {
                        bestValue = v
                        bestToken = i
                    }
                }
            }
        }
        guard bestToken >= 0 else {
            throw NSError(domain: "BadAppleANEFixed", code: 10)
        }
        return Int32(bestToken)
    }
}

@available(macOS 15.0, *)
private enum BadAppleANEBackend {
    case monolithic(BadAppleANECore)
    case sharded(BadAppleANEShardCore)
    case fixedFull(BadAppleANEFixedCore)

    func predictNext(_ tokens: UnsafePointer<Int32>, count: Int) -> Int32? {
        switch self {
        case .monolithic(let core): core.predictNext(tokens, count: count)
        case .sharded(let core): core.predictNext(tokens, count: count)
        case .fixedFull(let core): core.predictNext(tokens, count: count)
        }
    }

    func reset() {
        switch self {
        case .monolithic(let core): core.reset()
        case .sharded(let core): core.reset()
        case .fixedFull(let core): core.reset()
        }
    }

    /// Bulk prompt commit. Only the sharded backend has dedicated
    /// batched-prefill shards; the others fall back to the token loop.
    func prefill(_ tokens: UnsafePointer<Int32>, count: Int) -> Bool {
        switch self {
        case .sharded(let core): return core.prefill(tokens, count: count)
        case .monolithic(let core):
            for index in 0..<count {
                if core.predictNext(tokens.advanced(by: index), count: 1) == nil { return false }
            }
            return true
        case .fixedFull(let core):
            for index in 0..<count {
                if core.predictNext(tokens.advanced(by: index), count: 1) == nil { return false }
            }
            return true
        }
    }

    /// KV rewind for speculative decode. Only the position-indexed
    /// cores support it; the monolithic backend has no partial state.
    func rewind(to newPosition: Int32) -> Bool {
        switch self {
        case .monolithic: return false
        case .sharded(let core): return core.rewind(to: newPosition)
        case .fixedFull(let core): return core.rewind(to: newPosition)
        }
    }

    var position: Int32 {
        switch self {
        case .monolithic: return -1
        case .sharded(let core): return Int32(core.position)
        case .fixedFull(let core): return Int32(core.position)
        }
    }

    func prewarm() -> Bool {
        switch self {
        case .monolithic(let core): core.prewarm()
        case .sharded(let core): core.prewarm()
        case .fixedFull(let core): core.prewarm()
        }
    }

    var placementRatio: Double {
        switch self {
        case .monolithic(let core): core.placementRatio
        case .sharded(let core): core.placementRatio
        case .fixedFull(let core): core.placementRatio
        }
    }

    var computeUnitsRawValue: Int32 {
        switch self {
        case .monolithic(let core): Int32(core.configuration.computeUnits.rawValue)
        case .sharded(let core): Int32(core.configuration.computeUnits.rawValue)
        case .fixedFull(let core): Int32(core.configuration.computeUnits.rawValue)
        }
    }

    var selectionLatencies: (TimeInterval, TimeInterval) {
        switch self {
        case .monolithic(let core): (core.selectedLoadLatency, core.selectedPrewarmLatency)
        case .sharded(let core): (core.selectedLoadLatency, core.selectedPrewarmLatency)
        case .fixedFull(let core): (core.selectedLoadLatency, core.selectedPrewarmLatency)
        }
    }

    /// Mesh-brain tail-rank seam — only the sharded backend can consume
    /// upstream hidden states; monolithic/fixed cores own their embedding.
    func forwardHidden(_ bytes: UnsafeRawPointer, count: Int) -> Int32? {
        switch self {
        case .sharded(let core): core.forwardHidden(bytes, count: count)
        case .monolithic, .fixedFull: nil
        }
    }

    /// Speculative-verify seam — per-row argmaxes. Sharded backend only.
    func forwardHiddenTokens(
        _ bytes: UnsafeRawPointer,
        count: Int,
        out: UnsafeMutableBufferPointer<Int32>
    ) -> Int {
        switch self {
        case .sharded(let core): core.forwardHiddenTokens(bytes, count: count, out: out)
        case .monolithic, .fixedFull: -1
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
            let root = (try? JSONSerialization.jsonObject(with: Data(contentsOf: url))) as? [String: Any]
            if root?["fixed_model"] != nil {
                backend = .fixedFull(try BadAppleANEFixedCore.load(manifestURL: url))
            } else {
                backend = .sharded(try BadAppleANEShardCore.load(manifestURL: url))
            }
        } else {
            backend = .monolithic(try BadAppleANECore.load(modelURL: url))
        }
        return Unmanaged.passRetained(BadAppleANEHandle(backend: backend)).toOpaque()
    } catch {
        NSLog("%@", "🏴‍☠️  BAD APPLE // ANE backend creation failed: \(error)")
        return nil
    }
}

@_cdecl("bad_apple_ane_prefill")
public func badAppleANEPrefill(
    _ handle: UnsafeMutableRawPointer?,
    _ tokens: UnsafePointer<Int32>?,
    _ count: Int32
) -> Bool {
    guard #available(macOS 15.0, *), let handle, let tokens, count > 0 else { return false }
    let h = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return h.backend.prefill(tokens, count: Int(count))
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

/// Mesh-brain tail-rank seam: `bytes` is `count` fp16 bytes forming
/// [n, hidden] rows of upstream hidden states. Each row advances the KV
/// frontier; the last row is projected to a token. -1 on failure.
@_cdecl("bad_apple_ane_forward_hidden")
public func badAppleANEForwardHidden(
    _ handle: UnsafeMutableRawPointer?,
    _ bytes: UnsafeRawPointer?,
    _ count: Int
) -> Int32 {
    guard #available(macOS 15.0, *), let handle, let bytes, count > 0 else { return -1 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.forwardHidden(bytes, count: count) ?? -1
}

/// Speculative-verify seam: same input contract as `bad_apple_ane_forward_hidden`
/// but writes the per-row argmax into `outTokens` (capacity `outCapacity`
/// i32s). Returns the row count processed, or -1 on failure.
@_cdecl("bad_apple_ane_forward_hidden_multi")
public func badAppleANEForwardHiddenMulti(
    _ handle: UnsafeMutableRawPointer?,
    _ bytes: UnsafeRawPointer?,
    _ count: Int,
    _ outTokens: UnsafeMutablePointer<Int32>?,
    _ outCapacity: Int
) -> Int32 {
    guard #available(macOS 15.0, *), let handle, let bytes, count > 0,
          let outTokens, outCapacity > 0 else { return -1 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return Int32(core.backend.forwardHiddenTokens(
        bytes,
        count: count,
        out: UnsafeMutableBufferPointer(start: outTokens, count: outCapacity)
    ))
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

/// Rewind the KV frontier after speculative rejection — the next
/// predict writes the accepted continuation over the discarded drafts.
@_cdecl("bad_apple_ane_rewind")
public func badAppleANERewind(
    _ handle: UnsafeMutableRawPointer?,
    _ position: Int32
) -> Bool {
    guard #available(macOS 15.0, *), let handle else { return false }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.rewind(to: position)
}

/// Current KV frontier — diagnostics for the speculative decode loop.
@_cdecl("bad_apple_ane_position")
public func badAppleANEPosition(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard #available(macOS 15.0, *), let handle else { return -1 }
    let core = Unmanaged<BadAppleANEHandle>.fromOpaque(handle).takeUnretainedValue()
    return core.backend.position
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
