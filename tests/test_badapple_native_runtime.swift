import Foundation
import Darwin

@main
enum BadAppleNativeRuntimeTests {
    static func main() async {
        let runtime = BadAppleNativeRuntime(
            metricWindowSize: 2,
            circuitFailureThreshold: 2,
            circuitRecoverySeconds: 5
        )

        await testModelTransitions(runtime)
        await testModelLoadFailure(runtime)
        testAvailableMemory()
        await testMemoryReservations()
        await testModelOperationGate()
        await testMetrics(runtime)
        await testCircuitBreaker(runtime)
        await testResetAndJSON(runtime)

        print("Bad Apple native runtime tests passed")
    }

    private static func testModelTransitions(_ runtime: BadAppleNativeRuntime) async {
        expect(await runtime.modelStatus("brain") == .unloaded, "unknown model is unloaded")
        await runtime.markModelLoading("brain")
        expect(await runtime.modelStatus("brain") == .loading, "model loading transition")
        await runtime.markModelReady("brain")
        expect(await runtime.modelStatus("brain") == .ready, "model ready transition")
        expect(await runtime.activeModelIDs() == ["brain"], "ready model is active")

        await runtime.markModelFailed("brain", error: "weights rejected")
        expect(await runtime.modelStatus("brain") == .failed, "model failed transition")
        expect(await runtime.activeModelIDs().isEmpty, "failed model is inactive")
        var status = await runtime.runtimeStatus()
        expect(status["last_error"] as? String == "weights rejected", "model failure updates last error")

        await runtime.markModelUnloaded("brain")
        expect(await runtime.modelStatus("brain") == .unloaded, "model unloaded transition")
        status = await runtime.runtimeStatus()
        let models = status["model_status"] as? [String: [String: Any]]
        expect(models?["brain"]?["status"] as? String == "unloaded", "snapshot has model status")
    }

    private static func testModelLoadFailure(_ runtime: BadAppleNativeRuntime) async {
        let failure = "VRAM admission denied: Not enough system memory: 6GB needed, 5GB available"
        await runtime.markModelFailed("brain", error: failure)
        await runtime.markModelFailed("embedding", error: "unrelated failure")
        expect(await runtime.modelLoadError("brain") == "Could not load brain: \(failure)", "load error preserves the selected model's failure")
        await runtime.markModelLoading("brain")
        expect(await runtime.modelLoadError("brain") == "The AI model is still loading. Please wait and try again.", "loading does not report stale failure")
        await runtime.markModelReady("brain")
        expect(await runtime.modelLoadError("brain") == nil, "ready model clears its load error")
        await runtime.markModelUnloaded("brain")
        expect(await runtime.modelLoadError("brain") == "The AI model is not loaded. Please select a cached model and try again.", "unloaded model does not report stale failure")
        expect(await runtime.modelLoadError("unknown") != nil, "unknown model is not ready")
    }

    private static func testAvailableMemory() {
        var stats = vm_statistics64_data_t()
        stats.compressor_page_count = 400
        expect(BadAppleNativeRuntime.availableMemoryBytes(stats, totalBytes: 16_000, pageSize: 16) == 0, "compressed pages are occupied memory")
        stats.free_count = 100
        stats.inactive_count = 200
        stats.speculative_count = 50
        stats.purgeable_count = 25
        expect(BadAppleNativeRuntime.availableMemoryBytes(stats, totalBytes: 16_000, pageSize: 16) == 6_000, "compressed pages do not increase admission headroom")
        expect(BadAppleNativeRuntime.availableMemoryBytes(stats, totalBytes: 1_000, pageSize: 16) == 1_000, "available memory is bounded by physical memory")
        expect(BadAppleNativeRuntime.availableMemoryBytes(stats, totalBytes: UInt64.max, pageSize: UInt64.max) == 0, "overflow cannot admit a model")
    }

    private static func testMemoryReservations() async {
        let runtime = BadAppleNativeRuntime()
        await runtime.setVRAMBudget(16)
        expect(await runtime.reserveModelMemory("main", bytes: 8) == nil, "first model reserves its measured bytes")
        expect(await runtime.reserveModelMemory("fast", bytes: 9) != nil, "second model cannot exceed the shared budget")
        var status = await runtime.vramStatus()
        expect(status["used_bytes"] as? UInt64 == 8, "failed reservation does not consume budget")
        expect(await runtime.canFitModel(estimatedBytes: UInt64.max) != nil, "overflow is rejected rather than crashing")
        await runtime.releaseModelMemory(8)
        status = await runtime.vramStatus()
        expect(status["used_bytes"] as? UInt64 == 0, "unload releases the original measured reservation")
        expect(await runtime.reserveModelMemory("fast", bytes: 16) == nil, "released budget is available again")
    }

    private actor OperationProbe {
        var active = false
        var overlapped = false
        var completed = 0

        func run() async {
            if active { overlapped = true }
            active = true
            await Task.yield()
            completed += 1
            active = false
        }
    }

    private static func testModelOperationGate() async {
        let gate = BadAppleModelOperationGate()
        let probe = OperationProbe()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask { await gate.withLock { await probe.run() } }
            }
        }
        expect(await probe.completed == 50, "all queued model operations complete")
        expect(!(await probe.overlapped), "load and unload cannot overlap across suspensions")
    }

    private static func testMetrics(_ runtime: BadAppleNativeRuntime) async {
        await runtime.clearMetrics()
        await runtime.recordQuery(latencySeconds: 1, tokenCount: 10, succeeded: true)
        await runtime.recordQuery(latencySeconds: 2, tokenCount: 10, succeeded: false, error: "decode failed")
        await runtime.recordQuery(latencySeconds: 4, tokenCount: 8, succeeded: true)

        let status = await runtime.runtimeStatus()
        let counts = requireDictionary(status["query_counts"], "query count dictionary")
        expect(counts["total"] as? Int == 3, "total query count")
        expect(counts["succeeded"] as? Int == 2, "successful query count")
        expect(counts["failed"] as? Int == 1, "failed query count")

        let metrics = requireDictionary(status["rolling_metrics"], "rolling metrics dictionary")
        expect(metrics["sample_count"] as? Int == 2, "rolling window is bounded")
        expect(close(metrics["average_latency_seconds"] as? Double, 3), "rolling average latency")
        expect(close(metrics["average_token_rate"] as? Double, 3.5), "rolling average token rate")
        expect(close(metrics["latest_token_rate"] as? Double, 2), "latest token rate")
        expect(status["last_error"] as? String == "decode failed", "query failure updates last error")

        await runtime.clearMetrics()
        let cleared = await runtime.runtimeStatus()
        let clearedCounts = requireDictionary(cleared["query_counts"], "cleared query counts")
        let clearedMetrics = requireDictionary(cleared["rolling_metrics"], "cleared metrics")
        expect(clearedCounts["total"] as? Int == 0, "clear metrics resets counts")
        expect(clearedMetrics["sample_count"] as? Int == 0, "clear metrics resets samples")
    }

    private static func testCircuitBreaker(_ runtime: BadAppleNativeRuntime) async {
        await runtime.clearCircuits()
        expect(await runtime.allowCircuit("inference", nowUptime: 9), "closed circuit allows request")
        await runtime.recordCircuitFailure("inference", nowUptime: 10)
        expect(await runtime.circuitState("inference") == .closed, "failure below threshold stays closed")
        await runtime.recordCircuitFailure("inference", nowUptime: 11)
        expect(await runtime.circuitState("inference") == .open, "threshold opens circuit")
        expect(!(await runtime.allowCircuit("inference", nowUptime: 15)), "open circuit blocks before cooldown")
        expect(await runtime.allowCircuit("inference", nowUptime: 16), "cooldown permits half-open probe")
        expect(await runtime.circuitState("inference") == .halfOpen, "probe changes circuit to half-open")
        expect(!(await runtime.allowCircuit("inference", nowUptime: 16)), "only one half-open probe is allowed")

        await runtime.recordCircuitFailure("inference", nowUptime: 16)
        expect(await runtime.circuitState("inference") == .open, "failed probe reopens circuit")
        let openStatus = await runtime.runtimeStatus()
        let circuits = openStatus["circuits"] as? [String: [String: Any]]
        expect(circuits?["inference"]?["failures"] as? Int == 2, "failure count is bounded at threshold")

        expect(await runtime.allowCircuit("inference", nowUptime: 21), "second cooldown permits probe")
        await runtime.recordCircuitSuccess("inference")
        expect(await runtime.circuitState("inference") == .closed, "successful probe closes circuit")

        await runtime.resetCircuit("inference")
        expect(await runtime.circuitState("inference") == .closed, "reset removes circuit state")
    }

    private static func testResetAndJSON(_ runtime: BadAppleNativeRuntime) async {
        await runtime.markModelReady("brain")
        await runtime.recordQuery(latencySeconds: 0.5, tokenCount: 5, succeeded: true)
        await runtime.recordCircuitFailure("tools", nowUptime: 1, error: "tool failed")
        var status = await runtime.runtimeStatus()
        expect(JSONSerialization.isValidJSONObject(status), "runtime status is JSON-compatible")

        let memory = requireDictionary(status["memory"], "memory dictionary")
        expect((memory["total_bytes"] as? UInt64 ?? 0) > 0, "physical memory is reported")
        expect(memory["pressure"] is String, "memory pressure is reported")

        await runtime.reset()
        status = await runtime.runtimeStatus()
        expect((status["active_model_ids"] as? [String])?.isEmpty == true, "reset clears active models")
        expect(status["last_error"] is NSNull, "reset clears last error")
        expect((status["circuits"] as? [String: Any])?.isEmpty == true, "reset clears circuits")
        expect(JSONSerialization.isValidJSONObject(status), "reset status remains JSON-compatible")
    }

    private static func requireDictionary(_ value: Any?, _ name: String) -> [String: Any] {
        guard let dictionary = value as? [String: Any] else { fail(name) }
        return dictionary
    }

    private static func close(_ value: Double?, _ expected: Double) -> Bool {
        guard let value else { return false }
        return abs(value - expected) < 0.000_001
    }

    private static func expect(_ condition: Bool, _ name: String) {
        guard condition else { fail(name) }
    }

    private static func fail(_ name: String) -> Never {
        fputs("FAIL: \(name)\n", stderr)
        exit(1)
    }
}
