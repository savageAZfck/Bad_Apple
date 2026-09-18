import Foundation
import Darwin

actor BadAppleModelOperationGate {
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func withLock(_ operation: @Sendable () async -> Void) async {
        if locked {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            locked = true
        }
        await operation()
        if waiters.isEmpty {
            locked = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}

/// Native, process-local ownership of model state, telemetry, memory pressure, and
/// capability circuit breakers. Actor isolation makes every snapshot internally
/// consistent without requiring callers to provide their own locking.
public actor BadAppleNativeRuntime {
    public enum ModelStatus: String, Sendable {
        case loading
        case ready
        case failed
        case unloaded
    }

    public enum CircuitState: String, Sendable {
        case closed
        case open
        case halfOpen = "half-open"
    }

    private struct ModelRecord: Sendable {
        var status: ModelStatus
        var error: String?
    }

    private struct MetricSample: Sendable {
        let latencySeconds: Double
        let tokenRate: Double
    }

    private struct CircuitBreaker: Sendable {
        let failureThreshold: Int
        let recoverySeconds: TimeInterval
        var failures = 0
        var openedAt: TimeInterval?
        var halfOpenProbeInFlight = false

        var state: CircuitState {
            if openedAt == nil { return .closed }
            return halfOpenProbeInFlight ? .halfOpen : .open
        }

        mutating func allow(now: TimeInterval) -> Bool {
            guard let openedAt else { return true }
            guard now - openedAt >= recoverySeconds, !halfOpenProbeInFlight else {
                return false
            }
            halfOpenProbeInFlight = true
            return true
        }

        mutating func recordSuccess() {
            failures = 0
            openedAt = nil
            halfOpenProbeInFlight = false
        }

        mutating func recordFailure(now: TimeInterval) {
            halfOpenProbeInFlight = false
            failures = min(failureThreshold, failures + 1)
            if failures >= failureThreshold {
                openedAt = now
            }
        }

        func retryAfter(now: TimeInterval) -> TimeInterval {
            guard let openedAt, !halfOpenProbeInFlight else { return 0 }
            return max(0, recoverySeconds - (now - openedAt))
        }
    }

    private struct MemorySnapshot: Sendable {
        let usedBytes: UInt64
        let totalBytes: UInt64
        let ratio: Double
        let pressure: String
    }

    private let metricWindowSize: Int
    private let defaultFailureThreshold: Int
    private let defaultRecoverySeconds: TimeInterval
    private var models: [String: ModelRecord] = [:]
    private var samples: [MetricSample] = []
    private var totalQueries = 0
    private var successfulQueries = 0
    private var failedQueries = 0
    private var lastError: String?
    private var breakers: [String: CircuitBreaker] = [:]

    // MARK: - Hibernation

    /// Idle threshold (in seconds) before the runtime recommends hibernation.
    public static let defaultIdleThreshold: TimeInterval = 300
    private let idleThreshold: TimeInterval
    private var lastActivityTime: TimeInterval = ProcessInfo.processInfo.systemUptime
    private var isHibernating = false

    // MARK: - VRAM Governor

    /// Estimated memory budget for all loaded models (in bytes).
    private var vramBudgetBytes: UInt64 = 0
    /// Estimated memory used by currently loaded models (in bytes).
    private var vramUsedBytes: UInt64 = 0
    /// Memory reserved for safety/audit/runtime governance subsystems (bytes).
    private var safetyReservationBytes: UInt64 = 0
    /// Memory reserved for vision/VLM inferences (bytes).
    private var visionReservationBytes: UInt64 = 0
    /// Whether VRAM admission checks are enforced.
    private var vramAdmissionEnabled = true

    // MARK: - Health Checks

    public struct HealthCheck: Sendable {
        public let name: String
        public let level: String  // "liveness", "readiness", "correctness"
        public let ok: Bool
        public let detail: String
    }
    private var healthChecks: [String: HealthCheck] = [:]

    public init(
        metricWindowSize: Int = 100,
        circuitFailureThreshold: Int = 3,
        circuitRecoverySeconds: TimeInterval = 30,
        idleThreshold: TimeInterval = BadAppleNativeRuntime.defaultIdleThreshold
    ) {
        self.metricWindowSize = min(10_000, max(1, metricWindowSize))
        self.defaultFailureThreshold = min(100, max(1, circuitFailureThreshold))
        self.defaultRecoverySeconds = min(86_400, max(0.001, circuitRecoverySeconds))
        self.idleThreshold = min(86_400, max(1, idleThreshold))
        // Default VRAM budget: 80% of physical memory.
        self.vramBudgetBytes = UInt64(Double(ProcessInfo.processInfo.physicalMemory) * 0.8)
        // Reserved memory pools for safety/audit and vision/VLM workloads.
        let env = ProcessInfo.processInfo.environment
        if let raw = env["BADAPPLE_SAFETY_MEMORY_MB"], let mb = UInt64(raw) {
            self.safetyReservationBytes = mb * 1_048_576
        } else {
            // Default 512 MB for governance subsystems.
            self.safetyReservationBytes = 512 * 1_048_576
        }
        if let raw = env["BADAPPLE_VISION_MEMORY_MB"], let mb = UInt64(raw) {
            self.visionReservationBytes = mb * 1_048_576
        } else {
            // Default 768 MB for VLM workloads.
            self.visionReservationBytes = 768 * 1_048_576
        }
    }

    // MARK: Model lifecycle

    public func markModelLoading(_ modelID: String) {
        guard !modelID.isEmpty else { return }
        models[modelID] = ModelRecord(status: .loading, error: nil)
    }

    public func markModelReady(_ modelID: String) {
        guard !modelID.isEmpty else { return }
        models[modelID] = ModelRecord(status: .ready, error: nil)
    }

    public func markModelFailed(_ modelID: String, error: String) {
        guard !modelID.isEmpty else { return }
        models[modelID] = ModelRecord(status: .failed, error: error)
        lastError = error
    }

    public func markModelUnloaded(_ modelID: String) {
        guard !modelID.isEmpty else { return }
        models[modelID] = ModelRecord(status: .unloaded, error: nil)
    }

    public func modelStatus(_ modelID: String) -> ModelStatus {
        models[modelID]?.status ?? .unloaded
    }

    public func modelLoadError(_ modelID: String) -> String? {
        switch models[modelID]?.status ?? .unloaded {
        case .ready:
            return nil
        case .loading:
            return "The AI model is still loading. Please wait and try again."
        case .failed:
            return "Could not load \(modelID): \(models[modelID]?.error ?? "Unknown loading error")"
        case .unloaded:
            return "The AI model is not loaded. Please select a cached model and try again."
        }
    }

    public func activeModelIDs() -> [String] {
        models.compactMap { $0.value.status == .ready ? $0.key : nil }.sorted()
    }

    public func clearModel(_ modelID: String) {
        models.removeValue(forKey: modelID)
    }

    public func clearModels() {
        models.removeAll(keepingCapacity: false)
    }

    // MARK: Query telemetry

    public func recordQuery(latencySeconds: Double, tokenCount: Int, succeeded: Bool, error: String? = nil) {
        totalQueries += 1
        if succeeded {
            successfulQueries += 1
        } else {
            failedQueries += 1
            if let error, !error.isEmpty { lastError = error }
        }

        markActivity()

        guard latencySeconds.isFinite, latencySeconds >= 0 else { return }
        let safeTokens = max(0, tokenCount)
        let tokenRate = latencySeconds > 0 ? Double(safeTokens) / latencySeconds : 0
        samples.append(MetricSample(latencySeconds: latencySeconds, tokenRate: tokenRate))
        if samples.count > metricWindowSize {
            samples.removeFirst(samples.count - metricWindowSize)
        }
    }

    // MARK: Hibernation

    /// Mark activity (resets the idle timer and exits hibernation).
    public func markActivity() {
        lastActivityTime = ProcessInfo.processInfo.systemUptime
        if isHibernating {
            isHibernating = false
        }
    }

    /// Check if the runtime has been idle long enough to hibernate.
    public func shouldHibernate(nowUptime: TimeInterval? = nil) -> Bool {
        let now = nowUptime ?? ProcessInfo.processInfo.systemUptime
        return !isHibernating && (now - lastActivityTime) >= idleThreshold
    }

    /// Mark the runtime as hibernating.
    public func enterHibernation() {
        isHibernating = true
    }

    /// Whether the runtime is currently hibernating.
    public var hibernating: Bool {
        isHibernating
    }

    /// Seconds since the last activity.
    public func idleSeconds(nowUptime: TimeInterval? = nil) -> TimeInterval {
        let now = nowUptime ?? ProcessInfo.processInfo.systemUptime
        return max(0, now - lastActivityTime)
    }

    // MARK: VRAM Governor

    /// Set the VRAM budget in bytes.
    public func setVRAMBudget(_ bytes: UInt64) {
        vramBudgetBytes = bytes
    }

    /// Set the reserved safety/audit memory in bytes.
    public func setSafetyReservation(_ bytes: UInt64) {
        safetyReservationBytes = bytes
    }

    /// Set the reserved vision/VLM memory in bytes.
    public func setVisionReservation(_ bytes: UInt64) {
        visionReservationBytes = bytes
    }

    /// Track memory used by a loaded model.
    public func trackModelMemory(_ modelID: String, bytes: UInt64) {
        vramUsedBytes += bytes
    }

    /// Release memory tracked for a model.
    public func releaseModelMemory(_ bytes: UInt64) {
        vramUsedBytes = vramUsedBytes > bytes ? vramUsedBytes - bytes : 0
    }

    /// Check if a model with the given estimated size can be loaded within the VRAM budget.
    /// Returns nil if admitted, or an error message explaining why not.
    public func canFitModel(estimatedBytes: UInt64) -> String? {
        guard vramAdmissionEnabled else { return nil }
        let memory = Self.readMemorySnapshot()
        let availableSystem = memory.totalBytes > memory.usedBytes ? memory.totalBytes - memory.usedBytes : 0
        let reserved = safetyReservationBytes + visionReservationBytes
        let effectiveBudget = vramBudgetBytes > reserved ? vramBudgetBytes - reserved : 0
        let effectiveSystem = availableSystem > reserved ? availableSystem - reserved : 0
        let projected = vramUsedBytes.addingReportingOverflow(estimatedBytes)
        guard !projected.overflow else { return "Model memory requirements exceed the VRAM budget" }
        if projected.partialValue > effectiveBudget {
            return String(format: "Model would exceed effective VRAM budget: %.2f GiB projected vs %.2f GiB budget (%.2f GiB reserved)", Double(projected.partialValue) / 1_073_741_824, Double(effectiveBudget) / 1_073_741_824, Double(reserved) / 1_073_741_824)
        }
        if estimatedBytes > effectiveSystem {
            return String(format: "Not enough free memory to load the model — it needs %.2f GiB but only %.2f GiB is available (%.2f GiB held in reserve). Close some apps to free memory, then try again; Bad Apple loads the model automatically on the next query.", Double(estimatedBytes) / 1_073_741_824, Double(effectiveSystem) / 1_073_741_824, Double(reserved) / 1_073_741_824)
        }
        return nil
    }

    public func reserveModelMemory(_ modelID: String, bytes: UInt64) -> String? {
        if let reason = canFitModel(estimatedBytes: bytes) { return reason }
        trackModelMemory(modelID, bytes: bytes)
        return nil
    }

    /// Toggle VRAM admission checks.
    public func setVRAMAdmission(enabled: Bool) {
        vramAdmissionEnabled = enabled
    }

    /// Current VRAM usage summary.
    public func vramStatus() -> [String: Any] {
        let budgetGB = Double(vramBudgetBytes) / 1_073_741_824
        let usedGB = Double(vramUsedBytes) / 1_073_741_824
        let reservedGB = Double(safetyReservationBytes + visionReservationBytes) / 1_073_741_824
        return [
            "budget_bytes": vramBudgetBytes,
            "used_bytes": vramUsedBytes,
            "budget_gb": budgetGB,
            "used_gb": usedGB,
            "reserved_gb": reservedGB,
            "effective_budget_gb": budgetGB - reservedGB,
            "ratio": vramBudgetBytes > 0 ? Double(vramUsedBytes) / Double(vramBudgetBytes) : 0,
            "admission_enabled": vramAdmissionEnabled
        ]
    }

    // MARK: Health Checks

    /// Register or update a health check.
    public func registerHealthCheck(_ check: HealthCheck) {
        healthChecks[check.name] = check
    }

    /// Get all registered health checks.
    public func healthCheckResults() -> [HealthCheck] {
        Array(healthChecks.values).sorted { $0.name < $1.name }
    }

    /// Run built-in health checks and update their status.
    public func runBuiltinHealthChecks() {
        // Process check (always passes — we're running).
        registerHealthCheck(HealthCheck(name: "process", level: "liveness", ok: true, detail: "running"))

        // Memory pressure check.
        let memory = Self.readMemorySnapshot()
        let memoryOK = memory.ratio < 0.95
        registerHealthCheck(HealthCheck(
            name: "memory",
            level: "readiness",
            ok: memoryOK,
            detail: memory.pressure
        ))

        // Model readiness check.
        let hasReadyModel = models.values.contains { $0.status == .ready }
        registerHealthCheck(HealthCheck(
            name: "main_model",
            level: "readiness",
            ok: hasReadyModel,
            detail: hasReadyModel ? "loaded" : "not loaded"
        ))

        // Software canary heartbeat: touch a timestamp file the supervisor can watch.
        let canaryPath = "/var/lib/bad_apple/.canary"
        let canaryData = "\(Date().timeIntervalSince1970)\n".data(using: .utf8)
        let canaryOK: Bool
        if let data = canaryData {
            FileManager.default.createFile(atPath: canaryPath, contents: data, attributes: nil)
            canaryOK = FileManager.default.fileExists(atPath: canaryPath)
        } else {
            canaryOK = false
        }
        registerHealthCheck(HealthCheck(
            name: "canary",
            level: "liveness",
            ok: canaryOK,
            detail: canaryOK ? "heartbeat written" : "heartbeat failed"
        ))

        // Audit ledger check (verify chain integrity).
        let ledger = BadAppleAuditLedger()
        let ledgerOK = ledger.verify()
        registerHealthCheck(HealthCheck(
            name: "audit_ledger",
            level: "correctness",
            ok: ledgerOK,
            detail: ledgerOK ? "chain valid" : "chain broken"
        ))
    }

    public func setLastError(_ error: String?) {
        lastError = error?.isEmpty == true ? nil : error
    }

    public func clearLastError() {
        lastError = nil
    }

    public func clearMetrics() {
        samples.removeAll(keepingCapacity: false)
        totalQueries = 0
        successfulQueries = 0
        failedQueries = 0
    }

    // MARK: Circuit breakers

    public func configureCircuit(
        _ name: String,
        failureThreshold: Int? = nil,
        recoverySeconds: TimeInterval? = nil
    ) {
        guard !name.isEmpty else { return }
        let threshold = min(100, max(1, failureThreshold ?? defaultFailureThreshold))
        let recovery = min(86_400, max(0.001, recoverySeconds ?? defaultRecoverySeconds))
        breakers[name] = CircuitBreaker(failureThreshold: threshold, recoverySeconds: recovery)
    }

    public func allowCircuit(_ name: String, nowUptime: TimeInterval? = nil) -> Bool {
        guard !name.isEmpty else { return false }
        ensureCircuit(name)
        let now = nowUptime ?? ProcessInfo.processInfo.systemUptime
        return breakers[name]!.allow(now: now)
    }

    public func recordCircuitSuccess(_ name: String) {
        guard !name.isEmpty else { return }
        ensureCircuit(name)
        breakers[name]!.recordSuccess()
    }

    public func recordCircuitFailure(_ name: String, nowUptime: TimeInterval? = nil, error: String? = nil) {
        guard !name.isEmpty else { return }
        ensureCircuit(name)
        let now = nowUptime ?? ProcessInfo.processInfo.systemUptime
        breakers[name]!.recordFailure(now: now)
        if let error, !error.isEmpty { lastError = error }
    }

    public func circuitState(_ name: String) -> CircuitState {
        breakers[name]?.state ?? .closed
    }

    public func resetCircuit(_ name: String) {
        breakers.removeValue(forKey: name)
    }

    public func clearCircuits() {
        breakers.removeAll(keepingCapacity: false)
    }

    // MARK: Resource status

    /// Returns only JSONSerialization-compatible keys and values.
    public func runtimeStatus() -> [String: Any] {
        let now = ProcessInfo.processInfo.systemUptime
        let memory = Self.readMemorySnapshot()
        let modelStatuses = Dictionary(uniqueKeysWithValues: models.map { key, value in
            (key, [
                "status": value.status.rawValue,
                "error": value.error ?? NSNull()
            ] as [String: Any])
        })
        let circuitStatuses = Dictionary(uniqueKeysWithValues: breakers.map { key, breaker in
            (key, [
                "state": breaker.state.rawValue,
                "failures": breaker.failures,
                "failure_threshold": breaker.failureThreshold,
                "recovery_seconds": breaker.recoverySeconds,
                "retry_after_seconds": breaker.retryAfter(now: now)
            ] as [String: Any])
        })

        let latencyValues = samples.map(\.latencySeconds)
        let rateValues = samples.map(\.tokenRate)
        return [
            "model_status": modelStatuses,
            "active_model_ids": activeModelIDs(),
            "last_error": lastError ?? NSNull(),
            "query_counts": [
                "total": totalQueries,
                "succeeded": successfulQueries,
                "failed": failedQueries
            ],
            "rolling_metrics": [
                "window_size": metricWindowSize,
                "sample_count": samples.count,
                "average_latency_seconds": Self.average(latencyValues),
                "average_token_rate": Self.average(rateValues),
                "latest_latency_seconds": latencyValues.last ?? 0,
                "latest_token_rate": rateValues.last ?? 0
            ],
            "memory": [
                "used_bytes": memory.usedBytes,
                "total_bytes": memory.totalBytes,
                "used_ratio": memory.ratio,
                "pressure": memory.pressure
            ],
            "circuits": circuitStatuses,
            "hibernation": [
                "active": isHibernating,
                "idle_seconds": idleSeconds(nowUptime: now),
                "idle_threshold_seconds": idleThreshold
            ],
            "vram": vramStatus(),
            "health_checks": Dictionary(uniqueKeysWithValues: healthChecks.map { name, check in
                (name, [
                    "level": check.level,
                    "ok": check.ok,
                    "detail": check.detail
                ] as [String: Any])
            })
        ]
    }

    public func reset() {
        clearModels()
        clearMetrics()
        clearCircuits()
        clearLastError()
        vramUsedBytes = 0
        healthChecks.removeAll()
    }

    private func ensureCircuit(_ name: String) {
        if breakers[name] == nil {
            breakers[name] = CircuitBreaker(
                failureThreshold: defaultFailureThreshold,
                recoverySeconds: defaultRecoverySeconds
            )
        }
    }

    private static func average(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    static func availableMemoryBytes(_ statistics: vm_statistics64_data_t, totalBytes: UInt64, pageSize: UInt64) -> UInt64 {
        let reclaimablePages = UInt64(statistics.free_count)
            + UInt64(statistics.inactive_count)
            + UInt64(statistics.speculative_count)
            + UInt64(statistics.purgeable_count)
        let bytes = reclaimablePages.multipliedReportingOverflow(by: pageSize)
        return bytes.overflow ? 0 : min(totalBytes, bytes.partialValue)
    }

    private static func readMemorySnapshot() -> MemorySnapshot {
        let total = ProcessInfo.processInfo.physicalMemory
        guard total > 0 else {
            return MemorySnapshot(usedBytes: 0, totalBytes: 0, ratio: 0, pressure: "unknown")
        }

        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        var statistics = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &statistics) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return MemorySnapshot(usedBytes: 0, totalBytes: total, ratio: 0, pressure: "unknown")
        }

        let available = Self.availableMemoryBytes(statistics, totalBytes: total, pageSize: UInt64(vm_page_size))
        let used = total - available
        let ratio = Double(used) / Double(total)

        let environment = ProcessInfo.processInfo.environment
        let elevated = Self.threshold(environment["BADAPPLE_MEMORY_ELEVATED"], fallback: 0.80)
        let unhealthy = max(elevated, Self.threshold(environment["BADAPPLE_MEMORY_UNHEALTHY"], fallback: 0.90))
        let critical = max(unhealthy, Self.threshold(environment["BADAPPLE_MEMORY_CRITICAL"], fallback: 0.95))
        let pressure: String
        if ratio >= critical {
            pressure = "critical"
        } else if ratio >= unhealthy {
            pressure = "unhealthy"
        } else if ratio >= elevated {
            pressure = "elevated"
        } else {
            pressure = "normal"
        }
        return MemorySnapshot(usedBytes: used, totalBytes: total, ratio: ratio, pressure: pressure)
    }

    private static func threshold(_ value: String?, fallback: Double) -> Double {
        guard let value, let parsed = Double(value), parsed.isFinite else { return fallback }
        return min(1, max(0, parsed))
    }
}
