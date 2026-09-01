import Foundation
import Darwin

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
            ]
        ]
    }

    public func reset() {
        clearModels()
        clearMetrics()
        clearCircuits()
        clearLastError()
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

        let reclaimablePages = UInt64(statistics.free_count)
            + UInt64(statistics.inactive_count)
            + UInt64(statistics.speculative_count)
            + UInt64(statistics.purgeable_count)
        let reclaimableBytes = reclaimablePages.multipliedReportingOverflow(by: UInt64(vm_page_size))
        let available = reclaimableBytes.overflow ? total : min(total, reclaimableBytes.partialValue)
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
