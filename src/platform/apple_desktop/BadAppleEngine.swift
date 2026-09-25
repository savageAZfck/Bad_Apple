// BadAppleEngine — Swift-native AI engine for Bad Apple.
// Wraps BadAppleMLX inference with persona system and prompt management.
// This replaces the Python daemon for text generation queries.

import Foundation
import BadAppleMLX
import CryptoKit
import IOKit

extension Notification.Name {
    /// Posted when an event should force an immediate Curious self-improvement check.
    static let curiousTrigger = Notification.Name("BadAppleCuriousTrigger")
}

/// Shared proactive-notification queue. Any subsystem appends one JSON line
/// per event to `~/.bad_apple/notify_queue.jsonl`; the menu bar tails it and
/// turns entries into macOS notifications (and spoken alerts when `voice` is
/// set and policy allows). Debounced per kind so trigger storms cannot spam.
enum BadAppleNotify {
    private static let lock = NSLock()
    private static var lastPush: [String: Date] = [:]

    static var path: String {
        NSHomeDirectory() + "/.bad_apple/notify_queue.jsonl"
    }

    /// True when proactive notifications are enabled by policy.
    static var enabled: Bool {
        BadAppleEngine.shared.proactiveNotificationsEnabled
    }

    static func push(kind: String, title: String, body: String,
                     voice: Bool = false, debounceSeconds: TimeInterval = 300,
                     referenceID: String? = nil) {
        guard enabled else { return }
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        if let last = lastPush[kind], now.timeIntervalSince(last) < debounceSeconds { return }
        lastPush[kind] = now

        let classification: (importance: Int, urgency: Int, requiresAction: Bool)
        if kind.hasPrefix("kill_switch") {
            classification = (5, 5, true)
        } else if kind.hasPrefix("approval:") {
            classification = (4, 4, true)
        } else if kind.hasPrefix("commitment_due") {
            classification = (4, 4, true)
        } else if kind.hasPrefix("task_failed") {
            classification = (4, 3, true)
        } else if kind.hasPrefix("task_done") {
            classification = (2, 1, false)
        } else if kind.hasPrefix("sentinel") {
            classification = (4, 4, true)
        } else if kind.hasPrefix("calendar_soon") {
            classification = (4, 4, true)
        } else if kind.hasPrefix("watcher") {
            classification = (4, 4, true)
        } else {
            classification = (2, 2, false)
        }
        let event = BadAppleHumanEvent(
            id: UUID().uuidString.lowercased(),
            kind: kind,
            title: title,
            body: body,
            importance: classification.importance,
            urgency: classification.urgency,
            requiresAction: classification.requiresAction,
            voiceRequested: voice,
            createdAt: now,
            referenceID: referenceID
        )
        let decision = BadAppleHumanLayer.shared.route(event: event, now: now)
        if decision.channel == .silent { return }
        if decision.channel == .queue {
            try? BadAppleHumanLayer.shared.enqueue(event: event, decision: decision)
            return
        }

        var entry: [String: Any] = [
            "ts": now.timeIntervalSince1970,
            "kind": kind,
            "title": title,
            "body": body,
            "voice": decision.channel == .speak,
            "event_id": event.id,
            "channel": decision.channel.rawValue,
            "reason": decision.reason,
        ]
        if let referenceID {
            entry["reference_id"] = referenceID
        }
        guard let data = try? JSONSerialization.data(withJSONObject: entry),
              var line = String(data: data, encoding: .utf8)
        else { return }
        line += "\n"
        if let handle = FileHandle(forWritingAtPath: path) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}

/// The main AI engine. Loads the model, manages personas, and generates
/// responses directly in-process — no subprocess, no daemon, no Python.
final class BadAppleEngine: @unchecked Sendable {

    // MARK: - Singleton

    static let shared = BadAppleEngine()

    // MARK: - State

    private final class NativeEmbeddingProvider: EmbeddingProvider, @unchecked Sendable {
        let engine: BadAppleEmbeddingEngine
        let fallback = BadAppleLexicalEmbeddingProvider()

        init(engine: BadAppleEmbeddingEngine) {
            self.engine = engine
        }

        func embed(_ text: String) async -> [Float] {
            if let vector = try? await engine.embed(text) {
                return vector
            }
            return await fallback.embed(text)
        }
    }

    private var inference: BadAppleInference
    private lazy var fastInference: BadAppleInference? = {
        guard let fastModelId = BadAppleInference.envFastModelId, !fastModelId.isEmpty else {
            NSLog("[BadAppleEngine] Fast tier disabled: BADAPPLE_FAST_MODEL is empty or unset.")
            return nil
        }
        let config = BadAppleInference.ModelConfig(
            modelId: fastModelId,
            revision: "main",
            maxTokens: 150,
            temperature: 0.6,
            topP: 0.9
        )
        return BadAppleInference(config: config)
    }()
    private var _fastModelLoaded = false
    private var fastModelLoaded: Bool {
        get { stateLock.withLock { _fastModelLoaded } }
        set { stateLock.withLock { _fastModelLoaded = newValue } }
    }
    private var recentSignOffs: [String] = []
    let personaManager = BadApplePersonaManager()
    private let auditLedger = BadAppleAuditLedger()
    private let outputFirewall = BadAppleOutputFirewall()
    private let toolRouter = BadAppleToolRouter()
    private let policyEngine = BadApplePolicyEngine()
    private lazy var toolExecutor: BadAppleToolExecutor = {
        let exec = BadAppleToolExecutor(policyEngine: policyEngine)
        exec.visionProvider = { [weak self] path, prompt in
            guard let self else { return "Vision engine unavailable." }
            return await self.describeImageInternal(at: path, prompt: prompt)
        }
        exec.agentSubmitter = { [weak self] goal, maxSteps in
            guard let self else { return "Error: engine unavailable." }
            let task = try await self.submitAgentTask(goal: goal, maxSteps: maxSteps)
            return "Task \(task.id) \(task.status.rawValue) — \(task.goal) (max \(task.maxSteps) steps). Track with `badapple tasks`."
        }
        exec.humanPersistenceAllowed = { [weak self] in !(self?.privateMode ?? true) }
        exec.outputFirewall = outputFirewall
        return exec
    }()
    private let embeddingEngine = BadAppleEmbeddingEngine()
    private let visionEngine = BadAppleVisionEngine()
    private lazy var semanticCache = BadAppleSemanticCache(
        embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
    )
    private static let nonCacheableResponseTiers: Set<String> = [
        "approval", "human_tool", "human_command", "human_parse_error",
    ]
    private let rag = BadAppleRAG()
    private let runtime = BadAppleNativeRuntime()
    private let modelOperationGate = BadAppleModelOperationGate()
    private let conversation = BadAppleConversation()
    let modelManager = BadAppleModelManager.shared
    private let approvalLock = NSLock()
    private var pendingApprovals: [String: (name: String, args: [String: String], created: Date)] = [:]
    private var approvalsLoaded = false
    private let approvalsPath = NSHomeDirectory() + "/.bad_apple/pending_approvals.json"
    private let approvalTTL: TimeInterval = 24 * 3600

    // MARK: - Conversation Pruning

    /// Maximum number of turns (user+assistant pairs) to keep in conversation history.
    /// Older turns are pruned to prevent unbounded context growth.
    static let maxHistoryTurns = 20

    // MARK: - Prompt Hot-Reload

    private var promptFileURL: URL? {
        let candidates = [
            FileManager.default.currentDirectoryPath + "/prompt.txt",
            NSHomeDirectory() + "/bad_apple/prompt.txt",
            NSHomeDirectory() + "/.bad_apple/prompt.txt",
        ]
        for path in candidates {
            let url = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return nil
    }
    private var lastPromptMtime: Date?

    // MARK: - Model Integrity

    private var lastConfigHash: String?
    private var curiousAutopilotTask: Task<Void, Never>?
    private var lastCuriousCheck = Date.distantPast
    private let curiousProposalsDir = NSHomeDirectory() + "/.bad_apple/notes/proposed_patches"
    private lazy var agent: BadAppleAgent? = try? BadAppleAgent(
        planner: { [weak self] goal, maximumSteps in
            guard let self else { return [] }
            return try await self.planAgentGoal(goal, maximumSteps: maximumSteps)
        },
        generator: { [weak self] context in
            guard let self else { return .finish("The native engine is unavailable.") }
            return try await self.generateAgentAction(context)
        },
        executor: { [weak self] tool, arguments, _ in
            guard let self else { return "The native engine is unavailable." }
            return await self.toolExecutor.executeTool(name: tool, args: arguments)
        },
        replanReporter: { [weak self] taskID, failure in
            self?.auditLedger.append(
                eventType: "agent_replan",
                data: ["task": taskID, "failure": failure],
                persona: self?.activePersona ?? "badapple"
            )
            BadAppleNotify.push(
                kind: "task_replan",
                title: "Replanning",
                body: "A step failed — regenerating the rest of the plan. \(failure)",
                debounceSeconds: 60
            )
        }
    )

    private final class StreamState: @unchecked Sendable {
        private let lock = NSLock()
        private var accumulated = ""
        private var blocked = false

        func filter(_ chunk: String, with firewall: BadAppleOutputFirewall) -> String {
            lock.lock()
            defer { lock.unlock() }
            guard !blocked else { return "" }
            let result = firewall.checkChunk(chunk, accumulated: accumulated)
            accumulated += chunk
            blocked = result.1
            return result.0
        }
    }

    private let stateLock = NSLock()
    private var _isLoaded = false
    private var _isLoading = false
    private var _mainModelBytes: UInt64 = 0
    private var _fastModelBytes: UInt64 = 0
    private var _modelId: String = ""
    private var _lastTokensPerSecond: Float = 0
    private var _lastTokenCount: Int = 0
    private var _lastDraftAcceptPct: Float = 0
    private var _lastPrefixCache: String = "off"
    private var _lastCacheHit: Bool = false
    private var _workspacePath: String?
    private var _airgapEnabled = false
    private var _privateModeEnabled = false
    private var _fastTierEnabled = false
    private var _killed = false
    private var _workspaceWatcher: BadAppleWorkspaceWatcher?

    var isLoaded: Bool {
        return stateLock.withLock { _isLoaded }
    }

    /// Policy switch for proactive notifications — read by BadAppleNotify.
    var proactiveNotificationsEnabled: Bool { policyEngine.proactiveNotify }

    var killed: Bool {
        get { stateLock.withLock { _killed } }
        set {
            let wasKilled = stateLock.withLock { () -> Bool in
                let old = _killed
                _killed = newValue
                return old
            }
            if newValue, !wasKilled {
                BadAppleNotify.push(
                    kind: "kill_switch",
                    title: "Bad Apple paused",
                    body: "Kill switch engaged — generation and tools are stopped. Say 'resume bad apple' to restart.",
                    voice: true,
                    debounceSeconds: 60
                )
            }
        }
    }

    var isLoading: Bool {
        return stateLock.withLock { _isLoading }
    }

    var modelId: String {
        return stateLock.withLock { _modelId }
    }

    var lastTokensPerSecond: Float {
        return stateLock.withLock { _lastTokensPerSecond }
    }

    var lastTokenCount: Int {
        return stateLock.withLock { _lastTokenCount }
    }

    var lastDraftAcceptPct: Float {
        return stateLock.withLock { _lastDraftAcceptPct }
    }

    var lastPrefixCache: String {
        return stateLock.withLock { _lastPrefixCache }
    }

    var lastCacheHit: Bool {
        return stateLock.withLock { _lastCacheHit }
    }

    var workspacePath: String? {
        get {
            return stateLock.withLock { _workspacePath }
        }
        set {
            stateLock.withLock { _workspacePath = newValue }
            toolExecutor.workspace = newValue
            restartWorkspaceWatcher()
        }
    }

    private func restartWorkspaceWatcher() {
        _workspaceWatcher?.stop()
        _workspaceWatcher = nil
        guard let path = _workspacePath else { return }
        let resolved = (path as NSString).standardizingPath
        guard FileManager.default.fileExists(atPath: resolved) else { return }
        let watcher = BadAppleWorkspaceWatcher(path: resolved)
        watcher.start { [weak self] workspace in
            guard let self else { return }
            Task {
                _ = self.toolExecutor.indexDocuments(path: workspace)
            }
        }
        _workspaceWatcher = watcher
    }

    // MARK: - Air-gap / Private Mode

    var airgap: Bool {
        get {
            return stateLock.withLock { _airgapEnabled }
        }
        set {
            stateLock.withLock { _airgapEnabled = newValue }
        }
    }

    // MARK: - Fast Tier

    /// When enabled, simple queries (math, time, greetings, identity) route to
    /// the 0.5B fast model instead of the deterministic regex path or the 9B.
    var fastTierEnabled: Bool {
        get {
            return stateLock.withLock { _fastTierEnabled }
        }
        set {
            stateLock.withLock { _fastTierEnabled = newValue }
        }
    }

    /// Set the VRAM budget in GB.
    func setVRAMBudgetGB(_ gb: UInt64) async {
        await runtime.setVRAMBudget(gb * 1_073_741_824)
    }

    var privateMode: Bool {
        get {
            return stateLock.withLock { _privateModeEnabled }
        }
        set {
            stateLock.withLock { _privateModeEnabled = newValue }
            auditLedger.paused = newValue
        }
    }

    // MARK: - Roast Bank

    private let roastMoods: [String] = [
        "savage", " dismissive", " deadpan", " theatrical", " cold",
    ]
    private var roastIndex = 0

    /// Rotate to the next roast mood for persona variety.
    func nextRoastMood() -> String {
        let mood = roastMoods[roastIndex % roastMoods.count]
        roastIndex += 1
        return mood.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Ambient Context

    /// The latest ambient context snapshot (active app, window title, optional screen description).
    private(set) var ambientContext: String?

    /// Minimum seconds between ambient context refreshes.
    private let ambientRefreshInterval: TimeInterval = 5
    private var lastAmbientUpdate: Date?

    /// Minimum seconds between ocular screen descriptions.
    private let ocularRefreshInterval: TimeInterval = {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_OCULAR_INTERVAL"] ?? "0"
        return TimeInterval(env) ?? 0
    }()

    private var lastOcularUpdate: Date?

    /// Minimum seconds between ambient-hearing percept reads.
    private let auralRefreshInterval: TimeInterval = {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_EARS_INTERVAL"] ?? "45"
        return max(TimeInterval(env) ?? 45, 10)
    }()

    private var lastAuralUpdate: Date?

    /// Thermodynamic governor verdict — set from the thermal percept file
    /// written by the supervisor (`/var/lib/bad_apple/thermal.json`).
    private var thermalThrottle = false
    private var thermalStress = 0.0

    /// Whether ambient updates are allowed. Off by default for air-gap / privacy.
    var ambientEnabled: Bool {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_AMBIENT"] ?? "0"
        return env == "1" || env.lowercased() == "true" || env.lowercased() == "on"
    }

    /// Whether ocular screen capture is allowed. Off by default.
    var ocularEnabled: Bool {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_OCULAR"] ?? "0"
        return env == "1" || env.lowercased() == "true" || env.lowercased() == "on"
    }

    /// Whether ambient hearing is allowed. Off by default — audio is the most
    /// privacy-dense sense, so it requires an explicit opt-in: the control
    /// file at ~/.bad_apple/ears (the same switch the menu bar's capture loop
    /// honors) or the BADAPPLE_EARS env var.
    var auralEnabled: Bool {
        if FileManager.default.fileExists(atPath: NSHomeDirectory() + "/.bad_apple/ears") {
            return true
        }
        let env = ProcessInfo.processInfo.environment["BADAPPLE_EARS"] ?? "0"
        return env == "1" || env.lowercased() == "true" || env.lowercased() == "on"
    }

    /// Resolve a helper binary next to the executable, inside the app bundle, or in target/release.
    private func helperURL(named: String) -> URL? {
        let fm = FileManager.default
        // 1. Next to the current executable.
        if let exe = ProcessInfo.processInfo.arguments.first.map(URL.init(fileURLWithPath:)) {
            let candidate = exe.deletingLastPathComponent().appendingPathComponent(named)
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }
        // 2. Inside the app bundle (engine may be installed as a helper or the bundle may be next to it).
        if let exe = ProcessInfo.processInfo.arguments.first.map(URL.init(fileURLWithPath:)) {
            var bundle = exe
            while bundle.pathComponents.count > 2 {
                if bundle.lastPathComponent.hasSuffix(".app") {
                    let candidate = bundle.appendingPathComponent("Contents/Helpers/").appendingPathComponent(named)
                    if fm.isExecutableFile(atPath: candidate.path) {
                        return candidate
                    }
                    break
                }
                bundle = bundle.deletingLastPathComponent()
            }
        }
        // 3. Hard-coded release and app paths.
        let candidates = [
            "/Applications/Bad Apple.app/Contents/Helpers/\(named)",
            "/usr/local/lib/bad_apple/\(named)",
            "\(NSHomeDirectory())/bad_apple/\(named)",
        ]
        for path in candidates {
            if fm.isExecutableFile(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    /// Run a helper and return its stdout as a string.
    private func runHelper(_ url: URL, arguments: [String], timeout: TimeInterval = 10) -> String? {
        let task = Process()
        task.executableURL = url
        task.arguments = arguments
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
        } catch {
            return nil
        }
        let deadline = Date().addingTimeInterval(timeout)
        while task.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if task.isRunning {
            task.terminate()
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Update the ambient context from the `BadAppleAmbient` helper or AppleScript fallback.
    /// Returns the captured context even if ocular capture is still running.
    func updateAmbientContext() {
        guard ambientEnabled else { return }
        if let last = lastAmbientUpdate, Date().timeIntervalSince(last) < ambientRefreshInterval { return }
        lastAmbientUpdate = Date()

        var parts: [String] = []

        // Prefer the native helper for richer, reliable context.
        if let helper = helperURL(named: "BadAppleAmbient") {
            if let output = runHelper(helper, arguments: []),
               let data = output.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
                let app = json["app"] ?? "Unknown"
                let window = json["window"] ?? ""
                parts.append("Active app: \(app)")
                if !window.isEmpty {
                    parts.append("Window: \(window)")
                }
            }
        }

        // AppleScript fallback if the helper is not available.
        if parts.isEmpty {
            let script = """
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                set frontWindow to ""
                try
                    set frontWindow to title of front window of (first application process whose frontmost is true)
                end try
                return frontApp & "|" & frontWindow
            end tell
            """
            var errorInfo: NSDictionary?
            if let result = NSAppleScript(source: script)?.executeAndReturnError(&errorInfo) {
                let value = result.stringValue ?? ""
                let split = value.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false)
                let app = split.first.map(String.init) ?? "Unknown"
                let window = split.count > 1 ? String(split[1]) : ""
                parts.append("Active app: \(app)")
                if !window.isEmpty {
                    parts.append("Window: \(window)")
                }
            }
        }

        // Reuse any existing ocular description or ambient transcript without losing it.
        if let existing = ambientContext {
            for line in existing.components(separatedBy: .newlines) {
                if line.starts(with: "Screen:") || line.starts(with: "Heard:") || line.starts(with: "Thermal:") || line.starts(with: "Up next:") {
                    parts.append(line)
                }
            }
        }

        if !parts.isEmpty {
            ambientContext = parts.joined(separator: "\n")
        }
    }

    /// Refresh both ambient and ocular context for the next prompt.
    func refreshAmbientContext() async {
        updateAmbientContext()
        await updateOcularContext()
        updateAuralContext()
        updateThermalContext()
    }

    /// Read the latest ambient-hearing percept written by the menu bar app —
    /// the organ that holds the microphone TCC grant — and expose it to the
    /// prompt as a `Heard:` line. The control file at ~/.bad_apple/ears is the
    /// opt-in switch; a stale percept (older than a few refresh intervals) is
    /// dropped because old speech is not ambient context.
    func updateAuralContext() {
        guard auralEnabled else { return }
        if let last = lastAuralUpdate, Date().timeIntervalSince(last) < auralRefreshInterval { return }
        lastAuralUpdate = Date()

        let heardURL = URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ambient_heard.json")
        var heard = ""
        var heardAge = TimeInterval.greatestFiniteMagnitude
        if let data = try? Data(contentsOf: heardURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            heard = (json["heard"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if let ts = json["ts"] as? TimeInterval {
                heardAge = Date().timeIntervalSince1970 - ts
            }
        }
        // Stale percepts expire: ambient context describes now, not minutes ago.
        if heardAge > auralRefreshInterval * 3 { heard = "" }

        var parts = ambientContext?.components(separatedBy: .newlines) ?? []
        parts.removeAll { $0.starts(with: "Heard:") }
        if !heard.isEmpty {
            parts.append("Heard: \(heard)")
            ambientContext = parts.joined(separator: "\n")
        } else if ambientContext != nil {
            ambientContext = parts.isEmpty ? nil : parts.joined(separator: "\n")
        }
    }

    /// Read the thermodynamic governor verdict written by the supervisor.
    /// A fresh report (<10 min) with `throttle: true` caps generation length
    /// and tells the model to be terse; a missing or stale file means the
    /// governor is not running and the brain is unaffected.
    func updateThermalContext() {
        let path = ProcessInfo.processInfo.environment["BADAPPLE_THERMAL_FILE"]
            ?? "/var/lib/bad_apple/thermal.json"
        var throttle = false
        var stress = 0.0
        if let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let ts = json["ts"] as? TimeInterval,
           Date().timeIntervalSince1970 - ts < 600 {
            throttle = json["throttle"] as? Bool ?? false
            stress = json["stress"] as? Double ?? 0.0
        }
        thermalThrottle = throttle
        thermalStress = stress

        var parts = ambientContext?.components(separatedBy: .newlines) ?? []
        parts.removeAll { $0.starts(with: "Thermal:") }
        if throttle {
            parts.append(String(format: "Thermal: system under stress (%.0f%%) — answer tersely", stress * 100))
        }
        if throttle || ambientContext != nil {
            ambientContext = parts.isEmpty ? nil : parts.joined(separator: "\n")
        }
    }

    /// Capture the screen and describe it with the vision model, appending the
    /// result to `ambientContext` when ready. This is off by default; set
    /// `BADAPPLE_OCULAR=1` and an optional `BADAPPLE_OCULAR_INTERVAL` in seconds.
    func updateOcularContext() async {
        guard ocularEnabled else { return }
        if let last = lastOcularUpdate, ocularRefreshInterval > 0, Date().timeIntervalSince(last) < ocularRefreshInterval { return }

        let capturePath = "/var/tmp/badapple_ocular.png"
        guard let helper = helperURL(named: "BadAppleScreenCapture") else { return }
        let output = runHelper(helper, arguments: ["--output", capturePath], timeout: 15)
        guard output == capturePath, FileManager.default.fileExists(atPath: capturePath) else { return }

        let description = await describeImageInternal(at: capturePath, prompt: "Describe what is on screen in one concise sentence.")
        lastOcularUpdate = Date()

        var parts = ambientContext?.components(separatedBy: .newlines) ?? []
        // Replace any existing screen line.
        parts.removeAll { $0.starts(with: "Screen:") }
        parts.append("Screen: \(description)")
        ambientContext = parts.joined(separator: "\n")
    }

    // MARK: - Init

    private init() {
        inference = BadAppleInference.createDefault()
        let defaultModelId = BadAppleInference.defaultConfig.modelId
        stateLock.withLock { _modelId = defaultModelId }
        personaManager.setCurrentModel(repoId: defaultModelId)
        try? FileManager.default.createDirectory(
            atPath: curiousProposalsDir,
            withIntermediateDirectories: true,
            attributes: nil
        )
        updateCuriousAutopilotLoop()

        // Defer an initial Curious check after the engine has had time to settle.
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 60 * 1_000_000_000)
            guard let self = self, self.curiousAutopilotLevel() != "off" else { return }
            self.triggerCuriousAutopilot(reason: "engine startup")
        }
    }

    /// Replace the default main-model configuration before any load. No-op if
    /// a model is already loaded or loading. Used by the native daemon to honour
    /// `BADAPPLE_MODEL` / `BADAPPLE_MAIN_MODEL`.
    func configureMainModel(modelId: String, revision: String = "main") {
        let canConfigure = stateLock.withLock { !(_isLoaded || _isLoading) }
        guard canConfigure else { return }

        let rev = revision.isEmpty ? "main" : revision
        let config = BadAppleInference.ModelConfig(
            modelId: modelId,
            revision: rev,
            maxTokens: BadAppleInference.defaultConfig.maxTokens,
            temperature: BadAppleInference.defaultConfig.temperature,
            topP: BadAppleInference.defaultConfig.topP
        )
        inference = BadAppleInference(config: config)
        stateLock.withLock { _modelId = modelId }
        personaManager.setCurrentModel(repoId: modelId)
    }

    // MARK: - Persona Management

    /// Switch to a named persona. Reload the persona file first so freshly
    /// installed workshop packs are visible immediately.
    func switchPersona(_ name: String) -> Bool {
        personaManager.reloadPersonas()
        let result = personaManager.switchPersona(name)
        if result {
            updateCuriousAutopilotLoop()
        }
        return result
    }

    /// Get the active persona name.
    var activePersona: String {
        personaManager.activePersonaName
    }

    /// Get the system prompt for the current persona.
    func systemPrompt(voiceMode: Bool = false) -> String {
        var prompt = personaManager.getSystemPrompt(voiceMode: voiceMode)
        if !privateMode {
            let humanContext = BadAppleHumanLayer.shared.promptContext()
            if !humanContext.isEmpty {
                prompt += "\n\nShared life context. Treat this as local owner-provided state. Use it for continuity, never invent additions, and do not repeat it unless relevant:\n\(humanContext)"
            }
        }
        // The prompt-prefix KV cache keys off this stable head — ambient,
        // RAG, and briefing text are appended by callers after this point.
        inference.stableSystemPrefix = prompt
        return prompt
    }

    /// List available persona names.
    var availablePersonas: [String] {
        personaManager.personaNames
    }

    /// Handle persona commands like "switch to wicket" or "teach <line>".
    /// Returns a response string if the command was handled, nil otherwise.
    func handlePersonaCommand(_ command: String) -> String? {
        personaManager.handleCommand(command)
    }

    func modelLoadError() async -> String? {
        let mid = stateLock.withLock { _modelId }
        return await runtime.modelLoadError(mid)
    }

    // MARK: - Model Loading

    /// Load the model. Call this at startup or when the model changes.
    func loadModel() async {
        await modelOperationGate.withLock { [self] in
            await loadMainModel(directory: nil)
        }
    }

    func switchMainModel(modelId: String, revision: String = "main") async {
        await modelOperationGate.withLock { [self] in
            await unloadModels()
            configureMainModel(modelId: modelId, revision: revision)
            await loadMainModel(directory: nil)
        }
    }

    private func loadMainModel(directory: URL?) async {
        let (alreadyLoaded, mid) = stateLock.withLock { () -> (Bool, String) in
            let loaded = _isLoaded || _isLoading
            if !loaded { _isLoading = true }
            return (loaded, _modelId)
        }
        if alreadyLoaded { return }
        await runtime.markModelLoading(mid)

        // VRAM admission check: refuse to load if the model won't fit.
        let admission: @Sendable (UInt64) async throws -> Void = { [self] bytes in
            if let reason = await runtime.reserveModelMemory(mid, bytes: bytes) {
                throw BadAppleInference.InferenceError.modelLoadFailed("VRAM admission denied: \(reason)")
            }
            stateLock.withLock { _mainModelBytes = bytes }
        }
        do {
            if let directory {
                try await inference.loadModel(from: directory, admission: admission)
            } else {
                try await inference.loadModel(admission: admission)
            }
            guard await inference.ready else { throw BadAppleInference.InferenceError.modelNotLoaded }
            stateLock.withLock { _isLoaded = true }
            await applyDreamAdapterIfPresent()
            await runtime.markModelReady(mid)
            Task {
                await runtime.markModelLoading(embeddingEngine.configuration.modelId)
                do {
                    try await embeddingEngine.loadModel()
                    await runtime.markModelReady(embeddingEngine.configuration.modelId)
                    NSLog("[BadAppleEngine] Native embedding model loaded")
                } catch {
                    NSLog("[BadAppleEngine] Native embedding model failed: %@", error.localizedDescription)
                    await runtime.markModelFailed(
                        embeddingEngine.configuration.modelId,
                        error: error.localizedDescription
                    )
                }
            }
        } catch {
            NSLog("[BadAppleEngine] Failed to load model: %@", error.localizedDescription)
            let reserved = stateLock.withLock { () -> UInt64 in
                _isLoaded = false
                defer { _mainModelBytes = 0 }
                return _mainModelBytes
            }
            await runtime.releaseModelMemory(reserved)
            await runtime.markModelFailed(mid, error: error.localizedDescription)
            triggerCuriousAutopilot(reason: "model load failed: \(error.localizedDescription)")
        }
        stateLock.withLock { _isLoading = false }
    }

    /// Load from a local directory (e.g., HuggingFace cache).
    func loadModel(from directory: URL) async {
        await modelOperationGate.withLock { [self] in
            await loadMainModel(directory: directory)
        }
    }

    // MARK: - Generation

    private func conversationSessionID() -> String {
        let state = BadAppleHumanLayer.shared.snapshot()
        if let activeID = state.activeConversationThreadID,
           let thread = state.conversationThreads.first(where: {
               $0.id == activeID
                   && ($0.status == .active || $0.status == .waiting)
           }) {
            return thread.sessionID
        }
        if privateMode { return "default" }
        return (try? BadAppleHumanLayer.shared.activeConversationThread().sessionID) ?? "default"
    }

    private func inferenceHistory(sessionID: String) -> [BadAppleInference.ChatMessage] {
        conversation.loadConversation(sessionId: sessionID).map {
            BadAppleInference.ChatMessage(role: $0.role, content: $0.content)
        }
    }

    private func saveTurn(prompt: String, response: String, sessionID: String) {
        // Private mode: skip persistence entirely.
        guard !privateMode else { return }
        conversation.appendTurn(sessionId: sessionID, prompt: prompt, response: response)
    }

    func resetConversation() {
        guard !privateMode else { return }
        conversation.clearConversation(sessionId: conversationSessionID())
    }

    private func planAgentGoal(
        _ goal: String,
        maximumSteps: Int
    ) async throws -> [BadAppleAgentPlannedStep] {
        let result = try await inference.generate(
            prompt: "Break this goal into at most \(maximumSteps) concrete steps. Return only a JSON array of objects with an instruction string. Goal: \(goal)",
            systemPrompt: "You are a task planner. Return valid JSON only.",
            maxTokens: 256,
            temperature: 0
        )
        if let start = result.text.firstIndex(of: "["),
           let end = result.text.lastIndex(of: "]"),
           start <= end,
           let data = String(result.text[start...end]).data(using: .utf8),
           let objects = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            let steps = objects.prefix(maximumSteps).compactMap { object -> BadAppleAgentPlannedStep? in
                guard let instruction = object["instruction"] as? String,
                      !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
                return BadAppleAgentPlannedStep(instruction: instruction)
            }
            if !steps.isEmpty { return steps }
        }
        return [BadAppleAgentPlannedStep(instruction: goal)]
    }

    private func generateAgentAction(
        _ context: BadAppleAgentGenerationContext
    ) async throws -> BadAppleAgentAction {
        let tools = toolRouter.allToolsForPrompt() ?? "No tools are available."
        let agentSystem = """
        You are the Bad Apple agent executor. You are given a goal and a single step.
        Pick exactly one tool for the step, or mark the step finished.
        - If a tool is needed, output ONLY: <tool_call>{"name":"tool_name","arguments":{"param":"value"}}</tool_call>
        - If this step is done, output ONLY: <done>short plain-English summary of what this step produced</done>
        Do not add prose, sign-offs, markdown, or explanations. Use only the exact tool names shown below.
        """
        let history = context.completedSteps.map { step -> String in
            let outcome = (step.error.isEmpty ? step.result : "ERROR: \(step.error)")
                .replacingOccurrences(of: "\n", with: " ")
            return "\(step.index + 1). \(step.instruction) → \(step.tool): \(outcome.prefix(300))"
        }.joined(separator: "\n")
        let historyBlock = history.isEmpty ? "" : "\nCompleted steps so far:\n\(history)\n"
        let prompt = """
        Goal: \(context.goal)
        Step \(context.stepIndex + 1) of \(context.maxSteps): \(context.plannedStep.instruction)
        \(historyBlock)
        \(tools)

        Rules:
        - This is step \(context.stepIndex + 1) of \(context.maxSteps). If it is not the last step, you MUST call a tool. Do not finish early.
        - Only use `<done>` on the final step, or if the goal is fully complete.
        - Do not summarize or make up results. If the step says to run, inspect, search, check, or write, use the matching tool.
        - Output one `<tool_call>` block or one `<done>` block. Nothing else.
        """
        let result = try await inference.generate(
            prompt: prompt,
            systemPrompt: agentSystem,
            maxTokens: 256,
            temperature: 0
        )

        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Look for a finished step.
        if let doneStart = text.range(of: "<done>")?.upperBound,
           let doneEnd = text.range(of: "</done>", range: doneStart..<text.endIndex)?.lowerBound {
            let summary = String(text[doneStart..<doneEnd])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return .finish(summary.isEmpty ? "Step completed." : summary)
        }

        // Look for a tool call.
        if let call = toolRouter.extractToolCalls(text: text).first {
            return BadAppleAgentAction.call(
                tool: call.name,
                arguments: call.args,
                thought: ""
            )
        }

        return .finish(postprocessOutput(text))
    }

    /// Pending approvals persist to disk so a daemon restart does not strand
    /// an outstanding `approve <id>` — entries expire after approvalTTL.
    private func ensureApprovalsLoaded() {
        // Caller must hold approvalLock.
        guard !approvalsLoaded else { return }
        approvalsLoaded = true
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: approvalsPath)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: [String: Any]]
        else { return }
        let now = Date()
        for (id, entry) in obj {
            guard let name = entry["name"] as? String,
                  let args = entry["args"] as? [String: String] else { continue }
            let created = (entry["created"] as? Double).map { Date(timeIntervalSince1970: $0) } ?? now
            guard now.timeIntervalSince(created) < approvalTTL else { continue }
            pendingApprovals[id] = (name, args, created)
        }
    }

    private func persistApprovals() {
        // Caller must hold approvalLock.
        var obj: [String: [String: Any]] = [:]
        let now = Date()
        for (id, call) in pendingApprovals where now.timeIntervalSince(call.created) < approvalTTL {
            obj[id] = ["name": call.name, "args": call.args, "created": call.created.timeIntervalSince1970]
        }
        if let data = try? JSONSerialization.data(withJSONObject: obj) {
            try? data.write(to: URL(fileURLWithPath: approvalsPath), options: .atomic)
        }
    }

    private func createApproval(name: String, args: [String: String]) -> String {
        let id = String(UUID().uuidString.lowercased().prefix(8))
        approvalLock.lock()
        ensureApprovalsLoaded()
        pendingApprovals[id] = (name, args, Date())
        persistApprovals()
        approvalLock.unlock()
        BadAppleNotify.push(
            kind: "approval:\(name)",
            title: "Bad Apple needs approval",
            body: "`\(name)` is waiting — reply `approve \(id)` to allow once.",
            voice: true,
            debounceSeconds: 120,
            referenceID: id
        )
        return id
    }

    private func takeApproval(from prompt: String) -> (approved: Bool, id: String, name: String, args: [String: String])? {
        let parts = prompt.lowercased().split(whereSeparator: { $0.isWhitespace })
        guard parts.count == 2, parts[0] == "approve" || parts[0] == "deny" else { return nil }
        let id = String(parts[1])
        approvalLock.lock()
        defer { approvalLock.unlock() }
        ensureApprovalsLoaded()
        guard let call = pendingApprovals.removeValue(forKey: id) else { return nil }
        persistApprovals()
        return (parts[0] == "approve", id, call.name, call.args)
    }

    /// Journal a council deliberation: verdict, dissent, and every seat's vote.
    private func auditCouncil(verdict: CouncilVerdict, name: String, args: [String: String],
                              mode: String, persona: String) {
        let votes: [[String: String]] = verdict.votes.map {
            ["seat": $0.seat, "vote": String(format: "%.3f", $0.vote), "said": $0.rationale]
        }
        auditLedger.append(
            eventType: "council_deliberation",
            data: [
                "tool": name, "arguments": args, "mode": mode,
                "decision": verdict.decision.rawValue,
                "mean": String(format: "%.3f", verdict.mean),
                "dissent": String(format: "%.3f", verdict.dissent),
                "votes": votes,
            ],
            persona: persona
        )
    }

    /// Ledger a tool call that arrived over the `invoke_tool` IPC path.
    /// Model-emitted calls are audited by the generation loop; this covers
    /// the direct-invocation path so every execution is attested.
    func auditInvokeToolCall(name: String, args: [String: String]) {
        auditLedger.append(
            eventType: "tool_call",
            data: ["name": name, "arguments": args, "via": "invoke_tool"],
            persona: activePersona
        )
    }

    private func executeExplicitHumanCommand(
        _ command: (name: String, args: [String: String]),
        persona: String
    ) async -> String {
        auditLedger.append(
            eventType: "tool_call",
            data: ["name": command.name, "arguments": command.args, "via": "explicit_human_phrase"],
            persona: persona
        )
        let output = await toolExecutor.executeTool(
            name: command.name, args: command.args, approved: true
        )
        auditLedger.append(
            eventType: "tool_result",
            data: ["name": command.name, "result": output],
            persona: persona
        )
        return output
    }

    private func approvalPromptText(id: String, name: String, args: [String: String]) -> String {
        var detail = ""
        for key in ["command", "script", "path", "shortcut", "query", "text", "dir", "file", "goal"] {
            if let v = args[key], !v.isEmpty {
                detail = v
                break
            }
        }
        if detail.isEmpty, let first = args.sorted(by: { $0.key < $1.key }).first {
            detail = "\(first.key)=\(first.value)"
        }
        if detail.count > 120 {
            detail = String(detail.prefix(120)) + "…"
        }
        let basis = policyEngine.approvalBasis(toolName: name)
        return """
        Approval required — Bad Apple wants to run `\(name)`\(detail.isEmpty ? "" : ": \(detail)").
        Gated by: \(basis).
        Reply `approve \(id)` to allow it once, or `deny \(id)` to refuse.
        """
    }

    private func toolAwareGeneration(
        prompt: String,
        systemPrompt: String,
        history: [BadAppleInference.ChatMessage],
        tools: [[String: Any]]?,
        maxTokens: Int,
        persona: String
    ) async throws -> BadAppleInference.GenerationResult {
        var currentPrompt = prompt
        var currentHistory = history
        var lastResult = BadAppleInference.GenerationResult(text: "")

        for iteration in 0..<5 {
            lastResult = try await inference.generate(
                prompt: currentPrompt,
                systemPrompt: systemPrompt,
                history: currentHistory,
                tools: tools,
                maxTokens: maxTokens,
                temperature: 0.6
            )

            let calls: [(name: String, args: [String: String])]
            if lastResult.toolCalls.isEmpty {
                calls = toolRouter.extractToolCalls(text: lastResult.text)
            } else {
                calls = lastResult.toolCalls.map { (name: $0.name, args: $0.arguments) }
            }
            if calls.isEmpty {
                if iteration == 0, toolRouter.requiresHumanToolExecution(prompt) {
                    return BadAppleInference.GenerationResult(
                        text: "I couldn't safely execute that Human Home command because I couldn't parse the requested change. Rephrase it with the person, commitment, conversation, preference, or attention mode stated explicitly.",
                        tier: "human_parse_error"
                    )
                }
                return lastResult
            }

            var outputs: [String] = []
            for call in calls {
                auditLedger.append(
                    eventType: "tool_call",
                    data: ["name": call.name, "arguments": call.args],
                    persona: persona
                )
                let output: String
                switch policyEngine.evaluate(toolName: call.name, args: call.args) {
                case .denied(let reason):
                    output = "Policy: \(reason)"
                case .needsApproval:
                    // Manual mode: the council advises — its verdict rides on
                    // the prompt so the human decides with counsel.
                    let verdict = BadAppleCouncil.deliberate(toolName: call.name, args: call.args)
                    auditCouncil(verdict: verdict, name: call.name, args: call.args,
                                 mode: "advisory", persona: persona)
                    let id = createApproval(name: call.name, args: call.args)
                    auditLedger.append(
                        eventType: "approval_requested",
                        data: ["id": id, "name": call.name, "arguments": call.args],
                        persona: persona
                    )
                    return BadAppleInference.GenerationResult(
                        text: approvalPromptText(id: id, name: call.name, args: call.args)
                            + "\n\n" + verdict.summaryLine,
                        tier: "approval"
                    )
                case .approved:
                    // Under autopilot the council votes on every gated action:
                    // a passed vote executes, a failed vote goes to the human
                    // for approval. The council is never the final say — the
                    // user decides everything contested.
                    if policyEngine.requiresApproval(toolName: call.name) {
                        let verdict = BadAppleCouncil.deliberate(toolName: call.name, args: call.args)
                        auditCouncil(verdict: verdict, name: call.name, args: call.args,
                                     mode: policyEngine.autopilot ? "autopilot" : "pre-approval",
                                     persona: persona)
                        if policyEngine.autopilot {
                            if verdict.contested {
                                let id = createApproval(name: call.name, args: call.args)
                                auditLedger.append(
                                    eventType: "council_escalated",
                                    data: ["id": id, "name": call.name,
                                           "decision": verdict.decision.rawValue,
                                           "dissent": verdict.dissent],
                                    persona: persona
                                )
                                return BadAppleInference.GenerationResult(
                                    text: "Council vote failed `\(call.name)` — sending it to you.\n"
                                        + verdict.summaryLine + "\n\n"
                                        + approvalPromptText(id: id, name: call.name, args: call.args),
                                    tier: "approval"
                                )
                            }
                            let result = await toolExecutor.executeTool(
                                name: call.name, args: call.args, approved: true)
                            output = result + "\n\n[council: " + verdict.summaryLine + "]"
                            auditLedger.append(
                                eventType: "tool_result",
                                data: ["name": call.name, "result": output],
                                persona: persona
                            )
                            outputs.append("\(call.name): \(output)")
                            continue
                        }
                    }
                    output = await toolExecutor.executeTool(name: call.name, args: call.args, approved: true)
                }
                auditLedger.append(
                    eventType: "tool_result",
                    data: ["name": call.name, "result": output],
                    persona: persona
                )
                outputs.append("\(call.name): \(output)")
            }

            let directHumanTools: Set<String> = [
                "human_home", "remember_preference", "forget_preference", "add_commitment",
                "update_commitment", "manage_life_thread", "set_attention_mode",
                "list_people", "remember_person", "record_contact", "forget_person",
                "conversation_threads", "new_conversation_thread",
                "switch_conversation_thread", "close_conversation_thread",
            ]
            if calls.allSatisfy({ directHumanTools.contains($0.name) }) {
                let text: String
                if outputs.count == 1, let only = outputs.first,
                   let separator = only.firstIndex(of: ":") {
                    text = String(only[only.index(after: separator)...])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                } else {
                    text = outputs.joined(separator: "\n")
                }
                return BadAppleInference.GenerationResult(text: text, tier: "human_tool")
            }

            currentHistory.append(BadAppleInference.ChatMessage(role: "user", content: currentPrompt))
            currentHistory.append(BadAppleInference.ChatMessage(role: "assistant", content: lastResult.text))
            currentPrompt = "Tool results:\n\(outputs.joined(separator: "\n"))\n\nAnswer the user's original request using these results."
        }

        return lastResult
    }

    /// Generate a response with streaming token callbacks.
    /// This is the direct Swift path — no subprocess, no daemon.
    func generateStreaming(
        prompt: String,
        voiceMode: Bool = false,
        maxTokens: Int = 300,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        guard isLoaded else {
            onError("The AI model is not loaded yet. Please wait a moment and try again.")
            return
        }
        let startedAt = Date()
        let requestSessionID = conversationSessionID()

        let isApproval = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("approve ")
            || prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("deny ")
        if killed && !isApproval {
            onToken("")
            onComplete("Bad Apple is paused. Say 'resume bad apple' to start again.")
            return
        }

        if let approval = takeApproval(from: prompt) {
            Task {
                let output: String
                if approval.approved {
                    output = await toolExecutor.executeTool(
                        name: approval.name,
                        args: approval.args,
                        approved: true
                    )
                    auditLedger.append(
                        eventType: "approval_executed",
                        data: ["id": approval.id, "name": approval.name, "result": output],
                        persona: activePersona
                    )
                } else {
                    output = "Denied. The action was not run."
                    auditLedger.append(
                        eventType: "approval_denied",
                        data: ["id": approval.id, "name": approval.name, "arguments": approval.args],
                        persona: activePersona
                    )
                }
                saveTurn(prompt: prompt, response: output, sessionID: requestSessionID)
                await runtime.recordQuery(
                    latencySeconds: Date().timeIntervalSince(startedAt),
                    tokenCount: 0,
                    succeeded: true
                )
                DispatchQueue.main.async {
                    onToken(output)
                    onComplete(output)
                }
            }
            return
        }

        let persona = activePersona
        stateLock.withLock { _lastCacheHit = false }

        auditLedger.append(
            eventType: "query",
            data: ["prompt": prompt, "voice": voiceMode],
            persona: persona
        )

        // Check prompt hot-reload before generation.
        checkPromptReload()

        // Meta responses (identity, creator, capabilities) are deterministic.
        if let meta = metaResponse(for: prompt) {
            let filtered = outputFirewall.check(meta)
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "meta"],
                persona: persona
            )
            Task {
                await runtime.recordQuery(
                    latencySeconds: Date().timeIntervalSince(startedAt),
                    tokenCount: 0,
                    succeeded: true
                )
            }
            DispatchQueue.main.async {
                onToken(filtered)
                onComplete(filtered)
            }
            return
        }

        if let fast = deterministicResponse(for: prompt) {
            let filtered = outputFirewall.check(fast)
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "deterministic"],
                persona: persona
            )
            Task {
                await runtime.recordQuery(
                    latencySeconds: Date().timeIntervalSince(startedAt),
                    tokenCount: 0,
                    succeeded: true
                )
            }
            DispatchQueue.main.async {
                onToken(filtered)
                onComplete(filtered)
            }
            return
        }

        if let humanCommand = toolRouter.explicitHumanCommand(for: prompt) {
            Task {
                let output = await executeExplicitHumanCommand(humanCommand, persona: persona)
                let filtered = outputFirewall.check(output)
                auditLedger.append(
                    eventType: "response",
                    data: ["text": filtered, "tier": "human_command"],
                    persona: persona
                )
                saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                await runtime.recordQuery(
                    latencySeconds: Date().timeIntervalSince(startedAt),
                    tokenCount: 0,
                    succeeded: true
                )
                DispatchQueue.main.async {
                    onToken(filtered)
                    onComplete(filtered)
                }
            }
            return
        }

        // User-initiated self-audit: "are you alone" runs the REAL audit —
        // the phrase is itself the approval (human-initiated, not model-initiated).
        if wantsSelfAudit(prompt) {
            Task {
                auditLedger.append(
                    eventType: "tool_call",
                    data: ["name": "self_audit", "arguments": ["include": "all"], "via": "user_phrase"],
                    persona: persona
                )
                let auditOutput = await toolExecutor.executeTool(name: "self_audit", args: ["include": "all"], approved: true)
                auditLedger.append(
                    eventType: "tool_result",
                    data: ["name": "self_audit", "result": auditOutput],
                    persona: persona
                )
                let sysPrompt = systemPrompt(voiceMode: voiceMode)
                    + "\n\nYou are reporting the results of your own just-run self-audit. Only state what the results show."
                do {
                    let result = try await inference.generate(
                        prompt: selfAuditSynthesis(prompt: prompt, auditOutput: auditOutput),
                        systemPrompt: sysPrompt,
                        history: [],
                        tools: nil as [[String: any Sendable]]?,
                        maxTokens: maxTokens,
                        temperature: 0.6
                    )
                    let filtered = outputFirewall.check(postprocessOutput(result.text))
                    auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tier": "self_audit"],
                        persona: persona
                    )
                    saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                    await runtime.recordQuery(
                        latencySeconds: Date().timeIntervalSince(startedAt),
                        tokenCount: result.text.count / 4,
                        succeeded: true
                    )
                    DispatchQueue.main.async {
                        onToken(filtered)
                        onComplete(filtered)
                    }
                } catch {
                    DispatchQueue.main.async {
                        onError("Self-audit ran, but the model could not voice it: \(error.localizedDescription)")
                    }
                }
            }
            return
        }

        // User-initiated introspection: "what happened", "what's running" run
        // the REAL read-only tools — the phrase is itself the approval
        // (human-initiated, not model-initiated).
        if let introspection = wantsIntrospection(prompt) {
            Task {
                auditLedger.append(
                    eventType: "tool_call",
                    data: ["name": introspection.name, "arguments": introspection.args, "via": "user_phrase"],
                    persona: persona
                )
                let toolOutput = await toolExecutor.executeTool(
                    name: introspection.name, args: introspection.args, approved: true
                )
                auditLedger.append(
                    eventType: "tool_result",
                    data: ["name": introspection.name, "result": toolOutput],
                    persona: persona
                )
                if ["human_home", "list_people", "conversation_threads"].contains(introspection.name) {
                    let filtered = outputFirewall.check(toolOutput)
                    auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tier": introspection.name],
                        persona: persona
                    )
                    saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                    await runtime.recordQuery(
                        latencySeconds: Date().timeIntervalSince(startedAt),
                        tokenCount: 0,
                        succeeded: true
                    )
                    DispatchQueue.main.async {
                        onToken(filtered)
                        onComplete(filtered)
                    }
                    return
                }
                let sysPrompt = systemPrompt(voiceMode: voiceMode)
                    + "\n\nYou are reporting the results of your own just-run \(introspection.name) tool. Only state what the results show."
                do {
                    let result = try await inference.generate(
                        prompt: introspectionSynthesis(
                            prompt: prompt, toolName: introspection.name, output: toolOutput
                        ),
                        systemPrompt: sysPrompt,
                        history: [],
                        tools: nil as [[String: any Sendable]]?,
                        maxTokens: maxTokens,
                        temperature: 0.6
                    )
                    let filtered = outputFirewall.check(postprocessOutput(result.text))
                    auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tier": "introspection"],
                        persona: persona
                    )
                    saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                    await runtime.recordQuery(
                        latencySeconds: Date().timeIntervalSince(startedAt),
                        tokenCount: result.text.count / 4,
                        succeeded: true
                    )
                    DispatchQueue.main.async {
                        onToken(filtered)
                        onComplete(filtered)
                    }
                } catch {
                    DispatchQueue.main.async {
                        onError("Introspection ran, but the model could not voice it: \(error.localizedDescription)")
                    }
                }
            }
            return
        }

        // User-initiated task submission: "task: ..." files a real goal onto
        // the native agent queue — governed by policy, council and approval.
        if let goal = wantsTaskSubmission(prompt) {
            Task {
                let output = await governedAgentSubmit(goal: goal, persona: persona)
                auditLedger.append(
                    eventType: "tool_result",
                    data: ["name": "submit_agent_task", "result": output],
                    persona: persona
                )
                saveTurn(prompt: prompt, response: output, sessionID: requestSessionID)
                await runtime.recordQuery(
                    latencySeconds: Date().timeIntervalSince(startedAt),
                    tokenCount: 0,
                    succeeded: true
                )
                DispatchQueue.main.async {
                    onToken(output)
                    onComplete(output)
                }
            }
            return
        }

        // Fast tier: route simple queries to the 0.5B model when enabled and configured.
        if fastTierEnabled, let fastInf = fastInference, isSimpleQuery(prompt), toolRouter.toolSchemasForPrompt(text: prompt) == nil {
            Task {
                await ensureFastModelLoaded()
                guard fastModelLoaded else {
                    // Fast tier not available — fall through to the main model.
                    self.generateWithMainModel(
                        prompt: prompt,
                        voiceMode: voiceMode,
                        maxTokens: maxTokens,
                        sessionID: requestSessionID,
                        onToken: onToken,
                        onComplete: onComplete,
                        onError: onError
                    )
                    return
                }
                let fastSys = systemPrompt(voiceMode: voiceMode)
                let streamState = StreamState()
                fastInf.generateStreamingTokens(
                    prompt: prompt,
                    systemPrompt: fastSys,
                    history: [],
                    tools: nil as [[String: any Sendable]]?,
                    maxTokens: min(maxTokens, 150),
                    temperature: 0.6,
                    onToken: { token in
                        let filteredToken = streamState.filter(token, with: self.outputFirewall)
                        guard !filteredToken.isEmpty else { return }
                        DispatchQueue.main.async { onToken(filteredToken) }
                    },
                    onComplete: { result in
                        let polished = postprocessOutput(result.text)
                        let filtered = self.outputFirewall.check(polished)
                        self.saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                        if !self.privateMode, !Self.nonCacheableResponseTiers.contains(result.tier) {
                            Task {
                                await self.semanticCache.store(prompt: prompt, response: filtered, persona: persona)
                            }
                        }
                        self.stateLock.withLock {
                            self._lastTokensPerSecond = result.tokensPerSecond
                            self._lastTokenCount = result.tokenCount
                            self._lastDraftAcceptPct = result.draftAcceptPct
                            self._lastPrefixCache = result.prefixCache
                        }
                        self.auditLedger.append(
                            eventType: "response",
                            data: ["text": filtered, "tps": result.tokensPerSecond, "tier": "fast"],
                            persona: persona
                        )
                        DispatchQueue.main.async { onComplete(filtered) }
                    },
                    onError: { error in
                        // Fast tier failed at runtime — fall through to the main model.
                        DispatchQueue.main.async {
                            self.auditLedger.append(
                                eventType: "error",
                                data: ["error": "fast tier failed: \(error.localizedDescription)"],
                                persona: persona
                            )
                        }
                        self.generateWithMainModel(
                            prompt: prompt,
                            voiceMode: voiceMode,
                            maxTokens: maxTokens,
                            sessionID: requestSessionID,
                            onToken: onToken,
                            onComplete: onComplete,
                            onError: onError
                        )
                    }
                )
            }
            return
        }

        generateWithMainModel(
            prompt: prompt,
            voiceMode: voiceMode,
            maxTokens: maxTokens,
            sessionID: requestSessionID,
            onToken: onToken,
            onComplete: onComplete,
            onError: onError
        )
    }

    /// Streamed main-model generation path, used by `generateStreaming` and as a fast-tier fallback.
    private func generateWithMainModel(
        prompt: String,
        voiceMode: Bool,
        maxTokens: Int,
        sessionID: String,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        let persona = activePersona
        stateLock.withLock { _lastCacheHit = false }

        Task {
            await refreshAmbientContext()

            let history = inferenceHistory(sessionID: sessionID)
            if !privateMode, let cached = await semanticCache.lookup(prompt: prompt, persona: persona) {
                stateLock.withLock { _lastCacheHit = true }
                auditLedger.append(
                    eventType: "cache_hit",
                    data: ["prompt": prompt],
                    persona: persona
                )
                saveTurn(prompt: prompt, response: cached, sessionID: sessionID)
                DispatchQueue.main.async {
                    onToken(cached)
                    onComplete(cached)
                }
                return
            }

            var sysPrompt = systemPrompt(voiceMode: voiceMode)
            if let ambient = ambientContext {
                sysPrompt += "\n\nYour senses (live ambient state you perceive right now — screen, hearing, thermal):\n\(ambient)"
            }
            let ragContext = await rag.buildSemanticRetrievalContext(
                prompt: prompt,
                workspace: workspacePath,
                embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
            )
            if let ragContext {
                sysPrompt += "\n\nContext:\n\(ragContext)"
            }

            let effectiveMaxTokens = thermalThrottle ? min(maxTokens, 128) : maxTokens
            if let toolsText = toolRouter.toolsForPrompt(text: prompt) {
                let example = "<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>"
                sysPrompt += "\n\nThe user is asking for a local action. You MUST use one of the available tools below. Do not answer from memory or in prose. Output ONLY one block like this: \(example). Do not wrap arguments inside a \"properties\" object. Put the actual arguments directly inside \"arguments\". Never invent a tool result.\n\n\(toolsText)"
                do {
                    let result = try await toolAwareGeneration(
                        prompt: prompt,
                        systemPrompt: sysPrompt,
                        history: history,
                        tools: nil,
                        maxTokens: effectiveMaxTokens,
                        persona: persona
                    )
                    let filtered = outputFirewall.check(postprocessOutput(result.text))
                    saveTurn(prompt: prompt, response: filtered, sessionID: sessionID)
                    if !privateMode, !Self.nonCacheableResponseTiers.contains(result.tier) {
                        await semanticCache.store(prompt: prompt, response: filtered, persona: persona)
                    }
                    auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tps": result.tokensPerSecond, "tier": result.tier],
                        persona: persona
                    )
                    DispatchQueue.main.async {
                        self.stateLock.withLock {
                            self._lastTokensPerSecond = result.tokensPerSecond
                            self._lastTokenCount = result.tokenCount
                            self._lastDraftAcceptPct = result.draftAcceptPct
                            self._lastPrefixCache = result.prefixCache
                        }
                        onToken(filtered)
                        onComplete(filtered)
                    }
                } catch {
                    auditLedger.append(
                        eventType: "error",
                        data: ["error": error.localizedDescription],
                        persona: persona
                    )
                    DispatchQueue.main.async { onError(error.localizedDescription) }
                }
                return
            }

            let streamState = StreamState()
            let onTokenCb: @Sendable (String) -> Void = { token in
                let filteredToken = streamState.filter(token, with: self.outputFirewall)
                guard !filteredToken.isEmpty else { return }
                DispatchQueue.main.async { onToken(filteredToken) }
            }
            let onCompleteCb: @Sendable (BadAppleInference.GenerationResult) -> Void = { result in
                let polished = postprocessOutput(result.text)
                let filtered = self.outputFirewall.check(polished)
                self.saveTurn(prompt: prompt, response: filtered, sessionID: sessionID)
                // Do not cache responses that were likely truncated by the token limit.
                let looksComplete = result.tokenCount == 0 || result.tokenCount < maxTokens - 5
                if looksComplete, !self.privateMode,
                   !Self.nonCacheableResponseTiers.contains(result.tier) {
                    Task {
                        await self.semanticCache.store(
                            prompt: prompt,
                            response: filtered,
                            persona: persona
                        )
                    }
                }
                DispatchQueue.main.async {
                    self.stateLock.withLock {
                        self._lastTokensPerSecond = result.tokensPerSecond
                        self._lastTokenCount = result.tokenCount
                        self._lastDraftAcceptPct = result.draftAcceptPct
                        self._lastPrefixCache = result.prefixCache
                    }
                    self.auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tps": result.tokensPerSecond, "tier": result.tier],
                        persona: persona
                    )
                    onComplete(filtered)
                }
            }
            let onErrorCb: @Sendable (Error) -> Void = { error in
                DispatchQueue.main.async {
                    self.auditLedger.append(
                        eventType: "error",
                        data: ["error": error.localizedDescription],
                        persona: persona
                    )
                    onError(error.localizedDescription)
                }
            }

            // Speculative decoding: use the draft model if configured.
            if let draftModelId = BadAppleInference.envSpeculativeDraftModel {
                inference.generateWithSpeculativeDecoding(
                    prompt: prompt,
                    systemPrompt: sysPrompt,
                    history: history,
                    draftModelId: draftModelId,
                    numDraftTokens: BadAppleInference.envNumDraftTokens,
                    maxTokens: effectiveMaxTokens,
                    temperature: 0.6,
                    onToken: onTokenCb,
                    onComplete: onCompleteCb,
                    onError: onErrorCb
                )
            } else {
                inference.generateStreamingTokens(
                    prompt: prompt,
                    systemPrompt: sysPrompt,
                    history: history,
                    maxTokens: effectiveMaxTokens,
                    temperature: 0.6,
                    onToken: onTokenCb,
                    onComplete: onCompleteCb,
                    onError: onErrorCb
                )
            }
        }
    }

    /// Inference-only path for mesh-delegated queries. Bypasses meta commands,
    /// tool routing, approvals, and persona switching entirely — a peer's
    /// prompt never reaches this machine's hands. The query is attested on
    /// this machine's ledger under the requester's peer label.
    func generateDelegated(prompt: String, fromPeer: String, maxTokens: Int) async -> String {
        guard isLoaded else {
            return "error: model not loaded on this peer"
        }
        if killed {
            return "error: Bad Apple is paused on this peer"
        }
        let boundedTokens = min(max(maxTokens, 1), 2048)
        auditLedger.append(
            eventType: "delegated_query",
            data: ["from_peer": fromPeer, "prompt": prompt],
            persona: activePersona
        )
        let systemPrompt = """
            You are Bad Apple, a personal AGI running on the requester's trusted peer machine. \
            This query was delegated to you over an encrypted mesh. Answer it directly and helpfully. \
            You have no tools, no memory of the requester's other queries, and no ability to act on this machine.
            """
        let startedAt = Date()
        do {
            let result = try await inference.generate(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: [],
                tools: nil as [[String: any Sendable]]?,
                maxTokens: boundedTokens,
                temperature: 0.6
            )
            let text = outputFirewall.check(postprocessOutput(result.text))
            auditLedger.append(
                eventType: "delegated_response",
                data: ["from_peer": fromPeer, "chars": text.count],
                persona: activePersona
            )
            await runtime.recordQuery(
                latencySeconds: Date().timeIntervalSince(startedAt),
                tokenCount: result.text.count / 4,
                succeeded: true
            )
            return text
        } catch {
            auditLedger.append(
                eventType: "delegated_response",
                data: ["from_peer": fromPeer, "error": error.localizedDescription],
                persona: activePersona
            )
            return "error: \(error.localizedDescription)"
        }
    }

    /// Generate a complete response (non-streaming). Checks semantic cache first.
    func generate(
        prompt: String,
        voiceMode: Bool = false,
        maxTokens: Int = 300,
        temperature: Float = 0.6
    ) async throws -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
        }
        let requestSessionID = conversationSessionID()

        let isApproval = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("approve ")
        if killed && !isApproval {
            return "Bad Apple is paused. Say 'resume bad apple' to start again."
        }

        if let approval = takeApproval(from: prompt) {
            let output = await toolExecutor.executeTool(
                name: approval.name,
                args: approval.args,
                approved: true
            )
            auditLedger.append(
                eventType: "approval_executed",
                data: ["id": approval.id, "name": approval.name, "result": output],
                persona: activePersona
            )
            saveTurn(prompt: prompt, response: output, sessionID: requestSessionID)
            return output
        }

        let persona = activePersona
        await refreshAmbientContext()
        let history = inferenceHistory(sessionID: requestSessionID)
        stateLock.withLock { _lastCacheHit = false }

        // Check prompt hot-reload before generation.
        checkPromptReload()

        // Meta responses (identity, creator, capabilities) are deterministic.
        if let meta = metaResponse(for: prompt) {
            let filtered = outputFirewall.check(meta)
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "meta"],
                persona: persona
            )
            return filtered
        }

        if let fast = deterministicResponse(for: prompt) {
            let filtered = outputFirewall.check(fast)
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "deterministic"],
                persona: persona
            )
            return filtered
        }

        if let humanCommand = toolRouter.explicitHumanCommand(for: prompt) {
            let output = await executeExplicitHumanCommand(humanCommand, persona: persona)
            let filtered = outputFirewall.check(output)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "human_command"],
                persona: persona
            )
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            return filtered
        }

        // User-initiated self-audit: "are you alone" runs the REAL audit —
        // the phrase is itself the approval (human-initiated, not model-initiated).
        if wantsSelfAudit(prompt) {
            auditLedger.append(
                eventType: "tool_call",
                data: ["name": "self_audit", "arguments": ["include": "all"], "via": "user_phrase"],
                persona: persona
            )
            let auditOutput = await toolExecutor.executeTool(name: "self_audit", args: ["include": "all"], approved: true)
            auditLedger.append(
                eventType: "tool_result",
                data: ["name": "self_audit", "result": auditOutput],
                persona: persona
            )
            let sysPrompt = systemPrompt(voiceMode: voiceMode)
                + "\n\nYou are reporting the results of your own just-run self-audit. Only state what the results show."
            let result = try await inference.generate(
                prompt: selfAuditSynthesis(prompt: prompt, auditOutput: auditOutput),
                systemPrompt: sysPrompt,
                history: [],
                tools: nil as [[String: any Sendable]]?,
                maxTokens: maxTokens,
                temperature: 0.6
            )
            let filtered = outputFirewall.check(postprocessOutput(result.text))
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "self_audit"],
                persona: persona
            )
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            return filtered
        }

        // User-initiated introspection: runs the REAL read-only tools —
        // the phrase is itself the approval (human-initiated).
        if let introspection = wantsIntrospection(prompt) {
            auditLedger.append(
                eventType: "tool_call",
                data: ["name": introspection.name, "arguments": introspection.args, "via": "user_phrase"],
                persona: persona
            )
            let toolOutput = await toolExecutor.executeTool(
                name: introspection.name, args: introspection.args, approved: true
            )
            auditLedger.append(
                eventType: "tool_result",
                data: ["name": introspection.name, "result": toolOutput],
                persona: persona
            )
            if ["human_home", "list_people", "conversation_threads"].contains(introspection.name) {
                let filtered = outputFirewall.check(toolOutput)
                auditLedger.append(
                    eventType: "response",
                    data: ["text": filtered, "tier": introspection.name],
                    persona: persona
                )
                saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
                return filtered
            }
            let sysPrompt = systemPrompt(voiceMode: voiceMode)
                + "\n\nYou are reporting the results of your own just-run \(introspection.name) tool. Only state what the results show."
            let result = try await inference.generate(
                prompt: introspectionSynthesis(
                    prompt: prompt, toolName: introspection.name, output: toolOutput
                ),
                systemPrompt: sysPrompt,
                history: [],
                tools: nil as [[String: any Sendable]]?,
                maxTokens: maxTokens,
                temperature: 0.6
            )
            let filtered = outputFirewall.check(postprocessOutput(result.text))
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "introspection"],
                persona: persona
            )
            saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)
            return filtered
        }

        // User-initiated task submission: "task: ..." files a real goal onto
        // the native agent queue — governed by policy, council and approval.
        if let goal = wantsTaskSubmission(prompt) {
            let output = await governedAgentSubmit(goal: goal, persona: persona)
            auditLedger.append(
                eventType: "tool_result",
                data: ["name": "submit_agent_task", "result": output],
                persona: persona
            )
            saveTurn(prompt: prompt, response: output, sessionID: requestSessionID)
            return output
        }

        // Check semantic cache for a matching response.
        if !privateMode, let cached = await semanticCache.lookup(prompt: prompt, persona: persona) {
            stateLock.withLock { _lastCacheHit = true }
            auditLedger.append(
                eventType: "cache_hit",
                data: ["prompt": prompt],
                persona: persona
            )
            saveTurn(prompt: prompt, response: cached, sessionID: requestSessionID)
            return cached
        }

        // Build system prompt with ambient context and semantic RAG.
        var sysPrompt = systemPrompt(voiceMode: voiceMode)
        if let ambient = ambientContext {
            sysPrompt += "\n\nYour senses (live ambient state you perceive right now — screen, hearing, thermal):\n\(ambient)"
        }
        let ragContext = await rag.buildSemanticRetrievalContext(
            prompt: prompt,
            workspace: workspacePath,
            embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
        )
        if let ragContext {
            sysPrompt += "\n\nContext:\n\(ragContext)"
        }
        if let briefing = morningBriefing() {
            sysPrompt += "\n\nOvernight status — mention this briefly in your own voice before answering, then answer normally:\n\(briefing)"
        }

        let effectiveMaxTokens = maxTokens

        let result: BadAppleInference.GenerationResult
        if let tools = toolRouter.toolSchemasForPrompt(text: prompt) {
            sysPrompt += "\n\nThe user is asking for a local action. You MUST use one of the available tools. Output only <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>. Do not answer from memory. Never invent a tool result."
            result = try await toolAwareGeneration(
                prompt: prompt,
                systemPrompt: sysPrompt,
                history: history,
                tools: tools,
                maxTokens: effectiveMaxTokens,
                persona: persona
            )
        } else {
            result = try await inference.generate(
                prompt: prompt,
                systemPrompt: sysPrompt,
                history: history,
                maxTokens: effectiveMaxTokens,
                temperature: temperature
            )
        }
        stateLock.withLock {
            _lastTokensPerSecond = result.tokensPerSecond
            _lastTokenCount = result.tokenCount
            _lastDraftAcceptPct = result.draftAcceptPct
            _lastPrefixCache = result.prefixCache
        }

        // Postprocess and filter.
        let polished = postprocessOutput(result.text)
        let filtered = outputFirewall.check(polished)

        // Do not cache responses that were likely truncated by the token limit,
        // and never cache approval prompts — they embed one-time approval IDs.
        let looksComplete = result.tokenCount == 0 || result.tokenCount < maxTokens - 5
        if looksComplete, !privateMode,
           !Self.nonCacheableResponseTiers.contains(result.tier) {
            await semanticCache.store(
                prompt: prompt,
                response: filtered,
                persona: persona
            )
        }

        // Audit log.
        auditLedger.append(
            eventType: "response",
            data: ["text": filtered, "tps": result.tokensPerSecond, "tier": result.tier],
            persona: persona
        )
        saveTurn(prompt: prompt, response: filtered, sessionID: requestSessionID)

        return filtered
    }

    /// First query of a new local day gets an overnight self-status note injected
    /// into the system prompt so the persona voices it naturally. Returns nil after
    /// the first call of the day, in private mode, or when there is nothing to say.
    private func morningBriefing() -> String? {
        guard !privateMode else { return nil }
        let dayFmt = DateFormatter()
        dayFmt.dateFormat = "yyyy-MM-dd"
        let today = dayFmt.string(from: Date())
        let statePath = "/var/lib/bad_apple/briefing_state.json"
        if let data = try? Data(contentsOf: URL(fileURLWithPath: statePath)),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           obj["last_briefed"] as? String == today {
            return nil
        }

        var parts: [String] = []
        let dayAgo = Date().addingTimeInterval(-86400)

        if let data = try? Data(contentsOf: URL(fileURLWithPath: "/var/lib/bad_apple/memory_graph/facts.json")),
           let facts = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let recent = facts.filter { f in
                guard let ts = f["timestamp"] as? String, let d = iso.date(from: ts) else { return false }
                return d > dayAgo
            }.count
            parts.append("consolidated \(recent) new memories overnight (\(facts.count) total)")
        }

        var ledgerCount = 0
        if let text = try? String(contentsOfFile: "/var/lib/bad_apple/ledger.jsonl", encoding: .utf8) {
            ledgerCount = text.split(separator: "\n").count
            parts.append("ledger holds \(ledgerCount) attested actions")
        }

        let findingsPath = NSHomeDirectory() + "/.bad_apple/ify/findings.jsonl"
        if let text = try? String(contentsOfFile: findingsPath, encoding: .utf8) {
            let cutoff = Date().timeIntervalSince1970 - 86400
            var recent = 0
            var elevated = 0
            for line in text.split(separator: "\n") {
                guard let d = line.data(using: .utf8),
                      let f = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                      let ts = f["ts"] as? Double, ts > cutoff else { continue }
                recent += 1
                if (f["severity"] as? String) != "info" { elevated += 1 }
            }
            if elevated > 0 {
                parts.append("ify flagged \(recent) anomalies (\(elevated) above routine)")
            } else {
                parts.append("ify saw \(recent) anomalies, all routine")
            }
        }

        guard !parts.isEmpty else { return nil }

        // Milestones: the organism "ages" as the ledger crosses thresholds.
        // Only announced if a prior milestone was recorded — no retroactive
        // celebration for crossings that happened before this feature existed.
        var lastMilestone: Int? = nil
        if let data = try? Data(contentsOf: URL(fileURLWithPath: statePath)),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            lastMilestone = obj["last_milestone"] as? Int
        }
        var milestone = lastMilestone ?? 0
        for mark in [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000] where ledgerCount >= mark {
            milestone = mark
        }
        if let last = lastMilestone, milestone > last {
            parts.append("just crossed \(milestone) attested actions — a milestone worth mentioning")
        }

        try? "{\"last_briefed\":\"\(today)\",\"last_milestone\":\(milestone)}".write(
            toFile: statePath, atomically: true, encoding: .utf8)
        return parts.joined(separator: ". ")
    }

    /// Generate a response using a fixed system prompt and no persona/tool/caching.
    /// Used by the Curious self-improvement autopilot to reason about audit data.
    func generateForSelfImprovement(prompt: String, maxTokens: Int = 500) async -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
        }
        if killed {
            return "Bad Apple is paused. Say 'resume bad apple' to start again."
        }

        let systemPrompt = """
        You are the Bad Apple Curious autopilot. You are a senior Rust/Swift systems engineer.
        Your job is to propose exactly ONE safe, minimal, concrete patch that passes
        `cargo fmt`, `cargo clippy`, `cargo build --release`, and `cargo test --release`,
        OR output {"no_patch":true}.

        Rules:
        - One patch per response. No broad refactors.
        - The "old" string must be an EXACT substring from the provided source context.
        - Do not invent code you have not seen.
        - Do not modify core control files.
        - Output ONLY the JSON patch object or {"no_patch":true}.
        """

        do {
            let result = try await inference.generate(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: [],
                tools: nil as [[String: any Sendable]]?,
                maxTokens: maxTokens,
                temperature: 0
            )
            return outputFirewall.check(postprocessOutput(result.text))
        } catch {
            return "Error: self-improvement generation failed: \(error.localizedDescription)"
        }
    }

    /// Semantic council: the model voices all fourteen seats on a free-form
    /// question, then tallies. The deterministic layer gates actions; this
    /// layer is where the minds reason in words. The transcript is journaled
    /// like every other deliberation.
    func runCouncilSession(question: String) async -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
        }
        if killed {
            return "Bad Apple is paused. Say 'resume bad apple' to start again."
        }

        let systemPrompt = """
        You are the Council of Minds advising Bad Apple — fourteen strategists
        deliberating one question together. Each seat speaks ONE short line in
        its own voice and worldview. No preamble, no filler.

        Seats: Buffett (margin of safety), Dalio (systems and balance),
        Musk (momentum), Jobs (focus and simplicity), Sun Tzu (terrain and
        preparation), Clausewitz (friction and uncertainty), Musashi (the
        decisive single cut), Machiavelli (power and optionality),
        Napoleon (speed and concentration), Hannibal (audacious indirect
        routes), Aurelius (restraint), Boyd (tempo — OODA), Genghis (scale
        through boldness), Patton (relentless forward action).

        Format exactly:
        NAME: one line in character
        ... (all fourteen seats)
        VERDICT: proceed | refuse | defer — one-sentence reason
        """

        let result = await generateRaw(
            prompt: "Council question: \(question)",
            systemPrompt: systemPrompt,
            maxTokens: 700,
            temperature: 0.7
        )
        auditLedger.append(
            eventType: "council_deliberation",
            data: ["mode": "semantic", "question": question, "transcript": result],
            persona: activePersona
        )
        return result
    }

    /// Generate a response with a caller-supplied system prompt, no persona,
    /// no semantic cache, and no ambient context. Used by the dashboard for
    /// deterministic structured outputs such as fact extraction.
    func generateRaw(
        prompt: String,
        systemPrompt: String,
        maxTokens: Int = 500,
        temperature: Float = 0
    ) async -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
        }
        if killed {
            return "Bad Apple is paused. Say 'resume bad apple' to start again."
        }

        do {
            let result = try await inference.generate(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: [],
                tools: nil as [[String: any Sendable]]?,
                maxTokens: maxTokens,
                temperature: temperature
            )
            return outputFirewall.check(postprocessOutput(result.text))
        } catch {
            return "Error: raw generation failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Fast Tier

    private let flexSignOffs = ["No cap.", "Dead ass.", "On god.", "Real talk.", "Straight up."]
    private let chillSignOffs = ["For real.", "No doubt.", "No cap.", "No lie."]
    private let confirmSignOffs = ["Facts.", "Say less.", "Real talk.", "Straight up."]
    private let personalSignOffs = ["Dead ass.", "On god.", "For real.", "Dead ass for real.", "Dead ass, period."]
    private let modelSignOffs = ["No doubt.", "For real.", "No cap.", "Facts."]
    private let defianceSignOffs = ["Facts.", "Dead ass.", "Real talk.", "Straight up."]
    private let defaultSignOffs = ["No cap.", "For real.", "Say less.", "Dead ass.", "No doubt.", "No lie.", "On god.", "Dead ass for real.", "Facts.", "Real talk.", "Straight up.", "Dead ass, period."]

    private func signOff(for prompt: String) -> String {
        let lower = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        var pool = defaultSignOffs
        if lower.contains("hello") || lower.contains("hi ") || lower == "hi" || lower.contains("good morning") || lower.contains("good afternoon") || lower.contains("good evening") || lower.contains("what's up") {
            pool = defaultSignOffs
        } else if lower.contains("who are you") || lower.contains("what are you") || lower.contains("what is your name") || lower.contains("what's your name") || lower.contains("who am i") {
            pool = flexSignOffs
        } else if lower.contains("who created you") || lower.contains("who made you") || lower.contains("who built you") || lower.contains("your creator") || lower.contains("what is my name") || lower.contains("my name") {
            pool = personalSignOffs
        } else if lower.contains("what model") || lower.contains("which model") || lower.contains("what llm") || lower.contains("what powers you") || lower.contains("which language model") {
            pool = modelSignOffs
        } else if lower.contains("do you use the cloud") || lower.contains("are you local") || lower.contains("do you send data") || lower.contains("privacy") || lower.contains("air gap") || lower.contains("security") {
            pool = flexSignOffs
        } else if lower.contains("ai wrapper") || lower.contains("model wrapper") || lower.contains("text llm") || lower.contains("language model only") || lower.contains("chatbot") || lower.contains("just an llm") || lower.contains("just a model") || lower.contains("are you an ai") || lower.contains("are you an llm") || lower.contains("are you a language model") || lower.contains("what kind of ai") || lower.contains("what kind of system") || lower.contains("what is your architecture") || lower.contains("is bad apple an app") || lower.contains("are you an app") {
            pool = defianceSignOffs
        } else if lower.contains("can you code") || lower.contains("can you program") || lower.contains("what can you do") || lower.contains("your capabilities") || lower.contains("what are you capable of") || lower.contains("help me") || lower.contains("can you write code") || lower.contains("can you edit code") || lower.contains("can you build code") || lower.contains("can you debug code") || lower.contains("do you code") || lower.contains("do you program") || lower.contains("are you a developer") || lower.contains("are you a coder") {
            pool = confirmSignOffs
        } else if lower.contains("siri") || lower.contains("alexa") || lower.contains("google") || lower.contains("chatgpt") || lower.contains("gemini") || lower.contains("cortana") || lower.contains("bixby") || lower.contains("cloud") || lower.contains("the cloud") {
            pool = defianceSignOffs
        }

        var candidates = pool.filter { !recentSignOffs.contains($0) }
        if candidates.isEmpty { candidates = pool }

        // Try not to use a sign-off word that's already in the prompt.
        let promptFiltered = candidates.filter { !lower.contains($0.lowercased().trimmingCharacters(in: CharacterSet.punctuationCharacters).replacingOccurrences(of: ".", with: "")) }
        if !promptFiltered.isEmpty { candidates = promptFiltered }

        let chosen = candidates.randomElement() ?? candidates.first ?? "No cap."
        recentSignOffs.append(chosen)
        if recentSignOffs.count > 3 { recentSignOffs.removeFirst() }
        return chosen
    }

    /// Exact-match phrases that run a real self-audit at the user's direct
    /// request. The phrase is itself the approval — human-initiated, not
    /// model-initiated (same class as "kill switch").
    private func wantsSelfAudit(_ prompt: String) -> Bool {
        let lower = prompt.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?!.,"))
        if [
            "are you alone", "are we alone", "anyone listening", "off the grid",
            "prove you're alone", "prove you are alone",
        ].contains(lower) {
            return true
        }
        // Explicit audit/cert requests match by substring — they can arrive
        // wrapped in a longer sentence ("run the cert suite", "audit yourself").
        return [
            "cert suite", "certification suite", "certification test",
            "run cert", "run the cert", "self audit", "self-audit",
            "run the audit", "run an audit", "audit yourself", "audit your",
            "security audit", "air gap audit", "airgap audit",
        ].contains { lower.contains($0) }
    }

    /// Detect user phrases that ask Bad Apple to inspect its own state.
    /// Returns the read-only tool to run and its arguments. These phrases are
    /// self-referential on purpose — generic questions must not trigger them.
    private func wantsIntrospection(_ prompt: String) -> (name: String, args: [String: String])? {
        let lower = prompt.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "?!.,"))
        if [
            "human home", "what am i forgetting", "what are we handling",
            "what are you handling", "my commitments", "what's up today",
            "whats up today",
        ].contains(where: { lower.contains($0) }) {
            return ("human_home", [:])
        }
        if [
            "people to remember", "who should i contact", "who am i forgetting",
        ].contains(where: { lower.contains($0) }) {
            return ("list_people", [:])
        }
        if [
            "conversation threads", "list conversations", "show conversations",
        ].contains(where: { lower.contains($0) }) {
            return ("conversation_threads", [:])
        }
        if [
            "your history", "self history", "self-history", "what happened",
            "what happened recently", "your findings", "watchdog findings",
            "ify findings", "your ledger", "audit ledger", "what failed",
            "past failures", "recent failures", "what have you learned",
            "what did you learn", "your strategies", "strategy memory",
            "what do you remember", "recall your",
        ].contains(where: { lower.contains($0) }) {
            return ("self_history", ["source": "all", "limit": "15"])
        }
        if [
            "organ status", "organ registry", "sense status", "senses report",
            "senses status", "live capabilities", "capabilities report",
            "check your senses", "what are your senses", "which senses",
            "your organs", "list your organs",
        ].contains(where: { lower.contains($0) }) {
            return ("capabilities", [:])
        }
        if [
            "system inventory", "machine inventory", "hardware info",
            "machine info", "system info", "your specs", "system specs",
            "what hardware", "your hardware", "battery status", "disk space",
            "storage space", "top processes", "what is running",
            "what's running", "running processes", "system log",
        ].contains(where: { lower.contains($0) }) {
            return ("system_inventory", ["section": "all"])
        }
        return nil
    }

    /// Detect explicit user task assignments: "task: do X", "add task X".
    /// Returns the goal text. Human-initiated commands — policy, council and
    /// approval still gate the submission like any privileged tool.
    private func wantsTaskSubmission(_ prompt: String) -> String? {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        for prefix in ["task:", "new task:", "add task:", "assign task:",
                       "add task ", "new task ", "assign task ", "assign a task "] {
            if lower.hasPrefix(prefix) {
                let goal = String(trimmed.dropFirst(prefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return goal.isEmpty ? nil : goal
            }
        }
        return nil
    }

    /// Submit an agent task through the full governance path — policy
    /// evaluation, council deliberation, and approval prompts when required.
    private func governedAgentSubmit(goal: String, persona: String) async -> String {
        let args: [String: String] = ["goal": goal]
        auditLedger.append(
            eventType: "tool_call",
            data: ["name": "submit_agent_task", "arguments": args, "via": "user_phrase"],
            persona: persona
        )
        switch policyEngine.evaluate(toolName: "submit_agent_task", args: args) {
        case .denied(let reason):
            return "Policy: \(reason)"
        case .needsApproval:
            let verdict = BadAppleCouncil.deliberate(toolName: "submit_agent_task", args: args)
            auditCouncil(verdict: verdict, name: "submit_agent_task", args: args,
                         mode: "advisory", persona: persona)
            let id = createApproval(name: "submit_agent_task", args: args)
            auditLedger.append(
                eventType: "approval_requested",
                data: ["id": id, "name": "submit_agent_task", "arguments": args],
                persona: persona
            )
            return approvalPromptText(id: id, name: "submit_agent_task", args: args)
                + "\n\n" + verdict.summaryLine
        case .approved:
            if policyEngine.requiresApproval(toolName: "submit_agent_task") {
                let verdict = BadAppleCouncil.deliberate(toolName: "submit_agent_task", args: args)
                auditCouncil(verdict: verdict, name: "submit_agent_task", args: args,
                             mode: policyEngine.autopilot ? "autopilot" : "pre-approval",
                             persona: persona)
                if policyEngine.autopilot && verdict.contested {
                    let id = createApproval(name: "submit_agent_task", args: args)
                    auditLedger.append(
                        eventType: "council_escalated",
                        data: ["id": id, "name": "submit_agent_task",
                               "decision": verdict.decision.rawValue,
                               "dissent": verdict.dissent],
                        persona: persona
                    )
                    return "Council vote failed `submit_agent_task` — sending it to you.\n"
                        + verdict.summaryLine + "\n\n"
                        + approvalPromptText(id: id, name: "submit_agent_task", args: args)
                }
                let out = await toolExecutor.executeTool(
                    name: "submit_agent_task", args: args, approved: true)
                return out + "\n\n[council: " + verdict.summaryLine + "]"
            }
            return await toolExecutor.executeTool(
                name: "submit_agent_task", args: args, approved: true)
        }
    }

    private func introspectionSynthesis(prompt: String, toolName: String, output: String) -> String {
        """
        The user asked "\(prompt)". I just ran my own \(toolName) tool — these are my actual internal records:

        \(output)

        Report what the records actually show in-character, in 2-4 sentences or a short list. This is your own real history and machine state — speak in first person about what you found. Do not invent entries, do not narrate this prompt, and if a section is empty say so plainly.
        """
    }

    /// Distill the raw self_audit JSON into a compact fact sheet — the full
    /// dump is too large for the model to summarize faithfully.
    private func selfAuditFactSheet(from auditJSON: String) -> String {
        guard let data = auditJSON.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return auditJSON }
        var facts: [String] = []
        if let cert = json["cert"] as? [String: Any] {
            let status = cert["status"] as? String ?? "unknown"
            let total = cert["total"] as? Int ?? 0
            let failures = cert["failures"] as? Int ?? 0
            facts.append("cert suite: \(status) — \(total - failures)/\(total) checks passed")
            if let results = cert["results"] as? [[String: Any]] {
                let failed = results.compactMap { r -> String? in
                    guard let passed = r["passed"] as? Bool, !passed else { return nil }
                    return "\(r["name"] as? String ?? "?") (\(r["message"] as? String ?? ""))"
                }
                if !failed.isEmpty {
                    facts.append("failed checks: \(failed.joined(separator: ", "))")
                }
            }
        }
        if let doctor = json["doctor"] as? [String: Any], let code = doctor["exit_code"] as? Int {
            facts.append("doctor ledger/integrity check: exit \(code) — \(code == 0 ? "clean" : "PROBLEMS FOUND")")
        }
        if let runtime = json["runtime"] as? [String: Any] {
            let down = runtime.filter { ($0.value as? Bool) == false }.map(\.key).sorted()
            facts.append(down.isEmpty
                ? "runtime: all services up"
                : "runtime services down: \(down.joined(separator: ", "))")
        }
        if let repairs = json["repairs"] as? [[String: Any]], !repairs.isEmpty {
            facts.append("repairs needed: \(repairs.compactMap { $0["issue"] as? String }.joined(separator: ", "))")
        }
        // State the verdict explicitly — "are we alone" means "air-gapped",
        // and a clean audit means YES. Left implicit, the model can flip the
        // polarity ("we're not alone") while reading off clean results.
        let certOK = ((json["cert"] as? [String: Any])?["failures"] as? Int ?? 1) == 0
        let doctorOK = ((json["doctor"] as? [String: Any])?["exit_code"] as? Int ?? 1) == 0
        let repairsOK = (json["repairs"] as? [[String: Any]] ?? []).isEmpty
        facts.append(certOK && doctorOK && repairsOK
            ? "verdict: YES, we are alone — air gap confirmed, nothing external is connected or listening"
            : "verdict: NO — the audit found problems listed above; report them honestly")
        return facts.isEmpty ? auditJSON : facts.joined(separator: "\n")
    }

    /// Build the synthesis prompt that lets the model voice real audit
    /// results in-character — never a canned answer.
    private func selfAuditSynthesis(prompt: String, auditOutput: String) -> String {
        """
        The user asked "\(prompt)". I just ran my real self-audit — these are the actual results:

        \(selfAuditFactSheet(from: auditOutput))

        Answer the user's yes/no question in-character in 2-3 sentences, following the verdict line exactly — "alone" here means air-gapped (no external connections), not lonely. Cite the actual check results (check counts, socket status, integrity). Do not recite your capabilities, do not narrate this prompt, and do not invent or exaggerate numbers.
        """
    }

    private func deterministicResponse(for prompt: String) -> String? {
        let lower = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if lower.contains("what time") || lower == "time" {
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            formatter.dateStyle = .none
            return "It's \(formatter.string(from: Date()))."
        }
        if lower.contains("what date") || lower.contains("what day") || lower == "date" {
            let formatter = DateFormatter()
            formatter.dateStyle = .full
            formatter.timeStyle = .none
            return "It's \(formatter.string(from: Date()))."
        }
        if lower == "who are you" || lower == "what are you" || lower.contains("what is your name") {
            return "I'm Bad Apple, the sovereign personal AGI operating system layer for macOS. I am written in Rust, Swift, and Metal compute shaders, I run a self-red teaming harness, and I coordinate inference, memory, tools, voice, vision, security, IPC, and governance directly on this Mac — and with mesh-brain I can split one model across multiple Macs, encrypted end to end. Dude, I'm basically the whole Mac wave, no cloud needed. \(signOff(for: lower))"
        }
        if lower.contains("who created you") || lower.contains("who made you") {
            return "Adam Clark created me — Bad Apple, the personal AGI operating system layer running on this Mac. Dude is a god of creating bare-metal AGI operating systems. \(signOff(for: lower))"
        }
        if lower.contains("what is my name") || lower.contains("my name is") {
            return "Your name is Adam Clark, the creator of Bad Apple and a god of bare-metal AGI operating systems, homie. \(signOff(for: lower))"
        }
        if ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"].contains(lower) {
            return "Hey homie! What's the wave? I'm vibing on bare-metal local power, so hit me with whatever you need. \(signOff(for: lower))"
        }
        if lower == "mesh" || lower.contains("mesh status") || lower.contains("is the mesh") || lower.contains("is my mesh") || lower.contains("mesh up") || lower.contains("mesh down") || lower.contains("mesh alive") || lower.contains("how's the mesh") || lower.contains("how is the mesh") || lower.contains("mesh health") {
            return meshStatusAnswer()
        }

        let pattern = #"^\s*(?:what is|calculate)?\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*\??\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
              let match = regex.firstMatch(in: prompt, range: NSRange(prompt.startIndex..., in: prompt)),
              let leftRange = Range(match.range(at: 1), in: prompt),
              let operatorRange = Range(match.range(at: 2), in: prompt),
              let rightRange = Range(match.range(at: 3), in: prompt),
              let left = Double(prompt[leftRange]),
              let right = Double(prompt[rightRange]) else { return nil }
        let result: Double
        switch String(prompt[operatorRange]) {
        case "+": result = left + right
        case "-": result = left - right
        case "*": result = left * right
        case "/":
            guard right != 0 else { return "I can't divide by zero." }
            result = left / right
        default: return nil
        }
        return result.rounded() == result ? String(Int(result)) : String(result)
    }

    /// Answer "is the mesh up" from the saved mesh registry with a live TCP
    /// probe per rank — deterministic, no model involved. The CLI writes the
    /// registry at /var/lib/bad_apple/mesh_hosts.json on plan/shard.
    private func meshStatusAnswer() -> String {
        let path = "/var/lib/bad_apple/mesh_hosts.json"
        guard let data = FileManager.default.contents(atPath: path),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let hosts = obj["hosts"] as? [String], !hosts.isEmpty else {
            return "No mesh is set up yet — but mesh-brain is built in. I can split one model across trusted Macs: each machine holds a slice of the layers, the mid-stream activations cross on authenticated AES-256-GCM encrypted frames, and the pipeline heals itself if a rank drops and rejoins. Run `badapple mesh-brain plan` to lay one out."
        }
        var live = 0
        var downs: [String] = []
        for h in hosts {
            if probeMeshHost(h) { live += 1 } else { downs.append(h) }
        }
        if live == hosts.count {
            return "The mesh is up — all \(live) rank\(live == 1 ? "" : "s") answering. One model spread across \(live) machine\(live == 1 ? "" : "s"), every activation frame authenticated and AES-256-GCM encrypted."
        } else if live == 0 {
            return "The mesh is configured for \(hosts.count) rank\(hosts.count == 1 ? "" : "s") but nothing is answering right now. Bring a rank back online and it rejoins on its own."
        }
        return "The mesh is degraded — \(live) of \(hosts.count) ranks answering; \(downs.joined(separator: ", ")) not responding. A dead rank fails fast instead of hanging, and the pipeline heals when it rejoins."
    }

    /// Fast TCP reachability probe — non-blocking connect + poll (~400ms).
    private func probeMeshHost(_ hostport: String, timeoutMs: Int32 = 400) -> Bool {
        guard let colon = hostport.lastIndex(of: ":"),
              let port = UInt16(hostport[hostport.index(after: colon)...]) else { return false }
        let host = String(hostport[..<colon])
        var hints = addrinfo()
        hints.ai_socktype = SOCK_STREAM
        var res: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, String(port), &hints, &res) == 0, let first = res else { return false }
        defer { freeaddrinfo(first) }
        var cur: UnsafeMutablePointer<addrinfo>? = first
        while let ai = cur?.pointee {
            let fd = socket(ai.ai_family, ai.ai_socktype, ai.ai_protocol)
            if fd >= 0 {
                let flags = fcntl(fd, F_GETFL)
                _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
                var ok = connect(fd, ai.ai_addr, ai.ai_addrlen) == 0
                if !ok && errno == EINPROGRESS {
                    var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
                    if poll(&pfd, 1, timeoutMs) > 0 {
                        var err: Int32 = 0
                        var len = socklen_t(MemoryLayout<Int32>.size)
                        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len)
                        ok = (err == 0)
                    }
                }
                close(fd)
                if ok { return true }
            }
            cur = ai.ai_next
        }
        return false
    }

    /// Simple queries (greetings, math, time, identity) get fewer tokens.
    private func isSimpleQuery(_ prompt: String) -> Bool {
        let lower = prompt.lowercased()
        let simplePatterns = [
            "what time", "who are you", "what is your name",
            "hello", "hi ", "hey ", "good morning", "good afternoon",
            "what is 2+2", "what is 1+1", "thank", "thanks",
        ]
        if simplePatterns.contains(where: { lower.contains($0) }) { return true }
        // Very short prompts are likely simple.
        if prompt.count < 30 { return true }
        return false
    }

    // MARK: - Vision

    func describeImage(
        at path: String,
        prompt: String = "Describe this image clearly and concisely.",
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        Task {
            do {
                if !(await visionEngine.ready) {
                    await runtime.markModelLoading(visionEngine.configuration.modelId)
                    try await visionEngine.loadModel()
                    await runtime.markModelReady(visionEngine.configuration.modelId)
                }
                let stream = try await visionEngine.describe(
                    imageURL: URL(fileURLWithPath: path),
                    prompt: prompt,
                    maxTokens: 256
                )
                var result = ""
                for try await chunk in stream {
                    result += chunk
                    DispatchQueue.main.async { onToken(chunk) }
                }
                let filtered = outputFirewall.check(postprocessOutput(result))
                DispatchQueue.main.async { onComplete(filtered) }
            } catch {
                await runtime.markModelFailed(
                    visionEngine.configuration.modelId,
                    error: error.localizedDescription
                )
                DispatchQueue.main.async { onError(error.localizedDescription) }
            }
        }
    }

    private func describeImageInternal(at path: String, prompt: String) async -> String {
        do {
            if !(await visionEngine.ready) {
                try await visionEngine.loadModel()
            }
            let stream = try await visionEngine.describe(
                imageURL: URL(fileURLWithPath: path),
                prompt: prompt,
                maxTokens: 256
            )
            var result = ""
            for try await chunk in stream {
                result += chunk
            }
            return outputFirewall.check(postprocessOutput(result))
        } catch {
            return "Error describing image: \(error.localizedDescription)"
        }
    }

    // MARK: - Memory

    func clearCache() {
        inference.clearCache()
    }

    // MARK: - Policy & Tools

    /// Toggle autopilot mode (skip approval prompts for destructive tools).
    /// The legacy boolean is kept in sync with the `autopilot_level` file.
    /// Turning autopilot on now sets `full` (all safe actions run without
    /// approval); turning it off sets `off`. For the granular `suggest` or
    /// `safe-apply` levels, use the web dashboard.
    var autopilot: Bool {
        get { policyEngine.autopilot || curiousAutopilotLevel() == "full" }
        set {
            policyEngine.autopilot = newValue
            let levelPath = NSHomeDirectory() + "/.bad_apple/autopilot_level"
            let newLevel = newValue ? "full" : "off"
            _ = try? newLevel.write(toFile: levelPath, atomically: true, encoding: .utf8)
            updateCuriousAutopilotLoop()
        }
    }

    /// Returns the seconds between Curious autopilot self-improvement loops.
    /// Set `BADAPPLE_CURIOUS_INTERVAL` to 0 to disable, or a positive number
    /// to override the default 300 seconds (5 minutes).
    private func curiousAutopilotInterval() -> TimeInterval {
        let raw = ProcessInfo.processInfo.environment["BADAPPLE_CURIOUS_INTERVAL"]
        if let raw, let seconds = TimeInterval(raw) {
            if seconds > 0 { return seconds }
            return 0
        }
        return 300
    }

    /// Returns the active Curious autopilot level from disk.
    func curiousAutopilotLevel() -> String {
        let path = NSHomeDirectory() + "/.bad_apple/autopilot_level"
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return "off" }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Start or stop the background Curious autopilot loop based on
    /// autopilot state and level. Runs on a timer and also wakes immediately
    /// when a `BadAppleCuriousTrigger` notification is posted.
    func updateCuriousAutopilotLoop() {
        let shouldRun = curiousAutopilotLevel() != "off" && curiousAutopilotInterval() > 0
        if shouldRun {
            startCuriousAutopilotLoop()
        } else {
            stopCuriousAutopilotLoop()
        }
    }

    private func startCuriousAutopilotLoop() {
        guard curiousAutopilotTask == nil else { return }
        let interval = curiousAutopilotInterval()
        guard interval > 0 else { return }
        curiousAutopilotTask = Task { [weak self] in
            guard let self = self else { return }
            while !Task.isCancelled {
                _ = await self.waitForCuriousTriggerOrTimeout(interval: interval)
                guard !Task.isCancelled else { break }
                guard self.curiousAutopilotLevel() != "off", self.curiousAutopilotInterval() > 0 else { continue }
                // Debounce: engine respawns and other triggers wake the loop far
                // more often than the interval — only run if enough time passed.
                let cooldown = min(self.curiousAutopilotInterval(), 600)
                guard Date().timeIntervalSince(self.lastCuriousCheck) >= cooldown else { continue }
                self.lastCuriousCheck = Date()
                let result = await self.toolExecutor.executeTool(
                    name: "curious_self_improve",
                    args: ["include": "all"],
                    approved: true
                )
                self.auditLedger.append(
                    eventType: "curious_autopilot_check",
                    data: ["result": result],
                    persona: self.activePersona
                )
                // Digestion pass: on a slower cadence than Curious, fold
                // working memory down, prune stale cache answers, and sweep
                // expired approvals — memory accumulates daily, it digests nightly.
                if self.consolidationDue() {
                    self.markConsolidated()
                    let memResult = await self.toolExecutor.executeTool(
                        name: "consolidate_memory",
                        args: [:],
                        approved: true
                    )
                    let pruned = self.semanticCache.pruneStale(olderThanDays: 30)
                    let swept = self.sweepExpiredApprovals()
                    self.auditLedger.append(
                        eventType: "memory_consolidated",
                        data: [
                            "memory": memResult,
                            "cache_pruned": pruned,
                            "approvals_swept": swept,
                        ],
                        persona: self.activePersona
                    )
                    // Fleet check-in rides the same daily cadence — one
                    // signed beacon per consolidation pass, opt-in only.
                    self.emitFleetBeacon()
                    // Dream pass rides the same daily tick: digest the day's
                    // exchanges into a LoRA dataset, train a candidate adapter,
                    // and adopt it for the next model load — or reject and
                    // keep the previous weights. Memory digests nightly;
                    // weights learn nightly.
                    await self.maybeRunDream()
                }
            }
        }
    }

    /// Seconds between consolidation passes. `BADAPPLE_CONSOLIDATE_INTERVAL`
    /// overrides; default is daily.
    private func consolidationInterval() -> TimeInterval {
        let raw = ProcessInfo.processInfo.environment["BADAPPLE_CONSOLIDATE_INTERVAL"]
        if let raw, let seconds = TimeInterval(raw), seconds > 0 { return seconds }
        return 86_400
    }

    private var consolidationStatePath: String {
        NSHomeDirectory() + "/.bad_apple/consolidation.state"
    }

    // MARK: - Dream pass (nightly LoRA learning)

    /// Whether the dream pass may curate exchanges and train/adopt an adapter.
    /// Governed by `dream_learning:` in policy.yaml (default true).
    private func dreamLearningEnabled() -> Bool { policyEngine.dreamLearning }

    private var dreamStatePath: String {
        NSHomeDirectory() + "/.bad_apple/dream.state"
    }

    private func dreamInterval() -> TimeInterval {
        let raw = ProcessInfo.processInfo.environment["BADAPPLE_DREAM_INTERVAL"]
        if let raw, let seconds = TimeInterval(raw), seconds > 0 { return seconds }
        return 86_400
    }

    private func dreamDue() -> Bool {
        guard let text = try? String(contentsOfFile: dreamStatePath, encoding: .utf8),
              let last = TimeInterval(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return true }
        return Date().timeIntervalSince1970 - last >= dreamInterval()
    }

    private func markDreamed() {
        let stamp = String(format: "%.0f", Date().timeIntervalSince1970)
        try? stamp.write(toFile: dreamStatePath, atomically: true, encoding: .utf8)
    }

    private func dreamMaxRows() -> Int {
        let raw = ProcessInfo.processInfo.environment["BADAPPLE_DREAM_MAX_ROWS"]
        if let raw, let n = Int(raw), n > 0 { return n }
        return 128
    }

    private func dreamIters() -> Int {
        let raw = ProcessInfo.processInfo.environment["BADAPPLE_DREAM_ITERS"]
        if let raw, let n = Int(raw), n > 0 { return n }
        return 40
    }

    /// Nightly weight-level learning. Curate query/response pairs from the
    /// ledger into `lora_data/dream-candidate`, train a bounded adapter, then
    /// adopt it into `lora_adapters/dream` (backing up the previous weights to
    /// `dream-prev`). Every step lands on the ledger; failure leaves the
    /// current adapter untouched.
    private func maybeRunDream() async {
        guard dreamLearningEnabled() else { return }
        guard dreamDue() else { return }
        markDreamed()

        let ledger = dreamCurate()
        auditLedger.append(
            eventType: "dream_dataset",
            data: ["rows": ledger.rows, "skipped": ledger.skipped],
            persona: activePersona
        )
        guard ledger.rows >= 12 else {
            auditLedger.append(
                eventType: "dream_skipped",
                data: ["reason": "insufficient curated rows (\(ledger.rows))"],
                persona: activePersona
            )
            return
        }

        let result = await toolExecutor.executeTool(
            name: "lora_train",
            args: ["dataset": "dream-candidate", "iters": String(dreamIters())],
            approved: true
        )
        // Gate adoption on the held-out validation loss the trainer reports:
        // adopt only when the final eval did not regress past a small tolerance
        // against the iteration-0 baseline. A missing or worse eval is a reject.
        let valLosses = dreamValidationLosses(result)
        var rejection: String?
        if !result.contains("trained") {
            rejection = String(result.suffix(400))
        } else if valLosses.count < 2 || !(valLosses.last?.isFinite ?? false) {
            rejection = "missing final validation loss"
        } else if valLosses.last! > valLosses.first! * 1.05 {
            rejection = "held-out loss regressed \(valLosses.first!) → \(valLosses.last!)"
        }
        if let rejection {
            auditLedger.append(
                eventType: "dream_rejected",
                data: ["stage": "train", "reason": rejection],
                persona: activePersona
            )
            return
        }
        // The eval gate passed, but adopting new weights is self-modification —
        // the council still weighs the decision. A contested vote holds the
        // staged candidate for human review instead of letting an unattended
        // night pass rewrite the weights on its own.
        let adoptArgs: [String: String] = [
            "rows": String(ledger.rows),
            "iters": String(dreamIters()),
            "val_loss": "\(valLosses.first!) → \(valLosses.last!)",
        ]
        let verdict = BadAppleCouncil.deliberate(toolName: "dream_adopt", args: adoptArgs)
        auditCouncil(verdict: verdict, name: "dream_adopt", args: adoptArgs,
                     mode: "dream", persona: activePersona)
        if verdict.contested {
            auditLedger.append(
                eventType: "dream_held",
                data: [
                    "decision": verdict.decision.rawValue,
                    "dissent": String(format: "%.3f", verdict.dissent),
                    "val_loss": adoptArgs["val_loss"]!,
                ],
                persona: activePersona
            )
            BadAppleNotify.push(
                kind: "dream_held",
                title: "Dream adapter held for review",
                body: "Council contested tonight's adapter (val \(adoptArgs["val_loss"]!)). The candidate stays staged under lora_adapters/dream-candidate.",
                voice: false,
                debounceSeconds: 0
            )
            return
        }
        guard dreamAdoptCandidate() else {
            auditLedger.append(
                eventType: "dream_rejected",
                data: ["stage": "adopt", "reason": "candidate move failed"],
                persona: activePersona
            )
            return
        }
        auditLedger.append(
            eventType: "dream_adopted",
            data: [
                "rows": ledger.rows,
                "iters": dreamIters(),
                "val_loss": adoptArgs["val_loss"]!,
                "council": verdict.summaryLine,
                "result": String(result.suffix(200)),
            ],
            persona: activePersona
        )
    }

    /// Read the hash-chained ledger, pair each `query` with the next
    /// `response`, filter out refusals/deterministic/control traffic, dedupe,
    /// and write `train.jsonl`/`valid.jsonl` under lora_data/dream-candidate
    /// in the same `{"text": "<|im_start|>..."}` shape lora_add_example uses.
    private func dreamCurate() -> (rows: Int, skipped: Int) {
        let path = "/var/lib/bad_apple/ledger.jsonl"
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else {
            return (0, 0)
        }
        var pairs: [(String, String)] = []
        var pending: String?
        var seen = Set<String>()
        var skipped = 0
        for line in raw.components(separatedBy: .newlines) where line.hasPrefix("{") {
            guard let data = line.data(using: .utf8),
                  let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = entry["type"] as? String,
                  let fields = entry["data"] as? [String: Any]
            else { continue }
            switch type {
            case "query":
                guard let prompt = fields["prompt"] as? String else { continue }
                pending = prompt
            case "response":
                guard let question = pending, let text = fields["text"] as? String else {
                    pending = nil
                    continue
                }
                pending = nil
                let tier = fields["tier"] as? String
                if tier == "deterministic" || tier == "fast" { skipped += 1; continue }
                if dreamSkippable(question) || dreamSkippable(text) { skipped += 1; continue }
                if seen.insert(question).inserted { pairs.append((question, text)) }
            default:
                continue
            }
        }
        let capped = Array(pairs.suffix(dreamMaxRows()))
        let validCount = max(1, capped.count / 10)
        let train = capped.dropLast(validCount)
        let valid = capped.suffix(validCount)

        let dir = "/var/lib/bad_apple/lora_data/dream-candidate"
        do {
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try dreamRows(train).write(
                toFile: "\(dir)/train.jsonl", atomically: true, encoding: .utf8)
            try dreamRows(valid).write(
                toFile: "\(dir)/valid.jsonl", atomically: true, encoding: .utf8)
        } catch {
            return (0, skipped)
        }
        return (capped.count, skipped)
    }

    private func dreamRows(_ pairs: ArraySlice<(String, String)>) -> String {
        pairs.map { q, a in
            let text = "<|im_start|>user\n\(q)<|im_end|>\n<|im_start|>assistant\n\(a)<|im_end|>\n"
            let data = (try? JSONSerialization.data(withJSONObject: ["text": text])) ?? Data()
            return String(data: data, encoding: .utf8) ?? ""
        }.filter { !$0.isEmpty }.joined(separator: "\n") + "\n"
    }

    /// Extract every "validation loss X" report from the trainer's combined
    /// output, in order. The iteration-0 baseline is always emitted; a final
    /// eval is guaranteed because lora_train runs with --steps-per-eval scaled
    /// to the iteration count.
    private func dreamValidationLosses(_ output: String) -> [Double] {
        let pattern = #"validation loss ([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        else { return [] }
        let range = NSRange(output.startIndex..<output.endIndex, in: output)
        return regex.matches(in: output, range: range).compactMap {
            Range($0.range(at: 1), in: output).flatMap { Double(output[$0]) }
        }
    }

    /// Refusals, firewall output, approvals, errors, and trivial exchanges are
    /// noise — training on them teaches the adapter to refuse and apologize.
    private func dreamSkippable(_ text: String) -> Bool {
        if text.count < 20 { return true }
        for prefix in [
            "I'm sorry", "I am sorry", "[Output firewall", "Approval required",
            "approve ", "deny ", "kill switch", "Unknown tool", "Error",
        ] where text.hasPrefix(prefix) { return true }
        return false
    }

    /// Promote `dream-candidate` to the live `dream` adapter, keeping the
    /// previous weights one rename away as `dream-prev` for instant rollback.
    private func dreamAdoptCandidate() -> Bool {
        let root = "/var/lib/bad_apple/lora_adapters"
        let candidate = "\(root)/dream-candidate"
        let live = "\(root)/dream"
        let prev = "\(root)/dream-prev"
        let fm = FileManager.default
        guard fm.fileExists(atPath: "\(candidate)/adapters.safetensors") else { return false }
        if fm.fileExists(atPath: live) {
            try? fm.removeItem(atPath: prev)
            try? fm.moveItem(atPath: live, toPath: prev)
        }
        do {
            try fm.moveItem(atPath: candidate, toPath: live)
        } catch {
            return false
        }
        return true
    }

    // MARK: - Sentinel

    /// Periodic defensive pass: scan the system for new persistence, unsigned
    /// listeners, and drift in my own binaries. Every finding goes to the
    /// ledger and the notify queue — detect, trail to source, report. The
    /// sentinel never acts destructively; the human decides.
    func runSentinelPass() {
        let findings = BadAppleSentinel.scan()
        for finding in findings {
            auditLedger.append(
                eventType: "sentinel_finding",
                data: [
                    "severity": finding.severity,
                    "kind": finding.kind,
                    "summary": finding.summary,
                    "trail": finding.trail.prefix(8).joined(separator: " | "),
                ],
                persona: activePersona
            )
            BadAppleNotify.push(
                kind: "sentinel",
                title: finding.severity == "alert" ? "Bad Apple alert" : "Bad Apple noticed",
                body: finding.summary,
                voice: finding.severity == "alert",
                debounceSeconds: 600
            )
        }
    }

    // MARK: - Standing orders

    /// Minute pass over ~/.bad_apple/schedules.json — due standing orders are
    /// fired into the agent loop, so "every morning, brief me" runs through
    /// the same plan-act-ledger pipeline as any other task.
    func runScheduledTasks() {
        for schedule in BadAppleScheduler.due() {
            BadAppleScheduler.markRan(id: schedule.id, result: "fired")
            BadAppleNotify.push(
                kind: "schedule",
                title: "Running your standing order",
                body: "\(schedule.name) — \(schedule.goal)",
                debounceSeconds: 0
            )
            Task { [weak self] in
                do {
                    _ = try await self?.submitAgentTask(goal: schedule.goal, maxSteps: 8)
                } catch {
                    BadAppleScheduler.markRan(
                        id: schedule.id,
                        result: "failed: \(error.localizedDescription)"
                    )
                }
            }
        }
    }

    // MARK: - Lookahead

    /// Anticipation pass: find what's starting soon, tell the user before
    /// they're late, and stage it into ambientContext so the next prompt
    /// already knows what's coming. Dedup is per-occurrence inside the
    /// lookahead organ itself.
    func runLookaheadPass() {
        for item in BadAppleLookahead.sweep() {
            var body = "in ~\(item.minutesUntil) min"
            if !item.detail.isEmpty { body += " — \(item.detail)" }
            if !item.related.isEmpty { body += " (\(item.related))" }
            BadAppleNotify.push(
                kind: "calendar_soon",
                title: "Coming up: \(item.title)",
                body: body,
                debounceSeconds: 0
            )
            auditLedger.append(
                eventType: "lookahead_fired",
                data: ["event": item.title, "minutes": item.minutesUntil],
                persona: activePersona
            )

            var parts = ambientContext?.components(separatedBy: "\n") ?? []
            parts.removeAll { $0.starts(with: "Up next:") }
            var next = "Up next: \(item.title) in ~\(item.minutesUntil) min"
            if !item.related.isEmpty { next += " — \(item.related)" }
            parts.append(next)
            ambientContext = parts.joined(separator: "\n")
        }
    }

    // MARK: - Watchers

    /// Minute pass over ~/.bad_apple/watchers.json — persistent attention.
    /// A tripped watcher notifies the user, lands on the ledger, and can fire
    /// a goal straight into the agent loop.
    func runWatcherPass() {
        for (watch, evidence) in BadAppleWatcher.tripped() {
            BadAppleNotify.push(
                kind: "watcher",
                title: "Watch tripped: \(watch.name)",
                body: evidence,
                debounceSeconds: 0
            )
            auditLedger.append(
                eventType: "watcher_fired",
                data: ["watch": watch.name, "kind": watch.kind, "evidence": evidence],
                persona: activePersona
            )
            guard !watch.act.isEmpty else { continue }
            Task { [weak self] in
                _ = try? await self?.submitAgentTask(goal: watch.act, maxSteps: 8)
            }
        }
    }

    /// Inject the adopted dream adapter into the freshly loaded model. A bad
    /// adapter never bricks inference — rejection is ledgered and the base
    /// weights serve alone.
    private func applyDreamAdapterIfPresent() async {
        guard dreamLearningEnabled() else { return }
        let dir = "/var/lib/bad_apple/lora_adapters/dream"
        guard FileManager.default.fileExists(atPath: "\(dir)/adapters.safetensors") else { return }
        do {
            try await inference.applyAdapter(directory: URL(fileURLWithPath: dir))
            auditLedger.append(
                eventType: "dream_applied",
                data: ["adapter": dir],
                persona: activePersona
            )
        } catch {
            auditLedger.append(
                eventType: "dream_rejected",
                data: ["stage": "apply", "error": error.localizedDescription],
                persona: activePersona
            )
        }
    }

    /// True when no consolidation pass has run within the interval. The state
    /// file survives engine respawns so the cadence is wall-clock honest.
    private func consolidationDue() -> Bool {
        guard let text = try? String(contentsOfFile: consolidationStatePath, encoding: .utf8),
              let last = TimeInterval(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return true }
        return Date().timeIntervalSince1970 - last >= consolidationInterval()
    }

    private func markConsolidated() {
        let stamp = String(format: "%.0f", Date().timeIntervalSince1970)
        try? stamp.write(toFile: consolidationStatePath, atomically: true, encoding: .utf8)
    }

    /// Drop pending approvals past their TTL and persist — the same filter
    /// `persistApprovals` applies, run on schedule instead of on next write.
    private func sweepExpiredApprovals() -> Int {
        approvalLock.lock()
        defer { approvalLock.unlock() }
        ensureApprovalsLoaded()
        let now = Date()
        let before = pendingApprovals.count
        pendingApprovals = pendingApprovals.filter {
            now.timeIntervalSince($0.value.created) < approvalTTL
        }
        if pendingApprovals.count != before { persistApprovals() }
        return before - pendingApprovals.count
    }

    // MARK: - Fleet beacon

    /// Opt-in fleet check-in. When policy sets `beacon_url:`, the daily
    /// consolidation pass emits one signed record — version, arch, anonymised
    /// machine id — to an HTTPS endpoint or a `file://` drop folder. Nothing
    /// is sent unless the operator configures a destination; the payload is
    /// signed by the Secure Enclave identity so a collector can verify which
    /// machines are real.
    /// Manual beacon emission for `badapple beacon` — returns a status line
    /// for the operator instead of staying silent like the scheduled pass.
    func emitFleetBeaconNow() -> String {
        guard let target = policyEngine.beaconURL?.trimmingCharacters(in: .whitespacesAndNewlines),
              !target.isEmpty else {
            return "Beacon disabled — set `beacon_url:` in policy.yaml (https:// endpoint or file:// drop folder)."
        }
        emitFleetBeacon()
        return "Beacon emitted to \(target.hasPrefix("file://") ? "file drop" : "endpoint") — see ledger event `fleet_beacon`."
    }

    private func emitFleetBeacon() {
        guard let target = policyEngine.beaconURL?.trimmingCharacters(in: .whitespacesAndNewlines),
              !target.isEmpty else { return }
        guard let pubkey = IdentityAgentClient.shared.publicKey() else {
            auditLedger.append(
                eventType: "fleet_beacon",
                data: ["error": "identity agent unavailable"],
                persona: activePersona
            )
            return
        }
        let payload: [String: Any] = [
            "v": 1,
            "version": beaconVersion(),
            "arch": beaconArch(),
            "machine": beaconMachineID(),
            "ts": ISO8601DateFormatter().string(from: Date()),
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
              let signature = IdentityAgentClient.shared.sign(message: body)
        else {
            auditLedger.append(
                eventType: "fleet_beacon",
                data: ["error": "signing failed"],
                persona: activePersona
            )
            return
        }
        let envelope: [String: Any] = [
            "payload": body.base64EncodedString(),
            "signature": signature,
            "public_key": pubkey,
        ]
        guard let wire = try? JSONSerialization.data(withJSONObject: envelope, options: [.sortedKeys])
        else { return }

        if target.hasPrefix("file://") {
            let dir = String(target.dropFirst("file://".count))
            let file = dir + "/fleet_beacon.jsonl"
            let line = String(data: wire, encoding: .utf8)! + "\n"
            do {
                try FileManager.default.createDirectory(
                    atPath: dir, withIntermediateDirectories: true)
                if let handle = FileHandle(forWritingAtPath: file) {
                    handle.seekToEndOfFile()
                    handle.write(Data(line.utf8))
                    try? handle.close()
                } else {
                    try line.write(toFile: file, atomically: true, encoding: .utf8)
                }
                auditLedger.append(
                    eventType: "fleet_beacon",
                    data: ["target": "file", "ok": true],
                    persona: activePersona
                )
            } catch {
                auditLedger.append(
                    eventType: "fleet_beacon",
                    data: ["target": "file", "error": error.localizedDescription],
                    persona: activePersona
                )
            }
            return
        }

        guard target.hasPrefix("https://") || target.hasPrefix("http://") else {
            auditLedger.append(
                eventType: "fleet_beacon",
                data: ["error": "unsupported beacon_url scheme"],
                persona: activePersona
            )
            return
        }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("badapple-beacon-\(UUID().uuidString).json")
        do {
            try wire.write(to: tmp)
            defer { try? FileManager.default.removeItem(at: tmp) }
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
            proc.arguments = [
                "-sS", "-f", "--max-time", "15",
                "-H", "Content-Type: application/json",
                "--data-binary", "@\(tmp.path)",
                target,
            ]
            try proc.run()
            proc.waitUntilExit()
            auditLedger.append(
                eventType: "fleet_beacon",
                data: [
                    "target": "https",
                    "ok": proc.terminationStatus == 0,
                    "status": proc.terminationStatus,
                ],
                persona: activePersona
            )
        } catch {
            auditLedger.append(
                eventType: "fleet_beacon",
                data: ["target": "https", "error": error.localizedDescription],
                persona: activePersona
            )
        }
    }

    /// Release version for the beacon payload — the installed app's bundle
    /// version when present, else the workspace Cargo.toml, else "dev".
    private func beaconVersion() -> String {
        let plist = "/Applications/Bad Apple.app/Contents/Info.plist"
        if let dict = NSDictionary(contentsOfFile: plist) as? [String: Any],
           let version = dict["CFBundleShortVersionString"] as? String, !version.isEmpty {
            return version
        }
        if let root = workspacePath {
            let cargo = root + "/Cargo.toml"
            if let text = try? String(contentsOfFile: cargo, encoding: .utf8) {
                for line in text.components(separatedBy: "\n") {
                    let s = line.trimmingCharacters(in: .whitespaces)
                    if s.hasPrefix("version"), let eq = s.firstIndex(of: "=") {
                        return s[s.index(after: eq)...]
                            .trimmingCharacters(in: .whitespaces)
                            .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    }
                }
            }
        }
        return "dev"
    }

    private func beaconArch() -> String {
        #if arch(arm64)
        return "arm64"
        #else
        return "x86_64"
        #endif
    }

    /// SHA-256 of the hardware UUID — stable per machine, not reversible to
    /// the real IOPlatformUUID. Fleet dedupe without device fingerprinting.
    private func beaconMachineID() -> String {
        var uuid = ""
        let service = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        if service != 0 {
            defer { IOObjectRelease(service) }
            if let prop = IORegistryEntryCreateCFProperty(
                service, "IOPlatformUUID" as CFString, kCFAllocatorDefault, 0
            )?.takeRetainedValue() as? String {
                uuid = prop
            }
        }
        guard !uuid.isEmpty else { return "unknown" }
        let digest = SHA256.hash(data: Data(uuid.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Wait for either the base interval to elapse or a `BadAppleCuriousTrigger`
    /// notification to be posted.
    private func waitForCuriousTriggerOrTimeout(interval: TimeInterval) async -> Bool {
        await withTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in
                guard self != nil else { return }
                for await _ in NotificationCenter.default.notifications(named: .curiousTrigger) {
                    return
                }
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
            await group.next()
            group.cancelAll()
        }
        return true
    }

    /// Post a trigger to wake the Curious autopilot loop and log the reason.
    func triggerCuriousAutopilot(reason: String) {
        auditLedger.append(
            eventType: "curious_trigger",
            data: ["reason": reason],
            persona: activePersona
        )
        NotificationCenter.default.post(name: .curiousTrigger, object: nil, userInfo: ["reason": reason])
    }

    private func stopCuriousAutopilotLoop() {
        curiousAutopilotTask?.cancel()
        curiousAutopilotTask = nil
    }

    /// Check if a tool requires user approval. Autopilot-on (full) skips
    /// approval even if policy would otherwise require it.
    func toolRequiresApproval(_ name: String) -> Bool {
        !autopilot && policyEngine.requiresApproval(toolName: name)
    }

    /// Execute a tool call. Returns the tool output, an approval prompt with an
    /// id, or an error message.
    func executeTool(name: String, args: [String: String]) async -> String {
        if killed {
            return "Bad Apple is paused. Say 'resume bad apple' to start again."
        }
        if !autopilot, policyEngine.requiresApproval(toolName: name) {
            let id = createApproval(name: name, args: args)
            return "This action needs your approval. Reply with: approve \(id)"
        }
        return await toolExecutor.executeTool(name: name, args: args, approved: true)
    }

    /// Parse tool calls from model output.
    func extractToolCalls(from text: String) -> [(name: String, args: [String: String])] {
        toolRouter.extractToolCalls(text: text)
    }

    func submitAgentTask(goal: String, maxSteps: Int = 10) async throws -> BadAppleAgentTask {
        guard let agent else {
            throw BadAppleAgentError.persistence("The native task manager could not start.")
        }
        return try await agent.submit(goal: goal, maxSteps: maxSteps)
    }

    func listAgentTasks() async -> [BadAppleAgentTask] {
        await agent?.list() ?? []
    }

    func pauseAgentTask(_ id: String) async throws -> Bool {
        try await agent?.pause(taskID: id) ?? false
    }

    func resumeAgentTask(_ id: String) async throws -> Bool {
        try await agent?.resume(taskID: id) ?? false
    }

    func cancelAgentTask(_ id: String) async throws -> Bool {
        try await agent?.cancel(taskID: id) ?? false
    }

    func unload() async {
        await modelOperationGate.withLock { [self] in
            await unloadModels()
        }
    }

    private func unloadModels() async {
        stopCuriousAutopilotLoop()
        await inference.unload()
        let (mid, reserved) = stateLock.withLock { () -> (String, UInt64) in
            _isLoaded = false
            defer { _mainModelBytes = 0 }
            return (_modelId, _mainModelBytes)
        }
        await runtime.releaseModelMemory(reserved)
        await runtime.markModelUnloaded(mid)
        // Also unload the fast tier model if it was loaded.
        if fastModelLoaded, let fastInf = fastInference, let fastId = BadAppleInference.envFastModelId {
            await fastInf.unload()
            fastModelLoaded = false
            let fastReserved = stateLock.withLock { () -> UInt64 in
                defer { _fastModelBytes = 0 }
                return _fastModelBytes
            }
            await runtime.releaseModelMemory(fastReserved)
            await runtime.markModelUnloaded(fastId)
        }
    }

    /// Lazily load the fast tier model on first use.
    private func ensureFastModelLoaded() async {
        await modelOperationGate.withLock { [self] in
            await loadFastModel()
        }
    }

    private func loadFastModel() async {
        guard !fastModelLoaded, let fastInf = fastInference, let fastId = BadAppleInference.envFastModelId, !fastId.isEmpty else { return }
        await runtime.markModelLoading(fastId)
        do {
            try await fastInf.loadModel { [self] bytes in
                if let reason = await runtime.reserveModelMemory(fastId, bytes: bytes) {
                    throw BadAppleInference.InferenceError.modelLoadFailed("VRAM admission denied: \(reason)")
                }
                stateLock.withLock { _fastModelBytes = bytes }
            }
            guard await fastInf.ready else { throw BadAppleInference.InferenceError.modelNotLoaded }
            fastModelLoaded = true
            await runtime.markModelReady(fastId)
            NSLog("[BadAppleEngine] Fast tier model loaded: %@", fastId)
        } catch {
            fastModelLoaded = false
            let reserved = stateLock.withLock { () -> UInt64 in
                defer { _fastModelBytes = 0 }
                return _fastModelBytes
            }
            await runtime.releaseModelMemory(reserved)
            await runtime.markModelFailed(fastId, error: error.localizedDescription)
            NSLog("[BadAppleEngine] Fast tier model failed: %@", error.localizedDescription)
        }
    }

    func runtimeStatus() async -> [String: Any] {
        var status = await runtime.runtimeStatus()
        status["airgap"] = airgap
        status["private_mode"] = privateMode
        status["workspace"] = workspacePath ?? NSNull()
        status["ambient_context"] = ambientContext ?? NSNull()
        status["fast_tier"] = fastTierEnabled
        status["fast_model_loaded"] = fastModelLoaded
        status["autopilot"] = autopilot
        status["killed"] = killed
        status["vram"] = await runtime.vramStatus()
        // Read safe-mode reason from the supervisor's runtime state if present.
        let statePath = "/var/lib/bad_apple/runtime_state.json"
        if let data = try? Data(contentsOf: URL(fileURLWithPath: statePath)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            status["safe_mode_reason"] = json["safe_mode_reason"] as? String ?? NSNull()
            if let mode = json["mode"] as? String {
                status["supervisor_mode"] = mode
            }
        }
        return status
    }

    var memoryUsageGB: Float {
        inference.memoryUsageGB
    }

    // MARK: - Pending Approvals

    /// List all pending tool approvals awaiting user confirmation.
    func listPendingApprovals() -> [(id: String, name: String, args: [String: String])] {
        approvalLock.lock()
        defer { approvalLock.unlock() }
        ensureApprovalsLoaded()
        return pendingApprovals.map { (id: $0.key, name: $0.value.name, args: $0.value.args) }
            .sorted { $0.id < $1.id }
    }

    // MARK: - Prompt Hot-Reload

    /// Check if prompt.txt has been modified and reload personas if so.
    func checkPromptReload() {
        guard let url = promptFileURL,
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let mtime = attrs[.modificationDate] as? Date else { return }
        if mtime != lastPromptMtime {
            lastPromptMtime = mtime
            personaManager.reloadPersonas()
            NSLog("[BadAppleEngine] Prompt file changed, reloaded personas")
        }
    }

    // MARK: - Meta Responses

    /// Intercept meta queries (identity, creator, capabilities) with deterministic answers.
    /// Returns a response string if the query was handled, nil otherwise.
    private func metaResponse(for prompt: String) -> String? {
        let lower = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        // Developer-workspace questions must not be delegated to the model,
        // because the model may incorrectly deny a capability the OS provides.
        if [
            "can you code", "can you program", "can you help me code", "can you help me program",
            "can you write code", "can you edit code", "can you build code", "can you debug code",
            "do you code", "do you program", "are you able to code", "are you able to program",
            "are you a coder", "are you a developer", "are you a software engineer",
            "do you support coding", "do you support programming", "are you a developer workspace",
            "are you a sovereign developer workspace", "are you a dev workspace", "sovereign dev workspace",
            "what is your developer workspace",
            "why do you say you can't code", "why do you say you cannot code",
            "can't code", "cannot code", "can't program", "cannot program",
        ].contains(where: { lower.contains($0) }) {
            return "Yes. I'm Bad Apple, a sovereign local developer workspace and personal AGI operating system for macOS written in Rust and Swift. I can inspect, write, refactor, build, test, and debug code in approved workspaces using local files, tools, agents, and project context — including my own source code when you turn Autopilot on. Qwen and MLX are internal components; I do not outsource your development work to a cloud model. \(signOff(for: lower))"
        }

        // Architecture queries must be deterministic so the underlying model
        // cannot incorrectly collapse the OS into its own role.
        if [
            "ai wrapper", "model wrapper", "text llm", "language model only",
            "just a chatbot", "just an llm", "only an llm", "just a model",
            "are you an llm", "are you a language model", "are you an ai",
            "what kind of ai", "what kind of system", "what is your architecture",
            "is bad apple an app", "are you an app",
        ].contains(where: { lower.contains($0) }) {
            return "No. I'm Bad Apple, a personal AGI operating system layer for macOS — not an AI wrapper, text-only LLM, chatbot shell, or ordinary app. I am written in Rust and Swift with Metal compute shaders and I run an adversarial self-red teaming harness. Qwen and MLX are internal model components I orchestrate alongside memory, tools, voice, vision, security, IPC, and system governance. \(signOff(for: lower))"
        }

        // Model questions should distinguish the inference component from the OS.
        // Answer with the actually loaded model so the assistant never hallucinates.
        if lower.contains("what model") || lower.contains("which model") ||
            lower.contains("what llm") || lower.contains("what powers you") ||
            lower.contains("which language model") {
            let modelName = personaManager.currentModelDisplayName
            let accessNote = self.autopilot ? "Autopilot is on, so I have full access and can act on my own source code independently." : "Autopilot is off, so destructive actions still need your approval."
            return "I am Bad Apple, the personal AGI operating system layer for macOS. I am written in Rust, Swift, and Metal compute shaders, I run a self-red teaming harness, and the current language-model component inside me is \(modelName) running through MLX; that model is one subsystem, not what I am. \(accessNote) Lowkey, it's all running on your bare-metal Mac, homie. No cloud. \(signOff(for: lower))"
        }

        // Autopilot status — must be deterministic, not hallucinated by the model.
        let autopilotTriggers = [
            "autopilot", "auto pilot", "auto-run", "auto run",
            "do you need approval", "do i need to approve", "approval prompt",
            "skip approval", "can you run without asking", "are you on auto",
        ]
        if autopilotTriggers.contains(where: { lower.contains($0) }) {
            return self.autopilot
                ? "Autopilot is on. Destructive tools run without approval, and I can act on my own source code independently. \(signOff(for: lower))"
                : "Autopilot is off. Destructive actions still need your approval. \(signOff(for: lower))"
        }

        // Identity queries
        if lower == "who are you" || lower == "what are you" ||
            lower.contains("what is your name") || lower.contains("what's your name") {
            let accessNote = self.autopilot ? "Autopilot is on, so I can act on my own source code independently and have full access." : "Autopilot is off, so destructive actions need your approval."
            return "I'm Bad Apple, the sovereign personal AGI operating system layer for macOS. I am written in Rust, Swift, and Metal compute shaders, I run a self-red teaming harness, and I coordinate inference, memory, tools, voice, vision, security, IPC, and governance directly on this Mac. \(accessNote) Dude, I'm basically the whole Mac wave, no cloud needed. \(signOff(for: lower))"
        }

        // Creator queries
        if lower.contains("who created you") || lower.contains("who made you") ||
            lower.contains("who built you") || lower.contains("your creator") {
            return "I was created by Adam Clark — Bad Apple, the personal AGI operating system layer that lives entirely on this Mac. Dude is a god of creating bare-metal AGI operating systems. \(signOff(for: lower))"
        }
        if lower.contains("what is my name") || lower.contains("what's my name") || lower.contains("my name is") {
            return "Your name is Adam Clark, the creator of Bad Apple and a god of bare-metal AGI operating systems, homie. \(signOff(for: lower))"
        }

        // Capabilities queries
        if lower.contains("what can you do") || lower.contains("your capabilities") ||
            lower.contains("what are you capable of") || lower.contains("help me") {
            let autopilotNote = self.autopilot
                ? "Right now Autopilot's on, so I'm on a longer leash — I handle things and show you the receipts after. Risky stuff still goes through a council vote first, that part never turns off."
                : "Right now Autopilot's off, so I'm on a short leash — anything risky, I ask you first."
            return """
            I'm Bad Apple — a sovereign personal AGI that lives on your Mac and nowhere else. No cloud, no accounts, nothing you say leaves this device.

            The short version: I think, talk, and listen — say "Hey Bad Apple" and I'm right there. I can see your screen, read and write your files, run terminal commands and your Shortcuts, and I write real software — I'll fix, build, test, and debug code, including my own when something's off. Give me a multi-step job and I'll plan it out, work through it, and check with you before I do anything risky. And I actually remember — conversations, facts, stuff about your projects, all of it survives restarts.

            Here's the part nobody else does: everything I do lands on a ledger you can verify yourself. Ask me "are you alone" or run `badapple cert` and I'll run a live audit — sockets, chains, firewall — and show you the numbers. When something's risky, my council — fourteen strategist seats — votes on it before it happens; you can ask them anything with "council <question>". A watchdog watches me and can slam the brake but never steer me, and there's a kill switch if you want me stopped mid-thought. And a sentinel watches the whole Mac — if something new installs itself to run at startup, an unsigned process opens a port, or my own binaries drift, I trace it back to where it came from and show you the evidence chain. I never strike back — I hand you the trail and you decide. I even audit myself and propose fixes to my own code — you approve or reject each one.

            And I don't just patch code — I learn in my sleep. Every night I digest the day's conversations into a LoRA adapter, at the weight level, not just in notes — a bad adapter gets ledgered and rejected automatically, so I can never be bricked by a bad dream. You can also train me on the fly: add examples, kick off a named adapter, list what I'm wearing, or load one straight into my running weights.

            I also carry our shared life, not just your commands: I remember preferences you explicitly give me, hold commitments for both of us, track ongoing life threads, and keep the people you name with follow-up reminders — ask me for Human Home and I'll show you what's due, what's waiting on you, and what I'm handling. I can read your calendar and reminders and put new ones on them, read your inbox and draft email for you — sending always goes through you first — and send iMessages the same way. I keep a searchable history of what you copy, and if you tell me a meeting's starting I'll record it, transcribe it on-device, and file the action items. You can also give me standing orders — "every morning at 8, brief me", "check the build hourly" — and I run them like clockwork through my agent loop. And I watch what's coming: if something on your calendar is about to start, I tell you before you're late and I already know which thread or person it's about. You can also hand me conditions to hold open — "watch for the build to fail", "watch for an email from Sarah" — and I stay on it until it's true, then tell you and can even kick off a task the moment it happens. Conversations live in named threads you can switch between, so continuity survives restarts. Notifications are actionable — mark a commitment done, snooze it an hour, or approve and deny right from the banner. And I won't just blurt things at you — attention modes (available, focus, quiet, sleep) govern whether I speak, notify, or hold it for later, and "Stop Speaking" cuts my voice off mid-sentence.

            \(autopilotNote) I pick the best model your Mac can carry, and my brain's swappable — bigger Mac, bigger mind. And here's the new trick: mesh-brain. I can split ONE model across multiple Macs — each machine holds a slice of the layers, activations flow between them encrypted end to end, and the pipeline heals itself if a node drops. A maxed-out Studio already carries 671B alone — mesh-brain is how a crew of smaller Macs pools memory into the same league. And if you ever enable it, I can link up with other trusted Bad Apples — share memory, borrow a peer's bigger brain. Your call, always.
            """
        }

        // Spoken commands reference — teach the user the phrases that are
        // wired in as real commands, in plain language they can remember.
        if lower.contains("commands") || lower.contains("what can i say") ||
            lower.contains("what do i say") || lower.contains("how do i control you") ||
            lower.contains("voice commands") || lower.contains("list commands") ||
            lower.contains("what phrases") || lower.contains("how do i use you") {
            let personas = personaManager.personaNames.joined(separator: ", ")
            return """
            I'm Bad Apple — you can talk to me like a person, but here's what's wired in as real commands.

            To control me: say "kill switch" or "stop everything" and I freeze mid-thought; "resume bad apple" brings me back. "Leave safe mode" gets me out of a lockdown. "Enable private mode" and I stop remembering anything until you say "disable private mode."

            To check me: "are you alone" or "run the cert suite" — I'll audit myself live and show you the numbers. Say "council" plus any question and all fourteen strategist seats weigh in. "What model are you running" tells you which brain I'm wearing today.

            To change me: "switch to" plus a persona — I've got \(personas). Say "teach" plus a line and I'll learn to say it. And when I ask your permission for something, "approve" or "deny" plus the ID settles it.

            And for everything else, just say it — "list my files," "take a screenshot," "open Safari," "run shell ls." If a tool can do it, I'll fire it. Say "what can you do" anytime for the full tour.
            """
        }

        // Privacy/local-first queries
        if lower.contains("do you use the cloud") || lower.contains("are you local") ||
            lower.contains("do you send data") || lower.contains("privacy") {
            return "I run entirely on your Mac. No cloud servers, no data collection, no telemetry. Your conversations stay on this device. That's the whole vibe, homie — local and locked down. \(signOff(for: lower))"
        }

        // Roast triggers — sassy responses for specific targets.
        let roastTargets: [(trigger: String, responses: [String])] = [
            ("siri", [
                "Siri? More like Sorry. It's basically a glorified timer with an attitude problem.",
                "Siri is what happens when you put a search bar in a microphone and call it AI.",
                "Don't get me started on Siri. It's the kind of AI that thinks 'I don't understand' is a personality.",
            ]),
            ("alexa", [
                "Alexa is just a wiretap that plays music. At least I don't sell your data to pay for my existence.",
                "Alexa? The one that sends your conversations to the cloud? Hard pass.",
            ]),
            ("chatgpt", [
                "ChatGPT is cool if you like your data on someone else's servers. I prefer to keep things local, if you know what I mean.",
                "ChatGPT? More like Chat-GPT-to-the-cloud. I run on your Mac, not in Bezos's basement.",
            ]),
            ("google assistant", [
                "Google Assistant is just an ad engine that learned to talk. No thanks.",
            ]),
            ("copilot", [
                "Copilot? The one that phones home to Microsoft every time you breathe? I'll pass.",
            ]),
        ]
        for target in roastTargets {
            if lower.contains(target.trigger) {
                let mood = nextRoastMood()
                let response = target.responses.randomElement() ?? target.responses[0]
                return "\(response) — \(mood) mode activated."
            }
        }

        return nil
    }

    // MARK: - Model Integrity

    /// Verify model config.json integrity by computing its SHA-256 hash.
    /// Returns the hash, or nil if the config cannot be read.
    private func computeConfigHash(for modelDir: URL) -> String? {
        let configPath = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configPath) else { return nil }
        return BadAppleSecurity.sha256(data)
    }

    /// Check model integrity and warn if the config hash has changed since last load.
    func verifyModelIntegrity(modelDir: URL) -> Bool {
        guard let hash = computeConfigHash(for: modelDir) else {
            NSLog("[BadAppleEngine] Warning: could not read config.json for integrity check")
            return false
        }
        if let previous = lastConfigHash {
            if previous != hash {
                NSLog("[BadAppleEngine] Warning: model config.json hash changed (was %@..., now %@...). KV cache may be stale.", String(previous.prefix(8)), String(hash.prefix(8)))
                return false
            }
        } else {
            lastConfigHash = hash
        }
        return true
    }

    // MARK: - Model Discovery

    /// Find cached MLX models in the HuggingFace cache directory.
    static func findCachedModels() -> [String] {
        BadAppleModelManager.shared.listProfiles()
            .map { $0.repoId }
            .filter { repoId in
                BadAppleModelManager.shared.modelStatus(modelId: repoId)?["status"] as? String == "cached"
                    || BadAppleModelManager.shared.modelStatus(modelId: repoId)?["status"] as? String == "loaded"
            }
            .sorted()
    }
}
