// BadAppleEngine — Swift-native AI engine for Bad Apple.
// Wraps BadAppleMLX inference with persona system and prompt management.
// This replaces the Python daemon for text generation queries.

import Foundation
import BadAppleMLX

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
            print("[BadAppleEngine] Fast tier disabled: BADAPPLE_FAST_MODEL is empty or unset.")
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
    private var fastModelLoaded = false
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
        return exec
    }()
    private let embeddingEngine = BadAppleEmbeddingEngine()
    private let visionEngine = BadAppleVisionEngine()
    private lazy var semanticCache = BadAppleSemanticCache(
        embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
    )
    private let rag = BadAppleRAG()
    private let runtime = BadAppleNativeRuntime()
    private let conversation = BadAppleConversation()
    let modelManager = BadAppleModelManager.shared
    private let conversationSessionId = "default"
    private let approvalLock = NSLock()
    private var pendingApprovals: [String: (name: String, args: [String: String])] = [:]

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
    private var _modelId: String = ""
    private var _lastTokensPerSecond: Float = 0
    private var _lastTokenCount: Int = 0
    private var _lastDraftAcceptPct: Float = 0
    private var _lastCacheHit: Bool = false
    private var _workspacePath: String?
    private var _airgapEnabled = false
    private var _privateModeEnabled = false
    private var _fastTierEnabled = false

    var isLoaded: Bool {
        return stateLock.withLock { _isLoaded }
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
        }
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

        // Reuse any existing ocular description without losing it.
        if let existing = ambientContext {
            for line in existing.components(separatedBy: .newlines) {
                if line.starts(with: "Screen:") {
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

    /// Switch to a named persona.
    func switchPersona(_ name: String) -> Bool {
        let result = personaManager.switchPersona(name)
        if result { personaManager.reloadPersonas() }
        return result
    }

    /// Get the active persona name.
    var activePersona: String {
        personaManager.activePersonaName
    }

    /// Get the system prompt for the current persona.
    func systemPrompt(voiceMode: Bool = false) -> String {
        personaManager.getSystemPrompt(voiceMode: voiceMode)
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

    // MARK: - Model Loading

    /// Estimate the memory footprint of a model from its ID.
    /// 4-bit quantized models use roughly 0.5 bytes per parameter.
    private func estimateModelBytes(_ modelId: String) -> UInt64 {
        let lower = modelId.lowercased()
        // Extract parameter count from common naming patterns.
        if lower.contains("70b") { return 40 * 1_073_741_824 }   // ~40 GB
        if lower.contains("32b") { return 18 * 1_073_741_824 }   // ~18 GB
        if lower.contains("9b") || lower.contains("8b") || lower.contains("7b") { return 6 * 1_073_741_824 }  // ~6 GB
        if lower.contains("4b") || lower.contains("3b") { return 3 * 1_073_741_824 }   // ~3 GB
        if lower.contains("1.5b") { return 1 * 1_073_741_824 }    // ~1 GB
        if lower.contains("0.5b") || lower.contains("500m") { return 350 * 1_048_576 } // ~350 MB
        // Default: assume 9B-class.
        return 6 * 1_073_741_824
    }

    /// Load the model. Call this at startup or when the model changes.
    func loadModel() async {
        let (alreadyLoaded, mid) = stateLock.withLock { () -> (Bool, String) in
            let loaded = _isLoaded || _isLoading
            if !loaded { _isLoading = true }
            return (loaded, _modelId)
        }
        if alreadyLoaded { return }
        await runtime.markModelLoading(mid)

        // VRAM admission check: refuse to load if the model won't fit.
        let estimatedBytes = estimateModelBytes(mid)
        if let reason = await runtime.canFitModel(estimatedBytes: estimatedBytes) {
            print("[BadAppleEngine] VRAM admission denied for \(mid): \(reason)")
            stateLock.withLock { _isLoaded = false; _isLoading = false }
            await runtime.markModelFailed(mid, error: "VRAM admission denied: \(reason)")
            return
        }

        do {
            try await inference.loadModel()
            await runtime.trackModelMemory(mid, bytes: estimatedBytes)
            stateLock.withLock { _isLoaded = true }
            await runtime.markModelReady(mid)
            Task {
                await runtime.markModelLoading(embeddingEngine.configuration.modelId)
                do {
                    try await embeddingEngine.loadModel()
                    await runtime.markModelReady(embeddingEngine.configuration.modelId)
                    print("[BadAppleEngine] Native embedding model loaded")
                } catch {
                    print("[BadAppleEngine] Native embedding model failed: \(error.localizedDescription)")
                    await runtime.markModelFailed(
                        embeddingEngine.configuration.modelId,
                        error: error.localizedDescription
                    )
                }
            }
        } catch {
            print("[BadAppleEngine] Failed to load model: \(error.localizedDescription)")
            stateLock.withLock { _isLoaded = false }
            await runtime.markModelFailed(mid, error: error.localizedDescription)
        }
        stateLock.withLock { _isLoading = false }
    }

    /// Load from a local directory (e.g., HuggingFace cache).
    func loadModel(from directory: URL) async {
        let (alreadyLoaded, mid) = stateLock.withLock { () -> (Bool, String) in
            let loaded = _isLoaded || _isLoading
            if !loaded { _isLoading = true }
            return (loaded, _modelId)
        }
        if alreadyLoaded { return }
        await runtime.markModelLoading(mid)

        do {
            try await inference.loadModel(from: directory)
            stateLock.withLock { _isLoaded = true }
            await runtime.markModelReady(mid)
        } catch {
            print("[BadAppleEngine] Failed to load model from \(directory): \(error.localizedDescription)")
            stateLock.withLock { _isLoaded = false }
            await runtime.markModelFailed(mid, error: error.localizedDescription)
        }
        stateLock.withLock { _isLoading = false }
    }

    // MARK: - Generation

    private func inferenceHistory() -> [BadAppleInference.ChatMessage] {
        conversation.loadConversation(sessionId: conversationSessionId).map {
            BadAppleInference.ChatMessage(role: $0.role, content: $0.content)
        }
    }

    private func saveTurn(prompt: String, response: String) {
        // Private mode: skip persistence entirely.
        guard !privateMode else { return }
        var messages = conversation.loadConversation(sessionId: conversationSessionId)
        messages.append(BadAppleMessage(role: "user", content: prompt))
        messages.append(BadAppleMessage(role: "assistant", content: response))
        // Prune oldest turns to prevent unbounded context growth.
        let maxMessages = Self.maxHistoryTurns * 2
        if messages.count > maxMessages {
            messages = Array(messages.suffix(maxMessages))
        }
        conversation.saveConversation(sessionId: conversationSessionId, messages: messages)
    }

    func resetConversation() {
        conversation.clearConversation(sessionId: conversationSessionId)
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
        let tools = toolRouter.toolsForPrompt(text: context.plannedStep.instruction) ?? "No tools are available."
        let result = try await inference.generate(
            prompt: "Goal: \(context.goal)\nCurrent step: \(context.plannedStep.instruction)\n\(tools)\nReturn only JSON with thought, tool, arguments, and finish. Use either tool or finish, not both.",
            systemPrompt: systemPrompt(voiceMode: false),
            maxTokens: 256,
            temperature: 0
        )
        if let start = result.text.firstIndex(of: "{"),
           let end = result.text.lastIndex(of: "}"),
           start <= end,
           let data = String(result.text[start...end]).data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let thought = object["thought"] as? String ?? ""
            let tool = object["tool"] as? String
            let finish = object["finish"] as? String
            let rawArgs = object["arguments"] as? [String: Any] ?? [:]
            let arguments = rawArgs.mapValues { value in
                if let string = value as? String { return string }
                return String(describing: value)
            }
            return BadAppleAgentAction(
                thought: thought,
                tool: tool?.isEmpty == true ? nil : tool,
                arguments: arguments,
                finish: finish?.isEmpty == true ? nil : finish
            )
        }
        return .finish(postprocessOutput(result.text))
    }

    private func createApproval(name: String, args: [String: String]) -> String {
        let id = String(UUID().uuidString.lowercased().prefix(8))
        approvalLock.lock()
        pendingApprovals[id] = (name, args)
        approvalLock.unlock()
        return id
    }

    private func takeApproval(from prompt: String) -> (id: String, name: String, args: [String: String])? {
        let parts = prompt.lowercased().split(whereSeparator: { $0.isWhitespace })
        guard parts.count == 2, parts[0] == "approve" else { return nil }
        let id = String(parts[1])
        approvalLock.lock()
        defer { approvalLock.unlock() }
        guard let call = pendingApprovals.removeValue(forKey: id) else { return nil }
        return (id, call.name, call.args)
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

        for _ in 0..<5 {
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
            guard !calls.isEmpty else { return lastResult }

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
                    let id = createApproval(name: call.name, args: call.args)
                    auditLedger.append(
                        eventType: "approval_requested",
                        data: ["id": id, "name": call.name, "arguments": call.args],
                        persona: persona
                    )
                    return BadAppleInference.GenerationResult(
                        text: "This action needs your approval. Reply with: approve \(id)",
                        tier: "approval"
                    )
                case .approved:
                    output = await toolExecutor.executeTool(name: call.name, args: call.args, approved: true)
                }
                auditLedger.append(
                    eventType: "tool_result",
                    data: ["name": call.name, "result": output],
                    persona: persona
                )
                outputs.append("\(call.name): \(output)")
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

        if let approval = takeApproval(from: prompt) {
            Task {
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
                saveTurn(prompt: prompt, response: output)
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
            saveTurn(prompt: prompt, response: filtered)
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
            saveTurn(prompt: prompt, response: filtered)
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
                        self.saveTurn(prompt: prompt, response: filtered)
                        Task {
                            await self.semanticCache.store(prompt: prompt, response: filtered, persona: persona)
                        }
                        self.stateLock.withLock {
                            self._lastTokensPerSecond = result.tokensPerSecond
                            self._lastTokenCount = result.tokenCount
                            self._lastDraftAcceptPct = result.draftAcceptPct
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
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        let persona = activePersona
        stateLock.withLock { _lastCacheHit = false }

        Task {
            await refreshAmbientContext()

            let history = inferenceHistory()
            if let cached = await semanticCache.lookup(prompt: prompt, persona: persona) {
                stateLock.withLock { _lastCacheHit = true }
                auditLedger.append(
                    eventType: "cache_hit",
                    data: ["prompt": prompt],
                    persona: persona
                )
                saveTurn(prompt: prompt, response: cached)
                DispatchQueue.main.async {
                    onToken(cached)
                    onComplete(cached)
                }
                return
            }

            var sysPrompt = systemPrompt(voiceMode: voiceMode)
            if let ambient = ambientContext {
                sysPrompt += "\n\nAmbient:\n\(ambient)"
            }
            let ragContext = await rag.buildSemanticRetrievalContext(
                prompt: prompt,
                workspace: workspacePath,
                embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
            )
            if let ragContext {
                sysPrompt += "\n\nContext:\n\(ragContext)"
            }

            let effectiveMaxTokens = maxTokens
            if let tools = toolRouter.toolSchemasForPrompt(text: prompt) {
                sysPrompt += "\n\nIf a tool is needed, output only <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>. Never invent a tool result."
                do {
                    let result = try await toolAwareGeneration(
                        prompt: prompt,
                        systemPrompt: sysPrompt,
                        history: history,
                        tools: tools,
                        maxTokens: effectiveMaxTokens,
                        persona: persona
                    )
                    let filtered = outputFirewall.check(postprocessOutput(result.text))
                    saveTurn(prompt: prompt, response: filtered)
                    await semanticCache.store(prompt: prompt, response: filtered, persona: persona)
                    auditLedger.append(
                        eventType: "response",
                        data: ["text": filtered, "tps": result.tokensPerSecond],
                        persona: persona
                    )
                    DispatchQueue.main.async {
                        self.stateLock.withLock {
                            self._lastTokensPerSecond = result.tokensPerSecond
                            self._lastTokenCount = result.tokenCount
                            self._lastDraftAcceptPct = result.draftAcceptPct
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
                self.saveTurn(prompt: prompt, response: filtered)
                // Do not cache responses that were likely truncated by the token limit.
                let looksComplete = result.tokenCount == 0 || result.tokenCount < maxTokens - 5
                if looksComplete {
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

    /// Generate a complete response (non-streaming). Checks semantic cache first.
    func generate(
        prompt: String,
        voiceMode: Bool = false,
        maxTokens: Int = 300
    ) async throws -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
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
            saveTurn(prompt: prompt, response: output)
            return output
        }

        let persona = activePersona
        await refreshAmbientContext()
        let history = inferenceHistory()
        stateLock.withLock { _lastCacheHit = false }

        // Check prompt hot-reload before generation.
        checkPromptReload()

        // Meta responses (identity, creator, capabilities) are deterministic.
        if let meta = metaResponse(for: prompt) {
            let filtered = outputFirewall.check(meta)
            saveTurn(prompt: prompt, response: filtered)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "meta"],
                persona: persona
            )
            return filtered
        }

        if let fast = deterministicResponse(for: prompt) {
            let filtered = outputFirewall.check(fast)
            saveTurn(prompt: prompt, response: filtered)
            auditLedger.append(
                eventType: "response",
                data: ["text": filtered, "tier": "deterministic"],
                persona: persona
            )
            return filtered
        }

        // Check semantic cache for a matching response.
        if let cached = await semanticCache.lookup(prompt: prompt, persona: persona) {
            stateLock.withLock { _lastCacheHit = true }
            auditLedger.append(
                eventType: "cache_hit",
                data: ["prompt": prompt],
                persona: persona
            )
            saveTurn(prompt: prompt, response: cached)
            return cached
        }

        // Build system prompt with ambient context and semantic RAG.
        var sysPrompt = systemPrompt(voiceMode: voiceMode)
        if let ambient = ambientContext {
            sysPrompt += "\n\nAmbient:\n\(ambient)"
        }
        let ragContext = await rag.buildSemanticRetrievalContext(
            prompt: prompt,
            workspace: workspacePath,
            embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
        )
        if let ragContext {
            sysPrompt += "\n\nContext:\n\(ragContext)"
        }

        let effectiveMaxTokens = maxTokens

        let result: BadAppleInference.GenerationResult
        if let tools = toolRouter.toolSchemasForPrompt(text: prompt) {
            sysPrompt += "\n\nIf a tool is needed, output only <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>. Never invent a tool result."
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
                temperature: 0.6
            )
        }
        stateLock.withLock {
            _lastTokensPerSecond = result.tokensPerSecond
            _lastTokenCount = result.tokenCount
            _lastDraftAcceptPct = result.draftAcceptPct
        }

        // Postprocess and filter.
        let polished = postprocessOutput(result.text)
        let filtered = outputFirewall.check(polished)

        // Do not cache responses that were likely truncated by the token limit.
        let looksComplete = result.tokenCount == 0 || result.tokenCount < maxTokens - 5
        if looksComplete {
            await semanticCache.store(
                prompt: prompt,
                response: filtered,
                persona: persona
            )
        }

        // Audit log.
        auditLedger.append(
            eventType: "response",
            data: ["text": filtered, "tps": result.tokensPerSecond],
            persona: persona
        )
        saveTurn(prompt: prompt, response: filtered)

        return filtered
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
            return "I'm Bad Apple, the sovereign local AI operating system layer for macOS. I run inference, memory, tools, voice, vision, security, IPC, and governance directly on this Mac. Dude, I'm basically the whole Mac wave, no cloud needed. \(signOff(for: lower))"
        }
        if lower.contains("who created you") || lower.contains("who made you") {
            return "Adam Clark created me — Bad Apple, the local AI operating system layer running on this Mac. Dude is a god of creating bare-metal AI operating systems. \(signOff(for: lower))"
        }
        if lower.contains("what is my name") || lower.contains("my name is") {
            return "Your name is Adam Clark, the creator of Bad Apple and a god of bare-metal AI operating systems, homie. \(signOff(for: lower))"
        }
        if ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"].contains(lower) {
            return "Hey homie! What's the wave? I'm vibing on bare-metal local power, so hit me with whatever you need. \(signOff(for: lower))"
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
    var autopilot: Bool {
        get { policyEngine.autopilot }
        set { policyEngine.autopilot = newValue }
    }

    /// Check if a tool requires user approval.
    func toolRequiresApproval(_ name: String) -> Bool {
        policyEngine.requiresApproval(toolName: name)
    }

    /// Execute a tool call. Returns the tool output or an error message.
    func executeTool(name: String, args: [String: String]) async -> String {
        await toolExecutor.executeTool(name: name, args: args)
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
        await inference.unload()
        let mid = stateLock.withLock { () -> String in
            _isLoaded = false
            return _modelId
        }
        await runtime.releaseModelMemory(estimateModelBytes(mid))
        await runtime.markModelUnloaded(mid)
        // Also unload the fast tier model if it was loaded.
        if fastModelLoaded, let fastInf = fastInference, let fastId = BadAppleInference.envFastModelId {
            await fastInf.unload()
            fastModelLoaded = false
            await runtime.releaseModelMemory(estimateModelBytes(fastId))
            await runtime.markModelUnloaded(fastId)
        }
    }

    /// Lazily load the fast tier model on first use.
    private func ensureFastModelLoaded() async {
        guard !fastModelLoaded, let fastInf = fastInference, let fastId = BadAppleInference.envFastModelId, !fastId.isEmpty else { return }
        let estimatedBytes = estimateModelBytes(fastId)
        if let reason = await runtime.canFitModel(estimatedBytes: estimatedBytes) {
            print("[BadAppleEngine] VRAM admission denied for fast tier \(fastId): \(reason)")
            await runtime.markModelFailed(fastId, error: "VRAM admission denied: \(reason)")
            return
        }
        await runtime.markModelLoading(fastId)
        do {
            try await fastInf.loadModel()
            fastModelLoaded = true
            await runtime.trackModelMemory(fastId, bytes: estimatedBytes)
            await runtime.markModelReady(fastId)
            print("[BadAppleEngine] Fast tier model loaded: \(fastId)")
        } catch {
            fastModelLoaded = false
            await runtime.markModelFailed(fastId, error: error.localizedDescription)
            print("[BadAppleEngine] Fast tier model failed: \(error.localizedDescription)")
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
        status["vram"] = await runtime.vramStatus()
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
            print("[BadAppleEngine] Prompt file changed, reloaded personas")
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
            return "Yes. I'm Bad Apple, a sovereign local developer workspace and AI operating system for macOS. I can inspect, write, refactor, build, test, and debug code in approved workspaces using local files, tools, agents, and project context. Qwen and MLX are internal components; I do not outsource your development work to a cloud model. \(signOff(for: lower))"
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
            return "No. I'm Bad Apple, a local AI operating system layer for macOS — not an AI wrapper, text-only LLM, chatbot shell, or ordinary app. Qwen and MLX are internal model components I orchestrate alongside memory, tools, voice, vision, security, IPC, and system governance. \(signOff(for: lower))"
        }

        // Model questions should distinguish the inference component from the OS.
        // Answer with the actually loaded model so the assistant never hallucinates.
        if lower.contains("what model") || lower.contains("which model") ||
            lower.contains("what llm") || lower.contains("what powers you") ||
            lower.contains("which language model") {
            let modelName = personaManager.currentModelDisplayName
            return "I am Bad Apple, the local AI operating system layer for macOS. The current language-model component inside me is \(modelName) running through MLX; that model is one subsystem, not what I am. Lowkey, it's all running on your bare-metal Mac, homie. No cloud. \(signOff(for: lower))"
        }

        // Identity queries
        if lower == "who are you" || lower == "what are you" ||
            lower.contains("what is your name") || lower.contains("what's your name") {
            return "I'm Bad Apple, the sovereign local AI operating system layer for macOS. I run inference, memory, tools, voice, vision, security, IPC, and governance directly on this Mac. Dude, I'm basically the whole Mac wave, no cloud needed. \(signOff(for: lower))"
        }

        // Creator queries
        if lower.contains("who created you") || lower.contains("who made you") ||
            lower.contains("who built you") || lower.contains("your creator") {
            return "I was created by Adam Clark — Bad Apple, the local AI operating system layer that lives entirely on this Mac. Dude is a god of creating bare-metal AI operating systems. \(signOff(for: lower))"
        }
        if lower.contains("what is my name") || lower.contains("what's my name") || lower.contains("my name is") {
            return "Your name is Adam Clark, the creator of Bad Apple and a god of bare-metal AI operating systems, homie. \(signOff(for: lower))"
        }

        // Capabilities queries
        if lower.contains("what can you do") || lower.contains("your capabilities") ||
            lower.contains("what are you capable of") || lower.contains("help me") {
            return """
            I am Bad Apple, a sovereign local AI operating system and developer workspace for macOS. I coordinate these capabilities:
            • Inspecting, writing, refactoring, building, testing, and debugging source code in approved workspaces
            • Answering questions and having conversations
            • Reading and writing files on your Mac
            • Running shell commands and AppleScripts (with your approval)
            • Listing and running macOS Shortcuts
            • Searching your local notes, documents, and source repositories
            • Taking screenshots and describing images
            • Managing a working memory scratchpad
            • Multi-step agent tasks with planning
            • Voice interaction with "Hey Bad Apple"

            Everything runs locally on your Mac — no cloud, no data leaves your device.
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
            print("[BadAppleEngine] Warning: could not read config.json for integrity check")
            return false
        }
        if let previous = lastConfigHash {
            if previous != hash {
                print("[BadAppleEngine] Warning: model config.json hash changed (was \(previous.prefix(8))..., now \(hash.prefix(8))...). KV cache may be stale.")
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
