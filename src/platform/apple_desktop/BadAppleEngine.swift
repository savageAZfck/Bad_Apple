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

    private let inference: BadAppleInference
    private let personaManager = BadApplePersonaManager()
    private let auditLedger = BadAppleAuditLedger()
    private let outputFirewall = BadAppleOutputFirewall()
    private let toolRouter = BadAppleToolRouter()
    private let policyEngine = BadApplePolicyEngine()
    private lazy var toolExecutor = BadAppleToolExecutor(policyEngine: policyEngine)
    private let embeddingEngine = BadAppleEmbeddingEngine()
    private let visionEngine = BadAppleVisionEngine()
    private lazy var semanticCache = BadAppleSemanticCache(
        embeddingProvider: NativeEmbeddingProvider(engine: embeddingEngine)
    )
    private let rag = BadAppleRAG()
    private let runtime = BadAppleNativeRuntime()
    private let conversation = BadAppleConversation()
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
            "/Users/savag3/bad_apple/prompt.txt",
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

    private(set) var isLoaded = false
    private(set) var isLoading = false
    private(set) var modelId: String = ""
    private(set) var lastTokensPerSecond: Float = 0
    private(set) var lastTokenCount: Int = 0
    private(set) var lastCacheHit: Bool = false
    var workspacePath: String?

    // MARK: - Air-gap / Private Mode

    /// When true, no network access is attempted and HuggingFace downloads are disabled.
    private var airgapEnabled = false
    /// When true, conversation and audit persistence is paused.
    private var privateModeEnabled = false

    var airgap: Bool {
        get { airgapEnabled }
        set { airgapEnabled = newValue }
    }

    var privateMode: Bool {
        get { privateModeEnabled }
        set {
            privateModeEnabled = newValue
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

    /// The latest ambient context snapshot (active app, window title).
    private(set) var ambientContext: String?

    /// Update the ambient context from the frontmost application.
    func updateAmbientContext() {
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
            let parts = value.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false)
            let app = parts.first.map(String.init) ?? "Unknown"
            let window = parts.count > 1 ? String(parts[1]) : ""
            ambientContext = "Active app: \(app)\nWindow: \(window)"
        }
    }

    // MARK: - Init

    private init() {
        inference = BadAppleInference.createDefault()
        modelId = BadAppleInference.defaultConfig.modelId
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

    /// Load the model. Call this at startup or when the model changes.
    func loadModel() async {
        guard !isLoaded && !isLoading else { return }
        isLoading = true
        await runtime.markModelLoading(modelId)

        do {
            try await inference.loadModel()
            isLoaded = true
            await runtime.markModelReady(modelId)
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
            isLoaded = false
            await runtime.markModelFailed(modelId, error: error.localizedDescription)
        }
        isLoading = false
    }

    /// Load from a local directory (e.g., HuggingFace cache).
    func loadModel(from directory: URL) async {
        guard !isLoaded && !isLoading else { return }
        isLoading = true
        await runtime.markModelLoading(modelId)

        do {
            try await inference.loadModel(from: directory)
            isLoaded = true
            await runtime.markModelReady(modelId)
        } catch {
            print("[BadAppleEngine] Failed to load model from \(directory): \(error.localizedDescription)")
            isLoaded = false
            await runtime.markModelFailed(modelId, error: error.localizedDescription)
        }
        isLoading = false
    }

    // MARK: - Generation

    private func inferenceHistory() -> [BadAppleInference.ChatMessage] {
        conversation.loadConversation(sessionId: conversationSessionId).map {
            BadAppleInference.ChatMessage(role: $0.role, content: $0.content)
        }
    }

    private func saveTurn(prompt: String, response: String) {
        // Private mode: skip persistence entirely.
        guard !privateModeEnabled else { return }
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
                maxTokens: maxTokens,
                temperature: 0.6
            )
            let calls = toolRouter.extractToolCalls(text: lastResult.text)
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
                case .denied:
                    output = "That action is blocked by your safety settings."
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
                    output = await toolExecutor.executeTool(name: call.name, args: call.args)
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
        lastCacheHit = false

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

        Task {
            let history = inferenceHistory()
            if let cached = await semanticCache.lookup(prompt: prompt, persona: persona) {
                lastCacheHit = true
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

            let effectiveMaxTokens = isSimpleQuery(prompt) ? min(maxTokens, 150) : maxTokens
            if let tools = toolRouter.toolsForPrompt(text: prompt) {
                sysPrompt += "\n\n\(tools)\nIf a tool is needed, output only <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>. Never invent a tool result."
                do {
                    let result = try await toolAwareGeneration(
                        prompt: prompt,
                        systemPrompt: sysPrompt,
                        history: history,
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
                        self.lastTokensPerSecond = result.tokensPerSecond
                        self.lastTokenCount = result.tokenCount
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

            inference.generateStreamingTokens(
                prompt: prompt,
                systemPrompt: sysPrompt,
                history: history,
                maxTokens: effectiveMaxTokens,
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
                        await self.semanticCache.store(
                            prompt: prompt,
                            response: filtered,
                            persona: persona
                        )
                    }
                    DispatchQueue.main.async {
                        self.lastTokensPerSecond = result.tokensPerSecond
                        self.lastTokenCount = result.tokenCount
                        self.auditLedger.append(
                            eventType: "response",
                            data: ["text": filtered, "tps": result.tokensPerSecond],
                            persona: persona
                        )
                        onComplete(filtered)
                    }
                },
                onError: { error in
                    DispatchQueue.main.async {
                        self.auditLedger.append(
                            eventType: "error",
                            data: ["error": error.localizedDescription],
                            persona: persona
                        )
                        onError(error.localizedDescription)
                    }
                }
            )
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
        let history = inferenceHistory()
        lastCacheHit = false

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
            lastCacheHit = true
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

        let effectiveMaxTokens = isSimpleQuery(prompt) ? min(maxTokens, 150) : maxTokens

        let result: BadAppleInference.GenerationResult
        if let tools = toolRouter.toolsForPrompt(text: prompt) {
            sysPrompt += "\n\n\(tools)\nIf a tool is needed, output only <tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>. Never invent a tool result."
            result = try await toolAwareGeneration(
                prompt: prompt,
                systemPrompt: sysPrompt,
                history: history,
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
        lastTokensPerSecond = result.tokensPerSecond
        lastTokenCount = result.tokenCount

        // Postprocess and filter.
        let polished = postprocessOutput(result.text)
        let filtered = outputFirewall.check(polished)

        await semanticCache.store(
            prompt: prompt,
            response: filtered,
            persona: persona
        )

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
            return "I'm Bad Apple, your private AI running directly on this Mac."
        }
        if lower.contains("who created you") || lower.contains("who made you") {
            return "You created me."
        }
        if ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"].contains(lower) {
            return "Hey! What can I help you with?"
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
        isLoaded = false
        await runtime.markModelUnloaded(modelId)
    }

    func runtimeStatus() async -> [String: Any] {
        var status = await runtime.runtimeStatus()
        status["airgap"] = airgapEnabled
        status["private_mode"] = privateModeEnabled
        status["workspace"] = workspacePath ?? NSNull()
        status["ambient_context"] = ambientContext ?? NSNull()
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

        // Identity queries
        if lower == "who are you" || lower == "what are you" ||
            lower.contains("what is your name") || lower.contains("what's your name") {
            return "I'm Bad Apple, your private AI running directly on this Mac. No cloud, no data mining — just bare metal intelligence."
        }

        // Creator queries
        if lower.contains("who created you") || lower.contains("who made you") ||
            lower.contains("who built you") || lower.contains("your creator") {
            return "I was created by my user. I'm a local-first AI assistant that lives entirely on this Mac."
        }

        // Capabilities queries
        if lower.contains("what can you do") || lower.contains("your capabilities") ||
            lower.contains("what are you capable of") || lower.contains("help me") {
            return """
            I can help you with:
            • Answering questions and having conversations
            • Reading and writing files on your Mac
            • Running shell commands and AppleScripts (with your approval)
            • Listing and running macOS Shortcuts
            • Searching your local notes and documents
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
            return "I run entirely on your Mac. No cloud servers, no data collection, no telemetry. Your conversations stay on this device."
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
        let cacheDir = NSHomeDirectory() + "/.cache/huggingface/hub"
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: cacheDir) else {
            return []
        }
        return entries
            .filter { $0.hasPrefix("models--") && $0.contains("Qwen") }
            .map { $0.replacingOccurrences(of: "models--", with: "").replacingOccurrences(of: "--", with: "/") }
            .sorted()
    }
}
