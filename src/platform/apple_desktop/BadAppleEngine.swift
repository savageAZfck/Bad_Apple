// BadAppleEngine — Swift-native AI engine for Bad Apple.
// Wraps BadAppleMLX inference with persona system and prompt management.
// This replaces the Python daemon for text generation queries.

import Foundation
import BadAppleMLX

/// The main AI engine. Loads the model, manages personas, and generates
/// responses directly in-process — no subprocess, no daemon, no Python.
final class BadAppleEngine {

    // MARK: - Singleton

    static let shared = BadAppleEngine()

    // MARK: - State

    private let inference: BadAppleInference
    private let personaManager = BadApplePersonaManager()
    private let auditLedger = BadAppleAuditLedger()
    private let outputFirewall = BadAppleOutputFirewall()
    private let toolRouter = BadAppleToolRouter()
    private let policyEngine = BadApplePolicyEngine()
    private let toolExecutor = BadAppleToolExecutor()

    private(set) var isLoaded = false
    private(set) var isLoading = false
    private(set) var modelId: String = ""
    private(set) var lastTokensPerSecond: Float = 0
    private(set) var lastTokenCount: Int = 0

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

        do {
            try await inference.loadModel()
            isLoaded = true
        } catch {
            print("[BadAppleEngine] Failed to load model: \(error.localizedDescription)")
            isLoaded = false
        }
        isLoading = false
    }

    /// Load from a local directory (e.g., HuggingFace cache).
    func loadModel(from directory: URL) async {
        guard !isLoaded && !isLoading else { return }
        isLoading = true

        do {
            try await inference.loadModel(from: directory)
            isLoaded = true
        } catch {
            print("[BadAppleEngine] Failed to load model from \(directory): \(error.localizedDescription)")
            isLoaded = false
        }
        isLoading = false
    }

    // MARK: - Generation

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

        let sysPrompt = systemPrompt(voiceMode: voiceMode)
        let persona = activePersona

        // Log the query to the audit ledger.
        auditLedger.append(
            eventType: "query",
            data: ["prompt": prompt, "voice": voiceMode],
            persona: persona
        )

        inference.generateStreamingTokens(
            prompt: prompt,
            systemPrompt: sysPrompt,
            maxTokens: maxTokens,
            temperature: 0.6,
            onToken: { token in
                DispatchQueue.main.async { onToken(token) }
            },
            onComplete: { result in
                DispatchQueue.main.async {
                    self.lastTokensPerSecond = result.tokensPerSecond
                    self.lastTokenCount = result.tokenCount
                    // Apply output firewall to the final response.
                    let filtered = self.outputFirewall.check(result.text)
                    // Log the response to the audit ledger.
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

    /// Generate a complete response (non-streaming).
    func generate(
        prompt: String,
        voiceMode: Bool = false,
        maxTokens: Int = 300
    ) async throws -> String {
        guard isLoaded else {
            return "The AI model is not loaded yet. Please wait a moment and try again."
        }

        let sysPrompt = systemPrompt(voiceMode: voiceMode)
        let result = try await inference.generate(
            prompt: prompt,
            systemPrompt: sysPrompt,
            maxTokens: maxTokens,
            temperature: 0.6
        )
        lastTokensPerSecond = result.tokensPerSecond
        lastTokenCount = result.tokenCount
        return result.text
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

    func unload() async {
        await inference.unload()
        isLoaded = false
    }

    var memoryUsageGB: Float {
        inference.memoryUsageGB
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
