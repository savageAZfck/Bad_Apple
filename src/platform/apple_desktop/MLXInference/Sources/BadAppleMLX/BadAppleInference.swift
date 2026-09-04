// BadAppleMLX — Swift-native MLX inference runtime for Bad Apple.
// Replaces the Python badapple_mlx_server.py with a single Swift module
// that loads the model, generates text, and streams tokens via callbacks.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import HuggingFace
import Tokenizers

/// A ``Downloader`` that bridges ``HuggingFace.HubClient`` to the
/// ``MLXLMCommon.Downloader`` protocol used by ``LLMModelFactory``.
public struct HuggingFaceDownloader: Sendable, Downloader {
    private let client: HuggingFace.HubClient

    public init(client: HuggingFace.HubClient = .default) {
        self.client = client
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        return try await client.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }
}

public enum HuggingFaceDownloaderError: LocalizedError {
    case invalidRepositoryID(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let id):
            return "Invalid Hugging Face repository ID: '\(id)'. Expected format 'namespace/name'."
        }
    }
}

/// A ``MLXLMCommon.TokenizerLoader`` that loads Hugging Face tokenizers
/// from a local directory using ``Tokenizers.AutoTokenizer``.
public struct TokenizersLoader: Sendable, MLXLMCommon.TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return HuggingFaceTokenizer(upstream: upstream)
    }
}

private struct HuggingFaceTokenizer: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

/// The main inference engine. Loads a model once and handles generation requests.
public final class BadAppleInference: @unchecked Sendable {

    // MARK: - Types

    public struct ModelConfig: Sendable {
        public let modelId: String
        public let revision: String
        public let maxTokens: Int
        public let temperature: Float
        public let topP: Float

        public init(
            modelId: String,
            revision: String = "main",
            maxTokens: Int = 300,
            temperature: Float = 0.6,
            topP: Float = 0.9
        ) {
            self.modelId = modelId
            self.revision = revision
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
        }
    }

    public struct ChatMessage: Sendable {
        public let role: String
        public let content: String

        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }
    }

    public struct GenerationResult: Sendable {
        public let text: String
        public let tokensPerSecond: Float
        public let tokenCount: Int
        public let tier: String
        public let toolCalls: [ToolCall]

        public init(
            text: String,
            tokensPerSecond: Float = 0,
            tokenCount: Int = 0,
            tier: String = "main",
            toolCalls: [ToolCall] = []
        ) {
            self.text = text
            self.tokensPerSecond = tokensPerSecond
            self.tokenCount = tokenCount
            self.tier = tier
            self.toolCalls = toolCalls
        }
    }

    /// A tool definition for chat-template based tool calling.
    public struct ToolDefinition: Sendable {
        public let type: String
        public let function: [String: any Sendable]

        public init(type: String = "function", function: [String: any Sendable]) {
            self.type = type
            self.function = function
        }

        public init(type: String = "function", function: [String: Any]) {
            self.type = type
            self.function = function.mapValues { BadAppleInference.sendableValue(from: $0) }
        }

        public var toolSpec: [String: any Sendable] {
            ["type": type, "function": function]
        }
    }

    /// A parsed tool call.
    public struct ToolCall: Sendable {
        public let name: String
        public let arguments: [String: String]

        public init(name: String, arguments: [String: String]) {
            self.name = name
            self.arguments = arguments
        }
    }

    public enum InferenceError: Error, LocalizedError {
        case modelNotLoaded
        case modelLoadFailed(String)
        case generationFailed(String)

        public var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "The AI model is not loaded yet. Please wait a moment and try again."
            case .modelLoadFailed(let detail):
                return "Could not load the AI model: \(detail)"
            case .generationFailed(let detail):
                return "The AI could not generate a response: \(detail)"
            }
        }
    }

    // MARK: - State (atomic via actor)

    private actor ModelState {
        var container: ModelContainer?
        var isLoaded = false
        var isLoading = false

        func setContainer(_ container: ModelContainer?) {
            self.container = container
            self.isLoaded = container != nil
            self.isLoading = false
        }

        func setLoading() -> Bool {
            guard !isLoaded && !isLoading else { return false }
            isLoading = true
            return true
        }

        func getContainer() -> ModelContainer? { container }
        func loaded() -> Bool { isLoaded }
    }

    private let state = ModelState()
    private let config: ModelConfig

    // MARK: - Initialization

    public init(config: ModelConfig) {
        self.config = config
    }

    // MARK: - Model Loading

    public func loadModel() async throws {
        guard await state.setLoading() else { return }

        do {
            let modelConfiguration = ModelConfiguration(
                id: config.modelId,
                revision: config.revision
            )

            let container = try await LLMModelFactory.shared.loadContainer(
                from: HuggingFaceDownloader(client: HubClient.default),
                using: TokenizersLoader(),
                configuration: modelConfiguration
            )

            await state.setContainer(container)
        } catch {
            await state.setContainer(nil)
            throw InferenceError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Load from a local directory of already-downloaded model weights.
    public func loadModel(from localDirectory: URL) async throws {
        guard await state.setLoading() else { return }

        do {
            let container = try await LLMModelFactory.shared.loadContainer(
                from: localDirectory,
                using: TokenizersLoader()
            )

            await state.setContainer(container)
        } catch {
            await state.setContainer(nil)
            throw InferenceError.modelLoadFailed(error.localizedDescription)
        }
    }

    public var ready: Bool {
        get async { await state.loaded() }
    }

    // MARK: - Generation (callback-based streaming)

    private func buildChatMessages(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = []
    ) -> [Chat.Message] {
        var messages: [Chat.Message] = []
        if let systemPrompt = systemPrompt {
            messages.append(.system(systemPrompt))
        }
        for message in history.suffix(12) {
            switch message.role.lowercased() {
            case "assistant":
                messages.append(.assistant(message.content))
            case "system":
                messages.append(.system(message.content))
            case "tool":
                messages.append(.tool(message.content))
            default:
                messages.append(.user(message.content))
            }
        }
        messages.append(.user(prompt))
        return messages
    }

    private func buildRawMessages(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = []
    ) -> [[String: any Sendable]] {
        var messages: [[String: any Sendable]] = []
        if let systemPrompt = systemPrompt {
            messages.append(["role": "system", "content": systemPrompt])
        }
        for message in history.suffix(12) {
            messages.append(["role": message.role, "content": message.content])
        }
        messages.append(["role": "user", "content": prompt])
        return messages
    }

    private static func sendableValue(from value: Any) -> any Sendable {
        if let string = value as? String { return string }
        if let int = value as? Int { return int }
        if let double = value as? Double { return double }
        if let bool = value as? Bool { return bool }
        if let array = value as? [Any] { return array.map { Self.sendableValue(from: $0) } }
        if let dict = value as? [String: Any] { return dict.mapValues { Self.sendableValue(from: $0) } }
        return String(describing: value)
    }

    private func prepareInput(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: any Sendable]]? = nil,
        container: ModelContainer
    ) async throws -> LMInput {
        let chatMessages = buildChatMessages(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history
        )

        if let tools = tools, !tools.isEmpty {
            let tokenizer = await container.tokenizer
            let messages = buildRawMessages(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: history
            )
            do {
                let tokenIds = try tokenizer.applyChatTemplate(
                    messages: messages,
                    tools: tools,
                    additionalContext: ["enable_thinking": false]
                )
                return LMInput(tokens: MLXArray(tokenIds))
            } catch MLXLMCommon.TokenizerError.missingChatTemplate {
                // Fall through to the UserInput path below.
            }
        }

        let userInput = UserInput(
            chat: chatMessages,
            additionalContext: ["enable_thinking": false]
        )
        return try await container.prepare(input: userInput)
    }

    private func jsonValueToString(_ value: MLXLMCommon.JSONValue) -> String {
        switch value {
        case .string(let s):
            return s
        case .bool(let b):
            return b ? "true" : "false"
        case .int(let i):
            return String(i)
        case .double(let d):
            return String(d)
        case .null:
            return ""
        case .array(let a):
            return a.map { jsonValueToString($0) }.joined(separator: ", ")
        case .object(let o):
            return o.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
        }
    }

    private func toolCallText(from call: MLXLMCommon.ToolCall) -> String? {
        let args = call.function.arguments.mapValues { jsonValueToString($0) }
        guard let data = try? JSONSerialization.data(
            withJSONObject: ["name": call.function.name, "arguments": args],
            options: []
        ),
        let json = String(data: data, encoding: .utf8) else {
            return nil
        }
        return "<tool_call>\(json)</tool_call>"
    }

    private func convertToolCall(_ call: MLXLMCommon.ToolCall) -> BadAppleInference.ToolCall? {
        let args = call.function.arguments.mapValues { jsonValueToString($0) }
        return BadAppleInference.ToolCall(name: call.function.name, arguments: args)
    }

    public func parseToolCalls(_ text: String) -> [BadAppleInference.ToolCall] {
        var calls: [BadAppleInference.ToolCall] = []
        guard let regex = try? NSRegularExpression(
            pattern: #"<tool_call>(.*?)</tool_call>"#,
            options: [.dotMatchesLineSeparators]
        ) else {
            return []
        }
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        for match in matches {
            guard match.numberOfRanges >= 2,
                  let jsonRange = Range(match.range(at: 1), in: text) else { continue }
            let jsonStr = String(text[jsonRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard let data = jsonStr.data(using: .utf8) else { continue }

            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let effective = (object["function"] as? [String: Any]) ?? object
                guard let name = (effective["name"] as? String) ?? (object["name"] as? String) else { continue }

                var rawArgs = effective["arguments"] as? [String: Any]
                if rawArgs == nil, let stringArgs = effective["arguments"] as? String,
                   let argData = stringArgs.data(using: .utf8) {
                    rawArgs = try? JSONSerialization.jsonObject(with: argData) as? [String: Any]
                }
                if rawArgs == nil, let paramData = effective["parameters"] as? [String: Any] {
                    rawArgs = paramData
                }

                let args: [String: String] = (rawArgs ?? [:]).mapValues {
                    if let string = $0 as? String { return string }
                    return String(describing: $0)
                }
                calls.append(BadAppleInference.ToolCall(name: name, arguments: args))
            }
        }
        return calls
    }

    public func generateStreamingTokens(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: any Sendable]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) {
        Task {
            do {
                guard let container = await state.getContainer() else {
                    onError(InferenceError.modelNotLoaded)
                    return
                }

                let lmInput = try await prepareInput(
                    prompt: prompt,
                    systemPrompt: systemPrompt,
                    history: history,
                    tools: tools,
                    container: container
                )
                let params = GenerateParameters(
                    maxTokens: maxTokens ?? config.maxTokens,
                    maxKVSize: BadAppleInference.envKVSize,
                    temperature: temperature ?? config.temperature,
                    topP: config.topP,
                    prefillStepSize: BadAppleInference.envPrefillStepSize
                )

                let stream = try await container.generate(input: lmInput, parameters: params)

                var fullText = ""
                var tps: Float = 0
                var tokenCount = 0
                var detectedToolCalls: [BadAppleInference.ToolCall] = []
                var stop = false

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        fullText += text
                        onToken(text)
                    case .info(let info):
                        tps = Float(info.tokensPerSecond)
                        tokenCount = info.generationTokenCount
                    case .toolCall(let call):
                        if let text = toolCallText(from: call),
                           let toolCall = convertToolCall(call) {
                            fullText += text
                            detectedToolCalls.append(toolCall)
                        }
                        stop = true
                    }

                    if tools != nil, fullText.contains("</tool_call>") {
                        stop = true
                    }

                    if stop {
                        break
                    }
                }

                let finalToolCalls = detectedToolCalls.isEmpty ? parseToolCalls(fullText) : detectedToolCalls
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main",
                    toolCalls: finalToolCalls
                ))
            } catch {
                onError(InferenceError.generationFailed(error.localizedDescription))
            }
        }
    }

    public func generateStreamingTokens(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [ToolDefinition]? = nil,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { $0.toolSpec }
        generateStreamingTokens(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature,
            onToken: onToken,
            onComplete: onComplete,
            onError: onError
        )
    }

    public func generateStreamingTokens(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: Any]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { dict in
            dict.mapValues { Self.sendableValue(from: $0) }
        }
        generateStreamingTokens(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature,
            onToken: onToken,
            onComplete: onComplete,
            onError: onError
        )
    }

    /// Generate a complete response (non-streaming).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: any Sendable]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> GenerationResult {
        try await withCheckedThrowingContinuation { continuation in
            generateStreamingTokens(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: history,
                tools: tools,
                maxTokens: maxTokens,
                temperature: temperature,
                onToken: { _ in },
                onComplete: { result in
                    continuation.resume(returning: result)
                },
                onError: { error in
                    continuation.resume(throwing: error)
                }
            )
        }
    }

    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [ToolDefinition]? = nil,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> GenerationResult {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { $0.toolSpec }
        return try await generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature
        )
    }

    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: Any]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> GenerationResult {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { dict in
            dict.mapValues { Self.sendableValue(from: $0) }
        }
        return try await generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature
        )
    }

    /// Generate with tool-call parsing, returning both text and any tool calls.
    public func generateWithTools(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: any Sendable]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> (text: String, toolCalls: [BadAppleInference.ToolCall]) {
        let result = try await generate(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: tools,
            maxTokens: maxTokens,
            temperature: temperature
        )
        return (result.text, result.toolCalls)
    }

    public func generateWithTools(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [ToolDefinition]? = nil,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> (text: String, toolCalls: [BadAppleInference.ToolCall]) {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { $0.toolSpec }
        return try await generateWithTools(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature
        )
    }

    public func generateWithTools(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: Any]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> (text: String, toolCalls: [BadAppleInference.ToolCall]) {
        let toolSchemas: [[String: any Sendable]]? = tools?.map { dict in
            dict.mapValues { Self.sendableValue(from: $0) }
        }
        return try await generateWithTools(
            prompt: prompt,
            systemPrompt: systemPrompt,
            history: history,
            tools: toolSchemas,
            maxTokens: maxTokens,
            temperature: temperature
        )
    }

    // MARK: - Speculative Decoding

    public func generateWithSpeculativeDecoding(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        draftModelId: String,
        numDraftTokens: Int = 2,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) {
        Task {
            do {
                guard let container = await state.getContainer() else {
                    onError(InferenceError.modelNotLoaded)
                    return
                }

                var messages: [Chat.Message] = []
                if let systemPrompt = systemPrompt {
                    messages.append(.system(systemPrompt))
                }
                for message in history.suffix(12) {
                    switch message.role.lowercased() {
                    case "assistant":
                        messages.append(.assistant(message.content))
                    case "system":
                        messages.append(.system(message.content))
                    default:
                        messages.append(.user(message.content))
                    }
                }
                messages.append(.user(prompt))

                let userInput = UserInput(
                    chat: messages,
                    additionalContext: ["enable_thinking": false]
                )
                let lmInput = try await container.prepare(input: userInput)

                let draftConfiguration = ModelConfiguration(id: draftModelId)
                let draftContext = try await LLMModelFactory.shared.load(
                    from: HuggingFaceDownloader(client: HubClient.default),
                    using: TokenizersLoader(),
                    configuration: draftConfiguration
                )

                let params = GenerateParameters(
                    maxTokens: maxTokens ?? config.maxTokens,
                    maxKVSize: BadAppleInference.envKVSize,
                    temperature: temperature ?? config.temperature,
                    topP: config.topP,
                    prefillStepSize: BadAppleInference.envPrefillStepSize
                )

                let stream: AsyncStream<Generation> = try await container.perform(
                    nonSendable: (draftContext, lmInput)
                ) { mainContext, payload in
                    let (draftCtx, input) = payload
                    return try MLXLMCommon.generate(
                        input: input,
                        cache: nil,
                        parameters: params,
                        context: mainContext,
                        draftModel: draftCtx.model,
                        draftCache: nil,
                        numDraftTokens: numDraftTokens
                    )
                }

                var fullText = ""
                var tps: Float = 0
                var tokenCount = 0

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        fullText += text
                        onToken(text)
                    case .info(let info):
                        tps = Float(info.tokensPerSecond)
                        tokenCount = info.generationTokenCount
                    case .toolCall:
                        break
                    }
                }

                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main"
                ))
            } catch {
                onError(InferenceError.generationFailed(error.localizedDescription))
            }
        }
    }

    // MARK: - Memory Management

    public func clearCache() {
        Memory.clearCache()
    }

    public func unload() async {
        await state.setContainer(nil)
        Memory.clearCache()
    }

    public var memoryUsageGB: Float {
        Float(Memory.activeMemory) / Float(1024 * 1024 * 1024)
    }
}

// MARK: - Convenience

public extension BadAppleInference {
    public static let defaultConfig = ModelConfig(
        modelId: "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
        revision: "5ae9734004d530171fd52f89e660c059b6e36efc",
        maxTokens: 300,
        temperature: 0.6,
        topP: 0.9
    )

    /// Default 0.5B fast-tier model for simple queries.
    public static let fastTierConfig = ModelConfig(
        modelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        revision: "main",
        maxTokens: 150,
        temperature: 0.6,
        topP: 0.9
    )

    static func createDefault() -> BadAppleInference {
        BadAppleInference(config: defaultConfig)
    }

    /// Read `BADAPPLE_MAX_KV_SIZE` from the environment (default 4096).
    public static var envKVSize: Int {
        if let raw = ProcessInfo.processInfo.environment["BADAPPLE_MAX_KV_SIZE"],
           let val = Int(raw), val > 0 {
            return val
        }
        return 4096
    }

    /// Read `BADAPPLE_PREFILL_STEP_SIZE` from the environment (default 4096).
    public static var envPrefillStepSize: Int {
        if let raw = ProcessInfo.processInfo.environment["BADAPPLE_PREFILL_STEP_SIZE"],
           let val = Int(raw), val > 0 {
            return val
        }
        return 4096
    }

    /// Read `BADAPPLE_FAST_MODEL` from the environment, falling back to the default 0.5B.
    public static var envFastModelId: String {
        ProcessInfo.processInfo.environment["BADAPPLE_FAST_MODEL"]
            ?? fastTierConfig.modelId
    }

    /// Read `BADAPPLE_SPECULATIVE_DRAFT` from the environment (empty = disabled).
    public static var envSpeculativeDraftModel: String? {
        guard let raw = ProcessInfo.processInfo.environment["BADAPPLE_SPECULATIVE_DRAFT"],
              !raw.isEmpty, raw != "0" else { return nil }
        return raw
    }

    /// Read `BADAPPLE_NUM_DRAFT_TOKENS` from the environment (default 2).
    public static var envNumDraftTokens: Int {
        if let raw = ProcessInfo.processInfo.environment["BADAPPLE_NUM_DRAFT_TOKENS"],
           let val = Int(raw), val > 0 {
            return val
        }
        return 2
    }
}
