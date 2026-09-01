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

        public init(
            text: String,
            tokensPerSecond: Float = 0,
            tokenCount: Int = 0,
            tier: String = "main"
        ) {
            self.text = text
            self.tokensPerSecond = tokensPerSecond
            self.tokenCount = tokenCount
            self.tier = tier
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

    public func generateStreamingTokens(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
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
                let params = GenerateParameters(
                    maxTokens: maxTokens ?? config.maxTokens,
                    maxKVSize: 4096,
                    temperature: temperature ?? config.temperature,
                    topP: config.topP,
                    prefillStepSize: 4096
                )

                let stream = try await container.generate(input: lmInput, parameters: params)

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

    /// Generate a complete response (non-streaming).
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> GenerationResult {
        try await withCheckedThrowingContinuation { continuation in
            generateStreamingTokens(
                prompt: prompt,
                systemPrompt: systemPrompt,
                history: history,
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
                    maxKVSize: 4096,
                    temperature: temperature ?? config.temperature,
                    topP: config.topP,
                    prefillStepSize: 4096
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
    static let defaultConfig = ModelConfig(
        modelId: "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
        revision: "5ae9734004d530171fd52f89e660c059b6e36efc",
        maxTokens: 300,
        temperature: 0.6,
        topP: 0.9
    )

    static func createDefault() -> BadAppleInference {
        BadAppleInference(config: defaultConfig)
    }
}
