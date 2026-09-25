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
        public let draftProposedTokens: Int
        public let draftAcceptedTokens: Int
        public let draftAcceptPct: Float
        /// `hit` = reused warmed prefix KV, `warm` = built it this turn,
        /// `off` = disabled or prefix mismatch.
        public let prefixCache: String

        public init(
            text: String,
            tokensPerSecond: Float = 0,
            tokenCount: Int = 0,
            tier: String = "main",
            toolCalls: [ToolCall] = [],
            draftProposedTokens: Int = 0,
            draftAcceptedTokens: Int = 0,
            prefixCache: String = "off"
        ) {
            self.text = text
            self.tokensPerSecond = tokensPerSecond
            self.tokenCount = tokenCount
            self.tier = tier
            self.toolCalls = toolCalls
            self.draftProposedTokens = draftProposedTokens
            self.draftAcceptedTokens = draftAcceptedTokens
            self.prefixCache = prefixCache
            self.draftAcceptPct = if draftProposedTokens > 0 {
                Float(draftAcceptedTokens) / Float(draftProposedTokens) * 100.0
            } else {
                0.0
            }
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

    // MARK: - Prompt-prefix KV cache
    //
    // The stable head of the system prompt (persona + shared-life context,
    // before ambient/RAG/briefing are appended) is byte-identical across
    // turns. Its KV state is warmed once and reused on later turns so prefill
    // only evaluates the new history + message. Safety model: the boundary is
    // located with a sentinel and verified token-for-token every call; the
    // cache is checked out for the duration of a turn and trimmed back to the
    // warmed boundary when the stream drains; any miss, offset mismatch, or
    // non-trimmable cache drops the entry and the turn simply runs uncached.
    // `BADAPPLE_PREFIX_CACHE=0` disables.

    /// The engine sets this to the stable system-prompt head each query.
    /// A stale value can't corrupt anything — the token-exact boundary check
    /// fails closed to a plain full-prefill turn.
    public var stableSystemPrefix: String?

    public static var envPrefixCache: Bool {
        ProcessInfo.processInfo.environment["BADAPPLE_PREFIX_CACHE"] != "0"
    }

    /// Rare delimiter appended to the stable head inside a probe render —
    /// control chars on both sides keep BPE from merging across it, so its
    /// standalone tokenization matches its in-context one. The boundary is
    /// the sentinel's position: the end of the shared head content, before
    /// `im_end`, history, and the current-turn markup.
    private static let prefixSentinel = "\u{7}\u{7}BA_PREFIX_SENTINEL\u{7}\u{7}"

    /// One turn's prefix-cache handoff: boundary + sliced inputs going in,
    /// the warmed/reused cache coming back out.
    private final class PrefixTicket {
        var key = ""
        var boundary = 0
        var prefixText: LMInput.Text!
        var suffixInput: LMInput!
        var checkedOut: [KVCache]?
        var cacheUsed: [KVCache]?
        var draftCacheUsed: [KVCache]?
    }

    private let prefixCacheLock = NSLock()
    private var prefixCacheKey = ""
    private var prefixCacheBoundary = 0
    private var prefixCache: [KVCache]?

    private static func firstIndex(of needle: [Int32], in haystack: [Int32]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        for i in 0...(haystack.count - needle.count) where haystack[i] == needle[0] {
            if Array(haystack[i ..< i + needle.count]) == needle { return i }
        }
        return nil
    }

    /// Takes the stored cache for this prefix (if valid) so the turn owns it
    /// exclusively; `returnPrefixCache` puts it back once it's trimmed.
    private func checkoutPrefixCache(key: String, boundary: Int) -> [KVCache]? {
        prefixCacheLock.lock(); defer { prefixCacheLock.unlock() }
        guard let cache = prefixCache,
              prefixCacheKey == key, prefixCacheBoundary == boundary,
              cache.first?.offset == boundary,
              canTrimPromptCache(cache) else { return nil }
        prefixCache = nil
        prefixCacheKey = ""
        return cache
    }

    private func returnPrefixCache(_ ticket: PrefixTicket) {
        guard let cache = ticket.cacheUsed else { return }
        let extra = (cache.first?.offset ?? 0) - ticket.boundary
        if extra > 0 { trimPromptCache(cache, numTokens: extra) }
        guard extra >= 0, cache.first?.offset == ticket.boundary else { return }
        prefixCacheLock.lock(); defer { prefixCacheLock.unlock() }
        prefixCacheKey = ticket.key
        prefixCacheBoundary = ticket.boundary
        prefixCache = cache
    }

    /// Sendable probe result — the async half of prefix matching. The caller
    /// verifies it against the real input's tokens and builds the ticket
    /// synchronously so `lmInput` never crosses an actor boundary.
    private struct PrefixProbe: Sendable {
        var key: String
        var boundary: Int
        var prefixTokens: [Int32]
    }

    /// Renders `head + sentinel` as the system message and locates the
    /// sentinel's token offset — the exact end of the shared prefix, before
    /// the system-close tag and any history or user markup that follow in
    /// real renders.
    private func probePrefixBoundary(head: String, container: ModelContainer) async -> PrefixProbe? {
        guard let probe = try? await prepareInput(
            prompt: "", systemPrompt: head + Self.prefixSentinel, history: [], tools: nil, container: container
        ) else { return nil }
        let probeTokens = probe.text.tokens.asType(.int32).asArray(Int32.self)
        let sentinelTokens = (await container.tokenizer).encode(text: Self.prefixSentinel).map { Int32($0) }
        guard !sentinelTokens.isEmpty,
              let boundary = Self.firstIndex(of: sentinelTokens, in: probeTokens),
              boundary > 0 else { return nil }
        return PrefixProbe(
            key: config.modelId + "\u{0}" + head,
            boundary: boundary,
            prefixTokens: Array(probeTokens[..<boundary])
        )
    }

    /// Token-exact verification + ticket construction. Takes the rendered
    /// input's token array rather than `LMInput` itself so the caller's
    /// non-Sendable input never crosses a boundary.
    private func buildPrefixTicket(probe: PrefixProbe, fullTokens: [Int32]) -> PrefixTicket? {
        guard fullTokens.count > probe.boundary,
              Array(fullTokens[..<probe.boundary]) == probe.prefixTokens else { return nil }
        let ticket = PrefixTicket()
        ticket.key = probe.key
        ticket.boundary = probe.boundary
        ticket.prefixText = LMInput.Text(tokens: MLXArray(probe.prefixTokens))
        ticket.suffixInput = LMInput(tokens: MLXArray(Array(fullTokens[probe.boundary...])))
        ticket.checkedOut = checkoutPrefixCache(key: probe.key, boundary: probe.boundary)
        return ticket
    }

    // MARK: - Draft model + draft prefix cache (speculative path)

    /// The draft `ModelContext` and its own warmed prefix cache, keyed by
    /// draft model id + prefix key. Keeps speculative decoding from paying a
    /// model reload and a full draft prefill on every query.
    private final class DraftBox {
        var modelId = ""
        var context: ModelContext?
        var prefixKey = ""
        var boundary = 0
        var cache: [KVCache]?
    }

    private let draftLock = NSLock()
    private let draftBox = DraftBox()

    private func cachedDraftContext(id draftModelId: String) -> ModelContext? {
        draftLock.lock(); defer { draftLock.unlock() }
        return draftBox.modelId == draftModelId ? draftBox.context : nil
    }

    private func storeDraftContext(id draftModelId: String, _ context: ModelContext) {
        draftLock.lock(); defer { draftLock.unlock() }
        if draftBox.modelId != draftModelId {
            // Model changed — stale draft KV must never cross models.
            draftBox.cache = nil
            draftBox.prefixKey = ""
            draftBox.boundary = 0
        }
        draftBox.modelId = draftModelId
        draftBox.context = context
    }

    private func loadDraftContext(id draftModelId: String) async throws -> ModelContext {
        if let cached = cachedDraftContext(id: draftModelId) { return cached }
        let draftContext = try await LLMModelFactory.shared.load(
            from: HuggingFaceDownloader(client: HubClient.default),
            using: TokenizersLoader(),
            configuration: ModelConfiguration(id: draftModelId)
        )
        storeDraftContext(id: draftModelId, draftContext)
        return draftContext
    }

    private func checkoutDraftPrefix(key: String, boundary: Int) -> [KVCache]? {
        draftLock.lock(); defer { draftLock.unlock() }
        guard let cache = draftBox.cache,
              draftBox.prefixKey == key, draftBox.boundary == boundary,
              cache.first?.offset == boundary,
              canTrimPromptCache(cache) else { return nil }
        draftBox.cache = nil
        draftBox.prefixKey = ""
        return cache
    }

    private func returnDraftPrefix(key: String, boundary: Int, cache: [KVCache]) {
        let extra = (cache.first?.offset ?? 0) - boundary
        if extra > 0 { trimPromptCache(cache, numTokens: extra) }
        guard extra >= 0, cache.first?.offset == boundary else { return }
        draftLock.lock(); defer { draftLock.unlock() }
        draftBox.prefixKey = key
        draftBox.boundary = boundary
        draftBox.cache = cache
    }

    // MARK: - Initialization

    public init(config: ModelConfig) {
        self.config = config
    }

    // MARK: - Model Loading

    public func loadModel(admission: @Sendable (UInt64) async throws -> Void = { _ in }) async throws {
        guard await state.setLoading() else { return }

        do {
            let modelConfiguration = ModelConfiguration(
                id: config.modelId,
                revision: config.revision
            )
            let resolved = try await resolve(
                configuration: modelConfiguration,
                from: HuggingFaceDownloader(client: HubClient.default),
                useLatest: false,
                progressHandler: { _ in }
            )
            try await admission(Self.estimatedModelMemory(in: resolved.modelDirectory))
            let container = try await LLMModelFactory.shared.loadContainer(
                from: resolved.modelDirectory,
                using: TokenizersLoader()
            )

            await state.setContainer(container)
        } catch {
            await state.setContainer(nil)
            Memory.clearCache()
            if let error = error as? InferenceError { throw error }
            throw InferenceError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Load from a local directory of already-downloaded model weights.
    public func loadModel(from localDirectory: URL, admission: @Sendable (UInt64) async throws -> Void = { _ in }) async throws {
        guard await state.setLoading() else { return }

        do {
            try await admission(Self.estimatedModelMemory(in: localDirectory))
            let container = try await LLMModelFactory.shared.loadContainer(
                from: localDirectory,
                using: TokenizersLoader()
            )

            await state.setContainer(container)
        } catch {
            await state.setContainer(nil)
            Memory.clearCache()
            if let error = error as? InferenceError { throw error }
            throw InferenceError.modelLoadFailed(error.localizedDescription)
        }
    }

    public static func estimatedModelMemory(
        in directory: URL, kvSize: Int = envKVSize, prefillStepSize: Int = envPrefillStepSize
    ) throws -> UInt64 {
        func invalid(_ detail: String) -> InferenceError {
            .modelLoadFailed("Cannot measure model memory: \(detail)")
        }
        func product(_ values: UInt64...) throws -> UInt64 {
            try values.reduce(1) { result, value in
                let next = result.multipliedReportingOverflow(by: value)
                guard !next.overflow else { throw invalid("size overflow") }
                return next.partialValue
            }
        }
        func sum(_ values: UInt64...) throws -> UInt64 {
            try values.reduce(0) { result, value in
                let next = result.addingReportingOverflow(value)
                guard !next.overflow else { throw invalid("size overflow") }
                return next.partialValue
            }
        }
        let fm = FileManager.default
        let directory = directory.resolvingSymlinksInPath().standardizedFileURL
        let configURL = directory.appendingPathComponent("config.json")
        guard let configSize = try configURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              configSize <= 1_048_576,
              let root = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]
        else { throw invalid("invalid config.json") }
        var config = root.merging(root["text_config"] as? [String: Any] ?? [:]) { _, nested in nested }
        for (key, aliases) in [
            "hidden_size": ["n_embd", "d_model"],
            "num_attention_heads": ["n_head", "n_heads"],
            "num_hidden_layers": ["n_layer", "n_layers"],
            "intermediate_size": ["n_inner", "d_ff"],
        ] where config[key] == nil {
            if let alias = aliases.first(where: { config[$0] != nil }) { config[key] = config[alias] }
        }
        func dimension(_ key: String, fallback: UInt64? = nil) throws -> UInt64 {
            if config[key] == nil, let fallback { return fallback }
            guard let number = config[key] as? NSNumber,
                  let value = UInt64(exactly: number.doubleValue), value > 0
            else { throw invalid("missing or invalid \(key)") }
            return value
        }
        guard kvSize > 0, prefillStepSize > 0 else { throw invalid("invalid context limits") }
        let hidden = try dimension("hidden_size")
        let heads = try dimension("num_attention_heads")
        guard config["head_dim"] != nil || hidden % heads == 0 else {
            throw invalid("head_dim is required when hidden_size is not divisible by num_attention_heads")
        }
        let headSize = try dimension("head_dim", fallback: hidden / heads)
        guard headSize > 0 else { throw invalid("invalid attention dimensions") }
        let kvHeads = try dimension("num_key_value_heads", fallback: heads)
        let layers = try dimension("num_hidden_layers")
        let intermediate = try dimension("intermediate_size")
        let vocabulary = try dimension("vocab_size")
        let scalarBytes: UInt64 = (config["torch_dtype"] as? String ?? config["dtype"] as? String) == "float32" ? 4 : 2
        guard let enumerator = fm.enumerator(atPath: directory.path) else {
            throw invalid("cannot enumerate cached weights")
        }
        let relativePaths = Set(enumerator.compactMap { $0 as? String }.filter { $0.hasSuffix(".safetensors") })
        let files = relativePaths.map { directory.appendingPathComponent($0) }
        guard !files.isEmpty else { throw invalid("no cached safetensors weights") }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            guard let indexSize = try indexURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                  indexSize <= 16_777_216,
                  let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
                  let weights = index["weight_map"] as? [String: String], !weights.isEmpty,
                  Set(weights.values).isSubset(of: relativePaths)
            else { throw invalid("missing shards or invalid weight index") }
        }
        var weightBytes: UInt64 = 0
        for file in files {
            let info = try file.resolvingSymlinksInPath().resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard info.isRegularFile == true, let size = info.fileSize, size > 0 else {
                throw invalid("missing or empty weight shard")
            }
            weightBytes = try sum(weightBytes, UInt64(size))
        }
        let kvBytes = try product(2, layers, kvHeads, headSize, UInt64(kvSize), scalarBytes)
        let activationWidth = try sum(hidden, product(2, intermediate))
        let workspaceBytes = try product(UInt64(prefillStepSize), activationWidth, 4)
        let logitsBytes = try product(vocabulary, 4)
        return try sum(weightBytes, kvBytes, workspaceBytes, logitsBytes)
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

                // Prompt-prefix KV reuse: if the stable system-prompt head is
                // set and matches this render token-for-token, generate over a
                // warmed cache and evaluate only the suffix.
                var ticket: PrefixTicket?
                if Self.envPrefixCache, let head = stableSystemPrefix, !head.isEmpty,
                   let probe = await probePrefixBoundary(head: head, container: container) {
                    ticket = buildPrefixTicket(
                        probe: probe,
                        fullTokens: lmInput.text.tokens.asType(.int32).asArray(Int32.self))
                }

                let stream: AsyncStream<Generation>
                if let ticket {
                    stream = try await container.perform(nonSendable: (ticket, params)) { context, payload in
                        let (ticket, params) = payload
                        // Deliberately unbounded (parameters: nil): the
                        // prefix must survive whole, and per-turn growth is
                        // trimmed back to the boundary when the turn ends.
                        let cache = ticket.checkedOut
                            ?? makePromptCache(model: context.model, parameters: nil)
                        if ticket.checkedOut == nil {
                            // First turn for this prefix — warm the cache by
                            // evaluating just the stable head. Tokens are
                            // 1-D; the model expects a batch axis.
                            _ = context.model(
                                ticket.prefixText[text: .newAxis], cache: cache, state: nil)
                            eval(cache)
                        }
                        ticket.cacheUsed = cache
                        return try MLXLMCommon.generate(
                            input: ticket.suffixInput, cache: cache,
                            parameters: params, context: context)
                    }
                } else {
                    stream = try await container.generate(input: lmInput, parameters: params)
                }

                var fullText = ""
                var tps: Float = 0
                var tokenCount = 0
                var draftProposed = 0
                var draftAccepted = 0
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
                        draftProposed = info.proposedDraftTokens ?? 0
                        draftAccepted = info.acceptedDraftTokens ?? 0
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
                // Restore the prefix cache for the next turn — trimmed back to
                // the warmed boundary, or dropped if anything drifted.
                if let ticket { returnPrefixCache(ticket) }
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main",
                    toolCalls: finalToolCalls,
                    draftProposedTokens: draftProposed,
                    draftAcceptedTokens: draftAccepted,
                    prefixCache: ticket.map { $0.checkedOut != nil ? "hit" : "warm" } ?? "off"
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

                let draftContext = try await loadDraftContext(id: draftModelId)

                let params = GenerateParameters(
                    maxTokens: maxTokens ?? config.maxTokens,
                    maxKVSize: BadAppleInference.envKVSize,
                    temperature: temperature ?? config.temperature,
                    topP: config.topP,
                    prefillStepSize: BadAppleInference.envPrefillStepSize
                )

                // Same prefix reuse as the plain path — the main model gets a
                // warmed cache, and the draft gets its own warmed cache so its
                // proposals see the full system context too.
                var ticket: PrefixTicket?
                if Self.envPrefixCache, let head = stableSystemPrefix, !head.isEmpty,
                   let probe = await probePrefixBoundary(head: head, container: container) {
                    ticket = buildPrefixTicket(
                        probe: probe,
                        fullTokens: lmInput.text.tokens.asType(.int32).asArray(Int32.self))
                }

                let stream: AsyncStream<Generation> = try await container.perform(
                    nonSendable: (draftContext, lmInput, ticket, params)
                ) { mainContext, payload in
                    let (draftCtx, input, ticket, params) = payload
                    var cache: [KVCache]? = nil
                    var draftCache: [KVCache]? = nil
                    var genInput = input
                    if let ticket {
                        let warmed = ticket.checkedOut
                            ?? makePromptCache(model: mainContext.model, parameters: nil)
                        if ticket.checkedOut == nil {
                            // Tokens are 1-D; the model expects a batch axis.
                            _ = mainContext.model(
                                ticket.prefixText[text: .newAxis], cache: warmed, state: nil)
                            eval(warmed)
                        }
                        ticket.cacheUsed = warmed
                        cache = warmed
                        genInput = ticket.suffixInput

                        let dWarmed = self.checkoutDraftPrefix(
                            key: ticket.key, boundary: ticket.boundary)
                            ?? draftCtx.model.newCache(parameters: nil)
                        if dWarmed.first?.offset != ticket.boundary {
                            _ = draftCtx.model(
                                ticket.prefixText[text: .newAxis], cache: dWarmed, state: nil)
                            eval(dWarmed)
                        }
                        ticket.draftCacheUsed = dWarmed
                        draftCache = dWarmed
                    }
                    return try MLXLMCommon.generate(
                        input: genInput,
                        cache: cache,
                        parameters: params,
                        context: mainContext,
                        draftModel: draftCtx.model,
                        draftCache: draftCache,
                        numDraftTokens: numDraftTokens
                    )
                }

                var fullText = ""
                var tps: Float = 0
                var tokenCount = 0
                var draftProposed = 0
                var draftAccepted = 0

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        fullText += text
                        onToken(text)
                    case .info(let info):
                        tps = Float(info.tokensPerSecond)
                        tokenCount = info.generationTokenCount
                        draftProposed = info.proposedDraftTokens ?? 0
                        draftAccepted = info.acceptedDraftTokens ?? 0
                    case .toolCall:
                        break
                    }
                }

                if let ticket {
                    returnPrefixCache(ticket)
                    if let draftCache = ticket.draftCacheUsed {
                        returnDraftPrefix(key: ticket.key, boundary: ticket.boundary, cache: draftCache)
                    }
                }
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main",
                    draftProposedTokens: draftProposed,
                    draftAcceptedTokens: draftAccepted,
                    prefixCache: ticket.map { $0.checkedOut != nil ? "hit" : "warm" } ?? "off"
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

    /// Inject a saved LoRA adapter (`adapters.safetensors` + `adapter_config.json`)
    /// into the loaded model. The container mutates the module in place, so the
    /// adapter stays applied for subsequent generations until unload or replace.
    /// Callers handle failure — a bad adapter must never brick inference.
    public func applyAdapter(directory: URL) async throws {
        guard let container = await state.getContainer() else {
            throw InferenceError.modelNotLoaded
        }
        try await container.perform { context in
            let adapter = try LoRAContainer.from(directory: directory)
            try adapter.load(into: context.model)
        }
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
    /// Default launchd model: Qwen2.5-Coder-7B 4-bit. The 9B is a switchable
    /// option via `badapple model use` or `BADAPPLE_MAIN_MODEL`.
    public static let defaultConfig = ModelConfig(
        modelId: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        revision: "main",
        maxTokens: 2048,
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

    /// Read `BADAPPLE_FAST_MODEL` from the environment.
    /// - If the variable is unset, the default 0.5B fast-tier model is used.
    /// - If the variable is empty or "0", fast tier is treated as disabled (nil).
    public static var envFastModelId: String? {
        guard let raw = ProcessInfo.processInfo.environment["BADAPPLE_FAST_MODEL"] else {
            return fastTierConfig.modelId
        }
        if raw.isEmpty || raw == "0" {
            return nil
        }
        return raw
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
