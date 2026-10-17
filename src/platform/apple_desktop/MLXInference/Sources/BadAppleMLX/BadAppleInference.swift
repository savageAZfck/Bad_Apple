// BadAppleMLX — Swift-native MLX inference runtime for Bad Apple.
// Replaces the Python badapple_mlx_server.py with a single Swift module
// that loads the model, generates text, and streams tokens via callbacks.

import Foundation
import MLX
import MLXNN
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
        /// Tokenized length of the assembled prompt the model saw.
        public let promptTokenCount: Int

        public init(
            text: String,
            tokensPerSecond: Float = 0,
            tokenCount: Int = 0,
            tier: String = "main",
            toolCalls: [ToolCall] = [],
            draftProposedTokens: Int = 0,
            draftAcceptedTokens: Int = 0,
            prefixCache: String = "off",
            promptTokenCount: Int = 0
        ) {
            self.text = text
            self.tokensPerSecond = tokensPerSecond
            self.tokenCount = tokenCount
            self.tier = tier
            self.toolCalls = toolCalls
            self.draftProposedTokens = draftProposedTokens
            self.draftAcceptedTokens = draftAcceptedTokens
            self.prefixCache = prefixCache
            self.promptTokenCount = promptTokenCount
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
        case adapterGeometryMismatch(adapterIns: Set<Int>, targetIns: Set<Int>)

        public var errorDescription: String? {
            switch self {
            case .modelNotLoaded:
                return "The AI model is not loaded yet. Please wait a moment and try again."
            case .modelLoadFailed(let detail):
                return "Could not load the AI model: \(detail)"
            case .generationFailed(let detail):
                return "The AI could not generate a response: \(detail)"
            case .adapterGeometryMismatch(let adapterIns, let targetIns):
                return "LoRA adapter trained for a different geometry "
                    + "(adapter input dims \(adapterIns.sorted()), "
                    + "model layer input dims \(targetIns.sorted())) — refusing to apply."
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

        /// Mesh-brain mode: no local container — a rank-0 engine process owns
        /// the weights and serves `generate` over the mesh protocol.
        func setMeshReady() {
            self.container = nil
            self.isLoaded = true
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

    // MARK: - Mesh-brain backend
    //
    // A model directory carrying a rank-0 `mesh_brain_rank.json` loads no
    // local weights: the GPU prefix engine (BADAPPLE_SHARD_DIR=dir) owns them
    // and serves `generate` on spec.host; downstream ranks take `step` frames.
    // The daemon treats the whole pipeline as one swappable brain.

    private var meshHost: String?
    private var meshTokenizer: (any MLXLMCommon.Tokenizer)?

    /// True when the active brain is a mesh rank set rather than a local
    /// MLX container. Secondary tiers (fast model, ANE drafter, speculative
    /// draft) stay off — the mesh is the whole brain.
    public var isMeshBacked: Bool { meshHost != nil }

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
    /// a per-layer state snapshot (not a live cache) to restore from.
    private final class PrefixTicket {
        var key = ""
        var boundary = 0
        var prefixText: LMInput.Text!
        var suffixInput: LMInput!
        var snapshot: [(state: [MLXArray], meta: [String])]?
        var draftCacheUsed: [KVCache]?
    }

    private let prefixCacheLock = NSLock()
    private var prefixCacheKey = ""
    private var prefixCacheBoundary = 0
    /// Checkpoint protocol: the pooled artifact is a frozen per-layer state
    /// snapshot taken at the prefix boundary — never a live cache object.
    /// Rotating (sliding-window) caches cannot rewind — a ring buffer's
    /// earlier bytes are gone — but every cache type can be teleported back
    /// to a frozen state. Each turn gets fresh cache objects restored to
    /// post-head; generation mutates only the fresh objects, so the shared
    /// snapshot is never touched and nothing needs trimming.
    private var prefixSnapshot: [(state: [MLXArray], meta: [String])]?

    private static func firstIndex(of needle: [Int32], in haystack: [Int32]) -> Int? {
        guard !needle.isEmpty, haystack.count >= needle.count else { return nil }
        for i in 0...(haystack.count - needle.count) where haystack[i] == needle[0] {
            if Array(haystack[i ..< i + needle.count]) == needle { return i }
        }
        return nil
    }

    /// Hands the stored snapshot for this prefix (if any) to a turn.
    /// The snapshot is shared read-only state — it is never taken out of
    /// the pool, so concurrent turns can all restore from it.
    private func checkoutPrefixSnapshot(key: String, boundary: Int)
        -> [(state: [MLXArray], meta: [String])]?
    {
        prefixCacheLock.lock(); defer { prefixCacheLock.unlock() }
        guard let layers = prefixSnapshot,
              prefixCacheKey == key, prefixCacheBoundary == boundary else { return nil }
        return layers
    }

    /// Freezes the just-warmed cache's per-layer state into the pool.
    private func storePrefixSnapshot(key: String, boundary: Int, cache: [KVCache]) {
        let layers: [(state: [MLXArray], meta: [String])] = cache.map { layer in
            let s = layer.state
            s.forEach { eval($0) }
            return (s, layer.metaState)
        }
        prefixCacheLock.lock(); defer { prefixCacheLock.unlock() }
        prefixCacheKey = key
        prefixCacheBoundary = boundary
        prefixSnapshot = layers
    }

    /// Teleports a fresh cache to the frozen post-head state.
    private func restorePrefixSnapshot(
        _ layers: [(state: [MLXArray], meta: [String])], into cache: [KVCache]
    ) {
        for i in cache.indices where i < layers.count {
            var layer = cache[i]
            layer.state = layers[i].state
            layer.metaState = layers[i].meta
        }
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
        // Content tokens only — BOS-prepending tokenizers (Gemma & family)
        // would otherwise inject an id that can never occur mid-render,
        // silently disabling prefix reuse on every such model.
        let sentinelTokens = (await container.tokenizer)
            .encode(text: Self.prefixSentinel, addSpecialTokens: false).map { Int32($0) }
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
        ticket.snapshot = checkoutPrefixSnapshot(key: probe.key, boundary: probe.boundary)
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

        // Mesh-brain rank dir: wire up the pipeline client instead of loading
        // weights — the rank-0 engine process owns them. Probe the endpoint
        // now so a dead mesh fails the load instead of the first query.
        if let spec = Self.meshRankSpec(in: localDirectory), spec.rank == 0, spec.hasEmbed {
            do {
                let link = try await MeshBrainLink(host: spec.host)
                let (resp, _) = try await link.request(
                    MeshBrainFrame(op: "ping"), payload: nil)
                link.close()
                if let err = resp.error {
                    throw InferenceError.modelLoadFailed("mesh rank 0 ping: \(err)")
                }
            } catch let error as InferenceError {
                throw error
            } catch {
                throw InferenceError.modelLoadFailed(
                    "mesh rank 0 at \(spec.host) unreachable: \(error.localizedDescription)")
            }
            meshHost = spec.host
            meshTokenizer = try? await TokenizersLoader().load(from: localDirectory)
            await state.setMeshReady()
            // The old local container is gone — drop any residual MLX/Metal
            // buffers so the mesh ranks keep the memory budget to themselves.
            Memory.clearCache()
            return
        }

        meshHost = nil
        meshTokenizer = nil
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
        // Mesh rank dirs hold a prefix's weights for the rank process, not the
        // caller — admission only needs tokenizer + connection headroom.
        if meshRankSpec(in: directory) != nil { return 256 * 1024 * 1024 }
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

    /// Parse `mesh_brain_rank.json` if `directory` is a mesh shard dir.
    private static func meshRankSpec(in directory: URL) -> MeshBrainRankSpec? {
        let url = directory.appendingPathComponent("mesh_brain_rank.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(MeshBrainRankSpec.self, from: data)
    }

    /// Render chat messages to raw prompt text for the mesh `generate` op —
    /// rank 0 re-tokenizes, so the rendered string round-trips through encode.
    private func renderMeshPrompt(
        prompt: String,
        systemPrompt: String?,
        history: [ChatMessage]
    ) -> String {
        let messages = buildRawMessages(
            prompt: prompt, systemPrompt: systemPrompt, history: history)
        if let tok = meshTokenizer,
           let ids = try? tok.applyChatTemplate(
               messages: messages, tools: nil,
               additionalContext: ["enable_thinking": false])
        {
            return tok.decode(tokenIds: ids, skipSpecialTokens: false)
        }
        var text = systemPrompt.map { $0 + "\n\n" } ?? ""
        for m in history.suffix(12) { text += "\(m.role): \(m.content)\n" }
        text += prompt
        return text
    }

    /// Drive the mesh pipeline: connect to the rank-0 engine, send one
    /// `generate` frame, stream the result back as a single chunk.
    private func generateViaMesh(
        prompt: String,
        systemPrompt: String?,
        history: [ChatMessage],
        tools: [[String: any Sendable]]?,
        maxTokens: Int?,
        host: String,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) async {
        do {
            let rendered = renderMeshPrompt(
                prompt: prompt, systemPrompt: systemPrompt, history: history)
            let link = try await MeshBrainLink(host: host)
            var frame = MeshBrainFrame(op: "generate")
            frame.prompt = rendered
            frame.maxTokens = maxTokens ?? config.maxTokens
            let started = ProcessInfo.processInfo.systemUptime
            // One frame carries the whole generation — and the prefill
            // dominates: a multi-thousand-token system prompt costs
            // ~0.15 s/row through the serial ANE tail. Scale the wire
            // budget by prompt rows + generation tokens, not tokens alone.
            let promptTokens = meshTokenizer?
                .encode(text: rendered, addSpecialTokens: false).count
                ?? rendered.count / 4
            let budget = meshTimeout
                + TimeInterval(promptTokens) * 0.5
                + TimeInterval(frame.maxTokens ?? 0) * 30
            let (resp, _) = try await link.request(
                frame, payload: nil, timeout: budget)
            link.close()
            let elapsed = ProcessInfo.processInfo.systemUptime - started
            if let err = resp.error { throw MeshBrainError.transport(err) }
            let text = resp.text ?? ""
            if !text.isEmpty { onToken(text) }
            // token count is approximated from the tokenizer when available —
            // the wire only carries the finished string.
            let count = meshTokenizer?.encode(text: text, addSpecialTokens: false).count ?? 0
            onComplete(GenerationResult(
                text: text,
                tokensPerSecond: elapsed > 0 ? Float(count) / Float(elapsed) : 0,
                tokenCount: count,
                tier: "mesh",
                toolCalls: parseToolCalls(text),
                prefixCache: "off"
            ))
        } catch {
            // localizedDescription on a bridged enum hides the message —
            // CustomStringConvertible carries it ("transport: downstream …").
            onError(InferenceError.generationFailed("\(error)"))
        }
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
                if let meshHost {
                    await generateViaMesh(
                        prompt: prompt,
                        systemPrompt: systemPrompt,
                        history: history,
                        tools: tools,
                        maxTokens: maxTokens,
                        host: meshHost,
                        onToken: onToken,
                        onComplete: onComplete,
                        onError: onError
                    )
                    return
                }
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
                        // Fresh cache objects every turn — restored from the
                        // checkpoint when present, warmed + checkpointed
                        // otherwise. Deliberately unbounded (parameters:
                        // nil): the prefix must survive whole.
                        let cache = makePromptCache(model: context.model, parameters: nil)
                        if let snapshot = ticket.snapshot {
                            restorePrefixSnapshot(snapshot, into: cache)
                        } else {
                            // First turn for this prefix — warm the cache by
                            // evaluating just the stable head, then freeze.
                            // Tokens are 1-D; the model expects a batch axis.
                            _ = context.model(
                                ticket.prefixText[text: .newAxis], cache: cache, state: nil)
                            eval(cache)
                            storePrefixSnapshot(
                                key: ticket.key, boundary: ticket.boundary, cache: cache)
                        }
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
                var promptTokens = 0

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        fullText += text
                        onToken(text)
                    case .info(let info):
                        tps = Float(info.tokensPerSecond)
                        tokenCount = info.generationTokenCount
                        promptTokens = info.promptTokenCount
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
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main",
                    toolCalls: finalToolCalls,
                    draftProposedTokens: draftProposed,
                    draftAcceptedTokens: draftAccepted,
                    prefixCache: ticket.map { $0.snapshot != nil ? "hit" : "warm" } ?? "off",
                    promptTokenCount: promptTokens
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
    /// Acceptance rate of the most recent speculative generation — the
    /// draft-depth adapter's feedback signal. 0 until a spec turn lands.
    private var lastSpecAcceptancePct: Float = 0

    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        tools: [[String: any Sendable]]?,
        maxTokens: Int? = nil,
        temperature: Float? = nil
    ) async throws -> GenerationResult {
        // Plain-text turns can run through a drafting path: the spec verify
        // loop emits identical tokens to serial greedy (argmax-checked), so
        // output is unchanged — just cheaper. Tool-calling turns stay on the
        // streaming path (it parses .toolCall events; spec doesn't yet).
        if tools == nil, !isMeshBacked {
            let numDrafts = Self.adaptiveNumDraftTokens(
                acceptPct: lastSpecAcceptancePct, envDefault: Self.envNumDraftTokens)
            if let aneSession = await BadAppleANEDrafter.shared.acquire() {
                return try await withCheckedThrowingContinuation { continuation in
                    generateWithANEDrafting(
                        prompt: prompt, systemPrompt: systemPrompt, history: history,
                        session: aneSession, numDraftTokens: numDrafts,
                        maxTokens: maxTokens,
                        onToken: { _ in },
                        onComplete: { result in
                            if result.draftProposedTokens > 0 {
                                self.lastSpecAcceptancePct = result.draftAcceptPct
                            }
                            continuation.resume(returning: result)
                        },
                        onError: { continuation.resume(throwing: $0) })
                }
            }
            if let draftId = Self.envSpeculativeDraftModel {
                return try await withCheckedThrowingContinuation { continuation in
                    generateWithSpeculativeDecoding(
                        prompt: prompt, systemPrompt: systemPrompt, history: history,
                        draftModelId: draftId, numDraftTokens: numDrafts,
                        maxTokens: maxTokens, temperature: temperature,
                        onToken: { _ in },
                        onComplete: { result in
                            if result.draftProposedTokens > 0 {
                                self.lastSpecAcceptancePct = result.draftAcceptPct
                            }
                            continuation.resume(returning: result)
                        },
                        onError: { continuation.resume(throwing: $0) })
                }
            }
        }
        return try await withCheckedThrowingContinuation { continuation in
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
                        let warmed = makePromptCache(model: mainContext.model, parameters: nil)
                        if let snapshot = ticket.snapshot {
                            restorePrefixSnapshot(snapshot, into: warmed)
                        } else {
                            // Tokens are 1-D; the model expects a batch axis.
                            _ = mainContext.model(
                                ticket.prefixText[text: .newAxis], cache: warmed, state: nil)
                            eval(warmed)
                            storePrefixSnapshot(
                                key: ticket.key, boundary: ticket.boundary, cache: warmed)
                        }
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
                var promptTokens = 0

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        fullText += text
                        onToken(text)
                    case .info(let info):
                        tps = Float(info.tokensPerSecond)
                        tokenCount = info.generationTokenCount
                        promptTokens = info.promptTokenCount
                        if let spec = info.speculativeDecodingTelemetry {
                            draftProposed = spec.draftTokenCount
                            draftAccepted = spec.acceptedDraftTokenCount
                        } else {
                            draftProposed = info.proposedDraftTokens ?? 0
                            draftAccepted = info.acceptedDraftTokens ?? 0
                        }
                    case .toolCall:
                        break
                    }
                }

                if let ticket, let draftCache = ticket.draftCacheUsed {
                    returnDraftPrefix(key: ticket.key, boundary: ticket.boundary, cache: draftCache)
                }
                let prefixDesc = ticket.map { $0.snapshot != nil ? "hit" : "warm" } ?? "off"
                NSLog("[BadAppleSpec] prompt=%d tokens gen=%d tps=%.1f drafts=%d accepted=%d prefix=%@",
                      promptTokens, tokenCount, tps, draftProposed, draftAccepted, prefixDesc)
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: tps,
                    tokenCount: tokenCount,
                    tier: "main+spec",
                    draftProposedTokens: draftProposed,
                    draftAcceptedTokens: draftAccepted,
                    prefixCache: prefixDesc,
                    promptTokenCount: promptTokens
                ))
            } catch {
                onError(InferenceError.generationFailed(error.localizedDescription))
            }
        }
    }

    // MARK: - Cross-Silicon Speculative Decoding (ANE draft → GPU verify)

    private struct ANESpecOutcome: Sendable {
        var emitted: [Int32] = []
        var proposed = 0
        var accepted = 0
        var rounds = 0
        var draftNs: UInt64 = 0
        var verifyNs: UInt64 = 0
        var prefillNs: UInt64 = 0
        var anePrefillNs: UInt64 = 0
        var stoppedOnEOS = false
        var draftingSurvived = true
    }

    /// ANE-drafted generation. The 0.6B on the Neural Engine proposes
    /// `numDraftTokens` candidates per round; the GPU target verifies
    /// them in one forward pass and commits the longest matching prefix
    /// plus its own correction/bonus token. Rejected state is rewound
    /// on both sides: `trimPromptCache` for GPU KV, the bridge's
    /// position-indexed rewind for the ANE KV.
    ///
    /// GREEDY ONLY: the bridge emits argmax, so verification compares
    /// argmax-vs-argmax. `temperature` is deliberately ignored — mixing
    /// a sampled verifier with argmax drafts would corrupt the
    /// acceptance measurement.
    ///
    /// If the ANE side faults mid-generation the loop degrades to plain
    /// greedy decode on the already-warmed GPU cache — the generation
    /// completes rather than dying with the drafter.
    public func generateWithANEDrafting(
        prompt: String,
        systemPrompt: String? = nil,
        history: [ChatMessage] = [],
        session: BadAppleANEDrafter.Session,
        numDraftTokens: Int = 4,
        maxTokens: Int? = nil,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (GenerationResult) -> Void,
        onError: @escaping @Sendable (Error) -> Void
    ) {
        Task {
            defer { Task { await BadAppleANEDrafter.shared.release() } }
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
                    case "assistant": messages.append(.assistant(message.content))
                    case "system": messages.append(.system(message.content))
                    default: messages.append(.user(message.content))
                    }
                }
                messages.append(.user(prompt))

                let userInput = UserInput(
                    chat: messages,
                    additionalContext: ["enable_thinking": false]
                )
                let lmInput = try await container.prepare(input: userInput)
                let promptIds = lmInput.text.tokens.asType(.int32).asArray(Int32.self)

                let maxTok = maxTokens ?? config.maxTokens
                // ANE sees a tail window of the prompt: drafts only need
                // recent context, the verifier holds the truth. Budget
                // leaves room for the whole generation.
                let aneBudget = session.contextLimit - maxTok - numDraftTokens - 8
                let minWindow = 128
                guard aneBudget >= minWindow else {
                    NSLog("[BadAppleSpecANE] no ANE window (context %d, maxTokens %d) — falling back to plain GPU",
                          session.contextLimit, maxTok)
                    self.generateStreamingTokens(
                        prompt: prompt,
                        systemPrompt: systemPrompt,
                        history: history,
                        maxTokens: maxTok,
                        temperature: 0.6,
                        onToken: onToken,
                        onComplete: onComplete,
                        onError: onError
                    )
                    return
                }
                let anePromptIds = Array(promptIds.suffix(aneBudget))
                if anePromptIds.count < promptIds.count {
                    NSLog("[BadAppleSpecANE] ANE prefill windowed: %d of %d prompt tokens",
                          anePromptIds.count, promptIds.count)
                }

                let outcome = try await container.perform(
                    nonSendable: (session, promptIds, anePromptIds, maxTok, numDraftTokens)
                ) { ctx, payload -> ANESpecOutcome in
                    let (drafter, promptIds, anePromptIds, maxTok, numDraft) = payload
                    var outcome = ANESpecOutcome()

                    var cache = makePromptCache(model: ctx.model, parameters: nil)
                    guard canTrimPromptCache(cache) else {
                        outcome.draftingSurvived = false
                        return outcome
                    }
                    var eosIds = Set(ctx.configuration.eosTokenIds.map(Int32.init))
                    eosIds.formUnion(
                        [ctx.tokenizer.eosToken, "<|im_end|>", "<|endoftext|>"]
                            .compactMap { $0 }
                            .compactMap { ctx.tokenizer.convertTokenToId($0) }
                            .map { Int32($0) })

                    func forward(_ ids: [Int32], _ state: inout LMOutput.State?) -> MLXArray {
                        let text = LMInput.Text(tokens: MLXArray(ids))
                        let out = ctx.model(text[text: .newAxis], cache: cache, state: state)
                        state = out.state
                        return out.logits
                    }

                    var emittedSoFar = ""
                    func emitDelta() {
                        let text = ctx.tokenizer.decode(
                            tokenIds: outcome.emitted.map(Int.init), skipSpecialTokens: true)
                        if text.count > emittedSoFar.count {
                            onToken(String(text.dropFirst(emittedSoFar.count)))
                            emittedSoFar = text
                        }
                    }

                    // GPU prefill — one forward over the whole prompt.
                    var t0 = DispatchTime.now()
                    var mstate: LMOutput.State? = nil
                    let plogits = forward(promptIds, &mstate)
                    eval(plogits)
                    var y = argMax(plogits[0..., -1, 0...]).item(Int32.self)
                    outcome.prefillNs = DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds
                    outcome.emitted.append(y)
                    emitDelta()
                    var stop = eosIds.contains(y)

                    // ANE prime: commit the windowed prompt in batched
                    // chunks (discard its proposal for y's slot — the
                    // GPU's argmax won), then commit y itself — its
                    // return is the first draft proposal.
                    t0 = DispatchTime.now()
                    _ = drafter.reset()
                    let anePrimed = drafter.prefill(anePromptIds)
                    outcome.anePrefillNs = DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds
                    var pending = anePrimed ? drafter.predictNext([y]) : -1
                    var draftingAlive = pending >= 0
                    outcome.draftNs = DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds
                        - outcome.anePrefillNs
                    outcome.draftingSurvived = draftingAlive

                    while outcome.emitted.count < maxTok && !stop {
                        if draftingAlive {
                            // Collect k drafts: each predict commits the
                            // previous draft and proposes the next slot.
                            let F = drafter.position   // pending proposes slot F
                            t0 = DispatchTime.now()
                            var drafts: [Int32] = [pending]
                            while drafts.count < numDraft {
                                let nxt = drafter.predictNext([drafts.last!])
                                if nxt < 0 { draftingAlive = false; break }
                                drafts.append(nxt)
                            }
                            outcome.draftNs += DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds
                            guard draftingAlive else {
                                _ = drafter.reset()
                                continue
                            }

                            // Verify: [y] + drafts in ONE GPU forward.
                            t0 = DispatchTime.now()
                            let vlogits = forward([y] + drafts, &mstate)
                            eval(vlogits)
                            let k = drafts.count
                            let vstart = vlogits.dim(1) - (k + 1)
                            let mts = argMax(vlogits[0..., vstart..., 0...]
                                .squeezed(axis: 0), axis: -1).asArray(Int32.self)
                            outcome.verifyNs += DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds

                            var n = 0
                            while n < k && mts[n] == drafts[n] { n += 1 }
                            outcome.proposed += k
                            outcome.accepted += n
                            outcome.rounds += 1

                            // GPU keeps the anchor + n accepted positions.
                            _ = trimPromptCache(cache, numTokens: k - n)

                            let remaining = maxTok - outcome.emitted.count
                            let roundTokens = Array(mts[0 ... Swift.min(n, k)]).prefix(remaining)
                            outcome.emitted.append(contentsOf: roundTokens)
                            emitDelta()
                            y = mts[n]
                            stop = roundTokens.contains(where: { eosIds.contains($0) })
                                || outcome.emitted.count >= maxTok
                            if stop { break }

                            // ANE resync: rewind over rejected commits,
                            // then feed the verifier's token so the next
                            // pending proposes the new frontier.
                            t0 = DispatchTime.now()
                            if n == k {
                                pending = drafter.predictNext([drafts[k - 1], mts[k]])
                            } else {
                                _ = drafter.rewind(to: F + Int32(n))
                                pending = drafter.predictNext([mts[n]])
                            }
                            outcome.draftNs += DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds
                            if pending < 0 {
                                draftingAlive = false
                                outcome.draftingSurvived = false
                                _ = drafter.reset()
                            }
                        } else {
                            // ANE died mid-generation: finish greedy on
                            // the warmed GPU cache rather than erroring.
                            let lg = forward([y], &mstate)
                            eval(lg)
                            y = argMax(lg[0..., -1, 0...]).item(Int32.self)
                            outcome.emitted.append(y)
                            emitDelta()
                            stop = eosIds.contains(y)
                        }
                    }
                    outcome.stoppedOnEOS = stop && eosIds.contains(y)
                    Memory.clearCache()
                    return outcome
                }

                await BadAppleANEDrafter.shared.record(
                    proposed: outcome.proposed, accepted: outcome.accepted, rounds: outcome.rounds)

                let tokenizer = await container.tokenizer
                let fullText = tokenizer.decode(
                    tokenIds: outcome.emitted.map(Int.init), skipSpecialTokens: true)
                let elapsedS = Double(outcome.prefillNs + outcome.anePrefillNs + outcome.draftNs + outcome.verifyNs) / 1e9
                NSLog(
                    "[BadAppleSpecANE] %d tok | %d rounds | proposed %d accepted %d (%.0f%%) | gpuPrefill %.0fms anePrefill %.0fms draft %.0fms verify %.0fms | drafting survived: %d",
                    outcome.emitted.count, outcome.rounds, outcome.proposed, outcome.accepted,
                    outcome.proposed > 0 ? 100.0 * Double(outcome.accepted) / Double(outcome.proposed) : 0,
                    Double(outcome.prefillNs) / 1e6, Double(outcome.anePrefillNs) / 1e6,
                    Double(outcome.draftNs) / 1e6,
                    Double(outcome.verifyNs) / 1e6, outcome.draftingSurvived)
                onComplete(GenerationResult(
                    text: fullText,
                    tokensPerSecond: elapsedS > 0 ? Float(Double(outcome.emitted.count) / elapsedS) : 0,
                    tokenCount: outcome.emitted.count,
                    tier: "main+ane-draft",
                    draftProposedTokens: outcome.proposed,
                    draftAcceptedTokens: outcome.accepted,
                    prefixCache: "ane:\(outcome.rounds)r",
                    promptTokenCount: promptIds.count
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

    /// Bound MLX's process-wide free-buffer cache. MLX defaults the cache
    /// limit to the memory limit, so freed Metal buffers from every model in
    /// the process (VLM, embedder, main) pile up until the machine swaps.
    public static func limitBufferCache(bytes: Int) {
        Memory.cacheLimit = max(0, bytes)
    }

    /// Drop all cached (free) Metal buffers. Live arrays are untouched.
    public static func releaseBufferCache() {
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
            // Preflight geometry: update(parameters:) assigns adapter weights
            // by name with no shape check, so a stale adapter (trained on a
            // model with different dims) injects cleanly and then traps the
            // first matmul. Reject before the weights touch the model: every
            // lora_a input dim must match the input dim of the layer it
            // targets by name.
            if let lora = context.model as? LoRAModel {
                let targetIns = lora.loraLayers
                    .flatMap { $0.namedModules() }
                    .compactMap { ($0.1 as? Linear)?.shape.1 }
                let adapterIns = adapter.parameters.flattened()
                    .filter { $0.0.hasSuffix("lora_a") }
                    .map { $0.1.dim(0) }
                let illegal = adapterIns.filter { !targetIns.contains($0) }
                guard illegal.isEmpty else {
                    throw InferenceError.adapterGeometryMismatch(
                        adapterIns: Set(adapterIns), targetIns: Set(targetIns))
                }
            }
            try adapter.load(into: context.model)
        }
    }

    public func unload() async {
        meshHost = nil
        meshTokenizer = nil
        prefixCacheLock.withLock {
            prefixCacheKey = ""
            prefixCacheBoundary = 0
            prefixSnapshot = nil
        }
        await state.setContainer(nil)
        Memory.clearCache()
    }

    public var memoryUsageGB: Float {
        Float(Memory.activeMemory) / Float(1024 * 1024 * 1024)
    }
}

// MARK: - Convenience

public extension BadAppleInference {
    /// Code-level default: Qwen2.5-Coder-7B 4-bit, used when
    /// `BADAPPLE_MAIN_MODEL` is unset. The shipped launchd plist pins
    /// `mlx-community/Qwen3-8B-4bit` as the live GPU brain; other models are
    /// switchable via `badapple model use` or `BADAPPLE_MAIN_MODEL`.
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

    /// Read `BADAPPLE_MAX_KV_SIZE` from the environment (default 8192).
    public static var envKVSize: Int {
        if let raw = ProcessInfo.processInfo.environment["BADAPPLE_MAX_KV_SIZE"],
           let val = Int(raw), val > 0 {
            return val
        }
        return 8192
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

    /// Scale draft depth to the measured acceptance rate. Open prose
    /// lands ~30-65% per-position — a single-token draft minimizes the
    /// verification window's wasted work — while structured output
    /// (lists, code, counting) runs ≥75% and profits from deeper chains.
    /// `acceptPct <= 0` means no signal yet; fall back to the env value.
    /// Env is the initial depth only — adaptation ranges 1...5.
    public static func adaptiveNumDraftTokens(acceptPct: Float, envDefault: Int) -> Int {
        // Measured on Qwen3-14B-4bit + Qwen3-0.6B-4bit (M-series, 16 GB):
        // prose at 63% acceptance ran 16.8 tok/s with n=1 but 12.7 with
        // n=2 — every extra draft slot costs a full draft step and widens
        // the verify window, so depth only pays once acceptance is high.
        if acceptPct <= 0 { return max(envDefault, 1) }
        if acceptPct < 80 { return 1 }
        if acceptPct < 92 { return 2 }
        return 4
    }
}
