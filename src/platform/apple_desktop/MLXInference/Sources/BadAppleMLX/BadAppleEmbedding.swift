import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// Swift-native sentence embeddings backed by MLXEmbedders.
///
/// The actor owns the loaded model container and serializes model loading and
/// inference. Returned vectors are L2-normalized by MLXEmbedders' pooling layer.
public actor BadAppleEmbeddingEngine {
    public struct Configuration: Sendable {
        public let modelId: String
        public let revision: String
        public let maxInputTokens: Int

        public init(
            modelId: String = "BAAI/bge-small-en-v1.5",
            revision: String = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            maxInputTokens: Int = 512
        ) {
            self.modelId = modelId
            self.revision = revision
            self.maxInputTokens = maxInputTokens
        }
    }

    public enum EmbeddingError: Error, LocalizedError {
        case invalidConfiguration(String)
        case modelNotLoaded
        case invalidLocalModel(String)
        case emptyText
        case modelLoadFailed(String)
        case embeddingFailed(String)
        case emptyEmbedding

        public var errorDescription: String? {
            switch self {
            case .invalidConfiguration(let detail):
                return "The embedding model configuration is invalid: \(detail)"
            case .modelNotLoaded:
                return "The embedding model is not loaded yet. Load it before requesting an embedding."
            case .invalidLocalModel(let detail):
                return "The local embedding model could not be used: \(detail)"
            case .emptyText:
                return "Text to embed cannot be empty."
            case .modelLoadFailed(let detail):
                return "Could not load the embedding model: \(detail)"
            case .embeddingFailed(let detail):
                return "Could not create the embedding: \(detail)"
            case .emptyEmbedding:
                return "The embedding model returned an empty vector."
            }
        }
    }

    public static let defaultConfiguration = Configuration()
    public static let maximumInputTokens = 512

    public nonisolated let configuration: Configuration
    private var container: EmbedderModelContainer?

    public init(configuration: Configuration = BadAppleEmbeddingEngine.defaultConfiguration) {
        self.configuration = configuration
    }

    public var ready: Bool { container != nil }

    /// Download (or use the Hugging Face cache for) the configured BGE model and load it.
    public func loadModel() async throws {
        try validateConfiguration()
        do {
            let modelConfiguration = ModelConfiguration(
                id: configuration.modelId,
                revision: configuration.revision
            )
            container = try await EmbedderModelFactory.shared.loadContainer(
                from: HuggingFaceDownloader(),
                using: TokenizersLoader(),
                configuration: modelConfiguration
            )
        } catch {
            container = nil
            throw EmbeddingError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Load an already-downloaded MLX/Hugging Face model directory.
    public func loadModel(from localDirectory: URL) async throws {
        try validateConfiguration()
        try Self.validateModelDirectory(localDirectory)
        do {
            container = try await EmbedderModelFactory.shared.loadContainer(
                from: localDirectory,
                using: TokenizersLoader()
            )
        } catch {
            container = nil
            throw EmbeddingError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Embed one string as a normalized vector.
    public func embed(_ text: String) async throws -> [Float] {
        guard let container else { throw EmbeddingError.modelNotLoaded }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EmbeddingError.emptyText
        }

        do {
            let tokenLimit = min(configuration.maxInputTokens, Self.maximumInputTokens)
            let vector: [Float] = await container.perform { context in
                var tokens = context.tokenizer.encode(text: text, addSpecialTokens: true)
                if tokens.count > tokenLimit {
                    tokens.removeSubrange(tokenLimit...)
                }

                let input = MLXArray(tokens).reshaped(1, tokens.count)
                let mask = MLXArray.ones(like: input)
                let tokenTypes = MLXArray.zeros(like: input)
                let output = context.model(
                    input,
                    positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask
                )
                let pooled = context.pooling(
                    output,
                    mask: mask,
                    normalize: true,
                    applyLayerNorm: true
                )
                pooled.eval()
                return pooled.reshaped(-1).asArray(Float.self)
            }
            guard !vector.isEmpty else { throw EmbeddingError.emptyEmbedding }
            return vector
        } catch let error as EmbeddingError {
            throw error
        } catch {
            throw EmbeddingError.embeddingFailed(error.localizedDescription)
        }
    }

    private func validateConfiguration() throws {
        guard !configuration.modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EmbeddingError.invalidConfiguration("modelId cannot be empty.")
        }
        guard !configuration.revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EmbeddingError.invalidConfiguration("revision cannot be empty.")
        }
        guard (1...Self.maximumInputTokens).contains(configuration.maxInputTokens) else {
            throw EmbeddingError.invalidConfiguration(
                "maxInputTokens must be between 1 and \(Self.maximumInputTokens)."
            )
        }
    }

    private static func validateModelDirectory(_ url: URL) throws {
        guard url.isFileURL else {
            throw EmbeddingError.invalidLocalModel("Use a local file URL.")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw EmbeddingError.invalidLocalModel("No directory exists at '\(url.path)'.")
        }
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw EmbeddingError.invalidLocalModel("The directory is not readable.")
        }
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("config.json").path
        ) else {
            throw EmbeddingError.invalidLocalModel("config.json is missing.")
        }
    }
}
