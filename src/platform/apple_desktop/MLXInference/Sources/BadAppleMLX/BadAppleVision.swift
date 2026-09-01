import CoreImage
import Foundation
import MLXLMCommon
import MLXVLM

/// Swift-native local-image description using Qwen2-VL through MLXVLM.
public actor BadAppleVisionEngine {
    public struct Configuration: Sendable {
        public let modelId: String
        public let revision: String
        public let defaultPrompt: String
        public let maxTokens: Int

        public init(
            modelId: String = "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            revision: String = "01af461cdb9574acc09084a0ef94e216e142b085",
            defaultPrompt: String = "Describe this image clearly and concisely.",
            maxTokens: Int = 256
        ) {
            self.modelId = modelId
            self.revision = revision
            self.defaultPrompt = defaultPrompt
            self.maxTokens = maxTokens
        }
    }

    public enum VisionError: Error, LocalizedError {
        case invalidConfiguration(String)
        case modelNotLoaded
        case invalidImage(String)
        case invalidLocalModel(String)
        case modelLoadFailed(String)
        case generationFailed(String)

        public var errorDescription: String? {
            switch self {
            case .invalidConfiguration(let detail):
                return "The vision model configuration is invalid: \(detail)"
            case .modelNotLoaded:
                return "The vision model is not loaded yet. Load it before describing an image."
            case .invalidImage(let detail):
                return "The image could not be used: \(detail)"
            case .invalidLocalModel(let detail):
                return "The local vision model could not be used: \(detail)"
            case .modelLoadFailed(let detail):
                return "Could not load the vision model: \(detail)"
            case .generationFailed(let detail):
                return "Could not describe the image: \(detail)"
            }
        }
    }

    public static let defaultConfiguration = Configuration()
    public static let maximumGenerationTokens = 512

    public nonisolated let configuration: Configuration
    private var container: ModelContainer?

    public init(configuration: Configuration = BadAppleVisionEngine.defaultConfiguration) {
        self.configuration = configuration
    }

    public var ready: Bool { container != nil }

    /// Download (or use the Hugging Face cache for) Qwen2-VL and load it.
    public func loadModel() async throws {
        try validateConfiguration()
        do {
            let modelConfiguration = ModelConfiguration(
                id: configuration.modelId,
                revision: configuration.revision,
                defaultPrompt: configuration.defaultPrompt,
                extraEOSTokens: ["<|im_end|>"]
            )
            container = try await VLMModelFactory.shared.loadContainer(
                from: HuggingFaceDownloader(),
                using: TokenizersLoader(),
                configuration: modelConfiguration
            )
        } catch {
            container = nil
            throw VisionError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Load an already-downloaded Qwen2-VL MLX model directory.
    public func loadModel(from localDirectory: URL) async throws {
        try validateConfiguration()
        try Self.validateModelDirectory(localDirectory)
        do {
            container = try await VLMModelFactory.shared.loadContainer(
                from: localDirectory,
                using: TokenizersLoader()
            )
        } catch {
            container = nil
            throw VisionError.modelLoadFailed(error.localizedDescription)
        }
    }

    /// Validate and describe a local image, yielding decoded text as it is generated.
    ///
    /// The returned stream terminates after at most `maxTokens` tokens. A consumer
    /// cancelling iteration also cancels its forwarding task.
    public func describe(
        imageURL: URL,
        prompt: String? = nil,
        maxTokens: Int? = nil
    ) async throws -> AsyncThrowingStream<String, Error> {
        guard let container else { throw VisionError.modelNotLoaded }
        try Self.validateImage(imageURL)

        let requestedPrompt = prompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalPrompt = requestedPrompt.flatMap { $0.isEmpty ? nil : $0 }
            ?? configuration.defaultPrompt
        guard !finalPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VisionError.invalidConfiguration("The image description prompt cannot be empty.")
        }

        let tokenLimit = maxTokens ?? configuration.maxTokens
        guard (1...Self.maximumGenerationTokens).contains(tokenLimit) else {
            throw VisionError.invalidConfiguration(
                "maxTokens must be between 1 and \(Self.maximumGenerationTokens)."
            )
        }

        do {
            let userInput = UserInput(prompt: finalPrompt, images: [.url(imageURL)])
            let input = try await container.prepare(input: userInput)
            let upstream = try await container.generate(
                input: input,
                parameters: GenerateParameters(maxTokens: tokenLimit, temperature: 0)
            )

            return AsyncThrowingStream { continuation in
                let task = Task {
                    for await event in upstream {
                        if Task.isCancelled { break }
                        if case .chunk(let text) = event, !text.isEmpty {
                            continuation.yield(text)
                        }
                    }
                    continuation.finish()
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        } catch {
            throw VisionError.generationFailed(error.localizedDescription)
        }
    }

    private func validateConfiguration() throws {
        guard !configuration.modelId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VisionError.invalidConfiguration("modelId cannot be empty.")
        }
        guard !configuration.revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VisionError.invalidConfiguration("revision cannot be empty.")
        }
        guard !configuration.defaultPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VisionError.invalidConfiguration("defaultPrompt cannot be empty.")
        }
        guard (1...Self.maximumGenerationTokens).contains(configuration.maxTokens) else {
            throw VisionError.invalidConfiguration(
                "maxTokens must be between 1 and \(Self.maximumGenerationTokens)."
            )
        }
    }

    private static func validateModelDirectory(_ url: URL) throws {
        guard url.isFileURL else {
            throw VisionError.invalidLocalModel("Use a local file URL.")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw VisionError.invalidLocalModel("No directory exists at '\(url.path)'.")
        }
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw VisionError.invalidLocalModel("The directory is not readable.")
        }
        guard FileManager.default.fileExists(
            atPath: url.appendingPathComponent("config.json").path
        ) else {
            throw VisionError.invalidLocalModel("config.json is missing.")
        }
    }

    private static func validateImage(_ url: URL) throws {
        guard url.isFileURL else {
            throw VisionError.invalidImage("Use a local file URL; remote image URLs are not accepted.")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw VisionError.invalidImage("No file exists at '\(url.path)'.")
        }
        guard FileManager.default.isReadableFile(atPath: url.path) else {
            throw VisionError.invalidImage("The file is not readable.")
        }
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
              (values.fileSize ?? 0) > 0 else {
            throw VisionError.invalidImage("The file is empty.")
        }
        guard CIImage(contentsOf: url) != nil else {
            throw VisionError.invalidImage("The file is not a supported or decodable image.")
        }
    }
}
