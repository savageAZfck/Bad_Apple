import BadAppleMLX
import Foundation

@main
struct BadAppleMLXSelfTest {
    static func main() async {
        let inference = BadAppleInference.defaultConfig
        check(
            inference.modelId == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            "unexpected inference model id"
        )
        check(
            inference.revision == "main",
            "unexpected inference model revision"
        )
        check(inference.maxTokens == 2048, "unexpected inference token limit")
        check(
            BadAppleInference.InferenceError.modelNotLoaded.errorDescription?
                .contains("not loaded yet") == true,
            "missing friendly inference error"
        )

        let embeddingConfig = BadAppleEmbeddingEngine.defaultConfiguration
        check(embeddingConfig.modelId == "BAAI/bge-small-en-v1.5", "unexpected embedding model id")
        check(
            embeddingConfig.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            "unexpected embedding model revision"
        )
        check(embeddingConfig.maxInputTokens == 512, "unexpected embedding token limit")
        check(
            BadAppleEmbeddingEngine.EmbeddingError.emptyText.errorDescription?
                .contains("cannot be empty") == true,
            "missing friendly embedding error"
        )

        let embeddingEngine = BadAppleEmbeddingEngine()
        let embeddingReady = await embeddingEngine.ready
        check(!embeddingReady, "embedding engine loaded weights during initialization")
        do {
            _ = try await embeddingEngine.embed("self-test")
            fail("embedding without weights unexpectedly succeeded")
        } catch BadAppleEmbeddingEngine.EmbeddingError.modelNotLoaded {
            // Expected: validates the public error surface without loading weights.
        } catch {
            fail("unexpected embedding error: \(error.localizedDescription)")
        }

        let invalidEmbedding = BadAppleEmbeddingEngine(
            configuration: .init(modelId: "", maxInputTokens: 0)
        )
        do {
            try await invalidEmbedding.loadModel()
            fail("invalid embedding configuration unexpectedly loaded")
        } catch BadAppleEmbeddingEngine.EmbeddingError.invalidConfiguration {
            // Expected before any download or weight loading.
        } catch {
            fail("unexpected embedding configuration error: \(error.localizedDescription)")
        }

        let visionConfig = BadAppleVisionEngine.defaultConfiguration
        check(
            visionConfig.modelId == "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "unexpected vision model id"
        )
        check(
            visionConfig.revision == "01af461cdb9574acc09084a0ef94e216e142b085",
            "unexpected vision model revision"
        )
        check(visionConfig.maxTokens == 256, "unexpected vision token limit")
        check(
            BadAppleVisionEngine.VisionError.invalidImage("test").errorDescription?
                .contains("image could not be used") == true,
            "missing friendly vision error"
        )

        let visionEngine = BadAppleVisionEngine()
        let visionReady = await visionEngine.ready
        check(!visionReady, "vision engine loaded weights during initialization")
        do {
            _ = try await visionEngine.describe(
                imageURL: URL(fileURLWithPath: "/definitely/missing/self-test.png")
            )
            fail("vision generation without weights unexpectedly succeeded")
        } catch BadAppleVisionEngine.VisionError.modelNotLoaded {
            // Expected: validates the public error surface without loading weights.
        } catch {
            fail("unexpected vision error: \(error.localizedDescription)")
        }

        let invalidVision = BadAppleVisionEngine(
            configuration: .init(modelId: "", defaultPrompt: "", maxTokens: 0)
        )
        do {
            try await invalidVision.loadModel()
            fail("invalid vision configuration unexpectedly loaded")
        } catch BadAppleVisionEngine.VisionError.invalidConfiguration {
            // Expected before any download or weight loading.
        } catch {
            fail("unexpected vision configuration error: \(error.localizedDescription)")
        }

        if ProcessInfo.processInfo.environment["BADAPPLE_TEST_EMBEDDING"] == "1" {
            let engine = BadAppleEmbeddingEngine()
            do {
                try await engine.loadModel()
                let vector = try await engine.embed("Bad Apple native embedding probe")
                check(!vector.isEmpty, "embedding probe returned no values")
                let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
                check(abs(norm - 1) < 0.01, "embedding probe was not normalized")
                print("BadAppleMLX embedding probe passed (\(vector.count) dimensions)")
            } catch {
                fail("embedding probe failed: \(error.localizedDescription)")
            }
        }

        if let imagePath = ProcessInfo.processInfo.environment["BADAPPLE_TEST_VISION_IMAGE"] {
            let engine = BadAppleVisionEngine()
            do {
                try await engine.loadModel()
                let stream = try await engine.describe(
                    imageURL: URL(fileURLWithPath: imagePath),
                    prompt: "Describe the image in one short sentence.",
                    maxTokens: 64
                )
                var response = ""
                for try await chunk in stream { response += chunk }
                check(!response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "vision probe returned no text")
                print("BadAppleMLX vision probe passed")
            } catch {
                fail("vision probe failed: \(error.localizedDescription)")
            }
        }

        print("BadAppleMLX self-test passed")
    }

    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() { fail(message) }
    }

    private static func fail(_ message: String) -> Never {
        fputs("\(message)\n", stderr)
        exit(1)
    }
}
