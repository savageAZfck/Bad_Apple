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

        do {
            try testEstimatedModelMemory()
        } catch {
            fail("model memory fixtures failed: \(error.localizedDescription)")
        }

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

    private static func testEstimatedModelMemory() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("BadAppleMemorySelfTest-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        let dimensions: [String: Any] = [
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 3,
            "intermediate_size": 32,
            "vocab_size": 128,
            "torch_dtype": "float16",
        ]
        func writeJSON(_ value: [String: Any], to url: URL) throws {
            try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]).write(to: url)
        }
        func sparseFile(_ url: URL, bytes: UInt64) throws {
            check(fm.createFile(atPath: url.path, contents: nil), "could not create sparse weight fixture")
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.truncate(atOffset: bytes)
        }
        func fixture(_ name: String, config: [String: Any]? = nil, bytes: UInt64 = 4096) throws -> URL {
            let directory = root.appendingPathComponent(name)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            try writeJSON(config ?? dimensions, to: directory.appendingPathComponent("config.json"))
            try sparseFile(directory.appendingPathComponent("model.safetensors"), bytes: bytes)
            return directory
        }
        func estimate(_ directory: URL, kv: Int = 8, prefill: Int = 4) throws -> UInt64 {
            try BadAppleInference.estimatedModelMemory(in: directory, kvSize: kv, prefillStepSize: prefill)
        }
        func reject(_ directory: URL, containing detail: String, kv: Int = 8, prefill: Int = 4) {
            do {
                _ = try estimate(directory, kv: kv, prefill: prefill)
                fail("invalid memory fixture unexpectedly accepted: \(directory.lastPathComponent)")
            } catch BadAppleInference.InferenceError.modelLoadFailed(let message) {
                check(message.contains("Cannot measure model memory:"), "missing memory estimation error context")
                check(message.contains(detail), "unexpected memory estimation error: \(message)")
            } catch {
                fail("unexpected memory estimation error type: \(error.localizedDescription)")
            }
        }

        let sevenB = try fixture("model-7B-4bit")
        let nineB = root.appendingPathComponent("model-9B-4bit")
        try fm.copyItem(at: sevenB, to: nineB)
        let baseline = try estimate(sevenB)
        let renamed = try estimate(nineB)
        check(baseline == 6656, "memory estimate did not equal payload + KV + workspace + logits")
        check(renamed == baseline, "memory estimate depends on 7B/9B model name rather than files")

        let larger = try fixture("larger-payload", bytes: 1_052_672)
        let largerEstimate = try estimate(larger)
        check(largerEstimate == baseline + 1_048_576, "sparse payload byte growth was not counted exactly")
        let doubledKV = try estimate(sevenB, kv: 16)
        check(doubledKV == baseline + 768, "KV limit did not scale only KV memory")
        let doubledPrefill = try estimate(sevenB, prefill: 8)
        check(doubledPrefill == baseline + 1280, "prefill limit did not scale only workspace memory")

        var float32 = dimensions
        float32["torch_dtype"] = "float32"
        let float32Estimate = try estimate(fixture("float32", config: float32))
        check(float32Estimate == baseline + 768, "float32 did not double only KV scalar bytes")
        var bfloat16 = dimensions
        bfloat16["torch_dtype"] = "bfloat16"
        let bfloat16Estimate = try estimate(fixture("bfloat16", config: bfloat16))
        check(bfloat16Estimate == baseline, "bfloat16 should use two-byte KV scalars")

        var explicitHead = dimensions
        explicitHead["head_dim"] = 8
        let explicitEstimate = try estimate(fixture("explicit-head", config: explicitHead))
        check(explicitEstimate == baseline + 768, "explicit head_dim was not used")
        var fallbackHeads = dimensions
        fallbackHeads.removeValue(forKey: "num_key_value_heads")
        let fallbackEstimate = try estimate(fixture("fallback-heads", config: fallbackHeads))
        check(fallbackEstimate == baseline + 768, "KV heads did not default to attention heads")

        var nested = dimensions
        nested["hidden_size"] = 0
        nested["torch_dtype"] = "float32"
        nested["text_config"] = dimensions
        let nestedEstimate = try estimate(fixture("nested-config", config: nested))
        check(nestedEstimate == baseline, "nested text_config dimensions and dtype did not override root")
        var nestedText = dimensions
        nestedText.removeValue(forKey: "torch_dtype")
        nested["text_config"] = nestedText
        let nestedFallback = try estimate(fixture("nested-dtype-fallback", config: nested))
        check(nestedFallback == baseline + 768, "nested config did not inherit root dtype")

        var partialNested = dimensions
        partialNested["text_config"] = ["hidden_size": 16]
        let partialEstimate = try estimate(fixture("partial-nested", config: partialNested))
        check(partialEstimate == baseline, "partial text_config did not inherit root dimensions")
        var aliases = dimensions
        for (key, alias) in [("hidden_size", "n_embd"), ("num_attention_heads", "n_head"), ("num_hidden_layers", "n_layer"), ("intermediate_size", "n_inner")] {
            aliases[alias] = aliases.removeValue(forKey: key)
        }
        let aliasEstimate = try estimate(fixture("aliased-dimensions", config: aliases))
        check(aliasEstimate == baseline, "dimension aliases changed the estimate")

        let linked = try fixture("linked-weights")
        let linkedWeight = linked.appendingPathComponent("model.safetensors")
        try fm.removeItem(at: linkedWeight)
        let blob = root.appendingPathComponent("cached-weight-blob")
        try sparseFile(blob, bytes: 1_052_672)
        try fm.createSymbolicLink(at: linkedWeight, withDestinationURL: blob)
        let linkedEstimate = try estimate(linked)
        check(linkedEstimate == largerEstimate, "symlink estimate used link size instead of target payload size")

        let indexed = try fixture("indexed")
        let indexURL = indexed.appendingPathComponent("model.safetensors.index.json")
        try writeJSON(["weight_map": ["layer.a": "model.safetensors", "layer.b": "second.safetensors"]], to: indexURL)
        reject(indexed, containing: "missing shards")
        try sparseFile(indexed.appendingPathComponent("second.safetensors"), bytes: 8192)
        let indexedEstimate = try estimate(indexed)
        check(indexedEstimate == baseline + 8192, "complete indexed shards were not summed")
        try writeJSON(["weight_map": ["layer.a": "model.safetensors", "layer.b": "model.safetensors", "layer.c": "second.safetensors"]], to: indexURL)
        let duplicateMapEstimate = try estimate(indexed)
        check(duplicateMapEstimate == indexedEstimate, "weight index counted the same shard twice")
        try writeJSON(["weight_map": [String: String]()], to: indexURL)
        reject(indexed, containing: "invalid weight index")

        try writeJSON(["weight_map": ["layer.a": "model.safetensors"]], to: indexURL)
        let allFilesEstimate = try estimate(indexed)
        check(allFilesEstimate == indexedEstimate, "estimate must cover all safetensors files loaded by MLX")
        let nestedDirectory = indexed.appendingPathComponent("weights")
        try fm.createDirectory(at: nestedDirectory, withIntermediateDirectories: true)
        try sparseFile(nestedDirectory.appendingPathComponent("nested.safetensors"), bytes: 4096)
        try writeJSON(["weight_map": ["layer.a": "model.safetensors", "layer.b": "weights/nested.safetensors"]], to: indexURL)
        let recursiveEstimate = try estimate(indexed)
        check(recursiveEstimate == indexedEstimate + 4096, "nested weight shards were omitted")
        try writeJSON(["weight_map": ["layer.a": "../outside.safetensors"]], to: indexURL)
        reject(indexed, containing: "missing shards")

        let missingWeights = try fixture("missing-weights")
        try fm.removeItem(at: missingWeights.appendingPathComponent("model.safetensors"))
        reject(missingWeights, containing: "no cached safetensors weights")
        let emptyWeights = try fixture("empty-weights", bytes: 0)
        reject(emptyWeights, containing: "missing or empty weight shard")

        for key in ["hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers", "intermediate_size", "vocab_size"] {
            for (index, value) in ([0, -1, 1.5, "16", NSNull()] as [Any]).enumerated() {
                var invalid = dimensions
                invalid[key] = value
                let directory = try fixture("invalid-\(key)-\(index)", config: invalid)
                reject(directory, containing: "missing or invalid \(key)")
            }
        }
        for key in ["hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size", "vocab_size"] {
            var missing = dimensions
            missing.removeValue(forKey: key)
            reject(try fixture("missing-\(key)", config: missing), containing: "missing or invalid \(key)")
        }
        var invalidAttention = dimensions
        invalidAttention["hidden_size"] = 1
        reject(try fixture("invalid-attention", config: invalidAttention), containing: "head_dim is required")
        for limit in [0, -1] {
            reject(sevenB, containing: "invalid context limits", kv: limit)
            reject(sevenB, containing: "invalid context limits", prefill: limit)
        }
        for key in ["num_hidden_layers", "intermediate_size", "vocab_size"] {
            var overflow = dimensions
            overflow[key] = UInt64(1) << 62
            reject(try fixture("overflow-\(key)", config: overflow), containing: "size overflow")
        }
        var widthOverflow = dimensions
        widthOverflow["hidden_size"] = UInt64(1) << 63
        widthOverflow["head_dim"] = 4
        widthOverflow["intermediate_size"] = UInt64(1) << 62
        reject(try fixture("overflow-width-sum", config: widthOverflow), containing: "size overflow")
        var totalOverflow = dimensions
        totalOverflow["intermediate_size"] = UInt64(1) << 60
        totalOverflow["vocab_size"] = UInt64(1) << 61
        reject(try fixture("overflow-total-sum", config: totalOverflow), containing: "size overflow", prefill: 1)
        reject(sevenB, containing: "size overflow", kv: Int.max)
        reject(sevenB, containing: "size overflow", prefill: Int.max)
        print("BadAppleMLX model memory estimation tests passed")
    }

    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() { fail(message) }
    }

    private static func fail(_ message: String) -> Never {
        fputs("\(message)\n", stderr)
        exit(1)
    }
}
