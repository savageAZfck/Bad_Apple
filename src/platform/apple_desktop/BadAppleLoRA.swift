// BadAppleLoRA — native LoRA trainer/generator for Bad Apple.
//
// This is a top-level executable (`@main`). It restores the four `lora_*`
// tools that were lost when the Python MLX server was purged: the policy
// entries and docs survived, but the executor was never ported. Training
// runs in a separate process so the resident engine stays responsive and
// the adapter optimizer state never shares memory with inference.
//
//   badapple-lora train    --model <id|dir> --data <dir> --out <dir> [opts]
//   badapple-lora generate --model <id|dir> --adapter <dir> --prompt "..."
//
// Output format on disk matches the mlx-lm convention the original tools
// produced: `<out>/adapters.safetensors` + `<out>/adapter_config.json`.

import Foundation
import BadAppleMLX
import HuggingFace
import MLX
import MLXLLM
import MLXLMCommon
import MLXOptimizers
import Tokenizers

private let USAGE = """
  badapple-lora — native LoRA trainer/generator

    train    --model <id|dir> --data <dir> --out <dir>
             [--iters N] [--batch N] [--rank N] [--scale F]
             [--lr F] [--num-layers N] [--steps-per-report N]
             [--steps-per-eval N]
    generate --model <id|dir> --adapter <dir> --prompt "..."
             [--max-tokens N] [--temperature F]
  """

private struct Args {
    var model = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    var data = ""
    var out = ""
    var adapter = ""
    var prompt = ""
    var iters = 100
    var batch = 1
    var rank = 8
    var scale: Float = 10.0
    var lr: Float = 1e-4
    var numLayers = 16
    var stepsPerReport = 10
    var stepsPerEval = 100
    var maxTokens = 256
    var temperature: Float = 0.6

    static func parse() -> (String, Args)? {
        let argv = CommandLine.arguments
        guard argv.count >= 2, argv[1] == "train" || argv[1] == "generate" else {
            return nil
        }
        var a = Args()
        var i = 2
        func value(_ i: inout Int) -> String? {
            i += 1
            return i < argv.count ? argv[i] : nil
        }
        while i < argv.count {
            switch argv[i] {
            case "--model": if let v = value(&i) { a.model = v }
            case "--data": if let v = value(&i) { a.data = v }
            case "--out": if let v = value(&i) { a.out = v }
            case "--adapter": if let v = value(&i) { a.adapter = v }
            case "--prompt": if let v = value(&i) { a.prompt = v }
            case "--iters": if let v = value(&i).flatMap(Int.init) { a.iters = v }
            case "--batch": if let v = value(&i).flatMap(Int.init) { a.batch = v }
            case "--rank": if let v = value(&i).flatMap(Int.init) { a.rank = v }
            case "--scale": if let v = value(&i).flatMap(Float.init) { a.scale = v }
            case "--lr": if let v = value(&i).flatMap(Float.init) { a.lr = v }
            case "--num-layers": if let v = value(&i).flatMap(Int.init) { a.numLayers = v }
            case "--steps-per-report": if let v = value(&i).flatMap(Int.init) { a.stepsPerReport = v }
            case "--steps-per-eval": if let v = value(&i).flatMap(Int.init) { a.stepsPerEval = v }
            case "--max-tokens": if let v = value(&i).flatMap(Int.init) { a.maxTokens = v }
            case "--temperature": if let v = value(&i).flatMap(Float.init) { a.temperature = v }
            default: break
            }
            i += 1
        }
        return (argv[1], a)
    }
}

/// Resolve `--model` as a local directory when it looks like a path,
/// otherwise treat it as a HuggingFace repo id and resolve through the
/// same downloader the engine uses (cached weights only need the network
/// once).
private func loadContainer(model: String) async throws -> ModelContainer {
    if model.hasPrefix("/") || model.hasPrefix("~") || model.hasPrefix(".") {
        let dir = URL(fileURLWithPath: (model as NSString).expandingTildeInPath)
        return try await LLMModelFactory.shared.loadContainer(
            from: dir, using: TokenizersLoader())
    }
    let configuration = ModelConfiguration(id: model, revision: "main")
    let resolved = try await resolve(
        configuration: configuration,
        from: HuggingFaceDownloader(client: HubClient.default),
        useLatest: false,
        progressHandler: { _ in }
    )
    return try await LLMModelFactory.shared.loadContainer(
        from: resolved.modelDirectory, using: TokenizersLoader())
}

/// The original datasets used `{"messages":[{role,content},...]}` lines
/// (mlx-lm chat format). `loadLoRAData` only understands `{"text":...}`,
/// so normalize legacy lines into the Qwen chat template here rather than
/// stranding the data that already exists under lora_data/.
private func loadDataset(directory: URL, name: String) throws -> [String] {
    if let rows = try? loadLoRAData(directory: directory, name: name), !rows.isEmpty {
        return rows
    }
    struct Msg: Codable { let role: String; let content: String }
    struct Line: Codable { let messages: [Msg] }
    for ext in ["jsonl"] {
        let url = directory.appending(component: "\(name).\(ext)")
        guard FileManager.default.fileExists(atPath: url.path()) else { continue }
        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = text.components(separatedBy: .newlines)
            .filter { $0.first == "{" }
            .compactMap { try? JSONDecoder().decode(Line.self, from: $0.data(using: .utf8)!) }
            .map { line in
                line.messages.map {
                    "<|im_start|>\($0.role)\n\($0.content)<|im_end|>\n"
                }.joined()
            }
            .filter { !$0.isEmpty }
        if !rows.isEmpty { return rows }
    }
    throw NSError(domain: "BadAppleLoRA", code: 3, userInfo: [
        NSLocalizedDescriptionKey: "no dataset file '\(name)' in \(directory.path())",
    ])
}

private func runTrain(_ a: Args) async throws {
    guard !a.data.isEmpty, !a.out.isEmpty else {
        throw NSError(domain: "BadAppleLoRA", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "train requires --data <dir> and --out <dir>",
        ])
    }
    let dataDir = URL(fileURLWithPath: a.data)
    let outDir = URL(fileURLWithPath: a.out)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    fputs("[lora] loading model \(a.model)\n", stderr)
    let container = try await loadContainer(model: a.model)

    let trainSet = try loadDataset(directory: dataDir, name: "train")
    let validSet = (try? loadDataset(directory: dataDir, name: "valid")) ?? trainSet
    fputs("[lora] dataset: \(trainSet.count) train / \(validSet.count) valid rows\n", stderr)

    let loraConfig = LoRAConfiguration(
        numLayers: a.numLayers,
        fineTuneType: .lora,
        loraParameters: .init(rank: a.rank, scale: a.scale)
    )
    let params = LoRATrain.Parameters(
        batchSize: a.batch,
        iterations: a.iters,
        stepsPerReport: a.stepsPerReport,
        stepsPerEval: a.stepsPerEval,
        validationBatches: 5,
        saveEvery: a.iters,
        adapterURL: outDir.appending(component: "adapters.safetensors")
    )

    try await container.perform { context in
        let containerAdapter = try LoRAContainer.from(
            model: context.model, configuration: loraConfig)
        try containerAdapter.load(into: context.model)

        let optimizer = AdamW(learningRate: a.lr)
        var lastReport = ""
        try LoRATrain.train(
            model: context.model,
            train: trainSet,
            validate: validSet,
            optimizer: optimizer,
            tokenizer: context.tokenizer,
            parameters: params
        ) { progress in
            lastReport = progress.description
            fputs("[lora] \(progress)\n", stderr)
            return .more
        }
        try LoRATrain.saveLoRAWeights(
            model: context.model,
            url: outDir.appending(component: "adapters.safetensors"))
        if !lastReport.isEmpty { print("[lora] final: \(lastReport)") }
    }

    // Write adapter_config.json in the same shape the Python trainer used,
    // so existing tooling and the P2P adapter format stay compatible.
    let config: [String: Any] = [
        "fine_tune_type": "lora",
        "num_layers": a.numLayers,
        "lora_parameters": ["rank": a.rank, "scale": a.scale, "dropout": 0.0],
        "iters": a.iters,
        "batch_size": a.batch,
        "learning_rate": a.lr,
        "model": a.model,
        "data": a.data,
    ]
    let configData = try JSONSerialization.data(
        withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try configData.write(to: outDir.appending(component: "adapter_config.json"))
    print("[lora] adapter saved to \(outDir.path())")
}

private func runGenerate(_ a: Args) async throws {
    guard !a.adapter.isEmpty, !a.prompt.isEmpty else {
        throw NSError(domain: "BadAppleLoRA", code: 2, userInfo: [
            NSLocalizedDescriptionKey: "generate requires --adapter <dir> and --prompt <text>",
        ])
    }
    let adapterDir = URL(fileURLWithPath: a.adapter)
    fputs("[lora] loading model \(a.model)\n", stderr)
    let container = try await loadContainer(model: a.model)

    // Inject the adapter into the loaded model, then generate through the
    // container's own path — LoRAContainer mutates the module in place, so
    // the layers stay applied after perform returns.
    let tokenIds = try await container.perform { context -> [Int] in
        let containerAdapter = try LoRAContainer.from(directory: adapterDir)
        try containerAdapter.load(into: context.model)
        return try context.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": a.prompt]],
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
    }

    let params = GenerateParameters(
        maxTokens: a.maxTokens,
        temperature: a.temperature
    )
    let stream = try await container.generate(
        input: LMInput(tokens: MLXArray(tokenIds)),
        parameters: params
    )
    for await event in stream {
        if case .chunk(let chunk) = event {
            FileHandle.standardOutput.write(Data(chunk.utf8))
        }
    }
    FileHandle.standardOutput.write(Data("\n".utf8))
}

@main
struct BadAppleLoRATool {
    static func main() async {
        guard let (sub, args) = Args.parse() else {
            fputs(USAGE, stderr)
            exit(2)
        }
        do {
            switch sub {
            case "train": try await runTrain(args)
            case "generate": try await runGenerate(args)
            default: break
            }
        } catch {
            fputs("[lora] error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
