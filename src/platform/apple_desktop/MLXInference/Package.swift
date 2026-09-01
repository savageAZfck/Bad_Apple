// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BadAppleMLX",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "BadAppleMLX", type: .dynamic, targets: ["BadAppleMLX"]),
        .executable(name: "BadAppleMLXSelfTest", targets: ["BadAppleMLXSelfTest"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "BadAppleMLX",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/BadAppleMLX"
        ),
        .executableTarget(
            name: "BadAppleMLXSelfTest",
            dependencies: ["BadAppleMLX"]
        ),
    ]
)
