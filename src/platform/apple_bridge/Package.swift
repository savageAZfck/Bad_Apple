// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FireflySiriBridge",
    platforms: [.macOS(.v15)],
    products: [
        .library(
            name: "FireflySiriBridge",
            type: .dynamic,
            targets: ["FireflySiriBridge"]
        ),
    ],
    targets: [
        .target(
            name: "FireflySiriBridge",
            path: ".",
            exclude: ["Package.swift", "build_apple_bridge.sh"],
            swiftSettings: [
                .enableExperimentalFeature("AccessLevelOnImport"),
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("FoundationModels", .when(platforms: [.macOS])),
            ]
        ),
    ]
)
