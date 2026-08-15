// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "BadAppleBridge",
    platforms: [.macOS(.v26)],
    products: [
        .library(
            name: "BadAppleBridge",
            type: .dynamic,
            targets: ["BadAppleBridge"]
        ),
    ],
    targets: [
        .target(
            name: "BadAppleBridge",
            path: ".",
            exclude: [
                "Package.swift",
                "build_apple_bridge.sh",
                "install_daemon.sh",
                "com.badapple.substrate.plist",
            ],
            swiftSettings: [
                .enableExperimentalFeature("AccessLevelOnImport"),
            ],
            linkerSettings: [
                .linkedFramework("AppIntents", .when(platforms: [.macOS])),
                .linkedFramework("CryptoKit", .when(platforms: [.macOS])),
                .linkedFramework("Foundation"),
                .linkedFramework("FoundationModels", .when(platforms: [.macOS])),
                .linkedFramework("Security", .when(platforms: [.macOS])),
                .unsafeFlags(["-Xlinker", "-undefined", "-Xlinker", "dynamic_lookup"]),
            ]
        ),
    ],
    swiftLanguageModes: [.v5]
)
