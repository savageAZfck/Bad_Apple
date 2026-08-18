// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "BadAppleMenuBar",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "BadAppleMenuBar", targets: ["BadAppleMenuBar"]),
    ],
    dependencies: [
        .package(path: "../apple_bridge"),
    ],
    targets: [
        .executableTarget(
            name: "BadAppleMenuBar",
            dependencies: [
                .product(name: "BadAppleBridge", package: "apple_bridge"),
            ],
            path: ".",
            exclude: ["Package.swift", "build_bad_apple_menu_bar.sh"],
            swiftSettings: [
                .enableExperimentalFeature("AccessLevelOnImport"),
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Speech"),
                .linkedLibrary("dl"),
            ]
        ),
    ],
    swiftLanguageModes: [.v5]
)
