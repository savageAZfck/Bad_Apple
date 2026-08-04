// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "FireflyMenuBar",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "FireflyMenuBar", targets: ["FireflyMenuBar"]),
    ],
    targets: [
        .executableTarget(
            name: "FireflyMenuBar",
            path: ".",
            exclude: ["Package.swift", "build_firefly_menu_bar.sh"],
            swiftSettings: [
                .enableExperimentalFeature("AccessLevelOnImport"),
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedLibrary("dl"),
            ]
        ),
    ]
)
