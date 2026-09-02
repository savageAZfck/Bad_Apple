import Foundation

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fputs("FAIL: \(message)\n", stderr)
        exit(1)
    }
}

@main
private enum BadAppleIdentityTests {
    static func main() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("badapple-identity-\(UUID().uuidString)", isDirectory: true)
        let promptFile = root.appendingPathComponent("prompt.txt")
        let personasFile = root.appendingPathComponent("personas.json")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try? "Use concise drill style.".write(to: promptFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: root) }

        let manager = BadApplePersonaManager(
            dataDirectory: root,
            promptFile: promptFile,
            personasFile: personasFile
        )
        let systemPrompt = manager.getSystemPrompt()
        require(
            systemPrompt.contains("local AI operating system layer for macOS"),
            "Swift system prompt must identify Bad Apple as an OS"
        )
        require(
            systemPrompt.contains("Qwen, MLX") && systemPrompt.contains("not merely a text LLM"),
            "Swift system prompt must separate the model from the OS"
        )
        require(
            systemPrompt.contains("Use concise drill style."),
            "Swift system prompt must retain persona style"
        )

        let voicePrompt = manager.getSystemPrompt(voiceMode: true)
        require(
            voicePrompt.contains("local AI operating system layer for macOS"),
            "Swift voice prompt must identify Bad Apple as an OS"
        )
        require(
            voicePrompt.contains("not merely a text LLM"),
            "Swift voice prompt must reject the wrapper identity"
        )

        print("BadApple identity tests passed")
    }
}
