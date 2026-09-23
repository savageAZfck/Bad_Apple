import Foundation

final class BadAppleEngine {
    static let shared = BadAppleEngine()
    var killed = false
    var modelId: String { "" }
    func curiousAutopilotLevel() -> String { "off" }
    func triggerCuriousAutopilot(reason: String) {}
    func updateCuriousAutopilotLoop() {}
    func generateForSelfImprovement(prompt: String, maxTokens: Int = 500) async -> String { "" }
}

extension NSAttributedString {
    struct DocumentType: RawRepresentable, Equatable {
        let rawValue: String
        static let rtf = DocumentType(rawValue: "rtf")
    }
    enum DocumentReadingOptionKey: String {
        case documentType
    }
    convenience init(data: Data, options: [DocumentReadingOptionKey: Any], documentAttributes: Any?) throws {
        self.init(string: String(data: data, encoding: .utf8) ?? "")
    }
}

@main
enum BadAppleNativeLogicTests {
    static func main() async {
        let router = BadAppleToolRouter()
        let jsonCalls = router.extractToolCalls(
            text: #"<tool_call>{"name":"list_directory","arguments":{"path":"/tmp"}}</tool_call>"#
        )
        guard jsonCalls.count == 1,
              jsonCalls[0].name == "list_directory",
              jsonCalls[0].args["path"] == "/tmp" else {
            fail("JSON tool-call parsing")
        }

        let qwenCalls = router.extractToolCalls(
            text: #"<tool_call><function=read_file>{"path":"/tmp/test"}</function></tool_call>"#
        )
        guard qwenCalls.count == 1,
              qwenCalls[0].name == "read_file",
              qwenCalls[0].args["path"] == "/tmp/test" else {
            fail("Qwen tool-call parsing")
        }

        let xmlCalls = router.extractToolCalls(
            text: #"<tool name="search_notes"><arg name="query">native tools</arg></tool>"#
        )
        guard xmlCalls.count == 1,
              xmlCalls[0].name == "search_notes",
              xmlCalls[0].args["query"] == "native tools" else {
            fail("XML tool-call parsing")
        }

        let expectedNativeTools: Set<String> = [
            "get_current_time", "run_applescript", "list_shortcuts", "run_shortcut",
            "index_documents", "search_notes", "read_working_memory",
            "write_working_memory", "clear_working_memory", "runtime_status",
            "human_home", "remember_preference", "forget_preference",
            "add_commitment", "update_commitment", "manage_life_thread",
            "set_attention_mode",
        ]
        let registeredTools = Set(router.registeredToolNames())
        guard expectedNativeTools.isSubset(of: registeredTools) else {
            fail("native tool registry presence")
        }
        guard router.toolsForPrompt(text: "show runtime status")?.contains("[runtime_status]") == true,
              router.toolsForPrompt(text: "list shortcuts")?.contains("[list_shortcuts]") == true,
              router.toolsForPrompt(text: "search my notes")?.contains("[search_notes]") == true,
              router.toolsForPrompt(text: "refactor this code and run tests")?.contains("[write_file]") == true,
              router.toolsForPrompt(text: "refactor this code and run tests")?.contains("[run_shell]") == true,
              router.toolsForPrompt(text: "show my human home")?.contains("[human_home]") == true,
              router.toolsForPrompt(text: "what am i forgetting")?.contains("[human_home]") == true,
              router.toolsForPrompt(text: "remember that i prefer tea")?.contains("[remember_preference]") == true,
              router.toolsForPrompt(text: "remind me to call mom")?.contains("[add_commitment]") == true,
              router.toolsForPrompt(text: "keep track of this situation")?.contains("[manage_life_thread]") == true,
              router.toolsForPrompt(text: "switch to quiet mode")?.contains("[set_attention_mode]") == true else {
            fail("native tool keyword routing")
        }

        let policy = BadApplePolicyEngine()
        let allowedWritePath = NSHomeDirectory() + "/.bad_apple/notes/test.txt"
        guard case .needsApproval = policy.evaluate(
            toolName: "write_file",
            args: ["path": allowedWritePath]
        ) else {
            fail("write_file approval policy")
        }
        let destructiveTools = [
            "run_shell", "run_applescript", "run_shortcut", "write_file",
            "write_working_memory", "clear_working_memory", "index_documents",
            "screen_capture",
        ]
        let sampleArgs: [String: [String: String]] = [
            "run_shell": ["command": "ls /tmp"],
            "run_applescript": ["script": "tell app \"Finder\" to activate"],
            "run_shortcut": ["name": "Test"],
            "write_file": ["path": allowedWritePath],
        ]
        for toolName in destructiveTools {
            guard policy.requiresApproval(toolName: toolName),
                  case .needsApproval = policy.evaluate(
                      toolName: toolName, args: sampleArgs[toolName] ?? [:]
                  ) else {
                fail("\(toolName) destructive approval policy")
            }
        }
        policy.autopilot = true
        guard case .approved = policy.evaluate(
            toolName: "write_file",
            args: ["path": allowedWritePath]
        ) else {
            fail("autopilot approval policy")
        }

        let executor = BadAppleToolExecutor(policyEngine: policy)
        let currentTime = await executor.executeTool(
            name: "get_current_time",
            args: [:],
            approved: false
        )
        guard !currentTime.isEmpty, !currentTime.hasPrefix("Unknown tool") else {
            fail("approved executeTool API and current time execution")
        }
        guard executor.jailPath("/etc/passwd") == nil else {
            fail("path jail escape rejection")
        }
        guard let temporaryDirectory = executor.jailPath("/tmp"),
              temporaryDirectory == "/tmp" || temporaryDirectory == "/private/tmp" else {
            fail("path jail temporary directory")
        }

        let humanDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("badapple-native-human-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: humanDir) }
        let humanLayer = BadAppleHumanLayer(directory: humanDir)
        let humanExecutor = BadAppleToolExecutor(humanLayer: humanLayer)
        let emptyHome = await humanExecutor.executeTool(
            name: "human_home", args: [:], approved: false
        )
        guard emptyHome.contains("Human Home is clear") else {
            fail("human_home empty read")
        }
        let remembered = await humanExecutor.executeTool(
            name: "remember_preference",
            args: ["key": "drink", "value": "tea"],
            approved: false
        )
        guard remembered.contains("Remembered drink: tea") else {
            fail("remember_preference write")
        }
        let homeAfter = await humanExecutor.executeTool(
            name: "human_home", args: [:], approved: false
        )
        guard homeAfter.contains("drink: tea") else {
            fail("human_home reads remembered preference")
        }
        humanExecutor.humanPersistenceAllowed = { false }
        let blocked = await humanExecutor.executeTool(
            name: "remember_preference",
            args: ["key": "other", "value": "x"],
            approved: false
        )
        guard blocked == "Private mode is on. I did not save that." else {
            fail("private mode blocks human-layer writes")
        }

        let firewall = BadAppleOutputFirewall()
        guard firewall.check("token sk-secret") == "token [Output firewall: blocked]secret" else {
            fail("output firewall")
        }

        let cachePath = "/tmp/badapple-native-cache-test.json"
        try? FileManager.default.removeItem(atPath: cachePath)
        let cache = BadAppleSemanticCache(cachePath: cachePath)
        await cache.store(
            prompt: "how do I build this project",
            response: "use the build script",
            persona: "default"
        )
        let cached = await cache.lookup(
            prompt: "how do I build this project",
            persona: "default"
        )
        guard cached == "use the build script" else {
            fail("semantic cache lookup")
        }
        try? FileManager.default.removeItem(atPath: cachePath)

        print("Bad Apple native logic tests passed")
    }

    private static func fail(_ name: String) -> Never {
        fputs("FAIL: \(name)\n", stderr)
        exit(1)
    }
}
