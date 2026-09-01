import Foundation

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
        ]
        let registeredTools = Set(router.registeredToolNames())
        guard expectedNativeTools.isSubset(of: registeredTools) else {
            fail("native tool registry presence")
        }
        guard router.toolsForPrompt(text: "show runtime status")?.contains("[runtime_status]") == true,
              router.toolsForPrompt(text: "list shortcuts")?.contains("[list_shortcuts]") == true,
              router.toolsForPrompt(text: "search my notes")?.contains("[search_notes]") == true else {
            fail("native tool keyword routing")
        }

        let policy = BadApplePolicyEngine()
        guard case .needsApproval = policy.evaluate(
            toolName: "write_file",
            args: ["path": "/tmp/test"]
        ) else {
            fail("write_file approval policy")
        }
        let destructiveTools = [
            "run_shell", "run_applescript", "run_shortcut", "write_file",
            "write_working_memory", "clear_working_memory", "index_documents",
            "screen_capture",
        ]
        for toolName in destructiveTools {
            guard policy.requiresApproval(toolName: toolName),
                  case .needsApproval = policy.evaluate(toolName: toolName, args: [:]) else {
                fail("\(toolName) destructive approval policy")
            }
        }
        policy.autopilot = true
        guard case .approved = policy.evaluate(
            toolName: "write_file",
            args: ["path": "/tmp/test"]
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
