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

        let selfClosingCalls = router.extractToolCalls(
            text: #"I will save this. <remember_person name="Maya" relationship="sister" />"#
        )
        guard selfClosingCalls.count == 1,
              selfClosingCalls[0].name == "remember_person",
              selfClosingCalls[0].args["name"] == "Maya",
              selfClosingCalls[0].args["relationship"] == "sister" else {
            fail("self-closing tool tag parsing")
        }

        let singleQuoteCalls = router.extractToolCalls(
            text: #"<add_commitment title='Call Mom' due_at='tomorrow 9am'/>"#
        )
        guard singleQuoteCalls.count == 1,
              singleQuoteCalls[0].name == "add_commitment",
              singleQuoteCalls[0].args["title"] == "Call Mom",
              singleQuoteCalls[0].args["due_at"] == "tomorrow 9am" else {
            fail("single-quoted self-closing tool tag parsing")
        }

        let entityCalls = router.extractToolCalls(
            text: #"<remember_person name="Tom &amp; Jerry" notes="he said &quot;hi&quot;"/>"#
        )
        guard entityCalls.count == 1,
              entityCalls[0].args["name"] == "Tom & Jerry",
              entityCalls[0].args["notes"] == #"he said "hi""# else {
            fail("self-closing tag entity decoding: \(entityCalls)")
        }

        let ignoredCalls = router.extractToolCalls(
            text: #"<recorded_event name="x" /><remember_person/><remember_person name="unclosed">"#
        )
        guard ignoredCalls.isEmpty else {
            fail("unknown/malformed/zero-attribute tags must yield no calls: \(ignoredCalls)")
        }

        let dedupCalls = router.extractToolCalls(
            text: #"<remember_person name="Maya" relationship="sister" /> <tool_call>{"name":"remember_person","arguments":{"name":"Maya","relationship":"sister"}}</tool_call>"#
        )
        guard dedupCalls.count == 1,
              dedupCalls[0].name == "remember_person" else {
            fail("cross-format dedup: \(dedupCalls)")
        }

        let explicitCases: [(String, String, [String: String])] = [
            ("set attention mode to focus", "set_attention_mode", ["mode": "focus"]),
            ("set attention mode quiet", "set_attention_mode", ["mode": "quiet"]),
            ("switch to sleep mode", "set_attention_mode", ["mode": "sleep"]),
            ("remember person Maya as sister", "remember_person", ["name": "Maya", "relationship": "sister"]),
            ("remember person Ada Lovelace as Mentor", "remember_person", ["name": "Ada Lovelace", "relationship": "Mentor"]),
            ("remember person Maya", "remember_person", ["name": "Maya"]),
            ("remember person named Maya as sister", "remember_person", ["name": "Maya", "relationship": "sister"]),
            ("remember this person Bob as friend", "remember_person", ["name": "Bob", "relationship": "friend"]),
            ("forget person Maya.", "forget_person", ["person": "Maya"]),
            ("remove person Maya", "forget_person", ["person": "Maya"]),
            ("record contact Maya", "record_contact", ["person": "Maya"]),
            ("i talked to Maya", "record_contact", ["person": "Maya"]),
            ("i spoke to Maya", "record_contact", ["person": "Maya"]),
            ("follow up with Maya tomorrow", "remember_person", ["name": "Maya", "follow_up_at": "tomorrow"]),
            ("follow up with Dr Smith in 2 days", "remember_person", ["name": "Dr Smith", "follow_up_at": "in 2 days"]),
            ("new conversation about Work", "new_conversation_thread", ["title": "Work"]),
            ("new conversation thread Side Project", "new_conversation_thread", ["title": "Side Project"]),
            ("start a conversation about Health", "new_conversation_thread", ["title": "Health"]),
            ("switch conversation General", "switch_conversation_thread", ["thread": "General"]),
            ("switch conversation to General", "switch_conversation_thread", ["thread": "General"]),
            ("open conversation Work", "switch_conversation_thread", ["thread": "Work"]),
            ("continue conversation Work", "switch_conversation_thread", ["thread": "Work"]),
            ("close conversation Work", "close_conversation_thread", ["thread": "Work"]),
            ("finish conversation Work", "close_conversation_thread", ["thread": "Work"]),
            ("mark complete abc123", "update_commitment", ["id": "abc123", "status": "completed"]),
            ("mark commitment abc123 complete", "update_commitment", ["id": "abc123", "status": "completed"]),
            ("mark commitment abc123 completed", "update_commitment", ["id": "abc123", "status": "completed"]),
            ("dismiss commitment abc123", "update_commitment", ["id": "abc123", "status": "dismissed"]),
            ("remind me to call mom tomorrow 9am", "add_commitment", ["title": "call mom", "due_at": "tomorrow 9am"]),
            ("remind me to call mom", "add_commitment", ["title": "call mom"]),
            ("don't let me forget to pay rent tomorrow", "add_commitment", ["title": "pay rent", "due_at": "tomorrow"]),
            ("dont let me forget to pay rent", "add_commitment", ["title": "pay rent"]),
            ("remind me to call mom at tomorrow", "add_commitment", ["title": "call mom", "due_at": "tomorrow"]),
            ("Remind me to Call Mom tomorrow at 9am.", "add_commitment", ["title": "Call Mom", "due_at": "tomorrow at 9am"]),
            ("follow up with Maya tomorrow at 9am", "remember_person", ["name": "Maya", "follow_up_at": "tomorrow at 9am"]),
            ("keep track of this situation: Mom's recovery", "manage_life_thread", ["title": "Mom's recovery", "status": "active"]),
            ("track this situation work relocation", "manage_life_thread", ["title": "work relocation", "status": "active"]),
            ("i prefer tea", "remember_preference", ["key": "preference: tea", "value": "tea"]),
            ("remember that i prefer coffee", "remember_preference", ["key": "preference: coffee", "value": "coffee"]),
            ("remember that my name is Adam", "remember_preference", ["key": "my name is Adam", "value": "true"]),
            ("forget preference drink", "forget_preference", ["key": "drink"]),
            ("forget that preference drink", "forget_preference", ["key": "drink"]),
            ("stop remembering drink", "forget_preference", ["key": "drink"]),
        ]
        for (input, expectedName, expectedArgs) in explicitCases {
            guard let command = router.explicitHumanCommand(for: input) else {
                fail("explicitHumanCommand returned nil for '\(input)'")
            }
            guard command.name == expectedName else {
                fail("explicitHumanCommand '\(input)' -> \(command.name), expected \(expectedName)")
            }
            for (key, value) in expectedArgs {
                guard command.args[key] == value else {
                    fail("explicitHumanCommand '\(input)' arg \(key)=\(command.args[key] ?? "nil"), expected '\(value)'")
                }
            }
            guard command.args.count == expectedArgs.count else {
                fail("explicitHumanCommand '\(input)' extra args: \(command.args)")
            }
        }
        for vague in [
            "commitment", "next action", "my preference", "remember",
            "set attention mode sideways", "follow up with maya",
            "remember person", "what time is it", "tell me a joke",
        ] {
            guard router.explicitHumanCommand(for: vague) == nil else {
                fail("explicitHumanCommand should return nil for '\(vague)'")
            }
        }
        for phrase in [
            "remember person maya", "follow up with maya tomorrow", "forget person maya",
            "remind me to call mom", "human home", "conversation threads",
            "switch to quiet mode", "close conversation work",
            "keep track of this situation", "i prefer tea", "stop remembering drink",
        ] {
            guard router.requiresHumanToolExecution(phrase) else {
                fail("requiresHumanToolExecution should be true for '\(phrase)'")
            }
        }
        for prose in [
            "what is the weather like today", "tell me about quantum computing",
            "remember to water the plants", "what happened",
        ] {
            guard !router.requiresHumanToolExecution(prose) else {
                fail("requiresHumanToolExecution should be false for '\(prose)'")
            }
        }

        let expectedNativeTools: Set<String> = [
            "get_current_time", "run_applescript", "list_shortcuts", "run_shortcut",
            "index_documents", "search_notes", "read_working_memory",
            "write_working_memory", "clear_working_memory", "runtime_status",
            "human_home", "remember_preference", "forget_preference",
            "add_commitment", "update_commitment", "manage_life_thread",
            "set_attention_mode", "list_people", "remember_person",
            "record_contact", "forget_person", "conversation_threads",
            "new_conversation_thread", "switch_conversation_thread",
            "close_conversation_thread",
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
              router.toolsForPrompt(text: "switch to quiet mode")?.contains("[set_attention_mode]") == true,
              router.toolsForPrompt(text: "people to remember")?.contains("[list_people]") == true,
              router.toolsForPrompt(text: "who should i contact")?.contains("[list_people]") == true,
              router.toolsForPrompt(text: "remember person maya")?.contains("[remember_person]") == true,
              router.toolsForPrompt(text: "follow up with maya tomorrow")?.contains("[remember_person]") == true,
              router.toolsForPrompt(text: "follow up with maya tomorrow")?.contains("[record_contact]") != true,
              router.toolsForPrompt(text: "i talked to maya")?.contains("[record_contact]") == true,
              router.toolsForPrompt(text: "forget person maya")?.contains("[forget_person]") == true,
              router.toolsForPrompt(text: "list conversations")?.contains("[conversation_threads]") == true,
              router.toolsForPrompt(text: "new conversation about work")?.contains("[new_conversation_thread]") == true,
              router.toolsForPrompt(text: "switch conversation")?.contains("[switch_conversation_thread]") == true,
              router.toolsForPrompt(text: "close conversation")?.contains("[close_conversation_thread]") == true else {
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
        _ = try? humanLayer.ensureDefaultConversationThread()
        let rememberedPerson = await humanExecutor.executeTool(
            name: "remember_person",
            args: ["name": "Maya", "relationship": "sister", "follow_up_at": "tomorrow 9am"],
            approved: false
        )
        guard rememberedPerson.contains("Remembered Maya") else {
            fail("remember_person write")
        }
        let peopleList = await humanExecutor.executeTool(
            name: "list_people", args: [:], approved: false
        )
        guard peopleList.contains("Maya"), peopleList.contains("sister") else {
            fail("list_people read")
        }
        let contact = await humanExecutor.executeTool(
            name: "record_contact",
            args: ["person": "Maya", "notes": "caught up"],
            approved: false
        )
        guard contact.contains("Recorded contact with Maya") else {
            fail("record_contact write")
        }
        let badDate = await humanExecutor.executeTool(
            name: "remember_person",
            args: ["name": "X", "follow_up_at": "banana"],
            approved: false
        )
        guard badDate.hasPrefix("Error:") else {
            fail("remember_person invalid follow_up_at")
        }
        let newThread = await humanExecutor.executeTool(
            name: "new_conversation_thread",
            args: ["title": "Work"],
            approved: false
        )
        guard newThread.contains("now active") else {
            fail("new_conversation_thread write")
        }
        let threadsAfter = await humanExecutor.executeTool(
            name: "conversation_threads", args: [:], approved: false
        )
        guard threadsAfter.contains("Work"), threadsAfter.contains("*") else {
            fail("conversation_threads read")
        }
        let switched = await humanExecutor.executeTool(
            name: "switch_conversation_thread",
            args: ["thread": "general"],
            approved: false
        )
        guard switched.contains("General") else {
            fail("switch_conversation_thread")
        }
        let closed = await humanExecutor.executeTool(
            name: "close_conversation_thread",
            args: ["thread": "general"],
            approved: false
        )
        guard closed.contains("Closed conversation 'General'") else {
            fail("close_conversation_thread")
        }
        let forgotten = await humanExecutor.executeTool(
            name: "forget_person",
            args: ["person": "Maya"],
            approved: false
        )
        guard forgotten == "Forgot that person." else {
            fail("forget_person write")
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
        for (name, args) in [
            ("remember_person", ["name": "Zed"]),
            ("record_contact", ["person": "x"]),
            ("forget_person", ["person": "x"]),
            ("new_conversation_thread", ["title": "x"]),
            ("switch_conversation_thread", ["thread": "x"]),
            ("close_conversation_thread", ["thread": "x"]),
        ] {
            let out = await humanExecutor.executeTool(name: name, args: args, approved: false)
            guard out == "Private mode is on. I did not save that." else {
                fail("private mode blocks \(name)")
            }
        }
        let readablePeople = await humanExecutor.executeTool(
            name: "list_people", args: [:], approved: false
        )
        let readableThreads = await humanExecutor.executeTool(
            name: "conversation_threads", args: [:], approved: false
        )
        guard !readablePeople.hasPrefix("Private mode"),
              !readableThreads.hasPrefix("Private mode") else {
            fail("read tools stay readable in private mode")
        }

        let freshDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("badapple-native-fresh-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: freshDir) }
        let freshLayer = BadAppleHumanLayer(directory: freshDir)
        let freshExecutor = BadAppleToolExecutor(humanLayer: freshLayer)
        freshExecutor.humanPersistenceAllowed = { false }
        let privateThreads = await freshExecutor.executeTool(
            name: "conversation_threads", args: [:], approved: false
        )
        guard privateThreads == "No conversation threads yet.",
              freshLayer.snapshot().conversationThreads.isEmpty else {
            fail("conversation_threads creates nothing in private mode")
        }
        freshExecutor.humanPersistenceAllowed = { true }
        let firstList = await freshExecutor.executeTool(
            name: "conversation_threads", args: [:], approved: false
        )
        guard firstList.contains("General"), firstList.contains("*") else {
            fail("conversation_threads lists General on first use")
        }

        var probe = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        var menuBarSource: String?
        var engineSource: String?
        for _ in 0..<8 {
            let base = probe.appendingPathComponent("src/platform/apple_desktop")
            if let text = try? String(
                contentsOf: base.appendingPathComponent("BadAppleMenuBar.swift"),
                encoding: .utf8
            ) {
                menuBarSource = text
            }
            if let text = try? String(
                contentsOf: base.appendingPathComponent("BadAppleEngine.swift"),
                encoding: .utf8
            ) {
                engineSource = text
            }
            if menuBarSource != nil && engineSource != nil { break }
            probe.deleteLastPathComponent()
        }
        if let engineSource {
            let requiredHumanTools: Set<String> = [
                "human_home", "remember_preference", "forget_preference", "add_commitment",
                "update_commitment", "manage_life_thread", "set_attention_mode",
                "list_people", "remember_person", "record_contact", "forget_person",
                "conversation_threads", "new_conversation_thread",
                "switch_conversation_thread", "close_conversation_thread",
            ]
            guard let setStart = engineSource.range(of: "directHumanTools: Set<String>"),
                  let setEnd = engineSource.range(of: "]", range: setStart.upperBound..<engineSource.endIndex) else {
                fail("directHumanTools set missing in engine source")
            }
            let setBody = engineSource[setStart.lowerBound..<setEnd.upperBound]
            for name in requiredHumanTools {
                guard setBody.contains("\"\(name)\"") else {
                    fail("directHumanTools missing \(name)")
                }
            }
            guard engineSource.contains("tier: \"human_tool\"") else {
                fail("human_tool tier missing in engine source")
            }
            if let directReturn = engineSource.range(of: "tier: \"human_tool\""),
               let historyAppend = engineSource.range(
                   of: "role: \"user\", content: currentPrompt"
               ) {
                guard directReturn.lowerBound < historyAppend.lowerBound else {
                    fail("direct human tool return must precede synthetic history append")
                }
            } else {
                fail("could not locate direct return or history append in engine source")
            }
            var explicitPositions: [String.Index] = []
            var searchStart = engineSource.startIndex
            while let found = engineSource.range(
                of: "explicitHumanCommand(for: prompt)", range: searchStart..<engineSource.endIndex
            ) {
                explicitPositions.append(found.lowerBound)
                searchStart = found.upperBound
            }
            var introspectionPositions: [String.Index] = []
            searchStart = engineSource.startIndex
            while let found = engineSource.range(
                of: "wantsIntrospection(prompt)", range: searchStart..<engineSource.endIndex
            ) {
                introspectionPositions.append(found.lowerBound)
                searchStart = found.upperBound
            }
            guard explicitPositions.count == 2, introspectionPositions.count == 2,
                  explicitPositions[0] < introspectionPositions[0],
                  explicitPositions[1] < introspectionPositions[1] else {
                fail("explicitHumanCommand must run before wantsIntrospection in both paths")
            }
            guard engineSource.contains("\"via\": \"explicit_human_phrase\""),
                  engineSource.contains("\"tier\": \"human_command\""),
                  engineSource.contains("tier: \"human_parse_error\""),
                  engineSource.contains("couldn't safely execute that Human Home command"),
                  engineSource.contains("requiresHumanToolExecution(prompt)") else {
                fail("explicit human command wiring / fail-closed path missing")
            }
            guard let parseError = engineSource.range(of: "tier: \"human_parse_error\""),
                  let modelReturn = engineSource.range(
                      of: "return lastResult", range: parseError.upperBound..<engineSource.endIndex
                  ) else {
                fail("could not locate human_parse_error return or model fallthrough")
            }
            _ = modelReturn
            guard engineSource.contains(
                "\"approval\", \"human_tool\", \"human_command\", \"human_parse_error\""
            ) else {
                fail("cache exclusion must cover all three human tiers")
            }
        }
        if let menuBarSource {
            guard menuBarSource.contains("identifier: \"APPROVE_ACTION\", title: \"Approve\""),
                  menuBarSource.contains("options: [.authenticationRequired]"),
                  menuBarSource.contains("var persistenceAllowed"),
                  menuBarSource.contains("private var humanWritesAllowed"),
                  menuBarSource.contains("guard onSave?(name, style, carry) ?? false else { return }") else {
                fail("menu bar source checks")
            }
            if let gate = menuBarSource.range(
                of: "guard onSave?(name, style, carry) ?? false else { return }"
            ) {
                let rest = menuBarSource[gate.upperBound...]
                guard rest.contains("forKey: \"BadAppleRelationshipOnboarded\"") else {
                    fail("onboarded key must be set only after onSave succeeds")
                }
            }
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
