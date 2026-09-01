// BadAppleTools — Swift translation of the Python tool router, policy engine,
// and tool executor for Bad Apple.
//
// This file provides the native Swift equivalent of badapple_mlx_tools.py and
// the tool-execution helpers in badapple_tools.py, adapted for the macOS
// menu-bar / desktop client.

import Foundation
import Dispatch

// MARK: - Regex Helpers

/// Replace all matches of a regex pattern in a string with a template.
private func regexReplace(
    _ pattern: String,
    in text: String,
    with template: String,
    options: NSRegularExpression.Options = []
) -> String {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
        return text
    }
    let range = NSRange(text.startIndex..., in: text)
    return regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: template)
}

/// Return the first match of a regex pattern, or nil if no match.
private func regexFirstMatch(
    _ pattern: String,
    in text: String,
    options: NSRegularExpression.Options = []
) -> NSTextCheckingResult? {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
        return nil
    }
    let range = NSRange(text.startIndex..., in: text)
    return regex.firstMatch(in: text, options: [], range: range)
}

/// Return all matches of a regex pattern.
private func regexAllMatches(
    _ pattern: String,
    in text: String,
    options: NSRegularExpression.Options = []
) -> [NSTextCheckingResult] {
    guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
        return []
    }
    let range = NSRange(text.startIndex..., in: text)
    return regex.matches(in: text, options: [], range: range)
}

/// Split a string on regex matches, keeping the text between delimiters.
private func regexSplit(
    _ pattern: String,
    in text: String,
    options: NSRegularExpression.Options = []
) -> [String] {
    let matches = regexAllMatches(pattern, in: text, options: options)
    if matches.isEmpty { return [text] }
    var result: [String] = []
    var lastEnd = text.startIndex
    for match in matches {
        if let r = Range(match.range, in: text) {
            if lastEnd < r.lowerBound {
                result.append(String(text[lastEnd..<r.lowerBound]))
            }
            lastEnd = r.upperBound
        }
    }
    if lastEnd < text.endIndex {
        result.append(String(text[lastEnd...]))
    }
    return result
}

// MARK: - Text Processing Helpers

/// Light cleanup for streaming model output.
/// Strips Qwen3 thinking blocks, special stop tokens, CJK artifacts, and IPA
/// characters that leak into English generation.
func polishText(_ text: String) -> String {
    var result = text

    // Strip Qwen3 thinking blocks and leaked stop tokens.
    result = regexReplace(#"^Thinking Process:.*?(?:Final Answer:|Answer:)\s*"#, in: result, with: "", options: [.dotMatchesLineSeparators, .caseInsensitive])
    result = regexReplace(#"\n?\s*<thinking>.*?\s*\n?"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"</s>|<\|endoftext\|>|</thinking>"#, in: result, with: "")

    // Normalize em-dash spacing and remove markdown asterisks.
    result = result.replacingOccurrences(of: "\u{2014} \u{2014}", with: "\u{2014}")
    result = result.replacingOccurrences(of: "*", with: "")

    // Replace CJK / fullwidth character runs with a space.
    result = regexReplace(#"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+"#, in: result, with: " ")

    // Replace stray IPA phonetic characters with ASCII approximations.
    let ipaMap: [Character: Character] = [
        "\u{028B}": "v", "\u{028C}": "v", "\u{0251}": "a", "\u{0252}": "o",
        "\u{025B}": "e", "\u{026A}": "i", "\u{028A}": "u", "\u{0254}": "o",
        "\u{0259}": "a", "\u{00E6}": "a",
    ]
    result = String(result.map { ipaMap[$0] ?? $0 })

    // Collapse runs of spaces/tabs and normalize em-dash spacing.
    result = regexReplace(#"[ \t]+"#, in: result, with: " ")
    result = regexReplace(#" ?\u{2014} ?"#, in: result, with: "\u{2014}")

    // Convert triple dots to ellipsis.
    result = result.replacingOccurrences(of: "...", with: "\u{2026}")

    // Remove space before sentence punctuation.
    result = regexReplace(#"\s+([.,!?;:])"#, in: result, with: "$1")

    // Ensure a space after sentence punctuation when the next token is a letter.
    result = regexReplace(#"([.!?\\u2026])([A-Za-z])"#, in: result, with: "$1 $2")

    // Rewrite "fr fr" / "frfr" to the full phrase for clear speech.
    result = regexReplace(#"\bfr fr\b"#, in: result, with: "for real for real", options: [.caseInsensitive])
    result = regexReplace(#"\bfrfr\b"#, in: result, with: "for real for real", options: [.caseInsensitive])

    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Post-process final model output: remove thinking tags, code fence artifacts,
/// normalize whitespace, collapse repeated sentences.
func postprocessOutput(_ text: String) -> String {
    var result = text

    // Strip Qwen3 thinking blocks.
    result = regexReplace(#"^Thinking Process:.*?(?:Final Answer:|Answer:)\s*"#, in: result, with: "", options: [.dotMatchesLineSeparators, .caseInsensitive])
    result = regexReplace(#"\n?\s*<thinking>.*?\s*\n?"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"</s>|<\|endoftext\|>|</thinking>"#, in: result, with: "")

    // Remove code fence artifacts (```lang ... ``` wrappers that leak into output).
    result = regexReplace(#"^```[a-zA-Z]*\n"#, in: result, with: "", options: [.anchorsMatchLines])
    result = regexReplace(#"\n```\s*$"#, in: result, with: "", options: [.anchorsMatchLines])
    result = result.replacingOccurrences(of: "```", with: "")

    // Normalize em-dash spacing.
    result = result.replacingOccurrences(of: "\u{2014} \u{2014}", with: "\u{2014}")

    // Replace CJK / fullwidth character runs with a space.
    result = regexReplace(#"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+"#, in: result, with: " ")

    // Replace stray IPA phonetic characters.
    let ipaMap: [Character: Character] = [
        "\u{028B}": "v", "\u{028C}": "v", "\u{0251}": "a", "\u{0252}": "o",
        "\u{025B}": "e", "\u{026A}": "i", "\u{028A}": "u", "\u{0254}": "o",
        "\u{0259}": "a", "\u{00E6}": "a",
    ]
    result = String(result.map { ipaMap[$0] ?? $0 })

    // Remove disallowed persona ticks and normalize endearments.
    result = regexReplace(#"\b[pP]+f+[tT]+\b"#, in: result, with: "")
    result = regexReplace(#"\bhon\b"#, in: result, with: "hun", options: [.caseInsensitive])

    // Collapse all whitespace runs to a single space.
    result = regexReplace(#"\s+"#, in: result, with: " ")
    result = result.trimmingCharacters(in: .whitespacesAndNewlines)

    // Clean up stray commas and punctuation spacing.
    result = regexReplace(#"\s*,\s*([.!?])"#, in: result, with: "$1")
    result = regexReplace(#"\s*,\s*,"#, in: result, with: ",")
    result = regexReplace(#"\s*,\s*\u{2014}"#, in: result, with: "\u{2014}")
    result = regexReplace(#"^,\s*"#, in: result, with: "")
    result = regexReplace(#"\s*,\s*$"#, in: result, with: "")
    result = regexReplace(#"\s+([.!?])"#, in: result, with: "$1")
    result = regexReplace(#"([.!?])([\u{2014}\-])"#, in: result, with: "$1 $2")

    // Collapse immediately repeated sentences.
    let sentences = regexSplit(#"(?<=[.!?\\u2026])\s+"#, in: result)
    var deduped: [String] = []
    let stripChars = CharacterSet(charactersIn: ".!?\u{2026}")
    for s in sentences {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }
        let low = trimmed.lowercased().trimmingCharacters(in: stripChars)
        if let last = deduped.last {
            let lastLow = last.lowercased().trimmingCharacters(in: stripChars)
            if low == lastLow { continue }
        }
        deduped.append(trimmed)
    }
    result = deduped.joined(separator: " ")

    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - BadAppleToolRouter

/// Routes user prompts to relevant tools, formats tool definitions for the
/// model prompt, and extracts tool-call blocks from model output.
final class BadAppleToolRouter: @unchecked Sendable {

    /// A single tool definition.
    struct BadAppleTool: Codable {
        let name: String
        let description: String
        let parameters: [Parameter]
        let requiresApproval: Bool

        /// A single named parameter for a tool.
        struct Parameter: Codable {
            let name: String
            let description: String
            let required: Bool
        }
    }

    // MARK: - Tool Registry

    /// Native tools exposed to the model.
    private let tools: [BadAppleTool] = [
        BadAppleTool(
            name: "get_current_time",
            description: "Get the current local date and time on the Mac.",
            parameters: [],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "read_file",
            description: "Read the text content of a local file. Only reads text files and stops at a size limit.",
            parameters: [
                .init(name: "path", description: "Absolute or tilde-expanded path to the file.", required: true),
                .init(name: "limit", description: "Max characters to return. Default 10000.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "list_directory",
            description: "List files and folders in a local directory. Defaults to the user's home directory.",
            parameters: [
                .init(name: "path", description: "Absolute or tilde-expanded path to the directory.", required: true),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "search_content",
            description: "Search for a text string inside files under a directory using grep. Returns matching lines with file paths.",
            parameters: [
                .init(name: "pattern", description: "The text pattern to search for.", required: true),
                .init(name: "path", description: "Directory to search. Defaults to the user's home directory.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "run_shell",
            description: "Run a read-only shell command from a safe allowlist (ls, cat, head, tail, find, grep, wc, file, pwd, mdfind, ps, df, du, echo, which). No redirection, pipes, or multiple commands.",
            parameters: [
                .init(name: "command", description: "The shell command to run. Must begin with an allowed command and contain no dangerous characters.", required: true),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "write_file",
            description: "Write text content to a local file. Create or overwrite.",
            parameters: [
                .init(name: "path", description: "Absolute or tilde-expanded path to the file to write.", required: true),
                .init(name: "content", description: "The text content to write.", required: true),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "screen_capture",
            description: "Capture the main Mac screen to a PNG and return the local file path.",
            parameters: [
                .init(name: "path", description: "Optional absolute path to save the screenshot. Defaults to a temp file.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "run_applescript",
            description: "Run a short AppleScript directly with /usr/bin/osascript. Shell commands and network download commands are rejected.",
            parameters: [
                .init(name: "script", description: "The AppleScript source to run.", required: true),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "list_shortcuts",
            description: "List the names of installed macOS Shortcuts.",
            parameters: [],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "run_shortcut",
            description: "Run a named macOS Shortcut. Shortcuts may mutate apps or data, so approval is required.",
            parameters: [
                .init(name: "name", description: "The exact name of the Shortcut to run.", required: true),
                .init(name: "input", description: "Optional text input for the Shortcut.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "index_documents",
            description: "Securely enumerate local text and code files under a jailed path and return an indexing summary.",
            parameters: [
                .init(name: "path", description: "A jailed directory or text file to enumerate.", required: true),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "search_notes",
            description: "Search text documents read-only within ~/Documents and the active workspace.",
            parameters: [
                .init(name: "query", description: "Text to search for in local notes and documents.", required: true),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "read_working_memory",
            description: "Read the assistant scratchpad at ~/.bad_apple/working_memory.txt.",
            parameters: [
                .init(name: "limit", description: "Maximum characters to return. Default 5000.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "write_working_memory",
            description: "Write or append to the assistant scratchpad at ~/.bad_apple/working_memory.txt.",
            parameters: [
                .init(name: "content", description: "The text to store.", required: true),
                .init(name: "mode", description: "replace (default) or append.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "clear_working_memory",
            description: "Clear the assistant scratchpad at ~/.bad_apple/working_memory.txt.",
            parameters: [],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "runtime_status",
            description: "Return native macOS, process, memory, uptime, and workspace status.",
            parameters: [],
            requiresApproval: false
        ),
    ]

    /// Names in the native registry, exposed for discovery and logic tests.
    func registeredToolNames() -> [String] {
        tools.map(\.name)
    }

    // MARK: - Keyword Maps

    /// Keywords that signal the user's prompt may benefit from tool use.
    private let toolKeywords: [String] = [
        // Time / date
        "what time", "current time", "time is it", "date and time", "today's date", "what day",
        // Files / directories
        "list files", "show files", "files in", "directory", "folder", "what's in",
        "read file", "read the file", "contents of", "show me the file",
        "write file", "save to file", "create a file", "append to file", "write a note",
        // Search
        "search for", "find file", "search content", "search in", "grep", "find text", "find in files",
        // Shell
        "run command", "run shell", "execute command", "shell command",
        // Screen
        "screenshot", "screen capture", "capture screen", "what's on my screen",
        // Math
        "calculate", "compute", "math", "how much", "how many",
        // Native automation and local knowledge
        "run applescript", "applescript", "run shortcut", "list shortcuts", "shortcut",
        "index documents", "index files", "search notes", "search my notes",
        "working memory", "scratchpad", "runtime status", "health status",
        // System
        "system", "process", "memory", "disk usage",
        // Code
        "code", "function", "compile", "build",
    ]

    /// Maps keyword groups to the tools they suggest.
    private let keywordToolMap: [(keywords: [String], toolNames: [String])] = [
        (["what time", "current time", "time is it", "date and time", "today's date", "what day"], ["get_current_time"]),
        (["list files", "show files", "files in", "directory", "folder", "what's in"], ["list_directory"]),
        (["read file", "read the file", "contents of", "show me the file"], ["read_file"]),
        (["write file", "save to file", "create a file", "append to file", "write a note"], ["write_file"]),
        (["run command", "run shell", "execute command", "shell command"], ["run_shell"]),
        (["search content", "search in", "grep", "find text", "find in files", "search for"], ["search_content"]),
        (["screenshot", "screen capture", "capture screen", "what's on my screen"], ["screen_capture"]),
        (["run applescript", "run script", "applescript"], ["run_applescript"]),
        (["list shortcuts"], ["list_shortcuts"]),
        (["run shortcut", "shortcut"], ["run_shortcut", "list_shortcuts"]),
        (["index documents", "index my", "index files"], ["index_documents"]),
        (["search my notes", "search notes", "what did I write"], ["search_notes"]),
        (["working memory", "scratchpad"], ["read_working_memory", "write_working_memory", "clear_working_memory"]),
        (["runtime status", "health status", "system status", "process", "memory"], ["runtime_status"]),
    ]

    // MARK: - Prompt Routing

    /// Return formatted tool definitions relevant to the prompt, or nil if no
    /// tools are needed.
    func toolsForPrompt(text: String) -> String? {
        let low = text.lowercased()
        var selectedNames = Set<String>()

        for (keywords, toolNames) in keywordToolMap {
            if keywords.contains(where: { low.contains($0) }) {
                selectedNames.formUnion(toolNames)
            }
        }

        guard !selectedNames.isEmpty else { return nil }

        let selected = tools.filter { selectedNames.contains($0.name) }
        guard !selected.isEmpty else { return nil }

        var lines: [String] = ["Available tools:"]
        for tool in selected {
            lines.append("")
            lines.append("[\(tool.name)] \u{2014} \(tool.description)")
            lines.append("  Parameters:")
            for param in tool.parameters {
                let req = param.required ? "required" : "optional"
                lines.append("    \(param.name) (\(req)): \(param.description)")
            }
            lines.append("  Requires approval: \(tool.requiresApproval ? "yes" : "no")")
        }
        return lines.joined(separator: "\n")
    }

    /// Heuristic: should the model be offered tools for this prompt?
    func shouldUseTools(prompt: String) -> Bool {
        let low = prompt.lowercased()
        return toolKeywords.contains { low.contains($0) }
    }

    // MARK: - Tool Call Extraction

    /// Extract tool calls from model output in the format:
    ///   `<tool name="read_file"><arg name="path">/etc/hosts</arg></tool>`
    ///
    /// Returns an array of (name, args) tuples. The args dictionary maps
    /// argument names to their string values.
    func extractToolCalls(text: String) -> [(name: String, args: [String: String])] {
        var calls: [(name: String, args: [String: String])] = []

        let toolMatches = regexAllMatches(
            #"<tool\s+name="([^"]*)">\s*(.*?)\s*</tool>"#,
            in: text,
            options: [.dotMatchesLineSeparators]
        )

        for toolMatch in toolMatches {
            guard toolMatch.numberOfRanges >= 3,
                  let nameRange = Range(toolMatch.range(at: 1), in: text),
                  let contentRange = Range(toolMatch.range(at: 2), in: text) else { continue }

            let name = String(text[nameRange])
            let content = String(text[contentRange])

            var args: [String: String] = [:]
            let argMatches = regexAllMatches(
                #"<arg\s+name="([^"]*)">\s*(.*?)\s*</arg>"#,
                in: content,
                options: [.dotMatchesLineSeparators]
            )
            for argMatch in argMatches {
                guard argMatch.numberOfRanges >= 3,
                      let argNameRange = Range(argMatch.range(at: 1), in: content),
                      let argValueRange = Range(argMatch.range(at: 2), in: content) else { continue }
                args[String(content[argNameRange])] = String(content[argValueRange])
            }

            calls.append((name: name, args: args))
        }

        let jsonMatches = regexAllMatches(
            #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#,
            in: text,
            options: [.dotMatchesLineSeparators]
        )
        for match in jsonMatches {
            guard match.numberOfRanges >= 2,
                  let jsonRange = Range(match.range(at: 1), in: text),
                  let data = String(text[jsonRange]).data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = object["name"] as? String else { continue }
            let rawArgs = object["arguments"] as? [String: Any] ?? [:]
            let args = rawArgs.mapValues { value in
                if let string = value as? String { return string }
                return String(describing: value)
            }
            calls.append((name: name, args: args))
        }

        let functionMatches = regexAllMatches(
            #"<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>"#,
            in: text,
            options: [.dotMatchesLineSeparators]
        )
        for match in functionMatches {
            guard match.numberOfRanges >= 3,
                  let nameRange = Range(match.range(at: 1), in: text),
                  let argsRange = Range(match.range(at: 2), in: text) else { continue }
            let name = String(text[nameRange])
            let raw = String(text[argsRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            let candidates = [raw, "{\(raw)}"]
            var args: [String: String] = [:]
            for candidate in candidates {
                guard let data = candidate.data(using: .utf8),
                      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                args = object.mapValues { value in
                    if let string = value as? String { return string }
                    return String(describing: value)
                }
                break
            }
            calls.append((name: name, args: args))
        }

        return calls
    }

    // MARK: - Multi-Step Detection

    /// Heuristic: does this prompt describe a multi-step task?
    func isMultiStep(text: String) -> Bool {
        let low = text.lowercased()
        let simpleChecks = [
            "and then", "and save", "and write", "and show",
            "and list", "and read", "and run",
            "plan", "step by step",
        ]
        if simpleChecks.contains(where: { low.contains($0) }) {
            return true
        }
        // Regex for "multi-step", "multistep", "multi-task", "multitask".
        if regexFirstMatch(#"\bmulti.?(?:step|task)\b"#, in: low) != nil {
            return true
        }
        return false
    }
}

// MARK: - BadApplePolicyEngine

/// Declarative policy engine that loads rules from a YAML file and decides
/// whether a tool call is approved, denied, or needs human approval.
final class BadApplePolicyEngine: @unchecked Sendable {

    /// The result of evaluating a tool call against policy.
    enum ApprovalDecision {
        case approved
        case denied
        case needsApproval
    }

    // MARK: - State

    private let lock = NSLock()
    private var _autopilot: Bool = false
    private var policyLoaded: Bool = false
    private var defaultRequireApproval: Bool = true
    private var defaultAllowed: Bool = true
    private var toolApproval: [String: Bool] = [:]
    private var toolAllowed: [String: Bool] = [:]

    /// Tools that always require approval regardless of policy file.
    private let hardcodedApprovalRequired: Set<String> = [
        "run_shell", "run_applescript", "run_shortcut", "write_file",
        "write_working_memory", "clear_working_memory", "index_documents",
        "screen_capture",
    ]

    /// Policy file path.
    private let policyPath = "/var/lib/bad_apple/policy.yaml"

    // MARK: - Init

    init() {
        loadPolicy()
    }

    // MARK: - Autopilot

    /// When true, destructive tools run without asking for approval.
    var autopilot: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _autopilot
        }
        set {
            lock.lock()
            _autopilot = newValue
            lock.unlock()
        }
    }

    // MARK: - Policy Checks

    /// Check whether a tool requires human approval.
    func requiresApproval(toolName: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        // Code-level safety requirements cannot be relaxed by a stale or
        // permissive policy file. Autopilot is handled separately by evaluate.
        if hardcodedApprovalRequired.contains(toolName) {
            return true
        }
        if let value = toolApproval[toolName] {
            return value
        }
        if policyLoaded {
            return defaultRequireApproval
        }
        return false
    }

    /// Check whether a tool is allowed at all by policy.
    func isAllowed(toolName: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        if let value = toolAllowed[toolName] {
            return value
        }
        if policyLoaded {
            return defaultAllowed
        }
        return true
    }

    /// Evaluate a tool call against the full policy and autopilot state.
    func evaluate(toolName: String, args: [String: String]) -> ApprovalDecision {
        if !isAllowed(toolName: toolName) {
            return .denied
        }
        if autopilot {
            return .approved
        }
        if requiresApproval(toolName: toolName) {
            return .needsApproval
        }
        return .approved
    }

    // MARK: - Policy Loading

    /// Load policy from the YAML file using simple line-based parsing.
    /// Parses `autopilot`, `defaults.require_approval`, `defaults.allowed`,
    /// and per-tool `require_approval` / `allowed` values.
    private func loadPolicy() {
        guard let content = try? String(contentsOfFile: policyPath, encoding: .utf8) else {
            return
        }

        var inDefaults = false
        var inTools = false
        var currentTool: String?

        for rawLine in content.components(separatedBy: "\n") {
            // Skip comments and blank lines.
            let stripped = rawLine.trimmingCharacters(in: .whitespaces)
            if stripped.isEmpty || stripped.hasPrefix("#") { continue }

            // Count leading spaces to determine nesting level.
            let leadingSpaces = rawLine.prefix(while: { $0 == " " }).count

            if leadingSpaces == 0 {
                // Top-level key.
                currentTool = nil
                inDefaults = false
                inTools = false

                if stripped == "defaults:" {
                    inDefaults = true
                } else if stripped == "tools:" {
                    inTools = true
                } else if stripped.contains(":") {
                    let parts = stripped.split(separator: ":", maxSplits: 1)
                    if parts.count == 2 {
                        let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
                        let value = String(parts[1]).trimmingCharacters(in: .whitespaces)
                        if key == "autopilot" {
                            lock.lock()
                            _autopilot = (value == "true")
                            lock.unlock()
                        }
                    }
                }
            } else if inDefaults {
                // Defaults section (indent 2).
                if stripped.contains(":") {
                    let parts = stripped.split(separator: ":", maxSplits: 1)
                    if parts.count == 2 {
                        let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
                        let value = String(parts[1]).trimmingCharacters(in: .whitespaces)
                        lock.lock()
                        if key == "require_approval" {
                            defaultRequireApproval = (value == "true")
                        } else if key == "allowed" {
                            defaultAllowed = (value == "true")
                        }
                        lock.unlock()
                    }
                }
            } else if inTools {
                if leadingSpaces == 2 && stripped.hasSuffix(":") {
                    // Tool name entry.
                    currentTool = String(stripped.dropLast())
                } else if let tool = currentTool, stripped.contains(":") {
                    let parts = stripped.split(separator: ":", maxSplits: 1)
                    if parts.count == 2 {
                        let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
                        let value = String(parts[1]).trimmingCharacters(in: .whitespaces)
                        lock.lock()
                        if key == "require_approval" {
                            toolApproval[tool] = (value == "true")
                        } else if key == "allowed" {
                            toolAllowed[tool] = (value == "true")
                        }
                        lock.unlock()
                    }
                }
            }
        }

        lock.lock()
        policyLoaded = true
        lock.unlock()
    }
}

// MARK: - BadAppleToolExecutor

/// Executes tool calls with filesystem jail protection and policy enforcement.
final class BadAppleToolExecutor: @unchecked Sendable {

    // MARK: - State

    private let lock = NSLock()
    private var _workspace: String?
    private let policyEngine: BadApplePolicyEngine?
    private let startedAt = Date()

    /// Optional workspace root. When set, paths within the workspace are
    /// allowed in addition to the home and temp directories.
    var workspace: String? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _workspace
        }
        set {
            lock.lock()
            _workspace = newValue
            lock.unlock()
        }
    }

    // MARK: - Init

    init(policyEngine: BadApplePolicyEngine? = nil) {
        self.policyEngine = policyEngine
    }

    // MARK: - Tool Execution

    /// Execute a tool by name with the given arguments.
    /// Returns the tool result as a string (or an error message).
    func executeTool(name: String, args: [String: String], approved: Bool = false) async -> String {
        // Check policy if a policy engine is attached.
        if !approved, let policy = policyEngine {
            let decision = policy.evaluate(toolName: name, args: args)
            switch decision {
            case .denied:
                return "Policy: tool '\(name)' is not allowed."
            case .needsApproval:
                return "Approval required before I can run \(name). Reply with 'approve' to proceed. (Set autopilot to skip these prompts.)"
            case .approved:
                break
            }
        }

        switch name {
        case "get_current_time":
            return getCurrentTime()
        case "read_file":
            return readFile(path: args["path"] ?? "")
        case "list_directory":
            return listDirectory(path: args["path"] ?? "")
        case "search_content":
            let pattern = args["pattern"] ?? args["query"] ?? ""
            let path = args["path"] ?? "~"
            return searchContent(pattern: pattern, path: path)
        case "run_shell":
            return runShell(command: args["command"] ?? "")
        case "write_file":
            return writeFile(path: args["path"] ?? "", content: args["content"] ?? "")
        case "screen_capture":
            return screenCapture(path: args["path"])
        case "run_applescript":
            return runAppleScript(script: args["script"] ?? "")
        case "list_shortcuts":
            return listShortcuts()
        case "run_shortcut":
            return runShortcut(name: args["name"] ?? "", input: args["input"])
        case "index_documents":
            return indexDocuments(path: args["path"] ?? "")
        case "search_notes":
            return searchNotes(query: args["query"] ?? "")
        case "read_working_memory":
            return readWorkingMemory(limit: parseLimit(args["limit"], defaultValue: 5_000, maximum: 50_000))
        case "write_working_memory":
            return writeWorkingMemory(content: args["content"] ?? "", mode: args["mode"] ?? "replace")
        case "clear_working_memory":
            return clearWorkingMemory()
        case "runtime_status":
            return runtimeStatus()
        default:
            return "Unknown tool: \(name)"
        }
    }

    // MARK: - Path Jail

    /// Resolve a path and verify it is within an allowed root (home, /tmp,
    /// /var/tmp, or the workspace). Rejects `..` path components and symlink
    /// escapes. Returns the resolved path or nil if the path is unsafe.
    func jailPath(_ path: String) -> String? {
        // Reject `..` as a path component to prevent traversal.
        let components = path.split(separator: "/").map { String($0) }
        if components.contains("..") {
            return nil
        }

        // Expand tilde to the home directory.
        let expanded: String
        if path.hasPrefix("~/") {
            expanded = NSHomeDirectory() + String(path.dropFirst(1))
        } else if path == "~" {
            expanded = NSHomeDirectory()
        } else {
            expanded = path
        }

        // Resolve symlinks for existing paths; for non-existing paths, resolve
        // the parent directory and re-append the filename.
        let url = URL(fileURLWithPath: expanded)
        let resolved: String
        if FileManager.default.fileExists(atPath: expanded) {
            resolved = url.resolvingSymlinksInPath().path
        } else {
            let parent = url.deletingLastPathComponent()
            let parentResolved = parent.resolvingSymlinksInPath().path
            resolved = parentResolved + "/" + url.lastPathComponent
        }

        // Normalize the resolved path (remove trailing slashes for comparison).
        let normalizedResolved = resolved.hasSuffix("/") && resolved != "/"
            ? String(resolved.dropLast())
            : resolved

        // Build the list of allowed roots.
        var allowedRoots: [String] = [NSHomeDirectory(), "/tmp", "/var/tmp"]
        if let ws = workspace, !ws.isEmpty {
            allowedRoots.append(ws)
        }

        for root in allowedRoots {
            let rootURL = URL(fileURLWithPath: root)
            let rootResolved: String
            if FileManager.default.fileExists(atPath: root) {
                rootResolved = rootURL.resolvingSymlinksInPath().path
            } else {
                rootResolved = rootURL.standardizedFileURL.path
            }
            let normalizedRoot = rootResolved.hasSuffix("/") && rootResolved != "/"
                ? String(rootResolved.dropLast())
                : rootResolved

            if normalizedResolved == normalizedRoot
                || normalizedResolved.hasPrefix(normalizedRoot + "/") {
                return resolved
            }
        }

        return nil
    }

    // MARK: - Individual Tool Implementations

    private let textFileExtensions: Set<String> = [
        "txt", "md", "markdown", "rst", "rtf", "csv", "tsv", "json", "jsonl",
        "yaml", "yml", "toml", "xml", "html", "htm", "swift", "py", "rs",
        "c", "h", "m", "mm", "cpp", "hpp", "js", "jsx", "ts", "tsx", "java",
        "kt", "go", "rb", "php", "sh", "zsh", "fish", "sql", "css", "scss",
    ]

    private var workingMemoryPath: String {
        NSHomeDirectory() + "/.bad_apple/working_memory.txt"
    }

    /// Return local time without invoking an external process.
    func getCurrentTime() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss Z"
        return formatter.string(from: Date())
    }

    /// Read the text content of a file, respecting the jail and a size limit.
    func readFile(path: String) -> String {
        guard let jailed = jailPath(path) else {
            return "Error: path '\(path)' is outside allowed roots"
        }

        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: jailed, isDirectory: &isDir) {
            return "Error: \(jailed) does not exist"
        }
        if isDir.boolValue {
            return "Error: \(jailed) is a directory, not a file"
        }

        guard let data = FileManager.default.contents(atPath: jailed) else {
            return "Error: could not read \(jailed)"
        }

        let text = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1)
            ?? ""

        let limit = 10_000
        if text.count > limit {
            return String(text.prefix(limit)) + "\n... (\(text.count) characters total)"
        }
        return text
    }

    /// List the contents of a directory (max 50 entries).
    func listDirectory(path: String) -> String {
        let resolvedPath: String
        if path.isEmpty || path == "~" {
            resolvedPath = NSHomeDirectory()
        } else {
            guard let jailed = jailPath(path) else {
                return "Error: path '\(path)' is outside allowed roots"
            }
            resolvedPath = jailed
        }

        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: resolvedPath, isDirectory: &isDir) {
            return "Error: \(resolvedPath) does not exist"
        }
        if !isDir.boolValue {
            return "Error: \(resolvedPath) is not a directory"
        }

        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: resolvedPath) else {
            return "Error: could not list \(resolvedPath)"
        }

        let sorted = entries.sorted()
        let limited = Array(sorted.prefix(50))
        return limited.isEmpty ? "(empty directory)" : limited.joined(separator: "\n")
    }

    /// Search for a text pattern inside files under a directory using grep.
    func searchContent(pattern: String, path: String) -> String {
        if pattern.isEmpty {
            return "Error: no search pattern provided"
        }

        let resolvedPath: String
        if path.isEmpty || path == "~" {
            resolvedPath = NSHomeDirectory()
        } else {
            guard let jailed = jailPath(path) else {
                return "Error: path '\(path)' is outside allowed roots"
            }
            resolvedPath = jailed
        }

        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: resolvedPath, isDirectory: &isDir) {
            return "Error: \(resolvedPath) does not exist"
        }
        if !isDir.boolValue {
            return "Error: \(resolvedPath) is not a directory"
        }

        let arguments = [
            "-r", "-n", "-i", "--max-count=1",
            "--binary-files=without-match",
            "--exclude-dir=.git", "--exclude-dir=target", "--exclude-dir=.build",
            "--exclude-dir=.venv", "--exclude-dir=node_modules", "--exclude-dir=Pods",
            "--", pattern, resolvedPath,
        ]

        let result = runProcess(launchPath: "/usr/bin/grep", arguments: arguments, timeout: 15)
        if result.exitCode != 0 && result.stdout.isEmpty {
            return "No matches found"
        }

        let lines = result.stdout
            .components(separatedBy: "\n")
            .filter { !$0.isEmpty }
        let limited = Array(lines.prefix(20))
        return limited.isEmpty ? "No matches found" : limited.joined(separator: "\n")
    }

    /// Run a shell command from a safe allowlist. Rejects dangerous characters
    /// and commands not in the allowlist.
    func runShell(command: String) -> String {
        if command.isEmpty {
            return "Error: no command"
        }

        // Reject dangerous shell metacharacters.
        let dangerousChars: Set<Character> = [
            ";", "|", "&", "$", "`", "\"", "'", "\n", "\r",
            "<", ">", "{", "}", "[", "]", "*", "?",
        ]
        if command.contains(where: { dangerousChars.contains($0) }) {
            return "Error: command contains dangerous characters or operators"
        }

        // Split on whitespace (safe since quotes are rejected above).
        let tokens = command.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        if tokens.isEmpty {
            return "Error: empty command"
        }

        let baseName = tokens[0]
        let allowedCommands: Set<String> = [
            "ls", "cat", "head", "tail", "find", "grep", "wc", "file",
            "pwd", "mdfind", "ps", "df", "du", "echo", "which", "whoami", "id",
        ]

        // If the user passed an absolute path, extract the base name and verify
        // it resolves to the trusted binary on PATH.
        let commandName: String
        if baseName.hasPrefix("/") {
            commandName = (baseName as NSString).lastPathComponent
            if !allowedCommands.contains(commandName) {
                return "Error: '\(commandName)' is not in the allowed command list"
            }
            guard let trusted = resolveCommand(commandName) else {
                return "Error: '\(commandName)' not found on PATH"
            }
            let resolvedInput = URL(fileURLWithPath: baseName).resolvingSymlinksInPath().path
            if resolvedInput != trusted {
                return "Error: '\(baseName)' does not resolve to the trusted '\(commandName)' on PATH"
            }
        } else {
            commandName = baseName
            if !allowedCommands.contains(commandName) {
                return "Error: '\(commandName)' is not in the allowed command list"
            }
        }

        guard let resolved = resolveCommand(commandName) else {
            return "Error: '\(commandName)' not found on PATH"
        }

        let restArgs = Array(tokens.dropFirst())
        let result = runProcess(launchPath: resolved, arguments: restArgs, timeout: 15)

        if result.exitCode != 0 {
            let err = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "Error (\(result.exitCode)): \(err.isEmpty ? "command failed" : err)"
        }

        let out = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if out.isEmpty {
            return "(no output)"
        }
        return String(out.prefix(5000))
    }

    /// Write text content to a file. The path must be within the jail.
    func writeFile(path: String, content: String) -> String {
        guard let jailed = jailPath(path) else {
            return "Error: path '\(path)' is outside allowed roots"
        }

        let url = URL(fileURLWithPath: jailed)

        // Ensure parent directory exists.
        let parent = url.deletingLastPathComponent()
        do {
            try FileManager.default.createDirectory(
                at: parent,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            return "Error: could not create parent directory: \(error.localizedDescription)"
        }

        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            return "Wrote \(jailed)"
        } catch {
            return "Error: could not write to \(jailed): \(error.localizedDescription)"
        }
    }

    /// Execute AppleScript directly. Arguments are passed to osascript without
    /// a shell, and script features that could bypass the command cage are denied.
    func runAppleScript(script: String) -> String {
        let source = script.trimmingCharacters(in: .whitespacesAndNewlines)
        if source.isEmpty { return "Error: no AppleScript provided" }
        if source.count > 10_000 || source.contains("\0") {
            return "Error: AppleScript is too large or contains invalid characters"
        }
        let lowered = source.lowercased()
        let denied = [
            "do shell script", "do script", "run script", "use framework",
            "current application's", "curl", "wget", "rm -rf",
        ]
        if let match = denied.first(where: { lowered.contains($0) }) {
            return "Error: AppleScript contains denied operation '\(match)'"
        }

        let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", source], timeout: 15)
        if result.exitCode != 0 {
            let error = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "Error running AppleScript: \(error.isEmpty ? "osascript failed" : error)"
        }
        let output = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return output.isEmpty ? "done" : String(output.prefix(10_000))
    }

    /// List Shortcuts using Apple's fixed command-line executable.
    func listShortcuts() -> String {
        let result = runProcess(launchPath: "/usr/bin/shortcuts", arguments: ["list"], timeout: 15)
        if result.exitCode != 0 {
            let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return "Error listing shortcuts: \(error.isEmpty ? "shortcuts failed" : error)"
        }
        let names = result.stdout.components(separatedBy: .newlines).filter { !$0.isEmpty }
        return names.isEmpty ? "No shortcuts found" : names.prefix(100).joined(separator: "\n")
    }

    /// Run an exact Shortcut name without shell interpolation.
    func runShortcut(name: String, input: String?) -> String {
        let shortcutName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if shortcutName.isEmpty || shortcutName.count > 255 || shortcutName.contains("\0") {
            return "Error: invalid shortcut name"
        }
        var arguments = ["run", shortcutName]
        var standardInput: Data?
        if let input, !input.isEmpty {
            guard input.utf8.count <= 100_000 else { return "Error: shortcut input is too large" }
            arguments.append(contentsOf: ["-i", "-"])
            standardInput = input.data(using: .utf8)
        }
        let result = runProcess(
            launchPath: "/usr/bin/shortcuts",
            arguments: arguments,
            timeout: 60,
            standardInput: standardInput
        )
        if result.exitCode != 0 {
            let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return "Error running shortcut: \(error.isEmpty ? "shortcuts failed" : error)"
        }
        let output = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return output.isEmpty ? "done" : String(output.prefix(20_000))
    }

    /// Enumerate supported text files under a jailed path and summarize them.
    func indexDocuments(path: String) -> String {
        guard !path.isEmpty, let jailed = jailPath(path) else {
            return "Error: path '\(path)' is outside allowed roots"
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: jailed, isDirectory: &isDirectory) else {
            return "Error: \(jailed) does not exist"
        }

        let files = secureTextFiles(at: jailed, maximum: 2_000)
        if files.isEmpty { return "No supported text files found under \(jailed)" }
        var totalBytes = 0
        for url in files {
            totalBytes += (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        let sample = files.prefix(20).map(\.path).joined(separator: "\n")
        let truncated = files.count == 2_000 ? " (enumeration limit reached)" : ""
        return "Indexed \(files.count) text files totaling \(totalBytes) bytes from \(jailed)\(truncated)\n\(sample)"
    }

    /// Read-only case-insensitive search limited to Documents and workspace.
    func searchNotes(query: String) -> String {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if needle.isEmpty { return "Error: no search query provided" }
        if needle.count > 500 || needle.contains("\0") { return "Error: invalid search query" }

        var roots: [String] = []
        let documents = NSHomeDirectory() + "/Documents"
        if let jailedDocuments = jailPath(documents), FileManager.default.fileExists(atPath: jailedDocuments) {
            roots.append(jailedDocuments)
        }
        if let workspace, let jailedWorkspace = jailPath(workspace), !roots.contains(jailedWorkspace),
           FileManager.default.fileExists(atPath: jailedWorkspace) {
            roots.append(jailedWorkspace)
        }
        if roots.isEmpty { return "No searchable Documents directory or workspace found" }

        var matches: [String] = []
        for root in roots {
            for url in secureTextFiles(at: root, maximum: 1_000) {
                guard matches.count < 20,
                      let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                      size <= 2_000_000,
                      let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
                for (offset, line) in text.components(separatedBy: .newlines).enumerated()
                    where line.localizedCaseInsensitiveContains(needle) {
                    let excerpt = String(line.trimmingCharacters(in: .whitespaces).prefix(500))
                    matches.append("\(url.path):\(offset + 1): \(excerpt)")
                    if matches.count == 20 { break }
                }
            }
            if matches.count == 20 { break }
        }
        return matches.isEmpty ? "No relevant notes found." : matches.joined(separator: "\n")
    }

    func readWorkingMemory(limit: Int) -> String {
        guard let jailed = jailPath(workingMemoryPath) else {
            return "Error: working memory path is outside allowed roots"
        }
        guard FileManager.default.fileExists(atPath: jailed) else { return "Working memory is empty." }
        guard let text = try? String(contentsOfFile: jailed, encoding: .utf8) else {
            return "Error: could not read working memory"
        }
        if text.isEmpty { return "Working memory is empty." }
        return text.count > limit ? String(text.prefix(limit)) + "\n... (truncated)" : text
    }

    func writeWorkingMemory(content: String, mode: String) -> String {
        guard content.utf8.count <= 1_000_000 else { return "Error: working memory content is too large" }
        guard mode == "replace" || mode == "append" else {
            return "Error: mode must be 'replace' or 'append'"
        }
        guard let jailed = jailPath(workingMemoryPath) else {
            return "Error: working memory path is outside allowed roots"
        }
        let url = URL(fileURLWithPath: jailed)
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            var output = content
            if mode == "append", FileManager.default.fileExists(atPath: jailed) {
                let existing = try String(contentsOf: url, encoding: .utf8)
                output = existing + (existing.isEmpty || content.isEmpty ? "" : "\n") + content
            }
            try output.write(to: url, atomically: true, encoding: .utf8)
            return "Working memory \(mode == "append" ? "updated" : "written") (\(output.count) characters)."
        } catch {
            return "Error writing working memory: \(error.localizedDescription)"
        }
    }

    func clearWorkingMemory() -> String {
        guard let jailed = jailPath(workingMemoryPath) else {
            return "Error: working memory path is outside allowed roots"
        }
        do {
            if FileManager.default.fileExists(atPath: jailed) {
                try FileManager.default.removeItem(atPath: jailed)
            }
            return "Working memory cleared."
        } catch {
            return "Error clearing working memory: \(error.localizedDescription)"
        }
    }

    /// Native process and host information, with no shell or Python dependency.
    func runtimeStatus() -> String {
        let info = ProcessInfo.processInfo
        let uptime = max(0, info.systemUptime)
        let workspaceValue = workspace ?? "(not set)"
        return [
            "status: ready",
            "process_id: \(info.processIdentifier)",
            "process_name: \(info.processName)",
            "host: \(info.hostName)",
            "operating_system: \(info.operatingSystemVersionString)",
            "processor_count: \(info.processorCount)",
            "active_processor_count: \(info.activeProcessorCount)",
            "physical_memory_bytes: \(info.physicalMemory)",
            "system_uptime_seconds: \(Int(uptime))",
            "process_uptime_seconds: \(Int(max(0, Date().timeIntervalSince(startedAt))))",
            "thermal_state: \(thermalStateDescription(info.thermalState))",
            "low_power_mode: \(info.isLowPowerModeEnabled)",
            "workspace: \(workspaceValue)",
        ].joined(separator: "\n")
    }

    // MARK: - Private Helpers

    /// Capture the screen to a PNG file using the `screencapture` command.
    private func screenCapture(path: String?) -> String {
        let outputPath = path ?? (NSTemporaryDirectory() + "badapple_screen.png")

        // Always jail the output path to prevent arbitrary file overwrite.
        if let requested = path {
            guard let jailed = jailPath(requested) else {
                return "Error: output path '\(requested)' is outside allowed roots"
            }
            // screencapture writes to the path we give it; use the jailed path.
            let result = runProcess(launchPath: "/usr/sbin/screencapture", arguments: ["-x", jailed], timeout: 30)
            if result.exitCode == 0 {
                return jailed
            }
            return "Error: screen capture failed: \(result.stderr)"
        }

        let result = runProcess(launchPath: "/usr/sbin/screencapture", arguments: ["-x", outputPath], timeout: 30)
        if result.exitCode == 0 {
            return outputPath
        }
        return "Error: screen capture failed: \(result.stderr)"
    }

    /// Return regular, non-symlink text files whose resolved paths remain jailed.
    private func secureTextFiles(at root: String, maximum: Int) -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory) else { return [] }

        if !isDirectory.boolValue {
            let url = URL(fileURLWithPath: root)
            guard textFileExtensions.contains(url.pathExtension.lowercased()),
                  let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]),
                  values.isRegularFile == true, values.isSymbolicLink != true,
                  jailPath(url.path) != nil else { return [] }
            return [url]
        }

        let keys: [URLResourceKey] = [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: root),
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { _, _ in true }
        ) else { return [] }

        var files: [URL] = []
        for case let url as URL in enumerator {
            if files.count >= maximum { break }
            guard textFileExtensions.contains(url.pathExtension.lowercased()),
                  let values = try? url.resourceValues(forKeys: Set(keys)),
                  values.isRegularFile == true, values.isSymbolicLink != true,
                  let jailed = jailPath(url.path), jailed == url.resolvingSymlinksInPath().path else { continue }
            files.append(URL(fileURLWithPath: jailed))
        }
        return files.sorted { $0.path < $1.path }
    }

    private func parseLimit(_ value: String?, defaultValue: Int, maximum: Int) -> Int {
        guard let value, let parsed = Int(value) else { return defaultValue }
        return min(max(parsed, 1), maximum)
    }

    private func thermalStateDescription(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Resolve a command name to its full path by checking standard PATH dirs.
    private func resolveCommand(_ name: String) -> String? {
        let searchPaths = ["/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin"]
        for dir in searchPaths {
            let candidate = "\(dir)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    /// Run a process with stdout/stderr capture and a timeout.
    /// Returns (stdout, stderr, exitCode).
    private func runProcess(
        launchPath: String,
        arguments: [String],
        timeout: TimeInterval,
        standardInput: Data? = nil
    ) -> (stdout: String, stderr: String, exitCode: Int) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        let stdinPipe = standardInput == nil ? nil : Pipe()
        process.standardInput = stdinPipe

        var stdoutData = Data()
        var stderrData = Data()

        // Read pipes in the background to prevent deadlock when the output
        // exceeds the pipe buffer size.
        let group = DispatchGroup()
        DispatchQueue.global().async(group: group) {
            stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        }
        DispatchQueue.global().async(group: group) {
            stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        }

        do {
            try process.run()
        } catch {
            stdinPipe?.fileHandleForWriting.closeFile()
            return ("", "Error: \(error.localizedDescription)", -1)
        }
        if let standardInput, let stdinPipe {
            stdinPipe.fileHandleForWriting.write(standardInput)
            stdinPipe.fileHandleForWriting.closeFile()
        }

        // Wait with timeout.
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }

        if process.isRunning {
            process.terminate()
            group.wait()
            return ("", "Error: command timed out after \(Int(timeout))s", -1)
        }

        // Wait for the background pipe readers to finish.
        group.wait()

        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""
        return (stdout, stderr, Int(process.terminationStatus))
    }
}
