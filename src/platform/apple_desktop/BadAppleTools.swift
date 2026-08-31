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

    /// The six core tools exposed to the model.
    private let tools: [BadAppleTool] = [
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
    ]

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
        // System
        "system", "process", "memory", "disk usage",
        // Code
        "code", "function", "compile", "build",
    ]

    /// Maps keyword groups to the tools they suggest.
    private let keywordToolMap: [(keywords: [String], toolNames: [String])] = [
        (["list files", "show files", "files in", "directory", "folder", "what's in"], ["list_directory"]),
        (["read file", "read the file", "contents of", "show me the file"], ["read_file"]),
        (["write file", "save to file", "create a file", "append to file", "write a note"], ["write_file"]),
        (["run command", "run shell", "execute command", "shell command"], ["run_shell"]),
        (["search content", "search in", "grep", "find text", "find in files", "search for"], ["search_content"]),
        (["screenshot", "screen capture", "capture screen", "what's on my screen"], ["screen_capture"]),
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
        "run_shell", "run_applescript", "write_file", "index_documents",
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

        if let value = toolApproval[toolName] {
            return value
        }
        if policyLoaded {
            return defaultRequireApproval
        }
        return hardcodedApprovalRequired.contains(toolName)
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
    func executeTool(name: String, args: [String: String]) async -> String {
        // Check policy if a policy engine is attached.
        if let policy = policyEngine {
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
        timeout: TimeInterval
    ) -> (stdout: String, stderr: String, exitCode: Int) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

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
            return ("", "Error: \(error.localizedDescription)", -1)
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
