// BadAppleTools — Swift translation of the Python tool router, policy engine,
// and tool executor for Bad Apple.
//
// This file provides the native Swift equivalent of badapple_mlx_tools.py and
// the tool-execution helpers in badapple_tools.py, adapted for the macOS
// menu-bar / desktop client.

import Foundation
import Dispatch
import CommonCrypto
import Darwin

// MARK: - Custom Workshop Tools

/// A user-defined tool created in the Control Center Workshop and stored in
/// `~/.bad_apple/custom_tools.json`. The engine loads these and treats them as
/// first-class tools the local 7B model can invoke.
struct WorkshopCustomTool: Codable, Equatable {
    let id: String
    let name: String
    let description: String
    let kind: String
    let command: String
    let args: [String]

    /// Parameter names exposed to the model. If `args` is empty the tool
    /// takes no parameters.
    var parameters: [BadAppleToolRouter.BadAppleTool.Parameter] {
        args.map { .init(name: $0, description: "Argument \($0)", required: true) }
    }

    /// Convert to a native tool definition for prompt schema generation.
    func asBadAppleTool() -> BadAppleToolRouter.BadAppleTool {
        BadAppleToolRouter.BadAppleTool(
            name: id,
            description: description.isEmpty ? "Custom tool \(id)" : description,
            parameters: parameters,
            requiresApproval: true
        )
    }
}

/// On-disk shape of `~/.bad_apple/custom_tools.json`: the id is the map key.
private struct WorkshopCustomToolValue: Codable, Equatable {
    let name: String
    let description: String
    let kind: String
    let command: String
    let args: [String]
}

/// Shared, thread-unsafe cache for custom tools. Reads the JSON file on demand
/// and caches by mtime. Callers must externally serialize or call from a single
/// actor/queue.
final class CustomToolStore {
    static let shared = CustomToolStore()

    private var lastMtime: Date?
    private var tools: [WorkshopCustomTool] = []
    private var fileURL: URL {
        // Align with the dashboard's data directory (`dirs::data_dir()/bad_apple`).
        // Use NSHomeDirectory() so the root daemon respects the HOME env var.
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let appSupport = home.appendingPathComponent("Library/Application Support")
        let dir = appSupport.appendingPathComponent("bad_apple")
        return dir.appendingPathComponent("custom_tools.json")
    }

    /// Return the current list of custom tools, reloading if the file changed.
    func reload() -> [WorkshopCustomTool] {
        let fm = FileManager.default
        let url = fileURL
        guard fm.fileExists(atPath: url.path) else {
            lastMtime = nil
            tools = []
            return []
        }

        do {
            let attrs = try fm.attributesOfItem(atPath: url.path)
            let mtime = attrs[.modificationDate] as? Date
            if let mtime, let last = lastMtime, mtime <= last {
                return tools
            }
            lastMtime = mtime
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([String: WorkshopCustomToolValue].self, from: data)
            tools = decoded.map { (id, value) in
                WorkshopCustomTool(
                    id: id,
                    name: value.name,
                    description: value.description,
                    kind: value.kind,
                    command: value.command,
                    args: value.args
                )
            }
            return tools
        } catch {
            return tools
        }
    }
}

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

/// Common first-line markers for source code across languages.
private let codeStartMarkers = [
    "def ", "class ", "import ", "from ",
    "fn ", "func ", "function ", "pub ", "public ", "private ",
    "struct ", "enum ", "impl ", "mod ", "use ", "package ",
    "#include", "#!/", "int ", "void ", "char ", "const ",
    "let ", "var ", "static ", "module ", "interface ",
    "<!DOCTYPE", "<?xml", "<html",
]

/// Returns `true` if `text` is primarily source code rather than prose.
private func isCodeLike(_ text: String) -> Bool {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return false }

    // If any non-empty line starts with a known code marker, treat as code.
    let lines = trimmed.components(separatedBy: .newlines)
    for line in lines {
        let stripped = line.trimmingCharacters(in: .whitespaces)
        let lower = stripped.lowercased()
        for marker in codeStartMarkers {
            if lower.hasPrefix(marker.lowercased()) { return true }
        }
    }

    // A single line containing a function definition pattern is also code.
    if trimmed.range(of: #"\bdef\s+\w+\s*\("#, options: .regularExpression) != nil
        || trimmed.range(of: #"\bfunction\s+\w+\s*\("#, options: .regularExpression) != nil
        || trimmed.range(of: #"\bfn\s+\w+\s*\("#, options: .regularExpression) != nil
        || trimmed.range(of: #"\bfunc\s+\w+\s*\("#, options: .regularExpression) != nil
        || trimmed.range(of: #"#include\s+["<]"#, options: .regularExpression) != nil {
        return true
    }

    // Multi-line text with indentation and code-like punctuation.
    let codeLines = lines.filter {
        let s = $0.trimmingCharacters(in: .whitespaces)
        return s.hasPrefix("def ") || s.hasPrefix("class ") || s.hasPrefix("fn ")
            || s.hasPrefix("func ") || s.hasPrefix("import ") || s.hasPrefix("from ")
            || s.hasPrefix("    ") || s.hasPrefix("\t")
    }
    let hasCodePunctuation = trimmed.rangeOfCharacter(from: CharacterSet(charactersIn: "(){}[]=:;.,<>/\\\"|&!")) != nil
    if codeLines.count >= 2 && hasCodePunctuation {
        return true
    }

    // Fenced markdown code blocks only count as code if the block dominates the text.
    if let start = trimmed.range(of: "```")?.lowerBound,
       let end = trimmed[start...].range(of: "```", range: start..<trimmed.endIndex)?.upperBound,
       end > start {
        let before = trimmed[trimmed.startIndex..<start]
        let after = trimmed[end...]
        let nonCode = (before + after).trimmingCharacters(in: .whitespacesAndNewlines)
        // If the prose before/after the fence is short, this is a code response.
        if nonCode.count < 200 {
            return true
        }
    }

    return false
}

/// Strip non-code trailing lines (e.g. persona bars) from source code output.
private func stripTrailingProse(_ text: String) -> String {
    var lines = text.components(separatedBy: .newlines)
    let codePunctuation = CharacterSet(charactersIn: "(){}[]=:;.,<>/\\\"|&!0123456789")
    while let last = lines.last {
        let trimmed = last.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty {
            lines.removeLast()
            continue
        }
        // Heuristic: a trailing line is prose if it is all lowercase and contains
        // no code-like punctuation or digits.
        let hasCodeChar = trimmed.rangeOfCharacter(from: codePunctuation) != nil
        let hasUppercase = trimmed.rangeOfCharacter(from: .uppercaseLetters) != nil
        let hasPersonaPhrase = trimmed.localizedCaseInsensitiveContains("hold the L")
            || trimmed.localizedCaseInsensitiveContains("talk facts")
            || trimmed.localizedCaseInsensitiveContains("no cap")
            || trimmed.localizedCaseInsensitiveContains("bare metal")
            || trimmed.localizedCaseInsensitiveContains("sovereign")
            || trimmed.localizedCaseInsensitiveContains("cloud")
        if hasPersonaPhrase || (!hasCodeChar && !hasUppercase) {
            lines.removeLast()
        } else {
            break
        }
    }
    return lines.joined(separator: "\n")
}

/// Clean source-code output without destroying newlines or indentation.
private func cleanCodeOutput(_ text: String) -> String {
    var result = text

    // Strip Qwen3 thinking blocks and leaked stop tokens.
    result = regexReplace(#"^Thinking Process:.*?(?:Final Answer:|Answer:)\s*"#, in: result, with: "", options: [.dotMatchesLineSeparators, .caseInsensitive])
    result = regexReplace(#"\n?\s*<thinking>.*?\s*\n?"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"\n?\s*\.\.\.thinking\s*.*?(?:</s>|$)"#, in: result, with: "", options: [.dotMatchesLineSeparators])
    result = regexReplace(#"</s>|<\|endoftext\|>|</thinking>"#, in: result, with: "")

    // If the response is wrapped in markdown code fences, keep only the fenced block.
    // This removes prose like "Here is the code:" before or after the fence.
    if let fenceRange = result.range(of: "```") {
        let afterStart = result.index(fenceRange.upperBound, offsetBy: 0)
        if let nextFence = result[afterStart...].range(of: "```") {
            // Skip the optional language tag on the opening line.
            var content = String(result[afterStart..<nextFence.lowerBound])
            if let firstNewline = content.firstIndex(of: "\n") {
                let before = content[content.startIndex..<firstNewline].trimmingCharacters(in: .whitespaces)
                if before.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "+" || $0 == "-" || $0 == "_" }) {
                    content = String(content[content.index(after: firstNewline)...])
                }
            }
            result = content
        } else {
            // Single unclosed fence: remove the opening marker and any trailing prose.
            result = regexReplace(#"^```[a-zA-Z0-9_+-]*\n"#, in: result, with: "", options: [.anchorsMatchLines])
            result = result.replacingOccurrences(of: "```", with: "")
        }
    } else {
        // No fences: remove stray fence artifacts just in case.
        result = regexReplace(#"^```[a-zA-Z0-9_+-]*\n"#, in: result, with: "", options: [.anchorsMatchLines])
        result = regexReplace(#"\n```\s*$"#, in: result, with: "", options: [.anchorsMatchLines])
        result = result.replacingOccurrences(of: "```", with: "")
    }

    // Replace CJK / fullwidth character runs with a space.
    result = regexReplace(#"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+"#, in: result, with: " ")

    // Replace stray IPA phonetic characters with ASCII approximations.
    let ipaMap: [Character: Character] = [
        "\u{028B}": "v", "\u{028C}": "v", "\u{0251}": "a", "\u{0252}": "o",
        "\u{025B}": "e", "\u{026A}": "i", "\u{028A}": "u", "\u{0254}": "o",
        "\u{0259}": "a", "\u{00E6}": "a",
    ]
    result = String(result.map { ipaMap[$0] ?? $0 })

    // Remove disallowed persona ticks.
    result = regexReplace(#"\b[pP]+f+[tT]+\b"#, in: result, with: "")

    // Strip leading prose before the first code-like line.
    result = stripLeadingProse(result)

    // Strip trailing persona/prose bars while preserving code.
    result = stripTrailingProse(result)

    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Strip non-code leading lines (explanations like "Here is the function...").
private func stripLeadingProse(_ text: String) -> String {
    let lines = text.components(separatedBy: .newlines)
    let codePunctuation = CharacterSet(charactersIn: "(){}[]=:;.,<>/\\\"|&!0123456789")
    var firstCodeIndex: Int? = nil
    for (i, line) in lines.enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { continue }
        // A line is considered code if it starts with a known marker or contains
        // code-like punctuation and at least some indentation.
        let lower = trimmed.lowercased()
        if codeStartMarkers.contains(where: { lower.hasPrefix($0) }) {
            firstCodeIndex = i
            break
        }
        let hasCodeChar = trimmed.rangeOfCharacter(from: codePunctuation) != nil
        let hasIndent = line.hasPrefix("    ") || line.hasPrefix("\t")
        if hasCodeChar && hasIndent {
            firstCodeIndex = i
            break
        }
    }
    guard let idx = firstCodeIndex else { return text }
    return lines[idx...].joined(separator: "\n")
}

/// Post-process final model output: remove thinking tags, code fence artifacts,
/// normalize whitespace, collapse repeated sentences.
/// For code output, preserves newlines and indentation instead of collapsing them.
func postprocessOutput(_ text: String) -> String {
    // Do not collapse whitespace for source code.
    if isCodeLike(text) {
        return cleanCodeOutput(text)
    }

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
    struct BadAppleTool: Codable, Equatable {
        let name: String
        let description: String
        let parameters: [Parameter]
        let requiresApproval: Bool

        /// A single named parameter for a tool.
        struct Parameter: Codable, Equatable {
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
        BadAppleTool(
            name: "describe_image",
            description: "Use the vision engine to describe an image file. Returns a text description.",
            parameters: [
                .init(name: "path", description: "Absolute or tilde-expanded path to the image file.", required: true),
                .init(name: "prompt", description: "Optional prompt guiding the description.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "image_generation",
            description: "Generate an image from a text prompt using the local mflux FLUX.2-klein 4B model. Returns the path to the generated PNG.",
            parameters: [
                .init(name: "prompt", description: "Text description of the image to generate.", required: true),
                .init(name: "width", description: "Image width in pixels (default 512).", required: false),
                .init(name: "height", description: "Image height in pixels (default 512).", required: false),
                .init(name: "steps", description: "Number of diffusion steps (default 4).", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "translate_text",
            description: "Translate text between languages. The model handles translation natively.",
            parameters: [
                .init(name: "text", description: "The text to translate.", required: true),
                .init(name: "target_language", description: "The target language for the translation.", required: true),
                .init(name: "source_language", description: "The source language. Defaults to 'auto'.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "consolidate_memory",
            description: "Consolidate working memory by summarizing and deduplicating facts.",
            parameters: [],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "workspace_status",
            description: "Return the current workspace path and a brief summary.",
            parameters: [],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "read_document",
            description: "Read a document file (txt, md, pdf, docx) with size limits.",
            parameters: [
                .init(name: "path", description: "Absolute or tilde-expanded path to the document.", required: true),
                .init(name: "max_chars", description: "Max characters to return. Default 10000.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "search_local_files",
            description: "Search for files by name pattern in a directory.",
            parameters: [
                .init(name: "pattern", description: "The file name pattern to search for.", required: true),
                .init(name: "path", description: "Directory to search. Defaults to the user's home directory.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "set_session_seed",
            description: "Set a session seed for deterministic generation.",
            parameters: [
                .init(name: "seed", description: "The session seed string to store.", required: true),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "get_session_seed",
            description: "Get the current session seed.",
            parameters: [],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "git_status",
            description: "Run `git status` in the active workspace or a given path and return the output.",
            parameters: [
                .init(name: "path", description: "Optional directory to run git status in. Defaults to the active workspace.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "inspect_output_firewall",
            description: "Inspect the active output firewall: built-in default patterns, on-disk blocklist patterns, and the blocklist file path.",
            parameters: [
                .init(name: "show_patterns", description: "If 'true', include the on-disk blocklist patterns in the response. Default 'false'.", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "update_output_firewall",
            description: "Add or remove a pattern in the output firewall blocklist file and reload the active pattern set. Patterns must be 3-256 characters with no control characters or newlines.",
            parameters: [
                .init(name: "pattern", description: "The literal pattern to add or remove.", required: true),
                .init(name: "action", description: "'add' (default) or 'remove'.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "undo_last",
            description: "Undo the most recent user action by removing the last appended note or generated image.",
            parameters: [
                .init(name: "kind", description: "What to undo: 'note' or 'image'. Default is 'note'.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "self_audit",
            description: "Run the local cert suite and doctor diagnostic and return a JSON summary. Use this to verify Bad Apple's own security, health, and air-gap posture.",
            parameters: [
                .init(name: "include", description: "Comma-separated list: 'cert', 'doctor', or 'all' (default 'all').", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "curious_self_improve",
            description: "Run a bounded Curious self-improvement check: self-audit, output firewall, git status, search for TODO/FIXME/HACK/XXX in the source, and write a proposal note under ~/.bad_apple/notes/proposed_patches/.",
            parameters: [
                .init(name: "include", description: "What to include in the self-audit: 'cert', 'doctor', 'runtime', or 'all' (default 'all').", required: false),
            ],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "repair_runtime_issue",
            description: "Execute a bounded, allowlisted runtime repair such as loading a user LaunchAgent or creating a missing data file. Unknown issues are rejected.",
            parameters: [
                .init(name: "issue", description: "The repair issue id from self_audit repairs: data_dir_missing, blocklist_missing, identity_agent_not_loaded, dashboard_agent_not_loaded, tts_agent_not_loaded, menubar_agent_not_loaded, app_not_installed.", required: true),
                .init(name: "target", description: "Optional target path for the repair.", required: false),
            ],
            requiresApproval: true
        ),
        BadAppleTool(
            name: "kill_switch",
            description: "Engage the Bad Apple kill switch. Immediately pauses all generation, tools, and ambient actions until the user says 'resume bad apple'.",
            parameters: [],
            requiresApproval: false
        ),
        BadAppleTool(
            name: "resume",
            description: "Disengage the Bad Apple kill switch and resume normal operation.",
            parameters: [],
            requiresApproval: false
        ),
    ]

    /// Workshop/custom tools loaded from `~/.bad_apple/custom_tools.json`.
    /// Mutable so we can hot-reload without restarting the model.
    private var customTools: [BadAppleTool] = []
    private var customToolKeywords: [String] = []
    private var customKeywordMap: [(keywords: [String], toolNames: [String])] = []
    private var customToolExampleArgs: [String: String] = [:]

    /// Names in the native registry, exposed for discovery and logic tests.
    func registeredToolNames() -> [String] {
        reloadCustomToolsIfNeeded()
        return tools.map(\.name) + customTools.map(\.name)
    }

    /// Load or refresh the custom tool definitions from disk.
    func reloadCustomToolsIfNeeded() {
        let loaded = CustomToolStore.shared.reload()
        let converted = loaded.map { $0.asBadAppleTool() }
        guard converted != customTools else { return }
        customTools = converted

        customKeywordMap = converted.map { tool in
            let nameWords = tool.name.split(separator: "_").map(String.init)
            let descWords = tool.description
                .lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty && $0.count > 2 }
            let keywords = Array(Set(nameWords + descWords)).sorted()
            return (keywords: keywords, toolNames: [tool.name])
        }

        customToolKeywords = customKeywordMap.flatMap { $0.keywords }

        customToolExampleArgs = Dictionary(uniqueKeysWithValues: loaded.map { tool in
            let pairs = tool.args.enumerated().map { index, arg in
                "\"\(arg)\":\"value\(index + 1)\""
            }
            return (tool.id, pairs.joined(separator: ","))
        })
    }

    /// All tool definitions, native + workshop, with a fresh custom-tool reload.
    private func allTools() -> [BadAppleTool] {
        reloadCustomToolsIfNeeded()
        return tools + customTools
    }

    /// Keyword maps for both native and custom tools.
    private func allKeywordMaps() -> [(keywords: [String], toolNames: [String])] {
        reloadCustomToolsIfNeeded()
        return keywordToolMap + customKeywordMap
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
        // Self-audit
        "self audit", "run diagnostics", "health check", "cert suite", "air gap check",
        // Output firewall
        "blocklist", "block pattern", "what is blocked", "what is output firewall", "inspect output firewall", "inspect firewall",
        "add pattern", "add output firewall", "add pattern to output firewall",
        "remove pattern", "remove output firewall", "remove pattern from output firewall",
        "block this pattern", "update firewall",
        // Curious self-improvement
        "curious", "self improve", "improve yourself", "improve bad apple", "curious check",
        // Code
        "code", "function", "compile", "build",
        // Kill switch / resume
        "kill switch", "stop everything", "emergency stop", "resume", "resume bad apple", "start again", "leave safe mode", "exit safe mode",
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
        (["code", "coding", "program", "programming", "developer", "refactor", "debug", "build project", "run tests"], ["read_file", "search_content", "write_file", "run_shell", "index_documents", "workspace_status"]),
        (["describe image", "image description", "what's in this image", "analyze image"], ["describe_image"]),
        (["generate image", "make an image", "create image", "draw", "image of"], ["image_generation"]),
        (["translate", "translation", "translate text"], ["translate_text"]),
        (["consolidate memory", "deduplicate memory", "summarize memory"], ["consolidate_memory"]),
        (["workspace status", "current workspace", "workspace path"], ["workspace_status"]),
        (["read document", "open document", "document content"], ["read_document"]),
        (["search files", "find files", "file search", "search local files"], ["search_local_files"]),
        (["session seed", "set seed", "deterministic seed"], ["set_session_seed", "get_session_seed"]),
        (["git status", "git diff", "what changed"], ["git_status"]),
        (["blocklist", "block pattern", "what is blocked", "what is output firewall", "inspect output firewall", "inspect firewall"], ["inspect_output_firewall"]),
        (["add pattern", "add output firewall", "add pattern to output firewall", "remove pattern", "remove output firewall", "remove pattern from output firewall", "block this pattern", "update firewall"], ["update_output_firewall"]),
        (["undo", "delete last", "remove last"], ["undo_last"]),
        (["self audit", "self_audit", "run self audit", "audit bad apple", "run diagnostics", "health check", "cert suite", "air gap check"], ["self_audit"]),
        (["curious", "self improve", "improve yourself", "improve bad apple", "curious check"], ["curious_self_improve"]),
        (["kill switch", "stop everything", "emergency stop", "panic stop"], ["kill_switch"]),
        (["resume", "resume bad apple", "start again", "leave safe mode", "exit safe mode"], ["resume"]),
    ]

    // MARK: - Prompt Routing

    /// Match if any keyword is a substring of `text` or if all words of the
    /// keyword appear in `text`. This catches "add testpattern to the output
    /// firewall" without matching on single words like "add" alone.
    private func matchesKeyword(_ keyword: String, in text: String) -> Bool {
        let low = text.lowercased()
        let words = Set(low.split(separator: " ").map { String($0) })
        if low.contains(keyword) { return true }
        let keywordWords = keyword.split(separator: " ").map { String($0) }
        guard !keywordWords.isEmpty else { return false }
        return Set(keywordWords).isSubset(of: words)
    }

    /// Return formatted definitions for all tools. Used by the agent, which
    /// may need to pick from any available tool for a planned step.
    func allToolsForPrompt() -> String? {
        let everyTool = allTools()
        guard !everyTool.isEmpty else { return nil }

        var lines: [String] = ["Available tools:"]
        for tool in everyTool {
            lines.append("")
            lines.append("[\(tool.name)] \u{2014} \(tool.description)")
            if !tool.parameters.isEmpty {
                lines.append("  Parameters:")
                for param in tool.parameters {
                    let req = param.required ? "required" : "optional"
                    lines.append("    \(param.name) (\(req)): \(param.description)")
                }
            }
            if tool.requiresApproval {
                lines.append("  Requires user approval before it runs.")
            }
            if let example = exampleForTool(tool) {
                lines.append("  Example: \(example)")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Return formatted tool definitions relevant to the prompt, or nil if no
    /// tools are needed. Uses plain English so a 7B local model can follow.
    func toolsForPrompt(text: String) -> String? {
        var selectedNames = Set<String>()

        for (keywords, toolNames) in allKeywordMaps() {
            if keywords.contains(where: { matchesKeyword($0, in: text) }) {
                selectedNames.formUnion(toolNames)
            }
        }

        guard !selectedNames.isEmpty else { return nil }

        let selected = allTools().filter { selectedNames.contains($0.name) }
        guard !selected.isEmpty else { return nil }

        var lines: [String] = ["Available tools:"]
        for tool in selected {
            lines.append("")
            lines.append("[\(tool.name)] \u{2014} \(tool.description)")
            if !tool.parameters.isEmpty {
                lines.append("  Parameters:")
                for param in tool.parameters {
                    let req = param.required ? "required" : "optional"
                    lines.append("    \(param.name) (\(req)): \(param.description)")
                }
            }
            if tool.requiresApproval {
                lines.append("  Requires user approval before it runs.")
            }
            if let example = exampleForTool(tool) {
                lines.append("  Example: \(example)")
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Return tool schemas in the chat-template format expected by
    /// `BadAppleInference.generateWithTools`.
    func toolSchemasForPrompt(text: String) -> [[String: Any]]? {
        var selectedNames = Set<String>()

        for (keywords, toolNames) in allKeywordMaps() {
            if keywords.contains(where: { matchesKeyword($0, in: text) }) {
                selectedNames.formUnion(toolNames)
            }
        }

        guard !selectedNames.isEmpty else { return nil }

        let selected = allTools().filter { selectedNames.contains($0.name) }
        guard !selected.isEmpty else { return nil }

        return selected.map { tool in
            var properties: [String: Any] = [:]
            var required: [String] = []
            for param in tool.parameters {
                properties[param.name] = [
                    "type": "string",
                    "description": param.description,
                ]
                if param.required {
                    required.append(param.name)
                }
            }

            let parameters: [String: Any] = [
                "type": "object",
                "properties": properties,
                "required": required,
            ]

            let function: [String: Any] = [
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            ]

            return [
                "type": "function",
                "function": function,
            ]
        }
    }

    /// Build a concrete `<tool_call>` example for a single tool, with all
    /// required parameters filled in. This keeps the local 7B model from
    /// nesting the schema under `arguments.properties`.
    private func exampleForTool(_ tool: BadAppleTool) -> String? {
        let home = NSHomeDirectory()
        let argExamples: [String: String] = [
            "read_file": "\"path\":\"/var/lib/bad_apple/blocklist.txt\"",
            "write_file": "\"path\":\"\(home)/.bad_apple/notes.txt\",\"content\":\"hello\"",
            "list_directory": "\"path\":\"\(home)/.bad_apple\"",
            "search_content": "\"pattern\":\"TODO\",\"path\":\"\(home)/Documents/src\"",
            "run_shell": "\"command\":\"ls /tmp\"",
            "run_applescript": "\"script\":\"tell app \\\"Finder\\\" to activate\"",
            "run_shortcut": "\"name\":\"Good Morning\"",
            "set_workspace": "\"path\":\"\(home)/Documents\"",
            "describe_image": "\"path\":\"/var/lib/bad_apple/generated_images/image.png\"",
            "image_generation": "\"prompt\":\"a red apple on a beach\"",
            "search_local_files": "\"pattern\":\"AGENTS.md\"",
            "index_documents": "\"path\":\"\(home)/Documents\"",
            "read_document": "\"path\":\"\(home)/Documents/README.md\"",
            "translate_text": "\"text\":\"hello\",\"to\":\"spanish\"",
            "consolidate_memory": "",
            "workspace_status": "",
            "set_session_seed": "\"seed\":\"42\"",
            "get_session_seed": "",
            "git_status": "",
            "inspect_output_firewall": "\"show_patterns\":\"false\"",
            "update_output_firewall": "\"pattern\":\"badword\",\"action\":\"add\"",
            "undo_last": "\"kind\":\"image\"",
            "self_audit": "\"include\":\"all\"",
            "curious_self_improve": "\"include\":\"all\"",
            "screen_capture": "",
            "kill_switch": "",
            "resume": "",
        ]

        let exampleArgs = argExamples[tool.name] ?? customToolExampleArgs[tool.name] ?? ""
        let args = exampleArgs.isEmpty ? "" : ",\"arguments\":{\(exampleArgs)}"
        let prefix = "<tool_call>{\"name\":\"" + tool.name + "\""
        let suffix = "\(args)}</tool_call>"
        return prefix + suffix
    }

    /// Heuristic: should the model be offered tools for this prompt?
    func shouldUseTools(prompt: String) -> Bool {
        let lowered = prompt.lowercased()
        if toolKeywords.contains(where: { matchesKeyword($0, in: lowered) }) { return true }
        reloadCustomToolsIfNeeded()
        return customToolKeywords.contains(where: { matchesKeyword($0, in: lowered) })
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

        // Some local models emit a bare JSON object instead of wrapping it in
        // <tool_call> tags. Extract top-level JSON objects and treat `name` +
        // `arguments` (or `arguments.properties` if the model nested the schema)
        // as a tool call.
        for candidate in plainJSONCandidates(text) {
            guard let data = candidate.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = object["name"] as? String else { continue }
            var rawArgs = object["arguments"] as? [String: Any] ?? [:]
            if rawArgs.count == 1, let props = rawArgs["properties"] as? [String: Any] {
                rawArgs = props
            }
            let args = rawArgs.mapValues { value in
                if let string = value as? String { return string }
                return String(describing: value)
            }
            calls.append((name: name, args: args))
        }

        return calls
    }

    /// Find balanced top-level JSON objects in `text` by brace counting.
    private func plainJSONCandidates(_ text: String) -> [String] {
        var candidates: [String] = []
        var depth = 0
        var start: String.Index?
        for index in text.indices {
            let c = text[index]
            if c == "{" {
                if depth == 0 { start = index }
                depth += 1
            } else if c == "}", depth > 0 {
                depth -= 1
                if depth == 0, let s = start {
                    candidates.append(String(text[s...index]))
                    start = nil
                }
            }
        }
        return candidates
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
        case denied(reason: String)
        case needsApproval
    }

    /// Per-tool policy settings parsed from policy.yaml.
    private struct ToolPolicy {
        var allowed: Bool = true
        var requireApproval: Bool = true
        var allowedPaths: [String] = []
        var deniedPatterns: [String] = []
        var allowedCommands: [String] = []
        var maxTimeout: Int = 0
        var allowedApps: [String] = []
        var allowedFiles: [String] = []
        var notesDir: String?
        var maxSize: Int?
    }

    private enum Section {
        case top, defaults, tools
    }

    // MARK: - State

    private let lock = NSLock()
    private var _autopilot: Bool = false
    private var policyLoaded: Bool = false
    private var defaultPolicy = ToolPolicy()
    private var toolPolicies: [String: ToolPolicy] = [:]

    /// Tools that always require approval regardless of policy file.
    private let hardcodedApprovalRequired: Set<String> = [
        "run_shell", "run_applescript", "run_shortcut", "write_file",
        "write_working_memory", "clear_working_memory", "index_documents",
        "screen_capture", "consolidate_memory",
    ]

    /// Policy file path.
    private let policyPath = "/var/lib/bad_apple/policy.yaml"

    // MARK: - Init

    init() {
        loadPolicy()
    }

    // MARK: - Autopilot

    /// In-memory autopilot override. The canonical persistent level is
    /// `~/.bad_apple/autopilot_level`; this property is used for the current
    /// process lifetime only.
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

        if hardcodedApprovalRequired.contains(toolName) { return true }
        let policy = toolPolicies[toolName] ?? defaultPolicy
        return policy.requireApproval
    }

    /// Check whether a tool is allowed at all by policy.
    func isAllowed(toolName: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let policy = toolPolicies[toolName] ?? defaultPolicy
        return policy.allowed
    }

    /// Maximum allowed timeout for a tool, in seconds.
    func maxTimeout(toolName: String, defaultTimeout: Int = 30) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let policy = toolPolicies[toolName] ?? defaultPolicy
        return policy.maxTimeout > 0 ? policy.maxTimeout : defaultTimeout
    }

    /// Maximum read size for tools that return file content.
    func maxSize(toolName: String, defaultSize: Int = 100_000) -> Int {
        lock.lock()
        defer { lock.unlock() }

        let policy = toolPolicies[toolName] ?? defaultPolicy
        return policy.maxSize ?? defaultSize
    }

    /// Evaluate a tool call against the full policy and autopilot state.
    func evaluate(toolName: String, args: [String: String]) -> ApprovalDecision {
        let policy: ToolPolicy
        let auto: Bool
        lock.lock()
        policy = toolPolicies[toolName] ?? defaultPolicy
        auto = _autopilot
        lock.unlock()

        guard policy.allowed else {
            return .denied(reason: "tool '\(toolName)' is disabled by policy")
        }

        if let reason = validateArgs(toolName: toolName, args: args, policy: policy) {
            return .denied(reason: reason)
        }

        if auto {
            return .approved
        }
        if hardcodedApprovalRequired.contains(toolName) || policy.requireApproval {
            return .needsApproval
        }
        return .approved
    }

    // MARK: - Argument Validation

    private func validateArgs(toolName: String, args: [String: String], policy: ToolPolicy) -> String? {
        switch toolName {
        case "read_file", "list_directory", "search_content", "search_local_files",
             "index_documents", "read_document":
            let path = args["path"] ?? "~"
            return validatePath(path, policy: policy, tool: toolName)

        case "write_file":
            let path = args["path"] ?? ""
            if path.isEmpty { return "write_file requires a path" }
            if let notesDir = policy.notesDir, !notesDir.isEmpty {
                let expandedNotes = expandPath(notesDir)
                let expandedPath = expandPath(path)
                guard isPath(expandedPath, under: [expandedNotes]) else {
                    return "write_file path must be under notes_dir \(notesDir)"
                }
            }
            return validatePath(path, policy: policy, tool: toolName)

        case "run_shell":
            let command = args["command"] ?? ""
            if command.isEmpty { return "run_shell requires a command" }
            if !policy.allowedCommands.isEmpty {
                let first = command.trimmingCharacters(in: .whitespaces)
                    .split(separator: " ", omittingEmptySubsequences: true)
                    .first.map(String.init) ?? ""
                let base = (first as NSString).lastPathComponent
                let name = base.isEmpty ? first : base
                let lowered = name.lowercased()
                guard policy.allowedCommands.map({ $0.lowercased() }).contains(lowered) else {
                    return "command '\(name)' is not in the policy allowed_commands list"
                }
            }
            for pattern in policy.deniedPatterns {
                if command.contains(pattern) {
                    return "command matches denied pattern '\(pattern)'"
                }
            }
            return nil

        case "run_applescript":
            let script = args["script"] ?? ""
            if script.isEmpty { return "run_applescript requires a script" }
            for pattern in policy.deniedPatterns {
                if script.lowercased().contains(pattern.lowercased()) {
                    return "AppleScript matches denied pattern '\(pattern)'"
                }
            }
            if !policy.allowedApps.isEmpty {
                let loweredAllowed = Set(policy.allowedApps.map { $0.lowercased() })
                let targeted = matchesAppTell(script)
                for app in targeted where !loweredAllowed.contains(app.lowercased()) {
                    return "AppleScript targets application '\(app)' which is not in allowed_apps"
                }
            }
            return nil

        case "run_shortcut":
            let name = args["name"] ?? ""
            if name.isEmpty { return "run_shortcut requires a name" }
            for pattern in policy.deniedPatterns {
                if name.contains(pattern) {
                    return "shortcut name matches denied pattern '\(pattern)'"
                }
            }
            return nil

        case "update_output_firewall":
            let pattern = args["pattern"] ?? ""
            if pattern.isEmpty { return "update_output_firewall requires a pattern" }
            if pattern.count < 3 || pattern.count > 256 {
                return "output firewall pattern must be 3-256 characters"
            }
            let action = (args["action"] ?? "add").lowercased()
            if action != "add" && action != "remove" {
                return "output firewall action must be 'add' or 'remove'"
            }
            if pattern.rangeOfCharacter(from: .controlCharacters) != nil || pattern.rangeOfCharacter(from: .newlines) != nil {
                return "output firewall pattern must not contain control characters or newlines"
            }
            return nil

        default:
            return nil
        }
    }

    private func matchesAppTell(_ script: String) -> [String] {
        var names: [String] = []
        let pattern = #"(?i)tell\s+(?:application|app)\s+\"([^\"]+)\""#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return names }
        let range = NSRange(script.startIndex..., in: script)
        regex.enumerateMatches(in: script, options: [], range: range) { match, _, _ in
            guard let match = match, let r = Range(match.range(at: 1), in: script) else { return }
            names.append(String(script[r]))
        }
        return names
    }

    private func validatePath(_ path: String, policy: ToolPolicy, tool: String) -> String? {
        let expanded = expandPath(path)
        guard !expanded.isEmpty else { return "\(tool) path is empty" }
        if expanded.contains("..") { return "\(tool) path contains path traversal '..'" }

        let targets = [path, expanded]
        for pattern in policy.deniedPatterns {
            for target in targets where target.contains(pattern) {
                return "\(tool) path matches denied pattern '\(pattern)'"
            }
        }

        if !policy.allowedPaths.isEmpty {
            let expandedRoots = policy.allowedPaths.map(expandPath)
            guard isPath(expanded, under: expandedRoots) else {
                return "\(tool) path '\(path)' is not under allowed_paths"
            }
        }
        return nil
    }

    private func isPath(_ path: String, under roots: [String]) -> Bool {
        let resolved = (path as NSString).standardizingPath
        for root in roots {
            let resolvedRoot = (root as NSString).standardizingPath
            if resolved == resolvedRoot || resolved.hasPrefix(resolvedRoot + "/") { return true }
        }
        return false
    }

    private func expandPath(_ path: String) -> String {
        if path == "~" { return NSHomeDirectory() }
        if path.hasPrefix("~/") { return NSHomeDirectory() + String(path.dropFirst(2)) }
        if path.hasPrefix("~") { return NSHomeDirectory() + String(path.dropFirst(1)) }
        return path
    }

    // MARK: - Policy Loading

    /// Load policy from the YAML file. Supports scalars, flow lists, and block lists.
    private func loadPolicy() {
        guard let content = try? String(contentsOfFile: policyPath, encoding: .utf8) else { return }

        var defaults = ToolPolicy()
        var tools: [String: ToolPolicy] = [:]
        var section: Section = .top
        var currentToolName: String?
        var currentPolicy: ToolPolicy?
        var currentListKey: String?
        var currentList: [String] = []

        let lines = content.components(separatedBy: "\n")
        for rawLine in lines {
            let stripped = rawLine.trimmingCharacters(in: .whitespaces)
            if stripped.isEmpty || stripped.hasPrefix("#") { continue }
            let indent = rawLine.prefix(while: { $0 == " " }).count

            if indent == 0 {
                // Flush any pending list/tool before changing section.
                if section == .defaults, let key = currentListKey, !currentList.isEmpty {
                    applyListField(key: key, list: currentList, policy: &defaults)
                } else if section == .tools, let key = currentListKey, !currentList.isEmpty,
                          var policy = currentPolicy, let name = currentToolName {
                    applyListField(key: key, list: currentList, policy: &policy)
                    tools[name] = policy
                    currentPolicy = policy
                }
                currentListKey = nil
                currentList = []
                if let name = currentToolName, let policy = currentPolicy {
                    tools[name] = policy
                }
                currentToolName = nil
                currentPolicy = nil

                section = .top

                if stripped == "defaults:" {
                    section = .defaults
                } else if stripped == "tools:" {
                    section = .tools
                } else if stripped.contains(":") {
                    let parts = stripped.split(separator: ":", maxSplits: 1)
                    if parts.count == 2 {
                        let key = String(parts[0]).trimmingCharacters(in: .whitespaces)
                        let value = String(parts[1]).trimmingCharacters(in: .whitespaces)
                        if key == "autopilot" {
                            lock.lock(); _autopilot = (value == "true"); lock.unlock()
                        }
                    }
                }
                continue
            }

            switch section {
            case .top:
                continue
            case .defaults:
                if let colon = stripped.firstIndex(of: ":") {
                    let key = String(stripped[..<colon]).trimmingCharacters(in: .whitespaces)
                    let value = String(stripped[stripped.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                    if value.isEmpty {
                        currentListKey = key
                        currentList = []
                    } else if value.hasPrefix("[") && value.hasSuffix("]") {
                        let inner = String(value.dropFirst().dropLast())
                        applyListField(key: key, list: parseFlowList(inner), policy: &defaults)
                    } else {
                        applyScalarField(key: key, value: value, policy: &defaults)
                    }
                } else if stripped.hasPrefix("- "), currentListKey != nil {
                    let item = String(stripped.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                    currentList.append(trimQuotes(item))
                }
            case .tools:
                if indent == 2 {
                    // New tool block.
                    if let name = currentToolName, var policy = currentPolicy {
                        if let key = currentListKey, !currentList.isEmpty {
                            applyListField(key: key, list: currentList, policy: &policy)
                        }
                        tools[name] = policy
                        currentListKey = nil
                        currentList = []
                    }
                    let toolName = stripped.hasSuffix(":") ? String(stripped.dropLast()) : stripped
                    currentToolName = toolName
                    currentPolicy = defaults
                    continue
                }
                guard var policy = currentPolicy, let name = currentToolName else { continue }
                if let colon = stripped.firstIndex(of: ":") {
                    let key = String(stripped[..<colon]).trimmingCharacters(in: .whitespaces)
                    let value = String(stripped[stripped.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                    if value.isEmpty {
                        currentListKey = key
                        currentList = []
                    } else if value.hasPrefix("[") && value.hasSuffix("]") {
                        let inner = String(value.dropFirst().dropLast())
                        applyListField(key: key, list: parseFlowList(inner), policy: &policy)
                    } else {
                        applyScalarField(key: key, value: value, policy: &policy)
                    }
                    tools[name] = policy
                    currentPolicy = policy
                } else if stripped.hasPrefix("- "), currentListKey != nil {
                    let item = String(stripped.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                    currentList.append(trimQuotes(item))
                }
            }
        }

        // Flush any trailing list/tool.
        if section == .tools, let key = currentListKey, !currentList.isEmpty,
           var policy = currentPolicy, let name = currentToolName {
            applyListField(key: key, list: currentList, policy: &policy)
            tools[name] = policy
        } else if section == .defaults, let key = currentListKey, !currentList.isEmpty {
            applyListField(key: key, list: currentList, policy: &defaults)
        }
        if let name = currentToolName, let policy = currentPolicy {
            tools[name] = policy
        }

        lock.lock()
        defaultPolicy = defaults
        toolPolicies = tools
        policyLoaded = true

        // Autopilot is intentionally in-memory only; the persistent level is
        // kept in ~/.bad_apple/autopilot_level by BadAppleEngine.
        lock.unlock()
    }

    private func parseFlowList(_ inner: String) -> [String] {
        var result: [String] = []
        var current = ""
        var inQuotes = false
        var quoteChar: Character?
        for char in inner {
            if inQuotes {
                if char == quoteChar {
                    inQuotes = false
                    quoteChar = nil
                } else {
                    current.append(char)
                }
            } else if char == "\"" || char == "'" {
                inQuotes = true
                quoteChar = char
            } else if char == "," {
                result.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(char)
            }
        }
        let last = current.trimmingCharacters(in: .whitespaces)
        if !last.isEmpty { result.append(last) }
        return result.filter { !$0.isEmpty }.map(trimQuotes)
    }

    private func applyScalarField(key: String, value: String, policy: inout ToolPolicy) {
        switch key {
        case "allowed":
            policy.allowed = parseBool(value)
        case "require_approval":
            policy.requireApproval = parseBool(value)
        case "max_timeout":
            if let i = Int(value) { policy.maxTimeout = i }
        case "max_size":
            if let i = Int(value) { policy.maxSize = i }
        case "notes_dir":
            policy.notesDir = value
        default:
            break
        }
    }

    private func applyListField(key: String, list: [String], policy: inout ToolPolicy) {
        switch key {
        case "allowed_paths":
            policy.allowedPaths = list
        case "denied_patterns":
            policy.deniedPatterns = list
        case "allowed_commands":
            policy.allowedCommands = list
        case "allowed_apps":
            policy.allowedApps = list
        case "allowed_files":
            policy.allowedFiles = list
        default:
            break
        }
    }

    private func parseBool(_ value: String) -> Bool {
        value.lowercased() == "true"
    }

    private func trimQuotes(_ s: String) -> String {
        var r = s
        if r.hasPrefix("\"") { r.removeFirst() }
        if r.hasPrefix("'") { r.removeFirst() }
        if r.hasSuffix("\"") { r.removeLast() }
        if r.hasSuffix("'") { r.removeLast() }
        return r.trimmingCharacters(in: .whitespaces)
    }
}

// MARK: - Custom Tool Execution

/// Standalone executor for user-defined workshop tools. Mirrors the dashboard's
/// Rust runner and uses the same defensive boundaries.
enum CustomToolExecutor {
    /// Replace `{{1}}`, `{{2}}`, ... placeholders in a template with the
    /// provided positional arguments.
    static func interpolate(template: String, args: [String]) -> String {
        var out = template
        for (index, arg) in args.enumerated() {
            let placeholder = String(repeating: "{", count: 2) + "\(index + 1)" + String(repeating: "}", count: 2)
            out = out.replacingOccurrences(of: placeholder, with: arg)
        }
        return out
    }

    /// Shell-escape a string for use inside single quotes.
    static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    /// Run an interpolated shell command via `/bin/sh -c`. Rejects command
    /// chaining metacharacters as a defensive boundary.
    static func runShellCommand(_ command: String) -> String {
        if command.contains(";") || command.contains("&&") || command.contains("||") || command.contains(">") {
            return "Error: custom shell command contains disallowed metacharacters"
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", command]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        var stdoutData = Data()
        var stderrData = Data()
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
            return "Error: \(error.localizedDescription)"
        }

        let timeout: TimeInterval = 30
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }

        if process.isRunning {
            process.terminate()
            group.wait()
            return "Error: custom shell command timed out"
        }

        group.wait()
        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""
        if process.terminationStatus != 0, !stderr.isEmpty {
            return "Error: \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
        return stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "(no output)"
            : String(stdout.prefix(10_000))
    }

    /// Run an interpolated AppleScript via `/usr/bin/osascript`. stdin is used
    /// so multi-line scripts and quotes are preserved.
    static func runAppleScript(_ script: String) -> String {
        if script.count > 10_000 || script.contains("\0") {
            return "Error: custom AppleScript is too large or contains invalid characters"
        }
        let lowered = script.lowercased()
        let denied = [
            "do shell script", "do script", "run script", "use framework",
            "do javascript", "open location",
            "current application", "current application's",
            "keystroke", "key code",
            "curl", "wget", "rm -rf",
        ]
        if let match = denied.first(where: { lowered.contains($0) }) {
            return "Error: custom AppleScript contains denied operation '\(match)'"
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-"]

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        let stdinPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        process.standardInput = stdinPipe

        var stdoutData = Data()
        var stderrData = Data()
        let group = DispatchGroup()
        DispatchQueue.global().async(group: group) {
            stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        }
        DispatchQueue.global().async(group: group) {
            stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        }

        do {
            try process.run()
            stdinPipe.fileHandleForWriting.write(script.data(using: .utf8) ?? Data())
            stdinPipe.fileHandleForWriting.closeFile()
        } catch {
            return "Error: \(error.localizedDescription)"
        }

        let timeout: TimeInterval = 30
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }

        if process.isRunning {
            process.terminate()
            group.wait()
            return "Error: custom AppleScript timed out"
        }

        group.wait()
        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""
        if process.terminationStatus != 0, !stderr.isEmpty {
            return "Error: \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
        return stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "done"
            : String(stdout.prefix(10_000))
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
    private var sessionSeed: String?
    private var isRunningCurious = false

    /// Optional vision provider closure. When set, `describe_image` delegates
    /// to this closure with (path, prompt) and returns the description.
    var visionProvider: ((String, String) async -> String)?

    /// Optional output firewall reference. When set, `inspect_output_firewall`
    /// and `update_output_firewall` can read and modify the active blocklist.
    var outputFirewall: BadAppleOutputFirewall?

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

    // MARK: - Tool Aliases

    /// Map common misnames the 7B model invents to the real tool names.
    private let toolAliases: [String: [String]] = [
        "update_output_firewall": ["add_output_firewall_pattern", "remove_output_firewall_pattern", "add_blocklist", "remove_blocklist"],
        "self_audit": ["run_diagnostics", "health_check", "diagnostics", "cert_suite"],
        "inspect_output_firewall": ["output_firewall", "firewall_patterns"],
    ]

    private let allKnownToolNames: [String] = [
        "get_current_time", "read_file", "write_file", "list_directory", "search_content",
        "run_shell", "run_applescript", "run_shortcut", "set_workspace", "workspace_status",
        "describe_image", "image_generation", "search_local_files", "index_documents", "read_document",
        "translate_text", "consolidate_memory", "set_session_seed", "get_session_seed", "git_status",
        "inspect_output_firewall", "update_output_firewall", "undo_last", "self_audit", "screen_capture",
        "kill_switch", "resume"
    ]

    private func resolveToolName(_ name: String) -> String {
        let lower = name.lowercased().replacingOccurrences(of: "-", with: "_")
        if allKnownToolNames.contains(lower) { return lower }
        for (real, aliases) in toolAliases {
            if aliases.contains(lower) { return real }
        }
        for real in allKnownToolNames.sorted(by: { $0.count > $1.count }) {
            if lower.contains(real) { return real }
        }

        // Finally, check the workshop custom tools. They can shadow native names
        // if the user created one with the same id.
        let custom = CustomToolStore.shared.reload()
        for tool in custom where tool.id == lower { return tool.id }
        return name
    }

    // MARK: - Tool Execution

    /// Execute a tool by name with the given arguments.
    /// Returns the tool result as a string (or an error message).
    func executeTool(name: String, args: [String: String], approved: Bool = false) async -> String {
        let resolved = resolveToolName(name)

        // Kill switch and resume bypass the normal policy/approval flow.
        switch resolved {
        case "kill_switch":
            BadAppleEngine.shared.killed = true
            return "Kill switch engaged. Bad Apple is paused. Say 'resume bad apple' to start again."
        case "resume":
            BadAppleEngine.shared.killed = false
            return "Bad Apple is back online. No cap."
        default:
            break
        }

        // Check policy if a policy engine is attached.
        if !approved, let policy = policyEngine {
            let decision = policy.evaluate(toolName: resolved, args: args)
            switch decision {
            case .denied(let reason):
                return "Policy: \(reason)"
            case .needsApproval:
                return "Approval required before I can run \(resolved). Reply with 'approve' to proceed. (Set autopilot to skip these prompts.)"
            case .approved:
                break
            }
        }

        let timeout = policyEngine?.maxTimeout(toolName: resolved) ?? 30

        switch resolved {
        case "get_current_time":
            return getCurrentTime()
        case "read_file":
            let readMax = policyEngine?.maxSize(toolName: "read_file") ?? 10_000
            let maxChars = parseLimit(args["max_chars"], defaultValue: 10_000, maximum: readMax)
            return readFile(path: args["path"] ?? "", maxChars: maxChars)
        case "list_directory":
            return listDirectory(path: args["path"] ?? "")
        case "search_content":
            let pattern = args["pattern"] ?? args["query"] ?? ""
            let path = args["path"] ?? "~"
            return searchContent(pattern: pattern, path: path, timeout: timeout)
        case "run_shell":
            return runShell(command: args["command"] ?? "", timeout: timeout)
        case "write_file":
            return writeFile(path: args["path"] ?? "", content: args["content"] ?? "")
        case "screen_capture":
            return screenCapture(path: args["path"])
        case "run_applescript":
            return runAppleScript(script: args["script"] ?? "", timeout: timeout)
        case "list_shortcuts":
            return listShortcuts()
        case "run_shortcut":
            return runShortcut(name: args["name"] ?? "", input: args["input"], timeout: timeout)
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
        case "describe_image":
            return await describeImage(
                path: args["path"] ?? "",
                prompt: args["prompt"] ?? "Describe this image."
            )
        case "image_generation":
            return generateImage(
                prompt: args["prompt"] ?? "",
                width: Int(args["width"] ?? "") ?? 512,
                height: Int(args["height"] ?? "") ?? 512,
                steps: Int(args["steps"] ?? "") ?? 4
            )
        case "translate_text":
            return translateText(
                text: args["text"] ?? "",
                targetLanguage: args["target_language"] ?? "",
                sourceLanguage: args["source_language"] ?? "auto"
            )
        case "consolidate_memory":
            return consolidateMemory()
        case "workspace_status":
            return workspaceStatus()
        case "read_document":
            let readMax = policyEngine?.maxSize(toolName: "read_document") ?? 100_000
            let maxChars = parseLimit(args["max_chars"], defaultValue: 10_000, maximum: readMax)
            return readDocument(
                path: args["path"] ?? "",
                maxChars: maxChars
            )
        case "search_local_files":
            return searchLocalFiles(
                pattern: args["pattern"] ?? "",
                path: args["path"] ?? "~",
                timeout: timeout
            )
        case "set_session_seed":
            return setSessionSeed(seed: args["seed"] ?? "")
        case "get_session_seed":
            return getSessionSeed()
        case "git_status":
            return gitStatus(path: args["path"] ?? "")
        case "inspect_output_firewall":
            return inspectOutputFirewall(showPatterns: args["show_patterns"] ?? "false")
        case "update_output_firewall":
            return updateOutputFirewall(pattern: args["pattern"] ?? "", action: args["action"] ?? "add")
        case "undo_last":
            return undoLast(kind: args["kind"] ?? "note")
        case "self_audit":
            return selfAudit(include: args["include"] ?? "all")
        case "curious_self_improve":
            return await curiousSelfImprove(include: args["include"] ?? "all", approved: approved)
        case "repair_runtime_issue":
            return repairRuntimeIssue(issue: args["issue"] ?? "", target: args["target"])
        default:
            if let result = await executeCustomTool(name: resolved, args: args) {
                return result
            }
            return "Unknown tool: \(resolved)"
        }
    }

    /// Run a workshop custom tool if it exists. Returns nil if no matching custom
    /// tool is registered.
    private func executeCustomTool(name: String, args: [String: String]) async -> String? {
        let custom = CustomToolStore.shared.reload()
        guard let tool = custom.first(where: { $0.id == name }) else { return nil }

        let argList = tool.args.compactMap { args[$0] }
        let interpolated = CustomToolExecutor.interpolate(template: tool.command, args: argList)

        switch tool.kind.lowercased() {
        case "shell":
            return CustomToolExecutor.runShellCommand(interpolated)
        case "applescript":
            return CustomToolExecutor.runAppleScript(interpolated)
        case "shortcut":
            return runShortcut(name: tool.command, input: argList.first)
        default:
            return "Error: unknown custom tool kind '\(tool.kind)'"
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
    func readFile(path: String, maxChars: Int = 10_000) -> String {
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

        let limit = max(1, maxChars)
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

        let entries: [String]
        do {
            entries = try FileManager.default.contentsOfDirectory(atPath: resolvedPath)
        } catch {
            return "Error: could not list \(resolvedPath): \(error.localizedDescription)"
        }

        let sorted = entries.sorted()
        let limited = Array(sorted.prefix(50))
        return limited.isEmpty ? "(empty directory)" : limited.joined(separator: "\n")
    }

    /// Search for a text pattern inside files under a directory using grep.
    func searchContent(pattern: String, path: String, timeout: Int = 15) -> String {
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
            "--exclude-dir=node_modules", "--exclude-dir=Pods",
            "--", pattern, resolvedPath,
        ]

        let result = runProcess(launchPath: "/usr/bin/grep", arguments: arguments, timeout: TimeInterval(timeout))
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
    func runShell(command: String, timeout: Int = 15) -> String {
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

        // Block execution-spawning arguments (find -exec/-ok, xargs, etc.).
        if let denied = restArgs.first(where: { arg in
            let lower = arg.lowercased()
            return lower.hasPrefix("-exec") || lower.hasPrefix("-ok") ||
                   lower == ";" || lower == "{}" ||
                   lower == "xargs" || lower == "-delete"
        }) {
            return "Error: denied argument '\(denied)'"
        }

        let result = runProcess(launchPath: resolved, arguments: restArgs, timeout: TimeInterval(timeout))

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
    func runAppleScript(script: String, timeout: Int = 15) -> String {
        let source = script.trimmingCharacters(in: .whitespacesAndNewlines)
        if source.isEmpty { return "Error: no AppleScript provided" }
        if source.count > 10_000 || source.contains("\0") {
            return "Error: AppleScript is too large or contains invalid characters"
        }
        let lowered = source.lowercased()
        let denied = [
            "do shell script", "do script", "run script", "use framework",
            "do javascript", "open location",
            "current application", "current application's",
            "keystroke", "key code",
            "curl", "wget", "rm -rf",
        ]
        if let match = denied.first(where: { lowered.contains($0) }) {
            return "Error: AppleScript contains denied operation '\(match)'"
        }

        let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", source], timeout: TimeInterval(timeout))
        if result.exitCode != 0 {
            let error = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return "Error running AppleScript: \(error.isEmpty ? "osascript failed" : error)"
        }
        let output = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return output.isEmpty ? "done" : String(output.prefix(10_000))
    }

    /// List Shortcuts using Apple's fixed command-line executable.
    func listShortcuts() -> String {
        let payload: [String: Any] = ["timeout": 15]
        if let response = callAqua(command: "list_shortcuts", payload: payload, timeout: 15),
           let ok = response["ok"] as? Bool {
            if ok, let shortcuts = response["shortcuts"] as? [String] {
                return shortcuts.isEmpty ? "No shortcuts found" : shortcuts.joined(separator: "\n")
            } else if let error = response["error"] as? String {
                return "Error listing shortcuts: \(error)"
            }
        }

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
    func runShortcut(name: String, input: String?, timeout: Int = 60) -> String {
        let shortcutName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if shortcutName.isEmpty || shortcutName.count > 255 || shortcutName.contains("\0") {
            return "Error: invalid shortcut name"
        }

        var payload: [String: Any] = ["name": shortcutName, "timeout": timeout]
        if let input, !input.isEmpty {
            guard input.utf8.count <= 100_000 else { return "Error: shortcut input is too large" }
            payload["input"] = input
        }
        if let response = callAqua(command: "run_shortcut", payload: payload, timeout: TimeInterval(timeout)),
           let ok = response["ok"] as? Bool {
            if ok, let output = response["output"] as? String {
                return output.isEmpty ? "done" : String(output.prefix(20_000))
            } else if let error = response["error"] as? String {
                return "Error running shortcut: \(error)"
            }
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
            timeout: TimeInterval(timeout),
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
        let text: String
        do {
            text = try String(contentsOfFile: jailed, encoding: .utf8)
        } catch {
            return "Error: could not read working memory: \(error.localizedDescription)"
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

    /// Describe an image file using the vision provider closure, or return a
    /// placeholder when no vision engine is attached.
    func describeImage(path: String, prompt: String) async -> String {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPath.isEmpty { return "Error: no image path provided" }
        guard let jailed = jailPath(trimmedPath) else {
            return "Error: path '\(trimmedPath)' is outside allowed roots"
        }
        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: jailed, isDirectory: &isDir) {
            return "Error: \(jailed) does not exist"
        }
        if isDir.boolValue {
            return "Error: \(jailed) is a directory, not a file"
        }
        if let provider = visionProvider {
            return await provider(jailed, prompt)
        }
        return "Vision engine is not available. Install or enable the vision model to describe images."
    }

    /// Generate an image from a prompt using the local `mflux-generate-flux2`
    /// binary. Returns the path to the generated PNG or an error string.
    func generateImage(prompt: String, width: Int, height: Int, steps: Int) -> String {
        guard !prompt.isEmpty else { return "Error: prompt is required" }
        let dataDir = ProcessInfo.processInfo.environment["BADAPPLE_DATA_DIR"] ?? "/var/lib/bad_apple"
        let outDir = (dataDir as NSString).appendingPathComponent("generated_images")
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

        let model = ProcessInfo.processInfo.environment["BADAPPLE_IMAGE_MODEL"] ?? "flux2-klein-4b"
        let seed = Int.random(in: 0...1_000_000_000)
        let output = (outDir as NSString).appendingPathComponent("badapple_gen_\(seed).png")

        // Search for the mflux binary in trusted locations only. Do not honour
        // BADAPPLE_MFLUX_PATH, since an arbitrary path could point to malware.
        let candidates = [
            "/usr/local/bin/mflux-generate-flux2",
            "/opt/homebrew/bin/mflux-generate-flux2",
            "\(NSHomeDirectory())/.local/bin/mflux-generate-flux2",
        ]
        let fm = FileManager.default
        guard let exe = candidates.first(where: { fm.isExecutableFile(atPath: $0) }) else {
            return "Error: mflux-generate-flux2 not found. Install mflux with `pip install mflux` to enable image generation."
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: exe)
        task.arguments = [
            "--model", model,
            "--prompt", prompt,
            "--output", output,
            "--width", String(width),
            "--height", String(height),
            "--steps", String(steps),
            "--quantize", "4",
            "--no-metadata",
            "--low-ram",
            "--seed", String(seed),
        ]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
            try task.run()
            task.waitUntilExit()
            if task.terminationStatus != 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                let err = String(data: data, encoding: .utf8) ?? "unknown error"
                return "Image generation failed:\n\(err)"
            }
            if fm.fileExists(atPath: output) {
                return "Generated image: \(output)"
            }
            return "Image generation completed but output file was not found."
        } catch {
            return "Image generation error: \(error.localizedDescription)"
        }
    }

    /// Translate text between languages. The LLM handles translation natively,
    /// so this returns a placeholder directing the model to perform it.
    func translateText(text: String, targetLanguage: String, sourceLanguage: String) -> String {
        if text.isEmpty { return "Error: no text provided" }
        if targetLanguage.isEmpty { return "Error: no target language provided" }
        return "Translation requires the model to handle this."
    }

    /// Consolidate working memory by deduplicating lines and returning a summary.
    func consolidateMemory() -> String {
        guard let jailed = jailPath(workingMemoryPath) else {
            return "Error: working memory path is outside allowed roots"
        }
        guard FileManager.default.fileExists(atPath: jailed) else {
            return "Working memory is empty. Nothing to consolidate."
        }
        guard let text = try? String(contentsOfFile: jailed, encoding: .utf8) else {
            return "Error: could not read working memory"
        }
        if text.isEmpty { return "Working memory is empty. Nothing to consolidate." }

        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        var seen = Set<String>()
        var deduped: [String] = []
        for line in lines {
            let low = line.lowercased()
            if !seen.contains(low) {
                seen.insert(low)
                deduped.append(line)
            }
        }
        let output = deduped.joined(separator: "\n")
        do {
            try output.write(toFile: jailed, atomically: true, encoding: .utf8)
        } catch {
            return "Error writing consolidated memory: \(error.localizedDescription)"
        }
        let removed = lines.count - deduped.count
        return "Consolidated working memory: \(deduped.count) unique facts (\(removed) duplicates removed)."
    }

    /// Return the current workspace path and a brief summary.
    func workspaceStatus() -> String {
        if let ws = workspace, !ws.isEmpty {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: ws, isDirectory: &isDir) {
                return "Workspace: \(ws)\nType: \(isDir.boolValue ? "directory" : "file")"
            }
            return "Workspace: \(ws) (not found on disk)"
        }
        return "No workspace set."
    }

    /// Read a document file (txt, md, pdf, docx) with size limits.
    func readDocument(path: String, maxChars: Int) -> String {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPath.isEmpty { return "Error: no document path provided" }
        guard let jailed = jailPath(trimmedPath) else {
            return "Error: path '\(trimmedPath)' is outside allowed roots"
        }
        var isDir: ObjCBool = false
        if !FileManager.default.fileExists(atPath: jailed, isDirectory: &isDir) {
            return "Error: \(jailed) does not exist"
        }
        if isDir.boolValue {
            return "Error: \(jailed) is a directory, not a file"
        }

        let ext = (jailed as NSString).pathExtension.lowercased()
        var text: String

        switch ext {
        case "pdf":
            text = extractPDFText(path: jailed)
        case "docx":
            text = extractDOCXText(path: jailed)
        case "rtf":
            text = extractRTFText(path: jailed)
        case "txt", "md", "markdown", "json", "xml", "csv", "yaml", "yml", "log", "swift", "rs", "py", "sh", "js", "ts", "html", "css":
            guard let data = FileManager.default.contents(atPath: jailed) else {
                return "Error: could not read \(jailed)"
            }
            text = String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
                ?? ""
        default:
            guard let data = FileManager.default.contents(atPath: jailed) else {
                return "Error: could not read \(jailed)"
            }
            text = String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1)
                ?? ""
        }

        if text.isEmpty {
            return "Error: could not extract text from \(jailed)"
        }
        if text.count > maxChars {
            return String(text.prefix(maxChars)) + "\n... (\(text.count) characters total)"
        }
        return text
    }

    /// Extract text from a PDF using PDFKit (built into macOS).
    private func extractPDFText(path: String) -> String {
        // Use NSClassFromString to avoid a hard import dependency at build time;
        // PDFKit is always available on macOS but not all build contexts link it.
        guard let pdfDocClass = NSClassFromString("PDFDocument") as? NSObject.Type else {
            return "Error: PDFKit is not available on this system"
        }
        let url = URL(fileURLWithPath: path)
        let selector = NSSelectorFromString("initWithURL:")
        guard pdfDocClass.responds(to: selector) else {
            return "Error: PDFDocument does not respond to initWithURL:"
        }
        let doc = pdfDocClass.perform(selector, with: url)?.takeUnretainedValue() as? NSObject
        guard let doc = doc else { return "Error: could not open PDF at \(path)" }
        // Check page count to verify it loaded.
        let countSel = NSSelectorFromString("pageCount")
        guard doc.responds(to: countSel) else {
            return "Error: PDFDocument does not respond to pageCount"
        }
        let pageCount = doc.perform(countSel)?.takeUnretainedValue() as? Int ?? 0
        if pageCount == 0 { return "Error: PDF has 0 pages or failed to load" }
        // Extract full text via `string` property.
        let stringSel = NSSelectorFromString("string")
        guard doc.responds(to: stringSel) else {
            return "Error: PDFDocument does not respond to string"
        }
        let text = doc.perform(stringSel)?.takeUnretainedValue() as? String ?? ""
        return text
    }

    /// Extract text from a .docx file by unzipping word/document.xml and
    /// stripping XML tags. Uses Foundation's NSData compression helpers.
    private func extractDOCXText(path: String) -> String {
        guard let data = FileManager.default.contents(atPath: path) else {
            return "Error: could not read \(path)"
        }
        // .docx is a ZIP archive. We need to extract word/document.xml.
        // Use Process with `unzip` as a fallback since Foundation doesn't
        // have a built-in ZIP reader on macOS without importing Compression.
        let tmpDir = NSTemporaryDirectory() + "badapple_docx_\(UUID().uuidString)"
        do {
            try FileManager.default.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
        } catch {
            return "Error: could not create temp dir for DOCX extraction"
        }
        defer { try? FileManager.default.removeItem(atPath: tmpDir) }

        // Write the docx to a temp file and unzip it.
        let zipPath = tmpDir + "/input.docx"
        do {
            try data.write(to: URL(fileURLWithPath: zipPath))
        } catch {
            return "Error: could not write temp docx file"
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", "-q", zipPath, "word/document.xml", "-d", tmpDir]
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return "Error: unzip failed: \(error.localizedDescription)"
        }

        let xmlPath = tmpDir + "/word/document.xml"
        guard let xmlData = FileManager.default.contents(atPath: xmlPath),
              let xml = String(data: xmlData, encoding: .utf8) else {
            return "Error: could not extract word/document.xml from DOCX"
        }

        // Strip XML tags, preserving paragraph breaks.
        var text = xml
        // Convert paragraph and break tags to newlines.
        text = text.replacingOccurrences(of: "</w:p>", with: "\n")
        text = text.replacingOccurrences(of: "<w:br/>", with: "\n")
        text = text.replacingOccurrences(of: "<w:tab/>", with: "\t")
        // Remove all remaining XML tags.
        while let range = text.range(of: "<[^>]+>", options: .regularExpression) {
            text.removeSubrange(range)
        }
        // Decode XML entities.
        text = text.replacingOccurrences(of: "&amp;", with: "&")
        text = text.replacingOccurrences(of: "&lt;", with: "<")
        text = text.replacingOccurrences(of: "&gt;", with: ">")
        text = text.replacingOccurrences(of: "&quot;", with: "\"")
        text = text.replacingOccurrences(of: "&apos;", with: "'")
        // Collapse excessive blank lines.
        while text.contains("\n\n\n") {
            text = text.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Extract text from an RTF file using NSAttributedString.
    private func extractRTFText(path: String) -> String {
        guard let data = FileManager.default.contents(atPath: path) else {
            return "Error: could not read \(path)"
        }
        guard let attrStr = try? NSAttributedString(
            data: data,
            options: [.documentType: NSAttributedString.DocumentType.rtf],
            documentAttributes: nil
        ) else {
            return "Error: could not parse RTF at \(path)"
        }
        return attrStr.string
    }

    /// Search for files by name pattern in a directory using FileManager.
    func searchLocalFiles(pattern: String, path: String, timeout: Int = 15) -> String {
        if pattern.isEmpty { return "Error: no search pattern provided" }

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

        let loweredPattern = pattern.lowercased()
        var matches: [String] = []
        let deadline = Date().addingTimeInterval(TimeInterval(timeout))
        let keys: [URLResourceKey] = [.isRegularFileKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: resolvedPath),
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants],
            errorHandler: { _, _ in true }
        ) else { return "Error: could not enumerate \(resolvedPath)" }

        for case let url as URL in enumerator {
            if matches.count >= 50 { break }
            if Date() > deadline { break }
            let name = url.lastPathComponent.lowercased()
            if name.contains(loweredPattern) {
                matches.append(url.path)
            }
        }
        return matches.isEmpty ? "No files matching '\(pattern)' found." : matches.joined(separator: "\n")
    }

    /// Set the session seed for deterministic generation.
    func setSessionSeed(seed: String) -> String {
        let trimmed = seed.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "Error: no session seed provided" }
        lock.lock()
        sessionSeed = trimmed
        lock.unlock()
        return "Session seed set to '\(trimmed)'."
    }

    /// Get the current session seed.
    func getSessionSeed() -> String {
        lock.lock()
        defer { lock.unlock() }
        if let seed = sessionSeed {
            return "Session seed: \(seed)"
        }
        return "No session seed set."
    }

    /// Run `git status` in the active workspace or the given path.
    func gitStatus(path: String) -> String {
        let dir: String
        if path.isEmpty {
            if let ws = workspace, !ws.isEmpty { dir = ws } else { return "No workspace set." }
        } else if let jailed = jailPath(path) {
            dir = jailed
        } else {
            return "Error: path '\(path)' is outside allowed roots"
        }
        let fm = FileManager.default
        if !fm.fileExists(atPath: (dir as NSString).appendingPathComponent(".git")) {
            return "No git repository found at \(dir)"
        }
        let result = runProcess(launchPath: "/usr/bin/git", arguments: ["-C", dir, "status", "--short"], timeout: 30)
        if result.exitCode == 0 {
            let out = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            return out.isEmpty ? "Working tree clean." : out
        }
        return "Error: git status failed: \(result.stderr)"
    }

    /// Inspect the output firewall pattern counts and optional blocklist contents.
    func inspectOutputFirewall(showPatterns: String) -> String {
        let show = showPatterns.lowercased() == "true"
        let fm = FileManager.default
        let path = BadAppleOutputFirewall.blocklistPath
        let exists = fm.fileExists(atPath: path)

        if !exists, !isRunningCurious, BadAppleEngine.shared.curiousAutopilotLevel() != "off" {
            BadAppleEngine.shared.triggerCuriousAutopilot(reason: "output firewall blocklist missing")
        }

        let total = outputFirewall?.totalPatternCount() ?? 0
        let defaults = outputFirewall?.defaultPatternCount() ?? 0
        let blocklist = outputFirewall?.blocklistPatternCount() ?? 0

        var lines: [String] = [
            "Output firewall",
            "  blocklist path: \(path)",
            "  blocklist exists: \(exists)",
            "  total patterns: \(total)",
            "  built-in defaults: \(defaults)",
            "  blocklist patterns: \(blocklist)",
        ]

        if show {
            let contents = outputFirewall?.blocklistContents() ?? ""
            let patterns = contents
                .split(separator: "\n", omittingEmptySubsequences: true)
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            lines.append("  on-disk patterns:")
            for pattern in patterns {
                lines.append("    - \(pattern)")
            }
        }

        return lines.joined(separator: "\n")
    }

    /// Add or remove a pattern in the output firewall blocklist and reload.
    func updateOutputFirewall(pattern: String, action: String) -> String {
        let path = BadAppleOutputFirewall.blocklistPath
        let fm = FileManager.default

        let dir = (path as NSString).deletingLastPathComponent
        if !fm.fileExists(atPath: dir) {
            do {
                try fm.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: [FileAttributeKey.posixPermissions: 0o770])
            } catch {
                return "Error: cannot create blocklist directory: \(error.localizedDescription)"
            }
        }

        let action = action.lowercased()
        if action == "add" {
            var existing = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
            let patterns = existing
                .split(separator: "\n", omittingEmptySubsequences: true)
                .map { String($0).trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            if patterns.contains(pattern) {
                return "Pattern '\(pattern)' is already in the blocklist."
            }
            if !existing.isEmpty && !existing.hasSuffix("\n") {
                existing.append("\n")
            }
            existing.append(pattern + "\n")
            do {
                try existing.write(toFile: path, atomically: true, encoding: .utf8)
                outputFirewall?.reload()
                return "Added pattern '\(pattern)' to the output firewall blocklist."
            } catch {
                return "Error: cannot write blocklist: \(error.localizedDescription)"
            }
        } else if action == "remove" {
            let contents = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
            var lines = contents.split(separator: "\n", omittingEmptySubsequences: false).map { String($0) }
            let originalCount = lines.count
            lines.removeAll { line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                return !trimmed.hasPrefix("#") && trimmed == pattern
            }
            if lines.count == originalCount {
                return "Pattern '\(pattern)' was not found in the blocklist."
            }
            do {
                try lines.joined(separator: "\n").write(toFile: path, atomically: true, encoding: .utf8)
                outputFirewall?.reload()
                return "Removed pattern '\(pattern)' from the output firewall blocklist."
            } catch {
                return "Error: cannot write blocklist: \(error.localizedDescription)"
            }
        }

        return "Error: unknown action '\(action)'"
    }

    /// Undo the most recent note or image generation.
    func undoLast(kind: String) -> String {
        let kind = kind.lowercased()
        if kind == "image" {
            let dir = "/var/lib/bad_apple/generated_images"
            let fm = FileManager.default
            guard let files = try? fm.contentsOfDirectory(atPath: dir) else {
                return "No generated images to undo."
            }
            let pngs = files
                .filter { $0.hasSuffix(".png") }
                .map { (dir as NSString).appendingPathComponent($0) }
                .compactMap { path -> (String, TimeInterval)? in
                    guard let attrs = try? fm.attributesOfItem(atPath: path),
                          let date = attrs[.modificationDate] as? Date else { return nil }
                    return (path, date.timeIntervalSince1970)
                }
                .sorted { $0.1 > $1.1 }
            guard let last = pngs.first?.0 else { return "No generated images to undo." }
            do {
                try fm.removeItem(atPath: last)
                return "Removed generated image: \(last)"
            } catch {
                return "Error removing image: \(error.localizedDescription)"
            }
        } else {
            let notesDir = NSHomeDirectory() + "/.bad_apple/notes"
            let fm = FileManager.default
            try? fm.createDirectory(atPath: notesDir, withIntermediateDirectories: true)
            guard let files = try? fm.contentsOfDirectory(atPath: notesDir) else {
                return "No notes to undo."
            }
            let candidates = files
                .map { (notesDir as NSString).appendingPathComponent($0) }
                .compactMap { path -> (String, TimeInterval)? in
                    guard let attrs = try? fm.attributesOfItem(atPath: path),
                          let date = attrs[.modificationDate] as? Date else { return nil }
                    return (path, date.timeIntervalSince1970)
                }
                .sorted { $0.1 > $1.1 }
            guard let last = candidates.first?.0 else { return "No notes to undo." }
            do {
                try fm.removeItem(atPath: last)
                return "Removed note: \(last)"
            } catch {
                return "Error removing note: \(error.localizedDescription)"
            }
        }
    }

    /// Derive a ranked list of actionable runtime repairs from the runtime health check.
    /// `safe` means the repair is bounded to user-owned files or user LaunchAgents and can
    /// be attempted automatically in `safe-apply` or `full` autopilot levels.
    private func deriveRuntimeRepairs(runtime: [String: Any], dataDir: String) -> [[String: Any]] {
        var repairs: [[String: Any]] = []
        if let exists = runtime["data_dir_exists"] as? Bool, !exists {
            repairs.append([
                "issue": "data_dir_missing",
                "safe": true,
                "command": "mkdir -p \(dataDir)",
                "why": "The Bad Apple data directory is missing; create it so notes, backups, and proposals have a home.",
            ])
        }
        if let exists = runtime["blocklist_exists"] as? Bool, !exists {
            repairs.append([
                "issue": "blocklist_missing",
                "safe": true,
                "command": "touch /var/lib/bad_apple/blocklist.txt",
                "why": "The output firewall blocklist is missing; create an empty file so the firewall stops reporting it absent.",
            ])
        }
        if let loaded = runtime["identity_agent_loaded"] as? Bool, !loaded {
            repairs.append([
                "issue": "identity_agent_not_loaded",
                "safe": true,
                "command": "launchctl bootstrap gui/\(getuid())/com.badapple.identity_agent \\(NSHomeDirectory())/Library/LaunchAgents/com.badapple.identity_agent.plist",
                "why": "The identity agent is not loaded; bootstrap it so SLICKS v2 hardware-rooted identity is available.",
            ])
        }
        if let loaded = runtime["dashboard_agent_loaded"] as? Bool, !loaded {
            repairs.append([
                "issue": "dashboard_agent_not_loaded",
                "safe": true,
                "command": "launchctl bootstrap gui/\(getuid())/com.badapple.dashboard \\(NSHomeDirectory())/Library/LaunchAgents/com.badapple.dashboard.plist",
                "why": "The dashboard agent is not loaded; bootstrap it so the web Control Center is available.",
            ])
        }
        if let loaded = runtime["tts_agent_loaded"] as? Bool, !loaded {
            repairs.append([
                "issue": "tts_agent_not_loaded",
                "safe": true,
                "command": "launchctl bootstrap gui/\(getuid())/com.badapple.tts \\(NSHomeDirectory())/Library/LaunchAgents/com.badapple.tts.plist",
                "why": "The TTS agent is not loaded; bootstrap it so spoken responses work.",
            ])
        }
        if let loaded = runtime["menubar_agent_loaded"] as? Bool, !loaded {
            repairs.append([
                "issue": "menubar_agent_not_loaded",
                "safe": true,
                "command": "launchctl bootstrap gui/\(getuid())/com.badapple.menubar \\(NSHomeDirectory())/Library/LaunchAgents/com.badapple.menubar.plist",
                "why": "The menu bar agent is not loaded; bootstrap it so the menu bar is available at login.",
            ])
        }
        if let installed = runtime["app_installed"] as? Bool, !installed {
            repairs.append([
                "issue": "app_not_installed",
                "safe": false,
                "command": "",
                "why": "The Bad Apple app bundle is not in /Applications; copy it from the package to /Applications and run the installer before launchd agents can be loaded.",
            ])
        }
        return repairs
    }

    /// Execute a bounded, allowlisted runtime repair. Unknown or unsafe issues are
    /// rejected. Returns the command output or an error message.
    private func repairRuntimeIssue(issue: String, target: String?) -> String {
        let uid = String(getuid())
        let home = NSHomeDirectory()
        let dataDir = ProcessInfo.processInfo.environment["BADAPPLE_DATA_DIR"] ?? home + "/.bad_apple"

        switch issue {
        case "data_dir_missing":
            do {
                try FileManager.default.createDirectory(
                    atPath: dataDir,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
                return "Created \(dataDir)."
            } catch {
                return "Error: could not create \(dataDir): \(error.localizedDescription)"
            }
        case "blocklist_missing":
            let path = "/var/lib/bad_apple/blocklist.txt"
            let parent = (path as NSString).deletingLastPathComponent
            try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true, attributes: nil)
            if FileManager.default.createFile(atPath: path, contents: Data(), attributes: nil) {
                return "Created \(path)."
            }
            return "Error: could not create \(path)."
        case "identity_agent_not_loaded":
            let plist = home + "/Library/LaunchAgents/com.badapple.identity_agent.plist"
            let r = runProcess(launchPath: "/bin/launchctl", arguments: ["bootstrap", "gui/\(uid)", plist], timeout: 15)
            return r.exitCode == 0 ? "Loaded identity agent." : "Error: \(r.stdout + r.stderr)"
        case "dashboard_agent_not_loaded":
            let plist = home + "/Library/LaunchAgents/com.badapple.dashboard.plist"
            let r = runProcess(launchPath: "/bin/launchctl", arguments: ["bootstrap", "gui/\(uid)", plist], timeout: 15)
            return r.exitCode == 0 ? "Loaded dashboard agent." : "Error: \(r.stdout + r.stderr)"
        case "tts_agent_not_loaded":
            let plist = home + "/Library/LaunchAgents/com.badapple.tts.plist"
            let r = runProcess(launchPath: "/bin/launchctl", arguments: ["bootstrap", "gui/\(uid)", plist], timeout: 15)
            return r.exitCode == 0 ? "Loaded TTS agent." : "Error: \(r.stdout + r.stderr)"
        case "menubar_agent_not_loaded":
            let plist = home + "/Library/LaunchAgents/com.badapple.menubar.plist"
            let r = runProcess(launchPath: "/bin/launchctl", arguments: ["bootstrap", "gui/\(uid)", plist], timeout: 15)
            return r.exitCode == 0 ? "Loaded menu bar agent." : "Error: \(r.stdout + r.stderr)"
        case "app_not_installed":
            return "Cannot auto-repair: install /Applications/Bad Apple.app manually (requires administrator privileges)."
        default:
            return "Error: unknown or unsupported runtime issue '\(issue)'."
        }
    }

    /// Run the local cert suite and/or doctor diagnostic and return a JSON summary.
    func selfAudit(include: String) -> String {
        guard let binary = badappleBinaryPath() else {
            return "Error: badapple binary not found in an approved location"
        }

        let parts = include
            .lowercased()
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
        let includeCert = parts.contains("all") || parts.contains("cert")
        let includeDoctor = parts.contains("all") || parts.contains("doctor")
        let includeRuntime = parts.contains("all") || parts.contains("runtime")

        var result: [String: Any] = [:]

        if includeCert {
            let r = runProcess(launchPath: binary, arguments: ["cert"], timeout: 60)
            if r.exitCode != 0 {
                result["cert"] = ["ok": false, "error": r.stderr + r.stdout]
            } else if let data = r.stdout.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) {
                result["cert"] = json
            } else {
                result["cert"] = ["ok": true, "text": r.stdout]
            }
        }

        if includeDoctor {
            let r = runProcess(launchPath: binary, arguments: ["--doctor"], timeout: 60)
            result["doctor"] = [
                "exit_code": r.exitCode,
                "output": (r.stdout + r.stderr).prefix(20_000),
            ]
        }

        if includeRuntime {
            var runtime: [String: Any] = [:]
            let appPath = "/Applications/Bad Apple.app"
            runtime["app_installed"] = FileManager.default.fileExists(atPath: appPath)
            let badappleRunning = runProcess(launchPath: "/bin/ps", arguments: ["-ef"], timeout: 5)
            runtime["badapple_engine_running"] = badappleRunning.stdout.contains("badapple-engine")
            runtime["badapple_dashboard_running"] = badappleRunning.stdout.contains("badapple-dashboard")
            let uid = String(getuid())
            let identityAgentRunning = runProcess(launchPath: "/bin/launchctl", arguments: ["print", "gui/\(uid)/com.badapple.identity_agent"], timeout: 5)
            runtime["identity_agent_loaded"] = identityAgentRunning.exitCode == 0
            let dashboardAgentRunning = runProcess(launchPath: "/bin/launchctl", arguments: ["print", "gui/\(uid)/com.badapple.dashboard"], timeout: 5)
            runtime["dashboard_agent_loaded"] = dashboardAgentRunning.exitCode == 0
            let ttsAgentRunning = runProcess(launchPath: "/bin/launchctl", arguments: ["print", "gui/\(uid)/com.badapple.tts"], timeout: 5)
            runtime["tts_agent_loaded"] = ttsAgentRunning.exitCode == 0
            let menubarAgentRunning = runProcess(launchPath: "/bin/launchctl", arguments: ["print", "gui/\(uid)/com.badapple.menubar"], timeout: 5)
            runtime["menubar_agent_loaded"] = menubarAgentRunning.exitCode == 0
            let dataDir = ProcessInfo.processInfo.environment["BADAPPLE_DATA_DIR"] ?? NSHomeDirectory() + "/.bad_apple"
            runtime["data_dir_exists"] = FileManager.default.fileExists(atPath: dataDir)
            runtime["blocklist_exists"] = FileManager.default.fileExists(atPath: "/var/lib/bad_apple/blocklist.txt")
            result["runtime"] = runtime
            result["repairs"] = deriveRuntimeRepairs(runtime: runtime, dataDir: dataDir)
        }

        // If this audit found real-world runtime repairs and we are not already
        // inside a Curious run, wake the autopilot loop so it acts on the event
        // rather than waiting for the next timer tick.
        if !isRunningCurious,
           let repairs = result["repairs"] as? [[String: Any]],
           !repairs.isEmpty,
           BadAppleEngine.shared.curiousAutopilotLevel() != "off" {
            let issues = repairs.compactMap { $0["issue"] as? String }.joined(separator: ", ")
            BadAppleEngine.shared.triggerCuriousAutopilot(reason: "self_audit repairs: \(issues)")
        }

        guard let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted),
              let text = String(data: data, encoding: .utf8) else {
            return "Error: could not encode self-audit result"
        }
        return text
    }

    /// Run a bounded Curious self-improvement check and write a proposal note.
    /// This is the tool the Curious autopilot invokes so it can improve Bad Apple
    /// on its own without relying on the 7B model for multi-step planning.
    ///
    /// When `approved` is true (i.e. Autopilot is on), the tool will also apply the
    /// generated patch after backing up the original and verifying the replacement.
    func curiousSelfImprove(include: String, approved: Bool = false) async -> String {
        isRunningCurious = true
        defer { isRunningCurious = false }

        // Try to find the Bad Apple repo by walking up from the running binary.
        let fallbackBase = projectRootFromBinary() ?? NSHomeDirectory()
        let base = workspace ?? fallbackBase

        let proposalsDir = NSHomeDirectory() + "/.bad_apple/notes/proposed_patches"
        let backupsDir = NSHomeDirectory() + "/.bad_apple/backups"
        do {
            try FileManager.default.createDirectory(
                atPath: proposalsDir,
                withIntermediateDirectories: true,
                attributes: nil
            )
            try FileManager.default.createDirectory(
                atPath: backupsDir,
                withIntermediateDirectories: true,
                attributes: nil
            )
        } catch {
            return "Error: could not create Curious directories: \(error.localizedDescription)"
        }

        let audit = selfAudit(include: include)

        // Parse runtime repairs and attempt safe ones when the autopilot level allows.
        let level = loadAutopilotLevel()
        let shouldAttemptRepairs = level == "safe-apply" || level == "full"
        var repairs: [[String: Any]] = []
        var attemptedRepairs: [String] = []
        if let auditData = audit.data(using: .utf8),
           let auditDict = try? JSONSerialization.jsonObject(with: auditData) as? [String: Any],
           let foundRepairs = auditDict["repairs"] as? [[String: Any]] {
            repairs = foundRepairs
            for repair in foundRepairs {
                guard let issue = repair["issue"] as? String,
                      let safe = repair["safe"] as? Bool
                else { continue }
                if shouldAttemptRepairs, safe {
                    let result = repairRuntimeIssue(issue: issue, target: repair["target"] as? String)
                    attemptedRepairs.append("- \(issue): \(result)")
                    if result.hasPrefix("Loaded") || result.hasPrefix("Created") {
                        recordCuriousFeedback(file: issue, why: repair["why"] as? String ?? "", status: "repaired")
                    } else {
                        recordCuriousFeedback(file: issue, why: repair["why"] as? String ?? "", status: "failed", error: result)
                    }
                } else if !safe && level == "full" {
                    let result = repairRuntimeIssue(issue: issue, target: repair["target"] as? String)
                    attemptedRepairs.append("- \(issue): \(result)")
                } else {
                    attemptedRepairs.append("- \(issue): needs approval or autopilot level is '\(level)'")
                }
            }
        }

        let firewall = inspectOutputFirewall(showPatterns: "false")
        let git = gitStatus(path: base)
        let rawCodeSearch = searchContent(pattern: "TODO|FIXME|HACK|XXX", path: base, timeout: 30)
        let codeSearch = rawCodeSearch
            .components(separatedBy: .newlines)
            .filter { !$0.contains("searchContent(pattern:") && !$0.contains("\"TODO|FIXME|HACK|XXX\"") }
            .joined(separator: "\n")
        let sourceContext = fullFileContextForMarkers(codeSearch, window: 5)
        let acceptedExamples = recentAcceptedPatchExamples(limit: 5)
        let feedbackSummary = loadCuriousFeedback()
            .filter { ($0["status"] as? String) != nil }
            .suffix(20)
            .map { entry in
                let status = entry["status"] as? String ?? "unknown"
                let file = entry["file"] as? String ?? "unknown"
                let why = entry["why"] as? String ?? ""
                let error = entry["error"] as? String ?? ""
                return "- \(status): \(file) — \(why) \(error.isEmpty ? "" : "(error: \(error))")"
            }
            .joined(separator: "\n")

        let analysisPrompt = """
        You are the Bad Apple Curious autopilot. Analyze the project at \(base) and propose exactly ONE safe, minimal, concrete patch.

        Rules:
        - Only one patch per response. If nothing safe is obvious, output exactly {"no_patch":true}.
        - The patch must be tiny: prefer a single-line or single-function change. Never refactor more than 20 lines or 1000 characters.
        - For existing files, the "old" string must be an EXACT substring you have seen in the Source context below.
        - For new files, use "old":"" and provide the full content in "new".
        - Do NOT invent code. Do NOT guess the contents of a file you have not seen.
        - Do NOT modify core control files: BadAppleEngine.swift, BadAppleTools.swift, BadAppleEngineDaemon.swift, BadApplePolicyEngine.swift, BadAppleMenuBar.swift, BadAppleMenuBarUIResponder.swift, BadAppleConversation.swift, BadAppleTTS.swift, badapple-dashboard.rs, lib.rs.
        - Patches are applied and then immediately verified with `cargo fmt`, `cargo clippy --release --tests`, `cargo build --release`, and `cargo test --release`. They MUST be valid Rust/Swift.
        - If the patch is to /var/lib/bad_apple/blocklist.txt and the output firewall says `blocklist exists: false`, you MUST create it as an empty file.

        Runtime repairs detected and any attempts made:
        \(repairs.map { entry in
            let issue = entry["issue"] as? String ?? "unknown"
            let why = entry["why"] as? String ?? ""
            let safe = entry["safe"] as? Bool ?? false
            return "- [\(safe ? "safe" : "manual")] \(issue): \(why)"
        }.joined(separator: "\n"))

        \(attemptedRepairs.isEmpty ? "" : "Repairs already attempted:\n" + attemptedRepairs.joined(separator: "\n"))

        Recent accepted patch examples:
        \(acceptedExamples)

        Recent proposal feedback:
        \(feedbackSummary)

        Self-audit:
        \(audit)

        Output firewall:
        \(firewall)

        Git status:
        \(git)

        Source markers with context:
        \(sourceContext)

        Output ONLY a JSON object in this exact form:
        {"patch":{"file":"/absolute/path/to/file","old":"exact text to replace","new":"exact replacement text","why":"one sentence reason"}}

        Example good patch:
        {"patch":{"file":"/Users/savag3/bad_apple/src/Foo.swift","old":"let x = 1","new":"let x = 1\nlet y = 2","why":"Add missing secondary binding."}}

        If no safe patch, output exactly: {"no_patch":true}
        """

        let proposal = await BadAppleEngine.shared.generateForSelfImprovement(
            prompt: analysisPrompt,
            maxTokens: 600
        )

        var parsed = parseProposedPatch(from: proposal)

        // Deterministic fallback: if the model did not propose a patch and the
        // output firewall reports blocklist.txt missing, create it. This keeps
        // the self-improvement loop from being useless when there are obvious
        // environment issues the model may be too conservative to touch.
        let blocklistPath = "/var/lib/bad_apple/blocklist.txt"
        if parsed == nil,
           !FileManager.default.fileExists(atPath: blocklistPath),
           firewall.contains("blocklist exists: false") {
            parsed = (blocklistPath, "", "", "Create the missing output firewall blocklist file.")
        }

        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime, .withTimeZone]
        let timestamp = dateFormatter.string(from: Date()).replacingOccurrences(of: ":", with: "")
        let proposalPath = (proposalsDir as NSString).appendingPathComponent("\(timestamp)-curious-proposal.md")

        var body = "# Curious self-improvement check\n\n"
        body += "**When:** \(dateFormatter.string(from: Date()))\n\n"
        body += "**Workspace:** \(base)\n\n"
        body += "## Self-audit\n\n```json\n"
        body += audit.prefix(2_000)
        body += "\n```\n\n"
        body += "## Runtime repairs\n\n"
        if repairs.isEmpty {
            body += "No runtime repairs detected.\n\n"
        } else {
            for repair in repairs {
                let issue = repair["issue"] as? String ?? "unknown"
                let why = repair["why"] as? String ?? ""
                let safe = repair["safe"] as? Bool ?? false
                body += "- [\(safe ? "safe" : "manual")] \(issue): \(why)\n"
            }
            body += "\n"
            if !attemptedRepairs.isEmpty {
                body += "### Attempted\n\n"
                for entry in attemptedRepairs {
                    body += "\(entry)\n"
                }
                body += "\n"
            }
        }
        body += "## Output firewall\n\n"
        body += firewall
        body += "\n\n"
        body += "## Git status\n\n```\n"
        body += git
        body += "\n```\n\n"
        body += "## Source markers (TODO/FIXME/HACK/XXX)\n\n```\n"
        body += codeSearch
        body += "\n```\n\n"
        body += "## Proposal\n\n"
        body += "```json\n"
        body += proposal
        body += "\n```\n\n"

        let shouldAutoApply = approved && (level == "safe-apply" || level == "full")
        let safeMode = level == "safe-apply"

        var applyResult: String? = nil
        if let patch = parsed, shouldAutoApply {
            if safeMode, isCuriousProtectedFile(patch.file) {
                body += "A patch was proposed but not applied because level is safe-apply and \(patch.file) is a protected control file.\n"
            } else {
                applyResult = applyProposedPatch(patch, backupsDir: backupsDir, base: base)
                body += "## Applied\n\n"
                body += applyResult ?? "No patch was applied."
                body += "\n"
            }
        } else if parsed != nil, !shouldAutoApply {
            body += "A patch was proposed but not applied because Curious autopilot is set to '\(level)'.\n"
        } else {
            body += "No patch proposed.\n"
        }

        if let data = body.data(using: .utf8) {
            do {
                try data.write(to: URL(fileURLWithPath: proposalPath), options: .atomic)
            } catch {
                return "Wrote findings, but could not save proposal: \(error.localizedDescription)\n\nAudit: \(audit.prefix(500))"
            }
        }

        // If the dashboard or CLI just changed the autopilot level, make sure
        // the background loop is in the right state.
        BadAppleEngine.shared.updateCuriousAutopilotLoop()

        var summary = "Curious self-improvement check complete. Proposal written to \(proposalPath)."
        if let applyResult {
            summary += " Apply result: \(applyResult)"
        }
        return summary
    }

    private func isCuriousProtectedFile(_ path: String) -> Bool {
        let protectedNames: Set<String> = [
            "BadAppleEngine.swift",
            "BadAppleTools.swift",
            "BadAppleEngineDaemon.swift",
            "BadApplePolicyEngine.swift",
            "BadAppleMenuBar.swift",
            "BadAppleMenuBarUIResponder.swift",
            "BadAppleConversation.swift",
            "BadAppleTTS.swift",
            "badapple-dashboard.rs",
            "lib.rs",
        ]
        let filename = (path as NSString).lastPathComponent
        return protectedNames.contains(filename)
    }

    /// Load the current Curious autopilot level from disk.
    /// Valid levels: off, suggest, safe-apply, full. Default is suggest.
    private func loadAutopilotLevel() -> String {
        let path = NSHomeDirectory() + "/.bad_apple/autopilot_level"
        guard let text = try? String(contentsOfFile: path, encoding: .utf8),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return "suggest" }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    /// Locate the project root by searching for a `.git` directory above the
    /// running `badapple` binary. Falls back to `~/bad_apple` if it exists.
    private func projectRootFromBinary() -> String? {
        let home = NSHomeDirectory()
        let manualRepo = home + "/bad_apple"
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: manualRepo, isDirectory: &isDir), isDir.boolValue {
            return manualRepo
        }

        guard let binary = badappleBinaryPath() else { return nil }
        var url = URL(fileURLWithPath: binary).deletingLastPathComponent()
        for _ in 0..<6 {
            let gitDir = url.appendingPathComponent(".git").path
            if FileManager.default.fileExists(atPath: gitDir) {
                return url.path
            }
            if url.path == "/" || url.path == "/Applications" { break }
            url = url.deletingLastPathComponent()
        }
        return nil
    }

    /// Parse a patch proposal from the model's JSON output.
    private func parseProposedPatch(from text: String) -> (file: String, old: String, new: String, why: String)? {
        guard let data = text.data(using: .utf8) else { return nil }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if json["no_patch"] as? Bool == true { return nil }
        guard let patch = json["patch"] as? [String: Any],
              let file = patch["file"] as? String,
              let old = patch["old"] as? String,
              let new = patch["new"] as? String,
              let why = patch["why"] as? String,
              !file.isEmpty, !why.isEmpty
        else { return nil }
        // For new files, `old` is empty; for edits, `old` must be non-empty and
        // `new` may be empty (deletion is not allowed, but no-op is ok).
        if !old.isEmpty, new.isEmpty {
            // Disallow pure deletions.
            return nil
        }
        return (file, old, new, why)
    }

    /// Load the human feedback record for Curious proposals.
    private func loadCuriousFeedback() -> [[String: Any]] {
        let path = NSHomeDirectory() + "/.bad_apple/curious_feedback.json"
        guard FileManager.default.fileExists(atPath: path),
              let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return [] }
        return json
    }

    /// Return a short few-shot string of the last N accepted patches.
    private func recentAcceptedPatchExamples(limit: Int = 5) -> String {
        let feedback = loadCuriousFeedback()
            .filter { ($0["status"] as? String) == "accepted" }
            .suffix(limit)
        guard !feedback.isEmpty else { return "No accepted patches yet." }
        return feedback.enumerated().map { idx, entry in
            let file = entry["file"] as? String ?? "unknown"
            let why = entry["why"] as? String ?? ""
            let status = entry["status"] as? String ?? ""
            return "Example \(idx + 1) (\(status)): file \(file), reason: \(why)"
        }.joined(separator: "\n")
    }

    /// Read a window of lines around each TODO/FIXME/HACK/XXX marker.
    private func fullFileContextForMarkers(_ searchOutput: String, window: Int = 5) -> String {
        let lines = searchOutput.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var contexts: [String] = []
        var seen = Set<String>()
        for line in lines.prefix(20) {
            let parts = line.components(separatedBy: ":")
            guard parts.count >= 2 else { continue }
            let file = parts[0]
            guard let lineNumber = Int(parts[1]) else { continue }
            if seen.contains(file) { continue }
            seen.insert(file)
            guard let content = try? String(contentsOfFile: file, encoding: .utf8) else { continue }
            let all = content.components(separatedBy: .newlines)
            let start = max(0, lineNumber - window - 1)
            let end = min(all.count, lineNumber + window)
            let windowLines = all[start..<end].enumerated().map { offset, l in
                "\(start + offset + 1): \(l)"
            }.joined(separator: "\n")
            contexts.append("File: \(file)\n```\n\(windowLines)\n```")
        }
        return contexts.isEmpty ? "No source contexts." : contexts.joined(separator: "\n\n")
    }

    /// Append a line to the public Curious build log.
    private func appendCuriousBuildLog(file: String, why: String, status: String) {
        let path = NSHomeDirectory() + "/.bad_apple/CURIOUS.md"
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "- **\(timestamp)** `\(file)` — \(why) — status: \(status)\n"
        var text = (try? String(contentsOfFile: path, encoding: .utf8)) ?? "# Curious build log\n\n"
        text += line
        do {
            try text.write(toFile: path, atomically: true, encoding: .utf8)
        } catch {
            NSLog("[BadAppleTools] could not append CURIOUS.md: %@", error.localizedDescription)
        }
    }

    /// Run `cargo fmt`, `cargo clippy`, `cargo build --release`, and `cargo test --release`.
    /// Returns an error string if any step fails, or nil on success.
    private func runCargoVerification(repoRoot: String, targetFile: String) -> String? {
        guard let cargo = resolveCommand("cargo") else {
            return "cargo not found in PATH"
        }

        // Resolve the target file relative to the repo root.
        let rel = targetFile.hasPrefix(repoRoot)
            ? String(targetFile.dropFirst(repoRoot.count).trimmingCharacters(in: CharacterSet(charactersIn: "/")))
            : targetFile

        var steps: [(String, [String])] = [
            (cargo, ["fmt", "--", rel]),
            (cargo, ["fmt", "--check"]),
            (cargo, ["clippy", "--release", "--tests"]),
            (cargo, ["build", "--release"]),
            (cargo, ["test", "--release"]),
        ]
        // In consumer installs the repo may not exist; if so, skip verification.
        if !FileManager.default.fileExists(atPath: repoRoot + "/Cargo.toml") {
            steps = []
        }

        for (launchPath, args) in steps {
            let result = runProcess(launchPath: launchPath, arguments: args, timeout: 600, workingDirectory: repoRoot)
            if result.exitCode != 0 {
                return "\(launchPath) \(args.joined(separator: " ")) failed:\nstdout:\n\(result.stdout)\nstderr:\n\(result.stderr)"
            }
        }
        return nil
    }

    /// Apply a parsed patch after jail-check, backup, and verification.
    /// Returns a human-readable result or error.
    private func applyProposedPatch(
        _ patch: (file: String, old: String, new: String, why: String),
        backupsDir: String,
        base: String
    ) -> String {
        // Allow Bad Apple's own data directory in addition to the normal jail.
        let dataDir = ProcessInfo.processInfo.environment["BADAPPLE_DATA_DIR"] ?? "/var/lib/bad_apple"

        // Reject any file outside the project, home, or allowed temp/data areas.
        guard jailPath(patch.file) != nil || patch.file.hasPrefix(dataDir + "/") || patch.file == dataDir else {
            return "Refused: \(patch.file) is outside allowed roots."
        }
        guard patch.file.hasPrefix(base)
            || patch.file.hasPrefix(NSHomeDirectory())
            || patch.file.hasPrefix("/tmp/")
            || patch.file.hasPrefix("/var/tmp/")
            || patch.file.hasPrefix(dataDir + "/")
            || patch.file == dataDir
        else {
            return "Refused: \(patch.file) is outside the project or allowed directories."
        }

        // The autopilot may propose patches to any file, but it must not
        // self-modify the core control/planner/policy files without a human
        // in the loop. This keeps the self-improvement loop bounded.
        let protectedNames: Set<String> = [
            "BadAppleEngine.swift",
            "BadAppleTools.swift",
            "BadAppleEngineDaemon.swift",
            "BadApplePolicyEngine.swift",
            "BadAppleMenuBar.swift",
            "BadAppleMenuBarUIResponder.swift",
            "BadAppleConversation.swift",
            "BadAppleTTS.swift",
            "badapple-dashboard.rs",
            "lib.rs",
        ]
        let filename = (patch.file as NSString).lastPathComponent
        if protectedNames.contains(filename) {
            return "Refused: autopilot will not apply patches to core control files such as \(filename). The patch was proposed and logged for review."
        }

        let fileExists = FileManager.default.fileExists(atPath: patch.file)
        let isNewFile = patch.old.isEmpty

        if isNewFile, fileExists {
            return "Refused: proposed new file \(patch.file) already exists. Use an edit with a non-empty old string."
        }
        if !isNewFile, !fileExists {
            return "Error: \(patch.file) does not exist; cannot apply an edit to a missing file."
        }

        let original: String
        let originalData: Data?
        if fileExists {
            guard let data = FileManager.default.contents(atPath: patch.file) else {
                return "Error: could not read \(patch.file)."
            }
            guard let text = String(data: data, encoding: .utf8) else {
                return "Error: could not decode \(patch.file) as UTF-8."
            }
            original = text
            originalData = data
        } else {
            original = ""
            originalData = nil
        }

        if !isNewFile, !original.contains(patch.old) {
            return "Error: old string not found in \(patch.file). The proposal may be stale."
        }

        // Backup with a timestamped subdirectory.
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime, .withTimeZone]
        let timestamp = dateFormatter.string(from: Date()).replacingOccurrences(of: ":", with: "")
        let backupSubdir = (backupsDir as NSString).appendingPathComponent(timestamp)
        try? FileManager.default.createDirectory(atPath: backupSubdir, withIntermediateDirectories: true, attributes: nil)
        let backupPath = (backupSubdir as NSString).appendingPathComponent((patch.file as NSString).lastPathComponent)
        if let originalData {
            do {
                try originalData.write(to: URL(fileURLWithPath: backupPath), options: .atomic)
            } catch {
                return "Error: could not back up \(patch.file): \(error.localizedDescription)"
            }
        } else {
            // For new files, write an empty placeholder backup for traceability.
            do {
                try Data().write(to: URL(fileURLWithPath: backupPath), options: .atomic)
            } catch {
                NSLog("[BadAppleTools] could not write empty backup placeholder: %@", error.localizedDescription)
            }
        }

        let updated: String
        if isNewFile {
            updated = patch.new
        } else {
            guard let range = original.range(of: patch.old) else {
                return "Error: old string not found during replacement."
            }
            var working = original
            working.replaceSubrange(range, with: patch.new)
            updated = working
        }

        guard updated != original || isNewFile else {
            return "Error: replacement produced no change."
        }

        guard let newData = updated.data(using: .utf8) else {
            return "Error: could not encode updated content."
        }

        // Ensure the parent directory exists when creating a new file.
        let parent = URL(fileURLWithPath: patch.file).deletingLastPathComponent().path
        do {
            try FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true, attributes: nil)
        } catch {
            return "Error: could not create parent directory for \(patch.file): \(error.localizedDescription)"
        }

        do {
            try newData.write(to: URL(fileURLWithPath: patch.file), options: .atomic)
        } catch {
            return "Error: could not write \(patch.file): \(error.localizedDescription)"
        }

        // Verify.
        guard let verifyData = FileManager.default.contents(atPath: patch.file) else {
            return "Error: verification failed: could not read \(patch.file) after write."
        }
        let verify = String(data: verifyData, encoding: .utf8) ?? ""
        if patch.new.isEmpty {
            // Creating an empty file; verify it exists and is empty.
            guard verifyData.isEmpty, FileManager.default.fileExists(atPath: patch.file) else {
                return "Error: verification failed: \(patch.file) should be empty."
            }
        } else {
            guard verify.contains(patch.new) else {
                do {
                    try FileManager.default.removeItem(atPath: patch.file)
                } catch {
                    NSLog("[BadAppleTools] could not remove malformed patch: %@", error.localizedDescription)
                }
                if let originalData {
                    do {
                        try originalData.write(to: URL(fileURLWithPath: patch.file), options: .atomic)
                    } catch {
                        return "Error: verification failed after writing \(patch.file); rollback also failed: \(error.localizedDescription)"
                    }
                }
                recordCuriousFeedback(file: patch.file, why: patch.why, status: "failed", error: "verification failed after writing")
                return "Error: verification failed after writing \(patch.file); rolled back."
            }
        }

        // Build/test verification for source files in the project.
        if patch.file.hasPrefix(base) {
            if let error = runCargoVerification(repoRoot: base, targetFile: patch.file) {
                // Roll back.
                do {
                    try FileManager.default.removeItem(atPath: patch.file)
                } catch {
                    NSLog("[BadAppleTools] could not remove unverified patch: %@", error.localizedDescription)
                }
                if let originalData, originalData.count > 0 {
                    do {
                        try originalData.write(to: URL(fileURLWithPath: patch.file), options: .atomic)
                    } catch let writeError {
                        return "Error: verification failed; could not roll back \(patch.file): \(writeError.localizedDescription). Build error: \(error)"
                    }
                }
                recordCuriousFeedback(file: patch.file, why: patch.why, status: "failed", error: error)
                return "Error: verification failed; patch was rolled back. \(error)"
            }
        }

        recordCuriousFeedback(file: patch.file, why: patch.why, status: "accepted")
        appendCuriousBuildLog(file: patch.file, why: patch.why, status: "applied")

        return "Applied and verified patch to \(patch.file). Backup: \(backupPath). Why: \(patch.why)"
    }

    private func recordCuriousFeedback(file: String, why: String, status: String, error: String = "") {
        let path = NSHomeDirectory() + "/.bad_apple/curious_feedback.json"
        var entries = loadCuriousFeedback()
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withTimeZone]
        let entry: [String: Any] = [
            "id": UUID().uuidString,
            "file": file,
            "why": why,
            "status": status,
            "timestamp": formatter.string(from: Date()),
            "error": error,
        ]
        entries.append(entry)
        if entries.count > 100 {
            entries = Array(entries.suffix(100))
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        } catch {
            NSLog("[BadAppleTools] could not write curious feedback: %@", error.localizedDescription)
        }
    }

    /// Locate the trusted `badapple` helper binary used for self-audit and CLI calls.
    private func badappleBinaryPath() -> String? {
        let fm = FileManager.default
        let candidates: [String] = [
            Bundle.main.bundleURL
                .appendingPathComponent("Contents/Helpers/badapple")
                .path,
            (ProcessInfo.processInfo.arguments.first.map {
                (URL(fileURLWithPath: $0).deletingLastPathComponent().appendingPathComponent("badapple")).path
            } ?? ""),
            "/usr/local/bin/badapple",
            fm.currentDirectoryPath + "/target/release/badapple",
        ]
        return candidates.first { !$0.isEmpty && fm.isExecutableFile(atPath: $0) }
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
        let searchPaths = [
            "/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin",
            NSHomeDirectory() + "/.cargo/bin",
            NSHomeDirectory() + "/.local/bin",
        ]
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
        standardInput: Data? = nil,
        workingDirectory: String? = nil
    ) -> (stdout: String, stderr: String, exitCode: Int) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        if let workingDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        }

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

// MARK: - Aqua helper client

private func aquaSocketPath() -> String {
    return ProcessInfo.processInfo.environment["BADAPPLE_AQUA_SOCKET"] ?? "/var/run/badapple/aqua_helper.sock"
}

private func aquaSocketExists(_ path: String) -> Bool {
    var st = stat()
    return stat(path, &st) == 0 && (st.st_mode & S_IFMT) == S_IFSOCK
}

private func loadAquaSlicksSecret() -> Data? {
    let env = ProcessInfo.processInfo.environment
    let raw: String?
    if let secretEnv = env["BADAPPLE_SLICKS_SECRET"], !secretEnv.isEmpty {
        raw = secretEnv
    } else {
        let keyPath = env["BADAPPLE_SLICKS_KEY_PATH"] ?? "/var/lib/bad_apple/slicks.key"
        raw = try? String(contentsOfFile: keyPath, encoding: .utf8)
    }
    guard let raw = raw else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.allSatisfy({ $0.isHexDigit }) && trimmed.count >= 32 {
        return aquaHexToData(trimmed)
    }
    return trimmed.data(using: .utf8)
}

private func aquaHexToData(_ hex: String) -> Data? {
    let lower = hex.lowercased()
    guard lower.count % 2 == 0 else { return nil }
    var data = Data(capacity: lower.count / 2)
    let chars = Array(lower)
    for i in stride(from: 0, to: chars.count, by: 2) {
        guard let high = chars[i].hexDigitValue,
              let low = chars[i + 1].hexDigitValue else { return nil }
        data.append(UInt8(high * 16 + low))
    }
    return data
}

private func aquaHmacSHA256Hex(_ key: Data, _ message: String) -> String? {
    var mac = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
    let msgData = Data(message.utf8)
    key.withUnsafeBytes { keyPtr in
        msgData.withUnsafeBytes { msgPtr in
            guard let keyBase = keyPtr.baseAddress, let msgBase = msgPtr.baseAddress else { return }
            CCHmac(
                CCHmacAlgorithm(kCCHmacAlgSHA256),
                keyBase,
                key.count,
                msgBase,
                msgData.count,
                &mac
            )
        }
    }
    return mac.map { String(format: "%02x", $0) }.joined()
}

private func aquaCanonicalJSON(_ value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject(value) else { return nil }
    guard let data = try? JSONSerialization.data(
        withJSONObject: value,
        options: [.sortedKeys, .withoutEscapingSlashes]
    ) else { return nil }
    return String(data: data, encoding: .utf8)
}

private func callAqua(command: String, payload: [String: Any], timeout: TimeInterval) -> [String: Any]? {
    let path = aquaSocketPath()
    guard aquaSocketExists(path) else { return nil }
    guard let secret = loadAquaSlicksSecret(), !secret.isEmpty else { return nil }

    var body = payload
    body["command"] = command
    body["timestamp_ms"] = Int(Date().timeIntervalSince1970 * 1000)
    body["nonce"] = UUID().uuidString

    guard let material = aquaCanonicalJSON(body),
          let proof = aquaHmacSHA256Hex(secret, material) else { return nil }

    var signed = body
    signed["proof"] = proof

    guard let requestText = aquaCanonicalJSON(signed),
          let requestData = (requestText + "\n").data(using: .utf8) else { return nil }

    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return nil }
    defer { close(fd) }

    var nosigpipe: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, socklen_t(MemoryLayout<Int32>.size))

    var tv = timeval(tv_sec: __darwin_time_t(timeout), tv_usec: 0)
    var tvRecv = timeval(tv_sec: __darwin_time_t(timeout), tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tvRecv, socklen_t(MemoryLayout<timeval>.size))

    var addr = sockaddr_un()
    memset(&addr, 0, MemoryLayout<sockaddr_un>.size)
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(path.utf8)
    let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard pathBytes.count < maxPath else { return nil }
    _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
        pathBytes.withUnsafeBufferPointer { src in
            memcpy(UnsafeMutableRawPointer(dst), src.baseAddress!, pathBytes.count)
        }
    }
    addr.sun_len = UInt8(2 + pathBytes.count + 1)
    let addrLen = socklen_t(addr.sun_len)

    let connectResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            connect(fd, sockaddrPtr, addrLen)
        }
    }
    guard connectResult == 0 else { return nil }

    guard writeAll(fd, data: requestData) else { return nil }

    var buffer = Data()
    while true {
        if let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let line = Data(buffer[buffer.startIndex..<newlineIndex])
            guard let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { return nil }
            return json
        }

        var chunk = [UInt8](repeating: 0, count: 4096)
        let n = read(fd, &chunk, 4096)
        if n > 0 {
            buffer.append(chunk, count: n)
        } else if n == 0 {
            return nil
        } else {
            let e = errno
            if e == EINTR { continue }
            return nil
        }
    }
}

private func writeAll(_ fd: Int32, data: Data) -> Bool {
    var total = 0
    return data.withUnsafeBytes { raw -> Bool in
        guard let base = raw.baseAddress else { return false }
        while total < data.count {
            let n = write(fd, base.advanced(by: total), data.count - total)
            if n < 0 {
                let e = errno
                if e == EINTR { continue }
                return false
            }
            if n == 0 { return false }
            total += n
        }
        return true
    }
}
