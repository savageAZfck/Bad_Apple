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
            name: "undo_last",
            description: "Undo the most recent user action by removing the last appended note or generated image.",
            parameters: [
                .init(name: "kind", description: "What to undo: 'note' or 'image'. Default is 'note'.", required: false),
            ],
            requiresApproval: true
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
        (["undo", "delete last", "remove last"], ["undo_last"]),
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

    /// Return tool schemas in the chat-template format expected by
    /// `BadAppleInference.generateWithTools`.
    func toolSchemasForPrompt(text: String) -> [[String: Any]]? {
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

// MARK: - BadAppleToolExecutor

/// Executes tool calls with filesystem jail protection and policy enforcement.
final class BadAppleToolExecutor: @unchecked Sendable {

    // MARK: - State

    private let lock = NSLock()
    private var _workspace: String?
    private let policyEngine: BadApplePolicyEngine?
    private let startedAt = Date()
    private var sessionSeed: String?

    /// Optional vision provider closure. When set, `describe_image` delegates
    /// to this closure with (path, prompt) and returns the description.
    var visionProvider: ((String, String) async -> String)?

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
            case .denied(let reason):
                return "Policy: \(reason)"
            case .needsApproval:
                return "Approval required before I can run \(name). Reply with 'approve' to proceed. (Set autopilot to skip these prompts.)"
            case .approved:
                break
            }
        }

        let timeout = policyEngine?.maxTimeout(toolName: name) ?? 30

        switch name {
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
        case "undo_last":
            return undoLast(kind: args["kind"] ?? "note")
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

        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: resolvedPath) else {
            return "Error: could not list \(resolvedPath)"
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
            "current application's", "curl", "wget", "rm -rf",
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

        // Search for the mflux binary.
        let env = ProcessInfo.processInfo.environment
        let candidates = [
            (env["BADAPPLE_MFLUX_PATH"] ?? ""),
            "mflux-generate-flux2",
            "/opt/homebrew/bin/mflux-generate-flux2",
            "/usr/local/bin/mflux-generate-flux2",
            (env["HOME"] ?? "/") + "/.local/bin/mflux-generate-flux2",
        ]
        let fm = FileManager.default
        guard let exe = candidates.first(where: { !$0.isEmpty && fm.fileExists(atPath: $0) }) else {
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
