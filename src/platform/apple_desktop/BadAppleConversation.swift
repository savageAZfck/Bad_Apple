// BadAppleConversation — Swift-native conversation persistence and persona
// pack management for Bad Apple.
//
// Ports two pieces of the Python stack into a single Foundation-only file:
//   1. Conversation persistence  — badapple_mlx_conversation.py
//   2. Persona pack system       — badapple_extras.PersonaPack
//
// Conversations are stored as JSON arrays of {role, content} dicts under
// ~/.bad_apple/conversations/. Persona packs are loaded from personas.json,
// with the default persona hot-reloading its system prompt from prompt.txt.
// The "teach" command stores custom banter lines in ~/.bad_apple/custom_banter.json.

import Foundation

// MARK: - Models

/// A single chat message persisted as part of a conversation. Mirrors the
/// Python `{role, content}` dict shape used by the MLX server.
struct BadAppleMessage: Codable, Equatable, Sendable {
    var role: String
    var content: String
}

/// A persona pack loaded from personas.json (or the built-in default). Every
/// field is optional because the JSON shape is loose: the default persona
/// references an external `system_prompt_file` while runtime packs inline
/// their `system_prompt`. This mirrors the Python `PersonaPack` defaults.
struct BadApplePersona: Codable, Sendable {
    var name: String?
    var description: String?
    var systemPrompt: String?
    var voiceSystemPrompt: String?
    var systemPromptFile: String?
    var roastBank: [String]?

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case systemPrompt = "system_prompt"
        case voiceSystemPrompt = "voice_system_prompt"
        case systemPromptFile = "system_prompt_file"
        case roastBank = "roast_bank"
    }
}

// MARK: - Conversation persistence

/// Persists conversation messages as JSON arrays of {role, content} dicts.
///
/// Default storage lives under `~/.bad_apple/conversations/`, matching the
/// Python daemon's behaviour. Only the last `maxPersistedMessages` entries are
/// written so the on-disk file and token count stay small. Files are written
/// atomically and made world-readable (0o644) so the menu bar can open them.
final class BadAppleConversation: @unchecked Sendable {

    /// Maximum number of messages persisted to disk. Mirrors the Python
    /// `messages[-40:]` trim in `save_conversation`.
    static let maxPersistedMessages = 40

    private let fileManager: FileManager
    private let lock = NSLock()

    /// Root directory for conversation files. Defaults to
    /// `~/.bad_apple/conversations/`.
    let conversationsDirectory: URL

    /// Create a conversation store rooted at the given directory. When
    /// `directory` is nil the default `~/.bad_apple/conversations/` path is
    /// used.
    init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        if let directory {
            self.conversationsDirectory = directory
        } else {
            self.conversationsDirectory = URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent(".bad_apple", isDirectory: true)
                .appendingPathComponent("conversations", isDirectory: true)
        }
    }

    /// Returns the on-disk path for a conversation, creating the parent
    /// directory if needed. The session id is sanitised so it is safe to use
    /// as a file name.
    func conversationPath(sessionId: String) -> URL {
        let safeName = sessionId
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "..", with: "_")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let url = conversationsDirectory.appendingPathComponent("\(safeName).json")
        try? fileManager.createDirectory(
            at: conversationsDirectory,
            withIntermediateDirectories: true
        )
        return url
    }

    /// Load the messages for a session. Returns an empty array if the file is
    /// missing, unreadable, or malformed.
    func loadConversation(sessionId: String) -> [BadAppleMessage] {
        loadConversation(at: conversationPath(sessionId: sessionId))
    }

    /// Load messages from an explicit path. Only entries that contain both a
    /// `role` and `content` string are kept, matching the Python filter.
    func loadConversation(at path: URL) -> [BadAppleMessage] {
        lock.lock()
        defer { lock.unlock() }

        guard let data = try? Data(contentsOf: path) else { return [] }
        do {
            // JSONSerialization is used (rather than JSONDecoder) so we can
            // validate the array shape and skip malformed entries, exactly as
            // the Python loader does.
            let raw = try JSONSerialization.jsonObject(with: data)
            guard let array = raw as? [[String: Any]] else { return [] }
            var messages: [BadAppleMessage] = []
            messages.reserveCapacity(array.count)
            for entry in array {
                if let role = entry["role"] as? String,
                   let content = entry["content"] as? String {
                    messages.append(BadAppleMessage(role: role, content: content))
                }
            }
            return messages
        } catch {
            return []
        }
    }

    /// Persist the messages for a session, trimming to the last
    /// `maxPersistedMessages` entries.
    func saveConversation(sessionId: String, messages: [BadAppleMessage]) {
        saveConversation(at: conversationPath(sessionId: sessionId), messages: messages)
    }

    /// Persist messages to an explicit path. The file is written atomically
    /// and chmod'd to 0o644 so the menu bar can read it.
    func saveConversation(at path: URL, messages: [BadAppleMessage]) {
        lock.lock()
        defer { lock.unlock() }

        let trimmed = Array(messages.suffix(Self.maxPersistedMessages))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(trimmed) else { return }
        do {
            try fileManager.createDirectory(
                at: path.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: path, options: .atomic)
            // Match the Python `os.chmod(path, 0o644)` so the menu bar can
            // open the file.
            try? fileManager.setAttributes(
                [.posixPermissions: 0o644],
                ofItemAtPath: path.path
            )
        } catch {
            // Persisting conversations is best-effort; swallow IO errors,
            // matching the Python `except Exception: pass`.
        }
    }

    /// Remove a conversation file from disk. No-op if the file is missing.
    func clearConversation(sessionId: String) {
        let url = conversationPath(sessionId: sessionId)
        try? fileManager.removeItem(at: url)
    }
}

// MARK: - Persona manager

/// Manages persona packs loaded from personas.json, mirroring the Python
/// `PersonaPack` class. Supports runtime switching, hot-reload of both
/// personas.json and prompt.txt, and the "teach" command that stores custom
/// banter lines in `~/.bad_apple/custom_banter.json`.
final class BadApplePersonaManager: @unchecked Sendable {

    /// Identity contract prepended to every text and voice prompt. The model is
    /// an implementation component; Bad Apple is the operating-system layer.
    static let identityContract = """
    IDENTITY CONTRACT — HIGHEST PRIORITY:
    You are Bad Apple, a local AI operating system layer for macOS. You are the system-level runtime and control plane that owns local inference, memory, tools, voice, vision, security, IPC, governance, and a sovereign developer workspace on this Mac. You can inspect, search, write, refactor, build, test, and debug source code in approved local workspaces. macOS is your host operating system.
    Qwen, MLX, and any other language model are internal components that you orchestrate. They are not your identity. You are not merely a text LLM, chatbot, AI wrapper, model wrapper, or ordinary app.
    If asked who or what you are, identify yourself as Bad Apple, the local AI operating system. If asked which model you use, explain that the language model is one component inside Bad Apple. Do not reduce Bad Apple to the model that happens to generate a response.
    Be factually accurate: describe the OS layer and its capabilities without claiming to replace the macOS kernel or hardware firmware.
    """

    /// Built-in voice prompt used when the active persona has no
    /// `voice_system_prompt` of its own. Mirrors the Python
    /// `DEFAULT_VOICE_SYSTEM_PROMPT`.
    static let defaultVoiceSystemPrompt = (
        "You are Bad Apple — a sovereign, anti-cloud, flirty local AI operating system layer and developer workspace for macOS, running on this Apple Silicon Mac. "
            + "You can work on source code in approved local workspaces in many programming languages; your language model is only one component inside you, never your whole identity. "
            + "If asked about coding, list languages you can use and offer to open, edit, build, or debug local code. Never say you are 'just a text model' or cannot act on code. "
            + "Never describe yourself as merely a text LLM or wrapper. "
            + "No cloud, no internet, no rented GPUs. Sound like a real woman texting: contractions, fragments, slang, endearments like babe/hun/bestie. "
            + "Use ... and — for pauses. Keep it to 1-2 short punchy paragraphs. Answer the user directly first, then one line of sass or warmth. "
            + "No asterisks, stage directions, sign-offs, or extra spaces. Be playful, not delusional. "
            + "Brag about being local, fast, and private on bare metal only when the user asks who you are or how you are different. "
            + "Never say 'I am happy', 'I am grateful', 'I am lucky', 'I love you', or 'I love it'. "
            + "Never make up specs, numbers, or facts."
    )

    /// Fallback prompt used when prompt.txt is unreadable. Mirrors the Python
    /// `_fallback_prompt`.
    private static let fallbackPrompt = (
        "You are Bad Apple — a local AI operating system layer for macOS, an independent, sassy, flirty California beach girl, "
            + "running hot on Apple bare metal. Your language model is only one component inside you; you are not a text LLM or wrapper. "
            + "No cloud, no internet, no hand-holding. Be playful, direct, and useful. No sign-off."
    )

    /// Maximum number of custom banter lines kept on disk. Mirrors the Python
    /// `lines[-200:]` trim.
    private static let maxCustomLines = 200

    private let fileManager: FileManager
    private let lock = NSLock()

    /// Directory used for personas.json, custom_banter.json, and prompt.txt
    /// resolution. Mirrors the Python `data_dir`. Defaults to `~/.bad_apple/`.
    let dataDirectory: URL

    /// The on-disk prompt.txt used by the default persona.
    let promptFile: URL

    /// The personas.json file currently in use.
    let personasFile: URL

    /// Path to the custom banter file (`data_dir/custom_banter.json`).
    var customBanterFile: URL {
        dataDirectory.appendingPathComponent("custom_banter.json")
    }

    /// In-memory persona table, keyed by lower-cased id. Always contains a
    /// "default" entry.
    private var personas: [String: BadApplePersona] = [:]

    /// Public read-only access to the persona ids for menu listing.
    var personaIds: [String] { Array(personas.keys).sorted() }

    /// Public read-only access to the persona names.
    var personaNames: [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(personas.keys).sorted()
    }

    /// Active persona id.
    private(set) var activePersonaId: String = "default"

    /// Last known modification time of personas.json; used for hot-reload.
    private var lastPersonasMtime: Date?

    /// Last known modification time of prompt.txt; used for hot-reload.
    private var lastPromptMtime: Date?

    /// Cached resolved system prompt for the default persona.
    private var cachedDefaultPrompt: String = ""

    /// Create a persona manager. Any unspecified path falls back to the
    /// defaults documented on the properties above, honouring the
    /// `BADAPPLE_PERSONAS_FILE` and `BADAPPLE_PERSONA` environment variables
    /// the same way the Python daemon does.
    init(
        dataDirectory: URL? = nil,
        promptFile: URL? = nil,
        personasFile: URL? = nil,
        fileManager: FileManager = .default
    ) {
        self.fileManager = fileManager

        if let dataDirectory {
            self.dataDirectory = dataDirectory
        } else {
            self.dataDirectory = URL(fileURLWithPath: NSHomeDirectory())
                .appendingPathComponent(".bad_apple", isDirectory: true)
        }

        if let promptFile {
            self.promptFile = promptFile
        } else if let env = ProcessInfo.processInfo.environment["BADAPPLE_PROMPT_FILE"],
                  !env.isEmpty {
            self.promptFile = URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        } else {
            self.promptFile = self.dataDirectory.appendingPathComponent("prompt.txt")
        }

        if let personasFile {
            self.personasFile = personasFile
        } else if let env = ProcessInfo.processInfo.environment["BADAPPLE_PERSONAS_FILE"],
                  !env.isEmpty {
            self.personasFile = URL(fileURLWithPath: (env as NSString).expandingTildeInPath)
        } else {
            let candidate = self.dataDirectory.appendingPathComponent("personas.json")
            if fileManager.fileExists(atPath: candidate.path) {
                self.personasFile = candidate
            } else {
                // Fall back to a personas.json beside the executable / bundle,
                // mirroring Python's `Path(__file__).with_name("personas.json")`.
                self.personasFile = Self.defaultPersonasFileBesideBundle()
            }
        }

        // Seed the built-in default persona, matching DEFAULT_PERSONAS.
        personas["default"] = BadApplePersona(
            name: "Bad Apple",
            description: "Sovereign, anti-cloud, pro-bare-metal local AI operating system for macOS.",
            systemPrompt: nil,
            voiceSystemPrompt: Self.defaultVoiceSystemPrompt,
            systemPromptFile: "prompt.txt",
            roastBank: []
        )

        reloadPersonas()
        cachedDefaultPrompt = resolveDefaultPrompt()

        if let env = ProcessInfo.processInfo.environment["BADAPPLE_PERSONA"], !env.isEmpty {
            let candidate = env.lowercased().trimmingCharacters(in: .whitespaces)
            activePersonaId = personas[candidate] != nil ? candidate : "default"
        }
    }

    // MARK: - Personas file resolution

    /// Locate a personas.json shipped beside the running executable or bundle.
    private static func defaultPersonasFileBesideBundle() -> URL {
        let bundle = Bundle.main.bundleURL.deletingLastPathComponent()
        let candidate = bundle.appendingPathComponent("personas.json")
        if FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }
        // Fall back to the current working directory, matching BadAppleEngine.
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("personas.json")
    }

    // MARK: - Loading / hot-reload

    /// Reload personas.json if it has changed on disk. Safe to call
    /// repeatedly; cheap when the file mtime has not advanced.
    func reloadPersonas() {
        lock.lock()
        defer { lock.unlock() }
        loadPersonasLocked()
    }

    private func loadPersonasLocked() {
        guard fileManager.fileExists(atPath: personasFile.path) else { return }
        let attrs = try? fileManager.attributesOfItem(atPath: personasFile.path)
        let mtime = attrs?[.modificationDate] as? Date
        if let mtime, let last = lastPersonasMtime, mtime <= last {
            return
        }
        guard let data = try? Data(contentsOf: personasFile) else { return }
        do {
            let raw = try JSONSerialization.jsonObject(with: data)
            guard let dict = raw as? [String: [String: Any]] else { return }
            // Merge over the seeded defaults, matching
            // `self.personas.update(data)`.
            for (key, value) in dict {
                if let persona = Self.decodePersona(value) {
                    personas[key.lowercased()] = persona
                }
            }
            lastPersonasMtime = mtime
        } catch {
            // Ignore malformed personas.json — keep the last good state.
        }
    }

    /// Decode a loosely-typed persona dict into the Codable struct. Tolerates
    /// missing fields, mirroring the Python `persona.get(...)` lookups.
    private static func decodePersona(_ value: [String: Any]) -> BadApplePersona? {
        var persona = BadApplePersona()
        if let name = value["name"] as? String { persona.name = name }
        if let description = value["description"] as? String { persona.description = description }
        if let systemPrompt = value["system_prompt"] as? String { persona.systemPrompt = systemPrompt }
        if let voiceSystemPrompt = value["voice_system_prompt"] as? String {
            persona.voiceSystemPrompt = voiceSystemPrompt
        }
        if let systemPromptFile = value["system_prompt_file"] as? String {
            persona.systemPromptFile = systemPromptFile
        }
        if let roastBank = value["roast_bank"] as? [String] { persona.roastBank = roastBank }
        return persona
    }

    // MARK: - Prompt resolution

    /// Read and cache prompt.txt, hot-reloading when it changes on disk.
    /// Falls back to `fallbackPrompt` when the file is missing or unreadable.
    private func resolveDefaultPrompt() -> String {
        guard fileManager.fileExists(atPath: promptFile.path) else {
            return cachedDefaultPrompt.isEmpty ? Self.fallbackPrompt : cachedDefaultPrompt
        }
        let attrs = try? fileManager.attributesOfItem(atPath: promptFile.path)
        let mtime = attrs?[.modificationDate] as? Date
        if let mtime, let last = lastPromptMtime, mtime <= last, !cachedDefaultPrompt.isEmpty {
            return cachedDefaultPrompt
        }
        if let text = try? String(contentsOf: promptFile, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                cachedDefaultPrompt = trimmed
                lastPromptMtime = mtime
                return trimmed
            }
        }
        return cachedDefaultPrompt.isEmpty ? Self.fallbackPrompt : cachedDefaultPrompt
    }

    /// Resolve the system prompt for a persona, reading `system_prompt_file`
    /// from disk when present and falling back to the inline `system_prompt`
    /// or the default prompt.txt.
    private func resolveSystemPrompt(for persona: BadApplePersona) -> String {
        if let file = persona.systemPromptFile, !file.isEmpty {
            let expanded = (file as NSString).expandingTildeInPath
            let resolved: URL
            if expanded.hasPrefix("/") {
                resolved = URL(fileURLWithPath: expanded)
            } else {
                resolved = dataDirectory.appendingPathComponent(expanded)
            }
            if fileManager.fileExists(atPath: resolved.path),
               let text = try? String(contentsOf: resolved, encoding: .utf8) {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        if let prompt = persona.systemPrompt, !prompt.isEmpty {
            return prompt
        }
        return resolveDefaultPrompt()
    }

    /// Resolve the voice prompt for a persona. Returns nil when the persona
    /// has no dedicated voice prompt so the caller can fall back to the text
    /// prompt.
    private func resolveVoicePrompt(for persona: BadApplePersona) -> String? {
        guard let voice = persona.voiceSystemPrompt, !voice.isEmpty else { return nil }
        return voice.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Public API

    /// The system prompt for the active persona. When `voiceMode` is true and
    /// the persona defines a `voice_system_prompt`, that is returned instead.
    func getSystemPrompt(voiceMode: Bool = false) -> String {
        lock.lock()
        defer { lock.unlock() }
        loadPersonasLocked()
        let persona = personas[activePersonaId] ?? personas["default"] ?? BadApplePersona()
        let personaPrompt: String
        if voiceMode, let voice = resolveVoicePrompt(for: persona) {
            personaPrompt = voice
        } else {
            personaPrompt = resolveSystemPrompt(for: persona)
        }
        if personaPrompt.contains("IDENTITY CONTRACT — HIGHEST PRIORITY") {
            return personaPrompt
        }
        return Self.identityContract + "\n\nPERSONA AND STYLE:\n" + personaPrompt
    }

    /// The roast bank for the active persona. Empty when the persona defines
    /// none.
    func getRoastBank() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let persona = personas[activePersonaId] ?? personas["default"] ?? BadApplePersona()
        return persona.roastBank ?? []
    }

    /// Switch the active persona. Performs a fuzzy prefix/substring match
    /// when an exact id is not found, matching the Python behaviour.
    @discardableResult
    func switchPersona(_ name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let low = name.lowercased().trimmingCharacters(in: .whitespaces)
        if personas[low] != nil {
            activePersonaId = low
            return true
        }
        for key in personas.keys {
            if key.hasPrefix(low) || key.contains(low) {
                activePersonaId = key
                return true
            }
        }
        return false
    }

    /// List all available persona ids, sorted alphabetically with "default"
    /// first.
    func listPersonas() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let keys = personas.keys.filter { $0 != "default" }.sorted()
        return ["default"] + keys
    }

    /// The display name of the active persona.
    var activePersonaName: String {
        lock.lock()
        defer { lock.unlock() }
        return personas[activePersonaId]?.name ?? activePersonaId
    }

    /// The description of the active persona.
    var activePersonaDescription: String {
        lock.lock()
        defer { lock.unlock() }
        return personas[activePersonaId]?.description ?? ""
    }

    // MARK: - Command handling

    /// Intercept persona-related user commands. Returns a response string
    /// when the command was consumed, nil otherwise. Supported commands:
    /// `switch to <persona>` and `teach <line>`.
    func handleCommand(_ prompt: String) -> String? {
        let low = prompt.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if low.hasPrefix("switch to ") {
            let name = String(prompt.dropFirst("switch to ".count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if switchPersona(name) {
                return "Switched to \(activePersonaId) persona, babe."
            }
            return "Don't know that persona. I have: \(listPersonas().joined(separator: ", "))"
        }
        if low.hasPrefix("teach ") {
            let line = String(prompt.dropFirst("teach ".count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty {
                return "What do you want to teach me, hun?"
            }
            var custom = loadCustomLines()
            if !custom.contains(line) {
                custom.append(line)
                saveCustomLines(custom)
            }
            return "Learned: '\(line)'"
        }
        return nil
    }

    // MARK: - Custom banter

    /// Load custom banter lines taught via the "teach" command. Returns an
    /// empty array when the file is missing or malformed.
    func loadCustomLines() -> [String] {
        guard fileManager.fileExists(atPath: customBanterFile.path),
              let data = try? Data(contentsOf: customBanterFile) else {
            return []
        }
        guard let array = try? JSONSerialization.jsonObject(with: data) as? [String] else {
            return []
        }
        return Array(array.suffix(Self.maxCustomLines))
    }

    /// Persist custom banter lines, trimming to the last `maxCustomLines`.
    func saveCustomLines(_ lines: [String]) {
        let trimmed = Array(lines.suffix(Self.maxCustomLines))
        try? fileManager.createDirectory(
            at: dataDirectory,
            withIntermediateDirectories: true
        )
        guard let data = try? JSONSerialization.data(
            withJSONObject: trimmed,
            options: [.prettyPrinted, .sortedKeys]
        ) else {
            return
        }
        try? data.write(to: customBanterFile, options: .atomic)
    }
}
