// BadAppleSecurity — Swift translation of the Python audit ledger and output
// firewall from badapple_extras.py.
//
//   BadAppleAuditLedger     Append-only, SHA-256 hash-chained, redacted audit
//                           log written to /var/lib/bad_apple/ledger.jsonl.
//   BadAppleOutputFirewall  Blocklist-driven output filter that replaces
//                           forbidden patterns with "[Output firewall: blocked]".
//
// Both classes are marked @unchecked Sendable; internal state is guarded by
// an NSLock, mirroring the threading.RLock / fcntl locking in the Python
// original.  Hashing uses CryptoKit's SHA256.

import Foundation
import CryptoKit

// MARK: - Audit Ledger

/// Append-only, SHA-256 hash-chained audit ledger with PII / secret redaction.
///
/// Ported from `AuditLedger` in badapple_extras.py.  Every prompt, response,
/// and tool call is written to `/var/lib/bad_apple/ledger.jsonl` as one JSON
/// line.  Each entry has:
///
///   - timestamp    ISO-8601 UTC timestamp
///   - event_type   "prompt", "response", "tool_call", "error", ...
///   - data         the redacted payload (typically prompt/response/tool_call)
///   - persona      active persona name
///   - hash         SHA256(previous_hash + canonical_entry_json)
///
/// The chain is rooted at a fixed genesis hash so tampering is detectable via
/// `verify()`.  Secrets and PII (SSNs, emails, phone numbers, API keys, bearer
/// tokens, long random tokens) are redacted before anything is written.
final class BadAppleAuditLedger: @unchecked Sendable {

    // MARK: - Paths & constants

    static let directoryPath = "/var/lib/bad_apple"
    static let ledgerPath = "/var/lib/bad_apple/ledger.jsonl"
    private static let genesis = "bad-apple-genesis-v1"

    // MARK: - Redaction markers

    private static let redactedKey = "[REDACTED_KEY]"
    private static let redactedHash = "[REDACTED_HASH]"
    private static let redactedEmail = "[REDACTED_EMAIL]"
    private static let redactedSSN = "[REDACTED_SSN]"
    private static let redactedPhone = "[REDACTED_PHONE]"
    private static let redactedBearer = "[REDACTED_BEARER]"
    private static let redactedToken = "[REDACTED_TOKEN]"

    // MARK: - State

    private let lock = NSLock()

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    // MARK: - Init

    init() {
        ensureDirectory()
    }

    // MARK: - Public API

    /// Append a redacted, hash-chained entry to the ledger.
    ///
    /// - Parameters:
    ///   - eventType: "prompt", "response", "tool_call", "error", etc.
    ///   - data: payload dictionary (typically containing prompt/response/
    ///     tool_call).  Secrets and PII are redacted before writing.
    ///   - persona: active persona name.
    func append(eventType: String, data: [String: Any], persona: String) {
        lock.lock()
        defer { lock.unlock() }

        ensureDirectory()

        let prevHash = lastHash()
        let safeData = redact(sanitize(data))
        let timestamp = Self.isoFormatter.string(from: Date())

        let body: [String: Any] = [
            "timestamp": timestamp,
            "event_type": eventType,
            "data": safeData,
            "persona": persona,
        ]

        guard let bodyJSON = canonicalJSON(body) else {
            Self.log("[audit] ledger write failed: could not serialize entry body")
            return
        }

        // hash = SHA256(previous_hash + canonical_entry_json)
        let hash = sha256Hex(prevHash + bodyJSON)

        var entry = body
        entry["hash"] = hash

        guard let lineJSON = canonicalJSON(entry),
              let lineData = (lineJSON + "\n").data(using: .utf8) else {
            Self.log("[audit] ledger write failed: could not serialize entry")
            return
        }

        appendLine(lineData)
    }

    /// Verify the integrity of the entire hash chain.
    ///
    /// Returns `true` only if every entry's stored hash matches
    /// `SHA256(previous_hash + canonical_entry_json)` and the chain links back
    /// to the genesis hash.  An empty or missing ledger is trivially valid.
    func verify() -> Bool {
        guard let lines = readLines() else {
            return true
        }

        var prev = genesisHash()
        for line in lines {
            guard let lineData = line.data(using: .utf8),
                  let entry = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                  let storedHash = entry["hash"] as? String else {
                return false
            }

            // Reconstruct the canonical body that was hashed on write.
            let body: [String: Any] = [
                "timestamp": entry["timestamp"] ?? "",
                "event_type": entry["event_type"] ?? "",
                "data": entry["data"] ?? NSNull(),
                "persona": entry["persona"] ?? "",
            ]

            guard let bodyJSON = canonicalJSON(body) else {
                return false
            }

            let expected = sha256Hex(prev + bodyJSON)
            if expected != storedHash {
                return false
            }
            prev = storedHash
        }
        return true
    }

    // MARK: - Hashing

    private func genesisHash() -> String {
        sha256Hex(Self.genesis)
    }

    private func sha256Hex(_ string: String) -> String {
        let digest = SHA256.hash(data: Data(string.utf8))
        return digest.map { String(format: "%02x", Int($0)) }.joined()
    }

    /// Canonical (sorted-key, UTF-8) JSON string for deterministic hashing.
    private func canonicalJSON(_ object: Any) -> String? {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                  withJSONObject: object,
                  options: [.sortedKeys]
              ) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    // MARK: - File I/O

    private func ensureDirectory() {
        let fm = FileManager.default
        if !fm.fileExists(atPath: Self.directoryPath) {
            try? fm.createDirectory(
                atPath: Self.directoryPath,
                withIntermediateDirectories: true
            )
        }
    }

    /// Hash of the current chain tip, or the genesis hash if the ledger is
    /// empty / missing / corrupt.
    private func lastHash() -> String {
        guard let lines = readLines(),
              let last = lines.last,
              let data = last.data(using: .utf8),
              let entry = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let hash = entry["hash"] as? String else {
            return genesisHash()
        }
        return hash
    }

    private func readLines() -> [String]? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: Self.ledgerPath)),
              let content = String(data: data, encoding: .utf8) else {
            return nil
        }
        return content
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
    }

    private func appendLine(_ data: Data) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: Self.ledgerPath) {
            fm.createFile(atPath: Self.ledgerPath, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: Self.ledgerPath)) else {
            Self.log("[audit] ledger write failed: could not open ledger")
            return
        }
        // Seek to the current end of file (append) without relying on the
        // deprecated seekToEndOfFile().
        var offset: UInt64 = 0
        if let attrs = try? fm.attributesOfItem(atPath: Self.ledgerPath),
           let size = attrs[.size] as? NSNumber {
            offset = size.uint64Value
        }
        try? handle.seek(toOffset: offset)
        try? handle.write(contentsOf: data)
        try? handle.close()
    }

    // MARK: - Redaction

    /// Recursively redact secrets / PII in a JSON-compatible value.
    private func redact(_ value: Any) -> Any {
        switch value {
        case let string as String:
            return redactString(string)
        case let dict as [String: Any]:
            return dict.mapValues { redact($0) }
        case let array as [Any]:
            return array.map { redact($0) }
        default:
            return value
        }
    }

    /// If a leaf string contains a recognized secret / PII pattern, the entire
    /// string is replaced with the matching redaction marker (matching the
    /// Python `_redact_value` whole-string replacement behaviour).
    private func redactString(_ value: String) -> String {
        let nsValue = value as NSString
        let fullRange = NSRange(location: 0, length: nsValue.length)
        for rule in Self.redactionRules {
            if rule.regex.firstMatch(in: value, options: [], range: fullRange) != nil {
                return rule.replacement
            }
        }
        return value
    }

    /// Convert non-JSON-native values (Date, URL, Data) to JSON-safe strings,
    /// mirroring Python's `default=str`.  Numbers, strings, and bools pass
    /// through unchanged so they keep their native JSON types.
    private func sanitize(_ value: Any) -> Any {
        switch value {
        case let date as Date:
            return Self.isoFormatter.string(from: date)
        case let url as URL:
            return url.absoluteString
        case let data as Data:
            return data.base64EncodedString()
        case let dict as [String: Any]:
            return dict.mapValues { sanitize($0) }
        case let array as [Any]:
            return array.map { sanitize($0) }
        default:
            return value
        }
    }

    // MARK: - Redaction rules

    private struct RedactionRule {
        let regex: NSRegularExpression
        let replacement: String
    }

    private static let redactionRules: [RedactionRule] = {
        func rule(_ pattern: String, _ replacement: String) -> RedactionRule {
            let regex = try! NSRegularExpression(pattern: pattern, options: [])
            return RedactionRule(regex: regex, replacement: replacement)
        }
        return [
            // API keys (OpenAI sk-..., Google AIza...).
            rule(#"\b(sk-[a-zA-Z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{35,})"#, redactedKey),
            // PEM private-key headers.
            rule(#"-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"#, redactedKey),
            // Long hex digests (64+ chars).
            rule(#"[0-9a-fA-F]{64,}"#, redactedHash),
            // Email addresses.
            rule(#"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#, redactedEmail),
            // SSNs (xxx-xx-xxxx) or bare 9-digit SSNs.
            rule(#"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b"#, redactedSSN),
            // North-American phone numbers.
            rule(#"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"#, redactedPhone),
            // Bearer tokens.
            rule(#"[Bb]earer\s+[A-Za-z0-9_\-\.=]+"#, redactedBearer),
            // Long random-looking tokens (40+ alphanumeric / _ - chars).
            rule(#"[A-Za-z0-9_\-]{40,}"#, redactedToken),
        ]
    }()

    // MARK: - Logging

    private static func log(_ message: String) {
        // Write straight to stderr (unbuffered) to mirror print(flush=True).
        try? FileHandle.standardError.write(contentsOf: Data((message + "\n").utf8))
    }
}

// MARK: - Output Firewall

/// Blocklist-driven output firewall.
///
/// Ported from `StreamingFirewall` / `AhoCorasickAutomaton` in
/// badapple_extras.py, simplified to plain `String.range(of:)` matching (no
/// Aho-Corasick needed in Swift).  Patterns are loaded from
/// `/var/lib/bad_apple/blocklist.txt` (one per line; `#` comments and blank
/// lines are ignored) and merged with built-in defaults so the firewall is
/// useful even before a blocklist exists.
final class BadAppleOutputFirewall: @unchecked Sendable {

    // MARK: - Paths & constants

    static let blocklistPath = "/var/lib/bad_apple/blocklist.txt"
    static let blockedMarker = "[Output firewall: blocked]"

    /// Built-in defaults merged with the on-disk blocklist.
    private static let defaultPatterns: [String] = [
        "sk-",
        "ssh-rsa",
        "-----BEGIN",
        "-----END",
        "BEGIN PRIVATE KEY",
        "BEGIN OPENSSH PRIVATE KEY",
    ]

    // MARK: - State

    private let lock = NSLock()
    private(set) var patterns: [String]

    // MARK: - Init

    init() {
        var combined = Self.defaultPatterns
        combined.append(contentsOf: Self.loadBlocklist())
        self.patterns = combined
    }

    /// Reload patterns from the blocklist file (re-merged with defaults).
    func reload() {
        var combined = Self.defaultPatterns
        combined.append(contentsOf: Self.loadBlocklist())
        lock.lock()
        patterns = combined
        lock.unlock()
    }

    // MARK: - Public API

    /// One-shot scan: replace every occurrence of any blocked pattern in
    /// `text` with the blocked marker.
    func check(_ text: String) -> String {
        let snapshot = snapshotPatterns()
        let marker = Self.blockedMarker
        let structuralText = Self.structuralForm(text)
        if snapshot.contains(where: {
            let pattern = Self.structuralForm($0)
            return pattern.count >= 8 && structuralText.contains(pattern)
        }) {
            return marker
        }
        var result = text
        for pattern in snapshot {
            guard !pattern.isEmpty else { continue }
            var cursor = result.startIndex
            while cursor < result.endIndex,
                  let range = result.range(
                    of: pattern,
                    options: [.caseInsensitive],
                    range: cursor..<result.endIndex
                  ) {
                result.replaceSubrange(range, with: marker)
                // Advance past the inserted marker so the marker itself is
                // never re-scanned (avoids infinite loops if a pattern is a
                // substring of the marker).
                if let next = result.index(
                    range.lowerBound,
                    offsetBy: marker.count,
                    limitedBy: result.endIndex
                ) {
                    cursor = next
                } else {
                    cursor = result.endIndex
                }
            }
        }
        return result
    }

    /// Streaming scan: check a new `chunk` against the `accumulated` text so
    /// that patterns spanning the boundary between already-emitted text and
    /// the new chunk are caught.
    ///
    /// - Returns: `(textToEmit, blocked)`.  When no pattern is found the chunk
    ///   is emitted unchanged with `blocked = false`.  When any pattern appears
    ///   in `accumulated + chunk`, the blocked marker is emitted and
    ///   `blocked = true`.
    func checkChunk(_ chunk: String, accumulated: String) -> (String, Bool) {
        let snapshot = snapshotPatterns()
        let combined = accumulated + chunk
        let structuralCombined = Self.structuralForm(combined)
        for pattern in snapshot {
            guard !pattern.isEmpty else { continue }
            if combined.range(of: pattern, options: [.caseInsensitive]) != nil {
                return (Self.blockedMarker, true)
            }
            let structuralPattern = Self.structuralForm(pattern)
            if structuralPattern.count >= 8 && structuralCombined.contains(structuralPattern) {
                return (Self.blockedMarker, true)
            }
        }
        return (chunk, false)
    }

    // MARK: - Private

    private static func structuralForm(_ text: String) -> String {
        String(text.lowercased().unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0)
        })
    }

    private func snapshotPatterns() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return patterns
    }

    private static func loadBlocklist() -> [String] {
        guard let content = try? String(contentsOfFile: blocklistPath, encoding: .utf8) else {
            return []
        }
        return content
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }
}
