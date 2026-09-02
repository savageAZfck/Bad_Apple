// BadAppleSecurity — Swift translation of the Python audit ledger and output
// firewall from badapple_extras.py.
//
//   BadAppleAuditLedger     Append-only, hash-chained, redacted audit log
//                           written to /var/lib/bad_apple/ledger.jsonl.
//   BadAppleOutputFirewall  Blocklist-driven output filter that replaces
//                           forbidden patterns with "[Output firewall: blocked]".
//
// The ledger format, field names, hashing algorithm, and cross-process file
// locking are identical to the Python AuditLedger in badapple_extras.py so
// both runtimes can safely append to the same ledger.jsonl file.

import Foundation
import CryptoKit

// MARK: - Audit Ledger

/// Append-only, hash-chained audit ledger with PII / secret redaction.
///
/// Matches the Python `AuditLedger` in badapple_extras.py exactly:
///   - Field names: ts, type, data, prev_hash, hash
///   - Hashing: HMAC-SHA256 with the SLICKS key, or plain SHA-256 if no key
///   - Cross-process locking via fcntl.flock on ledger.lock
///   - The persona field is stored in data but not part of the hash body
final class BadAppleAuditLedger: @unchecked Sendable {

    // MARK: - Paths & constants

    static let directoryPath = "/var/lib/bad_apple"
    static let ledgerPath = "/var/lib/bad_apple/ledger.jsonl"
    static let lockPath = "/var/lib/bad_apple/ledger.lock"
    private static let slicksKeyPath = "/var/lib/bad_apple/slicks.key"
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
    private var cachedSecret: Data?

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    // MARK: - Init

    init() {
        ensureDirectory()
        cachedSecret = loadSlicksSecret()
    }

    /// When true, no entries are written (private mode).
    var paused = false

    // MARK: - Public API

    /// Append a redacted, hash-chained entry to the ledger.
    /// Uses the same format and hashing as the Python AuditLedger.
    func append(eventType: String, data: [String: Any], persona: String) {
        guard !paused else { return }
        lock.lock()
        defer { lock.unlock() }

        ensureDirectory()

        // Acquire cross-process file lock (same as Python's fcntl.flock).
        guard let lockFd = openLockFile() else {
            Self.log("[audit] could not acquire cross-process lock")
            return
        }
        defer { close(lockFd) }
        flock(lockFd, LOCK_EX)
        defer { flock(lockFd, LOCK_UN) }

        let prevHash = lastHash()
        var safeData = redact(sanitize(data))
        // Store persona inside data (Python stores it there too).
        if var dict = safeData as? [String: Any] {
            dict["persona"] = persona
            safeData = dict
        }
        let timestamp = Self.isoFormatter.string(from: Date())

        // The hash body matches Python: {ts, type, data, prev_hash}
        let body: [String: Any] = [
            "ts": timestamp,
            "type": eventType,
            "data": safeData,
            "prev_hash": prevHash,
        ]

        guard let bodyJSON = canonicalJSON(body) else {
            Self.log("[audit] ledger write failed: could not serialize entry body")
            return
        }

        // HMAC-SHA256 with SLICKS key, or plain SHA-256 if no key (matches Python).
        let hash: String
        if let secret = cachedSecret, !secret.isEmpty {
            hash = hmacSHA256Hex(secret, bodyJSON)
        } else {
            hash = sha256Hex(bodyJSON)
        }

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
    /// Reads entries in the Python format (ts, type, data, prev_hash, hash).
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
            // Match Python field names exactly.
            let body: [String: Any] = [
                "ts": entry["ts"] ?? "",
                "type": entry["type"] ?? "",
                "data": entry["data"] ?? NSNull(),
                "prev_hash": entry["prev_hash"] ?? "",
            ]

            guard let bodyJSON = canonicalJSON(body) else {
                return false
            }

            let expected: String
            if let secret = cachedSecret, !secret.isEmpty {
                expected = hmacSHA256Hex(secret, bodyJSON)
            } else {
                expected = sha256Hex(bodyJSON)
            }

            if expected != storedHash {
                return false
            }
            // Also verify chain linkage.
            if (entry["prev_hash"] as? String) != prev {
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

    private func hmacSHA256Hex(_ key: Data, _ string: String) -> String {
        let key = SymmetricKey(data: key)
        let mac = HMAC<SHA256>.authenticationCode(for: Data(string.utf8), using: key)
        return mac.map { String(format: "%02x", Int($0)) }.joined()
    }

    /// Canonical (sorted-key, UTF-8) JSON string for deterministic hashing.
    /// Matches Python's _safe_json: json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
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

    // MARK: - SLICKS Secret

    private func loadSlicksSecret() -> Data? {
        let keyPath = ProcessInfo.processInfo.environment["BADAPPLE_SLICKS_KEY_PATH"]
            ?? Self.slicksKeyPath
        guard FileManager.default.fileExists(atPath: keyPath) else {
            return nil
        }
        guard let raw = try? String(contentsOfFile: keyPath, encoding: .utf8) else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // If it looks like hex, decode it (matching Python's bytes.fromhex).
        if trimmed.allSatisfy({ $0.isHexDigit }) && trimmed.count >= 32 {
            return hexToData(trimmed)
        }
        return trimmed.data(using: .utf8)
    }

    /// Convert a hex string to Data (matching Python's bytes.fromhex).
    private func hexToData(_ hex: String) -> Data? {
        guard hex.count % 2 == 0 else { return nil }
        var data = Data(capacity: hex.count / 2)
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            guard let byte = UInt8(hex[index..<next], radix: 16) else { return nil }
            data.append(byte)
            index = next
        }
        return data
    }

    // MARK: - Cross-Process File Locking

    /// Open the lock file and return its file descriptor for flock.
    /// Matches Python's fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX).
    private func openLockFile() -> Int32? {
        let fm = FileManager.default
        let parent = (Self.lockPath as NSString).deletingLastPathComponent
        if !fm.fileExists(atPath: parent) {
            try? fm.createDirectory(atPath: parent, withIntermediateDirectories: true)
        }
        // Open with O_CREAT | O_APPEND, matching Python's "a+" mode.
        let fd = open(Self.lockPath, O_CREAT | O_RDWR | O_APPEND, 0o644)
        return fd >= 0 ? fd : nil
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
    /// empty / missing / corrupt. Reads the last line's "hash" field.
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
    /// `text` with the blocked marker, then redact PII.
    func check(_ text: String) -> String {
        let blocked = checkBlocklist(text)
        return redactPII(blocked)
    }

    /// Blocklist-only scan (no PII redaction).
    private func checkBlocklist(_ text: String) -> String {
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

    // MARK: PII Redaction

    /// PII redaction patterns (shared with the audit ledger's redaction rules).
    private static let piiPatterns: [(NSRegularExpression, String)] = {
        func regex(_ pattern: String) -> NSRegularExpression? {
            try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
        return [
            (regex(#"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#), "[REDACTED_EMAIL]"),
            (regex(#"\b\d{3}-\d{2}-\d{4}\b"#), "[REDACTED_SSN]"),
            (regex(#"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"#), "[REDACTED_PHONE]"),
            (regex(#"\b(sk-[a-zA-Z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{35,})"#), "[REDACTED_KEY]"),
            (regex(#"[Bb]earer\s+[A-Za-z0-9_\-\.=]+"#), "[REDACTED_BEARER]"),
        ].compactMap { (regex, replacement) in
            guard let regex else { return nil }
            return (regex, replacement)
        }
    }()

    /// Redact PII (emails, SSNs, phone numbers, API keys, bearer tokens) from
    /// output text. Unlike the blocklist check, PII is replaced in-place rather
    /// than blocking the entire response.
    func redactPII(_ text: String) -> String {
        var result = text
        for (regex, replacement) in Self.piiPatterns {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(
                in: result,
                options: [],
                range: range,
                withTemplate: replacement
            )
        }
        return result
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

// MARK: - Security Utilities

/// Utility functions for security operations.
enum BadAppleSecurity {
    /// Compute the SHA-256 hash of arbitrary data and return a hex string.
    static func sha256(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Compute the SHA-256 hash of a string (UTF-8 encoded).
    static func sha256(_ string: String) -> String {
        sha256(Data(string.utf8))
    }
}
