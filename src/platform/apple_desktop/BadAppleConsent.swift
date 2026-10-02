// BadAppleConsent — append-only consent registry for capture surfaces.
//
// Ambient ears and meeting recording capture people who never agreed to
// anything. This organ keeps an operator-attested record of who consented to
// which scope (ambient_hearing, meeting, voice) at
// /var/lib/bad_apple/consent.jsonl. Every mutation is also mirrored into the
// audit ledger as a `consent_*` event, so the operational log is anchored in
// the hash chain. Capture paths check the registry and record an
// attestation when they start with no grants on file.

import Foundation

struct ConsentRecord: Codable {
    let ts: String
    let action: String   // "granted" | "revoked" | "attested"
    let subject: String  // who ("alice", "household", "operator")
    let scope: String    // "ambient_hearing" | "meeting" | "voice" | free-form
    let note: String
}

final class BadAppleConsent: @unchecked Sendable {
    static let shared = BadAppleConsent()

    static let path = "/var/lib/bad_apple/consent.jsonl"
    private let lock = NSLock()

    private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    /// Append a consent record. Returns false if serialization fails.
    @discardableResult
    func record(action: String, subject: String, scope: String, note: String = "") -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let rec = ConsentRecord(
            ts: Self.isoFormatter.string(from: Date()),
            action: action, subject: subject, scope: scope, note: note
        )
        guard let data = try? JSONEncoder().encode(rec) else { return false }
        let dir = (Self.path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: Self.path) {
            FileManager.default.createFile(atPath: Self.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: Self.path))
        else { return false }
        defer { try? handle.close() }
        handle.seekToEndOfFile()
        handle.write(data + Data([0x0A]))
        return true
    }

    /// Fold the log into currently-active grants: (subject, scope) pairs
    /// whose latest record is "granted" (or an unrevoked "attested" scope
    /// assertion by the operator).
    func current() -> [(subject: String, scope: String)] {
        lock.lock()
        defer { lock.unlock() }
        var state: [String: String] = [:]  // "subject|scope" -> action
        guard let text = try? String(contentsOfFile: Self.path, encoding: .utf8) else {
            return []
        }
        for line in text.split(separator: "\n") {
            guard let rec = try? JSONDecoder().decode(ConsentRecord.self, from: Data(line.utf8))
            else { continue }
            state["\(rec.subject)|\(rec.scope)"] = rec.action
        }
        return state.compactMap { key, action in
            guard action == "granted" || action == "attested" else { return nil }
            let parts = key.split(separator: "|", maxSplits: 1).map(String.init)
            return parts.count == 2 ? (parts[0], parts[1]) : nil
        }
    }

    /// Subjects with an active grant for a scope.
    func activeSubjects(scope: String) -> [String] {
        current().filter { $0.scope == scope }.map(\.subject).sorted()
    }
}
