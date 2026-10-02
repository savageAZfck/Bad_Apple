// BadAppleErasure — certified subject erasure with crypto-shredding.
//
// `forget <name>` / `erase <name>` now does four things: revokes consent
// grants (never silently deleted), purges incidental mentions from the
// free-text stores, seals-and-shreds the human-layer person record by
// destroying its per-subject DEK, and issues a signed certificate that
// verifies — not asserts — irrecoverability: after destruction the code
// attempts a decrypt and the certificate carries `decrypt_verified: true`
// only when that attempt provably fails.
//
// Why crypto-shredding exists here: respawn can revert both state roots,
// which resurrects any plaintext "deleted" data. Keys and destroyed-subject
// tombstones live in ~/.bad_apple_keys — outside both snapshot
// roots — so a revert can restore ciphertext but never the key, and can
// never restore the memory of the erasure itself. The erasure certificate
// hash still anchors into the audit chain; the ledger only ever carries
// the hashed subject reference.

import Foundation
import CryptoKit

final class BadAppleErasure: @unchecked Sendable {
    static let shared = BadAppleErasure()

    static let certificateDir = "/var/lib/bad_apple/erasure"
    private let lock = NSLock()

    private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private func subjectRef(_ subject: String) -> String {
        BadAppleSubjectVault.subjectRef(subject)
    }

    /// Re-apply incidental-mention purges for every destroyed subject.
    /// Runs at human-layer init: if respawn reverted state past an erasure,
    /// the tombstones still live in the key vault and the mentions die again.
    func replayForgotten() {
        let names = BadAppleSubjectVault.shared.replayNames()
        for name in names where !name.isEmpty {
            _ = purgeIncidentalMentions(subject: name)
        }
    }

    /// Purge + shred + certify. Returns a human-readable summary.
    func purge(subject rawSubject: String, persona: String) -> String {
        let subject = rawSubject.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !subject.isEmpty else { return "Forget who? `forget <name>`." }
        lock.lock()
        defer { lock.unlock() }

        var report: [String: Any] = [:]
        var removed: [String] = []
        let vault = BadAppleSubjectVault.shared

        // 0. Resolve the person record up front — the DEK is keyed by person
        //    id, and we capture a sealed probe blob now so we can prove
        //    decrypt-failure after the key is destroyed.
        let person = BadAppleHumanLayer.shared.snapshot().people.first {
            $0.name.lowercased() == subject.lowercased()
                || $0.id == subject || $0.id.hasPrefix(subject)
        }
        var probeBox: BadAppleSealedBox?
        var dekArmed = false
        if let person, !person.shredded {
            let pii: [String: String] = [
                "name": person.name,
                "relationship": person.relationship,
                "notes": person.notes,
            ]
            if let plain = try? JSONSerialization.data(
                withJSONObject: pii, options: [.sortedKeys]),
               let box = vault.seal(plain, keyID: person.id), !box.blob.isEmpty {
                probeBox = box
                dekArmed = true
            }
        }

        // 1. Human-layer person record.
        if let removedPerson = try? BadAppleHumanLayer.shared.forgetPerson(idOrName: subject),
           removedPerson {
            removed.append("person_record")
        }

        // 2-4. Incidental mentions in free-text stores + ambient/meeting captures.
        let incidental = purgeIncidentalMentions(subject: subject)
        for (key, value) in incidental { report[key] = value }
        removed.append(contentsOf: incidental["__removed__"] as? [String] ?? [])

        // 5. Consent grants — revoked, not erased (the erasure of the grant
        //    itself is part of the record).
        let scopes = BadAppleConsent.shared.current()
            .filter { $0.subject.localizedCaseInsensitiveContains(subject) }
            .map(\.scope)
        for scope in scopes {
            BadAppleConsent.shared.record(
                action: "revoked", subject: subject, scope: scope, note: "subject erasure")
        }
        if !scopes.isEmpty {
            removed.append("consent_revoked:\(scopes.count)")
            report["consent_revoked"] = scopes
        }

        // 6. Key destruction — the crypto-shred. The DEK lives outside both
        //    respawn roots: no state revert can bring it back. Verify the
        //    math by attempting a decrypt that must now fail.
        var dekDestroyed = false
        var decryptVerified = false
        if let person, dekArmed {
            dekDestroyed = vault.destroyKey(person.id)
            if dekDestroyed, let probeBox {
                decryptVerified = vault.open(probeBox, keyID: person.id) == nil
            }
            if dekDestroyed {
                removed.append("subject_dek_destroyed")
                vault.recordDestroyed(
                    subjectRef: subjectRef(subject), name: subject)
                BadAppleEngine.shared.auditDaemonEvent(type: "dek_destroyed", data: [
                    "subject_ref": subjectRef(subject),
                    "key": "subject-dek",
                    "verify": decryptVerified ? "decrypt-fails" : "unverified",
                ])
            }
        }

        // 7. Certificate — signed if the identity agent is up.
        let ref = subjectRef(subject)
        var cert: [String: Any] = [
            "kind": "badapple.erasure_certificate",
            "version": 2,
            "issued_at": Self.isoFormatter.string(from: Date()),
            "subject_ref": ref,
            "method": dekDestroyed
                ? "crypto-shred+purge+consent-revocation+signed_certificate"
                : "purge+revocation+signed_certificate",
            "dek_destroyed": dekDestroyed,
            "decrypt_verified": decryptVerified,
            "purged": removed,
            "detail": report,
        ]
        if let body = canonicalJSON(cert),
           IdentityAgentClient.shared.isAvailable,
           let sig = IdentityAgentClient.shared.sign(message: body) {
            cert["signature"] = sig
            cert["signature_scheme"] = "secure-enclave"
            cert["signed"] = true
        } else {
            cert["signed"] = false
        }

        try? FileManager.default.createDirectory(
            atPath: Self.certificateDir, withIntermediateDirectories: true)
        let stamp = Self.isoFormatter.string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: ".", with: "")
        let certPath = "\(Self.certificateDir)/\(stamp)-\(ref).json"
        var certHash = ""
        if let certData = canonicalJSON(cert) {
            certHash = SHA256.hash(data: certData)
                .map { String(format: "%02x", $0) }.joined()
            if let pretty = try? JSONSerialization.data(
                withJSONObject: cert, options: [.prettyPrinted, .sortedKeys]) {
                try? pretty.write(to: URL(fileURLWithPath: certPath), options: .atomic)
            }
        }

        BadAppleEngine.shared.auditDaemonEvent(type: "erasure_certified", data: [
            "subject_ref": ref,
            "certificate_sha256": certHash,
            "certificate": certPath,
            "signed": cert["signed"] as? Bool == true ? "true" : "false",
            "dek_destroyed": dekDestroyed ? "true" : "false",
            "purged": removed.joined(separator: ","),
        ])

        let summary = removed.isEmpty ? "nothing found referencing them" : removed.joined(separator: ", ")
        var reply = "Erased \(subject): \(summary). Certificate: \(certPath)"
            + (cert["signed"] as? Bool == true ? " (signed)" : " (unsigned)")
        if dekDestroyed {
            reply += decryptVerified
                ? " — subject key destroyed and verified: sealed copies are unreadable everywhere, including state snapshots."
                : " — subject key destroyed (decrypt-failure unverified)."
        }
        return reply + " — subject is recorded as a hash reference; the name does not re-enter the ledger."
    }

    /// Incidental mentions: working-memory lines, notes files, conversation
    /// logs, ambient hearing percepts, and meeting transcripts. Returns a
    /// detail report plus a "__removed__" key listing what was purged.
    @discardableResult
    private func purgeIncidentalMentions(subject: String) -> [String: Any] {
        var report: [String: Any] = [:]
        var removed: [String] = []

        // Working memory lines mentioning the subject.
        let wmPath = NSHomeDirectory() + "/.bad_apple/working_memory.txt"
        if let text = try? String(contentsOfFile: wmPath, encoding: .utf8) {
            let lines = text.components(separatedBy: "\n")
            let kept = lines.filter { !$0.localizedCaseInsensitiveContains(subject) }
            let dropped = lines.count - kept.count
            if dropped > 0 {
                try? kept.joined(separator: "\n")
                    .write(toFile: wmPath, atomically: true, encoding: .utf8)
                removed.append("working_memory_lines:\(dropped)")
                report["working_memory_lines"] = dropped
            }
        }

        // Notes files named for the subject.
        let notesDir = NSHomeDirectory() + "/.bad_apple/notes"
        if let files = try? FileManager.default.contentsOfDirectory(atPath: notesDir) {
            let matches = files.filter { $0.localizedCaseInsensitiveContains(subject) }
            for f in matches {
                try? FileManager.default.removeItem(atPath: notesDir + "/" + f)
            }
            if !matches.isEmpty {
                removed.append("notes_files:\(matches.count)")
                report["notes_files"] = matches
            }
        }

        // Conversation lines mentioning the subject.
        let convDir = NSHomeDirectory() + "/.bad_apple/conversations"
        var convLinesRemoved = 0
        if let files = try? FileManager.default.contentsOfDirectory(atPath: convDir) {
            for f in files where f.hasSuffix(".jsonl") || f.hasSuffix(".json") || f.hasSuffix(".txt") {
                let p = convDir + "/" + f
                guard let text = try? String(contentsOfFile: p, encoding: .utf8) else { continue }
                let lines = text.components(separatedBy: "\n")
                let kept = lines.filter { !$0.localizedCaseInsensitiveContains(subject) }
                let dropped = lines.count - kept.count
                if dropped > 0 {
                    try? kept.joined(separator: "\n")
                        .write(toFile: p, atomically: true, encoding: .utf8)
                    convLinesRemoved += dropped
                }
            }
        }
        if convLinesRemoved > 0 {
            removed.append("conversation_lines:\(convLinesRemoved)")
            report["conversation_lines"] = convLinesRemoved
        }

        // Ambient hearing percepts + meeting transcripts: redact the name
        // inside the records rather than dropping whole captures.
        var redacted = 0
        let heardPath = NSHomeDirectory() + "/.bad_apple/ambient_heard.json"
        if let text = try? String(contentsOfFile: heardPath, encoding: .utf8) {
            let out = redactSubject(in: text, subject: subject)
            if out != text {
                try? out.write(toFile: heardPath, atomically: true, encoding: .utf8)
                redacted += 1
            }
        }
        let meetDir = NSHomeDirectory() + "/.bad_apple/meetings"
        if let files = try? FileManager.default.contentsOfDirectory(atPath: meetDir) {
            for f in files where f.hasSuffix(".json") {
                let p = meetDir + "/" + f
                guard let text = try? String(contentsOfFile: p, encoding: .utf8) else { continue }
                let out = redactSubject(in: text, subject: subject)
                if out != text {
                    try? out.write(toFile: p, atomically: true, encoding: .utf8)
                    redacted += 1
                }
            }
        }
        if redacted > 0 {
            removed.append("capture_records_redacted:\(redacted)")
            report["capture_records_redacted"] = redacted
        }

        report["__removed__"] = removed
        return report
    }

    /// Case-insensitive literal redaction — replaces every occurrence of the
    /// subject name with a tombstone marker, preserving JSON structure.
    private func redactSubject(in text: String, subject: String) -> String {
        var out = text
        var search = out.startIndex..<out.endIndex
        while let range = out.range(of: subject, options: .caseInsensitive, range: search) {
            out.replaceSubrange(range, with: "[forgotten]")
            let next = out.index(range.lowerBound, offsetBy: "[forgotten]".count)
            search = next..<out.endIndex
        }
        return out
    }

    private func canonicalJSON(_ object: [String: Any]) -> Data? {
        guard JSONSerialization.isValidJSONObject(object) else { return nil }
        return try? JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
    }
}
