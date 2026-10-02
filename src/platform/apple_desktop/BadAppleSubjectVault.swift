// BadAppleSubjectVault — per-subject data-encryption-key custody for
// certified erasure (crypto-shredding).
//
// Subject-bearing records (human-layer people today) are sealed under a
// per-subject AES-256-GCM key at the persistence boundary. `forget`
// destroys the key: every copy of the ciphertext — including respawn
// snapshots and tape bundles — becomes permanently unreadable, because
// the keys live in ~/.bad_apple_keys, deliberately OUTSIDE both respawn
// state roots (~/.bad_apple and /var/lib/bad_apple). A state revert can
// resurrect ciphertext but never the key, and can never resurrect the
// destroyed-subject tombstones either: the memory of an erasure is
// revert-immune by construction.
//
// Layout:
//   deks/<keyfile>            per-subject DEKs (JSON {v,alg,key_b64})
//   household                 household DEK sealing tombstone names
//   destroyed/<subject_ref>   erasure tombstones {destroyed_at, sealed_name}
//
// Permissions: dir 0700 / key files 0600 by default — engine and menu bar
// run under the same account. BADAPPLE_KEY_VAULT overrides the root for
// deployments that split those contexts.

import Foundation
import CryptoKit
import Darwin

struct BadAppleSealedBox: Codable, Equatable, Sendable {
    var v: Int
    var alg: String
    var blob: String
}

final class BadAppleSubjectVault: @unchecked Sendable {
    static let shared = BadAppleSubjectVault()

    private let lock = NSLock()
    private var cachedKeys: [String: SymmetricKey] = [:]

    private let root: String
    private var dekDir: String { root + "/deks" }
    private var destroyedDir: String { root + "/destroyed" }
    private var householdPath: String { root + "/household" }

    private init() {
        // Sibling of ~/.bad_apple: outside both respawn roots (the learned
        // root and the platform root), reachable by the engine daemon and
        // the menu bar under the same account. BADAPPLE_KEY_VAULT overrides.
        root = ProcessInfo.processInfo.environment["BADAPPLE_KEY_VAULT"]
            ?? NSHomeDirectory() + "/.bad_apple_keys"
    }

    // MARK: - Directory custody

    private func ensureDirs() -> Bool {
        let fm = FileManager.default
        do {
            try fm.createDirectory(atPath: dekDir, withIntermediateDirectories: true,
                                   attributes: [.posixPermissions: 0o700])
            try fm.createDirectory(atPath: destroyedDir, withIntermediateDirectories: true,
                                   attributes: [.posixPermissions: 0o700])
            for path in [root, dekDir, destroyedDir] {
                _ = chmod(path, 0o700)
            }
            return true
        } catch {
            vlog("subject vault: dir setup failed: \(error)")
            return false
        }
    }

    private static func keyFileName(_ keyID: String) -> String {
        let digest = SHA256.hash(data: Data(("subject-dek:" + keyID).utf8))
        return digest.map { String(format: "%02x", $0) }.joined().prefix(32).description
    }

    private static func fixPerms(_ path: String) {
        _ = chmod(path, 0o600)
    }

    // MARK: - DEK lifecycle

    private func keyPath(_ keyID: String) -> String {
        keyID == "__household__"
            ? householdPath
            : dekDir + "/" + Self.keyFileName(keyID)
    }

    func hasKey(_ keyID: String) -> Bool {
        FileManager.default.fileExists(atPath: keyPath(keyID))
    }

    func loadKey(_ keyID: String) -> SymmetricKey? {
        lock.lock()
        defer { lock.unlock() }
        if let cached = cachedKeys[keyID] { return cached }
        guard let data = FileManager.default.contents(atPath: keyPath(keyID)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let b64 = json["key_b64"] as? String,
              let raw = Data(base64Encoded: b64), raw.count == 32 else { return nil }
        let key = SymmetricKey(data: raw)
        cachedKeys[keyID] = key
        return key
    }

    /// Load or lazily create the subject's DEK. Returns nil only when the
    /// vault is unreachable — callers treat nil as "cannot seal".
    func ensureKey(_ keyID: String) -> SymmetricKey? {
        if let existing = loadKey(keyID) { return existing }
        guard ensureDirs() else { return nil }
        let key = SymmetricKey(size: .bits256)
        let raw = key.withUnsafeBytes { Data($0) }
        let doc: [String: Any] = [
            "v": 1,
            "alg": "A256GCM",
            "key_id_sha256": Self.keyFileName(keyID),
            "created": ISO8601DateFormatter().string(from: Date()),
            "key_b64": raw.base64EncodedString(),
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: doc, options: [.sortedKeys]) else { return nil }
        let path = keyPath(keyID)
        guard FileManager.default.createFile(atPath: path, contents: data,
                                             attributes: [.posixPermissions: 0o600]) else {
            return nil
        }
        Self.fixPerms(path)
        lock.lock()
        cachedKeys[keyID] = key
        lock.unlock()
        return key
    }

    /// Destroy the subject's DEK. After this returns true, every ciphertext
    /// sealed under that key is unrecoverable on every copy of the state.
    func destroyKey(_ keyID: String) -> Bool {
        lock.lock()
        cachedKeys.removeValue(forKey: keyID)
        lock.unlock()
        let path = keyPath(keyID)
        guard FileManager.default.fileExists(atPath: path) else { return true }
        do {
            try FileManager.default.removeItem(atPath: path)
            return !FileManager.default.fileExists(atPath: path)
        } catch {
            vlog("subject vault: key destroy failed for \(keyID): \(error)")
            return false
        }
    }

    // MARK: - Seal / open

    func seal(_ plaintext: Data, keyID: String) -> BadAppleSealedBox? {
        guard let key = ensureKey(keyID),
              let box = try? AES.GCM.seal(plaintext, using: key) else { return nil }
        return BadAppleSealedBox(
            v: 1, alg: "A256GCM",
            blob: box.combined?.base64EncodedString() ?? "")
    }

    func open(_ box: BadAppleSealedBox, keyID: String) -> Data? {
        guard box.alg == "A256GCM",
              let combined = Data(base64Encoded: box.blob),
              let key = loadKey(keyID),
              let sealed = try? AES.GCM.SealedBox(combined: combined),
              let plain = try? AES.GCM.open(sealed, using: key) else { return nil }
        return plain
    }

    // MARK: - Human-layer person sealing

    private struct PersonPII: Codable {
        var name: String
        var relationship: String
        var notes: String
    }

    /// Returns a disk-form person: sealed PII box + blanked plaintext fields.
    /// Tombstoned persons pass through untouched. When the vault cannot seal,
    /// the person is returned unchanged (plaintext) — callers decide policy.
    func sealPersonForDisk(_ person: BadApplePerson) -> BadApplePerson {
        if person.sealed != nil || person.shredded { return person }
        guard let key = ensureKey(person.id) else { return person }
        let pii = PersonPII(
            name: person.name, relationship: person.relationship, notes: person.notes)
        guard let plain = try? JSONEncoder().encode(pii),
              let box = try? AES.GCM.seal(plain, using: key) else { return person }
        var disk = person
        disk.sealed = BadAppleSealedBox(
            v: 1, alg: "A256GCM",
            blob: box.combined?.base64EncodedString() ?? "")
        disk.name = ""
        disk.relationship = ""
        disk.notes = ""
        return disk
    }

    /// Inverse of sealPersonForDisk at load time. Missing DEK (destroyed or
    /// unreachable) produces a tombstone: "[forgotten]", dismissed, shredded.
    /// Plaintext persons whose subject_ref is in the destroyed registry —
    /// i.e. resurrected by a state revert — are tombstoned on sight.
    func unsealPerson(_ person: BadApplePerson) -> BadApplePerson {
        guard let box = person.sealed, !box.blob.isEmpty else {
            if !person.name.isEmpty,
               destroyedRefs().contains(Self.subjectRef(person.name)) {
                return tombstone(person)
            }
            return person
        }
        guard let data = open(box, keyID: person.id),
              let pii = try? JSONDecoder().decode(PersonPII.self, from: data) else {
            var dead = person
            dead.shredded = true
            dead.name = "[forgotten]"
            dead.relationship = ""
            dead.notes = ""
            dead.status = .dismissed
            return dead
        }
        var live = person
        live.name = pii.name
        live.relationship = pii.relationship
        live.notes = pii.notes
        live.sealed = nil
        return live
    }

    private func tombstone(_ person: BadApplePerson) -> BadApplePerson {
        var dead = person
        dead.name = "[forgotten]"
        dead.relationship = ""
        dead.notes = ""
        dead.status = .dismissed
        dead.shredded = true
        return dead
    }

    // MARK: - Destroyed-subject tombstones (the erasure that survives revert)

    static func subjectRef(_ subject: String) -> String {
        let norm = subject.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let digest = SHA256.hash(data: Data(norm.utf8))
        return String(digest.map { String(format: "%02x", $0) }.joined().prefix(16))
    }

    /// Record that a subject's DEK was destroyed. The name is sealed under
    /// the household key — recoverable only for re-purge replay, never
    /// stored in plaintext. Lives outside respawn scope: a revert cannot
    /// erase the memory of the erasure.
    func recordDestroyed(subjectRef ref: String, name: String) {
        guard ensureDirs() else { return }
        var doc: [String: Any] = [
            "subject_ref": ref,
            "destroyed_at": ISO8601DateFormatter().string(from: Date()),
        ]
        if let key = ensureKey("__household__"),
           let box = try? AES.GCM.seal(Data(name.lowercased().utf8), using: key),
           let combined = box.combined {
            doc["sealed_name"] = combined.base64EncodedString()
        }
        if let data = try? JSONSerialization.data(
            withJSONObject: doc, options: [.sortedKeys]) {
            let path = destroyedDir + "/" + ref + ".json"
            FileManager.default.createFile(atPath: path, contents: data,
                                           attributes: [.posixPermissions: 0o600])
            Self.fixPerms(path)
        }
    }

    func destroyedRefs() -> Set<String> {
        guard let files = try? FileManager.default
            .contentsOfDirectory(atPath: destroyedDir) else { return [] }
        return Set(files.compactMap { $0.hasSuffix(".json") ? String($0.dropLast(5)) : nil })
    }

    /// Names of destroyed subjects, unsealed under the household key — the
    /// replay set for re-purging incidental mentions after a state revert.
    /// Entries whose household key is gone yield nothing (documented nuclear
    /// state: no re-purge memory).
    func replayNames() -> [String] {
        guard let files = try? FileManager.default
            .contentsOfDirectory(atPath: destroyedDir) else { return [] }
        var names: [String] = []
        for f in files where f.hasSuffix(".json") {
            guard let data = FileManager.default
                    .contents(atPath: destroyedDir + "/" + f),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let sealed = json["sealed_name"] as? String,
                  let combined = Data(base64Encoded: sealed),
                  let key = loadKey("__household__"),
                  let box = try? AES.GCM.SealedBox(combined: combined),
                  let plain = try? AES.GCM.open(box, using: key),
                  let name = String(data: plain, encoding: .utf8), !name.isEmpty
            else { continue }
            names.append(name)
        }
        return names
    }
}

private func vlog(_ message: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    print("[\(ts)] \(message)")
    fflush(stdout)
}
