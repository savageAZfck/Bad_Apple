// BadAppleAmendment — constitutional self-modification.
//
// Every change to Bad Apple's own behavior-bearing state — weight grafts,
// strategy updates, prompt patches, policy changes — is an amendment:
// proposed as a signed artifact, deliberated by the council, ratified per
// its tier, applied bounded, measured, and anchored into a hash-chained
// journal at /var/lib/bad_apple/amendments.jsonl with the artifact under
// /var/lib/bad_apple/amendments/<id>.json.
//
// Two tiers:
//   statutory       — council unanimity auto-ratifies (dream_adapter,
//                     strategy_update, prompt_patch); anything contested
//                     or merely-not-unanimous holds for the human.
//   constitutional  — policy_patch, council_change, gate_change: human
//                     ratification only, always. The council can argue;
//                     it cannot amend the rules that bind it.
//
// The point: self-modification is legal when it leaves a signed,
// verifiable record — and structurally impossible otherwise. The dream
// adapter is the first amendment citizen; its graft is adopted only
// through this lifecycle.

import Foundation
import CryptoKit

final class BadAppleAmendment: @unchecked Sendable {
    static let shared = BadAppleAmendment()

    static let dir = "/var/lib/bad_apple/amendments"
    static let journalPath = "/var/lib/bad_apple/amendments.jsonl"
    private let lock = NSLock()

    static let constitutionalSurfaces: Set<String> = [
        "policy_patch", "council_change", "gate_change",
    ]
    static let statutorySurfaces: Set<String> = [
        "dream_adapter", "strategy_update", "prompt_patch", "ify_proposal",
    ]

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private init() {
        try? FileManager.default.createDirectory(
            atPath: Self.dir, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
    }

    // MARK: - Artifact

    private func artifactPath(_ id: String) -> String { "\(Self.dir)/\(id).json" }

    func load(id: String) -> [String: Any]? {
        guard let data = FileManager.default.contents(atPath: artifactPath(id)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj
    }

    private func save(_ artifact: [String: Any]) -> Bool {
        guard let data = try? JSONSerialization.data(
            withJSONObject: artifact, options: [.prettyPrinted, .sortedKeys]) else { return false }
        let path = artifactPath(artifact["id"] as? String ?? "unknown")
        return (try? data.write(to: URL(fileURLWithPath: path), options: .atomic)) != nil
    }

    private func canonicalJSON(_ object: [String: Any]) -> Data? {
        guard JSONSerialization.isValidJSONObject(object) else { return nil }
        return try? JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Journal (hash-chained, one line per transition)

    private func journalTip() -> String {
        guard let text = try? String(contentsOfFile: Self.journalPath, encoding: .utf8),
              let last = text.split(separator: "\n").last,
              let data = last.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let h = obj["entry_hash"] as? String else { return "genesis" }
        return h
    }

    private func journal(id: String, transition: String, artifact: [String: Any]) {
        let artifactHash = canonicalJSON(artifact).map { sha256Hex($0) } ?? ""
        var entry: [String: Any] = [
            "ts": Self.iso.string(from: Date()),
            "amendment_id": id,
            "transition": transition,
            "artifact_sha256": artifactHash,
            "prev_hash": journalTip(),
        ]
        if let data = canonicalJSON(entry) {
            entry["entry_hash"] = sha256Hex(data)
        }
        if let line = canonicalJSON(entry),
           let text = String(data: line, encoding: .utf8) {
            if let h = FileHandle(forWritingAtPath: Self.journalPath) {
                h.seekToEndOfFile(); h.write(text.data(using: .utf8)! + Data("\n".utf8)); try? h.close()
            } else {
                FileManager.default.createFile(
                    atPath: Self.journalPath, contents: (text + "\n").data(using: .utf8),
                    attributes: [.posixPermissions: 0o600])
            }
        }
        BadAppleEngine.shared.auditDaemonEvent(
            type: "amendment_\(transition)",
            data: ["amendment_id": id, "artifact_sha256": artifactHash])
    }

    // MARK: - Lifecycle

    /// File a new amendment. Returns the amendment id.
    @discardableResult
    func propose(surface: String, evidence: [String: Any], persona: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        let stamp = Self.iso.string(from: Date())
            .replacingOccurrences(of: ":", with: "-").replacingOccurrences(of: ".", with: "")
        let id = "amend-\(stamp)-\(String(format: "%06x", Int.random(in: 0...0xFFFFFF)))"
        let tier = Self.constitutionalSurfaces.contains(surface) ? "constitutional" : "statutory"
        let artifact: [String: Any] = [
            "kind": "badapple.amendment",
            "version": 1,
            "id": id,
            "surface": surface,
            "tier": tier,
            "status": "proposed",
            "proposed_at": Self.iso.string(from: Date()),
            "proposed_by": persona,
            "evidence": evidence,
        ]
        _ = save(artifact)
        journal(id: id, transition: "proposed", artifact: artifact)
        return id
    }

    /// Record the council verdict into the artifact. Statutory + unanimous
    /// auto-ratifies; constitutional or contested/non-unanimous holds for
    /// the human. Returns the resulting status.
    @discardableResult
    func recordCouncil(id: String, verdict: CouncilVerdict) -> String {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id) else { return "missing" }
        let votes: [[String: Any]] = verdict.votes.map {
            ["seat": $0.seat, "vote": $0.vote, "rationale": $0.rationale]
        }
        artifact["council"] = [
            "decision": verdict.decision.rawValue,
            "mean": verdict.mean,
            "dissent": verdict.dissent,
            "unanimous_approve": verdict.unanimousApprove,
            "summary": verdict.summaryLine,
            "votes": votes,
        ]
        let tier = artifact["tier"] as? String ?? "statutory"
        let status: String
        if tier == "constitutional" {
            status = "held"              // constitution never self-ratifies
        } else if verdict.unanimousApprove {
            status = "ratified"          // statutory + unanimous → law
            artifact["ratified_by"] = "council-unanimous"
            artifact["ratified_at"] = Self.iso.string(from: Date())
        } else {
            status = "held"              // contested or divided → human
        }
        artifact["status"] = status
        if status == "ratified", let body = canonicalJSON(artifact),
           IdentityAgentClient.shared.isAvailable,
           let sig = IdentityAgentClient.shared.sign(message: body) {
            artifact["signature"] = sig
            artifact["signature_scheme"] = "secure-enclave"
            artifact["signed"] = true
        }
        _ = save(artifact)
        journal(id: id, transition: status == "ratified" ? "deliberated+ratified" : "deliberated+held",
                artifact: artifact)
        return status
    }

    /// Human ratification — the only path for constitutional-tier and held
    /// amendments. Signs the artifact at ratification.
    @discardableResult
    func ratify(id: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id) else { return "no amendment \(id)" }
        let status = artifact["status"] as? String ?? ""
        guard status == "held" || status == "proposed" else {
            return "amendment \(id) is \(status) — only held amendments can be ratified"
        }
        artifact["status"] = "ratified"
        artifact["ratified_by"] = "human"
        artifact["ratified_at"] = Self.iso.string(from: Date())
        if let body = canonicalJSON(artifact),
           IdentityAgentClient.shared.isAvailable,
           let sig = IdentityAgentClient.shared.sign(message: body) {
            artifact["signature"] = sig
            artifact["signature_scheme"] = "secure-enclave"
            artifact["signed"] = true
        }
        _ = save(artifact)
        journal(id: id, transition: "ratified", artifact: artifact)
        return "ratified"
    }

    /// Mark the amendment applied — the surface mutation actually ran.
    /// `postState` carries fingerprints of what changed (adapter hashes etc).
    @discardableResult
    func markApplied(id: String, postState: [String: Any]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id),
              artifact["status"] as? String == "ratified" else { return false }
        artifact["status"] = "applied"
        artifact["applied_at"] = Self.iso.string(from: Date())
        artifact["post_state"] = postState
        _ = save(artifact)
        journal(id: id, transition: "applied", artifact: artifact)
        return true
    }

    /// Attach post-apply measurement — what the amendment actually did.
    @discardableResult
    func markMeasured(id: String, metrics: [String: Any]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id),
              artifact["status"] as? String == "applied" else { return false }
        artifact["measured"] = metrics
        artifact["measured_at"] = Self.iso.string(from: Date())
        _ = save(artifact)
        journal(id: id, transition: "measured", artifact: artifact)
        return true
    }

    /// Human rejection — a held amendment is killed.
    @discardableResult
    func reject(id: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id) else { return "no amendment \(id)" }
        let status = artifact["status"] as? String ?? ""
        guard status == "held" || status == "proposed" else {
            return "amendment \(id) is \(status) — only held amendments can be rejected"
        }
        artifact["status"] = "rejected"
        artifact["rejected_by"] = "human"
        artifact["rejected_at"] = Self.iso.string(from: Date())
        _ = save(artifact)
        journal(id: id, transition: "rejected", artifact: artifact)
        return "rejected"
    }

    /// Mark an applied amendment reverted after its surface mutation was
    /// rolled back (e.g. dream-prev restored).
    @discardableResult
    func markReverted(id: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard var artifact = load(id: id),
              artifact["status"] as? String == "applied" else { return false }
        artifact["status"] = "reverted"
        artifact["reverted_at"] = Self.iso.string(from: Date())
        _ = save(artifact)
        journal(id: id, transition: "reverted", artifact: artifact)
        return true
    }

    // MARK: - Listing

    func list(limit: Int = 10) -> String {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: Self.dir),
              !files.isEmpty else { return "No amendments on record." }
        let rows = files.filter { $0.hasSuffix(".json") }.sorted().reversed().prefix(limit)
        var lines: [String] = []
        for f in rows {
            guard let a = load(id: String(f.dropLast(5))) else { continue }
            let id = a["id"] as? String ?? f
            let status = a["status"] as? String ?? "?"
            let surface = a["surface"] as? String ?? "?"
            let tier = a["tier"] as? String ?? "?"
            let signed = a["signed"] as? Bool == true ? "signed" : "unsigned"
            lines.append("- \(id)  \(surface) [\(tier)]  \(status)  \(signed)")
        }
        return lines.isEmpty ? "No amendments on record."
            : "Amendments (latest \(lines.count)):\n" + lines.joined(separator: "\n")
    }

    func show(id: String) -> String {
        guard let data = FileManager.default.contents(atPath: artifactPath(id)),
              let text = String(data: data, encoding: .utf8) else {
            return "no amendment \(id)"
        }
        return text
    }
}
