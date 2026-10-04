// BadAppleProvenance — signed provenance manifest for the loaded brain.
//
// Every successful model load or swap appends one record to
// /var/lib/bad_apple/provenance.jsonl: which weights, from where, how large,
// which dream adapter was present, and which policy was in force. When the
// identity agent is running the record is signed with the Secure Enclave key;
// either way a `model_provenance` ledger event anchors the record's SHA-256
// into the audit hash chain, so the artifact cannot be silently rewritten.

import Foundation
import CryptoKit

final class BadAppleProvenance: @unchecked Sendable {
    static let shared = BadAppleProvenance()

    static let directoryPath = "/var/lib/bad_apple"
    static let logPath = "/var/lib/bad_apple/provenance.jsonl"

    private let lock = NSLock()

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    /// Record a brain load/swap. Called from the engine after the container
    /// is ready and the dream adapter pass has run.
    ///
    /// - Parameters:
    ///   - modelId: repo id (or local model id) now serving inference
    ///   - revision: HF revision ("main" for cache loads)
    ///   - directory: explicit local directory when the load was path-based;
    ///     nil means resolved through the HuggingFace cache
    ///   - modelBytes: bytes reserved by the VRAM admission check
    ///   - policyContext: ledgerContext() from the policy engine
    func recordLoad(
        modelId: String,
        revision: String,
        directory: String?,
        modelBytes: UInt64,
        policyContext: [String: Any]
    ) {
        lock.lock()
        defer { lock.unlock() }

        var record: [String: Any] = [
            "ts": Self.isoFormatter.string(from: Date()),
            "kind": "brain_manifest",
            "model_id": modelId,
            "revision": revision,
            "source": directory != nil ? "local_dir" : "hf_cache",
            "model_bytes": Int(modelBytes),
        ]
        if let directory { record["directory"] = directory }

        let dreamDir = "/var/lib/bad_apple/lora_adapters/dream"
        if FileManager.default.fileExists(atPath: "\(dreamDir)/adapters.safetensors") {
            record["dream_adapter"] = dreamDir
        }
        for (k, v) in policyContext { record[k] = v }

        // The manager owns the manifest registry (keyed by profile slot; the
        // engine reports repo ids). First sighting of an unmanifested brain
        // auto-hashes the serving weights so the fingerprint lands this load.
        if let manifestRoot = BadAppleModelManager.shared
            .weightsManifestFingerprint(modelIdOrRepoId: modelId) {
            record["weights_manifest_sha256"] = manifestRoot
        }

        // Sign the canonical body when the identity agent is up.
        guard let bodyData = canonicalJSONData(record) else { return }
        let signed = IdentityAgentClient.shared.isAvailable
        if signed, let sigB64 = IdentityAgentClient.shared.sign(message: bodyData) {
            record["signature"] = sigB64
            record["signature_scheme"] = "secure-enclave"
            if let pub = IdentityAgentClient.shared.publicKey() {
                record["public_key"] = pub
            }
        }
        record["signed"] = signed && record["signature"] != nil

        guard let lineData = canonicalJSONData(record) else { return }
        let recordHash = SHA256.hash(data: lineData)
            .map { String(format: "%02x", Int($0)) }.joined()

        appendLine(lineData + Data([0x0A]))

        // Anchor the record into the audit chain — the artifact file can be
        // verified against the ledger entry even if the log is later copied.
        BadAppleEngine.shared.auditProvenanceRecord(
            modelId: modelId, recordSHA256: recordHash, signed: record["signed"] as? Bool == true
        )
    }

    /// Attest a single output span: a signed record binding which brain
    /// produced which output under which governance. Prompt and output
    /// are recorded as SHA-256 hashes — the attestation is shareable
    /// without disclosing content; a holder of the plaintext can prove
    /// correspondence with `covers_output` semantics off the crate.
    ///
    /// - Parameters:
    ///   - modelId / revision: the brain that emitted the span
    ///   - prompt / output: plaintext, hashed into the record
    ///   - tokenStart / tokenEnd: range covered in the response stream
    ///   - policyContext: ledgerContext() from the policy engine
    ///   - ledgerAnchor: hash of the ledger entry the emission attached to
    func recordSpan(
        modelId: String,
        revision: String,
        prompt: String,
        output: String,
        tokenStart: Int,
        tokenEnd: Int,
        policyContext: [String: Any],
        ledgerAnchor: String?
    ) {
        lock.lock()
        defer { lock.unlock() }

        var record: [String: Any] = [
            "ts": Int(Date().timeIntervalSince1970),
            "kind": "output_span",
            "model_id": modelId,
            "revision": revision,
            "prompt_sha256": sha256Hex(prompt),
            "output_sha256": sha256Hex(output),
            "token_start": tokenStart,
            "token_end": tokenEnd,
        ]
        if let manifestRoot = BadAppleModelManager.shared
            .weightsManifestFingerprint(modelIdOrRepoId: modelId) {
            record["weights_manifest_sha256"] = manifestRoot
        }
        let dreamDir = "/var/lib/bad_apple/lora_adapters/dream"
        if FileManager.default.fileExists(atPath: "\(dreamDir)/adapters.safetensors") {
            record["dream_adapter"] = sha256Hex(dreamDir)
        }
        for (k, v) in policyContext { record[k] = v }
        if let ledgerAnchor { record["ledger_anchor"] = ledgerAnchor }

        guard let bodyData = canonicalJSONData(record) else { return }
        let signed = IdentityAgentClient.shared.isAvailable
        if signed, let sigB64 = IdentityAgentClient.shared.sign(message: bodyData) {
            record["signature"] = sigB64
            record["signature_scheme"] = "secure-enclave"
            if let pub = IdentityAgentClient.shared.publicKey() {
                record["public_key"] = pub
            }
        }
        record["signed"] = signed && record["signature"] != nil

        guard let lineData = canonicalJSONData(record) else { return }
        let recordHash = SHA256.hash(data: lineData)
            .map { String(format: "%02x", Int($0)) }.joined()
        appendLine(lineData + Data([0x0A]))

        BadAppleEngine.shared.auditSpanRecord(
            modelId: modelId, recordSHA256: recordHash,
            tokenStart: tokenStart, tokenEnd: tokenEnd,
            signed: record["signed"] as? Bool == true
        )
    }

    private func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", Int($0)) }.joined()
    }

    private func canonicalJSONData(_ object: [String: Any]) -> Data? {
        guard JSONSerialization.isValidJSONObject(object) else { return nil }
        return try? JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
    }

    private func appendLine(_ data: Data) {
        let dir = Self.directoryPath
        if !FileManager.default.fileExists(atPath: dir) {
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        }
        let path = Self.logPath
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: path)) else { return }
        defer { try? handle.close() }
        handle.seekToEndOfFile()
        handle.write(data)
    }
}
