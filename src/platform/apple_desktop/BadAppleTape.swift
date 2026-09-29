import Foundation

/// Flight-recorder intent drop-stream writer.
///
/// Appends compact decision intents to the flight_tape daemon's ingest file
/// (`/var/lib/bad_apple/tape/intent.jsonl`). Fire-and-forget: the daemon may
/// not be running, writes are append-only and best-effort — a full disk or a
/// missing dir must never stall the dispatch loop. Sensitive values are
/// redacted again on ingest by the frame layer; keep fields small anyway.
enum BadAppleTape {
    static let streamPath = "/var/lib/bad_apple/tape/intent.jsonl"

    /// Append one `{kind, ts, ...fields}` line. Values must be JSON-serializable.
    static func drop(kind: String, fields: [String: Any]) {
        var obj = fields
        obj["kind"] = kind
        obj["ts"] = Int(Date().timeIntervalSince1970)
        guard let data = try? JSONSerialization.data(
            withJSONObject: obj, options: []),
            let text = String(data: data, encoding: .utf8)
        else { return }

        let path = streamPath
        if !FileManager.default.fileExists(atPath: path) {
            try? FileManager.default.createDirectory(
                atPath: (path as NSString).deletingLastPathComponent,
                withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let fh = FileHandle(forWritingAtPath: path) else { return }
        fh.seekToEndOfFile()
        fh.write(Data((text + "\n").utf8))
        try? fh.close()
    }
}
