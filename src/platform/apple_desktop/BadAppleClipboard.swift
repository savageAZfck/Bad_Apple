import AppKit
import Foundation

// BadAppleClipboard — clipboard recall organ.
//
// The menu bar app polls the general pasteboard's changeCount and appends each
// new text clipping to ~/.bad_apple/clipboard_history.jsonl — one JSON object
// per line: {"ts","app","text"}. The engine's recall_clipboard tool reads the
// same file, so "that link I copied this morning" is a query, not a hunt.
//
// Privacy contract: the caller passes `allowed` (humanWritesAllowed — private
// mode off) and the watcher simply does not observe the pasteboard while it
// is false. History is bounded at maxEntries and each entry is truncated, so
// the file can never grow without bound.

final class BadAppleClipboardWatch {
    static let historyURL =
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/clipboard_history.jsonl")

    private static let maxEntries = 400
    private static let maxTextLength = 4000

    private var lastChangeCount: Int
    private var lastText = ""

    init() {
        lastChangeCount = NSPasteboard.general.changeCount
    }

    /// Poll the pasteboard; persists a new entry only when it changed and
    /// persistence is allowed. Cheap: no work at all unless changeCount moved.
    func tick(allowed: Bool) {
        let pb = NSPasteboard.general
        guard pb.changeCount != lastChangeCount else { return }
        lastChangeCount = pb.changeCount
        guard allowed else { return }
        guard let text = pb.string(forType: .string), !text.isEmpty else { return }
        let clipped = String(text.prefix(Self.maxTextLength))
        guard clipped != lastText else { return }
        lastText = clipped

        let app = NSWorkspace.shared.frontmostApplication?.localizedName ?? ""
        let entry: [String: Any] = [
            "ts": Date().timeIntervalSince1970,
            "app": app,
            "text": clipped,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: entry),
              var line = String(data: data, encoding: .utf8) else { return }
        line.append("\n")

        var existing = (try? String(contentsOf: Self.historyURL, encoding: .utf8)) ?? ""
        existing.append(line)
        var lines = existing.components(separatedBy: "\n").filter { !$0.isEmpty }
        if lines.count > Self.maxEntries {
            lines = Array(lines.suffix(Self.maxEntries))
        }
        try? (lines.joined(separator: "\n") + "\n").write(
            to: Self.historyURL, atomically: true, encoding: .utf8
        )
    }
}
