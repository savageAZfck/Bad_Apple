import Foundation

// BadAppleLookahead — the anticipation organ.
//
// Due items fire when they're due; this organ looks ahead. Every few minutes
// it pulls today's calendar through the aqua bridge and finds what's starting
// inside the lookahead window — then tells the user before they're late,
// enriches the event with related life context (a matching thread, the person
// it's about), and stages an "Up next:" line into ambientContext so the next
// prompt already knows what's coming.
//
// Dedup contract: each event fires once per occurrence — a key of
// "start|title" is recorded in ~/.bad_apple/lookahead_fired.json and pruned
// after 7 days, so a moved meeting re-alerts and a steady one never nags twice.

enum BadAppleLookahead {

    struct Item {
        var key: String
        var title: String
        var minutesUntil: Int
        var detail: String       // calendar name, location
        var related: String      // matched life thread / person, or ""
    }

    private static let firedPath =
        NSHomeDirectory() + "/.bad_apple/lookahead_fired.json"

    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm"
        return f
    }()

    /// Pull today's events via the aqua bridge and return the ones starting
    /// inside `windowMinutes` that haven't fired yet.
    static func sweep(windowMinutes: Int = 30) -> [Item] {
        guard let res = callAqua(
            command: "calendar_events", payload: ["days_ahead": 1], timeout: 20
        ), let events = res["events"] as? [String] else { return [] }

        let now = Date()
        var fired = loadFired()
        var items: [Item] = []

        for line in events {
            let parts = line.components(separatedBy: " | ")
            guard parts.count >= 2,
                  let start = stamp.date(from: parts[0]) else { continue }
            let delta = start.timeIntervalSince(now)
            guard delta > 0, delta <= TimeInterval(windowMinutes * 60) else { continue }

            let key = "\(parts[0])|\(parts[1])"
            guard fired[key] == nil else { continue }
            fired[key] = now.timeIntervalSince1970

            var detail = parts.count > 2 ? parts[2] : ""
            if parts.count > 3, !parts[3].isEmpty {
                detail += detail.isEmpty ? parts[3] : " at \(parts[3])"
            }
            items.append(Item(
                key: key,
                title: parts[1],
                minutesUntil: Int(delta / 60),
                detail: detail,
                related: relatedContext(for: parts[1])
            ))
        }
        saveFired(fired)
        return items
    }

    /// Match event words against life threads and people — "Standup with
    /// Sarah" should know it's about the 'Q3 launch' thread and that Sarah
    /// owes you the API spec.
    private static func relatedContext(for title: String) -> String {
        let words = title.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 3 }
        guard !words.isEmpty else { return "" }
        let state = BadAppleHumanLayer.shared.snapshot()
        for thread in state.threads where thread.status == .active {
            if words.contains(where: { thread.title.lowercased().contains($0) }) {
                var line = "thread '\(thread.title)'"
                if !thread.nextAction.isEmpty { line += " — next: \(thread.nextAction)" }
                return line
            }
        }
        for person in state.people where person.status == .active {
            if words.contains(where: { person.name.lowercased().contains($0) }) {
                var line = person.name
                if !person.relationship.isEmpty { line += " (\(person.relationship))" }
                return line
            }
        }
        return ""
    }

    private static func loadFired() -> [String: TimeInterval] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: firedPath)),
              let dict = try? JSONDecoder().decode([String: TimeInterval].self, from: data)
        else { return [:] }
        // Prune keys older than a week — stale entries can't suppress a
        // genuinely rescheduled event and the file stays bounded.
        let cutoff = Date().timeIntervalSince1970 - 7 * 24 * 3600
        return dict.filter { $0.value > cutoff }
    }

    private static func saveFired(_ fired: [String: TimeInterval]) {
        guard let data = try? JSONEncoder().encode(fired) else { return }
        try? FileManager.default.createDirectory(
            atPath: NSHomeDirectory() + "/.bad_apple",
            withIntermediateDirectories: true
        )
        try? data.write(to: URL(fileURLWithPath: firedPath), options: .atomic)
    }
}
