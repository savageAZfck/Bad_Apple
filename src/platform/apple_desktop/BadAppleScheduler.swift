import Foundation

// BadAppleScheduler — standing orders.
//
// "Every morning at 8, brief me." "Check the build status hourly." "Remind me
// about the lease every Monday." The user states a goal and a cadence once;
// the scheduler fires it into the agent loop when it comes due, so recurring
// work runs through the same plan-act-ledger pipeline as anything she does.
//
// Storage: ~/.bad_apple/schedules.json — one record per standing order with
// the next fire time precomputed. The daemon's minute timer loads the file,
// runs whatever is due, and writes back the updated records, so schedules
// survive restarts like everything else about her.

struct BadAppleSchedule: Codable, Equatable {
    var id: String
    var name: String
    var goal: String
    var every: String
    var nextRun: TimeInterval
    var lastRun: TimeInterval?
    var lastResult: String?
    var enabled: Bool
}

enum BadAppleScheduler {

    static let filePath = NSHomeDirectory() + "/.bad_apple/schedules.json"

    // MARK: - Persistence

    static func load() -> [BadAppleSchedule] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else { return [] }
        return (try? JSONDecoder().decode([BadAppleSchedule].self, from: data)) ?? []
    }

    static func save(_ items: [BadAppleSchedule]) {
        let capped = Array(items.suffix(200))
        guard let data = try? JSONEncoder().encode(capped) else { return }
        try? FileManager.default.createDirectory(
            atPath: NSHomeDirectory() + "/.bad_apple",
            withIntermediateDirectories: true
        )
        try? data.write(to: URL(fileURLWithPath: filePath), options: .atomic)
    }

    static func add(name: String, goal: String, every: String) throws -> BadAppleSchedule {
        guard let next = nextOccurrence(every: every, from: Date()) else {
            throw BadAppleScheduleError.unparseable(every)
        }
        var items = load()
        let schedule = BadAppleSchedule(
            id: UUID().uuidString.lowercased(),
            name: name.isEmpty ? String(goal.prefix(40)) : name,
            goal: goal,
            every: every,
            nextRun: next.timeIntervalSince1970,
            lastRun: nil,
            lastResult: nil,
            enabled: true
        )
        items.append(schedule)
        save(items)
        return schedule
    }

    static func remove(idOrName: String) -> BadAppleSchedule? {
        var items = load()
        let needle = idOrName.lowercased()
        guard let idx = items.firstIndex(where: {
            $0.id == needle || $0.id.hasPrefix(needle) || $0.name.lowercased().contains(needle)
        }) else { return nil }
        let removed = items.remove(at: idx)
        save(items)
        return removed
    }

    /// Schedules that are due right now.
    static func due(now: Date = Date()) -> [BadAppleSchedule] {
        load().filter { $0.enabled && $0.nextRun <= now.timeIntervalSince1970 }
    }

    /// Record a completed run and roll the schedule forward: recurring
    /// cadences get their next occurrence; one-shots ("in 30 minutes")
    /// disable themselves.
    static func markRan(id: String, result: String) {
        var items = load()
        guard let idx = items.firstIndex(where: { $0.id == id }) else { return }
        items[idx].lastRun = Date().timeIntervalSince1970
        items[idx].lastResult = String(result.prefix(200))
        if let next = nextOccurrence(every: items[idx].every, from: Date()) {
            items[idx].nextRun = next.timeIntervalSince1970
        } else {
            items[idx].enabled = false
        }
        save(items)
    }

    // MARK: - Cadence parsing

    /// Compute the next fire time for a cadence string. Returns nil for
    /// unparseable input — the caller turns that into a friendly error.
    /// Supported: "in N minutes|hours", "every N minutes|hours", "hourly",
    /// "daily [at] [H[:MM][am|pm]]", "weekly <weekday> [H[:MM][am|pm]]",
    /// "every morning" (8am), "every evening" (7pm).
    static func nextOccurrence(every: String, from: Date) -> Date? {
        let text = every.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        let cal = Calendar.current

        if let minutes = parseInterval(text, prefix: "in ") {
            return from.addingTimeInterval(minutes * 60)
        }
        if let minutes = parseInterval(text, prefix: "every ") {
            return from.addingTimeInterval(minutes * 60)
        }
        if text == "hourly" || text == "every hour" {
            return from.addingTimeInterval(3600)
        }
        if text == "every morning" {
            return nextWallTime(hour: 8, minute: 0, from: from, cal: cal)
        }
        if text == "every evening" {
            return nextWallTime(hour: 19, minute: 0, from: from, cal: cal)
        }
        if text.hasPrefix("daily") {
            let (hour, minute) = parseClock(text.replacingOccurrences(of: "daily", with: "")) ?? (8, 0)
            return nextWallTime(hour: hour, minute: minute, from: from, cal: cal)
        }
        if text.hasPrefix("weekly") {
            let rest = text.replacingOccurrences(of: "weekly", with: "")
                .trimmingCharacters(in: .whitespaces)
            let (hour, minute) = parseClock(rest) ?? (9, 0)
            let weekdayNames = ["sunday", "monday", "tuesday", "wednesday",
                                "thursday", "friday", "saturday"]
            for (i, name) in weekdayNames.enumerated() where rest.contains(name) {
                return nextWeekday(weekday: i + 1, hour: hour, minute: minute, from: from, cal: cal)
            }
            return nextWeekday(weekday: 2, hour: hour, minute: minute, from: from, cal: cal)
        }
        return nil
    }

    /// "in 30 minutes" / "every 2 hours" → minutes. Returns nil otherwise.
    private static func parseInterval(_ text: String, prefix: String) -> TimeInterval? {
        guard text.hasPrefix(prefix) else { return nil }
        let rest = String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
        let parts = rest.split(separator: " ")
        guard parts.count >= 2, let n = Double(parts[0]), n > 0 else { return nil }
        let unit = String(parts[1])
        if unit.hasPrefix("minute") { return n }
        if unit.hasPrefix("hour") { return n * 60 }
        if unit.hasPrefix("day") { return n * 60 * 24 }
        return nil
    }

    /// "8", "8:30", "at 8am", "3pm", "15:30" → (hour, minute) in 24h.
    private static func parseClock(_ text: String) -> (Int, Int)? {
        var t = text.trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "at ", with: "")
        guard !t.isEmpty else { return nil }
        var pm = false, am = false
        if t.hasSuffix("pm") { pm = true; t = String(t.dropLast(2)) }
        if t.hasSuffix("am") { am = true; t = String(t.dropLast(2)) }
        t = t.trimmingCharacters(in: .whitespaces)
        let halves = t.split(separator: ":")
        guard let hour = Int(halves.first ?? ""), (0...23).contains(hour) else { return nil }
        let minute = halves.count > 1 ? (Int(halves[1]) ?? 0) : 0
        guard (0...59).contains(minute) else { return nil }
        var h = hour
        if pm && h < 12 { h += 12 }
        if am && h == 12 { h = 0 }
        return (h, minute)
    }

    private static func nextWallTime(hour: Int, minute: Int, from: Date, cal: Calendar) -> Date? {
        var comps = cal.dateComponents([.year, .month, .day], from: from)
        comps.hour = hour
        comps.minute = minute
        guard let today = cal.date(from: comps) else { return nil }
        if today > from { return today }
        return cal.date(byAdding: .day, value: 1, to: today)
    }

    private static func nextWeekday(weekday: Int, hour: Int, minute: Int, from: Date, cal: Calendar) -> Date? {
        var comps = cal.dateComponents([.year, .month, .day], from: from)
        comps.hour = hour
        comps.minute = minute
        guard var candidate = cal.date(from: comps) else { return nil }
        for _ in 0..<8 {
            if cal.component(.weekday, from: candidate) == weekday && candidate > from {
                return candidate
            }
            candidate = cal.date(byAdding: .day, value: 1, to: candidate) ?? candidate
        }
        return nil
    }
}

enum BadAppleScheduleError: LocalizedError {
    case unparseable(String)
    var errorDescription: String? {
        switch self {
        case .unparseable(let text):
            return "could not parse cadence '\(text)'. Try 'in 30 minutes', 'every 2 hours', 'daily 8am', 'weekly monday 9am', or 'every morning'."
        }
    }
}
