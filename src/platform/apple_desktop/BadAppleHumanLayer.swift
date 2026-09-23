import Darwin
import Foundation

enum BadAppleAttentionMode: String, Codable, CaseIterable, Sendable {
    case available, focus, quiet, sleep
}

enum BadAppleHumanStatus: String, Codable, CaseIterable, Sendable {
    case active, waiting, completed, dismissed
}

enum BadAppleCommitmentOwner: String, Codable, CaseIterable, Sendable {
    case user
    case badApple = "bad_apple"
}

struct BadAppleHumanPreference: Codable, Equatable, Sendable {
    let id: String
    var key: String
    var value: String
    var source: String
    let createdAt: Date
    var updatedAt: Date
}

struct BadAppleHumanCommitment: Codable, Equatable, Sendable {
    let id: String
    var title: String
    var owner: BadAppleCommitmentOwner
    var status: BadAppleHumanStatus
    var dueAt: Date?
    var threadID: String?
    var detail: String
    let createdAt: Date
    var updatedAt: Date
    var lastNotifiedAt: Date?
}

struct BadAppleLifeThread: Codable, Equatable, Sendable {
    let id: String
    var title: String
    var summary: String
    var nextAction: String
    var status: BadAppleHumanStatus
    let createdAt: Date
    var updatedAt: Date
}

enum BadAppleInteractionChannel: String, Codable, Sendable {
    case silent, queue, notify, speak
}

struct BadAppleHumanEvent: Codable, Equatable, Sendable {
    let id: String
    let kind: String
    let title: String
    let body: String
    let importance: Int
    let urgency: Int
    let requiresAction: Bool
    let voiceRequested: Bool
    let createdAt: Date
}

struct BadAppleInteractionDecision: Codable, Equatable, Sendable {
    let channel: BadAppleInteractionChannel
    let reason: String
}

struct BadAppleHumanInboxItem: Codable, Equatable, Sendable {
    let event: BadAppleHumanEvent
    let decision: BadAppleInteractionDecision
    var status: BadAppleHumanStatus
}

struct BadAppleHumanState: Codable, Equatable, Sendable {
    var schemaVersion: Int
    var attentionMode: BadAppleAttentionMode
    var preferences: [BadAppleHumanPreference]
    var commitments: [BadAppleHumanCommitment]
    var threads: [BadAppleLifeThread]
    var inbox: [BadAppleHumanInboxItem]
}

enum BadAppleHumanLayerError: LocalizedError {
    case emptyField(String)
    case ambiguousID(String)
    case persistence(String)

    var errorDescription: String? {
        switch self {
        case .emptyField(let field):
            return "\(field) must not be empty."
        case .ambiguousID(let prefix):
            return "id prefix '\(prefix)' matches more than one entry."
        case .persistence(let detail):
            return "Human layer persistence failed: \(detail)"
        }
    }
}

final class BadAppleHumanLayer: @unchecked Sendable {
    static let shared = BadAppleHumanLayer()

    static let schemaVersion = 1
    private static let maxPreferences = 500
    private static let maxCommitments = 1000
    private static let maxThreads = 500
    private static let maxInbox = 500
    private static let renotifyInterval: TimeInterval = 24 * 3600

    private let lock = NSLock()
    private let fileManager: FileManager
    private let directory: URL
    private var stateURL: URL { directory.appendingPathComponent("state.json") }
    private var lockURL: URL { directory.appendingPathComponent("state.lock") }

    init(directory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.directory = directory ?? URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".bad_apple")
            .appendingPathComponent("human")
        try? fileManager.createDirectory(
            at: self.directory,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try? fileManager.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: self.directory.path
        )
    }

    private static func emptyState() -> BadAppleHumanState {
        BadAppleHumanState(
            schemaVersion: schemaVersion,
            attentionMode: .available,
            preferences: [],
            commitments: [],
            threads: [],
            inbox: []
        )
    }

    private func loadStateUnlocked() -> BadAppleHumanState {
        guard let data = fileManager.contents(atPath: stateURL.path) else {
            return Self.emptyState()
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let state = try? decoder.decode(BadAppleHumanState.self, from: data) else {
            return Self.emptyState()
        }
        return state
    }

    private func persistUnlocked(_ state: BadAppleHumanState) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        let tempURL = directory.appendingPathComponent(
            ".state.json.tmp.\(ProcessInfo.processInfo.processIdentifier).\(UUID().uuidString)"
        )
        guard fileManager.createFile(
            atPath: tempURL.path,
            contents: data,
            attributes: [.posixPermissions: 0o600]
        ) else {
            throw BadAppleHumanLayerError.persistence("could not create temp state file")
        }
        if rename(tempURL.path, stateURL.path) != 0 {
            try? fileManager.removeItem(at: tempURL)
            try data.write(to: stateURL, options: .atomic)
            try? fileManager.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: stateURL.path
            )
        }
    }

    private func withFileLock<T>(_ operation: Int32, _ body: () throws -> T) throws -> T {
        let fd = open(lockURL.path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else {
            throw BadAppleHumanLayerError.persistence(
                "open \(lockURL.lastPathComponent) failed: \(String(cString: strerror(errno)))"
            )
        }
        _ = fchmod(fd, 0o600)
        guard flock(fd, operation) == 0 else {
            let detail = String(cString: strerror(errno))
            _ = close(fd)
            throw BadAppleHumanLayerError.persistence("flock failed: \(detail)")
        }
        defer {
            _ = flock(fd, LOCK_UN)
            _ = close(fd)
        }
        return try body()
    }

    private func withReadLock<T>(_ body: () throws -> T) throws -> T {
        try withFileLock(LOCK_SH, body)
    }

    private func withWriteLock<T>(_ body: () throws -> T) throws -> T {
        try withFileLock(LOCK_EX, body)
    }

    private func mutate<T>(_ body: (inout BadAppleHumanState) throws -> T) throws -> T {
        lock.lock()
        defer { lock.unlock() }
        return try withWriteLock {
            var state = loadStateUnlocked()
            let original = state
            let result = try body(&state)
            if state != original { try persistUnlocked(state) }
            return result
        }
    }

    func snapshot() -> BadAppleHumanState {
        lock.lock()
        defer { lock.unlock() }
        do {
            return try withReadLock { loadStateUnlocked() }
        } catch {
            return Self.emptyState()
        }
    }

    @discardableResult
    func rememberPreference(key: String, value: String, source: String = "user") throws -> BadAppleHumanPreference {
        let normalizedKey = Self.normalize(key)
        let normalizedValue = Self.normalize(value)
        guard !normalizedKey.isEmpty else { throw BadAppleHumanLayerError.emptyField("key") }
        guard !normalizedValue.isEmpty else { throw BadAppleHumanLayerError.emptyField("value") }
        return try mutate { state in
            if let index = state.preferences.firstIndex(where: {
                $0.key.lowercased() == normalizedKey.lowercased()
            }) {
                state.preferences[index].value = normalizedValue
                state.preferences[index].source = Self.normalize(source).isEmpty ? "user" : Self.normalize(source)
                state.preferences[index].updatedAt = Date()
                return state.preferences[index]
            }
            let preference = BadAppleHumanPreference(
                id: Self.newID(),
                key: normalizedKey,
                value: normalizedValue,
                source: Self.normalize(source).isEmpty ? "user" : Self.normalize(source),
                createdAt: Date(),
                updatedAt: Date()
            )
            state.preferences.append(preference)
            Self.trimPreferences(&state.preferences)
            return preference
        }
    }

    @discardableResult
    func forgetPreference(key: String) throws -> Bool {
        let normalizedKey = Self.normalize(key)
        guard !normalizedKey.isEmpty else { throw BadAppleHumanLayerError.emptyField("key") }
        return try mutate { state in
            let before = state.preferences.count
            state.preferences.removeAll { $0.key.lowercased() == normalizedKey.lowercased() }
            return state.preferences.count != before
        }
    }

    @discardableResult
    func addCommitment(
        title: String,
        owner: BadAppleCommitmentOwner,
        dueAt: Date?,
        threadID: String?,
        detail: String = ""
    ) throws -> BadAppleHumanCommitment {
        let normalizedTitle = Self.normalize(title)
        guard !normalizedTitle.isEmpty else { throw BadAppleHumanLayerError.emptyField("title") }
        return try mutate { state in
            let commitment = BadAppleHumanCommitment(
                id: Self.newID(),
                title: normalizedTitle,
                owner: owner,
                status: .active,
                dueAt: dueAt,
                threadID: Self.normalize(threadID ?? "").isEmpty ? nil : Self.normalize(threadID ?? ""),
                detail: Self.normalize(detail),
                createdAt: Date(),
                updatedAt: Date(),
                lastNotifiedAt: nil
            )
            state.commitments.append(commitment)
            Self.trimCommitments(&state.commitments)
            return commitment
        }
    }

    @discardableResult
    func updateCommitment(id: String, status: BadAppleHumanStatus) throws -> BadAppleHumanCommitment? {
        let normalizedID = Self.normalize(id)
        guard !normalizedID.isEmpty else { throw BadAppleHumanLayerError.emptyField("id") }
        return try mutate { state in
            guard let index = try Self.resolveCommitmentIndex(normalizedID, in: state) else {
                return nil
            }
            state.commitments[index].status = status
            state.commitments[index].updatedAt = Date()
            return state.commitments[index]
        }
    }

    private static func resolveCommitmentIndex(_ id: String, in state: BadAppleHumanState) throws -> Int? {
        if let exact = state.commitments.firstIndex(where: { $0.id == id }) {
            return exact
        }
        let matches = state.commitments.indices.filter { state.commitments[$0].id.hasPrefix(id) }
        if matches.count > 1 { throw BadAppleHumanLayerError.ambiguousID(id) }
        return matches.first
    }

    @discardableResult
    func upsertThread(
        id: String?,
        title: String,
        summary: String,
        nextAction: String,
        status: BadAppleHumanStatus
    ) throws -> BadAppleLifeThread {
        let normalizedTitle = Self.normalize(title)
        let normalizedID = Self.normalize(id ?? "")
        return try mutate { state in
            var index: Int? = nil
            if !normalizedID.isEmpty {
                if let exact = state.threads.firstIndex(where: { $0.id == normalizedID }) {
                    index = exact
                } else {
                    let matches = state.threads.indices.filter {
                        state.threads[$0].id.hasPrefix(normalizedID)
                    }
                    if matches.count > 1 { throw BadAppleHumanLayerError.ambiguousID(normalizedID) }
                    index = matches.first
                }
            }
            if let index {
                if !normalizedTitle.isEmpty { state.threads[index].title = normalizedTitle }
                let normalizedSummary = Self.normalize(summary)
                let normalizedNext = Self.normalize(nextAction)
                if !normalizedSummary.isEmpty { state.threads[index].summary = normalizedSummary }
                if !normalizedNext.isEmpty { state.threads[index].nextAction = normalizedNext }
                state.threads[index].status = status
                state.threads[index].updatedAt = Date()
                return state.threads[index]
            }
            guard !normalizedTitle.isEmpty else { throw BadAppleHumanLayerError.emptyField("title") }
            let thread = BadAppleLifeThread(
                id: Self.newID(),
                title: normalizedTitle,
                summary: Self.normalize(summary),
                nextAction: Self.normalize(nextAction),
                status: status,
                createdAt: Date(),
                updatedAt: Date()
            )
            state.threads.append(thread)
            Self.trimThreads(&state.threads)
            return thread
        }
    }

    func setAttentionMode(_ mode: BadAppleAttentionMode) throws {
        try mutate { state in
            state.attentionMode = mode
        }
    }

    func route(event: BadAppleHumanEvent, now: Date = Date()) -> BadAppleInteractionDecision {
        _ = now
        let importance = min(5, max(1, event.importance))
        let urgency = min(5, max(1, event.urgency))
        if importance <= 1 && !event.requiresAction {
            return BadAppleInteractionDecision(
                channel: .silent,
                reason: "importance <= 1 with no action needed — dropped silently"
            )
        }
        switch snapshot().attentionMode {
        case .sleep:
            if urgency == 5 {
                return BadAppleInteractionDecision(
                    channel: .speak,
                    reason: "sleep mode: urgency 5 speaks, everything else waits"
                )
            }
            return BadAppleInteractionDecision(
                channel: .queue,
                reason: "sleep mode: queued for later"
            )
        case .quiet:
            if urgency == 5 {
                return BadAppleInteractionDecision(
                    channel: .speak,
                    reason: "quiet mode: urgency 5 speaks"
                )
            }
            if event.requiresAction && urgency >= 4 {
                return BadAppleInteractionDecision(
                    channel: .notify,
                    reason: "quiet mode: actionable event with urgency >= 4 notifies"
                )
            }
            return BadAppleInteractionDecision(
                channel: .queue,
                reason: "quiet mode: queued for later"
            )
        case .focus:
            if urgency == 5 {
                return BadAppleInteractionDecision(
                    channel: .speak,
                    reason: "focus mode: urgency 5 speaks"
                )
            }
            if event.requiresAction || urgency >= 4 {
                return BadAppleInteractionDecision(
                    channel: .notify,
                    reason: "focus mode: actionable or urgency >= 4 notifies"
                )
            }
            return BadAppleInteractionDecision(
                channel: .queue,
                reason: "focus mode: queued for later"
            )
        case .available:
            if event.voiceRequested && (event.requiresAction || urgency >= 3) {
                return BadAppleInteractionDecision(
                    channel: .speak,
                    reason: "available mode: voice requested for an actionable or urgent event"
                )
            }
            return BadAppleInteractionDecision(
                channel: .notify,
                reason: "available mode: banner notification"
            )
        }
    }

    func enqueue(event: BadAppleHumanEvent, decision: BadAppleInteractionDecision) throws {
        try mutate { state in
            let item = BadAppleHumanInboxItem(event: event, decision: decision, status: .waiting)
            if let index = state.inbox.firstIndex(where: { $0.event.id == event.id }) {
                state.inbox[index] = item
            } else {
                state.inbox.append(item)
            }
            Self.trimInbox(&state.inbox)
        }
    }

    func claimDueCommitments(now: Date = Date()) throws -> [BadAppleHumanCommitment] {
        try mutate { state in
            var claimed: [BadAppleHumanCommitment] = []
            for index in state.commitments.indices {
                let commitment = state.commitments[index]
                guard commitment.status == .active || commitment.status == .waiting,
                      let dueAt = commitment.dueAt, dueAt <= now
                else { continue }
                if let last = commitment.lastNotifiedAt,
                   now.timeIntervalSince(last) < Self.renotifyInterval {
                    continue
                }
                state.commitments[index].lastNotifiedAt = now
                state.commitments[index].updatedAt = now
                claimed.append(state.commitments[index])
            }
            return claimed
        }
    }

    func claimHeldEvents(limit: Int = 20) throws -> [BadAppleHumanEvent] {
        let capped = max(1, min(limit, 100))
        return try mutate { state in
            let indices = state.inbox.indices
                .filter { state.inbox[$0].status == .waiting }
                .sorted { state.inbox[$0].event.createdAt < state.inbox[$1].event.createdAt }
                .prefix(capped)
            var events: [BadAppleHumanEvent] = []
            for index in indices {
                state.inbox[index].status = .completed
                events.append(state.inbox[index].event)
            }
            return events
        }
    }

    static func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter.string(from: date)
    }

    static func shortID(_ id: String) -> String {
        String(id.prefix(8))
    }

    private static func isOpen(_ status: BadAppleHumanStatus) -> Bool {
        status == .active || status == .waiting
    }

    func homeText(now: Date = Date()) -> String {
        let state = snapshot()
        var sections: [String] = []

        let open = state.commitments.filter { Self.isOpen($0.status) }
        let horizon = now.addingTimeInterval(Self.renotifyInterval)
        let due = open.filter { ($0.dueAt ?? .distantFuture) <= horizon }
            .sorted { ($0.dueAt ?? .distantFuture) < ($1.dueAt ?? .distantFuture) }
        let dueIDs = Set(due.map(\.id))
        let rest = open.filter { !dueIDs.contains($0.id) }

        if !due.isEmpty {
            var lines = ["NOW"]
            for commitment in due {
                let when: String
                if let dueAt = commitment.dueAt {
                    when = dueAt <= now
                        ? "overdue since \(Self.formatDate(dueAt))"
                        : "due \(Self.formatDate(dueAt))"
                } else {
                    when = "due soon"
                }
                lines.append("- \(Self.shortID(commitment.id)) \(commitment.title) — \(when)")
            }
            sections.append(lines.joined(separator: "\n"))
        }

        let yours = rest.filter { $0.owner == .user }
        if !yours.isEmpty {
            var lines = ["WAITING ON YOU"]
            for commitment in yours {
                var line = "- \(Self.shortID(commitment.id)) \(commitment.title)"
                if let dueAt = commitment.dueAt {
                    line += " — due \(Self.formatDate(dueAt))"
                }
                lines.append(line)
            }
            sections.append(lines.joined(separator: "\n"))
        }

        let mine = rest.filter { $0.owner == .badApple }
        if !mine.isEmpty {
            var lines = ["I'M HANDLING"]
            for commitment in mine {
                var line = "- \(Self.shortID(commitment.id)) \(commitment.title)"
                if let dueAt = commitment.dueAt {
                    line += " — due \(Self.formatDate(dueAt))"
                }
                lines.append(line)
            }
            sections.append(lines.joined(separator: "\n"))
        }

        if !state.preferences.isEmpty {
            var lines = ["REMEMBERING"]
            for preference in state.preferences.sorted(by: { $0.key < $1.key }) {
                lines.append("- \(preference.key): \(preference.value)")
            }
            sections.append(lines.joined(separator: "\n"))
        }

        let openThreads = state.threads.filter { Self.isOpen($0.status) }
        if !openThreads.isEmpty {
            var lines = ["LIFE THREADS"]
            for thread in openThreads {
                var line = "- \(Self.shortID(thread.id)) \(thread.title)"
                if !thread.nextAction.isEmpty {
                    line += " — next: \(thread.nextAction)"
                }
                lines.append(line)
            }
            sections.append(lines.joined(separator: "\n"))
        }

        let held = state.inbox.filter { $0.status == .waiting }
        if !held.isEmpty {
            var lines = ["HELD FOR LATER"]
            for item in held {
                var line = "- \(item.event.title)"
                if !item.event.body.isEmpty {
                    line += ": \(item.event.body)"
                }
                lines.append(line)
            }
            sections.append(lines.joined(separator: "\n"))
        }

        return sections.isEmpty
            ? "Human Home is clear. Nothing is waiting on us."
            : sections.joined(separator: "\n\n")
    }

    func promptContext(now: Date = Date(), maximumCharacters: Int = 4_000) -> String {
        let state = snapshot()
        let openCommitments = state.commitments.filter { Self.isOpen($0.status) }
        let openThreads = state.threads.filter { Self.isOpen($0.status) }
        let held = state.inbox.filter { $0.status == .waiting }
        let meaningful = state.attentionMode != .available
            || !state.preferences.isEmpty
            || !openCommitments.isEmpty
            || !openThreads.isEmpty
            || !held.isEmpty
        guard meaningful else { return "" }

        var lines: [String] = ["attention_mode: \(state.attentionMode.rawValue)"]
        if !state.preferences.isEmpty {
            lines.append("preferences:")
            for preference in state.preferences.sorted(by: { $0.key < $1.key }) {
                lines.append("- \(preference.key): \(preference.value)")
            }
        }
        if !openCommitments.isEmpty {
            lines.append("commitments:")
            for commitment in openCommitments {
                var line = "- [\(Self.shortID(commitment.id))] \(commitment.title) (\(commitment.owner.rawValue), \(commitment.status.rawValue))"
                if let dueAt = commitment.dueAt {
                    line += " due \(Self.formatDate(dueAt))"
                }
                lines.append(line)
            }
        }
        if !openThreads.isEmpty {
            lines.append("life_threads:")
            for thread in openThreads {
                var line = "- [\(Self.shortID(thread.id))] \(thread.title) (\(thread.status.rawValue))"
                if !thread.nextAction.isEmpty {
                    line += " next: \(thread.nextAction)"
                }
                lines.append(line)
            }
        }
        if !held.isEmpty {
            lines.append("held_notifications:")
            for item in held {
                lines.append("- \(item.event.title)")
            }
        }

        var text = lines.joined(separator: "\n")
        if text.count > maximumCharacters {
            text = String(text.prefix(maximumCharacters))
        }
        return text
    }

    static func parseDueDate(_ text: String, now: Date = Date(), calendar: Calendar = .current) -> Date? {
        let trimmed = normalize(text)
        guard !trimmed.isEmpty else { return nil }
        let lower = trimmed.lowercased()

        if let relative = parseRelativeDate(lower, now: now, calendar: calendar) {
            return relative
        }

        let dayWords: [(String, Int)] = [("today", 0), ("tomorrow", 1)]
        for (word, offset) in dayWords {
            if lower == word {
                guard let day = calendar.date(byAdding: .day, value: offset, to: now) else { return nil }
                return calendar.startOfDay(for: day)
            }
            if lower.hasPrefix(word + " ") {
                let timeText = String(lower.dropFirst(word.count)).trimmingCharacters(in: .whitespaces)
                guard let day = calendar.date(byAdding: .day, value: offset, to: now),
                      let (hour, minute) = parseTimeComponent(timeText)
                else { return nil }
                return calendar.date(
                    bySettingHour: hour, minute: minute, second: 0,
                    of: calendar.startOfDay(for: day)
                )
            }
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: trimmed) { return date }
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = iso.date(from: trimmed) { return date }

        let formats = ["yyyy-MM-dd HH:mm", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd"]
        for format in formats {
            let formatter = DateFormatter()
            formatter.calendar = calendar
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = format
            if let date = formatter.date(from: trimmed) { return date }
        }
        return nil
    }

    private static func parseRelativeDate(_ lower: String, now: Date, calendar: Calendar) -> Date? {
        guard let regex = try? NSRegularExpression(
            pattern: #"^in\s+(\d+)\s+(minute|minutes|min|hour|hours|hr|day|days)\s*$"#,
            options: [.caseInsensitive]
        ), let match = regex.firstMatch(
            in: lower, options: [], range: NSRange(lower.startIndex..., in: lower)
        ), match.numberOfRanges == 3,
           let countRange = Range(match.range(at: 1), in: lower),
           let unitRange = Range(match.range(at: 2), in: lower),
           let count = Int(lower[countRange])
        else { return nil }
        let unit = lower[unitRange]
        let component: Calendar.Component
        if unit.hasPrefix("min") {
            component = .minute
        } else if unit.hasPrefix("h") {
            component = .hour
        } else {
            component = .day
        }
        return calendar.date(byAdding: component, value: count, to: now)
    }

    private static func parseTimeComponent(_ text: String) -> (Int, Int)? {
        guard let regex = try? NSRegularExpression(
            pattern: #"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$"#,
            options: [.caseInsensitive]
        ), let match = regex.firstMatch(
            in: text, options: [], range: NSRange(text.startIndex..., in: text)
        ), let hourRange = Range(match.range(at: 1), in: text),
           let rawHour = Int(text[hourRange])
        else { return nil }
        var minute = 0
        if let minuteRange = Range(match.range(at: 2), in: text) {
            guard let parsed = Int(text[minuteRange]), parsed <= 59 else { return nil }
            minute = parsed
        }
        var hour = rawHour
        if let meridiemRange = Range(match.range(at: 3), in: text) {
            let meridiem = text[meridiemRange].lowercased()
            guard (1...12).contains(hour) else { return nil }
            if meridiem == "am" {
                if hour == 12 { hour = 0 }
            } else if hour != 12 {
                hour += 12
            }
        } else {
            guard (0...23).contains(hour) else { return nil }
        }
        return (hour, minute)
    }

    private static func normalize(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func newID() -> String {
        UUID().uuidString.lowercased()
    }

    private static func isTerminal(_ status: BadAppleHumanStatus) -> Bool {
        status == .completed || status == .dismissed
    }

    private static func trimCommitments(_ list: inout [BadAppleHumanCommitment]) {
        while list.count > maxCommitments {
            if let index = list.indices
                .filter({ isTerminal(list[$0].status) })
                .min(by: { list[$0].updatedAt < list[$1].updatedAt }) {
                list.remove(at: index)
            } else if let index = list.indices.min(by: {
                list[$0].createdAt < list[$1].createdAt
            }) {
                list.remove(at: index)
            } else {
                break
            }
        }
    }

    private static func trimPreferences(_ list: inout [BadAppleHumanPreference]) {
        while list.count > maxPreferences {
            if let index = list.indices.min(by: { list[$0].updatedAt < list[$1].updatedAt }) {
                list.remove(at: index)
            } else {
                break
            }
        }
    }

    private static func trimThreads(_ list: inout [BadAppleLifeThread]) {
        while list.count > maxThreads {
            if let index = list.indices
                .filter({ isTerminal(list[$0].status) })
                .min(by: { list[$0].updatedAt < list[$1].updatedAt }) {
                list.remove(at: index)
            } else if let index = list.indices.min(by: {
                list[$0].createdAt < list[$1].createdAt
            }) {
                list.remove(at: index)
            } else {
                break
            }
        }
    }

    private static func trimInbox(_ list: inout [BadAppleHumanInboxItem]) {
        while list.count > maxInbox {
            if let index = list.indices
                .filter({ isTerminal(list[$0].status) })
                .min(by: { list[$0].event.createdAt < list[$1].event.createdAt }) {
                list.remove(at: index)
            } else if let index = list.indices.min(by: {
                list[$0].event.createdAt < list[$1].event.createdAt
            }) {
                list.remove(at: index)
            } else {
                break
            }
        }
    }
}
