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
    var referenceID: String?

    init(id: String, kind: String, title: String, body: String, importance: Int, urgency: Int, requiresAction: Bool, voiceRequested: Bool, createdAt: Date, referenceID: String? = nil) {
        self.id = id
        self.kind = kind
        self.title = title
        self.body = body
        self.importance = importance
        self.urgency = urgency
        self.requiresAction = requiresAction
        self.voiceRequested = voiceRequested
        self.createdAt = createdAt
        self.referenceID = referenceID
    }
}

struct BadAppleInteractionDecision: Codable, Equatable, Sendable {
    let channel: BadAppleInteractionChannel
    let reason: String
}

struct BadAppleHumanInboxItem: Codable, Equatable, Sendable {
    let event: BadAppleHumanEvent
    let decision: BadAppleInteractionDecision
    var status: BadAppleHumanStatus
    var availableAfter: Date?

    init(event: BadAppleHumanEvent, decision: BadAppleInteractionDecision, status: BadAppleHumanStatus, availableAfter: Date? = nil) {
        self.event = event
        self.decision = decision
        self.status = status
        self.availableAfter = availableAfter
    }
}

struct BadApplePerson: Codable, Equatable, Sendable {
    let id: String
    var name: String
    var relationship: String
    var notes: String
    var lastContactAt: Date?
    var nextContactAt: Date?
    var status: BadAppleHumanStatus
    let createdAt: Date
    var updatedAt: Date
    var lastNotifiedAt: Date?
}

struct BadAppleConversationThread: Codable, Equatable, Sendable {
    let id: String
    var title: String
    var summary: String
    let sessionID: String
    var status: BadAppleHumanStatus
    let createdAt: Date
    var updatedAt: Date
    var lastOpenedAt: Date
}

struct BadAppleHumanState: Codable, Equatable, Sendable {
    var schemaVersion: Int
    var attentionMode: BadAppleAttentionMode
    var preferences: [BadAppleHumanPreference]
    var commitments: [BadAppleHumanCommitment]
    var threads: [BadAppleLifeThread]
    var inbox: [BadAppleHumanInboxItem]
    var people: [BadApplePerson]
    var conversationThreads: [BadAppleConversationThread]
    var activeConversationThreadID: String?

    init(
        schemaVersion: Int,
        attentionMode: BadAppleAttentionMode,
        preferences: [BadAppleHumanPreference],
        commitments: [BadAppleHumanCommitment],
        threads: [BadAppleLifeThread],
        inbox: [BadAppleHumanInboxItem],
        people: [BadApplePerson],
        conversationThreads: [BadAppleConversationThread],
        activeConversationThreadID: String?
    ) {
        self.schemaVersion = schemaVersion
        self.attentionMode = attentionMode
        self.preferences = preferences
        self.commitments = commitments
        self.threads = threads
        self.inbox = inbox
        self.people = people
        self.conversationThreads = conversationThreads
        self.activeConversationThreadID = activeConversationThreadID
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 2
        schemaVersion = max(2, decodedVersion)
        attentionMode = try container.decodeIfPresent(BadAppleAttentionMode.self, forKey: .attentionMode) ?? .available
        preferences = try container.decodeIfPresent([BadAppleHumanPreference].self, forKey: .preferences) ?? []
        commitments = try container.decodeIfPresent([BadAppleHumanCommitment].self, forKey: .commitments) ?? []
        threads = try container.decodeIfPresent([BadAppleLifeThread].self, forKey: .threads) ?? []
        inbox = try container.decodeIfPresent([BadAppleHumanInboxItem].self, forKey: .inbox) ?? []
        people = try container.decodeIfPresent([BadApplePerson].self, forKey: .people) ?? []
        conversationThreads = try container.decodeIfPresent([BadAppleConversationThread].self, forKey: .conversationThreads) ?? []
        activeConversationThreadID = try container.decodeIfPresent(String.self, forKey: .activeConversationThreadID)
    }
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

    static let schemaVersion = 2
    private static let maxPreferences = 500
    private static let maxCommitments = 1000
    private static let maxThreads = 500
    private static let maxInbox = 500
    private static let maxPeople = 500
    private static let maxConversationThreads = 200
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
            inbox: [],
            people: [],
            conversationThreads: [],
            activeConversationThreadID: nil
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

    func claimHeldEvents(limit: Int = 20, now: Date = Date()) throws -> [BadAppleHumanEvent] {
        let capped = max(1, min(limit, 100))
        return try mutate { state in
            let indices = state.inbox.indices
                .filter {
                    state.inbox[$0].status == .waiting
                        && (state.inbox[$0].availableAfter ?? .distantPast) <= now
                }
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

    @discardableResult
    func snoozeCommitment(id: String, until: Date) throws -> BadAppleHumanCommitment? {
        let normalizedID = Self.normalize(id)
        guard !normalizedID.isEmpty else { throw BadAppleHumanLayerError.emptyField("id") }
        return try mutate { state in
            guard let index = try Self.resolveCommitmentIndex(normalizedID, in: state) else {
                return nil
            }
            state.commitments[index].dueAt = until
            state.commitments[index].lastNotifiedAt = nil
            state.commitments[index].status = .active
            state.commitments[index].updatedAt = Date()
            return state.commitments[index]
        }
    }

    func snooze(event: BadAppleHumanEvent, until: Date, decision: BadAppleInteractionDecision) throws {
        try mutate { state in
            let item = BadAppleHumanInboxItem(
                event: event, decision: decision, status: .waiting, availableAfter: until
            )
            if let index = state.inbox.firstIndex(where: { $0.event.id == event.id }) {
                state.inbox[index] = item
            } else {
                state.inbox.append(item)
            }
            Self.trimInbox(&state.inbox)
        }
    }

    @discardableResult
    func rememberPerson(name: String, relationship: String, notes: String, nextContactAt: Date?) throws -> BadApplePerson {
        let normalizedName = Self.normalize(name)
        guard !normalizedName.isEmpty else { throw BadAppleHumanLayerError.emptyField("name") }
        return try mutate { state in
            let normalizedRelationship = Self.normalize(relationship)
            let normalizedNotes = Self.normalize(notes)
            if let index = state.people.firstIndex(where: {
                $0.name.lowercased() == normalizedName.lowercased()
            }) {
                if !normalizedRelationship.isEmpty {
                    state.people[index].relationship = normalizedRelationship
                }
                if !normalizedNotes.isEmpty {
                    state.people[index].notes = normalizedNotes
                }
                if let nextContactAt {
                    state.people[index].nextContactAt = nextContactAt
                }
                if !Self.isOpen(state.people[index].status) {
                    state.people[index].status = .active
                }
                state.people[index].updatedAt = Date()
                return state.people[index]
            }
            let person = BadApplePerson(
                id: Self.newID(),
                name: normalizedName,
                relationship: normalizedRelationship,
                notes: normalizedNotes,
                lastContactAt: nil,
                nextContactAt: nextContactAt,
                status: .active,
                createdAt: Date(),
                updatedAt: Date(),
                lastNotifiedAt: nil
            )
            state.people.append(person)
            Self.trimPeople(&state.people)
            return person
        }
    }

    @discardableResult
    func recordContact(idOrName: String, notes: String, nextContactAt: Date?, at: Date = Date()) throws -> BadApplePerson? {
        let normalized = Self.normalize(idOrName)
        guard !normalized.isEmpty else { throw BadAppleHumanLayerError.emptyField("person") }
        return try mutate { state in
            guard let index = try Self.resolvePersonIndex(normalized, in: state) else {
                return nil
            }
            state.people[index].lastContactAt = at
            let normalizedNotes = Self.normalize(notes)
            if !normalizedNotes.isEmpty {
                state.people[index].notes = normalizedNotes
            }
            if let nextContactAt {
                state.people[index].nextContactAt = nextContactAt
            }
            state.people[index].lastNotifiedAt = nil
            state.people[index].updatedAt = at
            return state.people[index]
        }
    }

    @discardableResult
    func forgetPerson(idOrName: String) throws -> Bool {
        let normalized = Self.normalize(idOrName)
        guard !normalized.isEmpty else { throw BadAppleHumanLayerError.emptyField("person") }
        return try mutate { state in
            guard let index = try Self.resolvePersonIndex(normalized, in: state) else {
                return false
            }
            state.people.remove(at: index)
            return true
        }
    }

    func claimDuePeople(now: Date = Date()) throws -> [BadApplePerson] {
        try mutate { state in
            var claimed: [BadApplePerson] = []
            for index in state.people.indices {
                let person = state.people[index]
                guard person.status == .active || person.status == .waiting,
                      let nextContactAt = person.nextContactAt, nextContactAt <= now
                else { continue }
                if let last = person.lastNotifiedAt,
                   now.timeIntervalSince(last) < Self.renotifyInterval {
                    continue
                }
                state.people[index].lastNotifiedAt = now
                state.people[index].updatedAt = now
                claimed.append(state.people[index])
            }
            return claimed
        }
    }

    private static func resolvePersonIndex(_ key: String, in state: BadAppleHumanState) throws -> Int? {
        if let exact = state.people.firstIndex(where: { $0.id == key }) {
            return exact
        }
        if let byName = state.people.firstIndex(where: { $0.name.lowercased() == key.lowercased() }) {
            return byName
        }
        let matches = state.people.indices.filter { state.people[$0].id.hasPrefix(key) }
        if matches.count > 1 { throw BadAppleHumanLayerError.ambiguousID(key) }
        return matches.first
    }

    @discardableResult
    func ensureDefaultConversationThread() throws -> BadAppleConversationThread {
        try mutate { state in
            Self.ensureDefaultThreadLocked(&state)
        }
    }

    func activeConversationThread() throws -> BadAppleConversationThread {
        try mutate { state in
            if let activeID = state.activeConversationThreadID,
               let index = state.conversationThreads.firstIndex(where: {
                   $0.id == activeID && Self.isOpen($0.status)
               }) {
                return state.conversationThreads[index]
            }
            return Self.ensureDefaultThreadLocked(&state)
        }
    }

    @discardableResult
    func createConversationThread(title: String, summary: String = "") throws -> BadAppleConversationThread {
        let normalizedTitle = Self.normalize(title)
        guard !normalizedTitle.isEmpty else { throw BadAppleHumanLayerError.emptyField("title") }
        return try mutate { state in
            let now = Date()
            let thread = BadAppleConversationThread(
                id: Self.newID(),
                title: normalizedTitle,
                summary: Self.normalize(summary),
                sessionID: "thread-\(UUID().uuidString.lowercased())",
                status: .active,
                createdAt: now,
                updatedAt: now,
                lastOpenedAt: now
            )
            state.conversationThreads.append(thread)
            state.activeConversationThreadID = thread.id
            Self.trimConversationThreads(&state.conversationThreads)
            return thread
        }
    }

    @discardableResult
    func switchConversationThread(idOrTitle: String) throws -> BadAppleConversationThread? {
        let normalized = Self.normalize(idOrTitle)
        guard !normalized.isEmpty else { throw BadAppleHumanLayerError.emptyField("thread") }
        return try mutate { state in
            guard let index = try Self.resolveThreadIndex(normalized, in: state),
                  Self.isOpen(state.conversationThreads[index].status) else {
                return nil
            }
            let now = Date()
            state.conversationThreads[index].lastOpenedAt = now
            state.conversationThreads[index].updatedAt = now
            state.activeConversationThreadID = state.conversationThreads[index].id
            return state.conversationThreads[index]
        }
    }

    @discardableResult
    func closeConversationThread(idOrTitle: String) throws -> BadAppleConversationThread? {
        let normalized = Self.normalize(idOrTitle)
        guard !normalized.isEmpty else { throw BadAppleHumanLayerError.emptyField("thread") }
        return try mutate { state in
            guard let index = try Self.resolveThreadIndex(normalized, in: state) else {
                return nil
            }
            let now = Date()
            state.conversationThreads[index].status = .completed
            state.conversationThreads[index].updatedAt = now
            let closed = state.conversationThreads[index]
            if state.activeConversationThreadID == closed.id {
                if let fallback = state.conversationThreads.indices
                    .filter({ Self.isOpen(state.conversationThreads[$0].status) })
                    .max(by: {
                        state.conversationThreads[$0].lastOpenedAt < state.conversationThreads[$1].lastOpenedAt
                    }) {
                    state.activeConversationThreadID = state.conversationThreads[fallback].id
                } else {
                    _ = Self.ensureDefaultThreadLocked(&state)
                }
            }
            return closed
        }
    }

    func listConversationThreads() throws -> [BadAppleConversationThread] {
        lock.lock()
        defer { lock.unlock() }
        return try withReadLock {
            loadStateUnlocked().conversationThreads
                .filter { Self.isOpen($0.status) }
                .sorted { $0.lastOpenedAt > $1.lastOpenedAt }
        }
    }

    private static func resolveThreadIndex(_ key: String, in state: BadAppleHumanState) throws -> Int? {
        if let exact = state.conversationThreads.firstIndex(where: { $0.id == key }) {
            return exact
        }
        if let byTitle = state.conversationThreads.firstIndex(where: {
            $0.title.lowercased() == key.lowercased()
        }) {
            return byTitle
        }
        let matches = state.conversationThreads.indices.filter {
            state.conversationThreads[$0].id.hasPrefix(key)
        }
        if matches.count > 1 { throw BadAppleHumanLayerError.ambiguousID(key) }
        return matches.first
    }

    private static func ensureDefaultThreadLocked(_ state: inout BadAppleHumanState) -> BadAppleConversationThread {
        if let index = state.conversationThreads.firstIndex(where: { $0.sessionID == "default" }) {
            let thread = state.conversationThreads[index]
            if state.activeConversationThreadID != thread.id || !isOpen(thread.status) {
                let now = Date()
                state.conversationThreads[index].status = .active
                state.conversationThreads[index].updatedAt = now
                state.conversationThreads[index].lastOpenedAt = now
                state.activeConversationThreadID = thread.id
            }
            return state.conversationThreads[index]
        }
        let now = Date()
        let thread = BadAppleConversationThread(
            id: newID(),
            title: "General",
            summary: "",
            sessionID: "default",
            status: .active,
            createdAt: now,
            updatedAt: now,
            lastOpenedAt: now
        )
        state.conversationThreads.append(thread)
        state.activeConversationThreadID = thread.id
        trimConversationThreads(&state.conversationThreads)
        return thread
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

        let openPeople = state.people.filter { Self.isOpen($0.status) }
        if !openPeople.isEmpty {
            var lines = ["PEOPLE TO REMEMBER"]
            for person in openPeople.sorted(by: { $0.name < $1.name }) {
                var line = "- \(Self.shortID(person.id)) \(person.name)"
                if !person.relationship.isEmpty {
                    line += " (\(person.relationship))"
                }
                if let lastContactAt = person.lastContactAt {
                    line += " — last contact \(Self.formatDate(lastContactAt))"
                }
                if let nextContactAt = person.nextContactAt {
                    line += " — next \(Self.formatDate(nextContactAt))"
                }
                lines.append(line)
            }
            sections.append(lines.joined(separator: "\n"))
        }

        let openConversations = state.conversationThreads
            .filter { Self.isOpen($0.status) }
            .sorted { $0.lastOpenedAt > $1.lastOpenedAt }
        if !openConversations.isEmpty {
            var lines = ["CONVERSATIONS"]
            for thread in openConversations {
                let marker = thread.id == state.activeConversationThreadID ? "*" : "-"
                lines.append("\(marker) \(Self.shortID(thread.id)) \(thread.title)")
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
        let openPeople = state.people.filter { Self.isOpen($0.status) }
        let held = state.inbox.filter { $0.status == .waiting }
        let activeConversation = state.activeConversationThreadID.flatMap { activeID in
            state.conversationThreads.first { $0.id == activeID && Self.isOpen($0.status) }
        }
        let meaningful = state.attentionMode != .available
            || !state.preferences.isEmpty
            || !openCommitments.isEmpty
            || !openThreads.isEmpty
            || !openPeople.isEmpty
            || !held.isEmpty
            || activeConversation != nil
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
        if !openPeople.isEmpty {
            lines.append("people:")
            for person in openPeople.sorted(by: { $0.name < $1.name }) {
                var line = "- \(person.name)"
                if !person.relationship.isEmpty {
                    line += " (\(person.relationship))"
                }
                if let nextContactAt = person.nextContactAt {
                    line += " next contact \(Self.formatDate(nextContactAt))"
                }
                lines.append(line)
            }
        }
        if let activeConversation {
            var line = "conversation_thread: \(activeConversation.title)"
            if !activeConversation.summary.isEmpty {
                line += " — \(activeConversation.summary)"
            }
            lines.append(line)
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
                var timeText = String(lower.dropFirst(word.count)).trimmingCharacters(in: .whitespaces)
                if timeText.hasPrefix("at ") {
                    timeText = String(timeText.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
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

    private static func trimPeople(_ list: inout [BadApplePerson]) {
        while list.count > maxPeople {
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

    private static func trimConversationThreads(_ list: inout [BadAppleConversationThread]) {
        while list.count > maxConversationThreads {
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
}
