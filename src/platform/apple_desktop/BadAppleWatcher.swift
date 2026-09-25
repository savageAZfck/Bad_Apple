import Foundation

// BadAppleWatcher — persistent attention.
//
// Schedules run on time; watchers run on truth. "Watch for the build to fail",
// "watch for an email from Sarah", "watch that folder" — each is a checkable
// condition the daemon evaluates on a minute cadence, and when it trips the
// watcher notifies the user and can fire a goal into the agent loop.
//
// Storage: ~/.bad_apple/watchers.json — survives restarts like schedules.
// Conditions are deliberately primitive (file/process/mail checks) — cheap to
// evaluate every minute, no model burn, no network. One-shot by default;
// "keep watching" keeps a tripped watcher armed for the next occurrence.

struct BadAppleWatch: Codable, Equatable {
    var id: String
    var name: String
    var kind: String            // file_exists, file_changed, process_running, process_gone, text_present, mail_from
    var target: String          // path, process name, or sender substring
    var contains: String        // text_present payload (unused elsewhere)
    var act: String             // agent goal to run when tripped (may be "")
    var every: TimeInterval     // min seconds between checks
    var persistent: Bool        // re-arm after firing instead of disabling
    var enabled: Bool
    var lastCheck: TimeInterval
    var lastState: String       // for file_changed: the observed signature
    var firedAt: TimeInterval?
}

enum BadAppleWatcher {

    static let filePath = NSHomeDirectory() + "/.bad_apple/watchers.json"

    // MARK: - Persistence

    static func load() -> [BadAppleWatch] {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else { return [] }
        return (try? JSONDecoder().decode([BadAppleWatch].self, from: data)) ?? []
    }

    static func save(_ items: [BadAppleWatch]) {
        let capped = Array(items.suffix(100))
        guard let data = try? JSONEncoder().encode(capped) else { return }
        try? FileManager.default.createDirectory(
            atPath: NSHomeDirectory() + "/.bad_apple",
            withIntermediateDirectories: true
        )
        try? data.write(to: URL(fileURLWithPath: filePath), options: .atomic)
    }

    /// Parse "kind:target[:text]" style specs into a watcher.
    /// Kinds: file_exists <path>, file_changed <path>, process_running <name>,
    /// process_gone <name>, text_present <path> containing <text>,
    /// mail_from <sender>.
    static func add(
        kind: String, target: String, contains: String,
        act: String, every: TimeInterval, persistent: Bool, name: String
    ) throws -> BadAppleWatch {
        let normalizedKind = kind.lowercased().trimmingCharacters(in: .whitespaces)
        guard ["file_exists", "file_changed", "process_running", "process_gone",
               "text_present", "mail_from"].contains(normalizedKind) else {
            throw BadAppleWatchError.badKind(kind)
        }
        guard !target.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw BadAppleWatchError.emptyTarget
        }
        var items = load()
        let watch = BadAppleWatch(
            id: UUID().uuidString.lowercased(),
            name: name.isEmpty ? "\(normalizedKind) \(target)" : name,
            kind: normalizedKind,
            target: target,
            contains: contains,
            act: act,
            every: max(every, 60),
            persistent: persistent,
            enabled: true,
            lastCheck: 0,
            lastState: "",
            firedAt: nil
        )
        items.append(watch)
        save(items)
        return watch
    }

    static func remove(idOrName: String) -> BadAppleWatch? {
        var items = load()
        let needle = idOrName.lowercased()
        guard let idx = items.firstIndex(where: {
            $0.id == needle || $0.id.hasPrefix(needle) || $0.name.lowercased().contains(needle)
        }) else { return nil }
        let removed = items.remove(at: idx)
        save(items)
        return removed
    }

    // MARK: - Evaluation

    /// Evaluate every enabled watcher whose interval has elapsed. Returns
    /// (watcher, evidence line) pairs for the ones that tripped.
    static func tripped(now: Date = Date()) -> [(BadAppleWatch, String)] {
        var items = load()
        var hits: [(BadAppleWatch, String)] = []
        for (idx, watch) in items.enumerated() where watch.enabled {
            let ts = now.timeIntervalSince1970
            guard ts - watch.lastCheck >= watch.every else { continue }
            items[idx].lastCheck = ts
            let (tripped, evidence, state) = evaluate(watch)
            items[idx].lastState = state
            if tripped {
                items[idx].firedAt = ts
                if !watch.persistent { items[idx].enabled = false }
                hits.append((items[idx], evidence))
            }
        }
        save(items)
        return hits
    }

    /// Run one condition. Returns (tripped, evidence, newState). file_changed
    /// compares the observed signature to lastState — the baseline is set on
    /// first check so "changed" means "changed since you asked".
    private static func evaluate(_ w: BadAppleWatch) -> (Bool, String, String) {
        switch w.kind {
        case "file_exists":
            let path = expand(w.target)
            let exists = FileManager.default.fileExists(atPath: path)
            return (exists, exists ? "\(path) exists" : "", w.lastState)

        case "file_changed":
            let path = expand(w.target)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
                  let mtime = attrs[.modificationDate] as? Date,
                  let size = attrs[.size] as? Int else {
                return (false, "", w.lastState)
            }
            let signature = "\(mtime.timeIntervalSince1970):\(size)"
            if w.lastState.isEmpty {
                return (false, "", signature)  // first check sets the baseline
            }
            let changed = signature != w.lastState
            return (changed, changed ? "\(path) changed (now \(size) bytes)" : "", signature)

        case "process_running":
            let running = processExists(w.target)
            return (running, running ? "\(w.target) is running" : "", w.lastState)

        case "process_gone":
            let running = processExists(w.target)
            return (!running, !running ? "\(w.target) is no longer running" : "", w.lastState)

        case "text_present":
            let path = expand(w.target)
            guard let text = try? String(contentsOfFile: path, encoding: .utf8),
                  text.localizedCaseInsensitiveContains(w.contains) else {
                return (false, "", w.lastState)
            }
            return (true, "\(path) now contains '\(w.contains)'", w.lastState)

        case "mail_from":
            guard let res = callAqua(command: "mail_read", payload: ["limit": 25], timeout: 30),
                  let output = res["output"] as? String else { return (false, "", w.lastState) }
            let hit = output.lowercased().contains(w.target.lowercased())
            let state = output.hashValue.description
            if hit && state != w.lastState {
                return (true, "new mail matching '\(w.target)' in the inbox", state)
            }
            return (false, "", state)

        default:
            return (false, "", w.lastState)
        }
    }

    private static func expand(_ path: String) -> String {
        (path as NSString).expandingTildeInPath
    }

    private static func processExists(_ name: String) -> Bool {
        let r = watcherProcess(
            launchPath: "/usr/bin/pgrep",
            arguments: ["-il", name],
            timeout: 5
        )
        return r.exitCode == 0 && !r.stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

enum BadAppleWatchError: LocalizedError {
    case badKind(String)
    case emptyTarget
    var errorDescription: String? {
        switch self {
        case .badKind(let k):
            return "unknown watch kind '\(k)'. Use file_exists, file_changed, process_running, process_gone, text_present, or mail_from."
        case .emptyTarget:
            return "a watch target is required — a path, process name, or sender."
        }
    }
}

private func watcherProcess(
    launchPath: String,
    arguments: [String],
    timeout: TimeInterval
) -> (stdout: String, stderr: String, exitCode: Int) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: launchPath)
    process.arguments = arguments
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    var stdoutData = Data()
    var stderrData = Data()
    let group = DispatchGroup()
    DispatchQueue.global().async(group: group) {
        stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    }
    DispatchQueue.global().async(group: group) {
        stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    }
    do {
        try process.run()
    } catch {
        return ("", "Error: \(error.localizedDescription)", -1)
    }
    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.05)
    }
    if process.isRunning {
        process.terminate()
        group.wait()
        return ("", "Error: timed out", -1)
    }
    group.wait()
    return (
        String(data: stdoutData, encoding: .utf8) ?? "",
        String(data: stderrData, encoding: .utf8) ?? "",
        Int(process.terminationStatus)
    )
}
