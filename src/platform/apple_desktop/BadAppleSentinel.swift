import Foundation
import Darwin

// BadAppleSentinel — the defensive organ.
//
// Watches the system for hostile actors the way IFY watches Bad Apple
// herself: detect, gather the trail back to source, report to the user.
// Every check is read-only — the sentinel never kills, quarantines, or
// strikes back; it builds the evidence chain so the human decides.
//
// Coverage:
//   - persistence diff — new/changed LaunchAgents and LaunchDaemons
//   - unsigned network listeners — processes holding open TCP ports
//   - self-integrity — Bad Apple's own binaries drifting from baseline
//
// Every finding carries a trail: the spawning process chain, the binary's
// signature status, its quarantine origin (download source via Spotlight
// metadata), birth time, and the launchd item responsible for it — the
// path from "something fired" back to "where it came from".
//
// Baseline contract: the first scan snapshots current state silently and
// only later diffs alert, so the machine's known-good state is the zero.

enum BadAppleSentinel {

    struct Finding: Codable, Equatable {
        var ts: TimeInterval
        var severity: String      // "info" | "warn" | "alert"
        var kind: String          // launchd_new, listener_unsigned, self_drift, ...
        var summary: String       // one plain-English line for the user
        var trail: [String]       // evidence chain, source first
    }

    struct PersistenceItem: Codable, Equatable {
        var path: String
        var sha256: String
        var program: String
        var signature: String
        var origin: String
    }

    struct Snapshot: Codable, Equatable {
        var takenAt: TimeInterval
        var persistence: [PersistenceItem]
        var listeners: [String]          // binary paths holding listen sockets
        var selfHashes: [String: String] // path -> sha256
    }

    static let directoryPath = NSHomeDirectory() + "/.bad_apple/sentinel"
    static let findingsPath = directoryPath + "/findings.jsonl"
    static let snapshotPath = directoryPath + "/snapshot.json"

    // MARK: - Scan

    /// Run a full pass. Returns only NEW findings — diffs against the stored
    /// baseline. The first ever run just establishes the baseline.
    static func scan() -> [Finding] {
        let fm = FileManager.default
        try? fm.createDirectory(atPath: directoryPath, withIntermediateDirectories: true)

        let current = gather()
        guard let baseline = loadSnapshot() else {
            saveSnapshot(current)
            return []
        }

        var findings: [Finding] = []
        let now = Date().timeIntervalSince1970
        let baseItems = Dictionary(uniqueKeysWithValues: baseline.persistence.map { ($0.path, $0) })
        let curItems = Dictionary(uniqueKeysWithValues: current.persistence.map { ($0.path, $0) })

        for item in current.persistence {
            if let base = baseItems[item.path] {
                if base.sha256 != item.sha256 {
                    findings.append(Finding(
                        ts: now, severity: "warn", kind: "launchd_changed",
                        summary: "A startup item changed on disk: \(item.path)",
                        trail: tracePersistence(item, prior: base)
                    ))
                }
            } else {
                findings.append(Finding(
                    ts: now, severity: "warn", kind: "launchd_new",
                    summary: "Something new installed itself to run at startup: \(item.program.isEmpty ? item.path : item.program)",
                    trail: tracePersistence(item, prior: nil)
                ))
            }
        }
        for gone in baseline.persistence where curItems[gone.path] == nil {
            findings.append(Finding(
                ts: now, severity: "info", kind: "launchd_removed",
                summary: "A startup item was removed: \(gone.path)",
                trail: ["removed plist: \(gone.path)", "was launching: \(gone.program)"]
            ))
        }

        let baseListeners = Set(baseline.listeners)
        for path in current.listeners where !baseListeners.contains(path) {
            findings.append(Finding(
                ts: now, severity: "warn", kind: "listener_new",
                summary: "A process started listening on the network: \(path)",
                trail: traceBinary(path)
            ))
        }

        for (path, hash) in current.selfHashes {
            if let base = baseline.selfHashes[path], base != hash {
                findings.append(Finding(
                    ts: now, severity: "alert", kind: "self_drift",
                    summary: "One of my own binaries changed outside a build: \(path)",
                    trail: traceBinary(path)
                ))
            }
        }

        if !findings.isEmpty {
            appendFindings(findings)
        }
        saveSnapshot(current)
        return findings
    }

    // MARK: - Gather

    private static func gather() -> Snapshot {
        Snapshot(
            takenAt: Date().timeIntervalSince1970,
            persistence: gatherPersistence(),
            listeners: unsignedListeners(),
            selfHashes: selfHashes()
        )
    }

    private static let persistenceDirs = [
        NSHomeDirectory() + "/Library/LaunchAgents",
        "/Library/LaunchAgents",
        "/Library/LaunchDaemons",
    ]

    private static func gatherPersistence() -> [PersistenceItem] {
        var items: [PersistenceItem] = []
        let fm = FileManager.default
        for dir in persistenceDirs {
            guard let names = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for name in names where name.hasSuffix(".plist") {
                let path = dir + "/" + name
                let program = plistProgram(path: path)
                items.append(PersistenceItem(
                    path: path,
                    sha256: sha256File(path),
                    program: program,
                    signature: signatureOf(program),
                    origin: downloadOrigin(program.isEmpty ? path : program)
                ))
            }
        }
        return items
    }

    private static func plistProgram(path: String) -> String {
        for key in ["ProgramArguments.0", "Program"] {
            let sel = key == "ProgramArguments.0" ? "Print :ProgramArguments:0" : "Print :Program"
            let r = sentinelProcess(
                launchPath: "/usr/libexec/PlistBuddy",
                arguments: ["-c", sel, path],
                timeout: 5
            )
            if r.exitCode == 0 {
                let v = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                if !v.isEmpty { return v }
            }
        }
        return ""
    }

    /// TCP listeners whose binary is unsigned or adhoc-signed — the class of
    /// process that should not be holding a port open without explanation.
    private static func unsignedListeners() -> [String] {
        let r = sentinelProcess(
            launchPath: "/usr/sbin/lsof",
            arguments: ["-nP", "-iTCP", "-sTCP:LISTEN", "-Fpc"],
            timeout: 15
        )
        guard r.exitCode == 0 else { return [] }
        var paths: [String] = []
        var pid: pid_t = 0
        for line in r.stdout.components(separatedBy: "\n") {
            if line.hasPrefix("p") {
                pid = pid_t(Int(line.dropFirst()) ?? 0)
            } else if line.hasPrefix("c"), pid > 0 {
                if let path = binaryPath(pid: pid), isUnsigned(path) {
                    paths.append(path)
                }
            }
        }
        return Array(Set(paths)).sorted()
    }

    private static func selfHashes() -> [String: String] {
        var hashes: [String: String] = [:]
        let candidates = [
            "/Applications/Bad Apple.app/Contents/MacOS/BadAppleMenuBar",
            "/Applications/Bad Apple.app/Contents/Helpers/badapple-engine",
            "/usr/local/bin/badapple",
            "/var/lib/bad_apple/bin/badapple-engine",
        ]
        for path in candidates where FileManager.default.fileExists(atPath: path) {
            hashes[path] = sha256File(path)
        }
        return hashes
    }

    // MARK: - Trail following

    /// Follow the evidence chain for a user-supplied target — a PID or a
    /// file path — and return the trail lines in report order.
    static func trace(target: String) -> [String] {
        let t = target.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return ["no target supplied"] }
        if let pid = pid_t(t), pid > 0 {
            guard let path = binaryPath(pid: pid) else {
                return ["pid \(pid) — no such process or path unavailable"]
            }
            var trail = ["pid \(pid) → \(path)"]
            trail.append(contentsOf: ancestry(pid: pid))
            trail.append(contentsOf: binaryEvidence(path))
            trail.append(contentsOf: openConnections(pid: pid))
            trail.append(contentsOf: launchdOwners(program: path))
            return trail
        }
        guard FileManager.default.fileExists(atPath: t) else {
            return ["\(t) — no such file"]
        }
        var trail = binaryEvidence(t)
        trail.append(contentsOf: launchdOwners(program: t))
        return trail
    }

    /// Trail for a persistence item — plist itself, the binary it launches,
    /// where the binary came from, and who signed it.
    private static func tracePersistence(_ item: PersistenceItem, prior: PersistenceItem?) -> [String] {
        var trail: [String] = []
        trail.append("plist: \(item.path)")
        if prior != nil { trail.append("hash changed — the launchd entry was rewritten") }
        if !item.program.isEmpty { trail.append("launches: \(item.program)") }
        if !item.signature.isEmpty { trail.append("signature: \(item.signature)") }
        if !item.origin.isEmpty { trail.append("downloaded from: \(item.origin)") }
        if !item.program.isEmpty {
            trail.append(contentsOf: binaryEvidence(item.program))
        }
        return trail
    }

    /// The chain of evidence for a binary: when it appeared, where it was
    /// downloaded from, who signed it, and what process owns it now.
    private static func traceBinary(_ path: String) -> [String] {
        var trail: [String] = ["binary: \(path)"]
        trail.append(contentsOf: binaryEvidence(path))
        trail.append(contentsOf: launchdOwners(program: path))
        if let pid = runningPid(path: path) {
            trail.append("running as pid \(pid)")
            trail.append(contentsOf: ancestry(pid: pid))
            trail.append(contentsOf: openConnections(pid: pid))
        }
        return trail
    }

    private static func binaryEvidence(_ path: String) -> [String] {
        var out: [String] = []
        let stat = sentinelProcess(
            launchPath: "/usr/bin/stat",
            arguments: ["-f", "%SB", path],
            timeout: 5
        )
        let born = stat.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if stat.exitCode == 0, !born.isEmpty {
            out.append("first appeared on disk: \(born)")
        }
        let origin = downloadOrigin(path)
        if !origin.isEmpty { out.append("downloaded from: \(origin)") }
        let sig = signatureOf(path)
        out.append("signature: \(sig.isEmpty ? "none (unsigned)" : sig)")
        return out
    }

    /// Walk the parent chain back to launchd — who spawned whom, in order.
    private static func ancestry(pid: pid_t) -> [String] {
        var chain: [String] = []
        var current = pid
        var hops = 0
        while current > 1 && hops < 8 {
            let r = sentinelProcess(
                launchPath: "/bin/ps",
                arguments: ["-o", "ppid=,comm=", "-p", String(current)],
                timeout: 5
            )
            let fields = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                .split(separator: " ", maxSplits: 1).map { String($0) }
            guard fields.count == 2, let parent = pid_t(fields[0]) else { break }
            chain.append("spawned by: \(fields[1]) (pid \(parent))")
            current = parent
            hops += 1
        }
        return chain
    }

    private static func openConnections(pid: pid_t) -> [String] {
        let r = sentinelProcess(
            launchPath: "/usr/sbin/lsof",
            arguments: ["-nP", "-i", "-a", "-p", String(pid)],
            timeout: 10
        )
        guard r.exitCode == 0 else { return [] }
        var out: [String] = []
        for line in r.stdout.components(separatedBy: "\n").dropFirst() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.contains("TCP") || trimmed.contains("UDP") {
                out.append("network: \(trimmed)")
            }
        }
        return Array(out.prefix(5))
    }

    /// Which launchd items reference this program — the persistence claim.
    private static func launchdOwners(program: String) -> [String] {
        var out: [String] = []
        let base = (program as NSString).lastPathComponent
        for dir in persistenceDirs {
            guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
            for name in names where name.hasSuffix(".plist") {
                let path = dir + "/" + name
                guard let text = try? String(contentsOfFile: path, encoding: .utf8),
                      text.contains(program) || text.contains(base) else { continue }
                out.append("started by: \(path)")
            }
        }
        return out
    }

    // MARK: - Evidence helpers

    private static func binaryPath(pid: pid_t) -> String? {
        var buf = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        let n = proc_pidpath(pid, &buf, UInt32(buf.count))
        guard n > 0 else { return nil }
        return String(cString: buf)
    }

    private static func runningPid(path: String) -> pid_t? {
        let r = sentinelProcess(
            launchPath: "/usr/bin/pgrep",
            arguments: ["-x", (path as NSString).lastPathComponent],
            timeout: 5
        )
        let first = r.stdout.components(separatedBy: "\n").first?
            .trimmingCharacters(in: .whitespaces) ?? ""
        return pid_t(first)
    }

    private static func isUnsigned(_ path: String) -> Bool {
        let sig = signatureOf(path)
        return sig.isEmpty || sig.contains("adhoc") || sig.contains("unsigned")
    }

    private static func signatureOf(_ path: String) -> String {
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return "" }
        let r = sentinelProcess(
            launchPath: "/usr/bin/codesign",
            arguments: ["-dv", "--verbose=2", path],
            timeout: 10
        )
        guard r.exitCode == 0 else { return "" }
        // codesign -dv reports on stderr: Authority=..., TeamIdentifier=...
        var authorities: [String] = []
        var team = ""
        for line in r.stderr.components(separatedBy: "\n") {
            if line.hasPrefix("Authority=") {
                authorities.append(String(line.dropFirst("Authority=".count)))
            } else if line.hasPrefix("TeamIdentifier=") {
                team = String(line.dropFirst("TeamIdentifier=".count))
            }
        }
        if authorities.isEmpty { return "adhoc" }
        return authorities.first! + (team.isEmpty ? "" : " [\(team)]")
    }

    /// Where a file came from — Spotlight records the download source URL in
    /// kMDItemWhereFroms for quarantined files. This is the trail to source.
    private static func downloadOrigin(_ path: String) -> String {
        guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return "" }
        let r = sentinelProcess(
            launchPath: "/usr/bin/mdls",
            arguments: ["-name", "kMDItemWhereFroms", "-raw", path],
            timeout: 5
        )
        guard r.exitCode == 0 else { return "" }
        let out = r.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if out.isEmpty || out == "(null)" { return "" }
        return out.replacingOccurrences(of: "\n", with: ", ")
    }

    private static func sha256File(_ path: String) -> String {
        let r = sentinelProcess(
            launchPath: "/usr/bin/shasum",
            arguments: ["-a", "256", path],
            timeout: 10
        )
        guard r.exitCode == 0 else { return "" }
        return r.stdout.split(separator: " ").first.map(String.init) ?? ""
    }

    // MARK: - Persistence of findings

    static func appendFindings(_ findings: [Finding]) {
        let encoder = JSONEncoder()
        var block = ""
        for var f in findings {
            f.ts = f.ts == 0 ? Date().timeIntervalSince1970 : f.ts
            guard let data = try? encoder.encode(f),
                  let line = String(data: data, encoding: .utf8) else { continue }
            block += line + "\n"
        }
        guard !block.isEmpty else { return }
        if let handle = FileHandle(forWritingAtPath: findingsPath) {
            handle.seekToEndOfFile()
            handle.write(block.data(using: .utf8)!)
            try? handle.close()
        } else {
            try? block.write(toFile: findingsPath, atomically: true, encoding: .utf8)
        }
    }

    private static func loadSnapshot() -> Snapshot? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: snapshotPath)) else { return nil }
        return try? JSONDecoder().decode(Snapshot.self, from: data)
    }

    private static func saveSnapshot(_ snapshot: Snapshot) {
        guard let data = try? JSONEncoder().encode(snapshot) else { return }
        try? data.write(to: URL(fileURLWithPath: snapshotPath), options: .atomic)
    }

    static func baselineInfo() -> (takenAt: TimeInterval, items: Int, listeners: Int) {
        guard let s = loadSnapshot() else { return (0, 0, 0) }
        return (s.takenAt, s.persistence.count, s.listeners.count)
    }
}

// MARK: - Subprocess

private func sentinelProcess(
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
