// BadAppleAquaHelper.swift — native Swift Aqua session helper for the Bad Apple menu bar.
//
// Replaces badapple_aqua_helper.py. Runs inside the Bad Apple.app process, serves
// /var/run/badapple/aqua_helper.sock, and performs user-session actions such as
// Shortcuts, screen capture, UI automation, and AppleScript.

import EventKit
import Foundation
import CommonCrypto
import Dispatch
import Darwin

private let defaultSocketPath = "/var/run/badapple/aqua_helper.sock"
private let nonceMaxSize = 1024
private let nonceMaxAgeMs: Int64 = 120_000

// MARK: - Public lifecycle API

private var _aquaHelper: AquaHelper?

func startAquaHelper() {
    if _aquaHelper == nil {
        _aquaHelper = AquaHelper()
    }
    _aquaHelper?.start()
}

func stopAquaHelper() {
    _aquaHelper?.stop()
    _aquaHelper = nil
}

// MARK: - SLICKS helpers

private func socketPath() -> String {
    return ProcessInfo.processInfo.environment["BADAPPLE_AQUA_SOCKET"] ?? defaultSocketPath
}

private func loadSlicksSecret() -> Data? {
    let env = ProcessInfo.processInfo.environment
    let raw: String?
    if let secretEnv = env["BADAPPLE_SLICKS_SECRET"], !secretEnv.isEmpty {
        raw = secretEnv
    } else {
        let keyPath = env["BADAPPLE_SLICKS_KEY_PATH"] ?? "/var/lib/bad_apple/slicks.key"
        raw = try? String(contentsOfFile: keyPath, encoding: .utf8)
    }
    guard let raw = raw else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.allSatisfy({ $0.isHexDigit }) && trimmed.count >= 32 {
        return hexToData(trimmed)
    }
    return trimmed.data(using: .utf8)
}

private func hexToData(_ hex: String) -> Data? {
    let lower = hex.lowercased()
    guard lower.count % 2 == 0 else { return nil }
    var data = Data(capacity: lower.count / 2)
    let chars = Array(lower)
    for i in stride(from: 0, to: chars.count, by: 2) {
        guard let high = chars[i].hexDigitValue,
              let low = chars[i + 1].hexDigitValue else { return nil }
        data.append(UInt8(high * 16 + low))
    }
    return data
}

private func hmacSHA256Hex(_ key: Data, _ message: String) -> String {
    var mac = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
    let msgData = Data(message.utf8)
    key.withUnsafeBytes { keyPtr in
        msgData.withUnsafeBytes { msgPtr in
            CCHmac(
                CCHmacAlgorithm(kCCHmacAlgSHA256),
                keyPtr.baseAddress,
                key.count,
                msgPtr.baseAddress,
                msgData.count,
                &mac
            )
        }
    }
    return mac.map { String(format: "%02x", $0) }.joined()
}

private func constantTimeCompare(_ a: Data, _ b: Data) -> Bool {
    guard a.count == b.count else { return false }
    var diff: UInt8 = 0
    for i in 0..<a.count {
        diff |= a[i] ^ b[i]
    }
    return diff == 0
}

/// Canonical JSON for SLICKS v1 proof material: sorted keys, no extra spaces,
/// and forward slashes are not escaped. This must match the client serializer
/// in BadAppleTools.swift.
private func canonicalJSONString(_ value: Any) -> String? {
    guard JSONSerialization.isValidJSONObject(value) else { return nil }
    guard let data = try? JSONSerialization.data(
        withJSONObject: value,
        options: [.sortedKeys, .withoutEscapingSlashes]
    ) else { return nil }
    return String(data: data, encoding: .utf8)
}

// MARK: - AquaHelper server

final class AquaHelper {
    private let lock = NSLock()
    private var isRunning = false
    private var listener: Int32 = -1

    private var seenNonces = Set<String>()
    private let nonceLock = NSLock()

    func start() {
        lock.lock()
        if isRunning {
            lock.unlock()
            return
        }
        isRunning = true
        lock.unlock()

        let path = socketPath()
        removeStaleSocket(path)
        createParentDir(path)

        let fd = createListenSocket(path: path)
        guard fd >= 0 else {
            print("[AquaHelper] could not create listen socket for \(path)")
            lock.lock()
            isRunning = false
            lock.unlock()
            return
        }

        _ = chmod(path, 0o600)
        print("[AquaHelper] listening on \(path) (0o600)")

        lock.lock()
        listener = fd
        lock.unlock()

        DispatchQueue.global(qos: .default).async { [weak self] in
            self?.acceptLoop()
        }
    }

    func stop() {
        lock.lock()
        isRunning = false
        let fd = listener
        listener = -1
        lock.unlock()

        if fd >= 0 {
            close(fd)
        }
        removeStaleSocket(socketPath())
    }

    private func acceptLoop() {
        while true {
            lock.lock()
            let running = isRunning
            let fd = listener
            lock.unlock()
            guard running, fd >= 0 else { break }

            var addr = sockaddr_un()
            memset(&addr, 0, MemoryLayout<sockaddr_un>.size)
            var len = socklen_t(MemoryLayout<sockaddr_un>.size)
            let client = withUnsafeMutablePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                    accept(fd, sockaddrPtr, &len)
                }
            }
            if client < 0 {
                let e = errno
                if e == EINTR { continue }
                if !isRunning { break }
                print("[AquaHelper] accept failed: \(e)")
                continue
            }

            DispatchQueue.global(qos: .default).async { [weak self] in
                self?.handleClient(client)
            }
        }
    }

    private func handleClient(_ clientFd: Int32) {
        defer { close(clientFd) }

        var on: Int32 = 1
        setsockopt(clientFd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))

        var buffer = Data()
        while let line = readLine(from: clientFd, buffer: &buffer) {
            var resp: [String: Any]
            guard let req = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else {
                resp = ["ok": false, "error": "invalid JSON"]
                writeJSON(clientFd, resp)
                continue
            }

            guard let secret = loadSlicksSecret(), !secret.isEmpty else {
                resp = ["ok": false, "error": "aqua helper has no SLICKS secret configured"]
                writeJSON(clientFd, resp)
                continue
            }

            if !verifyProof(req, secret: secret) {
                resp = ["ok": false, "error": "unauthenticated: invalid or missing proof"]
                writeJSON(clientFd, resp)
                continue
            }

            var stripped = req
            stripped.removeValue(forKey: "proof")
            resp = handleRequest(stripped)
            writeJSON(clientFd, resp)
        }
    }

    private func readLine(from fd: Int32, buffer: inout Data) -> Data? {
        while true {
            if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                let next = buffer.index(after: newlineIndex)
                let line = Data(buffer[buffer.startIndex..<newlineIndex])
                buffer.removeSubrange(buffer.startIndex..<next)
                return line
            }

            var chunk = [UInt8](repeating: 0, count: 4096)
            let n = read(fd, &chunk, 4096)
            if n > 0 {
                buffer.append(chunk, count: n)
            } else if n == 0 {
                return nil
            } else {
                let e = errno
                if e == EINTR { continue }
                return nil
            }
        }
    }

    private func writeJSON(_ fd: Int32, _ value: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: []),
              let text = String(data: data, encoding: .utf8),
              let line = (text + "\n").data(using: .utf8) else {
            return
        }
        _ = writeAll(fd, data: line)
    }

    private func writeAll(_ fd: Int32, data: Data) -> Bool {
        var total = 0
        return data.withUnsafeBytes { raw -> Bool in
            guard let base = raw.baseAddress else { return false }
            while total < data.count {
                let n = write(fd, base.advanced(by: total), data.count - total)
                if n < 0 {
                    let e = errno
                    if e == EINTR { continue }
                    return false
                }
                if n == 0 { return false }
                total += n
            }
            return true
        }
    }

    private func verifyProof(_ req: [String: Any], secret: Data) -> Bool {
        guard let proof = req["proof"] as? String else { return false }
        guard let timestampValue = req["timestamp_ms"] else { return false }
        guard let nonce = req["nonce"] as? String, !nonce.isEmpty else { return false }

        let timestamp: Double
        if let t = timestampValue as? Double {
            timestamp = t
        } else if let t = timestampValue as? Int {
            timestamp = Double(t)
        } else if let t = timestampValue as? Int64 {
            timestamp = Double(t)
        } else {
            return false
        }

        let now = Date().timeIntervalSince1970 * 1000
        guard abs(now - timestamp) <= Double(nonceMaxAgeMs) else { return false }

        nonceLock.lock()
        if seenNonces.contains(nonce) {
            nonceLock.unlock()
            return false
        }
        if seenNonces.count >= nonceMaxSize {
            seenNonces.removeAll()
        }
        seenNonces.insert(nonce)
        nonceLock.unlock()

        var body = req
        body.removeValue(forKey: "proof")

        guard let material = canonicalJSONString(body),
              let macData = hexToData(hmacSHA256Hex(secret, material)),
              let proofData = hexToData(proof.lowercased()),
              macData.count == proofData.count else {
            return false
        }

        return constantTimeCompare(macData, proofData)
    }

    private func handleRequest(_ req: [String: Any]) -> [String: Any] {
        guard let command = req["command"] as? String else {
            return ["ok": false, "error": "missing command"]
        }

        switch command {
        case "list_shortcuts":
            let timeout = TimeInterval(req["timeout"] as? Double ?? Double(req["timeout"] as? Int ?? 15))
            return listShortcuts(timeout: timeout)
        case "run_shortcut":
            let name = req["name"] as? String ?? ""
            let input = req["input"] as? String ?? ""
            let timeout = TimeInterval(req["timeout"] as? Double ?? Double(req["timeout"] as? Int ?? 60))
            return runShortcut(name: name, input: input, timeout: timeout)
        case "capture_screen":
            let path = req["path"] as? String ?? ""
            let region = req["region"] as? String ?? ""
            return captureScreen(path: path, region: region)
        case "ui_info":
            return uiInfo()
        case "ui_click":
            return uiClick(target: req["target"] as? String ?? "", role: req["role"] as? String ?? "")
        case "ui_type":
            return uiType(target: req["target"] as? String ?? "", text: req["text"] as? String ?? "")
        case "ui_focus":
            return uiFocus(target: req["target"] as? String ?? "")
        case "calendar_events":
            let days = req["days_ahead"] as? Int ?? 7
            return calendarEvents(daysAhead: days)
        case "calendar_create":
            return calendarCreate(
                title: req["title"] as? String ?? "",
                start: req["start"] as? String ?? "",
                end: req["end"] as? String ?? "",
                notes: req["notes"] as? String ?? ""
            )
        case "reminders_list":
            return remindersList()
        case "reminder_create":
            return reminderCreate(
                title: req["title"] as? String ?? "",
                due: req["due"] as? String ?? "",
                notes: req["notes"] as? String ?? ""
            )
        case "reminder_complete":
            return reminderComplete(title: req["title"] as? String ?? "")
        case "mail_read":
            let limit = req["limit"] as? Int ?? 10
            return mailRead(limit: limit)
        case "mail_draft":
            return mailWrite(
                to: req["to"] as? String ?? "",
                subject: req["subject"] as? String ?? "",
                body: req["body"] as? String ?? "",
                send: false
            )
        case "mail_send":
            return mailWrite(
                to: req["to"] as? String ?? "",
                subject: req["subject"] as? String ?? "",
                body: req["body"] as? String ?? "",
                send: true
            )
        case "message_send":
            return messageSend(
                to: req["to"] as? String ?? "",
                text: req["text"] as? String ?? ""
            )
        default:
            return ["ok": false, "error": "unknown command '\(command)'"]
        }
    }
}

// MARK: - Socket setup

private func removeStaleSocket(_ path: String) {
    var st = stat()
    if stat(path, &st) == 0 {
        unlink(path)
    }
}

private func createParentDir(_ path: String) {
    let parent = (path as NSString).deletingLastPathComponent
    let fm = FileManager.default
    if !fm.fileExists(atPath: parent) {
        try? fm.createDirectory(
            atPath: parent,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
    }
    _ = chmod(parent, 0o700)
}

private func createListenSocket(path: String) -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return -1 }

    var on: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))

    var addr = sockaddr_un()
    memset(&addr, 0, MemoryLayout<sockaddr_un>.size)
    addr.sun_family = sa_family_t(AF_UNIX)

    let pathBytes = Array(path.utf8)
    let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard pathBytes.count < maxPath else {
        close(fd)
        return -1
    }

    _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
        pathBytes.withUnsafeBufferPointer { src in
            memcpy(UnsafeMutableRawPointer(dst), src.baseAddress!, pathBytes.count)
        }
    }
    addr.sun_len = UInt8(2 + pathBytes.count + 1)
    let addrLen = socklen_t(addr.sun_len)

    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            bind(fd, sockaddrPtr, addrLen)
        }
    }
    guard bindResult == 0 else {
        close(fd)
        return -1
    }

    guard listen(fd, 128) == 0 else {
        close(fd)
        return -1
    }

    return fd
}

// MARK: - Command implementations

private func listShortcuts(timeout: TimeInterval) -> [String: Any] {
    let result = runProcess(launchPath: "/usr/bin/shortcuts", arguments: ["list"], timeout: timeout)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "shortcuts list failed" : error]
    }
    let names = result.stdout
        .components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
    return ["ok": true, "shortcuts": Array(names.prefix(100))]
}

private func runShortcut(name: String, input: String, timeout: TimeInterval) -> [String: Any] {
    let shortcutName = name.trimmingCharacters(in: .whitespacesAndNewlines)
    if shortcutName.isEmpty || shortcutName.count > 255 || shortcutName.contains("\0") {
        return ["ok": false, "error": "invalid shortcut name"]
    }

    var arguments = ["run", shortcutName]
    var standardInput: Data?
    if !input.isEmpty {
        guard input.utf8.count <= 100_000 else {
            return ["ok": false, "error": "shortcut input is too large"]
        }
        arguments.append(contentsOf: ["-i", "-"])
        standardInput = input.data(using: .utf8)
    }

    let result = runProcess(
        launchPath: "/usr/bin/shortcuts",
        arguments: arguments,
        timeout: timeout,
        standardInput: standardInput
    )
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "shortcut failed" : error]
    }
    let output = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    return ["ok": true, "output": output]
}

private func captureScreen(path: String, region: String) -> [String: Any] {
    var outPath = path
    if outPath.isEmpty {
        outPath = "/tmp/badapple_\(UUID().uuidString).png"
    }

    let outURL = URL(fileURLWithPath: outPath)
    try? FileManager.default.createDirectory(
        at: outURL.deletingLastPathComponent(),
        withIntermediateDirectories: true,
        attributes: nil
    )

    if region.isEmpty, let helper = Bundle.main.url(forAuxiliaryExecutable: "BadAppleScreenCapture") {
        let result = runProcess(launchPath: helper.path, arguments: ["--output", outPath], timeout: 30)
        if result.exitCode == 0,
           FileManager.default.fileExists(atPath: outPath),
           fileSize(outPath) > 0 {
            return ["ok": true, "path": outPath]
        }
    }

    var arguments = ["-x"]
    if region.isEmpty {
        arguments.append("-S")
    } else {
        arguments.append(contentsOf: ["-R", region])
    }
    arguments.append(outPath)

    let result = runProcess(launchPath: "/usr/bin/screencapture", arguments: arguments, timeout: 30)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "screencapture failed" : error]
    }

    if !FileManager.default.fileExists(atPath: outPath) || fileSize(outPath) == 0 {
        return ["ok": false, "error": "screencapture produced no image"]
    }
    return ["ok": true, "path": outPath]
}

private func fileSize(_ path: String) -> Int {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
          let size = attrs[.size] as? Int else { return 0 }
    return size
}

// MARK: - UI automation

private func uiInfo() -> [String: Any] {
    let helperResult = runUIHelper(action: "info", arguments: [:])
    if helperResult.ok, let json = helperResult.json {
        if let error = json["error"] as? String, !error.isEmpty {
            return ["ok": false, "error": error]
        }
        let app = json["app"] as? String ?? ""
        let window = json["window"] as? String ?? ""
        var elements: [String] = []
        if let elArray = json["elements"] as? [[String: Any]] {
            flattenElements(elArray, into: &elements, limit: 100)
        }
        return ["ok": true, "app": app, "window": window, "elements": elements]
    }

    let script = uiInfoAppleScript()
    let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", script], timeout: 30)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "ui_info failed" : error]
    }

    let text = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    let parts = text.components(separatedBy: "|")
    guard parts.count >= 3 else {
        return ["ok": false, "error": "unexpected ui_info output"]
    }
    let elements = parts[2]
        .components(separatedBy: ", ")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    return ["ok": true, "app": parts[0], "window": parts[1], "elements": elements]
}

private func uiClick(target: String, role: String) -> [String: Any] {
    if target.isEmpty && role.isEmpty {
        return ["ok": false, "error": "target name or role is required"]
    }

    let helperResult = runUIHelper(action: "click", arguments: ["target": target, "role": role])
    if helperResult.ok, let json = helperResult.json, (json["ok"] as? Bool) == true {
        return ["ok": true, "result": json["result"] as? String ?? "clicked"]
    }

    let escapedTarget = appleScriptEscape(target)
    let condition: String
    if !target.isEmpty && !role.isEmpty {
        condition = "name of e is \"\(escapedTarget)\" and role of e is \"\(appleScriptEscape(role))\""
    } else if !target.isEmpty {
        condition = "name of e is \"\(escapedTarget)\""
    } else {
        condition = "role of e is \"\(appleScriptEscape(role))\""
    }

    let script = """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if \(condition) then
                    click e
                    return "clicked \(escapedTarget)"
                end if
            end try
        end repeat
        return "not found"
    end tell
    """

    let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", script], timeout: 30)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "ui_click failed" : error]
    }
    let text = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    if text == "not found" {
        return ["ok": false, "error": "element '\(target)' not found"]
    }
    return ["ok": true, "result": text]
}

private func uiType(target: String, text: String) -> [String: Any] {
    if target.isEmpty {
        return ["ok": false, "error": "target name and text are required"]
    }

    let helperResult = runUIHelper(action: "type", arguments: ["target": target, "text": text])
    if helperResult.ok, let json = helperResult.json, (json["ok"] as? Bool) == true {
        return ["ok": true, "result": json["result"] as? String ?? "typed"]
    }

    let escapedTarget = appleScriptEscape(target)
    let escapedText = appleScriptEscape(text, escapeNewlines: true)
    let script = """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "\(escapedTarget)" then
                    set value of e to "\(escapedText)"
                    return "typed into \(escapedTarget)"
                end if
            end try
        end repeat
        return "not found"
    end tell
    """

    let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", script], timeout: 30)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "ui_type failed" : error]
    }
    let out = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    if out == "not found" {
        return ["ok": false, "error": "text field '\(target)' not found"]
    }
    return ["ok": true, "result": out]
}

private func uiFocus(target: String) -> [String: Any] {
    if target.isEmpty {
        return ["ok": false, "error": "target name is required"]
    }

    let helperResult = runUIHelper(action: "focus", arguments: ["target": target])
    if helperResult.ok, let json = helperResult.json, (json["ok"] as? Bool) == true {
        return ["ok": true, "result": json["result"] as? String ?? "focused"]
    }

    let escapedTarget = appleScriptEscape(target)
    let script = """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set w to front window of p
        repeat with e in (entire contents of w)
            try
                if name of e is "\(escapedTarget)" then
                    set focused of e to true
                    return "focused \(escapedTarget)"
                end if
            end try
        end repeat
        return "not found"
    end tell
    """

    let result = runProcess(launchPath: "/usr/bin/osascript", arguments: ["-e", script], timeout: 30)
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "ui_focus failed" : error]
    }
    let out = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    if out == "not found" {
        return ["ok": false, "error": "element '\(target)' not found"]
    }
    return ["ok": true, "result": out]
}

private func runUIHelper(action: String, arguments: [String: String]) -> (ok: Bool, output: String, json: [String: Any]?) {
    guard let helper = Bundle.main.url(forAuxiliaryExecutable: "BadAppleUI") else {
        return (false, "BadAppleUI helper not found", nil)
    }
    var args = ["--action", action]
    for (key, value) in arguments {
        if !value.isEmpty {
            args.append(contentsOf: ["--\(key)", value])
        }
    }
    let result = runProcess(launchPath: helper.path, arguments: args, timeout: 30)
    if result.exitCode != 0 {
        let error = result.stderr.isEmpty ? result.stdout : result.stderr
        return (false, error.trimmingCharacters(in: .whitespacesAndNewlines), nil)
    }
    let text = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let data = text.data(using: .utf8),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return (false, "invalid JSON from BadAppleUI", nil)
    }
    return (true, text, json)
}

private func flattenElements(_ elements: [[String: Any]], into: inout [String], limit: Int) {
    for el in elements {
        guard into.count < limit else { break }
        let role = el["role"] as? String ?? ""
        let title = el["title"] as? String ?? ""
        let line = title.isEmpty ? role : "\(role): \(title)"
        into.append(line)
        if let children = el["children"] as? [[String: Any]] {
            flattenElements(children, into: &into, limit: limit)
        }
    }
}

private func uiInfoAppleScript() -> String {
    return """
    tell application "System Events"
        set p to first application process whose frontmost is true
        set appName to name of p
        set w to front window of p
        set winName to name of w
        set elements to {}
        set counter to 0
        repeat with e in (entire contents of w)
            try
                if counter > 100 then exit repeat
                if exists e then
                    set n to name of e
                    set r to role of e
                    if n is not missing value then
                        set end of elements to (r & ": " & n)
                        set counter to counter + 1
                    end if
                end if
            end try
        end repeat
        return appName & "|" & winName & "|" & (elements as string)
    end tell
    """
}

private func appleScriptEscape(_ s: String, escapeNewlines: Bool = false) -> String {
    var r = s.replacingOccurrences(of: "\\", with: "\\\\")
    r = r.replacingOccurrences(of: "\"", with: "\\\"")
    if escapeNewlines {
        r = r.replacingOccurrences(of: "\n", with: "\\n")
    }
    return r
}

// MARK: - Process runner

private func runProcess(
    launchPath: String,
    arguments: [String],
    timeout: TimeInterval,
    standardInput: Data? = nil
) -> (stdout: String, stderr: String, exitCode: Int) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: launchPath)
    process.arguments = arguments

    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    let stdinPipe = standardInput == nil ? nil : Pipe()
    process.standardInput = stdinPipe

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
        stdinPipe?.fileHandleForWriting.closeFile()
        return ("", "Error: \(error.localizedDescription)", -1)
    }

    if let standardInput, let stdinPipe {
        stdinPipe.fileHandleForWriting.write(standardInput)
        stdinPipe.fileHandleForWriting.closeFile()
    }

    let deadline = Date().addingTimeInterval(timeout)
    while process.isRunning && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.05)
    }

    if process.isRunning {
        process.terminate()
        group.wait()
        return ("", "Error: command timed out after \(Int(timeout))s", -1)
    }

    group.wait()

    let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""
    return (stdout, stderr, Int(process.terminationStatus))
}

// MARK: - Calendar, reminders, and comms
//
// These run inside the app process so EventKit and Automation consent
// resolve against the app's TCC identity — the daemon reaches them through
// the authenticated aqua socket instead of holding grants itself. Argument
// strings are passed to osascript via argv, never interpolated into source.

private func requestEventKitAccess(reminders: Bool) -> (store: EKEventStore?, error: String?) {
    let store = EKEventStore()
    let sem = DispatchSemaphore(value: 0)
    var granted = false
    var accessError: Error?
    if reminders {
        store.requestFullAccessToReminders { ok, err in
            granted = ok
            accessError = err
            sem.signal()
        }
    } else {
        store.requestFullAccessToEvents { ok, err in
            granted = ok
            accessError = err
            sem.signal()
        }
    }
    if sem.wait(timeout: .now() + 60) == .timedOut {
        return (nil, "timed out waiting for access")
    }
    if let accessError { return (nil, accessError.localizedDescription) }
    if !granted {
        let kind = reminders ? "Reminders" : "Calendar"
        return (nil, "\(kind) access not granted — approve it in System Settings > Privacy & Security")
    }
    return (store, nil)
}

private let dayStamp: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd HH:mm"
    return f
}()

private func parseDateTime(_ text: String) -> Date? {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return nil }
    let iso = ISO8601DateFormatter()
    if let d = iso.date(from: trimmed) { return d }
    let formats = ["yyyy-MM-dd HH:mm", "yyyy-MM-dd'T'HH:mm", "yyyy-MM-dd"]
    for fmt in formats {
        let f = DateFormatter()
        f.dateFormat = fmt
        if let d = f.date(from: trimmed) { return d }
    }
    return BadAppleHumanLayer.parseDueDate(trimmed)
}

private func calendarEvents(daysAhead: Int) -> [String: Any] {
    let (store, error) = requestEventKitAccess(reminders: false)
    guard let store else { return ["ok": false, "error": error ?? "calendar unavailable"] }
    let days = min(max(daysAhead, 1), 90)
    let start = Date()
    guard let end = Calendar.current.date(byAdding: .day, value: days, to: start) else {
        return ["ok": false, "error": "bad date range"]
    }
    let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
    let events = store.events(matching: predicate).prefix(50)
    let rows = events.map { ev -> String in
        var line = "\(dayStamp.string(from: ev.startDate)) | \(ev.title ?? "(untitled)") | \(ev.calendar.title)"
        if let loc = ev.location, !loc.isEmpty { line += " | \(loc)" }
        return line
    }
    return ["ok": true, "events": rows]
}

private func calendarCreate(title: String, start: String, end: String, notes: String) -> [String: Any] {
    let cleanTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !cleanTitle.isEmpty, cleanTitle.count <= 500 else {
        return ["ok": false, "error": "title required (max 500 chars)"]
    }
    guard let startDate = parseDateTime(start) else {
        return ["ok": false, "error": "could not parse start '\(start)'"]
    }
    let endDate = parseDateTime(end) ?? startDate.addingTimeInterval(3600)
    let (store, error) = requestEventKitAccess(reminders: false)
    guard let store else { return ["ok": false, "error": error ?? "calendar unavailable"] }
    let event = EKEvent(eventStore: store)
    event.title = cleanTitle
    event.startDate = startDate
    event.endDate = max(endDate, startDate.addingTimeInterval(60))
    event.calendar = store.defaultCalendarForNewEvents
    if !notes.isEmpty { event.notes = String(notes.prefix(2000)) }
    do {
        try store.save(event, span: .thisEvent)
        return ["ok": true, "result": "Created '\(cleanTitle)' on \(dayStamp.string(from: startDate))"]
    } catch {
        return ["ok": false, "error": error.localizedDescription]
    }
}

private func reminderDate(_ text: String) -> DateComponents? {
    guard let date = parseDateTime(text) else { return nil }
    return Calendar.current.dateComponents(
        [.year, .month, .day, .hour, .minute], from: date
    )
}

private func remindersList() -> [String: Any] {
    let (store, error) = requestEventKitAccess(reminders: true)
    guard let store else { return ["ok": false, "error": error ?? "reminders unavailable"] }
    let predicate = store.predicateForReminders(in: nil)
    let sem = DispatchSemaphore(value: 0)
    var found: [EKReminder] = []
    store.fetchReminders(matching: predicate) { reminders in
        found = reminders ?? []
        sem.signal()
    }
    if sem.wait(timeout: .now() + 30) == .timedOut {
        return ["ok": false, "error": "timed out reading reminders"]
    }
    let open = found.filter { !$0.isCompleted }.prefix(50)
    let rows = open.map { r -> String in
        var line = r.title ?? "(untitled)"
        if let due = r.dueDateComponents?.date {
            line += " — due \(dayStamp.string(from: due))"
        }
        return line
    }
    return ["ok": true, "reminders": rows]
}

private func reminderCreate(title: String, due: String, notes: String) -> [String: Any] {
    let cleanTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !cleanTitle.isEmpty, cleanTitle.count <= 500 else {
        return ["ok": false, "error": "title required (max 500 chars)"]
    }
    let (store, error) = requestEventKitAccess(reminders: true)
    guard let store else { return ["ok": false, "error": error ?? "reminders unavailable"] }
    let reminder = EKReminder(eventStore: store)
    reminder.title = cleanTitle
    reminder.calendar = store.defaultCalendarForNewReminders()
    if let dueComponents = reminderDate(due) {
        reminder.dueDateComponents = dueComponents
        reminder.addAlarm(EKAlarm(absoluteDate: dueComponents.date ?? Date()))
    }
    if !notes.isEmpty { reminder.notes = String(notes.prefix(2000)) }
    do {
        try store.save(reminder, commit: true)
        var reply = "Reminder created: \(cleanTitle)"
        if let due = reminder.dueDateComponents?.date {
            reply += " — due \(dayStamp.string(from: due))"
        }
        return ["ok": true, "result": reply]
    } catch {
        return ["ok": false, "error": error.localizedDescription]
    }
}

private func reminderComplete(title: String) -> [String: Any] {
    let needle = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !needle.isEmpty else { return ["ok": false, "error": "title required"] }
    let (store, error) = requestEventKitAccess(reminders: true)
    guard let store else { return ["ok": false, "error": error ?? "reminders unavailable"] }
    let predicate = store.predicateForReminders(in: nil)
    let sem = DispatchSemaphore(value: 0)
    var found: [EKReminder] = []
    store.fetchReminders(matching: predicate) { reminders in
        found = reminders ?? []
        sem.signal()
    }
    if sem.wait(timeout: .now() + 30) == .timedOut {
        return ["ok": false, "error": "timed out reading reminders"]
    }
    guard let match = found.first(where: {
        !$0.isCompleted && ($0.title ?? "").localizedCaseInsensitiveContains(needle)
    }) else {
        return ["ok": false, "error": "no open reminder matching '\(needle)'"]
    }
    match.isCompleted = true
    do {
        try store.save(match, commit: true)
        return ["ok": true, "result": "Completed: \(match.title ?? needle)"]
    } catch {
        return ["ok": false, "error": error.localizedDescription]
    }
}

private func osascriptArgv(_ script: String, args: [String], timeout: TimeInterval) -> [String: Any] {
    let result = runProcess(
        launchPath: "/usr/bin/osascript",
        arguments: ["-e", script] + args,
        timeout: timeout
    )
    if result.exitCode != 0 {
        let error = (result.stderr.isEmpty ? result.stdout : result.stderr)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return ["ok": false, "error": error.isEmpty ? "osascript failed" : error]
    }
    return ["ok": true, "output": result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)]
}

private func mailRead(limit: Int) -> [String: Any] {
    let n = min(max(limit, 1), 50)
    let script = """
    on run argv
      set maxCount to (item 1 of argv) as integer
      set out to ""
      tell application "Mail"
        set msgs to messages of inbox
        set total to count of msgs
        set shown to maxCount
        if total < shown then set shown to total
        repeat with i from total to (total - shown + 1) by -1
          set m to item i of msgs
          try
            set out to out & (date received of m as string) & " | " & (sender of m) & " | " & (subject of m) & linefeed
          end try
        end repeat
      end tell
      return out
    end run
    """
    let res = osascriptArgv(script, args: [String(n)], timeout: 30)
    if (res["ok"] as? Bool) == true {
        let out = (res["output"] as? String) ?? ""
        return ["ok": true, "output": out.isEmpty ? "inbox is empty" : out]
    }
    return res
}

private func mailWrite(to: String, subject: String, body: String, send: Bool) -> [String: Any] {
    let addr = to.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !addr.isEmpty, addr.count <= 320, !addr.contains("\n") else {
        return ["ok": false, "error": "a valid 'to' address is required"]
    }
    guard subject.count <= 500, body.count <= 50_000 else {
        return ["ok": false, "error": "subject or body too large"]
    }
    let sendLine = send ? "send msg" : ""
    let script = """
    on run argv
      set toAddr to item 1 of argv
      set subj to item 2 of argv
      set bodyText to item 3 of argv
      tell application "Mail"
        set msg to make new outgoing message with properties {subject:subj, content:bodyText, visible:false}
        tell msg to make new to recipient at end of to recipients with properties {address:toAddr}
        \(sendLine)
      end tell
      return "ok"
    end run
    """
    let res = osascriptArgv(script, args: [addr, subject, body], timeout: 30)
    if (res["ok"] as? Bool) == true {
        return ["ok": true, "result": send ? "Mail sent to \(addr)" : "Draft created for \(addr)"]
    }
    return res
}

private func messageSend(to: String, text: String) -> [String: Any] {
    let target = to.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !target.isEmpty, target.count <= 320, !target.contains("\n") else {
        return ["ok": false, "error": "a valid recipient is required"]
    }
    guard !text.isEmpty, text.count <= 10_000 else {
        return ["ok": false, "error": "message text required (max 10000 chars)"]
    }
    let script = """
    on run argv
      set target to item 1 of argv
      set msgText to item 2 of argv
      tell application "Messages"
        set svc to 1st service whose service type = iMessage
        send msgText to buddy target of svc
      end tell
      return "sent"
    end run
    """
    let res = osascriptArgv(script, args: [target, text], timeout: 30)
    if (res["ok"] as? Bool) == true {
        return ["ok": true, "result": "Message sent to \(target)"]
    }
    return res
}
