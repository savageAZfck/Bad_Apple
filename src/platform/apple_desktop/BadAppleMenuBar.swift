import AppKit
import ApplicationServices
import AudioToolbox
import AVFoundation
import BadAppleBridge
import Carbon
import Darwin
import Foundation
import ServiceManagement
import Speech

private let badAppleVoiceLogPath = "/tmp/badapple_voice_debug.log"

/// Append a line to the voice debug log using a raw POSIX `open()` with
/// `O_NOFOLLOW` and `O_EXCL` on first creation. `/tmp` is world-writable,
/// so anything else running as this user could pre-create this path as a
/// regular file with mode 0666; O_NOFOLLOW blocks symlinks, and we fchmod
/// to 0600 after open to ensure the file is private regardless of how it
/// was created.
private func badAppleVoiceLog(_ message: String) {
    let stamp = ISO8601DateFormatter().string(from: Date())
    let line = "\(stamp) \(message)\n"
    let fd = open(badAppleVoiceLogPath, O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0o600)
    if fd >= 0 {
        // Force 0600 in case the file was pre-created with wider permissions.
        fchmod(fd, 0o600)
        line.withCString { cstr in
            _ = Darwin.write(fd, cstr, strlen(cstr))
        }
        close(fd)
    } else {
        NSLog("badAppleVoiceLog failed: open() errno %d", errno)
    }
    NSLog("[BadAppleVoice] %@", message)
}

// MARK: - C function signatures from bad_apple_core.h

private typealias BadAppleInitFn = @convention(c) (UnsafePointer<CChar>?) -> OpaquePointer?
private typealias BadAppleFreeFn = @convention(c) (OpaquePointer?) -> Void
private typealias BadAppleGetActivePursuitsFn = @convention(c) (OpaquePointer?) -> UnsafeMutablePointer<CChar>?
private typealias BadApplePushPursuitFn = @convention(c) (OpaquePointer?, UnsafePointer<CChar>?) -> Bool
private typealias BadAppleGetAppleLatencyUsFn = @convention(c) () -> UInt64
private typealias BadAppleGenerateTextFn = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutablePointer<CChar>?
private typealias BadAppleFreeStringFn = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
private typealias InitBadAppleBridgeFn = @convention(c) () -> Void

// MARK: - Dynamic loader for libbad_apple.dylib

final class BadAppleFFI {
    static let shared = BadAppleFFI()

    private var handle: UnsafeMutableRawPointer?
    private(set) var context: OpaquePointer?

    private var bad_apple_init: BadAppleInitFn?
    private var bad_apple_free: BadAppleFreeFn?
    private var bad_apple_get_active_pursuits: BadAppleGetActivePursuitsFn?
    private var bad_apple_push_pursuit: BadApplePushPursuitFn?
    private var bad_apple_get_apple_latency_us: BadAppleGetAppleLatencyUsFn?
    private var bad_apple_generate_text: BadAppleGenerateTextFn?
    private var bad_apple_free_string: BadAppleFreeStringFn?

    private var lastError: String?

    var isLoaded: Bool { handle != nil && context != nil }

    func load() {
        let bundleFrameworks = Bundle.main.bundlePath + "/Contents/Frameworks"

        // Load the Rust core library first so its symbols are available when the
        // bridge is initialized (the bridge references register_apple_intelligence_oracle
        // through -undefined dynamic_lookup).
        // Security: only load from the signed app bundle's Frameworks
        // directory or well-known system paths. Never load from CWD, parent
        // directories, or relative paths that could be hijacked by a
        // malicious dylib dropped next to the .app bundle.
        let searchPaths = [
            bundleFrameworks + "/libbad_apple.dylib",
            "/usr/local/lib/libbad_apple.dylib",
        ]

        for path in searchPaths {
            if let h = dlopen(path, RTLD_LAZY) {
                handle = h
                break
            }
        }

        guard let h = handle else {
            lastError = "libbad_apple.dylib not found in any search path"
            return
        }

        // Validate every dlsym result before unsafeBitCast to avoid a bogus
        // function pointer that would crash on call.
        guard let s_init = dlsym(h, "bad_apple_init") else {
            lastError = "dlsym: bad_apple_init not found"
            return
        }
        guard let s_free = dlsym(h, "bad_apple_free") else {
            lastError = "dlsym: bad_apple_free not found"
            return
        }
        guard let s_pursuits = dlsym(h, "bad_apple_get_active_pursuits") else {
            lastError = "dlsym: bad_apple_get_active_pursuits not found"
            return
        }
        guard let s_push = dlsym(h, "bad_apple_push_pursuit") else {
            lastError = "dlsym: bad_apple_push_pursuit not found"
            return
        }
        guard let s_latency = dlsym(h, "bad_apple_get_apple_latency_us") else {
            lastError = "dlsym: bad_apple_get_apple_latency_us not found"
            return
        }
        guard let s_generate = dlsym(h, "bad_apple_generate_text") else {
            lastError = "dlsym: bad_apple_generate_text not found"
            return
        }
        guard let s_free_string = dlsym(h, "bad_apple_free_string") else {
            lastError = "dlsym: bad_apple_free_string not found"
            return
        }
        bad_apple_init = unsafeBitCast(s_init, to: BadAppleInitFn.self)
        bad_apple_free = unsafeBitCast(s_free, to: BadAppleFreeFn.self)
        bad_apple_get_active_pursuits = unsafeBitCast(s_pursuits, to: BadAppleGetActivePursuitsFn.self)
        bad_apple_push_pursuit = unsafeBitCast(s_push, to: BadApplePushPursuitFn.self)
        bad_apple_get_apple_latency_us = unsafeBitCast(s_latency, to: BadAppleGetAppleLatencyUsFn.self)
        bad_apple_generate_text = unsafeBitCast(s_generate, to: BadAppleGenerateTextFn.self)
        bad_apple_free_string = unsafeBitCast(s_free_string, to: BadAppleFreeStringFn.self)

        context = bad_apple_init?(nil)
        if context == nil {
            lastError = "bad_apple_init() returned nil"
            // Continue: the bridge may still be useful for the daemon client.
        }

        // Now that the Rust core is resident, load and initialize the Swift bridge.
        let bridgeSearchPaths = [
            bundleFrameworks + "/libBadAppleBridge.dylib",
            bundleFrameworks + "/../libBadAppleBridge.dylib",
        ]
        for path in bridgeSearchPaths {
            if let bridgeHandle = dlopen(path, RTLD_LAZY) {
                if let sym = dlsym(bridgeHandle, "init_bad_apple_bridge") {
                    let initBridge = unsafeBitCast(sym, to: InitBadAppleBridgeFn.self)
                    initBridge()
                }
                break
            }
        }
    }

    func freeString(_ ptr: UnsafeMutablePointer<CChar>?) {
        guard let ptr = ptr, let freeFn = bad_apple_free_string else { return }
        freeFn(ptr)
    }

    func activePursuits() -> [String] {
        guard let ctx = context, let fn = bad_apple_get_active_pursuits else { return [] }
        guard let raw = fn(ctx) else { return [] }
        defer { freeString(raw) }
        guard let cstr = String(cString: raw, encoding: .utf8) else { return [] }
        if let data = cstr.data(using: .utf8),
           let arr = try? JSONSerialization.jsonObject(with: data, options: []) as? [String] {
            return arr
        }
        return []
    }

    @discardableResult
    func pushPursuit(_ text: String) -> Bool {
        guard let ctx = context, let fn = bad_apple_push_pursuit else { return false }
        return text.withCString { cstr in
            fn(ctx, cstr)
        }
    }

    func appleLatencyUs() -> UInt64 {
        return bad_apple_get_apple_latency_us?() ?? 0
    }

    func generateText(_ prompt: String) -> String? {
        guard let fn = bad_apple_generate_text else { return nil }
        return prompt.withCString { cstr in
            guard let raw = fn(cstr) else { return nil }
            defer { freeString(raw) }
            return String(cString: raw, encoding: .utf8)
        }
    }

    deinit {
        if let ctx = context, let freeFn = bad_apple_free {
            freeFn(ctx)
        }
        if let h = handle {
            dlclose(h)
        }
    }
}

// MARK: - Local neural TTS playback controller (crossfade)

/// Threading contract: all public and internal methods must run on the main queue.
/// The controller crossfades queued WAVs and falls back to `afplay` when `AVAudioPlayer`
/// cannot handle the file.
private final class PiperTTSPlaybackController: NSObject, AVAudioPlayerDelegate {
    private let crossfadeDuration: TimeInterval = 0.05
    private let pollInterval: TimeInterval = 0.010

    private final class PlayItem {
        let url: URL
        var player: AVAudioPlayer?
        var afplayTask: Process?
        let completion: ((Bool) -> Void)?

        init(url: URL, player: AVAudioPlayer? = nil, afplayTask: Process? = nil, completion: ((Bool) -> Void)? = nil) {
            self.url = url
            self.player = player
            self.afplayTask = afplayTask
            self.completion = completion
        }
    }

    private var pending: [PlayItem] = []
    private var current: PlayItem?
    private var previous: [PlayItem] = []
    private var timer: Timer?
    private var isCrossfading = false

    private func ensureMain(_ work: @escaping () -> Void) {
        if Thread.isMainThread {
            work()
        } else {
            DispatchQueue.main.async(execute: work)
        }
    }

    func enqueue(_ url: URL, completion: ((Bool) -> Void)? = nil) {
        ensureMain { [weak self] in
            self?._enqueue(url, completion: completion)
        }
    }

    private func _enqueue(_ url: URL, completion: ((Bool) -> Void)?) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            badAppleVoiceLog("PiperTTSPlayback: missing file \(url.path)")
            notifyCompletion(completion, success: false)
            return
        }

        let item = makePlayItem(url: url, completion: completion)
        if current == nil {
            start(item)
        } else {
            pending.append(item)
        }
    }

    private func makePlayItem(url: URL, completion: ((Bool) -> Void)?) -> PlayItem {
        if let player = try? AVAudioPlayer(contentsOf: url) {
            player.delegate = self
            if !player.prepareToPlay() {
                badAppleVoiceLog("PiperTTSPlayback: prepareToPlay() failed for \(url.path), will attempt afplay fallback")
            }
            return PlayItem(url: url, player: player, completion: completion)
        }

        badAppleVoiceLog("PiperTTSPlayback: AVAudioPlayer failed for \(url.path), falling back to afplay")
        let item = PlayItem(url: url, completion: completion)
        item.afplayTask = makeAfplayTask(item)
        return item
    }

    func stop() {
        ensureMain { [weak self] in
            guard let self = self else { return }
            self.isCrossfading = false
            self.timer?.invalidate()
            self.timer = nil
            self.current?.player?.stop()
            self.current?.player?.delegate = nil
            self.current?.afplayTask?.terminate()
            self.previous.forEach {
                $0.player?.stop()
                $0.player?.delegate = nil
                $0.afplayTask?.terminate()
            }
            self.pending.removeAll()
            self.current = nil
            self.previous.removeAll()
        }
    }

    private func start(_ item: PlayItem) {
        if let player = item.player {
            player.volume = 1.0
            guard player.play() else {
                badAppleVoiceLog("PiperTTSPlayback: play() failed for \(item.url.path)")
                // Fall back to afplay for this item.
                item.player = nil
                item.afplayTask = makeAfplayTask(item)
                start(item)
                return
            }
            current = item
            isCrossfading = false
            startTimer()
            badAppleVoiceLog("PiperTTSPlayback: playing \(item.url.path)")
        } else if let task = item.afplayTask {
            current = item
            isCrossfading = false
            do {
                try task.run()
            } catch {
                badAppleVoiceLog("PiperTTSPlayback afplay launch failed: \(error.localizedDescription)")
                notifyCompletion(item.completion, success: false)
                current = nil
                advanceIfIdle()
            }
        } else {
            badAppleVoiceLog("PiperTTSPlayback: no player or afplay task for \(item.url.path)")
            notifyCompletion(item.completion, success: false)
            advanceIfIdle()
        }
    }

    private func makeAfplayTask(_ item: PlayItem) -> Process {
        let task = Process()
        // The menu bar is a user LaunchAgent in the Aqua session, so it can
        // play audio directly.  Only use launchctl asuser when running as root.
        if getuid() == 0 {
            task.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            let uid = getuid()
            task.arguments = ["asuser", "\(uid)", "/usr/bin/afplay", item.url.path]
        } else {
            task.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
            task.arguments = [item.url.path]
        }
        task.terminationHandler = { [weak self, weak item] task in
            guard let self = self, let item = item else { return }
            let code = task.terminationStatus
            if code != 0 && code != 15 && code != -1 {
                badAppleVoiceLog("PiperTTSPlayback afplay exited with \(code)")
            }
            DispatchQueue.main.async {
                self.finishCurrent(item, success: code == 0)
            }
        }
        return task
    }

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: pollInterval, repeats: true) { [weak self] _ in
            self?.tick()
        }
    }

    private func tick() {
        guard !isCrossfading, let item = current, let player = item.player, player.isPlaying else {
            return
        }
        let remaining = player.duration - player.currentTime
        guard !pending.isEmpty, remaining <= crossfadeDuration, remaining > 0 else { return }

        isCrossfading = true
        let next = pending.removeFirst()
        if let nextPlayer = next.player {
            nextPlayer.volume = 0.0
            guard nextPlayer.play() else {
                badAppleVoiceLog("PiperTTSPlayback: next play() failed for \(next.url.path)")
                pending.insert(next, at: 0)
                isCrossfading = false
                return
            }
            player.setVolume(0.0, fadeDuration: crossfadeDuration)
            nextPlayer.setVolume(1.0, fadeDuration: crossfadeDuration)
            previous.append(item)
            current = next
            badAppleVoiceLog("PiperTTSPlayback: crossfading to \(next.url.path)")
        } else if next.afplayTask != nil {
            // Can\'t crossfade to an afplay-backed item; play it sequentially after the current.
            pending.insert(next, at: 0)
            isCrossfading = false
        } else {
            badAppleVoiceLog("PiperTTSPlayback: next item has no player or afplay task")
            notifyCompletion(next.completion, success: false)
            isCrossfading = false
            advanceIfIdle()
        }
    }

    private func finishCurrent(_ item: PlayItem, success: Bool) {
        if current === item {
            current = nil
            timer?.invalidate()
            timer = nil
        } else if let idx = previous.firstIndex(where: { $0 === item }) {
            previous.remove(at: idx)
            // The crossfade transition is now complete.
            if previous.isEmpty {
                isCrossfading = false
            }
        }
        notifyCompletion(item.completion, success: success)
        advanceIfIdle()
    }

    private func notifyCompletion(_ completion: ((Bool) -> Void)?, success: Bool) {
        DispatchQueue.main.async { completion?(success) }
    }

    private func advanceIfIdle() {
        guard current == nil, !pending.isEmpty else { return }
        start(pending.removeFirst())
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        if let idx = previous.firstIndex(where: { $0.player === player }) {
            let finished = previous.remove(at: idx)
            if previous.isEmpty {
                isCrossfading = false
            }
            notifyCompletion(finished.completion, success: flag)
            return
        }
        guard let item = current, item.player === player else { return }
        finishCurrent(item, success: flag)
    }
}

// MARK: - Local neural TTS client (Piper)

final class PiperTTSClient {
    static let shared = PiperTTSClient()

    private let playback = PiperTTSPlaybackController()
    private let socketPath = "/tmp/badapple_tts.sock"
    private let requestTimeout: TimeInterval = 2.0
    private let responseTimeout: TimeInterval = 60.0
    private var itemCounter = 0

    // Simple queue for streaming TTS chunks in order.
    private struct QueueItem {
        let text: String
        let voice: String
        let id: Int
        let sessionID: Int
        let completion: ((Bool) -> Void)?
    }
    private var queue: [QueueItem] = []
    private let queueLock = NSLock()
    private var isProcessing = false
    private var sessionID = 0
    private var hasErrorInSession = false
    private var lastCompletion: ((Bool) -> Void)?
    // One-shot completion fired when the entire queue drains.  Used by the
    // streaming voice path to restart listening only after every streamed
    // sentence has finished playing.
    private var drainCompletion: ((Bool) -> Void)?
    // Cached reachability probe so repeated voice prompts do not reconnect.
    private var reachabilityCache: (date: Date, value: Bool)?
    private let reachabilityLock = NSLock()

    func stop() {
        queueLock.lock()
        queue.removeAll()
        isProcessing = false
        sessionID += 1
        hasErrorInSession = false
        lastCompletion = nil
        drainCompletion = nil
        queueLock.unlock()
        playback.stop()
    }

    /// Set a one-shot completion fired when the entire TTS queue drains (or
    /// immediately if the queue is already empty).  This lets the streaming
    /// voice path resume listening only after all queued sentences have played.
    func setQueueDrainCompletion(_ completion: @escaping (Bool) -> Void) {
        queueLock.lock()
        drainCompletion = completion
        let empty = queue.isEmpty && !isProcessing
        let sessionSuccess = !hasErrorInSession
        queueLock.unlock()
        if empty {
            DispatchQueue.main.async { completion(sessionSuccess) }
        }
    }

    /// Quick, cached probe of whether the Piper TTS Unix socket is accepting
    /// connections.  The result is cached for 5 s so repeated voice prompts do
    /// not pay the probe cost.  A local-domain connect is near-instant when the
    /// server is up and fails immediately (ENOENT/ECONNREFUSED) when it is down.
    func isReachable() -> Bool {
        reachabilityLock.lock()
        if let cache = reachabilityCache, Date().timeIntervalSince(cache.date) < 5.0 {
            reachabilityLock.unlock()
            return cache.value
        }
        reachabilityLock.unlock()

        let path = effectiveSocketPath()
        // Fast path: if the socket file does not exist the server is not
        // running, so skip the connect entirely.  This keeps the probe instant
        // and main-thread-safe when Piper is not configured.
        guard FileManager.default.fileExists(atPath: path) else {
            reachabilityLock.lock()
            reachabilityCache = (Date(), false)
            reachabilityLock.unlock()
            return false
        }
        var result = false
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        if fd >= 0 {
            defer { close(fd) }
            var tv = timeval(tv_sec: 1, tv_usec: 0)
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            let pathBytes = Array(path.utf8)
            let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
            if pathBytes.count < maxPath {
                pathBytes.withUnsafeBufferPointer { src in
                    _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
                        memcpy(dst, src.baseAddress!, pathBytes.count)
                    }
                }
                addr.sun_len = UInt8(2 + pathBytes.count + 1)
                let len = socklen_t(addr.sun_len)
                let connectResult = withUnsafePointer(to: &addr) { ptr in
                    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                        connect(fd, sockaddrPtr, len)
                    }
                }
                result = (connectResult == 0)
            }
        }
        reachabilityLock.lock()
        reachabilityCache = (Date(), result)
        reachabilityLock.unlock()
        return result
    }

    static let defaultVoice = ProcessInfo.processInfo.environment["BADAPPLE_TTS_VOICE"] ?? "Best"
    static let availableVoices = [
        "Best",
        "en_US-amy-medium",
        "Samantha",
        "com.apple.voice.compact.en-US.Samantha",
        "com.apple.voice.compact.en-GB.Daniel",
        "com.apple.voice.compact.en-AU.Karen",
    ]

    /// Pre-warm Piper by synthesizing a short silent phrase. This loads the voice model
    /// into memory so the first spoken response does not hit the cold-start timeout.
    func warmup() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            do {
                _ = try self.synthesize("hello", voice: PiperTTSClient.defaultVoice)
                badAppleVoiceLog("PiperTTS warmup complete")
            } catch {
                badAppleVoiceLog("PiperTTS warmup failed: \(error.localizedDescription)")
            }
        }
    }

    /// Split text into sentence-ish chunks so the first WAV is small and
    /// starts playing while the rest of the queue is still being synthesized.
    /// Uses the *rightmost* sentence boundary in each window so chunks end
    /// naturally and the next chunk starts on a new sentence.
    private func chunkText(_ text: String, maxLength: Int = 160) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        // Match ., ?, ! followed by whitespace or end of string, or a line break.
        let regex = try! NSRegularExpression(pattern: "[.!?]+(?:\\s+|$)|\\n+", options: [])

        var chunks: [String] = []
        var remaining = trimmed
        while !remaining.isEmpty {
            if remaining.count <= maxLength {
                chunks.append(remaining)
                break
            }

            let prefix = String(remaining.prefix(maxLength))
            let nsRange = NSRange(location: 0, length: (prefix as NSString).length)
            let matches = regex.matches(in: prefix, options: [], range: nsRange)
            var splitIndex: String.Index? = nil
            if let last = matches.last {
                splitIndex = Range(last.range, in: prefix)?.upperBound
            }

            // No sentence boundary in this window — fall back to the last whitespace.
            if splitIndex == nil || splitIndex! <= prefix.index(prefix.startIndex, offsetBy: 20) {
                if let spaceRange = prefix.range(of: " ", options: .backwards) {
                    splitIndex = spaceRange.upperBound
                }
            }

            // If nothing reasonable was found, hard split at the max length.
            if splitIndex == nil || splitIndex! <= prefix.index(prefix.startIndex, offsetBy: 5) {
                splitIndex = prefix.index(prefix.startIndex, offsetBy: maxLength)
            }

            let chunk = String(remaining[..<splitIndex!]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunk.isEmpty {
                chunks.append(chunk)
            }
            remaining = String(remaining[splitIndex!...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return chunks.isEmpty ? [trimmed] : chunks
    }

    /// Enqueue a chunk for synthesis. Chunks play in order so streaming stays smooth.
    /// Long text is broken into sentence chunks so the voice starts earlier.
    func speak(_ text: String, voice: String, completion: ((Bool) -> Void)? = nil) {
        let chunks = chunkText(text)
        guard !chunks.isEmpty else {
            DispatchQueue.main.async { completion?(false) }
            return
        }

        queueLock.lock()
        hasErrorInSession = false
        lastCompletion = completion
        let session = sessionID
        for (index, chunk) in chunks.enumerated() {
            let isLast = index == chunks.count - 1
            itemCounter += 1
            queue.append(QueueItem(text: chunk, voice: voice, id: itemCounter, sessionID: session, completion: isLast ? { [weak self] success in
                guard let self = self else { return }
                let finalSuccess = success && !self.hasErrorInSession
                self.lastCompletion?(finalSuccess)
                self.lastCompletion = nil
            } : nil))
        }
        let shouldStart = !isProcessing
        if shouldStart { isProcessing = true }
        queueLock.unlock()
        if shouldStart {
            processNext()
        }
    }

    private func processNext() {
        queueLock.lock()
        guard !queue.isEmpty else {
            isProcessing = false
            let drain = drainCompletion
            drainCompletion = nil
            let sessionSuccess = !hasErrorInSession
            queueLock.unlock()
            badAppleVoiceLog("PiperTTS queue empty, stopping")
            if let drain = drain {
                DispatchQueue.main.async { drain(sessionSuccess) }
            }
            return
        }
        let item = queue.removeFirst()
        queueLock.unlock()

        badAppleVoiceLog("PiperTTS processNext: \(queue.count) left, text='\(item.text.prefix(40))...'")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            do {
                badAppleVoiceLog("PiperTTS synthesizing \(item.text.prefix(40))...")
                let wavURL = try self.synthesize(item.text, voice: item.voice)
                badAppleVoiceLog("PiperTTS got wav \(wavURL.path)")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self, item.sessionID == self.sessionID else { return }
                    guard self.isProcessing else { return }
                    self.playback.enqueue(wavURL, completion: item.completion)
                    self.processNext()
                }
            } catch {
                badAppleVoiceLog("PiperTTS synthesize error: \(error.localizedDescription)")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self, item.sessionID == self.sessionID else { return }
                    self.hasErrorInSession = true
                    item.completion?(false)
                    self.processNext()
                }
            }
        }
    }

    private func synthesize(_ text: String, voice: String) throws -> URL {
        let safeVoice = voice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? PiperTTSClient.defaultVoice
            : voice
        let request: [String: Any] = ["text": text, "voice": safeVoice]
        let data = try JSONSerialization.data(withJSONObject: request, options: [])
        let response = try unixSocketRequest(data)
        guard let json = try JSONSerialization.jsonObject(with: response) as? [String: Any] else {
            throw NSError(domain: "PiperTTS", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid JSON"])
        }
        guard (json["ok"] as? Bool) == true, let wavPath = json["wav_path"] as? String else {
            let err = json["error"] as? String ?? "unknown"
            throw NSError(domain: "PiperTTS", code: 3, userInfo: [NSLocalizedDescriptionKey: err])
        }
        let url = URL(fileURLWithPath: wavPath)
        guard isTrustedTTSPath(url) else {
            throw NSError(domain: "PiperTTS", code: 4, userInfo: [NSLocalizedDescriptionKey: "untrusted wav path"])
        }
        return url
    }

    private func isTrustedTTSPath(_ url: URL) -> Bool {
        let path = url.path
        let normalized = (path as NSString).standardizingPath
        guard normalized.hasPrefix("/tmp/badapple_tts_") else { return false }
        let ext = (normalized as NSString).pathExtension.lowercased()
        guard ext == "wav" || ext == "caf" else { return false }
        return true
    }

    private func effectiveSocketPath() -> String {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_TTS_SOCKET"], !env.isEmpty {
            return env
        }
        return socketPath
    }

    private func unixSocketRequest(_ data: Data) throws -> Data {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw makeError(code: 11, "socket() failed")
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let resolvedPath = effectiveSocketPath()
        let pathBytes = Array(resolvedPath.utf8)
        let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
        guard pathBytes.count < maxPath else {
            throw makeError(code: 12, "socket path too long")
        }
        pathBytes.withUnsafeBufferPointer { src in
            _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
                memcpy(dst, src.baseAddress!, pathBytes.count)
            }
        }
        addr.sun_len = UInt8(2 + pathBytes.count + 1)

        var tv = timeval(tv_sec: __darwin_time_t(requestTimeout), tv_usec: 0)
        var tvRecv = timeval(tv_sec: __darwin_time_t(responseTimeout), tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tvRecv, socklen_t(MemoryLayout<timeval>.size))

        let len = socklen_t(addr.sun_len)
        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                connect(fd, sockaddrPtr, len)
            }
        }
        guard connectResult == 0 else {
            throw makeError(code: 13, "connect() failed: \(errno)")
        }

        try writeAll(fd, data: data)
        let newline: UInt8 = 0x0A
        try writeAll(fd, data: Data([newline]))

        var response = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
        defer { buffer.deallocate() }
        while true {
            let n = read(fd, buffer, 4096)
            if n > 0 {
                response.append(buffer, count: n)
                if response.contains(0x0A) { break }
            } else if n == 0 {
                break
            } else {
                let err = errno
                if err == EINTR { continue }
                throw makeError(code: 14, "read() failed: \(err)")
            }
        }
        guard !response.isEmpty else {
            throw makeError(code: 15, "empty response from TTS server")
        }
        return response
    }

    private func writeAll(_ fd: Int32, data: Data) throws {
        var total = 0
        try data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let ptr = UnsafeRawPointer(base)
            let count = data.count
            while total < count {
                let n = write(fd, ptr.advanced(by: total), count - total)
                if n < 0 {
                    let err = errno
                    if err == EINTR { continue }
                    throw makeError(code: 16, "write() failed: \(err)")
                }
                guard n > 0 else {
                    throw makeError(code: 16, "write() returned 0")
                }
                total += n
            }
        }
    }

    private func makeError(code: Int, _ message: String) -> NSError {
        return NSError(domain: "PiperTTS", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }

}

// MARK: - Native voice host

private final class BadAppleVoiceHost: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {
    enum State: Equatable {
        case disabled
        case requestingPermission
        case listening
        case awaitingPrompt
        case processing
        case speaking
        case unavailable(String)

        var isUnavailable: Bool {
            if case .unavailable = self { return true }
            return false
        }

        var label: String {
            switch self {
            case .disabled: return "Voice: Off"
            case .requestingPermission: return "Voice: Requesting Permission"
            case .listening: return "Voice: Listening for “hey bad apple”"
            case .awaitingPrompt: return "Voice: Waiting for Prompt"
            case .processing: return "Voice: Generating"
            case .speaking: return "Voice: Speaking"
            case .unavailable(let reason): return "Voice unavailable: \(reason)"
            }
        }
    }

    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    private var pendingSpeechUtterances = 0
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var tapInstalled = false
    private var restartWorkItem: DispatchWorkItem?
    private var awaitingNextUtterance = false
    private var enabled = false
    private var currentSpeakID = 0
    private var speechSafetyWorkItem: DispatchWorkItem?
    private var converter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: false)!
    // Tolerant wake pattern: allows the on-device recognizer to insert filler words
    // such as "at", "my", "and" between "hey", "bad", and "apple".
    // Wake pattern: optional "hey"-like prefix, then "bad apple".
    // Case-insensitive and tolerant of filler words in between.
    private var wakePattern = try! NSRegularExpression(
        pattern: "(?i)^(?:(?:hey|he|hay|my)(?:\\s+\\w+){0,3}\\s+)?bad(?:\\s+\\w+){0,2}\\s+apple\\b"
    )
    private var promptTimer: Timer?
    private var stablePrompt = ""
    private var recognitionTimer: Timer?
    private var lastTranscript = ""
    private var sessionId = 0
    private var attenuator: AVAudioMixerNode?

    var state: State = .disabled {
        didSet {
            guard oldValue != state else { return }
            if state == .awaitingPrompt, stablePrompt.isEmpty,
               UserDefaults.standard.object(forKey: "BadAppleWakeSoundEnabled") as? Bool ?? true {
                // Short "Tink" chime to confirm the wake phrase was heard.
                AudioServicesPlaySystemSound(1113)
            }
            DispatchQueue.main.async { [weak self] in self?.onStateChange?(self?.state ?? .disabled) }
        }
    }
    var onStateChange: ((State) -> Void)?
    var onPrompt: ((String) -> Void)?
    var onTranscript: ((String) -> Void)?
    var onError: ((String) -> Void)?
    var onWaveform: ((Float) -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
        if let phrase = UserDefaults.standard.string(forKey: "BadAppleWakePhrase") {
            setWakePhrase(phrase)
        }
    }

    func setEnabled(_ shouldEnable: Bool) {
        enabled = shouldEnable
        restartWorkItem?.cancel()
        stopPromptTimer()
        stopRecognitionTimer()
        stablePrompt = ""
        lastTranscript = ""
        guard shouldEnable else {
            awaitingNextUtterance = false
            stopRecognition()
            cancelSpeechSafetyTimer()
            synthesizer.stopSpeaking(at: .immediate)
            pendingSpeechUtterances = 0
            state = .disabled
            return
        }
        requestPermissionsAndStart()
    }

    func setMicGain(_ gain: Float) {
        UserDefaults.standard.set(gain, forKey: "BadAppleMicMixerGain")
        attenuator?.volume = gain
    }

    func triggerShortcut() {
        guard enabled else {
            setEnabled(true)
            return
        }
        stablePrompt = ""
        lastTranscript = ""
        scheduleRestart(after: 0.05)
    }

    func setWakePhrase(_ phrase: String) {
        UserDefaults.standard.set(phrase, forKey: "BadAppleWakePhrase")
        let words = phrase
            .components(separatedBy: .whitespacesAndNewlines)
            .map { $0.trimmingCharacters(in: .punctuationCharacters) }
            .filter { !$0.isEmpty }
        guard !words.isEmpty else { return }
        let escaped = words.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "(?:\\\\s+\\\\w+){0,2}\\\\s+")
        let pattern = "(?i)^\\\\b" + escaped + "\\\\b"
        if let regex = try? NSRegularExpression(pattern: pattern) {
            wakePattern = regex
        }
    }

    private func requestPermissionsAndStart() {
        continuePermissionFlow()
    }

    private func continuePermissionFlow() {
        let speechAuth = SFSpeechRecognizer.authorizationStatus()
        let micAuth = AVCaptureDevice.authorizationStatus(for: .audio)
        badAppleVoiceLog("voice auth status: speech=\(speechAuth.rawValue) mic=\(micAuth.rawValue)")

        if speechAuth == .authorized && micAuth == .authorized {
            DispatchQueue.main.async { [weak self] in self?.configureOnDeviceRecognizer() }
            return
        }

        if speechAuth == .denied || speechAuth == .restricted {
            failClosed("speech recognition permission denied")
            return
        }
        if micAuth == .denied || micAuth == .restricted {
            failClosed("microphone permission denied")
            return
        }

        if speechAuth == .notDetermined {
            state = .requestingPermission
            badAppleVoiceLog("requesting speech recognition authorization")
            SFSpeechRecognizer.requestAuthorization { [weak self] speechStatus in
                guard let self = self, self.enabled else { return }
                badAppleVoiceLog("speech authorization status: \(speechStatus.rawValue)")
                if speechStatus == .authorized {
                    self.continuePermissionFlow()
                } else {
                    self.failClosed("speech recognition permission denied")
                }
            }
            return
        }

        if micAuth == .notDetermined {
            state = .requestingPermission
            badAppleVoiceLog("requesting microphone access")
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                guard let self = self, self.enabled else { return }
                badAppleVoiceLog("microphone access: \(granted)")
                if granted {
                    self.continuePermissionFlow()
                } else {
                    self.failClosed("microphone permission denied")
                }
            }
            return
        }
    }

    private func configureOnDeviceRecognizer() {
        guard enabled else { return }
        // The wake phrase is English; keep recognition in en-US even if the system
        // TTS/Siri locale is set to pt-BR.
        guard let localRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")) else {
            failClosed("recognizer unavailable for en-US")
            return
        }
        badAppleVoiceLog("configureOnDeviceRecognizer: locale=\(Locale.current.identifier) supportsOnDevice=\(localRecognizer.supportsOnDeviceRecognition) isAvailable=\(localRecognizer.isAvailable)")
        guard localRecognizer.supportsOnDeviceRecognition else {
            failClosed("on-device recognition is not installed")
            return
        }
        guard localRecognizer.isAvailable else {
            failClosed("on-device recognizer is unavailable")
            return
        }
        recognizer = localRecognizer
        startRecognition()
    }

    private func startRecognition() {
        dispatchPrecondition(condition: .onQueue(.main))
        guard enabled, let recognizer = recognizer else { return }
        guard recognizer.supportsOnDeviceRecognition, recognizer.isAvailable else {
            failClosed("on-device recognizer became unavailable")
            return
        }
        stopRecognition()
        sessionId += 1
        let currentSession = sessionId

        let recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        recognitionRequest.requiresOnDeviceRecognition = true
        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.contextualStrings = [
            "hey bad apple", "bad apple", "bad_apple",
            "open my bad_apple workspace folder",
            "open workspace", "workspace folder",
        ]
        request = recognitionRequest

        let input = audioEngine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            failClosed("microphone input has no usable format")
            return
        }

        // Keep the audio engine running by wiring input -> attenuator -> mixer -> output,
        // but mute the mixer so the user doesn't hear the microphone fed back.
        // The attenuator lowers the mic level so the recognizer gets clean audio.
        let mixer = AVAudioMixerNode()
        mixer.volume = 1.0
        audioEngine.attach(mixer)
        attenuator = mixer
        audioEngine.connect(input, to: mixer, format: inputFormat)

        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false)!
        audioEngine.connect(mixer, to: audioEngine.mainMixerNode, format: inputFormat)
        audioEngine.connect(audioEngine.mainMixerNode, to: audioEngine.outputNode, format: outputFormat)
        audioEngine.mainMixerNode.volume = 0.0
        audioEngine.mainMixerNode.outputVolume = 0.0

        converter = AVAudioConverter(from: inputFormat, to: targetFormat)
        guard let converter = converter else {
            failClosed("could not create audio converter")
            return
        }
        badAppleVoiceLog("startRecognition: inputFormat sampleRate=\(inputFormat.sampleRate) channels=\(inputFormat.channelCount) target sampleRate=16000 channels=1")

        let target = targetFormat
        mixer.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self, weak recognitionRequest] buffer, _ in
            guard let request = recognitionRequest else { return }
            if let data = buffer.floatChannelData?[0], buffer.frameLength > 0 {
                let frames = Int(buffer.frameLength)
                var sum: Float = 0
                for i in 0..<frames { sum += data[i] * data[i] }
                let rms = sqrt(sum / Float(frames))
                let gain = UserDefaults.standard.object(forKey: "BadAppleMicMixerGain") as? Float ?? 1.0
                let level = min(1.0, rms * 4.0 * gain)
                DispatchQueue.main.async { [weak self] in
                    self?.onWaveform?(level)
                }
            }
            let expectedFrames = AVAudioFrameCount(Double(buffer.frameLength) * target.sampleRate / inputFormat.sampleRate)
            let outputFrames = expectedFrames + 1024
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outputFrames) else { return }
            // The converter does not always reset frameLength; tell it the capacity and then clamp to actual.
            outputBuffer.frameLength = outputBuffer.frameCapacity
            var error: NSError?
            var consumedFrames: AVAudioFrameCount = 0
            let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
                let remaining = buffer.frameLength - consumedFrames
                guard remaining > 0 else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                let frames = min(inNumPackets, remaining)
                guard frames > 0, let slice = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: frames) else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                slice.frameLength = frames
                if let src = buffer.floatChannelData, let dst = slice.floatChannelData {
                    for ch in 0..<Int(buffer.format.channelCount) {
                        memcpy(dst[ch], src[ch].advanced(by: Int(consumedFrames)), Int(frames) * MemoryLayout<Float>.size)
                    }
                } else if let src = buffer.int16ChannelData, let dst = slice.int16ChannelData {
                    for ch in 0..<Int(buffer.format.channelCount) {
                        memcpy(dst[ch], src[ch].advanced(by: Int(consumedFrames)), Int(frames) * MemoryLayout<Int16>.size)
                    }
                } else if let src = buffer.int32ChannelData, let dst = slice.int32ChannelData {
                    for ch in 0..<Int(buffer.format.channelCount) {
                        memcpy(dst[ch], src[ch].advanced(by: Int(consumedFrames)), Int(frames) * MemoryLayout<Int32>.size)
                    }
                } else {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumedFrames += frames
                outStatus.pointee = .haveData
                return slice
            }
            let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputBlock)
            outputBuffer.frameLength = min(outputBuffer.frameLength, expectedFrames)
            if let err = error {
                badAppleVoiceLog("converter error: \(err)")
            } else if status != .error, outputBuffer.frameLength > 0 {
                let ptr = outputBuffer.int16ChannelData?[0]
                let first = ptr.map { $0[0] } ?? 0
                let last = ptr.map { $0[Int(outputBuffer.frameLength - 1)] } ?? 0
                badAppleVoiceLog("audio tap: in=\(buffer.frameLength) out=\(outputBuffer.frameLength) expected=\(expectedFrames) first=\(first) last=\(last)")
                request.append(outputBuffer)
            }
        }
        tapInstalled = true

        lastTranscript = ""
        startRecognitionTimer()

        task = recognizer.recognitionTask(with: recognitionRequest) { [weak self, currentSession] result, error in
            DispatchQueue.main.async {
                guard let self = self, self.sessionId == currentSession else { return }
                self.consume(result: result, error: error)
            }
        }
        do {
            audioEngine.prepare()
            try audioEngine.start()
            state = awaitingNextUtterance ? .awaitingPrompt : .listening
            badAppleVoiceLog("audioEngine started, state=\(state)")
        } catch {
            badAppleVoiceLog("audioEngine.start failed: \(error)")
            failClosed("microphone start failed: \(error.localizedDescription)")
        }
    }

    private func consume(result: SFSpeechRecognitionResult?, error: Error?) {
        guard enabled, state != .processing, state != .speaking else { return }
        // Each partial or error means the recognition task is alive; reset the
        // watchdog so it does not tear down a session while the user is still
        // speaking or the recognizer is still thinking.
        startRecognitionTimer()

        if let result = result {
            let transcript = result.bestTranscription.formattedString
            if !transcript.isEmpty {
                lastTranscript = transcript
                onTranscript?(lastTranscript)
            }
            badAppleVoiceLog("consume: isFinal=\(result.isFinal) transcript='\(transcript)' last='\(lastTranscript)' awaitingNextUtterance=\(awaitingNextUtterance)")

            if awaitingNextUtterance {
                let prompt = (wakeSuffix(in: transcript) ?? transcript)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: .punctuationCharacters)
                if prompt != stablePrompt {
                    // Always refresh stablePrompt and reset the prompt timer, even when
                    // the prompt becomes empty, so a stale command is never delivered
                    // after the recognizer revises the transcript back to the wake phrase.
                    stablePrompt = prompt
                    if !prompt.isEmpty {
                        state = .awaitingPrompt
                    }
                    startPromptTimer()
                }
                if result.isFinal {
                    stopPromptTimer()
                    awaitingNextUtterance = false
                    let finalPrompt = stablePrompt
                    stablePrompt = ""
                    if !finalPrompt.isEmpty {
                        deliver(finalPrompt)
                    } else {
                        scheduleRestart(after: 0.15)
                    }
                    return
                }
            } else if !result.isFinal, let suffix = wakeSuffix(in: transcript) {
                // A wake phrase appeared in a partial result. Enter command mode and
                // let the user finish the command in the same or a fresh session.
                if suffix != stablePrompt {
                    if suffix.isEmpty {
                        awaitingNextUtterance = true
                        stablePrompt = ""
                        state = .awaitingPrompt
                    } else {
                        awaitingNextUtterance = true
                        stablePrompt = suffix
                        state = .awaitingPrompt
                    }
                    startPromptTimer()
                }
                return
            } else if result.isFinal {
                // The final transcript is often empty when the task is ended for a
                // restart, so fall back to the last non-empty partial.
                let source = transcript.isEmpty ? lastTranscript : transcript
                if let suffix = wakeSuffix(in: source) {
                    if suffix.isEmpty {
                        awaitingNextUtterance = true
                        stablePrompt = ""
                        state = .awaitingPrompt
                        // Restart quickly so the following command starts in a fresh
                        // recognition request without stale audio context.
                        scheduleRestart(after: 0.05)
                    } else {
                        deliver(suffix)
                    }
                } else {
                    scheduleRestart(after: 0.15)
                }
                return
            }
        }
        if let error = error {
            badAppleVoiceLog("consume error: \(error)")
            let nsError = error as NSError
            if nsError.localizedDescription.contains("No speech detected") {
                onError?(nsError.localizedDescription)
            }
            scheduleRestart(after: 0.5)
        }
    }

    private func startPromptTimer() {
        promptTimer?.invalidate()
        // Give the user a couple of seconds to finish the command after the
        // wake phrase; the timer resets each time the transcript changes.
        let timer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
            guard let self = self, self.enabled, self.awaitingNextUtterance, !self.stablePrompt.isEmpty else { return }
            let prompt = self.stablePrompt
            self.stopPromptTimer()
            self.awaitingNextUtterance = false
            self.stablePrompt = ""
            self.deliver(prompt)
        }
        promptTimer = timer
    }

    private func stopPromptTimer() {
        promptTimer?.invalidate()
        promptTimer = nil
    }

    private func startRecognitionTimer() {
        recognitionTimer?.invalidate()
        // 12 s is long enough for full commands like "hey bad apple open the
        // dashboard and then run the benchmark"; the timer is reset every time
        // the recognizer reports a partial or an error.
        let timer = Timer.scheduledTimer(withTimeInterval: 12.0, repeats: false) { [weak self] _ in
            guard let self = self, self.enabled else { return }
            badAppleVoiceLog("recognition timer fired, lastTranscript='\(self.lastTranscript)' stablePrompt='\(self.stablePrompt)'")
            if self.awaitingNextUtterance, !self.stablePrompt.isEmpty {
                let prompt = self.stablePrompt
                self.stopPromptTimer()
                self.awaitingNextUtterance = false
                self.stablePrompt = ""
                self.deliver(prompt)
            } else if let suffix = self.wakeSuffix(in: self.lastTranscript), !suffix.isEmpty {
                self.stopRecognitionTimer()
                self.deliver(suffix)
            } else if let suffix = self.wakeSuffix(in: self.lastTranscript), suffix.isEmpty {
                // Wake phrase only; restart quickly to capture the command in a clean
                // recognition request.
                self.stopRecognitionTimer()
                self.awaitingNextUtterance = true
                self.stablePrompt = ""
                self.scheduleRestart(after: 0.05)
            } else if self.awaitingNextUtterance {
                // Wake phrase was heard but no command followed. Stop waiting so
                // later unrelated speech is not misinterpreted as a command.
                self.stopRecognitionTimer()
                self.awaitingNextUtterance = false
                self.stablePrompt = ""
                self.scheduleRestart(after: 0.15)
            } else {
                self.stopRecognitionTimer()
                self.scheduleRestart(after: 0.15)
            }
        }
        recognitionTimer = timer
    }

    private func stopRecognitionTimer() {
        recognitionTimer?.invalidate()
        recognitionTimer = nil
    }

    private func wakeSuffix(in transcript: String) -> String? {
        let range = NSRange(transcript.startIndex..<transcript.endIndex, in: transcript)
        guard let match = wakePattern.firstMatch(in: transcript, range: range),
              let swiftRange = Range(match.range, in: transcript) else { return nil }
        return String(transcript[swiftRange.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
    }

    private func deliver(_ prompt: String) {
        badAppleVoiceLog("deliver: \(prompt)")
        stopRecognition()
        stopPromptTimer()
        stopRecognitionTimer()
        stablePrompt = ""
        lastTranscript = ""
        state = .processing
        onPrompt?(prompt)
    }

    private struct ProsodyChunk {
        let text: String
        let rate: Float
        let pitch: Float
        let postDelay: TimeInterval
    }

    /// Pick the highest-quality installed female voice for the selected
    /// California beach-girl accent. US English (Samantha/Ava) is default.
    private func bestVoice() -> AVSpeechSynthesisVoice {
        let accent = UserDefaults.standard.string(forKey: "BadAppleTTSAccent") ?? "en-US"

        switch accent {
        case "en-US":
            for id in [
                "com.apple.eloquence.en-US.Sandy",
                "com.apple.eloquence.en-US.Shelley",
                "com.apple.eloquence.en-US.Flo",
            ] { if let voice = AVSpeechSynthesisVoice(identifier: id) { return voice } }
            if let voice = AVSpeechSynthesisVoice(language: "en-US") { return voice }
        case "uk-UA":
            for id in [
                "com.apple.voice.premium.uk-UA.Lesya",
                "com.apple.voice.enhanced.uk-UA.Lesya",
                "com.apple.voice.superpremium.uk-UA.Lesya",
                "com.apple.voice.compact.uk-UA.Lesya",
            ] { if let voice = AVSpeechSynthesisVoice(identifier: id) { return voice } }
            if let voice = AVSpeechSynthesisVoice(language: "uk-UA") { return voice }
        case "sk-SK":
            for id in [
                "com.apple.voice.premium.sk-SK.Laura",
                "com.apple.voice.enhanced.sk-SK.Laura",
                "com.apple.voice.superpremium.sk-SK.Laura",
                "com.apple.voice.compact.sk-SK.Laura",
            ] { if let voice = AVSpeechSynthesisVoice(identifier: id) { return voice } }
            if let voice = AVSpeechSynthesisVoice(language: "sk-SK") { return voice }
        case "ru-RU":
            for id in [
                "com.apple.voice.premium.ru-RU.Milena",
                "com.apple.voice.enhanced.ru-RU.Milena",
                "com.apple.voice.superpremium.ru-RU.Milena",
                "com.apple.voice.compact.ru-RU.Milena",
            ] { if let voice = AVSpeechSynthesisVoice(identifier: id) { return voice } }
            if let voice = AVSpeechSynthesisVoice(language: "ru-RU") { return voice }
        default:
            for id in [
                "com.apple.voice.premium.es-MX.Paulina",
                "com.apple.voice.enhanced.es-MX.Paulina",
                "com.apple.voice.superpremium.es-MX.Paulina",
                "com.apple.voice.compact.es-MX.Paulina",
            ] { if let voice = AVSpeechSynthesisVoice(identifier: id) { return voice } }
            if let voice = AVSpeechSynthesisVoice(language: "es-MX") { return voice }
        }

        return AVSpeechSynthesisVoice(identifier: "com.apple.speech.synthesis.voice.Fred")
            ?? AVSpeechSynthesisVoice(language: "en-US")!
    }

    /// Normalize text before TTS so ellipses, em dashes, and run-on dashes do
    /// not create awkward dead-air pauses. Fold them into a comma breath.
    private func normalizeForTTS(_ text: String) -> String {
        var normalized = text
        normalized = normalized.replacingOccurrences(of: "\\.{3,}", with: ", ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "…", with: ", ")
        normalized = normalized.replacingOccurrences(of: "[—–]", with: ", ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "-{2,}", with: ", ", options: .regularExpression)
        while normalized.contains("  ") {
            normalized = normalized.replacingOccurrences(of: "  ", with: " ")
        }
        return normalized.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Context-aware prosody for each utterance. Pause, rate, and pitch vary
    /// with the length and punctuation of the chunk so the voice doesn't take
    /// the same breath after every phrase.
    private func prosodyForChunk(_ text: String, ending: Character?) -> ProsodyChunk {
        let wordCount = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
        let isLong = wordCount > 18
        let isShort = wordCount < 6

        let rate: Float
        let pitch: Float
        let postDelay: TimeInterval

        switch ending {
        case "?":
            pitch = 1.03
            postDelay = isLong ? 0.22 : (isShort ? 0.12 : 0.16)
            rate = isLong ? 0.44 : 0.46
        case "!":
            pitch = 1.02
            postDelay = isLong ? 0.20 : (isShort ? 0.10 : 0.14)
            rate = isLong ? 0.44 : 0.46
        case "\n":
            pitch = 0.96
            let extra = text.hasSuffix("\n\n") ? 0.10 : 0.0
            postDelay = 0.18 + extra
            rate = 0.46
        case ".":
            fallthrough
        default:
            pitch = 0.97
            postDelay = isLong ? 0.18 : (isShort ? 0.08 : 0.12)
            rate = isLong ? 0.44 : 0.46
        }

        return ProsodyChunk(text: text, rate: rate, pitch: pitch, postDelay: postDelay)
    }

    /// Split the response into chilled, beachy chunks. US voices stay relaxed
    /// with a slightly slower rate and a soft, natural pitch.
    private func prosodyChunks(from text: String) -> [ProsodyChunk] {
        var chunks: [ProsodyChunk] = []
        var current = ""

        func flush() {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                current = ""
                return
            }
            let ending = trimmed.last
            chunks.append(prosodyForChunk(trimmed, ending: ending))
            current = ""
        }

        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)

            if c == "?" || c == "!" || c == "." || c == "\n" {
                // only end a sentence if the next char is whitespace/EOL,
                // so decimals like "3.14" and mid-word punctuation don't split.
                let next = i + 1 < chars.count ? chars[i + 1] : nil
                if next == nil || next!.isWhitespace || next! == "\n" {
                    flush()
                }
            }

            i += 1
        }

        flush()
        return chunks.isEmpty ? [prosodyForChunk(text, ending: text.last)] : chunks
    }

    private var usePiperTTS: Bool {
        UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? true
    }

    private var selectedPiperVoice: String {
        let raw = UserDefaults.standard.string(forKey: "BadAppleTTSVoice") ?? PiperTTSClient.defaultVoice
        return raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? PiperTTSClient.defaultVoice
            : raw
    }

    /// Stop any in-flight audio so a new request does not stack on old output.
    func stopAllAudio() {
        synthesizer.stopSpeaking(at: .immediate)
        PiperTTSClient.shared.stop()
    }

    // MARK: - Streaming TTS

    // Accumulated raw token text waiting to be split into sentences and spoken.
    private var streamTTSBuffer = ""
    // True while the accumulator is inside a fenced action block (so the raw
    // action JSON is never spoken aloud).
    private var streamTTSInFence = false
    // True once at least one sentence has been queued during this stream.
    var streamTTSEmittedAny = false

    /// Whether streaming TTS is viable right now: Apple TTS is always usable,
    /// and Piper TTS is usable when its local socket is accepting connections.
    var streamingTTSAvailable: Bool {
        if !usePiperTTS { return true }
        return PiperTTSClient.shared.isReachable()
    }

    /// Reset the streaming TTS pipeline at the start of a new voice prompt so
    /// stale audio from the previous response is dropped and the queue is empty.
    func resetStreamingTTS() {
        streamTTSBuffer = ""
        streamTTSInFence = false
        streamTTSEmittedAny = false
        // Stop audio but don't let the delegate's didFinish fire and
        // decrement pendingSpeechUtterances into a stale state.  We set
        // the counter to 0 BEFORE stopping so any in-flight delegate
        // callbacks are no-ops.
        pendingSpeechUtterances = 0
        cancelSpeechSafetyTimer()
        synthesizer.stopSpeaking(at: .immediate)
        PiperTTSClient.shared.stop()
    }

    /// Feed a token chunk to the streaming TTS pipeline.  Complete sentences are
    /// handed to the TTS queue as soon as they arrive so playback starts while
    /// the model is still generating the rest of the response.
    func speakStreamingChunk(_ text: String) {
        guard enabled else { return }
        streamTTSBuffer += normalizeForTTS(text)
        pumpStreamTTS(final: false)
    }

    /// Flush any text remaining in the streaming buffer after generation ends,
    /// and arrange for listening to resume once the queued audio finishes.
    func flushStreamingTTS() {
        pumpStreamTTS(final: true)
        if usePiperTTS {
            // Piper: restart listening when the whole queue drains.  If nothing
            // was queued the completion fires immediately.
            PiperTTSClient.shared.setQueueDrainCompletion { [weak self] _ in
                DispatchQueue.main.async {
                    guard let self = self, self.enabled else { return }
                    self.scheduleRestart(after: 0.25)
                }
            }
        } else {
            // Apple TTS: if nothing was queued the synthesizer delegate will not
            // fire, so restart listening explicitly.  If something WAS queued,
            // the didFinish delegate will restart — but add a safety fallback
            // in case the delegate doesn't fire (e.g., synthesizer was stopped
            // before the utterance started).
            if !streamTTSEmittedAny {
                scheduleRestart(after: 0.1)
            } else {
                // Safety: if the delegate doesn't fire within 5s, restart anyway.
                DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
                    guard let self = self, self.enabled, self.state == .processing || self.state == .speaking else { return }
                    if self.pendingSpeechUtterances == 0 {
                        badAppleVoiceLog("flushStreamingTTS: safety timeout, restarting listening")
                        self.scheduleRestart(after: 0.1)
                    }
                }
            }
        }
    }

    /// Extract complete sentences from the buffer (skipping fenced action
    /// blocks) and queue them via ``speakChunk``.  When `final` is false a
    /// trailing partial line/sentence is kept in the buffer for the next chunk;
    /// when `final` is true everything remaining is emitted.
    private func pumpStreamTTS(final: Bool) {
        var remaining = streamTTSBuffer
        var inFence = streamTTSInFence
        var spoken = ""

        // Process complete lines so fence boundaries are detected correctly; a
        // trailing partial line is held back unless this is the final flush.
        while let nl = remaining.firstIndex(of: "\n") {
            let line = String(remaining[..<nl])
            remaining = String(remaining[remaining.index(after: nl)...])
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                inFence.toggle()
                continue
            }
            if inFence { continue }
            spoken += line + "\n"
        }

        if final {
            if !inFence {
                let trimmed = remaining.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty, !trimmed.hasPrefix("```") {
                    spoken += remaining
                }
            }
            remaining = ""
        }

        streamTTSInFence = inFence

        let (sentences, tail) = splitStreamSentences(spoken)
        for sentence in sentences {
            streamTTSEmittedAny = true
            speakChunk(sentence)
        }
        // Keep any partial sentence (tail) plus the unprocessed partial line so
        // the next chunk can complete them.  Tail precedes the partial line in
        // the original text order.
        streamTTSBuffer = tail + remaining
    }

    /// Split fence-filtered text into complete sentences terminated by `.`,
    /// `!`, `?` (only when followed by whitespace or end-of-string, so decimals
    /// such as "3.14" are not split) or a newline.  Returns the emitted
    /// sentences and the leftover partial sentence that has not yet reached a
    /// boundary.
    private func splitStreamSentences(_ text: String) -> ([String], String) {
        var sentences: [String] = []
        var last = text.startIndex
        var i = text.startIndex
        while i < text.endIndex {
            let ch = text[i]
            if ch == "\n" {
                var end = text.index(after: i)
                while end < text.endIndex, text[end] == " " || text[end] == "\t" {
                    end = text.index(after: end)
                }
                let sentence = String(text[last..<end])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !sentence.isEmpty {
                    sentences.append(sentence)
                }
                last = end
                i = end
                continue
            }
            if ch == "." || ch == "!" || ch == "?" {
                // Consume a run of sentence-ending punctuation.
                var end = text.index(after: i)
                while end < text.endIndex, text[end] == "." || text[end] == "!" || text[end] == "?" {
                    end = text.index(after: end)
                }
                // Only a boundary if the run is followed by whitespace or EOL,
                // so "3.14" or "U.S." are not split mid-token.
                let atEnd = end == text.endIndex
                let nextIsSpace = end < text.endIndex && (text[end] == " " || text[end] == "\t" || text[end] == "\n")
                if atEnd || nextIsSpace {
                    while end < text.endIndex, text[end] == " " || text[end] == "\t" {
                        end = text.index(after: end)
                    }
                    let sentence = String(text[last..<end])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !sentence.isEmpty {
                        sentences.append(sentence)
                    }
                    last = end
                    i = end
                    continue
                }
                // Not a boundary (e.g. the dot in "3.14"); keep scanning.
                i = text.index(after: i)
                continue
            }
            i = text.index(after: i)
        }
        let tail = String(text[last...])
        return (sentences, tail)
    }

    /// Queue a single streamed sentence chunk without stopping any in-flight audio.
    /// This keeps responses smooth while the model is still generating the next chunk.
    func speakChunk(_ text: String) {
        let spoken = normalizeForTTS(text)
        guard enabled else { return }
        guard !spoken.isEmpty else { return }
        state = .speaking

        if usePiperTTS {
            let voice = selectedPiperVoice
            badAppleVoiceLog("speakChunk using Piper TTS (voice=\(voice))")
            PiperTTSClient.shared.speak(spoken, voice: voice)
        } else {
            badAppleVoiceLog("speakChunk using Apple TTS")
            speakWithAppleChunk(spoken)
        }
    }

    private func speakWithAppleChunk(_ text: String) {
        guard enabled else { return }
        guard state == .speaking || state == .processing else { return }
        let voice = bestVoice()
        let normalized = normalizeForTTS(text)
        let chunks = prosodyChunks(from: normalized)
        for chunk in chunks {
            let utterance = AVSpeechUtterance(string: chunk.text)
            utterance.voice = voice
            utterance.rate = chunk.rate
            utterance.pitchMultiplier = chunk.pitch
            utterance.postUtteranceDelay = chunk.postDelay
            utterance.volume = 0.95
            synthesizer.speak(utterance)
        }
        pendingSpeechUtterances += chunks.count
    }

    /// Main entry point: try the local neural Piper TTS first, then fall back
    /// to the on-device Apple speech engine. The result is much more human at
    /// the cost of ~200-600 ms synthesis latency for the 9B response.
    func speak(_ text: String) {
        currentSpeakID += 1
        let id = currentSpeakID
        let spoken = normalizeForTTS(text)
        guard enabled else { return }
        guard !spoken.isEmpty else {
            scheduleRestart(after: 0.1)
            return
        }
        state = .speaking

        // Stop any in-flight audio so we do not stack responses.
        synthesizer.stopSpeaking(at: .immediate)
        PiperTTSClient.shared.stop()
        cancelSpeechSafetyTimer()

        let piperReachable = usePiperTTS && PiperTTSClient.shared.isReachable()
        if piperReachable {
            let voice = selectedPiperVoice
            badAppleVoiceLog("speak using Piper TTS (id=\(id), voice=\(voice))")
            state = .speaking
            PiperTTSClient.shared.speak(spoken, voice: voice) { [weak self] success in
                DispatchQueue.main.async {
                    guard let self = self, self.enabled, self.currentSpeakID == id else { return }
                    self.cancelSpeechSafetyTimer()
                    if success {
                        self.scheduleRestart(after: 0.25)
                    } else {
                        badAppleVoiceLog("Piper TTS failed, falling back to Apple TTS (id=\(id))")
                        self.speakWithApple(spoken, id: id)
                    }
                }
            }
        } else {
            if usePiperTTS {
                badAppleVoiceLog("speak: Piper TTS not reachable, using Apple TTS (id=\(id))")
            } else {
                badAppleVoiceLog("speak using Apple TTS (id=\(id))")
            }
            speakWithApple(spoken, id: id)
        }
    }

    private func speakWithApple(_ text: String, id: Int) {
        guard enabled, currentSpeakID == id else { return }
        let voice = bestVoice()
        let normalized = normalizeForTTS(text)
        let chunks = prosodyChunks(from: normalized)
        badAppleVoiceLog("speaking with \(chunks.count) chunk(s), voice: \(voice.identifier)")

        for chunk in chunks {
            let utterance = AVSpeechUtterance(string: chunk.text)
            utterance.voice = voice
            utterance.rate = chunk.rate
            utterance.pitchMultiplier = chunk.pitch
            utterance.postUtteranceDelay = chunk.postDelay
            utterance.volume = 0.95
            synthesizer.speak(utterance)
        }
        pendingSpeechUtterances = chunks.count
        startSpeechSafetyTimer(forID: id)
    }

    private func startSpeechSafetyTimer(forID id: Int) {
        cancelSpeechSafetyTimer()
        let item = DispatchWorkItem { [weak self] in
            guard let self = self, self.enabled, self.currentSpeakID == id else { return }
            guard self.state == .speaking || self.state == .processing else { return }
            badAppleVoiceLog("speechSafetyTimer: TTS did not finish, forcing restart (id=\(id), pending=\(self.pendingSpeechUtterances))")
            self.synthesizer.stopSpeaking(at: .immediate)
            self.pendingSpeechUtterances = 0
            self.scheduleRestart(after: 0.25)
        }
        speechSafetyWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + 7.0, execute: item)
    }

    private func cancelSpeechSafetyTimer() {
        speechSafetyWorkItem?.cancel()
        speechSafetyWorkItem = nil
    }

    func resumeAfterFailure() {
        guard enabled else { return }
        scheduleRestart(after: 0.5)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        cancelSpeechSafetyTimer()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        cancelSpeechSafetyTimer()
        pendingSpeechUtterances = max(0, pendingSpeechUtterances - 1)
        if pendingSpeechUtterances == 0 {
            // Only restart if we're in speaking state — if we already restarted
            // via the safety timeout or flushStreamingTTS, don't double-restart.
            if state == .speaking {
                scheduleRestart(after: 0.25)
            }
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        cancelSpeechSafetyTimer()
        pendingSpeechUtterances = max(0, pendingSpeechUtterances - 1)
        if state == .speaking, pendingSpeechUtterances == 0 {
            scheduleRestart(after: 0.25)
        }
    }

    private func scheduleRestart(after delay: TimeInterval) {
        stopRecognition()
        stopPromptTimer()
        stopRecognitionTimer()
        guard enabled else { return }
        restartWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in self?.startRecognition() }
        restartWorkItem = item
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: item)
    }

    private func stopRecognition() {
        stopPromptTimer()
        stopRecognitionTimer()
        request?.endAudio()
        task?.cancel()
        task = nil
        request = nil
        if audioEngine.isRunning { audioEngine.stop() }
        if tapInstalled, let attenuator = attenuator {
            attenuator.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.disconnectNodeOutput(audioEngine.inputNode)
        if let attenuator = attenuator {
            audioEngine.disconnectNodeOutput(attenuator)
        }
        audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
        attenuator = nil
    }

    private func failClosed(_ reason: String) {
        badAppleVoiceLog("failClosed: \(reason)")
        DispatchQueue.main.async { [weak self] in
            self?.stopRecognition()
            self?.stopPromptTimer()
            self?.stopRecognitionTimer()
            self?.cancelSpeechSafetyTimer()
            self?.stablePrompt = ""
            self?.lastTranscript = ""
            self?.state = .unavailable(reason)
            self?.onError?(reason)
        }
    }
}

// MARK: - Global shortcuts

private func globalShortcutCallback(_ nextHandler: EventHandlerCallRef?, _ event: EventRef?, _ userData: UnsafeMutableRawPointer?) -> OSStatus {
    guard let userData = userData else { return noErr }
    let shortcut = Unmanaged<BadAppleGlobalShortcut>.fromOpaque(userData).takeUnretainedValue()
    shortcut.trigger()
    return noErr
}

private final class BadAppleGlobalShortcut {
    private var eventHandler: EventHandlerRef?
    private var hotKey: EventHotKeyRef?
    private let hotKeyID: EventHotKeyID
    private let name: String
    private let keyCode: UInt32
    private let modifiers: UInt32
    private let action: () -> Void

    init(name: String, keyCode: UInt32, modifiers: UInt32, id: UInt32, action: @escaping () -> Void) {
        self.name = name
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.hotKeyID = EventHotKeyID(signature: OSType(0x42415641), id: id)
        self.action = action
        register()
    }

    deinit {
        unregister()
    }

    func register() {
        let target = GetApplicationEventTarget()
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        var handler: EventHandlerRef?
        let userData = Unmanaged.passUnretained(self).toOpaque()
        InstallEventHandler(target, globalShortcutCallback, 1, &spec, userData, &handler)
        eventHandler = handler

        var hk: EventHotKeyRef?
        let status = RegisterEventHotKey(keyCode, modifiers, hotKeyID, target, 0, &hk)
        if status == noErr {
            hotKey = hk
            badAppleVoiceLog("registered \(name) shortcut")
        } else {
            badAppleVoiceLog("failed to register \(name) shortcut: \(status)")
        }
    }

    func unregister() {
        if let hotKey = hotKey {
            UnregisterEventHotKey(hotKey)
            self.hotKey = nil
        }
        if let eventHandler = eventHandler {
            RemoveEventHandler(eventHandler)
            self.eventHandler = nil
        }
    }

    func trigger() {
        badAppleVoiceLog("\(name) shortcut triggered")
        DispatchQueue.main.async { [weak self] in
            self?.action()
        }
    }
}

// MARK: - Response actions

private enum BadAppleOperation: String {
    case openApp = "open_app"
    case openWorkspace = "open_workspace"
    case createFile = "create_file"
    case createDirectory = "create_directory"
    case copyFile = "copy_file"
    case moveFile = "move_file"
    case moveToTrash = "move_to_trash"

    var mutatesFiles: Bool {
        switch self {
        case .createFile, .createDirectory, .copyFile, .moveFile, .moveToTrash: return true
        case .openApp, .openWorkspace: return false
        }
    }
}

private struct BadAppleAction {
    let operation: BadAppleOperation
    let target: String?
    let path: String?
    let source: String?
    let destination: String?
    let originalFence: String

    var confirmationDescription: String {
        switch operation {
        case .openApp: return "Open installed application “\(target ?? "")”?"
        case .openWorkspace: return "Open workspace “\(target ?? "")”?"
        case .createFile: return "Create file “\(path ?? "")”?"
        case .createDirectory: return "Create directory “\(path ?? "")”?"
        case .copyFile: return "Copy “\(source ?? "")” to “\(destination ?? "")”?"
        case .moveFile: return "Move “\(source ?? "")” to “\(destination ?? "")”?"
        case .moveToTrash: return "Move “\(path ?? "")” to the Trash?"
        }
    }
}

private struct BadAppleActionParseResult {
    let spoken: String
    let actions: [BadAppleAction]
    let error: String?
}

/// Fast local fallback for common voice actions when the cognitive substrate
/// is unavailable or slow.  Keeps simple open/create/copy commands responsive.
///
/// These patterns are intentionally loose because the on-device speech
/// recognizer tends to stutter words ("Open open my ...").
private enum BadAppleActionResolver {
    // The target is the word immediately before "workspace"/"folder" when one
    // of those appears.  This resists the recognizer stutter ("Open open my ...").
    private static let openWorkspacePattern = try! NSRegularExpression(
        pattern: "(?i)\\b([a-z0-9_\\-]+)\\s+(?:workspace|folder|repo|repository)\\b"
    )
    private static let openAppPattern = try! NSRegularExpression(
        pattern: "(?i)\\b(open|launch)\\b(?:\\s+\\w+){0,2}\\s+(?:the\\s+)?([a-z0-9_\\-]+\\.?(?:app)?)"
    )
    private static let createDirPattern = try! NSRegularExpression(
        pattern: "(?i)\\bcreate\\b(?:\\s+\\w+){0,3}\\s+(?:directory|folder)\\s+(?:at\\s+)?([~/a-z0-9_\\.\\-\\s/]+)"
    )
    private static let stopWords: Set<String> = ["open", "show", "launch", "my", "the", "a", "an", "this", "that", "please"]

    static func resolve(_ prompt: String) -> BadAppleAction? {
        let lower = prompt.lowercased()
        let range = NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)

        if lower.contains("open") || lower.contains("show"), lower.contains("workspace") || lower.contains("folder") || lower.contains("repo") {
            if let match = openWorkspacePattern.firstMatch(in: prompt, range: range),
               let r = Range(match.range(at: 1), in: prompt) {
                let target = String(prompt[r]).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !target.isEmpty, !stopWords.contains(target.lowercased()) else { return nil }
                return BadAppleAction(
                    operation: .openWorkspace,
                    target: target,
                    path: nil, source: nil, destination: nil,
                    originalFence: ""
                )
            }
        }

        if let match = openAppPattern.firstMatch(in: prompt, range: range),
           let r = Range(match.range(at: 2), in: prompt) {
            return BadAppleAction(
                operation: .openApp,
                target: String(prompt[r]).trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: ".app", with: "", options: .caseInsensitive),
                path: nil, source: nil, destination: nil,
                originalFence: ""
            )
        }

        if let match = createDirPattern.firstMatch(in: prompt, range: range),
           let r = Range(match.range(at: 1), in: prompt) {
            return BadAppleAction(
                operation: .createDirectory,
                target: nil,
                path: String(prompt[r]).trimmingCharacters(in: .whitespacesAndNewlines),
                source: nil, destination: nil,
                originalFence: ""
            )
        }

        return nil
    }
}

private enum BadAppleActionParser {
    // This mirrors automation_cage::parse_actions: the opener and closer must
    // occupy their own lines (surrounding horizontal whitespace is allowed).
    // Accept either the canonical `badapple-action` label or a `json` label,
    // because small local models sometimes emit the action object inside a
    // generic `json` fence.
    private static let blockPattern = try! NSRegularExpression(
        pattern: "(?m)^[ \\t]*```(?:badapple-action|json)[ \\t]*\\r?$\\n([\\s\\S]*?)^[ \\t]*```[ \\t]*\\r?$"
    )
    private static let openerPattern = try! NSRegularExpression(
        pattern: "(?m)^[ \\t]*```(?:badapple-action|json)[ \\t]*\\r?$"
    )
    // Some small-context models emit a bare JSON object instead of a fenced
    // block.  This pattern finds the first well-formed badapple-action object.
    private static let plainPattern = try! NSRegularExpression(
        pattern: "\\{[^{}]*\\}"
    )

    static func parse(_ response: String) -> BadAppleActionParseResult {
        let fullRange = NSRange(response.startIndex..<response.endIndex, in: response)
        let matches = blockPattern.matches(in: response, range: fullRange)
        let openerCount = openerPattern.numberOfMatches(in: response, range: fullRange)
        let spoken = blockPattern.stringByReplacingMatches(
            in: response,
            range: fullRange,
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        guard openerCount == matches.count else {
            return BadAppleActionParseResult(
                spoken: spoken,
                actions: [],
                error: "An unterminated badapple-action or json block was rejected."
            )
        }
        if !matches.isEmpty {
            var actions: [BadAppleAction] = []
            for match in matches {
                guard let payloadRange = Range(match.range(at: 1), in: response),
                      let fenceRange = Range(match.range(at: 0), in: response) else {
                    return rejected(spoken, "A malformed badapple-action block was rejected.")
                }
                let payload = String(response[payloadRange])
                let fence = String(response[fenceRange])
                guard !payload.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                      let action = parseActionPayload(payload, fence: fence) else {
                    return rejected(spoken, "Malformed or unknown badapple-action JSON was rejected.")
                }
                actions.append(action)
            }
            return BadAppleActionParseResult(spoken: spoken, actions: actions, error: nil)
        }

        // Fallback: a bare JSON object without fences.
        if let match = plainPattern.firstMatch(in: response, range: fullRange),
           let payloadRange = Range(match.range(at: 0), in: response) {
            let payload = String(response[payloadRange])
            if let action = parseActionPayload(payload, fence: payload) {
                return BadAppleActionParseResult(spoken: spoken, actions: [action], error: nil)
            }
        }

        return BadAppleActionParseResult(spoken: response, actions: [], error: nil)
    }

    private static func parseActionPayload(_ payload: String, fence: String) -> BadAppleAction? {
        guard let data = payload.data(using: .utf8),
              let value = try? JSONSerialization.jsonObject(with: data, options: []),
              let object = value as? [String: Any],
              let operationName = object["operation"] as? String,
              let operation = BadAppleOperation(rawValue: operationName) else {
            return nil
        }

        let requiredKeys: Set<String>
        let target: String?
        let path: String?
        let source: String?
        let destination: String?
        switch operation {
        case .openApp, .openWorkspace:
            requiredKeys = ["operation", "target"]
            target = expandPath(nonemptyString(object["target"]))
            path = nil; source = nil; destination = nil
        case .createFile, .createDirectory, .moveToTrash:
            requiredKeys = ["operation", "path"]
            target = nil
            path = expandPath(nonemptyString(object["path"]))
            source = nil; destination = nil
        case .copyFile, .moveFile:
            requiredKeys = ["operation", "source", "destination"]
            target = nil; path = nil
            source = expandPath(nonemptyString(object["source"]))
            destination = expandPath(nonemptyString(object["destination"]))
        }
        guard Set(object.keys) == requiredKeys,
              requiredKeys.subtracting(["operation"]).allSatisfy({ nonemptyString(object[$0]) != nil }) else {
            return nil
        }
        return BadAppleAction(
            operation: operation,
            target: target,
            path: path,
            source: source,
            destination: destination,
            originalFence: fence
        )
    }

    private static func nonemptyString(_ value: Any?) -> String? {
        guard let string = value as? String else { return nil }
        let trimmed = string.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : string
    }

    private static func expandPath(_ path: String?) -> String? {
        guard let path = path else { return nil }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path.hasPrefix("~/") {
            return home + String(path.dropFirst(1))
        }
        if path.hasPrefix("/Users/"), let slash = path.dropFirst("/Users/".count).firstIndex(of: "/") {
            return home + String(path[slash...])
        }
        return path
    }

    private static func rejected(_ spoken: String, _ message: String) -> BadAppleActionParseResult {
        BadAppleActionParseResult(spoken: spoken, actions: [], error: message)
    }
}

private final class BadAppleActionExecutor {

    private enum WorkspaceResolution {
        case found(URL)
        case notFound
        case failure(String)
    }

    private let fileManager = FileManager.default

    func showParsingFailure(_ message: String) {
        showFailure(message)
    }

    func confirmAndExecute(_ action: BadAppleAction) {
        badAppleVoiceLog("confirmAndExecute: \(action)")
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Allow Bad Apple action?"
        alert.informativeText = action.confirmationDescription
        alert.addButton(withTitle: "Allow")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }

        switch action.operation {
        case .openApp:
            openApplication(named: action.target ?? "")
        case .openWorkspace:
            openWorkspace(named: action.target ?? "")
        case .createFile, .createDirectory, .copyFile, .moveFile, .moveToTrash:
            invokeAutomationHelper(for: action)
        }
    }

    private func openApplication(named requestedName: String) {
        let matches = installedApplications(named: requestedName)
        guard matches.count == 1, let appURL = matches.first else {
            let reason = matches.isEmpty
                ? "No installed application exactly matching “\(requestedName)” was found."
                : "More than one installed application matches “\(requestedName)”; nothing was opened."
            showFailure(reason)
            return
        }
        NSWorkspace.shared.openApplication(at: appURL, configuration: .init()) { _, error in
            if let error = error { DispatchQueue.main.async { self.showFailure(error.localizedDescription) } }
        }
    }

    private func installedApplications(named requestedName: String) -> [URL] {
        let wanted = requestedName.replacingOccurrences(
            of: ".app",
            with: "",
            options: [.caseInsensitive, .anchored, .backwards]
        )
        let roots = [URL(fileURLWithPath: "/Applications"),
                     URL(fileURLWithPath: "/System/Applications"),
                     fileManager.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
        var found: [String: URL] = [:]
        for root in roots where isDirectory(root) {
            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isApplicationKey],
                options: [.skipsHiddenFiles]
            ) else { continue }
            for case let url as URL in enumerator where url.pathExtension.caseInsensitiveCompare("app") == .orderedSame {
                enumerator.skipDescendants()
                if url.deletingPathExtension().lastPathComponent.caseInsensitiveCompare(wanted) == .orderedSame {
                    let canonical = url.standardizedFileURL.resolvingSymlinksInPath()
                    found[canonical.path] = canonical
                }
            }
        }
        return Array(found.values).sorted { $0.path < $1.path }
    }

    private func openWorkspace(named target: String) {
        switch workspaceFromScavengerCatalog(target) {
        case .found(let url):
            openWorkspaceURL(url)
        case .failure(let message):
            showFailure(message)
        case .notFound:
            switch workspaceFromSafeRoots(target) {
            case .found(let url): openWorkspaceURL(url)
            case .failure(let message): showFailure(message)
            case .notFound: showFailure("No unique workspace matching “\(target)” was found in the catalog or safe roots.")
            }
        }
    }

    private func workspaceFromScavengerCatalog(_ target: String) -> WorkspaceResolution {
        let path = ProcessInfo.processInfo.environment["BADAPPLE_SCAVENGER_CATALOG"]
            ?? "/var/lib/bad_apple/scavenger_paths.json"
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: NSString(string: path).expandingTildeInPath))
            guard let entries = try JSONSerialization.jsonObject(with: data) as? [String] else {
                return .failure("The Bad Apple scavenger catalog is not a JSON string array.")
            }
            var matches: [String: URL] = [:]
            for entry in entries {
                var candidate = URL(fileURLWithPath: NSString(string: entry).expandingTildeInPath)
                    .standardizedFileURL.resolvingSymlinksInPath()
                if !isDirectory(candidate) { candidate.deleteLastPathComponent() }
                while candidate.path != "/" {
                    if isDirectory(candidate), candidate.lastPathComponent.caseInsensitiveCompare(target) == .orderedSame {
                        matches[candidate.path] = candidate
                    }
                    candidate.deleteLastPathComponent()
                }
            }
            guard !matches.isEmpty else { return .notFound }
            if matches.count == 1 {
                return .found(matches.values.first!)
            }
            // Disambiguate by counting how many catalog entries live under each match.
            let home = fileManager.homeDirectoryForCurrentUser
            var scores: [URL: Int] = [:]
            for (_, url) in matches {
                let underHome = url.path.hasPrefix(home.path)
                let count = entries.reduce(0) { sum, entry in
                    let entryURL = URL(fileURLWithPath: NSString(string: entry).expandingTildeInPath)
                        .standardizedFileURL.resolvingSymlinksInPath()
                    return sum + (isDescendant(entryURL, of: url) ? 1 : 0)
                }
                scores[url] = count + (underHome ? 1000 : 0)
            }
            if let best = scores.max(by: { $0.value < $1.value })?.key {
                return .found(best)
            }
            return .failure("The scavenger catalog contains multiple workspaces matching “\(target)”.")
        } catch {
            return .failure("Unable to read the Bad Apple scavenger catalog: \(error.localizedDescription)")
        }
    }

    private func workspaceFromSafeRoots(_ target: String) -> WorkspaceResolution {
        var matches: [String: URL] = [:]
        for root in safeRoots() {
            if root.lastPathComponent.caseInsensitiveCompare(target) == .orderedSame {
                matches[root.path] = root
            }
            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            ) else { continue }
            for case let url as URL in enumerator where url.lastPathComponent.caseInsensitiveCompare(target) == .orderedSame {
                let canonical = url.standardizedFileURL.resolvingSymlinksInPath()
                if isDirectory(canonical), isDescendant(canonical, of: root) {
                    matches[canonical.path] = canonical
                }
            }
        }
        if matches.count > 1 {
            return .failure("Multiple safe-root workspaces match “\(target)”; nothing was opened.")
        }
        return matches.values.first.map(WorkspaceResolution.found) ?? .notFound
    }

    private func safeRoots() -> [URL] {
        let environment = ProcessInfo.processInfo.environment
        let configured = environment["BADAPPLE_AUTOMATION_ROOTS"]
            ?? environment["BADAPPLE_SOURCE_ROOTS"]
        let paths = configured.map { $0.split(separator: ":").map(String.init) }
            ?? ["Desktop", "Documents", "Downloads", "Developer", "Projects", "src"]
                .map { fileManager.homeDirectoryForCurrentUser.appendingPathComponent($0).path }
        var seen = Set<String>()
        return paths
            .map { URL(fileURLWithPath: NSString(string: $0).expandingTildeInPath) }
            .map { $0.standardizedFileURL.resolvingSymlinksInPath() }
            .filter { isDirectory($0) && seen.insert($0.path).inserted }
    }

    private func openWorkspaceURL(_ url: URL) {
        if !NSWorkspace.shared.open(url) {
            showFailure("macOS could not open workspace “\(url.path)”.")
        }
    }

    private func invokeAutomationHelper(for action: BadAppleAction) {
        guard action.operation.mutatesFiles else { return }
        guard let helper = automationHelperURL() else {
            showFailure("The badapple-automation helper was not found in an approved location.")
            return
        }
        let fencedInput = action.originalFence
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let process = Process()
            let input = Pipe()
            let combinedOutput = Pipe()
            process.executableURL = helper
            process.arguments = ["--confirm"]
            process.standardInput = input
            process.standardOutput = combinedOutput
            process.standardError = combinedOutput
            do {
                try process.run()
                input.fileHandleForWriting.write(Data(fencedInput.utf8))
                try input.fileHandleForWriting.close()
                let output = combinedOutput.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationReason == .exit, process.terminationStatus == 0 else {
                    let detail = String(data: output, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    DispatchQueue.main.async {
                        self?.showFailure(detail?.isEmpty == false ? detail! : "badapple-automation failed with status \(process.terminationStatus).")
                    }
                    return
                }
            } catch {
                try? input.fileHandleForWriting.close()
                DispatchQueue.main.async { self?.showFailure("Unable to run badapple-automation: \(error.localizedDescription)") }
            }
        }
    }

    private func automationHelperURL() -> URL? {
        var candidates = [URL(fileURLWithPath: "/usr/local/bin/badapple-automation")]
        if let executable = Bundle.main.executableURL {
            candidates.append(executable.deletingLastPathComponent().appendingPathComponent("badapple-automation"))
        }
        candidates.append(
            URL(fileURLWithPath: fileManager.currentDirectoryPath)
                .appendingPathComponent("target/release/badapple_automation")
        )
        return candidates.first { fileManager.isExecutableFile(atPath: $0.path) }
    }

    private func isDescendant(_ candidate: URL, of root: URL) -> Bool {
        candidate.path == root.path || candidate.path.hasPrefix(root.path + "/")
    }

    private func isDirectory(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    private func showFailure(_ message: String) {
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "Bad Apple action not executed"
        alert.informativeText = message
        alert.runModal()
    }
}

// MARK: - Memory governor telemetry

private func systemMemorySnapshot() -> (usedGB: Double, totalGB: Double, pressure: String) {
    var size = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
    var stats = vm_statistics64_data_t()
    let result = withUnsafeMutablePointer(to: &stats) { statsPtr -> kern_return_t in
        statsPtr.withMemoryRebound(to: integer_t.self, capacity: Int(size)) { rawPtr in
            host_statistics64(mach_host_self(), HOST_VM_INFO64, rawPtr, &size)
        }
    }
    let totalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_000_000_000.0
    var usedGB = totalGB
    var pressure = "normal"
    if result == KERN_SUCCESS {
        let pageSize = Double(vm_page_size)
        let freePages = UInt64(stats.free_count) + UInt64(stats.inactive_count) + UInt64(stats.speculative_count) + UInt64(stats.purgeable_count)
        let freeGB = Double(freePages) * pageSize / 1_000_000_000.0
        usedGB = max(0.0, totalGB - freeGB)
        let critical = Double(ProcessInfo.processInfo.environment["BADAPPLE_MEMORY_CRITICAL"] ?? "") ?? 0.95
        let unhealthy = Double(ProcessInfo.processInfo.environment["BADAPPLE_MEMORY_UNHEALTHY"] ?? "") ?? 0.90
        let elevated = Double(ProcessInfo.processInfo.environment["BADAPPLE_MEMORY_ELEVATED"] ?? "") ?? 0.80
        let ratio = usedGB / totalGB
        if ratio >= critical { pressure = "critical" }
        else if ratio >= unhealthy { pressure = "unhealthy" }
        else if ratio >= elevated { pressure = "elevated" }
    }
    return (usedGB, totalGB, pressure)
}

private final class MemoryGovernor {
    private var timer: Timer?
    private var pressureSource: DispatchSourceMemoryPressure?
    var autoPurge = true
    var onUpdate: ((Double, Double, String) -> Void)?
    var onCritical: (() -> Void)?

    func start() {
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.sample()
        }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: DispatchQueue.global(qos: .utility))
        source.setEventHandler { [weak self] in
            let data = source.data
            if data.contains(.critical) {
                self?.onCritical?()
            }
        }
        source.resume()
        pressureSource = source
        sample()
    }

    func stop() {
        timer?.invalidate()
        pressureSource?.cancel()
    }

    private func sample() {
        let (used, total, pressure) = systemMemorySnapshot()
        DispatchQueue.main.async { [weak self] in
            self?.onUpdate?(used, total, pressure)
            if pressure == "critical" && self?.autoPurge == true {
                self?.onCritical?()
            }
        }
    }
}

// MARK: - Menu-bar application

@main
struct BadAppleMenuBarApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}

private enum BadAppleBrain {
    static let fastSocket = "/var/run/badapple/substrate_fast.sock"
    static let deepSocket = "/var/run/badapple/substrate.sock"
    static let directSocket = "/var/run/badapple/substrate_mlx.sock"
    static let keyPath = "/var/lib/bad_apple/slicks.key"
    static let generatedImagesDir = "/var/lib/bad_apple/generated_images"
}

// MARK: - Voice HUD

/// Floating HUD that appears near the menu bar while voice is active.
/// Shows state (listening / awaiting prompt / processing / speaking) and the
/// live transcript, then fades out when the interaction ends.
private final class BadAppleVoiceHUD: NSWindow {
    private let statusDot = NSView()
    private let stateLabel = NSTextField(labelWithString: "Listening")
    private let transcriptLabel = NSTextField(wrappingLabelWithString: "")
    private var idleTimer: Timer?
    private var statusButton: NSButton?
    private var isFadingIn = false
    private var idleTimeout: TimeInterval = 2.5
    private var currentState: BadAppleVoiceHost.State = .disabled
    private var waveformSamples: [Float] = []
    private let waveLayer = CAShapeLayer()

    private var hudEnabled: Bool {
        UserDefaults.standard.object(forKey: "BadAppleVoiceHUDEnabled") as? Bool ?? true
    }

    private static let windowWidth: CGFloat = 360
    private static let baseHeight: CGFloat = 56

    init() {
        let rect = NSRect(
            x: 0,
            y: 0,
            width: BadAppleVoiceHUD.windowWidth,
            height: BadAppleVoiceHUD.baseHeight
        )
        super.init(contentRect: rect, styleMask: [.borderless], backing: .buffered, defer: false)
        isReleasedWhenClosed = false
        isOpaque = false
        hasShadow = true
        backgroundColor = .clear
        level = .mainMenu
        collectionBehavior = [.canJoinAllSpaces, .transient]
        ignoresMouseEvents = true

        let visual = NSVisualEffectView(frame: rect)
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.layer?.cornerRadius = 16
        visual.layer?.borderWidth = 0.5
        visual.layer?.borderColor = NSColor.systemGray.withAlphaComponent(0.3).cgColor

        visual.wantsLayer = true
        visual.layer?.addSublayer(waveLayer)
        waveLayer.frame = NSRect(x: 0, y: 0, width: rect.width, height: 8)
        waveLayer.fillColor = NSColor.systemGreen.withAlphaComponent(0.5).cgColor
        waveLayer.isHidden = true
        waveLayer.autoresizingMask = [.layerWidthSizable]

        contentView = visual

        statusDot.frame = NSRect(x: 16, y: 24, width: 8, height: 8)
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 4
        statusDot.layer?.backgroundColor = NSColor.systemGreen.cgColor
        visual.addSubview(statusDot)

        stateLabel.font = .systemFont(ofSize: 13, weight: .semibold)
        stateLabel.textColor = NSColor.labelColor
        stateLabel.frame = NSRect(x: 34, y: 34, width: 120, height: 18)
        visual.addSubview(stateLabel)

        transcriptLabel.font = .systemFont(ofSize: 14)
        transcriptLabel.textColor = NSColor.secondaryLabelColor
        transcriptLabel.alignment = .left
        transcriptLabel.lineBreakMode = .byWordWrapping
        transcriptLabel.isHidden = true
        transcriptLabel.frame = NSRect(x: 16, y: 8, width: BadAppleVoiceHUD.windowWidth - 32, height: 24)
        transcriptLabel.maximumNumberOfLines = 2
        visual.addSubview(transcriptLabel)

        alphaValue = 0
    }

    func attach(statusButton: NSButton?) {
        self.statusButton = statusButton
    }

    func updateState(_ state: BadAppleVoiceHost.State) {
        currentState = state
        stateLabel.stringValue = state.displayName
        statusDot.layer?.backgroundColor = state.tintColor.cgColor

        if !hudEnabled, state != .disabled, !state.isUnavailable {
            hideAnimated()
            return
        }

        switch state {
        case .disabled, .unavailable:
            hideAnimated()
        case .listening:
            if transcriptLabel.isHidden || transcriptLabel.stringValue.isEmpty {
                transcriptLabel.isHidden = true
                resizeToBase()
            }
            startIdleTimer()
        case .requestingPermission:
            showAnimated()
            cancelIdleTimer()
        case .awaitingPrompt:
            showAnimated()
            startIdleTimer()
        default:
            showAnimated()
            cancelIdleTimer()
        }
    }

    func updateTranscript(_ transcript: String) {
        guard !transcript.isEmpty, hudEnabled else { return }
        transcriptLabel.stringValue = transcript
        transcriptLabel.isHidden = false
        resizeToFitTranscript()
        if alphaValue == 0, !isFadingIn {
            showAnimated()
        }
        if currentState == .awaitingPrompt {
            stateLabel.stringValue = "Hearing command"
        }
        startIdleTimer()
    }

    func updateCommand(_ prompt: String) {
        guard !prompt.isEmpty, hudEnabled else { return }
        stateLabel.stringValue = "Command"
        transcriptLabel.stringValue = prompt
        transcriptLabel.isHidden = false
        resizeToFitTranscript()
        showAnimated()
    }

    func updateResponse(_ response: String) {
        guard !response.isEmpty, hudEnabled else { return }
        stateLabel.stringValue = BadAppleVoiceHost.State.speaking.displayName
        transcriptLabel.stringValue = response
        transcriptLabel.isHidden = false
        resizeToFitTranscript()
        showAnimated()
    }

    func showError(_ message: String) {
        guard hudEnabled else { return }
        stateLabel.stringValue = "Error"
        statusDot.layer?.backgroundColor = NSColor.systemRed.cgColor
        transcriptLabel.stringValue = message
        transcriptLabel.isHidden = false
        resizeToFitTranscript()
        showAnimated()
        cancelIdleTimer()
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { [weak self] in
            self?.startIdleTimer()
        }
    }

    func setIdleTimeout(_ timeout: TimeInterval) {
        idleTimeout = timeout
    }

    func addWaveform(_ level: Float) {
        guard currentState == .listening || currentState == .awaitingPrompt else {
            waveLayer.isHidden = true
            return
        }
        if !hudEnabled { return }
        waveLayer.isHidden = false
        if waveformSamples.count > 80 { waveformSamples.removeFirst() }
        waveformSamples.append(level)
        updateWaveformPath()
    }

    private func updateWaveformPath() {
        let bounds = waveLayer.bounds
        guard bounds.width > 0, bounds.height > 0 else { return }
        let maxBars = max(1, Int(bounds.width / 6))
        let bars = Array(waveformSamples.suffix(maxBars))
        let barWidth = bounds.width / CGFloat(bars.count)
        let path = CGMutablePath()
        for (i, level) in bars.enumerated() {
            let x = CGFloat(i) * barWidth
            let height = CGFloat(level) * bounds.height
            let rect = CGRect(x: x, y: 0, width: barWidth - 1, height: height)
            path.addRect(rect, transform: .identity)
        }
        waveLayer.path = path
    }

    private func positionNearStatusBar() {
        guard let button = statusButton, let window = button.window else { return }
        let buttonRect = button.convert(button.bounds, to: nil)
        let screenRect = window.convertToScreen(buttonRect)
        let x = screenRect.midX - (frame.width / 2)
        let y = screenRect.minY - frame.height - 8
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    private func resizeToBase() {
        let rect = NSRect(x: frame.origin.x, y: frame.origin.y, width: BadAppleVoiceHUD.windowWidth, height: BadAppleVoiceHUD.baseHeight)
        setFrame(rect, display: true)
        (contentView as? NSVisualEffectView)?.frame = contentView?.bounds ?? .zero
        stateLabel.frame = NSRect(x: 34, y: 34, width: 120, height: 18)
        statusDot.frame = NSRect(x: 16, y: 39, width: 8, height: 8)
        transcriptLabel.isHidden = true
    }

    private func resizeToFitTranscript() {
        let width = BadAppleVoiceHUD.windowWidth - 32
        let attributed = NSAttributedString(
            string: transcriptLabel.stringValue,
            attributes: [.font: transcriptLabel.font as Any]
        )
        let size = attributed.boundingRect(
            with: NSSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading]
        )
        let textHeight = min(max(size.height, 20), 48)
        let newHeight = BadAppleVoiceHUD.baseHeight + textHeight - 20

        var frame = self.frame
        frame.size.width = BadAppleVoiceHUD.windowWidth
        frame.size.height = newHeight
        setFrame(frame, display: true)
        (contentView as? NSVisualEffectView)?.frame = contentView?.bounds ?? .zero

        stateLabel.frame = NSRect(x: 34, y: newHeight - 34, width: 120, height: 18)
        statusDot.frame = NSRect(x: 16, y: newHeight - 30, width: 8, height: 8)
        transcriptLabel.frame = NSRect(x: 16, y: 10, width: width, height: textHeight)
    }

    private func showAnimated() {
        cancelIdleTimer()
        guard alphaValue == 0 else { return }
        isFadingIn = true
        positionNearStatusBar()
        orderFront(nil)
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.18
            self.animator().alphaValue = 1.0
        }, completionHandler: { [weak self] in
            self?.isFadingIn = false
        })
    }

    private func hideAnimated() {
        cancelIdleTimer()
        guard alphaValue > 0 else {
            orderOut(nil)
            return
        }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = 0.18
            self.animator().alphaValue = 0.0
        }, completionHandler: { [weak self] in
            self?.orderOut(nil)
        })
    }

    private func startIdleTimer() {
        cancelIdleTimer()
        idleTimer = Timer.scheduledTimer(withTimeInterval: idleTimeout, repeats: false) { [weak self] _ in
            self?.hideAnimated()
        }
    }

    private func cancelIdleTimer() {
        idleTimer?.invalidate()
        idleTimer = nil
    }
}

private extension BadAppleVoiceHost.State {
    var displayName: String {
        switch self {
        case .disabled: return "Voice Off"
        case .requestingPermission: return "Requesting Permission"
        case .listening: return "Listening"
        case .awaitingPrompt: return "Heard Wake"
        case .processing: return "Thinking"
        case .speaking: return "Speaking"
        case .unavailable: return "Unavailable"
        }
    }

    var tintColor: NSColor {
        switch self {
        case .disabled: return .systemGray
        case .requestingPermission: return .systemOrange
        case .listening: return .systemGreen
        case .awaitingPrompt: return .systemYellow
        case .processing: return .systemPurple
        case .speaking: return .systemBlue
        case .unavailable: return .systemRed
        }
    }
}

private final class BadAppleSplashWindow {
    private var window: NSWindow?
    private var statusField: NSTextField?
    private var progressBar: NSProgressIndicator?
    private var startTime = Date()
    private var pollTimer: Timer?

    func show() {
        startTime = Date()
        let size = NSSize(width: 420, height: 260)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        let w = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        w.backgroundColor = NSColor(red: 0.04, green: 0.04, blue: 0.04, alpha: 1.0)
        w.isReleasedWhenClosed = true
        w.isOpaque = false
        w.hasShadow = true
        w.level = .statusBar

        let view = NSView(frame: NSRect(origin: .zero, size: size))
        view.wantsLayer = true
        view.layer?.cornerRadius = 16
        view.layer?.backgroundColor = CGColor(red: 0.04, green: 0.04, blue: 0.04, alpha: 1.0)

        let logo = NSTextField(labelWithString: "🍎")
        logo.font = NSFont.systemFont(ofSize: 56)
        logo.alignment = .center
        logo.textColor = NSColor.white
        logo.frame = NSRect(x: (size.width - 80) / 2, y: 130, width: 80, height: 64)

        let title = NSTextField(labelWithString: "Bad Apple")
        title.font = NSFont.systemFont(ofSize: 22, weight: .semibold)
        title.alignment = .center
        title.textColor = NSColor.white
        title.frame = NSRect(x: 0, y: 95, width: size.width, height: 28)

        let status = NSTextField(labelWithString: "Starting local AI...")
        status.font = NSFont.systemFont(ofSize: 13)
        status.alignment = .center
        status.textColor = NSColor(red: 0.6, green: 0.6, blue: 0.6, alpha: 1.0)
        status.frame = NSRect(x: 0, y: 60, width: size.width, height: 20)

        let hint = NSTextField(labelWithString: "First launch can take ~45 seconds while the 9B model loads.")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.alignment = .center
        hint.textColor = NSColor(red: 0.45, green: 0.45, blue: 0.45, alpha: 1.0)
        hint.frame = NSRect(x: 20, y: 38, width: size.width - 40, height: 16)

        let progress = NSProgressIndicator()
        progress.style = .bar
        progress.isIndeterminate = false
        progress.doubleValue = 0
        progress.minValue = 0
        progress.maxValue = 100
        progress.frame = NSRect(x: 80, y: 22, width: size.width - 160, height: 6)

        view.addSubview(logo)
        view.addSubview(title)
        view.addSubview(status)
        view.addSubview(hint)
        view.addSubview(progress)
        w.contentView = view

        window = w
        statusField = status
        progressBar = progress
        w.makeKeyAndOrderFront(nil)

        // Fallback close after 90s in case the daemon never reports ready.
        Timer.scheduledTimer(withTimeInterval: 90.0, repeats: false) { [weak self] _ in self?.close() }
    }

    func update(status: [String: Any]) {
        guard let statusField = statusField, let progressBar = progressBar else { return }
        let runtime = status["runtime"] as? [String: Any]
        let mode = runtime?["mode"] as? String
        let health = status["health"] as? [String: Any]
        let checks = health?["checks"] as? [String: Any]
        let mainModel = (checks?["main_model"] as? [String: Any])?["ok"] as? Bool ?? false
        let elapsed = -startTime.timeIntervalSinceNow

        let msg: String
        let pct: Double

        if mode == "READY" && mainModel {
            msg = "Ready."
            pct = 100
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in self?.close() }
            pollTimer?.invalidate()
            pollTimer = nil
        } else if mainModel {
            msg = "Warming up..."
            pct = 85
        } else if elapsed > 30 {
            msg = "Loading the 9B model... \(Int(elapsed))s"
            pct = 70
        } else if elapsed > 10 {
            msg = "Loading the 9B model..."
            pct = 50
        } else {
            msg = "Waking up the local brain..."
            pct = 25
        }

        statusField.stringValue = msg
        progressBar.doubleValue = pct
    }

    func setOffline() {
        guard let statusField = statusField else { return }
        statusField.stringValue = "Waiting for daemon..."
    }

    func close() {
        pollTimer?.invalidate()
        pollTimer = nil
        window?.close()
        window = nil
        statusField = nil
        progressBar = nil
    }
}

// MARK: - Voice onboarding

/// First-run onboarding panel that explains the wake phrase, asks for mic/
/// speech-recognition permission, and lets the user skip if they want.
private final class BadAppleVoiceOnboarding {
    private var window: NSWindow?
    var onEnableVoice: (() -> Void)?
    var onNotNow: (() -> Void)?

    func showIfNeeded() {
        guard !UserDefaults.standard.bool(forKey: "BadAppleVoiceOnboarded") else { return }

        let size = NSSize(width: 480, height: 380)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        let w = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        w.backgroundColor = .clear
        w.isReleasedWhenClosed = false
        w.isOpaque = false
        w.hasShadow = true
        w.level = .statusBar

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.layer?.cornerRadius = 20
        visual.layer?.borderWidth = 0.5
        visual.layer?.borderColor = NSColor.systemGray.withAlphaComponent(0.3).cgColor

        let logo = NSTextField(labelWithString: "🎙")
        logo.font = .systemFont(ofSize: 48)
        logo.alignment = .center
        logo.textColor = .labelColor
        logo.frame = NSRect(x: (size.width - 80) / 2, y: 270, width: 80, height: 56)

        let title = NSTextField(labelWithString: "Talk to Bad Apple")
        title.font = .systemFont(ofSize: 22, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: 235, width: size.width, height: 28)

        let body = NSTextField(wrappingLabelWithString: "")
        body.font = .systemFont(ofSize: 14)
        body.textColor = .secondaryLabelColor
        body.alignment = .center
        body.stringValue = """
        Bad Apple can listen for “Hey bad apple” and answer out loud.

        Voice processing is done on this Mac, but macOS still requires permission to use the microphone and on-device speech recognition.

        Say your command, pause briefly, and let Bad Apple reply.
        """
        body.frame = NSRect(x: 40, y: 100, width: size.width - 80, height: 120)

        let enable = NSButton(title: "Enable Voice", target: self, action: #selector(enableVoice(_:)))
        enable.bezelStyle = .rounded
        enable.keyEquivalent = "\r"
        enable.frame = NSRect(x: 90, y: 30, width: 130, height: 32)

        let later = NSButton(title: "Not Now", target: self, action: #selector(dismiss(_:)))
        later.bezelStyle = .rounded
        later.frame = NSRect(x: size.width - 220, y: 30, width: 130, height: 32)

        visual.addSubview(logo)
        visual.addSubview(title)
        visual.addSubview(body)
        visual.addSubview(enable)
        visual.addSubview(later)
        w.contentView = visual

        window = w
        w.makeKeyAndOrderFront(nil)
    }

    @objc private func dismiss(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleVoiceOnboarded")
        window?.orderOut(nil)
        onNotNow?()
    }

    @objc private func enableVoice(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleVoiceOnboarded")
        window?.orderOut(nil)
        onEnableVoice?()
    }
}

// MARK: - First-run onboarding

/// First-run onboarding panel that explains Bad Apple, asks the user to finish
/// platform setup, and points them to the help page.
private final class BadAppleFirstRunOnboarding {
    private var window: NSWindow?
    var onInstall: (() -> Void)?
    var onOpenChat: (() -> Void)?
    var onOpenDashboard: (() -> Void)?
    var onDismiss: (() -> Void)?

    var isPlatformInstalled: Bool {
        // Check both the gatekeeper's front-door socket and the MLX daemon's
        // own socket -- the two run as independent LaunchDaemons, and only
        // checking the gatekeeper's socket can report "installed" even when
        // the MLX daemon itself failed to start.
        UserDefaults.standard.bool(forKey: "BadApplePlatformInstalled")
            || (FileManager.default.fileExists(atPath: "/Library/LaunchDaemons/com.badapple.mlx.plist")
                && FileManager.default.fileExists(atPath: "/Library/LaunchDaemons/com.badapple.gatekeeper.plist")
                && FileManager.default.fileExists(atPath: "/var/run/badapple/substrate.sock")
                && FileManager.default.fileExists(atPath: BadAppleBrain.directSocket))
    }

    func showIfNeeded() {
        guard !UserDefaults.standard.bool(forKey: "BadAppleFirstRunOnboarded") else { return }

        if isPlatformInstalled {
            showWelcomeBack()
        } else {
            showWelcome()
        }
    }

    private func makeWindow(title: String, body: String, buttons: [(title: String, action: Selector, key: Bool)]) {
        let size = NSSize(width: 520, height: 420)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        let w = NSWindow(
            contentRect: frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        w.backgroundColor = .clear
        w.isReleasedWhenClosed = false
        w.isOpaque = false
        w.hasShadow = true
        w.level = .statusBar

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.layer?.cornerRadius = 20
        visual.layer?.borderWidth = 0.5
        visual.layer?.borderColor = NSColor.systemGray.withAlphaComponent(0.3).cgColor

        let logo = NSTextField(labelWithString: "🍎")
        logo.font = .systemFont(ofSize: 48)
        logo.alignment = .center
        logo.textColor = .labelColor
        logo.frame = NSRect(x: (size.width - 80) / 2, y: 300, width: 80, height: 56)

        let titleField = NSTextField(labelWithString: title)
        titleField.font = .systemFont(ofSize: 22, weight: .semibold)
        titleField.alignment = .center
        titleField.textColor = .labelColor
        titleField.frame = NSRect(x: 0, y: 260, width: size.width, height: 32)

        let bodyField = NSTextField(wrappingLabelWithString: body)
        bodyField.font = .systemFont(ofSize: 14)
        bodyField.textColor = .secondaryLabelColor
        bodyField.alignment = .center
        bodyField.frame = NSRect(x: 36, y: 100, width: size.width - 72, height: 140)

        visual.addSubview(logo)
        visual.addSubview(titleField)
        visual.addSubview(bodyField)

        let buttonWidth: CGFloat = 130
        let buttonHeight: CGFloat = 32
        let totalWidth = CGFloat(buttons.count) * buttonWidth + CGFloat(buttons.count - 1) * 16
        var x = (size.width - totalWidth) / 2
        for button in buttons {
            let b = NSButton(title: button.title, target: self, action: button.action)
            b.bezelStyle = .rounded
            b.frame = NSRect(x: x, y: 30, width: buttonWidth, height: buttonHeight)
            if button.key {
                b.keyEquivalent = "\r"
            }
            visual.addSubview(b)
            x += buttonWidth + 16
        }

        w.contentView = visual
        window = w
        w.makeKeyAndOrderFront(nil)
    }

    private func showWelcome() {
        let body = """
        Bad Apple is a local AI operating system layer for macOS. It manages on-device inference, memory, tools, voice, vision, security, and governance entirely on your Mac — it does not send your prompts to the cloud.

        To finish setup, Bad Apple needs to install a small background helper. Your Mac will ask for your password.

        You can chat from the menu bar, type in Terminal, or use your voice.
        """
        makeWindow(
            title: "Welcome to Bad Apple",
            body: body,
            buttons: [
                (title: "Install Now", action: #selector(installNow(_:)), key: true),
                (title: "Later", action: #selector(dismiss(_:)), key: false),
                (title: "Learn More", action: #selector(learnMore(_:)), key: false),
            ]
        )
    }

    private func showWelcomeBack() {
        let body = """
        Bad Apple is installed and ready. Everything runs on your Mac, so your prompts stay private.

        You can start a chat, open the dashboard, or just type from the menu bar.
        """
        makeWindow(
            title: "Welcome back",
            body: body,
            buttons: [
                (title: "Open Chat", action: #selector(openChat(_:)), key: false),
                (title: "Open Dashboard", action: #selector(openDashboard(_:)), key: false),
                (title: "Done", action: #selector(dismiss(_:)), key: true),
            ]
        )
    }

    @objc private func dismiss(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleFirstRunOnboarded")
        window?.orderOut(nil)
        onDismiss?()
    }

    @objc private func installNow(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleFirstRunOnboarded")
        window?.orderOut(nil)
        onInstall?()
    }

    @objc private func openChat(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleFirstRunOnboarded")
        window?.orderOut(nil)
        onOpenChat?()
    }

    @objc private func openDashboard(_ sender: NSButton) {
        UserDefaults.standard.set(true, forKey: "BadAppleFirstRunOnboarded")
        window?.orderOut(nil)
        onOpenDashboard?()
    }

    @objc private func learnMore(_ sender: NSButton) {
        if let url = URL(string: "https://github.com/savageAZfck/Bad_Apple#readme") {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - Voice help window

/// Cheat sheet of useful voice commands. Opened from Voice > Voice Help…
private final class BadAppleVoiceHelpWindow: NSObject, NSWindowDelegate {
    private var window: NSWindow?

    func show() {
        if let window = window, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            return
        }

        let size = NSSize(width: 520, height: 440)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        let w = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        w.title = "Voice Help"
        w.isReleasedWhenClosed = false
        w.level = .normal
        w.delegate = self

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true

        let scroll = NSScrollView(frame: NSRect(x: 20, y: 20, width: size.width - 40, height: size.height - 40))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.borderType = .noBorder
        scroll.autoresizingMask = [.width, .height]

        let textView = NSTextView(frame: scroll.bounds)
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textColor = .labelColor
        textView.font = .systemFont(ofSize: 14)
        textView.autoresizingMask = [.width]

        let helpText = """
Wake phrase
  “Hey bad apple”

The status dot turns yellow and the HUD appears. Then say your command and pause briefly so the recognizer finalizes.

Examples
  • “Hey bad apple, what is the capital of France?”
  • “Hey bad apple, tell me a joke.”
  • “Hey bad apple, open Safari.”
  • “Hey bad apple, open my workspace.”
  • “Hey bad apple, create a folder called notes in /tmp.”
  • “Hey bad apple, what can you do on my Mac?”

Personas
  • “Hey bad apple, switch to wicket.”
  • “Hey bad apple, switch to drill.”
  • “Hey bad apple, switch to genz.”
  • “Hey bad apple, switch to midwest.”

System control
  • “Hey bad apple, run a benchmark.”
  • “Hey bad apple, flush vram.”
  • “Hey bad apple, unload all models.”
  • “Hey bad apple, enable private mode.”
  • “Hey bad apple, kill switch.”

Tips
  • Speak at a normal volume; the input gain is already optimized.
  • Pause after the command. If the HUD says “Heard Wake,” wait for it to flip to “Thinking.”
  • You can turn off the wake sound or HUD in Voice > Voice Settings later.
"""

        textView.string = helpText
        scroll.documentView = textView
        visual.addSubview(scroll)
        w.contentView = visual

        window = w
        w.makeKeyAndOrderFront(nil)
    }

    func windowWillClose(_ notification: Notification) {
        window = nil
    }
}

// MARK: - Voice log window

private final class BadAppleVoiceLogWindow: NSWindow {
    private let textView = NSTextView()
    private let logPath = "/tmp/badapple_voice.log"
    private var refreshTimer: Timer?

    init() {
        let size = NSSize(width: 640, height: 420)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        title = "Voice Log"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow

        let scroll = NSScrollView(frame: NSRect(x: 12, y: 48, width: size.width - 24, height: size.height - 60))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.autoresizingMask = [.width, .height]

        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        textView.textColor = .labelColor
        textView.autoresizingMask = [.width]
        scroll.documentView = textView
        visual.addSubview(scroll)

        let refreshButton = NSButton(title: "Refresh", target: self, action: #selector(refresh(_:)))
        refreshButton.bezelStyle = .rounded
        refreshButton.frame = NSRect(x: size.width - 92, y: 12, width: 80, height: 28)
        refreshButton.autoresizingMask = [.minXMargin]
        visual.addSubview(refreshButton)

        contentView = visual
        refresh(nil)
    }

    override func makeKeyAndOrderFront(_ sender: Any?) {
        super.makeKeyAndOrderFront(sender)
        startTimer()
    }

    override func orderOut(_ sender: Any?) {
        super.orderOut(sender)
        refreshTimer?.invalidate()
        refreshTimer = nil
    }

    private func startTimer() {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            self?.refresh(nil)
        }
    }

    @objc private func refresh(_ sender: Any?) {
        guard FileManager.default.fileExists(atPath: logPath) else {
            textView.string = "No voice log found."
            return
        }
        guard let data = FileManager.default.contents(atPath: logPath),
              let text = String(data: data, encoding: .utf8) else { return }
        let lines = text.components(separatedBy: .newlines).filter { !$0.isEmpty }
        textView.string = lines.suffix(500).joined(separator: "\n")
        textView.scrollToEndOfDocument(nil)
    }
}

// MARK: - Daily briefing window

private final class BadAppleBriefingWindow: NSWindow {
    private let textView = NSTextView()
    private let spinner = NSProgressIndicator()
    private var isRunning = false
    var onRun: ((@escaping (String) -> Void, @escaping () -> Void) -> Void)?

    init() {
        let size = NSSize(width: 560, height: 440)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        title = "Daily Briefing"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow

        let scroll = NSScrollView(frame: NSRect(x: 16, y: 56, width: size.width - 32, height: size.height - 72))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.autoresizingMask = [.width, .height]

        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.font = .systemFont(ofSize: 14)
        textView.textColor = .labelColor
        textView.autoresizingMask = [.width]
        scroll.documentView = textView
        visual.addSubview(scroll)

        let runButton = NSButton(title: "Run Briefing", target: self, action: #selector(run(_:)))
        runButton.bezelStyle = .rounded
        runButton.frame = NSRect(x: size.width - 130, y: 16, width: 110, height: 28)
        runButton.autoresizingMask = [.minXMargin]
        visual.addSubview(runButton)

        spinner.style = .spinning
        spinner.isIndeterminate = true
        spinner.isDisplayedWhenStopped = false
        spinner.frame = NSRect(x: size.width - 150, y: 18, width: 18, height: 18)
        spinner.autoresizingMask = [.minXMargin]
        visual.addSubview(spinner)

        contentView = visual
    }

    override func makeKeyAndOrderFront(_ sender: Any?) {
        super.makeKeyAndOrderFront(sender)
        if textView.string.isEmpty { run(nil) }
    }

    @objc private func run(_ sender: Any?) {
        guard !isRunning else { return }
        isRunning = true
        spinner.startAnimation(nil)
        let append: (String) -> Void = { [weak self] chunk in
            guard let self = self else { return }
            self.textView.string += chunk
            self.textView.scrollToEndOfDocument(nil)
        }
        let finish: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.isRunning = false
            self.spinner.stopAnimation(nil)
        }
        onRun?(append, finish)
    }
}

// MARK: - Image playground window

private final class BadAppleImagePlaygroundWindow: NSWindow, NSTextFieldDelegate {
    private let textField = NSTextField()
    private let statusLabel = NSTextField(labelWithString: "")
    private let imageView = NSImageView()
    private let spinner = NSProgressIndicator()
    private var isRunning = false
    var onGenerate: ((String, @escaping (String, String?) -> Void) -> Void)?

    init() {
        let size = NSSize(width: 520, height: 540)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        title = "Image Playground"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow

        textField.placeholderString = "A dark forest with a glowing path..."
        textField.bezelStyle = .roundedBezel
        textField.delegate = self
        textField.frame = NSRect(x: 16, y: size.height - 52, width: size.width - 120, height: 28)
        textField.autoresizingMask = [.width]
        visual.addSubview(textField)

        let generate = NSButton(title: "Generate", target: self, action: #selector(generate(_:)))
        generate.bezelStyle = .rounded
        generate.frame = NSRect(x: size.width - 96, y: size.height - 52, width: 80, height: 28)
        generate.autoresizingMask = [.minXMargin]
        visual.addSubview(generate)

        spinner.style = .spinning
        spinner.isIndeterminate = true
        spinner.isDisplayedWhenStopped = false
        spinner.frame = NSRect(x: size.width - 88, y: size.height - 50, width: 18, height: 18)
        spinner.autoresizingMask = [.minXMargin]
        visual.addSubview(spinner)

        statusLabel.font = .systemFont(ofSize: 12)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.frame = NSRect(x: 16, y: size.height - 84, width: size.width - 32, height: 20)
        statusLabel.autoresizingMask = [.width]
        visual.addSubview(statusLabel)

        imageView.imageFrameStyle = .none
        imageView.imageScaling = .scaleProportionallyDown
        imageView.frame = NSRect(x: 16, y: 16, width: size.width - 32, height: size.height - 112)
        imageView.autoresizingMask = [.width, .height]
        visual.addSubview(imageView)

        contentView = visual
    }

    override func makeKeyAndOrderFront(_ sender: Any?) {
        super.makeKeyAndOrderFront(sender)
        textField.becomeFirstResponder()
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            generate(nil)
            return true
        }
        return false
    }

    @objc private func generate(_ sender: Any?) {
        guard !isRunning else { return }
        let prompt = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        isRunning = true
        spinner.startAnimation(nil)
        statusLabel.stringValue = "Generating..."
        imageView.image = nil

        onGenerate?(prompt) { [weak self] status, imagePath in
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.spinner.stopAnimation(nil)
                self?.statusLabel.stringValue = status
                if let path = imagePath, let image = NSImage(contentsOfFile: path) {
                    self?.imageView.image = image
                }
            }
        }
    }
}

// MARK: - Status window

private final class BadAppleStatusWindow: NSWindow {
    private let textView = NSTextView()
    private var timer: Timer?

    struct Snapshot {
        let headline: String
        let brain: String
        let memory: String
        let p2p: String
        let mcp: String
        let lastProblem: String
    }

    init() {
        let size = NSSize(width: 420, height: 320)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        title = "Bad Apple Status"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow

        let scroll = NSScrollView(frame: NSRect(x: 20, y: 20, width: size.width - 40, height: size.height - 40))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.borderType = .noBorder
        scroll.autoresizingMask = [.width, .height]

        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textColor = .labelColor
        textView.font = .systemFont(ofSize: 13)
        textView.autoresizingMask = [.width]
        scroll.documentView = textView
        visual.addSubview(scroll)
        contentView = visual
    }

    override func makeKeyAndOrderFront(_ sender: Any?) {
        super.makeKeyAndOrderFront(sender)
        startTimer()
    }

    override func orderOut(_ sender: Any?) {
        super.orderOut(sender)
        timer?.invalidate()
        timer = nil
    }

    var onRefresh: (() -> Snapshot)?

    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.refresh()
        }
    }

    @objc private func refresh() {
        guard let snapshot = onRefresh?() else { return }
        update(snapshot: snapshot)
    }

    func update(snapshot: Snapshot) {
        let text = NSMutableAttributedString()

        let headlineAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 18, weight: .semibold),
            .foregroundColor: NSColor.labelColor,
        ]
        let labelAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13, weight: .semibold),
            .foregroundColor: NSColor.labelColor,
        ]
        let valueAttributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 13),
            .foregroundColor: NSColor.secondaryLabelColor,
        ]

        text.append(NSAttributedString(string: snapshot.headline, attributes: headlineAttributes))
        text.append(NSAttributedString(string: "\n\n"))

        let rows = [
            ("AI brain:", snapshot.brain),
            ("Memory:", snapshot.memory),
            ("P2P:", snapshot.p2p),
            ("MCP:", snapshot.mcp),
            ("Last problem:", snapshot.lastProblem),
        ]
        for (label, value) in rows {
            text.append(NSAttributedString(string: "\(label) ", attributes: labelAttributes))
            text.append(NSAttributedString(string: value, attributes: valueAttributes))
            text.append(NSAttributedString(string: "\n"))
        }

        textView.textStorage?.setAttributedString(text)
    }
}

// MARK: - Screen actions window

private final class BadAppleScreenActionsWindow: NSWindow, NSTextFieldDelegate {
    private let textField = NSTextField()
    private let responseView = NSTextView()
    private let spinner = NSProgressIndicator()
    private var isRunning = false
    var onRun: ((String, @escaping (String) -> Void, @escaping () -> Void) -> Void)?

    init() {
        let size = NSSize(width: 560, height: 460)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        title = "Screen Actions"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow

        let context = NSTextField(wrappingLabelWithString: "Ask Bad Apple to look at the current screen, describe it, read text, or perform an action.")
        context.font = .systemFont(ofSize: 13)
        context.textColor = .secondaryLabelColor
        context.frame = NSRect(x: 16, y: size.height - 72, width: size.width - 32, height: 44)
        context.autoresizingMask = [.width]
        visual.addSubview(context)

        textField.placeholderString = "e.g. What’s on my screen? or Click the OK button"
        textField.bezelStyle = .roundedBezel
        textField.delegate = self
        textField.frame = NSRect(x: 16, y: size.height - 108, width: size.width - 120, height: 28)
        textField.autoresizingMask = [.width]
        visual.addSubview(textField)

        let runButton = NSButton(title: "Run", target: self, action: #selector(run(_:)))
        runButton.bezelStyle = .rounded
        runButton.frame = NSRect(x: size.width - 96, y: size.height - 108, width: 80, height: 28)
        runButton.autoresizingMask = [.minXMargin]
        visual.addSubview(runButton)

        spinner.style = .spinning
        spinner.isIndeterminate = true
        spinner.isDisplayedWhenStopped = false
        spinner.frame = NSRect(x: size.width - 88, y: size.height - 106, width: 18, height: 18)
        spinner.autoresizingMask = [.minXMargin]
        visual.addSubview(spinner)

        let scroll = NSScrollView(frame: NSRect(x: 16, y: 16, width: size.width - 32, height: size.height - 136))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.autoresizingMask = [.width, .height]

        responseView.isEditable = false
        responseView.isSelectable = true
        responseView.drawsBackground = false
        responseView.font = .systemFont(ofSize: 14)
        responseView.textColor = .labelColor
        responseView.autoresizingMask = [.width]
        scroll.documentView = responseView
        visual.addSubview(scroll)

        contentView = visual
    }

    override func makeKeyAndOrderFront(_ sender: Any?) {
        super.makeKeyAndOrderFront(sender)
        textField.becomeFirstResponder()
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            run(nil)
            return true
        }
        return false
    }

    @objc private func run(_ sender: Any?) {
        guard !isRunning else { return }
        let prompt = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        isRunning = true
        spinner.startAnimation(nil)

        let append: (String) -> Void = { [weak self] chunk in
            guard let self = self else { return }
            self.responseView.string += chunk
            self.responseView.scrollToEndOfDocument(nil)
        }
        let finish: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.isRunning = false
            self.spinner.stopAnimation(nil)
        }
        onRun?(prompt, append, finish)
    }
}

// MARK: - Ask palette

private final class BadAppleAskPalette: NSWindow, NSTextFieldDelegate {
    private let textField = NSTextField()
    private let responseView = NSTextView()
    private let sendButton = NSButton()
    private let spinner = NSProgressIndicator()
    private var isSubmitting = false
    var onSubmit: ((String, @escaping (String) -> Void, @escaping () -> Void) -> Void)?

    init() {
        let size = NSSize(width: 560, height: 420)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
        title = "Ask Bad Apple"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.layer?.cornerRadius = 18

        textField.placeholderString = "Ask Bad Apple anything..."
        textField.bezelStyle = .roundedBezel
        textField.delegate = self
        textField.frame = NSRect(x: 16, y: size.height - 52, width: size.width - 120, height: 28)
        textField.autoresizingMask = [.width]
        visual.addSubview(textField)

        sendButton.title = "Ask"
        sendButton.bezelStyle = .rounded
        sendButton.target = self
        sendButton.action = #selector(ask(_:))
        sendButton.frame = NSRect(x: size.width - 96, y: size.height - 52, width: 80, height: 28)
        sendButton.keyEquivalent = "\r"
        sendButton.autoresizingMask = [.minXMargin]
        visual.addSubview(sendButton)

        spinner.style = .spinning
        spinner.isIndeterminate = true
        spinner.isDisplayedWhenStopped = false
        spinner.frame = NSRect(x: size.width - 88, y: size.height - 52, width: 20, height: 20)
        spinner.autoresizingMask = [.minXMargin]
        visual.addSubview(spinner)

        let scroll = NSScrollView(frame: NSRect(x: 16, y: 16, width: size.width - 32, height: size.height - 80))
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = false
        scroll.autoresizingMask = [.width, .height]

        responseView.isEditable = false
        responseView.isSelectable = true
        responseView.drawsBackground = false
        responseView.font = .systemFont(ofSize: 14)
        responseView.textColor = .labelColor
        responseView.autoresizingMask = [.width]
        scroll.documentView = responseView
        visual.addSubview(scroll)

        contentView = visual
    }

    func show() {
        if isVisible {
            makeKeyAndOrderFront(nil)
            textField.becomeFirstResponder()
            return
        }
        responseView.string = ""
        textField.stringValue = ""
        center()
        makeKeyAndOrderFront(nil)
        textField.becomeFirstResponder()
    }

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            ask(nil)
            return true
        }
        return false
    }

    @objc private func ask(_ sender: Any?) {
        guard !isSubmitting else { return }
        let prompt = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        isSubmitting = true
        sendButton.isHidden = true
        spinner.startAnimation(nil)
        responseView.string = ""

        let append: (String) -> Void = { [weak self] chunk in
            guard let self = self else { return }
            self.responseView.string += chunk
            self.responseView.scrollToEndOfDocument(nil)
        }

        let finish: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.isSubmitting = false
            self.sendButton.isHidden = false
            self.spinner.stopAnimation(nil)
            self.textField.becomeFirstResponder()
        }

        onSubmit?(prompt, append, finish)
    }
}

// MARK: - Voice settings window

private final class BadAppleVoiceSettingsWindow: NSWindow {
    private weak var voiceHost: BadAppleVoiceHost?
    private weak var voiceHUD: BadAppleVoiceHUD?
    private var micGainSlider: NSSlider?
    private var micGainLabel: NSTextField?
    private var idleSlider: NSSlider?
    private var idleLabel: NSTextField?

    init(voiceHost: BadAppleVoiceHost, voiceHUD: BadAppleVoiceHUD) {
        let size = NSSize(width: 460, height: 520)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - size.height / 2,
            width: size.width,
            height: size.height
        )
        super.init(contentRect: frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
        self.voiceHost = voiceHost
        self.voiceHUD = voiceHUD
        title = "Voice Settings"
        isReleasedWhenClosed = false

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.layer?.cornerRadius = 16

        let stack = NSStackView(frame: NSRect(x: 24, y: 24, width: size.width - 48, height: size.height - 48))
        stack.orientation = .vertical
        stack.alignment = .left
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = true

        stack.addArrangedSubview(label("Voice"))
        stack.addArrangedSubview(checkbox("Enable voice listening", key: "BadAppleVoiceEnabled", action: #selector(toggleVoiceDefault(_:))))
        stack.addArrangedSubview(checkbox("Show voice HUD", key: "BadAppleVoiceHUDEnabled", action: #selector(hudChanged(_:))))
        stack.addArrangedSubview(checkbox("Play wake sound", key: "BadAppleWakeSoundEnabled"))

        stack.addArrangedSubview(label("Microphone"))
        let micRow = NSStackView()
        micRow.orientation = .horizontal
        micRow.spacing = 12
        micRow.alignment = .centerY
        let micSlider = NSSlider(value: 1.0, minValue: 0.1, maxValue: 2.0, target: self, action: #selector(micGainChanged(_:)))
        micSlider.numberOfTickMarks = 0
        micRow.addArrangedSubview(micSlider)
        let micValue = NSTextField(labelWithString: "1.0x")
        micValue.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        micValue.textColor = .secondaryLabelColor
        micValue.alignment = .right
        micValue.frame = NSRect(x: 0, y: 0, width: 48, height: 18)
        micRow.addArrangedSubview(micValue)
        stack.addArrangedSubview(micRow)
        micGainSlider = micSlider
        micGainLabel = micValue

        stack.addArrangedSubview(label("HUD"))
        let idleRow = NSStackView()
        idleRow.orientation = .horizontal
        idleRow.spacing = 12
        idleRow.alignment = .centerY
        let idleSlider = NSSlider(value: 2.5, minValue: 1.0, maxValue: 10.0, target: self, action: #selector(idleTimeoutChanged(_:)))
        idleSlider.numberOfTickMarks = 0
        idleRow.addArrangedSubview(idleSlider)
        let idleValue = NSTextField(labelWithString: "2.5s")
        idleValue.font = .monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        idleValue.textColor = .secondaryLabelColor
        idleValue.alignment = .right
        idleValue.frame = NSRect(x: 0, y: 0, width: 48, height: 18)
        idleRow.addArrangedSubview(idleValue)
        stack.addArrangedSubview(idleRow)
        self.idleSlider = idleSlider
        idleLabel = idleValue

        stack.addArrangedSubview(label("Wake Phrase"))
        let wakeField = NSTextField(string: UserDefaults.standard.string(forKey: "BadAppleWakePhrase") ?? "Hey bad apple")
        wakeField.target = self
        wakeField.action = #selector(wakePhraseChanged(_:))
        wakeField.bezelStyle = .roundedBezel
        wakeField.cell?.sendsActionOnEndEditing = true
        stack.addArrangedSubview(wakeField)

        let hint = NSTextField(wrappingLabelWithString: "Mic gain applies on the next wake. The wake phrase can contain multiple words; a small number of filler words between them is allowed.")
        hint.font = .systemFont(ofSize: 12)
        hint.textColor = .secondaryLabelColor
        stack.addArrangedSubview(hint)

        let doneButton = NSButton(title: "Done", target: self, action: #selector(closeSettings(_:)))
        doneButton.bezelStyle = .rounded
        stack.addArrangedSubview(doneButton)

        visual.addSubview(stack)
        contentView = visual

        loadDefaults()
    }

    private func loadDefaults() {
        let micGain = UserDefaults.standard.object(forKey: "BadAppleMicMixerGain") as? Float ?? 1.0
        micGainSlider?.floatValue = micGain
        micGainLabel?.stringValue = String(format: "%.1fx", micGain)

        let idle = UserDefaults.standard.object(forKey: "BadAppleVoiceHUDIdleTimeout") as? Double ?? 2.5
        idleSlider?.doubleValue = idle
        idleLabel?.stringValue = String(format: "%.1fs", idle)
    }

    private func label(_ text: String) -> NSTextField {
        let tf = NSTextField(labelWithString: text)
        tf.font = .systemFont(ofSize: 14, weight: .semibold)
        tf.textColor = .labelColor
        return tf
    }

    private func checkbox(_ title: String, key: String, action: Selector? = #selector(anyCheckboxChanged(_:))) -> NSButton {
        let cb = NSButton(checkboxWithTitle: title, target: self, action: action)
        cb.state = UserDefaults.standard.bool(forKey: key) ? .on : .off
        cb.tag = key.hash
        cb.identifier = NSUserInterfaceItemIdentifier(key)
        return cb
    }

    @objc private func toggleVoiceDefault(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: "BadAppleVoiceEnabled")
        if !enabled {
            voiceHost?.setEnabled(false)
        } else {
            voiceHost?.setEnabled(true)
        }
    }

    @objc private func hudChanged(_ sender: NSButton) {
        UserDefaults.standard.set(sender.state == .on, forKey: "BadAppleVoiceHUDEnabled")
    }

    @objc private func micGainChanged(_ sender: NSSlider) {
        let gain = sender.floatValue
        UserDefaults.standard.set(gain, forKey: "BadAppleMicMixerGain")
        micGainLabel?.stringValue = String(format: "%.1fx", gain)
        voiceHost?.setMicGain(gain)
    }

    @objc private func idleTimeoutChanged(_ sender: NSSlider) {
        let timeout = sender.doubleValue
        UserDefaults.standard.set(timeout, forKey: "BadAppleVoiceHUDIdleTimeout")
        idleLabel?.stringValue = String(format: "%.1fs", timeout)
        voiceHUD?.setIdleTimeout(timeout)
    }

    @objc private func wakePhraseChanged(_ sender: NSTextField) {
        let phrase = sender.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !phrase.isEmpty else { return }
        UserDefaults.standard.set(phrase, forKey: "BadAppleWakePhrase")
        voiceHost?.setWakePhrase(phrase)
    }

    @objc private func closeSettings(_ sender: NSButton) {
        close()
    }

    @objc private func anyCheckboxChanged(_ sender: NSButton) {
        guard let key = sender.identifier?.rawValue else { return }
        UserDefaults.standard.set(sender.state == .on, forKey: key)
    }
}

// MARK: - Focus / Do Not Disturb manager

private final class BadAppleFocusManager {
    private static let assertionPath = NSHomeDirectory() + "/Library/DoNotDisturb/DB/Assertions.json"
    private static let dndMode = "com.apple.donotdisturb.mode.default"
    private static let clientID = "com.apple.focus.activity-manager"
    private static let appleEpoch = 978_307_200.0

    static var isEnabled: Bool {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: assertionPath)) else { return false }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        guard let dataArr = json["data"] as? [[String: Any]], let store = dataArr.first else { return false }
        let records = store["storeAssertionRecords"] as? [[String: Any]] ?? []
        return records.contains { record in
            guard let details = record["assertionDetails"] as? [String: Any] else { return false }
            return details["assertionDetailsModeIdentifier"] as? String == dndMode
        }
    }

    static func setEnabled(_ enabled: Bool) -> String {
        do {
            var root: [String: Any]
            if FileManager.default.fileExists(atPath: assertionPath),
               let data = try? Data(contentsOf: URL(fileURLWithPath: assertionPath)),
               let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                root = parsed
            } else {
                root = ["data": [["storeAssertionRecords": [], "storeInvalidationRecords": [], "storeInvalidationRequestRecords": []]]]
            }

            guard var dataArr = root["data"] as? [[String: Any]] else {
                return "Focus: could not parse assertion database."
            }
            if dataArr.isEmpty {
                dataArr = [["storeAssertionRecords": [], "storeInvalidationRecords": [], "storeInvalidationRequestRecords": []]]
            }

            var store = dataArr[0]
            var records = store["storeAssertionRecords"] as? [[String: Any]] ?? []

            if enabled {
                records = [buildRecord()]
            } else {
                records = records.filter { record in
                    guard let details = record["assertionDetails"] as? [String: Any] else { return true }
                    return details["assertionDetailsModeIdentifier"] as? String != dndMode
                }
            }

            store["storeAssertionRecords"] = records
            dataArr[0] = store
            root["data"] = dataArr

            let version = (root["header"] as? [String: Any])?["version"] as? Int ?? 8
            root["header"] = ["version": version, "timestamp": Date().timeIntervalSinceReferenceDate]

            let outData = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys, .prettyPrinted])
            try outData.write(to: URL(fileURLWithPath: assertionPath), options: .atomic)

            postNotifications(enabled: enabled)
            restartDNDDaemon()

            return enabled ? "Focus enabled (Do Not Disturb)." : "Focus disabled."
        } catch {
            return "Focus error: \(error.localizedDescription)"
        }
    }

    private static func buildRecord() -> [String: Any] {
        let uuid = UUID().uuidString.uppercased()
        let now = Date().timeIntervalSinceReferenceDate
        return [
            "assertionDetails": [
                "assertionDetailsIdentifier": clientID,
                "assertionDetailsModeIdentifier": dndMode,
                "assertionDetailsReason": "user-action"
            ],
            "assertionSource": [
                "assertionClientIdentifier": clientID
            ],
            "assertionStartDateTimestamp": now,
            "assertionUUID": uuid
        ]
    }

    private static func postNotifications(enabled: Bool) {
        let name = enabled ? "_NSDoNotDisturbEnabledNotification" : "_NSDoNotDisturbDisabledNotification"
        DistributedNotificationCenter.default().postNotificationName(
            NSNotification.Name(name),
            object: nil,
            userInfo: nil,
            deliverImmediately: true
        )
    }

    private static func restartDNDDaemon() {
        let uid = getuid()
        let domain = "gui/\(uid)"
        for label in ["com.apple.donotdisturbd", "com.apple.ControlCenter"] {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
            process.arguments = ["kickstart", "-k", "\(domain)/\(label)"]
            try? process.run()
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate, @unchecked Sendable {
    private var statusItem: NSStatusItem?
    private var menu: NSMenu?
    private var timer: Timer?
    private let voiceHost = BadAppleVoiceHost()
    // Dedicated serial queue for spawning the badapple CLI during voice
    // queries so the process spawn is not delayed by other global-queue work.
    private let voiceQueue = DispatchQueue(label: "com.badapple.voice", qos: .userInitiated)
    private let actionExecutor = BadAppleActionExecutor()
    private let chatHistoryWindow = ChatHistoryWindow()
    private let chatWindow = BadAppleChatWindow()
    private var streamedTokenCount = 0
    private var lastPrompt = ""
    private var lastError: String?
    private var isSubmittingVoicePrompt = false
    private var voicePromptTimeout: DispatchWorkItem?
    private var voiceStreamingTTSActive = false
    private var lastSpoken: String?
    private var openMenuCount = 0
    private var needsMenuRebuild = false
    private let memoryGovernor = MemoryGovernor()
    private var memoryUsedGB = 0.0
    private var memoryTotalGB = 0.0
    private var memoryPressure = "normal"
    private var activeModels: [String] = ["main_9b"]
    private var cachedModelList: [[String: Any]] = []
    private var cachedHFModelNames: [String] = []
    private var lastTelemetryTime: TimeInterval = 0
    private var lastRuntimeStatus: [String: Any] = [:]
    private var lastRuntimeReachable = false
    private var nativeAgentTasks: [BadAppleAgentTask] = []
    private var autoPurgeEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleAutoPurge") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleAutoPurge") }
    }
    private var fastTierOnly: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleFastTierOnly") as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleFastTierOnly") }
    }
    private var autopilotEnabled: Bool {
        get {
            if let value = UserDefaults.standard.object(forKey: "BadAppleAutopilot") as? Bool {
                return value
            }
            return UserDefaults.standard.object(forKey: "BadAppleSettingsAutopilot") as? Bool ?? false
        }
        set {
            UserDefaults.standard.set(newValue, forKey: "BadAppleAutopilot")
            UserDefaults.standard.set(newValue, forKey: "BadAppleSettingsAutopilot")
        }
    }
    private var focusEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleFocusEnabled") as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleFocusEnabled") }
    }
    private let voiceHUD = BadAppleVoiceHUD()
    private let voiceOnboarding = BadAppleVoiceOnboarding()
    private let firstRunOnboarding = BadAppleFirstRunOnboarding()
    private let onboardingWindow = BadAppleOnboardingWindow()
    private let settingsWindow = BadAppleSettingsWindow()
    private let voiceHelp = BadAppleVoiceHelpWindow()
    private let voiceLog = BadAppleVoiceLogWindow()
    private lazy var voiceShortcut = BadAppleGlobalShortcut(name: "voice", keyCode: UInt32(kVK_ANSI_A), modifiers: UInt32(cmdKey | shiftKey), id: 1) { [weak self] in
        self?.voiceHost.triggerShortcut()
    }
    private lazy var askShortcut = BadAppleGlobalShortcut(name: "ask-palette", keyCode: UInt32(kVK_Space), modifiers: UInt32(cmdKey | shiftKey | optionKey), id: 2) { [weak self] in
        self?.askPalette.show()
    }
    private let askPalette = BadAppleAskPalette()
    private let controlCenter = BadAppleControlCenter()
    private let briefingWindow = BadAppleBriefingWindow()
    private let screenActionsWindow = BadAppleScreenActionsWindow()
    private let imagePlayground = BadAppleImagePlaygroundWindow()
    private let statusWindow = BadAppleStatusWindow()
    private lazy var voiceSettings: BadAppleVoiceSettingsWindow = {
        BadAppleVoiceSettingsWindow(voiceHost: voiceHost, voiceHUD: voiceHUD)
    }()
    private var voiceEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleVoiceEnabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleVoiceEnabled") }
    }
    private var roastEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleRoastEnabled") as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleRoastEnabled") }
    }
    private var selectedPersona: String {
        get { UserDefaults.standard.object(forKey: "BadAppleSelectedPersona") as? String ?? "default" }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleSelectedPersona") }
    }
    private let splash = BadAppleSplashWindow()
    private var runtimeState: [String: Any] {
        // NATIVE ENGINE: When the Swift MLX engine is loaded, return the cached
        // runtime status (refreshed asynchronously via runtimeStatus()) instead
        // of reading the daemon's runtime_state.json file.  A background Task is
        // kicked off here to keep the cache fresh; callers see lastRuntimeStatus
        // immediately and a refreshed value on the next poll.
        if BadAppleEngine.shared.isLoaded {
            Task { @MainActor in
                let status = await BadAppleEngine.shared.runtimeStatus()
                self.lastRuntimeStatus = status
            }
            return lastRuntimeStatus
        }
        let path = URL(fileURLWithPath: "/var/lib/bad_apple/runtime_state.json")
        guard let data = try? Data(contentsOf: path),
              let state = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return state
    }

    private func systemBootTime() -> Date? {
        var bootTime = timeval()
        var mib = [CTL_KERN, KERN_BOOTTIME]
        var size = MemoryLayout<timeval>.size
        let result = sysctl(&mib, u_int(mib.count), &bootTime, &size, nil, 0)
        guard result == 0 else { return nil }
        return Date(timeIntervalSince1970: TimeInterval(bootTime.tv_sec))
    }

    private var shouldShowBootSplash: Bool {
        guard let bootTime = systemBootTime() else { return true }
        let defaults = UserDefaults.standard
        if let lastBoot = defaults.object(forKey: "BadAppleSplashBootTime") as? Date,
           lastBoot.compare(bootTime) == .orderedSame {
            return false
        }
        defaults.set(bootTime, forKey: "BadAppleSplashBootTime")
        return true
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Bail out if another Bad Apple menu bar is already running.
        let bundleID = Bundle.main.bundleIdentifier ?? "com.badapple.menubar"
        let alreadyRunning = NSWorkspace.shared.runningApplications.filter {
            $0.bundleIdentifier == bundleID && $0.processIdentifier != ProcessInfo.processInfo.processIdentifier
        }
        if !alreadyRunning.isEmpty {
            badAppleVoiceLog("Another Bad Apple menu bar is already running (pids: \(alreadyRunning.map { $0.processIdentifier })); terminating.")
            NSApp.terminate(nil)
            return
        }

        if shouldShowBootSplash {
            splash.show()
        }
        BadAppleFFI.shared.load()
        startAquaHelper()
        BadAppleMenuBarUIResponder.shared.start()
        registerSMAppService()
        let nativeEngine = BadAppleEngine.shared
        nativeEngine.autopilot = autopilotEnabled
        nativeEngine.workspacePath = UserDefaults.standard.string(forKey: "BadAppleSettingsWorkspace")
        _ = nativeEngine.switchPersona(roastEnabled ? "drill" : selectedPersona)
        // If the daemon is already running, the menu bar should not load a
        // second copy of the 9B model into the same 16 GB machine. Route
        // through the daemon instead and let the status dot go green.
        let daemonAlreadyRunning = FileManager.default.fileExists(atPath: BadAppleBrain.deepSocket)
        if daemonAlreadyRunning {
            badAppleVoiceLog("daemon already running on \(BadAppleBrain.deepSocket) — not loading in-process engine")
            syncAutopilotToDaemon()
        } else {
            Task.detached(priority: .background) {
                await nativeEngine.loadModel()
                await MainActor.run {
                    if BadAppleEngine.shared.isLoaded {
                        badAppleVoiceLog("Native Swift MLX engine loaded — queries will bypass the daemon")
                        self.rebuildMenu()
                    } else {
                        badAppleVoiceLog("Native MLX engine not loaded — falling back to daemon")
                    }
                }
            }
        }
        // Do not remove the legacy LaunchAgent while running; the app is still
        // distributed through a LaunchAgent on this install path, and bootout
        // would kill the menu bar before it can display.
        // removeLegacyLaunchAgent()
        // Prevent AppKit from treating this LSUIElement as idle and terminating it.
        ProcessInfo.processInfo.disableAutomaticTermination("Bad Apple menu bar host")
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "🍎"
        menu = NSMenu(title: "Bad Apple")
        menu?.delegate = self
        statusItem?.menu = menu

        voiceHUD.attach(statusButton: statusItem?.button)
        voiceHost.onStateChange = { [weak self] state in
            self?.rebuildMenu()
            self?.updateStatusIcon()
            self?.voiceHUD.updateState(state)
            // When the wake word is heard, pre-warm the badapple CLI binary into
            // the OS file cache so the process spawn is instant by the time the
            // user finishes speaking their command.
            if case .awaitingPrompt = state {
                self?.prewarmVoiceCLI()
            }
        }
        voiceHost.onTranscript = { [weak self] transcript in
            self?.voiceHUD.updateTranscript(transcript)
        }
        voiceHost.onWaveform = { [weak self] level in
            self?.voiceHUD.addWaveform(level)
        }
        voiceHost.onError = { [weak self] message in
            self?.voiceHUD.showError(message)
        }
        _ = voiceShortcut
        _ = askShortcut
        voiceHost.onPrompt = { [weak self] prompt in
            self?.voiceHUD.updateCommand(prompt)
            self?.submitVoicePrompt(prompt)
        }
        askPalette.onSubmit = { [weak self] prompt, append, finish in
            guard let self = self else { finish(); return }
            if BadAppleEngine.shared.isLoaded {
                BadAppleEngine.shared.generateStreaming(
                    prompt: prompt,
                    voiceMode: false,
                    maxTokens: 512
                ) { chunk in
                    DispatchQueue.main.async { append(chunk) }
                } onComplete: { _ in
                    DispatchQueue.main.async { finish() }
                } onError: { error in
                    DispatchQueue.main.async {
                        append("\n\nError: \(error)")
                        finish()
                    }
                }
                return
            }
            Task {
                do {
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: prompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 512
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("\n\nError: \(error.localizedDescription)") }
                }
                DispatchQueue.main.async { finish() }
            }
        }
        chatWindow.onSubmit = { [weak self] prompt, append, finish in
            guard let self = self else { finish(); return }
            // NATIVE ENGINE: If the Swift MLX engine is loaded, route directly
            // through it — no subprocess, no daemon, no Python.
            if BadAppleEngine.shared.isLoaded {
                BadAppleEngine.shared.generateStreaming(
                    prompt: prompt,
                    voiceMode: false,
                    maxTokens: 512
                ) { chunk in
                    DispatchQueue.main.async { append(chunk) }
                } onComplete: { _ in
                    DispatchQueue.main.async { finish() }
                } onError: { error in
                    DispatchQueue.main.async {
                        append("Error: \(error)")
                        finish()
                    }
                }
                return
            }
            Task {
                do {
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: prompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 512
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("Error: \(error.localizedDescription)") }
                }
                DispatchQueue.main.async { finish() }
            }
        }
        chatWindow.onNewChat = { [weak self] in
            self?.newChat()
        }
        chatWindow.onDescribeImage = { [weak self] imagePath, imageName, append, finish in
            guard let self = self else { finish(); return }
            // NATIVE ENGINE: route through Swift if loaded.
            if BadAppleEngine.shared.isLoaded {
                BadAppleEngine.shared.describeImage(
                    at: imagePath,
                    prompt: "Describe this image in detail."
                ) { chunk in
                    append(chunk)
                } onComplete: { _ in
                    finish()
                } onError: { error in
                    append("Error: \(error)")
                    finish()
                }
                return
            }
            Task {
                do {
                    let prompt = "Describe this image in detail: \(imagePath)"
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: prompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 512
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("Error: \(error.localizedDescription)") }
                }
                DispatchQueue.main.async { finish() }
            }
        }
        briefingWindow.onRun = { [weak self] append, finish in
            guard let self = self else { finish(); return }
            let formatter = DateFormatter()
            formatter.dateStyle = .full
            formatter.timeStyle = .short
            let today = formatter.string(from: Date())
            let prompt = "Give me a concise daily briefing for \(today). Use local tools to check today’s calendar events, upcoming events, reminders, unread emails, the current workspace, and any relevant context. Summarize what’s coming up and what I should prioritize."
            // NATIVE ENGINE: If the Swift MLX engine is loaded, route directly
            // through it — no subprocess, no daemon, no Python.
            if BadAppleEngine.shared.isLoaded {
                BadAppleEngine.shared.generateStreaming(
                    prompt: prompt,
                    voiceMode: false,
                    maxTokens: 500
                ) { chunk in
                    DispatchQueue.main.async { append(chunk) }
                } onComplete: { _ in
                    DispatchQueue.main.async { finish() }
                } onError: { error in
                    DispatchQueue.main.async {
                        append("\n\nError: \(error)")
                        finish()
                    }
                }
                return
            }
            Task {
                do {
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: prompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 500
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("\n\nError: \(error.localizedDescription)") }
                }
                DispatchQueue.main.async { finish() }
            }
        }
        screenActionsWindow.onRun = { [weak self] prompt, append, finish in
            guard let self = self else { finish(); return }
            let context = self.currentScreenContext()
            let fullPrompt = """
            Current Mac context:
            \(context)

            User request about the screen or current app:
            \(prompt)
            """
            // NATIVE ENGINE: If the Swift MLX engine is loaded, route directly
            // through it — no subprocess, no daemon, no Python.
            if BadAppleEngine.shared.isLoaded {
                BadAppleEngine.shared.generateStreaming(
                    prompt: fullPrompt,
                    voiceMode: false,
                    maxTokens: 400
                ) { chunk in
                    DispatchQueue.main.async { append(chunk) }
                } onComplete: { _ in
                    DispatchQueue.main.async { finish() }
                } onError: { error in
                    DispatchQueue.main.async {
                        append("\n\nError: \(error)")
                        finish()
                    }
                }
                return
            }
            Task {
                do {
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: fullPrompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 400
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("\n\nError: \(error.localizedDescription)") }
                }
                DispatchQueue.main.async { finish() }
            }
        }
        imagePlayground.onGenerate = { [weak self] prompt, completion in
            guard let self = self else { completion("Cancelled", nil); return }
            // Image generation is handled natively by the daemon through the
            // `image_generation` tool, which invokes the local `mflux-generate-flux2`
            // binary when mflux is installed.
            let fullPrompt = "generate an image of \(prompt)"
            let before = Date().timeIntervalSince1970
            Task {
                do {
                    let result = try await self.runBadAppleCLI(
                        prompt: fullPrompt,
                        socketPath: BadAppleBrain.directSocket,
                        maxTokens: 160,
                        timeout: 900,
                        extraEnv: ["BADAPPLE_FAST_TIER": "0"]
                    )
                    var path = self.extractImagePath(from: result)
                    if path == nil {
                        path = self.latestGeneratedImage(since: before)
                    }
                    DispatchQueue.main.async { completion(result, path) }
                } catch {
                    DispatchQueue.main.async { completion("Error: \(error.localizedDescription)", nil) }
                }
            }
        }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            let now = Date().timeIntervalSince1970
            if now - (self?.lastTelemetryTime ?? 0) >= 5.0 {
                self?.lastTelemetryTime = now
                self?.refreshTelemetry()
            }
        }
        rebuildMenu()
        voiceOnboarding.onEnableVoice = { [weak self] in
            UserDefaults.standard.set(true, forKey: "BadAppleVoiceEnabled")
            self?.voiceHost.setEnabled(true)
            self?.rebuildMenu()
        }
        voiceOnboarding.onNotNow = { [weak self] in
            UserDefaults.standard.set(false, forKey: "BadAppleVoiceEnabled")
            self?.voiceHost.setEnabled(false)
            self?.rebuildMenu()
        }
        if voiceEnabled {
            if UserDefaults.standard.bool(forKey: "BadAppleVoiceOnboarded") {
                voiceHost.setEnabled(true)
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                    self?.voiceOnboarding.showIfNeeded()
                }
            }
        } else {
            voiceHost.setEnabled(false)
        }

        firstRunOnboarding.onInstall = { [weak self] in
            self?.runFirstRunInstaller()
        }
        firstRunOnboarding.onOpenChat = { [weak self] in
            self?.chatWindow.show()
        }
        firstRunOnboarding.onOpenDashboard = { [weak self] in
            self?.openDashboard()
        }
        firstRunOnboarding.onDismiss = { [weak self] in
            self?.rebuildMenu()
        }

        memoryGovernor.autoPurge = autoPurgeEnabled
        memoryGovernor.onUpdate = { [weak self] used, total, pressure in
            guard let self = self else { return }
            self.memoryUsedGB = used
            self.memoryTotalGB = total
            self.memoryPressure = pressure
            self.rebuildMenu()
        }
        memoryGovernor.onCritical = { [weak self] in
            guard let self = self else { return }
            self.autoPurgeVRAM()
            self.unloadOptionalModels()
        }
        memoryGovernor.start()
        refreshTelemetry()
        // Scan for cached models on startup so the Model submenu is populated
        scanModels()
        scanHuggingFaceCache()

        if UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false {
            PiperTTSClient.shared.warmup()
        }

        controlCenter.onVoiceToggle = { [weak self] enabled in
            guard let self = self else { return }
            UserDefaults.standard.set(enabled, forKey: "BadAppleVoiceEnabled")
            self.voiceHost.setEnabled(enabled)
            self.rebuildMenu()
        }

        statusWindow.onRefresh = { [weak self] in
            guard let self = self else {
                return BadAppleStatusWindow.Snapshot(
                    headline: "Bad Apple is checking...",
                    brain: "not loaded",
                    memory: "calibrating...",
                    p2p: "off",
                    mcp: "off",
                    lastProblem: ""
                )
            }
            return self.makeStatusSnapshot()
        }

        NSApp.servicesProvider = self

        // Pause voice when the screen locks or the Mac sleeps; resume on unlock/wake.
        let dnc = DistributedNotificationCenter.default
        dnc.addObserver(self, selector: #selector(screenLocked), name: Notification.Name("com.apple.screenIsLocked"), object: nil)
        dnc.addObserver(self, selector: #selector(screenUnlocked), name: Notification.Name("com.apple.screenIsUnlocked"), object: nil)
        let workspace = NSWorkspace.shared.notificationCenter
        workspace.addObserver(self, selector: #selector(willSleep), name: NSWorkspace.willSleepNotification, object: nil)
        workspace.addObserver(self, selector: #selector(didWake), name: NSWorkspace.didWakeNotification, object: nil)
        showOnboardingIfNeeded()
    }

    /// Shows the guided first-run onboarding wizard if the user has not
    /// completed it yet. Uses a short delay so the boot splash can dismiss
    /// first. If the platform has never been installed, the compact install
    /// panel is shown first so the user can run the installer; the full
    /// wizard is shown on the next launch once the platform is installed.
    func showOnboardingIfNeeded() {
        if UserDefaults.standard.bool(forKey: "BadAppleOnboarded") {
            return
        }
        let installed = FileManager.default.fileExists(atPath: "/Library/LaunchDaemons/com.badapple.mlx.plist")
            && FileManager.default.fileExists(atPath: "/Library/LaunchDaemons/com.badapple.gatekeeper.plist")
            && FileManager.default.fileExists(atPath: "/var/run/badapple/substrate.sock")
            && FileManager.default.fileExists(atPath: BadAppleBrain.directSocket)

        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self = self else { return }
            if installed {
                self.onboardingWindow.show()
            } else {
                self.firstRunOnboarding.showIfNeeded()
            }
        }
    }

    @objc private func screenLocked() {
        badAppleVoiceLog("screen locked; pausing voice")
        voiceHost.setEnabled(false)
    }

    @objc private func screenUnlocked() {
        guard voiceEnabled else { return }
        badAppleVoiceLog("screen unlocked; resuming voice")
        voiceHost.setEnabled(true)
    }

    @objc private func willSleep() {
        badAppleVoiceLog("Mac will sleep; pausing voice")
        voiceHost.setEnabled(false)
    }

    @objc private func didWake() {
        guard voiceEnabled else { return }
        badAppleVoiceLog("Mac woke; resuming voice")
        voiceHost.setEnabled(true)
    }

    private func findBadAppleRepoRoot() -> String? {
        if let envRoot = ProcessInfo.processInfo.environment["BADAPPLE_ROOT"],
           FileManager.default.fileExists(atPath: "\(envRoot)/src/platform/apple_bridge/install_badapple_platform.sh") {
            return envRoot
        }

        let home = FileManager.default.homeDirectoryForCurrentUser
        let badAppleDir = home.appendingPathComponent(".bad_apple")
        if let versions = try? FileManager.default.contentsOfDirectory(at: badAppleDir, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) {
            for versionDir in versions.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
                let root = versionDir.appendingPathComponent("bad_apple").path
                if FileManager.default.fileExists(atPath: "\(root)/src/platform/apple_bridge/install_badapple_platform.sh") {
                    return root
                }
            }
        }

        // If the app bundle sits next to a source checkout, use that.
        let bundleParent = (Bundle.main.bundlePath as NSString).deletingLastPathComponent
        let candidate = (bundleParent as NSString).deletingLastPathComponent
        if FileManager.default.fileExists(atPath: "\(candidate)/src/platform/apple_bridge/install_badapple_platform.sh") {
            return candidate
        }

        return nil
    }

    private func runFirstRunInstaller() {
        guard let script = Bundle.main.path(forResource: "install_badapple", ofType: "sh"),
              !script.isEmpty else {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Installer not found"
            alert.informativeText = "The Bad Apple installer is missing from the app bundle. Please run the Install Bad Apple command from the Bad Apple source folder."
            alert.alertStyle = .warning
            alert.runModal()
            return
        }

        let repoRoot = findBadAppleRepoRoot()
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", script]
            var environment = ProcessInfo.processInfo.environment
            if let repoRoot = repoRoot {
                environment["BADAPPLE_ROOT"] = repoRoot
            }
            process.environment = environment

            do {
                try process.run()
                process.waitUntilExit()
                DispatchQueue.main.async {
                    if process.terminationStatus == 0 {
                        UserDefaults.standard.set(true, forKey: "BadAppleFirstRunOnboarded")
                        UserDefaults.standard.set(true, forKey: "BadApplePlatformInstalled")
                        self?.refreshTelemetry()
                    } else {
                        self?.lastError = "The Bad Apple installer exited with code \(process.terminationStatus)."
                        self?.rebuildMenu()
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self?.lastError = "Could not run the Bad Apple installer: \(error.localizedDescription)"
                    self?.rebuildMenu()
                }
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        voiceHost.setEnabled(false)
        timer?.invalidate()
        memoryGovernor.stop()
        stopAquaHelper()
    }

    private func registerSMAppService() {
        Task {
            let service = SMAppService.mainApp
            let status = service.status
            badAppleVoiceLog("SMAppService status: \(status)")
            if status == .enabled || status == .notRegistered {
                do {
                    try service.register()
                    badAppleVoiceLog("SMAppService registered successfully")
                } catch {
                    badAppleVoiceLog("SMAppService register error: \(error.localizedDescription)")
                }
            }
        }
    }

    private func removeLegacyLaunchAgent() {
        let label = "com.badapple.menubar"
        let plistPath = (NSHomeDirectory() as NSString).appendingPathComponent("Library/LaunchAgents/\(label).plist")
        guard FileManager.default.fileExists(atPath: plistPath) else { return }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["bootout", "gui/\(getuid())/\(label)"]
        do {
            try process.run()
            process.waitUntilExit()
            try? FileManager.default.removeItem(atPath: plistPath)
            badAppleVoiceLog("Removed legacy LaunchAgent")
        } catch {
            badAppleVoiceLog("Could not remove legacy LaunchAgent: \(error.localizedDescription)")
        }
    }

    private func refreshTelemetry() {
        if BadAppleEngine.shared.isLoaded {
            let ready = BadAppleEngine.shared.isLoaded
            let status: [String: Any] = [
                "mode": ready ? "READY" : "LOADING",
                "model_id": BadAppleEngine.shared.modelId,
                "active_models": ready ? ["main_9b"] : [],
                "native_engine": true,
                "memory_gb": BadAppleEngine.shared.memoryUsageGB,
                "tokens_per_second": BadAppleEngine.shared.lastTokensPerSecond,
                "fast_tier": true,
            ]
            activeModels = status["active_models"] as? [String] ?? []
            lastRuntimeStatus = status
            lastRuntimeReachable = true
            updateStatusIcon()
            rebuildMenu()
            splash.update(status: status)
            chatWindow.tierName = ready ? "9B Native" : "Loading"
            Task {
                let tasks = await BadAppleEngine.shared.listAgentTasks()
                await MainActor.run {
                    self.nativeAgentTasks = tasks
                }
            }
            return
        }
        Task {
            do {
                let output = try await runBadAppleCLI(prompt: "runtime status", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
                if let data = output.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    await MainActor.run {
                        if let active = json["active_models"] as? [String] {
                            self.activeModels = active
                        }
                        var status = json
                        if let modelStatus = json["model_status"] as? [String: [String: Any]] {
                            let allReady = modelStatus.values.allSatisfy { $0["status"] as? String == "ready" }
                            let hasModels = !(json["active_model_ids"] as? [String] ?? []).isEmpty
                            status["mode"] = (allReady && hasModels) ? "READY" : "STARTING"
                        } else if !(json["active_model_ids"] as? [String] ?? []).isEmpty {
                            status["mode"] = "READY"
                        } else {
                            status["mode"] = "STARTING"
                        }
                        self.lastRuntimeStatus = status
                        self.lastRuntimeReachable = true
                        self.updateStatusIcon()
                        self.rebuildMenu()
                        self.splash.update(status: json)
                        // Update chat window tier badge
                        let tier = (json["fast_tier"] as? Bool == true) ? "0.5B Fast" : "9B"
                        self.chatWindow.tierName = tier
                        // Update vision availability for image drop/paste support
                        let models = json["active_models"] as? [String] ?? self.activeModels
                        self.chatWindow.visionAvailable = models.contains { $0.contains("vision") }
                    }
                } else {
                    await MainActor.run {
                        self.lastRuntimeReachable = false
                        self.updateStatusIcon()
                        self.splash.setOffline()
                    }
                }
            } catch {
                await MainActor.run {
                    self.lastRuntimeReachable = false
                    self.updateStatusIcon()
                    self.splash.setOffline()
                }
            }
        }
    }

    private func autoPurgeVRAM() {
        badAppleVoiceLog("Memory critical: auto-purging VRAM")
        BadAppleEngine.shared.clearCache()
    }

    private func unloadOptionalModels() {
        badAppleVoiceLog("Memory critical: unloading optional models")
        BadAppleEngine.shared.clearCache()
    }

    @objc private func purgeVRAM() {
        BadAppleEngine.shared.clearCache()
        refreshTelemetry()
    }

    @objc private func unloadModels() {
        BadAppleEngine.shared.clearCache()
        refreshTelemetry()
    }

    @objc private func toggleAutoPurge() {
        autoPurgeEnabled.toggle()
        memoryGovernor.autoPurge = autoPurgeEnabled
        rebuildMenu()
    }

    @objc private func openDashboard() {
        if let url = URL(string: "http://127.0.0.1:8787") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func showStatus() {
        statusWindow.update(snapshot: makeStatusSnapshot())
        statusWindow.makeKeyAndOrderFront(nil)
    }

    private func makeStatusSnapshot() -> BadAppleStatusWindow.Snapshot {
        let runtime = runtimeState
        let reachable = lastRuntimeReachable
        let mode = (lastRuntimeStatus["mode"] as? String ?? runtime["mode"] as? String ?? "UNKNOWN").uppercased()
        let mainLoaded = lastRuntimeStatus["main_model_loaded"] as? Bool ?? runtime["main_model_loaded"] as? Bool ?? false
        let killed = lastRuntimeStatus["killed"] as? Bool ?? runtime["killed"] as? Bool ?? false
        let safeReason = lastRuntimeStatus["safe_mode_reason"] as? String ?? runtime["safe_mode_reason"] as? String

        let headline: String
        if !reachable {
            headline = "Bad Apple needs to be installed or restarted"
        } else if killed || mode == "SAFE_MODE" || safeReason != nil {
            headline = "Bad Apple is paused"
        } else if mainLoaded && mode == "READY" {
            headline = "Bad Apple is ready"
        } else if mode == "STARTING" || !mainLoaded {
            headline = "Bad Apple is loading the model..."
        } else {
            headline = "Bad Apple is waking up..."
        }

        let brain: String
        let active = activeModels.isEmpty ? ["none"] : activeModels
        brain = active.joined(separator: ", ")

        let freeGB = max(0, memoryTotalGB - memoryUsedGB)
        let memory = memoryTotalGB > 0
            ? String(format: "%.1f GB free of %.1f GB", freeGB, memoryTotalGB)
            : "calibrating..."

        let p2pEnabled = runtime["p2p_enabled"] as? Bool ?? false
        let peers = (runtime["p2p_peers"] as? [String] ?? []).count
        let p2p = p2pEnabled ? (peers == 0 ? "P2P: on, no peers nearby" : "P2P: \(peers) peer\(peers == 1 ? "" : "s") nearby") : "off"

        let mcpSocket = (lastRuntimeStatus["mcp_socket"] as? String ?? runtime["mcp_socket"] as? String ?? "").trimmingCharacters(in: .whitespaces)
        let mcp = mcpSocket.isEmpty ? "off" : "ready"

        let lastProblem: String
        if let error = lastError, !error.isEmpty {
            lastProblem = error
        } else if !reachable {
            lastProblem = "The background helper is not running. If you have not installed it yet, choose Install Now from the first-run welcome, or open Status for help."
        } else {
            lastProblem = "None"
        }

        return BadAppleStatusWindow.Snapshot(
            headline: headline,
            brain: brain,
            memory: memory,
            p2p: p2p,
            mcp: mcp,
            lastProblem: lastProblem
        )
    }

    @objc private func viewWorkingMemory() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "read working memory", socketPath: BadAppleBrain.deepSocket, maxTokens: 120)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    @objc private func clearWorkingMemory() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "clear working memory", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    @objc private func listShortcuts() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "list shortcuts", socketPath: BadAppleBrain.deepSocket, maxTokens: 120)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    @objc private func runShortcutPrompt() {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Run macOS Shortcut"
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 340, height: 24))
        alert.accessoryView = textField
        alert.addButton(withTitle: "Run")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            let name = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !name.isEmpty {
                Task {
                    _ = try? await runBadAppleCLI(prompt: "run shortcut \(name)", socketPath: BadAppleBrain.deepSocket, maxTokens: 120)
                    await MainActor.run { self.rebuildMenu() }
                }
            }
        }
    }

    @objc private func toggleFastTier() {
        fastTierOnly.toggle()
        Task {
            _ = try? await runBadAppleCLI(prompt: fastTierOnly ? "fast tier on" : "fast tier off", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    /// Push the persisted UserDefaults autopilot state to the daemon so the
    /// assistant's reported state matches the UI toggle after restarts.
    private func syncAutopilotToDaemon() {
        Task {
            _ = try? await runBadAppleCLI(prompt: autopilotEnabled ? "autopilot on" : "autopilot off", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
        }
    }

    @objc private func toggleAutopilot() {
        autopilotEnabled.toggle()
        Task {
            _ = try? await runBadAppleCLI(prompt: autopilotEnabled ? "autopilot on" : "autopilot off", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    @objc private func toggleFocus() {
        focusEnabled.toggle()
        applyFocusMode(focusEnabled)
        rebuildMenu()
    }

    private func applyFocusMode(_ enabled: Bool) {
        let result = BadAppleFocusManager.setEnabled(enabled)
        badAppleVoiceLog(result)
        if result.hasPrefix("Focus error") {
            // Fallback: try a user-created Shortcuts action. macOS hides the
            // DoNotDisturb database from TCC if Full Disk Access is missing.
            let shortcutName = enabled ? "Bad Apple Focus On" : "Bad Apple Focus Off"
            if runFocusShortcut(named: shortcutName) {
                badAppleVoiceLog("Focus toggled via Shortcuts: \(shortcutName)")
                return
            }
            focusEnabled = !enabled
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Focus Mode"
            alert.informativeText = """
            \(result)

            Option 1: Grant Bad Apple Full Disk Access in System Settings > Privacy & Security > Full Disk Access, then toggle again.

            Option 2: Create a Shortcut named "\(shortcutName)" with a "Set Focus" action (Do Not Disturb), and Bad Apple will run it for you.
            """
            alert.alertStyle = .warning
            alert.runModal()
        } else if !result.hasPrefix("Focus:") {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Focus Mode"
            alert.informativeText = result
            alert.alertStyle = .informational
            alert.runModal()
        }
    }

    private func runFocusShortcut(named: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/shortcuts")
        process.arguments = ["run", named]
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            badAppleVoiceLog("Focus shortcut '\(named)' error: \(error.localizedDescription)")
            return false
        }
    }

    @objc private func toggleAmbient() {
        let ambientRunning = runtimeState["ambient_running"] as? Bool ?? false
        Task { @MainActor in
            do {
                _ = try await runBadAppleCLI(
                    prompt: ambientRunning ? "stop ambient" : "start ambient",
                    socketPath: BadAppleBrain.deepSocket,
                    maxTokens: 32
                )
                self.rebuildMenu()
            } catch {
                self.lastError = error.localizedDescription
                self.rebuildMenu()
            }
        }
    }

    @objc private func setWorkspacePrompt() {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Set Workspace"
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 400, height: 24))
        textField.placeholderString = "/path/to/workspace"
        alert.accessoryView = textField
        alert.addButton(withTitle: "Set")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let path = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return }
        Task { @MainActor in
            do {
                let output = try await runBadAppleCLI(
                    prompt: "set workspace to \(path)",
                    socketPath: BadAppleBrain.deepSocket,
                    maxTokens: 80
                )
                let info = NSAlert()
                info.messageText = "Workspace Set"
                info.informativeText = output
                info.alertStyle = .informational
                _ = info.runModal()
                self.rebuildMenu()
            } catch {
                self.lastError = error.localizedDescription
                self.rebuildMenu()
            }
        }
    }

    @objc private func openCurrentWorkspace() {
        guard let path = runtimeState["workspace"] as? String, !path.isEmpty else {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "No Workspace Set"
            alert.informativeText = "There is no current workspace to open."
            alert.alertStyle = .warning
            _ = alert.runModal()
            return
        }
        let url = URL(fileURLWithPath: path)
        if !NSWorkspace.shared.open(url) {
            NSApp.activate(ignoringOtherApps: true)
            let alert = NSAlert()
            alert.messageText = "Could Not Open Workspace"
            alert.informativeText = "macOS could not open \"\(path)\"."
            alert.alertStyle = .critical
            _ = alert.runModal()
        }
    }

    @objc private func toggleP2P() {
        let p2pEnabled = runtimeState["p2p_enabled"] as? Bool ?? false
        Task { @MainActor in
            do {
                _ = try await runBadAppleCLI(
                    prompt: p2pEnabled ? "p2p off" : "p2p on",
                    socketPath: BadAppleBrain.deepSocket,
                    maxTokens: 32
                )
                self.rebuildMenu()
            } catch {
                self.lastError = error.localizedDescription
                self.rebuildMenu()
            }
        }
    }

    private func submitVoicePrompt(_ prompt: String) {
        guard !isSubmittingVoicePrompt else {
            badAppleVoiceLog("submitVoicePrompt: ignoring duplicate submission")
            return
        }
        isSubmittingVoicePrompt = true
        badAppleVoiceLog("submitVoicePrompt: \(prompt)")
        lastPrompt = prompt
        lastError = nil
        streamedTokenCount = 0
        // Reset the streaming TTS pipeline and clear the queue so stale audio
        // from the previous response is dropped before the new one starts.
        voiceHost.resetStreamingTTS()
        voiceStreamingTTSActive = voiceHost.streamingTTSAvailable
        rebuildMenu()

        // Safety net: if the cognitive substrate or TTS never calls back,
        // clear the lock and resume listening so the voice host doesn't die.
        voicePromptTimeout?.cancel()
        let timeout = DispatchWorkItem { [weak self] in
            guard let self = self, self.isSubmittingVoicePrompt else { return }
            badAppleVoiceLog("submitVoicePrompt: overall timeout, resuming listening")
            self.isSubmittingVoicePrompt = false
            self.lastError = "response took too long"
            self.voiceHost.speak("Sorry, that took too long. Try again.")
            self.rebuildMenu()
        }
        voicePromptTimeout = timeout
        DispatchQueue.main.asyncAfter(deadline: .now() + 45.0, execute: timeout)

        // Voice mode switch commands are handled without a daemon call.
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePrompt = trimmed

        // Fast local action resolver for common voice commands (open workspace,
        // launch app, create directory).  This keeps the assistant responsive
        // even when the cognitive substrate is slow or unavailable.
        if let local = BadAppleActionResolver.resolve(effectivePrompt) {
            badAppleVoiceLog("submitVoicePrompt resolved local action: \(local)")
            actionExecutor.confirmAndExecute(local)
            isSubmittingVoicePrompt = false
            voiceHost.resumeAfterFailure()
            return
        }

        // Fallback: ask the on-device daemon via the bundled badapple CLI.
        // The CLI streams token chunks as they are generated; we feed each
        // chunk to the TTS queue immediately so the first sentence starts
        // playing while the model is still finishing the rest of the response.

        // NATIVE ENGINE: If the Swift MLX engine is loaded, route directly
        // through it — no subprocess, no daemon, no Python. This cuts
        // 200-500ms of process overhead per query.
        if BadAppleEngine.shared.isLoaded {
            badAppleVoiceLog("voice: using native Swift engine (no subprocess)")
            BadAppleEngine.shared.generateStreaming(
                prompt: effectivePrompt,
                voiceMode: true,
                maxTokens: 500
            ) { chunk in
                self.streamedTokenCount += chunk.count
                if self.voiceStreamingTTSActive {
                    self.voiceHost.speakStreamingChunk(chunk)
                }
                self.rebuildMenu()
            } onComplete: { finalText in
                self.voicePromptTimeout?.cancel()
                self.voicePromptTimeout = nil
                self.isSubmittingVoicePrompt = false
                if self.voiceStreamingTTSActive {
                    self.voiceHost.flushStreamingTTS()
                }
                self.completeVoiceResponse(finalText, streamed: self.voiceStreamingTTSActive)
                self.rebuildMenu()
            } onError: { error in
                self.voicePromptTimeout?.cancel()
                self.voicePromptTimeout = nil
                self.isSubmittingVoicePrompt = false
                self.lastError = error
                self.voiceHost.speak("Sorry, something went wrong. \(error)")
                self.rebuildMenu()
            }
            return
        }

        var extraArgs: [String] = []
        if selectedPersona != "default" { extraArgs += ["--persona", selectedPersona] }
        if roastEnabled { extraArgs += ["--roast"] }

        let socket = BadAppleBrain.deepSocket
        let maxTokens = 500

        // Minimal environment for the voice CLI process: do not inherit the
        // full parent environment (which can carry attacker-set overrides) and
        // keep the spawn as lightweight as possible to cut per-query latency.
        var voiceEnv: [String: String] = [
            "BADAPPLE_SOCKET_PATH": socket,
            "BADAPPLE_SLICKS_KEY_PATH": BadAppleBrain.keyPath,
            "BADAPPLE_STREAM_JSON": "1",
            "BADAPPLE_VOICE": "1",
        ]
        // Short simple queries skip the cognitive governor for faster response.
        if effectivePrompt.count < 60 && isSimpleVoiceQuery(effectivePrompt) {
            voiceEnv["BADAPPLE_FAST_TIER"] = "1"
            voiceEnv["BADAPPLE_COGNITIVE"] = "0"
            badAppleVoiceLog("voice fast path: short prompt, skipping cognitive governor")
        }

        Task {
            do {
                let finalText = try await runBadAppleCLIStreaming(prompt: effectivePrompt, socketPath: socket, maxTokens: maxTokens, extraArgs: extraArgs, timeout: 120.0, extraEnv: voiceEnv, minimalEnv: true) { chunk in
                    DispatchQueue.main.async {
                        self.streamedTokenCount += chunk.count
                        if self.voiceStreamingTTSActive {
                            self.voiceHost.speakStreamingChunk(chunk)
                        }
                        self.rebuildMenu()
                    }
                }
                await MainActor.run {
                    self.voicePromptTimeout?.cancel()
                    self.voicePromptTimeout = nil
                    if self.voiceStreamingTTSActive {
                        self.voiceHost.flushStreamingTTS()
                    }
                    self.completeVoiceResponse(finalText, streamed: self.voiceStreamingTTSActive)
                    self.isSubmittingVoicePrompt = false
                }
            } catch {
                badAppleVoiceLog("submitVoicePrompt error: \(error)")
                await MainActor.run {
                    self.voicePromptTimeout?.cancel()
                    self.voicePromptTimeout = nil
                    self.lastError = error.localizedDescription
                    // Drop any partially-streamed audio before resuming so stale
                    // sentences do not play over the restarted listening session.
                    self.voiceHost.stopAllAudio()
                    self.voiceHost.resumeAfterFailure()
                    self.rebuildMenu()
                    self.isSubmittingVoicePrompt = false
                }
            }
        }
    }

    /// Heuristic for routing short voice queries to the fast 0.5B tier without
    /// paying the cognitive governor's classification overhead.  Matches
    /// greetings, identity/time questions, confirmations, and simple arithmetic.
    private func isSimpleVoiceQuery(_ prompt: String) -> Bool {
        let lower = prompt.lowercased()
        let simplePatterns = ["what time", "who are you", "what is", "hello", "hey", "hi ", "thanks", "thank you", "yes", "no", "ok"]
        if simplePatterns.contains(where: { lower.contains($0) }) {
            return true
        }
        // Simple arithmetic like "2 + 2" or "12 * 9".
        let mathPattern = try? NSRegularExpression(pattern: "\\d+\\s*[+\\-*/]\\s*\\d+")
        if let mathPattern = mathPattern {
            let range = NSRange(prompt.startIndex..<prompt.endIndex, in: prompt)
            if mathPattern.firstMatch(in: prompt, range: range) != nil {
                return true
            }
        }
        return false
    }

    /// Touch the bundled badapple CLI binary so the OS pages it into the file
    /// cache ahead of the actual spawn.  Called when the wake word is detected
    /// so the process launch is instant by the time the user finishes speaking.
    private func prewarmVoiceCLI() {
        let binary = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Helpers")
            .appendingPathComponent("badapple")
        voiceQueue.async {
            guard FileManager.default.fileExists(atPath: binary.path) else { return }
            if let handle = try? FileHandle(forReadingFrom: binary) {
                _ = handle.readData(ofLength: 64)
                try? handle.close()
            }
        }
    }

    @objc private func showChatHistory() {
        chatHistoryWindow.show()
    }

    @objc private func showChatWindow() {
        chatWindow.show()
    }

    @objc private func showSettings() {
        settingsWindow.show()
    }

    @objc private func newChat() {
        BadAppleEngine.shared.resetConversation()
        lastPrompt = "new chat"
        lastError = nil
        streamedTokenCount = 0
        voiceHost.speak("Started a new chat.")
        rebuildMenu()
    }

    @objc private func toggleRoast() {
        roastEnabled.toggle()
        _ = BadAppleEngine.shared.switchPersona(roastEnabled ? "drill" : selectedPersona)
        rebuildMenu()
    }

    @objc private func selectPersona(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        selectedPersona = name
        if !roastEnabled {
            _ = BadAppleEngine.shared.switchPersona(name)
        }
        let displayNames: [String: String] = ["default": "Default", "wicket": "Wicket", "genz": "Gen Z", "drill": "Drill", "midwest": "Midwest Aunt"]
        chatWindow.personaName = displayNames[name] ?? name.capitalized
        rebuildMenu()
    }

    /// Scans `~/.cache/huggingface/hub/` for directories starting with `models--`
    /// and extracts the model name (format: `models--org--model-name` → `org/model-name`).
    @objc private func scanHuggingFaceCache() {
        let home = NSHomeDirectory()
        let hubDir = "\(home)/.cache/huggingface/hub"
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: hubDir) else {
            badAppleVoiceLog("scanHuggingFaceCache: hub dir not found at \(hubDir)")
            return
        }
        var names: [String] = []
        for entry in entries where entry.hasPrefix("models--") {
            // Format: models--org--model-name (org and model can contain -- segments)
            let stripped = String(entry.dropFirst("models--".count))
            // The first -- separates org from model name
            if let dashRange = stripped.range(of: "--") {
                let org = String(stripped[..<dashRange.lowerBound])
                let modelName = String(stripped[dashRange.upperBound...])
                if !org.isEmpty && !modelName.isEmpty {
                    names.append("\(org)/\(modelName)")
                }
            }
        }
        names.sort()
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let added = Set(names).subtracting(Set(self.cachedHFModelNames))
            self.cachedHFModelNames = names
            if !added.isEmpty {
                badAppleVoiceLog("scanHuggingFaceCache: found \(names.count) models, \(added.count) new")
                self.rebuildMenu()
            }
        }
    }

    /// Determines the currently active model from multiple sources.
    private func currentActiveModel() -> String {
        // 1. Check runtime status (populated by refreshTelemetry)
        if let modelId = lastRuntimeStatus["model_id"] as? String, !modelId.isEmpty {
            return modelId
        }
        // 2. Check BADAPPLE_MAIN_MODEL env var
        if let envModel = ProcessInfo.processInfo.environment["BADAPPLE_MAIN_MODEL"],
           !envModel.isEmpty {
            return envModel
        }
        // 3. Read from model_config_hash.json in /var/lib/bad_apple/
        let hashPath = "/var/lib/bad_apple/model_config_hash.json"
        if let data = try? Data(contentsOf: URL(fileURLWithPath: hashPath)),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let model = json["model"] as? String, !model.isEmpty {
            return model
        }
        // 4. Fallback to the known default
        return "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    }

    /// Opens the model management dashboard at http://127.0.0.1:8787/models
    @objc private func openModelManager() {
        if let url = URL(string: "http://127.0.0.1:8787/models") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc private func scanModels() {
        Task {
            do {
                let output = try await runBadAppleCLI(args: ["model", "scan"])
                if let data = output.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let models = json["models"] as? [[String: Any]] {
                    await MainActor.run {
                        self.cachedModelList = models
                        self.scanHuggingFaceCache()
                        self.rebuildMenu()
                    }
                }
            } catch {
                badAppleVoiceLog("scanModels error: \(error)")
            }
        }
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let repo = sender.representedObject as? String else { return }
        let modelLabel = repo.split(separator: "/").last.map(String.init) ?? repo
        // Brief confirmation shown immediately while the switch runs
        lastPrompt = "Switching to \(modelLabel)..."
        rebuildMenu()
        Task {
            do {
                let output = try await runBadAppleCLI(args: ["model", "use", repo])
                let confirmText: String
                if let data = output.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let text = json["text"] as? String, !text.isEmpty {
                    confirmText = text
                } else {
                    confirmText = "Switched to \(repo)."
                }
                await MainActor.run {
                    self.lastError = nil
                    self.lastSpoken = confirmText
                    self.lastPrompt = "Switched to \(modelLabel)"
                    self.scanHuggingFaceCache()
                    self.rebuildMenu()
                    badAppleVoiceLog("Model switched: \(confirmText)")
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.lastPrompt = ""
                    self.rebuildMenu()
                }
            }
        }
    }

    @objc private func runBenchmark() {
        Task {
            do {
                let output = try await runBadAppleCLI(args: ["--benchmark"])
                await MainActor.run {
                    NSApp.activate(ignoringOtherApps: true)
                    let alert = NSAlert()
                    alert.messageText = "Bad Apple Benchmark"
                    alert.informativeText = output
                    alert.alertStyle = .informational
                    alert.runModal()
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.rebuildMenu()
                }
            }
        }
    }

    private func runBadAppleCLIStreaming(
        prompt: String,
        socketPath: String,
        maxTokens: Int,
        extraArgs: [String] = [],
        timeout: TimeInterval = 120.0,
        extraEnv: [String: String] = [:],
        minimalEnv: Bool = false,
        onChunk: @escaping (String) -> Void
    ) async throws -> String {
        let binary = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Helpers")
            .appendingPathComponent("badapple")
        let command = binary.path
        guard FileManager.default.fileExists(atPath: command) else {
            throw BadAppleMenuBarError("The badapple helper binary is missing from the app bundle.")
        }

        return try await withCheckedThrowingContinuation { continuation in
            // Spawn on the dedicated serial voice queue so the process launch
            // is not delayed by unrelated global-queue work.
            voiceQueue.async {
                let process = Process()
                let outputPipe = Pipe()
                process.executableURL = binary
                process.arguments = extraArgs + ["--max-tokens", String(maxTokens), prompt]
                process.standardOutput = outputPipe
                process.standardError = outputPipe
                var environment: [String: String]
                if minimalEnv {
                    // SECURITY/PERF: Build a minimal allow-list environment for
                    // the latency-sensitive voice path instead of inheriting the
                    // full parent environment (which can carry attacker-set
                    // overrides and adds fork/exec cost).  extraEnv supplies the
                    // voice-specific flags (socket, SLICKS key, stream, fast
                    // tier, etc.).
                    environment = [
                        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
                        "HOME": ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(),
                        "LANG": ProcessInfo.processInfo.environment["LANG"] ?? "en_US.UTF-8",
                        "BADAPPLE_SOCKET_PATH": socketPath,
                        "BADAPPLE_SLICKS_KEY_PATH": BadAppleBrain.keyPath,
                        "BADAPPLE_STREAM_JSON": "1",
                        "BADAPPLE_VOICE": "1",
                    ]
                } else {
                    environment = ProcessInfo.processInfo.environment
                    environment["BADAPPLE_SOCKET_PATH"] = socketPath
                    environment["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
                    environment["BADAPPLE_STREAM_JSON"] = "1"
                    environment["BADAPPLE_VOICE"] = "1"
                }
                for (key, value) in extraEnv {
                    environment[key] = value
                }
                process.environment = environment

                let sync = NSLock()
                var finished = false

                var buffer = ""
                var fullText = ""

                let timeoutTimer = DispatchSource.makeTimerSource(queue: self.voiceQueue)
                timeoutTimer.schedule(deadline: .now() + timeout)
                timeoutTimer.setEventHandler { [weak process] in
                    badAppleVoiceLog("runBadAppleCLIStreaming: timeout, terminating")
                    process?.terminate()
                }
                timeoutTimer.resume()

                func finish(result: Result<String, Error>) {
                    sync.lock()
                    guard !finished else { sync.unlock(); return }
                    finished = true
                    timeoutTimer.cancel()
                    outputPipe.fileHandleForReading.readabilityHandler = nil
                    sync.unlock()
                    switch result {
                    case .success(let text):
                        continuation.resume(returning: text)
                    case .failure(let error):
                        continuation.resume(throwing: error)
                    }
                }

                outputPipe.fileHandleForReading.readabilityHandler = { handle in
                    guard let str = String(data: handle.availableData, encoding: .utf8) else { return }
                    sync.lock()
                    buffer += str
                    var localFullText = ""
                    while let newlineIndex = buffer.firstIndex(of: "\n") {
                        let line = String(buffer[..<newlineIndex])
                        buffer = String(buffer[buffer.index(after: newlineIndex)...])
                        guard let data = line.data(using: .utf8) else { continue }
                        do {
                            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                                if let type = json["type"] as? String, type == "token", let text = json["text"] as? String {
                                    fullText += text
                                    localFullText += text
                                } else if let type = json["type"] as? String, type == "done", let text = json["text"] as? String {
                                    fullText = text
                                }
                            }
                        } catch {
                            badAppleVoiceLog("runBadAppleCLIStreaming: ignoring non-JSON line: \(line.prefix(100))")
                        }
                    }
                    sync.unlock()
                    if !localFullText.isEmpty {
                        onChunk(localFullText)
                    }
                }

                process.terminationHandler = { _ in
                    sync.lock()
                    let code = process.terminationStatus
                    let text = fullText
                    sync.unlock()
                    if code != 0, text.isEmpty {
                        finish(result: .failure(BadAppleMenuBarError("The Bad Apple helper exited with code \(code).")))
                    } else {
                        finish(result: .success(text))
                    }
                }

                do {
                    try process.run()
                } catch {
                    finish(result: .failure(error))
                }
            }
        }
    }

    private func runBadAppleCLI(prompt: String, socketPath: String, maxTokens: Int, timeout: TimeInterval = 120.0, extraEnv: [String: String] = [:]) async throws -> String {
        let binary = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Helpers")
            .appendingPathComponent("badapple")
        let command = binary.path
        guard FileManager.default.fileExists(atPath: command) else {
            throw BadAppleMenuBarError("The badapple helper binary is missing from the app bundle.")
        }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let outputPipe = Pipe()
                process.executableURL = binary
                process.arguments = ["--max-tokens", String(maxTokens), prompt]
                process.standardOutput = outputPipe
                process.standardError = outputPipe
                var environment = ProcessInfo.processInfo.environment
                environment["BADAPPLE_SOCKET_PATH"] = socketPath
                environment["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
                environment["BADAPPLE_VOICE"] = "1"
                for (k, v) in extraEnv {
                    environment[k] = v
                }
                process.environment = environment

                var timeoutTimer: DispatchSourceTimer?
                let timeoutQueue = DispatchQueue.global(qos: .userInitiated)
                timeoutTimer = DispatchSource.makeTimerSource(queue: timeoutQueue)
                timeoutTimer?.schedule(deadline: .now() + timeout)
                timeoutTimer?.setEventHandler { [weak process] in
                    badAppleVoiceLog("runBadAppleCLI: timeout, terminating")
                    process?.terminate()
                }
                timeoutTimer?.resume()

                do {
                    try process.run()
                    // Read the pipe concurrently to avoid a deadlock when the
                    // child writes more data than the pipe buffer can hold.
                    // waitUntilExit() before readDataToEndOfFile() would block
                    // the child on write() and hang forever.
                    let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    timeoutTimer?.cancel()
                    timeoutTimer = nil
                    let output = String(data: data, encoding: .utf8) ?? ""
                    badAppleVoiceLog("runBadAppleCLI: exit=\(process.terminationStatus) output=\(output.prefix(200))")
                    if process.terminationStatus != 0, output.isEmpty {
                        throw BadAppleMenuBarError("The Bad Apple helper exited with code \(process.terminationStatus).")
                    }
                    continuation.resume(returning: output)
                } catch {
                    timeoutTimer?.cancel()
                    timeoutTimer = nil
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func runBadAppleCLI(args: [String]) async throws -> String {
        let binary = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Helpers")
            .appendingPathComponent("badapple")
        let command = binary.path
        guard FileManager.default.fileExists(atPath: command) else {
            throw BadAppleMenuBarError("The badapple helper binary is missing from the app bundle.")
        }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let outputPipe = Pipe()
                process.executableURL = binary
                process.arguments = args
                process.standardOutput = outputPipe
                process.standardError = outputPipe
                var environment = ProcessInfo.processInfo.environment
                environment["BADAPPLE_SOCKET_PATH"] = BadAppleBrain.deepSocket
                environment["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
                process.environment = environment

                var timeoutTimer: DispatchSourceTimer?
                let timeoutQueue = DispatchQueue.global(qos: .userInitiated)
                timeoutTimer = DispatchSource.makeTimerSource(queue: timeoutQueue)
                timeoutTimer?.schedule(deadline: .now() + 120.0)
                timeoutTimer?.setEventHandler { [weak process] in
                    badAppleVoiceLog("runBadAppleCLI benchmark: timeout, terminating")
                    process?.terminate()
                }
                timeoutTimer?.resume()

                do {
                    try process.run()
                    // Read the pipe before waitUntilExit to avoid deadlock.
                    let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    timeoutTimer?.cancel()
                    timeoutTimer = nil
                    let output = String(data: data, encoding: .utf8) ?? ""
                    badAppleVoiceLog("runBadAppleCLI: exit=\(process.terminationStatus) output=\(output.prefix(200))")
                    if process.terminationStatus != 0, output.isEmpty {
                        throw BadAppleMenuBarError("The Bad Apple helper exited with code \(process.terminationStatus).")
                    }
                    continuation.resume(returning: output)
                } catch {
                    timeoutTimer?.cancel()
                    timeoutTimer = nil
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct BadAppleMenuBarError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { self.errorDescription = message }
    }

    private func completeVoiceResponse(_ response: String, streamed: Bool = false) {
        badAppleVoiceLog("completeVoiceResponse: \(response.prefix(200)) streamed=\(streamed)")
        let parsed = BadAppleActionParser.parse(response)
        lastSpoken = parsed.spoken
        voiceHUD.updateResponse(parsed.spoken)
        badAppleVoiceLog("parsed actions: \(parsed.actions.count) error: \(parsed.error ?? "nil") spoken: \(parsed.spoken)")
        // When the spoken text was already streamed to TTS during generation we
        // do not speak it again; only parse and execute any embedded actions.
        // BUT: if streaming was active but nothing was actually spoken (e.g.,
        // the response was entirely inside a code fence or was too short for a
        // sentence boundary), fall back to speaking the full text.
        if !streamed || !voiceHost.streamTTSEmittedAny {
            voiceHost.speak(parsed.spoken)
        }
        if let parseError = parsed.error {
            actionExecutor.showParsingFailure(parseError)
        } else {
            for action in parsed.actions {
                actionExecutor.confirmAndExecute(action)
            }
        }
        rebuildMenu()
    }

    private func friendlyStatusHeadline() -> String {
        if !lastRuntimeReachable {
            return "Bad Apple needs to be installed or restarted"
        }
        let mode = (lastRuntimeStatus["mode"] as? String ?? runtimeState["mode"] as? String ?? "UNKNOWN").uppercased()
        let killed = lastRuntimeStatus["killed"] as? Bool ?? runtimeState["killed"] as? Bool ?? false
        let safeReason = lastRuntimeStatus["safe_mode_reason"] as? String ?? runtimeState["safe_mode_reason"] as? String
        let mainLoaded = lastRuntimeStatus["main_model_loaded"] as? Bool ?? runtimeState["main_model_loaded"] as? Bool ?? false
        if killed || mode == "SAFE_MODE" || safeReason != nil {
            return "Bad Apple is paused"
        }
        if mainLoaded && mode == "READY" {
            return "Bad Apple is ready"
        }
        if mode == "STARTING" || !mainLoaded {
            return "Bad Apple is loading..."
        }
        return "Bad Apple is \(mode)"
    }

    private func updateStatusIcon() {
        guard let button = statusItem?.button else { return }
        let dotColor: NSColor
        if !lastRuntimeReachable {
            dotColor = .systemRed
        } else {
            let mode = (lastRuntimeStatus["mode"] as? String ?? runtimeState["mode"] as? String ?? "UNKNOWN").uppercased()
            let killed = lastRuntimeStatus["killed"] as? Bool ?? runtimeState["killed"] as? Bool ?? false
            let safeReason = lastRuntimeStatus["safe_mode_reason"] as? String ?? runtimeState["safe_mode_reason"] as? String
            if killed || mode == "SAFE_MODE" || safeReason != nil {
                dotColor = .systemYellow
            } else if mode == "READY" {
                dotColor = .systemGreen
            } else if mode == "STARTING" {
                dotColor = .systemYellow
            } else {
                dotColor = .systemGray
            }
        }

        let title = NSMutableAttributedString()
        title.append(NSAttributedString(string: "🍎 ", attributes: [.font: NSFont.systemFont(ofSize: 13)]))
        title.append(NSAttributedString(string: "●", attributes: [
            .font: NSFont.systemFont(ofSize: 8),
            .foregroundColor: dotColor,
            .baselineOffset: -2
        ]))
        button.attributedTitle = title
        button.toolTip = friendlyStatusHeadline()
    }

    func rebuildMenu() {
        guard let menu = menu else { return }
        // Rebuilding while any menu or submenu is displayed can tear down items
        // the user is about to click and cause use-after-free crashes. Defer
        // until all menus close instead.
        if openMenuCount > 0 {
            needsMenuRebuild = true
            return
        }
        needsMenuRebuild = false
        menu.removeAllItems()
        let header = NSMenuItem(title: "Bad Apple — Local AI Operating System", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)

        let voiceStatus = NSMenuItem(title: voiceHost.state.label.truncated(to: 90), action: nil, keyEquivalent: "")
        voiceStatus.isEnabled = false
        menu.addItem(voiceStatus)

        if !lastPrompt.isEmpty {
            let lastCommand = NSMenuItem(title: "You said: \(lastPrompt.truncated(to: 60))", action: nil, keyEquivalent: "")
            lastCommand.isEnabled = false
            menu.addItem(lastCommand)
        }
        if let lastSpoken, !lastSpoken.isEmpty {
            let lastResponse = NSMenuItem(title: "Last response: \(lastSpoken.truncated(to: 60))", action: nil, keyEquivalent: "")
            lastResponse.isEnabled = false
            menu.addItem(lastResponse)
        }
        if let lastError, !lastError.isEmpty {
            let errorItem = NSMenuItem(title: "Last error: \(lastError.truncated(to: 60))", action: nil, keyEquivalent: "")
            menu.addItem(errorItem)
        }

        let engineMode = BadAppleEngine.shared.isLoaded ? "Swift (Native)" : "Loading Swift Engine"
        let mode = NSMenuItem(title: "Inference Component: Qwen 3.5 9B — \(engineMode)", action: nil, keyEquivalent: "")
        mode.isEnabled = false
        menu.addItem(mode)
        let runtime = runtimeState
        let runtimeMode = runtime["mode"] as? String ?? "UNKNOWN"
        let runtimeItem = NSMenuItem(title: "Status: \(runtimeMode)", action: nil, keyEquivalent: "")
        runtimeItem.isEnabled = false
        menu.addItem(runtimeItem)

        let usedStr = String(format: "%.1f", memoryUsedGB)
        let totalStr = String(format: "%.1f", memoryTotalGB)
        let memTitle = memoryTotalGB > 0 ? "Memory: \(usedStr) / \(totalStr) GB — \(memoryPressure)" : "Memory: calibrating..."
        let memoryItem = NSMenuItem(title: memTitle, action: nil, keyEquivalent: "")
        memoryItem.isEnabled = false
        menu.addItem(memoryItem)

        let activeTitle = "Active models: \(activeModels.joined(separator: ", "))"
        let activeItem = NSMenuItem(title: activeTitle, action: nil, keyEquivalent: "")
        activeItem.isEnabled = false
        menu.addItem(activeItem)

        let performanceMenu = NSMenu(title: "Performance")
        let purgeItem = NSMenuItem(title: "Free Up Memory", action: #selector(purgeVRAM), keyEquivalent: "")
        purgeItem.toolTip = "Release cached GPU memory and Metal allocations."
        performanceMenu.addItem(purgeItem)
        let unloadItem = NSMenuItem(title: "Close Extra AI Models", action: #selector(unloadModels), keyEquivalent: "")
        unloadItem.toolTip = "Unload vision, image, and optional models to free RAM."
        performanceMenu.addItem(unloadItem)
        let performanceFastTierItem = NSMenuItem(title: "Quick Answers Only", action: #selector(toggleFastTier), keyEquivalent: "")
        performanceFastTierItem.state = fastTierOnly ? .on : .off
        performanceFastTierItem.toolTip = "Route simple queries to the 0.5B fast model."
        performanceMenu.addItem(performanceFastTierItem)
        let autoPurgeItem = NSMenuItem(title: "Auto Free Memory When Full", action: #selector(toggleAutoPurge), keyEquivalent: "")
        autoPurgeItem.state = autoPurgeEnabled ? .on : .off
        autoPurgeItem.toolTip = "Automatically purge VRAM when memory pressure is critical."
        performanceMenu.addItem(autoPurgeItem)
        let performanceParent = NSMenuItem(title: "Performance", action: nil, keyEquivalent: "")
        performanceParent.submenu = performanceMenu
        menu.addItem(performanceParent)
        menu.addItem(NSMenuItem.separator())

        let privateMode = runtime["private_mode"] as? Bool ?? false
        let privacyMenu = NSMenu(title: "Privacy")
        let privateToggle = NSMenuItem(title: "Don't Save My Chats", action: #selector(togglePrivateMode), keyEquivalent: "")
        privateToggle.state = privateMode ? .on : .off
        privateToggle.toolTip = "Pause persistence and audit logging for this session."
        privacyMenu.addItem(privateToggle)
        let autopilotItem = NSMenuItem(title: "Auto-Run Commands", action: #selector(toggleAutopilot), keyEquivalent: "")
        autopilotItem.state = autopilotEnabled ? .on : .off
        autopilotItem.toolTip = "Allow destructive tools to run without approval prompts."
        privacyMenu.addItem(autopilotItem)
        let focusItem = NSMenuItem(title: "Focus Mode", action: #selector(toggleFocus), keyEquivalent: "")
        focusItem.state = focusEnabled ? .on : .off
        focusItem.toolTip = "Toggle Do Not Disturb / Focus while Bad Apple is active."
        privacyMenu.addItem(focusItem)
        let privacyParent = NSMenuItem(title: "Privacy", action: nil, keyEquivalent: "")
        privacyParent.submenu = privacyMenu
        menu.addItem(privacyParent)
        if runtime["killed"] as? Bool ?? false {
            let resumeItem = NSMenuItem(title: "Start Again", action: #selector(resetKillSwitch), keyEquivalent: "")
            resumeItem.toolTip = "Reset the kill switch and resume generation and tools."
            menu.addItem(resumeItem)
        } else {
            let stopItem = NSMenuItem(title: "Stop Everything", action: #selector(engageKillSwitch), keyEquivalent: "")
            stopItem.toolTip = "Cancel generation, stop ambient capture, and block tools."
            menu.addItem(stopItem)
        }
        if !lastRuntimeReachable {
            let restartDaemonItem = NSMenuItem(title: "Restart Bad Apple", action: #selector(restartDaemon), keyEquivalent: "")
            restartDaemonItem.toolTip = "Unload and reload the Bad Apple launchd daemon."
            menu.addItem(restartDaemonItem)
        }
        let statusItemMenu = NSMenuItem(title: "Status...", action: #selector(showStatus), keyEquivalent: "")
        statusItemMenu.toolTip = "Show a plain-English summary of Bad Apple’s status."
        menu.addItem(statusItemMenu)
        let systemHealthItem = NSMenuItem(title: "Health Check...", action: #selector(showSystemHealth), keyEquivalent: "")
        systemHealthItem.toolTip = "Show the full runtime status JSON for advanced troubleshooting."
        menu.addItem(systemHealthItem)
        let controlCenterItem = NSMenuItem(title: "Dashboard", action: #selector(showControlCenter), keyEquivalent: "")
        controlCenterItem.toolTip = "Open the native glass control center window."
        menu.addItem(controlCenterItem)
        let chatItem = NSMenuItem(title: "Chat", action: #selector(showChatWindow), keyEquivalent: "c")
        chatItem.toolTip = "Open the native chat window."
        menu.addItem(chatItem)
        let settingsMenuItem = NSMenuItem(title: "Settings...", action: #selector(showSettings), keyEquivalent: ",")
        settingsMenuItem.toolTip = "Open the Bad Apple settings window."
        menu.addItem(settingsMenuItem)
        let newChatItem = NSMenuItem(title: "New Chat", action: #selector(newChat), keyEquivalent: "n")
        newChatItem.toolTip = "Start a new conversation."
        menu.addItem(newChatItem)
        let chatHistoryItem = NSMenuItem(title: "Past Chats", action: #selector(showChatHistory), keyEquivalent: "h")
        chatHistoryItem.toolTip = "Show the chat history window."
        menu.addItem(chatHistoryItem)
        let toggle = NSMenuItem(title: "Voice (Hey Bad Apple)", action: #selector(toggleVoice), keyEquivalent: "v")
        toggle.state = voiceEnabled ? .on : .off
        toggle.toolTip = "Toggle the local voice wake-word listener."
        menu.addItem(toggle)

        let roastToggle = NSMenuItem(title: "Roast Mode (Savage)", action: #selector(toggleRoast), keyEquivalent: "")
        roastToggle.state = roastEnabled ? .on : .off
        roastToggle.toolTip = "Switch to the drill persona for spicy roasts."
        menu.addItem(roastToggle)

        let personaMenu = NSMenu(title: "Personality")
        for (name, display) in [("default", "Default"), ("wicket", "Wicket"), ("genz", "Gen Z"), ("drill", "Drill"), ("midwest", "Midwest Aunt")] {
            let item = NSMenuItem(title: display, action: #selector(selectPersona(_:)), keyEquivalent: "")
            item.representedObject = name
            item.state = (selectedPersona == name) ? .on : .off
            personaMenu.addItem(item)
        }
        let personaParent = NSMenuItem(title: "Personality", action: nil, keyEquivalent: "")
        personaParent.submenu = personaMenu
        menu.addItem(personaParent)

        // Model selector submenu
        let modelMenu = NSMenu(title: "AI Model")
        let activeModel = currentActiveModel()

        // Active model (checkmarked, disabled) at top
        let activeShort = activeModel.split(separator: "/").last.map(String.init) ?? activeModel
        let activeModelItem = NSMenuItem(title: "✓ \(activeShort)", action: nil, keyEquivalent: "")
        activeModelItem.isEnabled = false
        activeModelItem.toolTip = "Currently active model: \(activeModel)"
        modelMenu.addItem(activeModelItem)
        modelMenu.addItem(NSMenuItem.separator())

        // Build a merged, de-duplicated list of all known cached models.
        // Sources: cachedModelList (from `badapple model scan` JSON) and
        // cachedHFModelNames (from scanning ~/.cache/huggingface/hub/).
        var seenRepos = Set<String>()
        var modelEntries: [(repo: String, label: String, status: String)] = []

        // 1. Entries from the daemon's model scan (may include status info)
        for m in cachedModelList.prefix(30) {
            let id = m["id"] as? String ?? "?"
            let repo = m["repo_id"] as? String ?? id
            let status = m["status"] as? String ?? ""
            if !seenRepos.contains(repo) {
                seenRepos.insert(repo)
                let label = status.isEmpty ? id : "\(id) — \(status)"
                modelEntries.append((repo, label, status))
            }
        }

        // 2. Entries from the local HF cache scan (repo names only)
        for name in cachedHFModelNames {
            if !seenRepos.contains(name) {
                seenRepos.insert(name)
                let short = name.split(separator: "/").last.map(String.init) ?? name
                modelEntries.append((name, short, ""))
            }
        }

        if modelEntries.isEmpty {
            let empty = NSMenuItem(title: "No models found — click Scan", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            modelMenu.addItem(empty)
        } else {
            for entry in modelEntries.prefix(30) {
                let item = NSMenuItem(title: entry.label, action: #selector(selectModel(_:)), keyEquivalent: "")
                item.representedObject = entry.repo
                item.state = (entry.repo == activeModel) ? .on : .off
                item.toolTip = "Switch to \(entry.repo)"
                modelMenu.addItem(item)
            }
        }

        modelMenu.addItem(NSMenuItem.separator())
        let scanItem = NSMenuItem(title: "Find Available Models...", action: #selector(scanModels), keyEquivalent: "")
        scanItem.toolTip = "Scan the HuggingFace cache for available MLX models."
        modelMenu.addItem(scanItem)
        let managerItem = NSMenuItem(title: "Download More Models...", action: #selector(openModelManager), keyEquivalent: "")
        managerItem.toolTip = "Open the model management dashboard in your browser."
        modelMenu.addItem(managerItem)

        let modelParent = NSMenuItem(title: "AI Model", action: nil, keyEquivalent: "m")
        modelParent.submenu = modelMenu
        menu.addItem(modelParent)

        let restartVoiceItem = NSMenuItem(title: "Fix Voice", action: #selector(restartVoice), keyEquivalent: "r")
        restartVoiceItem.toolTip = "Recycle the local speech recognizer pipeline."
        menu.addItem(restartVoiceItem)
        let benchmarkItem = NSMenuItem(title: "Speed Test", action: #selector(runBenchmark), keyEquivalent: "b")
        benchmarkItem.toolTip = "Run the standard benchmark suite."
        menu.addItem(benchmarkItem)

        let toolsMenu = NSMenu(title: "Tools")
        let dashboardItem = NSMenuItem(title: "Open Dashboard", action: #selector(openDashboard), keyEquivalent: "d")
        dashboardItem.toolTip = "Open the Bad Apple web dashboard in your browser."
        toolsMenu.addItem(dashboardItem)
        let briefingItem = NSMenuItem(title: "Daily Summary", action: #selector(showBriefing), keyEquivalent: "")
        briefingItem.toolTip = "Show the daily briefing window."
        toolsMenu.addItem(briefingItem)
        let screenActionsItem = NSMenuItem(title: "Read My Screen...", action: #selector(showScreenActions), keyEquivalent: "")
        screenActionsItem.toolTip = "Run actions based on the current screen content."
        toolsMenu.addItem(screenActionsItem)
        let imagePlaygroundItem = NSMenuItem(title: "Make an Image...", action: #selector(showImagePlayground), keyEquivalent: "")
        imagePlaygroundItem.toolTip = "Generate images from a text prompt."
        toolsMenu.addItem(imagePlaygroundItem)
        let viewWorkingMemoryItem = NSMenuItem(title: "What I'm Thinking About", action: #selector(viewWorkingMemory), keyEquivalent: "")
        viewWorkingMemoryItem.toolTip = "Inspect the working memory scratchpad."
        toolsMenu.addItem(viewWorkingMemoryItem)
        let clearWorkingMemoryItem = NSMenuItem(title: "Clear My Memory", action: #selector(clearWorkingMemory), keyEquivalent: "")
        clearWorkingMemoryItem.toolTip = "Erase the working memory scratchpad."
        toolsMenu.addItem(clearWorkingMemoryItem)
        toolsMenu.addItem(NSMenuItem.separator())
        let listShortcutsItem = NSMenuItem(title: "Show My Shortcuts", action: #selector(listShortcuts), keyEquivalent: "")
        listShortcutsItem.toolTip = "List available macOS Shortcuts."
        toolsMenu.addItem(listShortcutsItem)
        let runShortcutItem = NSMenuItem(title: "Run a Shortcut...", action: #selector(runShortcutPrompt), keyEquivalent: "")
        runShortcutItem.toolTip = "Prompt for a Shortcut name and run it."
        toolsMenu.addItem(runShortcutItem)
        let toolsParent = NSMenuItem(title: "Tools", action: nil, keyEquivalent: "")
        toolsParent.submenu = toolsMenu
        menu.addItem(toolsParent)

        let meshMenu = NSMenu(title: "Sync & Devices")
        let p2pEnabled = runtime["p2p_enabled"] as? Bool ?? false
        let p2pItem = NSMenuItem(title: "Sync With Other Devices", action: #selector(toggleP2P), keyEquivalent: "")
        p2pItem.state = p2pEnabled ? .on : .off
        p2pItem.toolTip = "Enable or disable encrypted link-local peer discovery and sync."
        meshMenu.addItem(p2pItem)
        let ambientRunning = runtime["ambient_running"] as? Bool ?? false
        let ambientItem = NSMenuItem(title: ambientRunning ? "Stop Watching" : "Start Watching What I Do", action: #selector(toggleAmbient), keyEquivalent: "")
        ambientItem.state = ambientRunning ? .on : .off
        ambientItem.toolTip = "Capture active app and window context locally for context."
        meshMenu.addItem(ambientItem)
        let setWorkspaceItem = NSMenuItem(title: "Choose My Project Folder...", action: #selector(setWorkspacePrompt), keyEquivalent: "")
        setWorkspaceItem.toolTip = "Set the current workspace path for project-mode context."
        meshMenu.addItem(setWorkspaceItem)
        let openWorkspaceItem = NSMenuItem(title: "Open My Project", action: #selector(openCurrentWorkspace), keyEquivalent: "")
        openWorkspaceItem.toolTip = "Open the configured workspace in Finder."
        meshMenu.addItem(openWorkspaceItem)
        let peers = runtime["p2p_peers"] as? [String] ?? []
        let peersItem = NSMenuItem(title: "Connected Devices: \(peers.count)", action: nil, keyEquivalent: "")
        peersItem.isEnabled = false
        meshMenu.addItem(peersItem)
        let mcpSocket = runtime["mcp_socket"] as? String
        let mcpTitle: String
        if let socket = mcpSocket, !socket.isEmpty {
            mcpTitle = "MCP: \(socket)".truncated(to: 60)
        } else {
            mcpTitle = "MCP: not connected"
        }
        let mcpItem = NSMenuItem(title: mcpTitle, action: nil, keyEquivalent: "")
        mcpItem.isEnabled = false
        mcpItem.toolTip = mcpSocket
        meshMenu.addItem(mcpItem)
        let meshParent = NSMenuItem(title: "Sync & Devices", action: nil, keyEquivalent: "")
        meshParent.submenu = meshMenu
        menu.addItem(meshParent)

        let voiceMenu = NSMenu(title: "Voice")

        let settingsItem = NSMenuItem(title: "Voice Settings...", action: #selector(showVoiceSettings), keyEquivalent: ",")
        voiceMenu.addItem(settingsItem)
        let helpItem = NSMenuItem(title: "How to Use Voice...", action: #selector(showVoiceHelp), keyEquivalent: "")
        voiceMenu.addItem(helpItem)
        let logItem = NSMenuItem(title: "Voice Debug Log...", action: #selector(showVoiceLog), keyEquivalent: "")
        voiceMenu.addItem(logItem)
        voiceMenu.addItem(NSMenuItem.separator())

        let usePiper = UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? true
        let engineToggle = NSMenuItem(title: "Use Native TTS", action: #selector(togglePiperTTS), keyEquivalent: "")
        engineToggle.state = usePiper ? .on : .off
        voiceMenu.addItem(engineToggle)
        voiceMenu.addItem(NSMenuItem.separator())

        for voice in PiperTTSClient.availableVoices {
            let display = voice
                .replacingOccurrences(of: "es_MX-", with: "")
                .replacingOccurrences(of: "en_US-", with: "")
                .replacingOccurrences(of: "-high", with: "")
                .replacingOccurrences(of: "-medium", with: "")
                .replacingOccurrences(of: "-", with: " ")
                .capitalized
            let item = NSMenuItem(title: display, action: #selector(selectVoice(_:)), keyEquivalent: "")
            item.representedObject = voice
            let current = UserDefaults.standard.string(forKey: "BadAppleTTSVoice") ?? PiperTTSClient.defaultVoice
            item.state = (current == voice) ? .on : .off
            item.isEnabled = usePiper
            voiceMenu.addItem(item)
        }
        let voiceParent = NSMenuItem(title: usePiper ? "Voice (Native TTS)" : "Voice (Apple)", action: nil, keyEquivalent: "")
        voiceParent.submenu = voiceMenu
        menu.addItem(voiceParent)

        let accentMenu = NSMenu(title: "Accent")
        for accent in [
            ("en-US", "California Beach (Sandy)"),
            ("es-MX", "Latina (Paulina)"),
            ("ru-RU", "Russian (Milena)"),
            ("uk-UA", "Ukrainian (Lesya)"),
            ("sk-SK", "Slovak (Laura)"),
        ] {
            let item = NSMenuItem(title: accent.1, action: #selector(selectAccent(_:)), keyEquivalent: "")
            item.representedObject = accent.0
            let current = UserDefaults.standard.string(forKey: "BadAppleTTSAccent") ?? "en-US"
            item.state = (current == accent.0) ? .on : .off
            item.isEnabled = !usePiper
            accentMenu.addItem(item)
        }
        let accentParent = NSMenuItem(title: "Accent", action: nil, keyEquivalent: "")
        accentParent.submenu = accentMenu
        menu.addItem(accentParent)

        if !lastPrompt.isEmpty {
            let prompt = NSMenuItem(title: "Last prompt: \(lastPrompt.truncated(to: 65))", action: nil, keyEquivalent: "")
            prompt.isEnabled = false
            prompt.toolTip = lastPrompt
            menu.addItem(prompt)
        }
        // Removed: "Streamed output: X characters" — nobody cares about this.
        if let lastError = lastError {
            let error = NSMenuItem(title: "Error: \(lastError.truncated(to: 75))", action: nil, keyEquivalent: "")
            error.isEnabled = false
            error.toolTip = lastError
            menu.addItem(error)
        }

        menu.addItem(NSMenuItem.separator())
        let ffiStatus = NSMenuItem(
            title: BadAppleFFI.shared.isLoaded
                ? String(format: "Apple Latency: %.2f ms", Double(BadAppleFFI.shared.appleLatencyUs()) / 1000.0)
                : "libbad_apple not loaded (voice uses SLICKS)",
            action: nil,
            keyEquivalent: ""
        )
        ffiStatus.isEnabled = false
        menu.addItem(ffiStatus)

        let subMenu = NSMenu(title: "Current Tasks")
        if nativeAgentTasks.isEmpty {
            let empty = NSMenuItem(title: "No active tasks", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            subMenu.addItem(empty)
        } else {
            for task in nativeAgentTasks.prefix(8) {
                let title = "[\(task.status.rawValue)] \(task.goal)".truncated(to: 70)
                let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
                item.isEnabled = false
                item.toolTip = task.summary.isEmpty ? task.goal : task.summary
                subMenu.addItem(item)
            }
        }
        let parent = NSMenuItem(title: "Current Tasks", action: nil, keyEquivalent: "")
        parent.submenu = subMenu
        menu.addItem(parent)
        menu.addItem(NSMenuItem(title: "Add a Task...", action: #selector(pushPursuit), keyEquivalent: "p"))
        menu.addItem(NSMenuItem.separator())
        let helpMenu = NSMenu(title: "Help & Fixes")
        let restartDaemonItem = NSMenuItem(title: "Restart Bad Apple", action: #selector(restartDaemon), keyEquivalent: "")
        restartDaemonItem.toolTip = "Unload and reload the Bad Apple system LaunchDaemons."
        helpMenu.addItem(restartDaemonItem)
        let openLogItem = NSMenuItem(title: "Open Debug Log", action: #selector(openLog), keyEquivalent: "")
        openLogItem.toolTip = "Open /var/log/bad_apple_mlx_server.log in the default editor."
        helpMenu.addItem(openLogItem)
        if let mcpSocket = runtime["mcp_socket"] as? String, !mcpSocket.isEmpty {
            let copyMCPItem = NSMenuItem(title: "Copy Connection Info", action: #selector(copyMCPSocket), keyEquivalent: "")
            copyMCPItem.toolTip = mcpSocket
            helpMenu.addItem(copyMCPItem)
        }
        let helpParent = NSMenuItem(title: "Help & Fixes", action: nil, keyEquivalent: "")
        helpParent.submenu = helpMenu
        menu.addItem(helpParent)

        menu.addItem(NSMenuItem.separator())
        let updateItem = NSMenuItem(title: "Check for Updates...", action: #selector(checkForUpdates), keyEquivalent: "")
        updateItem.toolTip = "Download and install the latest unsigned release from GitHub."
        menu.addItem(updateItem)
        let startAtLoginItem = NSMenuItem(title: "Start When I Log In", action: #selector(toggleStartAtLogin), keyEquivalent: "")
        startAtLoginItem.state = isStartAtLoginEnabled() ? .on : .off
        menu.addItem(startAtLoginItem)
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(terminate), keyEquivalent: "q"))
        assignMenuTargets(menu)
    }

    private var launchAgentPlist: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents/com.badapple.menubar.plist")
    }

    private func launchAgentPlistContents() -> String {
        let appPath = Bundle.main.bundlePath
        return """
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>com.badapple.menubar</string>
    <key>ProgramArguments</key>
    <array>
        <string>\(appPath)/Contents/MacOS/BadApple</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>LimitLoadToSessionType</key>
    <string>Aqua</string>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>/tmp/badapple_menubar.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/badapple_menubar.log</string>
</dict>
</plist>
"""
    }

    private func isStartAtLoginEnabled() -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: launchAgentPlist.path) else { return false }
        let task = Process()
        task.launchPath = "/bin/launchctl"
        task.arguments = ["list", "com.badapple.menubar"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
            try task.run()
            task.waitUntilExit()
            return task.terminationStatus == 0
        } catch {
            return false
        }
    }

    @objc private func toggleStartAtLogin() {
        let task = Process()
        task.launchPath = "/bin/launchctl"

        if isStartAtLoginEnabled() {
            task.arguments = ["unload", "-w", launchAgentPlist.path]
            do {
                try task.run()
                task.waitUntilExit()
                try? FileManager.default.removeItem(at: launchAgentPlist)
            } catch {
                lastError = "Failed to disable start at login: \(error)"
            }
        } else {
            let fm = FileManager.default
            let agentsDir = launchAgentPlist.deletingLastPathComponent()
            try? fm.createDirectory(at: agentsDir, withIntermediateDirectories: true)
            let contents = launchAgentPlistContents()
            do {
                try contents.write(to: launchAgentPlist, atomically: true, encoding: .utf8)
                let load = Process()
                load.launchPath = "/bin/launchctl"
                load.arguments = ["load", "-w", launchAgentPlist.path]
                try load.run()
                load.waitUntilExit()
            } catch {
                lastError = "Failed to enable start at login: \(error)"
            }
        }
        rebuildMenu()
    }

    @objc private func togglePrivateMode() {
        let enabled = runtimeState["private_mode"] as? Bool ?? false
        Task {
            do {
                _ = try await runBadAppleCLI(
                    prompt: enabled ? "private mode off" : "private mode on",
                    socketPath: BadAppleBrain.deepSocket,
                    maxTokens: 32
                )
                await MainActor.run { self.rebuildMenu() }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.rebuildMenu()
                }
            }
        }
    }

    @objc private func engageKillSwitch() {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Stop Bad Apple?"
        alert.informativeText = "This cancels generation, stops ambient capture, and blocks tools until you resume."
        alert.addButton(withTitle: "Stop Everything")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        Task {
            _ = try? await runBadAppleCLI(prompt: "stop everything", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run {
                self.voiceHost.stopAllAudio()
                self.rebuildMenu()
            }
        }
    }

    @objc private func resetKillSwitch() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "resume bad apple", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.rebuildMenu() }
        }
    }

    @objc private func restartDaemon() {
        let script = """
        do shell script "launchctl unload /Library/LaunchDaemons/com.badapple.mlx.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.gatekeeper.plist 2>/dev/null; launchctl unload /Library/LaunchDaemons/com.badapple.supervisor.plist 2>/dev/null; sleep 2; launchctl load -w /Library/LaunchDaemons/com.badapple.gatekeeper.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.mlx.plist; launchctl load -w /Library/LaunchDaemons/com.badapple.supervisor.plist" with administrator privileges
        """
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        task.arguments = ["-e", script]
        do {
            try task.run()
            badAppleVoiceLog("Daemon restart initiated")
        } catch {
            badAppleVoiceLog("Daemon restart failed: \(error.localizedDescription)")
        }
    }

    @objc private func showSystemHealth() {
        Task {
            do {
                let output = try await runBadAppleCLI(prompt: "runtime status", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
                await MainActor.run {
                    let alert = NSAlert()
                    alert.messageText = "Bad Apple System Health"
                    alert.informativeText = output
                    alert.alertStyle = .informational
                    alert.runModal()
                }
            } catch {
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.rebuildMenu()
                }
            }
        }
    }

    @objc private func showControlCenter() {
        controlCenter.show()
    }

    @objc private func toggleVoice() {
        voiceEnabled.toggle()
        voiceHost.setEnabled(voiceEnabled)
        rebuildMenu()
    }

    @objc private func restartVoice() {
        guard voiceEnabled else { return }
        voiceHost.setEnabled(false)
        voiceHost.setEnabled(true)
    }

    @objc private func selectVoice(_ sender: NSMenuItem) {
        guard let voice = sender.representedObject as? String,
              PiperTTSClient.availableVoices.contains(voice) else { return }
        UserDefaults.standard.set(voice, forKey: "BadAppleTTSVoice")
        badAppleVoiceLog("selected TTS voice: \(voice)")
        rebuildMenu()
    }

    @objc private func togglePiperTTS() {
        let current = UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false
        let next = !current
        UserDefaults.standard.set(next, forKey: "BadAppleUsePiperTTS")
        badAppleVoiceLog("Native TTS enabled: \(next)")
        if next {
            PiperTTSClient.shared.warmup()
        }
        rebuildMenu()
    }

    @objc private func showVoiceHelp() {
        voiceHelp.show()
    }

    @objc private func showVoiceLog() {
        voiceLog.makeKeyAndOrderFront(nil)
    }

    @objc private func showBriefing() {
        briefingWindow.makeKeyAndOrderFront(nil)
    }

    @objc private func showScreenActions() {
        screenActionsWindow.makeKeyAndOrderFront(nil)
    }

    @objc private func showImagePlayground() {
        imagePlayground.makeKeyAndOrderFront(nil)
    }

    private func extractImagePath(from text: String) -> String? {
        let pattern = "\\S+\\.(png|jpg|jpeg|webp)"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { return nil }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: range),
              let swiftRange = Range(match.range, in: text) else { return nil }
        let path = String(text[swiftRange])
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    private func latestGeneratedImage(since: TimeInterval) -> String? {
        let dir = URL(fileURLWithPath: BadAppleBrain.generatedImagesDir)
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey], options: []) else { return nil }
        var best: (url: URL, mtime: TimeInterval)?
        for url in files where url.pathExtension.lowercased() == "png" {
            guard let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey]),
                  let mtime = vals.contentModificationDate?.timeIntervalSince1970,
                  mtime >= since else { continue }
            if best == nil || mtime > best!.mtime {
                best = (url, mtime)
            }
        }
        return best?.url.path
    }

    private func currentScreenContext() -> String {
        guard let app = NSWorkspace.shared.frontmostApplication else { return "No frontmost application." }
        let appName = app.localizedName ?? "Unknown"
        let pid = app.processIdentifier
        let appElement = AXUIElementCreateApplication(pid)
        var out: CFTypeRef?
        var title = ""
        let result = AXUIElementCopyAttributeValue(appElement, kAXMainWindowAttribute as CFString, &out)
        if result == .success, let window = out {
            var titleRef: CFTypeRef?
            AXUIElementCopyAttributeValue(window as! AXUIElement, kAXTitleAttribute as CFString, &titleRef)
            title = (titleRef as? String) ?? ""
        }
        let context = title.isEmpty ? "Application: \(appName)" : "Application: \(appName)\nWindow: \(title)"
        return context
    }

    @objc private func runWritingToolService(_ pboard: NSPasteboard, instruction: String, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        guard let text = pboard.string(forType: .string), !text.isEmpty else {
            error.pointee = "No text was selected." as NSString
            return
        }
        let prompt = "\(instruction):\n\n\(text)"
        guard let result = runServiceSync(prompt: prompt) else {
            error.pointee = "Bad Apple did not return a response." as NSString
            return
        }
        pboard.clearContents()
        pboard.setString(result, forType: .string)
    }

    @objc private func rewriteWithBadApple(_ pboard: NSPasteboard, userData: String, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        runWritingToolService(pboard, instruction: "Rewrite the following text to be clearer and more natural", error: error)
    }

    @objc private func summarizeWithBadApple(_ pboard: NSPasteboard, userData: String, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        runWritingToolService(pboard, instruction: "Summarize the following text in one or two sentences", error: error)
    }

    @objc private func proofreadWithBadApple(_ pboard: NSPasteboard, userData: String, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        runWritingToolService(pboard, instruction: "Proofread the following text and return only the corrected version", error: error)
    }

    private func runServiceSync(prompt: String) -> String? {
        var result: String?
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                result = try await runBadAppleCLI(prompt: prompt, socketPath: BadAppleBrain.deepSocket, maxTokens: 300)
            } catch {
                badAppleVoiceLog("service prompt error: \(error)")
            }
            sem.signal()
        }
        sem.wait()
        return result
    }

    @objc private func showVoiceSettings() {
        voiceSettings.makeKeyAndOrderFront(nil)
    }

    @objc private func selectAccent(_ sender: NSMenuItem) {
        guard let accent = sender.representedObject as? String else { return }
        UserDefaults.standard.set(accent, forKey: "BadAppleTTSAccent")
        badAppleVoiceLog("selected TTS accent: \(accent)")
        rebuildMenu()
    }

    @objc private func pushPursuit() {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Add a task"
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 340, height: 24))
        alert.accessoryView = textField
        alert.addButton(withTitle: "Add")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            let text = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                Task {
                    do {
                        _ = try await BadAppleEngine.shared.submitAgentTask(goal: text)
                        let tasks = await BadAppleEngine.shared.listAgentTasks()
                        await MainActor.run {
                            self.nativeAgentTasks = tasks
                            self.rebuildMenu()
                        }
                    } catch {
                        await MainActor.run {
                            self.lastError = error.localizedDescription
                            self.rebuildMenu()
                        }
                    }
                }
            }
        }
    }

    // MARK: NSMenuDelegate

    func menuWillOpen(_ menu: NSMenu) {
        if menu === self.menu {
            needsMenuRebuild = false
            rebuildMenu()
        }
        openMenuCount += 1
    }

    func menuDidClose(_ menu: NSMenu) {
        openMenuCount = max(0, openMenuCount - 1)
        if openMenuCount == 0, needsMenuRebuild {
            rebuildMenu()
        }
    }

    private func assignMenuTargets(_ menu: NSMenu) {
        menu.delegate = self
        for item in menu.items {
            if item.action != nil && !item.isSeparatorItem {
                item.target = self
            }
            if let submenu = item.submenu {
                assignMenuTargets(submenu)
            }
        }
    }

    @objc private func checkForUpdates() {
        guard let script = Bundle.main.path(forResource: "update_bad_apple", ofType: "sh"),
              !script.isEmpty else {
            let err = NSAlert()
            err.messageText = "Update script not found"
            err.informativeText = "The updater is not bundled in this build."
            err.alertStyle = .critical
            err.runModal()
            return
        }

        let alert = NSAlert()
        alert.messageText = "Check for Bad Apple updates?"
        alert.informativeText = "This downloads the latest unsigned release from GitHub and replaces /Applications/Bad Apple.app. Requires administrator password."
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Update")
        alert.addButton(withTitle: "Cancel")
        let result = alert.runModal()
        if result == .alertFirstButtonReturn {
            let repo = "savag3/bad_apple"
            let cmd = "BADAPPLE_GH_REPO=\(repo) \\\"\(script)\\\""
            let appleScript = "do shell script \"\(cmd)\" with administrator privileges"
            var errorInfo: NSDictionary?
            NSAppleScript(source: appleScript)?.executeAndReturnError(&errorInfo)
            if let errorInfo = errorInfo {
                let msg = errorInfo[NSAppleScript.errorMessage] as? String ?? "unknown error"
                let err = NSAlert()
                err.messageText = "Update failed"
                err.informativeText = msg
                err.alertStyle = .critical
                err.runModal()
            }
        }
    }

    @objc private func terminate() { NSApp.terminate(nil) }

    @objc private func openLog() {
        let logURL = URL(fileURLWithPath: "/var/log/bad_apple_mlx_server.log")
        NSWorkspace.shared.open(logURL)
    }

    @objc private func copyMCPSocket() {
        guard let socket = lastRuntimeStatus["mcp_socket"] as? String, !socket.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(socket, forType: .string)
    }
}

// MARK: - Settings / Preferences window

/// Preferences window that exposes key Bad Apple settings as toggles and
/// dropdowns so non-technical users do not need to edit policy.yaml or use
/// CLI flags. Pure AppKit — no SwiftUI.
final class BadAppleSettingsWindow: NSObject {
    private var window: NSWindow?
    private let windowSize = NSSize(width: 480, height: 520)

    // Setting controls
    private var autopilotButton: NSButton?
    private var autopilotLabel: NSTextField?
    private var fastTierButton: NSButton?
    private var voiceModeButton: NSButton?
    private var p2pButton: NSButton?
    private var airGapButton: NSButton?
    private var personaPopup: NSPopUpButton?
    private var workspaceField: NSTextField?

    // Persona options
    private let personas: [(id: String, display: String)] = [
        ("default", "Default"),
        ("wicket", "Wicket"),
        ("genz", "Gen Z"),
        ("drill", "Drill"),
        ("midwest", "Midwest"),
    ]

    // UserDefaults keys
    private let autopilotKey = "BadAppleAutopilot"
    private let fastTierKey = "BadAppleSettingsFastTier"
    private let voiceModeKey = "BadAppleSettingsVoiceMode"
    private let p2pKey = "BadAppleSettingsP2PSync"
    private let airGapKey = "BadAppleSettingsAirGap"
    private let personaKey = "BadAppleSelectedPersona"
    private let workspaceKey = "BadAppleSettingsWorkspace"

    func show() {
        if window == nil { buildWindow() }
        loadSettings()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: - Window construction

    private func buildWindow() {
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - windowSize.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - windowSize.height / 2,
            width: windowSize.width,
            height: windowSize.height
        )
        let w = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        w.title = "Bad Apple Settings"
        w.isReleasedWhenClosed = false
        w.level = .normal
        w.minSize = windowSize
        w.maxSize = windowSize

        let visual = NSVisualEffectView(frame: NSRect(origin: .zero, size: windowSize))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true

        let contentWidth = windowSize.width - 48
        var y = windowSize.height - 24.0

        // --- General section ---
        let genHeader = sectionHeader("GENERAL")
        y -= 16
        genHeader.frame = NSRect(x: 24, y: y, width: contentWidth, height: 16)
        visual.addSubview(genHeader)
        y -= 4

        let (autoRow, autoBtn, autoLbl) = toggleRow(
            label: "Autopilot (skip approval prompts)",
            description: "When on, destructive tools run without approval prompts.",
            action: #selector(autopilotToggled(_:)),
            isOn: boolSetting(autopilotKey, defaultValue: false),
            warningWhenOn: true
        )
        autopilotButton = autoBtn
        autopilotLabel = autoLbl
        y -= 38
        autoRow.frame = NSRect(x: 24, y: y, width: contentWidth, height: 38)
        visual.addSubview(autoRow)
        y -= 4

        let (voiceRow, voiceBtn, _) = toggleRow(
            label: "Voice Mode (speak responses aloud)",
            description: "Controls whether text-to-speech reads responses out loud.",
            action: #selector(voiceModeToggled(_:)),
            isOn: boolSetting(voiceModeKey, defaultValue: false)
        )
        voiceModeButton = voiceBtn
        y -= 38
        voiceRow.frame = NSRect(x: 24, y: y, width: contentWidth, height: 38)
        visual.addSubview(voiceRow)
        y -= 10

        // --- Models section ---
        let modelsHeader = sectionHeader("MODELS")
        y -= 16
        modelsHeader.frame = NSRect(x: 24, y: y, width: contentWidth, height: 16)
        visual.addSubview(modelsHeader)
        y -= 4

        let (ftRow, ftBtn, _) = toggleRow(
            label: "Fast Tier (route simple queries to 0.5B model)",
            description: "Simple math, identity, time, and greeting queries skip the 9B model.",
            action: #selector(fastTierToggled(_:)),
            isOn: boolSetting(fastTierKey, defaultValue: false)
        )
        fastTierButton = ftBtn
        y -= 38
        ftRow.frame = NSRect(x: 24, y: y, width: contentWidth, height: 38)
        visual.addSubview(ftRow)
        y -= 8

        let personaRow = popupRow(
            label: "Persona",
            items: personas.map { $0.display },
            action: #selector(personaChanged(_:))
        )
        personaPopup = personaRow.popup
        y -= 24
        personaRow.view.frame = NSRect(x: 24, y: y, width: contentWidth, height: 24)
        visual.addSubview(personaRow.view)
        y -= 10

        // --- Network section ---
        let netHeader = sectionHeader("NETWORK")
        y -= 16
        netHeader.frame = NSRect(x: 24, y: y, width: contentWidth, height: 16)
        visual.addSubview(netHeader)
        y -= 4

        let (p2pRow, p2pBtn, _) = toggleRow(
            label: "P2P Sync (encrypted mesh with other Bad Apple peers)",
            description: "Link-local UDP/TCP, AES-256-GCM. Off by default for air-gap certification.",
            action: #selector(p2pToggled(_:)),
            isOn: boolSetting(p2pKey, defaultValue: false)
        )
        p2pButton = p2pBtn
        y -= 38
        p2pRow.frame = NSRect(x: 24, y: y, width: contentWidth, height: 38)
        visual.addSubview(p2pRow)
        y -= 4

        let (agRow, agBtn, _) = toggleRow(
            label: "Air Gap (block all network access)",
            description: "Prevents Bad Apple from making any network connections.",
            action: #selector(airGapToggled(_:)),
            isOn: boolSetting(airGapKey, defaultValue: false)
        )
        airGapButton = agBtn
        y -= 38
        agRow.frame = NSRect(x: 24, y: y, width: contentWidth, height: 38)
        visual.addSubview(agRow)
        y -= 10

        // --- Advanced section ---
        let advHeader = sectionHeader("ADVANCED")
        y -= 16
        advHeader.frame = NSRect(x: 24, y: y, width: contentWidth, height: 16)
        visual.addSubview(advHeader)
        y -= 4

        let wsRow = workspaceRow()
        workspaceField = wsRow.field
        y -= 64
        wsRow.view.frame = NSRect(x: 24, y: y, width: contentWidth, height: 64)
        visual.addSubview(wsRow.view)
        y -= 4

        // --- Done button ---
        let doneButton = NSButton(title: "Done", target: self, action: #selector(closeSettings(_:)))
        doneButton.bezelStyle = .rounded
        doneButton.keyEquivalent = "\r"
        doneButton.frame = NSRect(x: windowSize.width - 24 - 90, y: 12, width: 90, height: 28)
        visual.addSubview(doneButton)

        w.contentView = visual
        window = w
    }

    // MARK: - Row builders

    private func sectionHeader(_ title: String) -> NSTextField {
        let label = NSTextField(labelWithString: title)
        label.font = .systemFont(ofSize: 11, weight: .bold)
        label.textColor = .secondaryLabelColor
        return label
    }

    private func toggleRow(label: String, description: String, action: Selector, isOn: Bool, warningWhenOn: Bool = false) -> (NSView, NSButton, NSTextField) {
        let container = NSView()
        let contentWidth = windowSize.width - 48

        let labelField = NSTextField(labelWithString: label)
        labelField.font = .systemFont(ofSize: 13)
        labelField.textColor = (warningWhenOn && isOn) ? .systemOrange : .labelColor
        labelField.lineBreakMode = .byTruncatingTail
        labelField.cell?.truncatesLastVisibleLine = true
        labelField.frame = NSRect(x: 0, y: 20, width: contentWidth - 36, height: 18)
        container.addSubview(labelField)

        let toggle = NSButton(checkboxWithTitle: "", target: self, action: action)
        toggle.state = isOn ? .on : .off
        toggle.frame = NSRect(x: contentWidth - 30, y: 19, width: 30, height: 20)
        container.addSubview(toggle)

        let descLabel = NSTextField(wrappingLabelWithString: description)
        descLabel.font = .systemFont(ofSize: 11)
        descLabel.textColor = .secondaryLabelColor
        descLabel.frame = NSRect(x: 0, y: 0, width: contentWidth, height: 16)
        container.addSubview(descLabel)

        return (container, toggle, labelField)
    }

    private func popupRow(label: String, items: [String], action: Selector) -> (view: NSView, popup: NSPopUpButton) {
        let container = NSView()
        let contentWidth = windowSize.width - 48

        let labelField = NSTextField(labelWithString: label)
        labelField.font = .systemFont(ofSize: 13)
        labelField.textColor = .labelColor
        labelField.frame = NSRect(x: 0, y: 3, width: 120, height: 18)
        container.addSubview(labelField)

        let popup = NSPopUpButton()
        popup.target = self
        popup.action = action
        for item in items {
            popup.addItem(withTitle: item)
        }
        popup.frame = NSRect(x: contentWidth - 200, y: 0, width: 200, height: 24)
        container.addSubview(popup)

        return (container, popup)
    }

    private func workspaceRow() -> (view: NSView, field: NSTextField) {
        let container = NSView()
        let contentWidth = windowSize.width - 48

        let labelField = NSTextField(labelWithString: "Workspace path")
        labelField.font = .systemFont(ofSize: 13)
        labelField.textColor = .labelColor
        labelField.frame = NSRect(x: 0, y: 46, width: contentWidth, height: 18)
        container.addSubview(labelField)

        let field = NSTextField()
        field.placeholderString = "/path/to/workspace"
        field.bezelStyle = .roundedBezel
        field.frame = NSRect(x: 0, y: 18, width: contentWidth - 170, height: 24)
        container.addSubview(field)

        let browseButton = NSButton(title: "Browse…", target: self, action: #selector(browseWorkspace(_:)))
        browseButton.bezelStyle = .rounded
        browseButton.frame = NSRect(x: contentWidth - 160, y: 18, width: 75, height: 24)
        container.addSubview(browseButton)

        let setButton = NSButton(title: "Set", target: self, action: #selector(setWorkspace(_:)))
        setButton.bezelStyle = .rounded
        setButton.frame = NSRect(x: contentWidth - 80, y: 18, width: 80, height: 24)
        container.addSubview(setButton)

        let descLabel = NSTextField(wrappingLabelWithString: "Path to current workspace.")
        descLabel.font = .systemFont(ofSize: 11)
        descLabel.textColor = .secondaryLabelColor
        descLabel.frame = NSRect(x: 0, y: 0, width: contentWidth, height: 14)
        container.addSubview(descLabel)

        return (container, field)
    }

    // MARK: - Settings persistence

    private func boolSetting(_ key: String, defaultValue: Bool) -> Bool {
        UserDefaults.standard.object(forKey: key) as? Bool ?? defaultValue
    }

    private func loadSettings() {
        let autoOn = boolSetting(autopilotKey, defaultValue: false)
        autopilotButton?.state = autoOn ? .on : .off
        autopilotLabel?.textColor = autoOn ? .systemOrange : .labelColor

        fastTierButton?.state = boolSetting(fastTierKey, defaultValue: false) ? .on : .off
        voiceModeButton?.state = boolSetting(voiceModeKey, defaultValue: false) ? .on : .off
        p2pButton?.state = boolSetting(p2pKey, defaultValue: false) ? .on : .off
        airGapButton?.state = boolSetting(airGapKey, defaultValue: false) ? .on : .off

        let currentPersona = UserDefaults.standard.string(forKey: personaKey) ?? "default"
        if let index = personas.firstIndex(where: { $0.id == currentPersona }) {
            personaPopup?.selectItem(at: index)
        }

        workspaceField?.stringValue = UserDefaults.standard.string(forKey: workspaceKey) ?? ""
    }

    // MARK: - Actions

    @objc private func autopilotToggled(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: autopilotKey)
        UserDefaults.standard.set(enabled, forKey: "BadAppleSettingsAutopilot")
        autopilotLabel?.textColor = enabled ? .systemOrange : .labelColor
        BadAppleEngine.shared.autopilot = enabled
        sendCommand(enabled ? "enable autopilot" : "disable autopilot")
    }

    @objc private func fastTierToggled(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: fastTierKey)
        sendCommand(enabled ? "enable fast tier" : "disable fast tier")
    }

    @objc private func voiceModeToggled(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: voiceModeKey)
        // Voice mode is a local TTS setting; no daemon command needed.
    }

    @objc private func p2pToggled(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: p2pKey)
        sendCommand(enabled ? "p2p on" : "p2p off")
    }

    @objc private func airGapToggled(_ sender: NSButton) {
        let enabled = sender.state == .on
        UserDefaults.standard.set(enabled, forKey: airGapKey)
        sendCommand(enabled ? "enable air gap" : "disable air gap")
    }

    @objc private func personaChanged(_ sender: NSPopUpButton) {
        let index = sender.indexOfSelectedItem
        guard index < personas.count else { return }
        let personaId = personas[index].id
        UserDefaults.standard.set(personaId, forKey: personaKey)
        _ = BadAppleEngine.shared.switchPersona(personaId)
        sendCommand("switch to \(personaId)")
    }

    @objc private func browseWorkspace(_ sender: NSButton) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose Workspace"
        if panel.runModal() == .OK, let url = panel.url {
            workspaceField?.stringValue = url.path
        }
    }

    @objc private func setWorkspace(_ sender: NSButton) {
        let path = workspaceField?.stringValue.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !path.isEmpty else { return }
        UserDefaults.standard.set(path, forKey: workspaceKey)
        BadAppleEngine.shared.workspacePath = path
        sendCommand("set workspace to \(path)")
    }

    @objc private func closeSettings(_ sender: NSButton) {
        window?.orderOut(nil)
    }

    // MARK: - Daemon command sender

    /// Runs the bundled badapple CLI with the given prompt, setting the
    /// socket and SLICKS key environment variables. Runs asynchronously on
    /// a background queue so the UI stays responsive.
    ///
    /// NATIVE ENGINE: When the Swift MLX engine is loaded, control commands
    /// (autopilot, airgap, private mode, persona, workspace) are applied
    /// directly to the engine instead of spawning a CLI subprocess. Commands
    /// the native engine cannot handle (fast tier, p2p, kill switch) fall
    /// through to the daemon CLI path below.
    private func sendCommand(_ prompt: String) {
        // NATIVE ENGINE: handle control commands directly when loaded.
        if BadAppleEngine.shared.isLoaded {
            let lower = prompt.lowercased()
            if lower == "enable autopilot" {
                BadAppleEngine.shared.autopilot = true
                badAppleVoiceLog("BadAppleSettings: native engine — autopilot enabled")
                return
            }
            if lower == "disable autopilot" {
                BadAppleEngine.shared.autopilot = false
                badAppleVoiceLog("BadAppleSettings: native engine — autopilot disabled")
                return
            }
            if lower == "enable air gap" {
                BadAppleEngine.shared.airgap = true
                badAppleVoiceLog("BadAppleSettings: native engine — air gap enabled")
                return
            }
            if lower == "disable air gap" {
                BadAppleEngine.shared.airgap = false
                badAppleVoiceLog("BadAppleSettings: native engine — air gap disabled")
                return
            }
            if lower == "private mode on" {
                BadAppleEngine.shared.privateMode = true
                badAppleVoiceLog("BadAppleSettings: native engine — private mode enabled")
                return
            }
            if lower == "private mode off" {
                BadAppleEngine.shared.privateMode = false
                badAppleVoiceLog("BadAppleSettings: native engine — private mode disabled")
                return
            }
            if lower.hasPrefix("switch to ") {
                let persona = String(prompt.dropFirst("switch to ".count))
                _ = BadAppleEngine.shared.switchPersona(persona)
                badAppleVoiceLog("BadAppleSettings: native engine — switched to persona '\(persona)'")
                return
            }
            if lower.hasPrefix("set workspace to ") {
                let path = String(prompt.dropFirst("set workspace to ".count))
                BadAppleEngine.shared.workspacePath = path
                badAppleVoiceLog("BadAppleSettings: native engine — workspace set to '\(path)'")
                return
            }
            // Unrecognized commands (fast tier, p2p, kill switch) are not
            // supported by the native engine — fall through to the daemon CLI.
            badAppleVoiceLog("BadAppleSettings: native engine cannot handle '\(prompt)', falling back to daemon")
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let binary = Bundle.main.bundleURL
                .appendingPathComponent("Contents")
                .appendingPathComponent("Helpers")
                .appendingPathComponent("badapple")
            guard FileManager.default.fileExists(atPath: binary.path) else {
                badAppleVoiceLog("BadAppleSettings: badapple binary not found at \(binary.path)")
                return
            }
            let process = Process()
            process.executableURL = binary
            process.arguments = ["--max-tokens", "32", prompt]
            var environment = ProcessInfo.processInfo.environment
            environment["BADAPPLE_SOCKET_PATH"] = BadAppleBrain.deepSocket
            environment["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
            process.environment = environment
            do {
                try process.run()
                process.waitUntilExit()
                badAppleVoiceLog("BadAppleSettings: sent '\(prompt)' (exit=\(process.terminationStatus))")
            } catch {
                badAppleVoiceLog("BadAppleSettings: failed to send '\(prompt)': \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Guided first-run onboarding window

/// A multi-step guided onboarding wizard for first-time Bad Apple users.
/// Walks the user through welcome, privacy, model status, permissions, and
/// a first query. Pure AppKit — no SwiftUI.
final class BadAppleOnboardingWindow: NSObject, NSTextFieldDelegate {
    private var window: NSWindow?
    private var visual: NSVisualEffectView?
    private var contentContainer: NSView?
    private var dots: [NSView] = []
    private var backButton: NSButton?
    private var continueButton: NSButton?
    private var currentStep = 0
    private let totalSteps = 5
    private let windowSize = NSSize(width: 520, height: 580)

    /// Usable content area inside the window (excludes side padding and the
    /// bottom bar that holds the dots and navigation buttons).
    private var contentSize: NSSize {
        NSSize(width: windowSize.width - 56, height: windowSize.height - 124)
    }

    // Step 2 — model status
    private var modelStatusLabel: NSTextField?
    private var modelProgressIndicator: NSProgressIndicator?
    private var modelCheckTimer: Timer?
    private var modelIsReady = false
    private var daemonRunning = false

    // Step 3 — permissions
    private var permCheckTimer: Timer?
    private struct PermStatusRef {
        let dot: NSView
        let label: NSTextField
    }
    private var permStatusRefs: [PermStatusRef] = []

    // Step 4 — first query
    private var queryField: NSTextField?
    private var askButton: NSButton?
    private var responseTextView: NSTextView?
    private var querySpinner: NSProgressIndicator?
    private var isQuerying = false

    func show() {
        if window == nil { buildWindow() }
        currentStep = 0
        modelIsReady = false
        daemonRunning = false
        renderStep()
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    deinit {
        modelCheckTimer?.invalidate()
        permCheckTimer?.invalidate()
    }

    // MARK: - Window construction

    private func buildWindow() {
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 600) - windowSize.width / 2,
            y: (screen?.visibleFrame.midY ?? 400) - windowSize.height / 2,
            width: windowSize.width,
            height: windowSize.height
        )
        let wc = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        wc.title = "Bad Apple"
        wc.isReleasedWhenClosed = false
        wc.minSize = windowSize
        wc.maxSize = windowSize

        let v = NSVisualEffectView(frame: NSRect(origin: .zero, size: windowSize))
        v.material = .hudWindow
        v.state = .active
        v.blendingMode = .behindWindow
        v.wantsLayer = true

        let size = contentSize
        let container = NSView(frame: NSRect(x: 28, y: 96, width: size.width, height: size.height))
        container.wantsLayer = true
        v.addSubview(container)

        buildDots(in: v)

        backButton = NSButton(title: "Back", target: self, action: #selector(backPressed(_:)))
        backButton?.bezelStyle = .rounded
        backButton?.frame = NSRect(x: 28, y: 20, width: 84, height: 32)
        if let backButton = backButton { v.addSubview(backButton) }

        continueButton = NSButton(title: "Get Started", target: self, action: #selector(continuePressed(_:)))
        continueButton?.bezelStyle = .rounded
        continueButton?.keyEquivalent = "\r"
        continueButton?.frame = NSRect(x: windowSize.width - 28 - 130, y: 20, width: 130, height: 32)
        if let continueButton = continueButton { v.addSubview(continueButton) }

        wc.contentView = v
        visual = v
        contentContainer = container
        window = wc
    }

    private func buildDots(in v: NSVisualEffectView) {
        let dotSize: CGFloat = 8
        let spacing: CGFloat = 14
        let total = CGFloat(totalSteps) * dotSize + CGFloat(totalSteps - 1) * spacing
        let startX = (windowSize.width - total) / 2
        for i in 0..<totalSteps {
            let dot = NSView(frame: NSRect(x: startX + CGFloat(i) * (dotSize + spacing), y: 62, width: dotSize, height: dotSize))
            dot.wantsLayer = true
            dot.layer?.cornerRadius = dotSize / 2
            dot.layer?.backgroundColor = NSColor.systemGray.withAlphaComponent(0.4).cgColor
            v.addSubview(dot)
            dots.append(dot)
        }
    }

    // MARK: - Step rendering

    private func renderStep() {
        guard let container = contentContainer else { return }
        modelCheckTimer?.invalidate()
        modelCheckTimer = nil
        permCheckTimer?.invalidate()
        permCheckTimer = nil

        container.subviews.forEach { $0.removeFromSuperview() }
        permStatusRefs.removeAll()

        let size = contentSize
        let view: NSView
        switch currentStep {
        case 0: view = buildWelcomeStep(size: size)
        case 1: view = buildPrivacyStep(size: size)
        case 2: view = buildModelStatusStep(size: size)
        case 3: view = buildPermissionsStep(size: size)
        case 4: view = buildFirstQueryStep(size: size)
        default: view = NSView(frame: NSRect(origin: .zero, size: size))
        }

        container.addSubview(view)

        // Crossfade: fade in the new content
        view.wantsLayer = true
        view.layer?.opacity = 0
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.3
            view.animator().alphaValue = 1
        }

        updateDots()
        updateButtons()

        if currentStep == 4 {
            DispatchQueue.main.async { [weak self] in
                self?.queryField?.becomeFirstResponder()
            }
        }
    }

    private func updateDots() {
        for (i, dot) in dots.enumerated() {
            if i == currentStep {
                dot.layer?.backgroundColor = NSColor.controlAccentColor.cgColor
            } else if i < currentStep {
                dot.layer?.backgroundColor = NSColor.systemGreen.withAlphaComponent(0.7).cgColor
            } else {
                dot.layer?.backgroundColor = NSColor.systemGray.withAlphaComponent(0.4).cgColor
            }
        }
    }

    private func updateButtons() {
        backButton?.isHidden = currentStep == 0
        if currentStep == totalSteps - 1 {
            continueButton?.title = "Finish"
        } else if currentStep == 0 {
            continueButton?.title = "Get Started"
        } else {
            continueButton?.title = "Continue"
        }
        // On the model-status step the Continue button is disabled until the
        // model is ready or at least the daemon socket is present.
        if currentStep == 2 {
            continueButton?.isEnabled = modelIsReady || daemonRunning
        } else {
            continueButton?.isEnabled = true
        }
    }

    // MARK: - Step 0: Welcome

    private func buildWelcomeStep(size: NSSize) -> NSView {
        let v = NSView(frame: NSRect(origin: .zero, size: size))

        let logo = NSTextField(labelWithString: "🍎")
        logo.font = .systemFont(ofSize: 56)
        logo.alignment = .center
        logo.textColor = .labelColor
        logo.frame = NSRect(x: (size.width - 80) / 2, y: size.height - 90, width: 80, height: 70)
        v.addSubview(logo)

        let title = NSTextField(labelWithString: "Welcome to Bad Apple")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: size.height - 150, width: size.width, height: 34)
        v.addSubview(title)

        let body = NSTextField(wrappingLabelWithString: "Bad Apple is a local AI operating system layer for macOS. It manages on-device inference, memory, tools, voice, vision, security, and governance entirely on your Mac — without sending your data to the cloud.\n\nThis quick setup will take about a minute.")
        body.font = .systemFont(ofSize: 15)
        body.textColor = .secondaryLabelColor
        body.alignment = .center
        body.frame = NSRect(x: 24, y: size.height - 320, width: size.width - 48, height: 150)
        v.addSubview(body)

        return v
    }

    // MARK: - Step 1: Privacy

    private func buildPrivacyStep(size: NSSize) -> NSView {
        let v = NSView(frame: NSRect(origin: .zero, size: size))

        let title = NSTextField(labelWithString: "Your AI stays on your Mac")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: size.height - 50, width: size.width, height: 34)
        v.addSubview(title)

        let bullets: [(String, String, String)] = [
            ("🔒", "Local inference", "The AI model runs on your Mac's GPU. No prompts leave your machine."),
            ("☁️", "No cloud", "Bad Apple never sends your questions or data to remote servers."),
            ("📊", "No telemetry", "We don't collect analytics, track usage, or phone home."),
        ]

        var y = size.height - 120
        for (icon, heading, desc) in bullets {
            let row = makeBulletRow(icon: icon, heading: heading, desc: desc, width: size.width)
            row.frame = NSRect(x: 20, y: y - 76, width: size.width - 40, height: 76)
            v.addSubview(row)
            y -= 96
        }

        return v
    }

    private func makeBulletRow(icon: String, heading: String, desc: String, width: CGFloat) -> NSView {
        let row = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 76))

        let iconLabel = NSTextField(labelWithString: icon)
        iconLabel.font = .systemFont(ofSize: 28)
        iconLabel.alignment = .center
        iconLabel.frame = NSRect(x: 0, y: 18, width: 44, height: 44)
        row.addSubview(iconLabel)

        let headLabel = NSTextField(labelWithString: heading)
        headLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        headLabel.textColor = .labelColor
        headLabel.frame = NSRect(x: 52, y: 42, width: width - 60, height: 22)
        row.addSubview(headLabel)

        let descLabel = NSTextField(wrappingLabelWithString: desc)
        descLabel.font = .systemFont(ofSize: 13)
        descLabel.textColor = .secondaryLabelColor
        descLabel.frame = NSRect(x: 52, y: 8, width: width - 60, height: 32)
        row.addSubview(descLabel)

        return row
    }

    // MARK: - Step 2: Model status

    private func buildModelStatusStep(size: NSSize) -> NSView {
        let v = NSView(frame: NSRect(origin: .zero, size: size))

        let title = NSTextField(labelWithString: "Model Status")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: size.height - 50, width: size.width, height: 34)
        v.addSubview(title)

        let desc = NSTextField(wrappingLabelWithString: "Bad Apple needs to load its AI model before you can ask questions. This happens automatically once the daemon is running.")
        desc.font = .systemFont(ofSize: 14)
        desc.textColor = .secondaryLabelColor
        desc.alignment = .center
        desc.frame = NSRect(x: 20, y: size.height - 120, width: size.width - 40, height: 50)
        v.addSubview(desc)

        modelStatusLabel = NSTextField(wrappingLabelWithString: "Checking model status…")
        modelStatusLabel?.font = .systemFont(ofSize: 15, weight: .medium)
        modelStatusLabel?.textColor = .labelColor
        modelStatusLabel?.alignment = .center
        modelStatusLabel?.frame = NSRect(x: 20, y: size.height - 230, width: size.width - 40, height: 60)
        if let modelStatusLabel = modelStatusLabel { v.addSubview(modelStatusLabel) }

        modelProgressIndicator = NSProgressIndicator()
        modelProgressIndicator?.style = .spinning
        modelProgressIndicator?.isIndeterminate = true
        modelProgressIndicator?.isDisplayedWhenStopped = false
        modelProgressIndicator?.frame = NSRect(x: (size.width - 32) / 2, y: size.height - 280, width: 32, height: 32)
        if let modelProgressIndicator = modelProgressIndicator { v.addSubview(modelProgressIndicator) }

        checkModelStatus()

        return v
    }

    private func checkModelStatus() {
        daemonRunning = FileManager.default.fileExists(atPath: BadAppleBrain.deepSocket)
        if !daemonRunning {
            modelProgressIndicator?.stopAnimation(nil)
            modelStatusLabel?.stringValue = "The Bad Apple daemon is not running yet. If you just installed, please complete the installation from the menu bar first, then wait a moment."
            modelCheckTimer?.invalidate()
            modelCheckTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
                self?.checkModelStatus()
            }
            updateButtons()
            return
        }

        modelCheckTimer?.invalidate()
        modelCheckTimer = nil
        modelProgressIndicator?.startAnimation(nil)
        modelStatusLabel?.stringValue = "Daemon is running — checking if the model is loaded…"
        updateButtons()
        probeModelAsync()
    }

    private func probeModelAsync() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let success = self?.probeModel() ?? false
            DispatchQueue.main.async {
                guard let self = self else { return }
                if success {
                    self.modelIsReady = true
                    self.modelProgressIndicator?.stopAnimation(nil)
                    self.modelStatusLabel?.stringValue = "✓ Model ready — Bad Apple is online and ready to answer questions."
                    self.modelCheckTimer?.invalidate()
                    self.modelCheckTimer = nil
                } else {
                    self.modelIsReady = false
                    self.modelStatusLabel?.stringValue = "The model is still loading. This can take up to a minute on first launch. You can continue once the daemon is running."
                    self.modelCheckTimer?.invalidate()
                    self.modelCheckTimer = Timer.scheduledTimer(withTimeInterval: 4.0, repeats: false) { [weak self] _ in
                        self?.probeModelAsync()
                    }
                }
                self.updateButtons()
            }
        }
    }

    /// Runs `badapple -n 1 "hi"` to check whether the model is loaded and
    /// responsive. Returns true on exit code 0.
    private func probeModel() -> Bool {
        guard let binary = badAppleBinaryURL() else { return false }
        let process = Process()
        let pipe = Pipe()
        process.executableURL = binary
        process.arguments = ["-n", "1", "hi"]
        process.standardOutput = pipe
        process.standardError = pipe
        var env = ProcessInfo.processInfo.environment
        env["BADAPPLE_SOCKET_PATH"] = BadAppleBrain.deepSocket
        env["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
        process.environment = env

        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        timer.schedule(deadline: .now() + 20)
        timer.setEventHandler { [weak process] in process?.terminate() }
        timer.resume()

        do {
            try process.run()
            _ = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            timer.cancel()
            return process.terminationStatus == 0
        } catch {
            timer.cancel()
            return false
        }
    }

    private func badAppleBinaryURL() -> URL? {
        let bundled = Bundle.main.bundleURL
            .appendingPathComponent("Contents")
            .appendingPathComponent("Helpers")
            .appendingPathComponent("badapple")
        if FileManager.default.fileExists(atPath: bundled.path) {
            return bundled
        }
        for path in ["/usr/local/bin/badapple", "/opt/homebrew/bin/badapple"] {
            if FileManager.default.fileExists(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        return nil
    }

    // MARK: - Step 3: Permissions

    private func buildPermissionsStep(size: NSSize) -> NSView {
        let v = NSView(frame: NSRect(origin: .zero, size: size))

        let title = NSTextField(labelWithString: "Permissions")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: size.height - 50, width: size.width, height: 34)
        v.addSubview(title)

        let desc = NSTextField(wrappingLabelWithString: "Bad Apple needs a few system permissions to work fully. Grant them in System Settings, then return here — the status updates automatically.")
        desc.font = .systemFont(ofSize: 13)
        desc.textColor = .secondaryLabelColor
        desc.alignment = .center
        desc.frame = NSRect(x: 20, y: size.height - 100, width: size.width - 40, height: 40)
        v.addSubview(desc)

        let perms: [(name: String, desc: String, required: Bool, selector: Selector)] = [
            ("Accessibility", "Needed for UI automation and screen reading.", true, #selector(openAccessibilitySettings(_:))),
            ("Speech Recognition", "Needed for voice prompts and commands.", true, #selector(openSpeechSettings(_:))),
            ("Microphone", "Needed for voice input.", false, #selector(openMicrophoneSettings(_:))),
        ]

        var y = size.height - 120
        let rowHeight: CGFloat = 92
        for perm in perms {
            let row = makePermissionRow(
                name: perm.name, desc: perm.desc, required: perm.required,
                selector: perm.selector, width: size.width
            )
            row.frame = NSRect(x: 16, y: y - rowHeight, width: size.width - 32, height: rowHeight)
            v.addSubview(row)
            y -= rowHeight + 8
        }

        refreshPermissionStatuses()
        permCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            self?.refreshPermissionStatuses()
        }

        return v
    }

    private func makePermissionRow(name: String, desc: String, required: Bool, selector: Selector, width: CGFloat) -> NSView {
        let row = NSView(frame: NSRect(x: 0, y: 0, width: width, height: 92))
        row.wantsLayer = true
        row.layer?.cornerRadius = 10
        row.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.15).cgColor

        let nameLabel = NSTextField(labelWithString: name)
        nameLabel.font = .systemFont(ofSize: 14, weight: .semibold)
        nameLabel.textColor = .labelColor
        nameLabel.frame = NSRect(x: 14, y: 62, width: width - 150, height: 20)
        row.addSubview(nameLabel)

        let reqLabel = NSTextField(labelWithString: required ? "Required" : "Optional")
        reqLabel.font = .systemFont(ofSize: 11, weight: .medium)
        reqLabel.textColor = required ? .systemOrange : .secondaryLabelColor
        reqLabel.frame = NSRect(x: 14, y: 44, width: 80, height: 16)
        row.addSubview(reqLabel)

        let descLabel = NSTextField(wrappingLabelWithString: desc)
        descLabel.font = .systemFont(ofSize: 12)
        descLabel.textColor = .secondaryLabelColor
        descLabel.frame = NSRect(x: 14, y: 10, width: width - 150, height: 30)
        row.addSubview(descLabel)

        let statusDot = NSView(frame: NSRect(x: width - 78, y: 62, width: 12, height: 12))
        statusDot.wantsLayer = true
        statusDot.layer?.cornerRadius = 6
        statusDot.layer?.backgroundColor = NSColor.systemGray.cgColor
        row.addSubview(statusDot)

        let statusLabel = NSTextField(labelWithString: "Unknown")
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.alignment = .center
        statusLabel.frame = NSRect(x: width - 95, y: 44, width: 46, height: 16)
        row.addSubview(statusLabel)

        permStatusRefs.append(PermStatusRef(dot: statusDot, label: statusLabel))

        let openBtn = NSButton(title: "Open Settings", target: self, action: selector)
        openBtn.bezelStyle = .rounded
        openBtn.font = .systemFont(ofSize: 11)
        openBtn.frame = NSRect(x: width - 120, y: 8, width: 106, height: 24)
        row.addSubview(openBtn)

        return row
    }

    private enum PermState { case granted, denied, unknown }

    private func refreshPermissionStatuses() {
        // Accessibility — AXIsProcessTrusted() is a definitive yes/no.
        let axState: PermState = AXIsProcessTrusted() ? .granted : .unknown

        // Speech recognition — authorizationStatus() does not prompt.
        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        let speechState: PermState
        switch speechStatus {
        case .authorized: speechState = .granted
        case .denied, .restricted: speechState = .denied
        default: speechState = .unknown
        }

        // Microphone — AVCaptureDevice.authorizationStatus(for: .audio) does
        // not prompt and returns the current TCC state.
        let micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        let micState: PermState
        switch micStatus {
        case .authorized: micState = .granted
        case .denied, .restricted: micState = .denied
        case .notDetermined: micState = .unknown
        @unknown default: micState = .unknown
        }

        setPermStatus(index: 0, state: axState)
        setPermStatus(index: 1, state: speechState)
        setPermStatus(index: 2, state: micState)
    }

    private func setPermStatus(index: Int, state: PermState) {
        guard index < permStatusRefs.count else { return }
        let ref = permStatusRefs[index]
        switch state {
        case .granted:
            ref.dot.layer?.backgroundColor = NSColor.systemGreen.cgColor
            ref.label.stringValue = "Granted"
            ref.label.textColor = NSColor.systemGreen
        case .denied:
            ref.dot.layer?.backgroundColor = NSColor.systemRed.cgColor
            ref.label.stringValue = "Denied"
            ref.label.textColor = NSColor.systemRed
        case .unknown:
            ref.dot.layer?.backgroundColor = NSColor.systemGray.cgColor
            ref.label.stringValue = "Unknown"
            ref.label.textColor = .secondaryLabelColor
        }
    }

    @objc private func openAccessibilitySettings(_ sender: Any?) {
        openSettings("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
    }

    @objc private func openSpeechSettings(_ sender: Any?) {
        openSettings("x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition")
    }

    @objc private func openMicrophoneSettings(_ sender: Any?) {
        openSettings("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
    }

    private func openSettings(_ urlString: String) {
        if let url = URL(string: urlString) {
            NSWorkspace.shared.open(url)
        }
    }

    // MARK: - Step 4: First query

    private func buildFirstQueryStep(size: NSSize) -> NSView {
        let v = NSView(frame: NSRect(origin: .zero, size: size))

        let title = NSTextField(labelWithString: "Try your first query")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.alignment = .center
        title.textColor = .labelColor
        title.frame = NSRect(x: 0, y: size.height - 50, width: size.width, height: 34)
        v.addSubview(title)

        let desc = NSTextField(wrappingLabelWithString: "Ask Bad Apple anything. This is a test run — try a simple question like “What can you do?”")
        desc.font = .systemFont(ofSize: 14)
        desc.textColor = .secondaryLabelColor
        desc.alignment = .center
        desc.frame = NSRect(x: 20, y: size.height - 100, width: size.width - 40, height: 40)
        v.addSubview(desc)

        let field = NSTextField()
        field.placeholderString = "e.g. What can you do on my Mac?"
        field.bezelStyle = .roundedBezel
        field.delegate = self
        field.frame = NSRect(x: 20, y: size.height - 150, width: size.width - 120, height: 28)
        v.addSubview(field)
        queryField = field

        askButton = NSButton(title: "Ask", target: self, action: #selector(askBadApple(_:)))
        askButton?.bezelStyle = .rounded
        askButton?.frame = NSRect(x: size.width - 90, y: size.height - 150, width: 70, height: 28)
        if let askButton = askButton { v.addSubview(askButton) }

        querySpinner = NSProgressIndicator()
        querySpinner?.style = .spinning
        querySpinner?.isIndeterminate = true
        querySpinner?.isDisplayedWhenStopped = false
        querySpinner?.frame = NSRect(x: 20, y: size.height - 182, width: 20, height: 20)
        if let querySpinner = querySpinner { v.addSubview(querySpinner) }

        let responseScroll = NSScrollView(frame: NSRect(x: 20, y: 10, width: size.width - 40, height: size.height - 200))
        responseScroll.hasVerticalScroller = true
        responseScroll.autohidesScrollers = true
        responseScroll.drawsBackground = false
        responseScroll.borderType = .noBorder

        let tv = NSTextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.font = .systemFont(ofSize: 14)
        tv.textColor = .secondaryLabelColor
        tv.autoresizingMask = [.width]
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.containerSize = NSSize(width: size.width - 56, height: .greatestFiniteMagnitude)
        tv.textContainerInset = NSSize(width: 8, height: 8)
        tv.string = "Your response will appear here."
        responseScroll.documentView = tv
        v.addSubview(responseScroll)
        responseTextView = tv

        return v
    }

    @objc private func askBadApple(_ sender: Any?) {
        guard !isQuerying else { return }
        let prompt = queryField?.stringValue.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !prompt.isEmpty else { return }
        isQuerying = true
        askButton?.isEnabled = false
        querySpinner?.startAnimation(nil)
        responseTextView?.string = "Thinking…"
        responseTextView?.textColor = .secondaryLabelColor

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result = self?.runQuery(prompt: prompt) ?? "Could not reach Bad Apple."
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.isQuerying = false
                self.askButton?.isEnabled = true
                self.querySpinner?.stopAnimation(nil)
                self.responseTextView?.string = result
                self.responseTextView?.textColor = .labelColor
                self.responseTextView?.scrollToEndOfDocument(nil)
            }
        }
    }

    /// Sends the user's first query through the bundled `badapple` CLI binary,
    /// mirroring the same mechanism used by `runBadAppleCLI` in the controller.
    private func runQuery(prompt: String) -> String {
        guard let binary = badAppleBinaryURL() else {
            return "The badapple binary was not found. Please make sure Bad Apple is properly installed."
        }
        let process = Process()
        let pipe = Pipe()
        process.executableURL = binary
        process.arguments = ["--max-tokens", "300", prompt]
        process.standardOutput = pipe
        process.standardError = pipe
        var env = ProcessInfo.processInfo.environment
        env["BADAPPLE_SOCKET_PATH"] = BadAppleBrain.deepSocket
        env["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
        process.environment = env

        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        timer.schedule(deadline: .now() + 60)
        timer.setEventHandler { [weak process] in process?.terminate() }
        timer.resume()

        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            timer.cancel()
            let output = String(data: data, encoding: .utf8) ?? ""
            if process.terminationStatus != 0 && output.isEmpty {
                return "Bad Apple returned an error (exit code \(process.terminationStatus)). Make sure the daemon is running."
            }
            return output.isEmpty ? "(empty response)" : output
        } catch {
            timer.cancel()
            return "Could not run Bad Apple: \(error.localizedDescription)"
        }
    }

    // MARK: - Navigation

    @objc private func backPressed(_ sender: Any?) {
        guard currentStep > 0 else { return }
        currentStep -= 1
        renderStep()
    }

    @objc private func continuePressed(_ sender: Any?) {
        if currentStep == totalSteps - 1 {
            UserDefaults.standard.set(true, forKey: "BadAppleOnboarded")
            window?.orderOut(nil)
            return
        }
        currentStep += 1
        renderStep()
    }

    // MARK: - NSTextFieldDelegate

    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            askBadApple(nil)
            return true
        }
        return false
    }
}

// MARK: - Native chat window

/// A multi-turn chat window with streaming responses, message bubbles,
/// and persona/tier indicators. This is the consumer-grade chat surface
/// that complements the menu bar icon and CLI.
final class BadAppleChatWindow: NSObject, NSTextViewDelegate {
    // MARK: Public interface (unchanged — consumed by the menu bar controller)

    var onSubmit: ((String, @escaping (String) -> Void, @escaping () -> Void) -> Void)?
    var onNewChat: (() -> Void)?
    /// Called when the user drops or pastes an image. The callback receives
    /// the image file path, a display name, and the same append/finish streaming
    /// closures used by `onSubmit`.
    var onDescribeImage: ((String, String, @escaping (String) -> Void, @escaping () -> Void) -> Void)?
    /// Set by the AppDelegate from runtime telemetry so the chat window can
    /// show a helpful error when no VLM is loaded.
    var visionAvailable: Bool = false
    var personaName: String = "Default" {
        didSet { personaLabel?.stringValue = "Persona: \(personaName)" }
    }
    var tierName: String = "9B" {
        didSet { tierLabel?.stringValue = "Model: \(tierName)" }
    }

    func show() {
        if window == nil { buildWindow() }
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.window?.makeFirstResponder(self.inputTextView)
        }
    }

    // MARK: State

    private var window: NSWindow?
    private var contentView: ChatContentView?
    private var topBar: NSView?
    private var personaLabel: NSTextField?
    private var tierLabel: NSTextField?
    private var newChatButton: NSButton?
    private var scrollView: NSScrollView?
    private var transcriptDocument: NSView?
    private var inputBar: NSView?
    private var inputTextView: NSTextView?
    private var placeholderLabel: NSTextField?
    private var sendButton: NSButton?
    private var spinner: NSProgressIndicator?

    private var bubbleViews: [BubbleView] = []
    private var cachedHeights: [CGFloat] = []
    private var lastTranscriptWidth: CGFloat = -1
    private var currentAssistantText = ""
    private var isSubmitting = false
    private var cursorTimer: Timer?
    private var cursorVisible = true

    deinit {
        NotificationCenter.default.removeObserver(self)
        cursorTimer?.invalidate()
    }

    // MARK: Window construction

    private func buildWindow() {
        let size = NSSize(width: 720, height: 560)
        let screen = NSScreen.main ?? NSScreen.screens.first
        let frame = NSRect(
            x: (screen?.visibleFrame.midX ?? 700) - size.width / 2,
            y: (screen?.visibleFrame.midY ?? 500) - size.height / 2,
            width: size.width,
            height: size.height
        )
        let wc = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .resizable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        wc.title = "Bad Apple"
        wc.isReleasedWhenClosed = false
        wc.minSize = NSSize(width: 480, height: 400)

        let visual = ChatContentView(frame: NSRect(origin: .zero, size: size))
        visual.material = .hudWindow
        visual.state = .active
        visual.blendingMode = .behindWindow
        visual.wantsLayer = true
        visual.owner = self
        contentView = visual

        // Register for image drag-and-drop (file URLs and raw NSImage pasteboard types)
        let dragTypes: [NSPasteboard.PasteboardType] = [
            .fileURL,
            .png,
            .tiff,
            NSPasteboard.PasteboardType("public.jpeg"),
        ]
        visual.registerForDraggedTypes(dragTypes)

        // Top bar: persona + tier + new chat
        let top = NSView(frame: .zero)
        topBar = top

        personaLabel = NSTextField(labelWithString: "Persona: \(personaName)")
        personaLabel!.font = .systemFont(ofSize: 12, weight: .medium)
        personaLabel!.textColor = .secondaryLabelColor
        top.addSubview(personaLabel!)

        tierLabel = NSTextField(labelWithString: "Model: \(tierName)")
        tierLabel!.font = .systemFont(ofSize: 12, weight: .medium)
        tierLabel!.textColor = .secondaryLabelColor
        top.addSubview(tierLabel!)

        newChatButton = NSButton(title: "New Chat", target: self, action: #selector(newChat(_:)))
        newChatButton!.bezelStyle = .rounded
        newChatButton!.controlSize = .small
        top.addSubview(newChatButton!)
        visual.addSubview(top)

        // Transcript scroll view (document view is laid out manually)
        let scroll = NSScrollView(frame: .zero)
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        scroll.drawsBackground = false
        scroll.backgroundColor = .clear
        scroll.scrollerStyle = .overlay
        let clip = scroll.contentView
        clip.drawsBackground = false
        clip.backgroundColor = .clear
        let doc = NSView(frame: .zero)
        scroll.documentView = doc
        visual.addSubview(scroll)
        scrollView = scroll
        transcriptDocument = doc

        // Input bar
        let bar = NSView(frame: .zero)
        inputBar = bar

        let tv = ChatInputTextView()
        tv.font = .systemFont(ofSize: 14)
        tv.textColor = .labelColor
        tv.drawsBackground = false
        tv.isRichText = false
        tv.isEditable = true
        tv.isSelectable = true
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainerInset = NSSize(width: 8, height: 6)
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(width: 400, height: CGFloat.greatestFiniteMagnitude)
        tv.wantsLayer = true
        tv.layer?.cornerRadius = 8
        tv.layer?.backgroundColor = NSColor(white: 1, alpha: 0.08).cgColor
        tv.delegate = self
        tv.chatOwner = self
        bar.addSubview(tv)
        inputTextView = tv

        placeholderLabel = NSTextField(labelWithString: "Ask Bad Apple anything...")
        placeholderLabel!.font = .systemFont(ofSize: 14)
        placeholderLabel!.textColor = .placeholderTextColor
        placeholderLabel!.isBezeled = false
        placeholderLabel!.drawsBackground = false
        placeholderLabel!.isEditable = false
        placeholderLabel!.isSelectable = false
        bar.addSubview(placeholderLabel!)

        sendButton = NSButton(title: "Send", target: self, action: #selector(send(_:)))
        sendButton!.bezelStyle = .rounded
        sendButton!.controlSize = .small
        bar.addSubview(sendButton!)

        spinner = NSProgressIndicator()
        spinner!.style = .spinning
        spinner!.isIndeterminate = true
        spinner!.isDisplayedWhenStopped = false
        spinner!.controlSize = .small
        bar.addSubview(spinner!)

        visual.addSubview(bar)

        wc.contentView = visual
        window = wc
        relayout()
    }

    // MARK: Layout

    private func relayout() {
        guard let visual = contentView, visual.bounds.width > 0 else { return }
        let bounds = visual.bounds
        let topH: CGFloat = 40
        let inputPadding: CGFloat = 12
        let sendW: CGFloat = 64
        let spinnerW: CGFloat = 20
        let gap: CGFloat = 8

        let inputFont = inputTextView?.font ?? NSFont.systemFont(ofSize: 14)
        let lineH = max(ceil(inputFont.boundingRectForFont.height), 14)
        let maxInputLines: CGFloat = 4
        let inputWidth = max(0, bounds.width - inputPadding * 2 - gap - spinnerW - gap - sendW)
        let textWidth = max(0, inputWidth - 16) // minus horizontal text container inset (8*2)
        let attr = NSAttributedString(string: inputTextView?.string ?? "", attributes: [.font: inputFont])
        let (_, measuredH) = BadAppleChatWindow.measureText(attr, maxWidth: textWidth)
        let inputH = min(max(measuredH + 12, lineH + 12), lineH * maxInputLines + 12)
        let inputBarH = max(44, inputH + 12)

        topBar?.frame = NSRect(x: 0, y: bounds.height - topH, width: bounds.width, height: topH)
        personaLabel?.frame = NSRect(x: 16, y: 10, width: 190, height: 20)
        tierLabel?.frame = NSRect(x: 214, y: 10, width: 170, height: 20)
        let ncW: CGFloat = 94
        newChatButton?.frame = NSRect(x: bounds.width - 16 - ncW, y: 6, width: ncW, height: 28)

        inputBar?.frame = NSRect(x: 0, y: 0, width: bounds.width, height: inputBarH)
        let sendX = bounds.width - inputPadding - sendW
        sendButton?.frame = NSRect(x: sendX, y: (inputBarH - 24) / 2, width: sendW, height: 24)
        let spinnerX = sendX - gap - spinnerW
        spinner?.frame = NSRect(x: spinnerX, y: (inputBarH - spinnerW) / 2, width: spinnerW, height: spinnerW)
        let inputX = inputPadding
        let inputY = (inputBarH - inputH) / 2
        inputTextView?.frame = NSRect(x: inputX, y: inputY, width: inputWidth, height: inputH)
        inputTextView?.textContainer?.containerSize = NSSize(width: textWidth, height: CGFloat.greatestFiniteMagnitude)
        placeholderLabel?.frame = NSRect(x: inputX + 8, y: inputY + 6, width: textWidth, height: lineH)

        let scrollY = inputBarH
        let scrollH = bounds.height - topH - inputBarH
        scrollView?.frame = NSRect(x: 0, y: scrollY, width: bounds.width, height: max(0, scrollH))

        relayoutTranscript(force: false, scrollToBottom: false)
        updatePlaceholder()
    }

    private func relayoutTranscript(force: Bool, scrollToBottom: Bool) {
        guard let scroll = scrollView, let doc = transcriptDocument else { return }
        let clip = scroll.contentView
        let visibleWidth = clip.bounds.width
        guard visibleWidth > 0 else { return }
        let maxBubbleWidth = floor(visibleWidth * 0.75)
        if force || abs(visibleWidth - lastTranscriptWidth) > 0.5 {
            cachedHeights = bubbleViews.map { $0.reconfigure(maxWidth: maxBubbleWidth) }
            lastTranscriptWidth = visibleWidth
        }
        let topPad: CGFloat = 12, bottomPad: CGFloat = 12, gap: CGFloat = 8
        var total = topPad + bottomPad
        for h in cachedHeights { total += h }
        total += gap * CGFloat(max(0, bubbleViews.count - 1))
        doc.frame = NSRect(x: 0, y: 0, width: visibleWidth, height: total)
        var y = total - topPad
        for (i, bv) in bubbleViews.enumerated() {
            let h = cachedHeights[i]
            y -= h
            let bw = bv.bounds.width
            let x: CGFloat = bv.isUser ? (visibleWidth - bw - 12) : 12
            bv.frame.origin = NSPoint(x: x, y: y)
            y -= gap
        }
        let clipH = clip.bounds.height
        if scrollToBottom {
            if total > clipH { clip.setBoundsOrigin(NSPoint(x: 0, y: total - clipH)) }
            else { clip.setBoundsOrigin(.zero) }
        } else {
            let maxY = max(0, total - clipH)
            var origin = clip.bounds.origin
            if origin.y > maxY { origin.y = maxY; clip.setBoundsOrigin(origin) }
        }
    }

    // MARK: Actions

    @objc private func send(_ sender: Any?) {
        guard !isSubmitting, let tv = inputTextView else { return }
        let prompt = tv.string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        isSubmitting = true
        sendButton?.isEnabled = false
        spinner?.startAnimation(nil)
        tv.string = ""
        updatePlaceholder()
        relayout()

        appendMessage(role: "user", text: prompt)
        currentAssistantText = ""
        appendMessage(role: "assistant", text: "", streaming: true)
        startCursor()

        let append: (String) -> Void = { [weak self] chunk in
            guard let self = self else { return }
            self.currentAssistantText += chunk
            self.updateLastAssistantText(self.currentAssistantText)
        }

        let finish: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.isSubmitting = false
            self.sendButton?.isEnabled = true
            self.spinner?.stopAnimation(nil)
            self.stopCursor()
            self.window?.makeFirstResponder(self.inputTextView)
        }

        onSubmit?(prompt, append, finish)
    }

    @objc private func newChat(_ sender: Any?) {
        bubbleViews.forEach { $0.removeFromSuperview() }
        bubbleViews.removeAll()
        cachedHeights.removeAll()
        currentAssistantText = ""
        stopCursor()
        isSubmitting = false
        sendButton?.isEnabled = true
        spinner?.stopAnimation(nil)
        relayoutTranscript(force: true, scrollToBottom: false)
        onNewChat?()
        window?.makeFirstResponder(inputTextView)
    }

    func textDidChange(_ notification: Notification) {
        updatePlaceholder()
        relayout()
    }

    private func updatePlaceholder() {
        placeholderLabel?.isHidden = !(inputTextView?.string.isEmpty ?? true)
    }

    // MARK: Streaming cursor

    private func startCursor() {
        cursorVisible = true
        if let last = bubbleViews.last, !last.isUser { last.cursorVisible = true }
        cursorTimer?.invalidate()
        cursorTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.cursorVisible.toggle()
            if let last = self.bubbleViews.last, !last.isUser {
                last.cursorVisible = self.cursorVisible
                self.relayoutTranscript(force: true, scrollToBottom: true)
            }
        }
    }

    private func stopCursor() {
        cursorTimer?.invalidate()
        cursorTimer = nil
        if let last = bubbleViews.last, !last.isUser {
            last.streaming = false
            last.cursorVisible = false
        }
        relayoutTranscript(force: true, scrollToBottom: false)
    }

    // MARK: Transcript management

    private func appendMessage(role: String, text: String, streaming: Bool = false) {
        guard let doc = transcriptDocument else { return }
        let bv = BubbleView()
        bv.isUser = (role == "user")
        bv.text = text
        bv.streaming = streaming
        bv.owner = self
        doc.addSubview(bv)
        bubbleViews.append(bv)
        relayoutTranscript(force: true, scrollToBottom: true)
    }

    private func updateLastAssistantText(_ text: String) {
        guard let last = bubbleViews.last, !last.isUser else { return }
        last.text = text
        relayoutTranscript(force: true, scrollToBottom: true)
    }

    // MARK: - Image drop & paste support

    /// Checks whether a dragging session carries an image (file URL to an image
    /// or raw image pasteboard data).
    static func draggingInfoHasImage(_ sender: NSDraggingInfo) -> Bool {
        let pb = sender.draggingPasteboard
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: nil) as? [URL],
           !urls.isEmpty {
            return urls.contains { url in
                let ext = url.pathExtension.lowercased()
                return ["png", "jpg", "jpeg", "tiff", "gif", "bmp", "heic", "webp"].contains(ext)
            }
        }
        let types = pb.types ?? []
        if types.contains(.png) || types.contains(.tiff) ||
           types.contains(NSPasteboard.PasteboardType("public.jpeg")) {
            return true
        }
        return false
    }

    /// Handles an image dropped onto the chat content view.
    func handleImageDrop(_ sender: NSDraggingInfo) {
        let pb = sender.draggingPasteboard
        // Try file URL first (dragged image file from Finder)
        if let urls = pb.readObjects(forClasses: [NSURL.self], options: nil) as? [URL] {
            for url in urls {
                let ext = url.pathExtension.lowercased()
                if ["png", "jpg", "jpeg", "tiff", "gif", "bmp", "heic", "webp"].contains(ext) {
                    processImage(path: url.path, displayName: url.lastPathComponent)
                    return
                }
            }
        }
        // Fall back to raw image data
        if let image = NSImage(pasteboard: pb) {
            handlePastedImage(image)
        }
    }

    /// Handles an image pasted into the input text view.
    func handlePastedImage(_ image: NSImage) {
        guard let path = saveImageToTempFile(image) else { return }
        processImage(path: path, displayName: "pasted-image.png")
    }

    /// Saves an NSImage to a temporary PNG file and returns the path.
    private func saveImageToTempFile(_ image: NSImage) -> String? {
        guard let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return nil }
        let tempDir = NSTemporaryDirectory()
        let filename = "badapple_pasted_\(Int(Date().timeIntervalSince1970)).png"
        let path = (tempDir as NSString).appendingPathComponent(filename)
        do {
            try png.write(to: URL(fileURLWithPath: path))
            return path
        } catch {
            return nil
        }
    }

    /// Core image processing: shows a thumbnail preview as a user message,
    /// then sends the image to the daemon for description and streams the
    /// response back as an assistant message.
    func processImage(path: String, displayName: String) {
        guard !isSubmitting else { return }

        // Check vision availability before sending
        if !visionAvailable {
            appendMessage(role: "user", text: "📷 \(displayName)")
            appendMessage(
                role: "assistant",
                text: "Vision model not loaded. Run `badapple model add mlx-community/Qwen2-VL-7B-Instruct-4bit` to enable image understanding."
            )
            relayoutTranscript(force: true, scrollToBottom: true)
            return
        }

        isSubmitting = true
        sendButton?.isEnabled = false
        spinner?.startAnimation(nil)

        // Show a thumbnail preview as a user message
        appendMessage(role: "user", text: "📷 \(displayName)")
        currentAssistantText = ""
        appendMessage(role: "assistant", text: "", streaming: true)
        startCursor()

        let append: (String) -> Void = { [weak self] chunk in
            guard let self = self else { return }
            self.currentAssistantText += chunk
            self.updateLastAssistantText(self.currentAssistantText)
        }

        let finish: () -> Void = { [weak self] in
            guard let self = self else { return }
            self.isSubmitting = false
            self.sendButton?.isEnabled = true
            self.spinner?.stopAnimation(nil)
            self.stopCursor()
            self.window?.makeFirstResponder(self.inputTextView)
        }

        // If a dedicated image callback is wired, use it; otherwise fall back
        // to onSubmit with a describe prompt that triggers the describe_image tool.
        if let onDescribeImage = onDescribeImage {
            onDescribeImage(path, displayName, append, finish)
        } else {
            let prompt = "Describe this image in detail: \(path)"
            onSubmit?(prompt, append, finish)
        }
    }

    // MARK: NSTextViewDelegate

    func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) && textView === inputTextView {
            let shift = (NSApp.currentEvent?.modifierFlags ?? []).contains(.shift)
            if !shift {
                send(nil)
                return true
            }
        }
        return false
    }

    func textView(_ textView: NSTextView, clickedOnLink link: Any, at charIndex: Int) -> Bool {
        if let url = link as? URL { NSWorkspace.shared.open(url); return true }
        if let s = link as? String, let url = URL(string: s) { NSWorkspace.shared.open(url); return true }
        return false
    }

    // MARK: - Rendering helpers

    private static var userBubbleColor: NSColor { NSColor.controlAccentColor }
    private static var assistantBubbleColor: NSColor { NSColor(white: 1, alpha: 0.09) }
    private static var codeBlockColor: NSColor {
        NSColor(srgbRed: 0x1a / 255.0, green: 0x1a / 255.0, blue: 0x1a / 255.0, alpha: 1)
    }
    private static var inlineCodeColor: NSColor { NSColor.systemTeal }
    private static var inlineCodeBg: NSColor { NSColor(white: 0, alpha: 0.22) }
    private static var linkColor: NSColor { NSColor.controlAccentColor }

    private static let headerRegex = try! NSRegularExpression(pattern: "^(#{1,3})\\s+(.+)$")
    private static let inlineRegex = try! NSRegularExpression(
        pattern: "\\*\\*([^*]+)\\*\\*|\\*([^*]+)\\*|`([^`]+)`|\\[([^\\]]+)\\]\\(([^)]+)\\)"
    )

    /// Measures the wrapped size of an attributed string for a given max width.
    private static func measureText(_ attr: NSAttributedString, maxWidth: CGFloat) -> (CGFloat, CGFloat) {
        guard maxWidth > 0 else { return (0, 0) }
        let storage = NSTextStorage(attributedString: attr)
        let lm = NSLayoutManager()
        storage.addLayoutManager(lm)
        let container = NSTextContainer(containerSize: NSSize(width: maxWidth, height: CGFloat.greatestFiniteMagnitude))
        container.lineFragmentPadding = 0
        container.widthTracksTextView = false
        lm.addTextContainer(container)
        lm.ensureLayout(for: container)
        let rect = lm.usedRect(for: container)
        return (ceil(rect.width), ceil(rect.height))
    }

    private static func makeTextView(_ attr: NSAttributedString, width: CGFloat) -> NSTextView {
        let tv = NSTextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.isRichText = true
        tv.textContainerInset = .zero
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = false
        tv.textContainer?.containerSize = NSSize(width: width, height: CGFloat.greatestFiniteMagnitude)
        tv.textStorage?.setAttributedString(attr)
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = []
        return tv
    }

    private enum Segment {
        case text(String)
        case code(String, String?)
    }

    /// Splits markdown into code blocks (``` ... ```) and text segments. An
    /// unclosed code fence is emitted as a code segment so streaming code shows
    /// up correctly as it arrives.
    private static func splitMarkdown(_ text: String) -> [Segment] {
        var segments: [Segment] = []
        var inCode = false
        var lang = ""
        var codeBuf = ""
        var textBuf = ""
        func flushText() {
            if !textBuf.isEmpty { segments.append(.text(textBuf)); textBuf = "" }
        }
        for raw in text.components(separatedBy: "\n") {
            if raw.hasPrefix("```") {
                if inCode {
                    segments.append(.code(codeBuf, lang.isEmpty ? nil : lang))
                    codeBuf = ""; lang = ""; inCode = false
                } else {
                    flushText()
                    inCode = true
                    lang = String(raw.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
            } else if inCode {
                codeBuf += (codeBuf.isEmpty ? "" : "\n") + raw
            } else {
                textBuf += (textBuf.isEmpty ? "" : "\n") + raw
            }
        }
        if inCode {
            segments.append(.code(codeBuf, lang.isEmpty ? nil : lang))
        } else {
            flushText()
        }
        return segments
    }

    private static func renderUserText(_ text: String) -> NSAttributedString {
        let para = NSMutableParagraphStyle()
        para.lineSpacing = 1
        para.paragraphSpacing = 2
        return NSAttributedString(string: text, attributes: [
            .font: NSFont.systemFont(ofSize: 14),
            .foregroundColor: NSColor.white,
            .paragraphStyle: para,
        ])
    }

    /// Renders a non-code markdown segment into an attributed string with
    /// headers, bullet lists, bold, italic, inline code, and links.
    private static func renderMarkdown(_ text: String, baseColor: NSColor) -> NSMutableAttributedString {
        let result = NSMutableAttributedString()
        let bodyFont = NSFont.systemFont(ofSize: 14)
        let para = NSMutableParagraphStyle()
        para.lineSpacing = 1
        para.paragraphSpacing = 6
        var first = true
        for raw in text.components(separatedBy: "\n") {
            if !first { result.append(NSAttributedString(string: "\n")) }
            first = false
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }

            // Headers
            if let m = headerRegex.firstMatch(in: trimmed, range: NSRange(location: 0, length: (trimmed as NSString).length)) {
                let hashes = (trimmed as NSString).substring(with: m.range(at: 1))
                let content = (trimmed as NSString).substring(with: m.range(at: 2))
                let level = hashes.count
                let size: CGFloat = level == 1 ? 19 : (level == 2 ? 16 : 15)
                let hpara = NSMutableParagraphStyle()
                hpara.paragraphSpacingBefore = 10
                hpara.paragraphSpacing = 4
                let attr = inline(content, baseColor: baseColor, bodyFont: NSFont.boldSystemFont(ofSize: size))
                attr.addAttribute(.font, value: NSFont.boldSystemFont(ofSize: size), range: NSRange(location: 0, length: attr.length))
                attr.addAttribute(.foregroundColor, value: baseColor, range: NSRange(location: 0, length: attr.length))
                attr.addAttribute(.paragraphStyle, value: hpara, range: NSRange(location: 0, length: attr.length))
                result.append(attr)
                continue
            }

            // Bullet lists
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                let content = String(trimmed.dropFirst(2))
                let attr = NSMutableAttributedString(string: "•   ")
                attr.addAttribute(.font, value: bodyFont, range: NSRange(location: 0, length: attr.length))
                attr.addAttribute(.foregroundColor, value: baseColor, range: NSRange(location: 0, length: attr.length))
                attr.addAttribute(.paragraphStyle, value: para, range: NSRange(location: 0, length: attr.length))
                let body = inline(content, baseColor: baseColor, bodyFont: bodyFont)
                body.addAttribute(.paragraphStyle, value: para, range: NSRange(location: 0, length: body.length))
                attr.append(body)
                result.append(attr)
                continue
            }

            // Normal paragraph
            let attr = inline(raw, baseColor: baseColor, bodyFont: bodyFont)
            attr.addAttribute(.paragraphStyle, value: para, range: NSRange(location: 0, length: attr.length))
            result.append(attr)
        }
        return result
    }

    /// Parses inline markdown tokens: **bold**, *italic*, `code`, [text](url).
    private static func inline(_ s: String, baseColor: NSColor, bodyFont: NSFont) -> NSMutableAttributedString {
        let result = NSMutableAttributedString()
        let ns = s as NSString
        var last = 0
        inlineRegex.enumerateMatches(in: s, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match = match else { return }
            if match.range.location > last {
                result.append(plain(ns.substring(with: NSRange(location: last, length: match.range.location - last)),
                                    baseColor: baseColor, font: bodyFont))
            }
            let boldR = match.range(at: 1)
            let italR = match.range(at: 2)
            let codeR = match.range(at: 3)
            let linkTextR = match.range(at: 4)
            let linkUrlR = match.range(at: 5)
            if boldR.location != NSNotFound {
                result.append(plain(ns.substring(with: boldR), baseColor: baseColor,
                                    font: NSFont.boldSystemFont(ofSize: bodyFont.pointSize)))
            } else if italR.location != NSNotFound {
                let italicFont = NSFontManager.shared.convert(bodyFont, toHaveTrait: .italicFontMask)
                result.append(plain(ns.substring(with: italR), baseColor: baseColor, font: italicFont))
            } else if codeR.location != NSNotFound {
                result.append(NSAttributedString(string: ns.substring(with: codeR), attributes: [
                    .font: NSFont.monospacedSystemFont(ofSize: 12.5, weight: .regular),
                    .foregroundColor: inlineCodeColor,
                    .backgroundColor: inlineCodeBg,
                ]))
            } else if linkTextR.location != NSNotFound {
                let label = ns.substring(with: linkTextR)
                let url = ns.substring(with: linkUrlR)
                if let u = URL(string: url) {
                    result.append(NSAttributedString(string: label, attributes: [
                        .font: bodyFont,
                        .foregroundColor: linkColor,
                        .link: u,
                        .underlineStyle: NSUnderlineStyle.single.rawValue,
                    ]))
                } else {
                    result.append(plain(label, baseColor: baseColor, font: bodyFont))
                }
            }
            last = match.range.location + match.range.length
        }
        if last < ns.length {
            result.append(plain(ns.substring(from: last), baseColor: baseColor, font: bodyFont))
        }
        return result
    }

    private static func plain(_ s: String, baseColor: NSColor, font: NSFont) -> NSAttributedString {
        NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: baseColor])
    }

    private static func appendCursor(to attr: NSMutableAttributedString) {
        attr.append(NSAttributedString(string: "▊", attributes: [
            .font: NSFont.systemFont(ofSize: 14),
            .foregroundColor: NSColor.labelColor,
        ]))
    }

    // MARK: - Nested views

    /// Vibrant content view that re-flows the chat on resize.
    private final class ChatContentView: NSVisualEffectView {
        weak var owner: BadAppleChatWindow?
        override func setFrameSize(_ newSize: NSSize) {
            super.setFrameSize(newSize)
            owner?.relayout()
        }

        // MARK: - NSDraggingDestination (image drop)

        override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
            if BadAppleChatWindow.draggingInfoHasImage(sender) { return .copy }
            return []
        }

        override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
            if BadAppleChatWindow.draggingInfoHasImage(sender) { return .copy }
            return []
        }

        override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
            owner?.handleImageDrop(sender)
            return true
        }
    }

    /// Custom input text view that intercepts image pastes.
    private final class ChatInputTextView: NSTextView {
        weak var chatOwner: BadAppleChatWindow?

        override func paste(_ sender: Any?) {
            // Check for image data on the pasteboard before falling back to text.
            let pb = NSPasteboard.general
            if let image = NSImage(pasteboard: pb) {
                chatOwner?.handlePastedImage(image)
                return
            }
            super.paste(sender)
        }
    }

    /// A single chat message bubble. User bubbles hug their content and align
    /// right; assistant bubbles render markdown segments (text + code blocks).
    private final class BubbleView: NSView {
        var isUser = false
        var text = ""
        var streaming = false
        var cursorVisible = false
        weak var owner: BadAppleChatWindow?

        private let bg = NSView()
        private var segmentViews: [NSView] = []

        init() {
            super.init(frame: .zero)
            bg.wantsLayer = true
            addSubview(bg)
        }

        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

        /// Rebuilds the bubble for the given max width and returns its height.
        func reconfigure(maxWidth: CGFloat) -> CGFloat {
            segmentViews.forEach { $0.removeFromSuperview() }
            segmentViews = []
            let padding: CGFloat = 12
            let segSpacing: CGFloat = 8
            let innerMax = max(0, maxWidth - 2 * padding)
            bg.layer?.cornerRadius = 14

            if isUser {
                bg.layer?.backgroundColor = BadAppleChatWindow.userBubbleColor.cgColor
                let attr = BadAppleChatWindow.renderUserText(text)
                let (naturalW, _) = BadAppleChatWindow.measureText(attr, maxWidth: innerMax)
                let bubbleW = min(maxWidth, max(naturalW, 24) + 2 * padding)
                let innerW = bubbleW - 2 * padding
                let (_, h) = BadAppleChatWindow.measureText(attr, maxWidth: innerW)
                let total = h + 2 * padding
                self.frame.size = NSSize(width: bubbleW, height: total)
                bg.frame = self.bounds
                let tv = BadAppleChatWindow.makeTextView(attr, width: innerW)
                tv.frame = NSRect(x: padding, y: padding, width: innerW, height: h)
                addSubview(tv); segmentViews.append(tv)
                return total
            }

            bg.layer?.backgroundColor = BadAppleChatWindow.assistantBubbleColor.cgColor
            let segments = BadAppleChatWindow.splitMarkdown(text)

            // Empty assistant: show a thin cursor while streaming, nothing when done.
            if segments.isEmpty {
                if streaming {
                    let attr = NSMutableAttributedString(string: " ", attributes: [
                        .font: NSFont.systemFont(ofSize: 14), .foregroundColor: NSColor.labelColor,
                    ])
                    if cursorVisible { BadAppleChatWindow.appendCursor(to: attr) }
                    let (w, _) = BadAppleChatWindow.measureText(attr, maxWidth: innerMax)
                    let bubbleW = min(maxWidth, max(w, 24) + 2 * padding)
                    let innerW = bubbleW - 2 * padding
                    let (_, h) = BadAppleChatWindow.measureText(attr, maxWidth: innerW)
                    let total = h + 2 * padding
                    self.frame.size = NSSize(width: bubbleW, height: total)
                    bg.frame = self.bounds
                    let tv = BadAppleChatWindow.makeTextView(attr, width: innerW)
                    tv.frame = NSRect(x: padding, y: padding, width: innerW, height: h)
                    tv.delegate = owner
                    addSubview(tv); segmentViews.append(tv)
                    return total
                }
                self.frame.size = .zero
                bg.frame = .zero
                return 0
            }

            // Pass 1: render/measure each segment at the max inner width.
            struct R {
                var attr: NSAttributedString?
                var code: CodeBlockView?
                var cursorSuffix: String?
                var width: CGFloat
                var height: CGFloat
            }
            var rendered: [R] = []
            let lastIndex = segments.count - 1
            for (i, seg) in segments.enumerated() {
                let showCursor = streaming && cursorVisible && i == lastIndex
                switch seg {
                case .text(let t):
                    let attr = BadAppleChatWindow.renderMarkdown(t, baseColor: .labelColor)
                    if showCursor { BadAppleChatWindow.appendCursor(to: attr) }
                    let (w, h) = BadAppleChatWindow.measureText(attr, maxWidth: innerMax)
                    rendered.append(R(attr: attr, code: nil, cursorSuffix: nil, width: w, height: h))
                case .code(let code, let lang):
                    let cb = CodeBlockView()
                    let suffix: String? = showCursor ? "▊" : nil
                    let h = cb.configure(code: code, language: lang, width: innerMax, cursorSuffix: suffix)
                    rendered.append(R(attr: nil, code: cb, cursorSuffix: suffix, width: innerMax, height: h))
                }
            }

            var maxSegW: CGFloat = 0
            for r in rendered { maxSegW = max(maxSegW, r.width) }
            let bubbleW = min(maxWidth, maxSegW + 2 * padding)
            let innerW = bubbleW - 2 * padding

            // Pass 2: finalize heights for the chosen inner width.
            var heights: [CGFloat] = []
            for r in rendered {
                if let attr = r.attr {
                    let (_, h) = BadAppleChatWindow.measureText(attr, maxWidth: innerW)
                    heights.append(h)
                } else if let cb = r.code {
                    heights.append(cb.configure(code: cb.copyText, language: cb.language,
                                                width: innerW, cursorSuffix: r.cursorSuffix))
                }
            }

            var total = 2 * padding
            for (i, h) in heights.enumerated() { total += h; if i > 0 { total += segSpacing } }
            self.frame.size = NSSize(width: bubbleW, height: total)
            bg.frame = self.bounds

            // Place segments top-down.
            var y = total - padding
            for (i, r) in rendered.enumerated() {
                if i > 0 { y -= segSpacing }
                let h = heights[i]
                y -= h
                if let attr = r.attr {
                    let tv = BadAppleChatWindow.makeTextView(attr, width: innerW)
                    tv.frame = NSRect(x: padding, y: y, width: innerW, height: h)
                    tv.delegate = owner
                    addSubview(tv); segmentViews.append(tv)
                } else if let cb = r.code {
                    cb.frame = NSRect(x: padding, y: y, width: innerW, height: h)
                    addSubview(cb); segmentViews.append(cb)
                }
            }
            return total
        }
    }

    /// A fenced code block with a dark background, language label, and Copy button.
    private final class CodeBlockView: NSView {
        var copyText = ""
        var language: String?
        private weak var owner: BadAppleChatWindow?

        private let bg = NSView()
        private let textView = NSTextView()
        private let langLabel = NSTextField(labelWithString: "")
        private let copyButton = NSButton()

        init() {
            super.init(frame: .zero)
            bg.wantsLayer = true
            bg.layer?.backgroundColor = BadAppleChatWindow.codeBlockColor.cgColor
            bg.layer?.cornerRadius = 8
            addSubview(bg)

            textView.isEditable = false
            textView.isSelectable = true
            textView.drawsBackground = false
            textView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
            textView.textColor = NSColor(white: 0.9, alpha: 1)
            textView.textContainerInset = .zero
            textView.textContainer?.lineFragmentPadding = 0
            textView.textContainer?.widthTracksTextView = false
            textView.isVerticallyResizable = true
            textView.isHorizontallyResizable = false
            textView.autoresizingMask = []
            addSubview(textView)

            langLabel.font = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
            langLabel.textColor = NSColor(white: 0.55, alpha: 1)
            addSubview(langLabel)

            copyButton.isBordered = false
            copyButton.font = .systemFont(ofSize: 11, weight: .medium)
            copyButton.target = self
            copyButton.action = #selector(copyCode)
            setCopyTitle("Copy")
            addSubview(copyButton)
        }

        required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

        private func setCopyTitle(_ t: String) {
            copyButton.attributedTitle = NSAttributedString(string: t, attributes: [
                .font: NSFont.systemFont(ofSize: 11, weight: .medium),
                .foregroundColor: NSColor(white: 0.75, alpha: 1),
            ])
        }

        @objc func copyCode() {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(copyText, forType: .string)
            setCopyTitle("Copied!")
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
                self?.setCopyTitle("Copy")
            }
        }

        /// Lays out the block for the given width and returns its height.
        /// `cursorSuffix` is appended to the rendered (not copied) text so a
        /// streaming cursor can blink inside an in-progress code block.
        func configure(code: String, language: String?, width: CGFloat, cursorSuffix: String?) -> CGFloat {
            copyText = code
            self.language = language
            let pad: CGFloat = 10
            let headerH: CGFloat = 16
            let innerW = max(0, width - 2 * pad)

            langLabel.stringValue = language ?? "code"
            langLabel.sizeToFit()
            copyButton.sizeToFit()

            let display = cursorSuffix.map { code + $0 } ?? code
            let attr = NSAttributedString(string: display, attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 12, weight: .regular),
                .foregroundColor: NSColor(white: 0.9, alpha: 1),
            ])
            let (_, h) = BadAppleChatWindow.measureText(attr, maxWidth: innerW)
            textView.textContainer?.containerSize = NSSize(width: innerW, height: CGFloat.greatestFiniteMagnitude)
            textView.textStorage?.setAttributedString(attr)

            let total = pad + headerH + 4 + h + pad
            self.frame.size = NSSize(width: width, height: total)
            bg.frame = self.bounds
            langLabel.frame = NSRect(x: pad, y: total - pad - langLabel.bounds.height,
                                     width: min(langLabel.bounds.width, innerW), height: langLabel.bounds.height)
            copyButton.frame = NSRect(x: width - pad - copyButton.bounds.width,
                                      y: total - pad - copyButton.bounds.height,
                                      width: copyButton.bounds.width, height: copyButton.bounds.height)
            textView.frame = NSRect(x: pad, y: pad, width: innerW, height: h)
            return total
        }
    }
}

// MARK: - Chat history window

final class ChatHistoryWindow: NSObject {
    private var window: NSWindow?
    private var textView: NSTextView?
    private var timer: Timer?
    private let conversationPath = "/var/lib/bad_apple/conversation.json"

    func show() {
        if window == nil {
            let contentSize = NSSize(width: 640, height: 480)
            let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
            let origin = NSPoint(
                x: screenFrame.midX - contentSize.width / 2,
                y: screenFrame.midY - contentSize.height / 2
            )
            let text = NSTextView()
            text.isEditable = false
            text.isSelectable = true
            text.font = NSFont.systemFont(ofSize: 13)
            text.autoresizingMask = [.width, .height]
            text.textContainer?.widthTracksTextView = true
            text.textContainer?.containerSize = NSSize(width: contentSize.width, height: .greatestFiniteMagnitude)

            let scroll = NSScrollView(frame: NSRect(origin: .zero, size: contentSize))
            scroll.hasVerticalScroller = true
            scroll.autoresizingMask = [.width, .height]
            scroll.documentView = text

            let wc = NSWindow(
                contentRect: NSRect(origin: origin, size: contentSize),
                styleMask: [.titled, .closable, .resizable],
                backing: .buffered,
                defer: false
            )
            wc.title = "Bad Apple Chat History"
            wc.contentView = scroll
            wc.isReleasedWhenClosed = false
            wc.makeKeyAndOrderFront(nil)

            window = wc
            textView = text
            timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
                self?.reload()
            }
        }
        window?.makeKeyAndOrderFront(nil)
        reload()
    }

    private func reload() {
        guard let textView = textView else { return }
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: conversationPath))
            let messages = try JSONSerialization.jsonObject(with: data) as? [[String: String]] ?? []
            let lines = messages.map { m -> String in
                let role = m["role"] ?? "?"
                let content = m["content"] ?? ""
                switch role {
                case "system": return "🍎 Bad Apple (system)\n\(content)"
                case "user": return "You\n\(content)"
                case "assistant": return "Bad Apple\n\(content)"
                default: return "\(role)\n\(content)"
                }
            }
            let transcript = lines.joined(separator: "\n\n---\n\n")
            textView.string = transcript
        } catch {
            textView.string = "No chat history available yet."
        }
    }
}

private extension String {
    func truncated(to length: Int) -> String {
        count > length ? String(prefix(length)) + "..." : self
    }
}
