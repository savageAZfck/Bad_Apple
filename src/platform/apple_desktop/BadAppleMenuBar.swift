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

private let badAppleVoiceLogURL = URL(fileURLWithPath: "/tmp/badapple_voice_debug.log")

private func badAppleVoiceLog(_ message: String) {
    let stamp = ISO8601DateFormatter().string(from: Date())
    let line = "\(stamp) \(message)\n"
    do {
        if !FileManager.default.fileExists(atPath: badAppleVoiceLogURL.path) {
            try line.write(to: badAppleVoiceLogURL, atomically: true, encoding: .utf8)
        } else {
            let data = line.data(using: .utf8)!
            if let fh = try? FileHandle(forWritingTo: badAppleVoiceLogURL) {
                _ = try? fh.seekToEnd()
                fh.write(data)
                fh.closeFile()
            }
        }
    } catch {
        NSLog("badAppleVoiceLog failed: %@", error.localizedDescription)
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
        let searchPaths = [
            bundleFrameworks + "/libbad_apple.dylib",
            bundleFrameworks + "/../libbad_apple.dylib",
            (Bundle.main.bundlePath as NSString).deletingLastPathComponent + "/libbad_apple.dylib",
            (Bundle.main.bundlePath as NSString).deletingLastPathComponent + "/../libbad_apple.dylib",
            "libbad_apple.dylib",
            "./libbad_apple.dylib",
            "../libbad_apple.dylib",
            "target/release/libbad_apple.dylib",
            "target/debug/libbad_apple.dylib",
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

        bad_apple_init = unsafeBitCast(dlsym(h, "bad_apple_init"), to: BadAppleInitFn.self)
        bad_apple_free = unsafeBitCast(dlsym(h, "bad_apple_free"), to: BadAppleFreeFn.self)
        bad_apple_get_active_pursuits = unsafeBitCast(dlsym(h, "bad_apple_get_active_pursuits"), to: BadAppleGetActivePursuitsFn.self)
        bad_apple_push_pursuit = unsafeBitCast(dlsym(h, "bad_apple_push_pursuit"), to: BadApplePushPursuitFn.self)
        bad_apple_get_apple_latency_us = unsafeBitCast(dlsym(h, "bad_apple_get_apple_latency_us"), to: BadAppleGetAppleLatencyUsFn.self)
        bad_apple_generate_text = unsafeBitCast(dlsym(h, "bad_apple_generate_text"), to: BadAppleGenerateTextFn.self)
        bad_apple_free_string = unsafeBitCast(dlsym(h, "bad_apple_free_string"), to: BadAppleFreeStringFn.self)

        context = bad_apple_init?(nil)
        if context == nil {
            lastError = "bad_apple_init() returned nil"
            // Continue: the bridge may still be useful for the daemon client.
        }

        // Now that the Rust core is resident, load and initialize the Swift bridge.
        let bridgeSearchPaths = [
            bundleFrameworks + "/libBadAppleBridge.dylib",
            bundleFrameworks + "/../libBadAppleBridge.dylib",
            "libBadAppleBridge.dylib",
            "./libBadAppleBridge.dylib",
            "../libBadAppleBridge.dylib",
            "target/release/libBadAppleBridge.dylib",
            "target/debug/libBadAppleBridge.dylib",
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

    func enqueue(_ url: URL, completion: ((Bool) -> Void)? = nil) {
        if let player = try? AVAudioPlayer(contentsOf: url) {
            player.delegate = self
            player.prepareToPlay()
            let item = PlayItem(url: url, player: player, completion: completion)
            if current == nil {
                start(item)
            } else {
                pending.append(item)
            }
        } else {
            badAppleVoiceLog("PiperTTSPlayback: AVAudioPlayer failed for \(url.path), falling back to afplay")
            let item = PlayItem(url: url, completion: completion)
            item.afplayTask = makeAfplayTask(item)
            if current == nil {
                start(item)
            } else {
                pending.append(item)
            }
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
        current?.player?.stop()
        current?.afplayTask?.terminate()
        previous.forEach { $0.player?.stop(); $0.afplayTask?.terminate() }
        pending.removeAll()
        current = nil
        previous.removeAll()
    }

    private func start(_ item: PlayItem) {
        if let player = item.player {
            player.volume = 1.0
            guard player.play() else {
                badAppleVoiceLog("PiperTTSPlayback: play() failed for \(item.url.path)")
                notifyCompletion(item.completion, success: false)
                advanceIfIdle()
                return
            }
            current = item
            startTimer()
            badAppleVoiceLog("PiperTTSPlayback: playing \(item.url.path)")
        } else if let task = item.afplayTask {
            current = item
            do {
                try task.run()
            } catch {
                badAppleVoiceLog("PiperTTSPlayback afplay launch failed: \(error.localizedDescription)")
                notifyCompletion(item.completion, success: false)
                current = nil
                advanceIfIdle()
            }
        }
    }

    private func makeAfplayTask(_ item: PlayItem) -> Process {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        let uid = getuid()
        task.arguments = ["asuser", "\(uid)", "/usr/bin/afplay", item.url.path]
        task.terminationHandler = { [weak self, weak item] task in
            guard let self = self, let item = item else { return }
            let code = task.terminationStatus
            if code != 0 {
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
        guard let item = current, let player = item.player, player.isPlaying else {
            return
        }
        let remaining = player.duration - player.currentTime
        guard !pending.isEmpty, remaining <= crossfadeDuration else { return }

        let next = pending.removeFirst()
        if let nextPlayer = next.player {
            nextPlayer.volume = 0.0
            guard nextPlayer.play() else {
                badAppleVoiceLog("PiperTTSPlayback: next play() failed for \(next.url.path)")
                pending.insert(next, at: 0)
                return
            }
            player.setVolume(0.0, fadeDuration: crossfadeDuration)
            nextPlayer.setVolume(1.0, fadeDuration: crossfadeDuration)
            previous.append(item)
            current = next
            badAppleVoiceLog("PiperTTSPlayback: crossfading to \(next.url.path)")
        } else if next.afplayTask != nil {
            // Can't crossfade to an afplay-backed item; play it sequentially.
            pending.insert(next, at: 0)
        }
    }

    private func finishCurrent(_ item: PlayItem, success: Bool) {
        if current === item {
            current = nil
            timer?.invalidate()
            timer = nil
        } else if let idx = previous.firstIndex(where: { $0 === item }) {
            previous.remove(at: idx)
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
        let completion: ((Bool) -> Void)?
    }
    private var queue: [QueueItem] = []
    private let queueLock = NSLock()
    private var isProcessing = false

    func stop() {
        queueLock.lock()
        queue.removeAll()
        isProcessing = false
        queueLock.unlock()
        playback.stop()
    }

    static let defaultVoice = "en_US-amy-medium"
    static let availableVoices = [
        "en_US-amy-medium",
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

        queueLock.lock()
        for (index, chunk) in chunks.enumerated() {
            let isLast = index == chunks.count - 1
            itemCounter += 1
            queue.append(QueueItem(text: chunk, voice: voice, id: itemCounter, completion: isLast ? completion : nil))
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
            queueLock.unlock()
            badAppleVoiceLog("PiperTTS queue empty, stopping")
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
                    guard let self = self else { return }
                    self.playback.enqueue(wavURL, completion: item.completion)
                    self.processNext()
                }
            } catch {
                badAppleVoiceLog("PiperTTS synthesize error: \(error.localizedDescription)")
                item.completion?(false)
                DispatchQueue.main.async { [weak self] in
                    self?.processNext()
                }
            }
        }
    }

    private func synthesize(_ text: String, voice: String) throws -> URL {
        let request: [String: Any] = ["text": text, "voice": voice]
        let data = try JSONSerialization.data(withJSONObject: request, options: [])
        let response = try unixSocketRequest(data)
        guard let json = try JSONSerialization.jsonObject(with: response) as? [String: Any] else {
            throw NSError(domain: "PiperTTS", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid JSON"])
        }
        guard (json["ok"] as? Bool) == true, let wavPath = json["wav_path"] as? String else {
            let err = json["error"] as? String ?? "unknown"
            throw NSError(domain: "PiperTTS", code: 3, userInfo: [NSLocalizedDescriptionKey: err])
        }
        return URL(fileURLWithPath: wavPath)
    }

    private func unixSocketRequest(_ data: Data) throws -> Data {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(domain: "PiperTTS", code: 11, userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8)
        let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
        guard pathBytes.count < maxPath else {
            throw NSError(domain: "PiperTTS", code: 12, userInfo: [NSLocalizedDescriptionKey: "socket path too long"])
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
            throw NSError(domain: "PiperTTS", code: 13, userInfo: [NSLocalizedDescriptionKey: "connect() failed: \(errno)"])
        }

        _ = data.withUnsafeBytes { write(fd, $0.baseAddress!, $0.count) }
        var newline: UInt8 = 0x0A
        _ = write(fd, &newline, 1)

        var response = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
        defer { buffer.deallocate() }
        while true {
            let n = read(fd, buffer, 4096)
            if n <= 0 { break }
            response.append(buffer, count: n)
            if response.contains(0x0A) { break }
        }
        return response
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
        state = .requestingPermission
        badAppleVoiceLog("requesting speech recognition authorization")
        SFSpeechRecognizer.requestAuthorization { [weak self] speechStatus in
            guard let self = self, self.enabled else { return }
            badAppleVoiceLog("speech authorization status: \(speechStatus.rawValue)")
            guard speechStatus == .authorized else {
                self.failClosed("speech recognition permission denied")
                return
            }
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                guard let self = self, self.enabled else { return }
                badAppleVoiceLog("microphone access: \(granted)")
                guard granted else {
                    self.failClosed("microphone permission denied")
                    return
                }
                DispatchQueue.main.async { self.configureOnDeviceRecognizer() }
            }
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

    /// Split the response into chilled, beachy chunks. US voices stay relaxed
    /// with a slightly slower rate and a soft, natural pitch.
    private func prosodyChunks(from text: String) -> [ProsodyChunk] {
        var chunks: [ProsodyChunk] = []
        var current = ""

        func flush(_ postDelay: TimeInterval = 0.0, rate: Float = 0.46, pitch: Float = 0.96) {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                current = ""
                return
            }
            chunks.append(ProsodyChunk(text: trimmed, rate: rate, pitch: pitch, postDelay: postDelay))
            current = ""
        }

        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)

            if c == "…" || (c == "." && i + 1 < chars.count && chars[i + 1] == "." && i + 2 < chars.count && chars[i + 2] == ".") {
                // ellipsis: slow trailing breath
                flush(0.15, rate: 0.44, pitch: 0.95)
                if c == "." { i += 2 }
            } else if c == "?" {
                flush(0.15, rate: 0.46, pitch: 1.00)
            } else if c == "!" {
                flush(0.15, rate: 0.48, pitch: 1.00)
            } else if c == "." || c == "\n" {
                // only end a sentence if the next char is whitespace or we are at the end
                let next = i + 1 < chars.count ? chars[i + 1] : nil
                if next == nil || next!.isWhitespace || next! == "\n" {
                    flush(0.12)
                }
            }

            i += 1
        }

        flush(0.10)
        return chunks.isEmpty ? [ProsodyChunk(text: text, rate: 0.46, pitch: 0.96, postDelay: 0.10)] : chunks
    }

    private var usePiperTTS: Bool {
        UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? true
    }

    private var selectedPiperVoice: String {
        let raw = UserDefaults.standard.string(forKey: "BadAppleTTSVoice") ?? PiperTTSClient.defaultVoice
        return PiperTTSClient.availableVoices.contains(raw) ? raw : PiperTTSClient.defaultVoice
    }

    /// Stop any in-flight audio so a new request does not stack on old output.
    func stopAllAudio() {
        synthesizer.stopSpeaking(at: .immediate)
        PiperTTSClient.shared.stop()
    }

    /// Queue a single streamed sentence chunk without stopping any in-flight audio.
    /// This keeps responses smooth while the model is still generating the next chunk.
    func speakChunk(_ text: String) {
        let spoken = text.trimmingCharacters(in: .whitespacesAndNewlines)
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
        let voice = bestVoice()
        let chunks = prosodyChunks(from: text)
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
        let spoken = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard enabled else { return }
        guard !spoken.isEmpty else {
            scheduleRestart(after: 0.1)
            return
        }
        state = .speaking

        // Stop any in-flight audio so we do not stack responses.
        synthesizer.stopSpeaking(at: .immediate)
        PiperTTSClient.shared.stop()

        if usePiperTTS {
            let voice = selectedPiperVoice
            badAppleVoiceLog("speak using Piper TTS (id=\(id), voice=\(voice))")
            state = .speaking
            PiperTTSClient.shared.speak(spoken, voice: voice) { [weak self] success in
                DispatchQueue.main.async {
                    guard let self = self, self.enabled, self.currentSpeakID == id else { return }
                    if success {
                        self.scheduleRestart(after: 0.25)
                    } else {
                        badAppleVoiceLog("Piper TTS failed, falling back to Apple TTS (id=\(id))")
                        self.speakWithApple(spoken, id: id)
                    }
                }
            }
        } else {
            badAppleVoiceLog("speak using Apple TTS (id=\(id))")
            speakWithApple(spoken, id: id)
        }
    }

    private func speakWithApple(_ text: String, id: Int) {
        guard enabled, currentSpeakID == id else { return }
        let voice = bestVoice()
        let chunks = prosodyChunks(from: text)
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
    }

    func resumeAfterFailure() {
        guard enabled else { return }
        scheduleRestart(after: 0.5)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        pendingSpeechUtterances = max(0, pendingSpeechUtterances - 1)
        if pendingSpeechUtterances == 0 {
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
    private let actionExecutor = BadAppleActionExecutor()
    private let chatHistoryWindow = ChatHistoryWindow()
    private var streamedTokenCount = 0
    private var lastPrompt = ""
    private var lastError: String?
    private var isSubmittingVoicePrompt = false
    private var lastSpoken: String?
    private var openMenuCount = 0
    private var needsMenuRebuild = false
    private let memoryGovernor = MemoryGovernor()
    private var memoryUsedGB = 0.0
    private var memoryTotalGB = 0.0
    private var memoryPressure = "normal"
    private var activeModels: [String] = ["main_9b"]
    private var lastTelemetryTime: TimeInterval = 0
    private var lastRuntimeStatus: [String: Any] = [:]
    private var lastRuntimeReachable = false
    private var autoPurgeEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleAutoPurge") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleAutoPurge") }
    }
    private var fastTierOnly: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleFastTierOnly") as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleFastTierOnly") }
    }
    private var autopilotEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleAutopilot") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleAutopilot") }
    }
    private var focusEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleFocusEnabled") as? Bool ?? false }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleFocusEnabled") }
    }
    private let voiceHUD = BadAppleVoiceHUD()
    private let voiceOnboarding = BadAppleVoiceOnboarding()
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
    private var aquaHelperProcess: Process?
    private let splash = BadAppleSplashWindow()
    private var runtimeState: [String: Any] {
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
            Task {
                do {
                    _ = try await self.runBadAppleCLIStreaming(
                        prompt: prompt,
                        socketPath: BadAppleBrain.deepSocket,
                        maxTokens: 300
                    ) { chunk in
                        DispatchQueue.main.async { append(chunk) }
                    }
                } catch {
                    DispatchQueue.main.async { append("\n\nError: \(error.localizedDescription)") }
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

        if UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false {
            PiperTTSClient.shared.warmup()
        }

        controlCenter.onVoiceToggle = { [weak self] enabled in
            guard let self = self else { return }
            UserDefaults.standard.set(enabled, forKey: "BadAppleVoiceEnabled")
            self.voiceHost.setEnabled(enabled)
            self.rebuildMenu()
        }

        NSApp.servicesProvider = self

        // Pause voice when the screen locks or the Mac sleeps; resume on unlock/wake.
        let dnc = DistributedNotificationCenter.default
        dnc.addObserver(self, selector: #selector(screenLocked), name: Notification.Name("com.apple.screenIsLocked"), object: nil)
        dnc.addObserver(self, selector: #selector(screenUnlocked), name: Notification.Name("com.apple.screenIsUnlocked"), object: nil)
        let workspace = NSWorkspace.shared.notificationCenter
        workspace.addObserver(self, selector: #selector(willSleep), name: NSWorkspace.willSleepNotification, object: nil)
        workspace.addObserver(self, selector: #selector(didWake), name: NSWorkspace.didWakeNotification, object: nil)
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

    func applicationWillTerminate(_ notification: Notification) {
        voiceHost.setEnabled(false)
        timer?.invalidate()
        memoryGovernor.stop()
        stopAquaHelper()
    }

    private func findAquaHelper() -> (python: URL, script: URL)? {
        // Prefer a full platform install under ~/.bad_apple/bad_apple-<version>/bad_apple.
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let badAppleDir = home.appendingPathComponent(".bad_apple")
        if let versions = try? fm.contentsOfDirectory(at: badAppleDir, includingPropertiesForKeys: nil, options: .skipsHiddenFiles) {
            for versionDir in versions.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
                let script = versionDir.appendingPathComponent("bad_apple/badapple_aqua_helper.py")
                let python = versionDir.appendingPathComponent("bad_apple/.venv/bin/python3")
                if fm.isExecutableFile(atPath: python.path) && fm.fileExists(atPath: script.path) {
                    return (python, script)
                }
            }
        }

        // Fall back to the script bundled in the app bundle; it only needs the system python.
        if let bundledScript = Bundle.main.url(forResource: "badapple_aqua_helper", withExtension: "py") {
            return (URL(fileURLWithPath: "/usr/bin/python3"), bundledScript)
        }

        return nil
    }

    private func startAquaHelper() {
        guard aquaHelperProcess == nil else { return }
        guard let helper = findAquaHelper() else {
            print("Aqua helper script not found", terminator: "\n")
            return
        }
        let process = Process()
        process.executableURL = helper.python
        process.arguments = ["-u", helper.script.path]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
        env["BADAPPLE_AQUA_SOCKET"] = "/var/run/badapple/aqua_helper.sock"
        env["BADAPPLE_APP_BUNDLE"] = Bundle.main.bundlePath
        process.environment = env
        do {
            try process.run()
            aquaHelperProcess = process
            print("Started Aqua helper for Shortcuts", terminator: "\n")
        } catch {
            print("Failed to start Aqua helper: \(error)", terminator: "\n")
        }
    }

    private func stopAquaHelper() {
        guard let process = aquaHelperProcess, process.isRunning else { return }
        process.terminate()
        aquaHelperProcess = nil
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
        Task {
            do {
                let output = try await runBadAppleCLI(prompt: "runtime status", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
                if let data = output.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    await MainActor.run {
                        if let active = json["active_models"] as? [String] {
                            self.activeModels = active
                        }
                        self.lastRuntimeStatus = json
                        self.lastRuntimeReachable = true
                        self.updateStatusIcon()
                        self.rebuildMenu()
                        self.splash.update(status: json)
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
        Task {
            _ = try? await runBadAppleCLI(prompt: "flush vram", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
        }
    }

    private func unloadOptionalModels() {
        badAppleVoiceLog("Memory critical: unloading optional models")
        Task {
            _ = try? await runBadAppleCLI(prompt: "unload all models", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
        }
    }

    @objc private func purgeVRAM() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "flush vram", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.refreshTelemetry() }
        }
    }

    @objc private func unloadModels() {
        Task {
            _ = try? await runBadAppleCLI(prompt: "unload all models", socketPath: BadAppleBrain.deepSocket, maxTokens: 32)
            await MainActor.run { self.refreshTelemetry() }
        }
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
        voiceHost.stopAllAudio()
        rebuildMenu()

        // Voice mode switch commands are handled without a daemon call.
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectivePrompt = trimmed

        // Fast local action resolver for common voice commands (open workspace,
        // launch app, create directory).  This keeps the assistant responsive
        // even when the cognitive substrate is slow or unavailable.
        if let local = BadAppleActionResolver.resolve(effectivePrompt) {
            badAppleVoiceLog("submitVoicePrompt resolved local action: \(local)")
            actionExecutor.confirmAndExecute(local)
            return
        }

        // Fallback: ask the on-device daemon via the bundled badapple CLI.
        // The CLI streams sentence chunks as they are generated, so TTS starts
        // while the 9B model is still finishing the rest of the response.
        var extraArgs: [String] = []
        if selectedPersona != "default" { extraArgs += ["--persona", selectedPersona] }
        if roastEnabled { extraArgs += ["--roast"] }

        let socket = BadAppleBrain.deepSocket
        let maxTokens = 300
        Task {
            do {
                let finalText = try await runBadAppleCLIStreaming(prompt: effectivePrompt, socketPath: socket, maxTokens: maxTokens, extraArgs: extraArgs) { chunk in
                    DispatchQueue.main.async {
                        self.streamedTokenCount += chunk.count
                        self.rebuildMenu()
                    }
                }
                await MainActor.run {
                    self.completeVoiceResponse(finalText)
                    self.isSubmittingVoicePrompt = false
                }
            } catch {
                badAppleVoiceLog("submitVoicePrompt error: \(error)")
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.voiceHost.resumeAfterFailure()
                    self.rebuildMenu()
                    self.isSubmittingVoicePrompt = false
                }
            }
        }
    }

    @objc private func showChatHistory() {
        chatHistoryWindow.show()
    }

    @objc private func newChat() {
        Task {
            do {
                let response = try await runBadAppleCLI(prompt: "new chat", socketPath: BadAppleBrain.deepSocket, maxTokens: 80)
                await MainActor.run {
                    self.lastPrompt = "new chat"
                    self.lastError = nil
                    self.streamedTokenCount = 0
                    self.voiceHost.speak(response)
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

    @objc private func toggleRoast() {
        roastEnabled.toggle()
        rebuildMenu()
    }

    @objc private func selectPersona(_ sender: NSMenuItem) {
        guard let name = sender.representedObject as? String else { return }
        selectedPersona = name
        rebuildMenu()
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
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let outputPipe = Pipe()
                process.executableURL = binary
                process.arguments = extraArgs + ["--max-tokens", String(maxTokens), prompt]
                process.standardOutput = outputPipe
                process.standardError = outputPipe
                var environment = ProcessInfo.processInfo.environment
                environment["BADAPPLE_SOCKET_PATH"] = socketPath
                environment["BADAPPLE_SLICKS_KEY_PATH"] = BadAppleBrain.keyPath
                environment["BADAPPLE_STREAM_JSON"] = "1"
                environment["BADAPPLE_VOICE"] = "1"
                process.environment = environment

                let sync = NSLock()
                var timeoutTimer: Timer?
                timeoutTimer = Timer.scheduledTimer(withTimeInterval: 120.0, repeats: false) { _ in
                    badAppleVoiceLog("runBadAppleCLIStreaming: timeout, terminating")
                    process.terminate()
                }

                var buffer = ""
                var fullText = ""
                var resumed = false

                func finish(result: Result<String, Error>) {
                    sync.lock()
                    guard !resumed else { sync.unlock(); return }
                    resumed = true
                    timeoutTimer?.invalidate()
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

                var timeoutTimer: Timer?
                timeoutTimer = Timer.scheduledTimer(withTimeInterval: timeout, repeats: false) { _ in
                    badAppleVoiceLog("runBadAppleCLI: timeout, terminating")
                    process.terminate()
                }

                do {
                    try process.run()
                    process.waitUntilExit()
                    timeoutTimer?.invalidate()
                    let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: data, encoding: .utf8) ?? ""
                    badAppleVoiceLog("runBadAppleCLI: exit=\(process.terminationStatus) output=\(output.prefix(200))")
                    if process.terminationStatus != 0, output.isEmpty {
                        throw BadAppleMenuBarError("The Bad Apple helper exited with code \(process.terminationStatus).")
                    }
                    continuation.resume(returning: output)
                } catch {
                    timeoutTimer?.invalidate()
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

                var timeoutTimer: Timer?
                timeoutTimer = Timer.scheduledTimer(withTimeInterval: 120.0, repeats: false) { _ in
                    badAppleVoiceLog("runBadAppleCLI benchmark: timeout, terminating")
                    process.terminate()
                }

                do {
                    try process.run()
                    process.waitUntilExit()
                    timeoutTimer?.invalidate()
                    let data = outputPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: data, encoding: .utf8) ?? ""
                    badAppleVoiceLog("runBadAppleCLI: exit=\(process.terminationStatus) output=\(output.prefix(200))")
                    if process.terminationStatus != 0, output.isEmpty {
                        throw BadAppleMenuBarError("The Bad Apple helper exited with code \(process.terminationStatus).")
                    }
                    continuation.resume(returning: output)
                } catch {
                    timeoutTimer?.invalidate()
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private struct BadAppleMenuBarError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { self.errorDescription = message }
    }

    private func completeVoiceResponse(_ response: String) {
        badAppleVoiceLog("completeVoiceResponse: \(response.prefix(200))")
        let parsed = BadAppleActionParser.parse(response)
        lastSpoken = parsed.spoken
        voiceHUD.updateResponse(parsed.spoken)
        badAppleVoiceLog("parsed actions: \(parsed.actions.count) error: \(parsed.error ?? "nil") spoken: \(parsed.spoken)")
        voiceHost.speak(parsed.spoken)
        if let parseError = parsed.error {
            actionExecutor.showParsingFailure(parseError)
        } else {
            for action in parsed.actions {
                actionExecutor.confirmAndExecute(action)
            }
        }
        rebuildMenu()
    }

    private func updateStatusIcon() {
        guard let button = statusItem?.button else { return }
        let dotColor: NSColor
        if !lastRuntimeReachable {
            dotColor = .systemRed
        } else {
            let mode = lastRuntimeStatus["mode"] as? String ?? runtimeState["mode"] as? String ?? "UNKNOWN"
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
        let statusText = lastRuntimeReachable ? (lastRuntimeStatus["mode"] as? String ?? "unknown") : "offline"
        button.toolTip = "Bad Apple daemon status: \(statusText)"
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
        let header = NSMenuItem(title: "Bad Apple — 9B MLX + RAG", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)

        let voiceStatus = NSMenuItem(title: voiceHost.state.label.truncated(to: 90), action: nil, keyEquivalent: "")
        voiceStatus.isEnabled = false
        menu.addItem(voiceStatus)

        if !lastPrompt.isEmpty {
            let lastCommand = NSMenuItem(title: "Last command: \(lastPrompt.truncated(to: 60))", action: nil, keyEquivalent: "")
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

        let mode = NSMenuItem(title: "Brain: Qwen3.5 9B MLX + RAG", action: nil, keyEquivalent: "")
        mode.isEnabled = false
        menu.addItem(mode)
        let runtime = runtimeState
        let runtimeMode = runtime["mode"] as? String ?? "UNKNOWN"
        let runtimeItem = NSMenuItem(title: "Runtime: \(runtimeMode)", action: nil, keyEquivalent: "")
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
        let purgeItem = NSMenuItem(title: "Purge VRAM", action: #selector(purgeVRAM), keyEquivalent: "")
        purgeItem.toolTip = "Release cached GPU memory and Metal allocations."
        performanceMenu.addItem(purgeItem)
        let unloadItem = NSMenuItem(title: "Unload Optional Models", action: #selector(unloadModels), keyEquivalent: "")
        unloadItem.toolTip = "Unload vision, image, and optional models to free RAM."
        performanceMenu.addItem(unloadItem)
        let performanceFastTierItem = NSMenuItem(title: "Fast Tier Only", action: #selector(toggleFastTier), keyEquivalent: "")
        performanceFastTierItem.state = fastTierOnly ? .on : .off
        performanceFastTierItem.toolTip = "Route simple queries to the 0.5B fast model."
        performanceMenu.addItem(performanceFastTierItem)
        let autoPurgeItem = NSMenuItem(title: "Auto-Purge on Critical", action: #selector(toggleAutoPurge), keyEquivalent: "")
        autoPurgeItem.state = autoPurgeEnabled ? .on : .off
        autoPurgeItem.toolTip = "Automatically purge VRAM when memory pressure is critical."
        performanceMenu.addItem(autoPurgeItem)
        let performanceParent = NSMenuItem(title: "Performance", action: nil, keyEquivalent: "")
        performanceParent.submenu = performanceMenu
        menu.addItem(performanceParent)
        menu.addItem(NSMenuItem.separator())

        let privateMode = runtime["private_mode"] as? Bool ?? false
        let privacyMenu = NSMenu(title: "Privacy")
        let privateToggle = NSMenuItem(title: "Private Mode", action: #selector(togglePrivateMode), keyEquivalent: "")
        privateToggle.state = privateMode ? .on : .off
        privateToggle.toolTip = "Pause persistence and audit logging for this session."
        privacyMenu.addItem(privateToggle)
        let autopilotItem = NSMenuItem(title: "Autopilot", action: #selector(toggleAutopilot), keyEquivalent: "")
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
            let resumeItem = NSMenuItem(title: "Resume Bad Apple", action: #selector(resetKillSwitch), keyEquivalent: "")
            resumeItem.toolTip = "Reset the kill switch and resume generation and tools."
            menu.addItem(resumeItem)
        } else {
            let stopItem = NSMenuItem(title: "Emergency Stop", action: #selector(engageKillSwitch), keyEquivalent: "")
            stopItem.toolTip = "Cancel generation, stop ambient capture, and block tools."
            menu.addItem(stopItem)
        }
        if !lastRuntimeReachable {
            let restartDaemonItem = NSMenuItem(title: "Restart Daemon", action: #selector(restartDaemon), keyEquivalent: "")
            restartDaemonItem.toolTip = "Unload and reload the Bad Apple launchd daemon."
            menu.addItem(restartDaemonItem)
        }
        let systemHealthItem = NSMenuItem(title: "System Health...", action: #selector(showSystemHealth), keyEquivalent: "")
        systemHealthItem.toolTip = "Show the full runtime status JSON."
        menu.addItem(systemHealthItem)
        let controlCenterItem = NSMenuItem(title: "Control Center", action: #selector(showControlCenter), keyEquivalent: "")
        controlCenterItem.toolTip = "Open the native glass control center window."
        menu.addItem(controlCenterItem)
        let newChatItem = NSMenuItem(title: "New Chat", action: #selector(newChat), keyEquivalent: "n")
        newChatItem.toolTip = "Start a new conversation."
        menu.addItem(newChatItem)
        let chatHistoryItem = NSMenuItem(title: "Chat History", action: #selector(showChatHistory), keyEquivalent: "h")
        chatHistoryItem.toolTip = "Show the chat history window."
        menu.addItem(chatHistoryItem)
        let toggle = NSMenuItem(title: "Voice Listening", action: #selector(toggleVoice), keyEquivalent: "v")
        toggle.state = voiceEnabled ? .on : .off
        toggle.toolTip = "Toggle the local voice wake-word listener."
        menu.addItem(toggle)

        let roastToggle = NSMenuItem(title: "Roast Mode", action: #selector(toggleRoast), keyEquivalent: "")
        roastToggle.state = roastEnabled ? .on : .off
        roastToggle.toolTip = "Switch to the drill persona for spicy roasts."
        menu.addItem(roastToggle)

        let personaMenu = NSMenu(title: "Persona")
        for (name, display) in [("default", "Default"), ("wicket", "Wicket"), ("genz", "Gen Z"), ("drill", "Drill"), ("midwest", "Midwest Aunt")] {
            let item = NSMenuItem(title: display, action: #selector(selectPersona(_:)), keyEquivalent: "")
            item.representedObject = name
            item.state = (selectedPersona == name) ? .on : .off
            personaMenu.addItem(item)
        }
        let personaParent = NSMenuItem(title: "Persona", action: nil, keyEquivalent: "")
        personaParent.submenu = personaMenu
        menu.addItem(personaParent)

        let restartVoiceItem = NSMenuItem(title: "Restart Voice Recognition", action: #selector(restartVoice), keyEquivalent: "r")
        restartVoiceItem.toolTip = "Recycle the local speech recognizer pipeline."
        menu.addItem(restartVoiceItem)
        let benchmarkItem = NSMenuItem(title: "Benchmark", action: #selector(runBenchmark), keyEquivalent: "b")
        benchmarkItem.toolTip = "Run the standard benchmark suite."
        menu.addItem(benchmarkItem)

        let toolsMenu = NSMenu(title: "Tools")
        let dashboardItem = NSMenuItem(title: "Open Dashboard", action: #selector(openDashboard), keyEquivalent: "d")
        dashboardItem.toolTip = "Open the Bad Apple web dashboard in your browser."
        toolsMenu.addItem(dashboardItem)
        let briefingItem = NSMenuItem(title: "Daily Briefing", action: #selector(showBriefing), keyEquivalent: "")
        briefingItem.toolTip = "Show the daily briefing window."
        toolsMenu.addItem(briefingItem)
        let screenActionsItem = NSMenuItem(title: "Screen Actions...", action: #selector(showScreenActions), keyEquivalent: "")
        screenActionsItem.toolTip = "Run actions based on the current screen content."
        toolsMenu.addItem(screenActionsItem)
        let imagePlaygroundItem = NSMenuItem(title: "Image Playground...", action: #selector(showImagePlayground), keyEquivalent: "")
        imagePlaygroundItem.toolTip = "Generate images from a text prompt."
        toolsMenu.addItem(imagePlaygroundItem)
        let viewWorkingMemoryItem = NSMenuItem(title: "View Working Memory", action: #selector(viewWorkingMemory), keyEquivalent: "")
        viewWorkingMemoryItem.toolTip = "Inspect the working memory scratchpad."
        toolsMenu.addItem(viewWorkingMemoryItem)
        let clearWorkingMemoryItem = NSMenuItem(title: "Clear Working Memory", action: #selector(clearWorkingMemory), keyEquivalent: "")
        clearWorkingMemoryItem.toolTip = "Erase the working memory scratchpad."
        toolsMenu.addItem(clearWorkingMemoryItem)
        toolsMenu.addItem(NSMenuItem.separator())
        let listShortcutsItem = NSMenuItem(title: "List Shortcuts", action: #selector(listShortcuts), keyEquivalent: "")
        listShortcutsItem.toolTip = "List available macOS Shortcuts."
        toolsMenu.addItem(listShortcutsItem)
        let runShortcutItem = NSMenuItem(title: "Run Shortcut...", action: #selector(runShortcutPrompt), keyEquivalent: "")
        runShortcutItem.toolTip = "Prompt for a Shortcut name and run it."
        toolsMenu.addItem(runShortcutItem)
        let toolsParent = NSMenuItem(title: "Tools", action: nil, keyEquivalent: "")
        toolsParent.submenu = toolsMenu
        menu.addItem(toolsParent)

        let meshMenu = NSMenu(title: "Mesh")
        let p2pEnabled = runtime["p2p_enabled"] as? Bool ?? false
        let p2pItem = NSMenuItem(title: "P2P Sync", action: #selector(toggleP2P), keyEquivalent: "")
        p2pItem.state = p2pEnabled ? .on : .off
        p2pItem.toolTip = "Enable or disable encrypted link-local peer discovery and sync."
        meshMenu.addItem(p2pItem)
        let ambientRunning = runtime["ambient_running"] as? Bool ?? false
        let ambientItem = NSMenuItem(title: ambientRunning ? "Stop Ambient" : "Start Ambient", action: #selector(toggleAmbient), keyEquivalent: "")
        ambientItem.state = ambientRunning ? .on : .off
        ambientItem.toolTip = "Capture active app and window context locally for context."
        meshMenu.addItem(ambientItem)
        let setWorkspaceItem = NSMenuItem(title: "Set Workspace...", action: #selector(setWorkspacePrompt), keyEquivalent: "")
        setWorkspaceItem.toolTip = "Set the current workspace path for project-mode context."
        meshMenu.addItem(setWorkspaceItem)
        let openWorkspaceItem = NSMenuItem(title: "Open Workspace", action: #selector(openCurrentWorkspace), keyEquivalent: "")
        openWorkspaceItem.toolTip = "Open the configured workspace in Finder."
        meshMenu.addItem(openWorkspaceItem)
        let peers = runtime["p2p_peers"] as? [String] ?? []
        let peersItem = NSMenuItem(title: "Peers: \(peers.count)", action: nil, keyEquivalent: "")
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
        let meshParent = NSMenuItem(title: "Mesh", action: nil, keyEquivalent: "")
        meshParent.submenu = meshMenu
        menu.addItem(meshParent)

        let voiceMenu = NSMenu(title: "Voice")

        let settingsItem = NSMenuItem(title: "Voice Settings…", action: #selector(showVoiceSettings), keyEquivalent: ",")
        voiceMenu.addItem(settingsItem)
        let helpItem = NSMenuItem(title: "Voice Help…", action: #selector(showVoiceHelp), keyEquivalent: "")
        voiceMenu.addItem(helpItem)
        let logItem = NSMenuItem(title: "Voice Log…", action: #selector(showVoiceLog), keyEquivalent: "")
        voiceMenu.addItem(logItem)
        voiceMenu.addItem(NSMenuItem.separator())

        let usePiper = UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false
        let engineToggle = NSMenuItem(title: "Use Piper TTS (experimental)", action: #selector(togglePiperTTS), keyEquivalent: "")
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
        let voiceParent = NSMenuItem(title: usePiper ? "Voice (Piper)" : "Voice (Apple)", action: nil, keyEquivalent: "")
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
        if streamedTokenCount > 0 {
            let stream = NSMenuItem(title: "Streamed output: \(streamedTokenCount) characters", action: nil, keyEquivalent: "")
            stream.isEnabled = false
            menu.addItem(stream)
        }
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

        if BadAppleFFI.shared.isLoaded {
            let pursuits = BadAppleFFI.shared.activePursuits()
            let subMenu = NSMenu(title: "Active Pursuits")
            if pursuits.isEmpty {
                let empty = NSMenuItem(title: "No active pursuits", action: nil, keyEquivalent: "")
                empty.isEnabled = false
                subMenu.addItem(empty)
            } else {
                for pursuit in pursuits.suffix(8) {
                    let item = NSMenuItem(title: pursuit.truncated(to: 70), action: nil, keyEquivalent: "")
                    item.isEnabled = false
                    item.toolTip = pursuit
                    subMenu.addItem(item)
                }
            }
            let parent = NSMenuItem(title: "Active Pursuits", action: nil, keyEquivalent: "")
            parent.submenu = subMenu
            menu.addItem(parent)
            menu.addItem(NSMenuItem(title: "Push Pursuit...", action: #selector(pushPursuit), keyEquivalent: "p"))
        }
        menu.addItem(NSMenuItem.separator())
        let updateItem = NSMenuItem(title: "Check for Updates...", action: #selector(checkForUpdates), keyEquivalent: "")
        updateItem.toolTip = "Download and install the latest unsigned release from GitHub."
        menu.addItem(updateItem)
        let startAtLoginItem = NSMenuItem(title: "Start at Login", action: #selector(toggleStartAtLogin), keyEquivalent: "")
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
        badAppleVoiceLog("Piper TTS enabled: \(next)")
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
        guard BadAppleFFI.shared.isLoaded else { return }
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Push a new active pursuit"
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 340, height: 24))
        alert.accessoryView = textField
        alert.addButton(withTitle: "Push")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            let text = textField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { _ = BadAppleFFI.shared.pushPursuit(text) }
        }
        rebuildMenu()
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
