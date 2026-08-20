import AppKit
import AVFoundation
import BadAppleBridge
import Darwin
import Foundation
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

// MARK: - Local neural TTS client (Piper)

final class PiperTTSClient: NSObject, AVAudioPlayerDelegate {
    static let shared = PiperTTSClient()

    private let socketPath = "/tmp/badapple_tts.sock"
    private let requestTimeout: TimeInterval = 2.0
    private let responseTimeout: TimeInterval = 15.0
    private var player: AVAudioPlayer?
    private var onDidFinish: (() -> Void)?
    private var requestID = 0

    func stop() {
        player?.stop()
        player = nil
        onDidFinish = nil
    }

    static let defaultVoice = "es_MX-claude-high"
    static let availableVoices = [
        "es_MX-claude-high",
        "es_MX-cortana-19669-epoch-high",
    ]

    /// Try to speak through the local Piper TTS server. Calls `completion(true)`
    /// when audio finishes, or `completion(false)` if the server is unreachable,
    /// synthesis fails, or playback fails.
    func speak(_ text: String, voice: String, completion: @escaping (Bool) -> Void) {
        requestID += 1
        let myID = requestID
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async { completion(false) }
                return
            }
            do {
                let wavURL = try self.synthesize(text, voice: voice)
                DispatchQueue.main.async { [weak self] in
                    guard let self = self, self.requestID == myID else { return }
                    self.play(url: wavURL, completion: completion)
                }
            } catch {
                badAppleVoiceLog("PiperTTS synthesize error: \(error.localizedDescription)")
                DispatchQueue.main.async { [weak self] in
                    guard let self = self, self.requestID == myID else { return }
                    completion(false)
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

    /// Play the synthesized WAV as raw as possible. The Piper model already has
    /// warmth and cadence baked in; extra pitch/reverb effects make it sound
    /// processed. We just use a slightly slower playback rate to give it a
    /// little more breath and weight.
    private func play(url: URL, completion: @escaping (Bool) -> Void) {
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            player?.volume = 1.0
            player?.prepareToPlay()
            onDidFinish = { [weak self] in
                self?.player = nil
                completion(true)
            }
            guard player?.play() == true else {
                throw NSError(domain: "PiperTTS", code: 5, userInfo: [NSLocalizedDescriptionKey: "play() returned false"])
            }
        } catch {
            badAppleVoiceLog("PiperTTS play error: \(error.localizedDescription)")
            completion(false)
        }
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.onDidFinish?()
            self?.onDidFinish = nil
            self?.player = nil
        }
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
    private let wakePattern = try! NSRegularExpression(
        pattern: "(?i)(?:^|\\b)(?:(?:hey|he|hay|my)(?:\\s+\\w+){0,3}\\s+)?bad(?:\\s+\\w+){0,2}\\s+apple(?:\\b|$)"
    )
    private var promptTimer: Timer?
    private var stablePrompt = ""
    private var recognitionTimer: Timer?
    private var lastTranscript = ""

    var state: State = .disabled {
        didSet { DispatchQueue.main.async { [weak self] in self?.onStateChange?(self?.state ?? .disabled) } }
    }
    var onStateChange: ((State) -> Void)?
    var onPrompt: ((String) -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
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

        // Keep the audio engine running by wiring input -> mixer -> output, but
        // mute the mixer so the user doesn't hear the microphone fed back.
        let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false)!
        audioEngine.connect(input, to: audioEngine.mainMixerNode, format: inputFormat)
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
        input.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak recognitionRequest] buffer, _ in
            guard let request = recognitionRequest else { return }
            let expectedFrames = AVAudioFrameCount(Double(buffer.frameLength) * target.sampleRate / inputFormat.sampleRate)
            let outputFrames = expectedFrames + 1024
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outputFrames) else { return }
            // The converter does not always reset frameLength; tell it the capacity and then clamp to actual.
            outputBuffer.frameLength = outputBuffer.frameCapacity
            var error: NSError?
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
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

        task = recognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            DispatchQueue.main.async { self?.consume(result: result, error: error) }
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
        if let result = result {
            let transcript = result.bestTranscription.formattedString
            lastTranscript = transcript
            badAppleVoiceLog("consume: isFinal=\(result.isFinal) transcript='\(transcript)' awaitingNextUtterance=\(awaitingNextUtterance)")

            if awaitingNextUtterance {
                let prompt = (wakeSuffix(in: transcript) ?? transcript)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: .punctuationCharacters)
                if !prompt.isEmpty {
                    if prompt != stablePrompt {
                        stablePrompt = prompt
                        state = .awaitingPrompt
                        startPromptTimer()
                    }
                    if result.isFinal {
                        stopPromptTimer()
                        awaitingNextUtterance = false
                        stablePrompt = ""
                        deliver(prompt)
                        return
                    }
                }
                if result.isFinal {
                    stopPromptTimer()
                    awaitingNextUtterance = false
                    stablePrompt = ""
                    scheduleRestart(after: 0.15)
                    return
                }
            } else if let suffix = wakeSuffix(in: transcript) {
                state = .awaitingPrompt
                if suffix.isEmpty {
                    awaitingNextUtterance = true
                    stablePrompt = ""
                    if result.isFinal {
                        // Wake phrase heard. Restart quickly so the prompt is captured
                        // in a fresh request without stale audio context.
                        scheduleRestart(after: 0.05)
                        return
                    }
                } else {
                    awaitingNextUtterance = true
                    stablePrompt = ""
                    if result.isFinal {
                        stopPromptTimer()
                        awaitingNextUtterance = false
                        deliver(suffix)
                        return
                    }
                }
            } else if result.isFinal {
                scheduleRestart(after: 0.15)
                return
            }
        }
        if let error = error {
            badAppleVoiceLog("consume error: \(error)")
            scheduleRestart(after: 0.5)
        }
    }

    private func startPromptTimer() {
        promptTimer?.invalidate()
        let timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: false) { [weak self] _ in
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
        let timer = Timer.scheduledTimer(withTimeInterval: 3.5, repeats: false) { [weak self] _ in
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

    /// Pick the highest-quality installed Spanish/Mexican voice. Prefer
    /// premium/enhanced Paulina, then Siri-quality, then compact, then any
    /// es-MX voice. This single change removes most of the "robot" sheen.
    private func bestVoice() -> AVSpeechSynthesisVoice {
        let candidateIds = [
            "com.apple.voice.premium.es-MX.Paulina",
            "com.apple.voice.enhanced.es-MX.Paulina",
            "com.apple.voice.superpremium.es-MX.Paulina",
            "com.apple.voice.premium.es-ES.Monica",
            "com.apple.voice.enhanced.es-ES.Monica",
            "com.apple.voice.siri.es-MX",
            "com.apple.voice.compact.es-MX.Paulina",
        ]
        for id in candidateIds {
            if let voice = AVSpeechSynthesisVoice(identifier: id) {
                return voice
            }
        }
        return AVSpeechSynthesisVoice(language: "es-MX")
            ?? AVSpeechSynthesisVoice(identifier: "com.apple.speech.synthesis.voice.Fred")
            ?? AVSpeechSynthesisVoice(language: "en-US")!
    }

    /// Split the response into natural prosodic chunks on sentence/ellipsis
    /// boundaries only. Paulina handles the inner cadence (em-dashes,
    /// ellipses) better inside a single utterance; splitting every pause makes
    /// it choppy. Short post-delays keep the flow connected but breathable.
    private func prosodyChunks(from text: String) -> [ProsodyChunk] {
        var chunks: [ProsodyChunk] = []
        var current = ""

        func flush(_ postDelay: TimeInterval = 0.0, rate: Float = 0.50, pitch: Float = 0.98) {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                current = ""
                return
            }
            // sign-off "—besos" gets extra warmth and a long trailing breath
            let isSignOff = trimmed.lowercased().contains("besos")
            let finalRate: Float = isSignOff ? 0.46 : rate
            let finalPitch: Float = isSignOff ? 0.96 : pitch
            let finalDelay: TimeInterval = isSignOff ? max(postDelay, 0.35) : postDelay
            chunks.append(ProsodyChunk(text: trimmed, rate: finalRate, pitch: finalPitch, postDelay: finalDelay))
            current = ""
        }

        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            current.append(c)

            if c == "…" || (c == "." && i + 1 < chars.count && chars[i + 1] == "." && i + 2 < chars.count && chars[i + 2] == ".") {
                // ellipsis: tiny trailing breath, let the voice trail
                flush(0.10, rate: 0.48, pitch: 0.97)
                if c == "." { i += 2 }
            } else if c == "?" {
                flush(0.12, rate: 0.50, pitch: 1.01)
            } else if c == "!" {
                flush(0.12, rate: 0.52, pitch: 1.00)
            } else if c == "." || c == "\n" {
                // only end a sentence if the next char is whitespace or we are at the end
                let next = i + 1 < chars.count ? chars[i + 1] : nil
                if next == nil || next!.isWhitespace || next! == "\n" {
                    flush(0.08)
                }
            }

            i += 1
        }

        flush(0.05)
        return chunks.isEmpty ? [ProsodyChunk(text: text, rate: 0.50, pitch: 0.98, postDelay: 0.05)] : chunks
    }

    private var usePiperTTS: Bool {
        UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false
    }

    private var selectedPiperVoice: String {
        let raw = UserDefaults.standard.string(forKey: "BadAppleTTSVoice") ?? PiperTTSClient.defaultVoice
        return PiperTTSClient.availableVoices.contains(raw) ? raw : PiperTTSClient.defaultVoice
    }

    /// Main entry point: try the local neural Piper TTS first, then fall back
    /// to the on-device Apple speech engine. The result is much more human at
    /// the cost of ~200-600 ms synthesis latency for the 8B response.
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
        if tapInstalled {
            audioEngine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        audioEngine.disconnectNodeOutput(audioEngine.inputNode)
        audioEngine.disconnectNodeOutput(audioEngine.mainMixerNode)
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
    static let keyPath = "/var/lib/bad_apple/slicks.key"
}

final class AppDelegate: NSObject, NSApplicationDelegate, @unchecked Sendable {
    private var statusItem: NSStatusItem?
    private var menu: NSMenu?
    private var timer: Timer?
    private let voiceHost = BadAppleVoiceHost()
    private let actionExecutor = BadAppleActionExecutor()
    private var streamedTokenCount = 0
    private var lastPrompt = ""
    private var lastError: String?
    private var voiceEnabled: Bool {
        get { UserDefaults.standard.object(forKey: "BadAppleVoiceEnabled") as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: "BadAppleVoiceEnabled") }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        BadAppleFFI.shared.load()
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem?.button?.title = "🍎"
        menu = NSMenu(title: "Bad Apple")
        statusItem?.menu = menu

        voiceHost.onStateChange = { [weak self] _ in self?.rebuildMenu() }
        voiceHost.onPrompt = { [weak self] prompt in self?.submitVoicePrompt(prompt) }
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.rebuildMenu()
        }
        rebuildMenu()
        voiceHost.setEnabled(voiceEnabled)
    }

    func applicationWillTerminate(_ notification: Notification) {
        voiceHost.setEnabled(false)
        timer?.invalidate()
    }

    private func submitVoicePrompt(_ prompt: String) {
        badAppleVoiceLog("submitVoicePrompt: \(prompt)")
        lastPrompt = prompt
        lastError = nil
        streamedTokenCount = 0
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
        let socket = BadAppleBrain.deepSocket
        let maxTokens = 120
        Task {
            do {
                let response = try await runBadAppleCLI(prompt: effectivePrompt, socketPath: socket, maxTokens: maxTokens)
                await MainActor.run { self.completeVoiceResponse(response) }
            } catch {
                badAppleVoiceLog("submitVoicePrompt error: \(error)")
                await MainActor.run {
                    self.lastError = error.localizedDescription
                    self.voiceHost.resumeAfterFailure()
                    self.rebuildMenu()
                }
            }
        }
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

    private func runBadAppleCLI(prompt: String, socketPath: String, maxTokens: Int) async throws -> String {
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
                process.environment = environment

                var timeoutTimer: Timer?
                timeoutTimer = Timer.scheduledTimer(withTimeInterval: 120.0, repeats: false) { _ in
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

    private struct BadAppleMenuBarError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { self.errorDescription = message }
    }

    private func completeVoiceResponse(_ response: String) {
        badAppleVoiceLog("completeVoiceResponse: \(response.prefix(200))")
        let parsed = BadAppleActionParser.parse(response)
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

    func rebuildMenu() {
        guard let menu = menu else { return }
        menu.removeAllItems()
        let header = NSMenuItem(title: "Bad Apple — 8B MLX + RAG", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)

        let voiceStatus = NSMenuItem(title: voiceHost.state.label.truncated(to: 90), action: nil, keyEquivalent: "")
        voiceStatus.isEnabled = false
        menu.addItem(voiceStatus)
        let mode = NSMenuItem(title: "Brain: 8B MLX + RAG", action: nil, keyEquivalent: "")
        mode.isEnabled = false
        menu.addItem(mode)
        menu.addItem(NSMenuItem(title: "New Chat", action: #selector(newChat), keyEquivalent: "n"))
        let toggle = NSMenuItem(title: "Voice Listening", action: #selector(toggleVoice), keyEquivalent: "v")
        toggle.state = voiceEnabled ? .on : .off
        menu.addItem(toggle)
        menu.addItem(NSMenuItem(title: "Restart Voice Recognition", action: #selector(restartVoice), keyEquivalent: "r"))

        let voiceMenu = NSMenu(title: "Voice")

        let usePiper = UserDefaults.standard.object(forKey: "BadAppleUsePiperTTS") as? Bool ?? false
        let engineToggle = NSMenuItem(title: "Use Piper TTS (experimental)", action: #selector(togglePiperTTS), keyEquivalent: "")
        engineToggle.state = usePiper ? .on : .off
        voiceMenu.addItem(engineToggle)
        voiceMenu.addItem(NSMenuItem.separator())

        for voice in PiperTTSClient.availableVoices {
            let display = voice
                .replacingOccurrences(of: "es_MX-", with: "")
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
        let voiceParent = NSMenuItem(title: usePiper ? "Voice (Piper)" : "Voice (Paulina)", action: nil, keyEquivalent: "")
        voiceParent.submenu = voiceMenu
        menu.addItem(voiceParent)

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
        menu.addItem(NSMenuItem(title: "Quit", action: #selector(terminate), keyEquivalent: "q"))
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
        UserDefaults.standard.set(!current, forKey: "BadAppleUsePiperTTS")
        badAppleVoiceLog("Piper TTS enabled: \(!current)")
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

    @objc private func terminate() { NSApp.terminate(nil) }
}

private extension String {
    func truncated(to length: Int) -> String {
        count > length ? String(prefix(length)) + "..." : self
    }
}
