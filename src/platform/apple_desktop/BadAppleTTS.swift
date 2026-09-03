// Bad Apple native TTS server.
//
// Replaces the Python badapple_tts_server.py with an on-device AVSpeechSynthesizer
// server that writes synthesized speech to a .caf file and returns its path over
// a Unix socket.
//
// Socket protocol (line-delimited JSON):
//   request : {"text": "hello", "voice": "Samantha", "rate": 0.5, "volume": 1.0}
//   response: {"ok": true, "wav_path": "/tmp/badapple_tts_xxxx.caf", "sample_rate": 22050, "duration_ms": 1000}
//   error   : {"ok": false, "error": "..."}

import AVFoundation
import Foundation

// C-style signal state. The handler must be @convention(c) and access only globals.
private var gTTSShouldStop: sig_atomic_t = 0
private var gTTSListenFd: Int32 = -1

@main
enum BadAppleTTS {
    static func main() {
        let server = TTSServer()
        server.run()
    }
}

// MARK: - Errors

private enum TTSError: LocalizedError {
    case invalidRequest(String)
    case socketBindFailed(String)
    case socketListenFailed(String)
    case socketAcceptFailed(String)
    case socketReadFailed(String)
    case socketWriteFailed(String)
    case voiceNotFound(String)
    case synthesisFailed(String)
    case synthesisTimeout
    case fileWriteFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidRequest(let reason): return "invalid request: \(reason)"
        case .socketBindFailed(let reason): return "cannot bind socket: \(reason)"
        case .socketListenFailed(let reason): return "cannot listen on socket: \(reason)"
        case .socketAcceptFailed(let reason): return "accept failed: \(reason)"
        case .socketReadFailed(let reason): return "read failed: \(reason)"
        case .socketWriteFailed(let reason): return "write failed: \(reason)"
        case .voiceNotFound(let name): return "voice not found: \(name)"
        case .synthesisFailed(let reason): return "synthesis failed: \(reason)"
        case .synthesisTimeout: return "synthesis timed out"
        case .fileWriteFailed(let reason): return "audio file write failed: \(reason)"
        }
    }
}

// MARK: - Server

private final class TTSServer {
    private let socketPath: String
    private let defaultVoiceName: String
    private let defaultLengthScale: Double
    private let defaultVolume: Double

    private let synthesizer = AVSpeechSynthesizer()
    private let fileManager = FileManager.default

    private let cleanupQueue = DispatchQueue(label: "com.badapple.tts.cleanup")
    private let synthQueue = DispatchQueue(label: "com.badapple.tts.synth", qos: .userInitiated)
    private var lastOutputURL: URL?

    private var sigIntSource: DispatchSourceSignal?
    private var sigTermSource: DispatchSourceSignal?

    private let maxRequestBytes = 8_388_608
    private let maxTextLength = 2_000

    init() {
        socketPath = ProcessInfo.processInfo.environment["BADAPPLE_TTS_SOCKET"] ?? "/tmp/badapple_tts.sock"
        defaultVoiceName = ProcessInfo.processInfo.environment["BADAPPLE_TTS_VOICE"] ?? "Samantha"
        defaultLengthScale = Double(ProcessInfo.processInfo.environment["BADAPPLE_TTS_LENGTH_SCALE"] ?? "1.0") ?? 1.0
        defaultVolume = Double(ProcessInfo.processInfo.environment["BADAPPLE_TTS_VOLUME"] ?? "1.0") ?? 1.0
        gTTSShouldStop = 0
        gTTSListenFd = -1
    }

    func run() {
        setupSignalHandling()

        if !startSocket() {
            return
        }

        log("Bad Apple TTS server listening on \(socketPath)")

        let acceptQueue = DispatchQueue(label: "com.badapple.tts.accept", qos: .userInitiated)
        acceptQueue.async { [weak self] in
            guard let self else { return }
            self.acceptLoop()
        }

        // Run the main run loop. AVSpeechSynthesizer buffer callbacks are delivered
        // to this run loop, so the main thread must stay alive and pump it.
        while gTTSShouldStop == 0 {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.25))
        }

        sigIntSource?.cancel()
        sigTermSource?.cancel()
        sigIntSource = nil
        sigTermSource = nil

        stopSocket()
        log("Bad Apple TTS server stopped")
    }

    private func requestStop(cancelSources: Bool = false) {
        gTTSShouldStop = 1
        if gTTSListenFd >= 0 {
            let fd = gTTSListenFd
            gTTSListenFd = -1
            close(fd)
        }
        if cancelSources {
            sigIntSource?.cancel()
            sigTermSource?.cancel()
        }
    }

    // MARK: - Socket

    private func startSocket() -> Bool {
        // Remove stale socket from a previous run.
        try? fileManager.removeItem(atPath: socketPath)

        gTTSShouldStop = 0
        gTTSListenFd = -1

        let newFd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard newFd >= 0 else {
            logError("socket() failed: \(errnoDescription())")
            return false
        }

        var on: Int32 = 1
        _ = setsockopt(newFd, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
        setNoSigPipe(fd: newFd)

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(socketPath.utf8)
        let maxPath = MemoryLayout.size(ofValue: addr.sun_path) - 1
        guard pathBytes.count < maxPath else {
            logError("socket path too long")
            close(newFd)
            return false
        }

        pathBytes.withUnsafeBufferPointer { src in
            _ = withUnsafeMutablePointer(to: &addr.sun_path) { dst in
                memcpy(dst, src.baseAddress!, pathBytes.count)
            }
        }
        addr.sun_len = UInt8(2 + pathBytes.count + 1)
        let bindAddrLen = socklen_t(addr.sun_len)

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                bind(newFd, sockaddrPtr, bindAddrLen)
            }
        }

        guard bindResult == 0 else {
            logError("bind() failed: \(errnoDescription())")
            close(newFd)
            return false
        }

        guard listen(newFd, 4) == 0 else {
            logError("listen() failed: \(errnoDescription())")
            close(newFd)
            return false
        }

        let chmodResult = socketPath.withCString { cPath in
            chmod(cPath, 0o666)
        }
        if chmodResult != 0 {
            logError("chmod socket failed: \(errnoDescription())")
        }

        gTTSListenFd = newFd
        return true
    }

    private func stopSocket() {
        if gTTSListenFd >= 0 {
            let fd = gTTSListenFd
            gTTSListenFd = -1
            close(fd)
        }
        try? fileManager.removeItem(atPath: socketPath)
    }

    private func setNoSigPipe(fd: Int32) {
        var on: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
    }

    private func acceptLoop() {
        while gTTSShouldStop == 0 {
            var clientAddr = sockaddr_un()
            var clientLen = socklen_t(MemoryLayout<sockaddr_un>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) { addrPtr in
                addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                    withUnsafeMutablePointer(to: &clientLen) { lenPtr in
                        accept(gTTSListenFd, sockaddrPtr, lenPtr)
                    }
                }
            }

            if clientFd < 0 {
                let err = errno
                if err == EINTR { continue }
                if gTTSShouldStop != 0 { break }
                if err != EBADF {
                    logError("accept() failed: \(errnoDescription(err))")
                }
                break
            }

            setNoSigPipe(fd: clientFd)

            synthQueue.async { [weak self] in
                guard let self else {
                    close(clientFd)
                    return
                }
                self.handleConnection(fd: clientFd)
            }
        }
    }

    private func handleConnection(fd: Int32) {
        defer { close(fd) }

        // Prevent a spurious SIGPIPE if the client disconnects before we reply.
        setNoSigPipe(fd: fd)

        guard let requestData = readRequest(fd: fd) else {
            sendError(fd: fd, error: TTSError.socketReadFailed("empty or oversized request"))
            return
        }

        do {
            let json = try JSONSerialization.jsonObject(with: requestData, options: [])
            guard let dict = json as? [String: Any] else {
                sendError(fd: fd, error: TTSError.invalidRequest("JSON must be an object"))
                return
            }

            let result = try processRequest(dict)
            sendSuccess(fd: fd, result: result)
        } catch {
            if let ttsError = error as? TTSError {
                sendError(fd: fd, error: ttsError)
            } else {
                sendError(fd: fd, error: TTSError.invalidRequest(error.localizedDescription))
            }
        }
    }

    private func readRequest(fd: Int32) -> Data? {
        var data = Data()
        let bufferSize = 4096

        while data.count < maxRequestBytes {
            var chunk = Data(count: min(bufferSize, maxRequestBytes - data.count))
            let chunkCount = chunk.count
            let readCount = chunk.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress else { return 0 }
                return recv(fd, base, chunkCount, 0)
            }

            if readCount <= 0 {
                if readCount < 0, errno == EINTR { continue }
                break
            }

            if readCount < chunk.count {
                chunk = chunk.prefix(readCount)
            }
            data.append(chunk)

            if data.contains(0x0A) {
                break
            }
        }

        guard !data.isEmpty, data.contains(0x0A) else { return nil }
        if let newlineIndex = data.firstIndex(of: 0x0A) {
            return data.prefix(upTo: newlineIndex)
        }
        return nil
    }

    private func sendSuccess(fd: Int32, result: SynthesisResult) {
        var response: [String: Any] = [
            "ok": true,
            "wav_path": result.url.path,
            "sample_rate": Int(result.sampleRate),
            "duration_ms": result.durationMs,
        ]
        if let error = result.error {
            response = ["ok": false, "error": error.localizedDescription]
            try? fileManager.removeItem(at: result.url)
        }
        sendJSON(fd: fd, object: response)
    }

    private func sendError(fd: Int32, error: TTSError) {
        sendJSON(fd: fd, object: ["ok": false, "error": error.localizedDescription])
    }

    private func sendJSON(fd: Int32, object: [String: Any]) {
        do {
            let payload = try JSONSerialization.data(withJSONObject: object, options: [])
            var data = payload
            data.append(0x0A)
            let sendCount = data.count
            _ = data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return -1 }
                return send(fd, base, sendCount, 0)
            }
        } catch {
            logError("failed to encode JSON response: \(error)")
        }
    }

    // MARK: - Synthesis

    private func processRequest(_ request: [String: Any]) throws -> SynthesisResult {
        guard let text = request["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw TTSError.invalidRequest("missing or empty 'text' field")
        }
        guard text.count <= maxTextLength else {
            throw TTSError.invalidRequest("text too long (max \(maxTextLength) characters)")
        }

        let cleanText = sanitizeText(text)
        guard !cleanText.isEmpty else {
            throw TTSError.invalidRequest("text is empty after cleaning")
        }

        let voiceName = request["voice"] as? String ?? defaultVoiceName
        guard let voice = resolveVoice(voiceName) else {
            throw TTSError.voiceNotFound(voiceName)
        }

        let lengthScale: Double
        if let raw = request["length_scale"] as? Double {
            lengthScale = raw
        } else if let raw = request["lengthScale"] as? Double {
            lengthScale = raw
        } else {
            lengthScale = defaultLengthScale
        }

        let rate: Float
        if let raw = request["rate"] as? Double {
            rate = Float(clamp(raw, min: 0.0, max: 1.0))
        } else if lengthScale > 0 {
            rate = Float(clamp(0.5 / lengthScale, min: 0.0, max: 1.0))
        } else {
            rate = 0.5
        }

        let volume: Float
        if let raw = request["volume"] as? Double {
            volume = Float(clamp(raw, min: 0.0, max: 1.0))
        } else {
            volume = Float(clamp(defaultVolume, min: 0.0, max: 1.0))
        }

        // Delete the previous output file before creating a new one.
        if let previous = lastOutputURL {
            cleanupQueue.async { [weak self] in
                _ = try? self?.fileManager.removeItem(at: previous)
            }
        }

        let outputURL = URL(fileURLWithPath: "/tmp/badapple_tts_\(UUID().uuidString).caf")
        lastOutputURL = outputURL

        let result = synthesizeToFile(text: cleanText, voice: voice, rate: rate, volume: volume, outputURL: outputURL)

        if result.error == nil {
            // Schedule cleanup after a short TTL so /tmp does not fill up.
            cleanupQueue.asyncAfter(deadline: .now() + 120) { [weak self] in
                _ = try? self?.fileManager.removeItem(at: outputURL)
            }
        }

        return result
    }

    private func sanitizeText(_ text: String) -> String {
        let withoutURLs = text.replacingOccurrences(of: "https?://\\S+", with: "", options: .regularExpression)
        let withoutAsterisks = withoutURLs.replacingOccurrences(of: "*", with: "")

        var cleaned = ""
        for scalar in withoutAsterisks.unicodeScalars {
            let value = scalar.value
            if value == 0x09 || value == 0x0A || value == 0x0D || (value >= 0x20 && value <= 0x7E) {
                cleaned.append(Character(scalar))
            } else if scalar.properties.isWhitespace {
                cleaned.append(" ")
            }
        }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func resolveVoice(_ name: String) -> AVSpeechSynthesisVoice? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)

        if let voice = AVSpeechSynthesisVoice(identifier: trimmed) {
            return voice
        }

        if trimmed == "Samantha" {
            if let voice = AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-US.Samantha") {
                return voice
            }
        }

        // Try to derive a BCP-47 language tag from names like en_US-amy-medium.
        let dashPattern = trimmed.replacingOccurrences(of: "_", with: "-")
        let localePrefix = dashPattern.prefix { $0 != "-" }
        if !localePrefix.isEmpty, let voice = AVSpeechSynthesisVoice(language: String(localePrefix)) {
            return voice
        }

        // Fallback to the built-in US English voice.
        if let voice = AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-US.Samantha") {
            return voice
        }
        if let voice = AVSpeechSynthesisVoice(language: "en-US") {
            return voice
        }

        // Last resort: the default system voice.
        return AVSpeechSynthesisVoice(language: "en")
    }

    private func synthesizeToFile(text: String, voice: AVSpeechSynthesisVoice, rate: Float, volume: Float, outputURL: URL) -> SynthesisResult {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = rate
        utterance.volume = volume

        let synthesizer = self.synthesizer

        // write(_:toBufferCallback:) is asynchronous. Its callbacks are delivered to
        // the main run loop, which is kept alive and pumped by this process's main
        // thread. This background thread waits on a semaphore until the final buffer.
        var output: AVAudioFile?
        var sampleRate: Double = 0
        var totalFrames: AVAudioFramePosition = 0
        var writeError: Error?
        var done = false
        let finished = DispatchSemaphore(value: 0)

        synthesizer.write(utterance) { [weak self] (buffer: AVAudioBuffer) in
            guard self != nil, gTTSShouldStop == 0 else {
                done = true
                finished.signal()
                return
            }
            guard let pcm = buffer as? AVAudioPCMBuffer else { return }
            if pcm.frameLength <= 1 {
                done = true
                finished.signal()
                return
            }
            do {
                if output == nil {
                    output = try AVAudioFile(forWriting: outputURL,
                                             settings: pcm.format.settings,
                                             commonFormat: pcm.format.commonFormat,
                                             interleaved: false)
                    sampleRate = pcm.format.sampleRate
                }
                try output?.write(from: pcm)
                totalFrames += AVAudioFramePosition(pcm.frameLength)
            } catch {
                writeError = error
                done = true
                finished.signal()
            }
        }

        _ = finished.wait(timeout: .now() + 60.0)

        if let error = writeError {
            output = nil
            return SynthesisResult(url: outputURL, sampleRate: 0, durationMs: 0, error: error)
        }

        if !done {
            output = nil
            return SynthesisResult(url: outputURL, sampleRate: 0, durationMs: 0, error: TTSError.synthesisTimeout)
        }

        output = nil

        let durationMs = sampleRate > 0 ? Int(Double(totalFrames) / sampleRate * 1000.0) : 0
        return SynthesisResult(url: outputURL, sampleRate: sampleRate, durationMs: durationMs, error: nil)
    }

    // MARK: - Signal handling

    private func setupSignalHandling() {
        // GCD signal sources are the only reliable way to catch SIGINT/SIGTERM
        // while AVFoundation is loaded. Mask the default delivery so the kernel
        // holds the signal pending; libdispatch observes it via kqueue and
        // invokes the handler on the main queue.
        _ = signal(SIGINT, SIG_IGN)
        _ = signal(SIGTERM, SIG_IGN)
        _ = signal(SIGHUP, SIG_IGN)
        _ = signal(SIGPIPE, SIG_IGN)

        let s1 = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        let s2 = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)

        s1.setEventHandler { [unowned self] in
            self.requestStop(cancelSources: true)
        }
        s2.setEventHandler { [unowned self] in
            self.requestStop(cancelSources: true)
        }

        s1.resume()
        s2.resume()

        sigIntSource = s1
        sigTermSource = s2
    }

    // MARK: - Logging

    private func log(_ message: String) {
        fputs("badapple-tts: \(message)\n", stderr)
    }

    private func logError(_ message: String) {
        fputs("badapple-tts: error: \(message)\n", stderr)
    }

    private func errnoDescription(_ code: Int32 = errno) -> String {
        return String(cString: strerror(code))
    }
}

// MARK: - Synthesis result

private struct SynthesisResult {
    let url: URL
    let sampleRate: Double
    let durationMs: Int
    let error: Error?
}

// MARK: - Helpers

private func clamp<T: Comparable>(_ value: T, min: T, max: T) -> T {
    if value < min { return min }
    if value > max { return max }
    return value
}
