// Bad Apple native TTS server.
//
// Replaces the Python badapple_tts_server.py. It first tries a local Piper ONNX
// voice (e.g. en_US-amy-medium) for neural-quality speech, then falls back to
// on-device AVSpeechSynthesizer. It returns a .wav or .caf path over a Unix socket.
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
    /// Optional env override for Piper length scale. 0 means "compute from word count".
    private let envLengthScale: Double
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
        defaultVoiceName = ProcessInfo.processInfo.environment["BADAPPLE_TTS_VOICE"] ?? "Best"
        envLengthScale = Double(ProcessInfo.processInfo.environment["BADAPPLE_TTS_LENGTH_SCALE"] ?? "") ?? 0.0
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

        // Word-based prosody. Long phrases are slowed down, short punchy
        // phrases stay natural, and the env var overrides everything.
        let wordCount = cleanText.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
        let isLong = wordCount > 18
        let isShort = wordCount < 6

        let lengthScale: Double
        if let raw = request["length_scale"] as? Double {
            lengthScale = raw
        } else if let raw = request["lengthScale"] as? Double {
            lengthScale = raw
        } else if envLengthScale > 0 {
            lengthScale = envLengthScale
        } else {
            lengthScale = isLong ? 1.15 : (isShort ? 1.00 : 1.05)
        }

        let rate: Float
        if let raw = request["rate"] as? Double {
            rate = Float(clamp(raw, min: 0.0, max: 1.0))
        } else {
            rate = isLong ? 0.42 : (isShort ? 0.50 : 0.46)
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

        // Prefer a local Piper ONNX model (neural TTS). Fall back to the native
        // AVFoundation synthesizer only when no Piper voice is available.
        let outputURL: URL
        let result: SynthesisResult
        let sentenceSilence = sentenceSilenceFor(cleanText)
        if let piperResult = synthesizeWithPiperIfAvailable(text: cleanText, voice: voiceName, lengthScale: lengthScale, sentenceSilence: sentenceSilence, outputURL: URL(fileURLWithPath: "/tmp/badapple_tts_\(UUID().uuidString).wav")), piperResult.error == nil {
            outputURL = piperResult.url
            result = piperResult
        } else {
            guard let voice = resolveVoice(voiceName) else {
                throw TTSError.voiceNotFound(voiceName)
            }
            outputURL = URL(fileURLWithPath: "/tmp/badapple_tts_\(UUID().uuidString).caf")
            result = synthesizeToFile(text: cleanText, voice: voice, rate: rate, volume: volume, outputURL: outputURL)
        }

        lastOutputURL = outputURL

        if result.error == nil {
            // Schedule cleanup after a short TTL so /tmp does not fill up.
            cleanupQueue.asyncAfter(deadline: .now() + 120) { [weak self] in
                _ = try? self?.fileManager.removeItem(at: outputURL)
            }
        }

        return result
    }

    /// Prepare text for the TTS engine: remove markup, turn ellipses, dashes,
    /// colons and bullets into brief breaths, and convert line breaks into clean
    /// pauses. This also collapses duplicate punctuation so the voice does not
    /// "speak" raw commas, periods, or colons.
    private func sanitizeText(_ text: String) -> String {
        var normalized = text

        // Strip URLs and markdown markers.
        normalized = normalized.replacingOccurrences(of: "https?://\\S+", with: "", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "*", with: "")

        // Ellipses, em/en dashes, run-on hyphens, colons and semicolons become breaths.
        normalized = normalized.replacingOccurrences(of: "\\.\\.\\.+", with: ", ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "[—–]", with: ", ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "-{2,}", with: ", ", options: .regularExpression)
        normalized = normalized.replacingOccurrences(of: "[;:]", with: ", ", options: .regularExpression)

        // Paragraph breaks first, so they become real sentence pauses.
        normalized = normalized.replacingOccurrences(of: "\\n\\n+", with: ". ", options: .regularExpression)

        // Bullets that start a line become the separator, consuming the newline.
        normalized = normalized.replacingOccurrences(of: "\\n\\s*[•·]\\s*", with: ", ", options: .regularExpression)

        // Any remaining newline is a clause break.
        normalized = normalized.replacingOccurrences(of: "\\n", with: ", ", options: .regularExpression)

        // Leading bullet at the very start of the text.
        normalized = normalized.replacingOccurrences(of: "^[•·]\\s*", with: "", options: .regularExpression)

        // Stray bullets elsewhere.
        normalized = normalized.replacingOccurrences(of: "[•·]", with: ", ", options: .regularExpression)

        // Collapse duplicate/fused punctuation so the voice doesn't read
        // "comma comma" or "period comma".
        var changed = true
        while changed {
            let before = normalized
            normalized = normalized.replacingOccurrences(of: ",\\s*,", with: ", ", options: .regularExpression)
            normalized = normalized.replacingOccurrences(of: ",\\s*\\.", with: ". ", options: .regularExpression)
            normalized = normalized.replacingOccurrences(of: "\\.\\s*,", with: ". ", options: .regularExpression)
            normalized = normalized.replacingOccurrences(of: "\\.\\s*\\.", with: ". ", options: .regularExpression)
            normalized = normalized.replacingOccurrences(of: "^,\\s*", with: "", options: .regularExpression)
            normalized = normalized.replacingOccurrences(of: "^\\.\\s*", with: "", options: .regularExpression)
            changed = (normalized != before)
        }

        // Strip control characters, keep printable ASCII and whitespace.
        var cleaned = ""
        for scalar in normalized.unicodeScalars {
            let value = scalar.value
            if (value >= 0x20 && value <= 0x7E) || scalar.properties.isWhitespace {
                cleaned.append(Character(scalar))
            }
        }

        while cleaned.contains("  ") {
            cleaned = cleaned.replacingOccurrences(of: "  ", with: " ")
        }

        // Remove any leading punctuation that would be spoken.
        cleaned = cleaned.replacingOccurrences(of: "^[,.:;!?\\-–—\\s]+", with: "", options: .regularExpression)

        // If the chunk has no words, return empty so the engine stays silent.
        if !cleaned.contains(where: { $0.isLetter || $0.isNumber }) {
            return ""
        }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Context-aware silence between sentences and clauses. Longer thoughts get
    /// more room to breathe; short punchy phrases stay tight.
    private func sentenceSilenceFor(_ text: String) -> Double {
        let wordCount = text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .count
        let isLong = wordCount > 18
        let isShort = wordCount < 6

        switch text.last {
        case "?":
            return isLong ? 0.28 : (isShort ? 0.12 : 0.18)
        case "!":
            return isLong ? 0.26 : (isShort ? 0.11 : 0.16)
        case "\n":
            let extra = text.hasSuffix("\n\n") ? 0.12 : 0.0
            return 0.18 + extra
        case ".":
            return isLong ? 0.22 : (isShort ? 0.10 : 0.14)
        case ",":
            return isLong ? 0.16 : (isShort ? 0.06 : 0.10)
        default:
            return isLong ? 0.18 : (isShort ? 0.06 : 0.10)
        }
    }

    private func voiceScore(_ voice: AVSpeechSynthesisVoice) -> (Int, Int) {
        let quality: Int
        switch voice.quality {
        case .premium:
            quality = 4
        case .enhanced:
            quality = 3
        case .default:
            quality = 2
        @unknown default:
            quality = 1
        }

        let identifier = voice.identifier.lowercased()
        let identifierRank: Int
        if identifier.contains("premium") {
            identifierRank = 5
        } else if identifier.contains("enhanced") {
            identifierRank = 4
        } else if identifier.contains("eloquence") {
            identifierRank = 3
        } else if identifier.contains("compact") {
            identifierRank = 2
        } else if identifier.contains("com.apple.speech.synthesis.voice") {
            // Legacy novelty voices (Bahh, Bells, Boing, ...).
            identifierRank = 1
        } else {
            identifierRank = 0
        }

        return (quality, identifierRank)
    }

    private func bestVoice(for language: String, matching name: String? = nil) -> AVSpeechSynthesisVoice? {
        let allVoices = AVSpeechSynthesisVoice.speechVoices()
        let byLanguage = allVoices.filter { $0.language == language }
        guard !byLanguage.isEmpty else { return nil }

        if let name = name, !name.isEmpty {
            let lower = name.lowercased()
            let named = byLanguage.filter {
                $0.name.lowercased().contains(lower) ||
                $0.identifier.lowercased().contains(lower)
            }
            if !named.isEmpty {
                return named.sorted { voiceScore($0) > voiceScore($1) }.first
            }
        }

        return byLanguage.sorted { voiceScore($0) > voiceScore($1) }.first
    }

    private func resolveVoice(_ name: String) -> AVSpeechSynthesisVoice? {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)

        // If the caller passed a real AVFoundation voice identifier, use it.
        if let voice = AVSpeechSynthesisVoice(identifier: trimmed) {
            return voice
        }

        // Special names meaning "use the best available voice for this language".
        if trimmed == "Best" || trimmed == "Default" || trimmed == "en_US-amy-medium" {
            // `en_US-amy-medium` is a legacy Piper-style identifier. The native
            // AVFoundation synthesizer has no such voice; treat it as a request
            // for the highest-quality installed US English voice.
            if let voice = bestVoice(for: "en-US") {
                return voice
            }
        }

        // Common explicit names. Use the best available quality for that name
        // (premium/enhanced if the user has downloaded it, otherwise compact).
        let explicitNames = ["Samantha", "Samantha (Enhanced)", "Samantha (Premium)"]
        if explicitNames.contains(trimmed) {
            if let voice = bestVoice(for: "en-US", matching: "Samantha") {
                return voice
            }
        }

        // Try to derive a BCP-47 language tag from names like en_US-amy-medium.
        let dashPattern = trimmed.replacingOccurrences(of: "_", with: "-")
        let localePrefix = dashPattern.prefix { $0 != "-" }
        if !localePrefix.isEmpty {
            let lang = String(localePrefix)
            if let voice = bestVoice(for: lang, matching: trimmed) {
                return voice
            }
            if let voice = AVSpeechSynthesisVoice(language: lang) {
                return voice
            }
        }

        // Fallback to the best available US English voice.
        if let voice = bestVoice(for: "en-US") {
            return voice
        }
        if let voice = AVSpeechSynthesisVoice(language: "en-US") {
            return voice
        }

        // Last resort: the default system voice.
        return AVSpeechSynthesisVoice(language: "en")
    }

    private func piperBinaryPath() -> URL? {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_PIPER_BINARY"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }

        // Bundled/known venv location from the original Bad Apple install.
        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/Users/savag3"
        let knownPaths = [
            URL(fileURLWithPath: home).appendingPathComponent(".local/share/badapple/venv/bin/piper"),
            URL(fileURLWithPath: "/usr/local/bin/piper"),
            URL(fileURLWithPath: "/opt/homebrew/bin/piper"),
        ]
        for known in knownPaths {
            if fileManager.isExecutableFile(atPath: known.path) {
                return known
            }
        }

        // Fall back to PATH lookup.
        let pathEnv = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/local/bin:/usr/bin:/bin"
        for dir in pathEnv.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(dir)).appendingPathComponent("piper")
            if fileManager.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }

        return nil
    }

    private func piperVoiceSearchDirectories() -> [URL] {
        var dirs: [URL] = []

        if let env = ProcessInfo.processInfo.environment["BADAPPLE_VOICES_DIR"], !env.isEmpty {
            dirs.append(URL(fileURLWithPath: env))
        }

        let home = ProcessInfo.processInfo.environment["HOME"] ?? "/Users/savag3"
        dirs.append(URL(fileURLWithPath: home).appendingPathComponent(".local/share/badapple/voices"))
        dirs.append(URL(fileURLWithPath: home).appendingPathComponent(".bad_apple/voices"))

        // Try to locate the repo root from the binary's path.
        let exe = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0])
            .resolvingSymlinksInPath()
        var dir = exe.deletingLastPathComponent()
        for _ in 0..<8 {
            let voices = dir.appendingPathComponent("voices")
            if fileManager.fileExists(atPath: voices.path) {
                dirs.append(voices)
            }
            let parent = dir.deletingLastPathComponent()
            guard parent != dir else { break }
            dir = parent
        }

        // Sibling to the binary (e.g. app bundle Contents/Helpers/../Resources/voices).
        let sibling = exe.deletingLastPathComponent().appendingPathComponent("voices")
        if fileManager.fileExists(atPath: sibling.path) {
            dirs.append(sibling)
        }
        let resourcesSibling = exe.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Resources/voices")
        if fileManager.fileExists(atPath: resourcesSibling.path) {
            dirs.append(resourcesSibling)
        }

        return dirs
    }

    private func piperVoiceNames() -> [String] {
        var names = Set<String>()
        for voicesDir in piperVoiceSearchDirectories() {
            if let files = try? fileManager.contentsOfDirectory(atPath: voicesDir.path) {
                for file in files {
                    if file.hasSuffix(".onnx") {
                        names.insert((file as NSString).deletingPathExtension)
                    }
                }
            }
        }
        return Array(names).sorted()
    }

    private func piperModelURL(for voice: String) -> URL? {
        for voicesDir in piperVoiceSearchDirectories() {
            let model = voicesDir.appendingPathComponent("\(voice).onnx")
            if fileManager.fileExists(atPath: model.path) {
                return model
            }
        }
        return nil
    }

    private func synthesizeWithPiperIfAvailable(text: String, voice: String, lengthScale: Double, sentenceSilence: Double, outputURL: URL) -> SynthesisResult? {
        let normalized = voice.trimmingCharacters(in: .whitespacesAndNewlines)

        // Resolve the actual model name. "Best" and "Default" mean the first
        // available Piper voice (which is higher quality than AVFoundation).
        let modelName: String
        if normalized == "Best" || normalized == "Default" || normalized == "en_US-amy-medium" {
            if let first = piperVoiceNames().first {
                modelName = first
            } else {
                return nil
            }
        } else {
            guard piperModelURL(for: normalized) != nil else { return nil }
            modelName = normalized
        }

        guard let binary = piperBinaryPath(),
              let model = piperModelURL(for: modelName) else {
            return nil
        }

        // The config file is named <model>.onnx.json.
        let config = URL(fileURLWithPath: model.path + ".json")
        guard fileManager.fileExists(atPath: config.path) else { return nil }

        // Piper writes its own output file if we pass -f. We can then move it
        // to our chosen outputURL, or just pass outputURL directly. Use a temp
        // input file because passing text with special chars on the command
        // line is fragile.
        let inputURL = URL(fileURLWithPath: "/tmp/badapple_tts_input_\(UUID().uuidString).txt")
        do {
            try text.write(toFile: inputURL.path, atomically: true, encoding: .utf8)
        } catch {
            return SynthesisResult(url: outputURL, sampleRate: 0, durationMs: 0, error: error)
        }

        let process = Process()
        process.executableURL = binary
        process.arguments = [
            "-m", model.path,
            "-c", config.path,
            "-i", inputURL.path,
            "-f", outputURL.path,
            "--length-scale", "\(lengthScale)",
            "--sentence-silence", "\(sentenceSilence)",
        ]

        // Piper expects to find its espeak-ng data and libonnxruntime. Set a
        // reasonable working directory and PATH.
        let venvBin = binary.deletingLastPathComponent()
        let pathEnv = ProcessInfo.processInfo.environment["PATH"] ?? ""
        process.environment = ["PATH": "\(venvBin.path):\(pathEnv)"]
        process.currentDirectoryURL = binary.deletingLastPathComponent()

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            try? fileManager.removeItem(at: inputURL)
            return SynthesisResult(url: outputURL, sampleRate: 0, durationMs: 0, error: error)
        }

        try? fileManager.removeItem(at: inputURL)

        if process.terminationStatus != 0 {
            let stderr = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return SynthesisResult(url: outputURL, sampleRate: 0, durationMs: 0,
                                   error: TTSError.synthesisFailed("piper exited \(process.terminationStatus): \(stderr)"))
        }

        // Derive the sample rate and duration from the generated WAV.
        var sampleRate: Double = 0
        var totalFrames: AVAudioFramePosition = 0
        if let file = try? AVAudioFile(forReading: outputURL) {
            sampleRate = file.fileFormat.sampleRate
            totalFrames = file.length
        }

        let durationMs = sampleRate > 0 ? Int(Double(totalFrames) / sampleRate * 1000.0) : 0
        return SynthesisResult(url: outputURL, sampleRate: sampleRate, durationMs: durationMs, error: nil)
    }

    private func synthesizeToFile(text: String, voice: AVSpeechSynthesisVoice, rate: Float, volume: Float, outputURL: URL) -> SynthesisResult {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        utterance.rate = rate
        utterance.volume = volume
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = 0.0
        utterance.postUtteranceDelay = 0.0

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
