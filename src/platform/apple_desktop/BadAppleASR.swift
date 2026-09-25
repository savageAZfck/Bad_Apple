import AVFoundation
import Foundation

// BadAppleASR — backend-swappable speech recognition seam.
//
// Two backends:
//   - `apple-speech` — Apple's on-device SFSpeechRecognizer. Always available
//     once TCC grants exist; nothing leaves the Mac. The default.
//   - `whisper` — activated automatically when a runtime is present. The
//     runner contract is a single executable that takes a wav path as argv[1]
//     and prints the transcript on stdout. Resolution order:
//       1. `BADAPPLE_WHISPER_BIN` env var → executable path
//       2. `~/.bad_apple/whisper/run.sh` → user-dropped runner
//     A runner can wrap anything local: mlx_whisper (the
//     mlx-community/whisper-large-v3-turbo-asr-fp16 weights are already in
//     the HF cache), whisper.cpp `main`, faster-whisper, etc. Example:
//       #!/bin/sh
//       exec python3 -m mlx_whisper "$1" --model mlx-community/whisper-large-v3-turbo-asr-fp16
//
// Both ears (bounded ambient windows) and meetings (recorded wav files) route
// through `transcribeFile`, so dropping one runner upgrades every speech
// organ at once — and removing it falls straight back to Apple Speech.
enum BadAppleASR {

    /// Active backend for capability/self-report surfaces.
    static var backendName: String {
        whisperRunner != nil ? "whisper" : "apple-speech"
    }

    /// Path to the whisper runner executable, or nil when none is installed.
    static var whisperRunner: String? {
        if let env = ProcessInfo.processInfo.environment["BADAPPLE_WHISPER_BIN"],
           FileManager.default.isExecutableFile(atPath: env) {
            return env
        }
        let local = NSHomeDirectory() + "/.bad_apple/whisper/run.sh"
        return FileManager.default.isExecutableFile(atPath: local) ? local : nil
    }

    /// Transcribes a recorded wav through the whisper runner. Returns nil when
    /// no whisper runtime is installed or the run fails — callers fall back
    /// to the Apple on-device recognizer.
    static func transcribeFile(url: URL) -> String? {
        guard let runner = whisperRunner else { return nil }
        let process = Process()
        let out = Pipe()
        process.executableURL = URL(fileURLWithPath: runner)
        process.arguments = [url.path]
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }

        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { process.waitUntilExit(); done.signal() }
        // Whisper on a long meeting can take a while; bound at 10 min.
        guard done.wait(timeout: .now() + 600) == .success else {
            process.terminate()
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        let text = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return text.isEmpty ? nil : text
    }

    /// Records `seconds` of mic audio to a wav file (for whisper — the Apple
    /// streaming path feeds buffers directly). Returns nil on failure.
    /// Caller owns the returned file.
    static func recordWav(seconds: Int) -> URL? {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else { return nil }
        let window = min(max(TimeInterval(seconds), 2), 15)
        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { return nil }

        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("badapple-asr-\(UUID().uuidString).wav")
        guard let file = try? AVAudioFile(forWriting: url, settings: format.settings) else { return nil }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            try? file.write(from: buffer)
        }
        do {
            engine.prepare()
            try engine.start()
        } catch {
            return nil
        }
        let deadline = Date().addingTimeInterval(window)
        while Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
        }
        engine.stop()
        input.removeTap(onBus: 0)
        return url
    }
}
