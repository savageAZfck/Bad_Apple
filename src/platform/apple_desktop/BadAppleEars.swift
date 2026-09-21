import AVFoundation
import Foundation
import Speech

// BadAppleEars — bounded ambient hearing.
//
// The menu bar app owns this organ: it already holds the microphone and
// speech-recognition TCC grants, so the capture runs inside the app process
// rather than a standalone helper (which would carry no TCC identity and
// hard-abort on the first capture API call).
//
// On a timer the app checks ~/.bad_apple/ears — the control file is the single
// opt-in switch — and when present captures a bounded window, transcribes it
// with Apple's on-device recognizer (requiresOnDeviceRecognition = true, so no
// audio ever leaves the machine), and writes the percept to
// ~/.bad_apple/ambient_heard.json for the engine's ambient loop.
//
// The capture is synchronous and must run on a background thread — it pumps a
// private RunLoop for the window duration while AVAudioEngine and the
// recognition task deliver callbacks.

struct BadAppleEars {
    /// Capture `seconds` of microphone audio and return the best transcript.
    /// Returns "" on silence, recognizer failure, or missing permission —
    /// callers treat empty as "nothing heard", never as an error to surface.
    static func captureOnce(seconds: Int) -> String {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized,
              AVCaptureDevice.authorizationStatus(for: .audio) == .authorized,
              let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
              recognizer.isAvailable,
              recognizer.supportsOnDeviceRecognition else {
            return ""
        }

        let window = min(max(TimeInterval(seconds), 2), 15)
        let audioEngine = AVAudioEngine()
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true

        var bestTranscript = ""
        var finished = false
        let done = DispatchSemaphore(value: 0)

        let task = recognizer.recognitionTask(with: request) { result, error in
            if let result {
                bestTranscript = result.bestTranscription.formattedString
                if result.isFinal { finished = true; done.signal() }
            }
            if error != nil { finished = true; done.signal() }
        }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            task.cancel()
            return ""
        }
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            task.cancel()
            return ""
        }

        let deadline = Date().addingTimeInterval(window)
        while !finished && Date() < deadline {
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
        }

        request.endAudio()
        audioEngine.stop()
        inputNode.removeTap(onBus: 0)

        // Short tail so the recognizer can finalize its last hypothesis.
        _ = done.wait(timeout: .now() + 2.5)
        task.cancel()
        return bestTranscript
    }

    /// Whether ambient hearing is enabled. The control file is the single
    /// opt-in switch both the app (capture) and engine (injection) honor.
    static var enabled: Bool {
        FileManager.default.fileExists(atPath: NSHomeDirectory() + "/.bad_apple/ears")
    }

    /// The file the app writes its latest percept to and the engine reads.
    static var heardFileURL: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ambient_heard.json")
    }
}
