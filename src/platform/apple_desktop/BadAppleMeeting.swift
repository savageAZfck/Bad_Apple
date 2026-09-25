import AVFoundation
import Foundation
import Speech

// BadAppleMeeting — meeting capture organ.
//
// Like the ears, the menu bar app owns this because it holds the microphone
// and speech-recognition TCC grants. The engine's `meeting_start` tool writes
// the control file ~/.bad_apple/meeting_record; the app's poll sees it and
// starts recording mic audio to a wav in ~/.bad_apple/meetings/. Removing the
// control file (`meeting_stop`) ends the capture, transcribes the wav with
// Apple's on-device recognizer (SFSpeechURLRecognitionRequest —
// requiresOnDeviceRecognition = true, so no audio ever leaves the machine),
// writes the transcript JSON next to the wav, and deletes the audio.
//
// The wav is transient: only the transcript persists. Private mode blocks the
// control file at the tool layer, so recording never starts.

enum BadAppleMeeting {
    /// Presence of this file means "be recording"; absence means "stop".
    static let controlFileURL =
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/meeting_record")

    /// Finished transcripts land here as <stamp>.json plus latest.json.
    static let directoryURL =
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/meetings")

    static var isRequested: Bool {
        FileManager.default.fileExists(atPath: controlFileURL.path)
    }
}

/// Records the mic to a wav file while the meeting control file exists.
/// All capture must be driven from the app's meeting poll — the recorder is
/// not self-scheduling so the control file stays the single source of truth.
final class BadAppleMeetingRecorder {
    private var audioEngine: AVAudioEngine?
    private var audioFile: AVAudioFile?
    private var wavURL: URL?

    private(set) var isRecording = false
    private(set) var startedAt: Date?

    /// Start recording. Returns false (recording simply doesn't begin) when
    /// the mic grant is missing or the engine cannot start — callers treat
    /// false as "not recording", never an error to surface.
    @discardableResult
    func start() -> Bool {
        guard !isRecording else { return true }
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .authorized else {
            return false
        }

        try? FileManager.default.createDirectory(
            at: BadAppleMeeting.directoryURL,
            withIntermediateDirectories: true
        )
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let url = BadAppleMeeting.directoryURL
            .appendingPathComponent("meeting_\(stamp).caf")

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { return false }
        guard let file = try? AVAudioFile(forWriting: url, settings: format.settings) else {
            return false
        }

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
            try? file.write(from: buffer)
        }
        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            return false
        }

        audioEngine = engine
        audioFile = file
        wavURL = url
        startedAt = Date()
        isRecording = true
        return true
    }

    /// Stop recording and hand back the audio file for transcription.
    @discardableResult
    func stop() -> (audioURL: URL?, startedAt: Date?, duration: TimeInterval) {
        guard isRecording else { return (nil, nil, 0) }
        let duration = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        let started = startedAt
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        audioFile = nil
        isRecording = false
        startedAt = nil
        return (wavURL, started, duration)
    }
}

/// Transcribes a finished recording entirely on-device. Same privacy contract
/// as the ears: requiresOnDeviceRecognition, nothing leaves the Mac.
enum BadAppleMeetingTranscriber {
    static func transcribe(url: URL, completion: @escaping (String) -> Void) {
        // Whisper backend when a runner is installed; Apple Speech otherwise.
        if let text = BadAppleASR.transcribeFile(url: url) {
            completion(text)
            return
        }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized,
              let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
              recognizer.isAvailable,
              recognizer.supportsOnDeviceRecognition else {
            completion("")
            return
        }
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        recognizer.recognitionTask(with: request) { result, error in
            if let result, result.isFinal {
                completion(result.bestTranscription.formattedString)
            } else if error != nil {
                completion("")
            }
        }
    }
}
