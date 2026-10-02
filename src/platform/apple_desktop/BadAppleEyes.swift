// BadAppleEyes — ambient sight. The retina organ.
//
// The control file at ~/.bad_apple/eyes is the single opt-in switch —
// mirrors the ears organ. While present, the menu bar app samples the main
// display every few seconds through the BadAppleScreenCapture helper — the
// binary that carries the screen-recording TCC grant — downsamples the
// frame to a small grayscale grid, and diffs it against the previous
// sample. Only when the scene has changed meaningfully is the PNG moved to
// ~/.bad_apple/eyes_frame.png. The engine daemon — which owns the vision
// model — notices the mtime change, describes the frame against the
// previous scene (temporal context = video understanding), and lands a
// `Saw:` ambient percept. Sensing is cheap every sample; understanding is
// paid only on change. No image ever leaves the machine.

import AppKit
import CoreGraphics
import Foundation

enum BadAppleEyes {
    /// Whether ambient sight is enabled. The control file is the single
    /// opt-in switch both the app (capture) and engine (percept loop) honor.
    static var enabled: Bool {
        FileManager.default.fileExists(atPath: NSHomeDirectory() + "/.bad_apple/eyes")
    }

    /// The frame handoff file — retina stages changed frames here, the
    /// engine watches mtime and runs the VLM describe.
    static var frameURL: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/eyes_frame.png")
    }

    /// The percept file the engine writes after describing a frame —
    /// mirrors ambient_heard.json.
    static var seenFileURL: URL {
        URL(fileURLWithPath: NSHomeDirectory() + "/.bad_apple/ambient_seen.json")
    }

    private static let gridW = 64
    private static let gridH = 36
    /// Fraction of downsampled pixels that must shift before the frame is
    /// considered a real scene change worth a VLM call.
    static var changeThreshold: Double {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_EYES_CHANGE"] ?? "0.12"
        return min(max(Double(env) ?? 0.12, 0.01), 1.0)
    }

    private static var prevGrid: [UInt8]?
    private static var loggedFailure = false

    /// One-shot marker so the OS prompt fires once ever — a declined or
    /// pending grant must never turn into a nagging prompt loop.
    private static var accessRequestedFlag: String {
        NSHomeDirectory() + "/.bad_apple/eyes_access_requested"
    }

    /// Capture one sample of the main display via the helper binary (the
    /// TCC-blessed capture identity), diff against the previous sample, and
    /// stage the frame for the cortex only on salient change. Fails closed:
    /// capture errors are logged once and the organ simply idles.
    static func sampleOnce() -> Bool {
        // Without the Screen Recording grant macOS silently returns the
        // bare wallpaper — the organ would "work" while seeing nothing.
        // Preflight it and ask once; the OS prompt does the rest.
        guard CGPreflightScreenCaptureAccess() else {
            if !FileManager.default.fileExists(atPath: accessRequestedFlag) {
                FileManager.default.createFile(atPath: accessRequestedFlag, contents: Data())
                NSLog("[BadAppleEyes] Screen Recording permission missing — requesting once")
                CGRequestScreenCaptureAccess()
            }
            return false
        }
        guard let helper = screenCaptureHelper() else {
            if !loggedFailure { loggedFailure = true; NSLog("[BadAppleEyes] helper binary not found") }
            return false
        }
        let tmpPath = NSTemporaryDirectory() + "badapple_eyes_sample.png"
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        let proc = Process()
        proc.executableURL = helper
        proc.arguments = ["--output", tmpPath]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do { try proc.run() } catch {
            if !loggedFailure { loggedFailure = true; NSLog("[BadAppleEyes] helper spawn failed: \(error.localizedDescription)") }
            return false
        }
        // Bounded wait — a hung helper (e.g. stalled TCC consult) must not
        // wedge the organ; terminate and idle.
        let deadline = Date().addingTimeInterval(20)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        if proc.isRunning {
            proc.terminate()
            if !loggedFailure { loggedFailure = true; NSLog("[BadAppleEyes] helper timed out — terminated") }
            return false
        }
        guard proc.terminationStatus == 0,
              let data = FileManager.default.contents(atPath: tmpPath),
              let bitmap = NSBitmapImageRep(data: data),
              let cg = bitmap.cgImage else {
            if !loggedFailure {
                loggedFailure = true
                NSLog("[BadAppleEyes] helper exit=\(proc.terminationStatus) bytes=\(FileManager.default.contents(atPath: tmpPath)?.count ?? -1) — check Screen Recording permission")
            }
            return false
        }

        guard let grid = downsampleToGray(cg) else { return false }
        defer { prevGrid = grid }
        guard let prev = prevGrid, prev.count == grid.count else {
            // First sample: stage a frame so the cortex has context.
            return stageFrame(at: tmpPath)
        }
        var changed = 0
        for i in 0..<grid.count {
            if abs(Int(grid[i]) - Int(prev[i])) > 12 { changed += 1 }
        }
        let frac = Double(changed) / Double(grid.count)
        guard frac >= changeThreshold else { return false }
        if !stageFrame(at: tmpPath) {
            if !loggedFailure { loggedFailure = true; NSLog("[BadAppleEyes] stageFrame move failed") }
            return false
        }
        return true
    }

    /// The screen-recording-blessed helper inside this app bundle.
    private static func screenCaptureHelper() -> URL? {
        let helper = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers/BadAppleScreenCapture")
        if FileManager.default.isExecutableFile(atPath: helper.path) { return helper }
        // Dev layout: next to the binary.
        if let exe = ProcessInfo.processInfo.arguments.first.map({ URL(fileURLWithPath: $0) }) {
            let candidate = exe.deletingLastPathComponent().appendingPathComponent("BadAppleScreenCapture")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// Move the sampled PNG into the handoff position — the tmp file was
    /// written atomically by the helper and the move is atomic within $HOME.
    private static func stageFrame(at tmpPath: String) -> Bool {
        let fm = FileManager.default
        do {
            try? fm.removeItem(at: frameURL)
            try fm.moveItem(atPath: tmpPath, toPath: frameURL.path)
            return true
        } catch {
            return false
        }
    }

    /// Reduce the frame to a tiny grayscale luminance grid for diffing.
    private static func downsampleToGray(_ image: CGImage) -> [UInt8]? {
        let w = gridW, h = gridH
        var data = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(
            data: &data, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return nil }
        ctx.interpolationQuality = .low
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return data
    }
}
