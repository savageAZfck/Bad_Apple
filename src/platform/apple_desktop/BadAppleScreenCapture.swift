import AppKit
import Dispatch
import Foundation
import ScreenCaptureKit

@main
struct BadAppleScreenCapture {
    static func main() {
        let args = CommandLine.arguments
        guard let outputIndex = args.firstIndex(of: "--output"), outputIndex + 1 < args.count else {
            print("Usage: BadAppleScreenCapture --output <path>")
            exit(1)
        }
        let path = args[outputIndex + 1]
        let parent = URL(fileURLWithPath: path).deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true, attributes: nil)

        let semaphore = DispatchSemaphore(value: 0)
        var capturedImage: CGImage?
        var captureError: Error?

        var captureRect = CGRect.infinite
        if let mainScreen = NSScreen.main {
            captureRect = mainScreen.frame
        }

        SCScreenshotManager.captureImage(in: captureRect) { image, error in
            capturedImage = image
            captureError = error
            semaphore.signal()
        }

        _ = semaphore.wait(timeout: .now() + 30)

        if let error = captureError {
            print("Screen capture failed: \(error.localizedDescription)")
            exit(1)
        }

        guard let image = capturedImage else {
            print("Screen capture returned no image")
            exit(1)
        }

        let bitmap = NSBitmapImageRep(cgImage: image)
        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            print("Failed to encode PNG")
            exit(1)
        }

        do {
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
            print(path)
            exit(0)
        } catch {
            print("Failed to write image: \(error.localizedDescription)")
            exit(1)
        }
    }
}
