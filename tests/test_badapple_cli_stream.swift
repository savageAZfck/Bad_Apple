import Foundation
import Darwin

struct RunnerError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

func runProcess(_ executable: URL, arguments: [String], timeout: TimeInterval) throws {
    let process = Process()
    let ended = DispatchSemaphore(value: 0)
    process.executableURL = executable
    process.arguments = arguments
    process.standardInput = FileHandle.nullDevice
    process.terminationHandler = { _ in ended.signal() }
    try process.run()
    if ended.wait(timeout: .now() + timeout) == .timedOut {
        process.terminate()
        if ended.wait(timeout: .now() + 2) == .timedOut {
            kill(process.processIdentifier, SIGKILL)
            _ = ended.wait(timeout: .now() + 2)
        }
        throw RunnerError(message: "\(executable.lastPathComponent) timed out after \(Int(timeout)) seconds")
    }
    guard process.terminationReason == .exit, process.terminationStatus == 0 else {
        throw RunnerError(message: "\(executable.lastPathComponent) failed with status \(process.terminationStatus) (\(process.terminationReason))")
    }
}

let preamble = ##"""
import Foundation

struct BadAppleMenuBarError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
enum BadAppleBrain { static let keyPath = "/tmp/test-key" }
func badAppleVoiceLog(_ message: String) {}
final class Harness {
    let voiceQueue = DispatchQueue(label: "test.voice")
    let testBundleURL: URL
    init(_ root: URL) { testBundleURL = root }
"""##

let tests = ##"""
}
private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fputs("FAIL: \(message)\n", stderr)
        exit(1)
    }
}
@main
private enum StreamingTests {
    static func main() async throws {
        let workspace = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        func run(_ script: String, timeout: Double = 3, executable: Bool = true, slow: Bool = false) async -> (Result<String, Error>, String) {
            let root = workspace.appendingPathComponent("bundle-\(UUID().uuidString)")
            let helpers = root.appendingPathComponent("Contents/Helpers")
            try! FileManager.default.createDirectory(at: helpers, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let binary = helpers.appendingPathComponent("badapple")
            let harness = Harness(root)
            try! ("#!/bin/sh\n" + script).write(to: binary, atomically: true, encoding: .utf8)
            try! FileManager.default.setAttributes([.posixPermissions: executable ? 0o755 : 0o644], ofItemAtPath: binary.path)
            var chunks = ""
            let result: Result<String, Error>
            do {
                let text = try await harness.runBadAppleCLIStreaming(prompt: "test", socketPath: "/tmp/test-daemon.sock", maxTokens: 10, timeout: timeout, minimalEnv: true) { chunk in
                    if slow { Thread.sleep(forTimeInterval: 0.03) }
                    chunks += chunk
                }
                result = .success(text)
            } catch {
                result = .failure(error)
            }
            let completedChunks = chunks
            try? await Task.sleep(nanoseconds: 30_000_000)
            require(chunks == completedChunks, "no callbacks after resolution")
            return (result, chunks)
        }
        func failure(_ result: Result<String, Error>) -> String {
            guard case .failure(let error) = result else { fatalError("expected failure, got \(result)") }
            return error.localizedDescription
        }
        func success(_ result: Result<String, Error>) -> String {
            guard case .success(let text) = result else { fatalError("expected success, got \(result)") }
            return text
        }

        let stderrOnly = await run("printf 'Cannot connect to daemon: Permission denied' >&2; exit 1")
        let stderrMessage = failure(stderrOnly.0)
        require(stderrMessage.contains("code 1") && stderrMessage.contains("Permission denied"), "stderr-only failure detail")
        require(stderrMessage.contains("/tmp/test-daemon.sock") && stderrMessage.contains("badapple --doctor"), "actionable failure guidance")
        require(stderrOnly.1.isEmpty, "stderr does not become tokens")

        let partial = await run(#"printf '%s\n' '{"type":"token","text":"partial"}'; printf 'generation failed' >&2; exit 1"#)
        require(failure(partial.0).contains("generation failed") && partial.1 == "partial", "partial text cannot hide nonzero exit")
        let doneFailure = await run(#"printf '%s\n' '{"type":"done","text":"complete"}'; exit 7"#)
        require(failure(doneFailure.0).contains("code 7"), "done cannot hide nonzero exit without stderr")

        let utf8 = await run(#"printf '{"type":"token","text":"caf\303'; sleep 0.05; printf '\251"}\n'; printf '{"type":"done","text":"caf\303'; sleep 0.05; printf '\251 final"}'"#)
        require(success(utf8.0) == "café final" && utf8.1 == "café", "split UTF-8 and unterminated done")
        let tailToken = await run(#"printf '%s' '{"type":"token","text":"tail"}'"#)
        require(success(tailToken.0) == "tail" && tailToken.1 == "tail", "unterminated token is delivered")
        let splitError = await run(#"printf 'caf\303' >&2; sleep 0.05; printf '\251 failure' >&2; exit 1"#)
        require(failure(splitError.0).contains("café failure"), "stderr UTF-8 split across reads")

        let largeError = await run(#"/usr/bin/awk 'BEGIN { for (i=0; i<200000; i++) printf "x"; printf "\nlast diagnostic\n" }' >&2; exit 1"#)
        let bounded = failure(largeError.0)
        require(bounded.contains("[stderr truncated]") && bounded.contains("last diagnostic"), "bounded stderr retains useful tail")
        require(bounded.utf8.count < 17000, "stderr error length bounded")

        let interleaved = await run(#"printf '%s\n' '{"type":"token","text":"a"}'; printf '%s\n' '{"type":"token","text":"not stdout"}' >&2; printf '%s\n' 'ordinary log' '{"type":"token","text":"b"}' '{"type":"done","text":"ab"}'"#)
        require(success(interleaved.0) == "ab" && interleaved.1 == "ab", "stderr JSON and stdout logs cannot corrupt token stream")
        let empty = await run("exit 0")
        require(success(empty.0).isEmpty && empty.1.isEmpty, "empty successful exit remains supported")

        for iteration in 0..<25 {
            let raced = await run(#"/usr/bin/awk 'BEGIN { for (i=0; i<4000; i++) print "{\"type\":\"token\",\"text\":\"x\"}"; printf "{\"type\":\"done\",\"text\":\"final\"}" }'"#, slow: true)
            require(success(raced.0) == "final" && raced.1 == String(repeating: "x", count: 4000), "EOF drain iteration \(iteration)")
        }

        let lateStderr = await run(#"(sleep 0.1; printf 'late diagnostic' >&2) & exit 1"#)
        require(failure(lateStderr.0).contains("late diagnostic"), "stderr drains after parent exit")
        let lateStdout = await run(#"(sleep 0.1; printf '%s' '{"type":"token","text":"late"}') & exit 0"#)
        require(success(lateStdout.0) == "late" && lateStdout.1 == "late", "stdout drains after parent exit")
        let timeout = await run(#"printf '%s\n' '{"type":"token","text":"partial"}'; printf 'still waiting' >&2; exec /bin/sleep 3"#, timeout: 1.0)
        require(failure(timeout.0).contains("timed out") && failure(timeout.0).contains("still waiting"), "timeout remains failure with partial text")
        let launch = await run("exit 0", executable: false)
        require(failure(launch.0).contains("Could not launch") && failure(launch.0).contains("badapple"), "launch failure is actionable")
        let missingRoot = workspace.appendingPathComponent("missing-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: missingRoot, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: missingRoot) }
        do {
            _ = try await Harness(missingRoot).runBadAppleCLIStreaming(prompt: "test", socketPath: "/tmp/test-daemon.sock", maxTokens: 10) { _ in }
            fatalError("missing helper must fail")
        } catch {
            require(error.localizedDescription.contains("missing from the app bundle"), "missing-helper error preserved")
        }
        print("Bad Apple CLI streaming isolated regressions passed (15 cases, 25 drain-race iterations)")
    }
}
"""##

func main() throws {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    let sourceURL = repo.appendingPathComponent("src/platform/apple_desktop/BadAppleMenuBar.swift")
    let source = try String(contentsOf: sourceURL, encoding: .utf8)
    let startMarker = "    private func runBadAppleCLIStreaming("
    let endMarker = "\n    private func runBadAppleCLI("
    guard let start = source.range(of: startMarker),
          let end = source.range(of: endMarker, range: start.upperBound..<source.endIndex) else {
        throw RunnerError(message: "Cannot locate streaming method boundaries in \(sourceURL.path)")
    }
    let extracted = String(source[start.lowerBound..<end.lowerBound])
    guard extracted.contains("Bundle.main.bundleURL") else {
        throw RunnerError(message: "Streaming method no longer contains the expected bundle injection point")
    }
    let method = extracted
        .replacingOccurrences(of: "private func runBadAppleCLIStreaming", with: "func runBadAppleCLIStreaming")
        .replacingOccurrences(of: "Bundle.main.bundleURL", with: "testBundleURL")
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent("badapple-stream-compile-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: workspace) }
    let swift = workspace.appendingPathComponent("Streaming.swift")
    let executable = workspace.appendingPathComponent("streaming-tests")
    try (preamble + "\n" + method + "\n" + tests + "\n").write(to: swift, atomically: true, encoding: .utf8)
    try runProcess(URL(fileURLWithPath: "/usr/bin/xcrun"), arguments: ["swiftc", "-swift-version", "5", "-parse-as-library", swift.path, "-o", executable.path], timeout: 90)
    try runProcess(executable, arguments: [workspace.path], timeout: 45)
}

do {
    try main()
} catch {
    fputs("FAIL: \(error.localizedDescription)\n", stderr)
    exit(1)
}
