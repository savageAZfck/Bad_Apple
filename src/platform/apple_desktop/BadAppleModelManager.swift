import Foundation
import CryptoKit
import MachO

// MARK: - Model Profile

/// Static description of a model the user can select or download.
struct ModelProfile: Codable, Identifiable {
    let id: String
    let name: String
    let repoId: String
    let sizeGB: Double
    let kind: String  // "text", "vision", "image", "draft"
    let loadedIn: String
}

// MARK: - Model State

/// Runtime download/load state for a single model.
final class ModelState: Codable {
    enum Status: String, Codable {
        case missing, queued, downloading, cached, loaded, error
    }

    var status: Status
    var progress: Double
    var error: String
    var localPath: String
    var downloadedBytes: Int64
    var totalBytes: Int64
    var lastUpdated: TimeInterval
    var verified: Bool

    init(
        status: Status = .missing,
        progress: Double = 0.0,
        error: String = "",
        localPath: String = "",
        downloadedBytes: Int64 = 0,
        totalBytes: Int64 = 0,
        lastUpdated: TimeInterval = Date().timeIntervalSince1970,
        verified: Bool = false
    ) {
        self.status = status
        self.progress = progress
        self.error = error
        self.localPath = localPath
        self.downloadedBytes = downloadedBytes
        self.totalBytes = totalBytes
        self.lastUpdated = lastUpdated
        self.verified = verified
    }
}

// MARK: - Model Manifest

/// Provenance manifest: SHA-256 fingerprints for every file in a cached model.
struct ModelFileEntry: Codable {
    let relative: String
    let size: Int64
    let mtime: TimeInterval
    let sha256: String
}

struct ModelManifest: Codable {
    let repoId: String
    let localPath: String
    let recordedAt: TimeInterval
    var files: [String: ModelFileEntry]
    var signature: String?
    var publicKey: String?
}

// MARK: - Model Manager

/// Background model download and status manager.
///
/// Tracks the known model profiles, scans the HuggingFace cache, queues
/// downloads, records SHA-256 provenance manifests, and recommends models
/// based on available memory.
final class BadAppleModelManager {
    static let shared = BadAppleModelManager()

    private static let knownProfiles: [ModelProfile] = [
        ModelProfile(
            id: "fast_0.5b",
            name: "Fast 0.5B",
            repoId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            sizeGB: 0.35,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "main_9b",
            name: "Deep 9B",
            repoId: "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
            sizeGB: 4.8,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "coder_7b",
            name: "Coder 7B",
            repoId: "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            sizeGB: 4.3,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "main_32b",
            name: "Deep 32B",
            repoId: "mlx-community/Qwen3.5-32B-MLX-4bit",
            sizeGB: 19.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "main_70b",
            name: "Deep 70B",
            repoId: "mlx-community/Llama-3.3-70B-Instruct-4bit",
            sizeGB: 40.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "moe_80b",
            name: "Hybrid 80B MoE",
            repoId: "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
            sizeGB: 45.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "oss_120b",
            name: "GPT-OSS 120B MoE",
            repoId: "mlx-community/gpt-oss-120b-MXFP4-Q4",
            sizeGB: 65.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "moe_235b",
            name: "Deep 235B MoE",
            repoId: "mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit",
            sizeGB: 132.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        // GLM4MOE arch is supported by the bundled mlx-swift-lm build;
        // Llama-4 (llama4 arch) is not, so this fills the ~200 GB slot.
        ModelProfile(
            id: "glm_355b",
            name: "GLM-4.5 355B MoE",
            repoId: "mlx-community/GLM-4.5-4bit",
            sizeGB: 200.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "coder_480b",
            name: "Coder 480B MoE",
            repoId: "mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit",
            sizeGB: 273.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        // Peak single-node brain: needs a maxed-out Studio (~512GB unified).
        ModelProfile(
            id: "r1_671b",
            name: "DeepSeek R1 671B",
            repoId: "mlx-community/DeepSeek-R1-0528-4bit",
            sizeGB: 380.0,
            kind: "text",
            loadedIn: "mlx_server"
        ),
        ModelProfile(
            id: "vision_2b",
            name: "Ocular Vision",
            repoId: "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            sizeGB: 1.4,
            kind: "vision",
            loadedIn: "vision_host"
        ),
        ModelProfile(
            id: "flux_4b",
            name: "FLUX.2-klein 4B",
            repoId: "mflux/flux2-klein-4b",
            sizeGB: 4.0,
            kind: "image",
            loadedIn: "mflux"
        ),
    ]

    private let dataDir: URL
    private let statusFile: URL
    private let manifestDir: URL
    private let lock = NSLock()
    private var profiles: [String: ModelProfile] = [:]
    private var state: [String: ModelState] = [:]
    private var downloadTasks: [String: Process] = [:]
    private let downloadQueue = OperationQueue()

    private let maxModelIdLength = 64
    private let maxRepoIdLength = 128
    private let defaultMaxDownloadGB = 50.0
    private let defaultDownloadTimeout: TimeInterval = 1800
    private let defaultProvenanceTimeout: TimeInterval = 300
    private let defaultHashCap: Int64 = 10 * 1024 * 1024 * 1024 // 10 GB total

    var allowDownloads: Bool {
        get { return lock.withLock { _allowDownloads || _onlineOverride } }
        set { lock.withLock { _allowDownloads = newValue; _onlineOverride = newValue } }
    }
    private var _allowDownloads: Bool = false
    private var _onlineOverride: Bool = false

    private var maxDownloadGB: Double {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_MAX_DOWNLOAD_GB"]
        if let raw = env, let val = Double(raw), val > 0 { return val }
        return defaultMaxDownloadGB
    }

    private var downloadTimeout: TimeInterval {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_DOWNLOAD_TIMEOUT_SECONDS"]
        if let raw = env, let val = Double(raw), val > 0 { return val }
        return defaultDownloadTimeout
    }

    private var provenanceTimeout: TimeInterval {
        let env = ProcessInfo.processInfo.environment["BADAPPLE_PROVENANCE_TIMEOUT_SECONDS"]
        if let raw = env, let val = Double(raw), val > 0 { return val }
        return defaultProvenanceTimeout
    }

    private var verifyHashes: Bool {
        ProcessInfo.processInfo.environment["BADAPPLE_VERIFY_MODEL_HASHES"] != "0"
    }

    /// Memory headroom multiplier for load-time fit checks. Dense models
    /// keep ~40% overhead for KV cache and runtime; giant MoE models have
    /// far less cache relative to parameter count, so a flat 1.4x would
    /// make a 380 GB model unreachable on a 512 GB machine.
    private func headroomFactor(_ sizeGB: Double) -> Double {
        sizeGB >= 100.0 ? 1.15 : 1.4
    }

    /// Provenance hashing budget scaled to the model's declared size.
    /// A flat 10 GB cap silently skips shards on 100 GB+ models.
    private func hashCapFor(_ profile: ModelProfile) -> Int64 {
        Int64(max(10.0, profile.sizeGB * 1.3) * 1_073_741_824.0)
    }

    /// Hashing a 380 GB model exceeds the default 5-minute budget even on
    /// fast NVMe; scale the deadline by model size (~2 s/GB floor).
    private func provenanceTimeoutFor(_ profile: ModelProfile) -> TimeInterval {
        max(provenanceTimeout, profile.sizeGB * 2.0)
    }

    /// Download deadline scaled to model size (~1 min/GB floor) so a
    /// multi-hundred-GB pull isn't killed at the 30-minute default.
    private func downloadTimeoutFor(_ profile: ModelProfile) -> TimeInterval {
        max(downloadTimeout, profile.sizeGB * 60.0)
    }

    private var hfCacheRoot: URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".cache/huggingface/hub")
    }

    init() {
        if let dataDirEnv = ProcessInfo.processInfo.environment["BADAPPLE_DATA_DIR"] {
            dataDir = URL(fileURLWithPath: dataDirEnv)
        } else {
            dataDir = URL(fileURLWithPath: "/var/lib/bad_apple")
        }
        try? FileManager.default.createDirectory(at: dataDir, withIntermediateDirectories: true)
        statusFile = dataDir.appendingPathComponent("model_manager.json")
        manifestDir = dataDir.appendingPathComponent("model_manifests")
        try? FileManager.default.createDirectory(at: manifestDir, withIntermediateDirectories: true)

        for profile in Self.knownProfiles {
            profiles[profile.id] = profile
            state[profile.id] = ModelState()
        }

        _allowDownloads = ProcessInfo.processInfo.environment["BADAPPLE_ALLOW_DOWNLOADS"] == "1"
        _onlineOverride = ProcessInfo.processInfo.environment["BADAPPLE_ONLINE_MODELS"] == "1"

        loadState()
        downloadQueue.maxConcurrentOperationCount = 2
        downloadQueue.name = "com.badapple.modelmanager.downloads"
    }

    // MARK: - Profiles and Status

    func listProfiles() -> [ModelProfile] {
        lock.lock(); defer { lock.unlock() }
        return Array(profiles.values)
    }

    func status(modelId: String? = nil) -> [[String: Any]] {
        lock.withLock {
            if let modelId = modelId {
                return [statusDictionary(for: modelId)]
            }
            return profiles.keys.sorted().map { statusDictionary(for: $0) }
        }
    }

    func modelStatus(modelId: String) -> [String: Any]? {
        lock.withLock {
            guard isSafeModelId(modelId), profiles[modelId] != nil else { return nil }
            return statusDictionary(for: modelId)
        }
    }

    private func statusDictionary(for modelId: String) -> [String: Any] {
        guard let profile = profiles[modelId], let state = state[modelId] else {
            return ["error": "unknown model \(modelId)"]
        }
        // Provenance verification does file I/O and hashing; status() is called
        // frequently while holding the model manager lock, so keep this quick.
        // The verified flag is updated by markLoaded/recordProvenance.
        var provenance: [String: Any] = ["status": "unknown"]
        if state.verified {
            provenance = ["status": "verified"]
        } else if !state.error.isEmpty {
            provenance = ["status": "error", "error": state.error]
        }
        return [
            "id": profile.id,
            "name": profile.name,
            "repo_id": profile.repoId,
            "kind": profile.kind,
            "size_gb": profile.sizeGB,
            "status": state.status.rawValue,
            "progress": round(state.progress * 1000) / 1000,
            "error": state.error,
            "local_path": state.localPath,
            "allow_downloads": _allowDownloads || _onlineOverride,
            "loaded_in": profile.loadedIn,
            "verified": state.verified,
            "provenance": provenance,
            "last_updated": state.lastUpdated,
        ]
    }

    // MARK: - Cache Scanning

    /// Find cached models in the HF cache, matching known profiles.
    func refreshCacheStatus(modelId: String) -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId), let profile = profiles[modelId], let state = self.state[modelId] else {
            return ["error": "unknown or invalid model_id: \(modelId)"]
        }
        if state.status == .downloading || state.status == .queued || state.status == .loaded {
            return statusDictionary(for: modelId)
        }
        if let path = resolveCachePath(repoId: profile.repoId, allowDownload: false) {
            state.localPath = path
            state.status = .cached
            state.progress = 1.0
            state.error = ""
            state.lastUpdated = Date().timeIntervalSince1970
        } else {
            state.localPath = ""
            state.status = .missing
            state.progress = 0.0
        }
        saveState()
        return statusDictionary(for: modelId)
    }

    func backgroundRefreshAll() {
        for modelId in profiles.keys {
            downloadQueue.addOperation { [weak self] in
                _ = self?.refreshCacheStatus(modelId: modelId)
            }
        }
    }

    // MARK: - Downloads

    /// Queue a background download using the native `badapple-fetch` Rust helper.
    /// Downloads only happen if `BADAPPLE_ALLOW_DOWNLOADS=1`.
    func startDownload(modelId: String) -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId), let profile = profiles[modelId], let state = self.state[modelId] else {
            return ["error": "unknown or invalid model_id: \(modelId)"]
        }
        if !(_allowDownloads || _onlineOverride) {
            return ["error": "Downloads are disabled. Set BADAPPLE_ALLOW_DOWNLOADS=1 or enable in the dashboard."]
        }
        if downloadTasks[modelId] != nil {
            return statusDictionary(for: modelId)
        }
        if state.status == .cached || state.status == .loaded {
            return statusDictionary(for: modelId)
        }

        state.status = .queued
        state.progress = 0.0
        state.error = ""
        state.lastUpdated = Date().timeIntervalSince1970
        saveState()

        // Use the native Rust helper. Falls back to environment overrides.
        let timeout = downloadTimeoutFor(profile)
        let repoId = profile.repoId
        guard let helperURL = fetchHelperURL() else {
            state.status = .error
            state.error = "badapple-fetch helper not found; run cargo build --release --bin badapple-fetch"
            state.progress = 0.0
            state.lastUpdated = Date().timeIntervalSince1970
            saveState()
            return statusDictionary(for: modelId)
        }

        let task = Process()
        task.executableURL = helperURL
        task.arguments = [repoId]

        // Inherit the parent environment, then explicitly allow the fetch helper
        // to reach the network. The daemon itself keeps HF_HUB_OFFLINE=1, but
        // when the user opts in with BADAPPLE_ALLOW_DOWNLOADS=1, the helper
        // must see HF_HUB_OFFLINE=0 or it will refuse immediately.
        var taskEnv = ProcessInfo.processInfo.environment
        taskEnv["HF_HUB_OFFLINE"] = "0"
        taskEnv["BADAPPLE_ALLOW_DOWNLOADS"] = "1"
        task.environment = taskEnv

        // Run the download in the background and poll isRunning (same pattern
        // BadAppleTools.runProcess uses; this works without a Foundation run loop).
        // A separate DispatchWorkItem acts as the timeout watchdog.
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        task.standardOutput = stdoutPipe
        task.standardError = stderrPipe

        let timeoutWorkItem = DispatchWorkItem { [weak self] in
            self?.downloadTimedOut(modelId: modelId)
        }
        DispatchQueue.global(qos: .utility).asyncAfter(
            deadline: .now() + timeout,
            execute: timeoutWorkItem
        )

        downloadTasks[modelId] = task
        DispatchQueue.global(qos: .utility).async { [weak self] in
            var stderrData = Data()
            let group = DispatchGroup()
            DispatchQueue.global().async(group: group) {
                _ = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            }
            DispatchQueue.global().async(group: group) {
                stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            }

            let semaphore = DispatchSemaphore(value: 0)
            task.terminationHandler = { _ in
                semaphore.signal()
            }

            do {
                try task.run()
                let waitResult = semaphore.wait(timeout: .now() + timeout)
                group.wait()
                timeoutWorkItem.cancel()
                let stderr = String(data: stderrData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if waitResult == .timedOut {
                    self?.downloadTimedOut(modelId: modelId)
                } else {
                    self?.downloadCompleted(
                        modelId: modelId,
                        exitCode: task.terminationStatus,
                        stderr: stderr
                    )
                }
            } catch {
                group.wait()
                timeoutWorkItem.cancel()
                self?.downloadFailedToStart(modelId: modelId, error: error)
            }
        }

        return statusDictionary(for: modelId)
    }

    private func downloadTimedOut(modelId: String) {
        lock.lock(); defer { lock.unlock() }
        if let task = downloadTasks[modelId] {
            task.terminate()
            downloadTasks.removeValue(forKey: modelId)
        }
        guard let state = state[modelId] else { return }
        state.status = .error
        state.error = "download timed out after \(downloadTimeout) seconds"
        state.lastUpdated = Date().timeIntervalSince1970
        saveState()
    }

    private func downloadFailedToStart(modelId: String, error: Error) {
        lock.lock(); defer { lock.unlock() }
        downloadTasks.removeValue(forKey: modelId)
        guard let state = state[modelId] else { return }
        state.status = .error
        state.error = "failed to start download: \(error.localizedDescription)"
        state.progress = 0.0
        state.lastUpdated = Date().timeIntervalSince1970
        saveState()
    }

    private func downloadCompleted(modelId: String, exitCode: Int32? = nil, stderr: String = "") {
        lock.lock(); defer { lock.unlock() }
        downloadTasks.removeValue(forKey: modelId)
        guard let profile = profiles[modelId], let state = self.state[modelId] else { return }

        if let path = resolveCachePath(repoId: profile.repoId, allowDownload: false) {
            state.localPath = path
            state.status = .cached
            state.progress = 1.0
            state.error = ""
            state.lastUpdated = Date().timeIntervalSince1970

            if verifyHashes {
                DispatchQueue.global(qos: .utility).async { [weak self] in
                    _ = self?.recordProvenance(modelId: modelId, localPath: path)
                }
            }
        } else if let exitCode = exitCode, exitCode != 0 {
            state.status = .error
            if state.error.isEmpty {
                let stderrHint = stderr.isEmpty ? "" : " (stderr: \(stderr))"
                state.error = "download failed (exit code \(exitCode))\(stderrHint). Make sure badapple-fetch is built and allow_downloads is enabled."
            }
            state.progress = 0.0
        } else {
            state.status = .error
            state.error = "download completed but model not found in HF cache"
            state.progress = 0.0
        }
        saveState()
    }

    /// Locate the native `badapple-fetch` Rust helper. Do not honour the
    /// BADAPPLE_FETCH env var; an untrusted path could be a malicious binary.
    /// Search the executable's directory (dev builds), the app bundle, and the
    /// standard PATH directories.
    private func fetchHelperURL() -> URL? {
        let fm = FileManager.default
        let exe = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[0])
        let sameDir = exe.deletingLastPathComponent().appendingPathComponent("badapple-fetch")
        if fm.isExecutableFile(atPath: sameDir.path) {
            return sameDir
        }

        // Build script installs helpers into Contents/Helpers alongside the main
        // executable. Bundled auxiliary paths may not cover this directory.
        let bundleHelpers = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers/badapple-fetch")
        if fm.isExecutableFile(atPath: bundleHelpers.path) {
            return bundleHelpers
        }

        if let bundled = Bundle.main.url(forAuxiliaryExecutable: "badapple-fetch"),
           fm.isExecutableFile(atPath: bundled.path) {
            return bundled
        }

        let paths = [
            "/usr/local/bin/badapple-fetch",
            "/opt/homebrew/bin/badapple-fetch",
            "/usr/bin/badapple-fetch",
        ]
        for path in paths where fm.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
        return nil
    }

    func cancelDownload(modelId: String) -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId), let state = self.state[modelId] else {
            return ["error": "unknown or invalid model_id: \(modelId)"]
        }
        if let task = downloadTasks[modelId] {
            task.terminate()
            downloadTasks.removeValue(forKey: modelId)
        }
        if state.status == .downloading || state.status == .queued {
            state.status = .missing
            state.progress = 0.0
            state.error = ""
            state.lastUpdated = Date().timeIntervalSince1970
            saveState()
        }
        return statusDictionary(for: modelId)
    }

    // MARK: - Add / Remove

    /// Add a local path as a model. The path must be under the HF cache root.
    func addModel(modelId: String, repoId: String, localPath: String) -> [String: Any] {
        guard isSafeModelId(modelId), isSafeRepoId(repoId) else {
            return ["error": "invalid model_id or repo_id"]
        }
        guard let root = safeLocalPath(localPath, mustExist: true), root.path.contains(hfCacheRoot.resolvingSymlinksInPath().path) else {
            return ["error": "local path must be under \(hfCacheRoot.path)"]
        }
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: root.path) else {
            return ["error": "cannot enumerate \(root.path)"]
        }
        let hasWeights = enumerator.compactMap({ $0 as? String }).contains {
            $0.hasSuffix(".safetensors") || $0.hasSuffix(".bin") || $0.hasSuffix(".gguf")
        }
        if !hasWeights {
            return ["error": "path does not contain model weight files (.safetensors/.bin/.gguf)"]
        }

        lock.lock(); defer { lock.unlock() }
        let sizeGB = directorySizeGB(root)
        let profile = ModelProfile(
            id: modelId,
            name: modelId,
            repoId: repoId,
            sizeGB: sizeGB,
            kind: "text",
            loadedIn: "mlx_server"
        )
        profiles[modelId] = profile
        let s = state[modelId] ?? ModelState()
        s.localPath = root.path
        s.status = .cached
        s.progress = 1.0
        s.error = ""
        s.lastUpdated = Date().timeIntervalSince1970
        state[modelId] = s
        saveState()
        return statusDictionary(for: modelId)
    }

    /// Remove a custom model from the manager. Does not delete files from HF cache.
    func removeModel(modelId: String) -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId) else {
            return ["error": "invalid model_id: \(modelId)"]
        }
        if Self.knownProfiles.contains(where: { $0.id == modelId }) {
            return ["error": "cannot remove built-in model \(modelId); mark it missing with refresh_cache_status"]
        }
        if state[modelId]?.status == .loaded {
            return ["error": "model \(modelId) is currently loaded; unload it first"]
        }
        profiles.removeValue(forKey: modelId)
        state.removeValue(forKey: modelId)
        saveState()
        return ["status": "removed", "model_id": modelId]
    }

    // MARK: - Load / Unload Tracking

    func markLoaded(modelId: String, localPath: String? = nil) {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId), let profile = profiles[modelId], let state = self.state[modelId] else { return }
        var path = localPath
        if path == nil || path!.isEmpty {
            path = resolveCachePath(repoId: profile.repoId, allowDownload: false)
        }
        if let p = path, let _ = safeLocalPath(p, mustExist: false) {
            state.localPath = p
            state.status = .loaded
            state.progress = 1.0
            state.error = ""
            state.verified = verifyProvenance(modelId: modelId, localPath: p)["status"] as? String == "verified"
        } else {
            state.status = .error
            state.error = "could not resolve local path for loaded model"
        }
        state.lastUpdated = Date().timeIntervalSince1970
        saveState()
    }

    func markUnloaded(modelId: String) {
        lock.lock(); defer { lock.unlock() }
        guard isSafeModelId(modelId), let state = self.state[modelId] else { return }
        state.status = state.localPath.isEmpty ? .missing : .cached
        state.lastUpdated = Date().timeIntervalSince1970
        saveState()
    }

    // MARK: - Recommendations

    /// Recommend the best model that fits in available memory.
    func recommendForMemory() -> [String: Any] {
        let availableGB = availableMemoryGB()
        let pick = recommendModelForMemory(availableGB: availableGB)
        let status = modelStatus(modelId: pick) ?? ["id": pick]
        return [
            "available_gb": round(availableGB * 100) / 100,
            "recommended_id": pick,
            "recommended": status,
            "message": "Bad Apple recommends \(profileName(pick)).",
            "reason": recommendationReason(availableGB: availableGB, pick: pick),
        ]
    }

    /// Recommend a model for a specific query and memory budget.
    func recommendForQuery(_ query: String) -> [String: Any] {
        let availableGB = availableMemoryGB()
        let pick = recommendModelForQuery(query, availableGB: availableGB)
        let status = modelStatus(modelId: pick) ?? ["id": pick]
        let low = query.lowercased()
        let wantsSmall = ["hi", "hello", "time", "weather", "joke", "quick", "short", "simple", "what is", "who is", "how are", "thanks", "ping"].contains { low.contains($0) }
        let wantsBig = ["reason", "deep", "complex", "analyze", "compare", "code review", "architecture", "design", "philosophy", "math proof", "debug"].contains { low.contains($0) }
        let reason: String
        if wantsSmall {
            reason = "This is a short or simple question, so a fast, tiny model is enough."
        } else if wantsBig {
            reason = "This looks like a reasoning or coding question, so a larger model is recommended if it fits. " + recommendationReason(availableGB: availableGB, pick: pick)
        } else {
            reason = recommendationReason(availableGB: availableGB, pick: pick)
        }
        return [
            "available_gb": round(availableGB * 100) / 100,
            "recommended_id": pick,
            "recommended": status,
            "message": "Bad Apple recommends \(profileName(pick)).",
            "reason": reason,
        ]
    }

    private func profileName(_ modelId: String) -> String {
        lock.lock(); defer { lock.unlock() }
        return profiles[modelId]?.name ?? modelId
    }

    /// Estimated memory required to load a model: size times a headroom
    /// factor for KV cache and runtime overhead (lower for giant MoE
    /// models whose cache is small relative to parameter count).
    func memoryRequiredGB(modelId: String) -> Double {
        lock.lock(); defer { lock.unlock() }
        let size = profiles[modelId]?.sizeGB ?? 0.0
        return size * headroomFactor(size)
    }

    // MARK: - Provenance

    /// Build a canonical, deterministic string that uniquely represents a
    /// manifest.  This is the exact message signed by the Secure Enclave when
    /// recording provenance and re-computed when verifying it.
    private func canonicalManifestMessage(repoId: String, localPath: String, recordedAt: TimeInterval, files: [String: ModelFileEntry]) -> String {
        var lines: [String] = []
        lines.append(repoId)
        lines.append(localPath)
        lines.append(String(format: "%.9f", recordedAt))
        for key in files.keys.sorted() {
            let e = files[key]!
            lines.append("\(key):\(e.size):\(String(format: "%.9f", e.mtime)):\(e.sha256)")
        }
        return lines.joined(separator: "\n")
    }

    /// Sign a canonical manifest message using the identity agent if available.
    /// Returns `(signature, publicKey)` base64 strings or `nil` if the agent
    /// is unavailable or refuses to sign.
    private func signManifest(_ manifest: ModelManifest) -> (signature: String, publicKey: String)? {
        guard IdentityAgentClient.shared.isAvailable else { return nil }
        let message = canonicalManifestMessage(
            repoId: manifest.repoId,
            localPath: manifest.localPath,
            recordedAt: manifest.recordedAt,
            files: manifest.files
        )
        guard let publicKey = IdentityAgentClient.shared.publicKey() else { return nil }
        guard let signature = IdentityAgentClient.shared.sign(message: Data(message.utf8)) else { return nil }
        return (signature, publicKey)
    }

    /// Compute and save a SHA-256 manifest for a cached model.
    func recordProvenance(modelId: String, localPath: String) -> [String: Any] {
        guard isSafeModelId(modelId) else {
            return ["status": "error", "error": "invalid model_id: \(modelId)"]
        }
        guard let root = safeLocalPath(localPath, mustExist: true) else {
            return ["status": "invalid_path", "error": "path is outside HF cache or does not exist"]
        }
        guard let profile = lock.withLock({ profiles[modelId] }) else {
            return ["status": "error", "error": "unknown model \(modelId)"]
        }
        let hashCap = hashCapFor(profile)
        let deadline = Date().addingTimeInterval(provenanceTimeoutFor(profile))

        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: root.path) else {
            return ["status": "error", "error": "cannot enumerate \(root.path)"]
        }

        var files: [String: ModelFileEntry] = [:]
        var totalHashed: Int64 = 0

        for case let file as String in enumerator {
            if file.hasPrefix(".") { continue }
            if file == "desktop.ini" || file == "Thumbs.db" { continue }
            let full = root.appendingPathComponent(file)
            guard fm.fileExists(atPath: full.path) else { continue }

            var isDir: ObjCBool = false
            fm.fileExists(atPath: full.path, isDirectory: &isDir)
            if isDir.boolValue { continue }

            if totalHashed > hashCap { break }

            var attributes: [FileAttributeKey: Any]?
            do {
                attributes = try fm.attributesOfItem(atPath: full.path)
            } catch { continue }
            let size = (attributes?[.size] as? Int64) ?? 0
            let mtime = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            if size > hashCap { continue }

            do {
                let sha256 = try sha256File(full, deadline: deadline)
                files[file] = ModelFileEntry(relative: file, size: size, mtime: mtime, sha256: sha256)
                totalHashed += size
            } catch {
                return ["status": "error", "error": "cannot hash \(file): \(error.localizedDescription)"]
            }
        }

        var manifest = ModelManifest(
            repoId: profile.repoId,
            localPath: root.path,
            recordedAt: Date().timeIntervalSince1970,
            files: files,
            signature: nil,
            publicKey: nil
        )

        // Sign the manifest with the Secure Enclave identity agent.
        if let signed = signManifest(manifest) {
            manifest.signature = signed.signature
            manifest.publicKey = signed.publicKey
        }

        saveManifest(modelId: modelId, manifest: manifest)
        var result: [String: Any] = [
            "status": "recorded",
            "files": files.count,
            "total_bytes": totalHashed
        ]
        if manifest.signature != nil {
            result["signed"] = true
        } else {
            result["signed"] = false
            result["note"] = "manifest recorded without Secure Enclave signature"
        }
        return result
    }

    /// Verify a cached model against the stored manifest.
    func verifyProvenance(modelId: String, localPath: String? = nil) -> [String: Any] {
        guard isSafeModelId(modelId) else {
            return ["status": "error", "error": "invalid model_id: \(modelId)"]
        }
        let manifestPath = manifestFilePath(modelId: modelId)
        guard FileManager.default.fileExists(atPath: manifestPath.path) else {
            return ["status": "unknown", "error": "no recorded manifest"]
        }

        let manifest: ModelManifest
        do {
            let data = try Data(contentsOf: manifestPath)
            manifest = try JSONDecoder().decode(ModelManifest.self, from: data)
        } catch {
            return ["status": "corrupt_manifest", "error": error.localizedDescription]
        }

        let root: URL
        if let localPath = localPath {
            guard let r = safeLocalPath(localPath, mustExist: false) else {
                return ["status": "invalid_path", "error": "path is outside HF cache"]
            }
            root = r
        } else {
            root = URL(fileURLWithPath: manifest.localPath)
        }
        guard FileManager.default.fileExists(atPath: root.path) else {
            return ["status": "missing", "error": "local path not found: \(root.path)"]
        }

        var mismatches: [String] = []
        var checked = 0
        let deadline = Date().addingTimeInterval(
            lock.withLock { profiles[modelId] }.map { provenanceTimeoutFor($0) } ?? provenanceTimeout
        )

        for (rel, entry) in manifest.files {
            let file = root.appendingPathComponent(rel)
            guard FileManager.default.fileExists(atPath: file.path) else {
                mismatches.append("missing: \(rel)")
                continue
            }

            var attributes: [FileAttributeKey: Any]?
            do {
                attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            } catch {
                mismatches.append("cannot stat \(rel): \(error.localizedDescription)")
                continue
            }
            let size = (attributes?[.size] as? Int64) ?? 0
            let mtime = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0

            if size != entry.size {
                mismatches.append("size changed: \(rel)")
                continue
            }
            if abs(mtime - entry.mtime) < 1e-9 {
                checked += 1
                continue
            }

            do {
                let digest = try sha256File(file, deadline: deadline)
                if digest != entry.sha256 {
                    mismatches.append("hash mismatch: \(rel)")
                    continue
                }
                checked += 1
            } catch {
                mismatches.append("cannot rehash \(rel): \(error.localizedDescription)")
            }
        }

        if !mismatches.isEmpty {
            return [
                "status": "mismatch",
                "error": mismatches.prefix(10).joined(separator: "; "),
                "mismatches": mismatches,
                "checked": checked,
            ]
        }

        // Verify the Secure Enclave signature if one was recorded.
        if let signatureB64 = manifest.signature, let publicKeyB64 = manifest.publicKey {
            guard let signatureData = Data(base64Encoded: signatureB64),
                  let publicKeyData = Data(base64Encoded: publicKeyB64) else {
                return ["status": "mismatch", "error": "manifest has malformed signature or public key"]
            }
            let message = canonicalManifestMessage(
                repoId: manifest.repoId,
                localPath: manifest.localPath,
                recordedAt: manifest.recordedAt,
                files: manifest.files
            )
            guard IdentityAgentClient.shared.verify(
                message: Data(message.utf8),
                signature: signatureData,
                publicKey: publicKeyData
            ) else {
                return ["status": "mismatch", "error": "Secure Enclave signature verification failed"]
            }
        }

        lock.lock(); defer { lock.unlock() }
        state[modelId]?.verified = true
        saveState()
        return [
            "status": "verified",
            "files": manifest.files.count,
            "checked": checked,
            "recorded_at": manifest.recordedAt,
            "signed": manifest.signature != nil,
        ]
    }

    // MARK: - Persistence

    private func loadState() {
        guard FileManager.default.fileExists(atPath: statusFile.path) else { return }
        do {
            let data = try Data(contentsOf: statusFile)
            let payload = try JSONDecoder().decode([String: [String: ModelState]].self, from: data)
            if let saved = payload["state"] {
                for (mid, s) in saved where state[mid] != nil {
                    // A download that was queued/downloading in a previous process
                    // was interrupted; do not claim it is still active.
                    if s.status == .downloading || s.status == .queued {
                        if !s.localPath.isEmpty && FileManager.default.fileExists(atPath: s.localPath) {
                            s.status = .cached
                            s.progress = 1.0
                            s.error = ""
                        } else {
                            s.status = .missing
                            s.progress = 0.0
                            s.localPath = ""
                            s.error = ""
                        }
                    }
                    state[mid] = s
                }
            }
        } catch {
            NSLog("[BadAppleModelManager] could not load state: %@", error.localizedDescription)
        }
    }

    private func saveState() {
        let payload: [String: [String: ModelState]] = ["state": state]
        do {
            let data = try JSONEncoder().encode(payload)
            let tmp = statusFile.appendingPathExtension("tmp")
            try data.write(to: tmp, options: .atomic)
            try? FileManager.default.removeItem(at: statusFile)
            try FileManager.default.moveItem(at: tmp, to: statusFile)
        } catch {
            NSLog("[BadAppleModelManager] could not save state: %@", error.localizedDescription)
        }
    }

    private func manifestFilePath(modelId: String) -> URL {
        let safe = modelId.replacingOccurrences(of: "/", with: "--")
        return manifestDir.appendingPathComponent("\(safe).json")
    }

    private func saveManifest(modelId: String, manifest: ModelManifest) {
        let path = manifestFilePath(modelId: modelId)
        let tmp = path.appendingPathExtension("tmp")
        do {
            let data = try JSONEncoder().encode(manifest)
            try data.write(to: tmp, options: .atomic)
            try? FileManager.default.removeItem(at: path)
            try FileManager.default.moveItem(at: tmp, to: path)
        } catch {
            NSLog("[BadAppleModelManager] could not save manifest: %@", error.localizedDescription)
        }
    }

    // MARK: - Helpers

    /// Resolve a HF repo ID to a local snapshot path. If `allowDownload` is false,
    /// only existing cache is scanned.
    private func resolveCachePath(repoId: String, allowDownload: Bool) -> String? {
        guard isSafeRepoId(repoId) else { return nil }

        let parts = repoId.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return nil }
        let modelDirName = "models--\(parts[0])--\(parts[1])"
        let modelDir = hfCacheRoot.appendingPathComponent(modelDirName)

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            return nil
        }

        do {
            let snapshotsDir = modelDir.appendingPathComponent("snapshots")
            let refsDir = modelDir.appendingPathComponent("refs")

            var resolvedSnapshot: String? = nil
            if let refs = try? FileManager.default.contentsOfDirectory(atPath: refsDir.path),
               let mainRef = refs.first {
                if let data = try? Data(contentsOf: refsDir.appendingPathComponent(mainRef)),
                   let ref = String(data: data, encoding: .utf8) {
                    resolvedSnapshot = ref.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }

            let snapshots = (try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path)) ?? []
            var candidatePaths: [URL] = []
            for snapshot in snapshots where !snapshot.hasPrefix(".") {
                let snapPath = snapshotsDir.appendingPathComponent(snapshot)
                let fm = FileManager.default
                guard let enumerator = fm.enumerator(atPath: snapPath.path) else { continue }
                let hasWeights = enumerator.compactMap({ $0 as? String }).contains {
                    $0.hasSuffix(".safetensors") || $0.hasSuffix(".bin") || $0.hasSuffix(".gguf")
                }
                if hasWeights {
                    if snapshot == resolvedSnapshot {
                        return snapPath.path
                    }
                    candidatePaths.append(snapPath)
                }
            }
            return candidatePaths.first?.path
        }
    }

    private func safeLocalPath(_ path: String, mustExist: Bool) -> URL? {
        let expanded = path.replacingOccurrences(of: "~", with: NSHomeDirectory())
        let url = URL(fileURLWithPath: expanded).resolvingSymlinksInPath()
        if mustExist && !FileManager.default.fileExists(atPath: url.path) {
            return nil
        }
        if !url.path.hasPrefix(hfCacheRoot.resolvingSymlinksInPath().path) {
            return nil
        }
        return url
    }

    private func isSafeModelId(_ id: String) -> Bool {
        if id.isEmpty || id.count > maxModelIdLength { return false }
        if id.contains("..") || id.contains("/") { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-."))
        return id.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    private func isSafeRepoId(_ repoId: String) -> Bool {
        if repoId.isEmpty || repoId.count > maxRepoIdLength * 2 + 1 { return false }
        if repoId.contains("..") { return false }
        let parts = repoId.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-."))
        return parts.allSatisfy { $0.unicodeScalars.allSatisfy { allowed.contains($0) } }
    }

    private func sha256File(_ url: URL, deadline: Date) throws -> String {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "BadAppleModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "file not found \(url.path)"])
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw NSError(domain: "BadAppleModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "cannot open \(url.path)"])
        }
        defer { try? handle.close() }

        var hasher = SHA256()
        let bufferSize = 1024 * 1024
        while Date() < deadline {
            let chunk = handle.readData(ofLength: bufferSize)
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        if Date() >= deadline {
            throw NSError(domain: "BadAppleModelManager", code: 2, userInfo: [NSLocalizedDescriptionKey: "hashing timed out"])
        }
        return hasher.finalize().compactMap { String(format: "%02x", $0) }.joined()
    }

    private func directorySizeGB(_ url: URL) -> Double {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: url.path) else { return 0.0 }
        var total: Int64 = 0
        for case let file as String in enumerator {
            let full = url.appendingPathComponent(file)
            guard fm.fileExists(atPath: full.path) else { continue }
            var isDir: ObjCBool = false
            fm.fileExists(atPath: full.path, isDirectory: &isDir)
            if isDir.boolValue { continue }
            if let attrs = try? fm.attributesOfItem(atPath: full.path),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }
        return Double(total) / 1_073_741_824.0
    }

    private func availableMemoryGB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        _ = kerr

        var vmStats = vm_statistics64_data_t()
        var statsCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let hostResult = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &statsCount)
            }
        }
        if hostResult == KERN_SUCCESS {
            let pageSize = Double(vm_page_size)
            let freePages = Double(vmStats.free_count) + Double(vmStats.inactive_count)
            return (freePages * pageSize) / 1_073_741_824.0
        }
        return 8.0
    }

    /// Pick the largest text model that fits in the available memory,
    /// walking the whole catalog — from 0.5B up to whatever the hardware
    /// can carry. Self-scaling: new catalog entries join the ladder
    /// automatically instead of needing a new hardcoded rung.
    private func recommendModelForMemory(availableGB: Double) -> String {
        lock.lock(); defer { lock.unlock() }
        let fitting = profiles.values.filter {
            $0.kind == "text" && $0.sizeGB * headroomFactor($0.sizeGB) <= availableGB
        }
        return fitting.max(by: { $0.sizeGB < $1.sizeGB })?.id ?? "fast_0.5b"
    }

    private func recommendModelForQuery(_ query: String, availableGB: Double) -> String {
        let low = query.lowercased()
        if ["hi", "hello", "time", "weather", "joke", "quick", "short", "simple", "what is", "who is", "how are", "thanks", "ping"].contains(where: { low.contains($0) }) {
            return "fast_0.5b"
        }
        return recommendModelForMemory(availableGB: availableGB)
    }

    private func recommendationReason(availableGB: Double, pick: String) -> String {
        let avail = String(format: "%.1f", availableGB)
        if pick == "fast_0.5b" {
            return "Only \(avail) GB of memory is free, so a tiny model is the safest choice."
        }
        return "You have \(avail) GB of free memory, so \(profileName(pick)) is the largest model that fits."
    }
}
