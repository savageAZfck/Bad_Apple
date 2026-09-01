// BadAppleRAG — Swift translation of the Python RAG stack.
//
// Ports three pieces of the Python backend into a single Foundation-only file:
//   1. Semantic cache  — badapple_extras.SemanticCache
//   2. RAG retrieval    — badapple_mlx_rag.build_retrieval_context + MemoryGraph
//   3. KV cache manager — badapple_mlx_rag (kv_cache_paths / save_kv_cache /
//      load_kv_cache / ensure_prompt_cache)
//
// The semantic cache stores query→response pairs keyed by embedding cosine
// similarity so repeated semantically-similar questions return instantly,
// scoped by active persona. The RAG builder assembles retrieved memory facts,
// local documents, and active-workspace context into a single prompt block.
// The KV cache manager persists the metadata that lets the MLX-Swift
// ModelContainer warm-load a primed system-prompt cache across restarts; the
// actual KV tensors are owned by ModelContainer, this class only tracks the
// cache keys and on-disk markers.
//
// Storage layout (mirrors the Python daemon under /var/lib/bad_apple):
//   /var/lib/bad_apple/semantic_cache.json     — semantic cache entries
//   /var/lib/bad_apple/memory_graph/facts.json — long-term memory facts
//   /var/lib/bad_apple/kv_cache/sys_<digest>.json — KV cache metadata markers

import Foundation

// MARK: - Shared helpers

/// Current UTC timestamp as an ISO-8601 string, matching the Python
/// `datetime.datetime.now(datetime.timezone.utc).isoformat()` shape used for
/// cache entries and memory facts.
private func isoTimestamp() -> String {
    ISO8601DateFormatter().string(from: Date())
}

/// Default cosine threshold for the semantic cache, used as the `lookup`
/// default argument. Kept at file scope (rather than a `static let` referenced
/// via `Self`) because default-argument expressions cannot reference the
/// covariant `Self` type. Mirrors `SemanticCache.DEFAULT_THRESHOLD`; the
/// effective cutoff can still be overridden at init time via the
/// `BADAPPLE_CACHE_THRESHOLD` environment variable.
private let badAppleCacheDefaultThreshold: Float = 0.92

/// FNV-1a 64-bit hash rendered as lowercase hex. Used to derive short,
/// filename-safe cache digests from arbitrary keys (the Python code uses
/// `sha256(...).hexdigest()[:16]`; FNV-1a produces the same 16-hex-char width
/// without pulling in CryptoKit, keeping this file Foundation-only).
private func fnv1aHex(_ string: String) -> String {
    var hash: UInt64 = 0xcbf29ce484222325
    for byte in string.utf8 {
        hash ^= UInt64(byte)
        hash &*= 0x100000001b3
    }
    return String(hash, radix: 16, uppercase: false)
}

// MARK: - Embedding provider

/// Abstraction over embedding generation. The Python cache embeds queries with
/// `BAAI/bge-small-en-v1.5`; in Swift this is backed by an MLX encoder at
/// runtime. `embed` is `async` because MLX inference runs off the calling
/// thread, so any method that needs a fresh embedding (`lookup`) is async too.
protocol EmbeddingProvider {
    func embed(_ text: String) async -> [Float]
}

struct BadAppleLexicalEmbeddingProvider: EmbeddingProvider {
    private let dimensions = 512

    func embed(_ text: String) async -> [Float] {
        let words = text.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
        guard !words.isEmpty else { return [] }
        var vector = [Float](repeating: 0, count: dimensions)
        let features = words + zip(words, words.dropFirst()).map { "\($0)_\($1)" }
        for feature in features {
            guard let hash = UInt64(fnv1aHex(feature), radix: 16) else { continue }
            let index = Int(hash % UInt64(dimensions))
            vector[index] += (hash & 1) == 0 ? 1 : -1
        }
        let magnitude = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
        guard magnitude > 0 else { return [] }
        return vector.map { $0 / magnitude }
    }
}

// MARK: - BadAppleSemanticCache

/// Query-to-response cache using embedding cosine similarity.
///
/// Before a generation, the engine checks whether a previous, semantically
/// similar query (scoped by persona) has already been answered. If the cosine
/// similarity between the new prompt embedding and a cached entry is above the
/// threshold, the cached response is returned instantly — a direct port of
/// `badapple_extras.SemanticCache`. Entries are persisted to
/// `/var/lib/bad_apple/semantic_cache.json` and capped at `maxCacheSize` (the
/// most-used entries are retained when the cache overflows, matching Python).
final class BadAppleSemanticCache: @unchecked Sendable {

    /// Default cosine threshold above which a cached response is returned.
    /// Mirrors `SemanticCache.DEFAULT_THRESHOLD`.
    static let defaultThreshold: Float = badAppleCacheDefaultThreshold

    /// Maximum number of entries kept on disk. Mirrors
    /// `SemanticCache.MAX_CACHE_SIZE`.
    static let maxCacheSize = 500

    /// A single cached response. The required fields mirror the Python entry
    /// shape (`query`/`embedding`/`response`/`persona`/`created`); `hits` and
    /// `lastHit` drive the least-used eviction policy. Decoding is tolerant of
    /// missing keys so a partially-written or legacy cache file still loads.
    struct Entry: Codable {
        var prompt: String
        var embedding: [Float]
        var response: String
        var persona: String
        var timestamp: String
        var hits: Int
        var lastHit: String?

        enum CodingKeys: String, CodingKey {
            case prompt, embedding, response, persona, timestamp, hits, lastHit
        }

        init(prompt: String,
             embedding: [Float],
             response: String,
             persona: String,
             timestamp: String,
             hits: Int = 0,
             lastHit: String? = nil) {
            self.prompt = prompt
            self.embedding = embedding
            self.response = response
            self.persona = persona
            self.timestamp = timestamp
            self.hits = hits
            self.lastHit = lastHit
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            prompt = try c.decodeIfPresent(String.self, forKey: .prompt) ?? ""
            embedding = try c.decodeIfPresent([Float].self, forKey: .embedding) ?? []
            response = try c.decodeIfPresent(String.self, forKey: .response) ?? ""
            persona = try c.decodeIfPresent(String.self, forKey: .persona) ?? "default"
            timestamp = try c.decodeIfPresent(String.self, forKey: .timestamp) ?? ""
            hits = try c.decodeIfPresent(Int.self, forKey: .hits) ?? 0
            lastHit = try c.decodeIfPresent(String.self, forKey: .lastHit)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(prompt, forKey: .prompt)
            try c.encode(embedding, forKey: .embedding)
            try c.encode(response, forKey: .response)
            try c.encode(persona, forKey: .persona)
            try c.encode(timestamp, forKey: .timestamp)
            try c.encode(hits, forKey: .hits)
            try c.encodeIfPresent(lastHit, forKey: .lastHit)
        }
    }

    private let fileManager: FileManager
    private let lock = NSLock()
    private let embeddingProvider: EmbeddingProvider?

    /// Instance-configured threshold (may come from `BADAPPLE_CACHE_THRESHOLD`
    /// or the `init` parameter). Used as the effective cutoff when `lookup` is
    /// called with its default argument, mirroring the Python `self.threshold`.
    private let threshold: Float

    /// On-disk path for the cache entries. Defaults to
    /// `/var/lib/bad_apple/semantic_cache.json`.
    let cacheURL: URL

    private var entries: [Entry] = []

    /// Create a semantic cache.
    ///
    /// - Parameters:
    ///   - embeddingProvider: Encoder used to embed prompts for `lookup`. When
    ///     nil, `lookup` cannot compute query embeddings and always returns
    ///     nil (matching the Python zero-vector fallback that never matches).
    ///   - threshold: Cosine threshold. Falls back to the
    ///     `BADAPPLE_CACHE_THRESHOLD` environment variable, then
    ///     `defaultThreshold` (0.92).
    ///   - cachePath: Override for the on-disk cache file.
    init(embeddingProvider: EmbeddingProvider? = BadAppleLexicalEmbeddingProvider(),
         threshold: Float? = nil,
         cachePath: String? = nil,
         fileManager: FileManager = .default) {
        self.embeddingProvider = embeddingProvider
        self.fileManager = fileManager
        if let threshold {
            self.threshold = threshold
        } else if let raw = ProcessInfo.processInfo.environment["BADAPPLE_CACHE_THRESHOLD"],
                  let parsed = Float(raw) {
            self.threshold = parsed
        } else {
            self.threshold = Self.defaultThreshold
        }
        if let cachePath {
            self.cacheURL = URL(fileURLWithPath: cachePath)
        } else {
            self.cacheURL = URL(fileURLWithPath: "/var/lib/bad_apple/semantic_cache.json")
        }
        load()
    }

    // MARK: - Lookup / store

    /// Return a cached response for `prompt` if a same-persona entry has cosine
    /// similarity >= `threshold`.
    ///
    /// Async because the prompt must be embedded via the (async)
    /// `EmbeddingProvider` before similarity can be computed — a direct port of
    /// the Python `SemanticCache.lookup`, which embeds the query then scans
    /// entries. When no provider is configured, nil is returned (the Python
    /// fallback uses zero vectors which never reach the threshold).
    ///
    /// The `threshold` parameter defaults to 0.92; when the caller leaves it at
    /// the default the instance-configured threshold (honouring
    /// `BADAPPLE_CACHE_THRESHOLD`) is used instead, so an explicit override
    /// always wins. The embedding is awaited *before* the lock is taken, and
    /// the locked scan runs in a synchronous helper so `NSLock` is never held
    /// across an await or acquired from an asynchronous context.
    func lookup(prompt: String,
                persona: String,
                threshold: Float = badAppleCacheDefaultThreshold) async -> String? {
        guard let provider = embeddingProvider else { return nil }
        let queryEmbedding = await provider.embed(prompt)
        guard !queryEmbedding.isEmpty else { return nil }

        let cutoff: Float = (threshold == badAppleCacheDefaultThreshold)
            ? self.threshold
            : threshold

        return lookupSync(queryEmbedding: queryEmbedding, persona: persona, cutoff: cutoff)
    }

    /// Synchronous, lock-protected scan used by `lookup` after the prompt
    /// embedding has been produced. Kept separate so the `NSLock` critical
    /// section never spans an `await` and is not acquired from an async context
    /// (which Swift 6 concurrency checking flags).
    private func lookupSync(queryEmbedding: [Float],
                            persona: String,
                            cutoff: Float) -> String? {
        lock.lock()
        defer { lock.unlock() }

        guard !entries.isEmpty else { return nil }
        var bestScore: Float = -1.0
        var bestIndex: Int = -1
        for (index, entry) in entries.enumerated() where entry.persona == persona {
            let score = cosineSimilarity(queryEmbedding, entry.embedding)
            if score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        guard bestIndex >= 0, bestScore >= cutoff else { return nil }

        // Bump the hit counter and record the last-hit timestamp so the
        // eviction policy keeps the most-useful entries, exactly as Python does.
        entries[bestIndex].hits += 1
        entries[bestIndex].lastHit = isoTimestamp()
        saveLocked()
        return entries[bestIndex].response
    }

    /// Add a cached response. The embedding is supplied by the caller (it is
    /// produced asynchronously by the `EmbeddingProvider` at generation time),
    /// mirroring the Python `SemanticCache.store` which embeds then appends.
    /// When the in-memory list exceeds twice the max size, the least-used
    /// entries are dropped so only the `maxCacheSize` most-hit remain.
    func store(prompt: String,
               response: String,
               persona: String,
               embedding: [Float]) {
        let entry = Entry(prompt: prompt,
                          embedding: embedding,
                          response: response,
                          persona: persona,
                          timestamp: isoTimestamp())
        lock.lock()
        defer { lock.unlock() }
        entries.append(entry)
        if entries.count > Self.maxCacheSize * 2 {
            // Keep the most-used half: sort ascending by hits, take the suffix.
            entries.sort { $0.hits < $1.hits }
            entries = Array(entries.suffix(Self.maxCacheSize))
        }
        saveLocked()
    }

    func store(prompt: String, response: String, persona: String) async {
        guard let embeddingProvider else { return }
        let embedding = await embeddingProvider.embed(prompt)
        guard !embedding.isEmpty else { return }
        store(prompt: prompt, response: response, persona: persona, embedding: embedding)
    }

    /// Cosine similarity between two vectors: dot(a,b) / (|a| * |b|). Returns 0
    /// for mismatched lengths or zero vectors. The Python encoder pre-normalises
    /// embeddings so a bare dot product suffices there; this implementation
    /// normalises explicitly so it works with un-normalised embeddings too.
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard !a.isEmpty, a.count == b.count else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let magnitudes = normA.squareRoot() * normB.squareRoot()
        guard magnitudes > 0 else { return 0 }
        return dot / magnitudes
    }

    /// Remove all in-memory and on-disk cache entries. Mirrors the Python
    /// `SemanticCache.clear`.
    func clear() -> String {
        lock.lock()
        defer { lock.unlock() }
        entries = []
        try? fileManager.removeItem(at: cacheURL)
        return "Semantic cache cleared."
    }

    // MARK: - Persistence

    /// Number of cached entries currently in memory.
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return entries.count
    }

    private func load() {
        guard let data = try? Data(contentsOf: cacheURL) else { return }
        if let parsed = try? JSONDecoder().decode([Entry].self, from: data) {
            entries = parsed
        }
    }

    /// Persist the entries. Must be called with `lock` held.
    private func saveLocked() {
        try? fileManager.createDirectory(
            at: cacheURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(entries) else { return }
        try? data.write(to: cacheURL, options: .atomic)
    }
}

// MARK: - BadAppleRAG

/// Builds retrieved context (memory facts, local documents, active workspace)
/// to inject into the model prompt — a Foundation-only port of
/// `badapple_mlx_rag.build_retrieval_context` and the `MemoryGraph` fact store.
///
/// Memory facts are stored as subject/predicate/object triples (with a
/// timestamp and source) under `/var/lib/bad_apple/memory_graph/facts.json`.
/// Retrieval uses keyword overlap — the same fallback the Python `MemoryGraph`
/// uses when no encoder is available — so `buildRetrievalContext` stays
/// synchronous and free of model dependencies.
final class BadAppleRAG: @unchecked Sendable {

    /// A long-term memory fact stored as a knowledge triple. Mirrors the
    /// structured relation records kept by the Python `MemoryGraph` (which
    /// stores free-text facts plus subject/relation/target relations); here the
    /// triple is first-class so retrieval and persistence are typed.
    struct MemoryFact: Codable {
        var subject: String
        var predicate: String
        var object: String
        var timestamp: String
        var source: String

        enum CodingKeys: String, CodingKey {
            case subject, predicate, object, timestamp, source
        }

        init(subject: String,
             predicate: String,
             object: String,
             timestamp: String,
             source: String) {
            self.subject = subject
            self.predicate = predicate
            self.object = object
            self.timestamp = timestamp
            self.source = source
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            subject = try c.decodeIfPresent(String.self, forKey: .subject) ?? ""
            predicate = try c.decodeIfPresent(String.self, forKey: .predicate) ?? ""
            object = try c.decodeIfPresent(String.self, forKey: .object) ?? ""
            timestamp = try c.decodeIfPresent(String.self, forKey: .timestamp) ?? ""
            source = try c.decodeIfPresent(String.self, forKey: .source) ?? "user"
        }
    }

    /// Root directory for the memory graph. Defaults to
    /// `/var/lib/bad_apple/memory_graph/`.
    static let defaultMemoryDirectory = "/var/lib/bad_apple/memory_graph"

    private let fileManager: FileManager
    private let lock = NSLock()

    /// Directory holding the memory graph files.
    let memoryDirectory: URL

    init(memoryDirectory: URL? = nil,
         fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.memoryDirectory = memoryDirectory
            ?? URL(fileURLWithPath: Self.defaultMemoryDirectory)
    }

    /// File storing the memory facts JSON array.
    private var factsURL: URL {
        memoryDirectory.appendingPathComponent("facts.json")
    }

    // MARK: - Memory graph persistence

    /// Load all memory facts from disk. Returns an empty array if the file is
    /// missing, unreadable, or malformed — mirroring the Python
    /// `MemoryGraph._load` best-effort behaviour.
    func loadMemoryGraph() -> [MemoryFact] {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: factsURL) else { return [] }
        return (try? JSONDecoder().decode([MemoryFact].self, from: data)) ?? []
    }

    /// Persist the full set of memory facts, overwriting any existing file.
    /// The parent directory is created on demand and the write is atomic,
    /// matching the Python `MemoryGraph._save` (which writes a temp file then
    /// `os.replace`s it).
    func saveMemoryGraph(_ facts: [MemoryFact]) {
        lock.lock()
        defer { lock.unlock() }
        try? fileManager.createDirectory(
            at: memoryDirectory,
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(facts) else { return }
        try? data.write(to: factsURL, options: .atomic)
    }

    // MARK: - Retrieval

    /// Build a single retrieved-context block for `prompt`, combining relevant
    /// memory facts, local documents from the active workspace, and a workspace
    /// summary. Returns nil when nothing relevant was found so the caller can
    /// skip injecting context entirely — the Swift analogue of
    /// `build_retrieval_context` returning the (possibly unmodified) message
    /// list.
    ///
    /// `workspace` is an optional absolute or tilde-expanded directory path;
    /// when present and valid its recent files, build system, git state, and
    /// README are summarised (a port of `Workspace.summary`), and its text
    /// files are scanned for keyword overlap with the prompt.
    func buildRetrievalContext(prompt: String, workspace: String?) -> String? {
        var blocks: [String] = []

        // 1. Long-term memory facts.
        let facts = loadMemoryGraph()
        let relevant = relevantFacts(prompt: prompt, facts: facts, k: 5)
        if !relevant.isEmpty {
            let lines = relevant.map { fact in
                "- \(fact.subject) \(fact.predicate) \(fact.object)"
            }
            blocks.append(
                "Things you remember about the user:\n" + lines.joined(separator: "\n")
            )
        }

        // 2. Active workspace context + local documents.
        if let workspace, !workspace.isEmpty {
            let expanded = (workspace as NSString).expandingTildeInPath
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
                if let summary = workspaceSummary(path: expanded) {
                    blocks.append("Active workspace:\n\(summary)")
                }
                if let docs = localDocumentContext(prompt: prompt, workspace: expanded) {
                    blocks.append("Relevant local documents:\n\(docs)")
                }
            }
        }

        guard !blocks.isEmpty else { return nil }
        return "Use this context if relevant:\n\n" + blocks.joined(separator: "\n\n")
    }

    // MARK: - Retrieval helpers

    /// Tokenise `text` into lowercased alphanumeric words longer than two
    /// characters — the same filter the Python `MemoryGraph.search` keyword
    /// fallback applies (`re.findall(r"\b\w+\b", ...) if len(w) > 2`).
    private func keywords(in text: String) -> Set<String> {
        var words = Set<String>()
        for raw in text.lowercased().split(whereSeparator: { ch in
            !ch.isLetter && !ch.isNumber
        }) {
            let word = String(raw)
            if word.count > 2 {
                words.insert(word)
            }
        }
        return words
    }

    /// Return the most relevant memory facts for `prompt` by keyword overlap
    /// — the same fallback the Python `MemoryGraph.search` uses when no encoder
    /// is available (`re.findall(r"\b\w+\b", ...) if len(w) > 2`). Semantic
    /// retrieval would require awaiting the `EmbeddingProvider` and is left to
    /// a higher-level async caller; `buildRetrievalContext` stays synchronous.
    private func relevantFacts(prompt: String,
                               facts: [MemoryFact],
                               k: Int) -> [MemoryFact] {
        guard !facts.isEmpty else { return [] }
        let queryWords = keywords(in: prompt)
        guard !queryWords.isEmpty else { return [] }

        var scored: [(Float, MemoryFact)] = []
        for fact in facts {
            let text = "\(fact.subject) \(fact.predicate) \(fact.object)"
            let overlap = Float(queryWords.intersection(keywords(in: text)).count)
            if overlap > 0 {
                scored.append((overlap, fact))
            }
        }
        scored.sort { $0.0 > $1.0 }
        return Array(scored.prefix(k).map { $0.1 })
    }

    /// Summarise a workspace directory: build-system detection, recent files,
    /// git state, and a README excerpt. A Foundation-only port of
    /// `badapple_extras.Workspace.summary`.
    private func workspaceSummary(path: String) -> String? {
        let url = URL(fileURLWithPath: path)
        var parts: [String] = ["Workspace: \(path)."]

        // Build-system detection.
        let buildMap: [(String, String)] = [
            ("Cargo.toml", "Rust/Cargo"),
            ("package.json", "Node/npm"),
            ("pyproject.toml", "Python"),
            ("setup.py", "Python setuptools"),
            ("Package.swift", "Swift Package"),
            ("Makefile", "Make"),
            ("CMakeLists.txt", "CMake"),
            ("build.gradle", "Gradle"),
            ("pom.xml", "Maven"),
        ]
        var buildSystems: [String] = []
        for (filename, label) in buildMap {
            if fileManager.fileExists(atPath: url.appendingPathComponent(filename).path) {
                buildSystems.append(label)
            }
        }
        // Glob-style Xcode detection.
        if let entries = try? fileManager.contentsOfDirectory(atPath: path) {
            if entries.contains(where: { $0.hasSuffix(".xcodeproj") }) {
                buildSystems.append("Xcode project")
            }
            if entries.contains(where: { $0.hasSuffix(".xcworkspace") }) {
                buildSystems.append("Xcode workspace")
            }
        }
        if !buildSystems.isEmpty {
            parts.append("Build system: \(buildSystems.joined(separator: ", ")).")
        }

        // Git state (best-effort, like the Python subprocess calls).
        if let branch = runGit(["branch", "--show-current"], in: path), !branch.isEmpty {
            let dirty = runGit(["status", "--porcelain"], in: path)?
                .split(separator: "\n")
                .count ?? 0
            let lastCommit = runGit(["log", "-1", "--oneline"], in: path) ?? "no commits"
            parts.append("git branch: \(branch), dirty: \(dirty), last: \(lastCommit).")
        }

        // README excerpt.
        for readmeName in ["README.md", "readme.md", "README.rst"] {
            let readmeURL = url.appendingPathComponent(readmeName)
            if let text = try? String(contentsOf: readmeURL, encoding: .utf8) {
                let excerpt = String(text.prefix(160))
                    .replacingOccurrences(of: "\n", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !excerpt.isEmpty {
                    parts.append("README: \(excerpt)...")
                }
                break
            }
        }

        // Recent files (top 10 by modification time, skipping dotfiles).
        let recent = recentFiles(in: path, limit: 10)
        if recent.isEmpty {
            parts.append("No files found.")
        } else {
            parts.append("Recent files: \(recent.joined(separator: ", ")).")
        }

        return parts.joined(separator: " ")
    }

    /// Return the `limit` most-recently-modified non-hidden files under `path`,
    /// expressed relative to `path` (matching the Python `rglob` + mtime sort).
    private func recentFiles(in path: String, limit: Int) -> [String] {
        let url = URL(fileURLWithPath: path)
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.contentModificationDateKey, .isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        var found: [(URL, Date)] = []
        for case let fileURL as URL in enumerator {
            let name = fileURL.lastPathComponent
            if name.hasPrefix(".") { continue }
            let values = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .contentModificationDateKey]
            )
            guard values?.isRegularFile == true,
                  let date = values?.contentModificationDate else {
                continue
            }
            found.append((fileURL, date))
        }
        found.sort { $0.1 > $1.1 }
        return found.prefix(limit).map { item in
            // Relative path when possible, otherwise the bare filename.
            if let relative = item.0.path.relativePath(from: url) {
                return relative
            }
            return item.0.lastPathComponent
        }
    }

    /// Scan text files under `workspace` for keyword overlap with `prompt` and
    /// return up to three short excerpts (capped at 160 characters, like the
    /// Python `rel_know` snippet `c[:160]`). Returns nil when nothing matches.
    private func localDocumentContext(prompt: String, workspace: String) -> String? {
        let queryWords = keywords(in: prompt)
        guard !queryWords.isEmpty else { return nil }
        let url = URL(fileURLWithPath: workspace)
        guard let enumerator = fileManager.enumerator(
            at: url,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        var scored: [(Int, String)] = []
        var scanned = 0
        let maxScan = 200
        for case let fileURL as URL in enumerator {
            if scanned >= maxScan { break }
            scanned += 1
            let name = fileURL.lastPathComponent
            if name.hasPrefix(".") { continue }
            guard isTextFile(name) else { continue }
            // Skip files larger than 256 KB to keep retrieval latency low.
            guard let attrs = try? fileManager.attributesOfItem(atPath: fileURL.path),
                  let size = attrs[.size] as? NSNumber,
                  size.intValue < 256_000 else {
                continue
            }
            guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
                continue
            }
            let overlap = queryWords.intersection(keywords(in: content)).count
            if overlap > 0 {
                let snippet = String(content.prefix(160))
                    .replacingOccurrences(of: "\n", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if !snippet.isEmpty {
                    scored.append((overlap, snippet))
                }
            }
        }
        guard !scored.isEmpty else { return nil }
        scored.sort { $0.0 > $1.0 }
        return scored.prefix(3).map { "- \($0.1)" }.joined(separator: "\n")
    }

    /// Best-effort git invocation. Returns trimmed stdout on success, nil on
    /// any failure — mirrors the Python `subprocess.run(..., check=False)`
    /// calls that simply produce empty strings when git is absent.
    private func runGit(_ arguments: [String], in directory: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = arguments
        process.currentDirectoryURL = URL(fileURLWithPath: directory)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return nil
        }
        guard process.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Heuristic: treat recognised source/text extensions as indexable.
    private func isTextFile(_ name: String) -> Bool {
        let extensions: Set<String> = [
            "txt", "md", "markdown", "rst",
            "swift", "py", "rs", "go", "ts", "tsx", "js", "jsx", "m", "mm",
            "c", "cc", "cpp", "h", "hpp", "java", "kt", "rb", "php",
            "json", "yaml", "yml", "toml", "ini", "cfg", "conf",
            "sh", "bash", "zsh", "fish", "ps1",
            "html", "htm", "css", "scss", "xml", "csv", "log",
        ]
        guard let dot = name.lastIndex(of: ".") else { return false }
        let ext = String(name[name.index(after: dot)...]).lowercased()
        return extensions.contains(ext)
    }
}

// MARK: - BadAppleKVCache

/// On-disk manager for system-prompt KV cache markers.
///
/// The actual KV tensors in MLX-Swift are owned by the model's `ModelContainer`
/// (the equivalent of the Python `make_prompt_cache` / `mx.save_safetensors`
/// round-trip). This class manages the cache *keys* and metadata on disk so the
/// engine can decide whether a primed system-prompt cache already exists
/// (`ensurePromptCache`) and whether a specific cache can be warm-loaded
/// (`loadKVCache`). It is a Foundation-only port of the metadata half of
/// `badapple_mlx_rag` (kv_cache_paths / save_kv_cache / load_kv_cache /
/// ensure_prompt_cache).
final class BadAppleKVCache: @unchecked Sendable {

    /// Directory holding the KV cache files. Defaults to
    /// `/var/lib/bad_apple/kv_cache/`.
    static let defaultCacheDirectory = "/var/lib/bad_apple/kv_cache"

    private let fileManager: FileManager
    private let lock = NSLock()

    /// Directory where KV cache metadata markers live.
    let cacheDirectory: URL

    init(cacheDirectory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.cacheDirectory = cacheDirectory
            ?? URL(fileURLWithPath: Self.defaultCacheDirectory)
    }

    /// List every saved KV cache file in the cache directory. Each primed
    /// system-prompt cache is represented by a `sys_<digest>.json` metadata
    /// marker (and, when MLX-Swift persists tensors, a matching
    /// `sys_<digest>.safetensors` file); this returns all regular files so the
    /// caller can enumerate or prune them. The directory is created on demand,
    /// matching the Python `server._kv_cache_dir.mkdir(parents=True, exist_ok=True)`.
    func kvCachePaths() -> [URL] {
        try? fileManager.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        guard let urls = try? fileManager.contentsOfDirectory(
            at: cacheDirectory,
            includingPropertiesForKeys: [.isRegularFileKey]
        ) else {
            return []
        }
        return urls
            .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Mark a cache as saved by writing its metadata marker to disk. The actual
    /// KV tensors are persisted by the MLX-Swift `ModelContainer`; this only
    /// records that a cache for `key` exists, so subsequent `loadKVCache(key:)`
    /// calls succeed. `key` is typically `<model>:<maxKVSize>:<systemHash>`,
    /// mirroring the Python `kv_cache_paths` composite key.
    func saveKVCache(key: String) {
        lock.lock()
        defer { lock.unlock() }
        try? fileManager.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        let metadata: [String: Any] = [
            "key": key,
            "saved_at": isoTimestamp(),
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: metadata,
            options: [.prettyPrinted, .sortedKeys]
        ) else {
            return
        }
        try? data.write(to: metadataURL(forKey: key), options: .atomic)
    }

    /// Check whether a cache for `key` has been saved. Returns true when the
    /// metadata marker exists on disk. The caller (the engine) is responsible
    /// for warm-loading the matching tensors from the `ModelContainer` — this
    /// only reports presence, mirroring the Python `load_kv_cache` early-exit
    /// when the weights/metadata files are missing.
    func loadKVCache(key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return fileManager.fileExists(atPath: metadataURL(forKey: key).path)
    }

    /// Check whether a primed cache for `systemPrompt` already exists. Returns
    /// true when a metadata marker keyed by the system prompt is present, so the
    /// engine can skip the expensive system-prefill (a port of the Python
    /// `ensure_prompt_cache` presence check; the re-priming itself is handled
    /// by the engine when this returns false).
    func ensurePromptCache(systemPrompt: String) -> Bool {
        loadKVCache(key: systemPrompt)
    }

    /// Remove a saved cache marker. Useful when the system prompt or model
    /// changes and the old cache is no longer valid.
    func removeKVCache(key: String) {
        lock.lock()
        defer { lock.unlock() }
        try? fileManager.removeItem(at: metadataURL(forKey: key))
    }

    /// Path of the metadata marker for `key`.
    private func metadataURL(forKey key: String) -> URL {
        cacheDirectory.appendingPathComponent("sys_\(digest(forKey: key)).json")
    }

    /// Short, filename-safe digest for a cache key (FNV-1a hex), matching the
    /// width of the Python `sha256(...).hexdigest()[:16]` digest.
    private func digest(forKey key: String) -> String {
        fnv1aHex(key)
    }
}

// MARK: - URL relative-path helper

private extension String {
    /// Return this path expressed relative to `base` when it is a descendant,
    /// otherwise nil. Used to render workspace file paths relative to the
    /// workspace root, like the Python `f.relative_to(p)`.
    func relativePath(from base: URL) -> String? {
        let baseURL = base.standardizedFileURL
        let targetURL = URL(fileURLWithPath: self).standardizedFileURL
        let basePath = baseURL.path
        let targetPath = targetURL.path
        guard targetPath.hasPrefix(basePath) else { return nil }
        let prefix = basePath.hasSuffix("/") ? basePath : basePath + "/"
        guard targetPath.hasPrefix(prefix) else { return nil }
        let relative = String(targetPath.dropFirst(prefix.count))
        return relative.isEmpty ? nil : relative
    }
}
