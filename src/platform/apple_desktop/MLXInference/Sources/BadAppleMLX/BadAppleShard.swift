import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import Network

// MARK: - Shardable model protocol
//
// The model files in MLXLLM carry `badapple-mesh-brain` extensions (see
// apply_mesh_brain_patch.sh) that expose embed/layers/norm/head slices of the
// forward pass. This protocol is the uniform handle the shard runtime drives;
// conformances are declared here while the witnesses live in the vendored
// model files (same-file extensions, so fileprivate members stay reachable).

public protocol BadAppleShardable {
    func badappleShardEmbed(_ ids: MLXArray) -> MLXArray
    func badappleShardLayers(_ h: MLXArray, cache: [KVCache]?) -> MLXArray
    func badappleShardNorm(_ h: MLXArray) -> MLXArray
    func badappleShardHead(_ h: MLXArray) -> MLXArray
}

extension Qwen2Model: BadAppleShardable {}
extension Qwen3Model: BadAppleShardable {}
extension LlamaModel: BadAppleShardable {}
extension Qwen3MoEModel: BadAppleShardable {}
extension GLM4MoEModel: BadAppleShardable {}
extension GLM4MoELiteModel: BadAppleShardable {}
extension DeepseekV3Model: BadAppleShardable {}
extension GPTOSSModel: BadAppleShardable {}

// MARK: - Rank spec (written by `badapple mesh-brain shard`)

public struct MeshBrainRankSpec: Codable, Sendable {
    public var rank: Int
    public var world: Int
    public var host: String        // this rank's listen address "host:port"
    public var nextHost: String?   // next rank's address, nil on the last rank
    public var hasEmbed: Bool
    public var hasHead: Bool
    public var layerStart: Int
    public var layerEnd: Int
}

// MARK: - Wire framing
//
// Frame = [u32 big-endian header length][header JSON][raw payload bytes].
// The payload is a contiguous activation tensor; the header carries
// shape/dtype so the receiver can rebuild an MLXArray without a copy.

struct MeshBrainFrame: Codable {
    var op: String                       // auth | auth-ok | step | reset | ping | generate | result | error
    var shape: [Int]? = nil
    var dtype: String? = nil
    var token: Int? = nil
    var done: Bool? = nil
    var maxTokens: Int? = nil
    var prompt: String? = nil
    var text: String? = nil
    var error: String? = nil
    var layerStart: Int? = nil
    var layerEnd: Int? = nil
}

// MARK: - Mutual auth
//
// Per-connection HMAC-SHA256 nonce challenge over the shared SLICKS secret.
// The server challenges; both sides prove. Resolved from
// BADAPPLE_MESH_KEY > BADAPPLE_P2P_SECRET > BADAPPLE_SLICKS_KEY_PATH >
// /var/lib/bad_apple/slicks.key. BADAPPLE_MESH_AUTH=0 disables (debug only).

enum MeshBrainAuth {
    static var enabled: Bool {
        ProcessInfo.processInfo.environment["BADAPPLE_MESH_AUTH"] != "0"
    }

    /// AES-256-GCM frames ride on top of the handshake — same SHA-256
    /// derivation as the P2P engram crypto. BADAPPLE_MESH_ENC=0 disables.
    static var encryptionEnabled: Bool {
        enabled
            && ProcessInfo.processInfo.environment["BADAPPLE_MESH_ENC"] != "0"
    }

    static func sessionKey() -> SymmetricKey? {
        guard encryptionEnabled, let s = secret else { return nil }
        return SymmetricKey(data: SHA256.hash(data: s))
    }

    static var secret: Data? {
        let env = ProcessInfo.processInfo.environment
        if let s = env["BADAPPLE_MESH_KEY"], !s.isEmpty { return Data(s.utf8) }
        if let s = env["BADAPPLE_P2P_SECRET"], !s.isEmpty { return Data(s.utf8) }
        let path = env["BADAPPLE_SLICKS_KEY_PATH"] ?? "/var/lib/bad_apple/slicks.key"
        guard let raw = try? String(contentsOfFile: path, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty else { return nil }
        return Data(raw.utf8)
    }

    static func nonce() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, 32, &bytes)
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func hmac(_ msg: String, key: Data) -> String {
        let mac = HMAC<SHA256>.authenticationCode(
            for: Data(msg.utf8), using: SymmetricKey(data: key))
        return mac.map { String(format: "%02x", $0) }.joined()
    }

    /// Server side: challenge the peer, verify proof, return counter-proof.
    static func serverHandshake(_ conn: NWConnection) async throws {
        guard enabled else { return }
        guard let key = secret else {
            throw MeshBrainError.transport(
                "mesh auth on but no secret — set BADAPPLE_MESH_KEY or slicks.key")
        }
        let n = nonce()
        var challenge = MeshBrainFrame(op: "auth")
        challenge.text = n
        try await MeshBrainWire.writeFrame(to: conn, header: challenge, payload: nil)
        let (resp, _) = try await MeshBrainWire.readFrame(from: conn)
        guard resp.op == "auth",
              resp.text == hmac("mb-c:" + n, key: key)
        else { throw MeshBrainError.transport("peer auth failed") }
        var ok = MeshBrainFrame(op: "auth-ok")
        ok.text = hmac("mb-s:" + n, key: key)
        try await MeshBrainWire.writeFrame(to: conn, header: ok, payload: nil)
    }

    /// Client side: answer the challenge, verify the counter-proof.
    static func clientHandshake(_ conn: NWConnection) async throws {
        guard enabled else { return }
        guard let key = secret else {
            throw MeshBrainError.transport(
                "mesh auth on but no secret — set BADAPPLE_MESH_KEY or slicks.key")
        }
        let (challenge, _) = try await MeshBrainWire.readFrame(from: conn)
        guard challenge.op == "auth", let n = challenge.text else {
            throw MeshBrainError.badFrame("expected auth challenge")
        }
        var resp = MeshBrainFrame(op: "auth")
        resp.text = hmac("mb-c:" + n, key: key)
        try await MeshBrainWire.writeFrame(to: conn, header: resp, payload: nil)
        let (ok, _) = try await MeshBrainWire.readFrame(from: conn)
        guard ok.op == "auth-ok",
              ok.text == hmac("mb-s:" + n, key: key)
        else { throw MeshBrainError.transport("server counter-proof failed") }
    }
}

enum MeshBrainWire {
    static func dtypeName(_ d: DType) -> String {
        switch d {
        case .bfloat16: return "bfloat16"
        case .float16: return "float16"
        case .float32: return "float32"
        case .float64: return "float64"
        case .int32: return "int32"
        case .int64: return "int64"
        case .uint32: return "uint32"
        default: return "\(d)"
        }
    }

    static func dtype(named name: String) -> DType? {
        switch name {
        case "bfloat16": return .bfloat16
        case "float16": return .float16
        case "float32": return .float32
        case "float64": return .float64
        case "int32": return .int32
        case "int64": return .int64
        case "uint32": return .uint32
        default: return nil
        }
    }

    static func packArray(_ array: MLXArray) throws -> (MeshBrainFrame, Data) {
        let d = array.contiguous().asData()
        var frame = MeshBrainFrame(op: "")
        frame.shape = d.shape
        frame.dtype = dtypeName(d.dType)
        return (frame, d.data)
    }

    static func unpackArray(_ frame: MeshBrainFrame, payload: Data) throws -> MLXArray {
        guard let shape = frame.shape, let name = frame.dtype,
              let dtype = dtype(named: name)
        else { throw MeshBrainError.badFrame("missing shape/dtype") }
        return MLXArray(payload, shape, dtype: dtype)
    }

    static func writeFrame(
        to conn: NWConnection, header: MeshBrainFrame, payload: Data?
    ) async throws {
        var headerData = try JSONEncoder().encode(header)
        var len = UInt32(headerData.count).bigEndian
        var out = Data(bytes: &len, count: 4)
        out.append(headerData)
        if let payload { out.append(payload) }
        headerData.removeAll()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: out, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    static func readExact(_ conn: NWConnection, count: Int) async throws -> Data {
        var buf = Data()
        while buf.count < count {
            let chunk: Data = try await withCheckedThrowingContinuation { cont in
                conn.receive(minimumIncompleteLength: 1,
                             maximumLength: count - buf.count) { data, _, _, error in
                    if let error { cont.resume(throwing: error); return }
                    cont.resume(returning: data ?? Data())
                }
            }
            if chunk.isEmpty { throw MeshBrainError.peerClosed }
            buf.append(chunk)
        }
        return buf
    }

    static func readFrame(
        from conn: NWConnection
    ) async throws -> (MeshBrainFrame, Data) {
        let lenData = try await readExact(conn, count: 4)
        let len = lenData.withUnsafeBytes { $0.load(as: UInt32.self) }.bigEndian
        guard len < 64 * 1024 else { throw MeshBrainError.badFrame("header too large") }
        let headerData = try await readExact(conn, count: Int(len))
        let header = try JSONDecoder().decode(MeshBrainFrame.self, from: headerData)
        var payload = Data()
        if let shape = header.shape {
            let elems = shape.reduce(1, *)
            let size = elems * (dtype(named: header.dtype ?? "")?.size ?? 2)
            if size > 0 { payload = try await readExact(conn, count: size) }
        }
        return (header, payload)
    }

    // MARK: encrypted packets
    //
    // Post-handshake the whole frame — header and tensor payload — travels
    // inside one AES-256-GCM envelope: [u32 ctLen][nonce||ct||tag].
    // `combined` layout matches Rust's P2PCipher.encrypt byte-for-byte.

    static func writePacket(
        to conn: NWConnection, header: MeshBrainFrame, payload: Data?,
        key: SymmetricKey?
    ) async throws {
        guard let key else {
            return try await writeFrame(to: conn, header: header, payload: payload)
        }
        let json = try JSONEncoder().encode(header)
        var inner = Data()
        var l = UInt32(json.count).bigEndian
        inner.append(Data(bytes: &l, count: 4))
        inner.append(json)
        if let payload { inner.append(payload) }
        let box = try AES.GCM.seal(inner, using: key)
        guard let combined = box.combined else {
            throw MeshBrainError.transport("gcm seal produced no combined box")
        }
        var out = Data()
        var cl = UInt32(combined.count).bigEndian
        out.append(Data(bytes: &cl, count: 4))
        out.append(combined)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: out, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) } else { cont.resume() }
            })
        }
    }

    static func readPacket(
        from conn: NWConnection, key: SymmetricKey?
    ) async throws -> (MeshBrainFrame, Data) {
        guard let key else { return try await readFrame(from: conn) }
        let lenData = try await readExact(conn, count: 4)
        let len = lenData.withUnsafeBytes { $0.load(as: UInt32.self) }.bigEndian
        guard len < 512 * 1024 * 1024 else {
            throw MeshBrainError.badFrame("encrypted frame too large")
        }
        let combined = try await readExact(conn, count: Int(len))
        let sealed = try AES.GCM.SealedBox(combined: combined)
        let inner = try AES.GCM.open(sealed, using: key)
        guard inner.count >= 4 else { throw MeshBrainError.badFrame("frame too short") }
        let jsonLen = Int(inner.withUnsafeBytes { $0.load(as: UInt32.self) }.bigEndian)
        guard inner.count >= 4 + jsonLen else {
            throw MeshBrainError.badFrame("frame truncated")
        }
        let header = try JSONDecoder().decode(
            MeshBrainFrame.self, from: inner[4 ..< 4 + jsonLen])
        return (header, inner[(4 + jsonLen)...])
    }
}

public enum MeshBrainError: Error, CustomStringConvertible {
    case badFrame(String)
    case peerClosed
    case unsupportedModel(String)
    case transport(String)
    public var description: String {
        switch self {
        case .badFrame(let s): return "bad frame: \(s)"
        case .peerClosed: return "peer closed connection"
        case .unsupportedModel(let s): return "unsupported shard model: \(s)"
        case .transport(let s): return "transport: \(s)"
        }
    }
}

/// Minimal async semaphore — serializes pipeline steps since KV caches are
/// stateful and must not interleave between requests.
actor AsyncSemaphore {
    private var permits: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []
    init(_ n: Int) { permits = n }
    func wait() async {
        if permits > 0 { permits -= 1; return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func signal() {
        if !waiters.isEmpty { waiters.removeFirst().resume() } else { permits += 1 }
    }
}

// MARK: - Downstream link (rank i -> rank i+1)

private var meshTimeout: TimeInterval {
    TimeInterval(
        ProcessInfo.processInfo.environment["BADAPPLE_MESH_TIMEOUT"] ?? "120")
        ?? 120
}

/// Timeout flag — checked by a watchdog task that cancels the connection to
/// wake pending sends/receives. NWConnection ops aren't Task-cancellable, so
/// the only way to interrupt a dead peer's read is killing the socket.
final class TimeoutFlag: @unchecked Sendable { var done = false }

final class MeshBrainLink: @unchecked Sendable {
    private let conn: NWConnection
    private var sessionKey: SymmetricKey?
    private let queue = DispatchQueue(label: "badapple.meshbrain.link")

    init(host: String) async throws {
        let parts = host.split(separator: ":")
        guard let h = parts.first, let p = parts.last, let port = NWEndpoint.Port(String(p))
        else { throw MeshBrainError.transport("bad host \(host)") }
        conn = NWConnection(
            host: NWEndpoint.Host(String(h)), port: port, using: .tcp)
        final class Once: @unchecked Sendable { var resumed = false }
        let once = Once()
        // Watchdog: unreachable hosts stall .waiting for 75s+ — cancel at 15s.
        let watchdog = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 15_000_000_000)
            guard !Task.isCancelled else { return }
            self?.conn.cancel()
        }
        defer { watchdog.cancel() }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if !once.resumed { once.resumed = true; cont.resume() }
                case .failed(let e), .waiting(let e):
                    if !once.resumed { once.resumed = true; cont.resume(throwing: e) }
                case .cancelled:
                    if !once.resumed {
                        once.resumed = true
                        cont.resume(throwing: MeshBrainError.transport("connect timeout"))
                    }
                default: break
                }
            }
            conn.start(queue: queue)
        }
        conn.stateUpdateHandler = nil
        try await MeshBrainAuth.clientHandshake(conn)
        sessionKey = MeshBrainAuth.sessionKey()
    }

    func request(_ header: MeshBrainFrame, payload: Data?) async throws
        -> (MeshBrainFrame, Data)
    {
        let flag = TimeoutFlag()
        let watchdog = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(meshTimeout * 1_000_000_000))
            guard !Task.isCancelled, !flag.done else { return }
            self?.conn.cancel()  // wakes pending send/receive with an error
        }
        defer { flag.done = true; watchdog.cancel() }
        do {
            try await MeshBrainWire.writePacket(
                to: conn, header: header, payload: payload, key: sessionKey)
            return try await MeshBrainWire.readPacket(from: conn, key: sessionKey)
        } catch {
            throw MeshBrainError.transport("\(error)")
        }
    }
}

// MARK: - Shard runtime

/// One pipeline rank: loads a shard directory produced by
/// `badapple mesh-brain shard` and serves hidden-state forwards over TCP.
/// Rank 0 additionally owns the embed and the tokenizer; the last rank owns
/// the final norm + lm_head and returns the next token.
public final class BadAppleShardRuntime: @unchecked Sendable {

    // Boxes keep non-Sendable MLX handles across @Sendable perform closures.
    final class Boxes: @unchecked Sendable {
        var shard: (any BadAppleShardable)?
        var model: (any LanguageModel)?
        var caches: [KVCache]?
        var eosTokenId: Int?
        // MLXArray isn't Sendable — tensors never leave perform() closures;
        // they travel between ops through these slots instead.
        var input: MLXArray?
        var output: MLXArray?
        var logits: MLXArray?
        var tokenIds: [Int] = []
        var sampledToken: Int = -1
        var outputText: String = ""
    }

    public let spec: MeshBrainRankSpec
    let container: ModelContainer
    let box = Boxes()
    private var link: MeshBrainLink?
    private var listener: NWListener?
    private let stepGate = AsyncSemaphore(1)
    public private(set) var isReady = false

    public init(shardDir: URL) async throws {
        let specURL = shardDir.appendingPathComponent("mesh_brain_rank.json")
        spec = try JSONDecoder().decode(
            MeshBrainRankSpec.self, from: Data(contentsOf: specURL))
        container = try await LLMModelFactory.shared.loadContainer(
            from: shardDir, using: TokenizersLoader())
        let b = box
        try await container.perform { ctx in
            guard let shard = ctx.model as? any BadAppleShardable else {
                throw MeshBrainError.unsupportedModel(
                    String(describing: type(of: ctx.model)))
            }
            b.shard = shard
            b.model = ctx.model
            b.caches = ctx.model.newCache(parameters: nil)
            b.eosTokenId = ctx.tokenizer.eosTokenId
        }
        isReady = true
    }

    // MARK: local forward ops

    /// ids -> embeddings, result in box.output (rank 0 only).
    private func embed(_ ids: [Int]) async throws {
        let b = box
        b.tokenIds = ids
        try await container.perform { _ in
            let arr = MLXArray(b.tokenIds.map { Int32($0) }, [1, b.tokenIds.count])
            let out = b.shard!.badappleShardEmbed(arr)
            out.eval()
            b.output = out
        }
    }

    /// box.input -> hidden states after local layer slice, into box.output.
    private func runLayers() async throws {
        let b = box
        try await container.perform { _ in
            let out = b.shard!.badappleShardLayers(b.input!, cache: b.caches)
            out.eval()
            b.output = out
        }
    }

    /// box.input -> norm + lm_head logits, into box.logits (last rank only).
    private func logits() async throws {
        let b = box
        try await container.perform { _ in
            let n = b.shard!.badappleShardNorm(b.input!)
            let out = b.shard!.badappleShardHead(n)
            out.eval()
            b.logits = out
        }
    }

    private func sampleNextToken(temperature: Float) async throws -> Int {
        let b = box
        await container.perform { _ in
            let lg = b.logits!
            let last = lg[0, lg.dim(1) - 1]
            let token: MLXArray
            if temperature <= 0 {
                token = last.argMax()
            } else {
                token = MLXRandom.categorical(last / temperature)
            }
            token.eval()
            b.sampledToken = token.item(Int.self)
        }
        return box.sampledToken
    }

    // MARK: pipeline

    /// Handle one `step` frame: run local layers, then either hand the hidden
    /// state downstream or produce the next token locally on the last rank.
    func handleStep(hidden: MLXArray) async throws -> (token: Int, done: Bool) {
        await stepGate.wait()
        defer { Task { await stepGate.signal() } }
        box.input = hidden
        try await runLayers()
        if let next = spec.nextHost {
            if link == nil { link = try await MeshBrainLink(host: next) }
            var (frame, payload) = try MeshBrainWire.packArray(box.output!)
            frame.op = "step"
            let resp: MeshBrainFrame
            do {
                (resp, _) = try await link!.request(frame, payload: payload)
            } catch {
                link = nil  // dead peer — next caller reconnects
                throw MeshBrainError.transport("downstream \(next): \(error)")
            }
            if let err = resp.error { throw MeshBrainError.transport(err) }
            guard let tok = resp.token else {
                throw MeshBrainError.badFrame("step response missing token")
            }
            return (tok, resp.done ?? false)
        }
        box.input = box.output
        try await logits()
        let tok = try await sampleNextToken(temperature: 0)
        return (tok, tok == box.eosTokenId)
    }

    func handleReset() async throws {
        await stepGate.wait()
        defer { Task { await stepGate.signal() } }
        try await resetPipeline()
    }

    /// Fresh KV caches locally and downstream. Caller must hold stepLock.
    private func resetPipeline() async throws {
        if let next = spec.nextHost {
            if link == nil { link = try await MeshBrainLink(host: next) }
            let f = MeshBrainFrame(op: "reset")
            _ = try await link!.request(f, payload: nil)
        }
        let b = box
        await container.perform { _ in
            b.caches = b.model!.newCache(parameters: nil)
        }
    }

    // MARK: server

    /// Listen on spec.host and serve step/reset/ping frames forever.
    public func serve() async throws {
        let parts = spec.host.split(separator: ":")
        guard let p = parts.last, let port = NWEndpoint.Port(String(p))
        else { throw MeshBrainError.transport("bad listen addr \(spec.host)") }
        let listener = try NWListener(using: .tcp, on: port)
        self.listener = listener
        let q = DispatchQueue(label: "badapple.meshbrain.serve")
        listener.newConnectionHandler = { [weak self] conn in
            Task { try? await self?.serveConnection(conn, queue: q) }
        }
        listener.start(queue: q)
        FileHandle.standardError.write(
            "mesh-brain rank \(spec.rank)/\(spec.world) serving \(spec.host)\n"
                .data(using: .utf8)!)
        try await Task.sleep(nanoseconds: .max)  // serve forever
    }

    private func serveConnection(_ conn: NWConnection, queue: DispatchQueue) async throws {
        final class Once: @unchecked Sendable { var resumed = false }
        let once = Once()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    if !once.resumed { once.resumed = true; cont.resume() }
                case .failed(let e), .waiting(let e):
                    if !once.resumed { once.resumed = true; cont.resume(throwing: e) }
                case .cancelled:
                    if !once.resumed {
                        once.resumed = true
                        cont.resume(throwing: MeshBrainError.transport("conn cancelled"))
                    }
                default: break
                }
            }
            conn.start(queue: queue)
        }
        conn.stateUpdateHandler = nil
        // Auth watchdog — a peer that connects and never proves itself gets
        // 15s, then the socket drops out from under the pending read.
        let authDone = TimeoutFlag()
        let authWatchdog = Task {
            try? await Task.sleep(nanoseconds: 15_000_000_000)
            guard !Task.isCancelled, !authDone.done else { return }
            conn.cancel()
        }
        do {
            try await MeshBrainAuth.serverHandshake(conn)
        } catch {
            var r = MeshBrainFrame(op: "error")
            r.error = "authentication failed"
            try? await MeshBrainWire.writeFrame(to: conn, header: r, payload: nil)
            conn.cancel()
            return
        }
        authDone.done = true
        authWatchdog.cancel()
        let connKey = MeshBrainAuth.sessionKey()
        while true {
            let header: MeshBrainFrame
            let payload: Data
            do {
                (header, payload) = try await MeshBrainWire.readPacket(
                    from: conn, key: connKey)
            } catch {
                return  // transport broke — nothing left to say
            }
            do {
                switch header.op {
                case "ping":
                    var r = MeshBrainFrame(op: "result")
                    r.token = spec.rank
                    r.layerStart = spec.layerStart
                    r.layerEnd = spec.layerEnd
                    try await MeshBrainWire.writePacket(
                        to: conn, header: r, payload: nil, key: connKey)
                case "reset":
                    try await handleReset()
                    try await MeshBrainWire.writePacket(
                        to: conn, header: MeshBrainFrame(op: "result"),
                        payload: nil, key: connKey)
                case "step":
                    let h = try MeshBrainWire.unpackArray(header, payload: payload)
                    let (tok, done) = try await handleStep(hidden: h)
                    var r = MeshBrainFrame(op: "result")
                    r.token = tok; r.done = done
                    try await MeshBrainWire.writePacket(
                        to: conn, header: r, payload: nil, key: connKey)
                case "generate":
                    let text = try await generate(
                        prompt: header.prompt ?? "",
                        maxTokens: header.maxTokens ?? 64)
                    var r = MeshBrainFrame(op: "result")
                    r.text = text
                    try await MeshBrainWire.writePacket(
                        to: conn, header: r, payload: nil, key: connKey)
                default:
                    var r = MeshBrainFrame(op: "error")
                    r.error = "unknown op \(header.op)"
                    try await MeshBrainWire.writePacket(
                        to: conn, header: r, payload: nil, key: connKey)
                }
            } catch {
                // Handler failure — tell the caller, keep the conn alive.
                var r = MeshBrainFrame(op: "error")
                r.error = "\(error)"
                try? await MeshBrainWire.writePacket(
                    to: conn, header: r, payload: nil, key: connKey)
            }
        }
    }

    // MARK: orchestration (rank 0)

    /// Full generate for rank 0: tokenize -> embed -> per-token pipeline.
    /// Only valid when spec.hasEmbed is true. A transport failure mid-token
    /// desyncs every downstream KV cache, so the whole pipeline resets and
    /// retries once before surfacing the error.
    public func generate(prompt: String, maxTokens: Int) async throws -> String {
        do {
            return try await generateOnce(prompt: prompt, maxTokens: maxTokens)
        } catch let e as MeshBrainError {
            if case .transport = e {} else if case .peerClosed = e {} else { throw e }
            link = nil  // dead — force reconnect on retry
            try? await resetPipeline()
            return try await generateOnce(prompt: prompt, maxTokens: maxTokens)
        }
    }

    private func generateOnce(prompt: String, maxTokens: Int) async throws -> String {
        guard spec.hasEmbed else {
            throw MeshBrainError.badFrame("generate must start on the embed rank")
        }
        await stepGate.wait()
        defer { Task { await stepGate.signal() } }
        try await resetPipeline()
        let b = box
        b.tokenIds = []
        try await container.perform { ctx in
            b.tokenIds = ctx.tokenizer.encode(text: prompt)
        }
        // Prefill: embed the whole prompt, run local layers over it.
        try await embed(b.tokenIds)
        box.input = box.output
        try await runLayers()
        var outTokens: [Int] = []
        var steps = 0
        while steps < maxTokens {
            let tok: Int
            if let next = spec.nextHost {
                if link == nil { link = try await MeshBrainLink(host: next) }
                var (frame, payload) = try MeshBrainWire.packArray(box.output!)
                frame.op = "step"
                let resp: MeshBrainFrame
                do {
                    (resp, _) = try await link!.request(frame, payload: payload)
                } catch {
                    link = nil
                    throw MeshBrainError.transport("downstream \(next): \(error)")
                }
                if let err = resp.error { throw MeshBrainError.transport(err) }
                guard let t = resp.token else {
                    throw MeshBrainError.badFrame("missing token")
                }
                tok = t
            } else {
                box.input = box.output
                try await logits()
                tok = try await sampleNextToken(temperature: 0)
            }
            if tok == box.eosTokenId { break }
            outTokens.append(tok)
            steps += 1
            // feed the sampled token back through embed -> local layers
            try await embed([tok])
            box.input = box.output
            try await runLayers()
        }
        b.tokenIds = outTokens
        await container.perform { ctx in
            b.outputText = ctx.tokenizer.decode(tokenIds: b.tokenIds)
        }
        return b.outputText
    }
}
