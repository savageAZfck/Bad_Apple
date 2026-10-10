// ane_conform.swift — CoreML/MLModel baseline decode of an ANE shard set.
// Mirrors BadAppleBridge's decode semantics exactly (embedding row copy,
// additive causal mask, one-hot kv_write_mask, per-position rope, per-head
// argmax with vocab_start offsets) so its token sequence is the reference
// `milc ane decode` must reproduce.
//
//   swift ane_conform.swift <shard_dir> <t1,t2,...> [max_new]
//
// Prints the generated token ids, one per line region: "tokens: a,b,c"

import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count >= 3 else {
    FileHandle.standardError.write("usage: ane_conform.swift <dir> <t1,t2> [max_new]\n".data(using: .utf8)!)
    exit(2)
}
let dir = URL(fileURLWithPath: args[1])
let prompt = args[2].split(separator: ",").compactMap { Int($0) }
let maxNew = args.count > 3 ? Int(args[3]) ?? 8 : 8

// ---- manifest ----
let manifestData = try Data(contentsOf: dir.appendingPathComponent("conversion_manifest.json"))
let root = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
let model = root["model"] as! [String: Any]
let hidden = (model["hidden_size"] as! NSNumber).intValue
let vocab = (model["vocab_size"] as! NSNumber).intValue
let seqLen = (model["seq_len"] as! NSNumber).intValue
let ropeHalf = (model["rope_dim"] as! NSNumber).intValue / 2
let ropeBase = (model["rope_freq_base"] as! NSNumber).doubleValue
let embeddingPath = dir.appendingPathComponent(
    ((root["shared"] as! [String: Any])["embedding"] as! [String: Any])["path"] as! String)
let headSpecs = ((root["shared"] as! [String: Any])["lm_head_shards"] as! [[String: Any]])
    .map { (start: ($0["vocab_start"] as! NSNumber).intValue,
            path: dir.appendingPathComponent($0["compiled_path"] as! String)) }
    .sorted { $0.start < $1.start }
let layerPaths = (root["shards"] as! [[String: Any]])
    .compactMap { $0["compiled_path"] as? String }
    .sorted()
    .map { dir.appendingPathComponent($0) }

// ---- load ----
let config = MLModelConfiguration()
config.computeUnits = .all
let embedding = try Data(contentsOf: embeddingPath, options: .mappedIfSafe)
let layers = try layerPaths.map { try MLModel(contentsOf: $0, configuration: config) }
let heads = try headSpecs.map { try MLModel(contentsOf: $0.path, configuration: config) }
let states = layers.map { $0.makeState() }
// zero states — MLState buffers are uninitialized
for (i, l) in layers.enumerated() {
    for name in l.modelDescription.stateDescriptionsByName.keys {
        states[i].withMultiArray(for: name) { arr in
            memset(arr.dataPointer, 0, arr.count * 2)
        }
    }
}

// ---- decode buffers ----
let attnMask = try MLMultiArray(shape: [1, 1, 8, seqLen] as [NSNumber], dataType: .float16)
let writeMask = try MLMultiArray(shape: [1, 1, seqLen, 1] as [NSNumber], dataType: .float16)
let ropeCos = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
let ropeSin = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
let invFreq = (0..<ropeHalf).map { 1.0 / pow(ropeBase, Double($0) / Double(ropeHalf)) }
var position = 0

func resetMasks() {
    for i in 0..<attnMask.count { attnMask[i] = NSNumber(value: Float(-10000)) }
    for i in 0..<writeMask.count { writeMask[i] = NSNumber(value: Float(0)) }
}
func updateInputs() {
    for i in 0..<ropeHalf {
        let angle = Double(position) * invFreq[i]
        ropeCos[i] = NSNumber(value: cos(angle))
        ropeSin[i] = NSNumber(value: sin(angle))
    }
    if position > 0 { writeMask[position - 1] = 0 }
    let rows = attnMask.count / seqLen
    for r in 0..<rows { attnMask[r * seqLen + position] = 0 }
    writeMask[position] = 1
}
func embed(_ token: Int) throws -> MLMultiArray {
    let x = try MLMultiArray(shape: [1, hidden, 1, 1] as [NSNumber], dataType: .float16)
    let off = token * hidden * 2
    embedding.withUnsafeBytes { src in
        memcpy(x.dataPointer, src.baseAddress!.advanced(by: off), hidden * 2)
    }
    return x
}
func logits(_ x: MLMultiArray) -> [(start: Int, arr: MLMultiArray)] {
    var out: [(Int, MLMultiArray)] = []
    for (i, h) in heads.enumerated() {
        let p = try! h.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x)]))
        let name = p.featureNames.first { p.featureValue(for: $0)?.multiArrayValue != nil }!
        out.append((headSpecs[i].start, p.featureValue(for: name)!.multiArrayValue!))
    }
    return out
}
func push(_ token: Int, project: Bool) throws -> Int? {
    var x = try embed(token)
    updateInputs()
    for (i, l) in layers.enumerated() {
        let p = try l.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: x),
            "rope_cos": MLFeatureValue(multiArray: ropeCos),
            "rope_sin": MLFeatureValue(multiArray: ropeSin),
            "attn_mask": MLFeatureValue(multiArray: attnMask),
            "kv_write_mask": MLFeatureValue(multiArray: writeMask),
        ]), using: states[i])
        let name = p.featureNames.first { p.featureValue(for: $0)?.multiArrayValue != nil }!
        x = p.featureValue(for: name)!.multiArrayValue!
    }
    position += 1
    guard project else { return nil }
    var best = -1
    var bestVal = -Float.infinity
    for (start, arr) in logits(x) {
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        // ANE head outputs are channel-strided — use the vocab dim's
        // element stride (32 on ANE), not a flat index.
        var stride = 1
        let dims = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        for d in 0..<dims.count where dims[d] == arr.count && d < strides.count {
            stride = strides[d]
        }
        for i in 0..<arr.count {
            let v = Float16(bitPattern: ptr[i * stride])
            if Float(v) > bestVal { bestVal = Float(v); best = start + i }
        }
    }
    return best >= 0 ? best : nil
}

// ---- run ----
resetMasks()
for t in prompt.dropLast() { _ = try push(t, project: false) }
var next = try push(prompt.last!, project: true)
var generated: [Int] = []
for _ in 0..<maxNew {
    guard let t = next else { break }
    generated.append(t)
    next = try push(t, project: true)
}
print("tokens: " + generated.map(String.init).joined(separator: ","))
