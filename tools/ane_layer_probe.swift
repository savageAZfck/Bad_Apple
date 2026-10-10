// ane_layer_probe.swift — single-layer MLModel probe matching e5_conform.
//   swift ane_layer_probe.swift <shard_dir> <token>
// Runs ONE layer on the MLModel path with pos-0 inputs and prints hidden.

import CoreML
import Foundation

let args = CommandLine.arguments
let dir = URL(fileURLWithPath: args[1])
let token = Int(args[2])!
let nLayers = args.count > 3 ? Int(args[3])! : 1
let hidden = 2560
let seqLen = 2048
let ropeHalf = 64

let manifestData = try Data(contentsOf: dir.appendingPathComponent("conversion_manifest.json"))
let root = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
let embeddingPath = dir.appendingPathComponent(
    ((root["shared"] as! [String: Any])["embedding"] as! [String: Any])["path"] as! String)
let layerPaths = ((root["shards"] as! [[String: Any]])
    .compactMap { $0["compiled_path"] as? String })
    .sorted().prefix(nLayers)
    .map { dir.appendingPathComponent($0) }

let config = MLModelConfiguration()
config.computeUnits = .all
let embedding = try Data(contentsOf: embeddingPath, options: .mappedIfSafe)
let layers = try layerPaths.map { try MLModel(contentsOf: $0, configuration: config) }
let states = layers.map { $0.makeState() }
for (i, l) in layers.enumerated() {
    for name in l.modelDescription.stateDescriptionsByName.keys {
        _ = states[i].withMultiArray(for: name) { arr in memset(arr.dataPointer, 0, arr.count * 2) }
    }
}

let x = try MLMultiArray(shape: [1, hidden, 1, 1] as [NSNumber], dataType: .float16)
_ = embedding.withUnsafeBytes { src in
    memcpy(x.dataPointer, src.baseAddress!.advanced(by: token * hidden * 2), hidden * 2)
}
let attnMask = try MLMultiArray(shape: [1, 1, 8, seqLen] as [NSNumber], dataType: .float16)
let writeMask = try MLMultiArray(shape: [1, 1, seqLen, 1] as [NSNumber], dataType: .float16)
let ropeCos = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
let ropeSin = try MLMultiArray(shape: [1, ropeHalf] as [NSNumber], dataType: .float16)
for i in 0..<attnMask.count { attnMask[i] = NSNumber(value: Float(-10000)) }
for r in 0..<8 { attnMask[r * seqLen] = 0 }
for i in 0..<writeMask.count { writeMask[i] = 0 }
writeMask[0] = 1
for i in 0..<ropeHalf { ropeCos[i] = 1; ropeSin[i] = 0 }

var out = x
for (i, l) in layers.enumerated() {
    let p = try l.prediction(from: MLDictionaryFeatureProvider(dictionary: [
        "x": MLFeatureValue(multiArray: out),
        "rope_cos": MLFeatureValue(multiArray: ropeCos),
        "rope_sin": MLFeatureValue(multiArray: ropeSin),
        "attn_mask": MLFeatureValue(multiArray: attnMask),
        "kv_write_mask": MLFeatureValue(multiArray: writeMask),
    ]), using: states[i])
    let name = p.featureNames.first { p.featureValue(for: $0)?.multiArrayValue != nil }!
    out = p.featureValue(for: name)!.multiArrayValue!
}
let ptr = out.dataPointer.assumingMemoryBound(to: UInt16.self)
var line = "hidden[0..16]:"
for i in 0..<16 { line += String(format: " %.6f", Float(Float16(bitPattern: ptr[i]))) }
print(line)
var nz = 0
for i in 0..<out.count { if ptr[i] != 0 { nz += 1 } }
print("nonzero: \(nz)/\(out.count)")
if ProcessInfo.processInfo.environment["HEADS"] != nil {
    let headSpecs = ((root["shared"] as! [String: Any])["lm_head_shards"] as! [[String: Any]])
        .map { (start: ($0["vocab_start"] as! NSNumber).intValue,
                path: dir.appendingPathComponent($0["compiled_path"] as! String)) }
        .sorted { $0.start < $1.start }
    var best = -1
    var bestVal = -Float.infinity
    var firstLogits = ""
    for (hi, spec) in headSpecs.enumerated() {
        let h = try MLModel(contentsOf: spec.path, configuration: config)
        let hp = try h.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "x": MLFeatureValue(multiArray: out)]))
        let hname = hp.featureNames.first { hp.featureValue(for: $0)?.multiArrayValue != nil }!
        let larr = hp.featureValue(for: hname)!.multiArrayValue!
        let lp = larr.dataPointer.assumingMemoryBound(to: UInt16.self)
        // ANE head outputs are channel-strided — the vocab dim's element
        // stride comes from MLMultiArray.strides (can be 32 on ANE).
        var stride = 1
        let dims = larr.shape.map { $0.intValue }
        let strides = larr.strides.map { $0.intValue }
        for d in 0..<dims.count where dims[d] == larr.count && d < strides.count {
            stride = strides[d]
        }
        if hi == 0 {
            for i in 0..<8 { firstLogits += String(format: " %.5f", Float(Float16(bitPattern: lp[i * stride]))) }
        }
        for i in 0..<larr.count {
            let v = Float(Float16(bitPattern: lp[i * stride]))
            if v > bestVal { bestVal = v; best = spec.start + i }
        }
    }
    print("argmax: \(best) (\(String(format: "%.5f", bestVal)))")
    print("logits[0..8]:\(firstLogits)")
}
_ = states[0].withMultiArray(for: "k_cache_0") { arr in
    let kp = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
    var s = "k_cache_0 head:"
    for i in 0..<8 { s += String(format: " %.5f", Float(Float16(bitPattern: kp[i]))) }
    var knz = 0
    for i in 0..<arr.count { if kp[i] != 0 { knz += 1 } }
    print("\(s)  nonzero \(knz)/\(arr.count)")
}
