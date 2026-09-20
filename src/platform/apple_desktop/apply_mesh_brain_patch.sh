#!/bin/bash
# apply_mesh_brain_patch.sh — append mesh-brain shard accessors to the
# pinned mlx-swift-lm checkout.
#
# Every supported model keeps its layer stack fileprivate/internal, which
# makes pipeline-parallel forward impossible from outside the module.
# `fileprivate` is file-scoped, so an extension appended to the same file
# sees everything — this patch is purely additive and re-applies cleanly
# after `swift package` re-resolves the checkout.
#
# Each extension conforms the outer model class to BadAppleShardable
# (declared in BadAppleShard.swift) so the shard runtime can drive a
# uniform embed → layers → norm → head path per rank.
#
# Covers: Qwen2, Qwen3, Llama, Qwen3MoE, GLM4MOE, GLM4MOELite,
# DeepseekV3, GPTOSS. (Qwen3Next skipped — its gated RMSNorm needs the
# head-side gate, not shardable with a plain hidden-state handoff.)

set -euo pipefail

CHECKOUT_ROOT="${1:-$(dirname "$0")/MLXInference/.build/checkouts/mlx-swift-lm/Libraries/MLXLLM/Models}"
MARKER="badapple-mesh-brain"

# append_ext <file> <OuterClass> <head-expr> [layers-body-file]
append_ext() {
    local file="$1" outer="$2" head="$3"
    local path="${CHECKOUT_ROOT}/${file}"
    if [[ ! -f "$path" ]]; then
        echo "  skip (missing): $file"
        return 0
    fi
    if grep -q "$MARKER" "$path"; then
        echo "  already patched: $file"
        return 0
    fi
    chmod u+w "$path" 2>/dev/null || true
    cat >> "$path" <<EOF

// MARK: - ${MARKER} (appended by apply_mesh_brain_patch.sh — do not edit)

public extension ${outer} {
    func badappleShardEmbed(_ ids: MLXArray) -> MLXArray { model.embedTokens(ids) }
    func badappleShardNorm(_ h: MLXArray) -> MLXArray { model.norm(h) }
    func badappleShardHead(_ h: MLXArray) -> MLXArray { ${head} }
    func badappleShardLayers(_ h: MLXArray, cache: [KVCache]?) -> MLXArray {
        let mask = createAttentionMask(h: h, cache: cache?.first)
        var x = h
        for (i, layer) in model.layers.enumerated() {
            x = layer(x, mask: mask, cache: cache?[i])
        }
        return x
    }
}
EOF
    echo "  patched: $file"
}

TIED_HEAD='if let lmHead { return lmHead(h) }; return model.embedTokens.asLinear(h)'
PLAIN_HEAD='return lmHead(h)'

append_ext "Qwen2.swift"       "Qwen2Model"       "$TIED_HEAD"
append_ext "Qwen3.swift"       "Qwen3Model"       "$TIED_HEAD"
append_ext "Llama.swift"       "LlamaModel"       "$TIED_HEAD"
append_ext "Qwen3MoE.swift"    "Qwen3MoEModel"    "$TIED_HEAD"
append_ext "GLM4MOE.swift"     "GLM4MoEModel"     "$TIED_HEAD"
append_ext "GLM4MOELite.swift" "GLM4MoELiteModel" "$PLAIN_HEAD"
append_ext "DeepseekV3.swift"  "DeepseekV3Model"  "$PLAIN_HEAD"

# GPTOSS alternates full/sliding attention — its mask logic is
# per-layer-type, so it gets its own layers body.
GPTOSS_FILE="${CHECKOUT_ROOT}/GPTOSS.swift"
if [[ -f "$GPTOSS_FILE" ]] && ! grep -q "$MARKER" "$GPTOSS_FILE"; then
    chmod u+w "$GPTOSS_FILE" 2>/dev/null || true
    cat >> "$GPTOSS_FILE" <<'EOF'

// MARK: - badapple-mesh-brain (appended by apply_mesh_brain_patch.sh — do not edit)

public extension GPTOSSModel {
    func badappleShardEmbed(_ ids: MLXArray) -> MLXArray { model.embedTokens(ids) }
    func badappleShardNorm(_ h: MLXArray) -> MLXArray { model.norm(h) }
    func badappleShardHead(_ h: MLXArray) -> MLXArray { lmHead(h) }
    func badappleShardLayers(_ h: MLXArray, cache: [KVCache]?) -> MLXArray {
        var x = h
        let caches: [KVCache?] = cache?.map { $0 } ?? [KVCache?](repeating: nil, count: model.layers.count)
        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?
        for (i, layer) in model.layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if model.layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(
                        n: seqLen, cache: caches[model.fullAttentionIndex], windowSize: nil)
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen, cache: caches[model.slidingAttentionIndex], windowSize: model.windowSize)
                }
                maskMode = slidingMask!
            }
            x = layer(x, mask: maskMode, cache: caches[i])
        }
        return x
    }
}
EOF
    echo "  patched: GPTOSS.swift"
elif [[ -f "$GPTOSS_FILE" ]]; then
    echo "  already patched: GPTOSS.swift"
fi

echo "mesh-brain patch applied under ${CHECKOUT_ROOT}"
