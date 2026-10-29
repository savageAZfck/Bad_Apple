cap fs
import "sys.cy"
import "strlib.cy"
import "mil.cy"

// emit_layer_shard — the stateful transformer-layer MIL shard, authored in
// Coyote. Weights come through the st_* externs (safetensors hex); the spec
// lowers through mil_spec on the Rust side. Debug knobs are cfg flags, not
// env vars.

host det fn cnum(c: Json, k: str, d: i64) -> i64 {
    var m = obj_of(c)
    if !has(m, k) { return d; }
    match m[k] {
        Json::Num(v) => { return i64(v); }
        _ => { return d; }
    }
}

host det fn cf64(c: Json, k: str, d: f64) -> f64 {
    var m = obj_of(c)
    if !has(m, k) { return d; }
    match m[k] {
        Json::Num(v) => { return v; }
        _ => { return d; }
    }
}

host det fn cbool(c: Json, k: str) -> bool {
    var m = obj_of(c)
    if !has(m, k) { return false; }
    match m[k] {
        Json::Bool(b) => { return b; }
        Json::Num(v) => { return v != 0.0; }
        _ => { return false; }
    }
}

host det fn cstr(c: Json, k: str) -> str {
    var m = obj_of(c)
    if !has(m, k) { return ""; }
    match m[k] {
        Json::Str(v) => { return v; }
        _ => { return ""; }
    }
}

host det fn spec_err(msg: str) -> Json {
    var m: {str: Json} = {}
    m["error"] = Json::Str(msg)
    return Json::Obj(m);
}

host det fn is_arch(c: Json, arch: str) -> bool {
    return cstr(c, "arch") == arch;
}

host det fn is_sliding(c: Json, i: i64) -> bool {
    if cbool(c, "gemma_no_sliding") { return false; }
    var pat = cnum(c, "sliding_pattern", 0)
    return cnum(c, "sliding_window", 0) > 0 && pat > 0 && ((i + 1) % pat) != 0;
}

host det fn rope_rows(c: Json) -> i64 {
    if cbool(c, "gemma_no_local_rope") { return 1; }
    if cf64(c, "rope_local_theta", 0.0) != cf64(c, "rope_theta", 0.0) {
        return 2;
    }
    return 1;
}

host det fn rope_row(c: Json, i: i64) -> i64 {
    if rope_rows(c) == 2 && is_sliding(c, i) { return 1; }
    return 0;
}

// norm weight: fp16 bytes with the Gemma (1+w) fold baked if norm_plus_one.
host fn norm_w(dir: str, plus_one: bool, name: str) -> str {
    var raw = st_f16(dir, name)
    if raw == "" { return ""; }
    if plus_one {
        return w_f16_plus_one(raw);
    }
    return raw;
}

// weight_const: fp16 blob or q8/q4 quantized pair per cfg.weight_bits.
host fn wt_const(spec: Json, name: str, hexs: str, shape: [i64], wbits: i64) -> Json {
    var spec = spec
    if hexs == "" {
        return spec_err("missing weight bytes for " + name);
    }
    if wbits == 8 {
        var q = w_q8_rows(hexs, shape[0])
        var qm = obj_of(q)
        var qh = jstr_of(q, "q")
        var sh = jstr_of(q, "s")
        var r = spec_weight_q8(spec, name, shape, qh, sh)
        return r;
    }
    if wbits == 4 {
        var q4 = w_q4_rows(hexs, shape[0])
        var r4 = spec_weight_q4(spec, name, shape, jstr_of(q4, "q"), jstr_of(q4, "lut"))
        return r4;
    }
    spec = spec_weight(spec, name, "fp16", shape, hexs)
    return pair(spec, name);
}

// rms_norm_big — ops-only into an existing spec. Returns {spec,name}.
host det fn rmsb(spec: Json, x: str, w: str, d: i64, eps: f64,
                 shape4: [i64], pfx: str) -> Json {
    var spec = spec
    var mshape: [i64] = [shape4[0], 1, shape4[2], shape4[3]]
    var k = f32_sqrt(f64(d))
    var inv = f32_round(1.0 / f32_round(8.0 * k))
    var e = f32_round(eps / f32_round(f32_round(64.0 * k) * k))
    spec = konst_f16(spec, pfx + "_bigik", inv)
    spec = mul2(spec, x, pfx + "_bigik", shape4, pfx + "_bigxs")
    spec = mul2(spec, pfx + "_bigxs", pfx + "_bigxs", shape4, pfx + "_bigsq")
    spec = konst_i32(spec, pfx + "_bigax", [1])
    spec = konst_bool(spec, pfx + "_bigkd", true)
    var ins = empty_map()
    ins["x"] = bin_var(pfx + "_bigsq")
    ins["axes"] = bin_var(pfx + "_bigax")
    ins["keep_dims"] = bin_var(pfx + "_bigkd")
    spec = spec_op(spec, "reduce_mean", ins, [pfx + "_bigm"], [mshape], ["fp16"], empty_map())
    spec = konst_f16(spec, pfx + "_bige", e)
    spec = add2(spec, pfx + "_bigm", pfx + "_bige", mshape, pfx + "_bigvp")
    spec = konst_f16(spec, pfx + "_bigfl", 0.0000001)
    ins = empty_map()
    ins["x"] = bin_var(pfx + "_bigvp")
    ins["y"] = bin_var(pfx + "_bigfl")
    spec = spec_op(spec, "maximum", ins, [pfx + "_bigvm"], [mshape], ["fp16"], empty_map())
    spec = konst_f16(spec, pfx + "_bigz", 0.0)
    ins = empty_map()
    ins["x"] = bin_var(pfx + "_bigvm")
    ins["epsilon"] = bin_var(pfx + "_bigz")
    spec = spec_op(spec, "rsqrt", ins, [pfx + "_bigr"], [mshape], ["fp16"], empty_map())
    spec = mul2(spec, pfx + "_bigxs", pfx + "_bigr", shape4, pfx + "_xn")
    spec = mul2(spec, pfx + "_xn", w, shape4, pfx + "_out")
    return pair(spec, pfx + "_out");
}

// head_rms — per-head norm over axis 1; returns {spec,name=pfx_xn}
// (the weight mul is applied by the caller, like the Rust version).
host det fn hrms(spec: Json, x: str, d: i64, eps: f64, shape4: [i64], pfx: str) -> Json {
    var spec = spec
    var mshape: [i64] = [shape4[0], 1, 1, shape4[3]]
    var k = f32_sqrt(f64(d))
    var inv = f32_round(1.0 / f32_round(8.0 * k))
    var e = f32_round(eps / f32_round(f32_round(64.0 * k) * k))
    spec = konst_f16(spec, pfx + "_ik", inv)
    spec = mul2(spec, x, pfx + "_ik", shape4, pfx + "_xs")
    spec = mul2(spec, pfx + "_xs", pfx + "_xs", shape4, pfx + "_sq")
    spec = konst_i32(spec, pfx + "_ax", [1])
    spec = konst_bool(spec, pfx + "_kd", true)
    var ins = empty_map()
    ins["x"] = bin_var(pfx + "_sq")
    ins["axes"] = bin_var(pfx + "_ax")
    ins["keep_dims"] = bin_var(pfx + "_kd")
    spec = spec_op(spec, "reduce_mean", ins, [pfx + "_m"], [mshape], ["fp16"], empty_map())
    spec = konst_f16(spec, pfx + "_e", e)
    spec = add2(spec, pfx + "_m", pfx + "_e", mshape, pfx + "_vp")
    spec = konst_f16(spec, pfx + "_fl", 0.0000001)
    ins = empty_map()
    ins["x"] = bin_var(pfx + "_vp")
    ins["y"] = bin_var(pfx + "_fl")
    spec = spec_op(spec, "maximum", ins, [pfx + "_vm"], [mshape], ["fp16"], empty_map())
    spec = konst_f16(spec, pfx + "_z", 0.0)
    ins = empty_map()
    ins["x"] = bin_var(pfx + "_vm")
    ins["epsilon"] = bin_var(pfx + "_z")
    spec = spec_op(spec, "rsqrt", ins, [pfx + "_r"], [mshape], ["fp16"], empty_map())
    spec = mul2(spec, pfx + "_xs", pfx + "_r", shape4, pfx + "_xn")
    return pair(spec, pfx + "_xn");
}

// linear_w: reshape→reshape→matmul→reshape (matmul-form projection).
host det fn linearw2(spec: Json, x: str, w: str, d: i64, cout: i64,
                     s: i64, name: str) -> Json {
    var spec = spec
    var r = reshape2(spec, x, [1, 1, d, s], name + "_xr")
    spec = jobj_of(r, "spec")
    r = reshape2(spec, w, [1, 1, cout, d], name + "_wr")
    spec = jobj_of(r, "spec")
    r = matmul2(spec, name + "_wr", name + "_xr", false, [1, 1, cout, s], name)
    spec = jobj_of(r, "spec")
    r = reshape2(spec, name, [1, cout, 1, s], name + "_b")
    spec = jobj_of(r, "spec")
    return pair(spec, name + "_b");
}

// conv1x1_w: s==1 → blk.conv1x1 (fresh consts, out [1,cout,1,1]);
// s>1 → hand-rolled conv with {name}_c* consts, out [1,cout,1,s].
host det fn convw2(spec: Json, x: str, w: str, cout: i64, s: i64, name: str) -> Json {
    var spec = spec
    if s == 1 {
        return conv1x1_s(spec, x, w, "", false, cout, 1, name);
    }
    spec = konst_i32(spec, name + "_cstrides", [1, 1])
    spec = konst_str(spec, name + "_cpadtype", "valid")
    spec = konst_i32(spec, name + "_cpad", [0, 0, 0, 0])
    spec = konst_i32(spec, name + "_cdil", [1, 1])
    spec = konst_scalar_i32(spec, name + "_cgrp", 1)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["weight"] = bin_var(w)
    ins["strides"] = bin_var(name + "_cstrides")
    ins["pad_type"] = bin_var(name + "_cpadtype")
    ins["pad"] = bin_var(name + "_cpad")
    ins["dilations"] = bin_var(name + "_cdil")
    ins["groups"] = bin_var(name + "_cgrp")
    spec = spec_op(spec, "conv", ins, [name], [[1, cout, 1, s]], ["fp16"], empty_map())
    return pair(spec, name);
}

// RoPE for s==1 — rank-3 layout; cos/sin arrive (1, rope_half).
host det fn rope1(spec: Json, xf: str, n_h: i64, dim: i64, total: i64,
                  rope_dim: i64, rope_half: i64, cos_in: str, sin_in: str,
                  tag: str) -> Json {
    var spec = spec
    var r = reshape2(spec, xf, [1, n_h, dim], tag + "_r")
    spec = jobj_of(r, "spec")
    var has_pass = rope_dim < dim
    var x_rot = tag + "_r"
    var x_pass = ""
    if has_pass {
        r = slice2(spec, tag + "_r", [0, 0, 0], [1, n_h, rope_dim],
                   [1, n_h, rope_dim], tag + "_rot")
        spec = jobj_of(r, "spec")
        x_rot = tag + "_rot"
        r = slice2(spec, tag + "_r", [0, 0, rope_dim], [1, n_h, dim],
                   [1, n_h, dim - rope_dim], tag + "_pass")
        spec = jobj_of(r, "spec")
        x_pass = tag + "_pass"
    }
    r = slice2(spec, x_rot, [0, 0, 0], [1, n_h, rope_half],
               [1, n_h, rope_half], tag + "_lo")
    spec = jobj_of(r, "spec")
    r = slice2(spec, x_rot, [0, 0, rope_half], [1, n_h, rope_dim],
               [1, n_h, rope_half], tag + "_hi")
    spec = jobj_of(r, "spec")
    r = reshape2(spec, cos_in, [1, 1, rope_half], tag + "_cos")
    spec = jobj_of(r, "spec")
    r = reshape2(spec, sin_in, [1, 1, rope_half], tag + "_sin")
    spec = jobj_of(r, "spec")
    spec = mul2(spec, tag + "_lo", tag + "_cos", [1, n_h, rope_half], tag + "_a")
    spec = mul2(spec, tag + "_hi", tag + "_sin", [1, n_h, rope_half], tag + "_b")
    spec = sub2(spec, tag + "_a", tag + "_b", [1, n_h, rope_half], tag + "_rlo")
    spec = mul2(spec, tag + "_lo", tag + "_sin", [1, n_h, rope_half], tag + "_c")
    spec = mul2(spec, tag + "_hi", tag + "_cos", [1, n_h, rope_half], tag + "_d")
    spec = add2(spec, tag + "_c", tag + "_d", [1, n_h, rope_half], tag + "_rhi")
    if has_pass {
        r = concat2(spec, [tag + "_rlo", tag + "_rhi", x_pass], -1,
                    [1, n_h, dim], tag + "_cat")
    } else {
        r = concat2(spec, [tag + "_rlo", tag + "_rhi"], -1,
                    [1, n_h, dim], tag + "_cat")
    }
    spec = jobj_of(r, "spec")
    r = reshape2(spec, tag + "_cat", [1, total, 1, 1], tag + "_flat")
    spec = jobj_of(r, "spec")
    return pair(spec, tag + "_flat");
}

// RoPE for s>1 — rank-4 layout; cos/sin already (1,1,rope_half,s).
host det fn ropep(spec: Json, xf: str, n_h: i64, dim: i64, total: i64,
                  rope_dim: i64, rope_half: i64, cos_in: str, sin_in: str,
                  s: i64, tag: str) -> Json {
    var spec = spec
    var r = reshape2(spec, xf, [1, n_h, dim, s], tag + "_r")
    spec = jobj_of(r, "spec")
    var has_pass = rope_dim < dim
    var x_rot = tag + "_r"
    var x_pass = ""
    if has_pass {
        r = slice2(spec, tag + "_r", [0, 0, 0, 0], [1, n_h, rope_dim, s],
                   [1, n_h, rope_dim, s], tag + "_rot")
        spec = jobj_of(r, "spec")
        x_rot = tag + "_rot"
        r = slice2(spec, tag + "_r", [0, 0, rope_dim, 0], [1, n_h, dim, s],
                   [1, n_h, dim - rope_dim, s], tag + "_pass")
        spec = jobj_of(r, "spec")
        x_pass = tag + "_pass"
    }
    r = slice2(spec, x_rot, [0, 0, 0, 0], [1, n_h, rope_half, s],
               [1, n_h, rope_half, s], tag + "_lo")
    spec = jobj_of(r, "spec")
    r = slice2(spec, x_rot, [0, 0, rope_half, 0], [1, n_h, rope_dim, s],
               [1, n_h, rope_half, s], tag + "_hi")
    spec = jobj_of(r, "spec")
    spec = mul2(spec, tag + "_lo", cos_in, [1, n_h, rope_half, s], tag + "_a")
    spec = mul2(spec, tag + "_hi", sin_in, [1, n_h, rope_half, s], tag + "_b")
    spec = sub2(spec, tag + "_a", tag + "_b", [1, n_h, rope_half, s], tag + "_rlo")
    spec = mul2(spec, tag + "_lo", sin_in, [1, n_h, rope_half, s], tag + "_c")
    spec = mul2(spec, tag + "_hi", cos_in, [1, n_h, rope_half, s], tag + "_d")
    spec = add2(spec, tag + "_c", tag + "_d", [1, n_h, rope_half, s], tag + "_rhi")
    if has_pass {
        r = concat2(spec, [tag + "_rlo", tag + "_rhi", x_pass], 2,
                    [1, n_h, dim, s], tag + "_cat")
    } else {
        r = concat2(spec, [tag + "_rlo", tag + "_rhi"], 2,
                    [1, n_h, dim, s], tag + "_cat")
    }
    spec = jobj_of(r, "spec")
    r = reshape2(spec, tag + "_cat", [1, total, 1, s], tag + "_flat")
    spec = jobj_of(r, "spec")
    return pair(spec, tag + "_flat");
}

// concat4 — four named i32 inputs → i32[4] (slice_update begin/end vecs).
host det fn concat4(spec: Json, pfx: str, v0: str, v1: str,
                    v2: str, v3: str) -> Json {
    var spec = spec
    spec = konst_scalar_i32(spec, pfx + "_ax", 0)
    spec = konst_bool(spec, pfx + "_il", false)
    var ins = empty_map()
    var vs: [Json] = []
    push(vs, bin_var(v0))
    push(vs, bin_var(v1))
    push(vs, bin_var(v2))
    push(vs, bin_var(v3))
    ins["values"] = Json::Arr(vs)
    ins["axis"] = bin_var(pfx + "_ax")
    ins["interleave"] = bin_var(pfx + "_il")
    spec = spec_op(spec, "concat", ins, [pfx], [[4]], ["int32"], empty_map())
    return pair(spec, pfx);
}

// Per-update mask/stride consts for slice_update (fresh consts per update —
// reused consts fail ANE plan-build).
host det fn mk_masks(spec: Json, pfx: str) -> Json {
    var spec = spec
    spec = konst_i32(spec, pfx + "_st", [1, 1, 1, 1])
    spec = konst(spec, pfx + "_bm",
                 boolsv(bool_n(4, false)), [4], "bool")
    var r: {str: Json} = {}
    r["spec"] = spec
    r["st"] = Json::Str(pfx + "_st")
    r["bm"] = Json::Str(pfx + "_bm")
    return Json::Obj(r);
}

// slice_update o1.
host det fn slice_upd(spec: Json, x: str, upd: str, b: str, e: str,
                      stm: str, bm: str, kvshape: [i64], name: str) -> Json {
    var spec = spec
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["update"] = bin_var(upd)
    ins["begin"] = bin_var(b)
    ins["end"] = bin_var(e)
    ins["stride"] = bin_var(stm)
    ins["begin_mask"] = bin_var(bm)
    ins["end_mask"] = bin_var(bm)
    ins["squeeze_mask"] = bin_var(bm)
    spec = spec_op(spec, "slice_update", ins, [name], [kvshape], ["fp16"], empty_map())
    return pair(spec, name);
}

// ── emit_layer_shard ────────────────────────────────────────────────────
// Returns a mil spec Json {ops,weights,inputs,outputs,states,fn_inputs,
// meta,block_outputs} or {"error": msg}. cfg carries the Rust Cfg fields
// plus the debug flags as bools (cut_at/stop_at as str, split_gu as i64).
host fn mil_layer_spec(cfg: Json, dir: str, ls: i64, le: i64,
                       seq: i64, s: i64, is_prefill: bool) -> Json {
    var spec = spec_new()
    var d = cnum(cfg, "d_model", 0)
    var nh = cnum(cfg, "n_heads", 0)
    var nkv = cnum(cfg, "n_kv", 0)
    var dh = cnum(cfg, "d_head", 0)
    var rope_dim = cnum(cfg, "rope_dim", dh)
    var rope_half = rope_dim / 2
    var dff = cnum(cfg, "d_ff", 0)
    var eps = cf64(cfg, "eps", 0.000001)
    var attn_scale = cf64(cfg, "attn_scale", 0.0)
    var wbits = cnum(cfg, "weight_bits", 16)
    var plus_one = cbool(cfg, "norm_plus_one")
    var has_qk = cbool(cfg, "has_qk_norm")
    var q_dim = nh * dh
    var kv_dim = nkv * dh
    var hpk = nh / nkv
    var n_state = le - ls
    var shape_x: [i64] = [1, d, 1, s]
    var kvshape: [i64] = [2 * n_state, nkv, seq, dh]
    var split_ffn = cbool(cfg, "split_ffn")
    var cut_at = cstr(cfg, "cut_at")
    var gemma = is_arch(cfg, "gemma3_text")
    var no_postnorm = cbool(cfg, "no_postnorm")

    // Packed KV state: ONE buffer per shard, rows 2l (K) / 2l+1 (V).
    var rs = read_state2(spec, "kv_cache", kvshape, "kv_init")
    var spec = jobj_of(rs, "spec")
    var cur_kv = jstr_of(rs, "name")

    var cur = "x"
    var cut_shape: [i64] = []
    var cut_extra: [Json] = []
    var out_shape_w = 0
    for i in ls..le {
        var p = "model.layers." + str(i)
        // ---- rope input row select (two rows: global 0, local 1) ----
        var cos_in = "rope_cos"
        var sin_in = "rope_sin"
        if rope_rows(cfg) == 2 {
            var row = rope_row(cfg, i)
            if s == 1 {
                var rr = slice2(spec, "rope_cos", [row, 0], [row + 1, rope_half],
                                [1, rope_half], "l" + str(i) + "_cosr")
                spec = jobj_of(rr, "spec")
                cos_in = "l" + str(i) + "_cosr"
                rr = slice2(spec, "rope_sin", [row, 0], [row + 1, rope_half],
                            [1, rope_half], "l" + str(i) + "_sinr")
                spec = jobj_of(rr, "spec")
                sin_in = "l" + str(i) + "_sinr"
            } else {
                var rr = slice2(spec, "rope_cos", [row, 0, 0, 0],
                                [row + 1, 1, rope_half, s],
                                [1, 1, rope_half, s], "l" + str(i) + "_cosr")
                spec = jobj_of(rr, "spec")
                cos_in = "l" + str(i) + "_cosr"
                rr = slice2(spec, "rope_sin", [row, 0, 0, 0],
                            [row + 1, 1, rope_half, s],
                            [1, 1, rope_half, s], "l" + str(i) + "_sinr")
                spec = jobj_of(rr, "spec")
                sin_in = "l" + str(i) + "_sinr"
            }
        }
        // ---- attn_norm ----
        var w = norm_w(dir, plus_one, p + ".input_layernorm.weight")
        if w == "" { return spec_err("missing " + p + ".input_layernorm.weight"); }
        spec = spec_weight(spec, "w_attn_norm_" + str(i), "fp16", [d, 1, 1], w)
        var rn = rmsb(spec, cur, "w_attn_norm_" + str(i), d, eps, shape_x,
                      "l" + str(i) + "_an")
        spec = jobj_of(rn, "spec")
        var normed = jstr_of(rn, "name")
        // cut!("an")
        if cut_at == "an" && n_state == 1 {
            spec = konst_f16(spec, "cutz_an", 0.0)
            spec = add2(spec, normed, "cutz_an", shape_x, "hidden")
            cur = "hidden"
            cut_shape = shape_x
            break
        }
        // ---- fused QKV projection ----
        var qw = st_f16(dir, p + ".self_attn.q_proj.weight")
        var kw = st_f16(dir, p + ".self_attn.k_proj.weight")
        var vw = st_f16(dir, p + ".self_attn.v_proj.weight")
        if qw == "" || kw == "" || vw == "" {
            return spec_err("missing qkv weight layer " + str(i));
        }
        // hex concat == byte concat
        var qkv_w = qw + kw + vw
        var wc = wt_const(spec, "w_qkv_" + str(i), qkv_w,
                          [q_dim + 2 * kv_dim, d, 1, 1], wbits)
        if jhas(wc, "error") { return wc; }
        spec = jobj_of(wc, "spec")
        var wqkv = jstr_of(wc, "name")
        var qkv = ""
        if cbool(cfg, "qkv_input") {
            qkv = "qkv"
        } else if s == 1 {
            var lp = linearw2(spec, normed, wqkv, d, q_dim + 2 * kv_dim, s,
                              "l" + str(i) + "_qkv")
            spec = jobj_of(lp, "spec")
            qkv = jstr_of(lp, "name")
        } else {
            var cp = convw2(spec, normed, wqkv, q_dim + 2 * kv_dim, s,
                            "l" + str(i) + "_qkv")
            spec = jobj_of(cp, "spec")
            qkv = jstr_of(cp, "name")
        }
        var sr = slice2(spec, qkv, [0, 0, 0, 0], [1, q_dim, 1, s],
                        [1, q_dim, 1, s], "l" + str(i) + "_q")
        spec = jobj_of(sr, "spec")
        var q = jstr_of(sr, "name")
        sr = slice2(spec, qkv, [0, q_dim, 0, 0], [1, q_dim + kv_dim, 1, s],
                    [1, kv_dim, 1, s], "l" + str(i) + "_k")
        spec = jobj_of(sr, "spec")
        var kx = jstr_of(sr, "name")
        sr = slice2(spec, qkv, [0, q_dim + kv_dim, 0, 0],
                    [1, q_dim + 2 * kv_dim, 1, s],
                    [1, kv_dim, 1, s], "l" + str(i) + "_v")
        spec = jobj_of(sr, "spec")
        var v = jstr_of(sr, "name")
        if cut_at == "qkv" && n_state == 1 {
            spec = konst_f16(spec, "cutz_qkv", 0.0)
            spec = add2(spec, qkv, "cutz_qkv", [1, q_dim + 2 * kv_dim, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, q_dim + 2 * kv_dim, 1, s]
            break
        }
        // ---- per-head q/k RMSNorm ----
        var qn = q
        var kn = kx
        if has_qk && !cbool(cfg, "no_qknorm") {
            var qnw = norm_w(dir, plus_one, p + ".self_attn.q_norm.weight")
            var knw = norm_w(dir, plus_one, p + ".self_attn.k_norm.weight")
            if qnw == "" || knw == "" {
                return spec_err("missing qk norm weights layer " + str(i));
            }
            spec = spec_weight(spec, "w_qn_" + str(i), "fp16",
                               [q_dim, 1, 1], w_tile(qnw, nh))
            spec = spec_weight(spec, "w_kn_" + str(i), "fp16",
                               [kv_dim, 1, 1], w_tile(knw, nkv))
            var rp = reshape2(spec, q, [nh, dh, 1, s], "l" + str(i) + "_qh")
            spec = jobj_of(rp, "spec")
            var hn = hrms(spec, "l" + str(i) + "_qh", dh, eps,
                          [nh, dh, 1, s], "l" + str(i) + "_qn")
            spec = jobj_of(hn, "spec")
            rp = reshape2(spec, jstr_of(hn, "name"), [1, q_dim, 1, s],
                          "l" + str(i) + "_qnb")
            spec = jobj_of(rp, "spec")
            spec = mul2(spec, "l" + str(i) + "_qnb", "w_qn_" + str(i),
                        [1, q_dim, 1, s], "l" + str(i) + "_qo")
            qn = "l" + str(i) + "_qo"
            rp = reshape2(spec, kx, [nkv, dh, 1, s], "l" + str(i) + "_kh")
            spec = jobj_of(rp, "spec")
            hn = hrms(spec, "l" + str(i) + "_kh", dh, eps,
                      [nkv, dh, 1, s], "l" + str(i) + "_kn")
            spec = jobj_of(hn, "spec")
            rp = reshape2(spec, jstr_of(hn, "name"), [1, kv_dim, 1, s],
                          "l" + str(i) + "_knb")
            spec = jobj_of(rp, "spec")
            spec = mul2(spec, "l" + str(i) + "_knb", "w_kn_" + str(i),
                        [1, kv_dim, 1, s], "l" + str(i) + "_ko")
            kn = "l" + str(i) + "_ko"
        }
        if cut_at == "qn" && n_state == 1 {
            spec = konst_f16(spec, "cutz_qn", 0.0)
            spec = add2(spec, qn, "cutz_qn", [1, q_dim, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, q_dim, 1, s]
            break
        }
        // ---- RoPE ----
        var q_r = qn
        var k_r = kn
        if cbool(cfg, "no_rope") {
            q_r = qn
            k_r = kn
        } else if s == 1 {
            var rop = rope1(spec, qn, nh, dh, q_dim, rope_dim, rope_half,
                            cos_in, sin_in, "l" + str(i) + "_rq")
            spec = jobj_of(rop, "spec")
            q_r = jstr_of(rop, "name")
            rop = rope1(spec, kn, nkv, dh, kv_dim, rope_dim, rope_half,
                        cos_in, sin_in, "l" + str(i) + "_rk")
            spec = jobj_of(rop, "spec")
            k_r = jstr_of(rop, "name")
        } else {
            var rop = ropep(spec, qn, nh, dh, q_dim, rope_dim, rope_half,
                            cos_in, sin_in, s, "l" + str(i) + "_rq")
            spec = jobj_of(rop, "spec")
            q_r = jstr_of(rop, "name")
            rop = ropep(spec, kn, nkv, dh, kv_dim, rope_dim, rope_half,
                        cos_in, sin_in, s, "l" + str(i) + "_rk")
            spec = jobj_of(rop, "spec")
            k_r = jstr_of(rop, "name")
        }
        if cut_at == "rq" && n_state == 1 {
            spec = konst_f16(spec, "cutz_rq", 0.0)
            spec = add2(spec, q_r, "cutz_rq", [1, q_dim, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, q_dim, 1, s]
            break
        }
        var new_k = ""
        var new_v = ""
        if s == 1 {
            var rk = reshape2(spec, k_r, [1, nkv, 1, dh], "l" + str(i) + "_nk")
            spec = jobj_of(rk, "spec")
            new_k = jstr_of(rk, "name")
            rk = reshape2(spec, v, [1, nkv, 1, dh], "l" + str(i) + "_nv")
            spec = jobj_of(rk, "spec")
            new_v = jstr_of(rk, "name")
        } else {
            var rk = reshape2(spec, k_r, [1, nkv, dh, s], "l" + str(i) + "_nkr")
            spec = jobj_of(rk, "spec")
            rk = transpose2(spec, jstr_of(rk, "name"), [0, 1, 3, 2],
                            [1, nkv, s, dh], "l" + str(i) + "_nk")
            spec = jobj_of(rk, "spec")
            new_k = jstr_of(rk, "name")
            var rv = reshape2(spec, v, [1, nkv, dh, s], "l" + str(i) + "_nvr")
            spec = jobj_of(rv, "spec")
            rv = transpose2(spec, jstr_of(rv, "name"), [0, 1, 3, 2],
                            [1, nkv, s, dh], "l" + str(i) + "_nv")
            spec = jobj_of(rv, "spec")
            new_v = jstr_of(rv, "name")
        }
        // ---- packed-KV state update + attention ----
        // The Rust emitter's `if true`: the runtime-pos write path runs for
        // every s — the prefill hybrid (kv_write_mask merge) is dead code
        // there, so it is dead here. pos drives begin/end of the row write.
        var local = i - ls
        var krow = 2 * local
        spec = konst_i32(spec, "l" + str(i) + "_psc", [s])
        var ins = empty_map()
        ins["x"] = bin_var("pos")
        ins["y"] = bin_var("l" + str(i) + "_psc")
        spec = spec_op(spec, "add", ins, ["l" + str(i) + "_ps"],
                       [[1]], ["int32"], empty_map())
        spec = konst_i32(spec, "l" + str(i) + "_c0", [0])
        spec = konst_i32(spec, "l" + str(i) + "_ck", [krow])
        spec = konst_i32(spec, "l" + str(i) + "_ck1", [krow + 1])
        spec = konst_i32(spec, "l" + str(i) + "_cv", [krow + 2])
        spec = konst_i32(spec, "l" + str(i) + "_cnkv", [nkv])
        spec = konst_i32(spec, "l" + str(i) + "_cdh", [dh])
        var mk = mk_masks(spec, "l" + str(i) + "_km")
        spec = jobj_of(mk, "spec")
        var kst = jstr_of(mk, "st")
        var kbm = jstr_of(mk, "bm")
        mk = mk_masks(spec, "l" + str(i) + "_vm")
        spec = jobj_of(mk, "spec")
        var vst = jstr_of(mk, "st")
        var vbm = jstr_of(mk, "bm")
        var c4 = concat4(spec, "l" + str(i) + "_kb",
                         "l" + str(i) + "_ck", "l" + str(i) + "_c0",
                         "pos", "l" + str(i) + "_c0")
        spec = jobj_of(c4, "spec")
        c4 = concat4(spec, "l" + str(i) + "_ke",
                     "l" + str(i) + "_ck1", "l" + str(i) + "_cnkv",
                     "l" + str(i) + "_ps", "l" + str(i) + "_cdh")
        spec = jobj_of(c4, "spec")
        var su = slice_upd(spec, cur_kv, new_k, "l" + str(i) + "_kb",
                           "l" + str(i) + "_ke", kst, kbm, kvshape,
                           "l" + str(i) + "_kup")
        spec = jobj_of(su, "spec")
        cur_kv = jstr_of(su, "name")
        if cut_at == "kupd" && n_state == 1 {
            spec = konst_f16(spec, "cutz_kupd", 0.0)
            spec = add2(spec, cur_kv, "cutz_kupd", kvshape, "hidden")
            cur = "hidden"
            cut_shape = kvshape
            break
        }
        var dup = cbool(cfg, "kv_dup")
        var vb0 = "l" + str(i) + "_ck1"
        var ve0 = "l" + str(i) + "_cv"
        var vupd = new_v
        if dup {
            vb0 = "l" + str(i) + "_ck"
            ve0 = "l" + str(i) + "_ck1"
            vupd = new_k
        }
        c4 = concat4(spec, "l" + str(i) + "_vb", vb0, "l" + str(i) + "_c0",
                     "pos", "l" + str(i) + "_c0")
        spec = jobj_of(c4, "spec")
        c4 = concat4(spec, "l" + str(i) + "_ve", ve0, "l" + str(i) + "_cnkv",
                     "l" + str(i) + "_ps", "l" + str(i) + "_cdh")
        spec = jobj_of(c4, "spec")
        su = slice_upd(spec, cur_kv, vupd, "l" + str(i) + "_vb",
                       "l" + str(i) + "_ve", vst, vbm, kvshape,
                       "l" + str(i) + "_vup")
        spec = jobj_of(su, "spec")
        cur_kv = jstr_of(su, "name")
        if cut_at == "vupd" && n_state == 1 {
            var ex: {str: Json} = {}
            ex["name"] = Json::Str(q_r)
            ex["shape"] = shape_arr([1, q_dim, 1, s])
            push(cut_extra, Json::Obj(ex))
            spec = konst_f16(spec, "cutz_vupd", 0.0)
            spec = add2(spec, cur_kv, "cutz_vupd", kvshape, "hidden")
            cur = "hidden"
            cut_shape = kvshape
            break
        }
        var ks = slice2(spec, cur_kv, [krow, 0, 0, 0],
                        [krow + 1, nkv, seq, dh], [1, nkv, seq, dh],
                        "l" + str(i) + "_kfull")
        spec = jobj_of(ks, "spec")
        var k_attn = jstr_of(ks, "name")
        ks = slice2(spec, cur_kv, [krow + 1, 0, 0, 0],
                    [krow + 2, nkv, seq, dh], [1, nkv, seq, dh],
                    "l" + str(i) + "_vfull")
        spec = jobj_of(ks, "spec")
        var v_attn = jstr_of(ks, "name")
        if cut_at == "kf" && n_state == 1 {
            spec = konst_f16(spec, "cutz_kf", 0.0)
            spec = add2(spec, k_attn, "cutz_kf", [1, nkv, seq, dh], "hidden")
            cur = "hidden"
            cut_shape = [1, nkv, seq, dh]
            break
        }
        if cut_at == "kv" && n_state == 1 {
            spec = konst_f16(spec, "cutz_kv", 0.0)
            spec = add2(spec, k_attn, "cutz_kv", [1, nkv, seq, dh], "hidden")
            cur = "hidden"
            cut_shape = [1, nkv, seq, dh]
            break
        }
        // ---- explicit attention (scores·scale·+mask·softmax·matmul) ----
        var rows = s * hpk
        var qf = ""
        if s == 1 {
            rows = 2 * hpk
            var q2 = reshape2(spec, q_r, [1, nkv, hpk, dh],
                              "l" + str(i) + "_qfold2")
            spec = jobj_of(q2, "spec")
            var qc = concat2(spec, [jstr_of(q2, "name"), jstr_of(q2, "name")],
                             2, [1, nkv, 2 * hpk, dh], "l" + str(i) + "_qfold")
            spec = jobj_of(qc, "spec")
            qf = "l" + str(i) + "_qfold"
        } else {
            var q4 = reshape2(spec, q_r, [nkv, hpk * dh, 1, s],
                              "l" + str(i) + "_qheads")
            spec = jobj_of(q4, "spec")
            var qt = transpose2(spec, jstr_of(q4, "name"), [0, 3, 2, 1],
                                [nkv, s, 1, hpk * dh], "l" + str(i) + "_qt")
            spec = jobj_of(qt, "spec")
            var qf2 = reshape2(spec, jstr_of(qt, "name"),
                               [1, nkv, s * hpk, dh], "l" + str(i) + "_qfold")
            spec = jobj_of(qf2, "spec")
            qf = "l" + str(i) + "_qfold"
        }
        var mm = matmul2(spec, qf, k_attn, true, [1, nkv, rows, seq],
                         "l" + str(i) + "_sc")
        spec = jobj_of(mm, "spec")
        spec = konst_f16(spec, "l" + str(i) + "_scl", attn_scale)
        spec = mul2(spec, "l" + str(i) + "_sc", "l" + str(i) + "_scl",
                    [1, nkv, rows, seq], "l" + str(i) + "_scm")
        if cut_at == "sc" && n_state == 1 {
            spec = konst_f16(spec, "cutz_sc", 0.0)
            spec = add2(spec, "l" + str(i) + "_scm", "cutz_sc",
                        [1, nkv, rows, seq], "hidden")
            cur = "hidden"
            cut_shape = [1, nkv, rows, seq]
            break
        }
        spec = add2(spec, "l" + str(i) + "_scm", "attn_mask",
                    [1, nkv, rows, seq], "l" + str(i) + "_msk")
        var masked = "l" + str(i) + "_msk"
        if is_sliding(cfg, i) {
            spec = add2(spec, masked, "attn_mask_slide",
                        [1, nkv, rows, seq], "l" + str(i) + "_mskw")
            masked = "l" + str(i) + "_mskw"
        }
        var pb = softmax2(spec, masked, 3, [1, nkv, rows, seq],
                          "l" + str(i) + "_pb", false)
        spec = jobj_of(pb, "spec")
        if cut_at == "pb" && n_state == 1 {
            spec = konst_f16(spec, "cutz_pb", 0.0)
            spec = add2(spec, "l" + str(i) + "_pb", "cutz_pb",
                        [1, nkv, rows, seq], "hidden")
            cur = "hidden"
            cut_shape = [1, nkv, rows, seq]
            break
        }
        var v_op = v_attn
        if s == 1 && cbool(cfg, "ones_attn") {
            spec = konst_f16(spec, "l" + str(i) + "_vzc", 0.0)
            spec = mul2(spec, v_attn, "l" + str(i) + "_vzc",
                        [1, nkv, seq, dh], "l" + str(i) + "_vz")
            spec = konst_f16(spec, "l" + str(i) + "_voc", 1.0)
            spec = add2(spec, "l" + str(i) + "_vz", "l" + str(i) + "_voc",
                        [1, nkv, seq, dh], "l" + str(i) + "_vones")
            v_op = "l" + str(i) + "_vones"
        }
        mm = matmul2(spec, "l" + str(i) + "_pb", v_op, false,
                     [1, nkv, rows, dh], "l" + str(i) + "_sdpa")
        spec = jobj_of(mm, "spec")
        if cut_at == "attn" && n_state == 1 {
            spec = konst_f16(spec, "cutz_attn", 0.0)
            spec = add2(spec, "l" + str(i) + "_sdpa", "cutz_attn",
                        [1, nkv, rows, dh], "hidden")
            cur = "hidden"
            cut_shape = [1, nkv, rows, dh]
            break
        }
        var attn4 = ""
        if s == 1 {
            var asl = slice2(spec, "l" + str(i) + "_sdpa", [0, 0, 0, 0],
                             [1, nkv, hpk, dh], [1, nkv, hpk, dh],
                             "l" + str(i) + "_aslice")
            spec = jobj_of(asl, "spec")
            var af = reshape2(spec, jstr_of(asl, "name"), [1, q_dim, 1, 1],
                              "l" + str(i) + "_attn4")
            spec = jobj_of(af, "spec")
            attn4 = "l" + str(i) + "_attn4"
        } else {
            var ar = reshape2(spec, "l" + str(i) + "_sdpa",
                              [nkv, s, hpk * dh, 1], "l" + str(i) + "_ar")
            spec = jobj_of(ar, "spec")
            var at = transpose2(spec, jstr_of(ar, "name"), [0, 2, 3, 1],
                                [nkv, hpk * dh, 1, s], "l" + str(i) + "_at")
            spec = jobj_of(at, "spec")
            var af = reshape2(spec, jstr_of(at, "name"), [1, q_dim, 1, s],
                              "l" + str(i) + "_attn4")
            spec = jobj_of(af, "spec")
            attn4 = "l" + str(i) + "_attn4"
        }
        // ---- o_proj + residual ----
        var ow = st_f16(dir, p + ".self_attn.o_proj.weight")
        if ow == "" { return spec_err("missing " + p + ".self_attn.o_proj.weight"); }
        var woc = wt_const(spec, "w_o_" + str(i), ow, [d, q_dim, 1, 1], wbits)
        if jhas(woc, "error") { return woc; }
        spec = jobj_of(woc, "spec")
        var wo = jstr_of(woc, "name")
        var oc = convw2(spec, attn4, wo, d, s, "l" + str(i) + "_o")
        spec = jobj_of(oc, "spec")
        var o_out = jstr_of(oc, "name")
        if cut_at == "o" && n_state == 1 {
            spec = konst_f16(spec, "cutz_o", 0.0)
            spec = add2(spec, o_out, "cutz_o", shape_x, "hidden")
            cur = "hidden"
            cut_shape = shape_x
            break
        }
        var o2 = o_out
        if gemma && !no_postnorm {
            var wp = norm_w(dir, plus_one, p + ".post_attention_layernorm.weight")
            if wp == "" { return spec_err("missing post_attention_layernorm l" + str(i)); }
            spec = spec_weight(spec, "w_post_attn_" + str(i), "fp16", [d, 1, 1], wp)
            var pn = rmsb(spec, o_out, "w_post_attn_" + str(i), d, eps,
                          shape_x, "l" + str(i) + "_pan")
            spec = jobj_of(pn, "spec")
            o2 = jstr_of(pn, "name")
        }
        spec = add2(spec, cur, o2, shape_x, "l" + str(i) + "_res1")
        var res1 = "l" + str(i) + "_res1"
        // ---- FFN norm ----
        var ffn_norm_name = "post_attention_layernorm"
        if gemma { ffn_norm_name = "pre_feedforward_layernorm" }
        var wfn = norm_w(dir, plus_one, p + "." + ffn_norm_name + ".weight")
        if wfn == "" { return spec_err("missing " + ffn_norm_name + " l" + str(i)); }
        spec = spec_weight(spec, "w_ffn_norm_" + str(i), "fp16", [d, 1, 1], wfn)
        if cut_at == "res1" && n_state == 1 {
            spec = konst_f16(spec, "cutz_res1", 0.0)
            spec = add2(spec, res1, "cutz_res1", shape_x, "hidden")
            cur = "hidden"
            cut_shape = shape_x
            break
        }
        // CUT=kvdbg: hidden = concat(K rows 0..8 read back post-update,
        // res1) — reads the state's actual contents through the terminal
        // output buffer.
        if cut_at == "kvdbg" && n_state == 1 {
            var k8 = slice2(spec, k_attn, [0, 0, 0, 0], [1, nkv, 8, dh],
                            [1, nkv, 8, dh], "l" + str(i) + "_ks8")
            spec = jobj_of(k8, "spec")
            var kf = reshape2(spec, jstr_of(k8, "name"),
                              [1, nkv * 8 * dh, 1, 1], "l" + str(i) + "_ks8f")
            spec = jobj_of(kf, "spec")
            var kcat = concat2(spec, [jstr_of(kf, "name"), res1], 1,
                               [1, nkv * 8 * dh + d, 1, s], "hidden")
            spec = jobj_of(kcat, "spec")
            cur = "hidden"
            cut_shape = [1, nkv * 8 * dh + d, 1, s]
            break
        }
        var fnn = rmsb(spec, res1, "w_ffn_norm_" + str(i), d, eps, shape_x,
                       "l" + str(i) + "_fn")
        spec = jobj_of(fnn, "spec")
        var n2 = jstr_of(fnn, "name")
        if split_ffn {
            var cats = concat2(spec, [res1, n2], 1, [1, 2 * d, 1, s], "hidden")
            spec = jobj_of(cats, "spec")
            cur = "hidden"
            cut_shape = [1, 2 * d, 1, s]
            break
        }
        if cut_at == "fn" && n_state == 1 {
            var exf: {str: Json} = {}
            exf["name"] = Json::Str(n2)
            exf["shape"] = shape_arr(shape_x)
            push(cut_extra, Json::Obj(exf))
            spec = konst_f16(spec, "cutz_fn", 0.0)
            spec = add2(spec, n2, "cutz_fn", shape_x, "hidden")
            cur = "hidden"
            cut_shape = shape_x
            break
        }
        // reverse_sequence round-trip: forces the norm output to physically
        // materialize before the wide conv reads it.
        spec = konst_i32(spec, "l" + str(i) + "_rsl", [d])
        spec = konst_scalar_i32(spec, "l" + str(i) + "_rsa", 1)
        spec = konst_scalar_i32(spec, "l" + str(i) + "_rba", 0)
        var rins = empty_map()
        rins["x"] = bin_var("l" + str(i) + "_fn_xn")
        rins["lengths"] = bin_var("l" + str(i) + "_rsl")
        rins["seq_axis"] = bin_var("l" + str(i) + "_rsa")
        rins["batch_axis"] = bin_var("l" + str(i) + "_rba")
        spec = spec_op(spec, "reverse_sequence", rins, ["l" + str(i) + "_fnr1"],
                       [shape_x], ["fp16"], empty_map())
        spec = konst_scalar_i32(spec, "l" + str(i) + "_rsa2", 1)
        spec = konst_scalar_i32(spec, "l" + str(i) + "_rba2", 0)
        rins = empty_map()
        rins["x"] = bin_var("l" + str(i) + "_fnr1")
        rins["lengths"] = bin_var("l" + str(i) + "_rsl")
        rins["seq_axis"] = bin_var("l" + str(i) + "_rsa2")
        rins["batch_axis"] = bin_var("l" + str(i) + "_rba2")
        spec = spec_op(spec, "reverse_sequence", rins, ["l" + str(i) + "_fnr2"],
                       [shape_x], ["fp16"], empty_map())
        var ffn_in = "l" + str(i) + "_fnr2"
        // column-fold the FFN norm weight into gate/up weights
        var gw0 = st_f16(dir, p + ".mlp.gate_proj.weight")
        var uw0 = st_f16(dir, p + ".mlp.up_proj.weight")
        if gw0 == "" || uw0 == "" {
            return spec_err("missing mlp weights l" + str(i));
        }
        var gw = w_f16_scale_cols(gw0, wfn)
        var uw = w_f16_scale_cols(uw0, wfn)
        var gate = ""
        var up = ""
        var split_n = cnum(cfg, "split_gu", 0)
        if split_n >= 1 {
            var chunk = dff / split_n
            var gouts: [str] = []
            var uouts: [str] = []
            for c in 0..split_n {
                var row_b = chunk * d * 2
                var lo = c * row_b
                var gc = wt_const(spec, "w_g" + str(c) + "_" + str(i),
                                  slice_str(gw, lo, lo + row_b),
                                  [chunk, d, 1, 1], wbits)
                if jhas(gc, "error") { return gc; }
                spec = jobj_of(gc, "spec")
                var gcv = convw2(spec, ffn_in, jstr_of(gc, "name"), chunk, s,
                                 "l" + str(i) + "_g" + str(c))
                spec = jobj_of(gcv, "spec")
                push(gouts, jstr_of(gcv, "name"))
                var uc = wt_const(spec, "w_u" + str(c) + "_" + str(i),
                                  slice_str(uw, lo, lo + row_b),
                                  [chunk, d, 1, 1], wbits)
                if jhas(uc, "error") { return uc; }
                spec = jobj_of(uc, "spec")
                var ucv = convw2(spec, ffn_in, jstr_of(uc, "name"), chunk, s,
                                 "l" + str(i) + "_u" + str(c))
                spec = jobj_of(ucv, "spec")
                push(uouts, jstr_of(ucv, "name"))
            }
            if split_n == 1 {
                gate = gouts[0]
                up = uouts[0]
            } else {
                var gcat = concat2(spec, gouts, 1, [1, dff, 1, s],
                                   "l" + str(i) + "_gcat")
                spec = jobj_of(gcat, "spec")
                gate = "l" + str(i) + "_gcat"
                var ucat = concat2(spec, uouts, 1, [1, dff, 1, s],
                                   "l" + str(i) + "_ucat")
                spec = jobj_of(ucat, "spec")
                up = "l" + str(i) + "_ucat"
            }
        } else {
            var gu_w = gw + uw
            var wc2 = wt_const(spec, "w_gu_" + str(i), gu_w,
                               [2 * dff, d, 1, 1], wbits)
            if jhas(wc2, "error") { return wc2; }
            spec = jobj_of(wc2, "spec")
            var wgu = jstr_of(wc2, "name")
            var gu = ""
            if cbool(cfg, "gu_mm") {
                var gl = linearw2(spec, ffn_in, wgu, d, 2 * dff, s,
                                  "l" + str(i) + "_gu")
                spec = jobj_of(gl, "spec")
                gu = jstr_of(gl, "name")
            } else {
                var gx = convw2(spec, ffn_in, wgu, 2 * dff, s,
                                "l" + str(i) + "_gu")
                spec = jobj_of(gx, "spec")
                gu = jstr_of(gx, "name")
            }
            // CUT=gufn: hidden = concat(attn4, o, res1, fn_xn, gu) — the
            // terminal concat pins all five chain values in one buffer.
            if cut_at == "gufn" && n_state == 1 {
                var gc = concat2(spec,
                                 ["l" + str(i) + "_attn4",
                                  "l" + str(i) + "_o",
                                  res1,
                                  "l" + str(i) + "_fn_xn",
                                  gu],
                                 1, [1, q_dim + 3 * d + 2 * dff, 1, s],
                                 "hidden")
                spec = jobj_of(gc, "spec")
                cur = "hidden"
                cut_shape = [1, q_dim + 3 * d + 2 * dff, 1, s]
                break
            }
            if cut_at == "gu" && n_state == 1 {
                spec = konst_f16(spec, "cutz_gu", 0.0)
                spec = add2(spec, gu, "cutz_gu", [1, 2 * dff, 1, s], "hidden")
                cur = "hidden"
                cut_shape = [1, 2 * dff, 1, s]
                break
            }
            var gs = slice2(spec, gu, [0, 0, 0, 0], [1, dff, 1, s],
                            [1, dff, 1, s], "l" + str(i) + "_gate")
            spec = jobj_of(gs, "spec")
            gate = "l" + str(i) + "_gate"
            var us = slice2(spec, gu, [0, dff, 0, 0], [1, 2 * dff, 1, s],
                            [1, dff, 1, s], "l" + str(i) + "_up")
            spec = jobj_of(us, "spec")
            up = "l" + str(i) + "_up"
        }
        // Gate activation: Qwen3 silu, Gemma3 gelu(tanh).
        if gemma && !cbool(cfg, "gemma_silu") {
            var gins = empty_map()
            gins["x"] = bin_var(gate)
            gins["mode"] = bin_const_val(strv("TANH_APPROXIMATION"))
            spec = spec_op(spec, "gelu", gins, ["l" + str(i) + "_gelu"],
                           [[1, dff, 1, s]], ["fp16"], empty_map())
            var gact = "l" + str(i) + "_gelu"
            spec = mul2(spec, gact, up, [1, dff, 1, s], "l" + str(i) + "_hid")
        } else {
            var sins = empty_map()
            sins["x"] = bin_var(gate)
            spec = spec_op(spec, "silu", sins, ["l" + str(i) + "_silu"],
                           [[1, dff, 1, s]], ["fp16"], empty_map())
            spec = mul2(spec, "l" + str(i) + "_silu", up,
                        [1, dff, 1, s], "l" + str(i) + "_hid")
        }
        var hidden = "l" + str(i) + "_hid"
        if cut_at == "hid" && n_state == 1 {
            spec = konst_f16(spec, "cutz_hid", 0.0)
            spec = add2(spec, hidden, "cutz_hid", [1, dff, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, dff, 1, s]
            break
        }
        var dw = st_f16(dir, p + ".mlp.down_proj.weight")
        if dw == "" { return spec_err("missing down_proj l" + str(i)); }
        var wdc = wt_const(spec, "w_down_" + str(i), dw, [d, dff, 1, 1], wbits)
        if jhas(wdc, "error") { return wdc; }
        spec = jobj_of(wdc, "spec")
        var wd = jstr_of(wdc, "name")
        // int8 path: pre-scale the hidden by 1/α, restore by α.
        var alpha = cf64(cfg, "down_prescale", 1.0)
        var ffn = ""
        if alpha > 1.0 {
            spec = konst_f16(spec, "l" + str(i) + "_dinv", f32_round(1.0 / f32_round(alpha)))
            spec = mul2(spec, hidden, "l" + str(i) + "_dinv",
                        [1, dff, 1, s], "l" + str(i) + "_hids")
            var dc = convw2(spec, "l" + str(i) + "_hids", wd, d, s,
                            "l" + str(i) + "_down_s")
            spec = jobj_of(dc, "spec")
            spec = konst_f16(spec, "l" + str(i) + "_dalpha", alpha)
            spec = mul2(spec, "l" + str(i) + "_down_s", "l" + str(i) + "_dalpha",
                        shape_x, "l" + str(i) + "_down")
            ffn = "l" + str(i) + "_down"
        } else {
            var dc = convw2(spec, hidden, wd, d, s, "l" + str(i) + "_down")
            spec = jobj_of(dc, "spec")
            ffn = "l" + str(i) + "_down"
        }
        if cut_at == "down" && n_state == 1 {
            spec = konst_f16(spec, "cutz_down", 0.0)
            spec = add2(spec, ffn, "cutz_down", shape_x, "hidden")
            cur = "hidden"
            cut_shape = shape_x
            break
        }
        var ffn_out = ffn
        if gemma && !no_postnorm {
            var wpf = norm_w(dir, plus_one, p + ".post_feedforward_layernorm.weight")
            if wpf == "" { return spec_err("missing post_ffn l" + str(i)); }
            spec = spec_weight(spec, "w_post_ffn_" + str(i), "fp16", [d, 1, 1], wpf)
            var pfn = rmsb(spec, ffn, "w_post_ffn_" + str(i), d, eps,
                           shape_x, "l" + str(i) + "_pfn")
            spec = jobj_of(pfn, "spec")
            ffn_out = jstr_of(pfn, "name")
        }
        var stop_at = cstr(cfg, "stop_at")
        var out_nm = "l" + str(i) + "_res2"
        if i == le - 1 && (stop_at == "" || s != 1) { out_nm = "hidden" }
        spec = add2(spec, res1, ffn_out, shape_x, out_nm)
        cur = out_nm
    }
    // Commit the chained packed-KV updates once per call.
    spec = write_state2(spec, "kv_cache", cur_kv)
    // BADAPPLE_STOP_AT=l0_<stage>: single output is a mid-block
    // intermediate re-bound through add(x, tail*0) so the whole graph
    // stays live (dead state writes fail plan-build -14).
    var stop_at = cstr(cfg, "stop_at")
    var stop_shape: [i64] = []
    if stop_at != "" && n_state == 1 && s == 1 {
        var i = ls
        var stop_names: [str] = ["an_out", "qo", "ko", "rq_flat", "rk_flat",
                               "attn4", "v", "sc", "msk", "pb", "o", "res1",
                               "fn_out", "hid"]
        var stop_shapes: [Json] = [
            shape_arr([1, d, 1, 1]),
            shape_arr([1, q_dim, 1, 1]),
            shape_arr([1, kv_dim, 1, 1]),
            shape_arr([1, q_dim, 1, 1]),
            shape_arr([1, kv_dim, 1, 1]),
            shape_arr([1, q_dim, 1, 1]),
            shape_arr([1, kv_dim, 1, 1]),
            shape_arr([1, nkv, 2 * hpk, seq]),
            shape_arr([1, nkv, 2 * hpk, seq]),
            shape_arr([1, nkv, 2 * hpk, seq]),
            shape_arr([1, d, 1, 1]),
            shape_arr([1, d, 1, 1]),
            shape_arr([1, d, 1, 1]),
            shape_arr([1, dff, 1, 1])
        ]
        for t in 0..len(stop_names) {
            var n = "l" + str(i) + "_" + stop_names[t]
            if stop_at == n {
                var shp = i64_vec(stop_shapes[t])
                var flat = 1
                for d2 in 0..len(shp) { flat = flat * shp[d2] }
                var src = n
                if !(len(shp) == 4 && shp[0] == 1 && shp[1] == flat
                     && shp[2] == 1 && shp[3] == 1) {
                    var fr = reshape2(spec, n, [1, flat, 1, 1],
                                      n + "_flat")
                    spec = jobj_of(fr, "spec")
                    src = jstr_of(fr, "name")
                }
                spec = konst_i32(spec, "stop_ax", [1])
                spec = konst_bool(spec, "stop_kd", true)
                var sins = empty_map()
                sins["x"] = bin_var(cur)
                sins["axes"] = bin_var("stop_ax")
                sins["keep_dims"] = bin_var("stop_kd")
                spec = spec_op(spec, "reduce_mean", sins, ["stop_tail"],
                               [[1, 1, 1, 1]], ["fp16"], empty_map())
                spec = konst_f16(spec, "stop_zero", 0.0)
                spec = mul2(spec, "stop_tail", "stop_zero",
                            [1, 1, 1, 1], "stop_keep")
                spec = add2(spec, src, "stop_keep", [1, flat, 1, 1], "hidden")
                cur = "hidden"
                stop_shape = [1, flat, 1, 1]
            }
        }
    }
    var m = obj_of(spec)
    var out_shape_decl = cut_shape
    if len(out_shape_decl) == 0 { out_shape_decl = stop_shape }
    if len(out_shape_decl) == 0 { out_shape_decl = shape_x }
    // ---- features ----
    spec = Json::Obj(m)
    spec = spec_add_input(spec, "x", shape_x, "fp16")
    var rrows = rope_rows(cfg)
    var rope_shape: [i64] = [rrows, rope_half]
    if s != 1 { rope_shape = [rrows, 1, rope_half, s] }
    spec = spec_add_input(spec, "rope_cos", rope_shape, "fp16")
    spec = spec_add_input(spec, "rope_sin", rope_shape, "fp16")
    var mask_rows = s * hpk
    if s == 1 { mask_rows = 2 * hpk }
    var mask_shape: [i64] = [1, 1, mask_rows, seq]
    spec = spec_add_input(spec, "attn_mask", mask_shape, "fp16")
    spec = spec_add_input(spec, "pos", [1], "int32")
    var has_sliding = false
    for i in ls..le {
        if is_sliding(cfg, i) { has_sliding = true }
    }
    if has_sliding {
        spec = spec_add_input(spec, "attn_mask_slide", mask_shape, "fp16")
    }
    if cbool(cfg, "qkv_input") {
        spec = spec_add_input(spec, "qkv", [1, q_dim + 2 * kv_dim, 1, s], "fp16")
    }
    if cbool(cfg, "gu_input") {
        spec = spec_add_input(spec, "wgu_g", [dff, d, 1, 1], "fp16")
        spec = spec_add_input(spec, "wgu_u", [dff, d, 1, 1], "fp16")
    }
    spec = spec_add_output(spec, "hidden", out_shape_decl, "fp16")
    for i in 0..len(cut_extra) {
        var ce = obj_of(cut_extra[i])
        spec = spec_add_output(spec, jstr_of(cut_extra[i], "name"),
                               i64_vec(jobj_of(cut_extra[i], "shape")), "fp16")
    }
    spec = spec_add_state(spec, "kv_cache", kvshape, "fp16")
    // ---- fn_inputs (same order as the Rust emitter) ----
    spec = spec_add_fn_input(spec, "x", shape_x, "fp16")
    spec = spec_add_fn_input(spec, "rope_cos", rope_shape, "fp16")
    spec = spec_add_fn_input(spec, "rope_sin", rope_shape, "fp16")
    spec = spec_add_fn_input(spec, "attn_mask", mask_shape, "fp16")
    spec = spec_add_fn_input(spec, "pos", [1], "int32")
    if has_sliding {
        spec = spec_add_fn_input(spec, "attn_mask_slide", mask_shape, "fp16")
    }
    spec = spec_add_fn_state(spec, "kv_cache", kvshape, "fp16")
    if cbool(cfg, "qkv_input") {
        spec = spec_add_fn_input(spec, "qkv", [1, q_dim + 2 * kv_dim, 1, s], "fp16")
    }
    if cbool(cfg, "gu_input") {
        spec = spec_add_fn_input(spec, "wgu_g", [dff, d, 1, 1], "fp16")
        spec = spec_add_fn_input(spec, "wgu_u", [dff, d, 1, 1], "fp16")
    }
    // ---- meta: decode shards plan under ios18/CoreML8; prefill ios19/CoreML9
    var mm2 = obj_of(spec)
    var meta = obj_of(mm2["meta"])
    var sv = 9
    var opset = "CoreML8"
    if s != 1 { sv = 10; opset = "CoreML9" }
    if cnum(cfg, "spec_version", 0) > 0 {
        sv = cnum(cfg, "spec_version", sv)
        opset = cstr(cfg, "opset")
    }
    meta["spec_version"] = Json::Num(f64(sv))
    meta["opset"] = Json::Str(opset)
    meta["creator"] = Json::Str("badapple-aneconvert")
    meta["description"] = Json::Str("Bad Apple native ANE shard")
    mm2["meta"] = Json::Obj(meta)
    var bo: [Json] = []
    push(bo, Json::Str(cur))
    for i in 0..len(cut_extra) {
        push(bo, Json::Str(jstr_of(cut_extra[i], "name")))
    }
    mm2["block_outputs"] = Json::Arr(bo)
    return Json::Obj(mm2);
}

// Json::Arr of i64 → [i64] (cut_extra shape restore).
host det fn i64_vec(j: Json) -> [i64] {
    var res: [i64] = []
    var a = arr_of(j)
    for i in 0..len(a) {
        match a[i] {
            Json::Num(v) => { push(res, i64(v)) }
            _ => { }
        }
    }
    return res;
}

// One-call emit: build the layer spec and lower it to {spec,weights} hex.
// Returns {"error": msg} on failure.
host fn mil_layer_emit(cfg: Json, dir: str, ls: i64, le: i64,
                       seq: i64, s: i64, is_prefill: bool) -> Json {
    var spec = mil_layer_spec(cfg, dir, ls, le, seq, s, is_prefill)
    if jhas(spec, "error") { return spec; }
    return mil_emit(spec);
}

// map-has for Json objs (jstr_of throws on missing keys).
host det fn jhas(j: Json, k: str) -> bool {
    return has(obj_of(j), k);
}

// ── emit_ffn_shard ──────────────────────────────────────────────────────
// Stateless FFN half of a split layer: consumes x = concat(res1, ffn_norm)
// [1,2d,1,s], emits res1 + post_ffn_norm(ffn). Gate/up and down run as
// matmul here (stateless shard — no slice_update -14 case; the conv form
// miscomputes at stride-32 in this plan).
host fn mil_ffn_spec(cfg: Json, dir: str, ls: i64, le: i64, s: i64) -> Json {
    if le - ls != 1 {
        return spec_err("split FFN shards are per-layer (le-ls must be 1)");
    }
    var spec = spec_new()
    var d = cnum(cfg, "d_model", 0)
    var dff = cnum(cfg, "d_ff", 0)
    var eps = cf64(cfg, "eps", 0.000001)
    var wbits = cnum(cfg, "weight_bits", 16)
    var gemma = is_arch(cfg, "gemma3_text")
    var no_postnorm = cbool(cfg, "no_postnorm")
    var ffn_cut = cstr(cfg, "ffn_cut")
    var shape_x: [i64] = [1, d, 1, s]
    var shape_2d: [i64] = [1, 2 * d, 1, s]
    var cut_shape: [i64] = []

    var sr = slice2(spec, "x", [0, 0, 0, 0], [1, d, 1, s],
                    shape_x, "in_res1")
    spec = jobj_of(sr, "spec")
    var res1 = "in_res1"
    sr = slice2(spec, "x", [0, d, 0, 0], [1, 2 * d, 1, s],
                shape_x, "in_fn")
    spec = jobj_of(sr, "spec")
    var n2 = "in_fn"
    var cur = res1
    for i in ls..le {
        var p = "model.layers." + str(i)
        var gw = st_f16(dir, p + ".mlp.gate_proj.weight")
        var uw = st_f16(dir, p + ".mlp.up_proj.weight")
        if gw == "" || uw == "" {
            return spec_err("missing mlp weights l" + str(i));
        }
        var gu_w = gw + uw
        var wc = wt_const(spec, "w_gu_" + str(i), gu_w, [2 * dff, d, 1, 1], wbits)
        if jhas(wc, "error") { return wc; }
        spec = jobj_of(wc, "spec")
        var wgu = jstr_of(wc, "name")
        var gl = linearw2(spec, n2, wgu, d, 2 * dff, s, "l" + str(i) + "_gu")
        spec = jobj_of(gl, "spec")
        var gu = jstr_of(gl, "name")
        if ffn_cut == "gu" {
            spec = konst_f16(spec, "fcz", 0.0)
            spec = add2(spec, gu, "fcz", [1, 2 * dff, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, 2 * dff, 1, s]
            break
        }
        var gs = slice2(spec, gu, [0, 0, 0, 0], [1, dff, 1, s],
                        [1, dff, 1, s], "l" + str(i) + "_gate")
        spec = jobj_of(gs, "spec")
        var gate = "l" + str(i) + "_gate"
        var us = slice2(spec, gu, [0, dff, 0, 0], [1, 2 * dff, 1, s],
                        [1, dff, 1, s], "l" + str(i) + "_up")
        spec = jobj_of(us, "spec")
        var up = "l" + str(i) + "_up"
        if gemma && !cbool(cfg, "gemma_silu") {
            var gins = empty_map()
            gins["x"] = bin_var(gate)
            gins["mode"] = bin_const_val(strv("TANH_APPROXIMATION"))
            spec = spec_op(spec, "gelu", gins, ["l" + str(i) + "_gelu"],
                           [[1, dff, 1, s]], ["fp16"], empty_map())
            spec = mul2(spec, "l" + str(i) + "_gelu", up,
                        [1, dff, 1, s], "l" + str(i) + "_hid")
        } else {
            var sins = empty_map()
            sins["x"] = bin_var(gate)
            spec = spec_op(spec, "silu", sins, ["l" + str(i) + "_silu"],
                           [[1, dff, 1, s]], ["fp16"], empty_map())
            spec = mul2(spec, "l" + str(i) + "_silu", up,
                        [1, dff, 1, s], "l" + str(i) + "_hid")
        }
        var hidden = "l" + str(i) + "_hid"
        if ffn_cut == "hid" {
            spec = konst_f16(spec, "fcz", 0.0)
            spec = add2(spec, hidden, "fcz", [1, dff, 1, s], "hidden")
            cur = "hidden"
            cut_shape = [1, dff, 1, s]
            break
        }
        var dw = st_f16(dir, p + ".mlp.down_proj.weight")
        if dw == "" { return spec_err("missing down_proj l" + str(i)); }
        var wdc = wt_const(spec, "w_down_" + str(i), dw, [d, dff, 1, 1], wbits)
        if jhas(wdc, "error") { return wdc; }
        spec = jobj_of(wdc, "spec")
        var wd = jstr_of(wdc, "name")
        var alpha = cf64(cfg, "down_prescale", 1.0)
        var ffn = ""
        if alpha > 1.0 {
            spec = konst_f16(spec, "l" + str(i) + "_dinv",
                             f32_round(1.0 / f32_round(alpha)))
            spec = mul2(spec, hidden, "l" + str(i) + "_dinv",
                        [1, dff, 1, s], "l" + str(i) + "_hids")
            var dr = linearw2(spec, "l" + str(i) + "_hids", wd, dff, d, s,
                              "l" + str(i) + "_down_s")
            spec = jobj_of(dr, "spec")
            spec = konst_f16(spec, "l" + str(i) + "_dalpha", alpha)
            spec = mul2(spec, jstr_of(dr, "name"),
                        "l" + str(i) + "_dalpha", shape_x, "l" + str(i) + "_down")
            ffn = "l" + str(i) + "_down"
        } else {
            var dr = linearw2(spec, hidden, wd, dff, d, s,
                              "l" + str(i) + "_down")
            spec = jobj_of(dr, "spec")
            ffn = jstr_of(dr, "name")
        }
        var ffn_out = ffn
        if gemma && !no_postnorm {
            var wpf = norm_w(dir, cbool(cfg, "norm_plus_one"),
                             p + ".post_feedforward_layernorm.weight")
            if wpf == "" { return spec_err("missing post_ffn l" + str(i)); }
            spec = spec_weight(spec, "w_post_ffn_" + str(i), "fp16", [d, 1, 1], wpf)
            var pfn = rmsb(spec, ffn, "w_post_ffn_" + str(i), d, eps,
                           shape_x, "l" + str(i) + "_pfn")
            spec = jobj_of(pfn, "spec")
            ffn_out = jstr_of(pfn, "name")
        }
        var out_nm = "l" + str(i) + "_res2"
        if i == le - 1 { out_nm = "hidden" }
        spec = add2(spec, res1, ffn_out, shape_x, out_nm)
        cur = out_nm
    }
    var m = obj_of(spec)
    var out_shape_decl = cut_shape
    if len(out_shape_decl) == 0 { out_shape_decl = shape_x }
    spec = Json::Obj(m)
    spec = spec_add_input(spec, "x", shape_2d, "fp16")
    spec = spec_add_output(spec, "hidden", out_shape_decl, "fp16")
    spec = spec_add_fn_input(spec, "x", shape_2d, "fp16")
    var mm2 = obj_of(spec)
    var meta = obj_of(mm2["meta"])
    var sv = 9
    var opset = "CoreML8"
    if s != 1 { sv = 10; opset = "CoreML9" }
    if cnum(cfg, "spec_version", 0) > 0 {
        sv = cnum(cfg, "spec_version", sv)
        opset = cstr(cfg, "opset")
    }
    meta["spec_version"] = Json::Num(f64(sv))
    meta["opset"] = Json::Str(opset)
    meta["creator"] = Json::Str("badapple-aneconvert")
    meta["description"] = Json::Str("Bad Apple native ANE FFN shard (split layer tail)")
    mm2["meta"] = Json::Obj(meta)
    var bo: [Json] = []
    push(bo, Json::Str(cur))
    mm2["block_outputs"] = Json::Arr(bo)
    return Json::Obj(mm2);
}

host fn mil_ffn_emit(cfg: Json, dir: str, ls: i64, le: i64, s: i64) -> Json {
    var spec = mil_ffn_spec(cfg, dir, ls, le, s)
    if jhas(spec, "error") { return spec; }
    return mil_emit(spec);
}

// ── emit_head_shard ─────────────────────────────────────────────────────
// Vocab-slice LM head: final RMSNorm + sliced conv projection.
host fn mil_head_spec(cfg: Json, dir: str, vstart: i64, vend: i64) -> Json {
    var spec = spec_new()
    var d = cnum(cfg, "d_model", 0)
    var eps = cf64(cfg, "eps", 0.000001)
    var wbits = cnum(cfg, "weight_bits", 16)
    var plus_one = cbool(cfg, "norm_plus_one")
    var vsz = vend - vstart
    var nw = norm_w(dir, plus_one, "model.norm.weight")
    if nw == "" { return spec_err("missing model.norm.weight"); }
    spec = spec_weight(spec, "w_out_norm", "fp16", [d, 1, 1], nw)
    var rn = rmsb(spec, "x", "w_out_norm", d, eps, [1, d, 1, 1], "hn")
    spec = jobj_of(rn, "spec")
    var normed = jstr_of(rn, "name")
    var lm_name = "lm_head.weight"
    if cbool(cfg, "tie_embeddings") ||
       st_f16_rows(dir, "lm_head.weight", 0, 1) == "" {
        lm_name = "model.embed_tokens.weight"
    }
    var slice_hex = st_f16_rows(dir, lm_name, vstart, vend)
    if slice_hex == "" { return spec_err("missing " + lm_name); }
    var wc = wt_const(spec, "w_lm", slice_hex, [vsz, d, 1, 1], wbits)
    if jhas(wc, "error") { return wc; }
    spec = jobj_of(wc, "spec")
    var wl = jstr_of(wc, "name")
    var lc = convw2(spec, normed, wl, vsz, 1, "logits")
    spec = jobj_of(lc, "spec")
    var logits = jstr_of(lc, "name")
    spec = spec_add_input(spec, "x", [1, d, 1, 1], "fp16")
    spec = spec_add_output(spec, "logits", [1, vsz, 1, 1], "fp16")
    spec = spec_add_fn_input(spec, "x", [1, d, 1, 1], "fp16")
    var mm2 = obj_of(spec)
    var meta = obj_of(mm2["meta"])
    var sv = 9
    var opset = "CoreML8"
    if cnum(cfg, "spec_version", 0) > 0 {
        sv = cnum(cfg, "spec_version", sv)
        opset = cstr(cfg, "opset")
    }
    meta["spec_version"] = Json::Num(f64(sv))
    meta["opset"] = Json::Str(opset)
    meta["creator"] = Json::Str("badapple-aneconvert")
    meta["description"] = Json::Str("Bad Apple native ANE shard")
    mm2["meta"] = Json::Obj(meta)
    var bo: [Json] = []
    push(bo, Json::Str(logits))
    mm2["block_outputs"] = Json::Arr(bo)
    return Json::Obj(mm2);
}

host fn mil_head_emit(cfg: Json, dir: str, vstart: i64, vend: i64) -> Json {
    var spec = mil_head_spec(cfg, dir, vstart, vend)
    if jhas(spec, "error") { return spec; }
    return mil_emit(spec);
}

// ── emit_probe ──────────────────────────────────────────────────────────
// Qwen3-only numeric-bisection packages: emit a graph that stops at `stage`
// so a failing layer can be isolated op by op. finish_probe's shared tail
// declares x/rope_cos/rope_sin and a single "out" feature.
host det fn probe_finish(spec: Json, out_nm: str, d: i64, out_dim: i64,
                         rope_half: i64) -> Json {
    var spec2 = spec
    spec2 = spec_add_input(spec2, "x", [1, d, 1, 1], "fp16")
    spec2 = spec_add_input(spec2, "rope_cos", [1, rope_half], "fp16")
    spec2 = spec_add_input(spec2, "rope_sin", [1, rope_half], "fp16")
    spec2 = spec_add_output(spec2, "out", [1, out_dim, 1, 1], "fp16")
    spec2 = spec_add_fn_input(spec2, "x", [1, d, 1, 1], "fp16")
    spec2 = spec_add_fn_input(spec2, "rope_cos", [1, rope_half], "fp16")
    spec2 = spec_add_fn_input(spec2, "rope_sin", [1, rope_half], "fp16")
    var mm2 = obj_of(spec2)
    var meta = obj_of(mm2["meta"])
    meta["spec_version"] = Json::Num(9.0)
    meta["opset"] = Json::Str("CoreML8")
    meta["creator"] = Json::Str("badapple-aneconvert")
    meta["description"] = Json::Str("Bad Apple native ANE shard")
    mm2["meta"] = Json::Obj(meta)
    var bo: [Json] = []
    push(bo, Json::Str(out_nm))
    mm2["block_outputs"] = Json::Arr(bo)
    return Json::Obj(mm2);
}

// probe-local rope: rank-3 reshape → rot/pass split → rotate-half → concat
// → flatten. Distinct from rope1/ropep (the layer-shard decode/prefill
// layouts) — this one feeds un-normalized rank-3 slices.
host det fn probe_rope(spec: Json, xf: str, n_h: i64, dim: i64, total: i64,
                       rope_dim: i64, rope_half: i64, tag: str) -> Json {
    var spec2 = jobj_of(reshape2(spec, xf, [1, n_h, dim], tag + "_r"), "spec")
    var rot_end = rope_dim
    var has_pass = rope_dim < dim
    var x_rot = tag + "_r"
    if has_pass {
        spec2 = jobj_of(slice2(spec2, x_rot, [0, 0, 0], [1, n_h, rot_end],                       [1, n_h, rope_dim], tag + "_rot"), "spec")
        x_rot = tag + "_rot"
    }
    var x_pass = ""
    if has_pass {
        spec2 = jobj_of(slice2(spec2, tag + "_r", [0, 0, rot_end], [1, n_h, dim],                       [1, n_h, dim - rope_dim], tag + "_pass"), "spec")
        x_pass = tag + "_pass"
    }
    spec2 = jobj_of(slice2(spec2, x_rot, [0, 0, 0], [1, n_h, rope_half],                   [1, n_h, rope_half], tag + "_lo"), "spec")
    spec2 = jobj_of(slice2(spec2, x_rot, [0, 0, rope_half], [1, n_h, rot_end],                   [1, n_h, rope_half], tag + "_hi"), "spec")
    spec2 = jobj_of(reshape2(spec2, "rope_cos", [1, 1, rope_half], tag + "_cos"), "spec")
    spec2 = jobj_of(reshape2(spec2, "rope_sin", [1, 1, rope_half], tag + "_sin"), "spec")
    spec2 = mul2(spec2, tag + "_lo", tag + "_cos", [1, n_h, rope_half], tag + "_a")
    spec2 = mul2(spec2, tag + "_hi", tag + "_sin", [1, n_h, rope_half], tag + "_b")
    spec2 = sub2(spec2, tag + "_a", tag + "_b", [1, n_h, rope_half], tag + "_rlo")
    spec2 = mul2(spec2, tag + "_lo", tag + "_sin", [1, n_h, rope_half], tag + "_c")
    spec2 = mul2(spec2, tag + "_hi", tag + "_cos", [1, n_h, rope_half], tag + "_d")
    spec2 = add2(spec2, tag + "_c", tag + "_d", [1, n_h, rope_half], tag + "_rhi")
    if has_pass {
        var xs: [str] = [tag + "_rlo", tag + "_rhi", x_pass]
        spec2 = jobj_of(concat2(spec2, xs, -1, [1, n_h, dim], tag + "_cat"), "spec")
    } else {
        var xs2: [str] = [tag + "_rlo", tag + "_rhi"]
        spec2 = jobj_of(concat2(spec2, xs2, -1, [1, n_h, dim], tag + "_cat"), "spec")
    }
    spec2 = jobj_of(reshape2(spec2, tag + "_cat", [1, total, 1, 1], tag + "_flat"), "spec")
    return spec2;
}

host fn mil_probe_spec(cfg: Json, dir: str, ls: i64, stage: str) -> Json {
    if !is_arch(cfg, "qwen3") {
        return spec_err("probe emitter is qwen3-only");
    }
    var spec = spec_new()
    var d = cnum(cfg, "d_model", 0)
    var nh = cnum(cfg, "n_heads", 0)
    var nkv = cnum(cfg, "n_kv", 0)
    var dh = cnum(cfg, "d_head", 0)
    var rope_dim = cnum(cfg, "rope_dim", 0)
    var eps = cf64(cfg, "eps", 0.000001)
    var rope_half = rope_dim / 2
    var q_dim = nh * dh
    var kv_dim = nkv * dh
    var p = "model.layers." + str(ls)
    var w = st_f16(dir, p + ".input_layernorm.weight")
    if w == "" { return spec_err("missing input_layernorm l" + str(ls)); }
    spec = spec_weight(spec, "w_attn_norm_" + str(ls), "fp16", [d, 1, 1], w)
    var rn = rmsb(spec, "x", "w_attn_norm_" + str(ls), d, eps,
                  [1, d, 1, 1], "l" + str(ls) + "_an")
    spec = jobj_of(rn, "spec")
    var normed = jstr_of(rn, "name")
    var out_shape: [i64] = [1, d, 1, 1]
    if stage == "an" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, normed, "pz", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "xw" {
        spec = mul2(spec, "x", "w_attn_norm_" + str(ls), out_shape, "xw")
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "xw", "pz", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "muls" {
        spec = konst_f16(spec, "cs", 0.5)
        spec = mul2(spec, "x", "cs", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "adds" {
        spec = konst_f16(spec, "cs", 1.5)
        spec = add2(spec, "x", "cs", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "id" {
        spec = add2(spec, "x", "x", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "xw4" {
        spec = spec_weight(spec, "w_attn_norm4", "fp16", [1, d, 1, 1], w)
        spec = mul2(spec, "x", "w_attn_norm4", out_shape, "xw4")
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "xw4", "pz", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }
    if stage == "anr" {
        var k = f32_sqrt(f64(d))
        spec = konst_f16(spec, "anr_ik", f32_round(1.0 / k))
        spec = mul2(spec, "x", "anr_ik", out_shape, "anr_xs")
        spec = mul2(spec, "anr_xs", "anr_xs", out_shape, "anr_sq")
        spec = konst_i32(spec, "anr_ax", [1])
        spec = konst_bool(spec, "anr_kd", true)
        var mins = empty_map()
        mins["x"] = bin_var("anr_sq")
        mins["axes"] = bin_var("anr_ax")
        mins["keep_dims"] = bin_var("anr_kd")
        spec = spec_op(spec, "reduce_mean", mins, ["anr_m"],
                       [[1, 1, 1, 1]], ["fp16"], empty_map())
        spec = konst_f16(spec, "anr_e", f32_round(eps / f64(d)))
        spec = add2(spec, "anr_m", "anr_e", [1, 1, 1, 1], "anr_vp")
        spec = konst_f16(spec, "anr_z", 0.0)
        var rins = empty_map()
        rins["x"] = bin_var("anr_vp")
        rins["epsilon"] = bin_var("anr_z")
        spec = spec_op(spec, "rsqrt", rins, ["anr_r"],
                       [[1, 1, 1, 1]], ["fp16"], empty_map())
        spec = mul2(spec, "anr_xs", "anr_r", out_shape, "anr_xn")
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "anr_xn", "pz", out_shape, "out")
        return probe_finish(spec, "out", d, d, rope_half);
    }

    var qw = st_f16(dir, p + ".self_attn.q_proj.weight")
    var kw = st_f16(dir, p + ".self_attn.k_proj.weight")
    var vw = st_f16(dir, p + ".self_attn.v_proj.weight")
    if qw == "" || kw == "" || vw == "" {
        return spec_err("missing qkv weights l" + str(ls));
    }
    var qkv_w = qw + kw + vw
    spec = spec_weight(spec, "w_qkv_" + str(ls), "fp16",
                       [q_dim + 2 * kv_dim, d, 1, 1], qkv_w)
    var qc = convw2(spec, normed, "w_qkv_" + str(ls), q_dim + 2 * kv_dim,
                    1, "l" + str(ls) + "_qkv")
    spec = jobj_of(qc, "spec")
    var qkv = jstr_of(qc, "name")
    spec = jobj_of(slice2(spec, qkv, [0, 0, 0, 0], [1, q_dim, 1, 1],                  [1, q_dim, 1, 1], "l" + str(ls) + "_q"), "spec")
    var q = "l" + str(ls) + "_q"
    spec = jobj_of(slice2(spec, qkv, [0, q_dim, 0, 0], [1, q_dim + kv_dim, 1, 1],                  [1, kv_dim, 1, 1], "l" + str(ls) + "_k"), "spec")
    var k = "l" + str(ls) + "_k"
    spec = jobj_of(slice2(spec, qkv, [0, q_dim + kv_dim, 0, 0],                  [1, q_dim + 2 * kv_dim, 1, 1], [1, kv_dim, 1, 1],                  "l" + str(ls) + "_v"), "spec")
    var v = "l" + str(ls) + "_v"
    if stage == "q" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, q, "pz", [1, q_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, q_dim, rope_half);
    }
    if stage == "v" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, v, "pz", [1, kv_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, kv_dim, rope_half);
    }

    var qnw = st_f16(dir, p + ".self_attn.q_norm.weight")
    var knw = st_f16(dir, p + ".self_attn.k_norm.weight")
    if qnw == "" || knw == "" {
        return spec_err("missing qk norm weights l" + str(ls));
    }
    var qn_tiled = w_tile(qnw, nh)
    var kn_tiled = w_tile(knw, nkv)
    spec = spec_weight(spec, "w_qn_" + str(ls), "fp16", [q_dim, 1, 1], qn_tiled)
    spec = spec_weight(spec, "w_kn_" + str(ls), "fp16", [kv_dim, 1, 1], kn_tiled)
    spec = jobj_of(reshape2(spec, q, [nh, dh, 1, 1], "l" + str(ls) + "_qh"), "spec")
    var qsh: [i64] = [nh, dh, 1, 1]
    var qn = hrms(spec, "l" + str(ls) + "_qh", dh, eps, qsh,
                  "l" + str(ls) + "_qn")
    spec = jobj_of(qn, "spec")
    spec = jobj_of(reshape2(spec, jstr_of(qn, "name"), [1, q_dim, 1, 1],
                    "l" + str(ls) + "_qnb"), "spec")
    spec = mul2(spec, "l" + str(ls) + "_qnb", "w_qn_" + str(ls),
                [1, q_dim, 1, 1], "l" + str(ls) + "_qo")
    var qo = "l" + str(ls) + "_qo"
    spec = jobj_of(reshape2(spec, k, [nkv, dh, 1, 1], "l" + str(ls) + "_kh"), "spec")
    var ksh: [i64] = [nkv, dh, 1, 1]
    var kn = hrms(spec, "l" + str(ls) + "_kh", dh, eps, ksh,
                  "l" + str(ls) + "_kn")
    spec = jobj_of(kn, "spec")
    spec = jobj_of(reshape2(spec, jstr_of(kn, "name"), [1, kv_dim, 1, 1],
                    "l" + str(ls) + "_knb"), "spec")
    spec = mul2(spec, "l" + str(ls) + "_knb", "w_kn_" + str(ls),
                [1, kv_dim, 1, 1], "l" + str(ls) + "_ko")
    var ko = "l" + str(ls) + "_ko"
    if stage == "qn" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, qo, "pz", [1, q_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, q_dim, rope_half);
    }
    if stage == "kn" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, ko, "pz", [1, kv_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, kv_dim, rope_half);
    }

    if stage == "kv" {
        var seq = 2048
        spec = jobj_of(reshape2(spec, ko, [1, nkv, 1, dh], "kv_nk"), "spec")
        spec = konst_f16(spec, "kv_one", 1.0)
        spec = sub2(spec, "kv_one", "kv_write_mask", [1, 1, seq, 1], "kv_om")
        var ksh2: [i64] = [1, nkv, seq, dh]
        spec = jobj_of(read_state2(spec, "k_cache_0", ksh2, "kv_kold"), "spec")
        spec = mul2(spec, "kv_kold", "kv_om", ksh2, "kv_kkeep")
        spec = mul2(spec, "kv_nk", "kv_write_mask", ksh2, "kv_knew")
        spec = add2(spec, "kv_kkeep", "kv_knew", ksh2, "kv_kfull")
        spec = write_state2(spec, "k_cache_0", "kv_kfull")
        spec = jobj_of(read_state2(spec, "k_cache_0", ksh2, "kv_kupd"), "spec")
        spec = jobj_of(slice2(spec, "kv_kupd", [0, 0, 0, 0], [1, nkv, 1, dh],                      [1, nkv, 1, dh], "kv_row0"), "spec")
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "kv_row0", "pz", [1, nkv, 1, dh], "out")
        // dedicated finisher: adds kv_write_mask input + k_cache_0 state
        var spec2 = spec
        spec2 = spec_add_input(spec2, "x", [1, d, 1, 1], "fp16")
        spec2 = spec_add_input(spec2, "rope_cos", [1, rope_half], "fp16")
        spec2 = spec_add_input(spec2, "rope_sin", [1, rope_half], "fp16")
        spec2 = spec_add_input(spec2, "kv_write_mask", [1, 1, seq, 1], "fp16")
        spec2 = spec_add_output(spec2, "out", [1, nkv, 1, dh], "fp16")
        spec2 = spec_add_state(spec2, "k_cache_0", ksh2, "fp16")
        spec2 = spec_add_fn_input(spec2, "x", [1, d, 1, 1], "fp16")
        spec2 = spec_add_fn_input(spec2, "rope_cos", [1, rope_half], "fp16")
        spec2 = spec_add_fn_input(spec2, "rope_sin", [1, rope_half], "fp16")
        spec2 = spec_add_fn_input(spec2, "kv_write_mask", [1, 1, seq, 1], "fp16")
        spec2 = spec_add_fn_state(spec2, "k_cache_0", ksh2, "fp16")
        var mm2 = obj_of(spec2)
        var meta = obj_of(mm2["meta"])
        meta["spec_version"] = Json::Num(9.0)
        meta["opset"] = Json::Str("CoreML8")
        meta["creator"] = Json::Str("badapple-aneconvert")
        meta["description"] = Json::Str("Bad Apple native ANE shard")
        mm2["meta"] = Json::Obj(meta)
        var bo: [Json] = []
        push(bo, Json::Str("out"))
        mm2["block_outputs"] = Json::Arr(bo)
        return Json::Obj(mm2);
    }

    spec = probe_rope(spec, qo, nh, dh, q_dim, rope_dim, rope_half,
                      "l" + str(ls) + "_rq")
    if stage == "rq" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "l" + str(ls) + "_rq_flat", "pz",
                    [1, q_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, q_dim, rope_half);
    }
    spec = probe_rope(spec, ko, nkv, dh, kv_dim, rope_dim, rope_half,
                      "l" + str(ls) + "_rk")
    if stage == "rk" {
        spec = konst_f16(spec, "pz", 0.0)
        spec = add2(spec, "l" + str(ls) + "_rk_flat", "pz",
                    [1, kv_dim, 1, 1], "out")
        return probe_finish(spec, "out", d, kv_dim, rope_half);
    }
    return spec_err("unknown probe stage " + stage);
}

host fn mil_probe_emit(cfg: Json, dir: str, ls: i64, stage: str) -> Json {
    var spec = mil_probe_spec(cfg, dir, ls, stage)
    if jhas(spec, "error") { return spec; }
    return mil_emit(spec);
}

// ── emit_embedding + convert orchestration ──────────────────────────────
host det fn pad2(n: i64) -> str {
    if n < 10 { return "0" + str(n); }
    return str(n);
}
host det fn pad5(n: i64) -> str {
    var s = str(n)
    var need = 5 - len(s)
    var r = ""
    for i in 0..need { r = r + "0" }
    return r + s;
}
host det fn conv_rel(p: str, base: str) -> str {
    if starts_with(p, base + "/") { return slice_str(p, len(base) + 1, len(p)); }
    return p;
}
host det fn conv_dir_size(path: str) -> i64 {
    var files = arr_of(fs_walk(path, 8))
    var total: i64 = 0
    for i in 0..len(files) {
        var f = obj_of(files[i])
        match f["size"] {
            Json::Num(v) => { total = total + i64(v) }
            _ => {}
        }
    }
    return total;
}

// emit_embedding: scaled fp16 table row-major → embedding.f16
host fn mil_embed(cfg: Json, dir: str, out_path: str) -> Json {
    var raw = st_f16(dir, "model.embed_tokens.weight")
    if raw == "" { return spec_err("missing model.embed_tokens.weight"); }
    var scale = cf64(cfg, "embed_scale", 1.0)
    if scale != 1.0 { raw = w_f16_scale_all(raw, scale) }
    var want = cnum(cfg, "vocab", 0) * cnum(cfg, "d_model", 0) * 4
    if len(raw) != want {
        return spec_err("embedding size " + str(len(raw) / 2) +
                        " != expected " + str(want / 2));
    }
    if !fs_write_hex(out_path, raw) {
        return spec_err("write failed: " + out_path);
    }
    var r = empty_map()
    r["ok"] = Json::Bool(true)
    return Json::Obj(r);
}

// coremlc compile + compiled_dir bookkeeping → manifest status fields.
host fn conv_compile(pkg_dir: str, stem: str, out_dir: str,
                     keep: bool) -> Json {
    var ent = empty_map()
    var mlpath = out_dir + "/" + stem + ".mlmodelc"
    if fs_exists(mlpath) { fs_remove_all(mlpath) }
    var r = sys_exec("xcrun", ["coremlc", "compile", pkg_dir, out_dir])
    var code = cnum(r, "code", -1)
    if code != 0 {
        ent["status"] = Json::Str("compile_failed")
        ent["last_error"] = Json::Str("coremlc failed for " + pkg_dir +
            " (status " + str(code) + "): " + cstr(r, "err"))
        return Json::Obj(ent);
    }
    if !fs_exists(mlpath) {
        ent["status"] = Json::Str("compile_failed")
        ent["last_error"] = Json::Str("compiled output missing: " + mlpath)
        return Json::Obj(ent);
    }
    var sz = conv_dir_size(mlpath)
    ent["status"] = Json::Str("compiled")
    ent["compiled_path"] = Json::Str(conv_rel(mlpath, out_dir))
    ent["compiled_size_bytes"] = Json::Num(f64(sz))
    if !keep { fs_remove_all(pkg_dir) }
    return Json::Obj(ent);
}

host det fn mset(m: {str: Json}, k: str, v: Json) -> {str: Json} {
    var m2 = m
    m2[k] = v
    return m2;
}

// One layer shard (decode or prefill): emit → package → compile → entry.
// split_ffn runs the paired stateless FFN shard too.
host fn conv_layer_entry(cfg: Json, dir: str, out_dir: str, ls: i64,
                         le: i64, seq: i64, s: i64, is_prefill: bool,
                         no_compile: bool, keep: bool,
                         split_ffn: bool) -> Json {
    var name = "layer_s" + pad2(ls) + "-" + pad2(le)
    if is_prefill { name = "prefill_s" + pad2(ls) + "-" + pad2(le) }
    var pkg = out_dir + "/" + name + ".mlpackage"
    var ent = empty_map()
    ent["layer_start"] = Json::Num(f64(ls))
    ent["layer_end"] = Json::Num(f64(le))
    var r = mil_layer_emit(cfg, dir, ls, le, seq, s, is_prefill)
    if jhas(r, "error") {
        ent["status"] = Json::Str("emit_failed")
        ent["last_error"] = Json::Str(cstr(r, "error"))
        return Json::Obj(ent);
    }
    if !mil_write_package(pkg, cstr(r, "spec"), cstr(r, "weights")) {
        ent["status"] = Json::Str("emit_failed")
        ent["last_error"] = Json::Str("package write failed: " + pkg)
        return Json::Obj(ent);
    }
    ent["name"] = Json::Str(name)
    if no_compile {
        ent["status"] = Json::Str("packaged")
        log_err("  shard [" + str(ls) + "," + str(le) + ") packaged")
    } else {
        var cr = conv_compile(pkg, name, out_dir, keep)
        var cro = obj_of(cr)
        ent["status"] = cro["status"]
        if cstr(cr, "status") == "compiled" {
            ent["compiled_path"] = cro["compiled_path"]
            ent["compiled_size_bytes"] = cro["compiled_size_bytes"]
            log_err("  shard [" + str(ls) + "," + str(le) + ") compiled")
        } else {
            ent["last_error"] = cro["last_error"]
            log_err("  shard [" + str(ls) + "," + str(le) + ") FAILED")
        }
    }
    if split_ffn {
        var fname = "ffn_s" + pad2(ls) + "-" + pad2(le)
        if is_prefill { fname = "ffn_p" + pad2(ls) + "-" + pad2(le) }
        var fpkg = out_dir + "/" + fname + ".mlpackage"
        var fr = mil_ffn_emit(cfg, dir, ls, le, s)
        if jhas(fr, "error") {
            ent["ffn_status"] = Json::Str("emit_failed")
            ent["ffn_last_error"] = Json::Str(cstr(fr, "error"))
        } else {
            if !mil_write_package(fpkg, cstr(fr, "spec"), cstr(fr, "weights")) {
                ent["ffn_status"] = Json::Str("emit_failed")
                ent["ffn_last_error"] = Json::Str("package write failed: " + fpkg)
            } else {
                ent["ffn_package_path"] = Json::Str(conv_rel(fpkg, out_dir))
                if no_compile {
                    ent["ffn_status"] = Json::Str("packaged")
                } else {
                    var fc = conv_compile(fpkg, fname, out_dir, keep)
                    var fco = obj_of(fc)
                    ent["ffn_status"] = fco["status"]
                    if cstr(fc, "status") == "compiled" {
                        ent["ffn_compiled_path"] = fco["compiled_path"]
                        ent["ffn_compiled_size_bytes"] = fco["compiled_size_bytes"]
                    } else {
                        ent["ffn_last_error"] = fco["last_error"]
                    }
                }
            }
        }
    }
    ent["package_path"] = Json::Str(conv_rel(pkg, out_dir))
    return Json::Obj(ent);
}

host fn conv_head_entry(cfg: Json, dir: str, out_dir: str, vs: i64,
                        ve: i64, no_compile: bool, keep: bool) -> Json {
    var name = "head_v" + pad5(vs) + "-" + pad5(ve)
    var pkg = out_dir + "/" + name + ".mlpackage"
    var ent = empty_map()
    ent["vocab_start"] = Json::Num(f64(vs))
    ent["vocab_end"] = Json::Num(f64(ve))
    var r = mil_head_emit(cfg, dir, vs, ve)
    if jhas(r, "error") {
        ent["status"] = Json::Str("emit_failed")
        ent["last_error"] = Json::Str(cstr(r, "error"))
        return Json::Obj(ent);
    }
    if !mil_write_package(pkg, cstr(r, "spec"), cstr(r, "weights")) {
        ent["status"] = Json::Str("emit_failed")
        ent["last_error"] = Json::Str("package write failed: " + pkg)
        return Json::Obj(ent);
    }
    ent["name"] = Json::Str(name)
    ent["package_path"] = Json::Str(conv_rel(pkg, out_dir))
    if no_compile {
        ent["status"] = Json::Str("packaged")
    } else {
        var cr = conv_compile(pkg, name, out_dir, keep)
        var cro = obj_of(cr)
        ent["status"] = cro["status"]
        if cstr(cr, "status") == "compiled" {
            ent["compiled_path"] = cro["compiled_path"]
            ent["compiled_size_bytes"] = cro["compiled_size_bytes"]
        } else {
            ent["last_error"] = cro["last_error"]
        }
    }
    return Json::Obj(ent);
}

host det fn manifest_save(out_dir: str, manifest: Json) -> unit {
    fs_write_hex(out_dir + "/conversion_manifest.json",
                 hex_encode(json_emit(manifest)))
    return;
}

// The whole aneconvert run: embedding → layer shards → prefill shards →
// head shards → manifest. Rust keeps arg parsing and the exit code.
host fn mil_convert_run(cfg: Json, dir: str, out_dir: str,
                        opts: Json) -> Json {
    var seq = cnum(opts, "seq", 2048)
    var lps = cnum(opts, "lps", 1)
    var head_shards = cnum(opts, "head_shards", 4)
    var only_lo = cnum(opts, "only_lo", -1)
    var only_hi = cnum(opts, "only_hi", -1)
    var no_compile = cbool(opts, "no_compile")
    var keep = cbool(opts, "keep_packages")
    var prefill_chunk = cnum(opts, "prefill_chunk", 0)
    var decode_width = cnum(opts, "decode_width", 1)
    var split_ffn = cbool(opts, "split_ffn")
    var probe_stage = cstr(opts, "probe_stage")
    var probe_layer = cnum(opts, "probe_layer", 0)
    var n_layers = cnum(cfg, "n_layers", 0)
    var vocab = cnum(cfg, "vocab", 0)
    var d = cnum(cfg, "d_model", 0)

    if split_ffn { lps = 1 }
    fs_mkdir(out_dir)
    fs_mkdir(out_dir + "/logs")

    if probe_stage != "" {
        var name = "probe_l" + str(probe_layer) + "_" + probe_stage
        var pkg = out_dir + "/" + name + ".mlpackage"
        var pr = mil_probe_emit(cfg, dir, probe_layer, probe_stage)
        var pe = empty_map()
        if jhas(pr, "error") {
            pe["status"] = Json::Str("emit_failed")
            pe["error"] = jobj_of(pr, "error")
        } else {
            if mil_write_package(pkg, cstr(pr, "spec"), cstr(pr, "weights")) {
                pe["status"] = Json::Str("packaged")
                pe["package"] = Json::Str(pkg)
                if !no_compile {
                    var pcr = conv_compile(pkg, name, out_dir, keep)
                    var pcro = obj_of(pcr)
                    pe["status"] = pcro["status"]
                    if has(pcro, "compiled_path") {
                        pe["compiled_path"] = pcro["compiled_path"]
                    }
                    if has(pcro, "last_error") {
                        pe["error"] = pcro["last_error"]
                    }
                }
            } else {
                pe["status"] = Json::Str("emit_failed")
                pe["error"] = Json::Str("package write failed")
            }
        }
        var pr2 = empty_map()
        pr2["probe"] = Json::Obj(pe)
        return Json::Obj(pr2);
    }

    var ls0 = only_lo
    var le0 = only_hi
    if ls0 < 0 { ls0 = 0 }
    if le0 < 0 { le0 = n_layers }

    var model = empty_map()
    model["architecture"] = Json::Str(cstr(cfg, "arch"))
    model["total_layers"] = Json::Num(f64(n_layers))
    model["hidden_size"] = Json::Num(f64(d))
    model["vocab_size"] = Json::Num(f64(vocab))
    model["seq_len"] = Json::Num(f64(seq))
    model["quant_bits"] = Json::Num(f64(cnum(cfg, "weight_bits", 16)))
    model["compute_units"] = Json::Str("all")
    model["layers_per_shard"] = Json::Num(f64(lps))
    model["decode_width"] = Json::Num(f64(decode_width))
    model["split_ffn"] = Json::Bool(split_ffn)
    model["stateful"] = Json::Bool(true)
    model["rope_dim"] = Json::Num(f64(cnum(cfg, "rope_dim", 0)))
    model["rope_freq_base"] = Json::Num(cf64(cfg, "rope_theta", 0.0))
    model["rms_norm_eps"] = Json::Num(cf64(cfg, "eps", 0.0))
    model["native"] = Json::Bool(true)
    model["rope_freq_base_local"] = Json::Num(cf64(cfg, "rope_local_theta", 0.0))
    model["sliding_window"] = Json::Num(f64(cnum(cfg, "sliding_window", 0)))
    model["sliding_window_pattern"] = Json::Num(f64(cnum(cfg, "sliding_pattern", 0)))
    if prefill_chunk > 0 {
        model["prefill_chunk"] = Json::Num(f64(prefill_chunk))
    }

    var emb = empty_map()
    emb["path"] = Json::Str("embedding.f16")
    var eshape: [Json] = [Json::Num(f64(vocab)), Json::Num(f64(d))]
    emb["shape"] = Json::Arr(eshape)
    emb["dtype"] = Json::Str("float16")
    var er = mil_embed(cfg, dir, out_dir + "/embedding.f16")
    if jhas(er, "error") {
        emb["status"] = Json::Str("failed")
        emb["error"] = jobj_of(er, "error")
        log_err("  embedding: " + cstr(er, "error"))
    } else {
        emb["status"] = Json::Str("complete")
        emb["size_bytes"] = Json::Num(f64(vocab * d * 2))
        log_err("  embedding.f16 written")
    }

    var shared = empty_map()
    shared["embedding"] = Json::Obj(emb)
    var heads_j: [Json] = []
    shared["lm_head_shards"] = Json::Arr(heads_j)

    var srcm = empty_map()
    srcm["path"] = Json::Str(dir)

    var manifest = empty_map()
    manifest["schema_version"] = Json::Num(1.0)
    manifest["status"] = Json::Str("pending")
    manifest["source"] = Json::Obj(srcm)
    manifest["model"] = Json::Obj(model)
    manifest["shared"] = Json::Obj(shared)
    var shards: [Json] = []
    manifest["shards"] = Json::Arr(shards)

    var ls = ls0
    while ls < le0 {
        var le = ls + lps
        if le > le0 { le = le0 }
        var ent = conv_layer_entry(cfg, dir, out_dir, ls, le, seq,
                                   decode_width, false, no_compile,
                                   keep, split_ffn)
        push(shards, ent)
        manifest["shards"] = Json::Arr(shards)
        manifest_save(out_dir, Json::Obj(manifest))
        ls = le
    }

    if prefill_chunk > 0 {
        var pentries: [Json] = []
        var pls = ls0
        while pls < le0 {
            var ple = pls + lps
            if ple > le0 { ple = le0 }
            var pent = conv_layer_entry(cfg, dir, out_dir, pls, ple, seq,
                                        prefill_chunk, true, no_compile,
                                        keep, split_ffn)
            push(pentries, pent)
            manifest["shards_prefill"] = Json::Arr(pentries)
            manifest_save(out_dir, Json::Obj(manifest))
            pls = ple
        }
    }

    var width = vocab
    if head_shards > 0 {
        width = (vocab + head_shards - 1) / head_shards
    }
    var vs = 0
    if head_shards == 0 { vs = vocab }
    var heads2: [Json] = []
    while vs < vocab {
        var ve = vs + width
        if ve > vocab { ve = vocab }
        var he = conv_head_entry(cfg, dir, out_dir, vs, ve, no_compile, keep)
        push(heads2, he)
        var sh2 = obj_of(manifest["shared"])
        sh2["lm_head_shards"] = Json::Arr(heads2)
        manifest["shared"] = Json::Obj(sh2)
        manifest_save(out_dir, Json::Obj(manifest))
        vs = ve
    }

    // tokenizer files ride along — runtimes load them from the artifact dir
    var toks: [str] = ["tokenizer.json", "tokenizer_config.json",
                       "tokenizer.model", "special_tokens_map.json",
                       "added_tokens.json"]
    for i in 0..len(toks) {
        var t = toks[i]
        var hexs = fs_read_hex(dir + "/" + t)
        if hexs != "" {
            fs_write_hex(out_dir + "/" + t, hexs)
        }
    }

    var all_ok = true
    for i in 0..len(shards) {
        var st2 = cstr(shards[i], "status")
        if st2 != "compiled" && st2 != "packaged" { all_ok = false }
    }
    if all_ok {
        manifest["status"] = Json::Str("complete")
    } else {
        manifest["status"] = Json::Str("partial")
    }
    manifest_save(out_dir, Json::Obj(manifest))
    var done = empty_map()
    done["all_ok"] = Json::Bool(all_ok)
    done["manifest"] = Json::Obj(manifest)
    return Json::Obj(done);
}
