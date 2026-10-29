cap proc
import "sys.cy"
import "strlib.cy"

// MIL graph spec builder — Coyote owns graph construction; Rust lowers
// this Json spec through mil_spec and serializes the model.

host det fn empty_vec() -> [Json] {
    var v: [Json] = []
    return v;
}

host det fn empty_map() -> {str: Json} {
    var m: {str: Json} = {}
    return m;
}

host det fn jarr0() -> Json {
    var a: [Json] = []
    return Json::Arr(a);
}

host det fn feature(name: str, shape: [i64], dtype: str, is_state: bool) -> Json {
    var dims: [Json] = []
    for i in 0..len(shape) {
        push(dims, Json::Num(f64(shape[i])))
    }
    var m: {str: Json} = {}
    m["name"] = Json::Str(name)
    m["shape"] = Json::Arr(dims)
    m["dtype"] = Json::Str(dtype)
    m["state"] = Json::Bool(is_state)
    return Json::Obj(m);
}

host det fn spec_new() -> Json {
    var m: {str: Json} = {}
    m["ops"] = jarr0()
    m["inputs"] = jarr0()
    m["outputs"] = jarr0()
    m["states"] = jarr0()
    m["fn_inputs"] = jarr0()
    var meta: {str: Json} = {}
    meta["spec_version"] = Json::Num(9.0)
    meta["opset"] = Json::Str("CoreML9")
    m["meta"] = Json::Obj(meta)
    return Json::Obj(m);
}

host det fn arr_of(j: Json) -> [Json] {
    match j {
        Json::Arr(a) => { return a; }
        _ => { return empty_vec(); }
    }
}

host det fn obj_of(j: Json) -> {str: Json} {
    match j {
        Json::Obj(m) => { return m; }
        _ => { return empty_map(); }
    }
}

host det fn spec_push_feature(spec: Json, key: str, name: str,
                              shape: [i64], dtype: str, is_state: bool) -> Json {
    var m = obj_of(spec)
    var v = arr_of(m[key])
    push(v, feature(name, shape, dtype, is_state))
    m[key] = Json::Arr(v)
    return Json::Obj(m);
}

host det fn spec_add_input(spec: Json, name: str, shape: [i64], dtype: str) -> Json {
    return spec_push_feature(spec, "inputs", name, shape, dtype, false);
}

host det fn spec_add_output(spec: Json, name: str, shape: [i64], dtype: str) -> Json {
    return spec_push_feature(spec, "outputs", name, shape, dtype, false);
}

host det fn spec_add_state(spec: Json, name: str, shape: [i64], dtype: str) -> Json {
    return spec_push_feature(spec, "states", name, shape, dtype, true);
}

host det fn spec_add_fn_input(spec: Json, name: str, shape: [i64], dtype: str) -> Json {
    return spec_push_feature(spec, "fn_inputs", name, shape, dtype, false);
}

host det fn spec_meta(spec: Json, creator: str, desc: str) -> Json {
    var m = obj_of(spec)
    var meta = obj_of(m["meta"])
    meta["creator"] = Json::Str(creator)
    meta["description"] = Json::Str(desc)
    m["meta"] = Json::Obj(meta)
    return Json::Obj(m);
}

host det fn shape_arr(shape: [i64]) -> Json {
    var a: [Json] = []
    for i in 0..len(shape) {
        push(a, Json::Num(f64(shape[i])))
    }
    return Json::Arr(a);
}

host det fn i32v(v: i64) -> Json {
    var m: {str: Json} = {}
    m["i32"] = Json::Num(f64(v))
    return Json::Obj(m);
}

host det fn i32sv(v: [i64]) -> Json {
    var m: {str: Json} = {}
    m["i32s"] = shape_arr(v)
    return Json::Obj(m);
}

host det fn i64s(v: [i64]) -> Json {
    var m: {str: Json} = {}
    m["i64s"] = shape_arr(v)
    return Json::Obj(m);
}

host det fn i64v(v: i64) -> Json {
    var m: {str: Json} = {}
    m["i64s"] = shape_arr([v])
    return Json::Obj(m);
}

host det fn f64v(v: f64) -> Json {
    var m: {str: Json} = {}
    m["f32"] = Json::Num(v)
    return Json::Obj(m);
}

host det fn boolsv(v: [bool]) -> Json {
    var a: [Json] = []
    for i in 0..len(v) {
        push(a, Json::Bool(v[i]))
    }
    var m: {str: Json} = {}
    m["bools"] = Json::Arr(a)
    return Json::Obj(m);
}

host det fn strv(s: str) -> Json {
    var m: {str: Json} = {}
    m["s"] = Json::Str(s)
    return Json::Obj(m);
}

host det fn blobv(off: i64, shape: [i64], dtype: str) -> Json {
    var b: {str: Json} = {}
    b["off"] = Json::Num(f64(off))
    b["shape"] = shape_arr(shape)
    b["dtype"] = Json::Str(dtype)
    var m: {str: Json} = {}
    m["blob"] = Json::Obj(b)
    return Json::Obj(m);
}

host det fn f16s_hex(data_hex: str, shape: [i64]) -> Json {
    var b: {str: Json} = {}
    b["hex"] = Json::Str(data_hex)
    b["shape"] = shape_arr(shape)
    b["dtype"] = Json::Str("fp16")
    var m: {str: Json} = {}
    m["f16s"] = Json::Obj(b)
    return Json::Obj(m);
}

host det fn bin_var(name: str) -> Json {
    return Json::Str(name);
}

host det fn bin_const(data_hex: str, dtype: str, shape: [i64]) -> Json {
    var b: {str: Json} = {}
    b["hex"] = Json::Str(data_hex)
    b["shape"] = shape_arr(shape)
    b["dtype"] = Json::Str(dtype)
    var c: {str: Json} = {}
    c["bytes"] = Json::Obj(b)
    var m: {str: Json} = {}
    m["c"] = Json::Obj(c)
    return Json::Obj(m);
}

host det fn bin_many(binds: [Json]) -> Json {
    return Json::Arr(binds);
}



host det fn spec_op(spec: Json, ty: str, inputs: {str: Json}, out_names: [str],
                    out_shapes: [[i64]], out_dtypes: [str], attrs: {str: Json}) -> Json {
    var m = obj_of(spec)
    var ops = arr_of(m["ops"])
    var o: {str: Json} = {}
    o["ty"] = Json::Str(ty)
    var im: {str: Json} = {}
    var ks = keys(inputs)
    for i in 0..len(ks) {
        im[ks[i]] = inputs[ks[i]]
    }
    o["in"] = Json::Obj(im)
    var outs: [Json] = []
    for i in 0..len(out_names) {
        var ot: {str: Json} = {}
        ot["name"] = Json::Str(out_names[i])
        ot["shape"] = shape_arr(out_shapes[i])
        ot["dtype"] = Json::Str(out_dtypes[i])
        push(outs, Json::Obj(ot))
    }
    o["out"] = Json::Arr(outs)
    if len(keys(attrs)) > 0 {
        o["attr"] = Json::Obj(attrs)
    }
    push(ops, Json::Obj(o))
    m["ops"] = Json::Arr(ops)
    return Json::Obj(m);
}

host det fn spec_op1(spec: Json, ty: str, in_name: str, in_bind: Json, out_name: str,
                     out_shape: [i64], out_dtype: str, attrs: {str: Json}) -> Json {
    var inputs: {str: Json} = {}
    inputs[in_name] = in_bind
    return spec_op(spec, ty, inputs, [out_name], [out_shape], [out_dtype], attrs);
}

// Emit the model spec through the Rust mil_spec bridge.
host fn mil_build(spec: Json) -> Json {
    return mil_emit(spec);
}

// Minimal single-op spec (add(x, const)) — the emit-path smoke shape.
host det fn mil_demo_spec() -> Json {
    var spec = spec_new()
    spec = spec_meta(spec, "badapple", "cy-mil demo")
    spec = spec_add_input(spec, "x", [1, 4], "fp16")
    spec = spec_add_output(spec, "y", [1, 4], "fp16")
    spec = spec_add_fn_input(spec, "x", [1, 4], "fp16")
    var ins: {str: Json} = {}
    ins["x"] = bin_var("x")
    ins["y"] = bin_const("00003c0000003c00", "fp16", [1, 4])
    var attrs: {str: Json} = {}
    return spec_op(spec, "add", ins, ["y"], [[1, 4]], ["fp16"], attrs);
}

// --- Block sugar: the same op expansion mil_spec::Block performs ---

// Fresh symbolic name — deterministic per spec ("{hint}_{n}").
// Returns {"spec": new_spec, "name": "hint_n"}.
host det fn spec_fresh(spec: Json, hint: str) -> Json {
    var m = obj_of(spec)
    var n = 0
    if has(m, "_fresh") {
        match m["_fresh"] {
            Json::Num(x) => { n = i64(x) }
            _ => { n = 0 }
        }
    }
    n = n + 1
    m["_fresh"] = Json::Num(f64(n))
    var r: {str: Json} = {}
    r["spec"] = Json::Obj(m)
    r["name"] = Json::Str(hint + "_" + str(n))
    return Json::Obj(r);
}

host det fn jstr_of(j: Json, k: str) -> str {
    match obj_of(j)[k] {
        Json::Str(s) => { return s; }
        _ => { return ""; }
    }
}

host det fn jobj_of(j: Json, k: str) -> Json {
    if has(obj_of(j), k) {
        return obj_of(j)[k];
    }
    return Json::Null;
}

host det fn i32_scalar(v: i64) -> Json {
    var m: {str: Json} = {}
    m["i32_scalar"] = Json::Num(f64(v))
    return Json::Obj(m);
}

host det fn bool1(v: bool) -> Json {
    var m: {str: Json} = {}
    m["bool1"] = Json::Bool(v)
    return Json::Obj(m);
}

host det fn f16_scalar(v: f64) -> Json {
    var m: {str: Json} = {}
    m["f16"] = Json::Num(v)
    return Json::Obj(m);
}

host det fn f32_scalar(v: f64) -> Json {
    var m: {str: Json} = {}
    m["f32"] = Json::Num(v)
    return Json::Obj(m);
}

// const op emitting a rank-0 or vector value.
host det fn empty_i64() -> [i64] {
    var v: [i64] = []
    return v;
}

host det fn konst(spec: Json, name: str, val: Json, shape: [i64], dtype: str) -> Json {
    var ins = empty_map()
    var attrs = empty_map()
    attrs["val"] = val
    return spec_op(spec, "const", ins, [name], [shape], [dtype], attrs);
}

host det fn konst_i32(spec: Json, name: str, vs: [i64]) -> Json {
    return konst(spec, name, i32sv(vs), [len(vs)], "int32");
}

host det fn konst_scalar_i32(spec: Json, name: str, v: i64) -> Json {
    return konst(spec, name, i32_scalar(v), empty_i64(), "int32");
}

host det fn konst_f16(spec: Json, name: str, v: f64) -> Json {
    return konst(spec, name, f16_scalar(v), empty_i64(), "fp16");
}

host det fn konst_f32(spec: Json, name: str, v: f64) -> Json {
    return konst(spec, name, f32_scalar(v), empty_i64(), "fp32");
}

host det fn konst_bool(spec: Json, name: str, v: bool) -> Json {
    return konst(spec, name, bool1(v), empty_i64(), "bool");
}

host det fn konst_str(spec: Json, name: str, v: str) -> Json {
    return konst(spec, name, strv(v), empty_i64(), "str");
}

// Two-input elementwise op with fp16 output.
host det fn ew2(spec: Json, ty: str, a: str, b: str, shape: [i64], name: str) -> Json {
    var ins = empty_map()
    ins["x"] = bin_var(a)
    ins["y"] = bin_var(b)
    return spec_op(spec, ty, ins, [name], [shape], ["fp16"], empty_map());
}

host det fn mul2(spec: Json, a: str, b: str, shape: [i64], name: str) -> Json {
    return ew2(spec, "mul", a, b, shape, name);
}

host det fn add2(spec: Json, a: str, b: str, shape: [i64], name: str) -> Json {
    return ew2(spec, "add", a, b, shape, name);
}

host det fn sub2(spec: Json, a: str, b: str, shape: [i64], name: str) -> Json {
    return ew2(spec, "sub", a, b, shape, name);
}

// Pair-returning helper: {"spec","name"} like Block fns returning String.
host det fn pair(spec: Json, name: str) -> Json {
    var r: {str: Json} = {}
    r["spec"] = spec
    r["name"] = Json::Str(name)
    return Json::Obj(r);
}

host det fn reshape2(spec: Json, x: str, shape: [i64], name: str) -> Json {
    var fr = spec_fresh(spec, "shape")
    var spec = jobj_of(fr, "spec")
    var sname = jstr_of(fr, "name")
    spec = konst_i32(spec, sname, shape)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["shape"] = bin_var(sname)
    spec = spec_op(spec, "reshape", ins, [name], [shape], ["fp16"], empty_map())
    return pair(spec, name);
}

host det fn matmul2(spec: Json, x: str, y: str, ty: bool, out_shape: [i64], name: str) -> Json {
    var fr = spec_fresh(spec, "tx")
    var spec = jobj_of(fr, "spec")
    var txn = jstr_of(fr, "name")
    spec = konst_bool(spec, txn, false)
    fr = spec_fresh(spec, "ty")
    spec = jobj_of(fr, "spec")
    var tyn = jstr_of(fr, "name")
    spec = konst_bool(spec, tyn, ty)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["y"] = bin_var(y)
    ins["transpose_x"] = bin_var(txn)
    ins["transpose_y"] = bin_var(tyn)
    spec = spec_op(spec, "matmul", ins, [name], [out_shape], ["fp16"], empty_map())
    return pair(spec, name);
}

// transpose by perm (perm const emitted via fresh).
host det fn transpose2(spec: Json, x: str, perm: [i64], out_shape: [i64], name: str) -> Json {
    var spec = spec
    var fr = spec_fresh(spec, "perm")
    spec = jobj_of(fr, "spec")
    var pn = jstr_of(fr, "name")
    spec = konst_i32(spec, pn, perm)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["perm"] = bin_var(pn)
    spec = spec_op(spec, "transpose", ins, [name], [out_shape], ["fp16"], empty_map())
    return pair(spec, name);
}

// expand_dims on axes.
host det fn expand_dims2(spec: Json, x: str, axes: [i64], out_shape: [i64], name: str) -> Json {
    var spec = spec
    var fr = spec_fresh(spec, "axes")
    spec = jobj_of(fr, "spec")
    var an = jstr_of(fr, "name")
    spec = konst_i32(spec, an, axes)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["axes"] = bin_var(an)
    spec = spec_op(spec, "expand_dims", ins, [name], [out_shape], ["fp16"], empty_map())
    return pair(spec, name);
}

// softmax on axis; fp32 selects the output dtype.
host det fn softmax2(spec: Json, x: str, axis: i64, out_shape: [i64], name: str, fp32: bool) -> Json {
    var spec = spec
    var fr = spec_fresh(spec, "axis")
    spec = jobj_of(fr, "spec")
    var an = jstr_of(fr, "name")
    spec = konst_scalar_i32(spec, an, axis)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["axis"] = bin_var(an)
    var dt = "fp16"
    if fp32 { dt = "fp32" }
    spec = spec_op(spec, "softmax", ins, [name], [out_shape], [dt], empty_map())
    return pair(spec, name);
}

// generic single-output op — the o1 pattern.
host det fn o1n(spec: Json, ty: str, ins: {str: Json}, name: str,
                out_shape: [i64], out_dtype: str) -> Json {
    var spec = spec
    spec = spec_op(spec, ty, ins, [name], [out_shape], [out_dtype], empty_map())
    return pair(spec, name);
}

// const binding of a value spec (non-bytes) — e.g. {"c":{"s":"TANH_APPROXIMATION"}}
host det fn bin_const_val(v: Json) -> Json {
    var m: {str: Json} = {}
    m["c"] = v
    return Json::Obj(m);
}

// conv1x1 (s=1) or conv1x1_s — emits the const prelude then the conv.
host det fn conv1x1_s(spec: Json, x: str, w: str, bias: str, has_bias: bool,
                      cout: i64, s: i64, name: str) -> Json {
    var fr = spec_fresh(spec, "stride")
    var spec = jobj_of(fr, "spec")
    var strides = jstr_of(fr, "name")
    spec = konst_i32(spec, strides, [1, 1])
    fr = spec_fresh(spec, "padtype")
    spec = jobj_of(fr, "spec")
    var padtype = jstr_of(fr, "name")
    spec = konst_str(spec, padtype, "valid")
    fr = spec_fresh(spec, "pad")
    spec = jobj_of(fr, "spec")
    var pads = jstr_of(fr, "name")
    spec = konst_i32(spec, pads, [0, 0, 0, 0])
    fr = spec_fresh(spec, "dil")
    spec = jobj_of(fr, "spec")
    var dil = jstr_of(fr, "name")
    spec = konst_i32(spec, dil, [1, 1])
    fr = spec_fresh(spec, "grp")
    spec = jobj_of(fr, "spec")
    var grp = jstr_of(fr, "name")
    spec = konst_scalar_i32(spec, grp, 1)
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["weight"] = bin_var(w)
    ins["strides"] = bin_var(strides)
    ins["pad_type"] = bin_var(padtype)
    ins["pad"] = bin_var(pads)
    ins["dilations"] = bin_var(dil)
    ins["groups"] = bin_var(grp)
    if has_bias {
        ins["bias"] = bin_var(bias)
    }
    spec = spec_op(spec, "conv", ins, [name], [[1, cout, s, 1]], ["fp16"], empty_map())
    return pair(spec, name);
}

// linear_w parity shape: reshape→reshape→matmul→reshape, exactly the
// Rust helper's op sequence (fresh names align: shape_1, shape_2, tx_3, ty_4).
host det fn mil_linear_spec(d: i64, cout: i64, s: i64) -> Json {
    var spec = spec_new()
    spec = spec_add_input(spec, "x", [1, d, 1, s], "fp16")
    spec = spec_add_input(spec, "w", [cout, d, 1, 1], "fp16")
    spec = spec_add_output(spec, "lin_b", [1, cout, 1, s], "fp16")
    spec = spec_add_fn_input(spec, "x", [1, d, 1, s], "fp16")
    spec = spec_add_fn_input(spec, "w", [cout, d, 1, 1], "fp16")
    var r = reshape2(spec, "x", [1, 1, d, s], "lin_xr")
    spec = jobj_of(r, "spec")
    r = reshape2(spec, "w", [1, 1, cout, d], "lin_wr")
    spec = jobj_of(r, "spec")
    r = matmul2(spec, "lin_wr", "lin_xr", false, [1, 1, cout, s], "lin")
    spec = jobj_of(r, "spec")
    r = reshape2(spec, "lin", [1, cout, 1, s], "lin_b")
    spec = jobj_of(r, "spec")
    var m = obj_of(spec)
    var bo: [Json] = []
    push(bo, Json::Str("lin_b"))
    m["block_outputs"] = Json::Arr(bo)
    return Json::Obj(m);
}

// rms_norm_big — hardened RMSNorm for massive activations.
// Mirrors src/bin/badapple-aneconvert.rs:rms_norm_big exactly, including
// f32 const math (f32_sqrt/f32_round keep const bits identical).
host det fn mil_rms_norm_spec(d: i64, eps: f64, s: i64, pfx: str) -> Json {
    var spec = spec_new()
    spec = spec_add_input(spec, "x", [1, d, 1, s], "fp16")
    spec = spec_add_input(spec, "w", [1, d, 1, 1], "fp16")
    var shape4: [i64] = [1, d, 1, s]
    var mshape: [i64] = [1, 1, 1, s]
    // f32 const math — identical rounding to the Rust emitter.
    var k = f32_sqrt(f64(d))
    var inv = f32_round(1.0 / f32_round(8.0 * k))
    var e = f32_round(eps / f32_round(64.0 * f32_round(k * k)))
    spec = konst_f16(spec, pfx + "_bigik", inv)
    spec = mul2(spec, "x", pfx + "_bigik", shape4, pfx + "_bigxs")
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
    spec = mul2(spec, pfx + "_xn", "w", shape4, pfx + "_out")
    spec = spec_add_output(spec, pfx + "_out", shape4, "fp16")
    var m = obj_of(spec)
    var bo: [Json] = []
    push(bo, Json::Str(pfx + "_out"))
    m["block_outputs"] = Json::Arr(bo)
    return Json::Obj(m);
}

// --- weight table + layer-shard primitives ---

// Declare a weight blob: appends {name,dtype,shape,hex} to spec.weights and
// emits the konst_blob const op. The lowering assigns the 64-byte-aligned
// offset in spec order — matches the Rust `blob()` closure's wb.put order.
host det fn spec_weight(spec: Json, name: str, dtype: str, shape: [i64], hexs: str) -> Json {
    var spec = spec
    var m = obj_of(spec)
    var ws = arr_of(jobj_of(spec, "weights"))
    var w: {str: Json} = {}
    w["name"] = Json::Str(name)
    w["dtype"] = Json::Str(dtype)
    w["shape"] = shape_arr(shape)
    w["hex"] = Json::Str(hexs)
    push(ws, Json::Obj(w))
    m["weights"] = Json::Arr(ws)
    // const op whose val is a blob_named ref — offset resolved at lowering
    var ins = empty_map()
    var attrs = empty_map()
    var bn: {str: Json} = {}
    bn["name"] = Json::Str(name)
    bn["shape"] = shape_arr(shape)
    bn["dtype"] = Json::Str(dtype)
    var v: {str: Json} = {}
    v["blob_named"] = Json::Obj(bn)
    attrs["val"] = Json::Obj(v)
    var spec2 = Json::Obj(m)
    return spec_op(spec2, "const", ins, [name], [shape], [dtype], attrs);
}

// read_state — blk.read_state(state, shape, name)
host det fn read_state2(spec: Json, st: str, shape: [i64], name: str) -> Json {
    var spec = spec
    var ins = empty_map()
    ins["input"] = bin_var(st)
    var attrs = empty_map()
    attrs["name"] = strv(name)
    spec = spec_op(spec, "read_state", ins, [name], [shape], ["fp16"], attrs)
    return pair(spec, name);
}

// write_state — blk.write_state(state, value); no outputs.
host det fn write_state2(spec: Json, st: str, value: str) -> Json {
    var spec = spec
    var ins = empty_map()
    ins["input"] = bin_var(st)
    ins["data"] = bin_var(value)
    var attrs = empty_map()
    attrs["name"] = strv(st + "_write_state")
    var outs: [str] = []
    var oshapes: [[i64]] = []
    var odts: [str] = []
    return spec_op(spec, "write_state", ins, outs, oshapes, odts, attrs);
}

// slice — blk.slice: six-const prelude + slice_by_index.
host det fn slice2(spec: Json, x: str, begin: [i64], endx: [i64],
                   out_shape: [i64], name: str) -> Json {
    var spec = spec
    var fr = spec_fresh(spec, "begin")
    var spec = jobj_of(fr, "spec")
    var b0 = jstr_of(fr, "name")
    spec = konst_i32(spec, b0, begin)
    fr = spec_fresh(spec, "end")
    spec = jobj_of(fr, "spec")
    var e0 = jstr_of(fr, "name")
    spec = konst_i32(spec, e0, endx)
    fr = spec_fresh(spec, "stride")
    spec = jobj_of(fr, "spec")
    var st = jstr_of(fr, "name")
    spec = konst_i32(spec, st, i64_ones(len(begin)))
    fr = spec_fresh(spec, "bmask")
    spec = jobj_of(fr, "spec")
    var bm = jstr_of(fr, "name")
    spec = konst(spec, bm, boolsv(bool_n(len(begin), false)), [len(begin)], "bool")
    fr = spec_fresh(spec, "emask")
    spec = jobj_of(fr, "spec")
    var em = jstr_of(fr, "name")
    spec = konst(spec, em, boolsv(bool_n(len(begin), false)), [len(begin)], "bool")
    fr = spec_fresh(spec, "smask")
    spec = jobj_of(fr, "spec")
    var sm = jstr_of(fr, "name")
    spec = konst(spec, sm, boolsv(bool_n(len(begin), false)), [len(begin)], "bool")
    var ins = empty_map()
    ins["x"] = bin_var(x)
    ins["begin"] = bin_var(b0)
    ins["end"] = bin_var(e0)
    ins["stride"] = bin_var(st)
    ins["begin_mask"] = bin_var(bm)
    ins["end_mask"] = bin_var(em)
    ins["squeeze_mask"] = bin_var(sm)
    spec = spec_op(spec, "slice_by_index", ins, [name], [out_shape], ["fp16"], empty_map())
    return pair(spec, name);
}

host det fn i64_ones(n: i64) -> [i64] {
    var v: [i64] = []
    for i in 0..n {
        push(v, 1)
    }
    return v;
}

host det fn bool_n(n: i64, v: bool) -> [bool] {
    var a: [bool] = []
    for i in 0..n {
        push(a, v)
    }
    return a;
}

// concat — blk.concat(xs, axis, out_shape, name)
host det fn concat2(spec: Json, xs: [str], axis: i64, out_shape: [i64], name: str) -> Json {
    var spec = spec
    var fr = spec_fresh(spec, "axis")
    var spec = jobj_of(fr, "spec")
    var ax = jstr_of(fr, "name")
    spec = konst_scalar_i32(spec, ax, axis)
    fr = spec_fresh(spec, "ilv")
    spec = jobj_of(fr, "spec")
    var il = jstr_of(fr, "name")
    spec = konst_bool(spec, il, false)
    var vs: [Json] = []
    for i in 0..len(xs) {
        push(vs, bin_var(xs[i]))
    }
    var ins = empty_map()
    ins["values"] = Json::Arr(vs)
    ins["axis"] = bin_var(ax)
    ins["interleave"] = bin_var(il)
    spec = spec_op(spec, "concat", ins, [name], [out_shape], ["fp16"], empty_map())
    return pair(spec, name);
}

// fn_input entry carrying the State type (KV cache etc).
host det fn spec_add_fn_state(spec: Json, name: str, shape: [i64], dtype: str) -> Json {
    var m = obj_of(spec)
    var ins = arr_of(m["fn_inputs"])
    push(ins, feature(name, shape, dtype, true))
    m["fn_inputs"] = Json::Arr(ins)
    return Json::Obj(m);
}

// weights-table entry only (no op) — for blob consts emitted separately.
host det fn spec_weight_only(spec: Json, name: str, dtype: str,
                             shape: [i64], hexs: str) -> Json {
    var spec = spec
    var m = obj_of(spec)
    var ws = arr_of(jobj_of(spec, "weights"))
    var w: {str: Json} = {}
    w["name"] = Json::Str(name)
    w["dtype"] = Json::Str(dtype)
    w["shape"] = shape_arr(shape)
    w["hex"] = Json::Str(hexs)
    push(ws, Json::Obj(w))
    m["weights"] = Json::Arr(ws)
    return Json::Obj(m);
}

// konst_blob for a weights-table entry (offset resolved at lowering).
host det fn blob_const(spec: Json, name: str, dtype: str, shape: [i64]) -> Json {
    var spec = spec
    var ins = empty_map()
    var attrs = empty_map()
    var bn: {str: Json} = {}
    bn["name"] = Json::Str(name)
    bn["shape"] = shape_arr(shape)
    bn["dtype"] = Json::Str(dtype)
    var v: {str: Json} = {}
    v["blob_named"] = Json::Obj(bn)
    attrs["val"] = Json::Obj(v)
    return spec_op(spec, "const", ins, [name], [shape], [dtype], attrs);
}

// weight_const for weight_bits==8: q8 data + fp16 row scales +
// constexpr_blockwise_shift_scale dequant op.
host det fn spec_weight_q8(spec: Json, name: str, shape: [i64],
                           q_hex: str, s_hex: str) -> Json {
    var spec = spec
    var sshape: [i64] = [shape[0]]
    for i in 1..len(shape) {
        push(sshape, 1)
    }
    spec = spec_weight_only(spec, name + "_q8", "int8", shape, q_hex)
    spec = spec_weight_only(spec, name + "_scale", "fp16", sshape, s_hex)
    spec = blob_const(spec, name + "_q8", "int8", shape)
    spec = blob_const(spec, name + "_scale", "fp16", sshape)
    var ins = empty_map()
    ins["data"] = bin_var(name + "_q8")
    ins["scale"] = bin_var(name + "_scale")
    spec = spec_op(spec, "constexpr_blockwise_shift_scale", ins,
                   [name], [shape], ["fp16"], empty_map())
    return pair(spec, name);
}

// weight_const for weight_bits==4: packed uint4 + fp16 LUT +
// constexpr_lut_to_dense dequant op.
host det fn spec_weight_q4(spec: Json, name: str, shape: [i64],
                           q_hex: str, lut_hex: str) -> Json {
    var spec = spec
    var lut_shape: [i64] = [shape[0]]
    for i in 1..len(shape) {
        push(lut_shape, 1)
    }
    push(lut_shape, 16)
    push(lut_shape, 1)
    spec = spec_weight_only(spec, name + "_q4", "uint4", shape, q_hex)
    spec = spec_weight_only(spec, name + "_lut", "fp16", lut_shape, lut_hex)
    spec = blob_const(spec, name + "_q4", "uint4", shape)
    spec = blob_const(spec, name + "_lut", "fp16", lut_shape)
    var ins = empty_map()
    ins["indices"] = bin_var(name + "_q4")
    ins["lut"] = bin_var(name + "_lut")
    spec = spec_op(spec, "constexpr_lut_to_dense", ins,
                   [name], [shape], ["fp16"], empty_map())
    return pair(spec, name);
}
