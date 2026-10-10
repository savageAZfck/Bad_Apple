// kvpack s1 n8 + shard-style provenance: updates come from
// slice+reshape of a fused "qkv" input instead of raw inputs.
use bad_apple::mil_spec::*;
use std::path::PathBuf;

fn f16(shape: &[i64]) -> ValueType {
    ValueType::Tensor(TensorType::f16(shape))
}

fn main() -> Result<(), String> {
    let out_dir = PathBuf::from(std::env::args().nth(1).expect("out dir"));
    let kv_shape = vec![2, 8, 2048, 128];
    let mut blk = Block::new();

    let one = blk.konst_i32("one", &[1]);
    let p1 = blk.o1(
        "add",
        vec![("x".into(), bind("pos").1), ("y".into(), bind(&one).1)],
        "p1",
        ValueType::Tensor(TensorType::i32v(1)),
    );

    let mk = |blk: &mut Block, pfx: &str, row0: i32, row1: i32| -> (String, String) {
        let b0 = blk.konst_i32(&format!("{pfx}_b0"), &[row0]);
        let b1 = blk.konst_i32(&format!("{pfx}_b1"), &[0]);
        let b3 = blk.konst_i32(&format!("{pfx}_b3"), &[0]);
        let ax = blk.konst_scalar_i32(&format!("{pfx}_axb"), 0);
        let il = blk.konst_bool(&format!("{pfx}_ilb"), false);
        let beg = blk.o1(
            "concat",
            vec![
                ("values".into(), bind_many(&[&b0, &b1, "pos", &b3])),
                ("axis".into(), bind(&ax).1),
                ("interleave".into(), bind(&il).1),
            ],
            &format!("{pfx}_beg"),
            ValueType::Tensor(TensorType::i32v(4)),
        );
        let e0 = blk.konst_i32(&format!("{pfx}_e0"), &[row1]);
        let e1 = blk.konst_i32(&format!("{pfx}_e1"), &[8]);
        let e3 = blk.konst_i32(&format!("{pfx}_e3"), &[128]);
        let axe = blk.konst_scalar_i32(&format!("{pfx}_axe"), 0);
        let ile = blk.konst_bool(&format!("{pfx}_ile"), false);
        let end = blk.o1(
            "concat",
            vec![
                ("values".into(), bind_many(&[&e0, &e1, &p1, &e3])),
                ("axis".into(), bind(&axe).1),
                ("interleave".into(), bind(&ile).1),
            ],
            &format!("{pfx}_end"),
            ValueType::Tensor(TensorType::i32v(4)),
        );
        (beg, end)
    };
    let (kb, ke) = mk(&mut blk, "k", 0, 1);
    let (vb, ve) = mk(&mut blk, "v", 1, 2);

    let masks = |blk: &mut Block, pfx: &str| -> (String, String, String, String) {
        let st = blk.konst_i32(&format!("{pfx}_st"), &[1, 1, 1, 1]);
        let mk = |blk: &mut Block, n: &str| -> String {
            blk.op(
                "const",
                vec![],
                vec![(
                    n,
                    ValueType::Tensor(TensorType {
                        dtype: DType::Bool,
                        shape: vec![4],
                    }),
                )],
                vec![("val".into(), Value::bools(&[false, false, false, false]))],
            )[0]
            .clone()
        };
        let bm = mk(blk, &format!("{pfx}_bm"));
        let em = mk(blk, &format!("{pfx}_em"));
        let sm = mk(blk, &format!("{pfx}_sm"));
        (st, bm, em, sm)
    };
    let (kst, kbm, kem_, ksm) = masks(&mut blk, "km");
    let (vst, vbm, vem, vsm) = masks(&mut blk, "vm");

    // shard-style provenance: qkv [1,2048,1,1] -> slices -> reshape [1,8,1,128]
    let k_sl = blk.slice(
        "qkv",
        &[0, 0, 0, 0],
        &[1, 1024, 1, 1],
        &[1, 1024, 1, 1],
        "k_sl",
    );
    let nk_t = blk.reshape(&k_sl, &[1, 8, 1, 128], "nk_t");
    let v_sl = blk.slice(
        "qkv",
        &[0, 1024, 0, 0],
        &[1, 2048, 1, 1],
        &[1, 1024, 1, 1],
        "v_sl",
    );
    let nv_t = blk.reshape(&v_sl, &[1, 8, 1, 128], "nv_t");

    let kv0 = blk.read_state("kv", &kv_shape, "kv0");
    let kv1 = blk.o1(
        "slice_update",
        vec![
            ("x".into(), bind(&kv0).1),
            ("update".into(), bind(&nk_t).1),
            ("begin".into(), bind(&kb).1),
            ("end".into(), bind(&ke).1),
            ("stride".into(), bind(&kst).1),
            ("begin_mask".into(), bind(&kbm).1),
            ("end_mask".into(), bind(&kem_).1),
            ("squeeze_mask".into(), bind(&ksm).1),
        ],
        "kv1",
        f16(&kv_shape),
    );
    let kv2 = blk.o1(
        "slice_update",
        vec![
            ("x".into(), bind(&kv1).1),
            ("update".into(), bind(&nv_t).1),
            ("begin".into(), bind(&vb).1),
            ("end".into(), bind(&ve).1),
            ("stride".into(), bind(&vst).1),
            ("begin_mask".into(), bind(&vbm).1),
            ("end_mask".into(), bind(&vem).1),
            ("squeeze_mask".into(), bind(&vsm).1),
        ],
        "kv2",
        f16(&kv_shape),
    );
    blk.write_state("kv", &kv2);

    let k_full = blk.slice(
        &kv2,
        &[0, 0, 0, 0],
        &[1, 8, 2048, 128],
        &[1, 8, 2048, 128],
        "k_full",
    );
    let scores = blk.matmul("q", &k_full, true, &[1, 8, 4, 2048], "scores");
    // big-FFN probe: state graph + a wide conv (gemma gu shape)
    let go = blk.conv1x1("xq", "wbig", None, 8192, "go");
    blk.outputs = vec![scores, k_full, kv2.clone(), go.clone()];

    let states = vec![Feature {
        name: "kv".into(),
        shape: kv_shape.clone(),
        dtype: DType::Fp16,
        is_state: true,
    }];
    let outputs = vec![
        Feature {
            name: "scores".into(),
            shape: vec![1, 8, 4, 2048],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "k_full".into(),
            shape: vec![1, 8, 2048, 128],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "kv2".into(),
            shape: kv_shape.clone(),
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "go".into(),
            shape: vec![1, 8192, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
    ];
    let inputs = vec![
        Feature {
            name: "qkv".into(),
            shape: vec![1, 2048, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "pos".into(),
            shape: vec![1],
            dtype: DType::Int32,
            is_state: false,
        },
        Feature {
            name: "q".into(),
            shape: vec![1, 8, 4, 128],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "xq".into(),
            shape: vec![1, 8, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
        Feature {
            name: "wbig".into(),
            shape: vec![8192, 8, 1, 1],
            dtype: DType::Fp16,
            is_state: false,
        },
    ];
    let fn_inputs = vec![
        NVT {
            name: "qkv".into(),
            ty: f16(&[1, 2048, 1, 1]),
        },
        NVT {
            name: "pos".into(),
            ty: ValueType::Tensor(TensorType::i32v(1)),
        },
        NVT {
            name: "q".into(),
            ty: f16(&[1, 8, 4, 128]),
        },
        NVT {
            name: "xq".into(),
            ty: f16(&[1, 8, 1, 1]),
        },
        NVT {
            name: "wbig".into(),
            ty: f16(&[8192, 8, 1, 1]),
        },
        NVT {
            name: "kv".into(),
            ty: ValueType::State(TensorType::f16(&kv_shape)),
        },
    ];
    let meta = ModelMeta::new(10, "CoreML9")
        .creator("mil_kvpack_s1n8")
        .description("packed KV slice_update probe s1 n8 prov");
    let spec = encode_model(&inputs, &outputs, &states, &blk, &fn_inputs, &meta);
    let pkg = out_dir.join("kvpack.mlpackage");
    write_mlpackage(&pkg, &spec, None).map_err(|e| e.to_string())?;
    println!("wrote {}", pkg.display());
    Ok(())
}
