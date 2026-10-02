// Minimal mlpackage emitter to probe opset/op support in coremlc.
// usage: mil_micro <out_dir> <op> [opset] [spec_version]
use bad_apple::mil_spec::*;
use std::path::PathBuf;

fn f16(shape: &[i64]) -> ValueType {
    ValueType::Tensor(TensorType::f16(shape))
}

fn main() -> Result<(), String> {
    let out_dir = PathBuf::from(std::env::args().nth(1).expect("out dir"));
    let which = std::env::args().nth(2).unwrap_or_else(|| "rms_norm".into());
    let opset = std::env::args().nth(3).unwrap_or_else(|| "CoreML8".into());
    let spec: i32 = std::env::args()
        .nth(4)
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);

    let mut blk = Block::new();
    let (inputs, fn_inputs, outputs): (Vec<Feature>, Vec<NVT>, Vec<Feature>);

    match which.as_str() {
        // rms_norm(x [1,2560,1,1], gamma [2560,1,1]) -> [1,2560,1,1]
        "rms_norm" => {
            let g = Feature {
                name: "g".into(),
                shape: vec![2560, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            };
            inputs = vec![
                Feature {
                    name: "x".into(),
                    shape: vec![1, 2560, 1, 1],
                    dtype: DType::Fp16,
                    is_state: false,
                },
                g,
            ];
            let out = blk.o1(
                "rms_norm",
                vec![("x".into(), bind("x").1), ("gamma".into(), bind("g").1)],
                "y",
                f16(&[1, 2560, 1, 1]),
            );
            blk.outputs = vec![out];
            fn_inputs = inputs
                .iter()
                .map(|f| NVT {
                    name: f.name.clone(),
                    ty: f16(&f.shape),
                })
                .collect();
            outputs = vec![Feature {
                name: "y".into(),
                shape: vec![1, 2560, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        // scaled_dot_product_attention q,k,v [1,32,1,128] mask [1,1,1,2048]
        "sdpa" => {
            let heads = Feature {
                name: "q".into(),
                shape: vec![1, 32, 1, 128],
                dtype: DType::Fp16,
                is_state: false,
            };
            let k = Feature {
                name: "k".into(),
                shape: vec![1, 8, 2048, 128],
                dtype: DType::Fp16,
                is_state: false,
            };
            let v = Feature {
                name: "v".into(),
                shape: vec![1, 8, 2048, 128],
                dtype: DType::Fp16,
                is_state: false,
            };
            let m = Feature {
                name: "m".into(),
                shape: vec![1, 1, 1, 2048],
                dtype: DType::Fp16,
                is_state: false,
            };
            inputs = vec![heads, k, v, m];
            let out = blk.o1(
                "scaled_dot_product_attention",
                vec![
                    ("q".into(), bind("q").1),
                    ("key".into(), bind("k").1),
                    ("value".into(), bind("v").1),
                    ("attn_mask".into(), bind("m").1),
                ],
                "y",
                f16(&[1, 32, 1, 128]),
            );
            blk.outputs = vec![out];
            fn_inputs = inputs
                .iter()
                .map(|f| NVT {
                    name: f.name.clone(),
                    ty: f16(&f.shape),
                })
                .collect();
            outputs = vec![Feature {
                name: "y".into(),
                shape: vec![1, 32, 1, 128],
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        // rope(x [1,32,1,128], cos [1,1,1,64]...)  — probe layout variants later
        "rope" => {
            inputs = vec![
                Feature {
                    name: "x".into(),
                    shape: vec![1, 32, 1, 128],
                    dtype: DType::Fp16,
                    is_state: false,
                },
                Feature {
                    name: "c".into(),
                    shape: vec![1, 1, 1, 64],
                    dtype: DType::Fp16,
                    is_state: false,
                },
                Feature {
                    name: "s".into(),
                    shape: vec![1, 1, 1, 64],
                    dtype: DType::Fp16,
                    is_state: false,
                },
            ];
            let out = blk.o1(
                "rope",
                vec![
                    ("x".into(), bind("x").1),
                    ("cos".into(), bind("c").1),
                    ("sin".into(), bind("s").1),
                ],
                "y",
                f16(&[1, 32, 1, 128]),
            );
            blk.outputs = vec![out];
            fn_inputs = inputs
                .iter()
                .map(|f| NVT {
                    name: f.name.clone(),
                    ty: f16(&f.shape),
                })
                .collect();
            outputs = vec![Feature {
                name: "y".into(),
                shape: vec![1, 32, 1, 128],
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        // SDPA param probe: send the params named in BADAPPLE_SDPA_PARAMS
        // (comma-separated); per-param shapes via BADAPPLE_SDPA_SHAPES
        // "value=1,8,2048,128;key=..." else all inputs share SHAPE.
        "sdpa_bare" => {
            let params: Vec<String> = std::env::var("BADAPPLE_SDPA_PARAMS")
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            let shapes_env = std::env::var("BADAPPLE_SDPA_SHAPES").unwrap_or_default();
            let shape_of = |p: &str| -> Vec<i64> {
                shapes_env
                    .split(';')
                    .filter_map(|kv| {
                        let (k, v) = kv.split_once('=')?;
                        if k == p {
                            Some(v.split(',').filter_map(|n| n.parse().ok()).collect())
                        } else {
                            None
                        }
                    })
                    .next()
                    .unwrap_or_else(|| vec![1, 32, 1, 128])
            };
            inputs = params
                .iter()
                .enumerate()
                .map(|(i, p)| Feature {
                    name: format!("in{i}"),
                    shape: shape_of(p),
                    dtype: DType::Fp16,
                    is_state: false,
                })
                .collect();
            let ins: Vec<(String, Argument)> = params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), bind(&format!("in{i}")).1))
                .collect();
            let oshape = shape_of("query");
            let out = blk.o1("scaled_dot_product_attention", ins, "y", f16(&oshape));
            blk.outputs = vec![out];
            fn_inputs = inputs
                .iter()
                .map(|f| NVT {
                    name: f.name.clone(),
                    ty: f16(&f.shape),
                })
                .collect();
            outputs = vec![Feature {
                name: "y".into(),
                shape: oshape,
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        "sdpa_bare" => {
            let params: Vec<String> = std::env::var("BADAPPLE_SDPA_PARAMS")
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            inputs = params
                .iter()
                .enumerate()
                .map(|(i, p)| Feature {
                    name: format!("in{i}"),
                    shape: vec![1, 32, 1, 128],
                    dtype: DType::Fp16,
                    is_state: false,
                })
                .collect();
            let ins: Vec<(String, Argument)> = params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.clone(), bind(&format!("in{i}")).1))
                .collect();
            let out = blk.o1(
                "scaled_dot_product_attention",
                ins,
                "y",
                f16(&[1, 32, 1, 128]),
            );
            blk.outputs = vec![out];
            fn_inputs = inputs
                .iter()
                .map(|f| NVT {
                    name: f.name.clone(),
                    ty: f16(&f.shape),
                })
                .collect();
            outputs = vec![Feature {
                name: "y".into(),
                shape: vec![1, 32, 1, 128],
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        // Chain of candidate ops on [1,64,1,1] to see preferred device per op.
        "opscan" => {
            inputs = vec![Feature {
                name: "x".into(),
                shape: vec![1, 64, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }];
            let ax = blk.konst_i32("ax", &[1]);
            let kd = blk.konst_bool("kd", true);
            let one = blk.konst_f16("one", 1.0);
            let negone = blk.konst_f16("negone", -1.0);
            let small = vec![1, 1, 1, 1];
            let full = vec![1, 64, 1, 1];
            let mx = blk.o1(
                "reduce_max",
                vec![
                    ("x".into(), bind("x").1),
                    ("axes".into(), bind(&ax).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                "mx",
                f16(&small),
            );
            let d = blk.sub("x", &mx, &full, "d");
            let e = blk.o1("exp", vec![("x".into(), bind(&d).1)], "e", f16(&full));
            let s = blk.o1(
                "reduce_sum",
                vec![
                    ("x".into(), bind(&e).1),
                    ("axes".into(), bind(&ax).1),
                    ("keep_dims".into(), bind(&kd).1),
                ],
                "s",
                f16(&small),
            );
            let sm = blk.o1(
                "real_div",
                vec![("x".into(), bind(&e).1), ("y".into(), bind(&s).1)],
                "sm",
                f16(&full),
            );
            let ne = blk.mul("x", &negone, &full, "ne");
            let ene = blk.o1("exp", vec![("x".into(), bind(&ne).1)], "ene", f16(&full));
            let den = blk.add(&ene, &one, &full, "den");
            let sig = blk.o1(
                "real_div",
                vec![("x".into(), bind(&one).1), ("y".into(), bind(&den).1)],
                "sig",
                f16(&full),
            );
            let sil = blk.mul("x", &sig, &full, "sil");
            let reps = blk.konst_i32("reps", &[1, 4, 1, 1]);
            let tl = blk.o1(
                "tile",
                vec![("x".into(), bind(&sil).1), ("reps".into(), bind(&reps).1)],
                "tl",
                f16(&[1, 256, 1, 1]),
            );
            let axs = blk.konst_scalar_i32("axs", 1);
            let sm2 = blk.o1(
                "softmax",
                vec![("x".into(), bind(&tl).1), ("axis".into(), bind(&axs).1)],
                "sm2",
                f16(&[1, 256, 1, 1]),
            );
            let sig2 = blk.o1(
                "sigmoid",
                vec![("x".into(), bind(&sm2).1)],
                "sig2",
                f16(&[1, 256, 1, 1]),
            );
            let sil2 = blk.o1(
                "silu",
                vec![("x".into(), bind(&sig2).1)],
                "sil2",
                f16(&[1, 256, 1, 1]),
            );
            blk.outputs = vec![sil2];
            fn_inputs = vec![NVT {
                name: "x".into(),
                ty: f16(&[1, 64, 1, 1]),
            }];
            outputs = vec![Feature {
                name: "sil2".into(),
                shape: vec![1, 256, 1, 1],
                dtype: DType::Fp16,
                is_state: false,
            }];
        }
        other => return Err(format!("unknown probe op {other}")),
    }

    let spec_bytes = encode_model(&inputs, &outputs, &[], &blk, &fn_inputs, spec, &opset);
    let pkg = out_dir.join(format!("{which}_{opset}_v{spec}.mlpackage"));
    write_mlpackage(&pkg, &spec_bytes, None).map_err(|e| e.to_string())?;
    println!("wrote {}", pkg.display());
    Ok(())
}
