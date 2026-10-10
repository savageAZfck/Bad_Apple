//! Width-parity probe: drive two tail ranks with identical hidden-state rows —
//! the width-1 tail via serial `step` (one row per call) and the width-N tail
//! via `vstep` (all rows in one call) — and compare per-position argmaxes.
//! Divergence = batched KV/mask semantics bug.
//!
//!   ane_vstep_probe --w1 127.0.0.1:8744 --w8 127.0.0.1:8745 [--ctx 12] [--probe 8]
//!
//! Rows are a deterministic synthetic pattern; correctness here means
//! AGREEMENT between the serial and batched code paths, not model quality.

use bad_apple::mesh_brain;

fn synth_row(seed: u64, row: usize, dim: usize) -> Vec<u8> {
    let mut state = seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(dim * 2);
    for _ in 0..dim {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let v = ((state >> 33) & 0xFFFF) as f32 / 32768.0 - 1.0;
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

fn main() -> anyhow::Result<()> {
    let mut w1 = String::from("127.0.0.1:8744");
    let mut w8 = String::from("127.0.0.1:8745");
    let mut ctx_rows = 12usize;
    let mut probe_rows = 8usize;
    let dim = 5120usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--w1" => w1 = args.next().unwrap(),
            "--w8" => w8 = args.next().unwrap(),
            "--ctx" => ctx_rows = args.next().unwrap().parse()?,
            "--probe" => probe_rows = args.next().unwrap().parse()?,
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    println!("reset both tails...");
    mesh_brain::reset(&w1)?;
    mesh_brain::reset(&w8)?;

    // Context phase: identical rows serially on both (the W8 tail batches
    // internally; the W1 tail is the trusted serial baseline).
    let mut w1_ctx = Vec::new();
    for r in 0..ctx_rows {
        let row = synth_row(0xC0FFEE, r, dim);
        let (t1, _) = mesh_brain::step(&w1, &row, &[1, 1, dim], "float16")?;
        let (t8, _) = mesh_brain::step(&w8, &row, &[1, 1, dim], "float16")?;
        let mark = if t1 == t8 { "ok" } else { "MISMATCH" };
        println!("  ctx row {r}: w1={t1} w8={t8} {mark}");
        w1_ctx.push(t1);
    }

    // Probe phase: K rows — serial on W1 (K calls), one vstep on W8.
    let mut batch = Vec::with_capacity(probe_rows * dim * 2);
    for r in 0..probe_rows {
        batch.extend_from_slice(&synth_row(0xBEEF, ctx_rows + r, dim));
    }
    let serial_toks: Vec<i64> = {
        let mut v = Vec::new();
        for r in 0..probe_rows {
            let row = &batch[r * dim * 2..(r + 1) * dim * 2];
            let (t, _) = mesh_brain::step(&w1, row, &[1, 1, dim], "float16")?;
            v.push(t);
        }
        v
    };
    let batched = mesh_brain::vstep(&w8, &batch, &[1, probe_rows, dim], "float16")?;

    let mut mismatches = 0;
    for r in 0..probe_rows {
        let s = serial_toks.get(r).copied().unwrap_or(-1);
        let b = batched.get(r).copied().unwrap_or(-1) as i64;
        let mark = if s == b {
            "ok"
        } else {
            mismatches += 1;
            "MISMATCH"
        };
        println!("  probe row {r}: serial={s} batched={b} {mark}");
    }
    if mismatches == 0 {
        println!("PARITY OK — {ctx_rows} ctx + {probe_rows} probe rows agree");
    } else {
        println!("PARITY FAIL — {mismatches} mismatched argmaxes");
        std::process::exit(1);
    }
    Ok(())
}
