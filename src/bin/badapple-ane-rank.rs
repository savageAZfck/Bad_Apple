//! ANE tail rank for mesh-brain: serve a converted ANE shard set on the
//! mesh wire protocol so a GPU prefix rank (badapple-engine with
//! BADAPPLE_SHARD_DIR) hands its suffix to the Neural Engine.
//!
//!   badapple-ane-rank --listen 127.0.0.1:8742 \
//!       --manifest /path/tail_manifest.json --tokenizer /path/tokenizer.json \
//!       [--rank 1 --world 2 --layers 20-40] [--eos 151645]

use anyhow::{bail, Context, Result};
use bad_apple::ane_core::{AneCore, AneCoreConfig};
use bad_apple::mesh_brain;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut listen = "127.0.0.1:8742".to_string();
    let mut manifest: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut rank = 1usize;
    let mut world = 2usize;
    let mut layer_start = 0usize;
    let mut layer_end = 0usize;
    // Qwen im_end token id; the driver also checks its own EOS table.
    let mut eos = 151645i64;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--listen" => {
                listen = args.get(i + 1).context("--listen needs h:p")?.clone();
                i += 2;
            }
            "--manifest" => {
                manifest = Some(args.get(i + 1).context("--manifest needs path")?.into());
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(args.get(i + 1).context("--tokenizer needs path")?.into());
                i += 2;
            }
            "--rank" => {
                rank = args.get(i + 1).context("--rank needs n")?.parse()?;
                i += 2;
            }
            "--world" => {
                world = args.get(i + 1).context("--world needs n")?.parse()?;
                i += 2;
            }
            "--layers" => {
                let v = args.get(i + 1).context("--layers needs A-B")?;
                let (a, b) = v.split_once('-').context("--layers needs A-B")?;
                layer_start = a.parse()?;
                layer_end = b.parse()?;
                i += 2;
            }
            "--eos" => {
                eos = args.get(i + 1).context("--eos needs id")?.parse()?;
                i += 2;
            }
            other => bail!("unknown arg {other}"),
        }
    }
    let manifest = manifest.context("--manifest is required")?;
    let tokenizer = tokenizer.context("--tokenizer is required")?;
    let config = AneCoreConfig {
        model_path: manifest,
        tokenizer_path: tokenizer,
        max_context_tokens: 4096,
        bucket_paths: HashMap::new(),
    };
    let mut core = AneCore::load(&config).context("ANE tail core load failed")?;
    if layer_end <= layer_start {
        // Labels only — used for ping replies; the core drives whatever
        // shards the manifest lists.
        layer_end = layer_start + 1;
    }
    mesh_brain::serve_ane_tail(&listen, &mut core, rank, world, layer_start, layer_end, eos)
}
