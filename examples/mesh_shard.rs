//! Build a mesh rank dir with an explicit layer range:
//!   mesh_shard <model_dir> <out_dir> --rank N --world N --layers A-B --host h:p [--next h:p] [--embed] [--head]

use bad_apple::mesh_brain::{self, ShardSpec};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = std::path::PathBuf::from(args.next().expect("model dir"));
    let out = std::path::PathBuf::from(args.next().expect("out dir"));
    let mut spec = ShardSpec {
        rank: 0,
        world: 1,
        host: String::new(),
        next_host: None,
        layer_start: 0,
        layer_end: 0,
        has_embed: false,
        has_head: false,
        tie_embeddings: false,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rank" => spec.rank = args.next().unwrap().parse()?,
            "--world" => spec.world = args.next().unwrap().parse()?,
            "--host" => spec.host = args.next().unwrap(),
            "--next" => spec.next_host = Some(args.next().unwrap()),
            "--layers" => {
                let v = args.next().unwrap();
                let (a, b) = v.split_once('-').expect("A-B");
                spec.layer_start = a.parse()?;
                spec.layer_end = b.parse()?;
            }
            "--embed" => spec.has_embed = true,
            "--head" => spec.has_head = true,
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    let n = mesh_brain::build_shard(&model, &out, &spec)?;
    println!("built {out:?}: {n} tensors");
    Ok(())
}
