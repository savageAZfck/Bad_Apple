//! Debug bisect client: send one hidden-state step to a mesh rank.
//!   mesh_step <host:port> --emb <embedding.f16> --token <id> [--hidden-dim N]
//!   mesh_step <host:port> --file <hidden.f16.bin> [--hidden-dim N]
//! Sends one [1,1,d] fp16 step and prints the returned token.

use bad_apple::mesh_brain;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let host = args.next().expect("host:port required");
    let mut emb: Option<String> = None;
    let mut token: usize = 0;
    let mut file: Option<String> = None;
    let mut dim = 5120usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--emb" => emb = args.next(),
            "--token" => token = args.next().unwrap().parse()?,
            "--file" => file = args.next(),
            "--hidden-dim" => dim = args.next().unwrap().parse()?,
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    let payload = if let Some(f) = file {
        std::fs::read(f)?
    } else if let Some(e) = emb {
        let data = std::fs::read(e)?;
        let off = token * dim * 2;
        data[off..off + dim * 2].to_vec()
    } else {
        vec![0u8; dim * 2]
    };
    let rows = payload.len() / (dim * 2);
    let (tok, done) = mesh_brain::step(&host, &payload, &[1, rows, dim], "float16")?;
    println!("token={tok} done={done}");
    Ok(())
}
