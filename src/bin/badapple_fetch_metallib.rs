//! Build-time helper: download the MLX Metal shader library from PyPI
//! without requiring Python.
//!
//! Replaces the old `python3 -m pip download ... && python3 -m zipfile -e ...`
//! pipeline in `build_bad_apple_menu_bar.sh`.
//!
//! Usage:
//!   badapple-fetch-metallib <version> <expected-sha256> <output-path>
//!
//! If `<output-path>` already exists and its SHA-256 matches, the download is
//! skipped entirely.

use std::fs;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <version> <expected-sha256> <output-path>",
            args.first()
                .map(|s| s.as_str())
                .unwrap_or("badapple-fetch-metallib")
        );
        std::process::exit(2);
    }

    let version = &args[1];
    let expected_sha = args[2].to_lowercase();
    let output_path = Path::new(&args[3]);

    if let Err(e) = run(version, &expected_sha, output_path) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(version: &str, expected_sha: &str, output_path: &Path) -> anyhow::Result<()> {
    // Fast path: cache is already valid.
    if output_path.exists() {
        if let Ok(data) = fs::read(output_path) {
            if sha256_hex(&data) == expected_sha {
                println!("metallib cache valid, skipping download");
                return Ok(());
            }
        }
    }

    // 1. Resolve the wheel URL via the PyPI JSON API.
    let api_url = format!("https://pypi.org/pypi/mlx-metal/{version}/json");
    eprintln!("Fetching PyPI metadata: {api_url}");
    let mut resp = ureq::get(&api_url).call()?;
    let body = resp.body_mut().read_to_string()?;
    let meta: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| anyhow::anyhow!("parse PyPI JSON: {e}"))?;

    let urls = meta["urls"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("PyPI JSON has no 'urls' array"))?;

    // mlx-metal publishes separate wheels per macOS SDK version (e.g.
    // macosx_14_0_arm64, macosx_15_0_arm64, macosx_26_0_arm64).  Pick the
    // one with the highest platform tag that we can run on — this mirrors
    // what `pip download` would select on the current machine.
    let mut best: Option<&serde_json::Value> = None;
    let mut best_tag = -1i32;
    for u in urls {
        let filename = match u["filename"].as_str() {
            Some(f) if f.ends_with(".whl") => f,
            _ => continue,
        };
        // Parse the macOS version tag from e.g. "mlx_metal-0.31.1-py3-none-macosx_26_0_arm64.whl"
        let tag = filename
            .split("macosx_")
            .nth(1)
            .and_then(|s| s.split('_').next())
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        if tag > best_tag {
            best_tag = tag;
            best = Some(u);
        }
    }

    let wheel = best
        .ok_or_else(|| anyhow::anyhow!("no .whl file in PyPI response for mlx-metal {version}"))?;

    let wheel_url = wheel["url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("wheel entry has no 'url'"))?;

    // 2. Download the wheel.
    eprintln!("Downloading wheel: {wheel_url}");
    let mut resp = ureq::get(wheel_url).call()?;
    let mut wheel_bytes = Vec::new();
    resp.body_mut().as_reader().read_to_end(&mut wheel_bytes)?;
    eprintln!("Downloaded {} bytes", wheel_bytes.len());

    // 3. Extract mlx/lib/mlx.metallib from the zip.
    let cursor = std::io::Cursor::new(&wheel_bytes);
    let mut archive =
        zip::ZipArchive::new(cursor).map_err(|e| anyhow::anyhow!("open wheel as zip: {e}"))?;

    let mut metallib: Option<Vec<u8>> = None;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        if name == "mlx/lib/mlx.metallib" {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            metallib = Some(buf);
            break;
        }
    }

    let metallib =
        metallib.ok_or_else(|| anyhow::anyhow!("mlx/lib/mlx.metallib not found in wheel"))?;

    // 4. Verify SHA-256.
    let actual = sha256_hex(&metallib);
    if actual != expected_sha {
        anyhow::bail!("SHA-256 mismatch: expected {expected_sha}, got {actual}");
    }

    // 5. Write to the output path.
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, &metallib)?;
    eprintln!(
        "Installed metallib ({} bytes) to {}",
        metallib.len(),
        output_path.display()
    );
    Ok(())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}
