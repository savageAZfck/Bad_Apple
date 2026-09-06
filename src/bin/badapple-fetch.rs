//! Native Rust HF Hub downloader. No Python, no venv.
//!
//! Usage:
//!   badapple-fetch <repo_id> [cache_dir]
//!
//! Environment:
//!   HF_ENDPOINT          - Hub base URL (default https://huggingface.co)
//!   HF_TOKEN             - Bearer token for gated/private models
//!   BADAPPLE_HF_REVISION - Branch/tag/commit to fetch (default main)
//!   HF_HUB_OFFLINE=1     - Refuse to download
//!
//! The cache layout matches huggingface_hub:
//!   <cache_dir>/models--<org>--<name>/refs/<revision>
//!   <cache_dir>/models--<org>--<name>/snapshots/<revision>/<file-path>

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use ureq::Agent;

#[derive(Deserialize, Debug)]
struct TreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    size: Option<u64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <repo_id> [cache_dir]", args[0]);
        std::process::exit(2);
    }
    if std::env::var("HF_HUB_OFFLINE").unwrap_or_default() == "1" {
        eprintln!("HF_HUB_OFFLINE=1; refusing download");
        std::process::exit(1);
    }
    if let Err(e) = run(&args[1], args.get(2).map(PathBuf::from)) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(repo_id: &str, cache_dir: Option<PathBuf>) -> Result<()> {
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let token = std::env::var("HF_TOKEN").ok();
    let revision = std::env::var("BADAPPLE_HF_REVISION").unwrap_or_else(|_| "main".into());

    let parts: Vec<&str> = repo_id.split('/').collect();
    if parts.len() != 2 {
        bail!("repo_id must be 'org/name', got: {repo_id}");
    }

    let cache = cache_dir
        .or_else(hf_hub_cache)
        .ok_or_else(|| anyhow!("could not determine HF cache directory"))?;

    let model_dir = cache.join(format!("models--{}--{}", parts[0], parts[1]));
    fs::create_dir_all(&model_dir)?;

    let refs_dir = model_dir.join("refs");
    fs::create_dir_all(&refs_dir)?;
    fs::write(refs_dir.join(&revision), &revision)?;

    let snapshot_dir = model_dir.join("snapshots").join(&revision);
    fs::create_dir_all(&snapshot_dir)?;

    let agent = Agent::new_with_config(
        Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(7200)))
            .build(),
    );

    let tree_url = format!(
        "{}/api/models/{}/tree/{}?recursive=true",
        endpoint, repo_id, revision
    );
    let entries = list_tree(&agent, &tree_url, token.as_deref())?;

    let total: u64 = entries.iter().filter_map(|e| e.size).sum();
    let mut downloaded: u64 = 0;
    let start = Instant::now();

    for (i, entry) in entries.iter().enumerate() {
        if entry.kind != "file" {
            continue;
        }
        let size = entry.size.unwrap_or(0);

        let rel_path = Path::new(&entry.path);
        if rel_path.is_absolute()
            || rel_path.components().any(|c| {
                matches!(
                    c,
                    std::path::Component::ParentDir | std::path::Component::RootDir
                )
            })
        {
            bail!("refusing unsafe file path: {}", entry.path);
        }

        let dst = snapshot_dir.join(rel_path);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }

        print_progress(
            downloaded,
            total,
            &format!("{}/{} {}", i + 1, entries.len(), entry.path),
        );

        download_file(
            &agent,
            &endpoint,
            repo_id,
            &revision,
            &entry.path,
            size,
            &dst,
            token.as_deref(),
        )
        .with_context(|| format!("download {}", entry.path))?;

        downloaded += size;
    }

    print_progress(downloaded, total, "done");
    eprintln!();
    eprintln!("fetched {} in {}s", repo_id, start.elapsed().as_secs());
    Ok(())
}

fn hf_hub_cache() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("HF_HUB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    // Match BadAppleModelManager.hfCacheRoot: ~/.cache/huggingface/hub
    dirs::home_dir().map(|d| d.join(".cache").join("huggingface").join("hub"))
}

fn list_tree(agent: &Agent, url: &str, token: Option<&str>) -> Result<Vec<TreeEntry>> {
    let mut req = agent.get(url).header("User-Agent", user_agent());
    if let Some(token) = token {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let mut resp = req.call().with_context(|| format!("list tree {url}"))?;
    let body = resp.body_mut().read_to_string()?;
    let entries: Vec<TreeEntry> =
        serde_json::from_str(&body).with_context(|| "parse tree response")?;
    Ok(entries)
}

#[allow(clippy::too_many_arguments)]
fn download_file(
    agent: &Agent,
    endpoint: &str,
    repo_id: &str,
    revision: &str,
    file_path: &str,
    expected: u64,
    dst: &Path,
    token: Option<&str>,
) -> Result<()> {
    let encoded_path = file_path
        .split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/");
    let url = format!(
        "{}/{}/resolve/{}/{}",
        endpoint, repo_id, revision, encoded_path
    );

    let part = dst.with_extension("part");
    if let Ok(meta) = fs::metadata(&part) {
        if meta.len() == expected {
            fs::rename(&part, dst)?;
            eprintln!("  cached {file_path}");
            return Ok(());
        }
    }

    let mut req = agent.get(&url).header("User-Agent", user_agent());
    if let Some(token) = token {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }

    let mut resp = req.call()?;

    let mut file = fs::File::create(&part)?;
    std::io::copy(&mut resp.body_mut().as_reader(), &mut file)?;
    file.flush()?;
    drop(file);

    if expected > 0 {
        let got = fs::metadata(&part)?.len();
        if got != expected {
            bail!("size mismatch for {file_path}: expected {expected}, got {got}");
        }
    }

    fs::rename(&part, dst)?;
    Ok(())
}

fn user_agent() -> String {
    format!("badapple-fetch/{} (no-python)", env!("CARGO_PKG_VERSION"))
}

fn print_progress(bytes: u64, total: u64, label: &str) {
    let pct = if total > 0 {
        (bytes as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    eprintln!("progress: {pct:5.1}% {label}");
    println!(
        "{{\"type\":\"progress\",\"bytes\":{},\"total\":{},\"pct\":{:.4}}}",
        bytes,
        total,
        pct / 100.0
    );
}
