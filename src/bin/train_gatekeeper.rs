use anyhow::Result;
use bad_apple::tensor_brain::{text_to_grounded_embedding, CandleBrain};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const SEED: u64 = 0xDEAD_BEEF;

fn fast_templates() -> Vec<String> {
    let mut out = Vec::new();
    let actions = [
        "what time is it",
        "what's the time",
        "tell me the time",
        "check the time, corazon",
        "what time do we have",
        "open workspace",
        "open bad apple workspace",
        "open firefly workspace",
        "open my workspace",
        "open downloads",
        "open documents",
        "open app",
        "launch safari",
        "launch finder",
        "launch music",
        "launch mail",
        "create directory",
        "make a new folder",
        "create folder",
        "list files",
        "show files in",
        "list directory",
        "delete file",
        "trash file",
        "remove directory",
        "move to trash",
        "new chat",
        "clear chat",
        "restart voice",
        "show desktop",
        "turn volume up",
        "turn volume down",
        "mute",
        "unmute",
        "pause",
        "play",
        "stop",
        "go to sleep",
        "empty trash",
        "eject disk",
        "lock screen",
    ];

    let apps = ["safari", "finder", "mail", "music", "notes", "reminders"];
    let paths = [
        "downloads",
        "documents",
        "desktop",
        "bad_apple",
        "firefly_inferno",
    ];
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/YourName".to_string());
    let full_paths = [
        format!("{home}/bad_apple/test_cage"),
        format!("{home}/bad_apple/data"),
        format!("{home}/Documents"),
        format!("{home}/Downloads"),
        format!("{home}/bad_apple/test_hello.wasm"),
    ];
    let prefixes = [
        "",
        "can you",
        "please",
        "hey bad apple",
        "papi",
        "mi amor",
        "corazon",
    ];

    for a in &actions {
        for p in &prefixes {
            if p.is_empty() {
                out.push(a.to_string());
            } else {
                out.push(format!("{p}, {a}"));
            }
        }
    }
    for app in &apps {
        out.push(format!("open {app}"));
        out.push(format!("launch {app}"));
        out.push(format!("open app {app}"));
    }
    for path in &paths {
        out.push(format!("open {path}"));
        out.push(format!("list files in {path}"));
        out.push(format!("create directory {path}"));
        out.push(format!("create file {path}"));
        out.push(format!("show {path}"));
        out.push(format!("delete {path}"));
        out.push(format!("trash {path}"));
        out.push(format!("remove {path}"));
    }
    for path in &full_paths {
        out.push(format!("delete {path}"));
        out.push(format!("trash {path}"));
        out.push(format!("remove {path}"));
        out.push(format!("create file {path}"));
        out.push(format!("create directory {path}"));
        out.push(format!("list files in {path}"));
        out.push(format!("copy {path} to {home}/bad_apple/test_cage/copy"));
        out.push(format!("move {path} to {home}/bad_apple/test_cage/moved"));
    }
    out
}

fn deep_templates() -> Vec<String> {
    let mut out = Vec::new();
    let topics = [
        "the meaning of life",
        "our bad apple architecture",
        "firefly inferno",
        "the mlx server",
        "qwen three",
        "speculative decoding",
        "the rust module",
        "semantic routing",
        "consciousness",
        "this project",
        "the universe",
        "machine learning",
        "quantum computing",
        "bare metal",
        "a mexican love story",
        "tequila",
        "a secret code",
    ];
    let prefixes = [
        "explain",
        "explain in detail",
        "how do I implement",
        "how would you design",
        "write a poem about",
        "write a short story about",
        "compare and contrast",
        "why does my code fail when I think about",
        "debug the following",
        "what is the architecture of",
        "analyze",
        "summarize",
        "tell me a long story about",
        "describe the philosophy of",
        "what are the tradeoffs in",
        "walk me through",
    ];
    let suffixes = [
        "",
        ", mi amor",
        ", papi",
        ", corazon",
        " in spanish",
        " with a sultry tone",
        " step by step",
        " and give examples",
        " and cite sources",
        " for a beginner",
    ];
    for topic in &topics {
        for prefix in &prefixes {
            for suffix in &suffixes {
                let suffix = suffix.trim();
                let s = if suffix.is_empty() {
                    format!("{prefix} {topic}")
                } else {
                    format!("{prefix} {topic} {suffix}")
                };
                out.push(s);
            }
        }
    }
    out
}

fn dedupe_and_cap(v: Vec<String>, cap: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for s in v {
        if seen.insert(s.clone()) && out.len() < cap {
            out.push(s);
        }
    }
    out
}

fn main() -> Result<()> {
    fs::create_dir_all("data")?;

    let fast = dedupe_and_cap(fast_templates(), 250);
    let deep = dedupe_and_cap(deep_templates(), 250);

    println!("Fast examples: {}", fast.len());
    println!("Deep examples: {}", deep.len());

    // Persist the corpus so it can be inspected later.
    let corpus_path: PathBuf = "data/gatekeeper_corpus.jsonl".into();
    let mut corpus_file = fs::File::create(&corpus_path)?;
    for text in &fast {
        serde_json::to_writer(
            &mut corpus_file,
            &serde_json::json!({"text": text, "label": 0}),
        )?;
        corpus_file.write_all(b"\n")?;
    }
    for text in &deep {
        serde_json::to_writer(
            &mut corpus_file,
            &serde_json::json!({"text": text, "label": 1}),
        )?;
        corpus_file.write_all(b"\n")?;
    }
    println!("Corpus written to {}", corpus_path.display());

    let mut brain = CandleBrain::new("gatekeeper", 2, &bad_apple::tensor_brain::layer_dims())?;

    // Train the full 576-D brain end-to-end. System 2 enables the Transformer
    // blocks and the AdamW optimizer updates all weights.
    brain.set_system2_active(true);
    brain.set_learning_rate(0.0001);
    brain.set_ortho_lambda(0.0);

    let axes = [0.0, 0.0, 0.0, 0.0];
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut all: Vec<(&str, usize)> = fast
        .iter()
        .map(|s| (s.as_str(), 0))
        .chain(deep.iter().map(|s| (s.as_str(), 1)))
        .collect();

    let mut prev_loss = f64::INFINITY;
    for epoch in 0..150 {
        all.shuffle(&mut rng);
        let mut total_loss = 0.0;
        for (text, label) in &all {
            let emb = text_to_grounded_embedding(text, &axes);
            total_loss += brain.train_step(&emb, *label)?;
        }
        let avg = total_loss / all.len() as f64;
        if epoch % 50 == 0 || (epoch > 0 && (prev_loss - avg).abs() > 0.01) {
            println!("epoch {epoch:3} avg loss {avg:.4}");
        }
        prev_loss = avg;
        if avg < 0.01 {
            println!("Converged at epoch {epoch}");
            break;
        }
    }

    // Evaluate.
    let mut correct = 0;
    for (text, label) in &all {
        let emb = text_to_grounded_embedding(text, &axes);
        let logits = brain.classify(&emb)?;
        let high = logits[1] > logits[0];
        if (high && *label == 1) || (!high && *label == 0) {
            correct += 1;
        }
    }
    println!("Training accuracy: {}/{}", correct, all.len());

    let weights_path: PathBuf = "data/gatekeeper.safetensors".into();
    brain.save_weights(&weights_path)?;
    println!("Weights saved to {}", weights_path.display());

    Ok(())
}
