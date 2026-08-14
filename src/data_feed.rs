use rand::seq::SliceRandom;
use std::fs;
use std::path::PathBuf;

/// A simple local-text curriculum feeder.
///
/// Scans configured directories for readable text files, splits them into
/// sentence-sized snippets, and serves random snippets as training material
/// for the cognitive loop.
#[derive(Debug)]
pub struct DataCurriculum {
    data_dirs: Vec<PathBuf>,
    corpus: Vec<String>,
}

impl DataCurriculum {
    pub fn new(data_dirs: Vec<PathBuf>) -> Self {
        let mut s = Self {
            data_dirs,
            corpus: Vec::new(),
        };
        s.reload();
        s
    }

    pub fn reload(&mut self) {
        self.corpus.clear();
        for dir in &self.data_dirs {
            if !dir.is_dir() {
                continue;
            }
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let is_text = path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| {
                            let e = e.to_lowercase();
                            matches!(
                                e.as_str(),
                                "txt"
                                    | "md"
                                    | "rs"
                                    | "py"
                                    | "json"
                                    | "toml"
                                    | "csv"
                                    | "log"
                                    | "html"
                                    | "xml"
                            )
                        })
                        .unwrap_or(false);
                    if !is_text {
                        continue;
                    }
                    if let Ok(text) = fs::read_to_string(&path) {
                        let snippets = Self::split_snippets(&text);
                        self.corpus.extend(snippets);
                    }
                }
            }
        }
        if self.corpus.is_empty() {
            self.corpus.push(
                "The firefly soul observes the world through sensors and learns from every cycle."
                    .to_string(),
            );
        }
    }

    fn split_snippets(text: &str) -> Vec<String> {
        let mut snippets = Vec::new();
        for raw in text.split(['.', '?', '!', '\n']) {
            let s = raw.trim().replace(|c: char| c.is_control(), " ");
            if s.len() > 20 && s.len() < 1200 {
                snippets.push(s);
            }
        }
        snippets
    }

    pub fn random_snippet(&self) -> String {
        let mut rng = rand::thread_rng();
        self.corpus
            .choose(&mut rng)
            .cloned()
            .unwrap_or_else(|| "Observation continues.".to_string())
    }

    pub fn snippet_count(&self) -> usize {
        self.corpus.len()
    }

    /// Add a new snippet to the live corpus and persist it to the first curriculum directory.
    pub fn add_snippet(&mut self, snippet: &str) {
        let cleaned = snippet.trim().replace(|c: char| c.is_control(), " ");
        if cleaned.len() < 20 || cleaned.len() > 1200 {
            return;
        }
        self.corpus.push(cleaned.clone());
        if let Some(dir) = self.data_dirs.first() {
            let path = dir.join("auto_curriculum.txt");
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                let _ = std::io::Write::write_fmt(&mut file, format_args!("{} ", cleaned));
            }
        }
    }
}

/// Sensible default curriculum directories for a Firefly instance.
pub fn default_curriculum_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(
            PathBuf::from(&home)
                .join("Firefly-EdgeOS")
                .join("curriculum"),
        );
        dirs.push(
            PathBuf::from(&home)
                .join("Documents")
                .join("firefly-curriculum"),
        );
    }
    dirs.push(PathBuf::from("curriculum"));
    dirs
}
