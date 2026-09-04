//! Workspace file watcher.
//!
//! Replaces the Python `watchdog` workspace observer with a native Rust
//! implementation. Watches a directory (recursively by default) and calls a
//! callback when text/code files are created, modified, or removed. The daemon
//! can use this to keep its RAG index fresh without re-scanning the whole
//! workspace on every query.

use anyhow::{Context, Result};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shared state protected by a single mutex to avoid lock-ordering deadlocks.
struct WatcherState {
    pending: Vec<std::path::PathBuf>,
    last_event: Instant,
}

/// A workspace watcher that reports file-system events after a short
/// coalescing delay.
pub struct WorkspaceWatcher {
    _watcher: RecommendedWatcher,
    running: Arc<AtomicBool>,
}

impl WorkspaceWatcher {
    /// Watch `path` recursively and call `on_change` for every batch of
    /// file-system events. The callback receives the workspace path and a list
    /// of normalized relative paths that changed.
    ///
    /// Events are coalesced for `coalesce` duration before the callback fires.
    /// If `filter` is provided, only paths matching it are reported.
    pub fn watch<F>(
        path: &Path,
        coalesce: Duration,
        filter: Option<fn(&std::path::Path) -> bool>,
        on_change: F,
    ) -> Result<Self>
    where
        F: Fn(&Path, Vec<std::path::PathBuf>) + Send + 'static,
    {
        // Use the canonical path. macOS FSEvents returns canonical /private/var
        // paths, while the caller may use /var. Canonicalization keeps both
        // sides aligned for strip_prefix.
        let root = std::fs::canonicalize(path)
            .with_context(|| format!("failed to canonicalize {path:?}"))?;
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        let (tx, rx) = crossbeam_channel::unbounded::<std::path::PathBuf>();
        let state = Arc::new(std::sync::Mutex::new(WatcherState {
            pending: Vec::new(),
            last_event: Instant::now(),
        }));

        let state_flush = state.clone();
        let root_flush = root.clone();
        let on_change_flush = Arc::new(std::sync::Mutex::new(Some(on_change)));
        let flush_send = tx.clone();
        let flush_running = r.clone();
        std::thread::spawn(move || {
            while flush_running.load(Ordering::SeqCst) {
                std::thread::sleep(coalesce);
                let (should_flush, to_send) = {
                    let mut s = state_flush.lock().unwrap();
                    if s.last_event.elapsed() >= coalesce && !s.pending.is_empty() {
                        let batch = std::mem::take(&mut s.pending);
                        s.last_event = Instant::now();
                        (true, batch)
                    } else {
                        (false, Vec::new())
                    }
                };
                if should_flush {
                    if let Some(f) = on_change_flush.lock().unwrap().take() {
                        f(&root_flush, to_send);
                        *on_change_flush.lock().unwrap() = Some(f);
                    }
                }
            }
            let _ = flush_send; // keep channel alive until the end
        });

        let state_recv = state.clone();
        std::thread::spawn(move || {
            for rel in rx {
                let mut s = state_recv.lock().unwrap();
                if !s.pending.contains(&rel) {
                    s.pending.push(rel);
                }
                s.last_event = Instant::now();
            }
        });

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            let Ok(event) = res else { return };
            let interesting = matches!(
                event.kind,
                notify::EventKind::Create(_)
                    | notify::EventKind::Modify(_)
                    | notify::EventKind::Remove(_)
            );
            if !interesting {
                return;
            }
            for path in event.paths {
                if let Some(rel) = path.strip_prefix(&root).ok().map(|p| p.to_path_buf()) {
                    if let Some(f) = filter {
                        if !f(&rel) {
                            continue;
                        }
                    }
                    let _ = tx.send(rel);
                }
            }
        })
        .map_err(|e| anyhow::anyhow!("failed to create watcher: {e}"))?;

        watcher
            .watch(path, RecursiveMode::Recursive)
            .with_context(|| format!("failed to watch {path:?}"))?;

        Ok(Self {
            _watcher: watcher,
            running,
        })
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

/// Default workspace filter: text and code documents that are safe to index.
pub fn default_file_filter(path: &std::path::Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let ext = ext.to_lowercase();
        return matches!(
            ext.as_str(),
            "txt"
                | "md"
                | "markdown"
                | "rs"
                | "swift"
                | "py"
                | "js"
                | "ts"
                | "tsx"
                | "jsx"
                | "c"
                | "cpp"
                | "h"
                | "hpp"
                | "m"
                | "mm"
                | "sh"
                | "zsh"
                | "bash"
                | "json"
                | "yaml"
                | "yml"
                | "toml"
                | "css"
                | "html"
                | "xml"
                | "plist"
                | "sql"
        );
    }
    // No extension but a file name (not just a dir ending with component).
    path.file_name().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_file_write() {
        let tmp =
            std::env::temp_dir().join(format!("badapple_watcher_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let changed = Arc::new(std::sync::Mutex::new(false));
        let c = changed.clone();
        let watcher = WorkspaceWatcher::watch(
            &tmp,
            Duration::from_millis(100),
            Some(default_file_filter),
            move |_, paths| {
                if paths
                    .iter()
                    .any(|p| p.to_string_lossy().contains("test.txt"))
                {
                    *c.lock().unwrap() = true;
                }
            },
        )
        .unwrap();

        std::thread::sleep(Duration::from_millis(500));
        let mut f = std::fs::File::create(tmp.join("test.txt")).unwrap();
        f.write_all(b"hello").unwrap();
        drop(f);

        for _ in 0..50 {
            if *changed.lock().unwrap() {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }

        watcher.stop();
        assert!(
            *changed.lock().unwrap(),
            "watcher did not detect file write"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn default_filter_accepts_code_and_text() {
        assert!(default_file_filter(std::path::Path::new("foo.rs")));
        assert!(default_file_filter(std::path::Path::new("README.md")));
        assert!(default_file_filter(std::path::Path::new("main.swift")));
        assert!(!default_file_filter(std::path::Path::new("binary.bin")));
    }
}
