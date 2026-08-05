use crossbeam_queue::ArrayQueue;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Micro-benchmark a block and return `(result, nanoseconds)`.
///
/// Uses `std::time::Instant` on Apple Silicon (no user-space TSC) and is
/// designed to be zero-allocation inside the measured section.
#[macro_export]
macro_rules! ns_latency {
    ($block:expr) => {{
        let _start = std::time::Instant::now();
        let _result = $block;
        let _ns = _start.elapsed().as_nanos() as u64;
        (_result, _ns)
    }};
}

/// Apple Silicon aligned cache-line padding.  Explicit 128-byte alignment
/// isolates the hot-path atomics so M1/M2/M3/M4 performance cores never
/// invalidate a shared cache line.
#[repr(align(128))]
pub struct CacheLinePadded<T> {
    value: T,
}

impl<T> CacheLinePadded<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> std::ops::Deref for CacheLinePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> std::ops::DerefMut for CacheLinePadded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// Lock-free, L2-cache-aligned atomic ring buffer for nanosecond latency
/// samples.  Stores a fixed capacity of `u64` nanosecond measurements and
/// can compute percentile / mean statistics on demand.
pub struct LatencyRingBuffer<const N: usize> {
    /// Cache-padded slot array to reduce false sharing under heavy contention.
    slots: CacheLinePadded<ArrayQueue<u64>>,
    /// Total samples pushed (monotonically increasing; may wrap, used only
    /// for diagnostics).
    samples: CacheLinePadded<AtomicU64>,
}

impl<const N: usize> LatencyRingBuffer<N> {
    pub fn new() -> Self {
        Self {
            slots: CacheLinePadded::new(ArrayQueue::new(N)),
            samples: CacheLinePadded::new(AtomicU64::new(0)),
        }
    }

    /// Push a nanosecond latency sample.  If the ring is full, the oldest
    /// sample is dropped (FIFO eviction).  This is the only writer path and
    /// is allocation-free.
    pub fn push(&self, nanos: u64) {
        self.samples.fetch_add(1, Ordering::Relaxed);
        if self.slots.is_full() {
            let _ = self.slots.pop();
        }
        let _ = self.slots.push(nanos);
    }

    /// Total number of samples ever pushed.
    pub fn total_samples(&self) -> u64 {
        self.samples.load(Ordering::Relaxed)
    }

    /// Return all currently stored samples (newest-first order is not
    /// guaranteed for ArrayQueue).  Allocates a `Vec` for statistics only.
    fn snapshot(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.slots.len());
        while let Some(v) = self.slots.pop() {
            out.push(v);
        }
        // Push them back so the ring remains intact.
        for v in &out {
            let _ = self.slots.push(*v);
        }
        out
    }

    /// Compute latency distribution statistics in nanoseconds.
    pub fn stats(&self) -> LatencyStats {
        let mut samples = self.snapshot();
        if samples.is_empty() {
            return LatencyStats::default();
        }
        samples.sort_unstable();
        let n = samples.len();
        let min = samples[0];
        let max = samples[n - 1];
        let p50 = percentile_sorted(&samples, 0.50);
        let p99 = percentile_sorted(&samples, 0.99);
        let p999 = percentile_sorted(&samples, 0.999);
        let mean = samples.iter().sum::<u64>() / n as u64;
        LatencyStats {
            samples: n as u64,
            min_ns: min,
            p50_ns: p50,
            p99_ns: p99,
            p99_9_ns: p999,
            max_ns: max,
            mean_ns: mean,
        }
    }
}

impl<const N: usize> Default for LatencyRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency distribution statistics in nanoseconds.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    pub samples: u64,
    pub min_ns: u64,
    pub p50_ns: u64,
    pub p99_ns: u64,
    pub p99_9_ns: u64,
    pub max_ns: u64,
    pub mean_ns: u64,
}

fn percentile_sorted(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p) as usize;
    sorted[idx.clamp(0, sorted.len() - 1)]
}

/// A single training / runtime metric snapshot.
#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct MetricsEntry {
    pub timestamp: u64,
    pub cycle: u64,
    pub conscience_loss: f64,
    pub language_loss: f64,
    pub goal_loss: Option<f64>,
    pub world_model_loss: Option<f64>,
    pub learning_rate: f64,
    pub critic_score: f64,
    pub active_goals: usize,
    pub memory_nodes: usize,
    pub local_conscience_conf: Option<f64>,
    pub local_conscience_agreement: bool,
    pub tokens_per_second: f64,
}

/// Persistent ring-buffered metrics logger with an HTML dashboard renderer.
#[derive(Clone)]
pub struct MetricsLogger {
    history: Vec<MetricsEntry>,
    path: PathBuf,
    max_in_memory: usize,
}

impl MetricsLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            history: Vec::new(),
            path: path.as_ref().to_path_buf(),
            max_in_memory: 500,
        }
    }

    /// Append an entry to the in-memory ring buffer and the persistent JSONL file.
    pub fn record(&mut self, entry: MetricsEntry) {
        self.history.push(entry.clone());
        if self.history.len() > self.max_in_memory {
            self.history.remove(0);
        }
        self.append_jsonl(&entry);
    }

    fn append_jsonl(&self, entry: &MetricsEntry) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            if let Ok(line) = serde_json::to_string(entry) {
                let _ = writeln!(file, "{}", line);
            }
        }
    }

    /// Load the entire persistent log from disk (last N entries, newest first).
    pub fn load_recent(&self, n: usize) -> Vec<MetricsEntry> {
        if let Ok(file) = File::open(&self.path) {
            let reader = BufReader::new(file);
            let mut entries: Vec<MetricsEntry> = reader
                .lines()
                .filter_map(|line| {
                    line.ok()
                        .and_then(|l| serde_json::from_str::<MetricsEntry>(&l).ok())
                })
                .collect();
            entries.sort_by(|a, b| {
                b.cycle
                    .cmp(&a.cycle)
                    .then_with(|| b.timestamp.cmp(&a.timestamp))
            });
            entries.into_iter().take(n).collect()
        } else {
            Vec::new()
        }
    }

    pub fn recent(&self, n: usize) -> Vec<MetricsEntry> {
        self.history.iter().rev().take(n).cloned().collect()
    }

    /// Basic statistics over the last N entries.
    pub fn summary(&self, n: usize) -> MetricsSummary {
        let entries = self.recent(n);
        if entries.is_empty() {
            return MetricsSummary::default();
        }
        let count = entries.len() as f64;
        let avg = |f: fn(&MetricsEntry) -> f64| entries.iter().map(f).sum::<f64>() / count;
        MetricsSummary {
            cycles: entries.len(),
            avg_conscience_loss: avg(|e| e.conscience_loss),
            last_conscience_loss: entries[0].conscience_loss,
            avg_language_loss: avg(|e| e.language_loss),
            last_language_loss: entries[0].language_loss,
            avg_goal_loss: entries.iter().filter_map(|e| e.goal_loss).sum::<f64>()
                / entries
                    .iter()
                    .filter(|e| e.goal_loss.is_some())
                    .count()
                    .max(1) as f64,
            avg_learning_rate: avg(|e| e.learning_rate),
            avg_critic_score: avg(|e| e.critic_score),
            local_agreement_rate: entries
                .iter()
                .filter(|e| e.local_conscience_agreement)
                .count() as f64
                / count,
        }
    }

    /// Render a simple standalone HTML dashboard with sparklines and a table.
    pub fn dashboard_html(&self) -> String {
        let data = self.recent(100);
        let sparkline = |f: fn(&MetricsEntry) -> f64, color: &str| -> String {
            if data.len() < 2 {
                return String::new();
            }
            let values: Vec<f64> = data.iter().map(f).collect();
            let (min, max) = values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(*v), hi.max(*v))
                });
            let range = (max - min).max(1e-6);
            let w = 800.0 / (values.len().max(1) as f64);
            let points: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let x = i as f64 * w;
                    let y = 100.0 - ((v - min) / range) * 100.0;
                    format!("{:.2},{:.2}", x, y)
                })
                .collect();
            format!(
                r#"<svg viewBox="0 0 800 120" class="sparkline" preserveAspectRatio="none"><polyline fill="none" stroke="{}" stroke-width="2" points="{}"/></svg>"#,
                color,
                points.join(" ")
            )
        };

        let rows: Vec<String> = data
            .iter()
            .map(|e| {
                format!(
                    "<tr><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:?}</td><td>{:.6}</td><td>{:.2}</td></tr>",
                    e.cycle,
                    e.conscience_loss,
                    e.language_loss,
                    e.goal_loss.unwrap_or(f64::NAN),
                    e.learning_rate,
                    e.critic_score
                )
            })
            .collect();

        let summary = self.summary(100);

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Sapient Soul Metrics Dashboard</title>
<style>
  body {{ font-family: monospace; background: #0b0b0f; color: #d0d0e0; padding: 20px; }}
  h1 {{ color: #7df; }}
  .panel {{ background: #15151a; border: 1px solid #334; padding: 15px; margin: 15px 0; border-radius: 6px; }}
  .sparkline {{ width: 100%; height: 120px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 6px; text-align: left; border-bottom: 1px solid #334; }}
  th {{ color: #8af; }}
  .summary {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .summary div {{ background: #1f1f2a; padding: 10px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Sapient Soul Evaluation Dashboard</h1>
<div class="summary">
  <div>Cycles: {}</div>
  <div>Avg Conscience Loss: {:.4}</div>
  <div>Last Conscience Loss: {:.4}</div>
  <div>Avg Language Loss: {:.4}</div>
  <div>Avg Goal Loss: {:.4}</div>
  <div>Avg LR: {:.6}</div>
  <div>Local Agreement: {:.1}%</div>
</div>
<div class="panel">
  <h2>Conscience Loss</h2>
  {}
</div>
<div class="panel">
  <h2>Language Head Loss</h2>
  {}
</div>
<div class="panel">
  <h2>Learning Rate</h2>
  {}
</div>
<div class="panel">
  <h2>Recent Cycles</h2>
  <table>
    <tr><th>Cycle</th><th>Conscience</th><th>Language</th><th>Goal</th><th>LR</th><th>Critic</th></tr>
    {}
  </table>
</div>
<p><a href="/telemetry" style="color:#8af">Telemetry JSON</a> | <a href="/metrics" style="color:#8af">Metrics JSON</a></p>
</body>
</html>"#,
            summary.cycles,
            summary.avg_conscience_loss,
            summary.last_conscience_loss,
            summary.avg_language_loss,
            summary.avg_goal_loss,
            summary.avg_learning_rate,
            summary.local_agreement_rate * 100.0,
            sparkline(|e| e.conscience_loss, "#7df"),
            sparkline(|e| e.language_loss, "#f7d"),
            sparkline(|e| e.learning_rate, "#7f7"),
            rows.join("\n")
        )
    }
}

#[derive(Clone, Serialize, Debug, Default)]
pub struct MetricsSummary {
    pub cycles: usize,
    pub avg_conscience_loss: f64,
    pub last_conscience_loss: f64,
    pub avg_language_loss: f64,
    pub last_language_loss: f64,
    pub avg_goal_loss: f64,
    pub avg_learning_rate: f64,
    pub avg_critic_score: f64,
    pub local_agreement_rate: f64,
}

/// Result of a single memory-drift sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryDrift {
    /// Absolute bytes in use at this sample.
    pub used_bytes: u64,
    /// Estimated bytes leaked per second, positive = growth.
    pub drift_bytes_per_sec: f64,
    /// Normalized 0..1 leak score, where 1.0 means a 1 MB/s sustained growth.
    pub leak_score: f64,
}

/// Background memory-leak profiling substrate.
///
/// Samples are pushed at a fixed interval (typically 5s).  A moving window
/// regression gives a live drift estimate without keeping the whole process
/// history.  Positive drift suggests a leak; negative drift suggests cleanup.
/// When the leak score stays at or above 0.5 for three consecutive samples,
/// `should_flush()` returns `true` and `trigger_flush()` can be called to ask
/// the host allocator to consolidate/return memory.
#[derive(Clone, Debug)]
pub struct MemoryProfiler {
    samples: VecDeque<(u64, u64)>,
    max_window: usize,
    /// Consecutive samples with a leak_score >= 0.5.
    consecutive_leak_samples: usize,
    /// Bytes used at the previous sample.
    pub last_used_bytes: u64,
}

impl MemoryProfiler {
    pub fn new(max_window: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_window),
            max_window: max_window.max(2),
            consecutive_leak_samples: 0,
            last_used_bytes: 0,
        }
    }

    /// Record a new memory-usage sample and return the computed drift.
    pub fn record(&mut self, used_bytes: u64, now_secs: u64) -> MemoryDrift {
        self.last_used_bytes = used_bytes;

        // Keep a bounded window so the drift is a recent moving average.
        if self.samples.len() >= self.max_window {
            self.samples.pop_front();
        }
        self.samples.push_back((now_secs, used_bytes));

        let drift = if self.samples.len() >= 2 {
            let first = self.samples.front().copied().unwrap();
            let last = self.samples.back().copied().unwrap();
            let dt = last.0.saturating_sub(first.0) as f64;
            if dt > 0.0 {
                (last.1 as f64 - first.1 as f64) / dt
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Score: 1.0 = 1 MB/sec sustained growth.  Clamp at 1.0; negative is 0.
        let score = (drift / 1_000_000.0).clamp(-1.0, 1.0);
        let leak_score = if score > 0.0 { score } else { 0.0 };

        if leak_score >= 0.5 {
            self.consecutive_leak_samples += 1;
        } else {
            self.consecutive_leak_samples = 0;
        }

        MemoryDrift {
            used_bytes,
            drift_bytes_per_sec: drift,
            leak_score,
        }
    }

    /// True when the leak score has been >= 0.5 for 3 or more consecutive samples.
    pub fn should_flush(&self) -> bool {
        self.consecutive_leak_samples >= 3
    }

    /// Number of consecutive high-leak samples seen.
    pub fn consecutive_leak_samples(&self) -> usize {
        self.consecutive_leak_samples
    }

    /// Ask the platform allocator to consolidate / return memory to the OS.
    /// Currently a best-effort pressure-relief call; it is safe to invoke
    /// repeatedly and will not panic.
    pub fn trigger_flush(&mut self) {
        // macOS: `malloc_zone_pressure_relief` hints the allocator to release
        // dirty pages.  Linux: `malloc_trim` releases top-most memory.
        // These calls are best-effort; if the symbol is absent the program
        // continues unaffected.
        #[cfg(target_os = "macos")]
        unsafe {
            extern "C" {
                fn malloc_zone_pressure_relief(zone: *mut c_void, nbytes: usize) -> usize;
            }
            let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
        #[cfg(not(target_os = "macos"))]
        unsafe {
            let _ = libc::malloc_trim(0);
        }
        self.consecutive_leak_samples = 0;
    }

    /// Slope over the current window in bytes/second.
    pub fn drift_bytes_per_sec(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first = self.samples.front().copied().unwrap();
        let last = self.samples.back().copied().unwrap();
        let dt = last.0.saturating_sub(first.0) as f64;
        if dt > 0.0 {
            (last.1 as f64 - first.1 as f64) / dt
        } else {
            0.0
        }
    }

    /// Current sample count in the moving window.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Helper type for thread-safe shared metrics.
pub type SharedMetrics = Arc<Mutex<MetricsLogger>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_ring_buffer_percentiles() {
        let ring = LatencyRingBuffer::<16>::new();
        for i in 1..=15 {
            ring.push(i as u64 * 10);
        }
        let stats = ring.stats();
        assert_eq!(stats.samples, 15);
        assert_eq!(stats.min_ns, 10);
        assert_eq!(stats.max_ns, 150);
        assert!(stats.p50_ns >= 70 && stats.p50_ns <= 80);
        assert!(stats.p99_ns >= 140);
    }

    #[test]
    fn ns_latency_macro_is_zero_allocation() {
        let ring = LatencyRingBuffer::<1024>::new();
        for _ in 0..100 {
            let (value, ns) = crate::ns_latency!(ring.push(42));
            assert_eq!(value, ());
            ring.push(ns);
        }
        let stats = ring.stats();
        assert_eq!(stats.samples, 200);
        assert!(stats.mean_ns > 0);
    }

    #[test]
    fn memory_profiler_detects_leak() {
        let mut p = MemoryProfiler::new(4);
        p.record(1_000_000, 0);
        p.record(2_000_000, 1);
        p.record(3_000_000, 2);
        let drift = p.record(4_000_000, 3);
        assert!(drift.leak_score >= 0.5);
        assert!(p.should_flush());
    }
}
