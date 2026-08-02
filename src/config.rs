//! Runtime configuration for the sapient_soul agent.
//!
//! Values are loaded from environment variables (prefixed with `FIREFLY_`) and
//! fall back to sensible defaults for local development.

use std::path::PathBuf;

/// Central runtime configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// Telemetry/dashboard HTTP port.
    pub telemetry_port: u16,
    /// First port in the multi-agent UDP range.
    pub multi_agent_port_start: u16,
    /// Last port in the multi-agent UDP range.
    pub multi_agent_port_end: u16,
    /// Local Ollama base URL.
    pub ollama_url: String,
    /// Ollama model used by the conscience oracle.
    pub ollama_model: String,
    /// Path to the persistent agent state file.
    pub state_file: PathBuf,
    /// Path to the local curriculum directory.
    pub curriculum_dir: PathBuf,
    /// Path to the watched wild workspace directory.
    pub wild_workspace_dir: PathBuf,
    /// Path to Sled strategy database.
    pub sled_db_path: PathBuf,
    /// Path to metrics JSONL log.
    pub metrics_log: PathBuf,
    /// Path to learned skills directory.
    pub skills_dir: PathBuf,
    /// Path to sandboxed tools directory.
    pub tools_dir: PathBuf,
    /// Cognitive clock interval in seconds.
    pub clock_interval_secs: u64,
    /// Default maximum engram batch size for multi-agent merge.
    pub max_engram_batch: usize,
    /// Default engram timeout in milliseconds.
    pub engram_timeout_ms: u64,
    /// Shared multi-agent signing secret (optional; default derived from hostname).
    pub multi_agent_secret: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            telemetry_port: env_u16("FIREFLY_TELEMETRY_PORT", 8080),
            multi_agent_port_start: env_u16("FIREFLY_MULTI_AGENT_PORT_START", 5001),
            multi_agent_port_end: env_u16("FIREFLY_MULTI_AGENT_PORT_END", 5010),
            ollama_url: env_or("FIREFLY_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model: env_or("FIREFLY_OLLAMA_MODEL", "mistral-nemo"),
            state_file: env_path("FIREFLY_STATE_FILE", "state.json"),
            curriculum_dir: env_path("FIREFLY_CURRICULUM_DIR", "curriculum"),
            wild_workspace_dir: env_path("FIREFLY_WILD_WORKSPACE_DIR", "wild_workspace"),
            sled_db_path: env_path("FIREFLY_SLED_DB_PATH", "strategy_db"),
            metrics_log: env_path("FIREFLY_METRICS_LOG", "metrics.jsonl"),
            skills_dir: env_path("FIREFLY_SKILLS_DIR", "skills"),
            tools_dir: env_path("FIREFLY_TOOLS_DIR", "tools"),
            clock_interval_secs: env_u64("FIREFLY_CLOCK_INTERVAL_SECS", 6),
            max_engram_batch: env_usize("FIREFLY_MAX_ENGRAM_BATCH", 64),
            engram_timeout_ms: env_u64("FIREFLY_ENGRAM_TIMEOUT_MS", 5),
            multi_agent_secret: std::env::var("FIREFLY_MULTI_AGENT_SECRET").ok(),
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_path(key: &str, default: &str) -> PathBuf {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}
