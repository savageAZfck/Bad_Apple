//! Runtime configuration for the bad_apple agent.
//!
//! Values are loaded from environment variables (prefixed with `BADAPPLE_`) and
//! fall back to sensible defaults for local development. A JSON config file
//! pointed at by `BADAPPLE_CONFIG_FILE` is overlaid on top of those defaults.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Central runtime configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Telemetry/dashboard HTTP port.
    pub telemetry_port: u16,
    /// First port in the multi-agent UDP range.
    pub multi_agent_port_start: u16,
    /// Last port in the multi-agent UDP range.
    pub multi_agent_port_end: u16,
    /// Native Apple Intelligence / oracle model identifier.
    pub apple_intelligence_model: String,
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
    /// Wide-area peer node addresses in `host:port` form.
    pub peer_nodes: Vec<String>,
    /// Optional path to a JSON file with extra config (overlays env defaults).
    pub config_file: Option<PathBuf>,
    /// TCP port to listen on for wide-area peer connections.
    pub wan_tcp_port: u16,
    /// WebSocket port to listen on for wide-area peer connections.
    pub wan_ws_port: u16,
    /// Maximum number of wide-area peer connections.
    pub max_wan_peers: usize,
    /// Base retry delay in milliseconds for peer reconnection (exponential backoff).
    pub peer_retry_base_ms: u64,
    /// Maximum retry delay in milliseconds for peer reconnection.
    pub peer_retry_max_ms: u64,
}

impl Config {
    pub fn from_env() -> Self {
        let config_file = std::env::var("BADAPPLE_CONFIG_FILE")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists());

        Self::from_env_raw(config_file)
    }

    /// Load config from a specific file path, falling back to env defaults.
    pub fn from_file(path: &Path) -> Self {
        let config_file = if path.exists() {
            Some(path.to_path_buf())
        } else {
            None
        };
        Self::from_env_raw(config_file)
    }

    fn from_env_raw(config_file: Option<PathBuf>) -> Self {
        let mut cfg = Self {
            telemetry_port: env_u16("BADAPPLE_TELEMETRY_PORT", 8080),
            multi_agent_port_start: env_u16("BADAPPLE_MULTI_AGENT_PORT_START", 5001),
            multi_agent_port_end: env_u16("BADAPPLE_MULTI_AGENT_PORT_END", 5010),
            apple_intelligence_model: env_or(
                "BADAPPLE_APPLE_INTELLIGENCE_MODEL",
                "apple-intelligence",
            ),
            state_file: env_path("BADAPPLE_STATE_FILE", "state.json"),
            curriculum_dir: env_path("BADAPPLE_CURRICULUM_DIR", "curriculum"),
            wild_workspace_dir: env_path("BADAPPLE_WILD_WORKSPACE_DIR", "wild_workspace"),
            sled_db_path: env_path("BADAPPLE_SLED_DB_PATH", "strategy_db"),
            metrics_log: env_path("BADAPPLE_METRICS_LOG", "metrics.jsonl"),
            skills_dir: env_path("BADAPPLE_SKILLS_DIR", "skills"),
            tools_dir: env_path("BADAPPLE_TOOLS_DIR", "tools"),
            clock_interval_secs: env_u64("BADAPPLE_CLOCK_INTERVAL_SECS", 6),
            max_engram_batch: env_usize("BADAPPLE_MAX_ENGRAM_BATCH", 64),
            engram_timeout_ms: env_u64("BADAPPLE_ENGRAM_TIMEOUT_MS", 5),
            multi_agent_secret: std::env::var("BADAPPLE_MULTI_AGENT_SECRET").ok(),
            peer_nodes: parse_peer_list(&env_or("BADAPPLE_PEERS", "")),
            config_file,
            wan_tcp_port: env_u16("BADAPPLE_WAN_TCP_PORT", 6001),
            wan_ws_port: env_u16("BADAPPLE_WAN_WS_PORT", 6002),
            max_wan_peers: env_usize("BADAPPLE_MAX_WAN_PEERS", 64),
            peer_retry_base_ms: env_u64("BADAPPLE_PEER_RETRY_BASE_MS", 250),
            peer_retry_max_ms: env_u64("BADAPPLE_PEER_RETRY_MAX_MS", 30_000),
        };

        // Overlay an active JSON configuration file if one is provided.
        if let Some(path) = &cfg.config_file {
            if let Ok(text) = std::fs::read_to_string(path) {
                if let Ok(overlay) = serde_json::from_str::<Config>(&text) {
                    // Overlay all non-default fields from the file. The env already won for
                    // values explicitly set; the file acts as a stable structured override.
                    merge_config(&mut cfg, overlay);
                }
            }
        }

        cfg
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
        .map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn parse_peer_list(s: &str) -> Vec<String> {
    s.split([',', ';', '\n', '\r'])
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

fn merge_config(base: &mut Config, overlay: Config) {
    if overlay.telemetry_port != 0 {
        base.telemetry_port = overlay.telemetry_port;
    }
    if overlay.multi_agent_port_start != 0 {
        base.multi_agent_port_start = overlay.multi_agent_port_start;
    }
    if overlay.multi_agent_port_end != 0 {
        base.multi_agent_port_end = overlay.multi_agent_port_end;
    }
    if !overlay.apple_intelligence_model.is_empty() {
        base.apple_intelligence_model = overlay.apple_intelligence_model;
    }
    if !overlay.state_file.as_os_str().is_empty() {
        base.state_file = overlay.state_file;
    }
    if !overlay.curriculum_dir.as_os_str().is_empty() {
        base.curriculum_dir = overlay.curriculum_dir;
    }
    if !overlay.wild_workspace_dir.as_os_str().is_empty() {
        base.wild_workspace_dir = overlay.wild_workspace_dir;
    }
    if !overlay.sled_db_path.as_os_str().is_empty() {
        base.sled_db_path = overlay.sled_db_path;
    }
    if !overlay.metrics_log.as_os_str().is_empty() {
        base.metrics_log = overlay.metrics_log;
    }
    if !overlay.skills_dir.as_os_str().is_empty() {
        base.skills_dir = overlay.skills_dir;
    }
    if !overlay.tools_dir.as_os_str().is_empty() {
        base.tools_dir = overlay.tools_dir;
    }
    if overlay.clock_interval_secs != 0 {
        base.clock_interval_secs = overlay.clock_interval_secs;
    }
    if overlay.max_engram_batch != 0 {
        base.max_engram_batch = overlay.max_engram_batch;
    }
    if overlay.engram_timeout_ms != 0 {
        base.engram_timeout_ms = overlay.engram_timeout_ms;
    }
    if overlay.multi_agent_secret.is_some() {
        base.multi_agent_secret = overlay.multi_agent_secret;
    }
    if !overlay.peer_nodes.is_empty() {
        base.peer_nodes = overlay.peer_nodes;
    }
    if overlay.wan_tcp_port != 0 {
        base.wan_tcp_port = overlay.wan_tcp_port;
    }
    if overlay.wan_ws_port != 0 {
        base.wan_ws_port = overlay.wan_ws_port;
    }
    if overlay.max_wan_peers != 0 {
        base.max_wan_peers = overlay.max_wan_peers;
    }
    if overlay.peer_retry_base_ms != 0 {
        base.peer_retry_base_ms = overlay.peer_retry_base_ms;
    }
    if overlay.peer_retry_max_ms != 0 {
        base.peer_retry_max_ms = overlay.peer_retry_max_ms;
    }
}
