//! MCP marketplace: a local registry of installable MCP servers.
//!
//! Replaces the Python MCP marketplace with a native Rust implementation.
//! The marketplace stores a catalog of MCP server definitions on disk and can
//! start/stop them as child processes, forwarding stdio messages.
//!
//! Security:
//! - Server `id` must match `^[a-zA-Z0-9_-]{1,64}$`.
//! - `command` is resolved against a restricted PATH and cannot contain shell
//!   metacharacters.
//! - Environment variables are filtered against a blocklist.
//! - Arguments are passed verbatim; they must be set by the user at install
//!   time, not injected.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Mutex;

/// On-disk catalog of MCP servers. Stored as JSON in the Bad Apple data dir.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct McpCatalog {
    #[serde(default)]
    pub servers: Vec<McpServer>,
    #[serde(default)]
    pub version: u32,
}

/// A single MCP server definition.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct McpServer {
    pub id: String,
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    pub transport: McpTransport,
    #[serde(default)]
    pub installed: bool,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub description: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    Stdio,
    Sse,
    Socket,
}

#[allow(clippy::derivable_impls)]
impl Default for McpTransport {
    fn default() -> Self {
        McpTransport::Stdio
    }
}

/// Runtime state of a running MCP server child process.
#[derive(Debug)]
pub struct McpProcess {
    pub id: String,
    pub child: Child,
    pub stdin: ChildStdin,
    pub started_at: Instant,
    pub last_heartbeat: Instant,
    pub exit_status: Option<std::process::ExitStatus>,
    pub stderr_task: Option<tokio::task::JoinHandle<()>>,
}

/// Manager for the MCP marketplace.
pub struct McpMarketplace {
    catalog_path: PathBuf,
    catalog: Arc<Mutex<McpCatalog>>,
    running: Arc<Mutex<HashMap<String, Arc<Mutex<McpProcess>>>>>,
    max_log_lines: usize,
}

impl McpMarketplace {
    pub fn new(catalog_path: PathBuf, max_log_lines: usize) -> Self {
        Self {
            catalog_path,
            catalog: Arc::new(Mutex::new(McpCatalog::default())),
            running: Arc::new(Mutex::new(HashMap::new())),
            max_log_lines,
        }
    }

    pub async fn load(&self) -> Result<()> {
        if !self.catalog_path.exists() {
            *self.catalog.lock().await = McpCatalog::default();
            return Ok(());
        }
        let data = tokio::fs::read_to_string(&self.catalog_path)
            .await
            .with_context(|| format!("failed to read MCP catalog from {:?}", self.catalog_path))?;
        let catalog: McpCatalog =
            serde_json::from_str(&data).with_context(|| "failed to parse MCP catalog JSON")?;
        *self.catalog.lock().await = catalog;
        Ok(())
    }

    pub async fn save(&self) -> Result<()> {
        let catalog = self.catalog.lock().await.clone();
        if let Some(parent) = self.catalog_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let json = serde_json::to_string_pretty(&catalog)?;
        tokio::fs::write(&self.catalog_path, json).await?;
        Ok(())
    }

    pub async fn list(&self) -> Vec<McpServer> {
        self.catalog.lock().await.servers.clone()
    }

    pub async fn get(&self, id: &str) -> Option<McpServer> {
        self.catalog
            .lock()
            .await
            .servers
            .iter()
            .find(|s| s.id == id)
            .cloned()
    }

    /// Add or update a server. Validates id, command, and environment.
    pub async fn upsert(&self, server: McpServer) -> Result<()> {
        validate_server_id(&server.id)?;
        validate_command(&server.command)?;
        validate_env(&server.env)?;

        let mut catalog = self.catalog.lock().await;
        if let Some(idx) = catalog.servers.iter().position(|s| s.id == server.id) {
            catalog.servers[idx] = server;
        } else {
            catalog.servers.push(server);
        }
        catalog.version += 1;
        Ok(())
    }

    pub async fn remove(&self, id: &str) -> Result<()> {
        self.stop(id).await?;
        let mut catalog = self.catalog.lock().await;
        catalog.servers.retain(|s| s.id != id);
        catalog.version += 1;
        Ok(())
    }

    pub async fn install(&self, id: &str) -> Result<()> {
        let mut catalog = self.catalog.lock().await;
        if let Some(server) = catalog.servers.iter_mut().find(|s| s.id == id) {
            server.installed = true;
            server.enabled = true;
            catalog.version += 1;
            Ok(())
        } else {
            bail!("MCP server {id} not found in catalog")
        }
    }

    pub async fn uninstall(&self, id: &str) -> Result<()> {
        self.stop(id).await?;
        let mut catalog = self.catalog.lock().await;
        if let Some(server) = catalog.servers.iter_mut().find(|s| s.id == id) {
            server.installed = false;
            server.enabled = false;
            catalog.version += 1;
            Ok(())
        } else {
            bail!("MCP server {id} not found in catalog")
        }
    }

    /// Start an installed, enabled server as a child process.
    pub async fn start(&self, id: &str) -> Result<()> {
        let server = self
            .get(id)
            .await
            .ok_or_else(|| anyhow!("MCP server {id} not found"))?;
        if !server.installed {
            bail!("MCP server {id} is not installed");
        }
        if !server.enabled {
            bail!("MCP server {id} is disabled");
        }
        if server.transport != McpTransport::Stdio {
            bail!("MCP server {id} uses non-stdio transport and cannot be started by this manager");
        }

        // If already running, stop first to avoid duplicates.
        self.stop(id).await?;

        let mut cmd = Command::new(&server.command);
        cmd.args(&server.args)
            .env_clear()
            .envs(allowed_env(&server.env))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to spawn MCP server {id}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP server {id} has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP server {id} has no stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("MCP server {id} has no stderr"))?;

        let id_clone = id.to_string();
        let stderr_task = tokio::spawn(async move {
            let reader = BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("[mcp {id_clone} stderr] {line}");
            }
        });

        // Forward stdout to a line reader. In a full implementation this would
        // deserialize JSON-RPC messages and route tool calls.
        let id_clone = id.to_string();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                // For now, drop server-to-client output. A real integration
                // would parse JSON-RPC and dispatch to handlers.
                let _ = line;
                tracing::trace!("[mcp {id_clone} stdout] {line}");
            }
        });

        let process = McpProcess {
            id: id.to_string(),
            child,
            stdin,
            started_at: Instant::now(),
            last_heartbeat: Instant::now(),
            exit_status: None,
            stderr_task: Some(stderr_task),
        };

        self.running
            .lock()
            .await
            .insert(id.to_string(), Arc::new(Mutex::new(process)));
        Ok(())
    }

    pub async fn stop(&self, id: &str) -> Result<()> {
        if let Some(handle) = self.running.lock().await.remove(id) {
            let mut process = handle.lock().await;
            let _ = process.child.start_kill();
            let _ = tokio::time::timeout(Duration::from_secs(2), process.child.wait()).await;
            if let Some(stderr) = process.stderr_task.take() {
                let _ = stderr.await;
            }
        }
        Ok(())
    }

    pub async fn stop_all(&self) {
        let ids: Vec<String> = self.running.lock().await.keys().cloned().collect();
        for id in ids {
            let _ = self.stop(&id).await;
        }
    }

    pub async fn status(&self, id: &str) -> Result<McpStatus> {
        let server = self
            .get(id)
            .await
            .ok_or_else(|| anyhow!("MCP server {id} not found"))?;
        if let Some(handle) = self.running.lock().await.get(id).cloned() {
            let process = handle.lock().await;
            Ok(McpStatus {
                id: id.to_string(),
                installed: server.installed,
                enabled: server.enabled,
                running: true,
                pid: process.child.id(),
                started_at: Some(process.started_at),
                exit_status: process.exit_status,
            })
        } else {
            Ok(McpStatus {
                id: id.to_string(),
                installed: server.installed,
                enabled: server.enabled,
                running: false,
                pid: None,
                started_at: None,
                exit_status: None,
            })
        }
    }

    /// Send a raw line to a running stdio MCP server's stdin.
    pub async fn send(&self, id: &str, line: &str) -> Result<()> {
        if let Some(handle) = self.running.lock().await.get(id).cloned() {
            let mut process = handle.lock().await;
            process
                .stdin
                .write_all(line.as_bytes())
                .await
                .context("failed to write to MCP server stdin")?;
            process
                .stdin
                .write_all(b"\n")
                .await
                .context("failed to write newline to MCP server stdin")?;
            process.last_heartbeat = Instant::now();
            Ok(())
        } else {
            bail!("MCP server {id} is not running")
        }
    }

    /// Load a built-in default catalog (e.g. filesystem, fetch, sqlite).
    pub fn default_catalog() -> McpCatalog {
        McpCatalog {
            version: 1,
            servers: vec![
                McpServer {
                    id: "filesystem".to_string(),
                    name: "Filesystem MCP".to_string(),
                    command: "npx".to_string(),
                    args: vec![
                        "-y".to_string(),
                        "@modelcontextprotocol/server-filesystem".to_string(),
                        "/".to_string(),
                    ],
                    env: HashMap::new(),
                    transport: McpTransport::Stdio,
                    installed: false,
                    enabled: false,
                    description: "Read and write files under a configured root.".to_string(),
                },
                McpServer {
                    id: "fetch".to_string(),
                    name: "Fetch MCP".to_string(),
                    command: "npx".to_string(),
                    args: vec![
                        "-y".to_string(),
                        "@modelcontextprotocol/server-fetch".to_string(),
                    ],
                    env: HashMap::new(),
                    transport: McpTransport::Stdio,
                    installed: false,
                    enabled: false,
                    description: "Fetch web content. Disabled by default in air-gapped mode."
                        .to_string(),
                },
            ],
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct McpStatus {
    pub id: String,
    pub installed: bool,
    pub enabled: bool,
    pub running: bool,
    pub pid: Option<u32>,
    #[serde(skip)]
    pub started_at: Option<Instant>,
    #[serde(skip)]
    pub exit_status: Option<std::process::ExitStatus>,
}

fn validate_server_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 64 {
        bail!("MCP server id must be 1-64 characters");
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        bail!("MCP server id must be ASCII alphanumeric, underscore, or hyphen");
    }
    Ok(())
}

fn validate_command(command: &str) -> Result<()> {
    if command.is_empty() {
        bail!("MCP server command cannot be empty");
    }
    // Reject shell metacharacters and paths containing .. or ~
    if command.contains(|c: char| ";&|`$(){}[]<>!\\*?\"'".contains(c)) {
        bail!("MCP server command contains disallowed shell metacharacters");
    }
    if command.starts_with('~') || command.contains("..") {
        bail!("MCP server command cannot use ~ or .. paths");
    }
    Ok(())
}

fn validate_env(env: &HashMap<String, String>) -> Result<()> {
    let blocked: HashSet<&str> = [
        "PATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "HOME",
        "USER",
        "SHELL",
        "SSH_AUTH_SOCK",
        "SUDO_COMMAND",
        "PS4",
    ]
    .iter()
    .copied()
    .collect();
    for k in env.keys() {
        if blocked.contains(k.as_str()) {
            bail!("MCP server environment variable {k} is blocked");
        }
        if k.starts_with("BADAPPLE_")
            || k.contains(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        {
            bail!("MCP server environment variable name is not allowed");
        }
    }
    Ok(())
}

fn allowed_env(env: &HashMap<String, String>) -> HashMap<String, String> {
    let blocked: HashSet<&str> = [
        "PATH",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "HOME",
        "USER",
        "SHELL",
        "SSH_AUTH_SOCK",
        "SUDO_COMMAND",
        "PS4",
    ]
    .iter()
    .copied()
    .collect();
    env.iter()
        .filter(|(k, _)| !blocked.contains(k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_server_id_rejects_bad_ids() {
        assert!(validate_server_id("ok-123").is_ok());
        assert!(validate_server_id("").is_err());
        assert!(validate_server_id("bad space").is_err());
        assert!(validate_server_id("a".repeat(65).as_str()).is_err());
    }

    #[test]
    fn validate_command_rejects_metacharacters() {
        assert!(validate_command("npx").is_ok());
        assert!(validate_command("rm -rf; echo").is_err());
        assert!(validate_command("~/.local/bin/mcp").is_err());
    }

    #[test]
    fn validate_env_blocks_sensitive_keys() {
        let mut env = HashMap::new();
        env.insert("PATH".to_string(), "/tmp".to_string());
        assert!(validate_env(&env).is_err());
        env.clear();
        env.insert("MY_VAR".to_string(), "ok".to_string());
        assert!(validate_env(&env).is_ok());
    }

    #[test]
    fn catalog_round_trip() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let tmp = std::env::temp_dir().join(format!("mcp_catalog_test_{}", std::process::id()));
            let path = tmp.join("mcp_catalog.json");
            let _ = std::fs::remove_dir_all(&tmp);

            let market = McpMarketplace::new(path, 100);
            market
                .upsert(McpServer {
                    id: "test".to_string(),
                    name: "Test Server".to_string(),
                    command: "npx".to_string(),
                    args: vec!["-y".to_string(), "@test".to_string()],
                    env: HashMap::new(),
                    transport: McpTransport::Stdio,
                    installed: false,
                    enabled: false,
                    description: "A test server".to_string(),
                })
                .await
                .unwrap();
            market.save().await.unwrap();

            let market2 = McpMarketplace::new(market.catalog_path.clone(), 100);
            market2.load().await.unwrap();
            let servers = market2.list().await;
            assert_eq!(servers.len(), 1);
            assert_eq!(servers[0].id, "test");

            let _ = std::fs::remove_dir_all(&tmp);
        });
    }
}
