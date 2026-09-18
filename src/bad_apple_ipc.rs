use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const SLICKS_VERSION: u8 = 1;
pub const SLICKS_VERSION_2: u8 = 2;
pub const DEFAULT_SOCKET_PATH: &str = "/var/run/badapple/substrate.sock";
pub const DEFAULT_MLX_SOCKET_PATH: &str = "/var/run/badapple/substrate_mlx.sock";
pub const DEFAULT_IDENTITY_AGENT_SOCKET: &str = "/var/run/badapple/identity.sock";
pub const DEFAULT_KEY_PATH: &str = "/var/lib/bad_apple/slicks.key";
pub const MAX_PROMPT_BYTES: usize = 64 * 1024;
pub const MAX_NEW_TOKENS: usize = 4096;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
pub const HANDSHAKE_MAX_SKEW: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientFrame {
    Hello {
        version: u8,
        timestamp_ms: u64,
        client_nonce: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        client_pubkey: Option<String>,
    },
    Execute {
        version: u8,
        timestamp_ms: u64,
        client_nonce: String,
        server_nonce: String,
        prompt: String,
        max_new_tokens: usize,
        proof: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        client_pubkey: Option<String>,
    },
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Metrics {
    pub tokens: usize,
    pub decode_tps: f64,
    pub total_tps: f64,
    pub draft_accept_pct: f64,
    pub peak_memory_gb: f64,
    /// Which model handled this query: "fast" (0.5B), "main" (9B), "fast_action"
    /// (gatekeeper resolved locally), or "unknown".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    Challenge {
        version: u8,
        server_nonce: String,
        proof: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_pubkey: Option<String>,
    },
    Accepted,
    Token {
        text: String,
    },
    Done {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metrics: Option<Metrics>,
    },
    Response {
        #[serde(default)]
        result: Value,
    },
    Error {
        message: String,
    },
}

pub fn socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_SOCKET_PATH")
        .map_or_else(|| PathBuf::from(DEFAULT_SOCKET_PATH), PathBuf::from)
}

pub fn mlx_socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_MLX_SOCKET_PATH")
        .map_or_else(|| PathBuf::from(DEFAULT_MLX_SOCKET_PATH), PathBuf::from)
}

pub fn identity_agent_socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_IDENTITY_AGENT_SOCKET").map_or_else(
        || PathBuf::from(DEFAULT_IDENTITY_AGENT_SOCKET),
        PathBuf::from,
    )
}

pub fn key_path() -> PathBuf {
    std::env::var_os("BADAPPLE_SLICKS_KEY_PATH")
        .map_or_else(|| PathBuf::from(DEFAULT_KEY_PATH), PathBuf::from)
}

pub fn load_slicks_secret() -> Result<Vec<u8>> {
    let raw = if let Some(secret) = std::env::var_os("BADAPPLE_SLICKS_SECRET") {
        secret.to_string_lossy().into_owned()
    } else {
        fs::read_to_string(key_path()).context("unable to read the Bad Apple SLICKS key")?
    };
    let trimmed = raw.trim();
    let secret = decode_hex(trimmed).unwrap_or_else(|| trimmed.as_bytes().to_vec());
    if secret.len() < 16 {
        bail!("Bad Apple SLICKS key must contain at least 128 bits");
    }
    Ok(secret)
}

pub fn now_unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis() as u64)
}

pub fn timestamp_is_fresh(timestamp_ms: u64) -> bool {
    let Ok(now) = now_unix_ms() else {
        return false;
    };
    now.abs_diff(timestamp_ms) <= HANDSHAKE_MAX_SKEW.as_millis() as u64
}

pub fn random_nonce() -> String {
    let mut nonce = [0_u8; 32];
    OsRng.fill_bytes(&mut nonce);
    hex_encode(&nonce)
}

pub fn nonce_is_valid(nonce: &str) -> bool {
    nonce.len() == 64 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn server_proof(
    secret: &[u8],
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
) -> String {
    sign(
        secret,
        format!(
            "BADAPPLE-SLICKS/{SLICKS_VERSION}|server|{timestamp_ms}|{client_nonce}|{server_nonce}"
        )
        .as_bytes(),
    )
}

pub fn verify_server_proof(
    secret: &[u8],
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    proof: &str,
) -> bool {
    verify(
        secret,
        format!(
            "BADAPPLE-SLICKS/{SLICKS_VERSION}|server|{timestamp_ms}|{client_nonce}|{server_nonce}"
        )
        .as_bytes(),
        proof,
    )
}

pub fn client_proof(
    secret: &[u8],
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    prompt: &str,
    max_new_tokens: usize,
) -> String {
    sign(
        secret,
        client_material(
            timestamp_ms,
            client_nonce,
            server_nonce,
            prompt,
            max_new_tokens,
        )
        .as_bytes(),
    )
}

pub fn verify_client_proof(
    secret: &[u8],
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    prompt: &str,
    max_new_tokens: usize,
    proof: &str,
) -> bool {
    verify(
        secret,
        client_material(
            timestamp_ms,
            client_nonce,
            server_nonce,
            prompt,
            max_new_tokens,
        )
        .as_bytes(),
        proof,
    )
}

pub fn validate_request(prompt: &str, max_new_tokens: usize) -> Result<()> {
    if prompt.trim().is_empty() {
        bail!("prompt must not be empty");
    }
    if prompt.len() > MAX_PROMPT_BYTES {
        bail!("prompt exceeds the {MAX_PROMPT_BYTES}-byte limit");
    }
    if !(1..=MAX_NEW_TOKENS).contains(&max_new_tokens) {
        bail!("max_new_tokens must be between 1 and {MAX_NEW_TOKENS}");
    }
    Ok(())
}

/// Thread-safe replay cache: stores consumed (client_nonce, server_nonce) pairs
/// to reject replayed Execute frames within the freshness window.
///
/// Eviction is FIFO: when the cache is full, only the OLDEST entry is removed.
/// (The previous implementation cleared the whole cache at capacity, which let
/// an attacker flush every seen nonce by flooding the cache and then replay a
/// captured frame inside the freshness window.)
pub struct ReplayCache {
    seen: Mutex<HashSet<(String, String)>>,
    order: Mutex<std::collections::VecDeque<(String, String)>>,
    max: usize,
}

impl ReplayCache {
    /// Create a new cache with the given bound on stored nonce pairs.
    pub fn new(max: usize) -> Self {
        Self {
            seen: Mutex::new(HashSet::new()),
            order: Mutex::new(std::collections::VecDeque::new()),
            max,
        }
    }

    /// Check if a nonce pair has been used, and insert it if not.
    /// Returns `true` if the pair is fresh (not a replay).
    pub fn check_and_insert(&self, client_nonce: &str, server_nonce: &str) -> bool {
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
        let key = (client_nonce.to_string(), server_nonce.to_string());
        if seen.contains(&key) {
            return false; // replay
        }
        while seen.len() >= self.max {
            match order.pop_front() {
                Some(oldest) => {
                    seen.remove(&oldest);
                }
                None => break,
            }
        }
        seen.insert(key.clone());
        order.push_back(key);
        true
    }
}

fn v2_server_material(timestamp_ms: u64, client_nonce: &str, server_nonce: &str) -> String {
    format!(
        "BADAPPLE-SLICKS/{SLICKS_VERSION_2}|server|{timestamp_ms}|{client_nonce}|{server_nonce}"
    )
}

fn v2_client_material(
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    prompt: &str,
    max_new_tokens: usize,
) -> String {
    let prompt_hash = Sha256::digest(prompt.as_bytes());
    format!(
        "BADAPPLE-SLICKS/{SLICKS_VERSION_2}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{}",
        hex_encode(&prompt_hash)
    )
}

/// Client to the long-lived identity agent that owns the Secure Enclave context.
pub struct IdentityAgentClient {
    socket_path: PathBuf,
}

impl IdentityAgentClient {
    pub fn new() -> Self {
        Self {
            socket_path: identity_agent_socket_path(),
        }
    }

    pub fn from_path<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            socket_path: path.into(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.socket_path.exists()
    }

    fn call(&self, request: Value) -> Result<Value> {
        if !self.is_available() {
            bail!("identity agent socket not found");
        }
        let stream = UnixStream::connect(&self.socket_path)
            .context("unable to connect to identity agent")?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        let mut writer = stream.try_clone()?;
        let mut reader = BufReader::new(stream);
        let mut frame = serde_json::to_vec(&request)?;
        frame.push(b'\n');
        writer.write_all(&frame)?;
        writer.flush()?;
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            bail!("identity agent closed the connection");
        }
        serde_json::from_str(&line).context("invalid identity agent response")
    }

    pub fn public_key(&self) -> Result<String> {
        let resp = self.call(serde_json::json!({"command": "public_key"}))?;
        if !resp
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "identity agent public_key failed: {}",
                resp.get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
        }
        resp.get("public_key")
            .and_then(|v| v.as_str())
            .map(String::from)
            .context("identity agent did not return a public key")
    }

    pub fn sign(&self, message_b64: &str) -> Result<String> {
        let resp = self.call(serde_json::json!({
            "command": "sign",
            "message_b64": message_b64,
        }))?;
        if !resp
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "identity agent sign failed: {}",
                resp.get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
        }
        resp.get("signature")
            .and_then(|v| v.as_str())
            .map(String::from)
            .context("identity agent did not return a signature")
    }

    pub fn verify(&self, message_b64: &str, signature: &str, public_key: &str) -> Result<bool> {
        let resp = self.call(serde_json::json!({
            "command": "verify",
            "message_b64": message_b64,
            "signature": signature,
            "public_key": public_key,
        }))?;
        if !resp
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!(
                "identity agent verify failed: {}",
                resp.get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
        }
        Ok(resp
            .get("valid")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false))
    }

    pub fn status(&self) -> Result<Value> {
        self.call(serde_json::json!({"command": "status"}))
    }
}

pub fn v2_client_proof(
    agent: &IdentityAgentClient,
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<String> {
    let material = v2_client_material(
        timestamp_ms,
        client_nonce,
        server_nonce,
        prompt,
        max_new_tokens,
    );
    let material_b64 = general_purpose::STANDARD.encode(material.as_bytes());
    agent.sign(&material_b64)
}

pub fn v2_verify_server_proof(
    agent: &IdentityAgentClient,
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    proof: &str,
    server_pubkey: &str,
) -> Result<bool> {
    // SECURITY: Verify the server's public key against a pinned trust store
    // before accepting the Challenge.  This prevents MITM attacks where an
    // attacker runs a fake daemon on a hijacked socket path and presents
    // their own keypair.
    if let Some(pinned) = v2_pinned_server_pubkey()? {
        if pinned != server_pubkey {
            bail!("SLICKS v2 server public key does not match the pinned trust store");
        }
    }
    let material = v2_server_material(timestamp_ms, client_nonce, server_nonce);
    let material_b64 = general_purpose::STANDARD.encode(material.as_bytes());
    agent.verify(&material_b64, proof, server_pubkey)
}

/// Load the pinned server public key from the trust store, if it exists.
///
/// The trust store lives at `/var/lib/bad_apple/keys/daemon.pub` (or the path
/// in `BADAPPLE_SERVER_KEY_PATH`) and contains a base64-encoded P-256 public
/// key.  If the file does not exist, key pinning is disabled (fail-open) —
/// the first connection writes the key so subsequent connections can verify
/// it.  This is TOFU (trust on first use).
pub fn v2_pinned_server_pubkey() -> Result<Option<String>> {
    let path = std::env::var_os("BADAPPLE_SERVER_KEY_PATH").map_or_else(
        || PathBuf::from("/var/lib/bad_apple/keys/daemon.pub"),
        PathBuf::from,
    );
    if !path.is_file() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("unable to read pinned server key at {}", path.display()))?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed))
}

/// Pin a server public key to the trust store (TOFU on first connection).
pub fn v2_pin_server_pubkey(pubkey: &str) -> Result<()> {
    let path = std::env::var_os("BADAPPLE_SERVER_KEY_PATH").map_or_else(
        || PathBuf::from("/var/lib/bad_apple/keys/daemon.pub"),
        PathBuf::from,
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, pubkey)?;
    // Set strict permissions on the trust store
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))?;
    }
    Ok(())
}

enum SlicksMode {
    V1(Vec<u8>),
    V2 {
        agent: IdentityAgentClient,
        client_pubkey: String,
    },
}

fn resolve_slicks_mode() -> Result<SlicksMode> {
    let agent = IdentityAgentClient::new();
    let agent_pubkey = if agent.is_available() {
        agent.public_key().ok()
    } else {
        None
    };

    match std::env::var("BADAPPLE_SLICKS2").as_deref() {
        Ok("0" | "false" | "no" | "off" | "disable" | "disabled") => {
            Ok(SlicksMode::V1(load_slicks_secret()?))
        }
        Ok("1" | "true" | "yes" | "on" | "enable" | "enabled") => {
            let client_pubkey =
                agent_pubkey.context("BADAPPLE_SLICKS2=1 requires a running identity agent")?;
            Ok(SlicksMode::V2 {
                agent,
                client_pubkey,
            })
        }
        Ok(other) => {
            // Reject unknown explicit values rather than silently forcing v2.
            bail!(
                "BADAPPLE_SLICKS2 has unrecognized value '{}'; expected one of \
                 0/1/false/true/no/yes/off/on/disable/enable/disabled/enabled",
                other
            );
        }
        Err(_) => {
            // Auto: prefer v2 when the identity agent is present, otherwise v1.
            if let Some(client_pubkey) = agent_pubkey {
                return Ok(SlicksMode::V2 {
                    agent,
                    client_pubkey,
                });
            }
            Ok(SlicksMode::V1(load_slicks_secret()?))
        }
    }
}

fn build_hello(mode: &SlicksMode, timestamp_ms: u64, client_nonce: String) -> Result<ClientFrame> {
    match mode {
        SlicksMode::V1(_) => Ok(ClientFrame::Hello {
            version: SLICKS_VERSION,
            timestamp_ms,
            client_nonce,
            client_pubkey: None,
        }),
        SlicksMode::V2 { client_pubkey, .. } => Ok(ClientFrame::Hello {
            version: SLICKS_VERSION_2,
            timestamp_ms,
            client_nonce,
            client_pubkey: Some(client_pubkey.clone()),
        }),
    }
}

fn build_execute(
    mode: &SlicksMode,
    timestamp_ms: u64,
    client_nonce: String,
    server_nonce: String,
    prompt: String,
    max_new_tokens: usize,
) -> Result<ClientFrame> {
    match mode {
        SlicksMode::V1(secret) => {
            let proof = client_proof(
                secret,
                timestamp_ms,
                &client_nonce,
                &server_nonce,
                &prompt,
                max_new_tokens,
            );
            Ok(ClientFrame::Execute {
                version: SLICKS_VERSION,
                timestamp_ms,
                client_nonce,
                server_nonce,
                prompt,
                max_new_tokens,
                proof,
                client_pubkey: None,
            })
        }
        SlicksMode::V2 {
            agent,
            client_pubkey,
        } => {
            let proof = v2_client_proof(
                agent,
                timestamp_ms,
                &client_nonce,
                &server_nonce,
                &prompt,
                max_new_tokens,
            )?;
            Ok(ClientFrame::Execute {
                version: SLICKS_VERSION_2,
                timestamp_ms,
                client_nonce,
                server_nonce,
                prompt,
                max_new_tokens,
                proof,
                client_pubkey: Some(client_pubkey.clone()),
            })
        }
    }
}

pub fn stream_query<F>(prompt: &str, max_new_tokens: usize, on_token: F) -> Result<String>
where
    F: FnMut(&str),
{
    query_with_metrics(prompt, max_new_tokens, on_token).map(|(text, _)| text)
}

pub fn call_agent(method: &str, params: Option<Value>, max_new_tokens: usize) -> Result<Value> {
    let mut req = serde_json::Map::new();
    req.insert("id".to_string(), Value::String("cli-1".to_string()));
    req.insert("method".to_string(), Value::String(method.to_string()));
    if let Some(p) = params {
        req.insert("params".to_string(), p);
    }
    let prompt = format!(
        "__BADAPPLE_AGENT__ {}",
        serde_json::to_string(&Value::Object(req))?
    );
    let (text, _) = query_with_metrics(&prompt, max_new_tokens, |_token| {})?;
    match serde_json::from_str(&text) {
        Ok(value) => Ok(value),
        Err(_) => {
            // Non-JSON or raw text fallback.
            Ok(Value::String(text))
        }
    }
}

pub fn query_with_metrics<F>(
    prompt: &str,
    max_new_tokens: usize,
    mut on_token: F,
) -> Result<(String, Option<Metrics>)>
where
    F: FnMut(&str),
{
    validate_request(prompt, max_new_tokens)?;
    let mode = resolve_slicks_mode()?;
    let timestamp_ms = now_unix_ms()?;
    let client_nonce = random_nonce();
    let mut stream =
        UnixStream::connect(socket_path()).context("unable to connect to Bad Apple")?;
    stream.set_read_timeout(Some(Duration::from_mins(15)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);

    let hello = build_hello(&mode, timestamp_ms, client_nonce.clone())?;
    write_frame(&mut stream, &hello)?;

    let challenge: ServerFrame = read_frame(&mut reader)?;
    let server_nonce = match (challenge, &mode) {
        (
            ServerFrame::Challenge {
                version,
                server_nonce,
                proof,
                server_pubkey: _,
            },
            SlicksMode::V1(secret),
        ) if version == SLICKS_VERSION && nonce_is_valid(&server_nonce) => {
            if !verify_server_proof(secret, timestamp_ms, &client_nonce, &server_nonce, &proof) {
                bail!("Bad Apple daemon failed SLICKS server authentication");
            }
            server_nonce
        }
        (
            ServerFrame::Challenge {
                version,
                server_nonce,
                proof,
                server_pubkey,
            },
            SlicksMode::V2 { agent, .. },
        ) if version == SLICKS_VERSION_2 && nonce_is_valid(&server_nonce) => {
            let server_pubkey = server_pubkey
                .as_ref()
                .context("SLICKS v2 challenge missing server public key")?;
            // TOFU: if no pinned key exists, pin this one on first use.
            // If a pinned key exists, v2_verify_server_proof checks it.
            if v2_pinned_server_pubkey()?.is_none() {
                if let Err(e) = v2_pin_server_pubkey(server_pubkey) {
                    eprintln!("[badapple] failed to pin server public key: {e}");
                }
            }
            if !v2_verify_server_proof(
                agent,
                timestamp_ms,
                &client_nonce,
                &server_nonce,
                &proof,
                server_pubkey,
            )? {
                bail!("Bad Apple daemon failed SLICKS v2 server authentication");
            }
            server_nonce
        }
        (ServerFrame::Error { message }, _) => bail!("Bad Apple rejected the handshake: {message}"),
        _ => bail!("Bad Apple returned an invalid SLICKS challenge"),
    };

    let execute = build_execute(
        &mode,
        timestamp_ms,
        client_nonce,
        server_nonce,
        prompt.to_string(),
        max_new_tokens,
    )?;
    write_frame(&mut stream, &execute)?;

    let mut accepted = false;
    loop {
        match read_frame::<_, ServerFrame>(&mut reader)? {
            ServerFrame::Accepted => accepted = true,
            ServerFrame::Token { text } if accepted => on_token(&text),
            ServerFrame::Done { text, metrics } if accepted => return Ok((text, metrics)),
            ServerFrame::Response { result } if accepted => {
                let text = serde_json::to_string(&result)?;
                return Ok((text, None));
            }
            ServerFrame::Error { message } => bail!("Bad Apple request failed: {message}"),
            _ => bail!("Bad Apple returned an out-of-order IPC frame"),
        }
    }
}

pub fn write_frame<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: Write,
    T: Serialize,
{
    let mut frame = serde_json::to_vec(value)?;
    if frame.len() > MAX_FRAME_BYTES {
        bail!("Bad Apple IPC frame exceeds the size limit");
    }
    frame.push(b'\n');
    writer.write_all(&frame)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame<R, T>(reader: &mut R) -> Result<T>
where
    R: BufRead,
    T: DeserializeOwned,
{
    let mut line = String::new();
    let mut limited = Read::take(reader, (MAX_FRAME_BYTES + 1) as u64);
    let read = limited.read_line(&mut line)?;
    if read == 0 {
        bail!("Bad Apple IPC connection closed");
    }
    if read > MAX_FRAME_BYTES || !line.ends_with('\n') {
        bail!("Bad Apple IPC frame exceeds the size limit or is unterminated");
    }
    serde_json::from_str(&line).context("invalid Bad Apple IPC frame")
}

pub async fn write_async_frame<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let mut frame = serde_json::to_vec(value)?;
    if frame.len() > MAX_FRAME_BYTES {
        bail!("Bad Apple IPC frame exceeds the size limit");
    }
    frame.push(b'\n');
    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read_async_frame<R, T>(reader: &mut R) -> Result<T>
where
    R: AsyncBufRead + Unpin,
    T: DeserializeOwned,
{
    let mut line = String::new();
    let mut limited = AsyncReadExt::take(reader, (MAX_FRAME_BYTES + 1) as u64);
    let read = limited.read_line(&mut line).await?;
    if read == 0 {
        bail!("Bad Apple IPC connection closed");
    }
    if read > MAX_FRAME_BYTES || !line.ends_with('\n') {
        bail!("Bad Apple IPC frame exceeds the size limit or is unterminated");
    }
    serde_json::from_str(&line).context("invalid Bad Apple IPC frame")
}

fn client_material(
    timestamp_ms: u64,
    client_nonce: &str,
    server_nonce: &str,
    prompt: &str,
    max_new_tokens: usize,
) -> String {
    let prompt_hash = Sha256::digest(prompt.as_bytes());
    format!(
        "BADAPPLE-SLICKS/{SLICKS_VERSION}|client|{timestamp_ms}|{client_nonce}|{server_nonce}|{max_new_tokens}|{}",
        hex_encode(&prompt_hash)
    )
}

fn sign(secret: &[u8], material: &[u8]) -> String {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(material);
    hex_encode(&mac.finalize().into_bytes())
}

fn verify(secret: &[u8], material: &[u8], proof: &str) -> bool {
    type HmacSha256 = Hmac<Sha256>;
    let Some(proof) = decode_hex(proof) else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret) else {
        return false;
    };
    mac.update(material);
    mac.verify_slice(&proof).is_ok()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(pair, 16).ok()
        })
        .collect()
}

pub fn unexpected_frame(frame: &ClientFrame) -> anyhow::Error {
    anyhow!("unexpected Bad Apple client frame: {frame:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicks_handshake_proofs_are_bound_to_prompt_and_nonces() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000;
        let client_nonce = "client";
        let server_nonce = "server";
        let server = server_proof(secret, timestamp, client_nonce, server_nonce);
        assert!(verify_server_proof(
            secret,
            timestamp,
            client_nonce,
            server_nonce,
            &server
        ));
        assert!(!verify_server_proof(
            secret,
            timestamp,
            client_nonce,
            "other",
            &server
        ));

        let client = client_proof(secret, timestamp, client_nonce, server_nonce, "hello", 32);
        assert!(verify_client_proof(
            secret,
            timestamp,
            client_nonce,
            server_nonce,
            "hello",
            32,
            &client
        ));
        assert!(!verify_client_proof(
            secret,
            timestamp,
            client_nonce,
            server_nonce,
            "tampered",
            32,
            &client
        ));
    }

    #[test]
    fn request_validation_rejects_empty_and_oversized_values() {
        assert!(validate_request("hello", 1).is_ok());
        assert!(validate_request("", 1).is_err());
        assert!(validate_request("hello", 0).is_err());
        assert!(validate_request("hello", MAX_NEW_TOKENS + 1).is_err());
    }

    #[test]
    fn nonce_validation_rejects_invalid_nonces() {
        assert!(nonce_is_valid(&random_nonce()));
        assert!(!nonce_is_valid(""));
        assert!(!nonce_is_valid("short"));
        assert!(!nonce_is_valid(
            "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg"
        ));
    }

    #[test]
    fn timestamp_freshness_rejects_old_and_future() {
        let now = now_unix_ms().unwrap();
        assert!(timestamp_is_fresh(now));
        assert!(!timestamp_is_fresh(now - 60_000));
        assert!(!timestamp_is_fresh(now + 60_000));
        assert!(timestamp_is_fresh(now - 10_000));
    }

    #[test]
    fn verify_rejects_tampered_proof() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let material = b"test material";
        let proof = sign(secret, material);
        assert!(verify(secret, material, &proof));
        // Tamper with proof — flip first hex char
        let mut tampered_bytes = proof.as_bytes().to_vec();
        tampered_bytes[0] = if tampered_bytes[0] == b'a' {
            b'b'
        } else {
            b'a'
        };
        let tampered = String::from_utf8(tampered_bytes).unwrap();
        assert!(!verify(secret, material, &tampered));
        // Wrong secret
        assert!(!verify(b"wrongsecret123456", material, &proof));
    }

    #[test]
    fn decode_hex_rejects_odd_and_invalid() {
        assert!(decode_hex("abcd").is_some());
        assert!(decode_hex("abc").is_none());
        assert!(decode_hex("").is_none());
        assert!(decode_hex("xy").is_none());
    }

    // =========================================================================
    // Security regression tests — red team findings
    // =========================================================================

    /// Verify that a v1 server_proof signed with one version cannot be verified
    /// with a different version.  The proof material embeds the SLICKS version,
    /// so a version mismatch must invalidate the MAC.
    #[test]
    fn v1_server_proof_rejects_wrong_version() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000;
        let client_nonce = "a".repeat(64);
        let server_nonce = "b".repeat(64);

        // Craft a proof with SLICKS_VERSION_2 in the material string.
        let wrong_material = format!(
            "BADAPPLE-SLICKS/{SLICKS_VERSION_2}|server|{timestamp}|{client_nonce}|{server_nonce}"
        );
        let proof = sign(secret, wrong_material.as_bytes());

        // verify_server_proof uses SLICKS_VERSION (v1), so the version mismatch
        // must cause verification to fail.
        assert!(!verify_server_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            &proof
        ));

        // A correct v1 proof should verify.
        let good_proof = server_proof(secret, timestamp, &client_nonce, &server_nonce);
        assert!(verify_server_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            &good_proof
        ));
    }

    /// Verify that changing max_new_tokens invalidates the client proof.
    #[test]
    fn v1_client_proof_is_bound_to_max_tokens() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000;
        let client_nonce = "a".repeat(64);
        let server_nonce = "b".repeat(64);

        let proof = client_proof(secret, timestamp, &client_nonce, &server_nonce, "hello", 32);

        // Different max_new_tokens must fail.
        assert!(!verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            "hello",
            33,
            &proof
        ));

        // Original max_new_tokens must pass.
        assert!(verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            "hello",
            32,
            &proof
        ));
    }

    /// Verify that changing the server_nonce invalidates the client proof.
    #[test]
    fn v1_client_proof_is_bound_to_server_nonce() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000;
        let client_nonce = "a".repeat(64);
        let server_nonce = "b".repeat(64);

        let proof = client_proof(secret, timestamp, &client_nonce, &server_nonce, "hello", 32);

        // Different server_nonce must fail.
        assert!(!verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &"c".repeat(64),
            "hello",
            32,
            &proof
        ));

        // Original server_nonce must pass.
        assert!(verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            "hello",
            32,
            &proof
        ));
    }

    /// Verify TOFU returns None when no trust store file exists.
    #[test]
    fn v2_pinned_server_pubkey_returns_none_when_missing() {
        // Point to a path in a nonexistent directory.
        std::env::set_var(
            "BADAPPLE_SERVER_KEY_PATH",
            "/tmp/bad_apple_ipc_test_nonexistent_999999/key.pub",
        );
        let result = v2_pinned_server_pubkey().unwrap();
        assert!(
            result.is_none(),
            "expected None when trust store is missing"
        );
        std::env::remove_var("BADAPPLE_SERVER_KEY_PATH");
    }

    /// Pin a server public key, then verify it is returned by v2_pinned_server_pubkey.
    #[test]
    fn v2_pin_and_verify_server_pubkey() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let key_path = std::env::temp_dir().join(format!(
            "bad_apple_ipc_test_key_{}_{}.pub",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&key_path);

        std::env::set_var("BADAPPLE_SERVER_KEY_PATH", &key_path);

        // Before pinning, no key should be present.
        assert!(v2_pinned_server_pubkey().unwrap().is_none());

        // Pin a key.
        let pubkey = "dGVzdF9wdWJsaWNfa2V5X2Jhc2U2NA==";
        v2_pin_server_pubkey(pubkey).unwrap();

        // After pinning, the key should be returned.
        let pinned = v2_pinned_server_pubkey().unwrap();
        assert_eq!(pinned.as_deref(), Some(pubkey));

        let _ = std::fs::remove_file(&key_path);
        std::env::remove_var("BADAPPLE_SERVER_KEY_PATH");
    }

    /// Verify that nonce_is_valid rejects a 64-character string with non-hex chars.
    #[test]
    fn nonce_is_valid_rejects_non_hex() {
        let non_hex = "ZZ".repeat(32); // 64 chars but 'Z' is not a hex digit
        assert_eq!(non_hex.len(), 64);
        assert!(!nonce_is_valid(&non_hex));
    }

    /// Verify the timestamp freshness boundary: exactly at the max skew is
    /// fresh, one ms beyond is not.
    #[test]
    fn timestamp_is_fresh_boundary() {
        let now = now_unix_ms().unwrap();
        let skew = HANDSHAKE_MAX_SKEW.as_millis() as u64;

        // Exactly at the skew boundary (30000ms) should be fresh.
        assert!(timestamp_is_fresh(now - skew));
        assert!(timestamp_is_fresh(now + skew));

        // One ms beyond the boundary (30001ms in the past) should not be fresh.
        assert!(!timestamp_is_fresh(now - skew - 1));
    }

    /// The replay cache must accept a fresh nonce pair and then reject a
    /// verbatim replay of the same pair.
    #[test]
    fn replay_cache_rejects_repeated_nonce_pair() {
        let cache = ReplayCache::new(100);
        let client_nonce = random_nonce();
        let server_nonce = random_nonce();

        assert!(
            cache.check_and_insert(&client_nonce, &server_nonce),
            "first use of a nonce pair must be accepted"
        );
        assert!(
            !cache.check_and_insert(&client_nonce, &server_nonce),
            "verbatim replay of the same pair must be rejected"
        );

        // A different pair is still accepted.
        let other_client = random_nonce();
        assert!(cache.check_and_insert(&other_client, &server_nonce));
    }

    /// When the replay cache hits its configured bound it must evict ONLY the
    /// oldest entry — flooding the cache must not make recent nonces
    /// replayable.
    #[test]
    fn replay_cache_evicts_oldest_at_capacity() {
        let cache = ReplayCache::new(2);
        let n1 = random_nonce();
        let n2 = random_nonce();
        let n3 = random_nonce();

        assert!(cache.check_and_insert(&n1, &n2));
        assert!(cache.check_and_insert(&n2, &n3));
        // Third insert evicts the oldest pair (n1, n2) and succeeds.
        assert!(cache.check_and_insert(&n3, &n1));
        // The evicted pair may be reinserted, but the still-cached pair must
        // still be rejected — the cache was not wiped.
        assert!(!cache.check_and_insert(&n2, &n3));
        // The oldest pair was evicted, so it is fresh again.
        assert!(cache.check_and_insert(&n1, &n2));
    }

    /// Reusing a server nonce with a different client nonce must produce a
    /// fresh, distinct proof and still verify.
    #[test]
    fn v1_client_proof_is_distinct_per_client_nonce() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let timestamp = 1_700_000_000_000;
        let client_nonce = random_nonce();
        let server_nonce = random_nonce();
        let prompt = "hello";

        let proof1 = client_proof(secret, timestamp, &client_nonce, &server_nonce, prompt, 64);
        assert!(verify_client_proof(
            secret,
            timestamp,
            &client_nonce,
            &server_nonce,
            prompt,
            64,
            &proof1
        ));

        let client_nonce2 = random_nonce();
        let proof2 = client_proof(secret, timestamp, &client_nonce2, &server_nonce, prompt, 64);
        assert!(verify_client_proof(
            secret,
            timestamp,
            &client_nonce2,
            &server_nonce,
            prompt,
            64,
            &proof2
        ));
        assert_ne!(
            proof1, proof2,
            "different client nonce must produce different proof"
        );
    }
}
