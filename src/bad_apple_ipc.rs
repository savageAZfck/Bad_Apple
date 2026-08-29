use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
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
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET_PATH))
}

pub fn mlx_socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_MLX_SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MLX_SOCKET_PATH))
}

pub fn identity_agent_socket_path() -> PathBuf {
    std::env::var_os("BADAPPLE_IDENTITY_AGENT_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_IDENTITY_AGENT_SOCKET))
}

pub fn key_path() -> PathBuf {
    std::env::var_os("BADAPPLE_SLICKS_KEY_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_KEY_PATH))
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
        if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
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
        if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
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
        if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            bail!(
                "identity agent verify failed: {}",
                resp.get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
            );
        }
        Ok(resp.get("valid").and_then(|v| v.as_bool()).unwrap_or(false))
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
    let material = v2_server_material(timestamp_ms, client_nonce, server_nonce);
    let material_b64 = general_purpose::STANDARD.encode(material.as_bytes());
    agent.verify(&material_b64, proof, server_pubkey)
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
        Ok("0" | "false" | "no") => {
            return Ok(SlicksMode::V1(load_slicks_secret()?));
        }
        Ok("1" | "true" | "yes") | Ok(_) => {
            let client_pubkey =
                agent_pubkey.context("BADAPPLE_SLICKS2=1 requires a running identity agent")?;
            return Ok(SlicksMode::V2 {
                agent,
                client_pubkey,
            });
        }
        Err(_) => {
            // Auto: prefer v2 when the identity agent is present, otherwise v1.
            if let Some(client_pubkey) = agent_pubkey {
                return Ok(SlicksMode::V2 {
                    agent,
                    client_pubkey,
                });
            }
            return Ok(SlicksMode::V1(load_slicks_secret()?));
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
    stream.set_read_timeout(Some(Duration::from_secs(900)))?;
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
}
