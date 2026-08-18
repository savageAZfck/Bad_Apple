use anyhow::{anyhow, bail, Context, Result};
use hmac::{Hmac, KeyInit, Mac};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const SLICKS_VERSION: u8 = 1;
pub const DEFAULT_SOCKET_PATH: &str = "/var/run/badapple/substrate.sock";
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
    },
    Execute {
        version: u8,
        timestamp_ms: u64,
        client_nonce: String,
        server_nonce: String,
        prompt: String,
        max_new_tokens: usize,
        proof: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerFrame {
    Challenge {
        version: u8,
        server_nonce: String,
        proof: String,
    },
    Accepted,
    Token {
        text: String,
    },
    Done {
        text: String,
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

pub fn stream_query<F>(prompt: &str, max_new_tokens: usize, mut on_token: F) -> Result<String>
where
    F: FnMut(&str),
{
    validate_request(prompt, max_new_tokens)?;
    let secret = load_slicks_secret()?;
    let timestamp_ms = now_unix_ms()?;
    let client_nonce = random_nonce();
    let mut stream =
        UnixStream::connect(socket_path()).context("unable to connect to Bad Apple")?;
    stream.set_read_timeout(Some(Duration::from_secs(900)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);

    write_frame(
        &mut stream,
        &ClientFrame::Hello {
            version: SLICKS_VERSION,
            timestamp_ms,
            client_nonce: client_nonce.clone(),
        },
    )?;

    let challenge: ServerFrame = read_frame(&mut reader)?;
    let (server_nonce, proof) = match challenge {
        ServerFrame::Challenge {
            version,
            server_nonce,
            proof,
        } if version == SLICKS_VERSION && nonce_is_valid(&server_nonce) => (server_nonce, proof),
        ServerFrame::Error { message } => bail!("Bad Apple rejected the handshake: {message}"),
        _ => bail!("Bad Apple returned an invalid SLICKS challenge"),
    };
    if !verify_server_proof(&secret, timestamp_ms, &client_nonce, &server_nonce, &proof) {
        bail!("Bad Apple daemon failed SLICKS server authentication");
    }

    let proof = client_proof(
        &secret,
        timestamp_ms,
        &client_nonce,
        &server_nonce,
        prompt,
        max_new_tokens,
    );
    write_frame(
        &mut stream,
        &ClientFrame::Execute {
            version: SLICKS_VERSION,
            timestamp_ms,
            client_nonce,
            server_nonce,
            prompt: prompt.to_string(),
            max_new_tokens,
            proof,
        },
    )?;

    let mut accepted = false;
    loop {
        match read_frame::<_, ServerFrame>(&mut reader)? {
            ServerFrame::Accepted => accepted = true,
            ServerFrame::Token { text } if accepted => on_token(&text),
            ServerFrame::Done { text } if accepted => return Ok(text),
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
