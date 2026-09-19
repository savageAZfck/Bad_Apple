//! Encrypted P2P model file transfer.
//!
//! This is the Rust-native replacement for the Python `badapple_p2p.py` model
//! pull/send/receive. It runs *alongside* the engram gossip mesh in
//! `protocol.rs` but uses a separate reliable TCP stream for file chunks,
//! because multi-GB model weights do not fit inside compact engrams.
//!
//! Protocol (all frames are length-prefixed and encrypted with the SLICKS/P2P
//! AES-256-GCM cipher):
//!
//! 1. Puller connects to sender's transfer port.
//! 2. Puller -> Sender: `Request { model_id, resume_chunks }`.
//! 3. Sender -> Puller: `Offer { model_id, file_name, total_bytes,
//!    total_chunks, chunk_size, sha256 }` or `Error`.
//! 4. Puller -> Sender: `Accept`.
//! 5. Sender -> Puller: `Chunk { index, total, bytes }`.
//! 6. Puller -> Sender: `Ack { index }` after each chunk.
//! 7. Sender -> Puller: `Done`.
//! 8. Puller verifies full SHA-256 and stores the file. Sender is done.
//!
//! If the puller has a partial file, it can resume by sending the set of
//! already-received chunk indices in `resume_chunks`.

use crate::p2p_crypto::P2PCipher;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// Default chunk size for model transfers.
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024; // 256 KiB

/// Maximum model file size we will transfer (80 GiB).
pub const MAX_MODEL_BYTES: u64 = 80 * 1024 * 1024 * 1024;

/// Encrypted, length-prefixed frame on the wire.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TransferFrame {
    Request(RequestFrame),
    PushRequest(PushRequestFrame),
    PushAccept,
    PushReject {
        reason: String,
    },
    Offer(OfferFrame),
    Accept,
    Reject {
        reason: String,
    },
    Chunk(ChunkFrame),
    Ack(AckFrame),
    Done,
    Error {
        message: String,
    },
    Cancel,
    /// Delegated inference: ask the peer's loaded engine to answer a prompt.
    /// Serving is opt-in per peer (`BADAPPLE_P2P_INFER=1`) and inference-only —
    /// the serving daemon bypasses meta/tool/approval routing entirely.
    InferRequest(InferRequestFrame),
    InferResponse(InferResponseFrame),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RequestFrame {
    pub model_id: String,
    #[serde(default)]
    pub resume_chunks: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PushRequestFrame {
    pub model_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OfferFrame {
    pub model_id: String,
    pub file_name: String,
    pub total_bytes: u64,
    pub total_chunks: usize,
    pub chunk_size: usize,
    pub sha256: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChunkFrame {
    pub index: usize,
    pub total: usize,
    pub bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AckFrame {
    pub index: usize,
}

/// Largest prompt a peer may delegate to this machine (32 KiB).
pub const MAX_DELEGATED_PROMPT_BYTES: usize = 32 * 1024;

/// Hard cap on tokens generated for a delegated request.
pub const MAX_DELEGATED_TOKENS: usize = 2048;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferRequestFrame {
    pub request_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    /// Requester's peer label, recorded in the serving machine's ledger.
    pub from_peer: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferResponseFrame {
    pub request_id: String,
    pub text: String,
    /// Model tier that served the request, when reported.
    pub tier: Option<String>,
    pub elapsed_ms: u64,
}

/// Manifest for a model that a peer can advertise/gossip.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelManifest {
    pub model_id: String,
    pub file_name: String,
    pub total_bytes: u64,
    pub sha256: String,
    pub origin_peer: String,
    pub chunk_size: usize,
}

/// A running P2P model transfer endpoint.
pub struct P2PModelTransfer {
    secret: Vec<u8>,
    model_dir: PathBuf,
    advertised: Arc<Mutex<Vec<ModelManifest>>>,
}

impl P2PModelTransfer {
    pub fn new(secret: Vec<u8>, model_dir: PathBuf) -> Self {
        Self {
            secret,
            model_dir,
            advertised: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Load the local model directory, compute SHA-256 for each file, and
    /// advertise the resulting manifests.
    pub async fn scan_and_advertise(&self, origin_peer: impl Into<String>) -> Result<()> {
        let origin = origin_peer.into();
        let mut manifests = Vec::new();
        if !self.model_dir.is_dir() {
            return Ok(());
        }
        for entry in fs::read_dir(&self.model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let meta = fs::metadata(&path)?;
            if meta.len() > MAX_MODEL_BYTES {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            let model_id = file_name_to_model_id(&file_name);
            let sha256 = sha256_file(&path)?;
            manifests.push(ModelManifest {
                model_id,
                file_name,
                total_bytes: meta.len(),
                sha256,
                origin_peer: origin.clone(),
                chunk_size: DEFAULT_CHUNK_SIZE,
            });
        }
        *self.advertised.lock().await = manifests;
        Ok(())
    }

    pub async fn advertised_manifests(&self) -> Vec<ModelManifest> {
        self.advertised.lock().await.clone()
    }

    pub fn cipher(&self) -> Result<P2PCipher> {
        let key = derive_key_from_secret(&self.secret);
        P2PCipher::new(&key).map_err(|e| anyhow!("failed to derive model transfer cipher: {e}"))
    }

    /// Start a TCP server and accept one or many model transfer connections.
    /// Returns the bound address and a join handle for the accept loop.
    pub async fn serve(
        &self,
        port: u16,
    ) -> Result<(std::net::SocketAddr, tokio::task::JoinHandle<Result<()>>)> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .with_context(|| format!("failed to bind P2P model transfer port {port}"))?;
        let addr = listener
            .local_addr()
            .context("failed to get local model transfer address")?;
        let this = self.clone();
        let handle = tokio::spawn(async move {
            loop {
                let (stream, peer) = listener
                    .accept()
                    .await
                    .context("failed to accept model transfer connection")?;
                let this = this.clone();
                tokio::spawn(async move {
                    if let Err(e) = this.handle_incoming(stream, peer.to_string()).await {
                        tracing::warn!("P2P model transfer from {peer} failed: {e}");
                    }
                });
            }
        });
        Ok((addr, handle))
    }

    async fn handle_incoming(&self, mut stream: TcpStream, peer: String) -> Result<()> {
        let cipher = self.cipher()?;

        let first = recv_frame(&mut stream, &cipher).await?;
        match first {
            TransferFrame::Request(r) => self.handle_pull_request(&mut stream, &cipher, r).await,
            TransferFrame::PushRequest(p) => {
                self.handle_push_receive(&mut stream, &cipher, p).await
            }
            TransferFrame::InferRequest(r) => {
                self.handle_infer_request(&mut stream, &cipher, r, &peer)
                    .await
            }
            other => bail!("expected Request, PushRequest, or InferRequest frame, got {other:?}"),
        }
    }

    /// Serve a delegated inference request through the local engine's
    /// delegated-only path. Disabled unless `BADAPPLE_P2P_INFER=1` — peers may
    /// transfer weights by default, but borrowing this machine's brain is an
    /// explicit opt-in. The serving daemon ledger-records the query under the
    /// requester's peer label and never routes it to tools or approvals.
    async fn handle_infer_request<S: AsyncRead + AsyncWrite + Unpin>(
        &self,
        stream: &mut S,
        cipher: &P2PCipher,
        request: InferRequestFrame,
        peer: &str,
    ) -> Result<()> {
        if std::env::var("BADAPPLE_P2P_INFER").ok().as_deref() != Some("1") {
            send_frame(
                stream,
                cipher,
                TransferFrame::Error {
                    message: "this peer does not serve delegated inference \
                              (set BADAPPLE_P2P_INFER=1 to enable)"
                        .to_string(),
                },
            )
            .await?;
            bail!("delegated inference refused — serving disabled on this peer");
        }
        if request.prompt.is_empty() || request.prompt.len() > MAX_DELEGATED_PROMPT_BYTES {
            send_frame(
                stream,
                cipher,
                TransferFrame::Error {
                    message: format!("prompt empty or exceeds {MAX_DELEGATED_PROMPT_BYTES} bytes"),
                },
            )
            .await?;
            bail!("delegated prompt rejected: {} bytes", request.prompt.len());
        }
        let max_tokens = request.max_tokens.clamp(1, MAX_DELEGATED_TOKENS);
        let started = std::time::Instant::now();
        let peer_tag = format!("{peer} ({})", request.from_peer);

        // The __BADAPPLE_DELEGATED__ envelope routes through the engine's
        // inference-only path — no meta commands, no tools, no approvals.
        let envelope = serde_json::json!({
            "prompt": request.prompt,
            "from_peer": request.from_peer,
            "max_tokens": max_tokens,
        });
        let daemon_prompt = format!("__BADAPPLE_DELEGATED__ {envelope}");
        let (text, metrics) = tokio::task::spawn_blocking(move || {
            crate::bad_apple_ipc::query_with_metrics(&daemon_prompt, max_tokens, |_| {})
        })
        .await
        .context("delegated inference task failed")??;
        let tier = metrics.and_then(|m| m.tier);

        send_frame(
            stream,
            cipher,
            TransferFrame::InferResponse(InferResponseFrame {
                request_id: request.request_id,
                text,
                tier,
                elapsed_ms: started.elapsed().as_millis() as u64,
            }),
        )
        .await?;
        tracing::info!("served delegated inference for {peer_tag}");
        Ok(())
    }

    async fn handle_pull_request<S: AsyncRead + AsyncWrite + Unpin>(
        &self,
        stream: &mut S,
        cipher: &P2PCipher,
        request: RequestFrame,
    ) -> Result<()> {
        // 1. Resolve the model file.
        let manifests = self.advertised.lock().await;
        let manifest = match manifests.iter().find(|m| m.model_id == request.model_id) {
            Some(m) => m.clone(),
            None => {
                send_frame(
                    stream,
                    cipher,
                    TransferFrame::Error {
                        message: format!("model {} not available from this peer", request.model_id),
                    },
                )
                .await?;
                bail!("model {} not found", request.model_id);
            }
        };
        drop(manifests);

        let path = self.model_dir.join(&manifest.file_name);
        let data =
            fs::read(&path).with_context(|| format!("failed to read model file {path:?}"))?;
        let total_chunks = data.len().div_ceil(manifest.chunk_size as usize);
        let resume: HashSet<usize> = request.resume_chunks.into_iter().collect();

        // 2. Offer.
        send_frame(
            stream,
            cipher,
            TransferFrame::Offer(OfferFrame {
                model_id: manifest.model_id,
                file_name: manifest.file_name,
                total_bytes: manifest.total_bytes,
                total_chunks,
                chunk_size: manifest.chunk_size as usize,
                sha256: manifest.sha256,
            }),
        )
        .await?;

        // 3. Wait for acceptance.
        match recv_frame(stream, cipher).await? {
            TransferFrame::Accept => {}
            TransferFrame::Reject { reason } => bail!("puller rejected offer: {reason}"),
            other => bail!("expected Accept, got {other:?}"),
        };

        // 4. Send chunks, skipping already-resumed ones.
        for index in 0..total_chunks {
            if resume.contains(&index) {
                continue;
            }
            let start = index * manifest.chunk_size as usize;
            let end = ((index + 1) * manifest.chunk_size as usize).min(data.len());
            let chunk = &data[start..end];

            send_frame(
                stream,
                cipher,
                TransferFrame::Chunk(ChunkFrame {
                    index,
                    total: total_chunks,
                    bytes: chunk.to_vec(),
                }),
            )
            .await?;

            // 5. Wait for ACK before continuing (flow control / retry).
            match recv_frame(stream, cipher).await? {
                TransferFrame::Ack(ack) if ack.index == index => {}
                other => bail!("expected Ack({index}), got {other:?}"),
            }
        }

        // 6. Done.
        send_frame(stream, cipher, TransferFrame::Done).await?;
        Ok(())
    }

    async fn handle_push_receive<S: AsyncRead + AsyncWrite + Unpin>(
        &self,
        stream: &mut S,
        cipher: &P2PCipher,
        _request: PushRequestFrame,
    ) -> Result<()> {
        // Accept the push intent.
        send_frame(stream, cipher, TransferFrame::PushAccept).await?;

        // Wait for the sender's offer.
        let offer = match recv_frame(stream, cipher).await? {
            TransferFrame::Offer(o) => o,
            TransferFrame::Error { message } => bail!("peer error: {message}"),
            other => bail!("expected Offer, got {other:?}"),
        };

        if offer.total_bytes > MAX_MODEL_BYTES {
            send_frame(
                stream,
                cipher,
                TransferFrame::Reject {
                    reason: "model is too large".to_string(),
                },
            )
            .await?;
            bail!("model is too large: {} bytes", offer.total_bytes);
        }

        // Confirm we will receive it.
        send_frame(stream, cipher, TransferFrame::Accept).await?;

        let part_path = self.model_dir.join(format!("{}.part", offer.file_name));
        fs::create_dir_all(&self.model_dir)?;

        // Initialize or open the partial file.
        let file_len = if part_path.exists() {
            fs::metadata(&part_path)?.len()
        } else {
            0
        };
        if file_len != offer.total_bytes {
            let f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&part_path)?;
            f.set_len(offer.total_bytes)?;
        }

        let mut received = HashSet::<usize>::new();
        let mut file = fs::OpenOptions::new().write(true).open(&part_path)?;
        let mut hasher = Sha256::new();

        // Receive chunks.
        loop {
            match recv_frame(stream, cipher).await? {
                TransferFrame::Chunk(chunk) => {
                    if chunk.index >= offer.total_chunks {
                        bail!("chunk index {} out of range", chunk.index);
                    }
                    let start = chunk.index * offer.chunk_size;
                    file.seek(std::io::SeekFrom::Start(start as u64))?;
                    file.write_all(&chunk.bytes)?;
                    hasher.update(&chunk.bytes);
                    received.insert(chunk.index);
                    send_frame(
                        stream,
                        cipher,
                        TransferFrame::Ack(AckFrame { index: chunk.index }),
                    )
                    .await?;
                }
                TransferFrame::Done => break,
                TransferFrame::Error { message } => bail!("peer error during transfer: {message}"),
                other => bail!("unexpected frame during push: {other:?}"),
            }
        }

        // Verify.
        if received.len() != offer.total_chunks {
            let missing: Vec<_> = (0..offer.total_chunks)
                .filter(|i| !received.contains(i))
                .collect();
            bail!("missing chunks: {missing:?}");
        }
        let actual = hex::encode(hasher.finalize());
        if actual != offer.sha256 {
            bail!(
                "SHA-256 mismatch: expected {}, got {}",
                offer.sha256,
                actual
            );
        }

        // Finalize.
        let final_path = self.model_dir.join(&offer.file_name);
        fs::rename(&part_path, &final_path)?;

        // Update the advertised manifest list so the model is now locally available.
        let manifest = ModelManifest {
            model_id: offer.model_id,
            file_name: offer.file_name,
            total_bytes: offer.total_bytes,
            sha256: offer.sha256,
            origin_peer: "local-push".to_string(),
            chunk_size: offer.chunk_size,
        };
        let mut manifests = self.advertised.lock().await;
        manifests.retain(|m| m.model_id != manifest.model_id);
        manifests.push(manifest);
        Ok(())
    }

    /// Pull a model from a remote peer. Returns the path to the downloaded file.
    pub async fn pull(
        &self,
        peer_addr: &str,
        model_id: &str,
        resume_chunks: &[usize],
        output_dir: &Path,
    ) -> Result<PathBuf> {
        let stream = TcpStream::connect(peer_addr)
            .await
            .with_context(|| format!("failed to connect to {peer_addr}"))?;
        self.pull_over_stream(stream, model_id, resume_chunks, output_dir)
            .await
    }

    async fn pull_over_stream<S>(
        &self,
        mut stream: S,
        model_id: &str,
        resume_chunks: &[usize],
        output_dir: &Path,
    ) -> Result<PathBuf>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        let cipher = self.cipher()?;

        // 1. Request.
        send_frame(
            &mut stream,
            &cipher,
            TransferFrame::Request(RequestFrame {
                model_id: model_id.to_string(),
                resume_chunks: resume_chunks.to_vec(),
            }),
        )
        .await?;

        // 2. Offer.
        let offer = match recv_frame(&mut stream, &cipher).await? {
            TransferFrame::Offer(o) => o,
            TransferFrame::Error { message } => bail!("peer error: {message}"),
            other => bail!("expected Offer, got {other:?}"),
        };

        // Validate offer.
        if offer.total_bytes > MAX_MODEL_BYTES {
            bail!("model is too large: {} bytes", offer.total_bytes);
        }

        // Accept.
        send_frame(&mut stream, &cipher, TransferFrame::Accept).await?;

        let part_path = output_dir.join(format!("{}.part", offer.file_name));
        fs::create_dir_all(output_dir)?;

        // Initialize or open the partial file.
        let file_len = if part_path.exists() {
            fs::metadata(&part_path)?.len()
        } else {
            0
        };
        if file_len != offer.total_bytes {
            // resize
            let f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&part_path)?;
            f.set_len(offer.total_bytes)?;
        }

        let mut received = std::collections::HashSet::<usize>::new();
        let mut file = fs::OpenOptions::new().write(true).open(&part_path)?;
        let mut hasher = Sha256::new();

        // Receive chunks.
        loop {
            match recv_frame(&mut stream, &cipher).await? {
                TransferFrame::Chunk(chunk) => {
                    if chunk.index >= offer.total_chunks {
                        bail!("chunk index {} out of range", chunk.index);
                    }
                    let start = chunk.index * offer.chunk_size;
                    file.seek(std::io::SeekFrom::Start(start as u64))?;
                    file.write_all(&chunk.bytes)?;
                    hasher.update(&chunk.bytes);
                    received.insert(chunk.index);
                    send_frame(
                        &mut stream,
                        &cipher,
                        TransferFrame::Ack(AckFrame { index: chunk.index }),
                    )
                    .await?;
                }
                TransferFrame::Done => break,
                TransferFrame::Error { message } => bail!("peer error during transfer: {message}"),
                other => bail!("unexpected frame during pull: {other:?}"),
            }
        }

        // Verify.
        if received.len() != offer.total_chunks {
            let missing: Vec<_> = (0..offer.total_chunks)
                .filter(|i| !received.contains(i))
                .collect();
            bail!("missing chunks: {missing:?}");
        }
        let actual = hex::encode(hasher.finalize());
        if actual != offer.sha256 {
            bail!(
                "SHA-256 mismatch: expected {}, got {}",
                offer.sha256,
                actual
            );
        }

        // Finalize.
        let final_path = output_dir.join(&offer.file_name);
        fs::rename(&part_path, &final_path)?;
        Ok(final_path)
    }

    /// Send a model to a remote peer that is already listening with `receive`.
    pub async fn push(&self, peer_addr: &str, model_id: &str) -> Result<()> {
        let path = self.find_model_file(model_id).with_context(|| {
            format!("model {model_id} not found in {}", self.model_dir.display())
        })?;

        let file_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let data =
            fs::read(&path).with_context(|| format!("failed to read model file {path:?}"))?;
        let sha256 = sha256_file(&path)?;
        let chunk_size = DEFAULT_CHUNK_SIZE;
        let total_chunks = data.len().div_ceil(chunk_size);
        let total_bytes = data.len() as u64;

        let mut stream = TcpStream::connect(peer_addr)
            .await
            .with_context(|| format!("failed to connect to {peer_addr}"))?;
        let cipher = self.cipher()?;

        // 1. Push request.
        send_frame(
            &mut stream,
            &cipher,
            TransferFrame::PushRequest(PushRequestFrame {
                model_id: model_id.to_string(),
            }),
        )
        .await?;

        // 2. Wait for acceptance.
        match recv_frame(&mut stream, &cipher).await? {
            TransferFrame::PushAccept => {}
            TransferFrame::PushReject { reason } => bail!("push rejected: {reason}"),
            TransferFrame::Reject { reason } => bail!("push rejected: {reason}"),
            other => bail!("expected PushAccept, got {other:?}"),
        };

        // 3. Offer.
        send_frame(
            &mut stream,
            &cipher,
            TransferFrame::Offer(OfferFrame {
                model_id: model_id.to_string(),
                file_name,
                total_bytes,
                total_chunks,
                chunk_size,
                sha256,
            }),
        )
        .await?;

        // 4. Wait for offer acceptance.
        match recv_frame(&mut stream, &cipher).await? {
            TransferFrame::Accept => {}
            TransferFrame::Reject { reason } => bail!("receiver rejected offer: {reason}"),
            other => bail!("expected Accept, got {other:?}"),
        };

        // 5. Send chunks.
        for index in 0..total_chunks {
            let start = index * chunk_size;
            let end = ((index + 1) * chunk_size).min(data.len());
            let chunk = &data[start..end];

            send_frame(
                &mut stream,
                &cipher,
                TransferFrame::Chunk(ChunkFrame {
                    index,
                    total: total_chunks,
                    bytes: chunk.to_vec(),
                }),
            )
            .await?;

            // 6. Wait for ACK.
            match recv_frame(&mut stream, &cipher).await? {
                TransferFrame::Ack(ack) if ack.index == index => {}
                other => bail!("expected Ack({index}), got {other:?}"),
            }
        }

        // 7. Done.
        send_frame(&mut stream, &cipher, TransferFrame::Done).await?;
        Ok(())
    }

    /// Ask a peer to run inference for us (delegated query). The serving peer
    /// must have `BADAPPLE_P2P_INFER=1` and a running engine; the response is
    /// the peer's answer text plus the tier that served it.
    pub async fn infer(
        &self,
        peer_addr: &str,
        prompt: &str,
        max_tokens: usize,
        from_peer: &str,
    ) -> Result<InferResponseFrame> {
        if prompt.is_empty() || prompt.len() > MAX_DELEGATED_PROMPT_BYTES {
            bail!("prompt empty or exceeds {MAX_DELEGATED_PROMPT_BYTES} bytes");
        }
        let cipher = self.cipher()?;
        let mut stream = TcpStream::connect(peer_addr)
            .await
            .with_context(|| format!("failed to reach peer at {peer_addr}"))?;
        let request_id = format!("{:016x}", rand::random::<u64>());
        send_frame(
            &mut stream,
            &cipher,
            TransferFrame::InferRequest(InferRequestFrame {
                request_id: request_id.clone(),
                prompt: prompt.to_string(),
                max_tokens: max_tokens.clamp(1, MAX_DELEGATED_TOKENS),
                from_peer: from_peer.to_string(),
            }),
        )
        .await?;
        match recv_frame(&mut stream, &cipher).await? {
            TransferFrame::InferResponse(r) if r.request_id == request_id => Ok(r),
            TransferFrame::InferResponse(_) => {
                bail!("infer response request_id mismatch")
            }
            TransferFrame::Error { message } => bail!("peer error: {message}"),
            other => bail!("expected InferResponse, got {other:?}"),
        }
    }

    fn find_model_file(&self, model_id: &str) -> Option<PathBuf> {
        if !self.model_dir.is_dir() {
            return None;
        }
        for entry in fs::read_dir(&self.model_dir).ok()? {
            let entry = entry.ok()?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if file_name_to_model_id(name) == model_id {
                return Some(path);
            }
        }
        None
    }
}

impl Clone for P2PModelTransfer {
    fn clone(&self) -> Self {
        Self {
            secret: self.secret.clone(),
            model_dir: self.model_dir.clone(),
            advertised: self.advertised.clone(),
        }
    }
}

/// Send a length-prefixed encrypted frame.
async fn send_frame<S>(stream: &mut S, cipher: &P2PCipher, frame: TransferFrame) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let mut plaintext = Vec::new();
    ciborium::into_writer(&frame, &mut plaintext)?;
    let ciphertext = cipher
        .encrypt(&plaintext)
        .map_err(|e| anyhow!("encrypt failed: {e}"))?;
    let len = ciphertext.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(&ciphertext).await?;
    Ok(())
}

/// Receive a length-prefixed encrypted frame.
async fn recv_frame<S>(stream: &mut S, cipher: &P2PCipher) -> Result<TransferFrame>
where
    S: AsyncRead + Unpin,
{
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len > MAX_FRAME_BYTES {
        bail!("model transfer frame exceeds size limit");
    }
    let mut ciphertext = vec![0u8; len];
    stream.read_exact(&mut ciphertext).await?;
    let plaintext = cipher
        .decrypt(&ciphertext)
        .map_err(|e| anyhow!("decrypt failed: {e}"))?;
    let frame: TransferFrame = ciborium::from_reader(&mut plaintext.as_slice())?;
    Ok(frame)
}

const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024; // 2 MiB encrypted frame

fn derive_key_from_secret(secret: &[u8]) -> [u8; 32] {
    crate::p2p_crypto::derive_key_from_bytes(secret)
}

fn file_name_to_model_id(file_name: &str) -> String {
    file_name
        .trim_end_matches(".safetensors")
        .trim_end_matches(".bin")
        .trim_end_matches(".gguf")
        .trim_end_matches(".pt")
        .trim_end_matches(".ckpt")
        .to_string()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn file_name_to_model_id_strips_extensions() {
        assert_eq!(file_name_to_model_id("qwen.safetensors"), "qwen");
        assert_eq!(file_name_to_model_id("foo.bin"), "foo");
        assert_eq!(file_name_to_model_id("bar"), "bar");
    }

    #[tokio::test]
    async fn loopback_pull_transfers_file() {
        let tmp =
            std::env::temp_dir().join(format!("badapple_p2p_model_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let sender_dir = tmp.join("sender");
        let receiver_dir = tmp.join("receiver");
        std::fs::create_dir_all(&sender_dir).unwrap();
        std::fs::create_dir_all(&receiver_dir).unwrap();

        let mut f = std::fs::File::create(sender_dir.join("test-model.safetensors")).unwrap();
        let payload = b"this is the model file content";
        f.write_all(payload).unwrap();
        drop(f);

        let secret = b"bad-apple-p2p-dev-secret-do-not-use-in-prod";
        let sender = P2PModelTransfer::new(secret.to_vec(), sender_dir.clone());
        sender.scan_and_advertise("sender-host").await.unwrap();

        // Spawn the server.
        let (addr, server) = sender.serve(0).await.unwrap();

        // Pull.
        let receiver = P2PModelTransfer::new(secret.to_vec(), receiver_dir.clone());
        let peer_addr = format!("127.0.0.1:{}", addr.port());
        let path = receiver
            .pull(&peer_addr, "test-model", &[], &receiver_dir)
            .await;

        // Stop the server.
        server.abort();

        let path = path.expect("pull should succeed");
        assert_eq!(path, receiver_dir.join("test-model.safetensors"));
        let received = std::fs::read(&path).unwrap();
        assert_eq!(received, payload);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
