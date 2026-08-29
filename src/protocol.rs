use crate::metrics::CacheLinePadded;
use crate::simd::{dot_f64_f32, magnitude_f64_f32};
use base64::{engine::general_purpose::STANDARD, Engine};
use crossbeam_queue::ArrayQueue;
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use sysinfo::System;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tokio::time::sleep;
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message, WebSocketStream};

/// Fixed dimension for compact multi-agent engram exchange.
/// Limits UDP payload size and network jitter.
pub const ENGRAM_DIM: usize = 100;

/// Full 2048-D grounded embedding dimension for similarity gating at the socket.
pub const EMBEDDING_DIM: usize = 2048;

/// Maximum number of concurrent wide-area peer connections.
pub const MAX_WAN_PEERS: usize = 64;

/// Largest single JSON frame we will accept over TCP/WebSocket (1 MiB).
const MAX_FRAME_BYTES: usize = 1_048_576;

/// Compact engram payload for multi-agent exchange.  Carries the sender's 2048-D
/// grounded embedding for socket-layer similarity gating and the 100-D brain
/// state for vector exchange.  The embedding may be regenerated from
/// `experiential_text` if a sender omits it.
///
/// `priority` is a bounded 0..=255 urgency hint.  Lower values are more
/// expendable; the lock-free ring backpressure guard will drop low-priority
/// packets when buffer occupancy exceeds 85%.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CompactEngramPacket {
    pub id: u64,
    pub timestamp: u64,
    pub experiential_text: String,
    pub emotional_state_snapshot: String,
    pub origin_instance: String,
    pub brain_state: Vec<f64>,
    #[serde(default)]
    pub embedding: Vec<f64>,
    #[serde(default = "default_priority")]
    pub priority: u8,
}

const fn default_priority() -> u8 {
    5
}

/// Trait used by the lock-free ring to decide whether an item is safe to drop
/// when the ring is under pressure.
pub trait Priority {
    fn priority(&self) -> u8;
}

impl Priority for CompactEngramPacket {
    fn priority(&self) -> u8 {
        self.priority
    }
}

// Outbound raw byte queues are always treated as high-priority control traffic.
impl Priority for Vec<u8> {
    fn priority(&self) -> u8 {
        u8::MAX
    }
}

/// Signed envelope for multi-agent engram exchange.
/// Uses HMAC-SHA256 for integrity and base64 for compact binary-safe encoding.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SignedUdpPacket {
    pub sender: String,
    pub payload_b64: String,
    pub signature_hex: String,
}

/// Transport-agnostic alias for the signed envelope.
pub type SignedPacket = SignedUdpPacket;

/// Fixed-size, cache-aligned, lock-free MPMC ring buffer for wide-area
/// ingestion hot paths. Uses `crossbeam_queue::ArrayQueue` for the lock-free
/// slots and `tokio::sync::Notify` for async wake-up.
///
/// A pressure-aware `push` selectively drops low-priority items when the buffer
/// is more than 85% full, protecting the main loop from backpressure spikes.
#[repr(align(128))]
pub struct LockFreeRing<T: Send + Priority> {
    queue: CacheLinePadded<ArrayQueue<T>>,
    closed: CacheLinePadded<AtomicBool>,
    dropped: CacheLinePadded<AtomicU64>,
    low_priority_dropped: CacheLinePadded<AtomicU64>,
    notify: Notify,
}

/// Occupancy threshold at which lower-priority items may be discarded.
const BACKPRESSURE_THRESHOLD: f64 = 0.85;

/// Minimum cosine similarity an incoming 2048-D embedding must have with any
/// active goal matrix before the engram is accepted into the ring.
pub const ENGRAM_SIMILARITY_THRESHOLD: f64 = 0.35;

/// Maximum characters of `experiential_text` that are sent in a compact engram
/// across the network.  This keeps JSON-serialized packets under the UDP
/// datagram size limit (~65 KB) while the full embedding and brain state are
/// still transmitted.
pub const MAX_ENGRAM_TEXT_CHARS: usize = 1024;

impl<T: Send + Priority> LockFreeRing<T> {
    pub fn new(cap: usize) -> Self {
        Self {
            queue: CacheLinePadded::new(ArrayQueue::new(cap)),
            closed: CacheLinePadded::new(AtomicBool::new(false)),
            dropped: CacheLinePadded::new(AtomicU64::new(0)),
            low_priority_dropped: CacheLinePadded::new(AtomicU64::new(0)),
            notify: Notify::new(),
        }
    }

    /// Push an item. If the ring is full or closed, the item is dropped and
    /// the internal drop counter is incremented.  If the ring is over 85% full
    /// and the item reports a low priority, it is discarded before enqueueing
    /// to bound network backpressure.
    pub fn push(&self, value: T) {
        if self.closed.load(Ordering::Acquire) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Backpressure guard: drop lower-priority traffic before the queue is
        // full so latency-sensitive / critical engrams remain available.
        let cap = self.queue.capacity();
        if cap > 0 {
            let occupancy = self.queue.len() as f64 / cap as f64;
            if occupancy > BACKPRESSURE_THRESHOLD && value.priority() < default_priority() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                self.low_priority_dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        if self.queue.push(value).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        } else {
            self.notify.notify_one();
        }
    }

    /// Try to pop an item without blocking.
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }

    /// Wait for an item, or return `None` once the ring is closed and empty.
    pub async fn pop_async(&self) -> Option<T> {
        loop {
            if let Some(value) = self.queue.pop() {
                return Some(value);
            }
            if self.closed.load(Ordering::Acquire) {
                return None;
            }
            self.notify.notified().await;
        }
    }

    /// Close the ring. Subsequent pushes are dropped, and any waiting pop
    /// returns `None` once drained.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn low_priority_dropped(&self) -> u64 {
        self.low_priority_dropped.load(Ordering::Relaxed)
    }
}

unsafe impl<T: Send + Priority> Send for LockFreeRing<T> {}
unsafe impl<T: Send + Priority> Sync for LockFreeRing<T> {}

/// Resolve a shared multi-agent signing secret.
/// Prefer the `MULTI_AGENT_SECRET` environment variable; otherwise derive a
/// default from the local hostname.  This keeps shared secrets out of source.
pub fn multi_agent_secret() -> Vec<u8> {
    let host = System::host_name().unwrap_or_else(|| "localhost".to_string());
    std::env::var("MULTI_AGENT_SECRET")
        .unwrap_or_else(|_| format!("bad-apple-{host}-default"))
        .into_bytes()
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().as_slice().to_vec()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Sign a compact engram payload, returning a JSON-safe signed packet.
pub fn sign_packet(sender: &str, payload: &[u8], secret: &[u8]) -> SignedPacket {
    let signature = hex_encode(&hmac_sha256(secret, payload));
    SignedPacket {
        sender: sender.to_string(),
        payload_b64: STANDARD.encode(payload),
        signature_hex: signature,
    }
}

/// Decode the base64 payload of a signed packet (does not verify integrity).
pub fn decode_payload(packet: &SignedPacket) -> Option<Vec<u8>> {
    STANDARD.decode(&packet.payload_b64).ok()
}

/// Verify the HMAC-SHA256 signature of a signed packet.
pub fn verify_packet(packet: &SignedPacket, secret: &[u8]) -> bool {
    let Some(payload) = decode_payload(packet) else {
        return false;
    };
    let expected = hex_encode(&hmac_sha256(secret, &payload));
    expected == packet.signature_hex
}

/// Live metrics for the wide-area swarm grid panel.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SwarmMetrics {
    pub peer_count: usize,
    pub bytes_in_total: u64,
    pub bytes_out_total: u64,
    pub prev_bytes_in: u64,
    pub prev_bytes_out: u64,
    #[serde(skip)]
    pub prev_sample_time: Option<Instant>,
    pub kbps_in: f64,
    pub kbps_out: f64,
    pub last_merge_latency_us: u64,
    pub last_merge_batch_size: usize,
    pub last_engram_receive_latency_us: u64,
    pub engrams_dropped_similarity: u64,
}

impl SwarmMetrics {
    pub fn sample(&mut self) {
        let now = Instant::now();
        if let Some(prev) = self.prev_sample_time {
            let dt = now.duration_since(prev).as_secs_f64();
            if dt > 0.0 {
                let in_delta =
                    self.bytes_in_total.saturating_sub(self.prev_bytes_in) as f64 / 1024.0;
                let out_delta =
                    self.bytes_out_total.saturating_sub(self.prev_bytes_out) as f64 / 1024.0;
                self.kbps_in = in_delta / dt;
                self.kbps_out = out_delta / dt;
            }
        }
        self.prev_bytes_in = self.bytes_in_total;
        self.prev_bytes_out = self.bytes_out_total;
        self.prev_sample_time = Some(now);
    }

    pub fn record_merge(&mut self, latency: Duration, batch_size: usize) {
        self.last_merge_latency_us = latency.as_micros() as u64;
        self.last_merge_batch_size = batch_size;
    }

    pub fn record_receive(&mut self, latency: Duration) {
        self.last_engram_receive_latency_us = latency.as_micros() as u64;
    }
}

/// Transport type for a wide-area peer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerTransport {
    Tcp,
    WebSocket,
}

/// Opaque identifier for a peer in the connection registry.
pub type PeerId = String;

/// Handle to a connected peer.
#[derive(Clone)]
pub struct PeerHandle {
    pub id: PeerId,
    pub transport: PeerTransport,
    pub addr: String,
    pub outbound: Arc<LockFreeRing<Vec<u8>>>,
    pub bytes_in: Arc<AtomicU64>,
    pub bytes_out: Arc<AtomicU64>,
    pub connected_at: Instant,
}

impl PeerHandle {
    fn new(
        id: PeerId,
        transport: PeerTransport,
        addr: String,
        outbound: Arc<LockFreeRing<Vec<u8>>>,
    ) -> Self {
        Self {
            id,
            transport,
            addr,
            outbound,
            bytes_in: Arc::new(AtomicU64::new(0)),
            bytes_out: Arc::new(AtomicU64::new(0)),
            connected_at: Instant::now(),
        }
    }
}

/// Persistent, non-blocking TCP/WebSocket connection manager for wide-area engram gossip.
///
/// * Maintains up to `MAX_WAN_PEERS` concurrent peer handles in a `DashMap`.
/// * Outbound connections use an exponential backoff retry state machine.
/// * Incoming connections are accepted on dedicated TCP and WebSocket ports.
/// * Verified engrams are forwarded to the lock-free `incoming` ring buffer.
#[derive(Clone)]
pub struct ConnectionManager {
    secret: Arc<Vec<u8>>,
    peers: Arc<DashMap<PeerId, PeerHandle>>,
    incoming: Arc<LockFreeRing<CompactEngramPacket>>,
    /// Lock-free outgoing ring for wild-workspace and other fire-and-forget
    /// broadcast producers.  Producers push synchronously; a single background
    /// sweeper drains the ring and calls `broadcast`.
    outgoing: Arc<LockFreeRing<CompactEngramPacket>>,
    pub metrics: Arc<Mutex<SwarmMetrics>>,
    max_peers: usize,
    retry_base: Duration,
    retry_max: Duration,
    listen_addr_tcp: Arc<Mutex<Option<SocketAddr>>>,
    listen_addr_ws: Arc<Mutex<Option<SocketAddr>>>,
    /// Active-goal 2048-D embeddings.  Empty means the similarity gate is open.
    goal_embeddings: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl ConnectionManager {
    pub fn new(
        secret: Vec<u8>,
        incoming: Arc<LockFreeRing<CompactEngramPacket>>,
        metrics: Arc<Mutex<SwarmMetrics>>,
        max_peers: usize,
        retry_base: Duration,
        retry_max: Duration,
    ) -> Self {
        Self {
            secret: Arc::new(secret),
            peers: Arc::new(DashMap::with_capacity(max_peers)),
            incoming,
            // Fire-and-forget outbound broadcast queue.  Capacity is sized to
            // absorb a full wild_workspace burst without backpressure.
            outgoing: Arc::new(LockFreeRing::new(256)),
            metrics,
            max_peers,
            retry_base,
            retry_max,
            listen_addr_tcp: Arc::new(Mutex::new(None)),
            listen_addr_ws: Arc::new(Mutex::new(None)),
            goal_embeddings: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Replace the active-goal embeddings used by the inbound similarity gate.
    pub fn set_goal_embeddings(&self, goals: Vec<Vec<f64>>) {
        if let Ok(mut guard) = self.goal_embeddings.lock() {
            *guard = goals;
        }
    }

    /// Spawn TCP and WebSocket accept loops and return the bound addresses.
    pub async fn start_server(
        &self,
        tcp_port: u16,
        ws_port: u16,
    ) -> (Result<SocketAddr, String>, Result<SocketAddr, String>) {
        let tcp_addr = self.start_tcp_listener(tcp_port).await;
        let ws_addr = self.start_ws_listener(ws_port).await;
        (tcp_addr, ws_addr)
    }

    /// Spawn outbound connection attempts for every configured peer.
    pub async fn connect_to_peers(&self, peers: Vec<String>) {
        for peer in peers {
            let cm = self.clone();
            tokio::spawn(async move {
                cm.connect_with_retry(peer).await;
            });
        }
    }

    /// Push a packet onto the lock-free outbound ring.  This is the zero-wait
    /// interface for wild_workspace and any other producers that must not block.
    pub fn push_outgoing(&self, packet: CompactEngramPacket) {
        self.outgoing.push(packet);
    }

    /// Spawn a single background sweeper that drains the outbound ring and
    /// broadcasts each packet.  Only one sweeper should be running per manager.
    pub fn start_outbound_sweeper(&self) {
        let cm = self.clone();
        tokio::spawn(async move {
            while let Some(packet) = cm.outgoing.pop_async().await {
                cm.broadcast(&packet).await;
            }
            // Ring is closed and empty.
        });
    }

    /// Broadcast a compact engram to every connected peer. The engram is signed
    /// with `origin_instance` as the sender identity.
    pub async fn broadcast(&self, packet: &CompactEngramPacket) {
        let payload = match serde_json::to_vec(packet) {
            Ok(v) => v,
            Err(_) => return,
        };
        let signed = sign_packet(&packet.origin_instance, &payload, &self.secret);
        let frame = match serde_json::to_vec(&signed) {
            Ok(v) => v,
            Err(_) => return,
        };

        let handles: Vec<PeerHandle> = self.peers.iter().map(|r| r.clone()).collect();
        for handle in handles {
            handle.outbound.push(frame.clone());
            handle
                .bytes_out
                .fetch_add(frame.len() as u64, Ordering::Relaxed);
        }

        if let Ok(mut m) = self.metrics.lock() {
            m.bytes_out_total += frame.len() as u64;
        }
    }

    /// Number of currently connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Active peer handles for telemetry/diagnostics.
    pub fn active_peers(&self) -> Vec<PeerHandle> {
        self.peers.iter().map(|r| r.clone()).collect()
    }

    /// Background sampler that recomputes throughput from peer byte counters.
    pub async fn run_metrics_sampler(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            if let Ok(mut m) = self.metrics.lock() {
                m.peer_count = self.peers.len();
                m.sample();
            }
        }
    }

    async fn start_tcp_listener(&self, port: u16) -> Result<SocketAddr, String> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .map_err(|e| format!("WAN TCP listener bind failed: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("WAN TCP listener local_addr failed: {e}"))?;
        if let Ok(mut guard) = self.listen_addr_tcp.lock() {
            *guard = Some(addr);
        }

        let cm = self.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        let cm = cm.clone();
                        tokio::spawn(async move {
                            cm.handle_tcp_stream(stream, peer_addr).await;
                        });
                    }
                    Err(e) => {
                        tracing::warn!("WAN TCP accept failed: {}", e);
                    }
                }
            }
        });
        Ok(addr)
    }

    async fn start_ws_listener(&self, port: u16) -> Result<SocketAddr, String> {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .map_err(|e| format!("WAN WebSocket listener bind failed: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| format!("WAN WebSocket listener local_addr failed: {e}"))?;
        if let Ok(mut guard) = self.listen_addr_ws.lock() {
            *guard = Some(addr);
        }

        let cm = self.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        let display = peer_addr.to_string();
                        let cm = cm.clone();
                        tokio::spawn(async move {
                            match accept_async(stream).await {
                                Ok(ws) => {
                                    cm.handle_ws_stream(ws, display).await;
                                }
                                Err(e) => {
                                    tracing::warn!("WAN WebSocket accept failed: {}", e);
                                }
                            }
                        });
                    }
                    Err(e) => {
                        tracing::warn!("WAN WebSocket accept failed: {}", e);
                    }
                }
            }
        });
        Ok(addr)
    }

    async fn connect_with_retry(&self, peer: String) {
        let mut attempt: u32 = 0;
        loop {
            if self.peers.len() >= self.max_peers {
                sleep(self.retry_max).await;
                continue;
            }

            if let Err(e) = self.try_connect(&peer).await {
                let delay = self
                    .retry_base
                    .mul_f64(2f64.powi(attempt as i32))
                    .min(self.retry_max);
                tracing::warn!(
                    "WAN peer {} connection failed (attempt {}): {}; retrying in {:?}",
                    peer,
                    attempt,
                    e,
                    delay
                );
                sleep(delay).await;
                attempt = attempt.saturating_add(1);
            } else {
                // If the connection closed, back off a bit before retrying.
                sleep(self.retry_base).await;
                attempt = 0;
            }
        }
    }

    async fn try_connect(&self, peer: &str) -> Result<(), String> {
        let (peer_id, transport) = parse_peer_spec(peer);
        match transport {
            PeerTransport::Tcp => {
                let stream = TcpStream::connect(&peer_id)
                    .await
                    .map_err(|e| e.to_string())?;
                let peer_addr = stream.peer_addr().map_err(|e| e.to_string())?;
                self.handle_tcp_stream(stream, peer_addr).await;
                Ok(())
            }
            PeerTransport::WebSocket => {
                let url = format!("ws://{peer_id}");
                let (ws, _) = connect_async(&url).await.map_err(|e| e.to_string())?;
                self.handle_ws_stream(ws, peer_id).await;
                Ok(())
            }
        }
    }

    async fn handle_tcp_stream(&self, stream: TcpStream, peer_addr: SocketAddr) {
        let (mut reader, mut writer) = stream.into_split();
        let outbound = Arc::new(LockFreeRing::<Vec<u8>>::new(256));
        let peer_id = format!("tcp:{peer_addr}");

        if self
            .register(
                peer_id.clone(),
                PeerTransport::Tcp,
                peer_addr.to_string(),
                outbound.clone(),
            )
            .await
            .is_none()
        {
            return;
        }

        let cm = self.clone();
        let peer_id_read = peer_id.clone();
        let read_task = tokio::spawn(async move {
            let mut buf = [0u8; 4];
            loop {
                if reader.read_exact(&mut buf).await.is_err() {
                    break;
                }
                let len = u32::from_be_bytes(buf) as usize;
                if len == 0 || len > MAX_FRAME_BYTES {
                    break;
                }
                let mut frame = vec![0u8; len];
                if reader.read_exact(&mut frame).await.is_err() {
                    break;
                }
                cm.handle_frame(&peer_id_read, &frame).await;
            }
            cm.unregister(&peer_id_read).await;
        });

        let cm = self.clone();
        let peer_id_write = peer_id.clone();
        let write_task = tokio::spawn(async move {
            while let Some(frame) = outbound.pop_async().await {
                let len = frame.len() as u32;
                if len == 0 {
                    continue;
                }
                let mut header = len.to_be_bytes().to_vec();
                header.extend_from_slice(&frame);
                if writer.write_all(&header).await.is_err() {
                    break;
                }
                if writer.flush().await.is_err() {
                    break;
                }
                if let Some(handle) = cm.peers.get(&peer_id_write) {
                    handle
                        .bytes_out
                        .fetch_add(header.len() as u64, Ordering::Relaxed);
                }
            }
            cm.unregister(&peer_id_write).await;
        });

        let _ = tokio::join!(read_task, write_task);
        self.unregister(&peer_id).await;
    }

    async fn handle_ws_stream<S>(&self, ws: WebSocketStream<S>, peer_display: String)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let outbound = Arc::new(LockFreeRing::<Vec<u8>>::new(256));
        let peer_id = format!("ws:{peer_display}");

        if self
            .register(
                peer_id.clone(),
                PeerTransport::WebSocket,
                peer_display,
                outbound.clone(),
            )
            .await
            .is_none()
        {
            return;
        }

        let (mut ws_sink, mut ws_stream) = ws.split();

        let cm = self.clone();
        let peer_id_read = peer_id.clone();
        let read_task = tokio::spawn(async move {
            while let Some(msg) = ws_stream.next().await {
                match msg {
                    Ok(Message::Binary(frame)) => {
                        cm.handle_frame(&peer_id_read, &frame).await;
                    }
                    Ok(Message::Text(text)) => {
                        cm.handle_frame(&peer_id_read, text.as_bytes()).await;
                    }
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
            cm.unregister(&peer_id_read).await;
        });

        let cm = self.clone();
        let peer_id_write = peer_id.clone();
        let write_task = tokio::spawn(async move {
            while let Some(frame) = outbound.pop_async().await {
                if ws_sink.send(Message::Binary(frame.clone())).await.is_err() {
                    break;
                }
                if let Some(handle) = cm.peers.get(&peer_id_write) {
                    handle
                        .bytes_out
                        .fetch_add(frame.len() as u64, Ordering::Relaxed);
                }
            }
            cm.unregister(&peer_id_write).await;
        });

        let _ = tokio::join!(read_task, write_task);
        self.unregister(&peer_id).await;
    }

    /// Cosine-similarity between an incoming 2048-D embedding and the best
    /// active-goal embedding.  Returns 1.0 when no goals are configured or the
    /// embedding is absent, effectively leaving the gate open.
    pub fn max_goal_similarity(&self, embedding: &[f64]) -> f64 {
        let Ok(goals) = self.goal_embeddings.lock() else {
            return 1.0;
        };
        if goals.is_empty() || embedding.is_empty() {
            return 1.0;
        }
        let norm_a = magnitude_f64_f32(embedding);
        if norm_a == 0.0 || !norm_a.is_finite() {
            return 0.0;
        }
        let mut best = -1.0_f64;
        for goal in goals.iter() {
            if goal.len() != embedding.len() {
                continue;
            }
            let norm_b = magnitude_f64_f32(goal);
            if norm_b == 0.0 || !norm_b.is_finite() {
                continue;
            }
            let dot = dot_f64_f32(embedding, goal);
            if !dot.is_finite() {
                continue;
            }
            let sim = dot / (norm_a * norm_b);
            if sim > best {
                best = sim;
            }
        }
        best
    }

    async fn handle_frame(&self, peer_id: &PeerId, frame: &[u8]) {
        let start = Instant::now();
        let Ok(packet) = serde_json::from_slice::<SignedPacket>(frame) else {
            return;
        };
        if !verify_packet(&packet, &self.secret) {
            tracing::info!(
                "🔒 Dropped unsigned / tampered wide-area packet from '{}'",
                packet.sender
            );
            return;
        }
        let Some(payload) = decode_payload(&packet) else {
            return;
        };
        let Ok(mut compact) = serde_json::from_slice::<CompactEngramPacket>(&payload) else {
            return;
        };
        compact.brain_state.truncate(ENGRAM_DIM);
        compact.embedding.truncate(EMBEDDING_DIM);

        if !compact.embedding.is_empty() {
            let sim = self.max_goal_similarity(&compact.embedding);
            if sim < ENGRAM_SIMILARITY_THRESHOLD {
                if let Ok(mut m) = self.metrics.lock() {
                    m.engrams_dropped_similarity += 1;
                }
                tracing::info!(
                    "🛡️ Dropped peer engram from '{}': similarity {:.3} < {}",
                    compact.origin_instance,
                    sim,
                    ENGRAM_SIMILARITY_THRESHOLD
                );
                return;
            }
        }

        self.incoming.push(compact);

        let frame_len = frame.len();
        if let Some(handle) = self.peers.get(peer_id) {
            handle
                .bytes_in
                .fetch_add(frame_len as u64, Ordering::Relaxed);
        }
        if let Ok(mut m) = self.metrics.lock() {
            m.bytes_in_total += frame_len as u64;
            m.record_receive(start.elapsed());
        }
    }

    async fn register(
        &self,
        peer_id: PeerId,
        transport: PeerTransport,
        addr: String,
        outbound: Arc<LockFreeRing<Vec<u8>>>,
    ) -> Option<PeerHandle> {
        if self.peers.len() >= self.max_peers {
            return None;
        }
        if self.peers.contains_key(&peer_id) {
            return None;
        }
        let handle = PeerHandle::new(peer_id.clone(), transport, addr, outbound);
        self.peers.insert(peer_id.clone(), handle.clone());
        tracing::info!(
            "🌐 [WAN PEER CONNECTED]: {} ({}); total peers {}",
            peer_id,
            handle.transport_label(),
            self.peers.len()
        );
        Some(handle)
    }

    async fn unregister(&self, peer_id: &PeerId) {
        if let Some((_, handle)) = self.peers.remove(peer_id) {
            handle.outbound.close();
            tracing::info!(
                "🌐 [WAN PEER DISCONNECTED]: {}; total peers {}",
                peer_id,
                self.peers.len()
            );
        }
    }
}

impl PeerHandle {
    fn transport_label(&self) -> &'static str {
        match self.transport {
            PeerTransport::Tcp => "TCP",
            PeerTransport::WebSocket => "WebSocket",
        }
    }
}

fn parse_peer_spec(peer: &str) -> (String, PeerTransport) {
    let peer = peer.trim();
    if let Some(rest) = peer.strip_prefix("ws://") {
        (rest.to_string(), PeerTransport::WebSocket)
    } else if let Some(rest) = peer.strip_prefix("wss://") {
        (rest.to_string(), PeerTransport::WebSocket)
    } else {
        (peer.to_string(), PeerTransport::Tcp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_roundtrip() {
        let secret = b"test-secret";
        let payload = b"hello engram";
        let signed = sign_packet("tester", payload, secret);
        assert!(verify_packet(&signed, secret));
        assert!(!verify_packet(&signed, b"wrong"));
    }

    #[test]
    fn tampered_payload_fails() {
        let secret = b"test-secret";
        let signed = sign_packet("tester", b"hello", secret);
        let mut tampered = signed.clone();
        tampered.payload_b64 = STANDARD.encode(b"different");
        assert!(!verify_packet(&tampered, secret));
    }

    #[test]
    fn parse_peer_spec_detects_ws() {
        let (addr, t) = parse_peer_spec("ws://1.2.3.4:9001");
        assert_eq!(addr, "1.2.3.4:9001");
        assert_eq!(t, PeerTransport::WebSocket);
    }

    #[test]
    fn parse_peer_spec_defaults_to_tcp() {
        let (addr, t) = parse_peer_spec("1.2.3.4:9001");
        assert_eq!(addr, "1.2.3.4:9001");
        assert_eq!(t, PeerTransport::Tcp);
    }

    #[test]
    fn compact_engram_truncate() {
        let mut p = CompactEngramPacket {
            id: 1,
            timestamp: 2,
            experiential_text: "t".to_string(),
            emotional_state_snapshot: "e".to_string(),
            origin_instance: "o".to_string(),
            brain_state: vec![1.0; 256],
            ..Default::default()
        };
        p.brain_state.truncate(ENGRAM_DIM);
        assert_eq!(p.brain_state.len(), ENGRAM_DIM);
    }

    #[tokio::test]
    async fn wan_tcp_roundtrip() {
        let ring = Arc::new(LockFreeRing::<CompactEngramPacket>::new(64));
        let metrics = Arc::new(Mutex::new(SwarmMetrics::default()));
        let cm = ConnectionManager::new(
            b"test-secret".to_vec(),
            ring.clone(),
            metrics,
            4,
            Duration::from_millis(10),
            Duration::from_millis(100),
        );
        let (tcp, _ws) = cm.start_server(0, 0).await;
        let addr = tcp.expect("TCP listener should bind");

        let cm_connect = cm.clone();
        let peer = format!("127.0.0.1:{}", addr.port());
        tokio::spawn(async move {
            cm_connect.connect_to_peers(vec![peer]).await;
        });
        sleep(Duration::from_millis(200)).await;

        let packet = CompactEngramPacket {
            id: 0xC0FFEE,
            timestamp: 1,
            experiential_text: "roundtrip".to_string(),
            emotional_state_snapshot: "test".to_string(),
            origin_instance: "self".to_string(),
            brain_state: vec![0.5; ENGRAM_DIM],
            ..Default::default()
        };
        cm.broadcast(&packet).await;

        let received = tokio::time::timeout(Duration::from_secs(5), ring.pop_async()).await;
        let received = received
            .expect("timeout waiting for engram")
            .expect("ring closed");
        assert_eq!(received.id, packet.id);
        assert_eq!(received.experiential_text, packet.experiential_text);
    }
}
