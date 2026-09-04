//! Cross-device private AI mesh sync.
//!
//! Synchronizes small encrypted documents (personas, prompts, settings)
//! across Bad Apple peers over the existing P2P engram mesh. All payloads
//! are encrypted and signed by `protocol::ConnectionManager` / `P2PCipher`.
//!
//! Conflict resolution is last-write-wins by wall-clock timestamp. Documents
//! are stored under `~/.bad_apple/mesh/` with one JSON file per document.

use crate::protocol::CompactEngramPacket;
use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Kinds of documents that can be synchronized across the mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeshDocKind {
    Personas,
    Prompt,
    Settings,
    ModelManifests,
}

impl std::fmt::Display for MeshDocKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshDocKind::Personas => write!(f, "personas"),
            MeshDocKind::Prompt => write!(f, "prompt"),
            MeshDocKind::Settings => write!(f, "settings"),
            MeshDocKind::ModelManifests => write!(f, "model_manifests"),
        }
    }
}

impl std::str::FromStr for MeshDocKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "personas" => Ok(MeshDocKind::Personas),
            "prompt" | "prompts" => Ok(MeshDocKind::Prompt),
            "settings" => Ok(MeshDocKind::Settings),
            "model_manifests" | "models" => Ok(MeshDocKind::ModelManifests),
            _ => Err(anyhow::anyhow!("unknown mesh document kind: {s}")),
        }
    }
}

impl MeshDocKind {
    /// Default filename used when the document is persisted in the mesh store.
    pub fn default_filename(&self) -> &'static str {
        match self {
            MeshDocKind::Personas => "personas.json",
            MeshDocKind::Prompt => "prompt.txt",
            MeshDocKind::Settings => "settings.json",
            MeshDocKind::ModelManifests => "model_manifests.json",
        }
    }

    /// Local source path on this device for the canonical document.
    pub fn local_source(&self) -> PathBuf {
        let base = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("/var/lib/bad_apple"))
            .join("bad_apple");
        match self {
            MeshDocKind::Personas => base.join("personas.json"),
            MeshDocKind::Prompt => base.join("prompt.txt"),
            MeshDocKind::Settings => base.join("settings.json"),
            MeshDocKind::ModelManifests => base.join("model_manifests.json"),
        }
    }
}

/// A synchronizable document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshDoc {
    pub kind: MeshDocKind,
    pub body: String,
    pub timestamp: u64,
    pub origin: String,
    pub version: u64,
}

impl MeshDoc {
    /// Stable document ID used as the mesh store key.
    pub fn doc_id(&self) -> String {
        format!("{}@{}", self.kind, self.origin)
    }

    /// Return a new `MeshDoc` with an incremented version and current timestamp.
    pub fn bumped(self) -> Self {
        Self {
            timestamp: now_secs(),
            version: self.version.saturating_add(1),
            ..self
        }
    }
}

/// Mesh sync protocol messages. Wrapped in `CompactEngramPacket::payload`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum MeshPacket {
    Push { doc_id: String, doc: MeshDoc },
    Pull { doc_id: String },
    Ack { doc_id: String, timestamp: u64 },
}

impl MeshPacket {
    pub fn doc_id(&self) -> &str {
        match self {
            MeshPacket::Push { doc_id, .. } => doc_id,
            MeshPacket::Pull { doc_id } => doc_id,
            MeshPacket::Ack { doc_id, .. } => doc_id,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Persistent local store for mesh-synced documents.
pub struct MeshStore {
    root: PathBuf,
    docs: Arc<Mutex<BTreeMap<String, MeshDoc>>>,
}

impl MeshStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create mesh store at {}", root.display()))?;
        let docs = Arc::new(Mutex::new(BTreeMap::new()));
        let store = Self { root, docs };
        store.load_all()?;
        Ok(store)
    }

    pub fn default_root() -> PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("/var/lib/bad_apple"))
            .join("bad_apple")
            .join("mesh")
    }

    pub fn load(&self, doc_id: &str) -> Option<MeshDoc> {
        self.docs.lock().ok()?.get(doc_id).cloned()
    }

    pub fn load_kind(&self, kind: MeshDocKind) -> Option<MeshDoc> {
        let guard = self.docs.lock().ok()?;
        guard.values().find(|d| d.kind == kind).cloned()
    }

    fn doc_path(&self, doc_id: &str) -> PathBuf {
        self.root.join(format!("{}.json", sanitize(doc_id)))
    }

    /// Merge a document into the store if it is newer than the existing copy.
    /// Returns `true` if the local copy was updated.
    pub fn merge(&self, doc: MeshDoc) -> bool {
        let mut guard = self.docs.lock().expect("mesh store poisoned");
        let id = doc.doc_id();
        let updated = if let Some(existing) = guard.get(&id) {
            doc.timestamp > existing.timestamp
                || (doc.timestamp == existing.timestamp && doc.version > existing.version)
        } else {
            true
        };
        if updated {
            guard.insert(id, doc);
            true
        } else {
            false
        }
    }

    /// Persist a document to disk.
    pub fn save(&self, doc: &MeshDoc) -> Result<()> {
        let path = self.doc_path(&doc.doc_id());
        let tmp = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(doc)?;
        let mut f = fs::File::create(&tmp)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Load all persisted documents from disk into memory.
    pub fn load_all(&self) -> Result<()> {
        let mut guard = self.docs.lock().expect("mesh store poisoned");
        guard.clear();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)?;
            if let Ok(doc) = serde_json::from_slice::<MeshDoc>(&bytes) {
                guard.insert(doc.doc_id(), doc);
            }
        }
        Ok(())
    }

    pub fn docs(&self) -> BTreeMap<String, MeshDoc> {
        self.docs.lock().ok().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn list(&self) -> Vec<MeshDoc> {
        self.docs().values().cloned().collect()
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}

/// Encode a `MeshPacket` into a base64 string suitable for `CompactEngramPacket::payload`.
pub fn encode_mesh_packet(packet: &MeshPacket) -> Result<String> {
    let json = serde_json::to_string(packet)?;
    Ok(STANDARD.encode(json.as_bytes()))
}

/// Decode a `MeshPacket` from a `CompactEngramPacket::payload` value.
pub fn decode_mesh_packet(payload: &str) -> Result<MeshPacket> {
    let bytes = STANDARD.decode(payload)?;
    let json = std::str::from_utf8(&bytes)?;
    Ok(serde_json::from_str(json)?)
}

/// Build a compact engram carrying a mesh sync packet.
///
/// The packet is stored in `payload` so it does not interfere with
/// `experiential_text`. The embedding is left empty; `ConnectionManager`
/// bypasses the similarity gate for packets that carry a `payload`.
pub fn build_engram(packet: &MeshPacket, origin: &str) -> Result<CompactEngramPacket> {
    let payload = encode_mesh_packet(packet)?;
    Ok(CompactEngramPacket {
        id: rand::random(),
        timestamp: now_secs(),
        experiential_text: format!("mesh:{}", packet.doc_id()),
        emotional_state_snapshot: String::new(),
        origin_instance: origin.to_string(),
        brain_state: vec![0.0; crate::protocol::ENGRAM_DIM],
        embedding: Vec::new(),
        priority: u8::MAX,
        payload: Some(payload),
    })
}

/// Read a mesh document from the local source path.
pub fn read_local_doc(kind: MeshDocKind) -> Result<MeshDoc> {
    let path = kind.local_source();
    let body = if path.exists() {
        std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
    let origin = std::env::var("BADAPPLE_ORIGIN_INSTANCE").unwrap_or_else(|_| hostname());
    Ok(MeshDoc {
        kind,
        body,
        timestamp: now_secs(),
        origin,
        version: 1,
    })
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Process an incoming compact engram and, if it contains a mesh packet,
/// merge it into the store and return an acknowledgement packet.
pub fn handle_incoming(packet: &CompactEngramPacket, store: &MeshStore) -> Option<MeshPacket> {
    let payload = packet.payload.as_ref()?;
    let mesh = decode_mesh_packet(payload).ok()?;
    match mesh {
        MeshPacket::Push { doc_id, doc } => {
            let updated = store.merge(doc.clone());
            if updated {
                let _ = store.save(&doc);
            }
            Some(MeshPacket::Ack {
                doc_id,
                timestamp: doc.timestamp,
            })
        }
        MeshPacket::Pull { doc_id } => {
            // Respond with a Push for the requested document if we have it.
            store
                .load(&doc_id)
                .map(|doc| MeshPacket::Push { doc_id, doc })
        }
        MeshPacket::Ack { .. } => None,
    }
}
