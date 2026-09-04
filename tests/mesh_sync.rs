use bad_apple::mesh_sync::{
    build_engram, decode_mesh_packet, handle_incoming, MeshDoc, MeshDocKind, MeshPacket, MeshStore,
};
use bad_apple::protocol::{CompactEngramPacket, ConnectionManager, LockFreeRing, SwarmMetrics};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join(format!("bad_apple_mesh_test_{}", rand::random::<u64>()))
}

#[test]
fn mesh_store_conflict_resolution() {
    let dir = temp_dir();
    let _ = std::fs::remove_dir_all(&dir);
    let store = MeshStore::new(&dir).unwrap();

    let doc1 = MeshDoc {
        kind: MeshDocKind::Prompt,
        body: "first".to_string(),
        timestamp: 1000,
        origin: "host-a".to_string(),
        version: 1,
    };
    let doc2 = MeshDoc {
        kind: MeshDocKind::Prompt,
        body: "second".to_string(),
        timestamp: 2000,
        origin: "host-a".to_string(),
        version: 1,
    };
    let doc3 = MeshDoc {
        kind: MeshDocKind::Prompt,
        body: "stale".to_string(),
        timestamp: 500,
        origin: "host-a".to_string(),
        version: 2,
    };

    assert!(store.merge(doc1.clone()));
    assert!(store.save(&doc1).is_ok());

    assert!(store.merge(doc2.clone()));
    assert!(store.save(&doc2).is_ok());

    assert!(!store.merge(doc3));

    let latest = store.load_kind(MeshDocKind::Prompt).unwrap();
    assert_eq!(latest.body, "second");
    assert_eq!(latest.timestamp, 2000);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mesh_packet_roundtrip() {
    let packet = MeshPacket::Push {
        doc_id: "prompt@host-a".to_string(),
        doc: MeshDoc {
            kind: MeshDocKind::Prompt,
            body: "hello".to_string(),
            timestamp: 1234,
            origin: "host-a".to_string(),
            version: 1,
        },
    };
    let encoded = bad_apple::mesh_sync::encode_mesh_packet(&packet).unwrap();
    let decoded = decode_mesh_packet(&encoded).unwrap();
    assert!(matches!(decoded, MeshPacket::Push { .. }));
}

#[tokio::test]
async fn mesh_sync_over_p2p_ring() {
    let dir = temp_dir();
    let _ = std::fs::remove_dir_all(&dir);

    let ring = Arc::new(LockFreeRing::<CompactEngramPacket>::new(64));
    let metrics = Arc::new(Mutex::new(SwarmMetrics::default()));
    let cm = ConnectionManager::new(
        b"mesh-test-secret".to_vec(),
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
    tokio::time::sleep(Duration::from_millis(200)).await;

    cm.start_outbound_sweeper();

    let doc = MeshDoc {
        kind: MeshDocKind::Personas,
        body: r#"{"default": "test"}"#.to_string(),
        timestamp: 12345,
        origin: "host-a".to_string(),
        version: 1,
    };
    let packet = MeshPacket::Push {
        doc_id: doc.doc_id(),
        doc,
    };
    let engram = build_engram(&packet, "host-a").unwrap();
    cm.push_outgoing(engram);

    let received = tokio::time::timeout(Duration::from_secs(5), ring.pop_async())
        .await
        .expect("timeout waiting for mesh packet")
        .expect("ring closed");

    assert!(received.payload.is_some());

    let store = MeshStore::new(&dir).unwrap();
    let reply = handle_incoming(&received, &store);
    assert!(reply.is_some(), "handle_incoming should produce an ack");
    assert!(matches!(reply, Some(MeshPacket::Ack { .. })));

    let saved = store.load_kind(MeshDocKind::Personas).unwrap();
    assert_eq!(saved.body, r#"{"default": "test"}"#);
    assert_eq!(saved.timestamp, 12345);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mesh_doc_kind_parsing() {
    assert_eq!(
        "personas".parse::<MeshDocKind>().unwrap(),
        MeshDocKind::Personas
    );
    assert_eq!(
        "prompts".parse::<MeshDocKind>().unwrap(),
        MeshDocKind::Prompt
    );
    assert_eq!(
        "models".parse::<MeshDocKind>().unwrap(),
        MeshDocKind::ModelManifests
    );
    assert!("unknown".parse::<MeshDocKind>().is_err());
}
