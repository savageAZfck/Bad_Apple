#![no_main]
use libfuzzer_sys::fuzz_target;

// Fuzz the multi-agent protocol frame parser with arbitrary bytes.
//
// The protocol module exchanges signed JSON envelopes (`SignedUdpPacket`)
// and compact engram packets (`CompactEngramPacket`) over TCP/WebSocket.
// All deserialisation and verification of untrusted input must be
// panic-free.  This target exercises:
//
// * Deserialising arbitrary bytes as `SignedUdpPacket`.
// * `decode_payload` – base64-decoding the packet's payload.
// * `verify_packet` – HMAC-SHA256 constant-time signature verification.
// * Deserialising arbitrary bytes as `CompactEngramPacket`.
// * Generic `serde_json::Value` parsing as a baseline.
fuzz_target!(|data: &[u8]| {
    use bad_apple::protocol;

    // A fixed dummy secret for signature verification.  The point is to
    // exercise the verify path, not to find valid signatures.
    let secret = b"fuzz-secret-key-not-used-in-prod";

    // --- SignedUdpPacket deserialisation + verify ---------------------
    // Try to parse the fuzzer data as a signed envelope.  If it parses,
    // exercise both decode_payload (base64) and verify_packet (HMAC).
    if let Ok(packet) = serde_json::from_slice::<protocol::SignedUdpPacket>(data) {
        // decode_payload does a base64 decode of payload_b64.
        let _ = protocol::decode_payload(&packet);

        // verify_packet decodes the payload, computes HMAC-SHA256, and
        // does a constant-time comparison with signature_hex.
        let _ = protocol::verify_packet(&packet, secret);
    }

    // --- CompactEngramPacket deserialisation --------------------------
    // Try to parse the fuzzer data as a compact engram.  These carry
    // brain_state (Vec<f64>) and embedding (Vec<f64>) vectors that could
    // trigger panics in downstream vector math if malformed.
    if let Ok(engram) = serde_json::from_slice::<protocol::CompactEngramPacket>(data) {
        // Verify that the deserialised fields don't cause issues when
        // accessed.  The brain_state vector length is used in downstream
        // similarity gating; ensure it doesn't panic.
        let _ = engram.brain_state.len();
        let _ = engram.embedding.len();
        let _ = engram.experiential_text.len();
        let _ = engram.priority;

        // The protocol truncates experiential_text to a limit; verify
        // that accessing the text doesn't panic.
        let _ = engram.experiential_text.chars().count();
    }

    // --- Generic JSON parsing baseline --------------------------------
    // Parse as a generic JSON value to catch serde_json-level panics
    // that the typed deserialisations above might not exercise.
    let _ = serde_json::from_slice::<serde_json::Value>(data);
});
