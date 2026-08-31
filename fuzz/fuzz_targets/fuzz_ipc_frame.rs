#![no_main]
use libfuzzer_sys::fuzz_target;

// Fuzz the SLICKS IPC frame parser with arbitrary bytes.
//
// The parser must never panic, hang, or cause undefined behaviour regardless
// of the input it receives.  This target exercises:
//
// * `read_frame` – newline-delimited JSON frame deserialisation for both
//   `ClientFrame` and `ServerFrame`.
// * `nonce_is_valid` – hex nonce format validation.
// * `timestamp_is_fresh` – handshake timestamp skew check.
// * `validate_request` – prompt and token-count bounds checking.
fuzz_target!(|data: &[u8]| {
    use bad_apple::bad_apple_ipc;

    // -----------------------------------------------------------------
    // read_frame: feed arbitrary bytes through the newline-delimited JSON
    // frame parser.  A Cursor is a cheap in-memory BufRead.  We try both
    // ClientFrame and ServerFrame since either may arrive on the wire.
    // -----------------------------------------------------------------
    {
        let mut reader = std::io::Cursor::new(data);
        let _ = bad_apple_ipc::read_frame::<_, bad_apple_ipc::ClientFrame>(&mut reader);
    }
    {
        let mut reader = std::io::Cursor::new(data);
        let _ = bad_apple_ipc::read_frame::<_, bad_apple_ipc::ServerFrame>(&mut reader);
    }

    // Also exercise the path where the data *does* end with a newline –
    // this is the common case on the wire and exercises the serde_json
    // deserialisation branch more deeply.
    if !data.ends_with(b"\n") {
        let mut buf = data.to_vec();
        buf.push(b'\n');
        let mut reader = std::io::Cursor::new(buf);
        let _ = bad_apple_ipc::read_frame::<_, bad_apple_ipc::ClientFrame>(&mut reader);
        let mut reader = std::io::Cursor::new(data.to_vec());
        reader.get_mut().push(b'\n');
        let _ = bad_apple_ipc::read_frame::<_, bad_apple_ipc::ServerFrame>(&mut reader);
    }

    // -----------------------------------------------------------------
    // nonce_is_valid: the nonce is a 64-char hex string.  Fuzz with
    // arbitrary UTF-8 to ensure the check never panics on odd input.
    // -----------------------------------------------------------------
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = bad_apple_ipc::nonce_is_valid(s);
    }

    // -----------------------------------------------------------------
    // timestamp_is_fresh: feed arbitrary 64-bit values as timestamps.
    // -----------------------------------------------------------------
    if data.len() >= 8 {
        let ts = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let _ = bad_apple_ipc::timestamp_is_fresh(ts);
    }

    // -----------------------------------------------------------------
    // validate_request: fuzz the prompt string and token count.  Split
    // the data: first half is the prompt, last byte maps to a token count.
    // -----------------------------------------------------------------
    if let Ok(s) = std::str::from_utf8(data) {
        // Use a deterministic mapping from the data to a token count so
        // the fuzzer explores both very small and very large values.
        let max_new_tokens = if data.len() >= 2 {
            (data[data.len() - 1] as usize) << 8 | (data[data.len() - 2] as usize)
        } else {
            data.len()
        };
        let _ = bad_apple_ipc::validate_request(s, max_new_tokens);
    }
});
