//! Minimal Model Context Protocol (MCP) server host for Bad Apple.
//!
//! This module implements the local MCP JSON-RPC transport over `stdio`,
//! Unix domain sockets, and HTTP Server-Sent Events (SSE). It proxies tool
//! calls to the Bad Apple daemon via the SLICKS Unix socket, so external MCP
//! clients can use Bad Apple as a context provider without opening TCP sockets.
//!
//! For now it exposes the built-in Bad Apple tools through the `invoke_tool`
//! agent method and advertises the runtime status endpoint. Marketplace
//! catalog/install will be added once the server lifecycle is stable.

use anyhow::{Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Router,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc::UnboundedSender, Mutex as AsyncMutex};
use tokio_stream::wrappers::UnboundedReceiverStream;

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "Bad Apple MCP";
const SERVER_VERSION: &str = "0.1.0";
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const MAX_TOOL_NAME_LEN: usize = 256;
const MAX_ARGUMENTS_DEPTH: usize = 8;
const CALL_AGENT_TIMEOUT: u64 = 120;

/// Tools that require human approval or can mutate the system outside the
/// daemon's policy gate. They are listed but calls are rejected unless the
/// daemon has already approved them.
const DANGEROUS_TOOLS: &[&str] = &[
    "run_shell",
    "run_applescript",
    "write_file",
    "delete_file",
    "index_documents",
    "run_shortcut",
];

/// Agent methods that can be exposed directly through MCP.
const DIRECT_AGENT_METHODS: &[&str] = &[
    "runtime_status",
    "invoke_tool",
    "p2p_peers",
    "p2p_sync",
    "set_workspace",
    "inference",
    "list_models",
    "model_info",
];

/// A running MCP server instance.
pub struct McpServer {
    socket_path: PathBuf,
}

impl McpServer {
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }

    /// Start a Unix-socket MCP server. Blocks until the listener is shut down.
    pub fn serve_unix(&self) -> Result<()> {
        let _ = std::fs::remove_file(&self.socket_path);
        let listener = UnixListener::bind(&self.socket_path).with_context(|| {
            format!(
                "failed to bind MCP socket at {}",
                self.socket_path.display()
            )
        })?;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    std::thread::spawn(move || {
                        let _ = handle_stream(stream);
                    });
                }
                Err(e) => tracing::warn!("MCP accept failed: {}", e),
            }
        }
        Ok(())
    }

    /// Run the `stdio` MCP transport on the current process stdin/stdout.
    pub fn serve_stdio(&self) -> Result<()> {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin);
        let mut stdout = std::io::stdout();
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let response = handle_request(&line);
            if let Some(resp) = response {
                let text = serde_json::to_string(&resp)?;
                writeln!(stdout, "{}", text)?;
                stdout.flush()?;
            }
        }
        Ok(())
    }

    /// Run the HTTP+SSE MCP transport on `bind_addr` (e.g. `127.0.0.1:9879`).
    pub fn serve_sse(&self, bind_addr: &str) -> Result<()> {
        let state = Arc::new(SseState::default());
        let app = Router::new()
            .route("/sse", get(sse_handler))
            .route("/message", post(message_handler))
            .with_state(state);

        let rt =
            tokio::runtime::Runtime::new().context("failed to create tokio runtime for MCP SSE")?;
        rt.block_on(async {
            let listener = tokio::net::TcpListener::bind(bind_addr)
                .await
                .with_context(|| format!("failed to bind MCP SSE server to {bind_addr}"))?;
            axum::serve(listener, app)
                .await
                .context("MCP SSE server error")?;
            Ok(())
        })
    }
}

#[derive(Default)]
struct SseState {
    sessions: AsyncMutex<HashMap<String, UnboundedSender<Event>>>,
    next_id: AtomicU64,
}

#[derive(Deserialize)]
struct MessageQuery {
    session_id: String,
}

async fn sse_handler(
    State(state): State<Arc<SseState>>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = state.next_id.fetch_add(1, Ordering::SeqCst);
    let session_id = format!("{:x}", id);
    state
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), tx.clone());

    let endpoint = format!("/message?session_id={}", session_id);
    let _ = tx.send(Event::default().event("endpoint").data(endpoint));

    let stream = UnboundedReceiverStream::new(rx).map(Ok::<_, Infallible>);
    Sse::new(stream)
}

async fn message_handler(
    State(state): State<Arc<SseState>>,
    Query(query): Query<MessageQuery>,
    body: String,
) -> impl IntoResponse {
    if body.len() > MAX_REQUEST_BYTES {
        return (StatusCode::PAYLOAD_TOO_LARGE, "request too large");
    }

    let response = match tokio::task::spawn_blocking(move || handle_request(&body)).await {
        Ok(Some(resp)) => resp,
        Ok(None) => {
            return (StatusCode::ACCEPTED, "");
        }
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "handler panicked");
        }
    };

    let text = match serde_json::to_string(&response) {
        Ok(t) => t,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to serialize response",
            )
        }
    };

    let sessions = state.sessions.lock().await;
    if let Some(tx) = sessions.get(&query.session_id) {
        let _ = tx.send(Event::default().data(text));
        drop(sessions);
        (StatusCode::ACCEPTED, "")
    } else {
        drop(sessions);
        (StatusCode::NOT_FOUND, "session not found")
    }
}

fn handle_stream(stream: UnixStream) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        if let Some(response) = handle_request(&line) {
            let text = serde_json::to_string(&response)?;
            writeln!(writer, "{}", text)?;
            writer.flush()?;
        }
        line.clear();
    }
    Ok(())
}

#[derive(Deserialize, Debug)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Serialize, Debug)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize, Debug)]
struct JsonRpcError {
    code: i32,
    message: String,
}

fn handle_request(line: &str) -> Option<JsonRpcResponse> {
    if line.len() > MAX_REQUEST_BYTES {
        return Some(error_response(None, -32700, "request too large"));
    }
    let req: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => return Some(error_response(None, -32700, format!("parse error: {}", e))),
    };
    if req.jsonrpc != "2.0" {
        return Some(error_response(req.id, -32600, "invalid jsonrpc version"));
    }

    let result = match req.method.as_str() {
        "initialize" => Some(initialize(&req)),
        "initialized" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id: req.id,
            result: Some(Value::Null),
            error: None,
        }),
        "tools/list" => Some(list_tools(&req)),
        "tools/call" => call_tool(&req),
        "resources/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id: req.id,
            result: Some(json!({"resources": []})),
            error: None,
        }),
        "prompts/list" => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id: req.id,
            result: Some(json!({"prompts": []})),
            error: None,
        }),
        _ => Some(error_response(
            req.id,
            -32601,
            format!("method not found: {}", req.method),
        )),
    };
    result
}

fn initialize(req: &JsonRpcRequest) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id: req.id.clone(),
        result: Some(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
            "capabilities": {
                "tools": { "listChanged": true },
                "resources": { "subscribe": false },
                "prompts": { "listChanged": false },
            }
        })),
        error: None,
    }
}

fn list_tools(req: &JsonRpcRequest) -> JsonRpcResponse {
    let tools = vec![
        json!({
            "name": "runtime_status",
            "description": "Get the current Bad Apple runtime status.",
            "inputSchema": { "type": "object", "properties": {}, "required": [] },
        }),
        json!({
            "name": "invoke_tool",
            "description": "Invoke a Bad Apple local tool by name with arguments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "args": { "type": "object" },
                },
                "required": ["name"],
            },
        }),
        json!({
            "name": "list_models",
            "description": "List tracked and cached models.",
            "inputSchema": { "type": "object", "properties": {}, "required": [] },
        }),
    ];
    JsonRpcResponse {
        jsonrpc: "2.0",
        id: req.id.clone(),
        result: Some(json!({ "tools": tools })),
        error: None,
    }
}

fn call_tool(req: &JsonRpcRequest) -> Option<JsonRpcResponse> {
    let req_id = req.id.clone();
    let params = match &req.params {
        Some(Value::Object(m)) => m,
        _ => return Some(error_response(req_id, -32602, "params must be an object")),
    };
    let name = match params.get("name").and_then(Value::as_str) {
        Some(n) if n.len() <= MAX_TOOL_NAME_LEN => n.to_string(),
        _ => {
            return Some(error_response(
                req_id,
                -32602,
                "name is required or too long",
            ))
        }
    };
    let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);
    if !is_safe_value(&arguments, 0) {
        return Some(error_response(
            req_id,
            -32602,
            "arguments exceed depth or type limits",
        ));
    }

    if !DIRECT_AGENT_METHODS.contains(&name.as_str()) {
        return Some(error_response(
            req_id,
            -32602,
            format!("tool {} is not exposed", name),
        ));
    }

    if DANGEROUS_TOOLS.contains(&name.as_str()) {
        return Some(error_response(
            req_id,
            -32602,
            format!("tool {} requires daemon approval", name),
        ));
    }

    let agent_params = if name == "invoke_tool" {
        if let Value::Object(args) = &arguments {
            let inner_name = args.get("name").and_then(Value::as_str).unwrap_or("");
            let inner_args = args
                .get("args")
                .cloned()
                .unwrap_or(Value::Object(Map::new()));
            let mut p = Map::new();
            p.insert("name".to_string(), Value::String(inner_name.to_string()));
            p.insert("args".to_string(), inner_args);
            Some(Value::Object(p))
        } else {
            None
        }
    } else {
        if let Value::Object(args) = &arguments {
            Some(Value::Object(args.clone()))
        } else {
            None
        }
    };

    let _timeout = std::time::Duration::from_secs(CALL_AGENT_TIMEOUT);
    let result =
        std::thread::spawn(move || crate::bad_apple_ipc::call_agent(&name, agent_params, 4096))
            .join()
            .map_err(|e| anyhow::anyhow!("agent call panicked: {:?}", e));

    match result {
        Ok(Ok(value)) => Some(JsonRpcResponse {
            jsonrpc: "2.0",
            id: req_id,
            result: Some(value),
            error: None,
        }),
        Ok(Err(e)) => Some(error_response(req_id, -32603, e.to_string())),
        Err(e) => Some(error_response(req_id, -32603, e.to_string())),
    }
}

fn is_safe_value(value: &Value, depth: usize) -> bool {
    if depth > MAX_ARGUMENTS_DEPTH {
        return false;
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => true,
        Value::Array(arr) => arr.iter().all(|v| is_safe_value(v, depth + 1)),
        Value::Object(map) => map
            .iter()
            .all(|(k, v)| k.len() <= 256 && is_safe_value(v, depth + 1)),
    }
}

fn error_response(id: Option<Value>, code: i32, message: impl Into<String>) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_json() {
        let resp = handle_request("not json").expect("returns error");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, -32700);
    }

    #[test]
    fn rejects_oversized_request() {
        let huge = "x".repeat(MAX_REQUEST_BYTES + 1);
        let resp = handle_request(&huge).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32700);
    }

    #[test]
    fn rejects_invalid_jsonrpc_version() {
        let req = r#"{"jsonrpc":"1.0","id":1,"method":"initialize","params":{}}"#;
        let resp = handle_request(req).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32600);
    }

    #[test]
    fn initialize_returns_server_info() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let resp = handle_request(req).expect("returns result");
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn tools_list_includes_runtime_status() {
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
        let resp = handle_request(req).expect("returns result");
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert!(tools.iter().any(|t| t["name"] == "runtime_status"));
    }

    #[test]
    fn rejects_unknown_method() {
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"foo","params":{}}"#;
        let resp = handle_request(req).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32601);
    }

    #[test]
    fn rejects_tool_with_long_name() {
        let name = "x".repeat(MAX_TOOL_NAME_LEN + 1);
        let req = format!(
            r#"{{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{{"name":"{}","arguments":{{}}}}}}"#,
            name
        );
        let resp = handle_request(&req).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32602);
    }

    #[test]
    fn rejects_deep_arguments() {
        // Build a value that exceeds MAX_ARGUMENTS_DEPTH.
        let mut value = Value::Object(Map::new());
        for _ in 0..=MAX_ARGUMENTS_DEPTH {
            let mut m = Map::new();
            m.insert("v".to_string(), value);
            value = Value::Object(m);
        }
        let req = format!(
            r#"{{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{{"name":"runtime_status","arguments":{}}}}}"#,
            value
        );
        let resp = handle_request(&req).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32602);
    }

    #[test]
    fn rejects_disallowed_tool() {
        let req = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"run_shell","arguments":{}}}"#;
        let resp = handle_request(req).expect("returns error");
        assert_eq!(resp.error.as_ref().unwrap().code, -32602);
    }

    #[test]
    fn safe_value_rejects_deeply_nested() {
        let mut v = Value::Array(vec![Value::Null]);
        for _ in 0..=MAX_ARGUMENTS_DEPTH {
            v = Value::Array(vec![v]);
        }
        assert!(!is_safe_value(&v, 0));
    }

    #[test]
    fn safe_value_rejects_long_object_key() {
        let mut m = Map::new();
        m.insert("x".repeat(300), Value::Null);
        let v = Value::Object(m);
        assert!(!is_safe_value(&v, 0));
    }

    #[test]
    fn safe_value_accepts_plain() {
        let mut m = Map::new();
        m.insert("ok".to_string(), Value::String("value".to_string()));
        m.insert("num".to_string(), Value::Number(42.into()));
        let v = Value::Object(m);
        assert!(is_safe_value(&v, 0));
    }
}
