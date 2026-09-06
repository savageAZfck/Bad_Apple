//! Bad Apple dashboard HTTP server.
//!
//! Serves the static `web/` single-page app and proxies API calls to the
//! native Swift daemon over the SLICKS Unix socket. This replaces the
//! deleted `badapple_dashboard.py` with a Rust `axum` implementation that
//! is off by default for air-gap certification.

use anyhow::{Context, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct DashboardState {
    web_root: PathBuf,
    helpers: HelperPaths,
    ocular: Arc<RwLock<OcularState>>,
}

#[derive(Clone)]
struct HelperPaths {
    ambient: PathBuf,
    screen_capture: PathBuf,
}

struct OcularState {
    running: bool,
    capture_interval: f64,
    describe_interval: f64,
    prompt: String,
    last_png: Option<Vec<u8>>,
    last_description: Option<String>,
    last_error: Option<String>,
    worker: Option<JoinHandle<()>>,
}

impl OcularState {
    fn new() -> Self {
        Self {
            running: false,
            capture_interval: 5.0,
            describe_interval: 0.0,
            prompt: "Describe what is currently on the screen in a few sentences, focusing on the active window and visible text.".to_string(),
            last_png: None,
            last_description: None,
            last_error: None,
            worker: None,
        }
    }
}

#[derive(Deserialize)]
struct ChatQuery {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct P2PAction {
    peer_id: Option<String>,
    model_id: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let port = std::env::var("BADAPPLE_DASHBOARD_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);

    let web_root = std::env::var("BADAPPLE_WEB_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("web"));

    if !web_root.is_dir() {
        anyhow::bail!("web root '{}' does not exist", web_root.display());
    }

    let helpers = helper_paths();
    let state = Arc::new(DashboardState {
        web_root: web_root.clone(),
        helpers,
        ocular: Arc::new(RwLock::new(OcularState::new())),
    });

    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    rt.block_on(run_server(port, web_root, state))
}

fn helper_paths() -> HelperPaths {
    let release_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("target")
        .join("release");

    let ambient = std::env::var("BADAPPLE_AMBIENT_HELPER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| release_dir.join("BadAppleAmbient"));

    let screen_capture = std::env::var("BADAPPLE_SCREEN_CAPTURE_HELPER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| release_dir.join("BadAppleScreenCapture"));

    HelperPaths {
        ambient,
        screen_capture,
    }
}

async fn run_server(port: u16, web_root: PathBuf, state: Arc<DashboardState>) -> Result<()> {
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/status", get(status_handler))
        .route(
            "/api/models",
            get(list_models_handler).post(models_action_handler),
        )
        .route("/api/models/:id", get(model_info_handler))
        .route("/api/models/:id/use", post(use_model_handler))
        .route("/api/models/:id/verify", post(verify_model_handler))
        .route("/api/agents", post(agents_action_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/p2p/peers", get(p2p_peers_handler))
        .route("/api/p2p/sync", post(p2p_sync_handler))
        .route("/api/p2p/models", get(p2p_models_handler))
        .route("/api/p2p/pull", post(p2p_pull_handler))
        .route("/api/p2p/send", post(p2p_send_handler))
        .route("/api/mcp/servers", get(mcp_servers_handler))
        .route("/api/mcp/servers", post(mcp_add_server_handler))
        .route("/api/mcp/servers/:id", delete(mcp_remove_server_handler))
        .route("/api/ambient", get(ambient_handler))
        .route("/api/ocular", get(ocular_handler))
        .route("/api/ocular", post(ocular_action_handler))
        .route("/api/ocular/screen.png", get(ocular_screen_handler))
        .nest_service("/static", ServeDir::new(web_root.join("static")))
        .fallback(static_handler)
        .with_state(state);

    // Bind to loopback only. The cert suite treats an all-interfaces listener
    // as an external network socket, and it is the safer default for a local
    // dashboard anyway.
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind dashboard to {}", addr))?;

    tracing::info!("Bad Apple dashboard listening on http://{}", addr);
    axum::serve(listener, app)
        .await
        .context("dashboard server error")?;

    Ok(())
}

async fn index_handler(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let path = state.web_root.join("index.html");
    match tokio::fs::read_to_string(&path).await {
        Ok(html) => Html(html),
        Err(e) => Html(format!("<h1>Dashboard unavailable</h1><pre>{}</pre>", e)),
    }
}

async fn static_handler(
    State(state): State<Arc<DashboardState>>,
    uri: axum::http::Uri,
) -> impl IntoResponse {
    let mut path = state.web_root.clone();
    let req_path = uri.path().trim_start_matches('/');
    for segment in req_path.split('/') {
        if segment.is_empty() || segment == ".." {
            continue;
        }
        path.push(segment);
    }

    // Standalone HTML pages (e.g. /models -> web/models.html).
    let html_path = path.with_extension("html");
    if tokio::fs::metadata(&html_path).await.is_ok() {
        return serve_file(&html_path).await;
    }

    // Static files / directories (e.g. /static/..., /models/...).
    if path.is_dir() {
        path.push("index.html");
    }
    if tokio::fs::metadata(&path).await.is_ok() {
        return serve_file(&path).await;
    }

    // SPA catch-all: index.html handles client-side routing for /chat, /settings, etc.
    let index = state.web_root.join("index.html");
    serve_file(&index).await
}

fn content_type_for(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|s| s.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

async fn serve_file(path: &std::path::Path) -> Response {
    match tokio::fs::read(path).await {
        Ok(bytes) => Response::builder()
            .header("Content-Type", content_type_for(path))
            .body(axum::body::Body::from(bytes))
            .unwrap()
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn status_handler(State(_): State<Arc<DashboardState>>) -> impl IntoResponse {
    match agent_call("runtime_status", None).await {
        Ok(v) => Json(adapt_runtime_status(v)),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// Adapt the native daemon's `runtime_status` schema to the one the web
/// dashboard's SPA expects.
fn adapt_runtime_status(mut v: Value) -> Value {
    let Some(obj) = v.as_object_mut() else {
        return v;
    };

    // The native runtime reports `model_status` keyed by model ID and
    // `active_model_ids`. The dashboard wants `active_models`, `runtime.mode`,
    // `main_model_loaded`, `health.checks.main_model.ok`, and a few flattened
    // hibernation/ambient fields.
    let active_ids = obj
        .get("active_model_ids")
        .and_then(|a| a.as_array())
        .and_then(|a| {
            a.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .into_iter()
                .next()
        })
        .unwrap_or_else(|| "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit".to_string());
    // Prefer the first non-embedding active model as the "main" model the
    // dashboard splash refers to. Embedding-only loads shouldn't be the hero.
    let main_id = obj
        .get("active_model_ids")
        .and_then(|a| a.as_array())
        .and_then(|a| {
            a.iter().filter_map(|x| x.as_str()).find(|s| {
                !s.to_lowercase().contains("bge") && !s.to_lowercase().contains("embedding")
            })
        })
        .unwrap_or(&active_ids)
        .to_string();

    let main_ready = obj
        .get("model_status")
        .and_then(|m| m.get(&main_id))
        .and_then(|m| m.get("status"))
        .and_then(|s| s.as_str())
        .map(|s| s == "ready")
        .unwrap_or(false);

    let private_mode = obj.get("private_mode").cloned().unwrap_or(json!(false));

    let mode = if main_ready { "READY" } else { "STARTING" };

    let runtime = json!({
        "mode": mode,
        "killed": false,
        "safe_mode_reason": null,
        "private_mode": private_mode,
    });

    let health = json!({
        "checks": {
            "main_model": {
                "ok": main_ready,
                "detail": main_id,
            }
        }
    });

    // Flatten hibernation fields for the dashboard cards.
    if let Some(hibernation) = obj.get("hibernation").cloned() {
        if let Some(hobj) = hibernation.as_object() {
            obj.insert(
                "hibernating".to_string(),
                hobj.get("active").cloned().unwrap_or(json!(false)),
            );
            obj.insert(
                "idle_seconds".to_string(),
                hobj.get("idle_seconds").cloned().unwrap_or(json!(0)),
            );
            obj.insert(
                "hibernate_after".to_string(),
                hobj.get("idle_threshold_seconds")
                    .cloned()
                    .unwrap_or(json!(300)),
            );
        }
    }

    // The dashboard uses `active_models` and `models.main_9b` for the splash.
    if let Some(active) = obj.get("active_model_ids").cloned() {
        obj.insert("active_models".to_string(), active.clone());
        // Expose the first active model under the legacy `main_9b` key the SPA
        // looks for during startup. The name is historical (9B used to be main).
        if let Some(first) = active
            .as_array()
            .and_then(|a| a.first())
            .and_then(|x| x.as_str())
        {
            if let Some(model_status) = obj.get("model_status").cloned() {
                if let Some(ms) = model_status.get(first).cloned() {
                    let models = json!({ "main_9b": ms });
                    obj.insert("models".to_string(), models);
                }
            }
        }
    }

    // Map ambient context to the field renderDashboard expects.
    let ambient = obj.get("ambient_context").cloned().unwrap_or(json!(null));
    obj.insert("ambient".to_string(), ambient);

    // Provide sensible defaults for fields the dashboard tests/renders.
    if !obj.contains_key("p2p_enabled") {
        obj.insert("p2p_enabled".to_string(), json!(false));
    }
    if !obj.contains_key("autopilot") {
        obj.insert("autopilot".to_string(), json!(false));
    }
    if !obj.contains_key("p2p_peers") {
        obj.insert(
            "p2p_peers".to_string(),
            json!("No peers on the local network."),
        );
    }
    if !obj.contains_key("ambient_running") {
        obj.insert("ambient_running".to_string(), json!(false));
    }

    obj.insert("runtime".to_string(), runtime);
    obj.insert("main_model_loaded".to_string(), json!(main_ready));
    obj.insert("health".to_string(), health);

    v
}

async fn list_models_handler() -> impl IntoResponse {
    match agent_call("list_models", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn model_info_handler(Path(id): Path<String>) -> impl IntoResponse {
    let mut params = Map::new();
    params.insert("model_id".to_string(), Value::String(id));
    match agent_call("model_info", Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn use_model_handler(Path(id): Path<String>) -> impl IntoResponse {
    let mut params = Map::new();
    params.insert("model_ref".to_string(), Value::String(id));
    match agent_call("switch_main_model", Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn verify_model_handler(Path(id): Path<String>) -> impl IntoResponse {
    let mut params = Map::new();
    params.insert("model_id".to_string(), Value::String(id));
    match agent_call("verify_models", Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn models_action_handler(Json(body): Json<Value>) -> impl IntoResponse {
    let action = body.get("action").and_then(|a| a.as_str()).unwrap_or("");
    match action {
        "status" => match agent_call("list_models", None).await {
            Ok(v) => Json(models_list_to_map(v)),
            Err(e) => Json(json!({"error": e.to_string()})),
        },
        "recommend" => {
            let query = body.get("query").and_then(|q| q.as_str()).unwrap_or("");
            let mut params = Map::new();
            params.insert("query".to_string(), Value::String(query.to_string()));
            match agent_call("recommend_model", Some(Value::Object(params))).await {
                Ok(v) => Json(v),
                Err(e) => Json(json!({"error": e.to_string()})),
            }
        }
        "switch" => {
            let model_ref = body.get("model_ref").and_then(|r| r.as_str()).unwrap_or("");
            let mut params = Map::new();
            params.insert(
                "model_ref".to_string(),
                Value::String(model_ref.to_string()),
            );
            match agent_call("switch_main_model", Some(Value::Object(params))).await {
                Ok(v) => Json(v),
                Err(e) => Json(json!({"error": e.to_string()})),
            }
        }
        "admit" => {
            let model_ref = body.get("model_ref").and_then(|r| r.as_str()).unwrap_or("");
            if model_ref.is_empty() {
                return Json(json!({"error": "model_ref is required"}));
            }
            let mut rec_params = Map::new();
            rec_params.insert("query".to_string(), Value::String(model_ref.to_string()));
            let mut info_params = Map::new();
            info_params.insert("model_id".to_string(), Value::String(model_ref.to_string()));
            let (rec, info) = tokio::join!(
                agent_call("recommend_model", Some(Value::Object(rec_params))),
                agent_call("model_info", Some(Value::Object(info_params)))
            );
            match (rec, info) {
                (Ok(r), Ok(i)) => {
                    let available = r
                        .get("available_gb")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let needed = i.get("size_gb").and_then(|v| v.as_f64()).unwrap_or(0.0) * 1.4;
                    let ok = available >= needed;
                    Json(json!({
                        "ok": ok,
                        "available_gb": round2(available),
                        "needed_gb": round2(needed),
                    }))
                }
                (Err(e), _) | (_, Err(e)) => Json(json!({"error": e.to_string()})),
            }
        }
        "refresh" => {
            let model_id = body.get("model_id").and_then(|r| r.as_str()).unwrap_or("");
            if model_id.is_empty() {
                match agent_call("scan_models", None).await {
                    Ok(v) => Json(v),
                    Err(e) => Json(json!({"error": e.to_string()})),
                }
            } else {
                let mut params = Map::new();
                params.insert("model_id".to_string(), Value::String(model_id.to_string()));
                match agent_call("model_info", Some(Value::Object(params))).await {
                    Ok(v) => Json(v),
                    Err(e) => Json(json!({"error": e.to_string()})),
                }
            }
        }
        "allow_downloads" => {
            let enabled = body
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let mut params = Map::new();
            params.insert("enabled".to_string(), json!(enabled));
            match agent_call("set_allow_downloads", Some(Value::Object(params))).await {
                Ok(v) => Json(v),
                Err(e) => Json(json!({"error": e.to_string()})),
            }
        }
        "download" => {
            let model_id = body.get("model_id").and_then(|r| r.as_str()).unwrap_or("");
            if model_id.is_empty() {
                return Json(json!({"error": "model_id is required"}));
            }
            let mut params = Map::new();
            params.insert("model_id".to_string(), Value::String(model_id.to_string()));
            match agent_call("download_model", Some(Value::Object(params))).await {
                Ok(status) => {
                    let s = status.get("status").cloned().unwrap_or(json!("missing"));
                    let size_gb = status
                        .get("size_gb")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    // Build a memory-check summary for the UI alert. The model
                    // manager queues the HF download; the check here only
                    // reflects the declared size, not real-time VRAM after load.
                    let mut mem = Map::new();
                    if s == "queued" || s == "downloading" || s == "cached" || s == "loaded" {
                        mem.insert("ok".to_string(), json!(true));
                        mem.insert(
                            "message".to_string(),
                            json!(format!("{size_gb} GB download queued")),
                        );
                    } else {
                        mem.insert("ok".to_string(), json!(false));
                        mem.insert(
                            "message".to_string(),
                            json!(status
                                .get("error")
                                .and_then(|e| e.as_str())
                                .unwrap_or("download did not start")),
                        );
                    }
                    Json(json!({
                        "status": s,
                        "memory_check": mem,
                    }))
                }
                Err(e) => Json(json!({"error": e.to_string()})),
            }
        }
        _ => Json(json!({"error": format!("unknown model action: {action}")})),
    }
}

fn models_list_to_map(v: Value) -> Value {
    let Some(obj) = v.as_object() else { return v };
    let mut map = Map::new();
    if let Some(arr) = obj.get("models").and_then(|m| m.as_array()) {
        for m in arr {
            if let Some(id) = m.get("id").and_then(|i| i.as_str()) {
                map.insert(id.to_string(), m.clone());
            }
        }
    }
    Value::Object(map)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

async fn agents_action_handler(Json(body): Json<Value>) -> impl IntoResponse {
    let action = body.get("action").and_then(|a| a.as_str()).unwrap_or("");
    let method = match action {
        "list" => "list_agent_tasks",
        "create" => "run_agent_task",
        "cancel" => "cancel_agent_task",
        "pause" => "pause_agent_task",
        "resume" => "resume_agent_task",
        "delete" => "cancel_agent_task", // engine has no delete; cancel stops it
        _ => return Json(json!({"error": format!("unknown agent action: {action}")})),
    };
    let mut params = Map::new();
    if let Some(goal) = body.get("goal") {
        params.insert("goal".to_string(), goal.clone());
    }
    if let Some(max_steps) = body.get("max_steps") {
        params.insert("max_steps".to_string(), max_steps.clone());
    }
    if let Some(task_id) = body.get("task_id") {
        params.insert("task_id".to_string(), task_id.clone());
    }
    match agent_call(method, Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn chat_handler(Json(query): Json<ChatQuery>) -> impl IntoResponse {
    let max_tokens = query.max_tokens.unwrap_or(512);
    let prompt = query.prompt;

    let result = tokio::task::spawn_blocking(move || {
        bad_apple::bad_apple_ipc::query_with_metrics(&prompt, max_tokens, |_token| {})
            .map(|(text, _)| text)
    })
    .await;

    match result {
        Ok(Ok(text)) => Json(json!({"text": text})).into_response(),
        Ok(Err(e)) => Json(json!({"error": e.to_string()})).into_response(),
        Err(e) => Json(json!({"error": e.to_string()})).into_response(),
    }
}

async fn p2p_peers_handler() -> impl IntoResponse {
    match agent_call("p2p_peers", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn p2p_sync_handler() -> impl IntoResponse {
    match agent_call("p2p_sync", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn p2p_models_handler() -> impl IntoResponse {
    match agent_call("p2p_models", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn p2p_pull_handler(Json(action): Json<P2PAction>) -> impl IntoResponse {
    let mut params = Map::new();
    if let Some(peer_id) = action.peer_id {
        params.insert("peer_id".to_string(), Value::String(peer_id));
    }
    if let Some(model_id) = action.model_id {
        params.insert("model_id".to_string(), Value::String(model_id));
    }
    match agent_call("p2p_pull_model", Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn p2p_send_handler(Json(action): Json<P2PAction>) -> impl IntoResponse {
    let mut params = Map::new();
    if let Some(peer_id) = action.peer_id {
        params.insert("peer_id".to_string(), Value::String(peer_id));
    }
    if let Some(model_id) = action.model_id {
        params.insert("model_id".to_string(), Value::String(model_id));
    }
    match agent_call("p2p_send_model", Some(Value::Object(params))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

fn mcp_catalog_path() -> PathBuf {
    std::env::var("BADAPPLE_MCP_CATALOG_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("/var/lib/bad_apple"))
                .join("bad_apple")
                .join("mcp_catalog.json")
        })
}

async fn load_marketplace() -> bad_apple::mcp_marketplace::McpMarketplace {
    let market = bad_apple::mcp_marketplace::McpMarketplace::new(mcp_catalog_path(), 100);
    let _ = market.load().await;
    market
}

async fn mcp_servers_handler() -> impl IntoResponse {
    let market = load_marketplace().await;
    Json(json!({"servers": market.list().await})).into_response()
}

#[derive(Deserialize)]
struct McpServerInput {
    id: String,
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: std::collections::HashMap<String, String>,
    #[serde(default = "default_mcp_transport")]
    transport: String,
    #[serde(default)]
    description: String,
}

fn default_mcp_transport() -> String {
    "stdio".to_string()
}

async fn mcp_add_server_handler(Json(input): Json<McpServerInput>) -> impl IntoResponse {
    let market = load_marketplace().await;
    let transport = match input.transport.as_str() {
        "sse" => bad_apple::mcp_marketplace::McpTransport::Sse,
        "socket" => bad_apple::mcp_marketplace::McpTransport::Socket,
        _ => bad_apple::mcp_marketplace::McpTransport::Stdio,
    };
    let server = bad_apple::mcp_marketplace::McpServer {
        id: input.id,
        name: input.name,
        command: input.command,
        args: input.args,
        env: input.env,
        transport,
        installed: true,
        enabled: true,
        description: input.description,
    };
    match market.upsert(server).await {
        Ok(()) => {
            let _ = market.save().await;
            Json(json!({"status": "ok"})).into_response()
        }
        Err(e) => Json(json!({"error": e.to_string()})).into_response(),
    }
}

async fn mcp_remove_server_handler(Path(id): Path<String>) -> impl IntoResponse {
    let market = load_marketplace().await;
    match market.remove(&id).await {
        Ok(()) => {
            let _ = market.save().await;
            Json(json!({"status": "ok"})).into_response()
        }
        Err(e) => Json(json!({"error": e.to_string()})).into_response(),
    }
}

async fn agent_call(method: &str, params: Option<Value>) -> Result<Value> {
    let method = method.to_string();
    tokio::task::spawn_blocking(move || bad_apple::bad_apple_ipc::call_agent(&method, params, 4096))
        .await
        .map_err(|e| anyhow::anyhow!("spawn blocking failed: {e}"))?
}

async fn ambient_handler(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let helper = state.helpers.ambient.clone();
    match tokio::process::Command::new(&helper).output().await {
        Ok(output) if output.status.success() => {
            match serde_json::from_slice::<Value>(&output.stdout) {
                Ok(v) => Json(v).into_response(),
                Err(e) => Json(json!({"error": format!("failed to parse ambient JSON: {e}")}))
                    .into_response(),
            }
        }
        Ok(output) => Json(json!({"error": String::from_utf8_lossy(&output.stderr).to_string()}))
            .into_response(),
        Err(e) => Json(json!({"error": e.to_string()})).into_response(),
    }
}

async fn ocular_handler(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let o = state.ocular.read().await;
    Json(json!({
        "running": o.running,
        "capture_interval": o.capture_interval,
        "describe_interval": o.describe_interval,
        "prompt": o.prompt,
        "last_description": o.last_description,
        "last_error": o.last_error,
    }))
}

#[derive(Deserialize)]
struct OcularAction {
    action: String,
    #[serde(default)]
    capture_interval: Option<f64>,
    #[serde(default)]
    describe_interval: Option<f64>,
    #[serde(default)]
    prompt: Option<String>,
}

async fn ocular_action_handler(
    State(state): State<Arc<DashboardState>>,
    Json(action): Json<OcularAction>,
) -> impl IntoResponse {
    let mut o = state.ocular.write().await;
    if let Some(interval) = action.capture_interval {
        o.capture_interval = interval.max(1.0);
    }
    if let Some(interval) = action.describe_interval {
        o.describe_interval = interval.max(0.0);
    }
    if let Some(prompt) = action.prompt {
        o.prompt = prompt;
    }

    match action.action.as_str() {
        "start" => {
            o.running = true;
            let ocular = Arc::clone(&state.ocular);
            let helpers = state.helpers.clone();
            let handle = tokio::spawn(ocular_worker(
                ocular,
                helpers,
                o.capture_interval,
                o.describe_interval,
            ));
            o.worker = Some(handle);
            Json(json!({"status": "started"})).into_response()
        }
        "stop" => {
            o.running = false;
            if let Some(handle) = o.worker.take() {
                handle.abort();
            }
            Json(json!({"status": "stopped"})).into_response()
        }
        "capture" => {
            drop(o);
            match capture_screen(&state.helpers.screen_capture).await {
                Ok(png) => {
                    let mut o = state.ocular.write().await;
                    o.last_png = Some(png);
                    o.last_error = None;
                    Json(json!({"status": "captured"})).into_response()
                }
                Err(e) => Json(json!({"error": e.to_string()})).into_response(),
            }
        }
        _ => Json(json!({"error": "unknown action"})).into_response(),
    }
}

async fn ocular_screen_handler(State(state): State<Arc<DashboardState>>) -> impl IntoResponse {
    let o = state.ocular.read().await;
    match &o.last_png {
        Some(png) => Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "image/png")
            .body(axum::body::Body::from(png.clone()))
            .unwrap()
            .into_response(),
        None => (StatusCode::NOT_FOUND, "no capture available").into_response(),
    }
}

async fn capture_screen(helper: &PathBuf) -> Result<Vec<u8>> {
    let mut path = std::env::temp_dir();
    path.push(format!("badapple_ocular_{}.png", std::process::id()));

    let output = tokio::process::Command::new(helper)
        .arg("--output")
        .arg(&path)
        .output()
        .await
        .context("failed to run screen capture helper")?;

    if !output.status.success() {
        anyhow::bail!(
            "screen capture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let bytes = tokio::fs::read(&path)
        .await
        .context("failed to read captured screen")?;
    let _ = tokio::fs::remove_file(&path).await;
    Ok(bytes)
}

async fn ocular_worker(
    state: Arc<RwLock<OcularState>>,
    helpers: HelperPaths,
    capture_interval: f64,
    describe_interval: f64,
) {
    let mut last_describe = std::time::Instant::now();
    loop {
        match capture_screen(&helpers.screen_capture).await {
            Ok(png) => {
                let mut o = state.write().await;
                o.last_png = Some(png);
                o.last_error = None;
            }
            Err(e) => {
                let mut o = state.write().await;
                o.last_error = Some(e.to_string());
                o.running = false;
                break;
            }
        }

        if describe_interval > 0.0 {
            let elapsed = last_describe.elapsed().as_secs_f64();
            if elapsed >= describe_interval {
                last_describe = std::time::Instant::now();
                // The dashboard could stream the image to the daemon for a
                // VLM description, but that requires a dedicated image-to-text
                // tool.  Leave the placeholder so the UI receives a friendly
                // "not yet" instead of an error.
                let prompt = state.read().await.prompt.clone();
                let _ = state.write().await.last_description;
                state.write().await.last_description = Some(format!(
                    "[VLM description not yet enabled; prompt: {}]",
                    prompt
                ));
            }
        }

        tokio::time::sleep(Duration::from_secs_f64(capture_interval)).await;

        let still_running = state.read().await.running;
        if !still_running {
            break;
        }
    }
}
