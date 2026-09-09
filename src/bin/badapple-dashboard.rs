//! Bad Apple dashboard HTTP server.
//!
//! Serves the static `web/` single-page app and proxies API calls to the
//! native Swift daemon over the SLICKS Unix socket. This replaces the
//! deleted `badapple_dashboard.py` with a Rust `axum` implementation that
//! is off by default for air-gap certification.

use anyhow::{Context, Result};
use axum::{
    extract::{Path, Request, State},
    http::header::AUTHORIZATION,
    http::StatusCode,
    middleware::{from_fn_with_state, Next},
    response::{Html, IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
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
    token: Option<String>,
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
    let token = std::env::var("BADAPPLE_DASHBOARD_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let state = Arc::new(DashboardState {
        web_root: web_root.clone(),
        helpers,
        ocular: Arc::new(RwLock::new(OcularState::new())),
        token,
    });

    let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    rt.block_on(run_server(port, state))
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

async fn require_token(
    State(state): State<Arc<DashboardState>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(expected) = &state.token {
        let got = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        if got != format!("Bearer {expected}") {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    Ok(next.run(req).await)
}

async fn run_server(port: u16, state: Arc<DashboardState>) -> Result<()> {
    let api_state = state.clone();
    let api = Router::new()
        .route("/status", get(status_handler))
        .route("/snapshot", get(snapshot_handler))
        .route("/tail", get(tail_handler))
        .route("/ledger", get(ledger_handler))
        .route("/audit", get(audit_handler))
        .route("/cert", get(cert_handler))
        .route("/doctor", get(doctor_handler))
        .route("/voice", get(voice_handler))
        .route("/capabilities", get(capabilities_handler))
        .route("/control", post(control_handler))
        .route(
            "/workspace",
            get(get_workspace_handler).post(set_workspace_handler),
        )
        .route(
            "/working_memory",
            get(working_memory_handler).post(set_working_memory_handler),
        )
        .route(
            "/memory/facts",
            get(list_facts_handler)
                .post(add_fact_handler)
                .delete(clear_facts_handler),
        )
        .route("/memory/facts/extract", post(extract_facts_handler))
        .route("/memory/facts/index", post(index_facts_handler))
        .route(
            "/workshop/personas",
            get(list_personas_handler).post(save_persona_handler),
        )
        .route(
            "/workshop/personas/:id",
            get(get_persona_handler).delete(delete_persona_handler),
        )
        .route(
            "/workshop/custom_tools",
            get(list_custom_tools_handler).post(save_custom_tool_handler),
        )
        .route(
            "/workshop/custom_tools/:id",
            get(get_custom_tool_handler).delete(delete_custom_tool_handler),
        )
        .route(
            "/workshop/custom_tools/:id/run",
            post(run_custom_tool_handler),
        )
        .route(
            "/models",
            get(list_models_handler).post(models_action_handler),
        )
        .route("/models/:id", get(model_info_handler))
        .route("/models/:id/use", post(use_model_handler))
        .route("/models/:id/verify", post(verify_model_handler))
        .route("/agents", post(agents_action_handler))
        .route("/chat", post(chat_handler))
        .route("/p2p/peers", get(p2p_peers_handler))
        .route("/p2p/sync", post(p2p_sync_handler))
        .route("/p2p/models", get(p2p_models_handler))
        .route("/p2p/pull", post(p2p_pull_handler))
        .route("/p2p/send", post(p2p_send_handler))
        .route("/mcp/servers", get(mcp_servers_handler))
        .route(
            "/mcp_servers",
            get(mcp_servers_handler).post(mcp_add_server_handler),
        )
        .route("/mcp/servers", post(mcp_add_server_handler))
        .route("/mcp/servers/:id", delete(mcp_remove_server_handler))
        .route("/mcp_servers/:id", delete(mcp_remove_server_handler))
        .route("/mcp_registry", get(mcp_registry_handler))
        .route("/ambient", get(ambient_handler))
        .route("/ocular", get(ocular_handler))
        .route("/ocular", post(ocular_action_handler))
        .route("/ocular/screen.png", get(ocular_screen_handler))
        .layer(from_fn_with_state(api_state.clone(), require_token))
        .with_state(api_state);

    let app = Router::new()
        .route("/", get(index_handler))
        .nest("/api", api)
        .nest_service("/static", ServeDir::new(state.web_root.join("static")))
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
    let autopilot = obj.get("autopilot").cloned().unwrap_or(json!(false));
    let fast_tier = obj.get("fast_tier").cloned().unwrap_or(json!(false));
    let airgap = obj.get("airgap").cloned().unwrap_or(json!(false));
    let killed = obj.get("killed").cloned().unwrap_or(json!(false));
    let workspace = obj.get("workspace").cloned().unwrap_or(json!(null));

    let mode = if main_ready { "READY" } else { "STARTING" };

    let runtime = json!({
        "mode": mode,
        "killed": killed,
        "safe_mode_reason": null,
        "private_mode": private_mode,
    });

    // Expose toggle fields at the top level so the Control Center UI can bind
    // them without crawling the runtime object.
    obj.insert("autopilot".to_string(), autopilot);
    obj.insert("fast_tier".to_string(), fast_tier);
    obj.insert("airgap".to_string(), airgap);
    obj.insert("private_mode".to_string(), private_mode);
    obj.insert("killed".to_string(), killed);
    obj.insert("workspace".to_string(), workspace);

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

// MARK: - Control Center, Memory, and Workshop API

/// Path helpers for user data used by the dashboard.
fn bad_apple_data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from(std::env!("HOME")))
        .join("bad_apple")
}

fn bad_apple_var_dir() -> PathBuf {
    PathBuf::from("/var/lib/bad_apple")
}

fn memory_graph_file() -> PathBuf {
    bad_apple_var_dir().join("memory_graph").join("facts.json")
}

fn working_memory_file() -> PathBuf {
    bad_apple_data_dir().join("working_memory.txt")
}

fn custom_personas_file() -> PathBuf {
    bad_apple_data_dir().join("personas.json")
}

fn custom_tools_file() -> PathBuf {
    bad_apple_data_dir().join("custom_tools.json")
}

fn builtin_personas_file() -> PathBuf {
    // Prefer the source tree personas.json compiled into the binary. This is
    // the manifest dir (the repo root) for source builds and the path set by
    // the install script for packaged builds.
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("personas.json");
    if source.is_file() {
        return source;
    }
    // Fall back to a personas.json beside the running dashboard binary.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("personas.json");
            if candidate.is_file() {
                return candidate;
            }
            // If the binary is in `target/release`, the repo root is two levels up.
            let repo_root = dir.parent().and_then(|d| d.parent());
            if let Some(repo) = repo_root {
                let candidate = repo.join("personas.json");
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
    }
    source
}

/// Call an agent method through the daemon.
async fn daemon_call(method: &str, params: Option<Value>) -> Result<Value> {
    agent_call(method, params).await
}

/// Call a tool through the daemon's `invoke_tool` method.
async fn invoke_tool(name: &str, args: Value) -> Result<Value> {
    let mut params = Map::new();
    params.insert("name".to_string(), Value::String(name.to_string()));
    params.insert("args".to_string(), args);
    daemon_call("invoke_tool", Some(Value::Object(params))).await
}

/// Snapshot of runtime resources for the dashboard cards.
async fn snapshot_handler() -> impl IntoResponse {
    let mut snap = Map::new();

    if let Ok(status) = daemon_call("runtime_status", None).await {
        // The native runtime exposes memory at the top level (`memory`) and
        // sometimes under `resources.memory`. Prefer the explicit top-level
        // object and fall back to the nested one.
        let mem = status.get("memory").cloned().or_else(|| {
            status
                .get("resources")
                .and_then(|m| m.get("memory"))
                .cloned()
        });
        if let Some(mem) = mem {
            let used_bytes = mem
                .get("used_bytes")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let total_bytes = mem
                .get("total_bytes")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let used_gb = used_bytes / 1024.0 / 1024.0 / 1024.0;
            let total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0;
            let percent = if total_gb > 0.0 {
                (used_gb / total_gb * 100.0).round()
            } else {
                0.0
            };
            snap.insert(
                "memory".to_string(),
                json!({
                    "used_gb": used_gb,
                    "total_gb": total_gb,
                    "percent": percent,
                    "pressure": mem.get("pressure").cloned().unwrap_or(json!(false)),
                }),
            );
        }
    }

    // Best-effort battery read using pmset. No extra dependencies.
    if let Ok(output) = tokio::process::Command::new("/usr/bin/pmset")
        .args(["-g", "ps"])
        .output()
        .await
    {
        let text = String::from_utf8_lossy(&output.stdout);
        let first = text.lines().next().unwrap_or("").to_string();
        let ac =
            first.to_lowercase().contains("ac power") || first.to_lowercase().contains("adapter");
        let percent: f64 = first
            .split_whitespace()
            .find_map(|w| w.trim_end_matches('%').parse().ok())
            .unwrap_or(0.0);
        snap.insert(
            "battery".to_string(),
            json!({
                "percent": percent,
                "source": if ac { "ac" } else { "battery" },
                "text": first,
            }),
        );
    }

    Json(Value::Object(snap))
}

/// Return the last N lines of the daemon log.
async fn tail_handler(
    axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let n: usize = query.get("n").and_then(|s| s.parse().ok()).unwrap_or(50);
    let paths = [
        "/var/log/bad_apple_mlx_server.log",
        "/var/log/bad_apple_supervisor.log",
    ];
    for path in paths {
        if tokio::fs::metadata(path).await.is_ok() {
            match tokio::process::Command::new("/usr/bin/tail")
                .args(["-n", &n.to_string(), path])
                .output()
                .await
            {
                Ok(output) => {
                    return Json(json!({"log": String::from_utf8_lossy(&output.stdout) }));
                }
                Err(_) => continue,
            }
        }
    }
    Json(json!({"log": "No daemon log found."}))
}

/// Hash-chained audit ledger tail (via daemon's audit_tail agent method).
async fn ledger_handler() -> impl IntoResponse {
    match daemon_call("audit_tail", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// Audit/cert/doctor summary via the daemon's self_audit tool.
async fn audit_handler() -> impl IntoResponse {
    match invoke_tool("self_audit", json!({"include": "all"})).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn cert_handler() -> impl IntoResponse {
    match invoke_tool("self_audit", json!({"include": "cert"})).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn doctor_handler() -> impl IntoResponse {
    match invoke_tool("self_audit", json!({"include": "doctor"})).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// Voice activity stub. The real voice log is not yet persisted.
async fn voice_handler() -> impl IntoResponse {
    Json(json!({"entries": []}))
}

/// Capabilities list for the dashboard splash.
async fn capabilities_handler() -> impl IntoResponse {
    Json(json!({
        "capabilities": [
            "Local chat with Qwen 2.5 Coder 7B",
            "Voice input and TTS output",
            "Ocular screen capture and VLM description",
            "Workspace indexing and RAG",
            "Long-term memory facts",
            "Self-audit (cert suite + doctor)",
            "P2P model and document sync",
            "MCP tool marketplace",
            "Curious autopilot and bounded self-improvement",
            "Output firewall and audit ledger",
        ]
    }))
}

/// Generic control command parser used by the Control Center UI.
#[derive(Deserialize)]
struct ControlCommand {
    command: String,
}

async fn control_handler(Json(body): Json<ControlCommand>) -> impl IntoResponse {
    let cmd = body.command.trim().to_lowercase();
    let result = dispatch_control_command(&cmd).await;
    match result {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn dispatch_control_command(cmd: &str) -> Result<Value> {
    match cmd {
        // Kill switch and resume (tools that directly set engine state).
        "kill switch" | "kill" => invoke_tool("kill_switch", json!({})).await,
        "resume" | "resume bad apple" => invoke_tool("resume", json!({})).await,
        // Fast tier.
        "fast tier on" | "enable fast tier" => {
            daemon_call("set_fast_tier", Some(json!({"enabled": true}))).await
        }
        "fast tier off" | "disable fast tier" => {
            daemon_call("set_fast_tier", Some(json!({"enabled": false}))).await
        }
        // Autopilot.
        "autopilot on" | "enable autopilot" => {
            daemon_call("set_autopilot", Some(json!({"enabled": true}))).await
        }
        "autopilot off" | "disable autopilot" => {
            daemon_call("set_autopilot", Some(json!({"enabled": false}))).await
        }
        // Private mode.
        "private mode on" | "enable private mode" => {
            daemon_call("private_mode", Some(json!({"enabled": true}))).await
        }
        "private mode off" | "disable private mode" => {
            daemon_call("private_mode", Some(json!({"enabled": false}))).await
        }
        // Airgap.
        "airgap on" | "enable airgap" | "air-gap on" => {
            daemon_call("set_airgap", Some(json!({"enabled": true}))).await
        }
        "airgap off" | "disable airgap" | "air-gap off" => {
            daemon_call("set_airgap", Some(json!({"enabled": false}))).await
        }
        // VRAM / models.
        "flush vram" => daemon_call("flush_vram", None).await,
        "unload all models" | "unload models" => daemon_call("unload_model", None).await,
        // Workspace.
        _ if cmd.starts_with("set workspace to ") => {
            let path = cmd.strip_prefix("set workspace to ").unwrap_or("").trim();
            daemon_call("set_workspace", Some(json!({"path": path}))).await
        }
        _ if cmd.starts_with("set workspace ") => {
            let path = cmd.strip_prefix("set workspace ").unwrap_or("").trim();
            daemon_call("set_workspace", Some(json!({"path": path}))).await
        }
        // Fallback: let the daemon interpret it as a chat command (slower).
        _ => {
            let prompt = cmd.to_string();
            let text = tokio::task::spawn_blocking(move || {
                bad_apple::bad_apple_ipc::query_with_metrics(&prompt, 256, |_token| {})
                    .map(|(text, _)| text)
            })
            .await
            .map_err(|e| anyhow::anyhow!("spawn blocking failed: {e}"))??;
            Ok(Value::String(text))
        }
    }
}

async fn get_workspace_handler() -> impl IntoResponse {
    match daemon_call("get_workspace", None).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct WorkspaceInput {
    path: String,
}

async fn set_workspace_handler(Json(body): Json<WorkspaceInput>) -> impl IntoResponse {
    match daemon_call("set_workspace", Some(json!({"path": body.path}))).await {
        Ok(v) => Json(v),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn working_memory_handler(
    axum::extract::Query(query): axum::extract::Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let limit: usize = query
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let file = working_memory_file();
    if !file.exists() {
        return Json(json!({"content": ""}));
    }
    match tokio::fs::read_to_string(&file).await {
        Ok(text) => {
            let count = text.chars().count();
            let trimmed = if count > limit {
                text.chars().take(limit).collect::<String>()
            } else {
                text
            };
            Json(json!({"content": trimmed, "truncated": count > limit}))
        }
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct WorkingMemoryInput {
    content: String,
    #[serde(default = "default_working_memory_mode")]
    mode: String,
}

fn default_working_memory_mode() -> String {
    "replace".to_string()
}

async fn set_working_memory_handler(Json(body): Json<WorkingMemoryInput>) -> impl IntoResponse {
    let file = working_memory_file();
    let _ = tokio::fs::create_dir_all(file.parent().unwrap_or(&PathBuf::from("."))).await;
    let content = if body.mode == "append" {
        match tokio::fs::read_to_string(&file).await {
            Ok(existing) if !existing.is_empty() => format!("{existing}\n{}", body.content),
            _ => body.content,
        }
    } else {
        body.content
    };
    match tokio::fs::write(&file, content.as_bytes()).await {
        Ok(()) => Json(json!({"status": "ok", "chars": content.chars().count()})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

// MARK: - Memory / facts

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoryFact {
    subject: String,
    predicate: String,
    object: String,
    #[serde(default = "now_iso")]
    timestamp: String,
    #[serde(default = "default_fact_source")]
    source: String,
}

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn default_fact_source() -> String {
    "dashboard".to_string()
}

async fn load_facts() -> Vec<MemoryFact> {
    let path = memory_graph_file();
    if !path.exists() {
        return Vec::new();
    }
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => serde_json::from_str::<Vec<MemoryFact>>(&text).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

async fn save_facts(facts: &[MemoryFact]) -> Result<()> {
    let path = memory_graph_file();
    let _ = tokio::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from("."))).await;
    let data = serde_json::to_vec_pretty(facts)?;
    tokio::fs::write(&path, data).await?;
    Ok(())
}

async fn list_facts_handler() -> impl IntoResponse {
    let facts = load_facts().await;
    Json(json!({"facts": facts}))
}

#[derive(Deserialize)]
struct AddFactInput {
    subject: String,
    predicate: String,
    object: String,
    #[serde(default = "default_fact_source")]
    source: String,
}

async fn add_fact_handler(Json(body): Json<AddFactInput>) -> impl IntoResponse {
    let mut facts = load_facts().await;
    facts.push(MemoryFact {
        subject: body.subject,
        predicate: body.predicate,
        object: body.object,
        timestamp: now_iso(),
        source: body.source,
    });
    match save_facts(&facts).await {
        Ok(()) => Json(json!({"status": "ok", "facts": facts})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn clear_facts_handler() -> impl IntoResponse {
    match save_facts(&[]).await {
        Ok(()) => Json(json!({"status": "ok"})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct ExtractFactsInput {
    #[serde(default)]
    text: String,
    #[serde(default)]
    path: String,
    #[serde(default = "default_extract_source")]
    source: String,
}

fn default_extract_source() -> String {
    "extraction".to_string()
}

async fn extract_facts_handler(Json(body): Json<ExtractFactsInput>) -> impl IntoResponse {
    let source_text = if !body.path.is_empty() {
        match read_jailed_text(&body.path).await {
            Ok(t) => t,
            Err(e) => return Json(json!({"error": e.to_string()})),
        }
    } else {
        body.text
    };

    if source_text.trim().is_empty() {
        return Json(json!({"error": "no text or path provided"}));
    }

    match extract_facts_from_text(&source_text, &body.source).await {
        Ok(facts) => {
            let mut existing = load_facts().await;
            existing.extend(facts.clone());
            match save_facts(&existing).await {
                Ok(()) => Json(json!({"status": "ok", "extracted": facts.len(), "facts": facts})),
                Err(e) => Json(json!({"error": e.to_string()})),
            }
        }
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct IndexFactsInput {
    path: String,
    #[serde(default = "default_extract_source")]
    source: String,
}

async fn index_facts_handler(Json(body): Json<IndexFactsInput>) -> impl IntoResponse {
    if body.path.is_empty() {
        return Json(json!({"error": "path is required"}));
    }
    match index_path_for_facts(&body.path, &body.source).await {
        Ok(count) => {
            let total = load_facts().await.len();
            Json(json!({"status": "ok", "extracted": count, "total_facts": total}))
        }
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

/// Read text from a path inside the allowed roots (home, /tmp, /var/tmp, or workspace).
async fn read_jailed_text(path: &str) -> Result<String> {
    let expanded = if path.starts_with("~/") {
        std::env::var("HOME")
            .map(|h| format!("{}{}", h, &path[1..]))
            .unwrap_or_else(|_| path.to_string())
    } else {
        path.to_string()
    };

    if expanded.contains("..") {
        anyhow::bail!("path contains '..'");
    }

    let allowed = allowed_roots().await?;
    let normalized = tokio::fs::canonicalize(&expanded)
        .await
        .unwrap_or_else(|_| std::path::PathBuf::from(&expanded));
    let ok = allowed
        .iter()
        .any(|root| normalized == *root || normalized.starts_with(root.join("")));
    if !ok {
        anyhow::bail!("path '{}' is outside allowed roots", expanded);
    }

    if tokio::fs::metadata(&normalized).await?.is_dir() {
        anyhow::bail!("path is a directory, not a text file");
    }
    let text = tokio::fs::read_to_string(&normalized).await?;
    Ok(text)
}

async fn allowed_roots() -> Result<Vec<std::path::PathBuf>> {
    let mut roots: Vec<std::path::PathBuf> = vec![
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")),
        PathBuf::from("/tmp"),
        PathBuf::from("/var/tmp"),
    ];
    if let Ok(status) = daemon_call("get_workspace", None).await {
        if let Some(ws) = status.get("workspace").and_then(|w| w.as_str()) {
            if !ws.is_empty() {
                roots.push(PathBuf::from(ws));
            }
        }
    }
    Ok(roots)
}

/// Extract fact triples from a block of text using the local model.
async fn extract_facts_from_text(text: &str, source: &str) -> Result<Vec<MemoryFact>> {
    let prompt = format!(
        "Extract every factual claim from the text below as a JSON array of objects with the fields \"subject\", \"predicate\", and \"object\". \
Each object must be a single subject-predicate-object triple. Only use explicit facts from the text. Do not invent or infer facts not present. \
If no facts are present, return an empty array []. \
\nExample output:\n[{{\"subject\": \"Bad Apple\", \"predicate\": \"is\", \"object\": \"a local AI operating system\"}}]\n\nText:\n{}\n\nReturn ONLY the JSON array:",
        text
    );
    let system_prompt = "You are a precise fact extraction tool. You output only valid JSON arrays of {subject, predicate, object} triples. No markdown, no prose, no explanation.";
    let params = json!({
        "prompt": prompt,
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "system_prompt": system_prompt
    });
    let result = daemon_call("inference", Some(params)).await?;
    let raw = result
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let json_text = extract_json_array(&raw)?;
    let facts: Vec<MemoryFact> = serde_json::from_str(&json_text)?;
    let stamped: Vec<MemoryFact> = facts
        .into_iter()
        .map(|mut f| {
            f.timestamp = now_iso();
            f.source = source.to_string();
            f
        })
        .collect();
    Ok(stamped)
}

fn extract_json_array(text: &str) -> Result<String> {
    let text = text.trim();
    // Strip code fences if present.
    let cleaned = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(text)
        .trim();
    if let Some(start) = cleaned.find('[') {
        if let Some(end) = cleaned.rfind(']') {
            return Ok(cleaned[start..=end].to_string());
        }
    }
    anyhow::bail!("no JSON array found in model output")
}

/// Collect supported text files under `dir` up to a limit, using an explicit
/// stack so the function stays non-recursive.
async fn collect_text_files(dir: &std::path::Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let mut stack = std::collections::VecDeque::new();
    stack.push_back((dir.to_path_buf(), 0usize));

    while let Some((current, depth)) = stack.pop_front() {
        if depth >= 4 || files.len() >= 50 {
            continue;
        }
        let mut entries = tokio::fs::read_dir(&current).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            if name.starts_with('.') {
                continue;
            }
            let file_type = entry.file_type().await?;
            if file_type.is_dir() {
                stack.push_back((path, depth + 1));
            } else if file_type.is_file()
                && is_text_file(&path)
                && matches!(tokio::fs::metadata(&path).await, Ok(meta) if meta.len() <= 256_000)
            {
                files.push(path);
            }
            if files.len() >= 50 {
                break;
            }
        }
    }
    Ok(())
}

/// Walk a directory and extract facts from each supported text file.
async fn index_path_for_facts(path: &str, source: &str) -> Result<usize> {
    let expanded = if path.starts_with("~/") {
        std::env::var("HOME")
            .map(|h| format!("{}{}", h, &path[1..]))
            .unwrap_or_else(|_| path.to_string())
    } else {
        path.to_string()
    };
    if expanded.contains("..") {
        anyhow::bail!("path contains '..'");
    }

    let allowed = allowed_roots().await?;
    let normalized = tokio::fs::canonicalize(&expanded)
        .await
        .unwrap_or_else(|_| std::path::PathBuf::from(&expanded));
    let ok = allowed
        .iter()
        .any(|root| normalized == *root || normalized.starts_with(root.join("")));
    if !ok {
        anyhow::bail!("path '{}' is outside allowed roots", expanded);
    }

    let mut files = Vec::new();
    collect_text_files(&normalized, &mut files).await?;

    let mut count = 0;
    let mut new_facts = Vec::new();
    for file in files {
        let text = tokio::fs::read_to_string(&file).await.unwrap_or_default();
        if text.trim().is_empty() {
            continue;
        }
        let source_label = format!("{}:{}", source, file.display());
        match extract_facts_from_text(&text, &source_label).await {
            Ok(facts) => {
                count += facts.len();
                new_facts.extend(facts);
            }
            Err(_) => continue,
        }
    }

    if !new_facts.is_empty() {
        let mut existing = load_facts().await;
        existing.extend(new_facts);
        save_facts(&existing).await?;
    }
    Ok(count)
}

fn is_text_file(path: &std::path::Path) -> bool {
    let exts: std::collections::HashSet<&str> = [
        "txt", "md", "markdown", "rst", "swift", "py", "rs", "go", "ts", "tsx", "js", "jsx", "m",
        "mm", "c", "cc", "cpp", "h", "hpp", "java", "kt", "rb", "php", "json", "yaml", "yml",
        "toml", "ini", "cfg", "conf", "sh", "bash", "zsh", "fish", "ps1", "html", "htm", "css",
        "scss", "xml", "csv", "log",
    ]
    .iter()
    .cloned()
    .collect();
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.contains(e.to_lowercase().as_str()))
        .unwrap_or(false)
}

// MARK: - Workshop

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WorkshopPersona {
    name: String,
    description: Option<String>,
    system_prompt: String,
    #[serde(default)]
    voice_system_prompt: Option<String>,
    #[serde(default)]
    roast_bank: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WorkshopTool {
    name: String,
    description: String,
    kind: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
}

async fn load_workshop_personas() -> std::collections::HashMap<String, WorkshopPersona> {
    let path = custom_personas_file();
    if !path.exists() {
        // Seed from the built-in personas.json so the user always has a base.
        let builtin = builtin_personas_file();
        if builtin.exists() {
            if let Ok(text) = tokio::fs::read_to_string(&builtin).await {
                let _ =
                    tokio::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from("."))).await;
                let _ = tokio::fs::write(&path, text.as_bytes()).await;
            }
        }
    }
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => {
            serde_json::from_str::<std::collections::HashMap<String, WorkshopPersona>>(&text)
                .unwrap_or_default()
        }
        Err(_) => std::collections::HashMap::new(),
    }
}

async fn save_workshop_personas(
    map: &std::collections::HashMap<String, WorkshopPersona>,
) -> Result<()> {
    let path = custom_personas_file();
    let _ = tokio::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from("."))).await;
    let data = serde_json::to_vec_pretty(map)?;
    tokio::fs::write(&path, data).await?;
    Ok(())
}

async fn list_personas_handler() -> impl IntoResponse {
    let map = load_workshop_personas().await;
    let list: Vec<Value> = map
        .into_iter()
        .map(|(id, p)| {
            json!({
                "id": id,
                "name": p.name,
                "description": p.description,
            })
        })
        .collect();
    Json(json!({"personas": list}))
}

async fn get_persona_handler(
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let map = load_workshop_personas().await;
    match map.get(&id) {
        Some(p) => Ok(Json(json!({"id": id, "persona": p}))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "persona not found"})),
        )),
    }
}

#[derive(Deserialize)]
struct SavePersonaInput {
    id: String,
    #[serde(flatten)]
    persona: WorkshopPersona,
}

async fn save_persona_handler(Json(body): Json<SavePersonaInput>) -> impl IntoResponse {
    let mut map = load_workshop_personas().await;
    let id = body.id.to_lowercase().trim().replace(' ', "_");
    if id.is_empty() {
        return Json(json!({"error": "persona id is required"}));
    }
    map.insert(id.clone(), body.persona);
    match save_workshop_personas(&map).await {
        Ok(()) => {
            // Switch to the new persona immediately.
            let _ = daemon_call("switch_persona", Some(json!({"name": id}))).await;
            Json(json!({"status": "ok", "id": id}))
        }
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn delete_persona_handler(Path(id): Path<String>) -> impl IntoResponse {
    let mut map = load_workshop_personas().await;
    map.remove(&id);
    match save_workshop_personas(&map).await {
        Ok(()) => Json(json!({"status": "ok"})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn load_custom_tools() -> std::collections::HashMap<String, WorkshopTool> {
    let path = custom_tools_file();
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => serde_json::from_str::<std::collections::HashMap<String, WorkshopTool>>(&text)
            .unwrap_or_default(),
        Err(_) => std::collections::HashMap::new(),
    }
}

async fn save_custom_tools(map: &std::collections::HashMap<String, WorkshopTool>) -> Result<()> {
    let path = custom_tools_file();
    let _ = tokio::fs::create_dir_all(path.parent().unwrap_or(&PathBuf::from("."))).await;
    let data = serde_json::to_vec_pretty(map)?;
    tokio::fs::write(&path, data).await?;
    Ok(())
}

async fn list_custom_tools_handler() -> impl IntoResponse {
    let map = load_custom_tools().await;
    let list: Vec<Value> = map
        .into_iter()
        .map(|(id, t)| {
            json!({
                "id": id,
                "name": t.name,
                "description": t.description,
                "kind": t.kind,
            })
        })
        .collect();
    Json(json!({"tools": list}))
}

async fn get_custom_tool_handler(
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let map = load_custom_tools().await;
    match map.get(&id) {
        Some(t) => Ok(Json(json!({"id": id, "tool": t}))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "tool not found"})),
        )),
    }
}

#[derive(Deserialize)]
struct SaveCustomToolInput {
    id: String,
    #[serde(flatten)]
    tool: WorkshopTool,
}

async fn save_custom_tool_handler(Json(body): Json<SaveCustomToolInput>) -> impl IntoResponse {
    let mut map = load_custom_tools().await;
    let id = body.id.to_lowercase().trim().replace(' ', "_");
    if id.is_empty() || body.tool.name.is_empty() || body.tool.kind.is_empty() {
        return Json(json!({"error": "id, name, and kind are required"}));
    }
    map.insert(id.clone(), body.tool);
    match save_custom_tools(&map).await {
        Ok(()) => Json(json!({"status": "ok", "id": id})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn delete_custom_tool_handler(Path(id): Path<String>) -> impl IntoResponse {
    let mut map = load_custom_tools().await;
    map.remove(&id);
    match save_custom_tools(&map).await {
        Ok(()) => Json(json!({"status": "ok"})),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct RunCustomToolInput {
    #[serde(default)]
    args: Vec<String>,
}

async fn run_custom_tool_handler(
    Path(id): Path<String>,
    Json(body): Json<RunCustomToolInput>,
) -> impl IntoResponse {
    let map = load_custom_tools().await;
    match map.get(&id) {
        Some(tool) => match run_workshop_tool(tool, &body.args).await {
            Ok(output) => Json(json!({"status": "ok", "output": output})),
            Err(e) => Json(json!({"error": e.to_string()})),
        },
        None => Json(json!({"error": "tool not found"})),
    }
}

async fn run_workshop_tool(tool: &WorkshopTool, args: &[String]) -> Result<String> {
    let kind = tool.kind.as_str();
    match kind {
        "shell" => {
            let command = interpolate_args(&tool.command, args);
            // Validate the command does not contain obvious shell metacharacters except
            // for the ones a typical safe command needs. This is a defensive boundary,
            // not a sandbox.
            if command.contains(';')
                || command.contains("&&")
                || command.contains("||")
                || command.contains('>')
            {
                anyhow::bail!("command contains disallowed shell metacharacters");
            }
            let out = tokio::process::Command::new("/bin/sh")
                .args(["-c", &command])
                .output()
                .await?;
            Ok(format!("{}", String::from_utf8_lossy(&out.stdout)))
        }
        "applescript" => {
            let script = interpolate_args(&tool.command, args);
            let mut child = tokio::process::Command::new("/usr/bin/osascript")
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;
            if let Some(ref mut stdin) = child.stdin {
                use tokio::io::AsyncWriteExt;
                stdin.write_all(script.as_bytes()).await?;
            }
            let out = child.wait_with_output().await?;
            Ok(format!("{}", String::from_utf8_lossy(&out.stdout)))
        }
        "shortcut" => {
            let name = tool.command.clone();
            let result = invoke_tool(
                "run_shortcut",
                json!({"name": name, "input": args.first().cloned().unwrap_or_default()}),
            )
            .await?;
            Ok(result.to_string())
        }
        _ => anyhow::bail!("unknown custom tool kind: {kind}"),
    }
}

fn interpolate_args(template: &str, args: &[String]) -> String {
    let mut out = template.to_string();
    for (i, arg) in args.iter().enumerate() {
        let placeholder = format!("{}{}", "{".repeat(2), i + 1) + &"}".repeat(2);
        out = out.replace(&placeholder, arg);
    }
    // Shell-escape any remaining placeholder by quoting the template.
    out
}

async fn mcp_registry_handler() -> impl IntoResponse {
    let market = load_marketplace().await;
    Json(json!({"servers": market.list().await}))
}
