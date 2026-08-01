use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::process::Command;
use std::path::PathBuf;
use std::fs;
use std::net::{TcpStream, SocketAddr};
use serde::{Serialize, Deserialize};
use axum::{Router, routing::{get, post}, Json, extract::State, http::StatusCode, response::IntoResponse, response::Html};
use sysinfo::{System, Components, Disks, Networks};
use chrono::Timelike;
use tokio::net::TcpListener;
use crate::metrics::MetricsLogger;

pub fn current_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct TelemetryState {
    pub start_time: u64,
    pub cycle_count: u64,
    pub last_forward_latency_ms: f64,
    pub last_backward_latency_ms: f64,
    pub last_total_latency_ms: f64,
    pub last_retrieval_latency_ms: f64,
    pub tokens_per_second: f64,
    pub last_loss: f64,
    pub avg_loss: f64,
    pub loss_history: Vec<f64>,
    pub memory_node_count: usize,
    pub active_goal_count: usize,
    pub last_llm_latency_ms: f64,
    pub llm_tokens_per_second: f64,
    pub tool_executions: u64,
    pub tool_successes: u64,
    pub tool_success_rate: f64,
    pub benchmark_score: f64,
    pub benchmark_attempts: u64,
    pub critic_score: f64,
}

impl TelemetryState {
    pub fn new() -> Self {
        Self {
            start_time: current_secs(),
            ..Default::default()
        }
    }

    pub fn record_forward(&mut self, ms: f64) {
        self.last_forward_latency_ms = ms;
    }

    pub fn record_backward(&mut self, ms: f64) {
        self.last_backward_latency_ms = ms;
    }

    pub fn record_retrieval(&mut self, ms: f64) {
        self.last_retrieval_latency_ms = ms;
    }

    pub fn record_total(&mut self, ms: f64) {
        self.last_total_latency_ms = ms;
    }

    pub fn record_tokens_per_second(&mut self, tokens: usize, elapsed_ms: f64) {
        if elapsed_ms > 0.0 {
            self.tokens_per_second = (tokens as f64) / (elapsed_ms / 1000.0);
        }
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.last_loss = loss;
        self.loss_history.push(loss);
        if self.loss_history.len() > 100 {
            self.loss_history.remove(0);
        }
        self.avg_loss = self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64;
    }

    pub fn record_tool_result(&mut self, success: bool) {
        self.tool_executions += 1;
        if success {
            self.tool_successes += 1;
        }
        self.tool_success_rate = self.tool_successes as f64 / self.tool_executions as f64;
    }

    pub fn record_benchmark(&mut self, score: f64, attempts: u64) {
        self.benchmark_score = score;
        self.benchmark_attempts = attempts;
    }

    pub fn record_critic(&mut self, score: f64) {
        self.critic_score = score;
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct SensorSnapshot {
    pub start_time: u64,
    pub cpu_usage_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub memory_pressure_percent: f64,
    pub cpu_temperature_celsius: f64,
    pub battery_percent: f64,
    pub battery_charging: bool,
    pub photons: f64,
    pub audio: f64,
    pub mass: f64,
    pub gravity: f64,
    pub uptime_seconds: u64,
    // 🌐 Richer embodiment
    pub cpu_core_count: usize,
    pub process_count: usize,
    pub disk_total_bytes: u64,
    pub disk_used_bytes: u64,
    pub network_localhost_reachable: bool,
    pub network_ollama_reachable: bool,
    pub network_rx_bytes_per_sec: f64,
    pub network_tx_bytes_per_sec: f64,
    pub top_process_name: String,
    pub top_process_cpu_percent: f64,
    pub disk_usage_percent: f64,
    pub curriculum_files: usize,
    pub curriculum_total_bytes: u64,
    #[serde(skip)]
    pub prev_network_rx: u64,
    #[serde(skip)]
    pub prev_network_tx: u64,
    #[serde(skip)]
    pub prev_network_time: u64,
    #[serde(skip)]
    pub refresh_counter: u8,
}

impl SensorSnapshot {
    pub fn new() -> Self {
        Self {
            start_time: current_secs(),
            gravity: 9.81,
            ..Default::default()
        }
    }
}

fn count_curriculum(dirs: &[PathBuf]) -> (usize, u64) {
    let mut files = 0usize;
    let mut bytes = 0u64;
    for dir in dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Ok(meta) = fs::metadata(&path) {
                        files += 1;
                        bytes += meta.len();
                    }
                }
            }
        }
    }
    (files, bytes)
}

fn is_host_reachable(addr: &str, port: u16) -> bool {
    let socket_addr = if let Ok(ip) = addr.parse::<std::net::IpAddr>() {
        std::net::SocketAddr::from((ip, port))
    } else {
        std::net::SocketAddr::from(([127,0,0,1], port))
    };
    TcpStream::connect_timeout(&socket_addr, Duration::from_millis(500)).is_ok()
}

pub fn update_sensor_snapshot(
    system: &mut System,
    battery_manager: &Option<starship_battery::Manager>,
    snapshot: &mut SensorSnapshot,
    curriculum_dirs: &[PathBuf],
) {
    snapshot.refresh_counter = snapshot.refresh_counter.wrapping_add(1);
    let slow_refresh = snapshot.refresh_counter % 5 == 0;

    system.refresh_cpu();
    system.refresh_memory();
    if slow_refresh {
        system.refresh_processes();
    }

    snapshot.cpu_usage_percent = system.global_cpu_info().cpu_usage() as f64;
    snapshot.memory_used_bytes = system.used_memory();
    snapshot.memory_total_bytes = system.total_memory();
    snapshot.memory_pressure_percent = if snapshot.memory_total_bytes > 0 {
        (snapshot.memory_used_bytes as f64 / snapshot.memory_total_bytes as f64) * 100.0
    } else {
        0.0
    };

    snapshot.cpu_core_count = system.cpus().len();
    snapshot.process_count = if slow_refresh { system.processes().len() } else { snapshot.process_count };

    if slow_refresh {
        // Disk usage
        let disks = Disks::new_with_refreshed_list();
        let (mut total, mut available) = (0u64, 0u64);
        for disk in disks.list() {
            total += disk.total_space();
            available += disk.available_space();
        }
        snapshot.disk_total_bytes = total;
        snapshot.disk_used_bytes = total.saturating_sub(available);
        snapshot.disk_usage_percent = if total > 0 {
            (snapshot.disk_used_bytes as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        // Network I/O rates
        let networks = Networks::new_with_refreshed_list();
        let (rx_total, tx_total) = networks.iter().fold((0u64, 0u64), |(rx, tx), (_, net)| {
            (rx + net.total_received(), tx + net.total_transmitted())
        });
        let now = current_secs();
        let dt = now.saturating_sub(snapshot.prev_network_time).max(1);
        if snapshot.prev_network_time > 0 {
            let rx_delta = rx_total.saturating_sub(snapshot.prev_network_rx);
            let tx_delta = tx_total.saturating_sub(snapshot.prev_network_tx);
            snapshot.network_rx_bytes_per_sec = rx_delta as f64 / dt as f64;
            snapshot.network_tx_bytes_per_sec = tx_delta as f64 / dt as f64;
        }
        snapshot.prev_network_rx = rx_total;
        snapshot.prev_network_tx = tx_total;
        snapshot.prev_network_time = now;

        // Top CPU process
        let mut top_cpu = 0.0f32;
        let mut top_name = String::new();
        for (_pid, process) in system.processes() {
            let cpu = process.cpu_usage();
            if cpu > top_cpu {
                top_cpu = cpu;
                top_name = process.name().to_string();
            }
        }
        snapshot.top_process_name = if top_name.is_empty() { "none".into() } else { top_name };
        snapshot.top_process_cpu_percent = top_cpu as f64;

        // CPU temperature
        let components = Components::new_with_refreshed_list();
        let mut temps = Vec::new();
        for component in components.iter() {
            let label = component.label().to_lowercase();
            if label.contains("cpu") || label.contains("soc") || label.contains("package") {
                temps.push(component.temperature() as f64);
            }
        }
        snapshot.cpu_temperature_celsius = if !temps.is_empty() {
            temps.iter().sum::<f64>() / temps.len() as f64
        } else {
            0.0
        };
    }

    // Network reachability (cheap, keep every cycle)
    snapshot.network_localhost_reachable = is_host_reachable("127.0.0.1", 8080);
    snapshot.network_ollama_reachable = is_host_reachable("127.0.0.1", 11434);

    // Curriculum footprint
    let (c_files, c_bytes) = count_curriculum(curriculum_dirs);
    snapshot.curriculum_files = c_files;
    snapshot.curriculum_total_bytes = c_bytes;

    // Battery state
    if let Some(ref manager) = battery_manager {
        if let Ok(mut batteries) = manager.batteries() {
            if let Some(Ok(bat)) = batteries.next() {
                snapshot.battery_percent = (bat.state_of_charge().get::<starship_battery::units::ratio::percent>()) as f64;
                snapshot.battery_charging = matches!(
                    bat.state(),
                    starship_battery::State::Charging | starship_battery::State::Full
                );
            }
        }
    }

    // Environmental proxies grounded in real system state
    let hour = chrono::Local::now().hour() as f64;
    snapshot.photons = if hour >= 6.0 && hour <= 18.0 {
        // Noon peak = 1.0, dawn/dusk = 0.0
        ((hour - 6.0) / 12.0 * std::f64::consts::PI).sin().max(0.0)
    } else {
        0.0
    };
    snapshot.audio = (snapshot.cpu_usage_percent / 100.0).clamp(0.0, 1.0);
    snapshot.mass = (snapshot.memory_pressure_percent / 100.0).clamp(0.0, 1.0);
    snapshot.uptime_seconds = current_secs().saturating_sub(snapshot.start_time);
}

#[derive(Clone)]
struct AppState {
    telemetry: Arc<Mutex<TelemetryState>>,
    sensors: Arc<Mutex<SensorSnapshot>>,
    metrics: Arc<Mutex<MetricsLogger>>,
}

async fn telemetry_handler(State(state): State<AppState>) -> impl IntoResponse {
    let telemetry = state.telemetry.lock().unwrap().clone();
    let sensors = state.sensors.lock().unwrap().clone();
    Json(serde_json::json!({
        "telemetry": telemetry,
        "sensors": sensors,
    }))
}

async fn metrics_json_handler(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.lock().unwrap();
    Json(serde_json::json!({
        "summary": metrics.summary(100),
        "recent": metrics.recent(100),
    }))
}

async fn metrics_dashboard_handler(State(state): State<AppState>) -> impl IntoResponse {
    let html = state.metrics.lock().unwrap().dashboard_html();
    Html(html)
}

#[derive(Deserialize)]
struct ToolRunRequest {
    name: String,
    code: String,
    language: String,
}

#[derive(Serialize)]
struct ToolRunResponse {
    status: String,
    output: Option<String>,
    error: Option<String>,
}

pub fn run_sandboxed_tool(name: &str, code: &str, language: &str) -> Result<String, String> {
    if !matches!(language, "python" | "sh") {
        return Err("Unsupported language. Use 'python' or 'sh'.".to_string());
    }

    let tools_dir = PathBuf::from("tools");
    fs::create_dir_all(&tools_dir).map_err(|e| format!("Cannot create tools dir: {}", e))?;

    let ext = if language == "python" { "py" } else { "sh" };
    let filename = tools_dir.join(format!("{}_{}.{}", name, current_secs(), ext));

    // For Python, inject a safe standard-library header so generated tools can use common modules.
    let code = if language == "python" {
        format!(
            "import json, math, random, statistics, datetime, itertools, collections, string, re\n{}\n",
            code
        )
    } else {
        code.to_string()
    };

    fs::write(&filename, code).map_err(|e| format!("Cannot write tool file: {}", e))?;

    let output = if language == "python" {
        Command::new("python3").arg(&filename).output()
    } else {
        Command::new("sh").arg(&filename).output()
    }
    .map_err(|e| format!("Failed to execute tool: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{}{}", stdout, stderr).trim().to_string();

    if !output.status.success() {
        return Err(combined);
    }
    Ok(combined)
}

async fn run_tool_handler(
    State(state): State<AppState>,
    Json(payload): Json<ToolRunRequest>,
) -> impl IntoResponse {
    let result = run_sandboxed_tool(&payload.name, &payload.code, &payload.language);
    let success = result.is_ok();
    state.telemetry.lock().unwrap().record_tool_result(success);

    match result {
        Ok(output) => (
            StatusCode::OK,
            Json(ToolRunResponse {
                status: "ok".to_string(),
                output: Some(output),
                error: None,
            }),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ToolRunResponse {
                status: "error".to_string(),
                output: None,
                error: Some(e),
            }),
        ),
    }
}

pub async fn start_telemetry_listener(port: u16) -> Result<TcpListener, Box<dyn std::error::Error>> {
    let fallback_base = port.saturating_add(1);
    for p in port..=port.saturating_add(15) {
        let addr = SocketAddr::from(([127, 0, 0, 1], p));
        match TcpListener::bind(addr).await {
            Ok(listener) => {
                if p == port {
                    println!("🛰️ [TELEMETRY ENGINE]: Online at http://{}/telemetry", addr);
                } else if p == fallback_base {
                    println!("⚠️ [PORT COLLISION]: Port {} active. Diverting stream to http://{}/telemetry", port, addr);
                } else {
                    println!("⚠️ [PORT COLLISION]: Diverting stream to http://{}/telemetry", addr);
                }
                return Ok(listener);
            }
            Err(_) => continue,
        }
    }
    Err(format!("All telemetry ports in range {}..{} are in use", port, port + 15).into())
}

pub async fn run_telemetry_server(
    telemetry: Arc<Mutex<TelemetryState>>,
    sensors: Arc<Mutex<SensorSnapshot>>,
    metrics: Arc<Mutex<MetricsLogger>>,
    port: u16,
) {
    let state = AppState { telemetry, sensors, metrics };
    let app = Router::new()
        .route("/telemetry", get(telemetry_handler))
        .route("/metrics", get(metrics_json_handler))
        .route("/dashboard", get(metrics_dashboard_handler))
        .route("/tools/run", post(run_tool_handler))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let listener = match start_telemetry_listener(port).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("⚠️ Telemetry server bind failed: {}. Continuing without HTTP telemetry.", e);
            return;
        }
    };

    let chosen_port = listener.local_addr().map(|a| a.port()).unwrap_or(port);
    println!("   POST http://127.0.0.1:{}/tools/run  {{\"name\",\"code\",\"language\"}}", chosen_port);
    axum::serve(listener, app).await.unwrap();
}
