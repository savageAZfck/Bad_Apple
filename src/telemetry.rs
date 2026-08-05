use crate::metrics::MetricsLogger;
use crate::ollama_client::OllamaClient;
use crate::protocol::SwarmMetrics;
use crate::strategy_library::StrategyLibrary;
use crate::{FullySapientSoulMatrix, Skill};
use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Timelike;
use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use sysinfo::{Components, Disks, Networks, System};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

pub fn current_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
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
    pub state_save_duration_ms: u64,
    pub last_llm_latency_ms: f64,
    pub llm_tokens_per_second: f64,
    pub tool_executions: u64,
    pub tool_successes: u64,
    pub tool_success_rate: f64,
    pub benchmark_score: f64,
    pub benchmark_attempts: u64,
    pub transfer_score: f64,
    pub transfer_attempts: u64,
    pub critic_score: f64,
    pub apple_intelligence_latency_us: u64,
    pub apple_intelligence_calls: u64,
    pub apple_intelligence_fails: u64,
    pub apple_intelligence_available: bool,
    pub memory_used_bytes: u64,
    pub memory_drift_bytes_per_sec: f64,
    pub memory_leak_score: f64,
    pub memory_sample_count: u64,
    pub curiosity_reward: f64,
    pub curiosity_cycles: u64,
    pub autonomously_discovered_strategies: u64,
    pub entropy_index: f64,
    pub resource_stress: f64,
    pub system2_active: bool,
    pub system2_cycles: u64,
    pub domain_mastery: std::collections::HashMap<String, f64>,
    pub synthesis_speedup: f64,
    pub power_reduction: f64,
    pub latency_stats: crate::metrics::LatencyStats,
    pub swarm_metrics: SwarmMetrics,
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

    pub fn record_state_save(&mut self, ms: u64) {
        self.state_save_duration_ms = ms;
    }

    pub fn record_benchmark(&mut self, score: f64, attempts: u64) {
        self.benchmark_score = score;
        self.benchmark_attempts = attempts;
    }

    pub fn record_transfer(&mut self, score: f64, attempts: u64) {
        self.transfer_score = score;
        self.transfer_attempts = attempts;
    }

    pub fn record_critic(&mut self, score: f64) {
        self.critic_score = score;
    }

    pub fn record_apple_intelligence(
        &mut self,
        latency_us: u64,
        calls: u64,
        fails: u64,
        available: bool,
    ) {
        self.apple_intelligence_latency_us = latency_us;
        self.apple_intelligence_calls = calls;
        self.apple_intelligence_fails = fails;
        self.apple_intelligence_available = available;
    }

    pub fn record_memory_drift(
        &mut self,
        used_bytes: u64,
        drift: f64,
        leak_score: f64,
        samples: u64,
    ) {
        self.memory_used_bytes = used_bytes;
        self.memory_drift_bytes_per_sec = drift;
        self.memory_leak_score = leak_score;
        self.memory_sample_count = samples;
    }

    pub fn record_curiosity(&mut self, reward: f64) {
        self.curiosity_reward = reward.clamp(0.0, 1.0);
    }

    pub fn increment_curiosity_cycle(&mut self) {
        self.curiosity_cycles += 1;
    }

    pub fn increment_discovered_strategy(&mut self) {
        self.autonomously_discovered_strategies += 1;
    }

    pub fn record_governor(
        &mut self,
        entropy: f64,
        stress: f64,
        system2: bool,
        system2_cycles: u64,
    ) {
        self.entropy_index = entropy.clamp(0.0, 1.0);
        self.resource_stress = stress.clamp(0.0, 1.0);
        self.system2_active = system2;
        self.system2_cycles = system2_cycles;
    }

    /// Record nanosecond ring-buffer latency statistics into the live
    /// telemetry dashboard.
    pub fn record_latency(&mut self, stats: crate::metrics::LatencyStats) {
        self.latency_stats = stats;
    }

    pub fn record_swarm(&mut self, swarm: &SwarmMetrics) {
        self.swarm_metrics = swarm.clone();
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
        std::net::SocketAddr::from(([127, 0, 0, 1], port))
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
    let slow_refresh = snapshot.refresh_counter.is_multiple_of(5);

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
    snapshot.process_count = if slow_refresh {
        system.processes().len()
    } else {
        snapshot.process_count
    };

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
        for process in system.processes().values() {
            let cpu = process.cpu_usage();
            if cpu > top_cpu {
                top_cpu = cpu;
                top_name = process.name().to_string();
            }
        }
        snapshot.top_process_name = if top_name.is_empty() {
            "none".into()
        } else {
            top_name
        };
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
                snapshot.battery_percent = (bat
                    .state_of_charge()
                    .get::<starship_battery::units::ratio::percent>())
                    as f64;
                snapshot.battery_charging = matches!(
                    bat.state(),
                    starship_battery::State::Charging | starship_battery::State::Full
                );
            }
        }
    }

    // Environmental proxies grounded in real system state
    let hour = chrono::Local::now().hour() as f64;
    snapshot.photons = if (6.0..=18.0).contains(&hour) {
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
    core_mind: Arc<Mutex<FullySapientSoulMatrix>>,
    ollama: Arc<OllamaClient>,
    strategy_library: Arc<StrategyLibrary>,
}

async fn telemetry_handler(State(state): State<AppState>) -> impl IntoResponse {
    let telemetry = state.telemetry.lock().await.clone();
    let sensors = state.sensors.lock().await.clone();
    Json(serde_json::json!({
        "telemetry": telemetry,
        "sensors": sensors,
    }))
}

async fn metrics_json_handler(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.lock().await;
    Json(serde_json::json!({
        "summary": metrics.summary(100),
        "recent": metrics.recent(100),
    }))
}

async fn metrics_dashboard_handler(State(state): State<AppState>) -> impl IntoResponse {
    let html = state.metrics.lock().await.dashboard_html();
    Html(html)
}

async fn live_dashboard_handler(State(state): State<AppState>) -> impl IntoResponse {
    let telemetry = state.telemetry.lock().await.clone();
    let sensors = state.sensors.lock().await.clone();
    Html(live_dashboard_html(&telemetry, &sensors))
}

/// Render the live Domain Competence Matrix and Thermodynamic Acceleration Panel
/// as a single HTML page with embedded SVGs. Updates are driven by the 6-second
/// clock loop in `main.rs`.
fn live_dashboard_html(telemetry: &TelemetryState, sensors: &SensorSnapshot) -> String {
    let competence_bars: Vec<String> = telemetry
        .domain_mastery
        .iter()
        .enumerate()
        .map(|(i, (domain, &mastery))| {
            let pct = (mastery * 100.0).clamp(0.0, 100.0);
            let y = 40 + i * 28;
            format!(
                r##"<g transform="translate(20, {})">
                   <text x="0" y="15" fill="#d0d0e0" font-size="12">{}</text>
                   <rect x="180" y="5" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
                   <rect x="180" y="5" width="{:.2}" height="12" fill="#7df" rx="2">
                       <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
                   </rect>
                   <text x="490" y="15" fill="#8af" font-size="12">{:.1}%</text>
                 </g>"##,
                y,
                html_escape(domain),
                pct * 3.0,
                pct * 3.0,
                pct
            )
        })
        .collect();

    let competence_svg = if competence_bars.is_empty() {
        r##"<svg viewBox="0 0 600 80"><text x="20" y="40" fill="#d0d0e0" font-size="14">No domain mastery data yet</text></svg>"##.to_string()
    } else {
        let h = 80 + telemetry.domain_mastery.len() * 28;
        format!(
            r##"<svg viewBox="0 0 600 {}" class="panel-svg">{}</svg>"##,
            h,
            competence_bars.join("\n")
        )
    };

    let speedup = (telemetry.synthesis_speedup * 100.0).clamp(0.0, 300.0);
    let power_pct = (telemetry.power_reduction * 100.0).clamp(0.0, 100.0);
    let cpu = sensors.cpu_usage_percent.clamp(0.0, 100.0);
    let mem = sensors.memory_pressure_percent.clamp(0.0, 100.0);

    let thermodynamic_svg = format!(
        r##"<svg viewBox="0 0 600 200" class="panel-svg">
           <text x="20" y="30" fill="#7df" font-size="16">Synthesis Speedup: {:.2}x</text>
           <rect x="20" y="45" width="500" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="20" y="45" width="{:.2}" height="12" fill="#7f7" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="95" fill="#7df" font-size="16">Power Reduction: {:.1}%</text>
           <rect x="20" y="110" width="500" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="20" y="110" width="{:.2}" height="12" fill="#f7d" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="160" fill="#8af" font-size="14">CPU: {:.1}%  |  Memory: {:.1}%</text>
         </svg>"##,
        telemetry.synthesis_speedup,
        speedup * 1.67,
        speedup * 1.67,
        power_pct,
        power_pct * 5.0,
        power_pct * 5.0,
        cpu,
        mem
    );

    let swarm = &telemetry.swarm_metrics;
    let peer_pct = (swarm.peer_count as f64 / 64.0 * 100.0).clamp(0.0, 100.0);
    let in_bar = swarm.kbps_in.clamp(0.0, 1000.0);
    let out_bar = swarm.kbps_out.clamp(0.0, 1000.0);
    let merge_us = swarm.last_merge_latency_us.clamp(0, 10_000) as f64;
    let receive_us = swarm.last_engram_receive_latency_us.clamp(0, 10_000) as f64;
    let swarm_svg = format!(
        r##"<svg viewBox="0 0 600 220" class="panel-svg">
           <text x="20" y="25" fill="#7df" font-size="16">Wide-Area Network Swarm Grid</text>

           <text x="20" y="55" fill="#d0d0e0" font-size="13">Active Peer Nodes: {}</text>
           <rect x="180" y="43" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="43" width="{:.2}" height="12" fill="#7df" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="90" fill="#d0d0e0" font-size="13">Inbound Throughput: {:.2} KB/s</text>
           <rect x="180" y="78" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="78" width="{:.2}" height="12" fill="#7f7" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="125" fill="#d0d0e0" font-size="13">Outbound Throughput: {:.2} KB/s</text>
           <rect x="180" y="113" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="113" width="{:.2}" height="12" fill="#f7d" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="160" fill="#d0d0e0" font-size="13">Merge Latency: {} us (batch {})</text>
           <rect x="180" y="148" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="148" width="{:.2}" height="12" fill="#8af" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="195" fill="#d0d0e0" font-size="13">Receive Latency: {} us</text>
           <rect x="180" y="183" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="183" width="{:.2}" height="12" fill="#aaf" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>
         </svg>"##,
        swarm.peer_count,
        peer_pct * 3.0,
        peer_pct * 3.0,
        swarm.kbps_in,
        in_bar * 0.3,
        in_bar * 0.3,
        swarm.kbps_out,
        out_bar * 0.3,
        out_bar * 0.3,
        swarm.last_merge_latency_us,
        swarm.last_merge_batch_size,
        merge_us * 0.03,
        merge_us * 0.03,
        swarm.last_engram_receive_latency_us,
        receive_us * 0.03,
        receive_us * 0.03
    );

    let ai_us = telemetry.apple_intelligence_latency_us.clamp(0, 10_000_000) as f64;
    let ai_ms = ai_us / 1000.0;
    let ai_bar = (ai_us / 100.0).clamp(0.0, 500.0);
    let ai_status = if telemetry.apple_intelligence_available {
        "ONLINE"
    } else {
        "OFFLINE"
    };
    let ai_svg = format!(
        r##"<svg viewBox="0 0 600 180" class="panel-svg">
           <text x="20" y="25" fill="#7df" font-size="16">Apple Intelligence Bridge</text>

           <text x="20" y="60" fill="#d0d0e0" font-size="13">Status: {}</text>

           <text x="20" y="100" fill="#d0d0e0" font-size="13">Last Latency: {:.2} ms</text>
           <rect x="180" y="88" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="88" width="{:.2}" height="12" fill="#7f7" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="140" fill="#d0d0e0" font-size="13">Calls: {} | Fails: {}</text>
         </svg>"##,
        ai_status,
        ai_ms,
        ai_bar,
        ai_bar,
        telemetry.apple_intelligence_calls,
        telemetry.apple_intelligence_fails
    );

    let used_mb = (telemetry.memory_used_bytes as f64) / (1024.0 * 1024.0);
    let drift = telemetry.memory_drift_bytes_per_sec;
    let drift_kb_s = drift / 1024.0;
    let leak_bar = (telemetry.memory_leak_score * 300.0).clamp(0.0, 300.0);
    let memory_svg = format!(
        r##"<svg viewBox="0 0 600 180" class="panel-svg">
           <text x="20" y="25" fill="#7df" font-size="16">Memory Leak Profiler</text>

           <text x="20" y="60" fill="#d0d0e0" font-size="13">Heap: {:.1} MB</text>

           <text x="20" y="100" fill="#d0d0e0" font-size="13">Drift: {:.2} KB/s (leak score {:.2})</text>
           <rect x="180" y="88" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="88" width="{:.2}" height="12" fill="#f7d" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="140" fill="#d0d0e0" font-size="13">Samples: {}</text>
         </svg>"##,
        used_mb,
        drift_kb_s,
        telemetry.memory_leak_score,
        leak_bar,
        leak_bar,
        telemetry.memory_sample_count
    );

    let reward_bar = (telemetry.curiosity_reward * 300.0).clamp(0.0, 300.0);
    let curiosity_svg = format!(
        r##"<svg viewBox="0 0 600 160" class="panel-svg">
           <text x="20" y="25" fill="#7df" font-size="16">Curiosity &amp; Autonomy Panel</text>

           <text x="20" y="60" fill="#d0d0e0" font-size="13">Reward: {:.2}</text>
           <rect x="180" y="48" width="300" height="12" fill="#1f1f2a" stroke="#334" rx="2"/>
           <rect x="180" y="48" width="{:.2}" height="12" fill="#7df" rx="2">
             <animate attributeName="width" from="0" to="{:.2}" dur="0.6s" fill="freeze"/>
           </rect>

           <text x="20" y="90" fill="#d0d0e0" font-size="13">Curiosity Cycles: {}</text>
           <text x="20" y="120" fill="#d0d0e0" font-size="13">Autonomously Discovered Strategies: {}</text>
         </svg>"##,
        telemetry.curiosity_reward,
        reward_bar,
        reward_bar,
        telemetry.curiosity_cycles,
        telemetry.autonomously_discovered_strategies
    );

    format!(
        r##"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Sapient Soul Live Efficiency Dashboard</title>
<meta http-equiv="refresh" content="6" />
<style>
  body {{ font-family: monospace; background: #0b0b0f; color: #d0d0e0; padding: 20px; }}
  h1 {{ color: #7df; }}
  h2 {{ color: #8af; }}
  .panel {{ background: #15151a; border: 1px solid #334; padding: 15px; margin: 15px 0; border-radius: 6px; }}
  .panel-svg {{ width: 100%; }}
</style>
</head>
<body>
<h1>Live Efficiency Dashboard</h1>
<div class="panel">
  <h2>Domain Competence Matrix</h2>
  {}
</div>
<div class="panel">
  <h2>Thermodynamic Acceleration Panel</h2>
  {}
</div>
<div class="panel">
  <h2>Wide-Area Network Swarm Grid Panel</h2>
  {}
</div>
<div class="panel">
  <h2>Apple Intelligence Bridge Panel</h2>
  {}
</div>
<div class="panel">
  <h2>Memory Leak Profiler Panel</h2>
  {}
</div>
<div class="panel">
  <h2>Curiosity &amp; Autonomy Panel</h2>
  {}
</div>
<p><a href="/" style="color:#8af">Metrics</a> | <a href="/telemetry" style="color:#8af">Telemetry JSON</a></p>
</body>
</html>"##,
        competence_svg, thermodynamic_svg, swarm_svg, ai_svg, memory_svg, curiosity_svg
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
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

    // For Python, inject a safe standard-library header and patch common
    // LLM mistakes (e.g., math.mean() does not exist; use statistics).
    let code = if language == "python" {
        let mut fixed = code.to_string();
        for name in [
            "mean",
            "stdev",
            "pstdev",
            "variance",
            "pvariance",
            "mode",
            "median",
            "harmonic_mean",
            "geometric_mean",
        ] {
            fixed = fixed.replace(&format!("math.{}(", name), &format!("statistics.{}(", name));
            fixed = fixed.replace(
                &format!("from math import {}", name),
                &format!("from statistics import {}", name),
            );
        }
        format!(
            "import json, math, random, statistics, datetime, itertools, collections, string, re\n{}\n",
            fixed
        )
    } else {
        code.to_string()
    };

    fs::write(&filename, code).map_err(|e| format!("Cannot write tool file: {}", e))?;

    // For Python, compile-check the file before executing.  This turns
    // bracket-mismatch and other SyntaxErrors into a clear early failure.
    if language == "python" {
        let check = Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(&filename)
            .output()
            .map_err(|e| format!("Failed to run py_compile: {}", e))?;
        if !check.status.success() {
            let _ = fs::remove_file(&filename);
            return Err(format!(
                "Python syntax error: {}",
                String::from_utf8_lossy(&check.stderr).trim()
            ));
        }
    }

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
    if payload.language == "python" && !crate::is_safe_agent_code(&payload.code) {
        state.telemetry.lock().await.record_tool_result(false);
        return (
            StatusCode::BAD_REQUEST,
            Json(ToolRunResponse {
                status: "error".to_string(),
                output: None,
                error: Some("Tool code failed safety check".to_string()),
            }),
        );
    }

    let result = run_sandboxed_tool(&payload.name, &payload.code, &payload.language);
    let success = result.is_ok();
    state.telemetry.lock().await.record_tool_result(success);

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

#[derive(Deserialize)]
struct SkillLearnRequest {
    description: String,
    example_input: String,
    example_output: String,
    model: Option<String>,
}

#[derive(Serialize)]
struct SkillLearnResponse {
    status: String,
    skill_key: Option<String>,
    code: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct SkillRunRequest {
    skill_key: String,
    input: String,
}

#[derive(Serialize)]
struct SkillRunResponse {
    status: String,
    output: Option<String>,
    error: Option<String>,
}

pub fn skill_key(description: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(description.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn strip_markdown_code(raw: &str) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let mut start = 0;
    let mut end = lines.len();
    while start < end && lines[start].trim().starts_with("```") {
        start += 1;
    }
    while end > start && lines[end - 1].trim().starts_with("```") {
        end -= 1;
    }
    lines[start..end].join("\n").trim().to_string()
}

/// Compile-check a snippet of Python 3 before it is executed.
/// This catches bracket-mismatch and other `SyntaxError`s produced by LLM output.
pub fn validate_python_syntax(code: &str) -> Result<(), String> {
    if code.trim().is_empty() {
        return Err("Empty Python code".to_string());
    }

    let tools_dir = PathBuf::from("tools");
    fs::create_dir_all(&tools_dir).map_err(|e| format!("Cannot create tools dir: {}", e))?;

    let stamp = current_secs();
    let source_path = tools_dir.join(format!(".syntax_check_{}.py", stamp));
    fs::write(&source_path, code).map_err(|e| format!("Cannot write syntax-check file: {}", e))?;

    let output = Command::new("python3")
        .arg("-m")
        .arg("py_compile")
        .arg(&source_path)
        .output()
        .map_err(|e| format!("Failed to invoke python3 -m py_compile: {}", e))?;

    // Best-effort cleanup of the temporary source file.
    let _ = fs::remove_file(&source_path);

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(())
}

async fn learn_skill_handler(
    State(state): State<AppState>,
    Json(payload): Json<SkillLearnRequest>,
) -> impl IntoResponse {
    let model = payload.model.as_deref().unwrap_or("llama3:latest");

    let prompt = format!(
        "You are a Python 3 code generator. Given a task description and one example, write a self-contained function named `skill(x)` that solves the task. The function must be read-only and computational, using only: math, random, statistics, json, datetime, itertools, collections, string, re. Do not use: network, shell, file write, exec, eval, subprocess. Do not include markdown or explanations. Return ONLY the function definition.

Task description: {}
Example input: {:?}
Example output: {:?}

Provide only the Python function `def skill(x): ...`",
        payload.description, payload.example_input, payload.example_output
    );

    let raw_code = match state
        .ollama
        .generate(
            model,
            &prompt,
            Some("Return a valid Python 3 function named skill(x) only."),
        )
        .await
    {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(SkillLearnResponse {
                    status: "error".to_string(),
                    skill_key: None,
                    code: None,
                    error: Some(format!("LLM generation failed: {}", e)),
                }),
            )
        }
    };

    let code = strip_markdown_code(&raw_code);
    if !crate::is_safe_agent_code(&code) || !code.to_lowercase().contains("def skill(") {
        return (
            StatusCode::BAD_REQUEST,
            Json(SkillLearnResponse {
                status: "error".to_string(),
                skill_key: None,
                code: Some(code),
                error: Some("Generated code must define a `def skill(x)` function".to_string()),
            }),
        );
    }
    let test_code = format!("{}\nprint(skill({:?}))", code, payload.example_input);

    match run_sandboxed_tool("skill_test", &test_code, "python") {
        Ok(output) => {
            let actual = output.trim();
            let expected = payload.example_output.trim();
            if actual == expected {
                let key = skill_key(&payload.description);
                let skill = Skill {
                    description: payload.description,
                    language: "python".to_string(),
                    code: code.clone(),
                    example_input: payload.example_input,
                    example_output: payload.example_output,
                    learned_at: current_secs(),
                    success_count: 1,
                };
                {
                    let mut mind = state.core_mind.lock().await;
                    mind.skill_memory.skills.insert(key.clone(), skill);
                }
                (
                    StatusCode::OK,
                    Json(SkillLearnResponse {
                        status: "ok".to_string(),
                        skill_key: Some(key),
                        code: Some(code),
                        error: None,
                    }),
                )
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    Json(SkillLearnResponse {
                        status: "error".to_string(),
                        skill_key: None,
                        code: Some(code),
                        error: Some(format!(
                            "Output mismatch: got {:?}, expected {:?}",
                            actual, expected
                        )),
                    }),
                )
            }
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(SkillLearnResponse {
                status: "error".to_string(),
                skill_key: None,
                code: Some(code),
                error: Some(format!("Execution failed: {}", e)),
            }),
        ),
    }
}

async fn run_skill_handler(
    State(state): State<AppState>,
    Json(payload): Json<SkillRunRequest>,
) -> impl IntoResponse {
    let skill = {
        let mind = state.core_mind.lock().await;
        mind.skill_memory.skills.get(&payload.skill_key).cloned()
    };

    match skill {
        Some(skill) => {
            if !crate::is_safe_agent_code(&skill.code)
                || !skill.code.to_lowercase().contains("def skill(")
            {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(SkillRunResponse {
                        status: "error".to_string(),
                        output: None,
                        error: Some("Stored skill code failed safety check".to_string()),
                    }),
                );
            }
            let code = format!("{}\nprint(skill({:?}))", skill.code, payload.input);
            match run_sandboxed_tool(&payload.skill_key, &code, "python") {
                Ok(output) => {
                    {
                        let mut mind = state.core_mind.lock().await;
                        if let Some(s) = mind.skill_memory.skills.get_mut(&payload.skill_key) {
                            s.success_count += 1;
                        }
                        let current = mind
                            .skill_reliability
                            .get(&payload.skill_key)
                            .copied()
                            .unwrap_or(0.5);
                        mind.skill_reliability
                            .insert(payload.skill_key.clone(), current * 0.7 + 0.3);
                    }
                    (
                        StatusCode::OK,
                        Json(SkillRunResponse {
                            status: "ok".to_string(),
                            output: Some(output.trim().to_string()),
                            error: None,
                        }),
                    )
                }
                Err(e) => {
                    {
                        let mut mind = state.core_mind.lock().await;
                        let current = mind
                            .skill_reliability
                            .get(&payload.skill_key)
                            .copied()
                            .unwrap_or(0.5);
                        mind.skill_reliability
                            .insert(payload.skill_key.clone(), current * 0.7);
                    }
                    (
                        StatusCode::BAD_REQUEST,
                        Json(SkillRunResponse {
                            status: "error".to_string(),
                            output: None,
                            error: Some(e),
                        }),
                    )
                }
            }
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(SkillRunResponse {
                status: "error".to_string(),
                output: None,
                error: Some(format!("Skill {} not found", payload.skill_key)),
            }),
        ),
    }
}

#[derive(Deserialize)]
struct AddPursuitRequest {
    pursuit: String,
}

#[derive(Serialize)]
struct AddPursuitResponse {
    status: String,
    active_pursuits: Vec<String>,
}

async fn add_pursuit_handler(
    State(state): State<AppState>,
    Json(payload): Json<AddPursuitRequest>,
) -> impl IntoResponse {
    let mut mind = state.core_mind.lock().await;
    mind.active_pursuits.push_front(payload.pursuit.clone());
    let pursuits: Vec<String> = mind.active_pursuits.iter().cloned().collect();
    (
        StatusCode::OK,
        Json(AddPursuitResponse {
            status: "ok".to_string(),
            active_pursuits: pursuits,
        }),
    )
}

#[derive(Deserialize)]
struct TransferEvaluateRequest {
    domain: String,
    description: String,
    train_input: String,
    train_output: String,
    test_input: String,
    expected_test_output: Option<String>,
    model: Option<String>,
}

#[derive(Serialize)]
struct TransferEvaluateResponse {
    status: String,
    output: Option<String>,
    passed: Option<bool>,
    code: Option<String>,
    error: Option<String>,
}

async fn transfer_evaluate_handler(
    State(state): State<AppState>,
    Json(payload): Json<TransferEvaluateRequest>,
) -> impl IntoResponse {
    let model = payload.model.as_deref().unwrap_or("llama3:latest");
    let mut previous_attempt: Option<String> = None;
    let mut previous_error: Option<String> = None;

    for attempt in 0..3 {
        let mut prompt = format!(
            "You are a Python 3 code generator. The task is from a NEW domain the system has never trained on: {}. Given a description and one training example, write a self-contained function named `skill(x)` that solves the task. The function must be read-only and computational, using only: math, random, statistics, json, datetime, itertools, collections, string, re. Do not use: network, shell, file write, exec, eval, subprocess. Do not include markdown or explanations. Return ONLY the function definition.

Domain: {}
Task description: {}
Training input: {:?}
Training output: {:?}",
            payload.domain, payload.domain, payload.description, payload.train_input, payload.train_output
        );
        if let (Some(prev), Some(err)) = (previous_attempt.as_ref(), previous_error.as_ref()) {
            prompt.push_str(&format!(
                "\n\nYour previous attempt failed: {}\nPrevious code:\n{}\n\nRewrite the function so it works for both the training example and any similar input. Provide only the corrected Python function `def skill(x): ...`",
                err, prev
            ));
        } else {
            prompt.push_str("\n\nProvide only the Python function `def skill(x): ...`");
        }

        let raw_code = match state
            .ollama
            .generate(
                model,
                &prompt,
                Some("Return a valid Python 3 function named skill(x) only."),
            )
            .await
        {
            Ok(c) => c,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(TransferEvaluateResponse {
                        status: "error".to_string(),
                        output: None,
                        passed: None,
                        code: None,
                        error: Some(format!("LLM generation failed: {}", e)),
                    }),
                )
            }
        };

        let code = strip_markdown_code(&raw_code);
        if !crate::is_safe_agent_code(&code) || !code.to_lowercase().contains("def skill(") {
            previous_attempt = Some(code.clone());
            previous_error =
                Some("Generated code must define a `def skill(x)` function".to_string());
            if attempt < 2 {
                continue;
            }
            return (
                StatusCode::BAD_REQUEST,
                Json(TransferEvaluateResponse {
                    status: "error".to_string(),
                    output: None,
                    passed: Some(false),
                    code: Some(code),
                    error: previous_error,
                }),
            );
        }
        let train_code = format!("{}\nprint(skill({:?}))", code, payload.train_input);

        match run_sandboxed_tool("transfer_train", &train_code, "python") {
            Ok(output) => {
                let actual = output.trim();
                let expected = payload.train_output.trim();
                if actual != expected {
                    previous_attempt = Some(code.clone());
                    previous_error = Some(format!(
                        "Training example mismatch: got {:?}, expected {:?}",
                        actual, expected
                    ));
                    if attempt < 2 {
                        continue;
                    }
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(TransferEvaluateResponse {
                            status: "error".to_string(),
                            output: None,
                            passed: Some(false),
                            code: Some(code),
                            error: previous_error,
                        }),
                    );
                }

                let test_code = format!("{}\nprint(skill({:?}))", code, payload.test_input);
                match run_sandboxed_tool("transfer_test", &test_code, "python") {
                    Ok(output) => {
                        let test_output = output.trim().to_string();
                        let passed = payload
                            .expected_test_output
                            .as_ref()
                            .map(|expected| test_output == expected.trim());
                        if passed == Some(false) && attempt < 2 {
                            previous_attempt = Some(code.clone());
                            previous_error = Some(format!(
                                "Test input {:?} produced {:?}, expected {:?}",
                                payload.test_input,
                                test_output,
                                payload
                                    .expected_test_output
                                    .as_deref()
                                    .unwrap_or("(no expected output)")
                            ));
                            continue;
                        }
                        return (
                            StatusCode::OK,
                            Json(TransferEvaluateResponse {
                                status: "ok".to_string(),
                                output: Some(test_output),
                                passed,
                                code: Some(code),
                                error: None,
                            }),
                        );
                    }
                    Err(e) => {
                        previous_attempt = Some(code.clone());
                        previous_error = Some(format!("Test execution failed: {}", e));
                        if attempt < 2 {
                            continue;
                        }
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(TransferEvaluateResponse {
                                status: "error".to_string(),
                                output: None,
                                passed: Some(false),
                                code: Some(code),
                                error: previous_error,
                            }),
                        );
                    }
                }
            }
            Err(e) => {
                previous_attempt = Some(code.clone());
                previous_error = Some(format!("Training execution failed: {}", e));
                if attempt < 2 {
                    continue;
                }
                return (
                    StatusCode::BAD_REQUEST,
                    Json(TransferEvaluateResponse {
                        status: "error".to_string(),
                        output: None,
                        passed: Some(false),
                        code: Some(code),
                        error: previous_error,
                    }),
                );
            }
        }
    }

    (
        StatusCode::BAD_REQUEST,
        Json(TransferEvaluateResponse {
            status: "error".to_string(),
            output: None,
            passed: Some(false),
            code: previous_attempt,
            error: Some("Exhausted transfer attempts".to_string()),
        }),
    )
}

async fn identity_handler(State(state): State<AppState>) -> impl IntoResponse {
    let mind = state.core_mind.lock().await;
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "narrative": mind.narrative_identity(),
            "born_at": mind.born_at,
            "journal_entries": mind.identity_journal.len(),
            "last_journal_entry": mind.last_journal_entry,
        })),
    )
}

pub async fn start_telemetry_listener(
    port: u16,
) -> Result<TcpListener, Box<dyn std::error::Error>> {
    let fallback_base = port.saturating_add(1);
    for p in port..=port.saturating_add(15) {
        let addr = SocketAddr::from(([127, 0, 0, 1], p));
        match TcpListener::bind(addr).await {
            Ok(listener) => {
                if p == port {
                    tracing::info!("🛰️ [TELEMETRY ENGINE]: Online at http://{}/telemetry", addr);
                } else if p == fallback_base {
                    tracing::warn!("⚠️ [PORT COLLISION]: Port {} active. Diverting stream to http://{}/telemetry", port, addr);
                } else {
                    tracing::warn!(
                        "⚠️ [PORT COLLISION]: Diverting stream to http://{}/telemetry",
                        addr
                    );
                }
                return Ok(listener);
            }
            Err(_) => continue,
        }
    }
    Err(format!(
        "All telemetry ports in range {}..{} are in use",
        port,
        port + 15
    )
    .into())
}

pub async fn run_telemetry_server(
    telemetry: Arc<Mutex<TelemetryState>>,
    sensors: Arc<Mutex<SensorSnapshot>>,
    metrics: Arc<Mutex<MetricsLogger>>,
    core_mind: Arc<Mutex<FullySapientSoulMatrix>>,
    ollama: Arc<OllamaClient>,
    strategy_library: Arc<StrategyLibrary>,
    port: u16,
) {
    let state = AppState {
        telemetry,
        sensors,
        metrics,
        core_mind,
        ollama,
        strategy_library,
    };
    let app = Router::new()
        .route("/telemetry", get(telemetry_handler))
        .route("/metrics", get(metrics_json_handler))
        .route("/dashboard", get(metrics_dashboard_handler))
        .route("/live", get(live_dashboard_handler))
        .route("/tools/run", post(run_tool_handler))
        .route("/skills/learn", post(learn_skill_handler))
        .route("/skills/run", post(run_skill_handler))
        .route("/pursuits/add", post(add_pursuit_handler))
        .route("/transfer/evaluate", post(transfer_evaluate_handler))
        .route("/identity", get(identity_handler))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let listener = match start_telemetry_listener(port).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(
                "⚠️ Telemetry server bind failed: {}. Continuing without HTTP telemetry.",
                e
            );
            return;
        }
    };

    let chosen_port = listener.local_addr().map(|a| a.port()).unwrap_or(port);
    tracing::info!(
        "   POST http://127.0.0.1:{}/tools/run  {{\"name\",\"code\",\"language\"}}",
        chosen_port
    );
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("telemetry server ended: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::current_secs;

    #[test]
    fn current_secs_is_monotonic_and_reasonable() {
        let a = current_secs();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = current_secs();
        assert!(b >= a, "current_secs must be monotonic");
        // Unix epoch was 1970; the value should be well past that and not in the far future.
        assert!(a > 1_000_000_000, "timestamp should be after year 2001");
        assert!(a < 3_000_000_000, "timestamp should be before year 2063");
    }
}
