use crate::metal_uma::UmaBuffer;
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};
use libloading::Library;
use serde_json::Value;
use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use tokenizers::Tokenizer;

type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type PredictFn = unsafe extern "C" fn(*mut c_void, *const i32, usize) -> i32;
type ResetFn = unsafe extern "C" fn(*mut c_void) -> bool;
type PrewarmFn = unsafe extern "C" fn(*mut c_void) -> bool;
type AneRatioFn = unsafe extern "C" fn(*mut c_void) -> f64;
type ComputeUnitsFn = unsafe extern "C" fn(*mut c_void) -> i32;
type SelectionLatencyFn = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> bool;
type ArtifactRatioFn = unsafe extern "C" fn(*const c_char, i32) -> f64;
type ProbeShardFn = unsafe extern "C" fn(*const c_char, i32, usize, *mut u64) -> bool;

struct BridgeSymbols {
    library: Library,
    create: CreateFn,
    destroy: DestroyFn,
    predict: PredictFn,
    reset: ResetFn,
    prewarm: PrewarmFn,
    ane_ratio: AneRatioFn,
    compute_units: ComputeUnitsFn,
    selection_latency: SelectionLatencyFn,
    artifact_ratio: ArtifactRatioFn,
    probe_shard: ProbeShardFn,
}

#[derive(Clone, Debug)]
pub struct AneCoreConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_context_tokens: usize,
}

impl AneCoreConfig {
    pub fn from_env() -> Option<Self> {
        let model_path = std::env::var_os("BADAPPLE_ANE_MODEL")?.into();
        let tokenizer_path = std::env::var_os("BADAPPLE_ANE_TOKENIZER")?.into();
        let max_context_tokens = std::env::var("BADAPPLE_ANE_MAX_CONTEXT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2048);
        Some(Self {
            model_path,
            tokenizer_path,
            max_context_tokens,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneExecutionPriority {
    Utility,
    UserInitiated,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AneGovernorDecision {
    pub should_prewarm: bool,
    pub max_context_tokens: usize,
    pub priority: AneExecutionPriority,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AneGovernorInput {
    pub critic_score: f64,
    pub previous_critic_score: f64,
    pub average_loss: f64,
    pub battery_percent: f64,
    pub battery_charging: bool,
    pub photons: f64,
    pub memory_pressure_percent: f64,
}

pub fn governor_decision(input: AneGovernorInput) -> AneGovernorDecision {
    let constrained =
        (input.battery_percent > 0.0 && input.battery_percent < 20.0 && !input.battery_charging)
            || input.photons <= 0.01
            || input.memory_pressure_percent >= 85.0;
    let confidence_falling = input.critic_score < input.previous_critic_score
        || input.critic_score < 0.65
        || input.average_loss >= 4.0;
    AneGovernorDecision {
        should_prewarm: confidence_falling && !constrained,
        max_context_tokens: if constrained { 512 } else { 2048 },
        priority: if constrained {
            AneExecutionPriority::Utility
        } else {
            AneExecutionPriority::UserInitiated
        },
    }
}

#[derive(Debug)]
pub enum AneCoreError {
    Unconfigured,
    MissingArtifact(PathBuf),
    Bridge(String),
    Tokenizer(String),
    Prediction,
}

impl std::fmt::Display for AneCoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unconfigured => {
                write!(f, "BADAPPLE_ANE_MODEL and BADAPPLE_ANE_TOKENIZER are unset")
            }
            Self::MissingArtifact(path) => write!(f, "missing ANE artifact: {}", path.display()),
            Self::Bridge(error) => write!(f, "ANE bridge: {error}"),
            Self::Tokenizer(error) => write!(f, "ANE tokenizer: {error}"),
            Self::Prediction => write!(f, "ANE prediction failed"),
        }
    }
}

impl std::error::Error for AneCoreError {}

pub struct AneCore {
    _library: Library,
    handle: NonNull<c_void>,
    destroy: DestroyFn,
    predict: PredictFn,
    reset: ResetFn,
    prewarm: PrewarmFn,
    ane_ratio: AneRatioFn,
    compute_units: ComputeUnitsFn,
    selection_latency: SelectionLatencyFn,
    tokenizer: Tokenizer,
    max_context_tokens: usize,
    chat_template: Option<ChatTemplate>,
    eos_tokens: Vec<u32>,
}

unsafe impl Send for AneCore {}

impl Drop for AneCore {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle.as_ptr()) };
    }
}

impl AneCore {
    pub fn load(config: &AneCoreConfig) -> Result<Self, AneCoreError> {
        if !config.model_path.exists() {
            return Err(AneCoreError::MissingArtifact(config.model_path.clone()));
        }
        if !config.tokenizer_path.is_file() {
            return Err(AneCoreError::MissingArtifact(config.tokenizer_path.clone()));
        }
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|error| AneCoreError::Tokenizer(error.to_string()))?;
        let (chat_template, tokenizer_config) = load_tokenizer_config(&config.tokenizer_path);
        let eos_tokens = collect_eos_tokens(&tokenizer, tokenizer_config.as_ref());
        let symbols = load_bridge_symbols()?;
        let model_path = CString::new(config.model_path.to_string_lossy().as_bytes())
            .map_err(|error| AneCoreError::Bridge(error.to_string()))?;
        let handle = NonNull::new(unsafe { (symbols.create)(model_path.as_ptr()) })
            .ok_or_else(|| AneCoreError::Bridge("model load returned null".to_string()))?;
        Ok(Self {
            _library: symbols.library,
            handle,
            destroy: symbols.destroy,
            predict: symbols.predict,
            reset: symbols.reset,
            prewarm: symbols.prewarm,
            ane_ratio: symbols.ane_ratio,
            compute_units: symbols.compute_units,
            selection_latency: symbols.selection_latency,
            tokenizer,
            max_context_tokens: config.max_context_tokens.max(1),
            chat_template,
            eos_tokens,
        })
    }

    pub fn prewarm(&mut self) -> bool {
        unsafe { (self.prewarm)(self.handle.as_ptr()) }
    }

    pub fn reset(&mut self) -> bool {
        unsafe { (self.reset)(self.handle.as_ptr()) }
    }

    pub fn ane_placement_ratio(&mut self) -> f64 {
        unsafe { (self.ane_ratio)(self.handle.as_ptr()) }
    }

    pub fn compute_units_raw_value(&self) -> i32 {
        unsafe { (self.compute_units)(self.handle.as_ptr()) }
    }

    pub fn selection_latency_us(&self) -> Option<(u64, u64)> {
        let mut load = 0;
        let mut prewarm = 0;
        unsafe {
            (self.selection_latency)(self.handle.as_ptr(), &mut load, &mut prewarm)
                .then_some((load, prewarm))
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        system: Option<&str>,
        max_new_tokens: usize,
        context_limit: usize,
    ) -> Result<String, AneCoreError> {
        self.generate_streaming(prompt, system, max_new_tokens, context_limit, |_| true)
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        system: Option<&str>,
        max_new_tokens: usize,
        context_limit: usize,
        mut on_token: F,
    ) -> Result<String, AneCoreError>
    where
        F: FnMut(&str) -> bool,
    {
        let rendered = match &self.chat_template {
            Some(template) => {
                let mut messages = Vec::with_capacity(2);
                if let Some(system) = system {
                    messages.push(Message::system(system));
                }
                messages.push(Message::user(prompt));
                template
                    .render_messages(&messages, true)
                    .map_err(|error| AneCoreError::Tokenizer(error.to_string()))?
            }
            None => match system {
                Some(system) => format!("{}\n\n{}", system, prompt),
                None => prompt.to_string(),
            },
        };

        // When a chat template is in use it already emits the model's special tokens,
        // so we must not ask the tokenizer to add them again.
        let add_special_tokens = self.chat_template.is_none();
        let encoding = self
            .tokenizer
            .encode(rendered.as_str(), add_special_tokens)
            .map_err(|error| AneCoreError::Tokenizer(error.to_string()))?;
        let limit = context_limit.min(self.max_context_tokens).max(1);
        let reserved = max_new_tokens.min(limit.saturating_sub(1));
        let prompt_limit = limit.saturating_sub(reserved).max(1);
        let ids = encoding.get_ids();
        let ids = &ids[ids.len().saturating_sub(prompt_limit)..];
        if ids.is_empty() {
            return Err(AneCoreError::Prediction);
        }

        self.reset();
        let prompt_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let mut next = self.predict_tokens(&prompt_ids)?;
        FIRST_TOKEN_LATENCY_US.store(
            LAST_TOKEN_LATENCY_US.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        let mut generated = Vec::with_capacity(max_new_tokens.min(4096));
        let mut decoded = String::new();
        for _ in 0..max_new_tokens {
            if next < 0 {
                return Err(AneCoreError::Prediction);
            }
            let token = next as u32;
            generated.push(token);
            let current = self
                .tokenizer
                .decode(&generated, true)
                .map_err(|error| AneCoreError::Tokenizer(error.to_string()))?;
            if let Some(delta) = current.strip_prefix(&decoded) {
                if !delta.is_empty() && !on_token(delta) {
                    decoded = current;
                    break;
                }
            }
            decoded = current;
            if self.eos_tokens.contains(&token) {
                break;
            }
            next = self.predict_tokens(&[next])?;
        }
        Ok(decoded)
    }

    fn predict_tokens(&mut self, tokens: &[i32]) -> Result<i32, AneCoreError> {
        let mut input = UmaBuffer::<i32>::new(tokens.len())
            .ok_or_else(|| AneCoreError::Bridge("Metal shared allocation failed".to_string()))?;
        input.as_mut_slice().copy_from_slice(tokens);
        let start = Instant::now();
        let output =
            unsafe { (self.predict)(self.handle.as_ptr(), input.as_slice().as_ptr(), input.len()) };
        LAST_TOKEN_LATENCY_US.store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        TOKEN_COUNT.fetch_add(1, Ordering::Relaxed);
        if output < 0 {
            PREDICTION_FAILURES.fetch_add(1, Ordering::Relaxed);
            Err(AneCoreError::Prediction)
        } else {
            Ok(output)
        }
    }
}

fn load_tokenizer_config(tokenizer_path: &Path) -> (Option<ChatTemplate>, Option<Value>) {
    let config_path = tokenizer_path.with_file_name("tokenizer_config.json");
    let raw = match std::fs::read_to_string(&config_path) {
        Ok(raw) => raw,
        Err(_) => return (None, None),
    };
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(_) => return (None, None),
    };
    let template = match serde_json::from_value::<TokenizerConfig>(value.clone()) {
        Ok(config) => ChatTemplate::from_tokenizer_config(&config).ok(),
        Err(_) => None,
    };
    (template, Some(value))
}

fn token_string_from_config(config: &Value, key: &str) -> Option<String> {
    match config.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Object(m) => m.get("content")?.as_str().map(String::from),
        _ => None,
    }
}

fn collect_eos_tokens(tokenizer: &Tokenizer, config: Option<&Value>) -> Vec<u32> {
    let mut ids = std::collections::HashSet::new();

    // Collect from the tokenizer's added special tokens.  This covers
    // end-of-turn markers such as Qwen's `।` even when the config calls
    // them `additional_special_tokens` rather than `eos_token`.
    for (id, token) in tokenizer.get_added_tokens_decoder() {
        if token.special && is_stop_token(&token.content) {
            ids.insert(id);
        }
    }

    if let Some(config) = config {
        for key in ["eos_token", "pad_token"] {
            if let Some(s) = token_string_from_config(config, key) {
                if let Some(id) = tokenizer.token_to_id(&s) {
                    ids.insert(id);
                }
            }
        }

        if let Some(Value::Array(arr)) = config.get("additional_special_tokens") {
            for value in arr {
                if let Some(s) = value.as_str() {
                    if let Some(id) = tokenizer.token_to_id(s) {
                        ids.insert(id);
                    }
                } else if let Some(s) = value.get("content").and_then(Value::as_str) {
                    if let Some(id) = tokenizer.token_to_id(s) {
                        ids.insert(id);
                    }
                }
            }
        }

        if let Some(Value::Object(decoder)) = config.get("added_tokens_decoder") {
            for (id, token) in decoder {
                let content = token
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if is_stop_token(content) {
                    if let Ok(id) = id.parse() {
                        ids.insert(id);
                    }
                }
            }
        }
    }

    ids.into_iter().collect()
}

fn is_stop_token(token: &str) -> bool {
    let lower = token.to_lowercase();
    matches!(
        lower.as_str(),
        "<|endoftext|>"
            | "<|end|>"
            | "</s>"
            | "<s>"
            | "<|eot_id|>"
            | "<|eom_id|>"
            | "<|im_end|>"
            | "।"
            | "<|assistant|>"
            | "[end]"
    )
}

fn bridge_paths() -> Vec<PathBuf> {
    let mut paths = vec![
        PathBuf::from("libBadAppleBridge.dylib"),
        PathBuf::from("target/release/libBadAppleBridge.dylib"),
        PathBuf::from("target/debug/libBadAppleBridge.dylib"),
    ];
    if let Ok(path) = std::env::var("BADAPPLE_ANE_BRIDGE") {
        paths.insert(0, PathBuf::from(path));
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            paths.insert(0, parent.join("libBadAppleBridge.dylib"));
        }
    }
    paths
}

fn load_bridge_symbols() -> Result<BridgeSymbols, AneCoreError> {
    let mut failures = Vec::new();
    for path in bridge_paths() {
        if !path.exists() {
            continue;
        }
        let library = match unsafe { Library::new(&path) } {
            Ok(library) => library,
            Err(error) => {
                failures.push(format!("{}: {error}", path.display()));
                continue;
            }
        };
        unsafe {
            let create = library.get::<CreateFn>(b"bad_apple_ane_create\0");
            let destroy = library.get::<DestroyFn>(b"bad_apple_ane_destroy\0");
            let predict = library.get::<PredictFn>(b"bad_apple_ane_predict_next\0");
            let reset = library.get::<ResetFn>(b"bad_apple_ane_reset\0");
            let prewarm = library.get::<PrewarmFn>(b"bad_apple_ane_prewarm\0");
            let ane_ratio = library.get::<AneRatioFn>(b"bad_apple_ane_placement_ratio\0");
            let compute_units =
                library.get::<ComputeUnitsFn>(b"bad_apple_ane_compute_units_raw_value\0");
            let selection_latency =
                library.get::<SelectionLatencyFn>(b"bad_apple_ane_selection_latency_us\0");
            let artifact_ratio =
                library.get::<ArtifactRatioFn>(b"bad_apple_coreml_placement_ratio\0");
            let probe_shard = library.get::<ProbeShardFn>(b"bad_apple_ane_probe_shard\0");
            if let (
                Ok(create),
                Ok(destroy),
                Ok(predict),
                Ok(reset),
                Ok(prewarm),
                Ok(ane_ratio),
                Ok(compute_units),
                Ok(selection_latency),
                Ok(artifact_ratio),
                Ok(probe_shard),
            ) = (
                create,
                destroy,
                predict,
                reset,
                prewarm,
                ane_ratio,
                compute_units,
                selection_latency,
                artifact_ratio,
                probe_shard,
            ) {
                let symbols = BridgeSymbols {
                    create: *create,
                    destroy: *destroy,
                    predict: *predict,
                    reset: *reset,
                    prewarm: *prewarm,
                    ane_ratio: *ane_ratio,
                    compute_units: *compute_units,
                    selection_latency: *selection_latency,
                    artifact_ratio: *artifact_ratio,
                    probe_shard: *probe_shard,
                    library,
                };
                return Ok(symbols);
            }
        }
        failures.push(format!("{}: missing ANE symbols", path.display()));
    }
    Err(AneCoreError::Bridge(if failures.is_empty() {
        "no in-process bridge dylib found".to_string()
    } else {
        failures.join("; ")
    }))
}

pub fn audit_artifact_placement(
    path: &Path,
    compute_units_raw_value: i32,
) -> Result<f64, AneCoreError> {
    let symbols = load_bridge_symbols()?;
    let path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|error| AneCoreError::Bridge(error.to_string()))?;
    let ratio = unsafe { (symbols.artifact_ratio)(path.as_ptr(), compute_units_raw_value) };
    if ratio < 0.0 {
        Err(AneCoreError::Bridge(
            "CoreML placement audit failed".to_string(),
        ))
    } else {
        Ok(ratio)
    }
}

pub fn probe_compiled_shard(
    path: &Path,
    compute_units_raw_value: i32,
    iterations: usize,
) -> Result<u64, AneCoreError> {
    let symbols = load_bridge_symbols()?;
    let path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|error| AneCoreError::Bridge(error.to_string()))?;
    let mut average_latency_us = 0;
    let succeeded = unsafe {
        (symbols.probe_shard)(
            path.as_ptr(),
            compute_units_raw_value,
            iterations.max(1),
            &mut average_latency_us,
        )
    };
    if succeeded {
        Ok(average_latency_us)
    } else {
        Err(AneCoreError::Prediction)
    }
}

static ENGINE: OnceLock<Mutex<AneCore>> = OnceLock::new();
static LAST_TOKEN_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static FIRST_TOKEN_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static TOKEN_COUNT: AtomicU64 = AtomicU64::new(0);
static PREDICTION_FAILURES: AtomicU64 = AtomicU64::new(0);
static CONTEXT_LIMIT: AtomicU64 = AtomicU64::new(2048);

pub fn initialize_from_env() -> Result<(), AneCoreError> {
    let config = AneCoreConfig::from_env().ok_or(AneCoreError::Unconfigured)?;
    let engine = AneCore::load(&config)?;
    ENGINE
        .set(Mutex::new(engine))
        .map_err(|_| AneCoreError::Bridge("engine already initialized".to_string()))
}

pub fn is_available() -> bool {
    ENGINE.get().is_some()
}

pub fn prewarm() -> bool {
    ENGINE
        .get()
        .and_then(|engine| engine.lock().ok())
        .is_some_and(|mut engine| engine.prewarm())
}

pub fn set_context_limit(context_limit: usize) {
    CONTEXT_LIMIT.store(context_limit.max(1) as u64, Ordering::Relaxed);
}

pub fn context_limit() -> usize {
    CONTEXT_LIMIT.load(Ordering::Relaxed) as usize
}

pub fn generate_sync(
    prompt: &str,
    max_new_tokens: usize,
    context_limit: usize,
) -> Result<String, AneCoreError> {
    generate_with_system(prompt, None, max_new_tokens, context_limit)
}

pub fn generate_with_system(
    prompt: &str,
    system: Option<&str>,
    max_new_tokens: usize,
    context_limit: usize,
) -> Result<String, AneCoreError> {
    let engine = ENGINE.get().ok_or(AneCoreError::Unconfigured)?;
    let mut engine = engine
        .lock()
        .map_err(|error| AneCoreError::Bridge(error.to_string()))?;
    engine.generate(prompt, system, max_new_tokens, context_limit)
}

pub fn generate_streaming<F>(
    prompt: &str,
    max_new_tokens: usize,
    context_limit: usize,
    on_token: F,
) -> Result<String, AneCoreError>
where
    F: FnMut(&str) -> bool,
{
    let engine = ENGINE.get().ok_or(AneCoreError::Unconfigured)?;
    let mut engine = engine
        .lock()
        .map_err(|error| AneCoreError::Bridge(error.to_string()))?;
    engine.generate_streaming(prompt, None, max_new_tokens, context_limit, on_token)
}

pub fn placement_ratio() -> Option<f64> {
    ENGINE
        .get()
        .and_then(|engine| engine.lock().ok())
        .map(|mut engine| engine.ane_placement_ratio())
}

pub fn compute_units_raw_value() -> Option<i32> {
    ENGINE
        .get()
        .and_then(|engine| engine.lock().ok())
        .map(|engine| engine.compute_units_raw_value())
}

pub fn selection_latency_us() -> Option<(u64, u64)> {
    ENGINE
        .get()
        .and_then(|engine| engine.lock().ok())
        .and_then(|engine| engine.selection_latency_us())
}

pub fn metrics() -> (u64, u64, u64) {
    (
        LAST_TOKEN_LATENCY_US.load(Ordering::Relaxed),
        TOKEN_COUNT.load(Ordering::Relaxed),
        PREDICTION_FAILURES.load(Ordering::Relaxed),
    )
}

pub fn first_token_latency_us() -> u64 {
    FIRST_TOKEN_LATENCY_US.load(Ordering::Relaxed)
}

pub fn reset_counters() {
    LAST_TOKEN_LATENCY_US.store(0, Ordering::Relaxed);
    FIRST_TOKEN_LATENCY_US.store(0, Ordering::Relaxed);
    TOKEN_COUNT.store(0, Ordering::Relaxed);
    PREDICTION_FAILURES.store(0, Ordering::Relaxed);
}

/// Live decode vitals for the Bad Apple boot banner.
/// Returns `(prewarm_latency_us, avg_token_latency_us, tokens_per_second)`.
/// `elapsed_us` is the wall time the caller measured around `generate_sync`.
pub fn decode_vitals(elapsed_us: u64) -> (u64, u64, f64) {
    let (last_token_us, token_count, _) = metrics();
    let first_token_us = first_token_latency_us();
    let prewarm_us = selection_latency_us()
        .map(|(_, prewarm)| prewarm)
        .unwrap_or(0);
    let avg_decode_us = if token_count > 1 {
        elapsed_us.saturating_sub(first_token_us) / (token_count - 1)
    } else {
        last_token_us
    };
    let tokens_per_second = if avg_decode_us > 0 {
        1_000_000.0 / avg_decode_us as f64
    } else {
        0.0
    };
    (prewarm_us, avg_decode_us, tokens_per_second)
}

pub fn artifact_is_compiled_model(path: &Path) -> bool {
    path.is_dir()
        && path
            .extension()
            .is_some_and(|extension| extension == "mlmodelc")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_power_policy_reduces_context_and_suppresses_prewarm() {
        let input = AneGovernorInput {
            critic_score: 0.2,
            previous_critic_score: 0.8,
            average_loss: 4.61,
            battery_percent: 10.0,
            battery_charging: false,
            photons: 0.0,
            ..AneGovernorInput::default()
        };
        let decision = governor_decision(input);
        assert_eq!(decision.max_context_tokens, 512);
        assert_eq!(decision.priority, AneExecutionPriority::Utility);
        assert!(!decision.should_prewarm);
    }

    #[test]
    fn falling_confidence_prewarms_when_resources_allow() {
        let input = AneGovernorInput {
            critic_score: 0.5,
            previous_critic_score: 0.8,
            average_loss: 4.61,
            battery_percent: 80.0,
            battery_charging: true,
            photons: 1.0,
            memory_pressure_percent: 20.0,
        };
        let decision = governor_decision(input);
        assert_eq!(decision.max_context_tokens, 2048);
        assert_eq!(decision.priority, AneExecutionPriority::UserInitiated);
        assert!(decision.should_prewarm);
    }

    #[test]
    fn missing_configuration_is_unavailable() {
        if AneCoreConfig::from_env().is_none() {
            assert!(!is_available());
        }
    }
}
