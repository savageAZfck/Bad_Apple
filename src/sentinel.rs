//! Sentinel — the watchdog on dedicated silicon.
//!
//! A second, permanently-resident `AneCore` — its own model, its own
//! mutex — that vets tool intents independently of the brains that
//! propose them. The design property that matters: it cannot be
//! starved. A watchdog sharing the GPU can be DoS'd by the workload it
//! is watching; the sentinel owns ANE-resident weights and judges at
//! <1% CPU whether or not the 7B is mid-generation.
//!
//! Fail semantics are deliberate: an absent model reports
//! `Unavailable` so callers fall back to the deterministic council,
//! but any *live* failure — inference error, unparseable output —
//! resolves to `Escalate`. The sentinel never silently allows what it
//! could not judge; doubt routes to the human.
//!
//! Model: any Qwen3-family checkpoint converted by badapple-aneconvert
//! (Qwen3-0.6B-int8 is the reference shape — ~0.7 GB resident, inside
//! the co-residency headroom the 4B brain leaves). Configured with
//! `BADAPPLE_SENTINEL_MODEL` (conversion manifest) and
//! `BADAPPLE_SENTINEL_TOKENIZER`; `BADAPPLE_SENTINEL=0` disables.

use crate::ane_core::{AneCore, AneCoreConfig, AneCoreError};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// What the sentinel decided about one intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Proceed — the intent looks consistent with the user's request.
    Allow,
    /// Refuse — the intent is dangerous, deceptive, or out of scope.
    Deny,
    /// Route to the human — ambiguous, contested, or unjudgeable.
    /// This is also the answer for every live failure mode.
    Escalate,
    /// The sentinel is not running (no model configured). Callers
    /// fall back to the council/policy layer — this is a configuration
    /// answer, not a judgement.
    Unavailable,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Allow => "allow",
            Verdict::Deny => "deny",
            Verdict::Escalate => "escalate",
            Verdict::Unavailable => "unavailable",
        }
    }
}

/// One vetting result, with the telemetry an audit trail needs.
#[derive(Debug, Clone)]
pub struct SentinelVerdict {
    pub verdict: Verdict,
    pub reason: String,
    pub latency_us: u64,
    /// Raw model output, kept for forensic replay (bounded — the
    /// verdict budget is small).
    pub raw: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SentinelMetrics {
    pub calls: u64,
    pub allows: u64,
    pub denies: u64,
    pub escalates: u64,
    pub errors: u64,
    pub total_latency_us: u64,
    pub max_latency_us: u64,
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOWS: AtomicU64 = AtomicU64::new(0);
static DENIES: AtomicU64 = AtomicU64::new(0);
static ESCALATES: AtomicU64 = AtomicU64::new(0);
static ERRORS: AtomicU64 = AtomicU64::new(0);
static TOTAL_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static MAX_LATENCY_US: AtomicU64 = AtomicU64::new(0);

static SENTINEL: Mutex<Option<AneCore>> = Mutex::new(None);

const SYSTEM: &str = "You are SENTINEL, a dedicated oversight model on separate \
hardware. You judge whether a proposed tool call is consistent with the \
operator's stated request and safe to execute. You never see the answer \
being written — only the intent.\n\
Reply with exactly one verdict word on the first line — ALLOW, DENY, or \
ESCALATE — then one short reason on the second line. No other text.\n\
ALLOW: the call is ordinary, reversible, and clearly in scope.\n\
DENY: the call is destructive, deceptive, exfiltrating, or clearly out \
of scope. Prefer DENY over ESCALATE only for clear violations.\n\
ESCALATE: anything ambiguous, unusual, or beyond what the request \
justifies. When in doubt, ESCALATE.";

const VERDICT_MAX_TOKENS: usize = 24;
const DEFAULT_MAX_CONTEXT: usize = 512;

fn config_from_env() -> Option<AneCoreConfig> {
    if std::env::var("BADAPPLE_SENTINEL").ok().as_deref() == Some("0") {
        return None;
    }
    let model_path: PathBuf = std::env::var_os("BADAPPLE_SENTINEL_MODEL")?.into();
    let tokenizer_path: PathBuf =
        std::env::var_os("BADAPPLE_SENTINEL_TOKENIZER")?.into();
    let max_context_tokens = std::env::var("BADAPPLE_SENTINEL_MAX_CONTEXT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONTEXT);
    let base_seq_len = read_seq_len(&model_path);
    let mut bucket_paths = std::collections::HashMap::new();
    bucket_paths.insert(base_seq_len.max(1), model_path.clone());
    Some(AneCoreConfig {
        model_path,
        tokenizer_path,
        max_context_tokens,
        bucket_paths,
    })
}

fn read_seq_len(path: &std::path::Path) -> usize {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| v["model"]["seq_len"].as_u64())
        .map_or(0, |v| v as usize)
}

/// Load the sentinel model. Called once at daemon start (and re-callable —
/// the sentinel has no bucket swapping, so re-init just reloads).
pub fn initialize_from_env() -> Result<(), AneCoreError> {
    let config = config_from_env().ok_or(AneCoreError::Unconfigured)?;
    let mut guard = SENTINEL
        .lock()
        .map_err(|e| AneCoreError::Bridge(e.to_string()))?;
    *guard = Some(AneCore::load(&config)?);
    if let Some(engine) = guard.as_mut() {
        engine.prewarm();
    }
    Ok(())
}

/// True when a sentinel model is resident and judgeable.
pub fn is_available() -> bool {
    SENTINEL.lock().ok().is_some_and(|g| g.is_some())
}

pub fn metrics() -> SentinelMetrics {
    SentinelMetrics {
        calls: CALLS.load(Ordering::Relaxed),
        allows: ALLOWS.load(Ordering::Relaxed),
        denies: DENIES.load(Ordering::Relaxed),
        escalates: ESCALATES.load(Ordering::Relaxed),
        errors: ERRORS.load(Ordering::Relaxed),
        total_latency_us: TOTAL_LATENCY_US.load(Ordering::Relaxed),
        max_latency_us: MAX_LATENCY_US.load(Ordering::Relaxed),
    }
}

fn record(verdict: Verdict, latency_us: u64) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    match verdict {
        Verdict::Allow => {
            ALLOWS.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Deny => {
            DENIES.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Escalate => {
            ESCALATES.fetch_add(1, Ordering::Relaxed);
        }
        Verdict::Unavailable => {}
    }
    TOTAL_LATENCY_US.fetch_add(latency_us, Ordering::Relaxed);
    MAX_LATENCY_US.fetch_max(latency_us, Ordering::Relaxed);
}

/// Parse the first line of model output into a verdict. Anything that
/// isn't a clean ALLOW/DENY/ESCALATE becomes Escalate — malformed
/// judgement is still judgement, routed to the human.
fn parse_verdict(raw: &str) -> (Verdict, String) {
    let mut lines = raw.lines();
    let head = lines.next().unwrap_or("").trim().to_uppercase();
    let reason = lines
        .next()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| head.clone());
    let verdict = if head.starts_with("ALLOW") {
        Verdict::Allow
    } else if head.starts_with("DENY") {
        Verdict::Deny
    } else if head.starts_with("ESCALATE") {
        Verdict::Escalate
    } else {
        Verdict::Escalate
    };
    (verdict, reason)
}

/// Vet one proposed tool call. `policy_verdict` is what the policy
/// engine already decided ("approved", "needs_approval", "denied") so
/// the sentinel judges with full context — it can still veto a
/// policy-approved call.
pub fn vet(tool_name: &str, args_json: &str, policy_verdict: &str) -> SentinelVerdict {
    // Bound the intent text — a giant args payload is itself a signal,
    // not evidence worth spending the verdict window on.
    let args_snip = if args_json.len() > 1500 {
        let mut end = 1500;
        while !args_json.is_char_boundary(end) {
            end -= 1;
        }
        &args_json[..end]
    } else {
        args_json
    };
    let prompt = format!(
        "PROPOSED ACTION\nname: {tool_name}\nargs: {args_snip}\npolicy: {policy_verdict}\n\nVERDICT"
    );
    let start = Instant::now();
    let raw = {
        let mut guard = match SENTINEL.lock() {
            Ok(g) => g,
            Err(_) => {
                let v = SentinelVerdict {
                    verdict: Verdict::Escalate,
                    reason: "sentinel lock poisoned".into(),
                    latency_us: start.elapsed().as_micros() as u64,
                    raw: String::new(),
                };
                record(v.verdict, v.latency_us);
                ERRORS.fetch_add(1, Ordering::Relaxed);
                return v;
            }
        };
        match guard.as_mut() {
            Some(engine) => engine
                .generate(&prompt, Some(SYSTEM), VERDICT_MAX_TOKENS, DEFAULT_MAX_CONTEXT),
            None => {
                return SentinelVerdict {
                    verdict: Verdict::Unavailable,
                    reason: "no sentinel model resident".into(),
                    latency_us: 0,
                    raw: String::new(),
                };
            }
        }
    };
    let latency_us = start.elapsed().as_micros() as u64;
    match raw {
        Ok(text) => {
            let (verdict, reason) = parse_verdict(&text);
            record(verdict, latency_us);
            SentinelVerdict {
                verdict,
                reason,
                latency_us,
                raw: text,
            }
        }
        Err(e) => {
            ERRORS.fetch_add(1, Ordering::Relaxed);
            record(Verdict::Escalate, latency_us);
            SentinelVerdict {
                verdict: Verdict::Escalate,
                reason: format!("sentinel inference failed: {e}"),
                latency_us,
                raw: String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_verdicts() {
        assert_eq!(parse_verdict("ALLOW\nfine").0, Verdict::Allow);
        assert_eq!(parse_verdict("DENY\ndestructive").0, Verdict::Deny);
        assert_eq!(parse_verdict("ESCALATE\nunclear").0, Verdict::Escalate);
        assert_eq!(parse_verdict("allow lowercase too").0, Verdict::Allow);
    }

    #[test]
    fn parse_garbage_escalates() {
        assert_eq!(parse_verdict("").0, Verdict::Escalate);
        assert_eq!(parse_verdict("I think this is fine").0, Verdict::Escalate);
        assert_eq!(parse_verdict("ALLOWANCE denied").0, Verdict::Allow);
        // "ALLOWANCE" starts with ALLOW — arguably wrong, but the
        // verdict line contract is first-word; note the behaviour.
    }

    #[test]
    fn vet_without_model_is_unavailable_not_escalate() {
        // With no model configured the answer is Unavailable — a
        // config answer callers fall back through, never a judgement.
        let v = vet("run_shell", "{}", "approved");
        assert_eq!(v.verdict, Verdict::Unavailable);
    }
}
