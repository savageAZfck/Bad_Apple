use crate::apple_intelligence;

/// Pure Apple Intelligence oracle wrapper.
///
/// The previous HTTP/Ollama socket path has been removed.  All generation
/// methods now route through the registered native Apple Intelligence callback
/// (`apple_intelligence::call`).  If the bridge is not loaded, generation
/// fails cleanly; the higher-level `ConscienceOracle` falls back to its
/// 2048-D cosine semantic matcher.
#[derive(Clone)]
pub struct OllamaClient;

impl OllamaClient {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate(
        &self,
        _model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        let full_prompt = build_prompt(system, prompt, None, None);
        if let Some(resp) = apple_intelligence::call(&full_prompt).await {
            return Ok(strip_code_fence(&resp));
        }
        Err("Apple Intelligence bridge not available".to_string())
    }

    pub async fn generate_constrained(
        &self,
        _model: &str,
        prompt: &str,
        system: Option<&str>,
        max_tokens: i32,
        temperature: f32,
    ) -> Result<String, String> {
        let full_prompt = build_prompt(system, prompt, Some(max_tokens), Some(temperature));
        if let Some(resp) = apple_intelligence::call(&full_prompt).await {
            return Ok(strip_code_fence(&resp));
        }
        Err("Apple Intelligence bridge not available".to_string())
    }

    pub async fn generate_structured(
        &self,
        _model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let json_system = "You must output only valid JSON. Do not add markdown, explanations, or any text outside the JSON object.";
        let combined_system = system
            .map(|s| format!("{} {}", json_system, s))
            .unwrap_or_else(|| json_system.to_string());
        let full_prompt = build_prompt(Some(&combined_system), prompt, None, None);

        if let Some(resp) = apple_intelligence::call(&full_prompt).await {
            let text = strip_code_fence(&resp);
            return serde_json::from_str::<serde_json::Value>(&text).map_err(|e| {
                format!(
                    "Failed to parse Apple Intelligence output as JSON: {}\nRaw: {}",
                    e, text
                )
            });
        }

        Err("Apple Intelligence bridge not available".to_string())
    }

    pub async fn is_available(&self) -> bool {
        apple_intelligence::is_available()
    }
}

fn build_prompt(
    system: Option<&str>,
    prompt: &str,
    max_tokens: Option<i32>,
    temperature: Option<f32>,
) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str(sys);
        out.push_str("\n\n");
    }
    out.push_str(prompt);
    if let Some(n) = max_tokens {
        out.push_str(&format!("\n\nRespond in at most {} tokens.", n));
    }
    if let Some(t) = temperature {
        let style = if t < 0.3 {
            "Be concise and deterministic."
        } else if t > 0.6 {
            "Be creative."
        } else {
            ""
        };
        if !style.is_empty() {
            out.push_str("\n\n");
            out.push_str(style);
        }
    }
    out
}

fn strip_code_fence(text: &str) -> String {
    let text = text.trim();
    if text.starts_with("```") {
        let without_start = text
            .strip_prefix("```json")
            .or_else(|| text.strip_prefix("```"))
            .unwrap_or(text);
        if let Some(end) = without_start.rfind("```") {
            without_start[..end].trim().to_string()
        } else {
            without_start.trim().to_string()
        }
    } else {
        text.to_string()
    }
}

#[derive(serde::Serialize)]
pub struct ReflectionPrompt {
    pub input: String,
    pub active_goals: Vec<String>,
    pub emotional_state: String,
    pub memory_count: usize,
    pub sensor_summary: String,
}

impl ReflectionPrompt {
    pub fn to_prompt(&self) -> String {
        format!(
            "You are a reflective AI agent. Respond in a single concise paragraph (max 3 sentences).\n\nCurrent input: '{}'\nActive goals: {:?}\nEmotional state: {}\nMemory count: {}\nSensor summary: {}\n\nReflect on what this means and what the agent should focus on next.",
            self.input, self.active_goals, self.emotional_state, self.memory_count, self.sensor_summary
        )
    }

    pub fn to_goal_prompt(&self) -> String {
        format!(
            "You are a planning AI. Given the current input, emotional state, active goals and memory count, produce at most 3 short, concrete sub-goal strings.\n\nCurrent input: '{}'\nActive goals: {:?}\nEmotional state: {}\nMemory count: {}\nSensor summary: {}\n\nSub-goals:",
            self.input, self.active_goals, self.emotional_state, self.memory_count, self.sensor_summary
        )
    }
}
