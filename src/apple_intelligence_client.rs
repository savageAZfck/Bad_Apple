use crate::apple_intelligence;

/// Native Apple Intelligence oracle wrapper.
///
/// All generation methods route through the registered native Apple
/// Intelligence callback (`apple_intelligence::call`).  If the bridge is not
/// loaded, generation fails cleanly; the higher-level `ConscienceOracle`
/// falls back to its 2048-D cosine semantic matcher.
#[derive(Clone)]
pub struct AppleIntelligenceClient;

impl AppleIntelligenceClient {
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
            return parse_or_repair_json(&text).map_err(|e| {
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

/// Try to parse JSON strictly; if that fails, attempt a structural repair
/// and then a field-level regex salvage for the common `name/language/code`
/// tool schema.
fn parse_or_repair_json(text: &str) -> Result<serde_json::Value, String> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        return Ok(v);
    }

    if let Some(repaired) = repair_truncated_json(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
            return Ok(v);
        }
    }

    if let Some(v) = salvage_tool_fields(text) {
        return Ok(v);
    }

    Err("unrepairable JSON".to_string())
}

/// Heuristic repair for the most common Apple Intelligence truncation: an
/// unclosed string at the end of an object. We track whether we are inside a
/// string, then close the string and balance any remaining `{` / `[` depth.
fn repair_truncated_json(s: &str) -> Option<String> {
    let mut in_string = false;
    let mut escaped = false;
    let mut object_depth: i32 = 0;
    let mut array_depth: i32 = 0;

    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && in_string {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => object_depth += 1,
            '}' => object_depth -= 1,
            '[' => array_depth += 1,
            ']' => array_depth -= 1,
            _ => {}
        }
    }

    if !in_string && object_depth <= 0 && array_depth <= 0 {
        return None;
    }

    let mut out = s.to_string();
    if in_string {
        // If the text ends while we are still inside a string, close it.
        out.push('"');
    }
    // Balance any unclosed objects or arrays.
    for _ in 0..object_depth.max(0) {
        out.push('}');
    }
    for _ in 0..array_depth.max(0) {
        out.push(']');
    }
    Some(out)
}

/// Last-ditch salvage for the tool-generation JSON schema.  Extracts
/// `name`, `language`, and `code` fields without external dependencies and
/// returns a valid `serde_json::Value`, even if the surrounding JSON is
/// mangled.
fn salvage_tool_fields(text: &str) -> Option<serde_json::Value> {
    let name = extract_json_string_value(text, "name");
    let language = extract_json_string_value(text, "language");
    let code = extract_json_string_value(text, "code");

    if name.is_none() && code.is_none() {
        return None;
    }

    let mut map = serde_json::Map::new();
    if let Some(n) = name {
        map.insert("name".to_string(), serde_json::Value::String(n));
    }
    if let Some(l) = language {
        map.insert("language".to_string(), serde_json::Value::String(l));
    }
    if let Some(c) = code {
        map.insert("code".to_string(), serde_json::Value::String(c));
    }
    Some(serde_json::Value::Object(map))
}

/// Extract the string value for a given JSON key using a simple text scan
/// that tolerates unclosed or truncated values.
fn extract_json_string_value(text: &str, key: &str) -> Option<String> {
    let key_pattern = format!("\"{}\"", key);
    let mut search = text;
    let start;
    loop {
        if let Some(pos) = search.find(&key_pattern) {
            let after = &search[pos + key_pattern.len()..];
            // Skip optional whitespace and the colon.
            let after = after.trim_start();
            if let Some(stripped) = after.strip_prefix(':') {
                start = stripped;
                break;
            }
            search = after;
        } else {
            return None;
        }
    }

    let rest = start.trim_start();
    // We expect the value to start with a quote.
    if !rest.starts_with('"') {
        return None;
    }
    let rest = &rest[1..];

    // Find the end of the string.  In a truncated response the closing quote
    // may be missing, so fall back to the end of a comma or closing brace.
    let mut value = String::new();
    let mut escaped = false;
    for c in rest.chars() {
        if escaped {
            value.push(c);
            escaped = false;
            continue;
        }
        if c == '\\' {
            value.push(c);
            escaped = true;
            continue;
        }
        if c == '"' {
            break;
        }
        if c == ',' || c == '}' {
            // A comma or brace before we closed the quote means the string
            // was truncated.  Stop and treat the accumulated value as the
            // intended string.
            break;
        }
        value.push(c);
    }

    let value = value.trim().to_string();
    if value.is_empty() {
        None
    } else {
        Some(value)
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
