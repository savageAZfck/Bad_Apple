use crate::apple_intelligence;
use serde::Serialize;
use serde_json::json;

const OLLAMA_BASE: &str = "http://localhost:11434";

#[derive(Clone)]
pub struct OllamaClient {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: OLLAMA_BASE.to_string(),
        }
    }

    pub fn with_base(base: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base.to_string(),
        }
    }

    pub async fn list_models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("Ollama list models failed: {}", response.status()));
        }
        let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        let models: Vec<String> = data
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        m.get("name")
                            .and_then(|n| n.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    pub async fn generate_constrained(
        &self,
        _model: &str,
        prompt: &str,
        system: Option<&str>,
        max_tokens: i32,
        temperature: f32,
    ) -> Result<String, String> {
        // Prefer the native Apple Intelligence bridge if it has been registered.
        let full_prompt = build_prompt(system, prompt, Some(max_tokens), Some(temperature));
        if let Some(resp) = apple_intelligence::call(&full_prompt).await {
            return Ok(resp.trim().to_string());
        }

        // Legacy Ollama fallback (only reached when the bridge is not loaded).
        let model = self.resolve_model(_model).await?;
        let url = format!("{}/api/generate", self.base_url);
        let mut body = json!({
            "model": &model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.8,
            }
        });
        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama constrained generate request failed: {}", e))?;
        if !response.status().is_success() {
            return Err(format!(
                "Ollama constrained generate returned {}",
                response.status()
            ));
        }
        let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        data.get("response")
            .and_then(|r| r.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| "Ollama response missing 'response' field".to_string())
    }

    pub async fn generate(
        &self,
        _model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        let full_prompt = build_prompt(system, prompt, None, None);
        if let Some(resp) = apple_intelligence::call(&full_prompt).await {
            return Ok(resp.trim().to_string());
        }

        let model = self.resolve_model(_model).await?;
        let url = format!("{}/api/generate", self.base_url);
        let mut body = json!({
            "model": &model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "num_predict": 512,
                "temperature": 0.1,
                "top_p": 0.5,
            }
        });
        if let Some(sys) = system {
            body["system"] = json!(sys);
        }
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama generate request failed: {}", e))?;
        if !response.status().is_success() {
            return Err(format!("Ollama generate returned {}", response.status()));
        }
        let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        data.get("response")
            .and_then(|r| r.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| "Ollama response missing 'response' field".to_string())
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
            let text = resp.trim().to_string();
            // Apple models may wrap JSON in markdown; strip a leading code fence if present.
            let text = strip_code_fence(&text);
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                return Ok(value);
            }
        }

        let model = self.resolve_model(_model).await?;
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": &model,
            "prompt": prompt,
            "stream": false,
            "format": "json",
            "system": system.unwrap_or(json_system),
            "options": {
                "num_predict": 512,
                "temperature": 0.05,
                "top_p": 0.5,
            }
        });
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama structured generate request failed: {}", e))?;
        if !response.status().is_success() {
            return Err(format!(
                "Ollama structured generate returned {}",
                response.status()
            ));
        }
        let data: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        let text = data
            .get("response")
            .and_then(|r| r.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| "Ollama response missing 'response' field".to_string())?;
        serde_json::from_str(&text).map_err(|e| {
            format!(
                "Failed to parse Ollama output as JSON: {}\nRaw: {}",
                e, text
            )
        })
    }

    pub async fn is_available(&self) -> bool {
        if apple_intelligence::is_available() {
            return true;
        }
        self.list_models()
            .await
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Resolve a requested model to an actually-installed local model.
    ///
    /// If `preferred` is present in `ollama list`, it is used verbatim. Otherwise,
    /// a small preference list of common general-purpose models is tried, and the
    /// first match is returned. If none match, the first available local model is
    /// used as a last resort.
    pub async fn resolve_model(&self, preferred: &str) -> Result<String, String> {
        let available = self.list_models().await?;
        if available.is_empty() {
            return Err("Ollama has no local models".to_string());
        }

        fn base_name(s: &str) -> &str {
            s.split(':').next().unwrap_or(s)
        }

        if let Some(m) = available
            .iter()
            .find(|m| m.as_str() == preferred || base_name(m) == preferred)
        {
            return Ok(m.clone());
        }

        const PREFERRED: &[&str] = &[
            "llama3", "llama3.1", "llama3.2", "mistral", "mixtral", "phi3", "gemma2", "gemma",
        ];
        for candidate in PREFERRED {
            if let Some(m) = available.iter().find(|m| {
                base_name(m).starts_with(candidate) && base_name(m).len() >= candidate.len()
            }) {
                return Ok(m.clone());
            }
        }

        available
            .first()
            .cloned()
            .ok_or_else(|| "Ollama has no local models".to_string())
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
        // Lower temperature -> more deterministic.
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

#[derive(Serialize)]
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
