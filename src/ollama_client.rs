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
        model: &str,
        prompt: &str,
        system: Option<&str>,
        max_tokens: i32,
        temperature: f32,
    ) -> Result<String, String> {
        let url = format!("{}/api/generate", self.base_url);
        let mut body = json!({
            "model": model,
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
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        let url = format!("{}/api/generate", self.base_url);
        let mut body = json!({
            "model": model,
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
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let url = format!("{}/api/generate", self.base_url);
        let json_system = "You must output only valid JSON. Do not add markdown, explanations, or any text outside the JSON object.";
        let mut body = json!({
            "model": model,
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
        if let Some(sys) = system {
            body["system"] = json!(format!("{} {}", json_system, sys));
        }
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
        self.list_models().await.is_ok()
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
            "You are a planning AI agent. Given the following state, return ONLY a valid JSON array of at most 3 concrete sub-goal strings.\n\nExample output (do not deviate from this format):\n[\"explore memory topology\", \"integrate new insight\", \"optimize learning rate\"]\n\nCurrent input: '{}'\nActive goals: {:?}\nEmotional state: {}\nMemory count: {}\nSensor summary: {}\n\nReturn only the JSON array. No explanation, no markdown, no objects, no keys, no repetition.",
            self.input, self.active_goals, self.emotional_state, self.memory_count, self.sensor_summary
        )
    }
}
