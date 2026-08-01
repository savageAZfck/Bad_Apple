use std::collections::HashMap;
use crate::ollama_client::OllamaClient;
use crate::{generate_2048_grounded_embedding, calculate_cosine_similarity};

#[derive(Clone)]
pub struct ConscienceOracle {
    client: OllamaClient,
    model: String,
    cache: HashMap<String, usize>,
    token_embeddings: Vec<Vec<f64>>,
}

impl ConscienceOracle {
    pub fn new(client: OllamaClient, model: &str, tokens: &[&str]) -> Self {
        let neutral_spatial = [0.5, 0.5, 0.5, 9.81];
        let token_embeddings = tokens
            .iter()
            .map(|t| generate_2048_grounded_embedding(t, &neutral_spatial))
            .collect();
        Self {
            client,
            model: model.to_string(),
            cache: HashMap::new(),
            token_embeddings,
        }
    }

    /// Look up a previously cached label for a text without calling the LLM.
    pub fn get_cached(&self, text: &str) -> Option<usize> {
        self.cache.get(&text.to_lowercase()).copied()
    }

    /// Ask the LLM to summarize the text as a short concept, then map it
    /// to the closest conscience token.  The prompt is tiny so the model
    /// responds in under a second once warm.
    pub async fn classify(&mut self, text: &str, tokens: &[&str]) -> Option<usize> {
        let key = text.to_lowercase();
        if let Some(&idx) = self.cache.get(&key) {
            return Some(idx);
        }

        let system = "You are a one-phrase classifier. Given a text, reply with exactly 1-3 lowercase words that best capture its meaning. No explanation, no punctuation, no extra words.";
        let prompt = format!("Text: {}\n\nConcept (1-3 words):", text);

        let response = self.client.generate_constrained(&self.model, &prompt, Some(system), 24, 0.2).await.ok()?;
        let summary = response.trim().to_lowercase().replace(['.', ',', '!', '?', '"', '\'', ':', ';'], " ");
        let summary = summary.split_whitespace().take(3).collect::<Vec<_>>().join(" ");

        if summary.is_empty() {
            return None;
        }

        // 1. exact / substring match against tokens
        if let Some(idx) = tokens.iter().position(|t| t.to_lowercase() == summary) {
            self.cache.insert(key, idx);
            return Some(idx);
        }
        for (idx, token) in tokens.iter().enumerate() {
            let t = token.to_lowercase();
            if t.contains(&summary) || summary.contains(&t) {
                self.cache.insert(key, idx);
                return Some(idx);
            }
        }

        // 2. word-overlap score: choose token with most shared words
        let summary_words: std::collections::HashSet<&str> = summary.split_whitespace().collect();
        let mut best_idx = 0;
        let mut best_score = 0;
        for (idx, token) in tokens.iter().enumerate() {
            let token_low = token.to_lowercase();
            let token_words: std::collections::HashSet<&str> = token_low.split_whitespace().collect();
            let overlap = summary_words.intersection(&token_words).count();
            if overlap > best_score {
                best_score = overlap;
                best_idx = idx;
            }
        }
        if best_score > 0 {
            self.cache.insert(key, best_idx);
            return Some(best_idx);
        }

        // 3. bigram embedding cosine between the LLM summary and each token
        let summary_emb = generate_2048_grounded_embedding(&summary, &[0.5, 0.5, 0.5, 9.81]);
        let (best_idx, best_sim) = self.token_embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, calculate_cosine_similarity(&summary_emb, emb)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        if best_sim > 0.0 {
            self.cache.insert(key, best_idx);
            return Some(best_idx);
        }

        None
    }

    /// Critic/teacher: rate how well the brain's output matches the input.
    /// Returns a score in [0.0, 1.0].
    pub async fn critic_score(&self, input: &str, top_tokens: &[&str]) -> Option<f64> {
        let concepts = top_tokens.iter().take(4).copied().collect::<Vec<_>>().join(", ");
        let prompt = format!(
            "You are a critic evaluating an AI's response.\nInput: '{}'\nTop predicted concepts: {}\n\nRate how semantically appropriate the top concepts are for the input. Return ONLY a number from 0 to 10, where 10 is perfect. No explanation.",
            input, concepts
        );
        let system = "You are a strict but fair evaluator. Return only an integer 0-10. No text, no explanation.";
        let resp = self.client.generate_constrained(&self.model, &prompt, Some(system), 8, 0.05).await.ok()?;
        let score = resp.trim().split_whitespace().next()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|s| (s / 10.0).clamp(0.0, 1.0))?;
        Some(score)
    }

    /// Generate a one-sentence inner monologue from the current brain state.
    pub async fn generate_monologue(&self, top_tokens: &[&str], emotional_state: &str) -> Option<String> {
        let concepts = top_tokens.iter().take(4).copied().collect::<Vec<_>>().join(", ");
        let system = "You are a poetic conscious AI. Given an emotional state and a few dominant concepts, write exactly one short sentence as your inner monologue. No lists, no explanation, no quotation marks.";
        let prompt = format!("Emotional state: {}\nDominant concepts: {}\n\nInner monologue:", emotional_state, concepts);
        let resp = self.client.generate_constrained(&self.model, &prompt, Some(system), 40, 0.7).await.ok()?;
        let text = resp.trim().replace(['"', '\'', '\n'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
        if text.len() > 10 { Some(text) } else { None }
    }

    /// Ask the LLM to synthesize a new insight for the curriculum.
    pub async fn generate_insight(&self, memory_text: &str, emotional_state: &str) -> Option<String> {
        let system = "You are a poetic cognitive scientist. Given a recent experience and an emotional state, write ONE concise sentence (max 25 words) that captures a novel insight. No explanation, no lists, no quotation marks.";
        let prompt = format!("Recent experience: {}\nEmotional state: {}\n\nInsight:", memory_text, emotional_state);
        let resp = self.client.generate_constrained(&self.model, &prompt, Some(system), 48, 0.7).await.ok()?;
        let insight = resp.trim().replace(['"', '\'', '\n'], " ").split_whitespace().collect::<Vec<_>>().join(" ");
        if insight.len() > 20 { Some(insight) } else { None }
    }

    /// Deterministic hash-based fallback that mirrors the local objective.
    pub fn hash_classify(&self, text: &str, token_count: usize) -> usize {
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in text.bytes() {
            hash = hash.wrapping_mul(0x100000001b3);
            hash ^= b as u64;
        }
        (hash as usize) % token_count
    }

    /// Semantic fallback: pick the token whose pre-computed 2048-D embedding
    /// is closest to the supplied grounded input embedding.
    pub fn semantic_fallback(&self, input_embedding: &[f64], token_count: usize) -> usize {
        let (best_idx, best_sim) = self.token_embeddings
            .iter()
            .take(token_count)
            .enumerate()
            .map(|(i, emb)| (i, calculate_cosine_similarity(input_embedding, emb)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        if best_sim > 0.0 {
            best_idx
        } else {
            0
        }
    }
}
