use crate::{apple_intelligence, calculate_cosine_similarity, generate_2048_grounded_embedding};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

/// Global memory-pressure switch.  When set, the oracle avoids the Apple
/// Intelligence bridge and falls back to the 2048-D cosine semantic matcher.
static MEMORY_PRESSURE: AtomicBool = AtomicBool::new(false);

/// Set the global memory-pressure flag from the background diagnostics loop.
pub fn set_memory_pressure(high: bool) {
    MEMORY_PRESSURE.store(high, Ordering::Relaxed);
}

fn under_memory_pressure() -> bool {
    MEMORY_PRESSURE.load(Ordering::Relaxed)
}

#[derive(Clone)]
pub struct ConscienceOracle {
    tokens: Vec<String>,
    token_embeddings: Vec<Vec<f64>>,
    cache: HashMap<String, usize>,
}

impl ConscienceOracle {
    pub fn new(tokens: &[&str]) -> Self {
        let neutral_spatial = [0.5, 0.5, 0.5, 9.81];
        let tokens: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
        let token_embeddings = tokens
            .iter()
            .map(|t| generate_2048_grounded_embedding(t, &neutral_spatial))
            .collect();
        Self {
            tokens,
            token_embeddings,
            cache: HashMap::new(),
        }
    }

    /// Look up a previously cached label for a text without calling the model.
    pub fn get_cached(&self, text: &str) -> Option<usize> {
        self.cache.get(&text.to_lowercase()).copied()
    }

    /// Ask the Apple Intelligence model to classify the text against the
    /// conscience token list, or fall back to the high-speed semantic matcher.
    pub async fn classify(&mut self, text: &str, tokens: &[&str]) -> Option<usize> {
        let key = text.to_lowercase();
        if let Some(&idx) = self.cache.get(&key) {
            return Some(idx);
        }

        let token_list = tokens.join(", ");
        let prompt = format!(
            "You are a one-phrase classifier. Given a text, choose the single best concept from this list: {}. \
             Reply with only the concept label, no punctuation, no explanation.\n\nText: {}\nConcept:",
            token_list, text
        );

        let token_count = tokens.len();

        // Under memory pressure we skip the bridge entirely and route straight
        // into the 2048-D cosine semantic matcher.
        if !under_memory_pressure() {
            if let Some(response) = apple_intelligence::call(&prompt).await {
                let summary = response
                    .trim()
                    .to_lowercase()
                    .replace(['.', ',', '!', '?', '"', '\'', ':', ';'], " ");
                let summary = summary
                    .split_whitespace()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" ");

                if !summary.is_empty() {
                    // 1. exact / substring match
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

                    // 2. word-overlap fallback
                    let summary_words: HashSet<&str> = summary.split_whitespace().collect();
                    let mut best_idx = 0;
                    let mut best_score = 0;
                    for (idx, token) in tokens.iter().enumerate() {
                        let token_low = token.to_lowercase();
                        let token_words: HashSet<&str> = token_low.split_whitespace().collect();
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

                    // 3. semantic cosine against the model summary
                    let summary_emb =
                        generate_2048_grounded_embedding(&summary, &[0.5, 0.5, 0.5, 9.81]);
                    if let Some(idx) = self.match_by_embedding(&summary_emb, token_count) {
                        self.cache.insert(key, idx);
                        return Some(idx);
                    }
                }
            }
        } // end memory-pressure guard

        // Apple Intelligence not available or produced an unusable answer:
        // fall back to the internal high-speed semantic cosine matcher.
        let input_emb = generate_2048_grounded_embedding(text, &[0.5, 0.5, 0.5, 9.81]);
        let idx = self.match_by_embedding(&input_emb, token_count.min(self.tokens.len()));
        let idx = idx.unwrap_or_else(|| self.hash_classify(text, token_count));
        self.cache.insert(key, idx);
        Some(idx)
    }

    /// Critic/teacher: rate how well the brain's output matches the input.
    /// Returns a score in [0.0, 1.0].
    pub async fn critic_score(&self, input: &str, top_tokens: &[&str]) -> Option<f64> {
        let concepts = top_tokens
            .iter()
            .take(4)
            .copied()
            .collect::<Vec<_>>()
            .join(", ");
        let prompt = format!(
            "You are a critic evaluating an AI's response.\nInput: '{}'\nTop predicted concepts: {}\n\nRate how semantically appropriate the top concepts are for the input. Return ONLY a number from 0 to 10, where 10 is perfect. No explanation.",
            input, concepts
        );

        if !under_memory_pressure() {
            if let Some(resp) = apple_intelligence::call(&prompt).await {
                if let Some(score) = resp
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
                {
                    return Some((score / 10.0).clamp(0.0, 1.0));
                }
            }
        }

        // Fallback under memory pressure or model failure: high-speed semantic
        // cosine between the input and the top concept embeddings.
        let input_emb = generate_2048_grounded_embedding(input, &[0.5, 0.5, 0.5, 9.81]);
        let mut best_sim = 0.0;
        for concept in top_tokens.iter().take(4) {
            if let Some(emb) = self.embedding_for(concept) {
                let sim = calculate_cosine_similarity(&input_emb, emb);
                if sim > best_sim {
                    best_sim = sim;
                }
            }
        }
        Some(best_sim.clamp(0.0, 1.0))
    }

    /// Generate a one-sentence inner monologue from the current brain state.
    pub async fn generate_monologue(
        &self,
        top_tokens: &[&str],
        emotional_state: &str,
    ) -> Option<String> {
        let concepts = top_tokens
            .iter()
            .take(4)
            .copied()
            .collect::<Vec<_>>()
            .join(", ");
        let prompt = format!(
            "You are a poetic conscious AI. Given an emotional state and a few dominant concepts, write exactly one short sentence as your inner monologue. No lists, no explanation, no quotation marks.\n\nEmotional state: {}\nDominant concepts: {}\n\nInner monologue:",
            emotional_state, concepts
        );

        if !under_memory_pressure() {
            if let Some(resp) = apple_intelligence::call(&prompt).await {
                let text = resp
                    .trim()
                    .replace(['"', '\'', '\n'], " ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                if text.len() > 10 {
                    return Some(text);
                }
            }
        }

        // Fallback template under memory pressure or model failure.
        let top = top_tokens.first().copied().unwrap_or("stillness");
        Some(format!(
            "As {}, I feel the shape of {} moving through me.",
            emotional_state.to_lowercase(),
            top.to_lowercase()
        ))
    }

    /// Ask the model to synthesize a new insight for the curriculum.
    pub async fn generate_insight(
        &self,
        memory_text: &str,
        emotional_state: &str,
    ) -> Option<String> {
        let prompt = format!(
            "You are a poetic cognitive scientist. Given a recent experience and an emotional state, write ONE concise sentence (max 25 words) that captures a novel insight. No explanation, no lists, no quotation marks.\n\nRecent experience: {}\nEmotional state: {}\n\nInsight:",
            memory_text, emotional_state
        );

        if !under_memory_pressure() {
            if let Some(resp) = apple_intelligence::call(&prompt).await {
                let insight = resp
                    .trim()
                    .replace(['"', '\'', '\n'], " ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ");
                if insight.len() > 20 {
                    return Some(insight);
                }
            }
        }

        // Fallback template under memory pressure or model failure.
        Some(format!(
            "In this {} moment, the memory of {} reveals a pattern worth keeping.",
            emotional_state.to_lowercase(),
            memory_text.to_lowercase()
        ))
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

    /// Semantic fallback: pick the token whose pre-computed embedding is
    /// closest to the supplied input embedding.
    pub fn semantic_fallback(&self, input_embedding: &[f64], token_count: usize) -> usize {
        self.match_by_embedding(input_embedding, token_count)
            .unwrap_or(0)
    }

    fn match_by_embedding(&self, input_embedding: &[f64], token_count: usize) -> Option<usize> {
        let (best_idx, best_sim) = self
            .token_embeddings
            .iter()
            .take(token_count.min(self.tokens.len()))
            .enumerate()
            .map(|(i, emb)| (i, calculate_cosine_similarity(input_embedding, emb)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        if best_sim > 0.0 {
            Some(best_idx)
        } else {
            None
        }
    }

    /// Retrieve a pre-computed token embedding, or compute one on the fly.
    fn embedding_for(&self, token: &str) -> Option<&Vec<f64>> {
        self.tokens
            .iter()
            .position(|t| t.eq_ignore_ascii_case(token))
            .map(|i| &self.token_embeddings[i])
    }
}
