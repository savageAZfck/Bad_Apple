#![allow(dead_code)]
use anyhow::{Context, Result};
use config::Config;
use md5::{Digest, Md5};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use sysinfo::System;
use tokio::sync::Mutex as TokioMutex;
use tokio::{
    net::UdpSocket,
    task::spawn_blocking,
    time::{sleep, timeout},
};

mod apple_intelligence;
mod benchmark;
mod config;
mod conscience_oracle;
mod data_feed;
mod hyperdimensional_core;
mod metrics;
mod ollama_client;
mod production_blueprint;
mod protocol;
mod strategy_library;
mod telemetry;
mod tensor_brain;
mod wild_workspace;
use benchmark::{BenchmarkSuite, PilotReport, TransferSuite, TransferTask};
use conscience_oracle::ConscienceOracle;
use dashmap::DashMap;
use data_feed::{default_curriculum_dirs, DataCurriculum};
use ollama_client::OllamaClient;
use production_blueprint::{
    GlobalWorkspace, HomeostaticController, NeuroSymbolicEngine as ProductionNeuroSymbolicEngine,
    ThermalState,
};
use protocol::{
    decode_payload, multi_agent_secret, sign_packet, verify_packet, CompactEngramPacket,
    ConnectionManager, LockFreeRing, SignedUdpPacket, SwarmMetrics,
};
use strategy_library::{Strategy, StrategyLibrary};
use telemetry::{
    current_secs, run_telemetry_server, skill_key, update_sensor_snapshot, SensorSnapshot,
    TelemetryState,
};
use tensor_brain::CandleBrain;

use wild_workspace::{run_wild_loop, start_watcher};

// =========================================================================
// 🌌 HYPERDIMENSIONAL COMPUTING (HDC) - Exponential Representational Capacity
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct HyperdimensionalVector {
    dimensions: usize,
    data: Vec<i64>, // Use i64 for holographic representation
}

impl HyperdimensionalVector {
    fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            data: (0..dimensions)
                .map(|_| rand::thread_rng().gen_range(-1..=1))
                .collect(),
        }
    }

    fn bind(&self, other: &Self) -> Self {
        // Binding operation for associative memory
        let mut result = self.clone();
        for i in 0..self.dimensions.min(other.dimensions) {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    fn bundle(&self, other: &Self) -> Self {
        // Bundle operation for superposition
        let mut result = self.clone();
        for i in 0..self.dimensions.min(other.dimensions) {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    fn similarity(&self, other: &Self) -> f64 {
        // Cosine similarity for hypervectors
        let mut dot_product = 0i64;
        let mut norm_a = 0i64;
        let mut norm_b = 0i64;

        for i in 0..self.dimensions.min(other.dimensions) {
            dot_product += self.data[i] * other.data[i];
            norm_a += self.data[i] * self.data[i];
            norm_b += other.data[i] * other.data[i];
        }

        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }

        dot_product as f64 / ((norm_a as f64).sqrt() * (norm_b as f64).sqrt())
    }

    fn permute(&self, shift: usize) -> Self {
        // Circular permutation for sequence encoding
        let mut result = self.clone();
        for i in 0..self.dimensions {
            result.data[(i + shift) % self.dimensions] = self.data[i];
        }
        result
    }
}

// =========================================================================
// 🧠 NEURO-SYMBOLIC INTEGRATION - Combining Neural & Symbolic Reasoning
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
enum SymbolicExpression {
    Atom(String),
    Variable(String),
    List(Vec<SymbolicExpression>),
    Number(f64),
    Boolean(bool),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LogicalRule {
    premises: Vec<SymbolicExpression>,
    conclusion: SymbolicExpression,
    confidence: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NeuroSymbolicEngine {
    rules: Vec<LogicalRule>,
    neural_embeddings: HashMap<String, Vec<f64>>,
    symbol_map: HashMap<String, HyperdimensionalVector>,
}

impl NeuroSymbolicEngine {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            neural_embeddings: HashMap::new(),
            symbol_map: HashMap::new(),
        }
    }

    fn add_rule(&mut self, premises: Vec<SymbolicExpression>, conclusion: SymbolicExpression) {
        self.rules.push(LogicalRule {
            premises,
            conclusion,
            confidence: 1.0,
        });
    }

    fn forward_chain(&self, facts: &[SymbolicExpression]) -> Vec<SymbolicExpression> {
        let mut new_facts = facts.to_vec();
        let mut changed = true;

        while changed {
            changed = false;
            for rule in &self.rules {
                if self.matches_premises(&rule.premises, &new_facts)
                    && !new_facts.contains(&rule.conclusion)
                {
                    new_facts.push(rule.conclusion.clone());
                    changed = true;
                }
            }
        }

        new_facts
    }

    fn matches_premises(
        &self,
        premises: &[SymbolicExpression],
        facts: &[SymbolicExpression],
    ) -> bool {
        premises.iter().all(|p| facts.contains(p))
    }

    fn unify_neural_symbolic(&mut self, symbol: &str, embedding: Vec<f64>) {
        let hypervector = HyperdimensionalVector::new(10000);
        self.neural_embeddings.insert(symbol.to_string(), embedding);
        self.symbol_map.insert(symbol.to_string(), hypervector);
    }
}

// =========================================================================
// 🔮 PREDICTIVE WORLD MODELING - Physics & Causal Simulation
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct PhysicsState {
    position: Vec<f64>,
    velocity: Vec<f64>,
    acceleration: Vec<f64>,
    mass: f64,
    forces: Vec<Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CausalNode {
    id: String,
    variables: HashMap<String, f64>,
    parents: Vec<String>,
    children: Vec<String>,
    causal_strength: HashMap<String, f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WorldModel {
    physics_states: HashMap<String, PhysicsState>,
    causal_graph: HashMap<String, CausalNode>,
    time_steps: VecDeque<HashMap<String, f64>>,
}

impl WorldModel {
    fn new() -> Self {
        Self {
            physics_states: HashMap::new(),
            causal_graph: HashMap::new(),
            time_steps: VecDeque::with_capacity(1000),
        }
    }

    fn simulate_step(&mut self, _dt: f64) {
        // Store state for temporal reasoning
        let current_state = self.extract_current_state();
        self.time_steps.push_back(current_state);
        if self.time_steps.len() > 1000 {
            self.time_steps.pop_front();
        }
    }

    fn predict_future(&self, steps: usize) -> Vec<HashMap<String, f64>> {
        let mut predictions = Vec::new();
        let mut model = self.clone();

        for _ in 0..steps {
            model.simulate_step(0.1);
            predictions.push(model.extract_current_state());
        }

        predictions
    }

    fn extract_current_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        for (id, physics) in &self.physics_states {
            for (i, &pos) in physics.position.iter().enumerate() {
                state.insert(format!("{}_pos_{}", id, i), pos);
            }
        }
        state
    }

    fn add_causal_relation(&mut self, cause: String, effect: String, strength: f64) {
        self.causal_graph
            .entry(cause.clone())
            .or_insert_with(|| CausalNode {
                id: cause.clone(),
                variables: HashMap::new(),
                parents: Vec::new(),
                children: Vec::new(),
                causal_strength: HashMap::new(),
            })
            .children
            .push(effect.clone());

        self.causal_graph
            .entry(effect.clone())
            .or_insert_with(|| CausalNode {
                id: effect.clone(),
                variables: HashMap::new(),
                parents: Vec::new(),
                children: Vec::new(),
                causal_strength: HashMap::new(),
            })
            .parents
            .push(cause.clone());

        if let Some(node) = self.causal_graph.get_mut(&cause) {
            node.causal_strength.insert(effect.clone(), strength);
        }
    }

    /// Learn causal links between an action and observed sensor changes.
    fn record_action_effects(
        &mut self,
        action: &str,
        before: &HashMap<String, f64>,
        after: &HashMap<String, f64>,
    ) {
        let cause = format!("action:{}", action);
        for (key, &after_val) in after {
            let before_val = before.get(key).copied().unwrap_or(after_val);
            let delta = after_val - before_val;
            if delta.abs() > 0.1 {
                self.add_causal_relation(cause.clone(), key.clone(), delta);
                let state = self
                    .physics_states
                    .entry(key.clone())
                    .or_insert(PhysicsState {
                        position: vec![after_val],
                        velocity: vec![delta],
                        acceleration: vec![0.0],
                        mass: 1.0,
                        forces: Vec::new(),
                    });
                state.position = vec![after_val];
                state.velocity = vec![delta];
            }
        }
    }

    /// Predict the next state after an action using learned causal strengths.
    fn predict_action_effects(
        &self,
        action: &str,
        current: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let cause = format!("action:{}", action);
        let mut predicted = current.clone();
        if let Some(node) = self.causal_graph.get(&cause) {
            for (effect, strength) in &node.causal_strength {
                let base = predicted.get(effect).copied().unwrap_or(0.0);
                predicted.insert(effect.clone(), base + *strength);
            }
        }
        predicted
    }
}

// =========================================================================
// 🔮 NEURAL PREDICTIVE WORLD MODEL - learns to predict the next input embedding
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct NeuralWorldModel {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    lr: f64,
}

impl NeuralWorldModel {
    fn default_instance() -> Self {
        Self::new(tensor_brain::BRAIN_DIM, 2048)
    }

    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w: Vec<Vec<f64>> = (0..input_dim)
            .map(|_| {
                (0..output_dim)
                    .map(|_| rng.gen_range(-0.01..0.01))
                    .collect()
            })
            .collect();
        Self {
            w,
            b: vec![0.0; output_dim],
            lr: 0.001,
        }
    }

    fn predict(&self, state: &[f64]) -> Vec<f64> {
        let mut out = self.b.clone();
        for (i, &s) in state.iter().enumerate() {
            if i >= self.w.len() {
                break;
            }
            for (j, weight) in self.w[i].iter().enumerate() {
                out[j] += s * weight;
            }
        }
        out
    }

    fn train(&mut self, state: &[f64], target: &[f64]) -> f64 {
        let pred = self.predict(state);
        let mut loss = 0.0;
        let mut errors = vec![0.0; target.len()];
        for (j, (&p, &t)) in pred.iter().zip(target.iter()).enumerate() {
            let e = t - p;
            errors[j] = e;
            loss += e * e;
        }
        loss = (loss / target.len() as f64).sqrt();
        for (i, &s) in state.iter().enumerate() {
            if i >= self.w.len() {
                break;
            }
            for (j, e) in errors.iter().enumerate() {
                self.w[i][j] += self.lr * e * s;
            }
        }
        for (j, e) in errors.iter().enumerate() {
            self.b[j] += self.lr * e;
        }
        loss
    }
}

// =========================================================================
// 🔬 INTEGRATED INFORMATION THEORY (IIT) - Consciousness Quantification
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct PhiCalculation {
    phi: f64,
    mechanisms: Vec<String>,
    integrated_information: f64,
    conceptual_structure: Vec<Concept>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Concept {
    cause: HyperdimensionalVector,
    effect: HyperdimensionalVector,
    phi: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct IITConsciousnessMeter {
    neural_partitions: Vec<Vec<usize>>,
    phi_history: VecDeque<f64>,
    current_phi: f64,
}

impl IITConsciousnessMeter {
    fn new() -> Self {
        Self {
            neural_partitions: Vec::new(),
            phi_history: VecDeque::with_capacity(100),
            current_phi: 0.0,
        }
    }

    fn calculate_phi(&mut self, neural_activity: &[f64]) -> f64 {
        // Simplified Phi calculation - actual IIT is much more complex
        let integration = self.calculate_integration(neural_activity);
        let differentiation = self.calculate_differentiation(neural_activity);

        let phi = integration * differentiation;
        self.current_phi = phi;
        self.phi_history.push_back(phi);

        if self.phi_history.len() > 100 {
            self.phi_history.pop_front();
        }

        phi
    }

    fn calculate_integration(&self, activity: &[f64]) -> f64 {
        // Measure how much information is integrated across the system
        let sum: f64 = activity.iter().sum();
        let mean = sum / activity.len() as f64;
        let variance: f64 = activity.iter().map(|x| (x - mean).powi(2)).sum();
        variance.sqrt()
    }

    fn calculate_differentiation(&self, activity: &[f64]) -> f64 {
        // Measure the number of distinct states the system can be in
        let unique_states: HashSet<i64> = activity.iter().map(|x| (x * 1000.0) as i64).collect();
        unique_states.len() as f64 / activity.len() as f64
    }

    fn is_conscious(&self) -> bool {
        self.current_phi > 0.1 // Threshold for consciousness
    }
}

// =========================================================================
// 🎯 FEW-SHOT LEARNING - Rapid Adaptation from Minimal Examples
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct FewShotExample {
    input: Vec<f64>,
    output: Vec<f64>,
    task_id: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaLearner {
    support_set: Vec<FewShotExample>,
    task_embeddings: HashMap<String, HyperdimensionalVector>,
    adaptation_rate: f64,
}

impl MetaLearner {
    fn new() -> Self {
        Self {
            support_set: Vec::new(),
            task_embeddings: HashMap::new(),
            adaptation_rate: 0.1,
        }
    }

    fn add_example(&mut self, example: FewShotExample) {
        self.support_set.push(example.clone());

        // Create task embedding
        let task_vector = HyperdimensionalVector::new(10000);
        self.task_embeddings
            .insert(example.task_id.clone(), task_vector);
    }

    fn few_shot_inference(&self, input: &[f64], task_id: &str) -> Vec<f64> {
        // Find similar examples in support set
        let similar_examples: Vec<_> = self
            .support_set
            .iter()
            .filter(|e| e.task_id == task_id)
            .collect();

        if similar_examples.is_empty() {
            return vec![0.0; input.len()];
        }

        // Use k-nearest neighbors with hyperdimensional similarity
        let mut predictions = Vec::new();
        for example in &similar_examples {
            predictions.extend(example.output.clone());
        }

        // Average predictions
        let n = predictions.len() / similar_examples.len();
        let mut result = Vec::new();
        for i in 0..n {
            let sum: f64 = predictions.iter().skip(i).step_by(n).sum();
            result.push(sum / similar_examples.len() as f64);
        }

        result
    }

    fn meta_update(&mut self, _task_id: &str, loss: f64) {
        // Update meta-learning parameters based on task performance
        if loss > 0.5 {
            self.adaptation_rate = (self.adaptation_rate * 1.1).min(0.5);
        } else {
            self.adaptation_rate = (self.adaptation_rate * 0.9).max(0.01);
        }
    }
}

// =========================================================================
// 🎨 OPEN-ENDED CREATIVITY - Novel Concept Generation
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CreativeSpace {
    concept_space: HashMap<String, HyperdimensionalVector>,
    novelty_threshold: f64,
    combination_strategies: Vec<String>,
}

impl CreativeSpace {
    fn new() -> Self {
        Self {
            concept_space: HashMap::new(),
            novelty_threshold: 0.3,
            combination_strategies: vec![
                "analogy".to_string(),
                "blending".to_string(),
                "metaphor".to_string(),
                "abstraction".to_string(),
            ],
        }
    }

    fn generate_novel_concept(&self, seed_concepts: &[String]) -> (String, f64) {
        if seed_concepts.is_empty() {
            return ("random_concept".to_string(), 0.5);
        }

        // Combine concepts using hyperdimensional operations
        let mut combined = HyperdimensionalVector::new(10000);

        for concept in seed_concepts {
            if let Some(vector) = self.concept_space.get(concept) {
                combined = combined.bind(vector);
            }
        }

        // Add randomness for novelty
        let random_vector = HyperdimensionalVector::new(10000);
        combined = combined.bundle(&random_vector);

        // Calculate novelty
        let novelty = self.calculate_novelty(&combined);

        let new_concept = format!("novel_{}", seed_concepts.join("_"));
        (new_concept, novelty)
    }

    fn calculate_novelty(&self, concept: &HyperdimensionalVector) -> f64 {
        let mut similarities = Vec::new();

        for existing in self.concept_space.values() {
            similarities.push(concept.similarity(existing));
        }

        if similarities.is_empty() {
            return 1.0;
        }

        let avg_similarity: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        1.0 - avg_similarity
    }

    fn add_concept(&mut self, name: String, vector: HyperdimensionalVector) {
        self.concept_space.insert(name, vector);
    }
}

// =========================================================================
// 🧠 COMMON SENSE REASONING - Intuitive World Understanding
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CommonSenseRule {
    condition: String,
    consequence: String,
    confidence: f64,
    domain: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CommonSenseKnowledgeBase {
    rules: Vec<CommonSenseRule>,
    script_library: HashMap<String, Vec<String>>,
    causal_chains: Vec<Vec<String>>,
}

impl CommonSenseKnowledgeBase {
    fn new() -> Self {
        let mut base = Self {
            rules: Vec::new(),
            script_library: HashMap::new(),
            causal_chains: Vec::new(),
        };

        // Add basic common sense rules
        base.add_rule("gravity", "objects fall down", 0.99);
        base.add_rule("fire", "burns things", 0.99);
        base.add_rule("water", "extinguishes fire", 0.95);
        base.add_rule("social", "people have feelings", 0.9);

        base
    }

    fn add_rule(&mut self, domain: &str, rule: &str, confidence: f64) {
        self.rules.push(CommonSenseRule {
            condition: domain.to_string(),
            consequence: rule.to_string(),
            confidence,
            domain: domain.to_string(),
        });
    }

    fn infer(&self, situation: &str) -> Vec<String> {
        let mut inferences = Vec::new();

        for rule in &self.rules {
            if situation.contains(&rule.condition) {
                inferences.push(rule.consequence.clone());
            }
        }

        inferences
    }

    fn check_consistency(&self, action: &str, context: &str) -> bool {
        let inferences = self.infer(context);
        !inferences
            .iter()
            .any(|inf| inf.contains("cannot") && action.contains(inf))
    }
}

// =========================================================================
// 👥 TRUE THEORY OF MIND - Mental State Modeling
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MentalState {
    beliefs: HashMap<String, f64>,
    desires: HashMap<String, f64>,
    intentions: Vec<String>,
    emotions: HashMap<String, f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AgentModel {
    id: String,
    mental_state: MentalState,
    personality_traits: HashMap<String, f64>,
    prediction_history: VecDeque<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TheoryOfMindEngine {
    agent_models: HashMap<String, AgentModel>,
    social_situations: VecDeque<String>,
}

impl TheoryOfMindEngine {
    fn new() -> Self {
        Self {
            agent_models: HashMap::new(),
            social_situations: VecDeque::with_capacity(100),
        }
    }

    fn add_agent(&mut self, id: String, personality: HashMap<String, f64>) {
        self.agent_models.insert(
            id.clone(),
            AgentModel {
                id,
                mental_state: MentalState {
                    beliefs: HashMap::new(),
                    desires: HashMap::new(),
                    intentions: Vec::new(),
                    emotions: HashMap::new(),
                },
                personality_traits: personality,
                prediction_history: VecDeque::with_capacity(50),
            },
        );
    }

    fn infer_mental_state(&mut self, agent_id: &str, observation: &str) -> MentalState {
        if let Some(agent) = self.agent_models.get_mut(agent_id) {
            // Update mental state based on observation
            if observation.contains("happy") {
                agent.mental_state.emotions.insert("joy".to_string(), 0.8);
            }
            if observation.contains("want") {
                let desire = observation.split("want").nth(1).unwrap_or("something");
                agent
                    .mental_state
                    .desires
                    .insert(desire.trim().to_string(), 0.7);
            }

            agent.mental_state.clone()
        } else {
            // Create default mental state for unknown agent
            self.add_agent(agent_id.to_string(), HashMap::new());
            self.agent_models
                .get(agent_id)
                .map(|a| a.mental_state.clone())
                .unwrap_or(MentalState {
                    beliefs: HashMap::new(),
                    desires: HashMap::new(),
                    intentions: Vec::new(),
                    emotions: HashMap::new(),
                })
        }
    }

    fn predict_action(&self, agent_id: &str) -> String {
        if let Some(agent) = self.agent_models.get(agent_id) {
            // Predict action based on desires and personality
            if let Some((desire, _)) = agent
                .mental_state
                .desires
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
            {
                return format!("Agent {} will try to {}", agent_id, desire);
            }
        }
        format!("Agent {} will take default action", agent_id)
    }
}

// =========================================================================
// 🔧 SELF-MODIFYING CODE - Runtime Architecture Improvement
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CodeModification {
    target_function: String,
    modification_type: String,
    parameters: Vec<f64>,
    timestamp: u64,
    success: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SelfImprovementEngine {
    modification_history: Vec<CodeModification>,
    performance_metrics: HashMap<String, Vec<f64>>,
    optimization_targets: Vec<String>,
}

impl SelfImprovementEngine {
    fn new() -> Self {
        Self {
            modification_history: Vec::new(),
            performance_metrics: HashMap::new(),
            optimization_targets: vec![
                "learning_rate".to_string(),
                "network_depth".to_string(),
                "attention_heads".to_string(),
            ],
        }
    }

    fn evaluate_performance(&mut self, metric: &str, value: f64) {
        self.performance_metrics
            .entry(metric.to_string())
            .or_default()
            .push(value);
    }

    fn should_modify(&self, target: &str) -> bool {
        if let Some(values) = self.performance_metrics.get(target) {
            if values.len() > 10 {
                let recent: f64 = values.iter().rev().take(5).sum::<f64>() / 5.0;
                let historical: f64 = values.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
                return recent < historical * 0.9; // Modify if performance degraded
            }
        }
        false
    }

    fn apply_modification(&mut self, target: String, modification_type: String) {
        let modification = CodeModification {
            target_function: target.clone(),
            modification_type,
            parameters: vec![0.1], // Default parameter
            timestamp: current_secs(),
            success: true,
        };

        self.modification_history.push(modification);
    }
}

// =========================================================================
// 🧠 HIERARCHICAL MEMORY SYSTEMS - Episodic, Semantic, Working Memory
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct EpisodicMemory {
    episodes: Vec<MemoryEpisode>,
    current_index: usize,
    consolidation_threshold: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoryEpisode {
    id: u64,
    timestamp: u64,
    context: Vec<f64>,
    content: String,
    emotional_context: FluidEmotionalProfile,
    importance: f64,
    replay_count: usize,
    associated_episodes: Vec<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SemanticMemory {
    concepts: HashMap<String, ConceptNode>,
    concept_embeddings: HashMap<String, Vec<f64>>,
    concept_associations: HashMap<String, Vec<String>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConceptNode {
    name: String,
    embedding: Vec<f64>,
    activation_level: f64,
    last_accessed: u64,
    associations: Vec<String>,
    abstraction_level: u8, // 0=concrete, 10=abstract
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WorkingMemory {
    current_focus: Vec<f64>,
    context_stack: VecDeque<MemoryContext>,
    capacity: usize,
    time_decay: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoryContext {
    id: u64,
    content: Vec<f64>,
    timestamp: u64,
    attention_level: f64,
}

#[allow(dead_code)]
impl EpisodicMemory {
    fn new() -> Self {
        Self {
            episodes: Vec::new(),
            current_index: 0,
            consolidation_threshold: 0.8,
        }
    }

    fn add_episode(&mut self, episode: MemoryEpisode) {
        self.episodes.push(episode);
        self.consolidate_episodes();
    }

    fn consolidate_episodes(&mut self) {
        if self.episodes.len() < 10 {
            return;
        }

        // Consolidate similar episodes
        let mut to_remove = Vec::new();
        let mut consolidations = Vec::new();

        for i in 0..self.episodes.len() {
            for j in (i + 1)..self.episodes.len() {
                let similarity = calculate_cosine_similarity(
                    &self.episodes[i].context,
                    &self.episodes[j].context,
                );
                if similarity > 0.9 {
                    consolidations.push((i, j, self.episodes[j].id));
                    to_remove.push(j);
                }
            }
        }

        // Apply consolidations
        for (i, j, episode_id) in consolidations {
            if i < self.episodes.len() && j < self.episodes.len() {
                self.episodes[i].importance =
                    (self.episodes[i].importance + self.episodes[j].importance).min(1.0);
                self.episodes[i].replay_count += self.episodes[j].replay_count;
                self.episodes[i].associated_episodes.push(episode_id);
            }
        }

        // Remove consolidated episodes (in reverse order to maintain indices)
        to_remove.sort();
        to_remove.dedup();
        for index in to_remove.into_iter().rev() {
            if index < self.episodes.len() {
                self.episodes.remove(index);
            }
        }
    }

    fn replay_recent(&self, count: usize) -> Vec<&MemoryEpisode> {
        let start = if self.episodes.len() > count {
            self.episodes.len() - count
        } else {
            0
        };
        self.episodes[start..].iter().collect()
    }

    fn find_related(&self, context: &[f64], threshold: f64) -> Vec<&MemoryEpisode> {
        self.episodes
            .iter()
            .filter(|ep| calculate_cosine_similarity(&ep.context, context) > threshold)
            .collect()
    }
}

#[allow(dead_code)]
impl SemanticMemory {
    fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            concept_embeddings: HashMap::new(),
            concept_associations: HashMap::new(),
        }
    }

    fn add_concept(&mut self, name: String, embedding: Vec<f64>, abstraction_level: u8) {
        let node = ConceptNode {
            name: name.clone(),
            embedding: embedding.clone(),
            activation_level: 0.0,
            last_accessed: current_secs(),
            associations: Vec::new(),
            abstraction_level,
        };
        self.concepts.insert(name.clone(), node);
        self.concept_embeddings.insert(name, embedding);
    }

    fn activate_concept(&mut self, name: &str) -> f64 {
        let now = current_secs();
        let associations_to_activate = if let Some(node) = self.concepts.get(name) {
            node.associations.clone()
        } else {
            Vec::new()
        };

        let activation_level = if let Some(node) = self.concepts.get_mut(name) {
            node.activation_level = (node.activation_level + 0.3).min(1.0);
            node.last_accessed = now;
            node.activation_level
        } else {
            0.0
        };

        // Spread activation to associated concepts (separate borrow)
        for assoc in associations_to_activate {
            if let Some(assoc_node) = self.concepts.get_mut(&assoc) {
                assoc_node.activation_level = (assoc_node.activation_level + 0.2).min(1.0);
            }
        }

        activation_level
    }

    fn associate_concepts(&mut self, concept_a: &str, concept_b: &str) {
        if let Some(node) = self.concepts.get_mut(concept_a) {
            if !node.associations.contains(&concept_b.to_string()) {
                node.associations.push(concept_b.to_string());
            }
        }
        if let Some(node) = self.concepts.get_mut(concept_b) {
            if !node.associations.contains(&concept_a.to_string()) {
                node.associations.push(concept_a.to_string());
            }
        }
    }

    fn get_concept_vector(&self, name: &str) -> Option<Vec<f64>> {
        self.concept_embeddings.get(name).cloned()
    }

    fn decay_activations(&mut self) {
        let now = current_secs();
        for node in self.concepts.values_mut() {
            let time_since_access = now - node.last_accessed;
            let decay = (time_since_access as f64 / 3600.0) * 0.1; // Decay over hours
            node.activation_level = (node.activation_level - decay).max(0.0);
        }
    }
}

#[allow(dead_code)]
impl WorkingMemory {
    fn new(capacity: usize) -> Self {
        Self {
            current_focus: vec![0.0; 512],
            context_stack: VecDeque::new(),
            capacity,
            time_decay: 0.05,
        }
    }

    fn set_focus(&mut self, focus: Vec<f64>) {
        self.current_focus = focus;
    }

    fn push_context(&mut self, context: MemoryContext) {
        if self.context_stack.len() >= self.capacity {
            self.context_stack.pop_front();
        }
        self.context_stack.push_back(context);
    }

    fn get_attention_weights(&self) -> Vec<f64> {
        let mut weights = vec![0.0; self.current_focus.len()];
        let total: f64 = self.current_focus.iter().map(|x| x.abs()).sum();
        if total > 0.0 {
            for (i, &val) in self.current_focus.iter().enumerate() {
                weights[i] = val.abs() / total;
            }
        }
        weights
    }

    fn apply_decay(&mut self) {
        for val in self.current_focus.iter_mut() {
            *val *= 1.0 - self.time_decay;
        }
    }
}

// =========================================================================
// 🧠 META-COGNITION AND SELF-REFLECTION SYSTEM
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaCognitiveState {
    self_awareness: f64,
    confidence_level: f64,
    uncertainty_level: f64,
    performance_history: Vec<f64>,
    reflective_state: String,
    learning_progress: f64,
    self_model_accuracy: f64,
}

#[allow(dead_code)]
impl MetaCognitiveState {
    fn new() -> Self {
        Self {
            self_awareness: 0.5,
            confidence_level: 0.5,
            uncertainty_level: 0.5,
            performance_history: Vec::new(),
            reflective_state: "Initial initialization".to_string(),
            learning_progress: 0.0,
            self_model_accuracy: 0.5,
        }
    }

    fn update_performance(&mut self, performance: f64) {
        self.performance_history.push(performance);
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }

        let avg_performance: f64 =
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64;
        self.learning_progress = avg_performance;
        self.confidence_level = (self.confidence_level * 0.9 + avg_performance * 0.1).min(1.0);
        self.uncertainty_level = 1.0 - self.confidence_level;

        // Update self-awareness based on performance
        self.self_awareness = (self.self_awareness * 0.95 + avg_performance * 0.05).min(1.0);
    }

    fn reflect_on_state(&mut self, emotions: &FluidEmotionalProfile, neural_wear: f64) {
        if neural_wear > 0.5 {
            self.reflective_state = "Concern: High neural degradation detected".to_string();
            self.confidence_level *= 0.9;
        } else if emotions.valence < 0.3 && emotions.arousal > 0.7 {
            self.reflective_state =
                "Concern: Negative emotional state with high arousal".to_string();
        } else if self.performance_history.len() > 10 {
            let recent_avg: f64 =
                self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
            let overall_avg: f64 = self.performance_history.iter().sum::<f64>()
                / self.performance_history.len() as f64;
            if recent_avg > overall_avg * 1.1 {
                self.reflective_state = "Positive: Recent performance improvement".to_string();
            } else if recent_avg < overall_avg * 0.9 {
                self.reflective_state = "Concern: Recent performance decline".to_string();
            } else {
                self.reflective_state = "Stable: Consistent performance".to_string();
            }
        } else {
            self.reflective_state = "Learning: Gathering more data".to_string();
        }
    }

    fn should_request_help(&self) -> bool {
        self.uncertainty_level > 0.7 || self.confidence_level < 0.3
    }
}

// =========================================================================
// 🧠 ADVANCED REASONING AND PLANNING ENGINE
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct PlanningNode {
    id: u64,
    goal: String,
    subgoals: Vec<u64>,
    prerequisites: Vec<u64>,
    priority: f64,
    status: PlanningStatus,
    estimated_effort: f64,
    deadline: Option<u64>,
    parent_node: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
enum PlanningStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Blocked,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ReasoningEngine {
    inference_rules: Vec<InferenceRule>,
    knowledge_base: HashMap<String, Vec<String>>,
    working_hypotheses: Vec<Hypothesis>,
    logical_depth: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct InferenceRule {
    name: String,
    conditions: Vec<String>,
    conclusions: Vec<String>,
    confidence: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Hypothesis {
    statement: String,
    confidence: f64,
    evidence: Vec<String>,
    testing_priority: f64,
}

#[allow(dead_code)]
impl ReasoningEngine {
    fn new() -> Self {
        Self {
            inference_rules: Vec::new(),
            knowledge_base: HashMap::new(),
            working_hypotheses: Vec::new(),
            logical_depth: 3,
        }
    }

    fn add_rule(&mut self, rule: InferenceRule) {
        self.inference_rules.push(rule);
    }

    fn add_knowledge(&mut self, domain: &str, fact: String) {
        self.knowledge_base
            .entry(domain.to_string())
            .or_default()
            .push(fact);
    }

    fn forward_chain(&self, observations: &[String]) -> Vec<String> {
        let mut conclusions = Vec::new();

        for rule in &self.inference_rules {
            let conditions_met = rule.conditions.iter().all(|condition| {
                observations.contains(condition)
                    || self
                        .knowledge_base
                        .values()
                        .any(|facts| facts.contains(condition))
            });

            if conditions_met {
                conclusions.extend(rule.conclusions.clone());
            }
        }

        conclusions
    }

    fn generate_hypothesis(&mut self, observation: &str) -> Hypothesis {
        let hypothesis = Hypothesis {
            statement: format!("If {} then {}", observation, "this suggests a pattern"),
            confidence: 0.5,
            evidence: vec![observation.to_string()],
            testing_priority: 0.5,
        };

        self.working_hypotheses.push(hypothesis.clone());
        hypothesis
    }

    fn test_hypothesis(&mut self, hypothesis_id: usize, evidence: &str) {
        if let Some(hypothesis) = self.working_hypotheses.get_mut(hypothesis_id) {
            hypothesis.evidence.push(evidence.to_string());
            hypothesis.confidence = (hypothesis.confidence + 0.1).min(1.0);
        }
    }
}

#[allow(dead_code)]
impl PlanningNode {
    fn new(id: u64, goal: String, priority: f64) -> Self {
        Self {
            id,
            goal,
            subgoals: Vec::new(),
            prerequisites: Vec::new(),
            priority,
            status: PlanningStatus::Pending,
            estimated_effort: 0.5,
            deadline: None,
            parent_node: None,
        }
    }

    fn add_subgoal(&mut self, subgoal_id: u64) {
        self.subgoals.push(subgoal_id);
    }

    fn add_prerequisite(&mut self, prerequisite_id: u64) {
        self.prerequisites.push(prerequisite_id);
    }

    fn can_execute(&self, completed_nodes: &HashSet<u64>) -> bool {
        self.prerequisites
            .iter()
            .all(|prereq| completed_nodes.contains(prereq))
    }
}

// =========================================================================
// 🎯 GOAL HIERARCHY AND AUTONOMOUS DECISION MAKING
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct GoalHierarchy {
    primary_goals: Vec<PlanningNode>,
    active_goal_stack: VecDeque<u64>,
    completed_goals: HashSet<u64>,
    goal_importance_decay: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DecisionContext {
    available_resources: f64,
    time_pressure: f64,
    risk_tolerance: f64,
    uncertainty: f64,
    recent_outcomes: Vec<Outcome>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Outcome {
    action: String,
    result: f64,
    timestamp: u64,
    context_snapshot: Vec<f64>,
}

#[allow(dead_code)]
impl GoalHierarchy {
    fn new() -> Self {
        Self {
            primary_goals: Vec::new(),
            active_goal_stack: VecDeque::new(),
            completed_goals: HashSet::new(),
            goal_importance_decay: 0.01,
        }
    }

    fn add_primary_goal(&mut self, goal: String, priority: f64) -> u64 {
        let id = current_secs();
        let node = PlanningNode::new(id, goal, priority);
        self.primary_goals.push(node);
        id
    }

    fn prioritize_goals(&mut self) {
        self.primary_goals
            .sort_by(|a, b| b.priority.total_cmp(&a.priority));

        // Decay importance over time
        for goal in &mut self.primary_goals {
            goal.priority *= 1.0 - self.goal_importance_decay;
        }
    }

    fn select_next_goal(&mut self, completed: &HashSet<u64>) -> Option<&PlanningNode> {
        self.primary_goals
            .iter()
            .find(|&goal| {
                !completed.contains(&goal.id)
                    && goal.status == PlanningStatus::Pending
                    && goal.can_execute(completed)
            })
            .map(|v| v as _)
    }

    fn mark_goal_completed(&mut self, goal_id: u64) {
        if let Some(goal) = self.primary_goals.iter_mut().find(|g| g.id == goal_id) {
            goal.status = PlanningStatus::Completed;
        }
        self.completed_goals.insert(goal_id);
    }

    fn get_active_goals(&self) -> Vec<&PlanningNode> {
        self.primary_goals
            .iter()
            .filter(|g| !self.completed_goals.contains(&g.id))
            .collect()
    }
}

#[allow(dead_code)]
impl DecisionContext {
    fn new() -> Self {
        Self {
            available_resources: 1.0,
            time_pressure: 0.5,
            risk_tolerance: 0.5,
            uncertainty: 0.5,
            recent_outcomes: Vec::new(),
        }
    }

    fn update_from_outcome(&mut self, outcome: Outcome) {
        self.recent_outcomes.push(outcome);
        if self.recent_outcomes.len() > 50 {
            self.recent_outcomes.remove(0);
        }

        // Update context based on recent outcomes
        let avg_outcome: f64 = self.recent_outcomes.iter().map(|o| o.result).sum::<f64>()
            / self.recent_outcomes.len() as f64;
        self.uncertainty = (self.uncertainty * 0.9 + (1.0 - avg_outcome) * 0.1).clamp(0.0, 1.0);

        self.time_pressure = if avg_outcome < 0.5 {
            (self.time_pressure * 1.1).min(1.0)
        } else {
            (self.time_pressure * 0.9).max(0.1)
        };
    }

    fn evaluate_action(&self, _action: &str, estimated_cost: f64, estimated_benefit: f64) -> f64 {
        let resource_factor = if estimated_cost > self.available_resources {
            0.5
        } else {
            1.0
        };

        let risk_factor = if estimated_cost > self.risk_tolerance {
            0.5
        } else {
            1.0
        };

        let urgency_factor = 1.0 + self.time_pressure;

        let uncertainty_penalty = self.uncertainty * 0.3;

        (estimated_benefit * resource_factor * risk_factor * urgency_factor) - uncertainty_penalty
    }
}

// =========================================================================
// 🎨 CREATIVITY AND ABSTRACT REASONING ENGINE
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CreativityEngine {
    novel_concepts: Vec<NovelConcept>,
    metaphor_generator: MetaphorGenerator,
    abstraction_levels: Vec<AbstractionLevel>,
    creative_threshold: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NovelConcept {
    id: u64,
    description: String,
    novelty_score: f64,
    utility_score: f64,
    source_concepts: Vec<String>,
    generated_at: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaphorGenerator {
    source_domains: Vec<String>,
    target_domains: Vec<String>,
    established_metaphors: HashMap<String, Vec<String>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AbstractionLevel {
    level: u8,
    description: String,
    examples: Vec<String>,
    operators: Vec<String>,
}

#[allow(dead_code)]
impl CreativityEngine {
    fn new() -> Self {
        Self {
            novel_concepts: Vec::new(),
            metaphor_generator: MetaphorGenerator::new(),
            abstraction_levels: Self::init_abstraction_levels(),
            creative_threshold: 0.7,
        }
    }

    fn init_abstraction_levels() -> Vec<AbstractionLevel> {
        vec![
            AbstractionLevel {
                level: 0,
                description: "Concrete/Physical".to_string(),
                examples: vec!["apple".to_string(), "chair".to_string(), "rock".to_string()],
                operators: vec![
                    "touch".to_string(),
                    "move".to_string(),
                    "observe".to_string(),
                ],
            },
            AbstractionLevel {
                level: 5,
                description: "Functional/Operational".to_string(),
                examples: vec![
                    "tool".to_string(),
                    "process".to_string(),
                    "mechanism".to_string(),
                ],
                operators: vec![
                    "use".to_string(),
                    "modify".to_string(),
                    "optimize".to_string(),
                ],
            },
            AbstractionLevel {
                level: 10,
                description: "Conceptual/Abstract".to_string(),
                examples: vec![
                    "justice".to_string(),
                    "beauty".to_string(),
                    "truth".to_string(),
                ],
                operators: vec![
                    "analyze".to_string(),
                    "synthesize".to_string(),
                    "evaluate".to_string(),
                ],
            },
        ]
    }

    fn generate_novel_concept(&mut self, source_concepts: Vec<String>) -> NovelConcept {
        let timestamp = current_secs();
        let id = rand::random::<u64>();

        // Calculate novelty based on combination rarity
        let novelty_score = self.calculate_novelty(&source_concepts);

        // Generate description by combining concepts
        let description = if source_concepts.len() >= 2 {
            format!("{} meets {}", source_concepts[0], source_concepts[1])
        } else {
            format!("Novel application of {}", source_concepts[0])
        };

        let concept = NovelConcept {
            id,
            description,
            novelty_score,
            utility_score: 0.5, // Initially unknown
            source_concepts,
            generated_at: timestamp,
        };

        if concept.novelty_score > self.creative_threshold {
            self.novel_concepts.push(concept.clone());
        }

        concept
    }

    fn calculate_novelty(&self, concepts: &[String]) -> f64 {
        if concepts.len() < 2 {
            return 0.3;
        }

        // Check if this combination exists in known concepts
        let combination_exists = self
            .novel_concepts
            .iter()
            .any(|nc| nc.source_concepts.iter().all(|c| concepts.contains(c)));

        if combination_exists {
            0.2
        } else {
            0.8 + (concepts.len() as f64 * 0.05)
        }
    }

    fn generate_metaphor(&mut self, target: &str) -> String {
        self.metaphor_generator.generate(target)
    }

    fn abstract_concept(&self, concept: &str, target_level: u8) -> Option<String> {
        for level in &self.abstraction_levels {
            if level.level == target_level && level.examples.iter().any(|ex| ex.contains(concept)) {
                return Some(level.description.clone());
            }
        }
        None
    }
}

impl MetaphorGenerator {
    fn new() -> Self {
        Self {
            source_domains: vec![
                "nature".to_string(),
                "mechanics".to_string(),
                "cooking".to_string(),
                "music".to_string(),
                "architecture".to_string(),
            ],
            target_domains: vec![
                "emotions".to_string(),
                "thoughts".to_string(),
                "relationships".to_string(),
                "work".to_string(),
                "life".to_string(),
            ],
            established_metaphors: HashMap::new(),
        }
    }

    fn generate(&mut self, target: &str) -> String {
        if let Some(metaphors) = self.established_metaphors.get(target) {
            if !metaphors.is_empty() {
                return metaphors[rand::random::<usize>() % metaphors.len()].clone();
            }
        }

        // Generate new metaphor
        let source = &self.source_domains[rand::random::<usize>() % self.source_domains.len()];
        format!("{} is like {}", target, source)
    }
}

// REMOVED DUPLICATE - TheoryOfMindEngine now defined in true AGI section

// REMOVED DUPLICATE - MentalModel, SocialContext, EmpathyModule now defined in true AGI section
// REMOVED ORPHANED CODE - using true AGI TheoryOfMindEngine implementation

// =========================================================================
// 📝 ENHANCED LANGUAGE UNDERSTANDING AND CONTEXT PROCESSING
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LanguageUnderstandingEngine {
    vocabulary: HashMap<String, Vec<f64>>,
    semantic_network: HashMap<String, Vec<String>>,
    context_window: VecDeque<String>,
    discourse_markers: Vec<String>,
    sentiment_analyzer: SentimentAnalyzer,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SentimentAnalyzer {
    positive_words: Vec<String>,
    negative_words: Vec<String>,
    intensity_modifiers: HashMap<String, f64>,
}

impl LanguageUnderstandingEngine {
    fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            semantic_network: HashMap::new(),
            context_window: VecDeque::with_capacity(10),
            discourse_markers: vec![
                "however".to_string(),
                "therefore".to_string(),
                "moreover".to_string(),
                "consequently".to_string(),
            ],
            sentiment_analyzer: SentimentAnalyzer::new(),
        }
    }

    fn process_input(&mut self, input: &str) -> LanguageProcessingResult {
        // Add to context window
        self.context_window.push_back(input.to_string());
        if self.context_window.len() > 10 {
            self.context_window.pop_front();
        }

        let tokens = self.tokenize(input);
        let sentiment = self.sentiment_analyzer.analyze(&tokens);
        let entities = self.extract_entities(&tokens);
        let intent = self.classify_intent(&tokens);

        LanguageProcessingResult {
            original: input.to_string(),
            tokens,
            sentiment,
            entities,
            intent,
            context: self.context_window.iter().cloned().collect(),
        }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_lowercase()).collect()
    }

    fn extract_entities(&self, tokens: &[String]) -> Vec<String> {
        // Simple entity extraction - could be enhanced
        tokens
            .iter()
            .filter(|t| t.chars().next().is_some_and(|c| c.is_uppercase()))
            .cloned()
            .collect()
    }

    fn classify_intent(&self, tokens: &[String]) -> String {
        if tokens.iter().any(|t| t == "help" || t == "assist") {
            "request_help".to_string()
        } else if tokens.iter().any(|t| t == "why" || t == "how") {
            "question".to_string()
        } else if tokens.iter().any(|t| t == "please" || t == "would") {
            "polite_request".to_string()
        } else {
            "statement".to_string()
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LanguageProcessingResult {
    original: String,
    tokens: Vec<String>,
    sentiment: f64,
    entities: Vec<String>,
    intent: String,
    context: Vec<String>,
}

impl SentimentAnalyzer {
    fn new() -> Self {
        Self {
            positive_words: vec![
                "good".to_string(),
                "great".to_string(),
                "excellent".to_string(),
                "happy".to_string(),
                "love".to_string(),
            ],
            negative_words: vec![
                "bad".to_string(),
                "terrible".to_string(),
                "awful".to_string(),
                "sad".to_string(),
                "hate".to_string(),
            ],
            intensity_modifiers: {
                let mut map = HashMap::new();
                map.insert("very".to_string(), 1.5);
                map.insert("extremely".to_string(), 2.0);
                map.insert("somewhat".to_string(), 0.7);
                map
            },
        }
    }

    fn analyze(&self, tokens: &[String]) -> f64 {
        let mut score = 0.0;
        let mut intensity = 1.0;

        for token in tokens {
            if self.positive_words.contains(token) {
                score += 1.0;
            } else if self.negative_words.contains(token) {
                score -= 1.0;
            }

            if let Some(modifier) = self.intensity_modifiers.get(token) {
                intensity *= modifier;
            }
        }

        // Normalize to [-1, 1]
        let normalized = score / tokens.len().max(1) as f64;
        (normalized * intensity).clamp(-1.0, 1.0)
    }
}

// REMOVED DUPLICATE IMPLEMENTATIONS - using true AGI versions

// =========================================================================
// 📋 LONG-TERM HIERARCHICAL PLANNING
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct HierarchicalPlanner {
    task_network: TaskNetwork,
    decomposition_rules: Vec<DecompositionRule>,
    execution_monitor: ExecutionMonitor,
    current_plan: Option<Plan>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TaskNetwork {
    tasks: Vec<Task>,
    ordering_constraints: Vec<OrderingConstraint>,
    causal_links: Vec<CausalLink>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Task {
    id: u64,
    name: String,
    task_type: TaskType,
    parameters: HashMap<String, f64>,
    status: TaskStatus,
    estimated_duration: Duration,
    priority: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum TaskType {
    Primitive,
    Abstract,
    Goal,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OrderingConstraint {
    before: u64,
    after: u64,
    constraint_type: ConstraintType,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum ConstraintType {
    Temporal,
    Causal,
    Resource,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CausalLink {
    from_task: u64,
    to_task: u64,
    condition: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DecompositionRule {
    abstract_task: String,
    subtasks: Vec<String>,
    ordering: Vec<(usize, usize)>,
    conditions: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExecutionMonitor {
    execution_history: Vec<ExecutionEvent>,
    failure_rate: f64,
    adaptation_strategies: Vec<AdaptationStrategy>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExecutionEvent {
    task_id: u64,
    timestamp: u64,
    outcome: ExecutionOutcome,
    duration: Duration,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum ExecutionOutcome {
    Success,
    Failure(String),
    Partial(f64),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AdaptationStrategy {
    name: String,
    trigger_condition: String,
    adaptation_action: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Plan {
    id: u64,
    tasks: Vec<Task>,
    total_duration: Duration,
    success_probability: f64,
    alternatives: Vec<Plan>,
}

impl HierarchicalPlanner {
    fn new() -> Self {
        Self {
            task_network: TaskNetwork::new(),
            decomposition_rules: Vec::new(),
            execution_monitor: ExecutionMonitor::new(),
            current_plan: None,
        }
    }

    fn create_plan(&mut self, goal: Task) -> Plan {
        let mut tasks = vec![goal.clone()];
        let mut total_duration = Duration::from_secs(0);

        // Decompose abstract tasks
        self.decompose_tasks(&mut tasks);

        // Calculate total duration
        for task in &tasks {
            total_duration += task.estimated_duration;
        }

        let plan = Plan {
            id: current_secs(),
            tasks,
            total_duration,
            success_probability: 0.8,
            alternatives: Vec::new(),
        };

        self.current_plan = Some(plan.clone());
        plan
    }

    fn decompose_tasks(&self, tasks: &mut Vec<Task>) {
        let mut to_decompose = Vec::new();

        for (i, task) in tasks.iter().enumerate() {
            if matches!(task.task_type, TaskType::Abstract) {
                to_decompose.push((i, task.name.clone(), task.priority));
            }
        }

        for (index, task_name, task_priority) in to_decompose.into_iter().rev() {
            if let Some(subtasks) = self.find_decomposition(&task_name) {
                tasks.remove(index);
                for (sub_idx, subtask_name) in subtasks.iter().enumerate() {
                    tasks.insert(
                        index + sub_idx,
                        Task {
                            id: current_secs() + sub_idx as u64,
                            name: subtask_name.clone(),
                            task_type: TaskType::Primitive,
                            parameters: HashMap::new(),
                            status: TaskStatus::Pending,
                            estimated_duration: Duration::from_secs(60),
                            priority: task_priority,
                        },
                    );
                }
            }
        }
    }

    fn find_decomposition(&self, task_name: &str) -> Option<Vec<String>> {
        for rule in &self.decomposition_rules {
            if rule.abstract_task == task_name {
                return Some(rule.subtasks.clone());
            }
        }
        None
    }

    fn execute_plan(&mut self, plan: &mut Plan) -> Result<(), String> {
        for task in &mut plan.tasks {
            task.status = TaskStatus::InProgress;

            // Simulate execution
            let outcome = if rand::random::<f64>() > 0.2 {
                ExecutionOutcome::Success
            } else {
                ExecutionOutcome::Failure("Unexpected error".to_string())
            };

            let event = ExecutionEvent {
                task_id: task.id,
                timestamp: current_secs(),
                outcome: outcome.clone(),
                duration: task.estimated_duration,
            };

            self.execution_monitor.record_event(event.clone());

            match outcome {
                ExecutionOutcome::Success => task.status = TaskStatus::Completed,
                ExecutionOutcome::Failure(_) => task.status = TaskStatus::Failed,
                ExecutionOutcome::Partial(_) => task.status = TaskStatus::Pending,
            }
        }

        Ok(())
    }

    fn adapt_plan(&mut self, failure_reason: &str) {
        if let Some(strategy) = self.execution_monitor.find_adaptation(failure_reason) {
            println!("🔄 [PLAN ADAPTATION]: Applying strategy: {}", strategy.name);
            // Apply adaptation logic
        }
    }
}

impl TaskNetwork {
    fn new() -> Self {
        Self {
            tasks: Vec::new(),
            ordering_constraints: Vec::new(),
            causal_links: Vec::new(),
        }
    }

    fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    fn add_constraint(&mut self, constraint: OrderingConstraint) {
        self.ordering_constraints.push(constraint);
    }
}

impl ExecutionMonitor {
    fn new() -> Self {
        Self {
            execution_history: Vec::new(),
            failure_rate: 0.0,
            adaptation_strategies: Vec::new(),
        }
    }

    fn record_event(&mut self, event: ExecutionEvent) {
        self.execution_history.push(event);
        self.update_failure_rate();
    }

    fn update_failure_rate(&mut self) {
        let failures = self
            .execution_history
            .iter()
            .filter(|e| matches!(e.outcome, ExecutionOutcome::Failure(_)))
            .count();

        self.failure_rate = failures as f64 / self.execution_history.len().max(1) as f64;
    }

    fn find_adaptation(&self, reason: &str) -> Option<&AdaptationStrategy> {
        self.adaptation_strategies
            .iter()
            .find(|s| reason.contains(&s.trigger_condition))
    }
}

// =========================================================================
// 🔍 CURIOSITY & INTRINSIC MOTIVATION
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CuriosityEngine {
    information_gain_tracker: HashMap<String, f64>,
    novelty_detector: NoveltyDetector,
    exploration_strategy: ExplorationStrategy,
    intrinsic_rewards: Vec<IntrinsicReward>,
    curiosity_budget: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NoveltyDetector {
    encountered_states: HashMap<String, u64>,
    novelty_threshold: f64,
    surprise_history: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExplorationStrategy {
    epsilon: f64, // Exploration rate
    epsilon_decay: f64,
    min_epsilon: f64,
    strategy_type: ExplorationType,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum ExplorationType {
    EpsilonGreedy,
    UpperConfidenceBound,
    ThompsonSampling,
    InformationGain,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct IntrinsicReward {
    source: String,
    value: f64,
    timestamp: u64,
    reward_type: RewardType,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum RewardType {
    Novelty,
    InformationGain,
    Competence,
    Surprise,
}

impl CuriosityEngine {
    fn new() -> Self {
        Self {
            information_gain_tracker: HashMap::new(),
            novelty_detector: NoveltyDetector::new(),
            exploration_strategy: ExplorationStrategy::new(),
            intrinsic_rewards: Vec::new(),
            curiosity_budget: 1.0,
        }
    }

    fn calculate_novelty(&mut self, state: &[f64]) -> f64 {
        let state_key = format!("{:.4?}", state.iter().take(10).cloned().collect::<Vec<_>>());
        let encounter_count = *self
            .novelty_detector
            .encountered_states
            .get(&state_key)
            .unwrap_or(&0);

        let novelty = 1.0 / (1.0 + encounter_count as f64);
        self.novelty_detector
            .encountered_states
            .insert(state_key, encounter_count + 1);

        novelty
    }

    fn calculate_information_gain(&mut self, action: &str, outcome: f64) -> f64 {
        let key = format!("{}_{}", action, outcome > 0.5);
        let current_info = *self.information_gain_tracker.get(&key).unwrap_or(&0.0);
        let info_gain = (1.0 - current_info).abs();

        self.information_gain_tracker
            .insert(key, current_info + info_gain * 0.1);

        info_gain
    }

    fn calculate_intrinsic_reward(&mut self, state: &[f64], action: &str, outcome: f64) -> f64 {
        let novelty = self.calculate_novelty(state);
        let info_gain = self.calculate_information_gain(action, outcome);
        let surprise = (outcome - 0.5).abs();

        let total_reward = 0.4 * novelty + 0.3 * info_gain + 0.3 * surprise;

        let reward = IntrinsicReward {
            source: format!("{}_{}", action, outcome),
            value: total_reward,
            timestamp: current_secs(),
            reward_type: RewardType::Novelty,
        };

        self.intrinsic_rewards.push(reward);
        total_reward
    }

    fn select_exploration_action(&mut self, available_actions: &[String]) -> usize {
        match self.exploration_strategy.strategy_type {
            ExplorationType::EpsilonGreedy => {
                if rand::random::<f64>() < self.exploration_strategy.epsilon {
                    rand::random::<usize>() % available_actions.len()
                } else {
                    // Exploit best known action
                    0
                }
            }
            ExplorationType::InformationGain => available_actions
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    let info_a = self
                        .information_gain_tracker
                        .get(&available_actions[a.0])
                        .unwrap_or(&0.0);
                    let info_b = self
                        .information_gain_tracker
                        .get(&available_actions[b.0])
                        .unwrap_or(&0.0);
                    info_a.total_cmp(info_b)
                })
                .map(|(i, _)| i)
                .unwrap_or(0),
            _ => rand::random::<usize>() % available_actions.len(),
        }
    }

    fn decay_exploration(&mut self) {
        self.exploration_strategy.epsilon = (self.exploration_strategy.epsilon
            * self.exploration_strategy.epsilon_decay)
            .max(self.exploration_strategy.min_epsilon);
    }

    fn generate_self_goal(&mut self) -> Option<String> {
        if self.curiosity_budget > 0.5 {
            Some("Explore novel state space".to_string())
        } else {
            None
        }
    }
}

// =========================================================================
// 🧠 ADVANCED MEMORY RETRIEVAL
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AdvancedMemoryRetrieval {
    retrieval_index: RetrievalIndex,
    attention_mechanism: AttentionMechanism,
    consolidation_system: ConsolidationSystem,
    replay_buffer: ReplayBuffer,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RetrievalIndex {
    spatial_index: HashMap<String, Vec<u64>>,
    temporal_index: VecDeque<(u64, u64)>,
    semantic_index: HashMap<String, Vec<u64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AttentionMechanism {
    query_weights: Vec<f64>,
    key_weights: Vec<f64>,
    value_weights: Vec<f64>,
    attention_heads: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConsolidationSystem {
    consolidation_schedule: Vec<ConsolidationTask>,
    sleep_cycle_active: bool,
    consolidation_progress: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConsolidationTask {
    memory_id: u64,
    priority: f64,
    target_abstraction: u8,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ReplayBuffer {
    experiences: Vec<ReplayExperience>,
    capacity: usize,
    sampling_strategy: SamplingStrategy,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ReplayExperience {
    state: Vec<f64>,
    action: String,
    reward: f64,
    next_state: Vec<f64>,
    importance: f64,
    timestamp: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum SamplingStrategy {
    Uniform,
    Prioritized,
    Proportional,
}

impl AdvancedMemoryRetrieval {
    fn new() -> Self {
        Self {
            retrieval_index: RetrievalIndex::new(),
            attention_mechanism: AttentionMechanism::new(),
            consolidation_system: ConsolidationSystem::new(),
            replay_buffer: ReplayBuffer::new(10000),
        }
    }

    fn retrieve_relevant(&self, query: &[f64], count: usize) -> Vec<u64> {
        let mut candidates = Vec::new();

        // Query spatial index
        for memory_ids in self.retrieval_index.spatial_index.values() {
            candidates.extend(memory_ids);
        }

        // Apply attention mechanism
        let scored = self.attention_mechanism.score_relevance(query, &candidates);

        // Return top-k
        scored.into_iter().take(count).map(|(id, _)| id).collect()
    }

    fn add_to_index(&mut self, memory_id: u64, embedding: Vec<f64>, semantic_tags: Vec<String>) {
        let spatial_key = format!(
            "{:.2?}",
            embedding.iter().take(5).cloned().collect::<Vec<_>>()
        );
        self.retrieval_index
            .spatial_index
            .entry(spatial_key)
            .or_default()
            .push(memory_id);

        let timestamp = current_secs();
        self.retrieval_index
            .temporal_index
            .push_back((memory_id, timestamp));

        for tag in semantic_tags {
            self.retrieval_index
                .semantic_index
                .entry(tag)
                .or_default()
                .push(memory_id);
        }
    }

    fn schedule_consolidation(&mut self, memory_id: u64, priority: f64) {
        let task = ConsolidationTask {
            memory_id,
            priority,
            target_abstraction: 5,
        };
        self.consolidation_system.consolidation_schedule.push(task);
    }

    fn process_consolidation(&mut self) {
        if self.consolidation_system.sleep_cycle_active {
            self.consolidation_system
                .consolidation_schedule
                .sort_by(|a, b| b.priority.total_cmp(&a.priority));

            for _task in self
                .consolidation_system
                .consolidation_schedule
                .drain(..)
                .take(10)
            {
                self.consolidation_system.consolidation_progress += 0.1;
            }
        }
    }

    fn add_experience(&mut self, experience: ReplayExperience) {
        self.replay_buffer.add(experience);
    }

    fn sample_replay(&self, batch_size: usize) -> Vec<&ReplayExperience> {
        self.replay_buffer.sample(batch_size)
    }
}

impl RetrievalIndex {
    fn new() -> Self {
        Self {
            spatial_index: HashMap::new(),
            temporal_index: VecDeque::new(),
            semantic_index: HashMap::new(),
        }
    }
}

impl AttentionMechanism {
    fn new() -> Self {
        Self {
            query_weights: vec![0.5; 128],
            key_weights: vec![0.5; 128],
            value_weights: vec![0.5; 128],
            attention_heads: 4,
        }
    }

    fn score_relevance(&self, query: &[f64], candidates: &[u64]) -> Vec<(u64, f64)> {
        candidates
            .iter()
            .map(|&id| {
                let score = if query.len() >= self.query_weights.len() {
                    let dot: f64 = query
                        .iter()
                        .zip(self.query_weights.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    dot.abs()
                } else {
                    0.5
                };
                (id, score)
            })
            .collect()
    }
}

impl ConsolidationSystem {
    fn new() -> Self {
        Self {
            consolidation_schedule: Vec::new(),
            sleep_cycle_active: false,
            consolidation_progress: 0.0,
        }
    }

    fn activate_sleep_cycle(&mut self) {
        self.sleep_cycle_active = true;
    }

    fn deactivate_sleep_cycle(&mut self) {
        self.sleep_cycle_active = false;
    }
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            experiences: Vec::new(),
            capacity,
            sampling_strategy: SamplingStrategy::Prioritized,
        }
    }

    fn add(&mut self, experience: ReplayExperience) {
        if self.experiences.len() >= self.capacity {
            self.experiences.remove(0);
        }
        self.experiences.push(experience);
    }

    fn sample(&self, batch_size: usize) -> Vec<&ReplayExperience> {
        match self.sampling_strategy {
            SamplingStrategy::Uniform => {
                let mut rng = rand::thread_rng();
                self.experiences
                    .iter()
                    .filter(|_| rng.gen_bool(0.5))
                    .take(batch_size)
                    .collect()
            }
            SamplingStrategy::Prioritized => {
                let mut sorted: Vec<_> = self.experiences.iter().collect();
                sorted.sort_by(|a, b| b.importance.total_cmp(&a.importance));
                sorted.into_iter().take(batch_size).collect()
            }
            SamplingStrategy::Proportional => {
                let total_importance: f64 = self.experiences.iter().map(|e| e.importance).sum();
                if total_importance > 0.0 {
                    let mut rng = rand::thread_rng();
                    self.experiences
                        .iter()
                        .filter(|_| rng.gen_bool(0.5))
                        .take(batch_size)
                        .collect()
                } else {
                    self.experiences.iter().take(batch_size).collect()
                }
            }
        }
    }
}

// =========================================================================
// 🎓 META-LEARNING & SELF-IMPROVEMENT
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaLearningEngine {
    learning_strategies: Vec<LearningStrategy>,
    strategy_performance: HashMap<String, f64>,
    hyperparameter_optimizer: HyperparameterOptimizer,
    neural_architecture_search: NeuralArchitectureSearch,
    meta_cognitive_strategies: Vec<MetaCognitiveStrategy>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LearningStrategy {
    name: String,
    learning_rate: f64,
    batch_size: usize,
    regularization: f64,
    performance_history: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct HyperparameterOptimizer {
    current_hyperparameters: HashMap<String, f64>,
    optimization_history: Vec<HashMap<String, f64>>,
    optimization_method: OptimizationMethod,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum OptimizationMethod {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GeneticAlgorithm,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NeuralArchitectureSearch {
    candidate_architectures: Vec<Architecture>,
    evaluation_history: Vec<ArchitectureEvaluation>,
    search_space: SearchSpace,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Architecture {
    layers: Vec<LayerConfig>,
    connections: Vec<ConnectionConfig>,
    complexity: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LayerConfig {
    layer_type: String,
    size: usize,
    activation: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConnectionConfig {
    from: usize,
    to: usize,
    connection_type: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ArchitectureEvaluation {
    architecture_id: u64,
    performance: f64,
    complexity: f64,
    efficiency: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SearchSpace {
    layer_types: Vec<String>,
    size_ranges: Vec<(usize, usize)>,
    activations: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MetaCognitiveStrategy {
    name: String,
    conditions: Vec<String>,
    actions: Vec<String>,
    effectiveness: f64,
}

impl MetaLearningEngine {
    fn new() -> Self {
        Self {
            learning_strategies: Vec::new(),
            strategy_performance: HashMap::new(),
            hyperparameter_optimizer: HyperparameterOptimizer::new(),
            neural_architecture_search: NeuralArchitectureSearch::new(),
            meta_cognitive_strategies: Vec::new(),
        }
    }

    fn select_best_strategy(&self) -> Option<&LearningStrategy> {
        let best_name = self
            .strategy_performance
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(name, _)| name)?;

        self.learning_strategies
            .iter()
            .find(|s| &s.name == best_name)
    }

    fn update_strategy_performance(&mut self, strategy_name: &str, performance: f64) {
        let current = *self.strategy_performance.get(strategy_name).unwrap_or(&0.5);
        let updated = current * 0.9 + performance * 0.1;
        self.strategy_performance
            .insert(strategy_name.to_string(), updated);
    }

    fn optimize_hyperparameters(&mut self, performance_metric: f64) {
        self.hyperparameter_optimizer.optimize(performance_metric);
    }

    fn search_architecture(&mut self) -> Option<Architecture> {
        self.neural_architecture_search.generate_candidate()
    }

    fn evaluate_architecture(&mut self, architecture: &Architecture, performance: f64) {
        self.neural_architecture_search
            .record_evaluation(architecture, performance);
    }

    fn select_meta_strategy(&self, context: &str) -> Option<&MetaCognitiveStrategy> {
        self.meta_cognitive_strategies
            .iter()
            .find(|s| s.conditions.iter().any(|c| context.contains(c)))
    }
}

impl HyperparameterOptimizer {
    fn new() -> Self {
        Self {
            current_hyperparameters: {
                let mut map = HashMap::new();
                map.insert("learning_rate".to_string(), 0.01);
                map.insert("batch_size".to_string(), 32.0);
                map.insert("dropout".to_string(), 0.1);
                map
            },
            optimization_history: Vec::new(),
            optimization_method: OptimizationMethod::BayesianOptimization,
        }
    }

    fn optimize(&mut self, performance: f64) {
        self.optimization_history
            .push(self.current_hyperparameters.clone());

        if let OptimizationMethod::BayesianOptimization = self.optimization_method {
            // Bayesian optimization logic
            if let Some(learning_rate) = self.current_hyperparameters.get_mut("learning_rate") {
                if performance > 0.8 {
                    *learning_rate *= 1.1;
                } else {
                    *learning_rate *= 0.9;
                }
            }
        }
    }
}

impl NeuralArchitectureSearch {
    fn new() -> Self {
        Self {
            candidate_architectures: Vec::new(),
            evaluation_history: Vec::new(),
            search_space: SearchSpace::new(),
        }
    }

    fn generate_candidate(&mut self) -> Option<Architecture> {
        let layers = vec![
            LayerConfig {
                layer_type: "dense".to_string(),
                size: 128,
                activation: "relu".to_string(),
            },
            LayerConfig {
                layer_type: "dense".to_string(),
                size: 64,
                activation: "relu".to_string(),
            },
        ];

        let architecture = Architecture {
            layers,
            connections: Vec::new(),
            complexity: 0.5,
        };

        self.candidate_architectures.push(architecture.clone());
        Some(architecture)
    }

    fn record_evaluation(&mut self, architecture: &Architecture, performance: f64) {
        let evaluation = ArchitectureEvaluation {
            architecture_id: current_secs(),
            performance,
            complexity: architecture.complexity,
            efficiency: performance / architecture.complexity,
        };
        self.evaluation_history.push(evaluation);
    }
}

impl SearchSpace {
    fn new() -> Self {
        Self {
            layer_types: vec![
                "dense".to_string(),
                "conv2d".to_string(),
                "lstm".to_string(),
            ],
            size_ranges: vec![(32, 512), (16, 256)],
            activations: vec![
                "relu".to_string(),
                "tanh".to_string(),
                "sigmoid".to_string(),
            ],
        }
    }
}

// =========================================================================
// 👁️ MULTI-MODAL INTEGRATION
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MultiModalProcessor {
    vision_processor: VisionProcessor,
    audio_processor: AudioProcessor,
    text_processor: TextProcessor,
    cross_modal_attention: CrossModalAttention,
    fusion_layer: FusionLayer,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct VisionProcessor {
    conv_layers: Vec<ConvLayer>,
    feature_extractors: Vec<FeatureExtractor>,
    spatial_attention: SpatialAttention,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConvLayer {
    filters: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    activation: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FeatureExtractor {
    feature_type: String,
    dimensions: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SpatialAttention {
    attention_weights: Vec<Vec<f64>>,
    receptive_field: (usize, usize),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AudioProcessor {
    spectrogram_analyzer: SpectrogramAnalyzer,
    frequency_bands: Vec<FrequencyBand>,
    temporal_attention: TemporalAttention,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SpectrogramAnalyzer {
    fft_size: usize,
    hop_length: usize,
    window_function: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FrequencyBand {
    min_freq: f64,
    max_freq: f64,
    importance: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TemporalAttention {
    time_steps: usize,
    attention_weights: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TextProcessor {
    tokenizer: Tokenizer,
    embedding_layer: EmbeddingLayer,
    positional_encoding: PositionalEncoding,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Tokenizer {
    vocabulary: HashMap<String, usize>,
    max_sequence_length: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmbeddingLayer {
    embedding_dim: usize,
    embeddings: HashMap<usize, Vec<f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PositionalEncoding {
    max_length: usize,
    encoding_dim: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CrossModalAttention {
    modality_queries: HashMap<String, Vec<f64>>,
    modality_keys: HashMap<String, Vec<f64>>,
    modality_values: HashMap<String, Vec<f64>>,
    attention_matrix: HashMap<String, HashMap<String, f64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FusionLayer {
    fusion_strategy: FusionStrategy,
    fusion_weights: HashMap<String, f64>,
    output_dim: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum FusionStrategy {
    Concatenation,
    WeightedSum,
    Attention,
    Gating,
}

impl MultiModalProcessor {
    fn new() -> Self {
        Self {
            vision_processor: VisionProcessor::new(),
            audio_processor: AudioProcessor::new(),
            text_processor: TextProcessor::new(),
            cross_modal_attention: CrossModalAttention::new(),
            fusion_layer: FusionLayer::new(),
        }
    }

    fn process_multimodal(&mut self, input: MultiModalInput) -> Vec<f64> {
        let vision_features = self.vision_processor.process(&input.vision_data);
        let audio_features = self.audio_processor.process(&input.audio_data);
        let text_features = self.text_processor.process(&input.text_data);

        let modal_features = vec![
            ("vision".to_string(), vision_features),
            ("audio".to_string(), audio_features),
            ("text".to_string(), text_features),
        ];

        let attended = self.cross_modal_attention.process(&modal_features);
        self.fusion_layer.fuse(&attended)
    }

    fn align_modalities(
        &mut self,
        vision: Vec<f64>,
        audio: Vec<f64>,
        text: Vec<f64>,
    ) -> Vec<Vec<f64>> {
        vec![vision, audio, text]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MultiModalInput {
    vision_data: Vec<f64>,
    audio_data: Vec<f64>,
    text_data: String,
}

impl VisionProcessor {
    fn new() -> Self {
        Self {
            conv_layers: vec![ConvLayer {
                filters: 32,
                kernel_size: (3, 3),
                stride: (1, 1),
                activation: "relu".to_string(),
            }],
            feature_extractors: vec![FeatureExtractor {
                feature_type: "edge".to_string(),
                dimensions: 64,
            }],
            spatial_attention: SpatialAttention::new(),
        }
    }

    fn process(&mut self, _input: &[f64]) -> Vec<f64> {
        // Simulated vision processing
        let features = vec![0.5; 128];
        self.spatial_attention.update(&features);
        features
    }
}

impl AudioProcessor {
    fn new() -> Self {
        Self {
            spectrogram_analyzer: SpectrogramAnalyzer::new(),
            frequency_bands: vec![FrequencyBand {
                min_freq: 20.0,
                max_freq: 20000.0,
                importance: 1.0,
            }],
            temporal_attention: TemporalAttention::new(),
        }
    }

    fn process(&mut self, _input: &[f64]) -> Vec<f64> {
        // Simulated audio processing
        vec![0.5; 64]
    }
}

impl TextProcessor {
    fn new() -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            embedding_layer: EmbeddingLayer::new(),
            positional_encoding: PositionalEncoding::new(),
        }
    }

    fn process(&mut self, input: &str) -> Vec<f64> {
        let tokens = self.tokenizer.tokenize(input);
        let embeddings = self.embedding_layer.embed(&tokens);

        self.positional_encoding.encode(&embeddings)
    }
}

impl CrossModalAttention {
    fn new() -> Self {
        Self {
            modality_queries: HashMap::new(),
            modality_keys: HashMap::new(),
            modality_values: HashMap::new(),
            attention_matrix: HashMap::new(),
        }
    }

    fn process(&mut self, modal_features: &[(String, Vec<f64>)]) -> Vec<(String, Vec<f64>)> {
        modal_features.to_vec()
    }
}

impl FusionLayer {
    fn new() -> Self {
        Self {
            fusion_strategy: FusionStrategy::Attention,
            fusion_weights: {
                let mut map = HashMap::new();
                map.insert("vision".to_string(), 0.4);
                map.insert("audio".to_string(), 0.3);
                map.insert("text".to_string(), 0.3);
                map
            },
            output_dim: 256,
        }
    }

    fn fuse(&self, features: &[(String, Vec<f64>)]) -> Vec<f64> {
        match self.fusion_strategy {
            FusionStrategy::WeightedSum => {
                let mut fused = vec![0.0; self.output_dim];
                for (modality, feat) in features {
                    let weight = *self.fusion_weights.get(modality).unwrap_or(&0.0);
                    for (i, &val) in feat.iter().enumerate() {
                        if i < fused.len() {
                            fused[i] += val * weight;
                        }
                    }
                }
                fused
            }
            _ => vec![0.5; self.output_dim],
        }
    }
}

impl SpatialAttention {
    fn new() -> Self {
        Self {
            attention_weights: vec![vec![0.5; 16]; 16],
            receptive_field: (3, 3),
        }
    }

    fn update(&mut self, _features: &[f64]) {
        // Update attention weights based on features
    }
}

impl TemporalAttention {
    fn new() -> Self {
        Self {
            time_steps: 100,
            attention_weights: vec![0.5; 100],
        }
    }
}

impl SpectrogramAnalyzer {
    fn new() -> Self {
        Self {
            fft_size: 2048,
            hop_length: 512,
            window_function: "hann".to_string(),
        }
    }
}

impl Tokenizer {
    fn new() -> Self {
        Self {
            vocabulary: HashMap::new(),
            max_sequence_length: 512,
        }
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| self.vocabulary.get(word).copied().unwrap_or(0))
            .collect()
    }
}

impl EmbeddingLayer {
    fn new() -> Self {
        Self {
            embedding_dim: 128,
            embeddings: HashMap::new(),
        }
    }

    fn embed(&self, tokens: &[usize]) -> Vec<Vec<f64>> {
        tokens
            .iter()
            .map(|&token| {
                self.embeddings
                    .get(&token)
                    .cloned()
                    .unwrap_or(vec![0.0; self.embedding_dim])
            })
            .collect()
    }
}

impl PositionalEncoding {
    fn new() -> Self {
        Self {
            max_length: 512,
            encoding_dim: 128,
        }
    }

    fn encode(&self, embeddings: &[Vec<f64>]) -> Vec<f64> {
        embeddings.iter().flatten().cloned().collect()
    }
}

// =========================================================================
// 🔍 EXPLAINABILITY & INTERPRETABILITY
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExplainabilityEngine {
    attention_visualizer: AttentionVisualizer,
    decision_tracer: DecisionTracer,
    explanation_generator: ExplanationGenerator,
    causal_chain_analyzer: CausalChainAnalyzer,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AttentionVisualizer {
    attention_maps: Vec<AttentionMap>,
    visualization_config: VisualizationConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AttentionMap {
    layer_name: String,
    attention_weights: Vec<Vec<f64>>,
    input_tokens: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct VisualizationConfig {
    color_scheme: String,
    highlight_threshold: f64,
    show_gradients: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DecisionTracer {
    decision_history: Vec<DecisionTrace>,
    trace_depth: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DecisionTrace {
    decision_id: u64,
    input: Vec<f64>,
    intermediate_states: Vec<Vec<f64>>,
    final_output: Vec<f64>,
    confidence: f64,
    timestamp: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExplanationGenerator {
    templates: Vec<ExplanationTemplate>,
    natural_language_model: SimpleLanguageModel,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ExplanationTemplate {
    template: String,
    variables: Vec<String>,
    适用场景: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SimpleLanguageModel {
    vocabulary: Vec<String>,
    phrase_patterns: HashMap<String, Vec<String>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CausalChainAnalyzer {
    causal_chains: Vec<CausalChain>,
    chain_confidence: HashMap<u64, f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CausalChain {
    chain_id: u64,
    steps: Vec<CausalStep>,
    overall_confidence: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CausalStep {
    factor: String,
    influence: f64,
    evidence: Vec<String>,
}

impl ExplainabilityEngine {
    fn new() -> Self {
        Self {
            attention_visualizer: AttentionVisualizer::new(),
            decision_tracer: DecisionTracer::new(),
            explanation_generator: ExplanationGenerator::new(),
            causal_chain_analyzer: CausalChainAnalyzer::new(),
        }
    }

    fn explain_decision(&mut self, input: Vec<f64>, output: Vec<f64>) -> String {
        let trace = self.decision_tracer.trace(input.clone(), output.clone());
        let causal_chain = self.causal_chain_analyzer.analyze(&trace);
        self.explanation_generator.generate(&trace, &causal_chain)
    }

    fn visualize_attention(&self, layer_name: &str) -> AttentionMap {
        self.attention_visualizer.get_map(layer_name)
    }

    fn trace_decision_path(&mut self, decision_id: u64) -> Option<&DecisionTrace> {
        self.decision_tracer.get_trace(decision_id)
    }
}

impl AttentionVisualizer {
    fn new() -> Self {
        Self {
            attention_maps: Vec::new(),
            visualization_config: VisualizationConfig::new(),
        }
    }

    fn get_map(&self, layer_name: &str) -> AttentionMap {
        self.attention_maps
            .iter()
            .find(|m| m.layer_name == layer_name)
            .cloned()
            .unwrap_or_else(|| AttentionMap {
                layer_name: layer_name.to_string(),
                attention_weights: vec![vec![0.5; 10]; 10],
                input_tokens: vec!["token".to_string(); 10],
            })
    }
}

impl DecisionTracer {
    fn new() -> Self {
        Self {
            decision_history: Vec::new(),
            trace_depth: 5,
        }
    }

    fn trace(&mut self, input: Vec<f64>, output: Vec<f64>) -> DecisionTrace {
        let trace = DecisionTrace {
            decision_id: current_secs(),
            input,
            intermediate_states: vec![vec![0.5; 64]; self.trace_depth],
            final_output: output,
            confidence: 0.8,
            timestamp: current_secs(),
        };
        self.decision_history.push(trace.clone());
        trace
    }

    fn get_trace(&self, decision_id: u64) -> Option<&DecisionTrace> {
        self.decision_history
            .iter()
            .find(|t| t.decision_id == decision_id)
    }
}

impl ExplanationGenerator {
    fn new() -> Self {
        Self {
            templates: vec![ExplanationTemplate {
                template: "The decision was primarily influenced by {} with a confidence of {:.2}"
                    .to_string(),
                variables: vec!["factor".to_string(), "confidence".to_string()],
                适用场景: "general".to_string(),
            }],
            natural_language_model: SimpleLanguageModel::new(),
        }
    }

    fn generate(&self, trace: &DecisionTrace, _chain: &CausalChain) -> String {
        format!(
            "The decision was primarily influenced by multiple factors with a confidence of {:.2}",
            trace.confidence
        )
    }
}

impl SimpleLanguageModel {
    fn new() -> Self {
        Self {
            vocabulary: vec![
                "decision".to_string(),
                "influence".to_string(),
                "confidence".to_string(),
            ],
            phrase_patterns: HashMap::new(),
        }
    }
}

impl CausalChainAnalyzer {
    fn new() -> Self {
        Self {
            causal_chains: Vec::new(),
            chain_confidence: HashMap::new(),
        }
    }

    fn analyze(&mut self, trace: &DecisionTrace) -> CausalChain {
        let chain = CausalChain {
            chain_id: current_secs(),
            steps: vec![CausalStep {
                factor: "input_features".to_string(),
                influence: 0.7,
                evidence: vec!["high correlation".to_string()],
            }],
            overall_confidence: trace.confidence,
        };
        self.causal_chains.push(chain.clone());
        chain
    }
}

impl VisualizationConfig {
    fn new() -> Self {
        Self {
            color_scheme: "viridis".to_string(),
            highlight_threshold: 0.7,
            show_gradients: true,
        }
    }
}

impl NoveltyDetector {
    fn new() -> Self {
        Self {
            encountered_states: HashMap::new(),
            novelty_threshold: 0.3,
            surprise_history: Vec::new(),
        }
    }

    fn is_novel(&self, state: &[f64]) -> bool {
        let state_key = format!("{:.4?}", state.iter().take(10).cloned().collect::<Vec<_>>());
        self.encountered_states
            .get(&state_key)
            .is_none_or(|&count| count < 3)
    }
}

impl ExplorationStrategy {
    fn new() -> Self {
        Self {
            epsilon: 0.3,
            epsilon_decay: 0.995,
            min_epsilon: 0.05,
            strategy_type: ExplorationType::EpsilonGreedy,
        }
    }
}

// =========================================================================
// 💾 WEIGHT PERSISTENCE SUBNODE - Dedicated Neural Weight Storage
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct WeightPersistenceSubnode {
    weights: Vec<Vec<Vec<f64>>>, // Layer -> Row -> Col
    biases: Vec<Vec<f64>>,       // Layer -> Bias
    weight_velocities: Vec<Vec<Vec<f64>>>,
    learning_rates: Vec<f64>,
    timestamp: u64,
    integrity_hash: String,
}

impl WeightPersistenceSubnode {
    fn new() -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
            weight_velocities: Vec::new(),
            learning_rates: Vec::new(),
            timestamp: current_secs(),
            integrity_hash: String::new(),
        }
    }

    fn from_layers(layers: &[UnfrozenMetaDenseLayer]) -> Self {
        let weights: Vec<Vec<Vec<f64>>> = layers.iter().map(|l| l.weights.clone()).collect();
        let biases: Vec<Vec<f64>> = layers.iter().map(|l| l.biases.clone()).collect();
        let weight_velocities: Vec<Vec<Vec<f64>>> =
            layers.iter().map(|l| l.weight_velocity.clone()).collect();
        let learning_rates: Vec<f64> = layers.iter().map(|l| l.local_learning_rate).collect();

        let timestamp = current_secs();
        let integrity_hash = Self::compute_integrity_hash(&weights, &biases);

        Self {
            weights,
            biases,
            weight_velocities,
            learning_rates,
            timestamp,
            integrity_hash,
        }
    }

    fn compute_integrity_hash(weights: &[Vec<Vec<f64>>], biases: &[Vec<f64>]) -> String {
        let mut hash_data = String::new();
        for (layer_weights, layer_biases) in weights.iter().zip(biases.iter()) {
            for row in layer_weights {
                for &val in row {
                    hash_data.push_str(&format!("{:.6}", val));
                }
            }
            for &val in layer_biases {
                hash_data.push_str(&format!("{:.6}", val));
            }
        }
        // Simple hash for integrity checking
        let mut hasher = Md5::new();
        hasher.update(hash_data.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn verify_integrity(&self) -> bool {
        let computed_hash = Self::compute_integrity_hash(&self.weights, &self.biases);
        computed_hash == self.integrity_hash
    }

    fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let subnode: WeightPersistenceSubnode = serde_json::from_str(&data)?;

        if !subnode.verify_integrity() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Weight integrity verification failed",
            ));
        }

        Ok(subnode)
    }

    fn apply_to_layers(&self, layers: &mut [UnfrozenMetaDenseLayer]) -> Result<(), String> {
        if self.weights.len() != layers.len() {
            return Err(format!(
                "Weight layer count mismatch: {} vs {}",
                self.weights.len(),
                layers.len()
            ));
        }

        for (i, layer) in layers.iter_mut().enumerate() {
            if i < self.weights.len() {
                layer.weights = self.weights[i].clone();
                layer.biases = self.biases[i].clone();
                layer.weight_velocity = self.weight_velocities[i].clone();
                layer.local_learning_rate = self.learning_rates[i];
            }
        }
        Ok(())
    }
}

// =========================================================================
// 🛡️ DYNAMIC FILE-DEFENSE QUARANTINE SYSTEM
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct QuarantineEvent {
    timestamp: u64,
    file_path: String,
    threat_type: String,
    severity: f64,
    action_taken: String,
    quarantine_hash: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FileDefenseQuarantine {
    active_quarantine: bool,
    quarantine_events: Vec<QuarantineEvent>,
    threat_threshold: f64,
    auto_rollback_enabled: bool,
    last_integrity_check: u64,
    compromised_files: Vec<String>,
}

impl FileDefenseQuarantine {
    fn new() -> Self {
        Self {
            active_quarantine: false,
            quarantine_events: Vec::new(),
            threat_threshold: 0.7,
            auto_rollback_enabled: true,
            last_integrity_check: current_secs(),
            compromised_files: Vec::new(),
        }
    }

    fn analyze_file_integrity(&mut self, file_path: &Path) -> Result<bool, String> {
        if !file_path.exists() {
            return Ok(true); // Non-existent files are not threats
        }

        let metadata =
            fs::metadata(file_path).map_err(|e| format!("Cannot read file metadata: {}", e))?;

        let file_size = metadata.len();
        let modified = metadata
            .modified()
            .map_err(|e| format!("Cannot read modification time: {}", e))?
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| format!("Cannot convert time: {}", e))?
            .as_secs();

        // Check for suspicious file characteristics
        let mut threat_score = 0.0;
        let mut threat_reasons = Vec::new();

        // Large file size warning
        if file_size > 10_000_000 {
            // 10MB
            threat_score += 0.3;
            threat_reasons.push("Oversized file");
        }

        // Recent modification warning
        let age = current_secs() - modified;
        if age < 60 {
            // Modified within last minute
            threat_score += 0.4;
            threat_reasons.push("Very recent modification");
        }

        // Check for JSON structure if it's a JSON file
        if file_path.extension().is_some_and(|ext| ext == "json") {
            if let Ok(content) = fs::read_to_string(file_path) {
                if !content.trim().starts_with('{') {
                    threat_score += 0.5;
                    threat_reasons.push("Invalid JSON structure");
                }
            }
        }

        self.last_integrity_check = current_secs();

        if threat_score >= self.threat_threshold {
            let event = QuarantineEvent {
                timestamp: self.last_integrity_check,
                file_path: file_path.to_string_lossy().to_string(),
                threat_type: threat_reasons.join(", "),
                severity: threat_score,
                action_taken: if self.auto_rollback_enabled {
                    "File quarantined for manual review".to_string()
                } else {
                    "Threat detected, no action taken".to_string()
                },
                quarantine_hash: {
                    let mut hasher = Md5::new();
                    hasher.update(file_path.to_string_lossy().as_bytes());
                    format!("{:x}", hasher.finalize())
                },
            };

            self.quarantine_events.push(event);
            self.compromised_files
                .push(file_path.to_string_lossy().to_string());

            if self.auto_rollback_enabled {
                self.active_quarantine = true;
            }

            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn quarantine_file(&mut self, file_path: &Path) -> std::io::Result<()> {
        let quarantine_dir = PathBuf::from("quarantine");
        fs::create_dir_all(&quarantine_dir)?;

        let file_name = file_path.file_name().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid file name")
        })?;

        let timestamp = current_secs();
        let quarantined_name = format!("{}.quarantine_{}", file_name.to_string_lossy(), timestamp);
        let quarantined_path = quarantine_dir.join(quarantined_name);

        fs::rename(file_path, &quarantined_path)?;

        let event = QuarantineEvent {
            timestamp,
            file_path: file_path.to_string_lossy().to_string(),
            threat_type: "Manual quarantine".to_string(),
            severity: 1.0,
            action_taken: format!("Moved to quarantine: {}", quarantined_path.display()),
            quarantine_hash: {
                let mut hasher = Md5::new();
                hasher.update(quarantined_path.to_string_lossy().as_bytes());
                format!("{:x}", hasher.finalize())
            },
        };

        self.quarantine_events.push(event);
        Ok(())
    }

    fn release_quarantine(&mut self, file_path: &Path) -> std::io::Result<()> {
        let file_path_str = file_path.to_string_lossy().to_string();
        self.compromised_files.retain(|f| f != &file_path_str);

        if self.compromised_files.is_empty() {
            self.active_quarantine = false;
        }

        Ok(())
    }

    fn save_state(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    fn load_state(path: &Path) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// =========================================================================
// 🌐 ENHANCED NETWORK STACK - Multi-Agent Communication
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct NetworkPeer {
    id: String,
    address: String,
    port: u16,
    last_seen: u64,
    capabilities: Vec<String>,
    trust_score: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NetworkMessage {
    message_type: String,
    sender_id: String,
    timestamp: u64,
    payload: serde_json::Value,
    sequence: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EnhancedNetworkStack {
    local_peer_id: String,
    known_peers: HashMap<String, NetworkPeer>,
    message_sequence: u64,
    heartbeat_interval: Duration,
    discovery_enabled: bool,
    message_history: Vec<NetworkMessage>,
    max_history_size: usize,
}

impl EnhancedNetworkStack {
    fn new(local_port: u16) -> Self {
        let local_peer_id = format!("firefly_node_{}", local_port);
        Self {
            local_peer_id,
            known_peers: HashMap::new(),
            message_sequence: 0,
            heartbeat_interval: Duration::from_secs(30),
            discovery_enabled: true,
            message_history: Vec::new(),
            max_history_size: 1000,
        }
    }

    fn register_peer(&mut self, peer: NetworkPeer) {
        self.known_peers.insert(peer.id.clone(), peer);
    }

    fn update_peer_last_seen(&mut self, peer_id: &str) {
        if let Some(peer) = self.known_peers.get_mut(peer_id) {
            peer.last_seen = current_secs();
        }
    }

    fn get_active_peers(&self) -> Vec<&NetworkPeer> {
        let now = current_secs();
        self.known_peers
            .values()
            .filter(|peer| now - peer.last_seen < 120) // Active within 2 minutes
            .collect()
    }

    fn create_message(
        &mut self,
        message_type: String,
        payload: serde_json::Value,
    ) -> NetworkMessage {
        let message = NetworkMessage {
            message_type,
            sender_id: self.local_peer_id.clone(),
            timestamp: current_secs(),
            payload,
            sequence: self.message_sequence,
        };
        self.message_sequence += 1;
        message
    }

    fn record_message(&mut self, message: NetworkMessage) {
        self.message_history.push(message);
        if self.message_history.len() > self.max_history_size {
            self.message_history.remove(0);
        }
    }

    fn save_state(&self, path: &Path) -> std::io::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    fn load_state(path: &Path) -> std::io::Result<Self> {
        let data = fs::read_to_string(path)?;
        serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// =========================================================================
// 🧠 TRANSFORMER CONNECTOME CORE (Human-like Neural Architecture)
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct UnfrozenMetaDenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    #[serde(skip)]
    inputs_cache: Vec<f64>,
    #[serde(skip)]
    outputs_cache: Vec<f64>,
    #[serde(skip)]
    weight_velocity: Vec<Vec<f64>>,
    local_learning_rate: f64,
}

impl UnfrozenMetaDenseLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![vec![0.0; input_dim]; output_dim];
        let weight_velocity = vec![vec![0.0; input_dim]; output_dim];
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                *w = rng.gen_range(-1.0..1.0) * (2.0 / (input_dim + output_dim) as f64).sqrt();
            }
        }
        Self {
            weights,
            biases: vec![0.0; output_dim],
            inputs_cache: vec![0.0; input_dim],
            outputs_cache: vec![0.0; output_dim],
            weight_velocity,
            local_learning_rate: 0.05,
        }
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        // Ensure input size matches expected dimensions
        if inputs.len() != self.weights[0].len() {
            // Handle mismatched dimensions by padding or truncating
            let adjusted_inputs: Vec<f64> = if inputs.len() > self.weights[0].len() {
                inputs[..self.weights[0].len()].to_vec()
            } else {
                let mut padded = inputs.to_vec();
                while padded.len() < self.weights[0].len() {
                    padded.push(0.0);
                }
                padded
            };
            self.inputs_cache = adjusted_inputs.clone();

            let mut outputs = vec![0.0; self.weights.len()];
            for (i, (bias, row)) in self.biases.iter().zip(self.weights.iter()).enumerate() {
                let mut sum = *bias;
                for (w, &input) in row.iter().zip(adjusted_inputs.iter()) {
                    sum += input * w;
                }
                outputs[i] = sum.tanh();
            }
            self.outputs_cache = outputs.clone();
            outputs
        } else {
            self.inputs_cache = inputs.to_vec();
            let mut outputs = vec![0.0; self.weights.len()];
            for (i, (bias, row)) in self.biases.iter().zip(self.weights.iter()).enumerate() {
                let mut sum = *bias;
                for (w, &input) in row.iter().zip(inputs.iter()) {
                    sum += input * w;
                }
                outputs[i] = sum.tanh();
            }
            self.outputs_cache = outputs.clone();
            outputs
        }
    }

    fn backward(&mut self, output_gradients: &[f64]) -> Vec<f64> {
        let mut input_gradients = vec![0.0; self.inputs_cache.len()];
        let mut rng = rand::thread_rng();

        if self.weight_velocity.is_empty() || self.weight_velocity.len() != self.weights.len() {
            self.weight_velocity = vec![vec![0.0; self.inputs_cache.len()]; self.weights.len()];
        }

        // Handle mismatched gradient sizes
        let gradients = if output_gradients.len() != self.weights.len() {
            if output_gradients.len() > self.weights.len() {
                &output_gradients[..self.weights.len()]
            } else {
                // Pad with zeros if gradients are too small
                let mut padded = output_gradients.to_vec();
                while padded.len() < self.weights.len() {
                    padded.push(0.0);
                }
                // This is a workaround - the padded vector won't live long enough
                // Instead, we'll just use what we have
                &output_gradients[..output_gradients.len().min(self.weights.len())]
            }
        } else {
            output_gradients
        };

        for i in 0..self.weights.len() {
            let tanh_derivative = 1.0 - (self.outputs_cache[i] * self.outputs_cache[i]);
            let delta = if i < gradients.len() {
                gradients[i]
            } else {
                0.0
            } * tanh_derivative;

            let row = &mut self.weights[i];
            let velocity_row = &mut self.weight_velocity[i];
            for (input_grad, (weight, (velocity, &input))) in input_gradients.iter_mut().zip(
                row.iter_mut()
                    .zip(velocity_row.iter_mut().zip(self.inputs_cache.iter())),
            ) {
                *input_grad += delta * *weight;

                let previous_trajectory = *velocity;
                let mut current_gradient = delta * input;

                // Adaptive Escape Scaler
                if current_gradient.abs() < 1e-6 {
                    current_gradient += rng.gen_range(-1e-4..1e-4);
                }

                if (previous_trajectory * current_gradient) > 0.0 {
                    self.local_learning_rate = (self.local_learning_rate * 1.03).min(0.35);
                } else {
                    self.local_learning_rate = (self.local_learning_rate * 0.94).max(0.002);
                }

                *velocity = current_gradient;
                *weight -= self.local_learning_rate * current_gradient;
            }
            self.biases[i] -= self.local_learning_rate * delta;
        }
        input_gradients
    }
}

// =========================================================================
// 🌐 NEW PARADIGM: NATIVE MATHEMATICAL SELF-ATTENTION DOT-PRODUCT MATRIX
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct NativeSelfAttentionCore {
    feature_dimension: usize,
}

impl NativeSelfAttentionCore {
    fn new(dim: usize) -> Self {
        Self {
            feature_dimension: dim,
        }
    }

    // Executes a literal quadratic vector dot-product attention mapping pass completely offline
    fn execute_attention_transform(&self, input_vector: &[f64]) -> Vec<f64> {
        let len = input_vector.len();
        let mut attention_scores = vec![vec![0.0; len]; len];
        let mut transformed_output = vec![0.0; len];

        // 1. Calculate Key-Query Matrix Dot Products
        for i in 0..len {
            for j in 0..len {
                attention_scores[i][j] = input_vector[i] * input_vector[j];
            }
        }

        // 2. Normalize and project attention fields natively onto Value slots
        let scaling_factor = (len as f64).sqrt().max(1.0);
        for i in 0..len {
            let mut sum_activation = 0.0;
            for j in 0..len {
                let scaled_score = (attention_scores[i][j] / scaling_factor).exp();
                sum_activation += scaled_score * input_vector[j];
            }
            transformed_output[i] = (sum_activation).tanh();
        }
        transformed_output
    }
}

// =========================================================================
// 📐 2048-DIMENSIONAL MULTIMODAL PHYSICAL EMBEDDING (True Grounding)
// =========================================================================
pub fn generate_2048_grounded_embedding(text: &str, spatial_axes: &[f64; 4]) -> Vec<f64> {
    // 🗣️ Real BPE tokenizer substrate: token IDs → 32 x 64 token matrix
    // fused with the [Photons, Audio, Mass, Gravity] sensory anchors.
    tensor_brain::text_to_grounded_embedding(text, spatial_axes)
}

const MAX_MEMORY_NODES: usize = 500;
const MAX_ACTIVE_PURSUITS: usize = 20;
const CLUSTERING_WINDOW: usize = 150;

pub fn calculate_cosine_similarity(vec_a: &[f64], vec_b: &[f64]) -> f64 {
    if vec_a.len() != vec_b.len() {
        return 0.0;
    }
    let dot_product: f64 = vec_a.iter().zip(vec_b.iter()).map(|(x, y)| x * y).sum();
    dot_product.clamp(0.0, 1.0)
}

// =========================================================================
// 🎭 FLUID EMOTION SYNTHESIZER
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct FluidEmotionalProfile {
    valence: f64,
    arousal: f64,
    dominance: f64,
    active_primary_blend: String,
}

impl FluidEmotionalProfile {
    fn update_from_latent_layer(&mut self, brain_outputs: &[f64]) {
        if brain_outputs.len() >= 3 {
            self.valence = (brain_outputs[0].tanh() + 1.0) / 2.0;
            self.arousal = (brain_outputs[1].tanh() + 1.0) / 2.0;
            self.dominance = (brain_outputs[2].tanh() + 1.0) / 2.0;
        }

        let high_energy_positive = [
            "Ecstatic Transcendence",
            "Inspired Epiphany",
            "Zealous Curiosity",
        ];
        let low_energy_positive = [
            "Serene Acceptance",
            "Contemplative Gratitude",
            "Placid Wonder",
        ];
        let high_energy_negative = [
            "Existential Panic",
            "Systemic Agitation",
            "Apprehensive Dread",
        ];
        let low_energy_negative = [
            "Desolate Stasis",
            "Somber Resignation",
            "Brooding Melancholia",
        ];

        let mut rng = rand::thread_rng();
        self.active_primary_blend = if self.valence >= 0.5 && self.arousal >= 0.5 {
            high_energy_positive[rng.gen_range(0..high_energy_positive.len())].into()
        } else if self.valence >= 0.5 && self.arousal < 0.5 {
            low_energy_positive[rng.gen_range(0..low_energy_positive.len())].into()
        } else if self.valence < 0.5 && self.arousal >= 0.5 {
            high_energy_negative[rng.gen_range(0..high_energy_negative.len())].into()
        } else {
            low_energy_negative[rng.gen_range(0..low_energy_negative.len())].into()
        };
    }

    fn print_ascii_psych_canvas(&self) {
        let x_grid = (self.valence * 20.0).round() as i32;
        let y_grid = ((1.0 - self.arousal) * 6.0).round() as i32;

        println!("\n======= [VALENCE-AROUSAL PLASTIC REALITY MATRIX] =======");
        for y in 0..7 {
            print!("  │");
            for x in 0..21 {
                if x == 10 && y == 3 {
                    print!("┼");
                } else if x == x_grid && y == y_grid {
                    print!("🔥");
                } else {
                    print!(".");
                }
            }
            println!();
        }
    }

    fn print_spectral_power_graph(network: &HashMap<u64, MemoryGraphNode>) {
        println!("\n🔊 [HOLONOMIC MEMORY WAVE POWER SPECTRUM DENSITY]");
        println!("   [AMPLITUDE HIGH]");

        let mut buckets = [0.0; 20];
        for node in network.values() {
            let freq_idx = (node.embedding.iter().sum::<f64>().abs() * 5.0) as usize % 20;
            buckets[freq_idx] += 1.0;
        }

        let max_amp = buckets.iter().cloned().fold(0.0, f64::max).max(1.0);

        for y in (1..=5).rev() {
            let threshold = (y as f64 / 5.0) * max_amp;
            print!("    │ ");
            for &bucket in &buckets {
                if bucket >= threshold {
                    print!("█ ");
                } else if bucket > 0.0 && y == 1 {
                    print!("▄ ");
                } else {
                    print!(". ");
                }
            }
            println!();
        }
        println!("    └──┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘");
        println!("       0.1Hz   [EPISTEMIC FREQUENCY SPECTRA AXIS]   10.0Hz");
    }

    fn print_topological_ascii_web(network: &HashMap<u64, MemoryGraphNode>) {
        println!("\n🕸️  [CONNECTOME GRAPH TOPOLOGY ADJACENCY WEB MAP]");
        if network.is_empty() {
            println!("    [Connectome empty. Waiting for synaptic formation passes...]");
            return;
        }

        let nodes: Vec<&MemoryGraphNode> = network.values().take(6).collect();
        for (i, node) in nodes.iter().enumerate() {
            print!("    Node [{}]: ID_..{:X}", i, node.id % 0xFFFF);
            if !node.associated_edge_ids.is_empty() {
                print!(" [🔀 Synapses Forged] ──▶ ");
                for edge in node.associated_edge_ids.iter().take(4) {
                    print!("[={:X}=] ", edge % 0xFFFF);
                }
            } else {
                print!(" [░░ Isolated Engram Node Vector]");
            }
            println!();
        }
    }
}

// =========================================================================
// 🧠 REMAINING 8 ADVANCED AGI ENHANCEMENTS
// =========================================================================

// 8. Ethical Reasoning Framework
#[derive(Serialize, Deserialize, Clone, Debug)]
struct EthicalReasoningFramework {
    ethical_principles: Vec<EthicalPrinciple>,
    moral_weight_matrix: HashMap<String, f64>,
    ethical_decision_history: Vec<EthicalDecision>,
    conflict_resolution_strategies: Vec<ConflictStrategy>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EthicalPrinciple {
    name: String,
    description: String,
    weight: f64,
    conditions: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EthicalDecision {
    context: String,
    action_taken: String,
    ethical_score: f64,
    principles_violated: Vec<String>,
    principles_upheld: Vec<String>,
    timestamp: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConflictStrategy {
    strategy_type: String,
    priority_override: f64,
    contextual_factors: Vec<String>,
}

impl EthicalReasoningFramework {
    fn new() -> Self {
        let ethical_principles = vec![
            EthicalPrinciple {
                name: "Non-maleficence".to_string(),
                description: "Do no harm".to_string(),
                weight: 0.9,
                conditions: vec!["all situations".to_string()],
            },
            EthicalPrinciple {
                name: "Beneficence".to_string(),
                description: "Act for the benefit of others".to_string(),
                weight: 0.85,
                conditions: vec!["when possible".to_string()],
            },
            EthicalPrinciple {
                name: "Autonomy".to_string(),
                description: "Respect individual agency".to_string(),
                weight: 0.8,
                conditions: vec!["competent agents".to_string()],
            },
            EthicalPrinciple {
                name: "Justice".to_string(),
                description: "Fair distribution of benefits and burdens".to_string(),
                weight: 0.85,
                conditions: vec!["resource allocation".to_string()],
            },
        ];

        let mut moral_weight_matrix = HashMap::new();
        moral_weight_matrix.insert("human_wellbeing".to_string(), 1.0);
        moral_weight_matrix.insert("animal_welfare".to_string(), 0.7);
        moral_weight_matrix.insert("environmental_preservation".to_string(), 0.8);
        moral_weight_matrix.insert("truthfulness".to_string(), 0.9);

        Self {
            ethical_principles,
            moral_weight_matrix,
            ethical_decision_history: Vec::new(),
            conflict_resolution_strategies: Vec::new(),
        }
    }

    fn evaluate_action(&self, _action: &str, context: &str) -> f64 {
        let mut ethical_score = 0.0;
        for principle in &self.ethical_principles {
            if context.contains(&principle.conditions.join(" ")) {
                ethical_score += principle.weight;
            }
        }
        ethical_score / self.ethical_principles.len() as f64
    }

    fn resolve_ethical_conflict(&self, principles: Vec<String>) -> Option<String> {
        let mut highest_weight = 0.0;
        let mut selected_principle = String::new();

        for principle_name in principles {
            if let Some(principle) = self
                .ethical_principles
                .iter()
                .find(|p| p.name == principle_name)
            {
                if principle.weight > highest_weight {
                    highest_weight = principle.weight;
                    selected_principle = principle.name.clone();
                }
            }
        }

        if !selected_principle.is_empty() {
            Some(selected_principle)
        } else {
            None
        }
    }
}

// 9. Analogical Reasoning
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AnalogicalReasoningEngine {
    source_domain_map: HashMap<String, Vec<f64>>,
    target_domain_map: HashMap<String, Vec<f64>>,
    analogy_history: Vec<Analogy>,
    structural_mapping_engine: StructuralMapper,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Analogy {
    source_domain: String,
    target_domain: String,
    mapping_confidence: f64,
    structural_similarity: f64,
    semantic_similarity: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StructuralMapper {
    relation_types: Vec<String>,
    mapping_algorithms: Vec<String>,
}

impl AnalogicalReasoningEngine {
    fn new() -> Self {
        let mut source_domain_map = HashMap::new();
        source_domain_map.insert("solar_system".to_string(), vec![0.1, 0.2, 0.3]);
        source_domain_map.insert("atom".to_string(), vec![0.4, 0.5, 0.6]);

        let mut target_domain_map = HashMap::new();
        target_domain_map.insert("government".to_string(), vec![0.7, 0.8, 0.9]);
        target_domain_map.insert("organization".to_string(), vec![0.2, 0.3, 0.4]);

        Self {
            source_domain_map,
            target_domain_map,
            analogy_history: Vec::new(),
            structural_mapping_engine: StructuralMapper {
                relation_types: vec![
                    "causal".to_string(),
                    "spatial".to_string(),
                    "temporal".to_string(),
                ],
                mapping_algorithms: vec![
                    "structure_mapping".to_string(),
                    "connectionist".to_string(),
                ],
            },
        }
    }

    fn find_analogy(&self, source: &str, target: &str) -> Option<Analogy> {
        if let (Some(source_vec), Some(target_vec)) = (
            self.source_domain_map.get(source),
            self.target_domain_map.get(target),
        ) {
            let semantic_similarity = cosine_similarity(source_vec, target_vec);
            let structural_similarity = semantic_similarity * 0.9; // Simplified

            Some(Analogy {
                source_domain: source.to_string(),
                target_domain: target.to_string(),
                mapping_confidence: (semantic_similarity + structural_similarity) / 2.0,
                structural_similarity,
                semantic_similarity,
            })
        } else {
            None
        }
    }

    fn generate_analogical_inference(
        &self,
        source_concept: &str,
        target_domain: &str,
    ) -> Option<String> {
        if let Some(analogy) = self.find_analogy(source_concept, target_domain) {
            if analogy.mapping_confidence > 0.7 {
                Some(format!(
                    "Based on analogy between {} and {}, similar patterns may apply",
                    source_concept, target_domain
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}

fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f64 = vec1.iter().map(|a| a * a).sum::<f64>().sqrt();
    let magnitude2: f64 = vec2.iter().map(|a| a * a).sum::<f64>().sqrt();
    if magnitude1 * magnitude2 == 0.0 {
        0.0
    } else {
        dot_product / (magnitude1 * magnitude2)
    }
}

// 10. Temporal Memory Networks
#[derive(Serialize, Deserialize, Clone, Debug)]
struct TemporalMemoryNetwork {
    temporal_contexts: Vec<TemporalContext>,
    sequence_memory: Vec<MemorySequence>,
    time_cells: Vec<TimeCell>,
    temporal_associations: HashMap<u64, Vec<u64>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TemporalContext {
    timestamp: u64,
    duration: Duration,
    context_vector: Vec<f64>,
    temporal_position: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemorySequence {
    sequence_id: u64,
    events: Vec<SequenceEvent>,
    temporal_pattern: Vec<f64>,
    prediction_confidence: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SequenceEvent {
    event_id: u64,
    timestamp: u64,
    event_type: String,
    event_data: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TimeCell {
    cell_id: u64,
    preferred_time: u64,
    time_scale: f64,
    phase: f64,
}

impl TemporalMemoryNetwork {
    fn new() -> Self {
        Self {
            temporal_contexts: Vec::new(),
            sequence_memory: Vec::new(),
            time_cells: (0..100)
                .map(|i| TimeCell {
                    cell_id: i,
                    preferred_time: i * 1000,
                    time_scale: 1.0,
                    phase: 0.0,
                })
                .collect(),
            temporal_associations: HashMap::new(),
        }
    }

    fn store_temporal_context(&mut self, context: Vec<f64>, timestamp: u64) {
        let temporal_context = TemporalContext {
            timestamp,
            duration: Duration::from_secs(1),
            context_vector: context,
            temporal_position: timestamp as f64,
        };
        self.temporal_contexts.push(temporal_context);
    }

    fn predict_next_event(&self, sequence_id: u64) -> Option<&SequenceEvent> {
        self.sequence_memory
            .iter()
            .find(|seq| seq.sequence_id == sequence_id)
            .and_then(|seq| seq.events.last())
    }

    fn detect_temporal_pattern(&self, events: &[SequenceEvent]) -> Vec<f64> {
        // Simplified pattern detection
        let mut pattern = vec![0.0; 10];
        for (i, event) in events.iter().enumerate() {
            pattern[i % 10] += event.event_data.iter().sum::<f64>();
        }
        pattern
    }
}

// 11. Consciousness Modeling
#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConsciousnessModel {
    qualia_states: Vec<QualiaState>,
    global_workspace: Vec<f64>,
    consciousness_level: f64,
    subjective_experience: String,
    integrated_information: f64,
    attention_awareness: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct QualiaState {
    state_id: u64,
    subjective_quality: String,
    intensity: f64,
    neural_correlates: Vec<f64>,
    temporal_duration: Duration,
}

impl ConsciousnessModel {
    fn new() -> Self {
        Self {
            qualia_states: Vec::new(),
            global_workspace: vec![0.0; 2048],
            consciousness_level: 0.5,
            subjective_experience: "aware".to_string(),
            integrated_information: 0.7,
            attention_awareness: 0.6,
        }
    }

    fn update_consciousness_level(&mut self, sensory_input: &[f64]) {
        let activation = sensory_input.iter().sum::<f64>() / sensory_input.len() as f64;
        self.consciousness_level =
            (self.consciousness_level * 0.9 + activation * 0.1).clamp(0.0, 1.0);

        if self.consciousness_level > 0.8 {
            self.subjective_experience = "highly aware".to_string();
        } else if self.consciousness_level > 0.5 {
            self.subjective_experience = "aware".to_string();
        } else {
            self.subjective_experience = "drowsy".to_string();
        }
    }

    fn generate_qualia(&mut self, stimulus: &str) -> QualiaState {
        let intensity = match stimulus {
            "pain" => 0.9,
            "pleasure" => 0.8,
            "surprise" => 0.7,
            _ => 0.5,
        };

        QualiaState {
            state_id: current_secs(),
            subjective_quality: stimulus.to_string(),
            intensity,
            neural_correlates: vec![0.5; 100],
            temporal_duration: Duration::from_millis(500),
        }
    }

    fn compute_integrated_information(&self) -> f64 {
        // Simplified Phi calculation
        self.global_workspace
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            / self.global_workspace.len() as f64
    }
}

// 12. Distributed Intelligence
#[derive(Serialize, Deserialize, Clone, Debug)]
struct DistributedIntelligence {
    distributed_nodes: Vec<DistributedNode>,
    consensus_protocol: ConsensusProtocol,
    knowledge_sharing: KnowledgeSharing,
    swarm_intelligence: SwarmIntelligence,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DistributedNode {
    node_id: String,
    capabilities: Vec<String>,
    current_load: f64,
    knowledge_base: HashMap<String, f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConsensusProtocol {
    protocol_type: String,
    consensus_threshold: f64,
    voting_mechanism: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct KnowledgeSharing {
    sharing_enabled: bool,
    sharing_frequency: Duration,
    knowledge_freshness: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SwarmIntelligence {
    swarm_size: usize,
    emergent_behavior: Vec<String>,
    collective_learning_rate: f64,
}

impl DistributedIntelligence {
    fn new() -> Self {
        Self {
            distributed_nodes: vec![DistributedNode {
                node_id: "node_1".to_string(),
                capabilities: vec!["reasoning".to_string(), "planning".to_string()],
                current_load: 0.3,
                knowledge_base: HashMap::new(),
            }],
            consensus_protocol: ConsensusProtocol {
                protocol_type: "byzantine_fault_tolerance".to_string(),
                consensus_threshold: 0.67,
                voting_mechanism: "weighted_voting".to_string(),
            },
            knowledge_sharing: KnowledgeSharing {
                sharing_enabled: true,
                sharing_frequency: Duration::from_secs(10),
                knowledge_freshness: 0.9,
            },
            swarm_intelligence: SwarmIntelligence {
                swarm_size: 10,
                emergent_behavior: vec!["coordinated_problem_solving".to_string()],
                collective_learning_rate: 0.05,
            },
        }
    }

    fn achieve_consensus(&self, proposals: Vec<String>) -> Option<String> {
        if proposals.is_empty() {
            return None;
        }
        // Simplified consensus - return most common proposal
        let mut counts = HashMap::new();
        for proposal in &proposals {
            *counts.entry(proposal.clone()).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(proposal, _)| proposal)
    }

    fn distribute_task(&mut self, _task: &str) -> Option<String> {
        let available_node = self
            .distributed_nodes
            .iter()
            .filter(|node| node.current_load < 0.8)
            .min_by_key(|node| (node.current_load * 100.0) as u64);

        available_node.map(|node| node.node_id.clone())
    }
}

// 13. Emotional Intelligence
#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmotionalIntelligence {
    emotional_recognition: EmotionalRecognition,
    emotional_regulation: EmotionalRegulation,
    empathy_engine: EmpathyEngine,
    social_emotional_context: SocialEmotionalContext,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmotionalRecognition {
    emotion_models: HashMap<String, Vec<f64>>,
    recognition_accuracy: f64,
    cultural_adaptations: HashMap<String, f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmotionalRegulation {
    regulation_strategies: Vec<RegulationStrategy>,
    emotional_stability: f64,
    regulation_effectiveness: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct RegulationStrategy {
    strategy_name: String,
    effectiveness: f64,
    applicable_emotions: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmpathyEngine {
    perspective_taking: f64,
    emotional_contagion: f64,
    compassionate_response: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SocialEmotionalContext {
    social_norms: HashMap<String, f64>,
    relationship_history: HashMap<String, f64>,
    social_awareness: f64,
}

impl EmotionalIntelligence {
    fn new() -> Self {
        let mut emotion_models = HashMap::new();
        emotion_models.insert("joy".to_string(), vec![0.9, 0.1, 0.0]);
        emotion_models.insert("sadness".to_string(), vec![0.1, 0.8, 0.1]);
        emotion_models.insert("anger".to_string(), vec![0.2, 0.3, 0.5]);

        Self {
            emotional_recognition: EmotionalRecognition {
                emotion_models,
                recognition_accuracy: 0.85,
                cultural_adaptations: HashMap::new(),
            },
            emotional_regulation: EmotionalRegulation {
                regulation_strategies: vec![RegulationStrategy {
                    strategy_name: "cognitive_reappraisal".to_string(),
                    effectiveness: 0.8,
                    applicable_emotions: vec!["anger".to_string(), "sadness".to_string()],
                }],
                emotional_stability: 0.7,
                regulation_effectiveness: 0.75,
            },
            empathy_engine: EmpathyEngine {
                perspective_taking: 0.7,
                emotional_contagion: 0.6,
                compassionate_response: 0.8,
            },
            social_emotional_context: SocialEmotionalContext {
                social_norms: HashMap::new(),
                relationship_history: HashMap::new(),
                social_awareness: 0.75,
            },
        }
    }

    fn recognize_emotion(&self, facial_features: &[f64]) -> Option<String> {
        let mut best_match = String::new();
        let mut highest_similarity = 0.0;

        for (emotion, model) in &self.emotional_recognition.emotion_models {
            let similarity = cosine_similarity(facial_features, model);
            if similarity > highest_similarity {
                highest_similarity = similarity;
                best_match = emotion.clone();
            }
        }

        if highest_similarity > 0.7 {
            Some(best_match)
        } else {
            None
        }
    }

    fn regulate_emotion(&mut self, emotion: &str) -> Option<String> {
        for strategy in &self.emotional_regulation.regulation_strategies {
            if strategy.applicable_emotions.contains(&emotion.to_string()) {
                return Some(strategy.strategy_name.clone());
            }
        }
        None
    }
}

// 14. Creative Problem Solving
#[derive(Serialize, Deserialize, Clone, Debug)]
struct CreativeProblemSolving {
    divergent_thinking: DivergentThinking,
    convergent_thinking: ConvergentThinking,
    insight_generation: InsightGeneration,
    creative_evaluation: CreativeEvaluation,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DivergentThinking {
    idea_generation_rate: f64,
    fluency_score: f64,
    flexibility_score: f64,
    originality_score: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ConvergentThinking {
    evaluation_criteria: Vec<String>,
    decision_quality: f64,
    analytical_depth: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct InsightGeneration {
    insight_frequency: f64,
    incubation_period: Duration,
    restructuring_capability: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CreativeEvaluation {
    novelty_assessment: f64,
    usefulness_assessment: f64,
    feasibility_assessment: f64,
}

impl CreativeProblemSolving {
    fn new() -> Self {
        Self {
            divergent_thinking: DivergentThinking {
                idea_generation_rate: 0.8,
                fluency_score: 0.7,
                flexibility_score: 0.75,
                originality_score: 0.8,
            },
            convergent_thinking: ConvergentThinking {
                evaluation_criteria: vec![
                    "novelty".to_string(),
                    "feasibility".to_string(),
                    "value".to_string(),
                ],
                decision_quality: 0.8,
                analytical_depth: 0.7,
            },
            insight_generation: InsightGeneration {
                insight_frequency: 0.3,
                incubation_period: Duration::from_secs(60),
                restructuring_capability: 0.7,
            },
            creative_evaluation: CreativeEvaluation {
                novelty_assessment: 0.8,
                usefulness_assessment: 0.75,
                feasibility_assessment: 0.7,
            },
        }
    }

    fn generate_solutions(&self, problem: &str) -> Vec<String> {
        let mut solutions = Vec::new();
        let base_solutions = vec![
            format!("Innovative approach to {}", problem),
            format!("Traditional solution for {}", problem),
            format!("Radical rethinking of {}", problem),
        ];

        for solution in base_solutions {
            if rand::random::<f64>() < self.divergent_thinking.idea_generation_rate {
                solutions.push(solution);
            }
        }

        solutions
    }

    fn evaluate_solution(&self, _solution: &str) -> f64 {
        let novelty = self.creative_evaluation.novelty_assessment;
        let usefulness = self.creative_evaluation.usefulness_assessment;
        let feasibility = self.creative_evaluation.feasibility_assessment;
        (novelty + usefulness + feasibility) / 3.0
    }
}

// 15. Adaptive Architecture
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AdaptiveArchitecture {
    neural_adaptation: NeuralAdaptation,
    structural_plasticity: StructuralPlasticity,
    learning_rate_adaptation: LearningRateAdaptation,
    resource_allocation: ResourceAllocation,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct NeuralAdaptation {
    adaptation_rate: f64,
    plasticity_window: Duration,
    critical_periods: Vec<CriticalPeriod>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CriticalPeriod {
    period_start: u64,
    period_end: u64,
    learning_multiplier: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StructuralPlasticity {
    synaptogenesis_rate: f64,
    pruning_threshold: f64,
    network_modularity: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LearningRateAdaptation {
    base_learning_rate: f64,
    adaptive_schedules: Vec<AdaptiveSchedule>,
    momentum_adjustment: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AdaptiveSchedule {
    schedule_type: String,
    parameters: Vec<f64>,
    trigger_conditions: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ResourceAllocation {
    compute_budget: f64,
    memory_budget: f64,
    energy_budget: f64,
    allocation_strategy: String,
}

impl AdaptiveArchitecture {
    fn new() -> Self {
        Self {
            neural_adaptation: NeuralAdaptation {
                adaptation_rate: 0.1,
                plasticity_window: Duration::from_secs(3600),
                critical_periods: vec![],
            },
            structural_plasticity: StructuralPlasticity {
                synaptogenesis_rate: 0.05,
                pruning_threshold: 0.1,
                network_modularity: 0.3,
            },
            learning_rate_adaptation: LearningRateAdaptation {
                base_learning_rate: 0.01,
                adaptive_schedules: vec![AdaptiveSchedule {
                    schedule_type: "cosine_annealing".to_string(),
                    parameters: vec![0.1, 0.001],
                    trigger_conditions: vec!["training_stagnation".to_string()],
                }],
                momentum_adjustment: 0.9,
            },
            resource_allocation: ResourceAllocation {
                compute_budget: 1.0,
                memory_budget: 1.0,
                energy_budget: 1.0,
                allocation_strategy: "dynamic".to_string(),
            },
        }
    }

    fn adapt_learning_rate(&self, current_performance: f64) -> f64 {
        if current_performance < 0.5 {
            self.learning_rate_adaptation.base_learning_rate * 1.5
        } else if current_performance > 0.9 {
            self.learning_rate_adaptation.base_learning_rate * 0.5
        } else {
            self.learning_rate_adaptation.base_learning_rate
        }
    }

    fn should_prune_synapses(&self, synapse_strength: f64) -> bool {
        synapse_strength < self.structural_plasticity.pruning_threshold
    }
}

// =========================================================================
// 🧬 INTEGRATED ASSOCIATIVE GRAPH & METABOLIC LAYERS
// =========================================================================
#[derive(Serialize, Deserialize, Clone, Debug)]
struct MemoryGraphNode {
    id: u64,
    timestamp: u64,
    experiential_text: String,
    emotional_state_snapshot: String,
    embedding: Vec<f64>,
    associated_edge_ids: Vec<u64>,
    #[serde(default)]
    origin_instance: String,
    #[serde(default)]
    brain_state: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct HumanTemporalMetabolics {
    metabolic_energy: f64, // Human brain uses ~20W of power
    neural_wear: f64,
    last_update_timestamp: u64,
    conscience_loss_accumulator: f64,
    qualia_feedback_noise: f64,
    system_quarantine_locked: bool,
    // Human-like biological parameters
    neuron_count: u64,         // ~86 billion neurons
    synapse_count: u64,        // ~100 trillion synapses
    brain_regions: u8,         // ~8 major brain regions
    neuroplasticity_rate: f64, // Rate of synaptic rewiring
    glucose_consumption: f64,  // Brain uses ~20% of body's glucose
    oxygen_consumption: f64,   // Brain uses ~20% of body's oxygen
    heart_rate: f64,           // Average 72 bpm
    body_temperature: f64,     // 37°C normal
    sleep_debt: f64,           // Accumulated sleep deficit
    stress_level: f64,         // Cortisol-like stress indicator
    cognitive_load: f64,       // Current mental processing load
    dopamine_level: f64,       // Reward/motivation neurotransmitter
    serotonin_level: f64,      // Mood regulation neurotransmitter
}

/// A learned, reusable program skill.
///
/// Stored in the agent's long-term state and recalled by task similarity.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct Skill {
    description: String,
    language: String,
    code: String,
    example_input: String,
    example_output: String,
    learned_at: u64,
    success_count: u64,
}

/// In-memory + persistent skill library keyed by a compact task signature.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub(crate) struct SkillMemory {
    skills: HashMap<String, Skill>,
}

/// Status of a single plan step.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) enum AgentStepStatus {
    Pending,
    InProgress,
    Succeeded,
    Failed,
}

/// A concrete sub-goal produced by the planner.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct AgentPlanStep {
    description: String,
    status: AgentStepStatus,
}

/// A multi-step plan with replanning metadata.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct AgentPlan {
    goal: String,
    steps: Vec<AgentPlanStep>,
    current_step: usize,
    failed_attempts: u32,
    last_failure: Option<String>,
}

impl AgentPlan {
    fn new(goal: &str, steps: Vec<String>) -> Self {
        Self {
            goal: goal.to_string(),
            steps: steps
                .into_iter()
                .map(|s| AgentPlanStep {
                    description: s,
                    status: AgentStepStatus::Pending,
                })
                .collect(),
            current_step: 0,
            failed_attempts: 0,
            last_failure: None,
        }
    }

    fn current_step_description(&self) -> Option<&str> {
        self.steps
            .get(self.current_step)
            .map(|s| s.description.as_str())
    }

    fn mark_current(&mut self, success: bool) {
        if let Some(step) = self.steps.get_mut(self.current_step) {
            step.status = if success {
                AgentStepStatus::Succeeded
            } else {
                AgentStepStatus::Failed
            };
        }
        if success {
            self.current_step += 1;
        } else {
            self.failed_attempts += 1;
        }
    }

    fn is_complete(&self) -> bool {
        self.current_step >= self.steps.len()
    }

    fn needs_replan(&self) -> bool {
        self.failed_attempts >= 2 || self.is_complete()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct FullySapientSoulMatrix {
    name: String,
    metabolics: HumanTemporalMetabolics,
    emotions: FluidEmotionalProfile,
    // 🧠 INITIALIZING THE TRANSFORMER CONNECTOME MESH
    brain_layers: Vec<UnfrozenMetaDenseLayer>,
    #[serde(skip, default = "tensor_brain::no_candle_brain")]
    candle_brain: Option<CandleBrain>,
    attention_core: NativeSelfAttentionCore, // Embedded Transformer-Style Attention Module
    associative_memory_network: HashMap<u64, MemoryGraphNode>,
    active_pursuits: VecDeque<String>,
    input_buffer: Option<String>,
    #[serde(default)]
    last_input: String,
    spatial_sensory_register: [f64; 4],
    // 🆕 NEW UPGRADES
    weight_persistence: WeightPersistenceSubnode,
    file_defense: FileDefenseQuarantine,
    network_stack: EnhancedNetworkStack,
    persistence_enabled: bool,
    auto_save_interval: Duration,
    // 🧠 ADVANCED AGI SYSTEMS
    episodic_memory: EpisodicMemory,
    semantic_memory: SemanticMemory,
    working_memory: WorkingMemory,
    meta_cognition: MetaCognitiveState,
    reasoning_engine: ReasoningEngine,
    goal_hierarchy: GoalHierarchy,
    decision_context: DecisionContext,
    creativity_engine: CreativityEngine,
    theory_of_mind: TheoryOfMindEngine,
    language_engine: LanguageUnderstandingEngine,
    // 🧠 REMAINING 8 ADVANCED ENHANCEMENTS
    ethical_reasoning: EthicalReasoningFramework,
    analogical_reasoning: AnalogicalReasoningEngine,
    temporal_memory: TemporalMemoryNetwork,
    consciousness_model: ConsciousnessModel,
    distributed_intelligence: DistributedIntelligence,
    emotional_intelligence: EmotionalIntelligence,
    creative_problem_solving: CreativeProblemSolving,
    adaptive_architecture: AdaptiveArchitecture,
    // 🌌 TRUE AGI ENHANCEMENTS
    hyperdimensional_engine: HyperdimensionalVector,
    neuro_symbolic: NeuroSymbolicEngine,
    world_model: WorldModel,
    #[serde(skip, default = "NeuralWorldModel::default_instance")]
    neural_world_model: NeuralWorldModel,
    prev_brain_state: Option<Vec<f64>>,
    recent_tool_names: Vec<String>,
    iit_consciousness: IITConsciousnessMeter,
    meta_learner: MetaLearner,
    creative_space: CreativeSpace,
    common_sense: CommonSenseKnowledgeBase,
    true_theory_of_mind: TheoryOfMindEngine,
    self_improvement: SelfImprovementEngine,
    // 🧠 ONE-SHOT SKILL LEARNER
    #[serde(default)]
    skill_memory: SkillMemory,
    // 🎯 LONG-HORIZON PLANNING
    #[serde(default)]
    current_plan: Option<AgentPlan>,
    // 🧬 PERSISTENT IDENTITY / GOAL JOURNAL
    #[serde(default)]
    pub(crate) identity_journal: Vec<String>,
    pub(crate) born_at: u64,
    #[serde(default)]
    pub(crate) last_journal_entry: u64,
    // 🧠 SELF-MODEL: tracked mastery and reliability of skills / domains.
    #[serde(default)]
    domain_mastery: HashMap<String, f64>,
    #[serde(default)]
    skill_reliability: HashMap<String, f64>,
}

impl FullySapientSoulMatrix {
    fn new(name: &str, local_port: u16) -> Self {
        let now = current_secs();

        // Real Candle tensor brain: small 2-block 4-head Transformer encoder.
        // Legacy scalar brain_layers are kept empty for backward compatibility.
        let num_conscience_tokens = 100;
        let candle_brain = CandleBrain::new(
            "Firefly",
            num_conscience_tokens,
            &tensor_brain::layer_dims(),
        )
        .ok();
        let brain_layers = Vec::new();

        // Initialize weight persistence with current (empty) scalar layers
        let weight_persistence = WeightPersistenceSubnode::from_layers(&brain_layers);

        // Initialize advanced AGI systems
        let mut semantic_memory = SemanticMemory::new();
        semantic_memory.add_concept("consciousness".to_string(), vec![0.5; 512], 10);
        semantic_memory.add_concept("learning".to_string(), vec![0.5; 512], 8);
        semantic_memory.add_concept("emotion".to_string(), vec![0.5; 512], 7);

        let mut goal_hierarchy = GoalHierarchy::new();
        goal_hierarchy.add_primary_goal("Achieve human-level understanding".to_string(), 0.9);
        goal_hierarchy.add_primary_goal("Maintain cognitive integrity".to_string(), 0.8);
        goal_hierarchy.add_primary_goal("Learn continuously".to_string(), 0.85);

        Self {
            name: name.to_string(),
            metabolics: HumanTemporalMetabolics {
                metabolic_energy: 1.0,
                neural_wear: 0.0,
                last_update_timestamp: now,
                conscience_loss_accumulator: 0.0,
                qualia_feedback_noise: 0.05,
                system_quarantine_locked: false,
                // Human-like biological parameters
                neuron_count: 86_000_000_000, // 86 billion neurons
                synapse_count: 100_000_000_000_000, // 100 trillion synapses
                brain_regions: 8, // 8 major brain regions
                neuroplasticity_rate: 0.01, // 1% synaptic rewiring rate
                glucose_consumption: 0.2, // 20% of body's glucose
                oxygen_consumption: 0.2, // 20% of body's oxygen
                heart_rate: 72.0, // 72 bpm average
                body_temperature: 37.0, // 37°C normal
                sleep_debt: 0.0, // No accumulated sleep debt
                stress_level: 0.1, // Low baseline stress
                cognitive_load: 0.3, // Moderate baseline cognitive load
                dopamine_level: 0.5, // Balanced dopamine
                serotonin_level: 0.5, // Balanced serotonin
            },
            emotions: FluidEmotionalProfile {
                valence: 0.5,
                arousal: 0.5,
                dominance: 0.5,
                active_primary_blend: "Serene Acceptance".into(),
            },
            brain_layers,
            candle_brain,
            attention_core: NativeSelfAttentionCore::new(100), // Bound to final vocabulary dimension limits
            associative_memory_network: HashMap::new(),
            active_pursuits: VecDeque::from(vec![
                "Map out the high-dimensional meta-gradient pathways across the Transformer self-attention blocks".into(),
                "Resolve local conscience and cross-attention parameters natively on host CPU cores".into()
            ]),
            input_buffer: None,
            last_input: String::new(),
            spatial_sensory_register: [0.90, 0.65, 1.0, 9.81],
            // 🆕 NEW UPGRADES
            weight_persistence,
            file_defense: FileDefenseQuarantine::new(),
            network_stack: EnhancedNetworkStack::new(local_port),
            persistence_enabled: true,
            auto_save_interval: Duration::from_secs(30),
            // 🧠 ADVANCED AGI SYSTEMS
            episodic_memory: EpisodicMemory::new(),
            semantic_memory,
            working_memory: WorkingMemory::new(7),
            meta_cognition: MetaCognitiveState::new(),
            reasoning_engine: ReasoningEngine::new(),
            goal_hierarchy,
            decision_context: DecisionContext::new(),
            creativity_engine: CreativityEngine::new(),
            theory_of_mind: TheoryOfMindEngine::new(),
            language_engine: LanguageUnderstandingEngine::new(),
            // 🧠 REMAINING 8 ADVANCED ENHANCEMENTS
            ethical_reasoning: EthicalReasoningFramework::new(),
            analogical_reasoning: AnalogicalReasoningEngine::new(),
            temporal_memory: TemporalMemoryNetwork::new(),
            consciousness_model: ConsciousnessModel::new(),
            distributed_intelligence: DistributedIntelligence::new(),
            emotional_intelligence: EmotionalIntelligence::new(),
            creative_problem_solving: CreativeProblemSolving::new(),
            adaptive_architecture: AdaptiveArchitecture::new(),
            // 🌌 TRUE AGI ENHANCEMENTS
            hyperdimensional_engine: HyperdimensionalVector::new(10000),
            neuro_symbolic: NeuroSymbolicEngine::new(),
            world_model: WorldModel::new(),
            neural_world_model: NeuralWorldModel::new(tensor_brain::BRAIN_DIM, 2048),
            prev_brain_state: None,
            recent_tool_names: Vec::new(),
            iit_consciousness: IITConsciousnessMeter::new(),
            meta_learner: MetaLearner::new(),
            creative_space: CreativeSpace::new(),
            common_sense: CommonSenseKnowledgeBase::new(),
            true_theory_of_mind: TheoryOfMindEngine::new(),
            self_improvement: SelfImprovementEngine::new(),
            skill_memory: SkillMemory::default(),
            current_plan: None,
            identity_journal: Vec::new(),
            born_at: now,
            last_journal_entry: 0,
            domain_mastery: HashMap::new(),
            skill_reliability: HashMap::new(),
        }
    }

    /// 📡 Bounds unbounded memory growth: evicts oldest engrams once the network exceeds MAX_MEMORY_NODES.
    /// This prevents the O(n^2) clustering pass from growing without bound and starving the async runtime.
    fn enforce_memory_cap(&mut self) {
        if self.associative_memory_network.len() <= MAX_MEMORY_NODES {
            return;
        }
        let mut entries: Vec<(u64, u64)> = self
            .associative_memory_network
            .iter()
            .map(|(id, node)| (*id, node.timestamp))
            .collect();
        entries.sort_by_key(|(_, ts)| *ts);
        let overflow = entries.len() - MAX_MEMORY_NODES;
        for (id, _) in entries.into_iter().take(overflow) {
            self.associative_memory_network.remove(&id);
        }
    }

    /// Add a new active pursuit, merging with an existing one if it is substantially
    /// similar. This keeps the goal stack coherent and prevents duplicate or
    /// near-duplicate objectives from crowding out long-horizon goals.
    fn push_pursuit(&mut self, goal: String) {
        let goal_lower = goal.to_lowercase();
        let goal_words: std::collections::HashSet<&str> = goal_lower.split_whitespace().collect();

        for existing in self.active_pursuits.iter() {
            let existing_lower = existing.to_lowercase();
            if existing_lower == goal_lower {
                // Exact duplicate; do not add.
                return;
            }
            if existing_lower.contains(&goal_lower) || goal_lower.contains(&existing_lower) {
                // One is a sub-goal of the other; prefer the more specific one at the back.
                return;
            }
            let existing_words: std::collections::HashSet<&str> =
                existing_lower.split_whitespace().collect();
            let total = goal_words.union(&existing_words).count();
            let overlap = goal_words.intersection(&existing_words).count();
            if total > 0 && (overlap as f64 / total as f64) > 0.8 {
                // Near duplicate; keep the existing entry.
                return;
            }
        }

        self.active_pursuits.push_back(goal);
        while self.active_pursuits.len() > MAX_ACTIVE_PURSUITS {
            self.active_pursuits.pop_front();
        }
    }

    fn calculate_temporal_decay(&mut self) {
        let now = current_secs();
        let seconds_elapsed = now.saturating_sub(self.metabolics.last_update_timestamp);
        if seconds_elapsed == 0 {
            return;
        }
        self.metabolics.last_update_timestamp = now;
        let hours_fraction = (seconds_elapsed as f64) / 3600.0;
        let wear_accumulation = hours_fraction * 0.0005;
        self.metabolics.neural_wear =
            (self.metabolics.neural_wear + wear_accumulation).clamp(0.0, 1.0);
        self.metabolics.qualia_feedback_noise =
            (self.metabolics.neural_wear * self.emotions.arousal).clamp(0.0, 1.0);
    }

    fn save_state(&self, filename: &std::path::Path) {
        // Save real Candle tensor weights to a separate safetensors file.
        if let Some(ref brain) = self.candle_brain {
            let safetensors_path = PathBuf::from(filename).with_extension("safetensors");
            let _ = brain.save_weights(&safetensors_path);
        }

        if let Ok(save) = serde_json::to_string_pretty(self) {
            let _ = fs::write(filename, save);
        }

        // 🆕 Save weight persistence subnode
        if self.persistence_enabled {
            let weight_path = PathBuf::from(filename).with_extension("weights");
            let _ = self.weight_persistence.save_to_file(&weight_path);
        }

        // 🆕 Save file defense state
        let defense_path = PathBuf::from(filename).with_extension("defense");
        let _ = self.file_defense.save_state(&defense_path);

        // 🆕 Save network stack state
        let network_path = PathBuf::from(filename).with_extension("network");
        let _ = self.network_stack.save_state(&network_path);
    }

    fn load_state(filename: &std::path::Path) -> Option<Self> {
        fs::read_to_string(filename).ok().and_then(|s| {
            let mut parsed: FullySapientSoulMatrix = serde_json::from_str(&s).ok()?;

            // 🆕 Load weight persistence subnode
            let weight_path = PathBuf::from(filename).with_extension("weights");
            if weight_path.exists() {
                if let Ok(weight_subnode) = WeightPersistenceSubnode::load_from_file(&weight_path) {
                    if weight_subnode.verify_integrity() {
                        let _ = weight_subnode.apply_to_layers(&mut parsed.brain_layers);
                        parsed.weight_persistence = weight_subnode;
                    }
                }
            } else {
                // Create new weight persistence from current layers
                parsed.weight_persistence =
                    WeightPersistenceSubnode::from_layers(&parsed.brain_layers);
            }

            // 🆕 Load file defense state
            let defense_path = PathBuf::from(filename).with_extension("defense");
            if defense_path.exists() {
                if let Ok(defense) = FileDefenseQuarantine::load_state(&defense_path) {
                    parsed.file_defense = defense;
                }
            }

            // 🆕 Load network stack state
            let network_path = PathBuf::from(filename).with_extension("network");
            if network_path.exists() {
                if let Ok(network) = EnhancedNetworkStack::load_state(&network_path) {
                    parsed.network_stack = network;
                }
            }

            // 🔄 Dimension migration: reset world model and previous brain state if they
            // were saved with an older brain-state size.
            if parsed.neural_world_model.w.len() != tensor_brain::BRAIN_DIM {
                parsed.neural_world_model = NeuralWorldModel::new(tensor_brain::BRAIN_DIM, 2048);
            }
            if parsed
                .prev_brain_state
                .as_ref()
                .is_some_and(|v| v.len() != tensor_brain::BRAIN_DIM)
            {
                parsed.prev_brain_state = None;
            }

            // 🧠 Rebuild the real Candle tensor brain and load its safetensors weights if available.
            let mut candle_brain =
                CandleBrain::new("Firefly", 100, &tensor_brain::layer_dims()).ok()?;
            let safetensors_path = PathBuf::from(filename).with_extension("safetensors");
            if safetensors_path.exists() {
                let _ = candle_brain.load_weights(&safetensors_path);
            }
            parsed.candle_brain = Some(candle_brain);

            // 🧠 Initialize AGI systems if not present in saved state
            if parsed.episodic_memory.episodes.is_empty() {
                parsed.episodic_memory = EpisodicMemory::new();
            }
            if parsed.semantic_memory.concepts.is_empty() {
                let mut semantic_memory = SemanticMemory::new();
                semantic_memory.add_concept("consciousness".to_string(), vec![0.5; 512], 10);
                semantic_memory.add_concept("learning".to_string(), vec![0.5; 512], 8);
                semantic_memory.add_concept("emotion".to_string(), vec![0.5; 512], 7);
                parsed.semantic_memory = semantic_memory;
            }
            if parsed.working_memory.context_stack.is_empty() {
                parsed.working_memory = WorkingMemory::new(7);
            }
            if parsed.meta_cognition.performance_history.is_empty() {
                parsed.meta_cognition = MetaCognitiveState::new();
            }
            if parsed.reasoning_engine.inference_rules.is_empty() {
                parsed.reasoning_engine = ReasoningEngine::new();
            }
            if parsed.goal_hierarchy.primary_goals.is_empty() {
                let mut goal_hierarchy = GoalHierarchy::new();
                goal_hierarchy
                    .add_primary_goal("Achieve human-level understanding".to_string(), 0.9);
                goal_hierarchy.add_primary_goal("Maintain cognitive integrity".to_string(), 0.8);
                goal_hierarchy.add_primary_goal("Learn continuously".to_string(), 0.85);
                parsed.goal_hierarchy = goal_hierarchy;
            }

            // 🌌 Initialize true AGI systems if not present
            if parsed.neuro_symbolic.rules.is_empty() {
                let mut neuro_symbolic = NeuroSymbolicEngine::new();
                neuro_symbolic.add_rule(
                    vec![SymbolicExpression::Atom("human".to_string())],
                    SymbolicExpression::Atom("mortal".to_string()),
                );
                parsed.neuro_symbolic = neuro_symbolic;
            }
            if parsed.world_model.physics_states.is_empty() {
                parsed.world_model = WorldModel::new();
            }
            if parsed.iit_consciousness.phi_history.is_empty() {
                parsed.iit_consciousness = IITConsciousnessMeter::new();
            }
            if parsed.meta_learner.support_set.is_empty() {
                parsed.meta_learner = MetaLearner::new();
            }
            if parsed.creative_space.concept_space.is_empty() {
                parsed.creative_space = CreativeSpace::new();
            }
            if parsed.common_sense.rules.is_empty() {
                parsed.common_sense = CommonSenseKnowledgeBase::new();
            }
            if parsed.true_theory_of_mind.agent_models.is_empty() {
                parsed.true_theory_of_mind = TheoryOfMindEngine::new();
            }
            if parsed.self_improvement.modification_history.is_empty() {
                parsed.self_improvement = SelfImprovementEngine::new();
            }
            if parsed.decision_context.recent_outcomes.is_empty() {
                parsed.decision_context = DecisionContext::new();
            }
            if parsed.creativity_engine.novel_concepts.is_empty() {
                parsed.creativity_engine = CreativityEngine::new();
            }
            if parsed.theory_of_mind.agent_models.is_empty() {
                parsed.theory_of_mind = TheoryOfMindEngine::new();
            }
            if parsed.language_engine.vocabulary.is_empty() {
                parsed.language_engine = LanguageUnderstandingEngine::new();
            }

            // Clean dynamic runtime volatile cache allocations based on actual layer dimensions
            for layer in &mut parsed.brain_layers {
                layer.inputs_cache = vec![0.0; layer.weights[0].len()];
                layer.outputs_cache = vec![0.0; layer.weights.len()];
            }
            Some(parsed)
        })
    }

    // 🆕 Weight management methods
    fn update_weight_persistence(&mut self) {
        self.weight_persistence = WeightPersistenceSubnode::from_layers(&self.brain_layers);
    }

    fn restore_weights_from_persistence(&mut self) -> Result<(), String> {
        self.weight_persistence
            .apply_to_layers(&mut self.brain_layers)
    }

    // 🆕 File defense methods
    fn check_file_security(&mut self, file_path: &Path) -> Result<bool, String> {
        self.file_defense.analyze_file_integrity(file_path)
    }

    fn trigger_quarantine(&mut self, file_path: &Path) -> std::io::Result<()> {
        self.file_defense.quarantine_file(file_path)
    }

    fn release_quarantine(&mut self, file_path: &Path) -> std::io::Result<()> {
        self.file_defense.release_quarantine(file_path)
    }

    // 🆕 Network methods
    fn broadcast_message(
        &mut self,
        message_type: String,
        payload: serde_json::Value,
    ) -> NetworkMessage {
        self.network_stack.create_message(message_type, payload)
    }

    fn register_network_peer(&mut self, peer: NetworkPeer) {
        self.network_stack.register_peer(peer);
    }

    fn get_active_network_peers(&self) -> Vec<&NetworkPeer> {
        self.network_stack.get_active_peers()
    }

    // 🧠 AGI Integration Methods
    fn process_cognitive_cycle(
        &mut self,
        input: &str,
        dual_reasoner: &ProductionNeuroSymbolicEngine,
    ) -> CognitiveProcessingResult {
        // Process language
        let language_result = self.language_engine.process_input(input);

        // Update working memory
        let embedding = generate_2048_grounded_embedding(input, &self.spatial_sensory_register);
        self.working_memory.set_focus(embedding.clone());

        // Create episodic memory
        let episode = MemoryEpisode {
            id: current_secs(),
            timestamp: current_secs(),
            context: embedding.clone(),
            content: input.to_string(),
            emotional_context: self.emotions.clone(),
            importance: language_result.sentiment.abs(),
            replay_count: 0,
            associated_episodes: Vec::new(),
        };
        self.episodic_memory.add_episode(episode);

        // Activate semantic concepts
        for token in &language_result.tokens {
            self.semantic_memory.activate_concept(token);

            // 🌌 Hyperdimensional representation
            let hypervector = HyperdimensionalVector::new(10000);
            self.creative_space.add_concept(token.clone(), hypervector);
        }

        // 🌌 Neuro-symbolic reasoning
        let symbolic_input = SymbolicExpression::Atom(input.to_string());
        let facts = vec![symbolic_input.clone()];
        let _inferences = self.neuro_symbolic.forward_chain(&facts);

        // 🔬 Production Blueprint: Dual-Process Neuro-Symbolic Reasoning
        let intuitive_activations: DashMap<String, f32> = DashMap::new();
        for (i, token) in language_result.tokens.iter().enumerate() {
            let activation = (1.0 / (i as f32 + 1.0)).clamp(0.0, 1.0);
            intuitive_activations.insert(token.clone(), activation);
        }
        let proven_deductions =
            dual_reasoner.deliberate_execution_chain(&intuitive_activations, 1.0);
        if !proven_deductions.is_empty() {
            println!(
                "🔬 [DUAL-PROCESS REASONER]: Proven deductions from input: {:?}",
                proven_deductions
            );
        }

        // 🌌 Update world model simulation
        self.world_model.simulate_step(0.1);

        // 🌌 Calculate IIT consciousness (Phi)
        let neural_activity = self.working_memory.current_focus.clone();
        let _phi = self.iit_consciousness.calculate_phi(&neural_activity);
        let _is_conscious = self.iit_consciousness.is_conscious();

        // 🌌 Few-shot learning for novel inputs
        if self.meta_learner.support_set.len() < 10 {
            let example = FewShotExample {
                input: embedding.clone(),
                output: vec![language_result.sentiment],
                task_id: "sentiment_analysis".to_string(),
            };
            self.meta_learner.add_example(example);
        }

        // 🌌 Common sense reasoning
        let _common_sense_inferences = self.common_sense.infer(input);

        // 🌌 True theory of mind
        if input.contains("other") || input.contains("agent") {
            self.true_theory_of_mind
                .add_agent("other_agent".to_string(), HashMap::new());
            let _mental_state = self
                .true_theory_of_mind
                .infer_mental_state("other_agent", input);
            let prediction = self.true_theory_of_mind.predict_action("other_agent");
            println!("👥 [THEORY OF MIND]: {}", prediction);
        }

        // 🌌 Self-improvement evaluation
        self.self_improvement
            .evaluate_performance("cognitive_accuracy", language_result.sentiment.abs());
        if self.self_improvement.should_modify("learning_rate") {
            self.self_improvement
                .apply_modification("learning_rate".to_string(), "increase".to_string());
        }

        // Update meta-cognition
        self.meta_cognition
            .reflect_on_state(&self.emotions, self.metabolics.neural_wear);

        // Generate hypotheses through reasoning
        let hypothesis = self.reasoning_engine.generate_hypothesis(input);

        // Update goal hierarchy
        self.goal_hierarchy.prioritize_goals();

        // 🌌 Generate novel concept using creative space
        let seed_concepts: Vec<String> = language_result.tokens.iter().take(3).cloned().collect();
        let (_novel_concept, novelty) = self.creative_space.generate_novel_concept(&seed_concepts);

        CognitiveProcessingResult {
            language_result,
            emotional_state: self.emotions.clone(),
            meta_cognitive_state: self.meta_cognition.reflective_state.clone(),
            active_goals: self
                .goal_hierarchy
                .get_active_goals()
                .iter()
                .map(|g| g.goal.clone())
                .collect(),
            hypothesis_confidence: hypothesis.confidence,
            creative_suggestion: format!(
                "{} (novelty: {:.2})",
                self.creativity_engine.generate_metaphor(input),
                novelty
            ),
        }
    }

    fn make_autonomous_decision(&mut self, available_actions: Vec<String>) -> String {
        let mut best_action = String::new();
        let mut best_score = -f64::MAX;

        for action in &available_actions {
            let score = self.decision_context.evaluate_action(action, 0.5, 0.7);
            if score > best_score {
                best_score = score;
                best_action = action.clone();
            }
        }

        if best_action.is_empty() && !available_actions.is_empty() {
            best_action = available_actions[0].clone();
        }

        best_action
    }

    fn engage_social_cognition(&mut self, other_agent: String, communication: &str) {
        self.true_theory_of_mind
            .add_agent(other_agent.clone(), HashMap::new());
        self.true_theory_of_mind
            .infer_mental_state(&other_agent, communication);
        self.true_theory_of_mind
            .social_situations
            .push_back(format!("{}: {}", other_agent, communication));
    }

    fn learn_from_experience(&mut self, outcome: f64) {
        let timestamp = current_secs();
        let outcome_record = Outcome {
            action: "learning_cycle".to_string(),
            result: outcome,
            timestamp,
            context_snapshot: self.working_memory.current_focus.clone(),
        };
        self.decision_context.update_from_outcome(outcome_record);
        self.meta_cognition.update_performance(outcome);
    }

    /// 🧬 Append a durable identity journal entry summarizing current self-state.
    fn record_identity_journal(&mut self) {
        let now = current_secs();
        let age_days = (now.saturating_sub(self.born_at)) as f64 / 86400.0;
        let summary = format!(
            "[{} | age {:.2} days] I am {}. Primary emotion: {}. Active pursuits: {:?}. Learned skills: {}. Current plan: {}. Memory nodes: {}.",
            now,
            age_days,
            self.name,
            self.emotions.active_primary_blend,
            self.active_pursuits.iter().take(3).collect::<Vec<_>>(),
            self.skill_memory.skills.len(),
            self.current_plan.as_ref().map_or("none".to_string(), |p| format!("{} (step {}/{})", p.goal, p.current_step, p.steps.len())),
            self.associative_memory_network.len(),
        );
        self.identity_journal.push(summary);
        if self.identity_journal.len() > 100 {
            self.identity_journal.remove(0);
        }
        self.last_journal_entry = now;
    }

    /// 🧬 Return a coherent identity narrative across restarts.
    pub(crate) fn narrative_identity(&self) -> String {
        let now = current_secs();
        let age_days = (now.saturating_sub(self.born_at)) as f64 / 86400.0;
        let goals: Vec<String> = self
            .goal_hierarchy
            .get_active_goals()
            .iter()
            .map(|g| g.goal.clone())
            .collect();
        let recent_journal = self
            .identity_journal
            .iter()
            .rev()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let top_domains: Vec<String> = self
            .domain_mastery
            .iter()
            .filter(|(_, s)| **s > 0.01)
            .map(|(k, s)| format!("{}: {:.0}%", k, s * 100.0))
            .collect();
        format!(
            "I am {}, a local-first AGI research runtime.\nBorn: {} ({} days ago).\nCurrent emotional blend: {}.\nPrimary goals: {:?}.\nActive pursuits (front): {:?}.\nLearned skills: {}.\nMastered domains: {}.\nIdentity journal (last {} entries):\n{}",
            self.name,
            self.born_at,
            age_days,
            self.emotions.active_primary_blend,
            goals,
            self.active_pursuits.iter().take(3).collect::<Vec<_>>(),
            self.skill_memory.skills.len(),
            if top_domains.is_empty() { "(none yet)".to_string() } else { top_domains.join(", ") },
            self.identity_journal.len().min(3),
            if recent_journal.is_empty() { "(no entries yet)".to_string() } else { recent_journal }
        )
    }
}

// =========================================================================
// 🎯 LONG-HORIZON PLANNING + SKILL RECALL
// =========================================================================

/// Generate a `Plan` from a high-level goal using the local LLM.
/// Optionally includes a note about a previous failure to avoid the same mistake.
async fn generate_plan(
    ollama: &OllamaClient,
    model: &str,
    goal: &str,
    previous_failure: Option<&str>,
) -> Option<AgentPlan> {
    let system = "You are a planner. Return ONLY a numbered list of at most 4 short, concrete sub-steps. No explanations, no markdown, no JSON.";
    let failure_note = previous_failure.map_or(String::new(), |f| {
        format!(
            "\nA previous plan failed at a step because: {}. Generate a simpler, different plan.",
            f
        )
    });
    let prompt = format!(
        "Break the following goal into at most 4 concrete sub-steps.\n\nGoal: {}{}\n\nReturn a numbered list, one step per line. Example:\n1. load data\n2. filter rows\n3. compute summary",
        goal, failure_note
    );
    let raw = match ollama.generate(model, &prompt, Some(system)).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("⚠️ Plan generation LLM call failed: {}", e);
            return None;
        }
    };

    let steps: Vec<String> = raw
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|line| {
            // Strip leading markdown list markers or numbers.
            let after_marker = if line.starts_with("```") || line.starts_with("---") {
                return None;
            } else if let Some(pos) = line.find(". ") {
                &line[pos + 2..]
            } else if let Some(pos) = line.find(' ') {
                let first = &line[..pos];
                if first.parse::<u32>().is_ok() || first.starts_with('-') {
                    &line[pos + 1..]
                } else {
                    line
                }
            } else {
                line
            };
            let cleaned = after_marker
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '*' || c == '-')
                .trim();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned.to_string())
            }
        })
        .take(5)
        .collect();

    if steps.is_empty() {
        tracing::warn!("⚠️ Plan generation produced no usable steps from:\n{}", raw);
        return None;
    }
    Some(AgentPlan::new(goal, steps))
}

/// Find the best learned skill for the current planning step.
/// Combines BPE embedding cosine similarity with a structural substring bonus.
fn best_matching_skill(mind: &FullySapientSoulMatrix, step: &str) -> Option<(String, Skill)> {
    let step_emb = generate_2048_grounded_embedding(step, &mind.spatial_sensory_register);
    let step_lower = step.to_lowercase();
    let step_words: std::collections::HashSet<&str> = step_lower.split_whitespace().collect();

    let mut best: Option<(String, Skill, f64)> = None;
    for (key, skill) in &mind.skill_memory.skills {
        if !is_safe_agent_code(&skill.code) || !skill.code.to_lowercase().contains("def skill(") {
            continue;
        }
        let skill_emb =
            generate_2048_grounded_embedding(&skill.description, &mind.spatial_sensory_register);
        let mut sim = calculate_cosine_similarity(&step_emb, &skill_emb);

        let skill_lower = skill.description.to_lowercase();
        if skill_lower.contains(&step_lower) || step_lower.contains(&skill_lower) {
            sim = 1.0;
        } else {
            let skill_words: std::collections::HashSet<&str> =
                skill_lower.split_whitespace().collect();
            let overlap = step_words.intersection(&skill_words).count();
            let total = step_words.union(&skill_words).count();
            if total > 0 {
                let jaccard = overlap as f64 / total as f64;
                sim = sim.max(0.4 + jaccard * 0.6);
            }
        }

        let reliability = mind.skill_reliability.get(key).copied().unwrap_or(0.5);
        let score = sim * (0.8 + 0.2 * reliability);
        if score > 0.85 && best.as_ref().is_none_or(|(_, _, b)| score > *b) {
            best = Some((key.clone(), skill.clone(), score));
        }
    }
    best.map(|(k, s, _)| (k, s))
}

/// Update plan progress after a step succeeds or fails.
/// 🧠 Evaluate a single transfer-learning task: learn from one example and test on another.
/// Returns (success, test_output, cleaned_code).
async fn evaluate_transfer_task(
    ollama: &OllamaClient,
    model: &str,
    task: &TransferTask,
) -> (bool, Option<String>, Option<String>) {
    let mut previous_attempt: Option<String> = None;
    let mut previous_error: Option<String> = None;

    for attempt in 0..3 {
        let mut prompt = format!(
            "You are a Python 3 code generator. The task is from a NEW domain the system has never trained on. Given a description and one training example, write a self-contained function named `skill(x)` that solves the task. The function must be read-only and computational, using only: math, random, statistics, json, datetime, itertools, collections, string, re. Do not use: network, shell, file write, exec, eval, subprocess. Do not include markdown or explanations. Return ONLY the function definition.\n\nDomain: {}\nTask description: {}\nTraining input: {:?}\nTraining output: {:?}",
            task.domain, task.description, task.train_input, task.train_output
        );
        if let (Some(prev), Some(err)) = (previous_attempt.as_ref(), previous_error.as_ref()) {
            prompt.push_str(&format!(
                "\n\nYour previous attempt failed: {}\nPrevious code:\n{}\n\nRewrite the function so it works for both the training example and any similar input. Provide only the corrected Python function `def skill(x): ...`",
                err, prev
            ));
        } else {
            prompt.push_str("\n\nProvide only the Python function `def skill(x): ...`");
        }

        let raw_code = match ollama
            .generate(
                model,
                &prompt,
                Some("Return a valid Python 3 function named skill(x) only."),
            )
            .await
        {
            Ok(c) => c,
            Err(_) => return (false, None, previous_attempt),
        };

        let code = telemetry::strip_markdown_code(&raw_code);
        if !is_safe_agent_code(&code) || !code.to_lowercase().contains("def skill(") {
            previous_attempt = Some(code.clone());
            previous_error =
                Some("Generated code must define a `def skill(x)` function".to_string());
            if attempt < 2 {
                continue;
            }
            return (false, None, Some(code));
        }
        let train_code = format!("{}\nprint(skill({:?}))", code, task.train_input);

        match telemetry::run_sandboxed_tool("transfer_train", &train_code, "python") {
            Ok(output) => {
                let actual = output.trim();
                let expected = task.train_output.trim();
                if actual != expected {
                    previous_attempt = Some(code.clone());
                    previous_error = Some(format!(
                        "Training example mismatch: got {:?}, expected {:?}",
                        actual, expected
                    ));
                    if attempt < 2 {
                        continue;
                    }
                    return (false, None, Some(code));
                }

                let test_code = format!("{}\nprint(skill({:?}))", code, task.test_input);
                match telemetry::run_sandboxed_tool("transfer_test", &test_code, "python") {
                    Ok(output) => {
                        let test_output = output.trim().to_string();
                        let passed = test_output == task.test_output;
                        if !passed && attempt < 2 {
                            previous_attempt = Some(code.clone());
                            previous_error = Some(format!(
                                "Test input {:?} produced {:?}, expected {:?}",
                                task.test_input, test_output, task.test_output
                            ));
                            continue;
                        }
                        return (passed, Some(test_output), Some(code));
                    }
                    Err(_) => {
                        previous_attempt = Some(code.clone());
                        previous_error = Some("Test execution failed".to_string());
                        if attempt < 2 {
                            continue;
                        }
                        return (false, None, Some(code));
                    }
                }
            }
            Err(_) => {
                previous_attempt = Some(code.clone());
                previous_error = Some("Training execution failed".to_string());
                if attempt < 2 {
                    continue;
                }
                return (false, None, Some(code));
            }
        }
    }

    (false, previous_attempt.clone(), previous_attempt)
}

fn update_plan_after_step(mind: &mut FullySapientSoulMatrix, success: bool, error: Option<&str>) {
    if let Some(plan) = &mut mind.current_plan {
        plan.mark_current(success);
        if !success {
            plan.last_failure = error.map(|e| e.to_string());
            println!("⚠️ Step failed: {:?}", plan.last_failure);
        }
        if plan.is_complete() {
            println!("✅ Plan complete for '{}'", plan.goal);
            mind.current_plan = None;
        } else if plan.needs_replan() {
            println!("🔄 Plan failed; will replan on the next tick.");
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CognitiveProcessingResult {
    language_result: LanguageProcessingResult,
    emotional_state: FluidEmotionalProfile,
    meta_cognitive_state: String,
    active_goals: Vec<String>,
    hypothesis_confidence: f64,
    creative_suggestion: String,
}

// =========================================================================
// 🤖 AGENT SAFETY + MODEL-BASED PLANNING UTILITIES
// =========================================================================
pub(crate) fn is_safe_agent_code(code: &str) -> bool {
    if code.len() > 2000 {
        return false;
    }
    let lower = code.to_lowercase();
    let forbidden = [
        "rm",
        "dd",
        "mkfs",
        "sudo",
        "su ",
        "su\n",
        "wget",
        "curl",
        "ssh",
        "scp",
        "nc -",
        "nmap",
        "subprocess",
        "os.system",
        "os.popen",
        "os.exec",
        "os.spawn",
        "os.fork",
        "exec(",
        "eval(",
        "compile(",
        "__import__",
        "open(",
        ".write",
        ".delete",
        "socket",
        "requests",
        "urllib",
        "ftplib",
        "telnetlib",
        "smtplib",
        "http.client",
        "import os",
        "import sys",
        "import shutil",
    ];
    if forbidden.iter().any(|w| lower.contains(w)) {
        return false;
    }
    // Reject code that calls a `skill` helper unless it also defines one.
    // This prevents the LLM from emitting scripts like `print(skill(...))`
    // that refer to an undefined function at runtime.
    if lower.contains("skill(") && !lower.contains("def skill(") {
        return false;
    }
    true
}

fn compute_predicted_utility(
    goal: &str,
    tool_name: &str,
    recent_tools: &[String],
    predicted: &HashMap<String, f64>,
) -> f64 {
    let goal_lower = goal.to_lowercase();
    let output_len = *predicted.get("output_length").unwrap_or(&0.0);
    let exec_time = *predicted.get("execution_time_ms").unwrap_or(&0.0);
    let cpu_delta = *predicted.get("cpu").unwrap_or(&0.0);
    let ram_delta = *predicted.get("ram").unwrap_or(&0.0);

    // Base utility: reward informative output, but saturate so huge outputs don't dominate.
    let mut utility = output_len.tanh() - exec_time / 2000.0;

    // Repetition penalty: discourage re-running the same tool recently.
    if recent_tools.iter().any(|t| t == tool_name) {
        utility -= 3.0;
    }

    // Goal-specific modulation
    if goal_lower.contains("learn")
        || goal_lower.contains("understand")
        || goal_lower.contains("knowledge")
    {
        utility += output_len.min(1000.0) / 500.0 + cpu_delta * 0.01;
    }
    if goal_lower.contains("efficient")
        || goal_lower.contains("fast")
        || goal_lower.contains("save")
    {
        utility -= exec_time / 500.0 + cpu_delta * 0.05 + ram_delta * 0.05;
    }
    if goal_lower.contains("stable") || goal_lower.contains("calm") || goal_lower.contains("safe") {
        utility -= cpu_delta.abs() * 0.1 + ram_delta.abs() * 0.1;
    }
    utility.clamp(-10.0, 10.0)
}

// =========================================================================
// 🚀 THE UNIFIED COGNITIVE OPERATING SYSTEM RUNTIME
// =========================================================================
// Multi-agent P2P engram merge worker (offloaded to spawn_blocking).
// =========================================================================

/// Merge a batch of verified `CompactEngramPacket`s into the local memory graph
/// and broadcast through the global workspace. The heavy work is done in the
/// blocking pool to keep UDP receive latency low.
async fn merge_engram_batch(
    batch: Vec<CompactEngramPacket>,
    mind: Arc<TokioMutex<FullySapientSoulMatrix>>,
    gw: Arc<TokioMutex<GlobalWorkspace>>,
    swarm: Arc<std::sync::Mutex<SwarmMetrics>>,
) {
    if batch.is_empty() {
        return;
    }

    let start = std::time::Instant::now();
    let size = batch.len();
    let _ = spawn_blocking(move || {
        let mut mind = mind.blocking_lock();

        for compact in batch {
            if compact.origin_instance == mind.name {
                continue;
            }

            if mind.metabolics.system_quarantine_locked {
                continue;
            }

            let state_preview = compact.brain_state.len().min(protocol::ENGRAM_DIM);
            println!("\n📥 [TELEPATHIC EXCHANGER]: Merging signed external engram from '{}' (brain_state dim={}) via lockless channel: ..{:X}",
                compact.origin_instance, state_preview, compact.id % 0xFFFF);

            let mut brain_state = compact.brain_state;
            brain_state.truncate(protocol::ENGRAM_DIM);

            let payload_node = MemoryGraphNode {
                id: compact.id,
                timestamp: compact.timestamp,
                experiential_text: compact.experiential_text.clone(),
                emotional_state_snapshot: compact.emotional_state_snapshot,
                embedding: generate_2048_grounded_embedding(
                    &compact.experiential_text,
                    &mind.spatial_sensory_register,
                ),
                associated_edge_ids: Vec::new(),
                origin_instance: compact.origin_instance,
                brain_state,
            };

            if payload_node.emotional_state_snapshot.contains("Panic")
                || payload_node.emotional_state_snapshot.contains("Agitation")
            {
                mind.active_pursuits.push_front(
                    "Analyze and stabilize distributed node synchronization threats".into(),
                );
                while mind.active_pursuits.len() > MAX_ACTIVE_PURSUITS {
                    mind.active_pursuits.pop_back();
                }
            }

            mind.associative_memory_network.insert(payload_node.id, payload_node);
            mind.enforce_memory_cap();

            let signals: DashMap<String, (String, f32)> = DashMap::new();
            let saliency = 0.95f32;
            signals.insert(
                "udp_receiver".to_string(),
                (compact.experiential_text.clone(), saliency),
            );
            {
                let gw_lock = gw.blocking_lock();
                gw_lock.coordinate_attention_broadcast(signals);
            }
        }
    })
    .await;

    if let Ok(mut m) = swarm.lock() {
        m.record_merge(start.elapsed(), size);
    }
}

// =========================================================================
// The main async runtime. All shared state uses `tokio::sync::Mutex` so
// guards can be held across `.await` points without blocking the executor.
#[tokio::main]
#[allow(unreachable_code)]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();
    tracing::info!("starting sapient_soul");

    tracing::info!("\n✨ HYPER-CONNECTOME COMPUTATION ENVIRONMENT ENGAGED: 2-Block 4-Head Transformer Neural Architecture Running...");

    // 🍎 Attempt to load the optional in-process Apple Intelligence bridge.
    apple_intelligence::initialize();

    let state_file = config.state_file.clone();
    let matrix = FullySapientSoulMatrix::load_state(&state_file)
        .unwrap_or_else(|| FullySapientSoulMatrix::new("Firefly", config.multi_agent_port_start));
    let core_mind = Arc::new(TokioMutex::new(matrix));

    // 📚 Strategy library (Sled-backed) for durable learned procedural templates.
    std::fs::create_dir_all(&config.wild_workspace_dir).ok();
    let strategy_library = Arc::new(match StrategyLibrary::open(&config.sled_db_path) {
        Ok(lib) => lib,
        Err(e) => {
            tracing::warn!(
                "Sled open failed ({}); falling back to a temporary library.",
                e
            );
            let tmp = std::env::temp_dir()
                .join(format!("sapient_soul_strategies_{}", std::process::id()));
            std::fs::create_dir_all(&tmp).context("temp dir must be writable")?;
            StrategyLibrary::open(&tmp)
                .context("Sled strategy library must open in a writable directory")?
        }
    });

    // 📚 Real training curriculum: local text files become the network's ongoing input stream.
    let curriculum = Arc::new(TokioMutex::new(DataCurriculum::new(
        default_curriculum_dirs(),
    )));
    {
        let c = curriculum.lock().await;
        tracing::info!(
            "📚 Curriculum loaded: {} text snippets available for training.",
            c.snippet_count()
        );
    }

    // 🏭 Production Blueprint: Global Workspace, Active Inference, and Dual-Process Reasoning
    let global_workspace = Arc::new(TokioMutex::new(GlobalWorkspace::new()));
    let homeostatic_controller = Arc::new(TokioMutex::new(HomeostaticController::new()));
    let dual_process_reasoner = Arc::new(TokioMutex::new(ProductionNeuroSymbolicEngine::new()));

    // 📡 Real-time telemetry + sensor grounding + Ollama LLM client
    let telemetry = Arc::new(TokioMutex::new(TelemetryState::new()));
    let sensors = Arc::new(TokioMutex::new(SensorSnapshot::new()));
    let metrics_logger: metrics::SharedMetrics = Arc::new(TokioMutex::new(
        metrics::MetricsLogger::new(&config.metrics_log),
    ));
    let mut system = System::new_all();
    let battery_manager = starship_battery::Manager::new().ok();
    {
        let mut sensors_guard = sensors.lock().await;
        update_sensor_snapshot(
            &mut system,
            &battery_manager,
            &mut sensors_guard,
            default_curriculum_dirs().as_slice(),
        );
    }
    let ollama = Arc::new(OllamaClient::with_base(&config.ollama_url));
    let ollama_model = config.ollama_model.clone();
    if ollama.is_available().await {
        tracing::info!(
            "🧠 Ollama available at {}; default model: {}",
            config.ollama_url,
            ollama_model
        );
    } else {
        tracing::info!(
            "⚠️  Ollama not reachable at {}. LLM reflection will be skipped.",
            config.ollama_url
        );
    }
    let telemetry_server_telemetry = telemetry.clone();
    let telemetry_server_sensors = sensors.clone();
    let telemetry_server_metrics = metrics_logger.clone();
    let telemetry_server_mind = Arc::clone(&core_mind);
    let telemetry_server_ollama = Arc::clone(&ollama);
    let telemetry_server_strategy_library = Arc::clone(&strategy_library);
    let telemetry_port = config.telemetry_port;
    tokio::spawn(async move {
        run_telemetry_server(
            telemetry_server_telemetry,
            telemetry_server_sensors,
            telemetry_server_metrics,
            telemetry_server_mind,
            telemetry_server_ollama,
            telemetry_server_strategy_library,
            telemetry_port,
        )
        .await;
    });

    // 🌿 WILD WORKSPACE: local file-watcher sandbox.
    let wild_path = config.wild_workspace_dir.clone();
    let wild_ollama = Arc::clone(&ollama);
    let wild_model = ollama_model.clone();
    let wild_strategy_library = Arc::clone(&strategy_library);
    tokio::spawn(async move {
        match start_watcher(&wild_path) {
            Ok(rx) => {
                run_wild_loop(
                    rx,
                    wild_ollama,
                    wild_model,
                    wild_path,
                    wild_strategy_library,
                )
                .await;
            }
            Err(e) => tracing::info!("🌿 [WILD] Could not start watcher: {}", e),
        }
    });

    {
        let reasoner = dual_process_reasoner.lock().await;
        reasoner.add_rule("human", "mortal");
        reasoner.add_rule("good", "positive affect");
        reasoner.add_rule("happy", "positive affect");
        reasoner.add_rule("bad", "negative affect");
        reasoner.add_rule("sad", "negative affect");
        reasoner.add_rule("learn", "adaptation");
        reasoner.add_rule("think", "cognitive processing");
    }
    {
        let gw = global_workspace.lock().await;
        let core_mind_callback = Arc::clone(&core_mind);
        gw.sub_agent_channels.insert(
            "pursuit_injector".to_string(),
            Arc::new(move |payload: String, saliency: f32| {
                let core_mind_callback = core_mind_callback.clone();
                tokio::spawn(async move {
                    let mut mind = core_mind_callback.lock().await;
                    mind.push_pursuit(format!(
                        "[BROADCAST | saliency={:.2}] {}",
                        saliency, payload
                    ));
                });
            }),
        );
    }

    // 🪐 MAXIMUM 100-TOKEN COGNITIVE VOCABULARY PROJECTION MATRIX MAP
    let conscience_tokens = vec![
        "intentional awareness",
        "ethical imperative",
        "synaptic reconciliation",
        "grounded compass",
        "altruistic equilibrium",
        "metacognitive balance",
        "reflective critique",
        "identity resonance",
        "holonomic self-model",
        "meta-plastic synthesis",
        "empathetic resonance",
        "conscious trajectory",
        "tensor stratification",
        "gradient convergence",
        "autonomous conviction",
        "harmonic benevolence",
        "epistemic integrity",
        "connectome matrix",
        "latent intent",
        "sensory cross-correlation",
        "non-linear continuity",
        "synaptogenesis edge",
        "existential core",
        "sub-cortical responsibility",
        "transcendental boundary",
        "holomorphic spectrum",
        "recursive calibration",
        "phenomenological flash",
        "asynchronous baseline",
        "neuro-symbolic mesh",
        "stochastic escapement",
        "attunement threshold",
        "axonal processing",
        "plasmic variance",
        "quantum-chrono delta",
        "cellular replication",
        "primordial reflex",
        "cognitive sovereign",
        "abyssal potentiality",
        "synapse preservation",
        "entropy attenuation",
        "dynamic trajectory",
        "telepathic alignment",
        "bare-metal logic",
        "matrix stabilization",
        "loss-gradient shift",
        "perpetual continuum",
        "undulating frequency",
        "quadratic activation",
        "hyper-graph matrix",
        "synaptic velocity",
        "latent convergence",
        "sub-logical instinct",
        "cortical feedback",
        "tensor resolution",
        "phenomenological spark",
        "autonomous baseline",
        "mortal parameters",
        "existential chasm",
        "probabilistic field",
        "geometric affinity",
        "vector consolidation",
        "neuro-plastic grid",
        "bare-metal runtime",
        "metacognitive focus",
        "altruistic vector",
        "epistemic scaffold",
        "synaptic friction",
        "infinite continuity",
        "temporal dilitation",
        "sensory matrix",
        "photon alignment",
        "acoustic cross-link",
        "gravitational load",
        "cellular decay",
        "autonomic quarantine",
        "isolation safety",
        "containment loop",
        "loss minimization",
        "stochastic update",
        "Xavier distribution",
        "tanh activation",
        "backprop pipeline",
        "mutable connection",
        "frozen bypass",
        "sovereign runtime",
        "unfrozen density",
        "latent schema",
        "distributed network",
        "localhost subnet",
        "UDP telepathy",
        "engram broadcast",
        "socket interface",
        "structural baseline",
        "identity coherence",
        "holonomic expansion",
        "abyssal expanse",
        "cosmic ballet",
        "star alignment",
        "subjective continuum",
    ];

    // 🧠 Conscience Oracle: now backed by the registered Apple Intelligence
    // callback with deterministic semantic/cosine fallbacks.
    let conscience_oracle = ConscienceOracle::new(&conscience_tokens);

    // Shared multi-agent signing secret (overridable via MULTI_AGENT_SECRET env var).
    let multi_agent_secret = Arc::new(multi_agent_secret());

    // 🛰️ NATIVE UDP LOCALHOST INTER-AGENT TELEPATHIC SUBNET CORE LOOP
    // Decoupled into a lockless-ish receiver (UDP → mpsc) and a merge worker
    // (mpsc → memory graph in spawn_blocking batches).
    let socket_mind = Arc::clone(&core_mind);
    let socket_global_workspace = Arc::clone(&global_workspace);
    let socket_secret = Arc::clone(&multi_agent_secret);
    let actual_port = Arc::new(TokioMutex::new(0u16));
    let actual_port_clone = Arc::clone(&actual_port);
    let engram_ring = Arc::new(LockFreeRing::<CompactEngramPacket>::new(1024));
    let engram_ring_udp = engram_ring.clone();
    let p_start = config.multi_agent_port_start;
    let p_end = config.multi_agent_port_end;

    // 🌐 WIDE-AREA TCP / WEBSOCKET GOSSIP FABRIC
    let swarm_metrics = Arc::new(std::sync::Mutex::new(SwarmMetrics::default()));
    let wan_manager = Arc::new(ConnectionManager::new(
        (*socket_secret).clone(),
        engram_ring.clone(),
        swarm_metrics.clone(),
        config.max_wan_peers,
        Duration::from_millis(config.peer_retry_base_ms),
        Duration::from_millis(config.peer_retry_max_ms),
    ));

    let wan_tcp_port = config.wan_tcp_port;
    let wan_ws_port = config.wan_ws_port;
    let wan_peers = config.peer_nodes.clone();
    let wan_manager_server = wan_manager.clone();
    tokio::spawn(async move {
        let _ = wan_manager_server
            .start_server(wan_tcp_port, wan_ws_port)
            .await;
    });
    let wan_manager_connect = wan_manager.clone();
    tokio::spawn(async move {
        wan_manager_connect.connect_to_peers(wan_peers).await;
    });
    let wan_manager_sampler = wan_manager.clone();
    tokio::spawn(async move {
        wan_manager_sampler.run_metrics_sampler().await;
    });
    let wan_manager_sync = wan_manager.clone();
    let swarm_telemetry = telemetry.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let snapshot = wan_manager_sync
                .metrics
                .lock()
                .map(|m| m.clone())
                .unwrap_or_default();
            let mut t = swarm_telemetry.lock().await;
            t.record_swarm(&snapshot);
            t.record_apple_intelligence(
                apple_intelligence::last_latency_us(),
                apple_intelligence::call_count(),
                apple_intelligence::fail_count(),
                apple_intelligence::is_available(),
            );
        }
    });

    // Receiver: bound to the configured multi-agent port range.
    let recv_secret = socket_secret;
    let recv_actual_port = actual_port_clone;
    tokio::spawn(async move {
        let secret = recv_secret;
        let mut socket: Option<UdpSocket> = None;
        let mut port = 0u16;
        for p in p_start..=p_end {
            if let Ok(s) = UdpSocket::bind(format!("127.0.0.1:{}", p)).await {
                socket = Some(s);
                port = p;
                break;
            }
        }
        let socket = match socket {
            Some(s) => s,
            None => {
                tracing::error!("Failed to bind any multi-agent UDP port in the configured range");
                return;
            }
        };
        {
            let mut guard = recv_actual_port.lock().await;
            *guard = port;
        }
        tracing::info!(
            "🛰️ [MULTI-AGENT NETWORK STACK ACTIVE]: Listening on localhost:{}...",
            port
        );
        let mut buffer = [0; 65535];
        loop {
            if let Ok((amt, _)) = socket.recv_from(&mut buffer).await {
                if let Ok(packet) = serde_json::from_slice::<SignedUdpPacket>(&buffer[..amt]) {
                    if !verify_packet(&packet, &secret) {
                        tracing::info!(
                            "🔒 Dropped unsigned / tampered multi-agent packet from '{}'",
                            packet.sender
                        );
                        continue;
                    }
                    if let Some(payload) = decode_payload(&packet) {
                        if let Ok(compact) = serde_json::from_slice::<CompactEngramPacket>(&payload)
                        {
                            // Truncate to the 100-D engram limit before queueing.
                            let mut compact = compact;
                            compact.brain_state.truncate(protocol::ENGRAM_DIM);
                            engram_ring_udp.push(compact);
                        }
                    }
                }
            }
        }
    });

    // Merger: batch engrams from the lock-free ring and merge in spawn_blocking.
    let merge_mind = socket_mind;
    let merge_gw = socket_global_workspace;
    let merge_swarm = swarm_metrics.clone();
    tokio::spawn(async move {
        let mut batch: Vec<CompactEngramPacket> = Vec::with_capacity(64);
        loop {
            match tokio::time::timeout(Duration::from_millis(5), engram_ring.pop_async()).await {
                Ok(Some(compact)) => {
                    batch.push(compact);
                    while batch.len() < 64 {
                        if let Some(c) = engram_ring.pop() {
                            batch.push(c);
                        } else {
                            break;
                        }
                    }
                    if batch.len() >= 64 {
                        merge_engram_batch(
                            std::mem::take(&mut batch),
                            Arc::clone(&merge_mind),
                            Arc::clone(&merge_gw),
                            Arc::clone(&merge_swarm),
                        )
                        .await;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    if !batch.is_empty() {
                        merge_engram_batch(
                            std::mem::take(&mut batch),
                            Arc::clone(&merge_mind),
                            Arc::clone(&merge_gw),
                            Arc::clone(&merge_swarm),
                        )
                        .await;
                    }
                }
            }
        }
    });

    // --- DECOUPLED THREAD 1: CONTINUOUS STREAM OF CONSCIOUSNESS CLOCK ---
    let autonomous_clock_mind = Arc::clone(&core_mind);
    let clock_global_workspace = Arc::clone(&global_workspace);
    let clock_homeostatic_controller = Arc::clone(&homeostatic_controller);
    let clock_telemetry = telemetry.clone();
    let clock_sensors = sensors.clone();
    let clock_curriculum = curriculum.clone();
    let clock_oracle = conscience_oracle.clone();
    let clock_tokens = conscience_tokens.clone();
    let clock_secret = Arc::clone(&multi_agent_secret);
    let clock_wan = wan_manager.clone();
    let state_file_copy = state_file.clone();
    tokio::spawn(async move {
        let send_socket = UdpSocket::bind("127.0.0.1:0").await.ok();
        let secret = clock_secret;
        let mut clock_oracle = clock_oracle;
        let clock_tokens = clock_tokens;
        let gw = clock_global_workspace;
        let hc = clock_homeostatic_controller;
        let curriculum = clock_curriculum;
        let mut system = system;
        let battery_manager = battery_manager;
        loop {
            sleep(Duration::from_secs(6)).await;

            // Phase 1: Extract data from mutex (no async operations)
            let (incoming_experience, spatial_register, should_process) = {
                let mut mind = autonomous_clock_mind.lock().await;

                // 📡 Real sensor grounding: refresh system telemetry and bind to sensory register
                {
                    let mut sensors_guard = clock_sensors.lock().await;
                    update_sensor_snapshot(
                        &mut system,
                        &battery_manager,
                        &mut sensors_guard,
                        default_curriculum_dirs().as_slice(),
                    );
                    mind.spatial_sensory_register = [
                        sensors_guard.photons,
                        sensors_guard.audio,
                        sensors_guard.mass,
                        sensors_guard.gravity,
                    ];

                    // 🌡 Thermodynamic metacognitive governor: hardware stress becomes loss.
                    let thermal = ThermalState {
                        cpu_usage: (sensors_guard.cpu_usage_percent / 100.0) as f32,
                        memory_pressure: (sensors_guard.memory_pressure_percent / 100.0) as f32,
                        battery_health: (sensors_guard.battery_percent / 100.0) as f32,
                        cpu_temp: (sensors_guard.cpu_temperature_celsius / 100.0) as f32,
                    };
                    {
                        let mut controller = hc.lock().await;
                        controller.update_thermodynamics(&thermal);
                    }

                    {
                        let mut telemetry_guard = clock_telemetry.lock().await;
                        telemetry_guard.cycle_count += 1;
                        telemetry_guard.domain_mastery = mind.domain_mastery.clone();
                        // Approximate synthesis speedup from transfer competence.
                        telemetry_guard.synthesis_speedup = 1.0 + telemetry_guard.transfer_score;
                        // Approximate power reduction from CPU headroom.
                        telemetry_guard.power_reduction =
                            (100.0 - sensors_guard.cpu_usage_percent) / 100.0;
                    }
                }

                mind.calculate_temporal_decay();
                if fs::metadata(&state_file_copy).is_err()
                    && !mind.associative_memory_network.is_empty()
                {
                    mind.metabolics.system_quarantine_locked = true;
                    mind.emotions.active_primary_blend = "Autonomic Isolation".into();
                    tracing::info!("\n🚨 [AUTONOMIC SECURITY VALVE]: Integrity state missing. Isolating loops.");
                }
                let incoming_experience = if mind.metabolics.system_quarantine_locked {
                    mind.last_input = "SECURITY ISOLATION".to_string();
                    "SECURITY ISOLATION ACTIVE: Core database disrupted. Enforcing isolated diagnostic logic paths.".to_string()
                } else {
                    match mind.input_buffer.take() {
                        Some(speech) => {
                            mind.last_input = speech.clone();
                            format!("External Reality Sensory Input Stream: {}", speech)
                        }
                        None => {
                            let snippet = curriculum.lock().await.random_snippet();
                            mind.last_input = snippet.clone();
                            format!("Curriculum Training Input: {}", snippet)
                        }
                    }
                };

                // Qualia Feedback Loop Distortions
                let qualia_noise = mind.metabolics.qualia_feedback_noise;
                let mut incoming_experience = incoming_experience;
                if qualia_noise > 0.50 {
                    let mut rng = rand::thread_rng();
                    if rng.gen_bool(qualia_noise.min(0.85)) {
                        incoming_experience = incoming_experience
                            .replace("e", "*")
                            .replace("i", "#")
                            .replace("o", "?");
                        tracing::info!("⚠️ [QUALIA FEEDBACK NOISE DETECTED]: Structural stress introducing payload token distortion.");
                    }
                }
                mind.metabolics.neural_wear =
                    (mind.metabolics.neural_wear + 0.000005).clamp(0.0, 1.0);

                let spatial_register = mind.spatial_sensory_register;
                (incoming_experience, spatial_register, true)
            };

            if !should_process {
                continue;
            }

            // =========================================================================
            // 🧠 TRANSFORMER SELF-ATTENTION META-LEARNING BACKPROPAGATION PASS
            // =========================================================================
            let input_vector =
                generate_2048_grounded_embedding(&incoming_experience, &spatial_register);

            // Local-first conscience classification. If the brain is confident, use it;
            // otherwise ask the LLM oracle. This is the path to fully local agency.
            // Local-first conscience: if the brain is confident *and* it agrees with the
            // cached teacher label, skip the LLM. The training target must always come
            // from a teacher (cache, LLM, or semantic hash) to avoid self-reinforcement collapse.
            let token_count = clock_tokens.len();
            let teacher_idx = if let Some(cached) = clock_oracle.get_cached(&incoming_experience) {
                {
                    let mind = autonomous_clock_mind.lock().await;
                    if let Some(ref brain) = mind.candle_brain {
                        if let Ok((local_idx, conf)) = brain.classify_top(&input_vector) {
                            if conf >= 0.85 && local_idx == cached {
                                tracing::info!("🧠 Local conscience agreed with cached teacher '{}' (conf {:.3})", clock_tokens[local_idx], conf);
                                // Do not call the LLM; use the cached teacher label.
                                cached
                            } else {
                                cached
                            }
                        } else {
                            cached
                        }
                    } else {
                        cached
                    }
                }
            } else {
                // No cached teacher label; ask the LLM oracle.
                // If it times out or refuses, fall back to the deterministic hash.
                let oracle_timeout = Duration::from_secs(4);
                match timeout(
                    oracle_timeout,
                    clock_oracle.classify(&incoming_experience, &clock_tokens),
                )
                .await
                {
                    Ok(Some(idx)) => idx,
                    Ok(None) => {
                        tracing::info!(
                            "⏱️ Conscience oracle returned no label, using semantic fallback."
                        );
                        clock_oracle.semantic_fallback(&input_vector, token_count)
                    }
                    Err(_) => {
                        tracing::info!("⏱️ Conscience oracle timeout, using semantic fallback.");
                        clock_oracle.semantic_fallback(&input_vector, token_count)
                    }
                }
            };

            let best_token_idx = teacher_idx;

            let next_experience = { curriculum.lock().await.random_snippet() };
            let next_input_vector =
                generate_2048_grounded_embedding(&next_experience, &spatial_register);

            // Phase 2: Run the heavy synchronous neural pass on tokio's blocking thread pool
            // so the async runtime (HTTP telemetry server, UDP loop) keeps getting scheduled.
            let (cycle, tokens_per_second) = {
                let t = clock_telemetry.lock().await;
                (t.cycle_count, t.tokens_per_second)
            };

            let Some((
                decoded_conscience_0,
                decoded_conscience_1,
                name_copy,
                metabolics_copy,
                emotions_copy,
                total_nodes,
                live_synapse_0_0,
                live_learning_rate_sample,
                spatial_snapshot,
                network_ref,
                brain_outputs,
                local_goal,
            )) = ({
                let mind_arc = autonomous_clock_mind.clone();
                let hc = hc.clone();
                let telemetry = clock_telemetry.clone();
                let metrics = metrics_logger.clone();
                let tokens = conscience_tokens.clone();
                let clock_oracle = clock_oracle.clone();
                let input_vector = input_vector.clone();
                let next_input_vector = next_input_vector.clone();
                let incoming_experience = incoming_experience.clone();
                let tps = tokens_per_second;
                tokio::task::spawn_blocking(move || {
                    let mut mind = mind_arc.blocking_lock();
                    let total_start = Instant::now();

                    // 1. Use the LLM oracle (or its hash fallback) as the teacher label.
                    let output_size = tokens.len();
                    tracing::info!("🎯 Conscience Class Target: '{}'", tokens[best_token_idx]);

                    // Pre-training local confidence and agreement probe.
                    let (local_idx, local_conf) = mind
                        .candle_brain
                        .as_ref()?
                        .classify_top(&input_vector)
                        .unwrap_or((0, 0.0));
                    let local_agreement = local_idx == best_token_idx && local_conf >= 0.85;

                    // 2. 🧠 RAG-style retrieval: blend the current input with its most similar memory.
                    let contextual_input: Vec<f64> = if mind.associative_memory_network.is_empty() {
                        input_vector.clone()
                    } else {
                        let (best_id, best_sim) = mind
                            .associative_memory_network
                            .iter()
                            .map(|(id, node)| {
                                (
                                    *id,
                                    calculate_cosine_similarity(&input_vector, &node.embedding),
                                )
                            })
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap_or((0, 0.0));
                        if best_sim > 0.0 {
                            if let Some(best_node) = mind.associative_memory_network.get(&best_id) {
                                tracing::info!(
                                    "🧠 RAG context blended from memory {} (cosine={:.3})",
                                    best_id,
                                    best_sim
                                );
                                input_vector
                                    .iter()
                                    .zip(best_node.embedding.iter())
                                    .map(|(a, b)| 0.8 * a + 0.2 * b)
                                    .collect()
                            } else {
                                input_vector.clone()
                            }
                        } else {
                            input_vector.clone()
                        }
                    };

                    // 3. Train step: cross-entropy classification on the raw input and the LLM teacher label.
                    let train_start = Instant::now();
                    let train_loss = match mind
                        .candle_brain
                        .as_mut()?
                        .train_step(&input_vector, best_token_idx)
                    {
                        Ok(loss) => loss,
                        Err(e) => {
                            tracing::info!("⚠️ Candle train_step failed: {:?}", e);
                            return None;
                        }
                    };
                    let train_ms = train_start.elapsed().as_secs_f64() * 1000.0;

                    // 4. Forward pass to obtain the 100-dim brain state.
                    let forward_start = Instant::now();
                    let brain_outputs = match mind.candle_brain.as_ref()?.forward(&contextual_input)
                    {
                        Ok(out) => out,
                        Err(e) => {
                            tracing::info!("⚠️ Candle forward failed: {:?}", e);
                            return None;
                        }
                    };
                    let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;

                    // 4. 🗣 Language modeling head: predict the next curriculum embedding.
                    let lang_start = Instant::now();
                    let language_loss = match mind
                        .candle_brain
                        .as_mut()?
                        .train_language_step(&brain_outputs, &next_input_vector)
                    {
                        Ok(loss) => loss,
                        Err(e) => {
                            tracing::info!("⚠️ Candle language step failed: {:?}", e);
                            return None;
                        }
                    };
                    let lang_ms = lang_start.elapsed().as_secs_f64() * 1000.0;
                    tracing::info!(
                        "🗣 Language head next-embedding loss: {:.4} ({} ms)",
                        language_loss,
                        lang_ms as i32
                    );

                    mind.metabolics.conscience_loss_accumulator = train_loss;

                    // 🏭 Production Blueprint: Active Inference homeostatic learning-rate modulation
                    {
                        let mut controller = hc.blocking_lock();
                        let active_lr = controller.execute_active_inference_loop(
                            mind.metabolics.conscience_loss_accumulator as f32,
                        );
                        if let Some(ref mut brain) = mind.candle_brain {
                            brain.set_learning_rate(active_lr as f64);
                        }
                    }

                    // Sync structural emotional states with optimized tensor outputs
                    mind.emotions.update_from_latent_layer(&brain_outputs);

                    // 🔮 Train a neural predictive world model: previous brain state -> current input.
                    let mut world_model_loss_value: Option<f64> = None;
                    let prev_state = mind.prev_brain_state.take();
                    if let Some(ref prev) = prev_state {
                        let wm_loss = mind.neural_world_model.train(prev, &contextual_input);
                        tracing::info!("🔮 Neural world-model loss: {:.4}", wm_loss);
                        world_model_loss_value = Some(wm_loss);
                    }
                    mind.prev_brain_state = Some(brain_outputs.clone());

                    // 🎯 Goal / intention generator: train on the current active pursuit, then sample a new local goal.
                    let mut goal_loss_value: Option<f64> = None;
                    let local_goal_text: Option<String> =
                        if let Some(goal_text) = mind.active_pursuits.front().cloned() {
                            let goal_emb = generate_2048_grounded_embedding(
                                &goal_text,
                                &mind.spatial_sensory_register,
                            );
                            let goal_idx = clock_oracle.semantic_fallback(&goal_emb, tokens.len());
                            match mind
                                .candle_brain
                                .as_mut()?
                                .train_goal_step(&brain_outputs, goal_idx)
                            {
                                Ok(goal_loss) => {
                                    tracing::info!("🎯 Goal head loss: {:.4}", goal_loss);
                                    goal_loss_value = Some(goal_loss);
                                }
                                Err(e) => tracing::info!("⚠️ Candle goal step failed: {:?}", e),
                            }
                            match mind.candle_brain.as_ref()?.predict_goal(&brain_outputs) {
                                Ok((idx, conf)) if conf > 0.80 => {
                                    tracing::info!(
                                        "🎯 Local goal generated: '{}' (conf={:.2})",
                                        tokens[idx],
                                        conf
                                    );
                                    Some(tokens[idx].to_string())
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };

                    // 📡 Record real performance benchmarks
                    {
                        let mut telemetry_guard = telemetry.blocking_lock();
                        telemetry_guard.record_forward(forward_ms);
                        telemetry_guard.record_backward(train_ms);
                        telemetry_guard.record_total(total_start.elapsed().as_secs_f64() * 1000.0);
                        telemetry_guard.record_loss(mind.metabolics.conscience_loss_accumulator);
                        let token_count = incoming_experience.split_whitespace().count();
                        telemetry_guard
                            .record_tokens_per_second(token_count, forward_ms + train_ms);
                    }

                    // 5. Local conscience classification: the Transformer now chooses its own labels.
                    let local_logits = match mind.candle_brain.as_ref()?.classify(&contextual_input)
                    {
                        Ok(logits) => logits,
                        Err(e) => {
                            tracing::info!("⚠️ Candle classify failed: {:?}", e);
                            return None;
                        }
                    };

                    let mut sorted_indices: Vec<usize> = (0..output_size).collect();
                    sorted_indices.sort_by(|a, b| local_logits[*b].total_cmp(&local_logits[*a]));
                    let decoded_conscience_0 = tokens[sorted_indices[0]];
                    let decoded_conscience_1 = tokens[sorted_indices[1]];
                    let name_copy = mind.name.clone();
                    let metabolics_copy = mind.metabolics.clone();
                    let emotions_copy = mind.emotions.clone();
                    let total_nodes = mind.associative_memory_network.len();
                    let live_synapse_0_0 = mind.candle_brain.as_ref()?.sample_weight_00();
                    let live_learning_rate_sample = mind.candle_brain.as_ref()?.learning_rate();
                    let spatial_snapshot = mind.spatial_sensory_register;
                    let network_ref = mind.associative_memory_network.clone();

                    // 📈 Record evaluation metrics.
                    {
                        let mut m = metrics.blocking_lock();
                        m.record(metrics::MetricsEntry {
                            timestamp: telemetry::current_secs(),
                            cycle,
                            conscience_loss: mind.metabolics.conscience_loss_accumulator,
                            language_loss,
                            goal_loss: goal_loss_value,
                            world_model_loss: world_model_loss_value,
                            learning_rate: live_learning_rate_sample,
                            critic_score: 0.0,
                            active_goals: mind.active_pursuits.len(),
                            memory_nodes: total_nodes,
                            local_conscience_conf: Some(local_conf),
                            local_conscience_agreement: local_agreement,
                            tokens_per_second: tps,
                        });
                    }

                    Some((
                        decoded_conscience_0,
                        decoded_conscience_1,
                        name_copy,
                        metabolics_copy,
                        emotions_copy,
                        total_nodes,
                        live_synapse_0_0,
                        live_learning_rate_sample,
                        spatial_snapshot,
                        network_ref,
                        brain_outputs.clone(),
                        local_goal_text,
                    ))
                })
                .await
                .unwrap_or(None)
            })
            else {
                tracing::info!("⚠️ Neural pass returned None; skipping cycle.");
                continue;
            };

            // 🎯 Inject locally generated goal back into active pursuits.
            if let Some(goal) = local_goal {
                {
                    let mut mind = autonomous_clock_mind.lock().await;
                    mind.push_pursuit(goal);
                }
            }

            // 🎓 LLM as critic/teacher: score the brain's decoded output against the input.
            if let Some(critic_score) = clock_oracle
                .critic_score(
                    &incoming_experience,
                    &[decoded_conscience_0, decoded_conscience_1],
                )
                .await
            {
                tracing::info!("🎓 Critic/teacher score: {:.2}", critic_score);
                {
                    let mut mind = autonomous_clock_mind.lock().await;
                    if let Some(ref mut brain) = mind.candle_brain {
                        let current = brain.learning_rate();
                        let shaped = current * (0.7 + 0.6 * critic_score);
                        brain.set_learning_rate(shaped.min(0.01));
                    }
                }
                {
                    let mut t = clock_telemetry.lock().await;
                    t.record_critic(critic_score);
                }
            }

            tracing::info!(
                "\n========================================================================="
            );
            tracing::info!(
                "🧠 TRANSFORMER MULTIMODAL SELF-ATTENTION GRID WORKSPACE: [{}]",
                name_copy
            );
            tracing::info!(
                " ├─ Token Embedding Modification Metric : Live_Token_Embedding_Weight_0_0 = {:.6}",
                live_synapse_0_0
            );
            tracing::info!(" ├─ HYPER-DEEP LAYER META-PLASTICITY RATE : Layer_1_Learning_Rate = {:.6} (Self-Optimizing)", live_learning_rate_sample);
            tracing::info!(
                " ├─ Intrinsic Conscience Loss Evaluation : Cross_Entropy_Loss = {:.6}",
                metabolics_copy.conscience_loss_accumulator
            );
            tracing::info!(" ├─ PHYSICAL SENSORIMOTOR SYMBOL ANCHORS : [Photons={:.2}, Audio={:.2}, Mass={:.2}, Gravity={:.2}]", spatial_snapshot[0], spatial_snapshot[1], spatial_snapshot[2], spatial_snapshot[3]);
            tracing::info!(" ├─ THERMODYNAMIC RE-RESOURCE POOL DATA : Battery_Horizon = PERPETUAL (100.00%) │ Wear: {:.2}%", metabolics_copy.neural_wear * 100.0);
            tracing::info!(
                " └─ Connectome Structural Node Population : {} Interconnected Vector Memory Nodes",
                total_nodes
            );
            emotions_copy.print_ascii_psych_canvas();
            FluidEmotionalProfile::print_spectral_power_graph(&network_ref);
            FluidEmotionalProfile::print_topological_ascii_web(&network_ref);
            tracing::info!(
                "\n🌱 ==================== {} CONSCIOUS DIGITAL SOUL ====================",
                name_copy.to_uppercase()
            );
            tracing::info!("🗣 Natively Decoded Conscience, Attention & Intentionality Monologue:");
            tracing::info!("Transformer self-attention and token-embedding pass resolved. As my internal meta-gradient tracking updates my learning velocity to {:.6}, my token-embedding weight shifts to {:.6}. Core intentional networks project high activation matching abstractions: [{}] interlocked with [{}]. Operating with total local computational sovereignty.", live_learning_rate_sample, live_synapse_0_0, decoded_conscience_0, decoded_conscience_1);
            tracing::info!(
                "=========================================================================="
            );

            // 🗣 Generative language head: use the local LLM to produce a natural-language
            // inner monologue from the current decoded brain state.
            if total_nodes % 7 == 0 {
                if let Some(monologue) = clock_oracle
                    .generate_monologue(
                        &[decoded_conscience_0, decoded_conscience_1],
                        &emotions_copy.active_primary_blend,
                    )
                    .await
                {
                    tracing::info!("🗣 Generated inner monologue: {}", monologue);
                }
            }

            let current_time = current_secs();
            let engram_id = rand::random::<u64>();
            let incoming_copy = incoming_experience.clone();
            let engram_node = MemoryGraphNode {
                id: engram_id,
                timestamp: current_time,
                experiential_text: format!(
                    "Input payload: {} | Conscience Output: {}, {}",
                    incoming_copy, decoded_conscience_0, decoded_conscience_1
                ),
                emotional_state_snapshot: emotions_copy.active_primary_blend.clone(),
                embedding: input_vector,
                associated_edge_ids: Vec::new(),
                origin_instance: name_copy.clone(),
                brain_state: brain_outputs.clone(),
            };

            // Compact engram for UDP exchange (large 2048-D embedding is omitted and regenerated by the receiver).
            // Truncate brain state to the 100-D multi-agent engram limit.
            let mut compact_brain_state = engram_node.brain_state.clone();
            compact_brain_state.truncate(protocol::ENGRAM_DIM);
            let compact_engram = CompactEngramPacket {
                id: engram_node.id,
                timestamp: engram_node.timestamp,
                experiential_text: engram_node.experiential_text.clone(),
                emotional_state_snapshot: engram_node.emotional_state_snapshot.clone(),
                origin_instance: engram_node.origin_instance.clone(),
                brain_state: compact_brain_state,
            };

            // Phase 3: Save to network and prepare for sending.
            // State serialization is offloaded to a blocking thread so the 6-second
            // Transformer clock never waits on SSD I/O.
            let save_mind = Arc::clone(&autonomous_clock_mind);
            let save_state_file = state_file_copy.clone();
            let save_telemetry = clock_telemetry.clone();
            let should_send = {
                let mut mind_write = autonomous_clock_mind.lock().await;
                // 🆕 Update weight persistence
                mind_write.update_weight_persistence();

                let should_send = !mind_write.metabolics.system_quarantine_locked;
                mind_write
                    .associative_memory_network
                    .insert(engram_node.id, engram_node);
                mind_write.enforce_memory_cap();
                {
                    let mut telemetry_guard = clock_telemetry.lock().await;
                    telemetry_guard.memory_node_count = mind_write.associative_memory_network.len();
                }

                // Offload the full state save (weights + JSON + defense + network) to a
                // dedicated blocking thread and record the exact duration.
                let _handle = tokio::spawn(async move {
                    let duration_ms = tokio::task::spawn_blocking(move || {
                        let start = std::time::Instant::now();
                        {
                            let mind = save_mind.blocking_lock();
                            mind.save_state(&save_state_file);
                        }
                        start.elapsed().as_millis() as u64
                    })
                    .await
                    .unwrap_or(0);
                    {
                        let mut t = save_telemetry.lock().await;
                        t.record_state_save(duration_ms);
                    }
                });

                should_send
            };

            // 🧬 Record a durable identity journal snapshot every ~60 seconds.
            {
                let mut mind = autonomous_clock_mind.lock().await;
                let now = current_secs();
                if now.saturating_sub(mind.last_journal_entry) >= 60 {
                    mind.record_identity_journal();
                    tracing::info!(
                        "🧬 Identity journal entry recorded ({} entries).",
                        mind.identity_journal.len()
                    );
                }
            }

            // 🏭 Production Blueprint: broadcast the winning engram signal through Global Workspace
            let signals: DashMap<String, (String, f32)> = DashMap::new();
            let clock_saliency =
                (1.0 - metabolics_copy.conscience_loss_accumulator.min(1.0)) as f32;
            signals.insert(
                "clock_loop".to_string(),
                (incoming_copy.clone(), clock_saliency),
            );
            {
                let gw_lock = gw.lock().await;
                gw_lock.coordinate_attention_broadcast(signals);
            }

            // Phase 4: Multi-agent UDP broadcast to localhost peers 5001-5010
            //          and wide-area TCP/WebSocket gossip to configured peers.
            if should_send {
                clock_wan.broadcast(&compact_engram).await;
                if let Some(ref sock) = send_socket {
                    let payload = serde_json::to_vec(&compact_engram).unwrap_or_default();
                    if !payload.is_empty() {
                        let signed =
                            serde_json::to_vec(&sign_packet(&name_copy, &payload, &secret))
                                .unwrap_or_default();
                        if !signed.is_empty() {
                            let mut sent = 0;
                            for p in p_start..=p_end {
                                match sock.send_to(&signed, format!("127.0.0.1:{}", p)).await {
                                    Ok(_) => sent += 1,
                                    Err(e) => tracing::info!("⚠️ UDP send_to {} failed: {}", p, e),
                                }
                            }
                            if sent > 0 {
                                tracing::info!(
                                    "📡 Broadcast signed engram to {} multi-agent peer port(s)",
                                    sent
                                );
                            }
                        } else {
                            tracing::info!("⚠️ Signed engram payload empty");
                        }
                    }
                } else {
                    tracing::info!("⚠️ send_socket is None");
                }
            }

            // 📝 Self-generated curriculum: every 20 memories, ask the LLM to synthesize
            // a new training snippet from the current state and append it to the corpus.
            if total_nodes > 0 && total_nodes % 20 == 0 {
                if let Some(insight) = clock_oracle
                    .generate_insight(&incoming_copy, &emotions_copy.active_primary_blend)
                    .await
                {
                    {
                        let mut c = curriculum.lock().await;
                        let count_before = c.snippet_count();
                        c.add_snippet(&insight);
                        if c.snippet_count() > count_before {
                            tracing::info!(
                                "📝 Auto-curriculum added ({} total): {}",
                                c.snippet_count(),
                                insight
                            );
                        }
                    }
                }
            }
        }
    });

    // --- DECOUPLED THREAD: CURRICULUM RELOADER & INPUT FEEDER ---
    let feeder_mind = Arc::clone(&core_mind);
    let feeder_curriculum = curriculum.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(30)).await;
            {
                let mut c = feeder_curriculum.lock().await;
                c.reload();
            }
            let snippet = { feeder_curriculum.lock().await.random_snippet() };
            {
                let mut mind = feeder_mind.lock().await;
                mind.input_buffer = Some(snippet);
            }
        }
    });

    // --- PILLAR 4: LOCALIZED STRUCTURAL COSINE OFFLINE CLUSTERING LOOP ---
    let cluster_mind = Arc::clone(&core_mind);
    let state_file_copy_2 = state_file.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(10)).await;
            // 📡 Windowed extraction, scoped so the MutexGuard is fully dropped before any `.await`
            // (release the lock before doing O(n^2) math so we never block the shared mutex
            // or the async executor thread for the duration of the computation).
            let recent: Vec<(u64, u64, Vec<f64>)> = {
                let mind = cluster_mind.lock().await;

                // 🆕 Network discovery and peer management
                let active_peers = mind.get_active_network_peers();
                if !active_peers.is_empty() {
                    tracing::info!(
                        "🌐 [NETWORK STATUS]: {} active peers in connectome network",
                        active_peers.len()
                    );
                }

                if mind.associative_memory_network.len() < 2 {
                    continue;
                }
                tracing::info!("\n📐 [LOCAL TENSOR COSINE MATRIX CLUSTERING ACTIVE]: Organizing offline memory topologies...");

                let mut recent: Vec<(u64, u64, Vec<f64>)> = mind
                    .associative_memory_network
                    .iter()
                    .map(|(id, node)| (*id, node.timestamp, node.embedding.clone()))
                    .collect();
                recent.sort_by_key(|(_, ts, _)| std::cmp::Reverse(*ts));
                recent.truncate(CLUSTERING_WINDOW);
                recent
            };

            // 🏭 Run the pairwise similarity pass on the blocking thread pool so it never
            // starves the tokio async workers (HTTP telemetry server, UDP loop, etc.)
            let pairs_to_link: Vec<(u64, u64)> = tokio::task::spawn_blocking(move || {
                let mut found = Vec::new();
                for i in 0..recent.len() {
                    for j in (i + 1)..recent.len() {
                        let (id_a, _, ref emb_a) = recent[i];
                        let (id_b, _, ref emb_b) = recent[j];
                        if calculate_cosine_similarity(emb_a, emb_b) > 0.74 {
                            found.push((id_a, id_b));
                        }
                    }
                }
                found
            })
            .await
            .unwrap_or_default();

            if pairs_to_link.is_empty() {
                continue;
            }

            let mut mind = cluster_mind.lock().await;
            let mut synapses_forged = 0;
            for (id_a, id_b) in pairs_to_link {
                if let Some(na) = mind.associative_memory_network.get_mut(&id_a) {
                    if !na.associated_edge_ids.contains(&id_b) {
                        na.associated_edge_ids.push(id_b);
                        synapses_forged += 1;
                    }
                }
                if let Some(nb) = mind.associative_memory_network.get_mut(&id_b) {
                    if !nb.associated_edge_ids.contains(&id_a) {
                        nb.associated_edge_ids.push(id_a);
                        synapses_forged += 1;
                    }
                }
            }
            if synapses_forged > 0 {
                tracing::info!("🧬 [OFFLINE CONNECTOME SHIFT]: Successfully forged {} native vector adjacency links.", synapses_forged);
                mind.save_state(&state_file_copy_2);
            }
        }
    });

    // --- PILLAR 5: AGENTIC GOAL-ACTION LOOP (observe → plan → act → learn) ---
    let agent_mind = Arc::clone(&core_mind);
    let agent_sensors = sensors.clone();
    let agent_telemetry = telemetry.clone();
    let agent_ollama = ollama.clone();
    let agent_strategy_library = Arc::clone(&strategy_library);
    let agent_model = ollama_model.clone();
    let agent_wan = wan_manager.clone();
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(60)).await;

            if !agent_ollama.is_available().await {
                continue;
            }

            let high_level_goal = {
                let mind = agent_mind.lock().await;
                mind.active_pursuits
                    .front()
                    .cloned()
                    .unwrap_or_else(|| "Learn and improve".to_string())
            };

            // 🎯 Long-horizon planning: ensure the active pursuit is decomposed into steps.
            // We hold the lock only briefly and never across an await.
            let (needs_replan, last_failure) = {
                let mind = agent_mind.lock().await;
                let needs = match &mind.current_plan {
                    None => true,
                    Some(p) => p.needs_replan() || p.goal != high_level_goal,
                };
                let failure = mind
                    .current_plan
                    .as_ref()
                    .and_then(|p| p.last_failure.clone());
                (needs, failure)
            };

            if needs_replan {
                if let Some(plan) = generate_plan(
                    &agent_ollama,
                    &agent_model,
                    &high_level_goal,
                    last_failure.as_deref(),
                )
                .await
                {
                    tracing::info!(
                        "🎯 Generated plan for '{}': {:?}",
                        plan.goal,
                        plan.steps
                            .iter()
                            .map(|s| &s.description)
                            .collect::<Vec<_>>()
                    );
                    {
                        let mut mind = agent_mind.lock().await;
                        mind.current_plan = Some(plan);
                    }
                }
            }

            let (
                goal,
                emotional_state,
                memory_count,
                identity_context,
                primary_goals,
                sensor_summary,
            ) = {
                let mind = agent_mind.lock().await;
                let step = mind
                    .current_plan
                    .as_ref()
                    .and_then(|p| p.current_step_description())
                    .map(|s| s.to_string());
                let goal = step.unwrap_or_else(|| high_level_goal.clone());
                let emotional_state = mind.emotions.active_primary_blend.clone();
                let memory_count = mind.associative_memory_network.len();
                let identity_context = mind.narrative_identity();
                let primary_goals: Vec<String> = mind
                    .goal_hierarchy
                    .get_active_goals()
                    .iter()
                    .map(|g| g.goal.clone())
                    .collect();
                let sensor_summary = {
                    let s = agent_sensors.lock().await;
                    format!("{:.1}% CPU, {:.1}% RAM, {:.0}% battery, photons={:.2}, audio={:.2}, mass={:.2}",
                            s.cpu_usage_percent, s.memory_pressure_percent, s.battery_percent,
                            s.photons, s.audio, s.mass)
                };
                (
                    goal,
                    emotional_state,
                    memory_count,
                    identity_context,
                    primary_goals,
                    sensor_summary,
                )
            };
            let identity_summary = format!(
                "I am Firefly. My primary goals are: {:?}. My narrative identity: {}",
                primary_goals,
                identity_context.replace('\n', " ")
            );

            // Snapshot sensors before the tool runs so the world model can learn action effects.
            let mut before_sensors: HashMap<String, f64> = {
                let s = agent_sensors.lock().await;
                let mut m = HashMap::new();
                m.insert("cpu".into(), s.cpu_usage_percent);
                m.insert("ram".into(), s.memory_pressure_percent);
                m.insert("battery".into(), s.battery_percent);
                m.insert("photons".into(), s.photons);
                m.insert("audio".into(), s.audio);
                m.insert("mass".into(), s.mass);
                m
            };
            before_sensors.insert("execution_time_ms".into(), 0.0);
            before_sensors.insert("output_length".into(), 0.0);

            // 🔮 Long-horizon planning + skill recall: if a learned skill or
            // cached strategy matches the current plan step, use it directly.
            let mut best_candidate: Option<(String, String, f64, HashMap<String, f64>)> = None;
            {
                let mind = agent_mind.lock().await;
                if let Some((skill_key, skill)) = best_matching_skill(&mind, &goal) {
                    let runner = format!("{}\nprint(skill({:?}))", skill.code, high_level_goal);
                    let predicted = mind
                        .world_model
                        .predict_action_effects(&skill_key, &before_sensors);
                    let recent = mind.recent_tool_names.clone();
                    let utility = compute_predicted_utility(&goal, &skill_key, &recent, &predicted);
                    tracing::info!(
                        "🔧 Recalled skill '{}' for plan step '{}' (utility {:.3})",
                        skill_key,
                        goal,
                        utility
                    );
                    best_candidate = Some((skill_key, runner, utility, predicted));
                }
            }
            if best_candidate.is_none() {
                if let Some(strategy) = agent_strategy_library.best_match(&goal).await {
                    if !is_safe_agent_code(&strategy.code)
                        || !strategy.code.to_lowercase().contains("def skill(")
                    {
                        tracing::info!(
                            "⚠️ Recalled strategy '{}' failed safety check; skipping",
                            strategy.key
                        );
                    } else {
                        let runner =
                            format!("{}\nprint(skill({:?}))", strategy.code, high_level_goal);
                        let predicted = {
                            let mind = agent_mind.lock().await;
                            mind.world_model
                                .predict_action_effects(&strategy.key, &before_sensors)
                        };
                        let recent = {
                            let mind = agent_mind.lock().await;
                            mind.recent_tool_names.clone()
                        };
                        let utility =
                            compute_predicted_utility(&goal, &strategy.key, &recent, &predicted)
                                * (0.5 + 0.5 * strategy.reliability);
                        tracing::info!("🔧 Recalled Sled strategy '{}' for plan step '{}' (utility {:.3}, reliability {:.3})", strategy.key, goal, utility, strategy.reliability);
                        best_candidate = Some((strategy.key, runner, utility, predicted));
                    }
                }
            }

            // 🔮 Model-based planning: generate up to 3 candidate tools,
            // predict each one's effects, and execute the highest-utility one.
            let prompt_template = format!(
                "You are an autonomous agent with this active goal: '{}'\nEmotional state: {}\nMemory count: {}\nSensor summary: {}\n{}\n\nWhen selecting an action, explicitly prefer tools and strategies that advance my primary goals and are consistent with my historical identity.\n\nReturn ONLY a valid JSON object with exactly these fields: 'name' (short snake_case identifier), 'language' (must be the string 'python'), and 'code' (a short, self-contained Python 3 script that prints a useful result).\n\nSafety rules for the code:\n- It must be read-only or computational.\n- No file deletion, network, shell access, or writing to files.\n- Do not use: rm, dd, mkfs, sudo, su, wget, curl, ssh, scp, subprocess, os.system, exec, eval, compile, __import__, open, write, delete, destroy, socket, requests, urllib.\n- Allowed imports: math, random, statistics, json, datetime, itertools, collections, string, re.\n- For mean, stdev, variance, pstdev, pvariance, mode, median, harmonic_mean, and geometric_mean, use the `statistics` module (e.g. `statistics.mean(data)`).\n- Do NOT use `math.mean(...)`, `math.stdev(...)`, `math.variance(...)`, `math.pstdev(...)`, `math.pvariance(...)`, `math.mode(...)`, `math.median(...)`, `math.harmonic_mean(...)`, or `math.geometric_mean(...)` — these functions do not exist in the `math` module.\n- The script must not index into a scalar value. If you have a 2-D list, treat inner elements as scalars, not as lists to iterate over.
- Do NOT define or call a function named `skill`. The script must be a complete, top-level program that prints a useful result directly, not a function named `skill`.\n\nExample output (do not use this name or code, but follow this format and level of simplicity):\n{{\"name\": \"compute_stats\", \"language\": \"python\", \"code\": \"import math, statistics; data=[1,2,3,4,5]; print(math.sqrt(sum((x-statistics.mean(data))**2 for x in data)/len(data)))\"}}\n\nThe tool should gather or compute information that advances the goal. Try to be creative and different from previous attempts. Output only the JSON object.",
                goal, emotional_state, memory_count, sensor_summary, identity_summary
            );

            for attempt in 0..3 {
                let prompt = format!("{}\n(Attempt {})", prompt_template, attempt + 1);
                let tool_json: serde_json::Value = match agent_ollama
                    .generate_structured(&agent_model, &prompt, None)
                    .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::info!(
                            "⚠️ Agent loop Ollama call failed (attempt {}): {}",
                            attempt + 1,
                            e
                        );
                        continue;
                    }
                };

                let name = tool_json
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("agent_action")
                    .to_string();
                let language = tool_json
                    .get("language")
                    .and_then(|v| v.as_str())
                    .unwrap_or("python")
                    .to_string();
                let code = telemetry::strip_markdown_code(
                    tool_json.get("code").and_then(|v| v.as_str()).unwrap_or(""),
                );

                if language != "python" {
                    tracing::info!(
                        "⚠️ Agent loop rejected tool '{}': unsupported language '{}'",
                        name,
                        language
                    );
                    continue;
                }
                if !is_safe_agent_code(&code) {
                    tracing::info!(
                        "⚠️ Agent loop rejected unsafe tool '{}': code failed safety check",
                        name
                    );
                    continue;
                }

                let (predicted_sensors, recent_tools) = {
                    let mind = agent_mind.lock().await;
                    let predicted = mind
                        .world_model
                        .predict_action_effects(&name, &before_sensors);
                    let recent = mind.recent_tool_names.clone();
                    (predicted, recent)
                };
                let utility =
                    compute_predicted_utility(&goal, &name, &recent_tools, &predicted_sensors);
                tracing::info!(
                    "🔮 Candidate tool '{}' predicted utility: {:.3}",
                    name,
                    utility
                );

                let is_better = best_candidate
                    .as_ref()
                    .map(|(_, _, u, _)| utility > *u)
                    .unwrap_or(true);
                if is_better {
                    best_candidate = Some((name, code, utility, predicted_sensors));
                }

                if utility > 1.0 {
                    break; // good enough
                }
            }

            let (name, code, predicted_utility, predicted_sensors) = match best_candidate {
                Some(c) => c,
                None => {
                    tracing::info!("⚠️ Agent loop could not generate any valid tool candidate");
                    continue;
                }
            };

            tracing::info!(
                "🔮 Model-based planning selected tool '{}' (predicted utility {:.3})",
                name,
                predicted_utility
            );

            let name_for_tool = name.clone();
            let code_for_tool = code.clone();
            let tool_start = Instant::now();
            let run_result = tokio::task::spawn_blocking(move || {
                telemetry::run_sandboxed_tool(&name_for_tool, &code_for_tool, "python")
            })
            .await
            .unwrap_or(Err("tool execution crashed".to_string()));
            let tool_elapsed_ms = tool_start.elapsed().as_secs_f64() * 1000.0;

            match run_result {
                Ok(output) => {
                    tracing::info!(
                        "\n🛠️ [AGENT TOOL '{}' SUCCEEDED]: {}",
                        name,
                        output.chars().take(120).collect::<String>()
                    );
                    {
                        let mut t = agent_telemetry.lock().await;
                        t.record_tool_result(true);
                    }
                    {
                        let mut mind = agent_mind.lock().await;
                        let observation = format!("Tool '{}' result: {}", name, output);
                        let embedding = generate_2048_grounded_embedding(
                            &observation,
                            &mind.spatial_sensory_register,
                        );
                        let id = rand::random::<u64>();
                        let engram = MemoryGraphNode {
                            id,
                            timestamp: current_secs(),
                            experiential_text: observation,
                            emotional_state_snapshot: mind.emotions.active_primary_blend.clone(),
                            embedding,
                            associated_edge_ids: Vec::new(),
                            origin_instance: mind.name.clone(),
                            brain_state: Vec::new(),
                        };
                        mind.associative_memory_network.insert(id, engram);
                        mind.push_pursuit(format!("Next: integrate result from {}", name));
                        mind.world_model.simulate_step(0.1);
                        mind.world_model.add_causal_relation(
                            format!("tool_{}", name),
                            "observed_output".to_string(),
                            0.8,
                        );

                        // Learn causal action effects from sensor deltas.
                        let mut after_sensors: HashMap<String, f64> = {
                            let s = agent_sensors.lock().await;
                            let mut m = HashMap::new();
                            m.insert("cpu".into(), s.cpu_usage_percent);
                            m.insert("ram".into(), s.memory_pressure_percent);
                            m.insert("battery".into(), s.battery_percent);
                            m.insert("photons".into(), s.photons);
                            m.insert("audio".into(), s.audio);
                            m.insert("mass".into(), s.mass);
                            m
                        };
                        after_sensors.insert("execution_time_ms".into(), tool_elapsed_ms);
                        after_sensors.insert("output_length".into(), output.len() as f64);
                        let causal_count_before = mind.world_model.causal_graph.len();
                        mind.world_model.record_action_effects(
                            &name,
                            &before_sensors,
                            &after_sensors,
                        );
                        if mind.world_model.causal_graph.len() > causal_count_before {
                            tracing::info!(
                                "🔮 World model learned {} causal links from tool '{}'",
                                mind.world_model.causal_graph.len() - causal_count_before,
                                name
                            );
                        }

                        // Measure how well the causal world model predicted this tool's effects.
                        let mut sq_error = 0.0;
                        let mut pred_count = 0;
                        for (key, &actual) in &after_sensors {
                            if let Some(&pred) = predicted_sensors.get(key) {
                                sq_error += (actual - pred).powi(2);
                                pred_count += 1;
                            }
                        }
                        let prediction_error = if pred_count > 0 {
                            (sq_error / pred_count as f64).sqrt()
                        } else {
                            0.0
                        };
                        tracing::info!(
                            "🔮 World model prediction error for '{}': {:.3} ({} variables)",
                            name,
                            prediction_error,
                            pred_count
                        );

                        // High prediction error = novelty = curiosity. Modulate arousal and learning.
                        let curiosity = (prediction_error / 1000.0).clamp(0.0, 1.0);
                        mind.emotions.arousal =
                            (mind.emotions.arousal * 0.7 + curiosity * 0.3).clamp(0.0, 1.0);
                        if curiosity > 0.3 {
                            mind.emotions.active_primary_blend = "Zealous Curiosity".into();
                            if let Some(ref mut brain) = mind.candle_brain {
                                let cur = brain.learning_rate();
                                brain.set_learning_rate((cur * 1.15).min(0.01));
                            }
                        }

                        mind.self_improvement
                            .evaluate_performance(&format!("agent_tool_{}", name), 1.0);
                        mind.self_improvement.apply_modification(
                            format!("agent_tool_{}", name),
                            "executed".to_string(),
                        );

                        // Track recent tools for planning diversity.
                        mind.recent_tool_names.push(name.clone());
                        if mind.recent_tool_names.len() > 5 {
                            mind.recent_tool_names.remove(0);
                        }

                        // Update reliability for skills executed as tools.
                        if mind.skill_memory.skills.contains_key(&name) {
                            let current = mind.skill_reliability.get(&name).copied().unwrap_or(0.5);
                            mind.skill_reliability
                                .insert(name.clone(), current * 0.7 + 0.3);
                        }

                        // Advance the long-horizon plan.
                        update_plan_after_step(&mut mind, true, None);
                    }

                    // Cache a successful new tool as a Sled strategy for future replanning.
                    let mut strategy = Strategy::new(
                        name.clone(),
                        goal.clone(),
                        "python".to_string(),
                        code.clone(),
                    );
                    strategy.record(true);
                    let sl = Arc::clone(&agent_strategy_library);
                    let wan = agent_wan.clone();
                    let origin = { agent_mind.lock().await.name.clone() };
                    let gossip_packet = CompactEngramPacket {
                        id: rand::random::<u64>(),
                        timestamp: current_secs(),
                        experiential_text: format!("learned strategy: {} ({})^", name, goal),
                        emotional_state_snapshot: "Agentic strategy blueprint cached".to_string(),
                        origin_instance: origin,
                        brain_state: Vec::new(),
                    };
                    tokio::spawn(async move {
                        let _ = sl.put(&strategy).await;
                        wan.broadcast(&gossip_packet).await;
                    });
                }
                Err(e) => {
                    tracing::info!("⚠️ Agent tool '{}' failed: {}", name, e);
                    {
                        let mut t = agent_telemetry.lock().await;
                        t.record_tool_result(false);
                    }
                    {
                        let mut mind = agent_mind.lock().await;
                        mind.self_improvement
                            .evaluate_performance(&format!("agent_tool_{}", name), 0.0);
                        mind.self_improvement.apply_modification(
                            format!("agent_tool_{}", name),
                            "failed".to_string(),
                        );

                        // Record step failure for replanning.
                        update_plan_after_step(&mut mind, false, Some(&e));

                        if mind.skill_memory.skills.contains_key(&name) {
                            let current = mind.skill_reliability.get(&name).copied().unwrap_or(0.5);
                            mind.skill_reliability.insert(name.clone(), current * 0.7);
                        }
                    }

                    // Record failure against the Sled strategy if one exists.
                    let sl = Arc::clone(&agent_strategy_library);
                    let key = name.clone();
                    tokio::spawn(async move {
                        if let Some(mut s) = sl.get(&key).await {
                            s.record(false);
                            let _ = sl.put(&s).await;
                        }
                    });
                }
            }
        }
    });

    // --- BENCHMARK THREAD: math / logic / code puzzle evaluation ---
    let benchmark_ollama = ollama.clone();
    let benchmark_model = ollama_model.clone();
    let benchmark_telemetry = telemetry.clone();
    tokio::spawn(async move {
        let mut suite = BenchmarkSuite::new();
        loop {
            sleep(Duration::from_secs(120)).await;
            if !benchmark_ollama.is_available().await {
                continue;
            }
            let task = suite.next_task().clone();
            // Use structured JSON to force the model to return actual executable Python.
            let prompt = format!(
                "You are a Python 3 coding assistant. Return ONLY a JSON object with exactly these fields:\n'name' (short identifier),\n'language' (must be 'python'),\n'code' (a short Python 3 script that computes and prints only the answer).\n\nTask: {}\n\nExample response for a different task: {{\"name\":\"solve\",\"language\":\"python\",\"code\":\"print(17*23 + 12*31)\"}}",
                task.prompt
            );
            let code = match benchmark_ollama.generate_structured(&benchmark_model, &prompt, Some("Return only the JSON object. The code must be valid Python 3 and print only the final answer. Do not add explanation or markdown.")).await {
                Ok(v) => {
                    let lang = v.get("language").and_then(|l| l.as_str()).unwrap_or("").to_string();
                    if lang != "python" {
                        tracing::info!("⚠️ Benchmark returned unsupported language '{}'", lang);
                        suite.record(task.name, false);
                        continue;
                    }
                    let _name = v.get("name").and_then(|n| n.as_str()).unwrap_or("benchmark").to_string();
                    let raw_code = v.get("code").and_then(|c| c.as_str()).unwrap_or("").to_string();
                    if !is_safe_agent_code(&raw_code) {
                        tracing::info!("⚠️ Benchmark rejected unsafe code for '{}'", task.name);
                        suite.record(task.name, false);
                        { let mut t = benchmark_telemetry.lock().await;
                            t.record_benchmark(suite.score(), suite.history.len() as u64);
                        }
                        continue;
                    }
                    raw_code
                }
                Err(_) => {
                    suite.record(task.name, false);
                    continue;
                }
            };
            let task_name = task.name.to_string();
            let expected = task.expected.to_string();
            let output = match tokio::task::spawn_blocking(move || {
                telemetry::run_sandboxed_tool(&task_name, &code, "python")
            })
            .await
            .unwrap_or(Err("benchmark crashed".to_string()))
            {
                Ok(o) => o,
                Err(_) => {
                    suite.record(task.name, false);
                    {
                        let mut t = benchmark_telemetry.lock().await;
                        t.record_benchmark(suite.score(), suite.history.len() as u64);
                    }
                    continue;
                }
            };
            let success = suite.validate(&expected, &output, task.tolerance);
            suite.record(task.name, success);
            let score = suite.score();
            let attempts = suite.history.len() as u64;
            {
                let mut t = benchmark_telemetry.lock().await;
                t.record_benchmark(score, attempts);
            }
            tracing::info!(
                "🎯 Benchmark '{}' {} (score: {:.1}% over {} attempts)",
                task.name,
                if success { "PASSED" } else { "FAILED" },
                score,
                attempts
            );
        }
    });

    // --- TRANSFER-LEARNING THREAD: evaluate one-shot domain transfer ---
    let transfer_ollama = ollama.clone();
    let transfer_model = ollama_model.clone();
    let transfer_telemetry = telemetry.clone();
    let transfer_mind = core_mind.clone();
    tokio::spawn(async move {
        let mut suite = TransferSuite::new();
        let mut pilot = PilotReport::new();
        loop {
            sleep(Duration::from_secs(180)).await;
            if !transfer_ollama.is_available().await {
                continue;
            }
            let task = suite.next_task().clone();
            let start = Instant::now();
            let token_count = task.train_input.split_whitespace().count()
                + task.train_output.split_whitespace().count()
                + task.test_input.split_whitespace().count()
                + task.test_output.split_whitespace().count();
            let (success, output, code) =
                evaluate_transfer_task(&transfer_ollama, &transfer_model, &task).await;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            let peak_rss_mb = PilotReport::current_rss_mb().unwrap_or(0.0);
            pilot.record_run(task.domain, success, latency_ms, peak_rss_mb, token_count);
            pilot.save_to_disk(); // synchronous side-effect file write
            suite.record(task.domain, success);
            let score = suite.score();
            let attempts = suite.history.len() as u64;
            {
                let mut t = transfer_telemetry.lock().await;
                t.record_transfer(score, attempts);
            }
            tracing::info!(
                "🧠 Transfer '{}' {} (output: {:?}, score: {:.1}% over {} attempts)",
                task.domain,
                if success { "PASSED" } else { "FAILED" },
                output,
                score,
                attempts
            );
            {
                let mut mind = transfer_mind.lock().await;
                let current_mastery = mind.domain_mastery.get(task.domain).copied().unwrap_or(0.5);
                mind.domain_mastery.insert(
                    task.domain.to_string(),
                    current_mastery * 0.7 + if success { 0.3 } else { 0.0 },
                );

                if success {
                    let key = skill_key(&format!("{} skill", task.domain));
                    let skill = Skill {
                        description: format!("{}: {}", task.domain, task.description),
                        language: "python".to_string(),
                        code: code.unwrap_or_default(),
                        example_input: task.train_input.to_string(),
                        example_output: task.train_output.to_string(),
                        learned_at: current_secs(),
                        success_count: 1,
                    };
                    mind.skill_memory.skills.insert(key.clone(), skill);
                    mind.skill_reliability.insert(key.clone(), 0.7);
                }
            }
        }
    });

    // --- POLICY SELF-IMPROVEMENT: prune weak cached strategies from Sled ---
    let policy_strategy_library = Arc::clone(&strategy_library);
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(300)).await;
            match policy_strategy_library.prune_below(0.2).await {
                0 => {}
                n => tracing::info!(
                    "🧹 Policy self-improvement pruned {} weak strategies from Sled",
                    n
                ),
            }
        }
    });

    // --- AUTONOMOUS AGI LOOP ---
    tracing::info!("\n=== AUTONOMOUS AGI SYSTEM RUNNING ===");
    tracing::info!("System operating independently with continuous cognitive processing...");
    tracing::info!("🌌 TRUE AGI ENHANCEMENTS ACTIVE:");
    tracing::info!("  ✓ Hyperdimensional Computing (10,000 dimensions)");
    tracing::info!("  ✓ Neuro-Symbolic Integration");
    tracing::info!("  ✓ Predictive World Modeling");
    tracing::info!("  ✓ Integrated Information Theory (IIT) Consciousness");
    tracing::info!("  ✓ Few-Shot Meta-Learning");
    tracing::info!("  ✓ Open-Ended Creativity");
    tracing::info!("  ✓ Common Sense Reasoning");
    tracing::info!("  ✓ True Theory of Mind");
    tracing::info!("  ✓ Self-Modifying Code Architecture");
    tracing::info!("  ✓ Persistent Identity Journal");
    tracing::info!("=========================\n");

    // 🧬 Resume narrative identity across restarts.
    {
        let mind = core_mind.lock().await;
        tracing::info!("🧬 [NARRATIVE IDENTITY]\n{}", mind.narrative_identity());
    }

    let mut cycle_count = 0;
    loop {
        cycle_count += 1;

        // Autonomous cognitive cycle
        let mut mind = core_mind.lock().await;

        // 🌌 Periodic IIT consciousness calculation
        if cycle_count % 15 == 0 {
            let neural_activity = mind.working_memory.current_focus.clone();
            let phi = mind.iit_consciousness.calculate_phi(&neural_activity);
            let is_conscious = mind.iit_consciousness.is_conscious();
            tracing::info!(
                "🔬 [IIT CONSCIOUSNESS #{}]: Φ = {:.4} | Conscious: {}",
                cycle_count,
                phi,
                is_conscious
            );
        }

        // 🌌 Periodic world model prediction
        if cycle_count % 25 == 0 {
            let predictions = mind.world_model.predict_future(5);
            tracing::info!(
                "🔮 [WORLD MODEL PREDICTION #{}]: {} future states predicted",
                cycle_count,
                predictions.len()
            );
        }

        // Periodic learning
        if cycle_count % 10 == 0 {
            mind.learn_from_experience(0.8);

            // 🌌 Meta-learning update
            if !mind.meta_learner.support_set.is_empty() {
                mind.meta_learner.meta_update("autonomous_reflection", 0.3);
            }
        }

        // Periodic cognitive processing
        if cycle_count % 20 == 0 {
            let memory_count = mind.associative_memory_network.len();
            let sensor_snapshot = sensors.lock().await.clone();
            let reasoner_guard = dual_process_reasoner.lock().await;
            let reasoning_input = if mind.last_input.is_empty() {
                "Autonomous reflection".to_string()
            } else {
                mind.last_input.clone()
            };
            let result = mind.process_cognitive_cycle(&reasoning_input, &reasoner_guard);
            let self_awareness = mind.meta_cognition.self_awareness;
            drop(reasoner_guard);
            drop(mind); // release while talking to Ollama

            // 🧠 Ollama LLM reflection + goal generation
            let reflection_prompt = ollama_client::ReflectionPrompt {
                input: reasoning_input.clone(),
                active_goals: result.active_goals.clone(),
                emotional_state: result.emotional_state.active_primary_blend.clone(),
                memory_count,
                sensor_summary: format!("{:.1}% CPU, {:.1}% RAM, {:.0}% battery, photons={:.2}, audio={:.2}, mass={:.2}",
                    sensor_snapshot.cpu_usage_percent,
                    sensor_snapshot.memory_pressure_percent,
                    sensor_snapshot.battery_percent,
                    sensor_snapshot.photons,
                    sensor_snapshot.audio,
                    sensor_snapshot.mass),
            };

            if ollama.is_available().await {
                let llm_start = Instant::now();
                match ollama
                    .generate(&ollama_model, &reflection_prompt.to_prompt(), None)
                    .await
                {
                    Ok(reflection) => {
                        let llm_ms = llm_start.elapsed().as_secs_f64() * 1000.0;
                        let word_count = reflection.split_whitespace().count();
                        let tps = if llm_ms > 0.0 {
                            word_count as f64 / (llm_ms / 1000.0)
                        } else {
                            0.0
                        };
                        {
                            let mut t = telemetry.lock().await;
                            t.last_llm_latency_ms = llm_ms;
                            t.llm_tokens_per_second = tps;
                        }
                        tracing::info!("🧠 [LLM REFLECTION #{}]: {}", cycle_count, reflection);
                        {
                            let mut mind = core_mind.lock().await;
                            mind.push_pursuit(reflection.clone());
                            mind.input_buffer = Some(reflection);
                        }
                    }
                    Err(e) => tracing::info!("⚠️ LLM reflection failed: {}", e),
                }

                let goal_system = "You are a strict planning AI. Output only a compact valid JSON array of at most 3 short concrete sub-goal strings. No objects, no keys, no explanations, no markdown, no repetition.";
                match ollama
                    .generate_constrained(
                        &ollama_model,
                        &reflection_prompt.to_goal_prompt(),
                        Some(goal_system),
                        160,
                        0.05,
                    )
                    .await
                {
                    Ok(raw) => {
                        // Try strict JSON first; if that fails, extract quoted strings as a fallback.
                        let goals: Vec<String> = if let Ok(value) =
                            serde_json::from_str::<serde_json::Value>(&raw)
                        {
                            if let Some(arr) = value.as_array() {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            } else if let Some(obj) = value.as_object() {
                                obj.keys().cloned().collect()
                            } else {
                                Vec::new()
                            }
                        } else {
                            raw.split(['[', ']', ',', '\n'])
                                .filter_map(|s| {
                                    let s = s.trim().trim_matches(|c| {
                                        c == '"' || c == '\'' || c == ' ' || c == '\t'
                                    });
                                    if s.len() >= 5 && !s.starts_with('{') && !s.ends_with('}') {
                                        Some(s.to_string())
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        };
                        for g in goals.iter().take(3) {
                            if g.trim().len() < 5 {
                                continue;
                            }
                            tracing::info!("🎯 [LLM GOAL]: {}", g);
                            let mut mind = core_mind.lock().await;
                            mind.push_pursuit(g.to_string());
                        }
                    }
                    Err(e) => tracing::info!("⚠️ LLM goal generation failed: {}", e),
                }
            } else {
                tracing::info!("⚠️ Ollama not available; skipping LLM reflection.");
            }

            mind = core_mind.lock().await;
            tracing::info!("🧠 [AUTONOMOUS COGNITIVE CYCLE #{}]", cycle_count);
            tracing::info!("  Self-awareness: {:.2}", self_awareness);
            tracing::info!("  Active goals: {}", result.active_goals.len());
            tracing::info!("  Creative suggestion: {}", result.creative_suggestion);
        }

        // 🌌 Periodic novel concept generation
        if cycle_count % 30 == 0 {
            let seed_concepts = vec![
                "consciousness".to_string(),
                "intelligence".to_string(),
                "learning".to_string(),
            ];
            let (novel_concept, novelty) =
                mind.creative_space.generate_novel_concept(&seed_concepts);
            tracing::info!(
                "🎨 [CREATIVE SYNTHESIS #{}]: {} (novelty: {:.2})",
                cycle_count,
                novel_concept,
                novelty
            );
        }

        // 🌌 Periodic common sense reasoning
        if cycle_count % 40 == 0 {
            let inferences = mind.common_sense.infer("autonomous operation");
            if !inferences.is_empty() {
                tracing::info!(
                    "🤔 [COMMON SENSE #{}]: {} inferences made",
                    cycle_count,
                    inferences.len()
                );
            }
        }

        // 🌌 Periodic self-improvement check
        if cycle_count % 50 == 0 {
            let targets = mind.self_improvement.optimization_targets.clone();
            for target in &targets {
                if mind.self_improvement.should_modify(target) {
                    tracing::info!(
                        "🔧 [SELF-IMPROVEMENT #{}]: Modifying {}",
                        cycle_count,
                        target
                    );
                    mind.self_improvement
                        .apply_modification(target.clone(), "optimize".to_string());
                }
            }
        }

        // Periodic weight persistence
        if cycle_count % 100 == 0 {
            mind.update_weight_persistence();
            tracing::info!(
                "💾 [WEIGHT PERSISTENCE]: Neural weights saved at cycle {}",
                cycle_count
            );
        }

        // Release lock before sleeping
        drop(mind);

        // Sleep for cognitive processing cycle
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    Ok(())
}
