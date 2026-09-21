#![allow(dead_code, unused_variables)]
use dashmap::DashMap;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// System Parameters for Optimal Cognitive Resonance
pub const HYPER_DIMENSIONS: usize = 10_000;
pub const ATTENTION_THRESHOLD: f32 = 0.82;
pub const COMPRESSION_RATIO: f32 = 0.05;

#[derive(Clone)]
pub struct EngramNode {
    pub id: String,
    pub space_vector: Array1<f32>,
    pub scalar_modalities: DashMap<String, f32>,
}

pub struct GlobalWorkspace {
    pub memory_graph: Arc<DashMap<String, EngramNode>>,
    pub sub_agent_channels: DashMap<String, Arc<dyn Fn(String, f32) + Send + Sync>>,
}

impl GlobalWorkspace {
    pub fn new() -> Self {
        Self {
            memory_graph: Arc::new(DashMap::new()),
            sub_agent_channels: DashMap::new(),
        }
    }

    /// Evaluates distributed incoming signals and broadcasts the highest priority state
    pub fn coordinate_attention_broadcast(&self, signals: DashMap<String, (String, f32)>) {
        if let Some(winning_entry) = signals.iter().max_by(|a, b| {
            a.value()
                .1
                .partial_cmp(&b.value().1)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let (payload, saliency) = winning_entry.value();

            if *saliency < ATTENTION_THRESHOLD {
                return;
            }

            for agent in &self.sub_agent_channels {
                let callback = Arc::clone(agent.value());
                let payload_clone = payload.clone();
                let saliency_val = *saliency;

                tokio::spawn(async move {
                    callback(payload_clone, saliency_val);
                });
            }
        }
    }

    /// Stores an engram into the shared hyperdimensional memory graph
    pub fn store_engram(&self, id: String, vector: Array1<f32>, modalities: Vec<(String, f32)>) {
        let scalar_modalities = DashMap::new();
        for (k, v) in modalities {
            scalar_modalities.insert(k, v);
        }
        let node = EngramNode {
            id: id.clone(),
            space_vector: vector,
            scalar_modalities,
        };
        self.memory_graph.insert(id, node);
    }
}

pub const COGNITIVE_VELOCITY: f32 = 0.00206;
pub const COGNITIVE_AROUSAL: f32 = 0.72;

pub struct NeuroSymbolicEngine {
    pub rule_register: DashMap<String, String>,
}

impl NeuroSymbolicEngine {
    pub fn new() -> Self {
        Self {
            rule_register: DashMap::new(),
        }
    }

    pub fn add_rule(&self, premise: &str, conclusion: &str) {
        self.rule_register
            .insert(premise.to_string(), conclusion.to_string());
    }

    /// Processes high-dimensional matrix insights and subjects them to exact analytical verifications
    pub fn deliberate_execution_chain(
        &self,
        intuitive_activations: &DashMap<String, f32>,
        learning_rate_modifier: f32,
    ) -> Vec<String> {
        let mut proven_deductions = Vec::new();
        let calibrated_learning_rate =
            COGNITIVE_VELOCITY * learning_rate_modifier * COGNITIVE_AROUSAL;

        for rule in &self.rule_register {
            let premise = rule.key();
            let conclusion = rule.value();

            if let Some(activation_level) = intuitive_activations.get(premise) {
                if *activation_level > 0.85 {
                    proven_deductions.push(conclusion.clone());
                    let _optimized_weight = calibrated_learning_rate * (*activation_level);
                }
            }
        }

        proven_deductions
    }
}

pub const TARGET_STABILITY: f32 = 2.5;
pub const MAXIMUM_META_RATE: f32 = 0.001000;

/// Snapshot of hardware stress used by the thermodynamic governor.
/// All values are normalized to [0, 1] where higher means more stress.
#[derive(Clone, Debug, Default)]
pub struct ThermalState {
    pub cpu_usage: f32,
    pub memory_pressure: f32,
    pub battery_health: f32,
    pub cpu_temp: f32,
}

/// Thermodynamic metacognitive governor.
///
/// Treats physical hardware stress as a primary loss signal. When the machine
/// is hot, memory-pressured, or on degraded battery, the governor suppresses
/// dense floating-point computation and shifts toward sparse, low-power
/// execution.
pub struct ThermodynamicGovernor {
    /// 0..1 stress level, where 1.0 is the most constrained.
    pub stress: f32,
    /// Target sparsity for weight execution (0.0 = dense, 1.0 = maximally sparse).
    pub target_sparsity: f32,
    /// Whether to throttle the cognitive tick rate.
    pub throttle: bool,
}

impl ThermodynamicGovernor {
    pub fn new() -> Self {
        Self {
            stress: 0.0,
            target_sparsity: 0.0,
            throttle: false,
        }
    }

    /// Update stress from a `ThermalState` snapshot. The formula is a
    /// weighted sum that emphasizes thermal and battery degradation.
    pub fn update(&mut self, state: &ThermalState) {
        self.stress = (state.cpu_usage * 0.25
            + state.memory_pressure * 0.25
            + (1.0 - state.battery_health) * 0.35
            + state.cpu_temp * 0.15)
            .clamp(0.0, 1.0);

        // As stress rises, shift toward sparse execution and throttling.
        self.target_sparsity = self.stress.clamp(0.0, 0.9);
        self.throttle = self.stress > 0.7;
    }

    /// Apply a sparse mask to a vector in place. Keeps the top
    /// `(1 - target_sparsity)` activations by magnitude and zeroes the rest.
    pub fn sparsify(&self, vector: &mut [f32]) {
        if self.target_sparsity <= 0.0 || vector.is_empty() {
            return;
        }

        // Build a small index/magnitude vector and find the keep threshold.
        let mut indexed: Vec<(usize, f32)> = vector
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, x.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let keep_ratio = (1.0 - self.target_sparsity).clamp(0.0, 1.0);
        let keep_count = ((vector.len() as f32) * keep_ratio).ceil() as usize;
        let mut keep = vec![false; vector.len()];
        for (idx, _) in indexed.iter().take(keep_count) {
            keep[*idx] = true;
        }

        for (i, v) in vector.iter_mut().enumerate() {
            if !keep[i] {
                *v = 0.0;
            }
        }
    }

    /// Scale a learning rate inversely with stress.
    pub fn throttle_learning_rate(&self, base_lr: f32) -> f32 {
        base_lr * (1.0 - self.stress * 0.8).clamp(0.0001, 1.0)
    }
}

impl Default for ThermodynamicGovernor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HomeostaticController {
    pub active_learning_rate: f32,
    pub governor: ThermodynamicGovernor,
}

impl HomeostaticController {
    pub fn new() -> Self {
        Self {
            active_learning_rate: 0.000500,
            governor: ThermodynamicGovernor::new(),
        }
    }

    /// Minimizes prediction errors by adjusting system learning rates based on
    /// internal entropy metrics and hardware thermodynamics.
    pub fn execute_active_inference_loop(&mut self, empirical_loss: f32) -> f32 {
        let prediction_error = empirical_loss - TARGET_STABILITY;

        if prediction_error > 0.0 {
            self.active_learning_rate += prediction_error * 0.001;
        } else {
            self.active_learning_rate -= prediction_error.abs() * 0.0005;
        }

        self.active_learning_rate = self.active_learning_rate.clamp(0.0001, MAXIMUM_META_RATE);
        // Thermodynamic throttle: reduce learning when the machine is stressed.
        self.active_learning_rate = self
            .governor
            .throttle_learning_rate(self.active_learning_rate);
        self.active_learning_rate
    }

    /// Feed hardware stress into the thermodynamic governor.
    pub fn update_thermodynamics(&mut self, state: &ThermalState) {
        self.governor.update(state);
    }
}

/// Logical relation between two nodes in the causal knowledge graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRelation {
    Enforces,
    Protects,
    Enables,
    Causes,
    Violates,
    DependsOn,
}

impl CausalRelation {
    /// Human-readable relation name.
    pub fn as_str(&self) -> &'static str {
        match self {
            CausalRelation::Enforces => "ENFORCES",
            CausalRelation::Protects => "PROTECTS",
            CausalRelation::Enables => "ENABLES",
            CausalRelation::Causes => "CAUSES",
            CausalRelation::Violates => "VIOLATES",
            CausalRelation::DependsOn => "DEPENDS_ON",
        }
    }

    /// Inverse used during backward failure traversal.
    pub fn inverse(&self) -> &'static str {
        match self {
            CausalRelation::Enforces => "VIOLATED BY",
            CausalRelation::Protects => "ENDANGERED BY",
            CausalRelation::Enables => "BLOCKED BY",
            CausalRelation::Causes => "RESULTED FROM",
            CausalRelation::Violates => "VIOLATES",
            CausalRelation::DependsOn => "REQUIRED BY",
        }
    }
}

/// A neuro-symbolic causal knowledge graph.
///
/// Stores explicit logical relations between system assets, skills, and
/// constraints.  When a plan or tool step fails, the graph can be traversed
/// backwards from the failed node to produce a plain-text causal explanation
/// of the broken primitive.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CausalGraph {
    nodes: Vec<String>,
    edges: Vec<(usize, CausalRelation, usize)>,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, name: &str) -> usize {
        if let Some(idx) = self.nodes.iter().position(|n| n == name) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(name.to_string());
        idx
    }

    /// Add a directed causal relation between two named nodes.
    pub fn add_relation(&mut self, src: &str, rel: CausalRelation, dst: &str) {
        let s = self.add_node(src);
        let d = self.add_node(dst);
        self.edges.push((s, rel, d));
    }

    /// Backward traversal from a failed node, returning a causal explanation.
    pub fn explain_failure(&self, failed_node: &str) -> Option<String> {
        let start = self.nodes.iter().position(|n| n == failed_node)?;

        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![(start, false)];
        let mut antecedents: Vec<String> = Vec::new();

        while let Some((idx, is_root)) = stack.pop() {
            if !visited.insert(idx) {
                continue;
            }

            for (s, rel, d) in &self.edges {
                if *d == idx && !visited.contains(s) {
                    let explanation = if is_root {
                        format!(
                            "[{}] {} [{}]",
                            self.nodes[*s],
                            rel.inverse(),
                            self.nodes[idx]
                        )
                    } else {
                        format!(
                            "[{}] {} [{}]",
                            self.nodes[*s],
                            rel.as_str(),
                            self.nodes[idx]
                        )
                    };
                    antecedents.push(explanation);
                    stack.push((*s, false));
                }
            }
        }

        if antecedents.is_empty() {
            Some(format!("Failure at isolated node [{failed_node}]"))
        } else {
            Some(format!(
                "Causal chain for failure at [{}]:\n{}",
                failed_node,
                antecedents.join("\n")
            ))
        }
    }

    /// Return the node name of the most likely broken primitive for a given
    /// failure string by looking for the first node name that appears in it.
    pub fn find_broken_primitive(&self, failure: &str) -> Option<&str> {
        self.nodes
            .iter()
            .find(|n| failure.to_lowercase().contains(&n.to_lowercase()))
            .map(std::string::String::as_str)
    }

    /// Node names, in `find_broken_primitive` match priority order.
    /// `find_broken_primitive` returns the first node whose name appears in
    /// the failure text, so specific technical terms must precede generic
    /// ones ("vram" before "memory", "identity" before "agent").
    pub const PRIMITIVE_PRIORITY: &'static [&'static str] = &[
        "vram",
        "tokenizer",
        "socket",
        "ledger",
        "identity",
        "slicks",
        "menubar",
        "dashboard",
        "tts",
        "ears",
        "microphone",
        "speech",
        "screen",
        "index",
        "blocklist",
        "permission",
        "launch",
        "agent",
        "disk",
        "mesh",
        "peer",
        "council",
        "approval",
        "firewall",
        "cache",
        "data_dir",
        "recall",
        "model",
        "engine",
        "memory",
    ];

    /// All node names registered in the graph.
    pub fn node_names(&self) -> &[String] {
        &self.nodes
    }

    /// Built-in causal map for the Bad Apple architecture.
    pub fn bad_apple_default() -> Self {
        let mut g = Self::new();
        for name in Self::PRIMITIVE_PRIORITY {
            g.add_node(name);
        }
        g.add_relation(
            "apple_intelligence_bridge",
            CausalRelation::Enables,
            "llm_oracle",
        );
        g.add_relation("llm_oracle", CausalRelation::Enables, "planning_head");
        g.add_relation("llm_oracle", CausalRelation::Enables, "tool_synthesis");
        g.add_relation("candle_brain", CausalRelation::Enables, "local_conscience");
        g.add_relation(
            "local_conscience",
            CausalRelation::Protects,
            "goal_integrity",
        );
        g.add_relation(
            "conscience_oracle",
            CausalRelation::Enforces,
            "safety_policy",
        );
        g.add_relation("safety_policy", CausalRelation::Protects, "system_asset");
        g.add_relation("wild_workspace", CausalRelation::Protects, "system_asset");
        g.add_relation(
            "transfer_evaluator",
            CausalRelation::Enables,
            "skill_memory",
        );
        g.add_relation("skill_memory", CausalRelation::Enables, "planning_head");
        g.add_relation(
            "strategy_library",
            CausalRelation::DependsOn,
            "skill_memory",
        );

        // Runtime anatomy: real primitives, named by the vocabulary that
        // appears in actual failure text. Two edge shapes per subsystem:
        // (prerequisite, Enables, dependent) so a dependent's failure names
        // its cause, and (dependent, DependsOn, prerequisite) so a
        // prerequisite's failure names what else it takes down.
        let runtime: &[(&str, CausalRelation, &str)] = &[
            // Daemon socket lifecycle — a stale engine generation races the
            // fd and shadows the socket for the new process.
            ("launch_plist", CausalRelation::Enables, "socket"),
            ("stale_engine_process", CausalRelation::Violates, "socket"),
            ("engine", CausalRelation::DependsOn, "socket"),
            ("gatekeeper", CausalRelation::DependsOn, "socket"),
            ("cli", CausalRelation::DependsOn, "socket"),
            // Model residency — VRAM admission is the classic failure;
            // stale engines compressing gigabytes were the observed cause.
            ("vram", CausalRelation::Enables, "model"),
            ("memory_pressure", CausalRelation::Violates, "vram"),
            (
                "stale_engine_process",
                CausalRelation::Causes,
                "memory_pressure",
            ),
            ("model", CausalRelation::Enables, "generation"),
            ("model", CausalRelation::Enables, "engine"),
            ("socket", CausalRelation::Enables, "engine"),
            ("memory_pressure", CausalRelation::Violates, "memory"),
            ("memory", CausalRelation::Enables, "vram"),
            ("model", CausalRelation::DependsOn, "vram"),
            // Retrieval stack — recall needs both a current index and a
            // helper binary that matches the running scorer.
            ("model_bundle", CausalRelation::Enables, "tokenizer"),
            ("tokenizer", CausalRelation::Enables, "index"),
            ("tokenizer", CausalRelation::Enables, "engine"),
            ("index", CausalRelation::DependsOn, "tokenizer"),
            ("engine", CausalRelation::DependsOn, "tokenizer"),
            ("index", CausalRelation::Enables, "recall"),
            ("helper_binary", CausalRelation::Enables, "recall"),
            ("recall", CausalRelation::DependsOn, "index"),
            ("hf_cache", CausalRelation::Enables, "model"),
            ("app_bundle", CausalRelation::Protects, "helper_binary"),
            // Identity and signing.
            ("secure_enclave", CausalRelation::Enables, "identity"),
            ("identity_agent", CausalRelation::Enables, "identity"),
            ("launch_plist", CausalRelation::Enables, "identity_agent"),
            ("identity", CausalRelation::Enables, "slicks"),
            ("slicks", CausalRelation::DependsOn, "identity"),
            ("peer_tls", CausalRelation::DependsOn, "identity"),
            ("launchd_service", CausalRelation::Enables, "launch"),
            ("launch_plist", CausalRelation::Enables, "agent"),
            // Persistence and audit.
            ("platform_installer", CausalRelation::Enables, "data_dir"),
            ("data_dir", CausalRelation::Enables, "ledger"),
            ("disk", CausalRelation::Enables, "ledger"),
            ("state_growth", CausalRelation::Violates, "disk"),
            ("ledger", CausalRelation::Enables, "ify"),
            ("ledger", CausalRelation::Enables, "audit"),
            ("ify", CausalRelation::DependsOn, "ledger"),
            ("audit", CausalRelation::DependsOn, "ledger"),
            ("bge_embedding_model", CausalRelation::Enables, "cache"),
            // Senses — permissions are owned by the entitled app process.
            ("microphone", CausalRelation::Enables, "ears"),
            ("speech", CausalRelation::Enables, "ears"),
            ("menubar", CausalRelation::Enables, "ears"),
            ("ears", CausalRelation::DependsOn, "menubar"),
            ("ears", CausalRelation::Enables, "ambient_hearing"),
            ("screen", CausalRelation::Enables, "ocular"),
            ("vision_model", CausalRelation::Enables, "ocular"),
            ("user_consent", CausalRelation::Enables, "permission"),
            ("permission", CausalRelation::Enables, "microphone"),
            ("permission", CausalRelation::Enables, "speech"),
            ("permission", CausalRelation::Enables, "screen"),
            // User agents ride on their launch plists.
            ("launch_plist", CausalRelation::Enables, "menubar"),
            ("launch_plist", CausalRelation::Enables, "tts"),
            ("launch_plist", CausalRelation::Enables, "dashboard"),
            ("launch_plist", CausalRelation::Enables, "ify"),
            // Governance.
            ("council", CausalRelation::Enables, "approval"),
            ("engine", CausalRelation::Enables, "council"),
            ("firewall", CausalRelation::Protects, "secrets"),
            ("firewall", CausalRelation::DependsOn, "blocklist"),
            ("blocklist", CausalRelation::Enables, "firewall"),
            // Mesh.
            ("identity", CausalRelation::Enables, "peer_tls"),
            ("peer_tls", CausalRelation::Enables, "peer"),
            ("peer", CausalRelation::Enables, "mesh"),
            ("mesh", CausalRelation::Enables, "delegated_inference"),
        ];
        for (src, rel, dst) in runtime {
            g.add_relation(src, rel.clone(), dst);
        }
        g
    }
}

#[cfg(test)]
mod causal_tests {
    use super::CausalGraph;

    #[test]
    fn explains_vram_admission_denied() {
        let g = CausalGraph::bad_apple_default();
        let primitive = g
            .find_broken_primitive("VRAM admission denied: Not enough free memory")
            .expect("vram should match");
        assert_eq!(primitive, "vram");
        let chain = g.explain_failure(primitive).unwrap();
        assert!(chain.contains("memory_pressure"));
        assert!(chain.contains("stale_engine_process"));
    }

    #[test]
    fn explains_socket_contention() {
        let g = CausalGraph::bad_apple_default();
        let primitive = g
            .find_broken_primitive("error binding socket: address already in use")
            .expect("socket should match");
        assert_eq!(primitive, "socket");
        let chain = g.explain_failure(primitive).unwrap();
        assert!(chain.contains("stale_engine_process"));
        assert!(chain.contains("DEPENDS_ON"));
    }

    #[test]
    fn explains_identity_agent_failure() {
        let g = CausalGraph::bad_apple_default();
        let primitive = g
            .find_broken_primitive("identity_agent_not_loaded")
            .expect("identity should match before agent");
        assert_eq!(primitive, "identity");
        let chain = g.explain_failure(primitive).unwrap();
        assert!(chain.contains("launch_plist"));
    }

    #[test]
    fn specific_terms_win_over_generic() {
        let g = CausalGraph::bad_apple_default();
        let primitive = g
            .find_broken_primitive("speech recognition permission denied")
            .expect("speech should match before permission");
        assert_eq!(primitive, "speech");
    }

    #[test]
    fn unknown_failures_report_isolation() {
        let g = CausalGraph::bad_apple_default();
        assert!(g.find_broken_primitive("quantum flux inverter").is_none());
    }
}
