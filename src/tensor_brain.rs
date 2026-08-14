use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, loss as nn_loss, ops as nn_ops, AdamW, Init, LayerNorm, Linear, Module,
    Optimizer, VarBuilder, VarMap,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::{whitespace::WhitespaceSplit, PreTokenizerWrapper};
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::TokenizerImpl;
use tokenizers::{AddedToken, Tokenizer};

/// Dimensionality of the Transformer hidden / brain state.
pub const BRAIN_DIM: usize = 576;

/// Legacy MLP layer-dimension list kept for API compatibility.
///
/// The real brain is now a small Transformer encoder; this function is no
/// longer used for architecture but is still exported so existing call sites
/// compile.
pub fn layer_dims() -> Vec<(usize, usize)> {
    let mut dims = Vec::with_capacity(99);
    for _ in 0..10 {
        dims.push((2048, 1024));
    }
    for _ in 0..30 {
        dims.push((1024, 1024));
    }
    for _ in 0..30 {
        dims.push((512, 512));
    }
    for _ in 0..16 {
        dims.push((256, 128));
    }
    for _ in 0..5 {
        dims.push((128, 128));
    }
    dims.push((128, 100));
    for _ in 0..7 {
        dims.push((100, 100));
    }
    dims
}

/// Default value used by serde for the skipped `candle_brain` field.
pub fn no_candle_brain() -> Option<CandleBrain> {
    None
}

/// Attention head dropout (DropHead).
///
/// Drops entire attention heads during training with probability `p`, while
/// scaling the surviving heads by `1 / (1 - p)` so that the expected value is
/// preserved.  This forces the 12 heads to stay robust rather than silently
/// co-adapting.
#[derive(Clone, Debug)]
struct DropHead {
    p: f64,
    scale: f64,
}

impl DropHead {
    fn new(p: f64) -> Self {
        let p = p.clamp(0.0, 0.999_999);
        Self {
            p,
            scale: 1.0 / (1.0 - p),
        }
    }

    /// Apply the same head mask to Q, K, and V tensors of shape
    /// `(batch, heads, seq, head_dim)`.
    fn apply_to_heads(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        training: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if !training || self.p == 0.0 {
            return Ok((q.clone(), k.clone(), v.clone()));
        }

        let num_heads = q.dim(1)?;
        let mut mask = vec![0.0f32; num_heads];
        let mut rng = rand::thread_rng();
        for m in mask.iter_mut() {
            if !rng.gen_bool(self.p) {
                *m = self.scale as f32;
            }
        }

        let mask_t = Tensor::new(mask, q.device())?.reshape((1, num_heads, 1, 1))?;
        let q_out = q.broadcast_mul(&mask_t)?;
        let k_out = k.broadcast_mul(&mask_t)?;
        let v_out = v.broadcast_mul(&mask_t)?;
        Ok((q_out, k_out, v_out))
    }
}

/// A small multi-head self-attention Transformer block.
struct TransformerBlock {
    dim: usize,
    ln1: LayerNorm,
    ln2: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ffn1: Linear,
    ffn2: Linear,
    num_heads: usize,
    head_dim: usize,
    /// Head-dropout regularizer; applied during training only.
    drop_head: DropHead,
}

impl TransformerBlock {
    fn new(dim: usize, num_heads: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!(dim % num_heads, 0, "dim must be divisible by num_heads");
        let head_dim = dim / num_heads;
        Ok(Self {
            dim,
            ln1: layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            q_proj: linear(dim, dim, vb.pp("q_proj"))?,
            k_proj: linear(dim, dim, vb.pp("k_proj"))?,
            v_proj: linear(dim, dim, vb.pp("v_proj"))?,
            o_proj: linear(dim, dim, vb.pp("o_proj"))?,
            ffn1: linear(dim, ffn_dim, vb.pp("ffn1"))?,
            ffn2: linear(ffn_dim, dim, vb.pp("ffn2"))?,
            num_heads,
            head_dim,
            drop_head: DropHead::new(0.1),
        })
    }

    /// Orthogonality penalty on the flattened Q and K head weight matrices.
    ///
    /// Computes `sum_{i != j} <W_i, W_j>^2` across heads, encouraging each of
    /// the 12 heads to occupy a distinct subspace of the 512-D geometry.
    fn orthogonality_penalty(&self, proj: &Linear) -> Result<Tensor> {
        // Weight shape is (dim, dim).  Reshape to (num_heads, head_dim, dim),
        // then flatten each head to a vector.
        let w = proj.weight();
        let w_3d = w.reshape((self.num_heads, self.head_dim, self.dim))?;
        let w_2d = w_3d.reshape((self.num_heads, self.head_dim * self.dim))?;

        // Gram matrix G = W @ W^T, then mask the diagonal and penalize the
        // off-diagonal energy.  Clamp G before squaring to prevent FPU
        // overflow when Q/K weights have grown large.
        let g = w_2d
            .matmul(&w_2d.transpose(D::Minus2, D::Minus1)?)?
            .clamp(-10.0, 10.0)?;
        let eye = Tensor::eye(self.num_heads, DType::F32, w.device())?;
        let ones = Tensor::ones((self.num_heads, self.num_heads), DType::F32, w.device())?;
        let off_diag_mask = ones.sub(&eye)?;
        let off_diag = g.mul(&off_diag_mask)?;
        off_diag.sqr()?.mean_all()?.reshape(())
    }

    /// Total orthogonality regularization for this block (Q + K).
    fn block_orthogonality_penalty(&self) -> Result<Tensor> {
        let q = self.orthogonality_penalty(&self.q_proj)?;
        let k = self.orthogonality_penalty(&self.k_proj)?;
        q.add(&k)?.reshape(())
    }

    /// Forward on a tensor of shape `(batch, seq, dim)`.
    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        let (_b, _s, d) = x.dims3()?;

        // Self-attention sub-layer.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((1, _s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((1, _s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((1, _s, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Drop entire attention heads during training.
        let (q, k, v) = self.drop_head.apply_to_heads(&q, &k, &v, training)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scale = (self.head_dim as f32).sqrt();
        let scale_t = Tensor::new(&[scale], q.device())?.reshape((1, 1, 1, 1))?;
        let scores = q.matmul(&k_t)?.broadcast_div(&scale_t)?;
        let attn = nn_ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((1, _s, d))?;
        let out = self.o_proj.forward(&out)?;

        let x = x.add(&out)?;
        let x = self.ln1.forward(&x)?;

        // Feed-forward sub-layer.
        let ffn = self.ffn1.forward(&x)?.gelu()?;
        let ffn = self.ffn2.forward(&ffn)?;
        let x = x.add(&ffn)?;
        self.ln2.forward(&x)
    }
}

/// Layer-wise learning-rate decay (LLRD) group.
///
/// Holds one AdamW optimizer for a subset of parameters and a multiplier
/// `gamma` that is applied on top of the global base learning rate set by the
/// Conscience Oracle.  Earlier blocks / embeddings get `gamma = 0.75`; later
/// blocks and downstream heads get `gamma = 1.0`.
struct OptimizerGroup {
    optimizer: AdamW,
    gamma: f64,
}

/// A real Candle tensor backend for the Firefly brain.
///
/// Replaces the legacy MLP with a small Transformer encoder that processes
/// the 2048-dim grounded embedding as a 32-token sequence (64-dim tokens).
pub struct CandleBrain {
    name: String,
    device: Device,
    varmap: VarMap,
    seq_len: usize,
    token_dim: usize,
    dim: usize,
    token_embedding: Linear,
    pos_embed: Tensor,
    transformer_blocks: Vec<TransformerBlock>,
    output_head: Linear,
    /// 100-class conscience classifier head on top of the brain state.
    conscience_head: Linear,
    /// 100-class goal / intention generator head on top of the brain state.
    goal_head: Linear,
    num_classes: usize,
    /// 2048-dim next-embedding language head on top of the last layer.
    language_head: Linear,
    /// Intrinsic curiosity reward used to modulate the goal-head loss.
    /// 0.0 = no bonus; 1.0 = maximum exploration bonus.
    curiosity_reward: f64,
    /// Dual-process governor switch.  When false, the Transformer blocks are
    /// bypassed and training becomes inference-only, keeping routine ticks fast.
    system2_active: bool,
    /// Learning-rate multiplier when System 2 is active.
    system2_lr_multiplier: f64,
    /// Unscaled learning rate set by the caller.
    base_lr: f64,
    /// Hard floor on the learning rate; raised when the conscience head is
    /// stuck at uniform cross-entropy.
    lr_floor: f64,
    /// Layer-wise optimizers for LLRD.
    optimizers: Vec<OptimizerGroup>,
    /// Coefficient for the Q/K orthogonality regularization term.
    ortho_lambda: f64,
    /// Current global cycle count (used for the 30-cycle post-boot LR dampener).
    cycle: u64,
    /// Maximum learning rate during the post-boot warmup.
    warmup_max_lr: f64,
    /// Number of cycles the LR dampener stays active after boot.
    warmup_cycles: u64,
    /// Maximum global gradient L2 norm.  Anything larger is rescaled to this
    /// ceiling to stop loss explosions during backpropagation.
    max_grad_norm: f64,
    /// Recent conscience losses used to detect a perfectly flat plateau.
    loss_history: VecDeque<f64>,
    /// Number of consecutive losses that must be flat to trigger a symmetry break.
    stagnation_window: usize,
    /// Two losses are considered "identical" if they differ by less than this.
    stagnation_epsilon: f64,
    /// Scale of the uniform noise injected into head weights during a symmetry break.
    symmetry_noise_scale: f64,
    /// Maximum learning-rate floor applied when a symmetry break fires.
    symmetry_lr_bump: f64,
}

impl fmt::Debug for CandleBrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CandleBrain")
            .field("name", &self.name)
            .field("device", &self.device)
            .field("dim", &self.dim)
            .field("transformer_blocks", &self.transformer_blocks.len())
            .field("num_classes", &self.num_classes)
            .field("learning_rate", &self.base_lr)
            .finish()
    }
}

/// Returns true for token / positional embeddings and Transformer Blocks 0 and 1.
///
/// These receive a reduced learning rate under LLRD so foundational grammar
/// and vocabulary stay locked, while later blocks and task heads can adapt
/// faster.
fn is_early_var(name: &str) -> bool {
    name.starts_with("token_embedding")
        || name == "pos_embed"
        || name.starts_with("0/")
        || name.starts_with("1/")
}

/// Xavier/Glorot uniform init clamped to a `max_bound` recovery band.
///
/// The standard fan-in/out bound is `sqrt(6 / (fan_in + fan_out))`; for the
/// small downstream task heads this is capped at `max_bound` (0.01) so the
/// initial predictions stay in the finite recovery band while still giving the
/// head enough asymmetry to break out of the uniform cross-entropy symmetry.
fn xavier_head(
    in_dim: usize,
    out_dim: usize,
    max_bound: f64,
    vb: VarBuilder<'_>,
) -> Result<Linear> {
    let xavier_bound = (6.0 / (in_dim + out_dim) as f64).sqrt();
    let bound = xavier_bound.min(max_bound);
    let ws = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        Init::Uniform {
            lo: -bound,
            up: bound,
        },
    )?;
    let bs = vb.get_with_hints(
        out_dim,
        "bias",
        Init::Uniform {
            lo: -bound,
            up: bound,
        },
    )?;
    Ok(Linear::new(ws, Some(bs)))
}

impl CandleBrain {
    /// Build a small Transformer encoder using `candle_nn` and `VarMap`.
    ///
    /// Tries Metal, falls back to CPU.  All internal weights are `F32`.
    pub fn new(name: &str, num_classes: usize, _layer_dims: &[(usize, usize)]) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        if matches!(device, Device::Metal(_)) {
            tracing::info!("[CandleBrain '{}' initialized on Apple Metal GPU]", name);
        } else {
            tracing::info!("[CandleBrain '{}' initialized on CPU]", name);
        }
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let seq_len = 32usize;
        let token_dim = 64usize;
        let dim = BRAIN_DIM;
        let num_heads = 12usize;
        let ffn_dim = dim * 4;
        let num_blocks = 4usize;

        // Embed 64-dim token vectors into the 576-D transformer trunk.
        let token_embedding = linear(token_dim, dim, vb.pp("token_embedding"))?;

        // Learnable positional embeddings.
        let pos_embed = vb.get((seq_len, dim), "pos_embed")?;

        // Four 12-head Transformer blocks.
        let mut transformer_blocks = Vec::new();
        for i in 0..num_blocks {
            let block = TransformerBlock::new(dim, num_heads, ffn_dim, vb.pp(i.to_string()))?;
            transformer_blocks.push(block);
        }

        // Final projection to the dim-dim conscience/classifier space.
        let output_head = linear(dim, dim, vb.pp("output_head"))?;

        // Conscience, goal, and language heads use a scaled Xavier/Glorot init
        // capped at the 0.01 stability band so the initial predictions are
        // finite and start with enough asymmetry to escape uniform logits.
        let max_head_bound = 0.01;
        let conscience_head =
            xavier_head(dim, num_classes, max_head_bound, vb.pp("conscience_head"))?;
        let goal_head = xavier_head(dim, num_classes, max_head_bound, vb.pp("goal_head"))?;
        let language_head = xavier_head(dim, 2048, max_head_bound, vb.pp("language_head"))?;

        // Layer-wise learning-rate decay: embeddings + Blocks 0/1 -> 0.75x,
        // Blocks 2/3 + all task heads -> 1.0x.
        let mut early = Vec::new();
        let mut late = Vec::new();
        {
            let data = varmap.data().lock().unwrap();
            for (name, var) in data.iter() {
                if is_early_var(name) {
                    early.push(var.clone());
                } else {
                    late.push(var.clone());
                }
            }
        }

        let optimizers = vec![
            OptimizerGroup {
                optimizer: AdamW::new_lr(early, 0.001 * 0.75)?,
                gamma: 0.75,
            },
            OptimizerGroup {
                optimizer: AdamW::new_lr(late, 0.001)?,
                gamma: 1.0,
            },
        ];

        Ok(Self {
            name: name.to_string(),
            device,
            varmap,
            seq_len,
            token_dim,
            dim,
            token_embedding,
            pos_embed,
            transformer_blocks,
            output_head,
            conscience_head,
            goal_head,
            num_classes,
            language_head,
            curiosity_reward: 0.0,
            system2_active: true,
            system2_lr_multiplier: 1.0,
            base_lr: 0.001,
            lr_floor: 0.0,
            optimizers,
            ortho_lambda: 1e-4,
            cycle: 0,
            // The post-boot cap used to be 0.00005, which is below the
            // HomeostaticController's minimum active rate and starved the
            // conscience head during the critical first 30 cycles. Allow it
            // to reach the controller's maximum (0.001) so symmetry can break.
            warmup_max_lr: 0.001,
            warmup_cycles: 30,
            max_grad_norm: 5.0,
            loss_history: VecDeque::with_capacity(8),
            stagnation_window: 4,
            stagnation_epsilon: 0.001,
            symmetry_noise_scale: 0.005,
            symmetry_lr_bump: 0.005,
        })
    }

    /// Set the intrinsic curiosity reward in [0.0, 1.0].
    /// A higher value increases exploratory pressure on the goal head.
    pub fn set_curiosity_reward(&mut self, reward: f64) {
        self.curiosity_reward = reward.clamp(0.0, 1.0);
    }

    /// Current curiosity reward used by the loss modulation.
    pub fn curiosity_reward(&self) -> f64 {
        self.curiosity_reward
    }

    /// Engage or disengage System 2 deep attention.
    ///
    /// When System 2 is active, the full 4-block Transformer runs, the
    /// effective context window is expanded, and the learning rate is scaled
    /// up.  When inactive, the Transformer blocks are bypassed and training
    /// becomes inference-only.
    pub fn set_system2_active(&mut self, active: bool) {
        self.system2_active = active;
        self.system2_lr_multiplier = if active { 1.5 } else { 1.0 };
        self.set_learning_rate(self.base_lr);
    }

    pub fn system2_active(&self) -> bool {
        self.system2_active
    }

    /// Reshape a flat 2048 input into `(1, seq_len, token_dim)`.
    fn prepare_input(&self, input: &[f64]) -> Result<Tensor> {
        let input_f32: Vec<f32> = input.iter().map(|v| *v as f32).collect();
        let t = Tensor::new(input_f32.as_slice(), &self.device)?;
        t.reshape((1, self.seq_len, self.token_dim))
    }

    /// Shared trunk: input -> dim-dim brain state.
    ///
    /// In System 2, the full 4-block Transformer processes the token sequence.
    /// In System 1, the Transformer blocks are bypassed for a fast embedding
    /// path while still projecting to the same `dim`-dimensional state space.
    fn brain_state(&self, input: &[f64], training: bool) -> Result<Tensor> {
        let mut x = self.prepare_input(input)?;
        x = self.token_embedding.forward(&x)?;
        x = x.add(&self.pos_embed.reshape((1, self.seq_len, self.dim))?)?;

        if self.system2_active {
            for block in &self.transformer_blocks {
                x = block.forward(&x, training)?;
            }
        }

        // Mean-pool over the sequence and project to `dim` output units.
        let pooled = x.mean(1)?.reshape((1, self.dim))?;
        let out = self.output_head.forward(&pooled)?;
        // Hard vector clamp keeps the 576-D trunk in a finite band, preventing
        // FPU microcode traps from cascading downstream heads and world model.
        out.clamp(-10.0, 10.0)
    }

    /// Run a full forward pass through the Transformer and project to dim-D.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.dim];
        self.forward_into(input, &mut out)?;
        Ok(out)
    }

    /// In-place `forward` that writes the 576-D brain state into a pre-allocated
    /// slice.  This avoids the `Vec<f64>` allocation for callers that already
    /// own a reusable buffer (e.g. a `MemoryArena` or a pre-sized `Vec`).
    pub fn forward_into(&self, input: &[f64], out: &mut [f64]) -> Result<()> {
        assert_eq!(out.len(), self.dim, "forward output must be dim-D");
        let out_tensor = self.brain_state(input, false)?;
        let values = out_tensor.squeeze(0)?.to_vec1::<f32>()?;
        for (i, &v) in values.iter().enumerate() {
            out[i] = v as f64;
        }
        Ok(())
    }

    /// Run a full forward pass and return num_classes conscience logits.
    pub fn classify(&self, input: &[f64]) -> Result<Vec<f64>> {
        let state = self.brain_state(input, false)?;
        let logits = self
            .conscience_head
            .forward(&state)?
            .squeeze(0)?
            .clamp(-10.0, 10.0)?;
        let values = logits.to_vec1::<f32>()?;
        Ok(values.into_iter().map(|v| v as f64).collect())
    }

    /// Run a full forward pass and return the top class with its probability.
    pub fn classify_top(&self, input: &[f64]) -> Result<(usize, f64)> {
        let logits = self.classify(input)?;
        let logits_t = Tensor::new(
            logits
                .iter()
                .map(|v| *v as f32)
                .collect::<Vec<_>>()
                .as_slice(),
            &self.device,
        )?;
        let probs = nn_ops::softmax(&logits_t, D::Minus1)?.to_vec1::<f32>()?;
        let (idx, &p) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok((idx, p as f64))
    }

    /// Sum the Q/K orthogonality penalties across all 4 Transformer blocks.
    fn orthogonality_penalty(&self) -> Result<Tensor> {
        let mut total = Tensor::zeros((), DType::F32, &self.device)?;
        for block in &self.transformer_blocks {
            total = total.add(&block.block_orthogonality_penalty()?)?;
        }
        Ok(total)
    }

    /// Shared gradient-backward step across the LLRD optimizer groups.
    ///
    /// Computes gradients, rescales them if their global L2 norm exceeds
    /// `max_grad_norm`, and then applies the AdamW optimizer.  Hard gradient
    /// clipping prevents the loss explosions seen during extended training.
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let mut grads = loss.backward()?;

        // Global L2 norm across all parameter gradients.
        let mut norm_sq = 0.0;
        let data = self.varmap.data().lock().unwrap();
        for var in data.values() {
            if let Some(g) = grads.get(var.as_tensor()) {
                let n = g.sqr()?.sum_all()?.to_vec0::<f32>()? as f64;
                norm_sq += n;
            }
        }
        drop(data);

        let norm = norm_sq.sqrt();
        let scale = if norm > self.max_grad_norm {
            self.max_grad_norm / norm
        } else {
            1.0
        };

        if scale < 1.0 {
            let data = self.varmap.data().lock().unwrap();
            for var in data.values() {
                if let Some(g) = grads.get(var.as_tensor()) {
                    let scaled = g.affine(scale, 0.0)?;
                    let _ = grads.insert(var.as_tensor(), scaled);
                }
            }
        }

        for g in &mut self.optimizers {
            let lr = self.base_lr * g.gamma * self.system2_lr_multiplier;
            g.optimizer.set_learning_rate(lr);
            g.optimizer.step(&grads)?;
        }
        Ok(())
    }

    /// Forward + cross-entropy classification loss + AdamW optimizer step.
    ///
    /// In System 1 the loss is computed for telemetry but no weights are
    /// updated, keeping routine ticks fast.
    pub fn train_step(&mut self, input: &[f64], target_idx: usize) -> Result<f64> {
        let target = Tensor::new(&[target_idx as u32], &self.device)?;

        let state = self.brain_state(input, true)?;
        let logits = self.conscience_head.forward(&state)?.clamp(-10.0, 10.0)?;

        let loss = nn_loss::cross_entropy(&logits, &target)?;

        // Add Q/K orthogonality regularization to keep the 12 heads separated.
        let ortho = self.orthogonality_penalty()?.reshape(())?;
        let lambda = Tensor::new(self.ortho_lambda as f32, &self.device)?;
        let total_loss = loss.add(&ortho.broadcast_mul(&lambda)?)?;

        if self.system2_active {
            self.backward_step(&total_loss)?;
        }
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar)
    }

    /// Train the goal / intention head to predict the teacher-assigned goal class from the brain state.
    ///
    /// The cross-entropy loss is modulated by the intrinsic curiosity reward:
    /// a higher curiosity reward scales the loss down, encouraging the goal
    /// head to prefer exploratory, non-greedy trajectories.
    ///
    /// In System 1 the loss is computed but no weights are updated.
    pub fn train_goal_step(&mut self, brain_state: &[f64], target_idx: usize) -> Result<f64> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let logits = self.goal_head.forward(&state_batch)?.clamp(-10.0, 10.0)?;
        let target = Tensor::new(&[target_idx as u32], &self.device)?;
        let loss = nn_loss::cross_entropy(&logits, &target)?;

        // Curiosity modulation: 0.0 -> scale 1.0, 1.0 -> scale 0.5.
        let scale = 1.0 - 0.5 * self.curiosity_reward as f32;
        let scale_t = Tensor::new(&[scale], &self.device)?;
        let scaled = loss.broadcast_mul(&scale_t)?;

        if self.system2_active {
            self.backward_step(&scaled)?;
        }
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar)
    }

    /// Predict the local goal class and its confidence from the brain state.
    pub fn predict_goal(&self, brain_state: &[f64]) -> Result<(usize, f64)> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let logits = self
            .goal_head
            .forward(&state_batch)?
            .squeeze(0)?
            .clamp(-10.0, 10.0)?;
        let logits_vec = logits.to_vec1::<f32>()?;
        let logits_t = Tensor::new(logits_vec.as_slice(), &self.device)?;
        let probs = nn_ops::softmax(&logits_t, D::Minus1)?.to_vec1::<f32>()?;
        let (idx, &p) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok((idx, p as f64))
    }

    /// Train the language head to predict the next input embedding from the brain state.
    ///
    /// In System 1 the language loss is computed for telemetry only.
    pub fn train_language_step(&mut self, brain_state: &[f64], next_target: &[f64]) -> Result<f64> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let target_f32: Vec<f32> = next_target.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let target_t = Tensor::new(target_f32.as_slice(), &self.device)?;

        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let pred = self
            .language_head
            .forward(&state_batch)?
            .squeeze(0)?
            .clamp(-10.0, 10.0)?;
        let target_t = target_t.clamp(-10.0, 10.0)?;

        let loss = pred.sub(&target_t)?.sqr()?.mean_all()?;
        if self.system2_active {
            self.backward_step(&loss)?;
        }
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar.sqrt())
    }

    /// Update the current global training cycle.
    pub fn set_cycle(&mut self, cycle: u64) {
        self.cycle = cycle;
    }

    /// Record a conscience loss and break symmetry if the loss has been
    /// perfectly flat for too many cycles.  The detector is triggered when the
    /// last `stagnation_window` values are all within `stagnation_epsilon` of
    /// each other and the loss is still high enough to be stuck in the uniform
    /// cross-entropy basin.  When triggered we:
    ///
    ///   1. raise the learning-rate floor to `symmetry_lr_bump`,
    ///   2. add small uniform noise to all task-head and output-head weights,
    ///   3. increase the noise scale so repeated stalls get stronger nudges.
    pub fn note_conscience_loss(&mut self, conscience_loss: f64) {
        self.loss_history.push_back(conscience_loss);
        if self.loss_history.len() > self.stagnation_window {
            self.loss_history.pop_front();
        }

        let uniform = (self.num_classes as f64).ln();
        let high = conscience_loss > uniform - 0.5;

        let flat = self.loss_history.len() >= self.stagnation_window
            && self
                .loss_history
                .iter()
                .all(|&v| (v - conscience_loss).abs() < self.stagnation_epsilon);

        if flat && high {
            tracing::info!(
                "conscience head stagnant at {:.6} for {} cycles; breaking symmetry",
                conscience_loss,
                self.stagnation_window
            );
            self.lr_floor = (self.lr_floor * 1.5).max(self.symmetry_lr_bump).min(0.01);
            self.symmetry_noise_scale = (self.symmetry_noise_scale * 1.2).min(0.05);
            if let Err(e) = self.apply_symmetry_break() {
                tracing::warn!("symmetry break failed: {:?}", e);
            }
        } else if conscience_loss < 3.5 && self.lr_floor > 0.0 {
            // Loss is clearly dropping; decay the hard floor so the
            // HomeostaticController can fine-tune once we leave the basin.
            self.lr_floor = (self.lr_floor * 0.5).max(0.0001);
        }
    }

    /// Inject small uniform noise into the task heads and the trunk output
    /// projection.  This is a direct parameter-space perturbation; it does not
    /// go through the optimizer, so it works even when AdamW momentum has
    /// stalled.
    fn apply_symmetry_break(&mut self) -> Result<()> {
        let data = self.varmap.data().lock().unwrap();
        let mut rng = StdRng::from_entropy();
        let prefixes = [
            "conscience_head",
            "goal_head",
            "language_head",
            "output_head",
        ];

        for (name, var) in data.iter() {
            if !prefixes.iter().any(|p| name.starts_with(p)) {
                continue;
            }

            let shape = var.as_tensor().shape().clone();
            let flat = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            let noise = self.symmetry_noise_scale as f32;
            let mut noisy = Vec::with_capacity(flat.len());
            for &v in &flat {
                let delta = (rng.gen::<f32>() * 2.0 - 1.0) * noise;
                noisy.push(v + delta);
            }

            let new_tensor = Tensor::new(noisy.as_slice(), &self.device)?.reshape(shape)?;
            var.set(&new_tensor)?;
        }

        Ok(())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        let max_lr = if self.cycle > 0 && self.cycle <= self.warmup_cycles {
            self.warmup_max_lr
        } else {
            f64::MAX
        };
        // The floor must not exceed the ceiling, otherwise `clamp` panics.
        let max_lr = max_lr.max(self.lr_floor);
        let lr = lr.clamp(self.lr_floor, max_lr);
        self.base_lr = lr;
        for g in &mut self.optimizers {
            let scaled = lr * g.gamma * self.system2_lr_multiplier;
            g.optimizer.set_learning_rate(scaled);
        }
    }

    /// Set the Q/K orthogonality regularization coefficient.
    pub fn set_ortho_lambda(&mut self, lambda: f64) {
        self.ortho_lambda = lambda.clamp(0.0, 1.0);
    }

    pub fn learning_rate(&self) -> f64 {
        self.base_lr
    }

    pub fn sample_weight_00(&self) -> f64 {
        self.token_embedding
            .weight()
            .get(0)
            .ok()
            .and_then(|row| row.get(0).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .unwrap_or(0.0) as f64
    }

    pub fn save_weights<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.varmap.save(path)
    }

    /// Return a lightweight, read-only view of every variable in the VarMap as
    /// a `name -> tensor` map.  The `Tensor` values are Arc-cloned (shallow),
    /// not deep-copied; the background `StateSaveWorker` performs the deep copy
    /// on its own thread once the main loop has resumed.  The main loop only
    /// mutates weights inside the cognitive tick, so the worker has the whole
    /// inter-tick interval to copy and flush safely.
    pub fn snapshot_weights(&self) -> Result<HashMap<String, Tensor>> {
        let data = self.varmap.data().lock().unwrap();
        let mut snapshot = HashMap::with_capacity(data.len());
        for (name, var) in data.iter() {
            // Shallow clone: increments the Arc to the underlying storage, so
            // the worker can deep-copy on its own thread without blocking the
            // main cognitive clock.
            snapshot.insert(name.clone(), var.as_tensor().clone());
        }
        Ok(snapshot)
    }

    pub fn load_weights<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.varmap.load(path)
    }
}

// =========================================================================
// 🗣️ BPE TOKENIZER-BASED GROUNDED EMBEDDING
// =========================================================================

/// Global BPE tokenizer loaded once from `tokenizer.json`.
///
/// The tokenizer maps raw text to real vocabulary token IDs.  A deterministic
/// 64-dimensional embedding table is generated from a fixed seed so the 32-token
/// sequence fed into the Transformer is semantically stable across runs.
static BPE_TOKENIZER: OnceLock<BpeTokenizer> = OnceLock::new();

struct BpeTokenizer {
    tokenizer: Tokenizer,
    /// vocab_size x 64 deterministic token embeddings.
    embeddings: Vec<Vec<f64>>,
}

impl BpeTokenizer {
    /// Load `tokenizer.json` if it exists and is valid; otherwise train a
    /// lightweight BPE model on the curriculum and save it to the same path.
    fn load_or_train<P: AsRef<Path>>(path: P) -> Self {
        match Tokenizer::from_file(&path) {
            Ok(tokenizer) => return Self::from_tokenizer(tokenizer),
            Err(e) => eprintln!(
                "⚠️ Could not load {:?}: {}. Training a fresh BPE tokenizer from curriculum...",
                path.as_ref(),
                e
            ),
        }

        let mut trainer = BpeTrainerBuilder::new()
            .vocab_size(128)
            .min_frequency(3)
            .show_progress(false)
            .limit_alphabet(256)
            .special_tokens(vec![
                AddedToken::from(String::from("<pad>"), true),
                AddedToken::from(String::from("<unk>"), true),
                AddedToken::from(String::from("<s>"), true),
                AddedToken::from(String::from("</s>"), true),
            ])
            .build();

        let mut tokenizer: TokenizerImpl<
            BPE,
            NormalizerWrapper,
            PreTokenizerWrapper,
            PostProcessorWrapper,
            DecoderWrapper,
        > = TokenizerImpl::new(BPE::default());
        tokenizer.with_pre_tokenizer(WhitespaceSplit);

        let files = vec!["curriculum/curriculum.txt".to_string()];
        tokenizer
            .train_from_files(&mut trainer, files)
            .expect("Failed to train BPE tokenizer from curriculum files");

        tokenizer
            .save(&path, false)
            .expect("Failed to save tokenizer.json");

        // Type-erase the concrete BPE implementation into the standard `Tokenizer` wrapper.
        Self::from_tokenizer(tokenizer.into())
    }

    fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let embeddings = Self::build_embeddings(vocab_size);
        Self {
            tokenizer,
            embeddings,
        }
    }

    fn build_embeddings(vocab_size: usize) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(0xF1A_F1E_F1A_F1E);
        let mut embeddings = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut row = Vec::with_capacity(64);
            for _ in 0..64 {
                row.push(rng.gen::<f64>() * 2.0 - 1.0);
            }
            embeddings.push(row);
        }
        embeddings
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let encoding = match self.tokenizer.encode(text, false) {
            Ok(enc) => enc,
            Err(e) => {
                eprintln!("⚠️ BPE tokenize failed ({}); returning empty.", e);
                return Vec::new();
            }
        };
        encoding.get_ids().to_vec()
    }

    fn encode_to_2048_into(&self, text: &str, spatial_axes: &[f64; 4], out: &mut [f64]) {
        const SEQ_LEN: usize = 32;
        const TOKEN_DIM: usize = 64;
        assert_eq!(out.len(), SEQ_LEN * TOKEN_DIM);

        out.fill(0.0);

        let encoding = self.tokenizer.encode(text, false).unwrap_or_else(|e| {
            eprintln!("⚠️ BPE encode failed ({}); using empty encoding.", e);
            // Return a zero-length encoding from an empty string.
            self.tokenizer
                .encode("", false)
                .expect("tokenizer must encode empty string")
        });

        let ids = encoding.get_ids();

        for (row, &id) in ids.iter().take(SEQ_LEN).enumerate() {
            let idx = (id as usize).min(self.embeddings.len().saturating_sub(1));
            let emb = &self.embeddings[idx];
            for (col, &val) in emb.iter().take(TOKEN_DIM).enumerate() {
                out[row * TOKEN_DIM + col] = val;
            }
        }

        // Blend the 4-D physical/sensory anchors into the first channel of
        // rows 0, 8, 16, and 24 (indices 0, 512, 1024, 1536).
        for (i, &axis) in spatial_axes.iter().enumerate().take(4) {
            let slot = i * 8 * TOKEN_DIM;
            out[slot] += axis * 5.0;
        }

        let magnitude: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for v in out.iter_mut() {
                *v /= magnitude;
            }
        }
    }

    fn encode_to_2048(&self, text: &str, spatial_axes: &[f64; 4]) -> Vec<f64> {
        let mut out = vec![0.0; 32 * 64];
        self.encode_to_2048_into(text, spatial_axes, &mut out);
        out
    }
}

/// Convert raw text into the 2048-dimensional grounded embedding expected by
/// `CandleBrain`.  The text is tokenized with the local BPE tokenizer, mapped
/// into a 32 x 64 token matrix, and then fused with the four physical anchors.
pub fn text_to_grounded_embedding(text: &str, spatial_axes: &[f64; 4]) -> Vec<f64> {
    let mut out = vec![0.0; 2048];
    text_to_grounded_embedding_into(text, spatial_axes, &mut out);
    out
}

/// In-place variant of `text_to_grounded_embedding`.  The caller must supply a
/// 2048-element `&mut [f64]` that is overwritten.  This eliminates the per-call
/// `Vec` allocation when a pre-allocated or arena-backed buffer is reused.
pub fn text_to_grounded_embedding_into(text: &str, spatial_axes: &[f64; 4], out: &mut [f64]) {
    assert_eq!(out.len(), 2048, "grounded embedding must be 2048-D");
    let bpe = BPE_TOKENIZER.get_or_init(|| BpeTokenizer::load_or_train("tokenizer.json"));
    bpe.encode_to_2048_into(text, spatial_axes, out);
}

/// Tokenize a string into BPE token IDs, returning the raw token sequence.
/// Used by higher-level modules (e.g., HDC script profiling).
pub fn tokenize_text(text: &str) -> Vec<u32> {
    let bpe = BPE_TOKENIZER.get_or_init(|| BpeTokenizer::load_or_train("tokenizer.json"));
    bpe.tokenize_to_ids(text)
}

// =========================================================================
// Continuous-Time Neural ODE / Liquid State Machine foundation
// =========================================================================

/// Liquid State Machine (LSM) with leaky continuous-time reservoir dynamics.
///
/// State evolves as `dx/dt = -leak * x + tanh(W_in * u + W_rec * x + b)`.
/// We use fixed-size, pre-allocated vectors and in-place Euler updates to keep
/// per-step allocations near zero.
pub struct LiquidStateMachine {
    state: Vec<f64>,
    bias: Vec<f64>,
    input_weights: Vec<Vec<f64>>,
    recurrent_weights: Vec<Vec<f64>>,
    reservoir_size: usize,
    input_dim: usize,
    leak_rate: f64,
    dt: f64,
    rng: StdRng,
}

impl LiquidStateMachine {
    /// Build a new LSM. Weights are sampled uniformly in `[-scale, scale]`.
    pub fn new(input_dim: usize, reservoir_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_in = 0.3;
        let scale_rec = 0.1;

        let mut input_weights = vec![Vec::with_capacity(input_dim); reservoir_size];
        for row in input_weights.iter_mut() {
            for _ in 0..input_dim {
                row.push(rng.gen_range(-scale_in..scale_in));
            }
        }

        let mut recurrent_weights = vec![Vec::with_capacity(reservoir_size); reservoir_size];
        for row in recurrent_weights.iter_mut() {
            for _ in 0..reservoir_size {
                row.push(rng.gen_range(-scale_rec..scale_rec));
            }
        }

        let mut bias = Vec::with_capacity(reservoir_size);
        for _ in 0..reservoir_size {
            bias.push(rng.gen_range(-0.1..0.1));
        }

        Self {
            state: vec![0.0; reservoir_size],
            bias,
            input_weights,
            recurrent_weights,
            reservoir_size,
            input_dim,
            leak_rate: 0.9,
            dt: 0.01,
            rng,
        }
    }

    /// One continuous-time Euler step of duration `dt`. Uses a pre-allocated
    /// scratch buffer to avoid inner allocations.
    pub fn step(&mut self, input: &[f64], scratch: &mut [f64]) {
        assert_eq!(input.len(), self.input_dim);
        assert_eq!(scratch.len(), self.reservoir_size);

        for (i, scratch_i) in scratch.iter_mut().enumerate().take(self.reservoir_size) {
            let mut drive = self.bias[i];
            for (j, &x) in input.iter().enumerate() {
                drive += self.input_weights[i][j] * x;
            }
            for (j, &s) in self.state.iter().enumerate() {
                drive += self.recurrent_weights[i][j] * s;
            }
            *scratch_i = drive.tanh();
        }

        for (i, s) in self.state.iter_mut().enumerate().take(self.reservoir_size) {
            *s = *s + self.dt * (-self.leak_rate * *s + scratch[i]);
        }
    }

    /// Run `n` integration steps for a given input.
    pub fn integrate(&mut self, input: &[f64], n: usize, scratch: &mut [f64]) {
        for _ in 0..n {
            self.step(input, scratch);
        }
    }

    /// Read a copy of the current state. The caller can use it for readout.
    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

/// Continuous-time brain that combines the LSM reservoir with a linear
/// readout layer. Pre-allocates all scratch memory.
pub struct ContinuousBrain {
    lsm: LiquidStateMachine,
    readout: Vec<Vec<f64>>,
    output_dim: usize,
    scratch: Vec<f64>,
    output: Vec<f64>,
}

impl ContinuousBrain {
    pub fn new(input_dim: usize, reservoir_size: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        let mut readout = vec![Vec::with_capacity(reservoir_size); output_dim];
        for row in readout.iter_mut() {
            for _ in 0..reservoir_size {
                row.push(rng.gen_range(-0.1..0.1));
            }
        }

        Self {
            lsm: LiquidStateMachine::new(input_dim, reservoir_size, seed),
            readout,
            output_dim,
            scratch: vec![0.0; reservoir_size],
            output: vec![0.0; output_dim],
        }
    }

    /// Advance the continuous state by `dt` per step for `n_steps` and produce
    /// an `output_dim` readout. All memory is pre-allocated.
    pub fn forward(&mut self, input: &[f64], n_steps: usize) -> &[f64] {
        self.lsm.integrate(input, n_steps, &mut self.scratch);
        let state = self.lsm.state();
        for (i, out) in self.output.iter_mut().enumerate() {
            *out = 0.0;
            for (j, &row_j) in self.readout[i].iter().enumerate() {
                *out += row_j * state[j];
            }
        }
        &self.output
    }

    /// Hebbian-like plasticity update on the readout: if `target` is provided,
    /// nudge readout weights by `eta * (target - out) * state`.
    pub fn adapt(&mut self, target: &[f64], eta: f64) {
        let state = self.lsm.state();
        for (i, row) in self.readout.iter_mut().enumerate() {
            let error = target.get(i).copied().unwrap_or(0.0) - self.output[i];
            for (j, w) in row.iter_mut().enumerate() {
                *w += eta * error * state[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_embedding_is_2048_and_normalised() {
        let embedding = text_to_grounded_embedding("hello world", &[0.5, 0.4, 0.3, 0.2]);
        assert_eq!(embedding.len(), 2048);
        let magnitude = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-9);
    }
}
