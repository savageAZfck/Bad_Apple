use std::fmt;
use std::path::Path;
use std::sync::OnceLock;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, layer_norm, ops as nn_ops, loss as nn_loss, AdamW, Linear, LayerNorm, Module, Optimizer, VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tokenizers::{AddedToken, Tokenizer};
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::{whitespace::WhitespaceSplit, PreTokenizerWrapper};
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::TokenizerImpl;

/// Dimensionality of the Transformer hidden / brain state.
pub const BRAIN_DIM: usize = 256;

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
        })
    }

    /// Forward on a tensor of shape `(batch, seq, dim)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _s, d) = x.dims3()?;

        // Self-attention sub-layer.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((1, _s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((1, _s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((1, _s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scale = (self.head_dim as f32).sqrt();
        let scale_t = Tensor::new(&[scale], &q.device())?.reshape((1, 1, 1, 1))?;
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
    optimizer: AdamW,
}

impl fmt::Debug for CandleBrain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CandleBrain")
            .field("name", &self.name)
            .field("device", &self.device)
            .field("dim", &self.dim)
            .field("transformer_blocks", &self.transformer_blocks.len())
            .field("num_classes", &self.num_classes)
            .field("learning_rate", &self.optimizer.learning_rate())
            .finish()
    }
}

impl CandleBrain {
    /// Build a small Transformer encoder using `candle_nn` and `VarMap`.
    ///
    /// Tries Metal, falls back to CPU.  All internal weights are `F32`.
    pub fn new(name: &str, num_classes: usize, _layer_dims: &[(usize, usize)]) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let seq_len = 32usize;
        let token_dim = 64usize;
        let dim = BRAIN_DIM;
        let num_heads = 8usize;
        let ffn_dim = 1024usize;
        let num_blocks = 4usize;

        // Embed 64-dim token vectors into the model dimension.
        let token_embedding = linear(token_dim, dim, vb.pp("token_embedding"))?;

        // Learnable positional embeddings.
        let pos_embed = vb.get((seq_len, dim), "pos_embed")?;

        // Four larger Transformer blocks.
        let mut transformer_blocks = Vec::new();
        for i in 0..num_blocks {
            let block = TransformerBlock::new(dim, num_heads, ffn_dim, vb.pp(i.to_string()))?;
            transformer_blocks.push(block);
        }

        // Final projection to the dim-dim conscience/classifier space.
        let output_head = linear(dim, dim, vb.pp("output_head"))?;

        // Conscience classifier: brain state -> num_classes logits.
        let conscience_head = linear(dim, num_classes, vb.pp("conscience_head"))?;

        // Goal / intention generator: brain state -> num_classes goal logits.
        let goal_head = linear(dim, num_classes, vb.pp("goal_head"))?;

        // Language head: pooled brain state -> 2048-dim next-input embedding.
        let language_head = linear(dim, 2048, vb.pp("language_head"))?;

        let optimizer = AdamW::new_lr(varmap.all_vars(), 0.001)?;

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
            optimizer,
        })
    }

    /// Reshape a flat 2048 input into `(1, seq_len, token_dim)`.
    fn prepare_input(&self, input: &[f64]) -> Result<Tensor> {
        let input_f32: Vec<f32> = input.iter().map(|v| *v as f32).collect();
        let t = Tensor::new(input_f32.as_slice(), &self.device)?;
        t.reshape((1, self.seq_len, self.token_dim))
    }

    /// Shared trunk: input -> dim-dim brain state.
    fn brain_state(&self, input: &[f64]) -> Result<Tensor> {
        let mut x = self.prepare_input(input)?;
        x = self.token_embedding.forward(&x)?;
        x = x.add(&self.pos_embed.reshape((1, self.seq_len, self.dim))?)?;

        for block in &self.transformer_blocks {
            x = block.forward(&x)?;
        }

        // Mean-pool over the sequence and project to `dim` output units.
        let pooled = x.mean(1)?.reshape((1, self.dim))?;
        self.output_head.forward(&pooled)
    }

    /// Run a full forward pass through the Transformer and project to dim-D.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let out = self.brain_state(input)?;
        let values = out.squeeze(0)?.to_vec1::<f32>()?;
        Ok(values.into_iter().map(|v| v as f64).collect())
    }

    /// Run a full forward pass and return num_classes conscience logits.
    pub fn classify(&self, input: &[f64]) -> Result<Vec<f64>> {
        let state = self.brain_state(input)?;
        let logits = self.conscience_head.forward(&state)?.squeeze(0)?;
        let values = logits.to_vec1::<f32>()?;
        Ok(values.into_iter().map(|v| v as f64).collect())
    }

    /// Run a full forward pass and return the top class with its probability.
    pub fn classify_top(&self, input: &[f64]) -> Result<(usize, f64)> {
        let logits = self.classify(input)?;
        let logits_t = Tensor::new(logits.iter().map(|v| *v as f32).collect::<Vec<_>>().as_slice(), &self.device)?;
        let probs = nn_ops::softmax(&logits_t, D::Minus1)?.to_vec1::<f32>()?;
        let (idx, &p) = probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        Ok((idx, p as f64))
    }

    /// Forward + cross-entropy classification loss + AdamW optimizer step.
    pub fn train_step(&mut self, input: &[f64], target_idx: usize) -> Result<f64> {
        let target = Tensor::new(&[target_idx as u32], &self.device)?;

        let state = self.brain_state(input)?;
        let logits = self.conscience_head.forward(&state)?;

        let loss = nn_loss::cross_entropy(&logits, &target)?;
        self.optimizer.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar)
    }

    /// Train the goal / intention head to predict the teacher-assigned goal class from the brain state.
    pub fn train_goal_step(&mut self, brain_state: &[f64], target_idx: usize) -> Result<f64> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let logits = self.goal_head.forward(&state_batch)?;
        let target = Tensor::new(&[target_idx as u32], &self.device)?;
        let loss = nn_loss::cross_entropy(&logits, &target)?;
        self.optimizer.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar)
    }

    /// Predict the local goal class and its confidence from the brain state.
    pub fn predict_goal(&self, brain_state: &[f64]) -> Result<(usize, f64)> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let logits = self.goal_head.forward(&state_batch)?.squeeze(0)?;
        let logits_vec = logits.to_vec1::<f32>()?;
        let logits_t = Tensor::new(logits_vec.as_slice(), &self.device)?;
        let probs = nn_ops::softmax(&logits_t, D::Minus1)?.to_vec1::<f32>()?;
        let (idx, &p) = probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        Ok((idx, p as f64))
    }

    /// Train the language head to predict the next input embedding from the brain state.
    pub fn train_language_step(&mut self, brain_state: &[f64], next_target: &[f64]) -> Result<f64> {
        let state_f32: Vec<f32> = brain_state.iter().map(|v| *v as f32).collect();
        let target_f32: Vec<f32> = next_target.iter().map(|v| *v as f32).collect();
        let state_t = Tensor::new(state_f32.as_slice(), &self.device)?;
        let target_t = Tensor::new(target_f32.as_slice(), &self.device)?;

        let state_batch = state_t.reshape((1, state_f32.len()))?;
        let pred = self.language_head.forward(&state_batch)?.squeeze(0)?;

        let loss = pred.sub(&target_t)?.sqr()?.mean_all()?;
        self.optimizer.backward_step(&loss)?;
        let loss_scalar = loss.to_vec0::<f32>()? as f64;
        Ok(loss_scalar.sqrt())
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
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
            Err(e) => eprintln!("⚠️ Could not load {:?}: {}. Training a fresh BPE tokenizer from curriculum...", path.as_ref(), e),
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

        let mut tokenizer: TokenizerImpl<BPE, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper> =
            TokenizerImpl::new(BPE::default());
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
        Self { tokenizer, embeddings }
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

    fn encode_to_2048(&self, text: &str, spatial_axes: &[f64; 4]) -> Vec<f64> {
        const SEQ_LEN: usize = 32;
        const TOKEN_DIM: usize = 64;

        let encoding = self.tokenizer.encode(text, false)
            .unwrap_or_else(|e| {
                eprintln!("⚠️ BPE encode failed ({}); using empty encoding.", e);
                // Return a zero-length encoding from an empty string.
                self.tokenizer.encode("", false).expect("tokenizer must encode empty string")
            });

        let ids = encoding.get_ids();
        let mut vec = vec![0.0; SEQ_LEN * TOKEN_DIM];

        for (row, &id) in ids.iter().take(SEQ_LEN).enumerate() {
            let idx = (id as usize).min(self.embeddings.len().saturating_sub(1));
            let emb = &self.embeddings[idx];
            for (col, &val) in emb.iter().take(TOKEN_DIM).enumerate() {
                vec[row * TOKEN_DIM + col] = val;
            }
        }

        // Blend the 4-D physical/sensory anchors into the first channel of
        // rows 0, 8, 16, and 24 (indices 0, 512, 1024, 1536).
        for i in 0..4 {
            let slot = i * 8 * TOKEN_DIM;
            vec[slot] += spatial_axes[i] * 5.0;
        }

        let magnitude: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for v in vec.iter_mut() { *v /= magnitude; }
        }

        vec
    }
}

/// Convert raw text into the 2048-dimensional grounded embedding expected by
/// `CandleBrain`.  The text is tokenized with the local BPE tokenizer, mapped
/// into a 32 x 64 token matrix, and then fused with the four physical anchors.
pub fn text_to_grounded_embedding(text: &str, spatial_axes: &[f64; 4]) -> Vec<f64> {
    let bpe = BPE_TOKENIZER.get_or_init(|| BpeTokenizer::load_or_train("tokenizer.json"));
    bpe.encode_to_2048(text, spatial_axes)
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
