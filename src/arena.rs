//! Pre-allocated scratchpad workspace guaranteeing zero heap allocations during decode.
//!
//! All activation buffers (`x`, `xb`, `q`, `k`, `v`, `hb`, `hb2`, `logits`) and the
//! quantized activation scratch buffer (`xq_bytes`) are sized once from `ModelConfig`
//! (supporting up to `max_batch` tokens during batched prefill, and slicing `0..dim` for
//! `batch = 1` decode).

use crate::quant::MAX_QUANT_BLOCK_BYTES;

/// Default maximum tokens processed in a single prefill batch.
/// 32 tokens keeps scratch buffers inside L2 cache (~1-2 MB for 0.5B-1.5B models)
/// while boosting arithmetic intensity by 32x over token-by-token prefill.
pub const DEFAULT_PREFILL_BATCH: usize = 32;

/// Reusable activation buffers for both batched prefill and single-token decode.
pub struct InferenceScratch {
    pub max_batch: usize,
    pub dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,

    /// Residual stream `[max_batch, dim]`
    pub x: Vec<f32>,
    /// Normalized / intermediate `[max_batch, dim]`
    pub xb: Vec<f32>,
    /// Projection output `[max_batch, dim]`
    pub xb2: Vec<f32>,
    /// Query projections `[max_batch, q_dim]`
    pub q: Vec<f32>,
    /// Key projections `[max_batch, kv_dim]`
    pub k: Vec<f32>,
    /// Value projections `[max_batch, kv_dim]`
    pub v: Vec<f32>,
    /// Attention output before `wo` `[max_batch, q_dim]`
    pub att_out: Vec<f32>,
    /// SwiGLU gate buffer `[max_batch, hidden_dim]`
    pub hb: Vec<f32>,
    /// SwiGLU up buffer `[max_batch, hidden_dim]`
    pub hb2: Vec<f32>,
    /// Output logits `[vocab_size]` (only computed for the last token of a batch)
    pub logits: Vec<f32>,
    /// Scratch buffer for on-the-fly quantized input activations (`Q8_0` / `Q8_K`),
    /// sized `[max_batch, max_in_blocks * MAX_QUANT_BLOCK_BYTES]` so integer SIMD
    /// matmuls never allocate per token or per layer.
    pub xq_stride: usize,
    pub xq_bytes: Vec<u8>,
}

impl InferenceScratch {
    pub fn new(
        max_batch: usize,
        dim: usize,
        q_dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
        vocab_size: usize,
    ) -> Self {
        let max_batch = max_batch.max(1);
        let max_in_dim = dim.max(q_dim).max(hidden_dim);
        // Round up to blocks of 32 (and 256 for k-quants) with generous byte stride.
        let padded_blocks = (max_in_dim + 255) / 32;
        let xq_stride = padded_blocks * MAX_QUANT_BLOCK_BYTES;

        Self {
            max_batch,
            dim,
            q_dim,
            kv_dim,
            hidden_dim,
            vocab_size,
            x: vec![0.0; max_batch * dim],
            xb: vec![0.0; max_batch * dim],
            xb2: vec![0.0; max_batch * dim],
            q: vec![0.0; max_batch * q_dim],
            k: vec![0.0; max_batch * kv_dim],
            v: vec![0.0; max_batch * kv_dim],
            att_out: vec![0.0; max_batch * q_dim],
            hb: vec![0.0; max_batch * hidden_dim],
            hb2: vec![0.0; max_batch * hidden_dim],
            logits: vec![0.0; vocab_size],
            xq_stride,
            xq_bytes: vec![0u8; max_batch * xq_stride],
        }
    }

    /// Total heap memory footprint of the scratchpad in bytes.
    pub fn byte_size(&self) -> usize {
        let floats = self.x.len()
            + self.xb.len()
            + self.xb2.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att_out.len()
            + self.hb.len()
            + self.hb2.len()
            + self.logits.len();
        floats * std::mem::size_of::<f32>() + self.xq_bytes.len()
    }
}
