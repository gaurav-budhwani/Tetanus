//! Transformer forward pass (Llama / Qwen2 / Qwen3) over zero-copy GGUF weights.
//!
//! Key architectural highlights:
//!   - Unified decoder supporting GQA, SwiGLU, RMSNorm, optional QKV biases (Qwen 2/2.5),
//!     optional per-head Q/K RMSNorm (Qwen 3), tied embeddings, and Interleaved vs.
//!     Split-Half RoPE (automatically selected from `general.architecture`).
//!   - Batched prefill (`DEFAULT_PREFILL_BATCH` tokens) with tiled online-softmax attention:
//!     processes attention in `TILE_KV = 64` blocks per head using stack buffers, keeping
//!     attention workspace `O(1)` in sequence length instead of `O(seq_len^2)`.
//!   - Zero heap allocations during `decode_token`: all scratch state lives in `InferenceScratch`.

use crate::arena::{InferenceScratch, DEFAULT_PREFILL_BATCH};
use crate::gguf::{GgmlType, GgufError, GgufFile};
use crate::kernels::{
    apply_rope, matmul_batch, rmsnorm, rmsnorm_inplace, silu_mul_inplace, RopeStyle,
    SimdBackend,
};
use crate::kv_cache::{KvCache, KvQuant};
use crate::pool::{shard_range, ThreadPool};
use crate::quant::dequantize_row;

const TILE_KV: usize = 64;
const MAX_HEAD_DIM: usize = 256;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_style: RopeStyle,
    pub tied_embeddings: bool,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let arch = gguf
            .arch()
            .ok_or_else(|| GgufError::Format("missing general.architecture".into()))?
            .to_owned();

        let dim = gguf
            .arch_u32("embedding_length")
            .ok_or_else(|| GgufError::Format(format!("missing {arch}.embedding_length")))?
            as usize;
        let hidden_dim = gguf
            .arch_u32("feed_forward_length")
            .ok_or_else(|| GgufError::Format(format!("missing {arch}.feed_forward_length")))?
            as usize;
        let n_layers = gguf
            .arch_u32("block_count")
            .ok_or_else(|| GgufError::Format(format!("missing {arch}.block_count")))?
            as usize;
        let n_heads = gguf
            .arch_u32("attention.head_count")
            .ok_or_else(|| GgufError::Format(format!("missing {arch}.attention.head_count")))?
            as usize;
        let n_kv_heads = gguf
            .arch_u32("attention.head_count_kv")
            .unwrap_or(n_heads as u32) as usize;
        let head_dim = gguf
            .arch_u32("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let rope_dims = gguf
            .arch_u32("rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let max_seq_len = gguf
            .arch_u32("context_length")
            .unwrap_or(4096) as usize;
        let rms_norm_eps = gguf
            .arch_f32("attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_theta = gguf.arch_f32("rope.freq_base").unwrap_or(10_000.0);

        // `llama` GGUFs permute Q/K weights for adjacent-pair RoPE (`NORM`);
        // `qwen2` / `qwen3` keep split-half (`NEOX`) order.
        let rope_style = if arch == "llama" {
            RopeStyle::Interleaved
        } else {
            RopeStyle::SplitHalf
        };

        let emb_info = gguf.tensor("token_embd.weight")?;
        let vocab_size = emb_info.dims.get(1).copied().ok_or_else(|| {
            GgufError::Format("token_embd.weight must be 2D".into())
        })?;
        let tied_embeddings = !gguf.tensors.contains_key("output.weight");

        if head_dim > MAX_HEAD_DIM {
            return Err(GgufError::Format(format!(
                "head_dim {head_dim} exceeds MAX_HEAD_DIM ({MAX_HEAD_DIM})"
            )));
        }

        Ok(Self {
            arch,
            vocab_size,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dims,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            rope_style,
            tied_embeddings,
        })
    }

    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// Zero-copy pointer into a 2D weight matrix inside `GgufFile`.
#[derive(Debug, Clone, Copy)]
pub struct WeightView {
    pub file_offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    /// `out_features` (`dims[1]` in GGUF)
    pub rows: usize,
    /// `in_features` (`dims[0]` in GGUF)
    pub cols: usize,
}

impl WeightView {
    fn from_gguf(gguf: &GgufFile, name: &str, expected_rows: usize, expected_cols: usize) -> Result<Self, GgufError> {
        let info = gguf.tensor(name)?;
        // Validate actual slice accessibility (also rejects unsupported GGML types upfront).
        let _ = gguf.tensor_bytes(info)?;
        let cols = *info.dims.first().unwrap_or(&0);
        let rows = *info.dims.get(1).unwrap_or(&0);
        if rows != expected_rows || cols != expected_cols {
            return Err(GgufError::Format(format!(
                "tensor '{name}' shape mismatch: got [{cols}, {rows}], expected [{expected_cols}, {expected_rows}]"
            )));
        }
        Ok(Self {
            file_offset: info.file_offset,
            byte_size: info.byte_size,
            dtype: info.dtype,
            rows,
            cols,
        })
    }

    #[inline]
    fn bytes<'a>(&self, file_bytes: &'a [u8]) -> &'a [u8] {
        &file_bytes[self.file_offset..self.file_offset + self.byte_size]
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: WeightView,
    wk: WeightView,
    wv: WeightView,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wo: WeightView,
    ffn_norm: Vec<f32>,
    w_gate: WeightView,
    w_up: WeightView,
    w_down: WeightView,
}

pub struct Transformer {
    pub config: ModelConfig,
    pub backend: SimdBackend,
    gguf: GgufFile,
    token_embd: WeightView,
    output_norm: Vec<f32>,
    output_head: WeightView,
    layers: Vec<LayerWeights>,
    inv_freqs: Vec<f32>,
}

impl Transformer {
    pub fn from_gguf(gguf: GgufFile) -> Result<Self, GgufError> {
        let cfg = ModelConfig::from_gguf(&gguf)?;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();

        let token_embd = WeightView::from_gguf(&gguf, "token_embd.weight", cfg.vocab_size, cfg.dim)?;
        let output_norm = load_f32_vec(&gguf, "output_norm.weight", cfg.dim)?;
        let output_head = if cfg.tied_embeddings {
            token_embd
        } else {
            WeightView::from_gguf(&gguf, "output.weight", cfg.vocab_size, cfg.dim)?
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let attn_norm = load_f32_vec(&gguf, &format!("blk.{i}.attn_norm.weight"), cfg.dim)?;
            let wq = WeightView::from_gguf(&gguf, &format!("blk.{i}.attn_q.weight"), q_dim, cfg.dim)?;
            let wk = WeightView::from_gguf(&gguf, &format!("blk.{i}.attn_k.weight"), kv_dim, cfg.dim)?;
            let wv = WeightView::from_gguf(&gguf, &format!("blk.{i}.attn_v.weight"), kv_dim, cfg.dim)?;

            let bq = load_opt_f32_vec(&gguf, &format!("blk.{i}.attn_q.bias"), q_dim)?;
            let bk = load_opt_f32_vec(&gguf, &format!("blk.{i}.attn_k.bias"), kv_dim)?;
            let bv = load_opt_f32_vec(&gguf, &format!("blk.{i}.attn_v.bias"), kv_dim)?;

            let q_norm = load_opt_f32_vec(&gguf, &format!("blk.{i}.attn_q_norm.weight"), cfg.head_dim)?;
            let k_norm = load_opt_f32_vec(&gguf, &format!("blk.{i}.attn_k_norm.weight"), cfg.head_dim)?;

            let wo = WeightView::from_gguf(&gguf, &format!("blk.{i}.attn_output.weight"), cfg.dim, q_dim)?;
            let ffn_norm = load_f32_vec(&gguf, &format!("blk.{i}.ffn_norm.weight"), cfg.dim)?;
            let w_gate = WeightView::from_gguf(&gguf, &format!("blk.{i}.ffn_gate.weight"), cfg.hidden_dim, cfg.dim)?;
            let w_up = WeightView::from_gguf(&gguf, &format!("blk.{i}.ffn_up.weight"), cfg.hidden_dim, cfg.dim)?;
            let w_down = WeightView::from_gguf(&gguf, &format!("blk.{i}.ffn_down.weight"), cfg.dim, cfg.hidden_dim)?;

            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                bq,
                bk,
                bv,
                q_norm,
                k_norm,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        let half_rope = cfg.rope_dims / 2;
        let inv_freqs: Vec<f32> = (0..half_rope)
            .map(|i| 1.0 / cfg.rope_theta.powf((2 * i) as f32 / (cfg.rope_dims as f32)))
            .collect();

        Ok(Self {
            config: cfg,
            backend: SimdBackend::detect(),
            gguf,
            token_embd,
            output_norm,
            output_head,
            layers,
            inv_freqs,
        })
    }

    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// Approximate bytes read per generated token (all layer weights + LM head + active KV cache).
    /// Useful for reporting realized DRAM bandwidth vs. theoretical hardware limits.
    pub fn bytes_per_token_decode(&self, seq_len: usize, kv_quant: KvQuant) -> usize {
        let mut weight_bytes = self.output_head.byte_size;
        for l in &self.layers {
            weight_bytes += l.wq.byte_size
                + l.wk.byte_size
                + l.wv.byte_size
                + l.wo.byte_size
                + l.w_gate.byte_size
                + l.w_up.byte_size
                + l.w_down.byte_size;
        }
        let slot = match kv_quant {
            KvQuant::Q8_0 if self.config.head_dim.is_multiple_of(32) => {
                (self.config.head_dim / 32) * 34
            }
            _ => self.config.head_dim * 2,
        };
        let kv_bytes = 2 * self.config.n_layers * self.config.n_kv_heads * seq_len * slot;
        weight_bytes + kv_bytes
    }

    pub fn new_scratch(&self, max_batch: usize) -> InferenceScratch {
        InferenceScratch::new(
            max_batch,
            self.config.dim,
            self.config.q_dim(),
            self.config.kv_dim(),
            self.config.hidden_dim,
            self.config.vocab_size,
        )
    }

    pub fn new_kv_cache(&self, max_seq_len: usize, quant: KvQuant) -> KvCache {
        let cap = max_seq_len.min(self.config.max_seq_len);
        KvCache::new(
            self.config.n_layers,
            self.config.n_kv_heads,
            cap,
            self.config.head_dim,
            quant,
        )
    }

    /// Prefills `tokens` starting at `start_pos` in chunks of `scratch.max_batch` (default 32),
    /// returning `&scratch.logits` for the final token in `tokens`.
    pub fn prefill<'a>(
        &self,
        tokens: &[u32],
        start_pos: usize,
        pool: &ThreadPool,
        kv: &mut KvCache,
        scratch: &'a mut InferenceScratch,
    ) -> &'a mut [f32] {
        assert!(!tokens.is_empty(), "prefill requires at least one token");
        let chunk_cap = scratch.max_batch.max(1);
        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + chunk_cap).min(tokens.len());
            let is_last_chunk = end == tokens.len();
            self.forward_batch(
                &tokens[offset..end],
                start_pos + offset,
                is_last_chunk,
                pool,
                kv,
                scratch,
            );
            offset = end;
        }
        &mut scratch.logits
    }

    /// Runs a single decode step at `pos` with **zero heap allocations**.
    #[inline]
    pub fn decode_token<'a>(
        &self,
        token: u32,
        pos: usize,
        pool: &ThreadPool,
        kv: &mut KvCache,
        scratch: &'a mut InferenceScratch,
    ) -> &'a mut [f32] {
        self.forward_batch(&[token], pos, true, pool, kv, scratch);
        &mut scratch.logits
    }

    /// Core unified forward pass for `n_tokens` (`1` for decode, up to `max_batch` for prefill).
    fn forward_batch(
        &self,
        tokens: &[u32],
        start_pos: usize,
        compute_logits: bool,
        pool: &ThreadPool,
        kv: &mut KvCache,
        s: &mut InferenceScratch,
    ) {
        let n_tokens = tokens.len();
        assert!(n_tokens > 0 && n_tokens <= s.max_batch);
        assert!(start_pos + n_tokens <= kv.max_seq_len);

        let cfg = &self.config;
        let dim = cfg.dim;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hidden_dim = cfg.hidden_dim;
        let mapped_all = self.gguf.raw_bytes();

        // 1. Token embedding lookup into `s.x[t * dim ..]`.
        let emb_bs = self.token_embd.dtype.block_size();
        let emb_row_bytes = (dim / emb_bs) * self.token_embd.dtype.type_size();
        let emb_base = self.token_embd.bytes(mapped_all);
        for (t, &tok_id) in tokens.iter().enumerate() {
            let tid = (tok_id as usize).min(cfg.vocab_size - 1);
            let row_slice = &emb_base[tid * emb_row_bytes..(tid + 1) * emb_row_bytes];
            dequantize_row(
                self.token_embd.dtype,
                row_slice,
                &mut s.x[t * dim..(t + 1) * dim],
            );
        }

        // 2. Transformer layers.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // RMSNorm before attention.
            for t in 0..n_tokens {
                rmsnorm(
                    &mut s.xb[t * dim..(t + 1) * dim],
                    &s.x[t * dim..(t + 1) * dim],
                    &layer.attn_norm,
                    cfg.rms_norm_eps,
                );
            }

            // Q, K, V projections.
            matmul_batch(
                pool,
                self.backend,
                layer.wq.dtype,
                layer.wq.bytes(mapped_all),
                q_dim,
                dim,
                &s.xb,
                dim,
                &mut s.q,
                q_dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );
            matmul_batch(
                pool,
                self.backend,
                layer.wk.dtype,
                layer.wk.bytes(mapped_all),
                kv_dim,
                dim,
                &s.xb,
                dim,
                &mut s.k,
                kv_dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );
            matmul_batch(
                pool,
                self.backend,
                layer.wv.dtype,
                layer.wv.bytes(mapped_all),
                kv_dim,
                dim,
                &s.xb,
                dim,
                &mut s.v,
                kv_dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );

            // Optional QKV biases (Qwen 2 / 2.5) + optional Q/K RMSNorm (Qwen 3) + RoPE + KV write.
            for t in 0..n_tokens {
                let q_row = &mut s.q[t * q_dim..(t + 1) * q_dim];
                let k_row = &mut s.k[t * kv_dim..(t + 1) * kv_dim];
                let v_row = &mut s.v[t * kv_dim..(t + 1) * kv_dim];

                if let Some(bq) = &layer.bq {
                    add_bias(q_row, bq);
                }
                if let Some(bk) = &layer.bk {
                    add_bias(k_row, bk);
                }
                if let Some(bv) = &layer.bv {
                    add_bias(v_row, bv);
                }

                if let Some(qn) = &layer.q_norm {
                    for h in 0..cfg.n_heads {
                        rmsnorm_inplace(
                            &mut q_row[h * cfg.head_dim..(h + 1) * cfg.head_dim],
                            qn,
                            cfg.rms_norm_eps,
                        );
                    }
                }
                if let Some(kn) = &layer.k_norm {
                    for h in 0..cfg.n_kv_heads {
                        rmsnorm_inplace(
                            &mut k_row[h * cfg.head_dim..(h + 1) * cfg.head_dim],
                            kn,
                            cfg.rms_norm_eps,
                        );
                    }
                }

                let pos = start_pos + t;
                apply_rope(
                    q_row,
                    pos,
                    cfg.n_heads,
                    cfg.head_dim,
                    &self.inv_freqs,
                    cfg.rope_style,
                );
                apply_rope(
                    k_row,
                    pos,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    &self.inv_freqs,
                    cfg.rope_style,
                );

                kv.store_token(layer_idx, pos, k_row, v_row);
            }

            // Tiled online-softmax GQA attention across `(token, head)` pairs.
            self.run_tiled_attention(layer_idx, start_pos, n_tokens, pool, kv, s);

            // Output projection + residual connection.
            matmul_batch(
                pool,
                self.backend,
                layer.wo.dtype,
                layer.wo.bytes(mapped_all),
                dim,
                q_dim,
                &s.att_out,
                q_dim,
                &mut s.xb2,
                dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );
            for t in 0..n_tokens {
                let x_row = &mut s.x[t * dim..(t + 1) * dim];
                let xb2_row = &s.xb2[t * dim..(t + 1) * dim];
                for i in 0..dim {
                    x_row[i] += xb2_row[i];
                }
            }

            // FFN RMSNorm + SwiGLU MLP + residual connection.
            for t in 0..n_tokens {
                rmsnorm(
                    &mut s.xb[t * dim..(t + 1) * dim],
                    &s.x[t * dim..(t + 1) * dim],
                    &layer.ffn_norm,
                    cfg.rms_norm_eps,
                );
            }

            matmul_batch(
                pool,
                self.backend,
                layer.w_gate.dtype,
                layer.w_gate.bytes(mapped_all),
                hidden_dim,
                dim,
                &s.xb,
                dim,
                &mut s.hb,
                hidden_dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );
            matmul_batch(
                pool,
                self.backend,
                layer.w_up.dtype,
                layer.w_up.bytes(mapped_all),
                hidden_dim,
                dim,
                &s.xb,
                dim,
                &mut s.hb2,
                hidden_dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );

            for t in 0..n_tokens {
                silu_mul_inplace(
                    &mut s.hb[t * hidden_dim..(t + 1) * hidden_dim],
                    &s.hb2[t * hidden_dim..(t + 1) * hidden_dim],
                );
            }

            matmul_batch(
                pool,
                self.backend,
                layer.w_down.dtype,
                layer.w_down.bytes(mapped_all),
                dim,
                hidden_dim,
                &s.hb,
                hidden_dim,
                &mut s.xb2,
                dim,
                n_tokens,
                &mut s.xq_bytes,
                s.xq_stride,
            );
            for t in 0..n_tokens {
                let x_row = &mut s.x[t * dim..(t + 1) * dim];
                let xb2_row = &s.xb2[t * dim..(t + 1) * dim];
                for i in 0..dim {
                    x_row[i] += xb2_row[i];
                }
            }
        }

        // 3. Final RMSNorm + LM head only for the final token when requested.
        if compute_logits {
            let last_t = n_tokens - 1;
            rmsnorm(
                &mut s.xb[..dim],
                &s.x[last_t * dim..(last_t + 1) * dim],
                &self.output_norm,
                cfg.rms_norm_eps,
            );
            matmul_batch(
                pool,
                self.backend,
                self.output_head.dtype,
                self.output_head.bytes(mapped_all),
                cfg.vocab_size,
                dim,
                &s.xb[..dim],
                dim,
                &mut s.logits,
                cfg.vocab_size,
                1,
                &mut s.xq_bytes,
                s.xq_stride,
            );
        }
    }

    /// Tiled online-softmax attention across `(t, head)` tasks.
    ///
    /// Processes keys/values in `TILE_KV = 64` blocks so each query head needs only stack
    /// arrays (`scores[64]`, `acc[MAX_HEAD_DIM]`, `kv_scratch[MAX_HEAD_DIM]`) — zero heap
    /// allocations and `O(1)` auxiliary memory regardless of sequence length.
    fn run_tiled_attention(
        &self,
        layer_idx: usize,
        start_pos: usize,
        n_tokens: usize,
        pool: &ThreadPool,
        kv: &KvCache,
        scratch: &mut InferenceScratch,
    ) {
        let cfg = &self.config;
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.q_dim();
        let kv_group = (n_heads / cfg.n_kv_heads).max(1);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_jobs = n_tokens * n_heads;

        let q_all: &[f32] = &scratch.q;
        let out_ptr = scratch.att_out.as_mut_ptr() as usize;

        let run = |tid: usize, n_threads: usize| {
            let range = shard_range(total_jobs, tid, n_threads);
            if range.is_empty() {
                return;
            }
            let out_raw = out_ptr as *mut f32;
            let mut kv_scratch = [0.0f32; MAX_HEAD_DIM];
            let mut acc = [0.0f32; MAX_HEAD_DIM];
            let mut scores = [0.0f32; TILE_KV];

            for job in range {
                let t = job / n_heads;
                let h = job % n_heads;
                let kv_h = h / kv_group;
                let causal_len = start_pos + t + 1;

                let q_head = &q_all[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                let view = kv.head_view(layer_idx, kv_h);

                acc[..head_dim].fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;

                let mut tile_start = 0;
                while tile_start < causal_len {
                    let tile_end = (tile_start + TILE_KV).min(causal_len);
                    let tile_len = tile_end - tile_start;

                    let mut tile_max = f32::NEG_INFINITY;
                    for (j, score_slot) in scores[..tile_len].iter_mut().enumerate() {
                        let s = view.score_k(tile_start + j, q_head, &mut kv_scratch) * scale;
                        *score_slot = s;
                        if s > tile_max {
                            tile_max = s;
                        }
                    }

                    let new_max = m.max(tile_max);
                    if m.is_finite() {
                        let alpha = (m - new_max).exp();
                        if (alpha - 1.0).abs() > 0.0 {
                            l *= alpha;
                            for d in &mut acc[..head_dim] {
                                *d *= alpha;
                            }
                        }
                    }

                    let mut tile_sum = 0.0f32;
                    for (j, &s_val) in scores[..tile_len].iter().enumerate() {
                        let p = (s_val - new_max).exp();
                        tile_sum += p;
                        view.accum_v(tile_start + j, p, &mut acc[..head_dim], &mut kv_scratch);
                    }
                    l += tile_sum;
                    m = new_max;
                    tile_start = tile_end;
                }

                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                let dst_off = t * q_dim + h * head_dim;
                for (d, &acc_val) in acc[..head_dim].iter().enumerate() {
                    unsafe {
                        *out_raw.add(dst_off + d) = acc_val * inv_l;
                    }
                }
            }
        };

        // Run inline on tiny jobs or single-thread; parallelize across query heads otherwise.
        if pool.num_threads() <= 1 || total_jobs < 2 {
            run(0, 1);
        } else {
            pool.parallel_for(run);
        }
    }
}

#[inline]
fn add_bias(dst: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(dst.len(), bias.len());
    for (d, &b) in dst.iter_mut().zip(bias.iter()) {
        *d += b;
    }
}

fn load_f32_vec(gguf: &GgufFile, name: &str, expected_len: usize) -> Result<Vec<f32>, GgufError> {
    let info = gguf.tensor(name)?;
    if info.n_elements != expected_len {
        return Err(GgufError::Format(format!(
            "tensor '{name}' length {} != expected {expected_len}",
            info.n_elements
        )));
    }
    let bytes = gguf.tensor_bytes(info)?;
    let mut out = vec![0.0f32; expected_len];
    dequantize_row(info.dtype, bytes, &mut out);
    Ok(out)
}

fn load_opt_f32_vec(
    gguf: &GgufFile,
    name: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>, GgufError> {
    if !gguf.tensors.contains_key(name) {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(gguf, name, expected_len)?))
}

/// Re-exported for convenience in `DEFAULT_PREFILL_BATCH`.
pub const PREFILL_BATCH: usize = DEFAULT_PREFILL_BATCH;
