//! Contiguous head-major KV cache supporting `F16` and `Q8_0` storage.
//!
//! Layout is `[n_layers, n_kv_heads, max_seq_len, head_dim]` rather than token-major
//! `[seq_len, n_layers, n_kv_heads, head_dim]`. During attention at sequence position `t`,
//! scanning `0..=t` for a single `(layer, kv_head)` hits contiguous sequential addresses
//! with minimal TLB/cache misses.
//!
//! Storing keys/values as `Q8_0` (`f16` scale + `i8` quants per 32 dims) nearly halves
//! KV memory footprint and KV bandwidth at long contexts when `head_dim % 32 == 0`.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::kernels::dot_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuant {
    F16,
    Q8_0,
}

impl KvQuant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }
}

pub struct KvCache {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub quant: KvQuant,
    /// Stride in bytes for one token vector (`head_dim` elements) in a KV head.
    slot_bytes: usize,
    /// Stride in bytes for one `(layer, kv_head)` timeline of length `max_seq_len`.
    head_bytes: usize,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
}

impl KvCache {
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        quant: KvQuant,
    ) -> Self {
        let effective_quant = match quant {
            KvQuant::Q8_0 if head_dim.is_multiple_of(32) => KvQuant::Q8_0,
            _ => KvQuant::F16,
        };
        let slot_bytes = match effective_quant {
            KvQuant::F16 => head_dim * 2,
            KvQuant::Q8_0 => (head_dim / 32) * 34,
        };
        let head_bytes = max_seq_len * slot_bytes;
        let total = n_layers * n_kv_heads * head_bytes;

        Self {
            n_layers,
            n_kv_heads,
            max_seq_len,
            head_dim,
            quant: effective_quant,
            slot_bytes,
            head_bytes,
            k_data: vec![0u8; total],
            v_data: vec![0u8; total],
        }
    }

    pub fn byte_size(&self) -> usize {
        self.k_data.len() + self.v_data.len()
    }

    #[inline]
    fn head_offset(&self, layer: usize, kv_head: usize) -> usize {
        (layer * self.n_kv_heads + kv_head) * self.head_bytes
    }

    /// Writes `k_all` and `v_all` (`[n_kv_heads * head_dim]`) at sequence position `pos`.
    pub fn store_token(&mut self, layer: usize, pos: usize, k_all: &[f32], v_all: &[f32]) {
        assert!(pos < self.max_seq_len, "KV cache overflow at pos {pos}");
        let hd = self.head_dim;
        for h in 0..self.n_kv_heads {
            let off = self.head_offset(layer, h) + pos * self.slot_bytes;
            let k_src = &k_all[h * hd..(h + 1) * hd];
            let v_src = &v_all[h * hd..(h + 1) * hd];
            encode_slot(
                self.quant,
                k_src,
                &mut self.k_data[off..off + self.slot_bytes],
            );
            encode_slot(
                self.quant,
                v_src,
                &mut self.v_data[off..off + self.slot_bytes],
            );
        }
    }

    /// Returns an immutable view over keys & values for `(layer, kv_head)` usable across
    /// parallel query-head worker threads.
    #[inline]
    pub fn head_view(&self, layer: usize, kv_head: usize) -> KvHeadView<'_> {
        let base = self.head_offset(layer, kv_head);
        KvHeadView {
            quant: self.quant,
            head_dim: self.head_dim,
            slot_bytes: self.slot_bytes,
            k_bytes: &self.k_data[base..base + self.head_bytes],
            v_bytes: &self.v_data[base..base + self.head_bytes],
        }
    }
}

pub struct KvHeadView<'a> {
    quant: KvQuant,
    head_dim: usize,
    slot_bytes: usize,
    k_bytes: &'a [u8],
    v_bytes: &'a [u8],
}

impl<'a> KvHeadView<'a> {
    /// Computes `dot(q, K[pos])` directly from `F16` or `Q8_0` without allocating.
    #[inline]
    pub fn score_k(&self, pos: usize, q: &[f32], scratch: &mut [f32]) -> f32 {
        let off = pos * self.slot_bytes;
        let slot = &self.k_bytes[off..off + self.slot_bytes];
        decode_slot(self.quant, slot, &mut scratch[..self.head_dim]);
        dot_f32(q, &scratch[..self.head_dim])
    }

    /// Accumulates `acc[d] += weight * V[pos, d]` directly into `acc`.
    #[inline]
    pub fn accum_v(&self, pos: usize, weight: f32, acc: &mut [f32], scratch: &mut [f32]) {
        let off = pos * self.slot_bytes;
        let slot = &self.v_bytes[off..off + self.slot_bytes];
        decode_slot(self.quant, slot, &mut scratch[..self.head_dim]);
        crate::kernels::axpy_f32(acc, weight, &scratch[..self.head_dim]);
    }
}

#[inline]
fn encode_slot(quant: KvQuant, src: &[f32], dst: &mut [u8]) {
    match quant {
        KvQuant::F16 => {
            for (i, &v) in src.iter().enumerate() {
                dst[i * 2..i * 2 + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        }
        KvQuant::Q8_0 => {
            for (b, chunk) in src.chunks_exact(32).enumerate() {
                let mut amax = 0.0f32;
                for &v in chunk {
                    amax = amax.max(v.abs());
                }
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let out = &mut dst[b * 34..(b + 1) * 34];
                out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                for j in 0..32 {
                    out[2 + j] = (chunk[j] * id).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        }
    }
}

#[inline]
fn decode_slot(quant: KvQuant, src: &[u8], dst: &mut [f32]) {
    match quant {
        KvQuant::F16 => {
            for (i, ch) in src.chunks_exact(2).enumerate() {
                dst[i] = f16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
            }
        }
        KvQuant::Q8_0 => {
            for (b, blk) in src.chunks_exact(34).enumerate() {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let out = &mut dst[b * 32..(b + 1) * 32];
                for j in 0..32 {
                    out[j] = d * (blk[2 + j] as i8 as f32);
                }
            }
        }
    }
}
