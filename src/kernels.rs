//! SIMD and multi-threaded tensor kernels.
//!
//! Architecture dispatch:
//!   - `aarch64`: baseline NEON (`vfmaq_f32`, `vmull_s8`/`vpaddlq_s16`) plus runtime-detected
//!     ARMv8.2 `+dotprod` (`sdot` emitted via `core::arch::asm!`, since `stdarch_neon_dotprod`
//!     is still unstable on Rust stable 1.95).
//!   - `x86_64`: runtime-detected `AVX2` + `FMA` (`_mm256_fmadd_ps`, `_mm256_maddubs_epi16`).
//!   - Portable scalar fallback for all other architectures or unit-parity validation.

use crate::gguf::GgmlType;
use crate::pool::{shard_range, ThreadPool};
use crate::quant::{quantize_row_q8_0, quantize_row_q8_k, scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Llama (`NORM`): rotates adjacent pairs `(2*i, 2*i + 1)`.
    Interleaved,
    /// Qwen2 / Qwen3 (`NEOX`): rotates split halves `(i, i + rope_dims/2)`.
    SplitHalf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    NeonDotprod,
    Neon,
    Avx2Fma,
    Scalar,
}

impl SimdBackend {
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return Self::NeonDotprod;
            }
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Self::Avx2Fma;
            }
            return Self::Scalar;
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        Self::Scalar
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::NeonDotprod => "aarch64-neon-sdot",
            Self::Neon => "aarch64-neon",
            Self::Avx2Fma => "x86_64-avx2-fma",
            Self::Scalar => "scalar",
        }
    }
}

/// F32 dot product dispatched to the fastest available SIMD backend.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::dot_f32_neon(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { x86::dot_f32_avx2(a, b) };
        }
    }
    #[allow(unreachable_code)]
    scalar::dot_f32(a, b)
}

/// Computes `dst[i] += scale * src[i]` (used in attention value accumulation).
#[inline]
pub fn axpy_f32(dst: &mut [f32], scale: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::axpy_f32_neon(dst, scale, src) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            unsafe { x86::axpy_f32_avx2(dst, scale, src) };
            return;
        }
    }
    #[allow(unreachable_code)]
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += scale * s;
    }
}

/// RMS normalization: `out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)`.
/// Supports aliasing (`out` and `x` can be updated via separate buffers or via `rmsnorm_inplace`).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(out.len(), weight.len());
    let ss = dot_f32(x, x);
    let inv_rms = 1.0 / (ss / (x.len() as f32) + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let ss = dot_f32(x, x);
    let inv_rms = 1.0 / (ss / (x.len() as f32) + eps).sqrt();
    for (xi, &wi) in x.iter_mut().zip(weight.iter()) {
        *xi *= inv_rms * wi;
    }
}

/// Numerically stable in-place softmax.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_v = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        let e = (*v - max_v).exp();
        *v = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// SwiGLU activation in-place: `gate[i] = silu(gate[i]) * up[i]`.
pub fn silu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        let val = *g;
        *g = (val / (1.0 + (-val).exp())) * u;
    }
}

/// Applies Rotary Position Embedding (RoPE) across `n_heads` heads inside `vec` (`[n_heads * head_dim]`).
/// Precomputes `(cos, sin)` once per frequency index into a small stack buffer instead of
/// materializing a giant `[max_seq_len, head_dim]` lookup table (important for 128K-context models).
pub fn apply_rope(
    vec: &mut [f32],
    pos: usize,
    n_heads: usize,
    head_dim: usize,
    inv_freqs: &[f32],
    style: RopeStyle,
) {
    let half = inv_freqs.len();
    debug_assert!(half * 2 <= head_dim);
    debug_assert_eq!(vec.len(), n_heads * head_dim);

    // Max `rope_dims <= 256` covers all standard Llama/Qwen models (`head_dim <= 256`).
    let mut cos_buf = [0.0f32; 128];
    let mut sin_buf = [0.0f32; 128];
    let pos_f = pos as f32;
    for i in 0..half {
        let (s, c) = (pos_f * inv_freqs[i]).sin_cos();
        cos_buf[i] = c;
        sin_buf[i] = s;
    }

    for h in 0..n_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        match style {
            RopeStyle::Interleaved => {
                for i in 0..half {
                    let x0 = head[2 * i];
                    let x1 = head[2 * i + 1];
                    let c = cos_buf[i];
                    let s = sin_buf[i];
                    head[2 * i] = x0 * c - x1 * s;
                    head[2 * i + 1] = x0 * s + x1 * c;
                }
            }
            RopeStyle::SplitHalf => {
                for i in 0..half {
                    let x0 = head[i];
                    let x1 = head[i + half];
                    let c = cos_buf[i];
                    let s = sin_buf[i];
                    head[i] = x0 * c - x1 * s;
                    head[i + half] = x0 * s + x1 * c;
                }
            }
        }
    }
}

/// Computes `out[t, r] = dot(W[r, :], xs[t, :])` for `t in 0..n_tokens` and `r in 0..rows`.
///
/// Key optimizations:
///   1. Quantizes each input vector `xs[t, :]` into `xq_buf` (`Q8_0` or `Q8_K`) ONCE upfront.
///   2. Shards output rows across the persistent `ThreadPool`.
///   3. During batched prefill (`n_tokens > 1`), iterates weight rows on the outer loop and
///      processes up to 4 tokens simultaneously per weight row so weight bytes stay in L1D/registers
///      instead of being re-streamed from DRAM `n_tokens` times.
#[allow(clippy::too_many_arguments)]
pub fn matmul_batch(
    pool: &ThreadPool,
    backend: SimdBackend,
    dtype: GgmlType,
    weight_bytes: &[u8],
    rows: usize,
    cols: usize,
    xs: &[f32],
    x_stride: usize,
    out: &mut [f32],
    out_stride: usize,
    n_tokens: usize,
    xq_buf: &mut [u8],
    xq_stride: usize,
) {
    if n_tokens == 0 || rows == 0 {
        return;
    }

    // Step 1: quantize input activations once if weight format uses integer dot products.
    match dtype {
        GgmlType::Q4_0 | GgmlType::Q8_0 => {
            for t in 0..n_tokens {
                let x_row = &xs[t * x_stride..t * x_stride + cols];
                let q_row = &mut xq_buf[t * xq_stride..(t + 1) * xq_stride];
                quantize_row_q8_0(x_row, q_row);
            }
        }
        GgmlType::Q4K | GgmlType::Q6K => {
            for t in 0..n_tokens {
                let x_row = &xs[t * x_stride..t * x_stride + cols];
                let q_row = &mut xq_buf[t * xq_stride..(t + 1) * xq_stride];
                quantize_row_q8_k(x_row, q_row);
            }
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {}
        GgmlType::Unsupported(c) => panic!("unsupported weight dtype code {c}"),
    }

    let bs = dtype.block_size();
    let row_bytes = (cols / bs) * dtype.type_size();
    debug_assert_eq!(weight_bytes.len(), rows * row_bytes);

    let out_ptr = out.as_mut_ptr() as usize;
    let xq_ro: &[u8] = xq_buf;

    // Step 2: parallel row-sharded matmul.
    // Safety: each worker writes strictly to `out[t * out_stride + row]` for its disjoint `row` shard.
    let run_shard = |tid: usize, n_threads: usize| {
        let range = shard_range(rows, tid, n_threads);
        if range.is_empty() {
            return;
        }
        let out_raw = out_ptr as *mut f32;

        if n_tokens == 1 {
            let x_f32 = &xs[..cols];
            let x_q = &xq_ro[..xq_stride];
            for r in range {
                let w_row = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
                let val = dot_weight_row(backend, dtype, w_row, x_f32, x_q, cols);
                unsafe {
                    *out_raw.add(r) = val;
                }
            }
        } else {
            // Prefill path: outer loop over `r` keeps each weight row hot in L1 across tokens.
            for r in range {
                let w_row = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
                let mut t = 0;
                while t + 4 <= n_tokens {
                    let x0 = &xs[t * x_stride..t * x_stride + cols];
                    let x1 = &xs[(t + 1) * x_stride..(t + 1) * x_stride + cols];
                    let x2 = &xs[(t + 2) * x_stride..(t + 2) * x_stride + cols];
                    let x3 = &xs[(t + 3) * x_stride..(t + 3) * x_stride + cols];

                    let q0 = &xq_ro[t * xq_stride..(t + 1) * xq_stride];
                    let q1 = &xq_ro[(t + 1) * xq_stride..(t + 2) * xq_stride];
                    let q2 = &xq_ro[(t + 2) * xq_stride..(t + 3) * xq_stride];
                    let q3 = &xq_ro[(t + 3) * xq_stride..(t + 4) * xq_stride];

                    let [v0, v1, v2, v3] = dot_weight_row_4(
                        backend,
                        dtype,
                        w_row,
                        [x0, x1, x2, x3],
                        [q0, q1, q2, q3],
                        cols,
                    );
                    unsafe {
                        *out_raw.add(t * out_stride + r) = v0;
                        *out_raw.add((t + 1) * out_stride + r) = v1;
                        *out_raw.add((t + 2) * out_stride + r) = v2;
                        *out_raw.add((t + 3) * out_stride + r) = v3;
                    }
                    t += 4;
                }
                while t < n_tokens {
                    let x_f32 = &xs[t * x_stride..t * x_stride + cols];
                    let x_q = &xq_ro[t * xq_stride..(t + 1) * xq_stride];
                    let val = dot_weight_row(backend, dtype, w_row, x_f32, x_q, cols);
                    unsafe {
                        *out_raw.add(t * out_stride + r) = val;
                    }
                    t += 1;
                }
            }
        }
    };

    // Skip pool synchronization on very small projections.
    if pool.num_threads() <= 1 || rows * n_tokens < 64 {
        run_shard(0, 1);
    } else {
        pool.parallel_for(run_shard);
    }
}

#[inline]
pub fn dot_weight_row(
    backend: SimdBackend,
    dtype: GgmlType,
    w_row: &[u8],
    x_f32: &[f32],
    x_q: &[u8],
    cols: usize,
) -> f32 {
    match dtype {
        GgmlType::F32 => {
            #[cfg(target_endian = "little")]
            {
                let w_f32 = unsafe {
                    std::slice::from_raw_parts(w_row.as_ptr().cast::<f32>(), cols)
                };
                dot_f32(w_f32, x_f32)
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut sum = 0.0f32;
                for (ch, &xv) in w_row.chunks_exact(4).zip(x_f32.iter()) {
                    sum += f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]) * xv;
                }
                sum
            }
        }
        GgmlType::F16 => {
            #[cfg(target_arch = "aarch64")]
            {
                return unsafe { arm::dot_f16_f32_neon(w_row, x_f32) };
            }
            #[allow(unreachable_code)]
            scalar::dot_f16_f32(w_row, x_f32)
        }
        GgmlType::BF16 => scalar::dot_bf16_f32(w_row, x_f32),
        GgmlType::Q8_0 => match backend {
            #[cfg(target_arch = "aarch64")]
            SimdBackend::NeonDotprod => unsafe { arm::dot_q8_0_q8_0_sdot(w_row, x_q, cols) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { arm::dot_q8_0_q8_0_neon(w_row, x_q, cols) },
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2Fma => unsafe { x86::dot_q8_0_q8_0_avx2(w_row, x_q, cols) },
            _ => scalar::dot_q8_0_q8_0(w_row, x_q, cols),
        },
        GgmlType::Q4_0 => match backend {
            #[cfg(target_arch = "aarch64")]
            SimdBackend::NeonDotprod => unsafe { arm::dot_q4_0_q8_0_sdot(w_row, x_q, cols) },
            #[cfg(target_arch = "aarch64")]
            SimdBackend::Neon => unsafe { arm::dot_q4_0_q8_0_neon(w_row, x_q, cols) },
            #[cfg(target_arch = "x86_64")]
            SimdBackend::Avx2Fma => unsafe { x86::dot_q4_0_q8_0_avx2(w_row, x_q, cols) },
            _ => scalar::dot_q4_0_q8_0(w_row, x_q, cols),
        },
        GgmlType::Q4K => match backend {
            #[cfg(target_arch = "aarch64")]
            SimdBackend::NeonDotprod => unsafe { arm::dot_q4_k_q8_k_sdot(w_row, x_q, cols) },
            _ => scalar::dot_q4_k_q8_k(w_row, x_q, cols),
        },
        GgmlType::Q6K => match backend {
            #[cfg(target_arch = "aarch64")]
            SimdBackend::NeonDotprod => unsafe { arm::dot_q6_k_q8_k_sdot(w_row, x_q, cols) },
            _ => scalar::dot_q6_k_q8_k(w_row, x_q, cols),
        },
        GgmlType::Unsupported(c) => panic!("unsupported dtype {c}"),
    }
}

/// Computes 4 dot products sharing the same weight row so each weight block is loaded once.
#[inline]
fn dot_weight_row_4(
    backend: SimdBackend,
    dtype: GgmlType,
    w_row: &[u8],
    xs: [&[f32]; 4],
    xqs: [&[u8]; 4],
    cols: usize,
) -> [f32; 4] {
    match dtype {
        GgmlType::Q4_0 if backend == SimdBackend::NeonDotprod => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                return arm::dot_q4_0_q8_0_sdot_4(w_row, xqs, cols);
            }
            #[allow(unreachable_code)]
            [
                dot_weight_row(backend, dtype, w_row, xs[0], xqs[0], cols),
                dot_weight_row(backend, dtype, w_row, xs[1], xqs[1], cols),
                dot_weight_row(backend, dtype, w_row, xs[2], xqs[2], cols),
                dot_weight_row(backend, dtype, w_row, xs[3], xqs[3], cols),
            ]
        }
        GgmlType::Q8_0 if backend == SimdBackend::NeonDotprod => {
            #[cfg(target_arch = "aarch64")]
            unsafe {
                return arm::dot_q8_0_q8_0_sdot_4(w_row, xqs, cols);
            }
            #[allow(unreachable_code)]
            [
                dot_weight_row(backend, dtype, w_row, xs[0], xqs[0], cols),
                dot_weight_row(backend, dtype, w_row, xs[1], xqs[1], cols),
                dot_weight_row(backend, dtype, w_row, xs[2], xqs[2], cols),
                dot_weight_row(backend, dtype, w_row, xs[3], xqs[3], cols),
            ]
        }
        _ => [
            dot_weight_row(backend, dtype, w_row, xs[0], xqs[0], cols),
            dot_weight_row(backend, dtype, w_row, xs[1], xqs[1], cols),
            dot_weight_row(backend, dtype, w_row, xs[2], xqs[2], cols),
            dot_weight_row(backend, dtype, w_row, xs[3], xqs[3], cols),
        ],
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;
    use std::arch::asm;

    use crate::f16::f16_to_f32;
    use crate::quant::{unpack_q4k_scale_min, ActQ8K, ActQ8_0};

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f16_f32_neon(w_bytes: &[u8], x: &[f32]) -> f32 {
        let len = x.len();
        let wp = w_bytes.as_ptr();
        let xp = x.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0usize;
        while i + 16 <= len {
            unsafe {
                let h0 = vld1q_u8(wp.add(i * 2));
                let h1 = vld1q_u8(wp.add((i + 8) * 2));
                let (f0, f1, f2, f3): (float32x4_t, float32x4_t, float32x4_t, float32x4_t);
                asm!(
                    "fcvtl {f0:v}.4s, {h0:v}.4h",
                    "fcvtl2 {f1:v}.4s, {h0:v}.8h",
                    "fcvtl {f2:v}.4s, {h1:v}.4h",
                    "fcvtl2 {f3:v}.4s, {h1:v}.8h",
                    h0 = in(vreg) h0,
                    h1 = in(vreg) h1,
                    f0 = out(vreg) f0,
                    f1 = out(vreg) f1,
                    f2 = out(vreg) f2,
                    f3 = out(vreg) f3,
                    options(pure, nomem, nostack)
                );
                acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp.add(i)));
                acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xp.add(i + 4)));
                acc2 = vfmaq_f32(acc2, f2, vld1q_f32(xp.add(i + 8)));
                acc3 = vfmaq_f32(acc3, f3, vld1q_f32(xp.add(i + 12)));
            }
            i += 16;
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 8 <= len {
            unsafe {
                let h0 = vld1q_u8(wp.add(i * 2));
                let (f0, f1): (float32x4_t, float32x4_t);
                asm!(
                    "fcvtl {f0:v}.4s, {h0:v}.4h",
                    "fcvtl2 {f1:v}.4s, {h0:v}.8h",
                    h0 = in(vreg) h0,
                    f0 = out(vreg) f0,
                    f1 = out(vreg) f1,
                    options(pure, nomem, nostack)
                );
                acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp.add(i)));
                acc0 = vfmaq_f32(acc0, f1, vld1q_f32(xp.add(i + 4)));
            }
            i += 8;
        }
        let mut sum = vaddvq_f32(acc0);
        while i < len {
            unsafe {
                sum += read_f16_le(wp.add(i * 2)) * *xp.add(i);
            }
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0usize;
        while i + 16 <= len {
            unsafe {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            }
            i += 16;
        }
        acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        while i + 4 <= len {
            unsafe {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            }
            i += 4;
        }
        let mut sum = vaddvq_f32(acc0);
        while i < len {
            unsafe {
                sum += *pa.add(i) * *pb.add(i);
            }
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn axpy_f32_neon(dst: &mut [f32], scale: f32, src: &[f32]) {
        let len = dst.len();
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        let vs = vdupq_n_f32(scale);
        let mut i = 0usize;
        while i + 16 <= len {
            unsafe {
                let d0 = vfmaq_f32(vld1q_f32(pd.add(i)), vs, vld1q_f32(ps.add(i)));
                let d1 = vfmaq_f32(vld1q_f32(pd.add(i + 4)), vs, vld1q_f32(ps.add(i + 4)));
                let d2 = vfmaq_f32(vld1q_f32(pd.add(i + 8)), vs, vld1q_f32(ps.add(i + 8)));
                let d3 = vfmaq_f32(vld1q_f32(pd.add(i + 12)), vs, vld1q_f32(ps.add(i + 12)));
                vst1q_f32(pd.add(i), d0);
                vst1q_f32(pd.add(i + 4), d1);
                vst1q_f32(pd.add(i + 8), d2);
                vst1q_f32(pd.add(i + 12), d3);
            }
            i += 16;
        }
        while i + 4 <= len {
            unsafe {
                let d0 = vfmaq_f32(vld1q_f32(pd.add(i)), vs, vld1q_f32(ps.add(i)));
                vst1q_f32(pd.add(i), d0);
            }
            i += 4;
        }
        while i < len {
            unsafe {
                *pd.add(i) += scale * *ps.add(i);
            }
            i += 1;
        }
    }

    #[inline(always)]
    unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        unsafe {
            asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack)
            );
        }
        acc
    }

    #[inline(always)]
    unsafe fn read_f16_le(ptr: *const u8) -> f32 {
        let bits = unsafe { u16::from_le_bytes([*ptr, *ptr.add(1)]) };
        f16_to_f32(bits)
    }

    #[inline(always)]
    unsafe fn read_f32_le(ptr: *const u8) -> f32 {
        unsafe { f32::from_le_bytes([*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q8_0_q8_0_neon(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let mut sumf = 0.0f32;

        for _ in 0..n_blocks {
            unsafe {
                let d_w = read_f16_le(wp);
                let d_x = read_f32_le(xp);
                let w0 = vld1q_s8(wp.add(2).cast::<i8>());
                let w1 = vld1q_s8(wp.add(18).cast::<i8>());
                let x0 = vld1q_s8(xp.add(4).cast::<i8>());
                let x1 = vld1q_s8(xp.add(20).cast::<i8>());

                let p0 = vaddq_s16(
                    vmull_s8(vget_low_s8(w0), vget_low_s8(x0)),
                    vmull_high_s8(w0, x0),
                );
                let p1 = vaddq_s16(
                    vmull_s8(vget_low_s8(w1), vget_low_s8(x1)),
                    vmull_high_s8(w1, x1),
                );
                let acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                sumf += (d_w * d_x) * (vaddvq_s32(acc) as f32);

                wp = wp.add(34);
                xp = xp.add(x_stride);
            }
        }
        sumf
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q8_0_q8_0_sdot(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;

        let mut b = 0usize;
        while b + 2 <= n_blocks {
            unsafe {
                let dw0 = read_f16_le(wp);
                let dx0 = read_f32_le(xp);
                let mut acc0 = vdupq_n_s32(0);
                acc0 = sdot(acc0, vld1q_s8(wp.add(2).cast()), vld1q_s8(xp.add(4).cast()));
                acc0 = sdot(acc0, vld1q_s8(wp.add(18).cast()), vld1q_s8(xp.add(20).cast()));

                let wp1 = wp.add(34);
                let xp1 = xp.add(x_stride);
                let dw1 = read_f16_le(wp1);
                let dx1 = read_f32_le(xp1);
                let mut acc1 = vdupq_n_s32(0);
                acc1 = sdot(acc1, vld1q_s8(wp1.add(2).cast()), vld1q_s8(xp1.add(4).cast()));
                acc1 = sdot(acc1, vld1q_s8(wp1.add(18).cast()), vld1q_s8(xp1.add(20).cast()));

                sum0 += (dw0 * dx0) * (vaddvq_s32(acc0) as f32);
                sum1 += (dw1 * dx1) * (vaddvq_s32(acc1) as f32);

                wp = wp.add(68);
                xp = xp.add(2 * x_stride);
            }
            b += 2;
        }
        if b < n_blocks {
            unsafe {
                let dw = read_f16_le(wp);
                let dx = read_f32_le(xp);
                let mut acc = vdupq_n_s32(0);
                acc = sdot(acc, vld1q_s8(wp.add(2).cast()), vld1q_s8(xp.add(4).cast()));
                acc = sdot(acc, vld1q_s8(wp.add(18).cast()), vld1q_s8(xp.add(20).cast()));
                sum0 += (dw * dx) * (vaddvq_s32(acc) as f32);
            }
        }
        sum0 + sum1
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q8_0_q8_0_sdot_4(
        w_bytes: &[u8],
        xqs: [&[u8]; 4],
        cols: usize,
    ) -> [f32; 4] {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp0 = xqs[0].as_ptr();
        let mut xp1 = xqs[1].as_ptr();
        let mut xp2 = xqs[2].as_ptr();
        let mut xp3 = xqs[3].as_ptr();
        let mut sums = [0.0f32; 4];

        for _ in 0..n_blocks {
            unsafe {
                let dw = read_f16_le(wp);
                let w_lo = vld1q_s8(wp.add(2).cast());
                let w_hi = vld1q_s8(wp.add(18).cast());

                let mut a0 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp0.add(4).cast()));
                a0 = sdot(a0, w_hi, vld1q_s8(xp0.add(20).cast()));
                let mut a1 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp1.add(4).cast()));
                a1 = sdot(a1, w_hi, vld1q_s8(xp1.add(20).cast()));
                let mut a2 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp2.add(4).cast()));
                a2 = sdot(a2, w_hi, vld1q_s8(xp2.add(20).cast()));
                let mut a3 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp3.add(4).cast()));
                a3 = sdot(a3, w_hi, vld1q_s8(xp3.add(20).cast()));

                sums[0] += (dw * read_f32_le(xp0)) * (vaddvq_s32(a0) as f32);
                sums[1] += (dw * read_f32_le(xp1)) * (vaddvq_s32(a1) as f32);
                sums[2] += (dw * read_f32_le(xp2)) * (vaddvq_s32(a2) as f32);
                sums[3] += (dw * read_f32_le(xp3)) * (vaddvq_s32(a3) as f32);

                wp = wp.add(34);
                xp0 = xp0.add(x_stride);
                xp1 = xp1.add(x_stride);
                xp2 = xp2.add(x_stride);
                xp3 = xp3.add(x_stride);
            }
        }
        sums
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn dot_q4_0_q8_0_neon(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let mask_low = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut sumf = 0.0f32;

        for _ in 0..n_blocks {
            unsafe {
                let d_w = read_f16_le(wp);
                let d_x = read_f32_le(xp);
                let q_bytes = vld1q_u8(wp.add(2));
                let w_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q_bytes, mask_low)), eight);
                let w_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q_bytes)), eight);

                let x_lo = vld1q_s8(xp.add(4).cast());
                let x_hi = vld1q_s8(xp.add(20).cast());

                let p0 = vaddq_s16(
                    vmull_s8(vget_low_s8(w_lo), vget_low_s8(x_lo)),
                    vmull_high_s8(w_lo, x_lo),
                );
                let p1 = vaddq_s16(
                    vmull_s8(vget_low_s8(w_hi), vget_low_s8(x_hi)),
                    vmull_high_s8(w_hi, x_hi),
                );
                let acc = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
                sumf += (d_w * d_x) * (vaddvq_s32(acc) as f32);

                wp = wp.add(18);
                xp = xp.add(x_stride);
            }
        }
        sumf
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_0_q8_0_sdot(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let mask_low = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;

        let mut b = 0usize;
        while b + 2 <= n_blocks {
            unsafe {
                let dw0 = read_f16_le(wp);
                let dx0 = read_f32_le(xp);
                let q0 = vld1q_u8(wp.add(2));
                let w0_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q0, mask_low)), eight);
                let w0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0)), eight);
                let mut a0 = sdot(vdupq_n_s32(0), w0_lo, vld1q_s8(xp.add(4).cast()));
                a0 = sdot(a0, w0_hi, vld1q_s8(xp.add(20).cast()));

                let wp1 = wp.add(18);
                let xp1 = xp.add(x_stride);
                let dw1 = read_f16_le(wp1);
                let dx1 = read_f32_le(xp1);
                let q1 = vld1q_u8(wp1.add(2));
                let w1_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q1, mask_low)), eight);
                let w1_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1)), eight);
                let mut a1 = sdot(vdupq_n_s32(0), w1_lo, vld1q_s8(xp1.add(4).cast()));
                a1 = sdot(a1, w1_hi, vld1q_s8(xp1.add(20).cast()));

                sum0 += (dw0 * dx0) * (vaddvq_s32(a0) as f32);
                sum1 += (dw1 * dx1) * (vaddvq_s32(a1) as f32);

                wp = wp.add(36);
                xp = xp.add(2 * x_stride);
            }
            b += 2;
        }
        if b < n_blocks {
            unsafe {
                let dw = read_f16_le(wp);
                let dx = read_f32_le(xp);
                let q = vld1q_u8(wp.add(2));
                let w_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, mask_low)), eight);
                let w_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), eight);
                let mut acc = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp.add(4).cast()));
                acc = sdot(acc, w_hi, vld1q_s8(xp.add(20).cast()));
                sum0 += (dw * dx) * (vaddvq_s32(acc) as f32);
            }
        }
        sum0 + sum1
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_0_q8_0_sdot_4(
        w_bytes: &[u8],
        xqs: [&[u8]; 4],
        cols: usize,
    ) -> [f32; 4] {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp0 = xqs[0].as_ptr();
        let mut xp1 = xqs[1].as_ptr();
        let mut xp2 = xqs[2].as_ptr();
        let mut xp3 = xqs[3].as_ptr();
        let mask_low = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);
        let mut sums = [0.0f32; 4];

        for _ in 0..n_blocks {
            unsafe {
                let dw = read_f16_le(wp);
                let q = vld1q_u8(wp.add(2));
                let w_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, mask_low)), eight);
                let w_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(q)), eight);

                let mut a0 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp0.add(4).cast()));
                a0 = sdot(a0, w_hi, vld1q_s8(xp0.add(20).cast()));
                let mut a1 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp1.add(4).cast()));
                a1 = sdot(a1, w_hi, vld1q_s8(xp1.add(20).cast()));
                let mut a2 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp2.add(4).cast()));
                a2 = sdot(a2, w_hi, vld1q_s8(xp2.add(20).cast()));
                let mut a3 = sdot(vdupq_n_s32(0), w_lo, vld1q_s8(xp3.add(4).cast()));
                a3 = sdot(a3, w_hi, vld1q_s8(xp3.add(20).cast()));

                sums[0] += (dw * read_f32_le(xp0)) * (vaddvq_s32(a0) as f32);
                sums[1] += (dw * read_f32_le(xp1)) * (vaddvq_s32(a1) as f32);
                sums[2] += (dw * read_f32_le(xp2)) * (vaddvq_s32(a2) as f32);
                sums[3] += (dw * read_f32_le(xp3)) * (vaddvq_s32(a3) as f32);

                wp = wp.add(18);
                xp0 = xp0.add(x_stride);
                xp1 = xp1.add(x_stride);
                xp2 = xp2.add(x_stride);
                xp3 = xp3.add(x_stride);
            }
        }
        sums
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q4_k_q8_k_sdot(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 256;
        let x_stride = std::mem::size_of::<ActQ8K>();
        let mask_low = vdupq_n_u8(0x0F);
        let mut sumf = 0.0f32;

        for b in 0..n_blocks {
            let wb = &w_bytes[b * 144..(b + 1) * 144];
            let xb = &x_bytes[b * x_stride..(b + 1) * x_stride];
            let d_w = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([wb[2], wb[3]]));
            let scales: &[u8; 12] = wb[4..16].try_into().unwrap();
            let qs = &wb[16..144];
            let d_x = f32::from_le_bytes([xb[0], xb[1], xb[2], xb[3]]);
            let x_qs = &xb[4..260];

            let mut sum_sc = 0i32;
            let mut sum_min = 0i32;
            for chunk in 0..4 {
                let is = chunk * 2;
                let (sc0, m0) = unpack_q4k_scale_min(is, scales);
                let (sc1, m1) = unpack_q4k_scale_min(is + 1, scales);

                let qp = unsafe { qs.as_ptr().add(chunk * 32) };
                let xp = unsafe { x_qs.as_ptr().add(chunk * 64).cast::<i8>() };

                unsafe {
                    let q0 = vld1q_u8(qp);
                    let q1 = vld1q_u8(qp.add(16));

                    // Values 0..15 fit in signed i8 without biasing.
                    let w0_a = vreinterpretq_s8_u8(vandq_u8(q0, mask_low));
                    let w0_b = vreinterpretq_s8_u8(vandq_u8(q1, mask_low));
                    let mut a0 = sdot(vdupq_n_s32(0), w0_a, vld1q_s8(xp));
                    a0 = sdot(a0, w0_b, vld1q_s8(xp.add(16)));

                    let w1_a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                    let w1_b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
                    let mut a1 = sdot(vdupq_n_s32(0), w1_a, vld1q_s8(xp.add(32)));
                    a1 = sdot(a1, w1_b, vld1q_s8(xp.add(48)));

                    let dot0 = vaddvq_s32(a0);
                    let dot1 = vaddvq_s32(a1);

                    let bs0 = i16::from_le_bytes([xb[260 + is * 4], xb[261 + is * 4]]) as i32
                        + i16::from_le_bytes([xb[262 + is * 4], xb[263 + is * 4]]) as i32;
                    let bs1 = i16::from_le_bytes([xb[264 + is * 4], xb[265 + is * 4]]) as i32
                        + i16::from_le_bytes([xb[266 + is * 4], xb[267 + is * 4]]) as i32;

                    sum_sc += (sc0 as i32) * dot0 + (sc1 as i32) * dot1;
                    sum_min += (m0 as i32) * bs0 + (m1 as i32) * bs1;
                }
            }
            sumf += (d_w * d_x) * (sum_sc as f32) - (dmin_w * d_x) * (sum_min as f32);
        }
        sumf
    }

    #[target_feature(enable = "neon,dotprod")]
    pub unsafe fn dot_q6_k_q8_k_sdot(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 256;
        let x_stride = std::mem::size_of::<ActQ8K>();
        let mask_0f = vdupq_n_u8(0x0F);
        let mask_03 = vdupq_n_u8(0x03);
        let mask_0c = vdupq_n_u8(0x0C);
        let mask_30 = vdupq_n_u8(0x30);
        let mask_c0 = vdupq_n_u8(0xC0);
        let c32 = vdupq_n_s8(32);
        let mut sumf = 0.0f32;

        for b in 0..n_blocks {
            let wb = &w_bytes[b * 210..(b + 1) * 210];
            let xb = &x_bytes[b * x_stride..(b + 1) * x_stride];
            let ql = &wb[0..128];
            let qh = &wb[128..192];
            let scales = &wb[192..208];
            let d_w = f16_to_f32(u16::from_le_bytes([wb[208], wb[209]]));
            let d_x = f32::from_le_bytes([xb[0], xb[1], xb[2], xb[3]]);
            let x_qs = &xb[4..260];

            let mut isum = 0i32;
            for n in 0..2 {
                let ql_h = &ql[n * 64..n * 64 + 64];
                let qh_h = &qh[n * 32..n * 32 + 32];
                let sc_h = &scales[n * 8..n * 8 + 8];
                let x_h = &x_qs[n * 128..n * 128 + 128];

                for is in 0..2 {
                    let off = is * 16;
                    unsafe {
                        let ql0 = vld1q_u8(ql_h.as_ptr().add(off));
                        let ql1 = vld1q_u8(ql_h.as_ptr().add(32 + off));
                        let qhv = vld1q_u8(qh_h.as_ptr().add(off));
                        let xp = x_h.as_ptr().add(off).cast::<i8>();

                        let q1 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(ql0, mask_0f),
                                vshlq_n_u8::<4>(vandq_u8(qhv, mask_03)),
                            )),
                            c32,
                        );
                        let q2 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(ql1, mask_0f),
                                vshlq_n_u8::<2>(vandq_u8(qhv, mask_0c)),
                            )),
                            c32,
                        );
                        let q3 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(ql0),
                                vandq_u8(qhv, mask_30),
                            )),
                            c32,
                        );
                        let q4 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(ql1),
                                vshrq_n_u8::<2>(vandq_u8(qhv, mask_c0)),
                            )),
                            c32,
                        );

                        let d1 = vaddvq_s32(sdot(vdupq_n_s32(0), q1, vld1q_s8(xp)));
                        let d2 = vaddvq_s32(sdot(vdupq_n_s32(0), q2, vld1q_s8(xp.add(32))));
                        let d3 = vaddvq_s32(sdot(vdupq_n_s32(0), q3, vld1q_s8(xp.add(64))));
                        let d4 = vaddvq_s32(sdot(vdupq_n_s32(0), q4, vld1q_s8(xp.add(96))));

                        isum += (sc_h[is] as i8 as i32) * d1
                            + (sc_h[is + 2] as i8 as i32) * d2
                            + (sc_h[is + 4] as i8 as i32) * d3
                            + (sc_h[is + 6] as i8 as i32) * d4;
                    }
                }
            }
            sumf += (d_w * d_x) * (isum as f32);
        }
        sumf
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use crate::f16::f16_to_f32;
    use crate::quant::ActQ8_0;

    #[inline(always)]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        unsafe {
            let hi = _mm256_extractf128_ps::<1>(v);
            let lo = _mm256_castps256_ps128(v);
            let sum = _mm_add_ps(lo, hi);
            let shuf = _mm_movehdup_ps(sum);
            let sums = _mm_add_ps(sum, shuf);
            let shuf2 = _mm_movehl_ps(sums, sums);
            _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
        }
    }

    #[inline(always)]
    unsafe fn hsum256_epi32(v: __m256i) -> i32 {
        unsafe {
            let hi = _mm256_extracti128_si256::<1>(v);
            let lo = _mm256_castsi256_si128(v);
            let sum = _mm_add_epi32(lo, hi);
            let sum2 = _mm_add_epi32(sum, _mm_shuffle_epi32::<0x4E>(sum));
            let sum3 = _mm_add_epi32(sum2, _mm_shuffle_epi32::<0xB1>(sum2));
            _mm_cvtsi128_si32(sum3)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= len {
            unsafe {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                    acc1,
                );
            }
            i += 16;
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        while i + 8 <= len {
            unsafe {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            }
            i += 8;
        }
        let mut sum = unsafe { hsum256_ps(acc0) };
        while i < len {
            unsafe {
                sum += *pa.add(i) * *pb.add(i);
            }
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_f32_avx2(dst: &mut [f32], scale: f32, src: &[f32]) {
        let len = dst.len();
        let pd = dst.as_mut_ptr();
        let ps = src.as_ptr();
        let vs = _mm256_set1_ps(scale);
        let mut i = 0usize;
        while i + 8 <= len {
            unsafe {
                let d = _mm256_fmadd_ps(vs, _mm256_loadu_ps(ps.add(i)), _mm256_loadu_ps(pd.add(i)));
                _mm256_storeu_ps(pd.add(i), d);
            }
            i += 8;
        }
        while i < len {
            unsafe {
                *pd.add(i) += scale * *ps.add(i);
            }
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8_0_q8_0_avx2(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let ones = _mm256_set1_epi16(1);
        let mut sumf = 0.0f32;

        for _ in 0..n_blocks {
            unsafe {
                let dw = f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)]));
                let dx = f32::from_le_bytes([*xp, *xp.add(1), *xp.add(2), *xp.add(3)]);
                let vw = _mm256_loadu_si256(wp.add(2).cast());
                let vx = _mm256_loadu_si256(xp.add(4).cast());
                // `ActQ8_0` clamps `vx` to [-127, 127], so `_mm256_sign_epi8(vx, vx)` cannot overflow -128.
                let ax = _mm256_sign_epi8(vx, vx);
                let sw = _mm256_sign_epi8(vw, vx);
                let p16 = _mm256_maddubs_epi16(ax, sw);
                let p32 = _mm256_madd_epi16(p16, ones);
                sumf += (dw * dx) * (hsum256_epi32(p32) as f32);

                wp = wp.add(34);
                xp = xp.add(x_stride);
            }
        }
        sumf
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4_0_q8_0_avx2(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut wp = w_bytes.as_ptr();
        let mut xp = x_bytes.as_ptr();
        let mask_low = _mm256_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut sumf = 0.0f32;

        for _ in 0..n_blocks {
            unsafe {
                let dw = f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)]));
                let dx = f32::from_le_bytes([*xp, *xp.add(1), *xp.add(2), *xp.add(3)]);
                let q128 = _mm_loadu_si128(wp.add(2).cast());
                let q_lo = _mm_and_si128(q128, _mm256_castsi256_si128(mask_low));
                let q_hi = _mm_and_si128(_mm_srli_epi16::<4>(q128), _mm256_castsi256_si128(mask_low));
                let q_u8 = _mm256_set_m128i(q_hi, q_lo);
                let q_i8 = _mm256_sub_epi8(q_u8, eight);

                let vx = _mm256_loadu_si256(xp.add(4).cast());
                let ax = _mm256_sign_epi8(vx, vx);
                let sq = _mm256_sign_epi8(q_i8, vx);
                let p16 = _mm256_maddubs_epi16(ax, sq);
                let p32 = _mm256_madd_epi16(p16, ones);
                sumf += (dw * dx) * (hsum256_epi32(p32) as f32);

                wp = wp.add(18);
                xp = xp.add(x_stride);
            }
        }
        sumf
    }
}
