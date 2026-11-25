//! GGML block quantization formats and on-the-fly activation quantizers.
//!
//! Supported weight formats:
//!   - `F32`, `F16`, `BF16`: unquantized scalars.
//!   - `Q8_0`: 32 weights/block (`f16` scale + 32 `i8` quants, 34 bytes).
//!   - `Q4_0`: 32 weights/block (`f16` scale + 16 packed bytes, 18 bytes).
//!     Note GGML split-nibble layout: byte `j` stores element `j` in low nibble
//!     and element `j + 16` in high nibble (NOT adjacent pairs).
//!   - `Q4_K`: 256-weight super-blocks (8x32 sub-blocks with packed 6-bit scales/mins, 144 bytes).
//!   - `Q6_K`: 256-weight super-blocks (16x16 sub-blocks, 6-bit signed quants, 210 bytes).
//!
//! Rather than dequantizing weights to `f32` in the inner matmul loop, we quantize the
//! activation vector once into `ActQ8_0` or `ActQ8K` inside `InferenceScratch`, then run
//! pure integer SIMD dot products (`sdot` / NEON widening on ARM64, `maddubs` on x86_64).

use crate::f16::{bf16_to_f32, f16_to_f32};
use crate::gguf::GgmlType;

/// Byte stride per 32-element activation slot inside `InferenceScratch`.
/// Sized so both `ActQ8_0` (36 bytes per 32 elems) and `ActQ8K` (292 bytes per 256 elems
/// = 36.5 bytes per 32 elems) fit comfortably without reallocation.
pub const MAX_QUANT_BLOCK_BYTES: usize = 40;

/// Scratch activation block for `Q4_0` and `Q8_0` dot products (32 elements).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQ8_0 {
    pub d: f32,
    pub qs: [i8; 32],
}

/// Scratch activation super-block for `Q4_K` and `Q6_K` dot products (256 elements).
/// `bsums[g]` caches `sum(qs[16*g .. 16*g+16])` so the `Q4_K` asymmetric `min` term
/// requires zero activation summation inside the inner weight-row loop.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

/// Unpacks 6-bit scale and min for sub-block `j` (`0..8`) of a `Q4_K` super-block.
#[inline]
pub fn unpack_q4k_scale_min(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Quantizes `x` (`x.len() % 32 == 0`) into `ActQ8_0` blocks inside `out_bytes`.
pub fn quantize_row_q8_0(x: &[f32], out_bytes: &mut [u8]) {
    debug_assert_eq!(x.len() % 32, 0);
    let n_blocks = x.len() / 32;
    let block_size = std::mem::size_of::<ActQ8_0>();
    debug_assert!(out_bytes.len() >= n_blocks * block_size);

    for (b, chunk) in x.chunks_exact(32).enumerate() {
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut block = ActQ8_0 { d, qs: [0; 32] };
        for (dst_q, &val) in block.qs.iter_mut().zip(chunk.iter()) {
            *dst_q = (val * id).round().clamp(-127.0, 127.0) as i8;
        }

        let dst = &mut out_bytes[b * block_size..(b + 1) * block_size];
        dst[0..4].copy_from_slice(&block.d.to_le_bytes());
        // Safety: `[i8; 32]` and `[u8; 32]` have identical size and alignment 1.
        let qs_u8: &[u8; 32] = unsafe { &*((&block.qs as *const [i8; 32]).cast::<[u8; 32]>()) };
        dst[4..36].copy_from_slice(qs_u8);
    }
}

/// Quantizes `x` (`x.len() % 256 == 0`) into `ActQ8K` super-blocks inside `out_bytes`.
pub fn quantize_row_q8_k(x: &[f32], out_bytes: &mut [u8]) {
    debug_assert_eq!(x.len() % 256, 0);
    let n_blocks = x.len() / 256;
    let block_size = std::mem::size_of::<ActQ8K>();
    debug_assert!(out_bytes.len() >= n_blocks * block_size);

    for (b, chunk) in x.chunks_exact(256).enumerate() {
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for (g, bsum) in bsums.iter_mut().enumerate() {
            let base = g * 16;
            let mut sum = 0i16;
            for j in 0..16 {
                let q = (chunk[base + j] * id).round().clamp(-127.0, 127.0) as i8;
                qs[base + j] = q;
                sum += q as i16;
            }
            *bsum = sum;
        }

        let dst = &mut out_bytes[b * block_size..(b + 1) * block_size];
        dst[0..4].copy_from_slice(&d.to_le_bytes());
        let qs_u8: &[u8; 256] = unsafe { &*((&qs as *const [i8; 256]).cast::<[u8; 256]>()) };
        dst[4..260].copy_from_slice(qs_u8);
        for g in 0..16 {
            dst[260 + g * 2..262 + g * 2].copy_from_slice(&bsums[g].to_le_bytes());
        }
    }
}

/// Dequantizes a single row of any supported `GgmlType` into `out` (`out.len() == cols`).
/// Used for embedding lookup, small 1D/norms conversion, and numerical parity tests.
pub fn dequantize_row(dtype: GgmlType, src: &[u8], out: &mut [f32]) {
    match dtype {
        GgmlType::F32 => {
            for (dst, ch) in out.iter_mut().zip(src.chunks_exact(4)) {
                *dst = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            }
        }
        GgmlType::F16 => {
            for (dst, ch) in out.iter_mut().zip(src.chunks_exact(2)) {
                *dst = f16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
            }
        }
        GgmlType::BF16 => {
            for (dst, ch) in out.iter_mut().zip(src.chunks_exact(2)) {
                *dst = bf16_to_f32(u16::from_le_bytes([ch[0], ch[1]]));
            }
        }
        GgmlType::Q8_0 => {
            for (blk, dst) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..32 {
                    dst[j] = d * (blk[2 + j] as i8 as f32);
                }
            }
        }
        GgmlType::Q4_0 => {
            for (blk, dst) in src.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..16 {
                    let byte = blk[2 + j];
                    let lo = ((byte & 0x0F) as i8) - 8;
                    let hi = ((byte >> 4) as i8) - 8;
                    dst[j] = d * (lo as f32);
                    dst[j + 16] = d * (hi as f32);
                }
            }
        }
        GgmlType::Q4K => {
            for (blk, dst) in src.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let scales: &[u8; 12] = blk[4..16].try_into().unwrap();
                let qs = &blk[16..144];

                for chunk in 0..4 {
                    let is = chunk * 2;
                    let (sc0, m0) = unpack_q4k_scale_min(is, scales);
                    let (sc1, m1) = unpack_q4k_scale_min(is + 1, scales);
                    let d1 = d * (sc0 as f32);
                    let min1 = dmin * (m0 as f32);
                    let d2 = d * (sc1 as f32);
                    let min2 = dmin * (m1 as f32);

                    let q = &qs[chunk * 32..chunk * 32 + 32];
                    let y = &mut dst[chunk * 64..chunk * 64 + 64];
                    for l in 0..32 {
                        y[l] = d1 * ((q[l] & 0x0F) as f32) - min1;
                        y[l + 32] = d2 * ((q[l] >> 4) as f32) - min2;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (blk, dst) in src.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208];
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

                for n in 0..2 {
                    let ql_h = &ql[n * 64..n * 64 + 64];
                    let qh_h = &qh[n * 32..n * 32 + 32];
                    let sc_h = &scales[n * 8..n * 8 + 8];
                    let y = &mut dst[n * 128..n * 128 + 128];

                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = (((ql_h[l] & 0x0F) | ((qh_h[l] & 0x03) << 4)) as i8) - 32;
                        let q2 = (((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 0x03) << 4)) as i8) - 32;
                        let q3 = (((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 0x03) << 4)) as i8) - 32;
                        let q4 = (((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 0x03) << 4)) as i8) - 32;

                        y[l] = d * (sc_h[is] as i8 as f32) * (q1 as f32);
                        y[l + 32] = d * (sc_h[is + 2] as i8 as f32) * (q2 as f32);
                        y[l + 64] = d * (sc_h[is + 4] as i8 as f32) * (q3 as f32);
                        y[l + 96] = d * (sc_h[is + 6] as i8 as f32) * (q4 as f32);
                    }
                }
            }
        }
        GgmlType::Unsupported(code) => panic!("cannot dequantize unsupported ggml type {code}"),
    }
}

/// Scalar reference dot products used on platforms without SIMD and in unit parity tests.
pub mod scalar {
    use super::*;

    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let k = i * 4;
            s0 += a[k] * b[k];
            s1 += a[k + 1] * b[k + 1];
            s2 += a[k + 2] * b[k + 2];
            s3 += a[k + 3] * b[k + 3];
        }
        let mut sum = (s0 + s1) + (s2 + s3);
        for i in chunks * 4..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    pub fn dot_f16_f32(w_bytes: &[u8], x: &[f32]) -> f32 {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let chunks = x.len() / 4;
        for i in 0..chunks {
            let k = i * 4;
            let b = k * 2;
            s0 += f16_to_f32(u16::from_le_bytes([w_bytes[b], w_bytes[b + 1]])) * x[k];
            s1 += f16_to_f32(u16::from_le_bytes([w_bytes[b + 2], w_bytes[b + 3]])) * x[k + 1];
            s2 += f16_to_f32(u16::from_le_bytes([w_bytes[b + 4], w_bytes[b + 5]])) * x[k + 2];
            s3 += f16_to_f32(u16::from_le_bytes([w_bytes[b + 6], w_bytes[b + 7]])) * x[k + 3];
        }
        let mut sum = (s0 + s1) + (s2 + s3);
        for (k, &xv) in x.iter().enumerate().skip(chunks * 4) {
            let b = k * 2;
            sum += f16_to_f32(u16::from_le_bytes([w_bytes[b], w_bytes[b + 1]])) * xv;
        }
        sum
    }

    pub fn dot_bf16_f32(w_bytes: &[u8], x: &[f32]) -> f32 {
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let chunks = x.len() / 4;
        for i in 0..chunks {
            let k = i * 4;
            let b = k * 2;
            s0 += bf16_to_f32(u16::from_le_bytes([w_bytes[b], w_bytes[b + 1]])) * x[k];
            s1 += bf16_to_f32(u16::from_le_bytes([w_bytes[b + 2], w_bytes[b + 3]])) * x[k + 1];
            s2 += bf16_to_f32(u16::from_le_bytes([w_bytes[b + 4], w_bytes[b + 5]])) * x[k + 2];
            s3 += bf16_to_f32(u16::from_le_bytes([w_bytes[b + 6], w_bytes[b + 7]])) * x[k + 3];
        }
        let mut sum = (s0 + s1) + (s2 + s3);
        for (k, &xv) in x.iter().enumerate().skip(chunks * 4) {
            let b = k * 2;
            sum += bf16_to_f32(u16::from_le_bytes([w_bytes[b], w_bytes[b + 1]])) * xv;
        }
        sum
    }

    pub fn dot_q8_0_q8_0(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut sumf = 0.0f32;
        for b in 0..n_blocks {
            let wb = &w_bytes[b * 34..(b + 1) * 34];
            let xb = &x_bytes[b * x_stride..(b + 1) * x_stride];
            let d_w = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let d_x = f32::from_le_bytes([xb[0], xb[1], xb[2], xb[3]]);
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (wb[2 + j] as i8 as i32) * (xb[4 + j] as i8 as i32);
            }
            sumf += (d_w * d_x) * (isum as f32);
        }
        sumf
    }

    pub fn dot_q4_0_q8_0(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 32;
        let x_stride = std::mem::size_of::<ActQ8_0>();
        let mut sumf = 0.0f32;
        for b in 0..n_blocks {
            let wb = &w_bytes[b * 18..(b + 1) * 18];
            let xb = &x_bytes[b * x_stride..(b + 1) * x_stride];
            let d_w = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let d_x = f32::from_le_bytes([xb[0], xb[1], xb[2], xb[3]]);
            let mut isum = 0i32;
            for j in 0..16 {
                let byte = wb[2 + j];
                let lo = ((byte & 0x0F) as i32) - 8;
                let hi = ((byte >> 4) as i32) - 8;
                isum += lo * (xb[4 + j] as i8 as i32) + hi * (xb[4 + 16 + j] as i8 as i32);
            }
            sumf += (d_w * d_x) * (isum as f32);
        }
        sumf
    }

    pub fn dot_q4_k_q8_k(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 256;
        let x_stride = std::mem::size_of::<ActQ8K>();
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

                let q = &qs[chunk * 32..chunk * 32 + 32];
                let x_c = &x_qs[chunk * 64..chunk * 64 + 64];
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for l in 0..32 {
                    dot0 += ((q[l] & 0x0F) as i32) * (x_c[l] as i8 as i32);
                    dot1 += ((q[l] >> 4) as i32) * (x_c[l + 32] as i8 as i32);
                }

                let bs0 = i16::from_le_bytes([xb[260 + is * 4], xb[261 + is * 4]]) as i32
                    + i16::from_le_bytes([xb[262 + is * 4], xb[263 + is * 4]]) as i32;
                let bs1 = i16::from_le_bytes([xb[264 + is * 4], xb[265 + is * 4]]) as i32
                    + i16::from_le_bytes([xb[266 + is * 4], xb[267 + is * 4]]) as i32;

                sum_sc += (sc0 as i32) * dot0 + (sc1 as i32) * dot1;
                sum_min += (m0 as i32) * bs0 + (m1 as i32) * bs1;
            }
            let d_all = d_w * d_x;
            let dmin_all = dmin_w * d_x;
            sumf += d_all * (sum_sc as f32) - dmin_all * (sum_min as f32);
        }
        sumf
    }

    pub fn dot_q6_k_q8_k(w_bytes: &[u8], x_bytes: &[u8], cols: usize) -> f32 {
        let n_blocks = cols / 256;
        let x_stride = std::mem::size_of::<ActQ8K>();
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
                    let mut d1 = 0i32;
                    let mut d2 = 0i32;
                    let mut d3 = 0i32;
                    let mut d4 = 0i32;
                    for j in 0..16 {
                        let l = off + j;
                        let q1 = (((ql_h[l] & 0x0F) | ((qh_h[l] & 0x03) << 4)) as i8) - 32;
                        let q2 = (((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 0x03) << 4)) as i8) - 32;
                        let q3 = (((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 0x03) << 4)) as i8) - 32;
                        let q4 = (((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 0x03) << 4)) as i8) - 32;

                        d1 += (q1 as i32) * (x_h[l] as i8 as i32);
                        d2 += (q2 as i32) * (x_h[l + 32] as i8 as i32);
                        d3 += (q3 as i32) * (x_h[l + 64] as i8 as i32);
                        d4 += (q4 as i32) * (x_h[l + 96] as i8 as i32);
                    }
                    isum += (sc_h[is] as i8 as i32) * d1
                        + (sc_h[is + 2] as i8 as i32) * d2
                        + (sc_h[is + 4] as i8 as i32) * d3
                        + (sc_h[is + 6] as i8 as i32) * d4;
                }
            }
            sumf += (d_w * d_x) * (isum as f32);
        }
        sumf
    }
}
