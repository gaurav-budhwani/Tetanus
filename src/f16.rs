//! Bit-level IEEE-754 half-precision (`f16`) and brain-float (`bf16`) conversions.
//!
//! We convert via pure register bit-manipulation instead of a 256 KB lookup table
//! because a 256 KB LUT thrashes Apple Silicon / x86 L1 data caches (64 KB on M4)
//! when streaming quantized weight scales during decode.

/// Converts an IEEE-754 binary16 bit pattern to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16: normalize by shifting the highest set mantissa bit into bit 10.
        // Because `mant` is nonzero and < 1024, `leading_zeros()` is in 22..=31.
        let shift = mant.leading_zeros() - 21;
        let norm_mant = (mant << shift) & 0x03FF;
        let exp_f32 = 113 - shift;
        return f32::from_bits(sign | (exp_f32 << 23) | (norm_mant << 13));
    }

    if exp == 31 {
        // Inf (mant == 0) or NaN (mant != 0).
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 13));
    }

    // Rebias exponent from f16 (+15) to f32 (+127): +112.
    f32::from_bits(sign | (((exp as u32) + 112) << 23) | (mant << 13))
}

/// Converts `f32` to IEEE-754 binary16 using round-to-nearest, ties-to-even.
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Preserve NaN payload (ensure quiet/non-zero mantissa so NaN doesn't turn into Inf).
        if mant != 0 {
            let payload = ((mant >> 13) as u16).max(1);
            return sign | 0x7C00 | payload;
        }
        return sign | 0x7C00;
    }

    let e16 = exp - 112;
    if e16 >= 31 {
        return sign | 0x7C00;
    }

    if e16 <= 0 {
        if e16 < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e16) as u32;
        let q = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round = u32::from(rem > half || (rem == half && (q & 1) != 0));
        return sign | ((q + round) as u16);
    }

    let q = mant >> 13;
    let rem = mant & 0x1FFF;
    let half = 0x1000;
    let round = u32::from(rem > half || (rem == half && (q & 1) != 0));

    // If `q + round == 0x0400`, adding into `e16 << 10` naturally carries into the exponent
    // (and saturates cleanly to Inf 0x7C00 when e16 == 30).
    sign | ((((e16 as u32) << 10) + q + round) as u16)
}

/// Converts a bfloat16 bit pattern to `f32` (top 16 bits of IEEE-754 f32).
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_special_and_normal_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());

        // Smallest positive f16 subnormal = 2^-24
        assert_eq!(f16_to_f32(0x0001), (-24.0f32).exp2());
        // Largest f16 subnormal = 1023 * 2^-24
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * (-24.0f32).exp2());

        for bits in 0u16..=u16::MAX {
            let v = f16_to_f32(bits);
            if v.is_nan() {
                assert!(f16_to_f32(f32_to_f16(v)).is_nan());
            } else {
                assert_eq!(f32_to_f16(v), bits, "mismatch at bits=0x{bits:04x}");
            }
        }
    }

    #[test]
    fn bf16_basic() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x4049), f32::from_bits(0x4049_0000));
    }
}
