//! Zero-allocation token sampler with greedy fast-path, repetition penalty,
//! O(V) partial-selection top-k, and top-p nucleus filtering.

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.0,
            repeat_window: 64,
            seed: 42,
        }
    }
}

pub struct Sampler {
    cfg: SamplerConfig,
    rng_state: u64,
    /// Pre-allocated scratch buffer sized once to `vocab_size` so sampling never allocates.
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(vocab_size: usize, cfg: SamplerConfig) -> Self {
        let seed = if cfg.seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            cfg.seed
        };
        Self {
            cfg,
            rng_state: seed,
            candidates: Vec::with_capacity(vocab_size),
        }
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        // Apply repetition penalty in-place over recent token window.
        if (self.cfg.repeat_penalty - 1.0).abs() > 1e-6 && !history.is_empty() {
            let start = history.len().saturating_sub(self.cfg.repeat_window);
            let penalty = self.cfg.repeat_penalty;
            for &tok in &history[start..] {
                if let Some(l) = logits.get_mut(tok as usize) {
                    if *l > 0.0 {
                        *l /= penalty;
                    } else {
                        *l *= penalty;
                    }
                }
            }
        }

        // Greedy decoding: single linear argmax pass with zero sorting or RNG.
        if self.cfg.temperature <= 0.0 {
            let mut best_id = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in logits.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_id = i as u32;
                }
            }
            return best_id;
        }

        // Reuse pre-allocated candidate buffer without heap reallocation.
        self.candidates.clear();
        let inv_temp = 1.0 / self.cfg.temperature;
        for (i, &val) in logits.iter().enumerate() {
            self.candidates.push((i as u32, val * inv_temp));
        }

        // O(V) top-k partition before sorting only the top-k survivors.
        let k = if self.cfg.top_k > 0 && self.cfg.top_k < self.candidates.len() {
            let pivot = self.cfg.top_k;
            self.candidates
                .select_nth_unstable_by(pivot, |a, b| b.1.total_cmp(&a.1));
            pivot
        } else {
            self.candidates.len()
        };

        let active = &mut self.candidates[..k];
        active.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max_logit = active[0].1;
        let mut sum_exp = 0.0f32;
        for item in active.iter_mut() {
            let p = (item.1 - max_logit).exp();
            item.1 = p;
            sum_exp += p;
        }

        // Top-p nucleus truncation.
        let mut cut = active.len();
        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let target = self.cfg.top_p * sum_exp;
            let mut cum = 0.0f32;
            for (i, item) in active.iter().enumerate() {
                cum += item.1;
                if cum >= target {
                    cut = i + 1;
                    sum_exp = cum;
                    break;
                }
            }
        }

        let r = next_f32(&mut self.rng_state) * sum_exp;
        let mut acc = 0.0f32;
        for &(id, prob) in &active[..cut] {
            acc += prob;
            if acc >= r {
                return id;
            }
        }
        active[cut - 1].0
    }
}

/// xorshift64* PRNG (deterministic given seed, zero external dependencies).
#[inline]
fn next_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
    (bits as f32) * (1.0 / ((1u32 << 24) as f32))
}
