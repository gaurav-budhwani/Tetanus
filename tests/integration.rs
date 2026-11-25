//! Integration tests covering:
//!   1. GGUF v3 serialization/deserialization and malformed-file guards.
//!   2. SIMD (`sdot` / NEON / AVX2) vs scalar vs dequantized reference parity
//!      across `F32`, `F16`, `Q8_0`, `Q4_0`, `Q4_K`, and `Q6_K`.
//!   3. KV-cache vs full-sequence recomputation equivalence for both Llama (interleaved RoPE)
//!      and Qwen2 (split-half RoPE + QKV biases), tested with both `F16` and `Q8_0` KV caches.
//!   4. GPT-2 byte-level BPE, SentencePiece BPE, special tokens, and streaming UTF-8 decoding.

use tetanus::arena::DEFAULT_PREFILL_BATCH;
use tetanus::f16::f32_to_f16;
use tetanus::gguf::{GgmlType, GgufFile};
use tetanus::kernels::{dot_weight_row, SimdBackend};
use tetanus::kv_cache::KvQuant;
use tetanus::model::Transformer;
use tetanus::pool::ThreadPool;
use tetanus::quant::{dequantize_row, quantize_row_q8_0, quantize_row_q8_k, scalar, ActQ8K, ActQ8_0};
use tetanus::tokenizer::{Tokenizer, Utf8Stream};

struct GgufBuilder {
    kvs: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl GgufBuilder {
    fn new() -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn kv_u32(&mut self, key: &str, val: u32) {
        self.kvs.push((key.to_owned(), 4, val.to_le_bytes().to_vec()));
    }

    fn kv_f32(&mut self, key: &str, val: f32) {
        self.kvs.push((key.to_owned(), 6, val.to_le_bytes().to_vec()));
    }

    fn kv_bool(&mut self, key: &str, val: bool) {
        self.kvs.push((key.to_owned(), 7, vec![u8::from(val)]));
    }

    fn kv_str(&mut self, key: &str, val: &str) {
        let mut buf = Vec::with_capacity(8 + val.len());
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        self.kvs.push((key.to_owned(), 8, buf));
    }

    fn kv_str_array(&mut self, key: &str, vals: &[&str]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for s in vals {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        self.kvs.push((key.to_owned(), 9, buf));
    }

    fn kv_f32_array(&mut self, key: &str, vals: &[f32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.kvs.push((key.to_owned(), 9, buf));
    }

    fn kv_u32_array(&mut self, key: &str, vals: &[u32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.kvs.push((key.to_owned(), 9, buf));
    }

    fn add_tensor(&mut self, name: &str, dims: &[usize], dtype: GgmlType, data: Vec<u8>) {
        let code = match dtype {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q6K => 14,
            GgmlType::BF16 => 30,
            GgmlType::Unsupported(c) => c,
        };
        self.tensors.push((
            name.to_owned(),
            dims.iter().map(|&d| d as u64).collect(),
            code,
            data,
        ));
    }

    fn build(self) -> Vec<u8> {
        let align = 32usize;
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());

        for (k, ty, val_bytes) in &self.kvs {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(val_bytes);
        }

        let mut rel_offset = 0usize;
        for (name, dims, code, data) in &self.tensors {
            out.extend_from_slice(&(name.len() as u64).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&code.to_le_bytes());
            out.extend_from_slice(&(rel_offset as u64).to_le_bytes());
            rel_offset += (data.len() + align - 1) & !(align - 1);
        }

        while out.len() % align != 0 {
            out.push(0);
        }
        for (_, _, _, data) in self.tensors {
            out.extend_from_slice(&data);
            while out.len() % align != 0 {
                out.push(0);
            }
        }
        out
    }
}

fn deterministic_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
            ((u as f32) / ((1u32 << 24) as f32) - 0.5) * 0.4
        })
        .collect()
}

fn encode_f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_q8_0_bytes(vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len() % 32, 0);
    let mut out = Vec::with_capacity((vals.len() / 32) * 34);
    for chunk in vals.chunks_exact(32) {
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for &v in chunk {
            out.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

fn encode_q4_0_bytes(vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len() % 32, 0);
    let mut out = Vec::with_capacity((vals.len() / 32) * 18);
    for chunk in vals.chunks_exact(32) {
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let lo = ((chunk[j] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            let hi = ((chunk[j + 16] * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            out.push((lo & 0x0F) | ((hi & 0x0F) << 4));
        }
    }
    out
}

fn build_test_model_gguf(arch: &str, with_qkv_bias: bool, quant_weights: GgmlType) -> Vec<u8> {
    let vocab = 64usize;
    let dim = 64usize;
    let hidden = 128usize;
    let n_layers = 2usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let head_dim = dim / n_heads; // 16
    let kv_dim = n_kv_heads * head_dim; // 32

    let mut b = GgufBuilder::new();
    b.kv_str("general.architecture", arch);
    b.kv_u32(&format!("{arch}.embedding_length"), dim as u32);
    b.kv_u32(&format!("{arch}.feed_forward_length"), hidden as u32);
    b.kv_u32(&format!("{arch}.block_count"), n_layers as u32);
    b.kv_u32(&format!("{arch}.attention.head_count"), n_heads as u32);
    b.kv_u32(&format!("{arch}.attention.head_count_kv"), n_kv_heads as u32);
    b.kv_u32(&format!("{arch}.context_length"), 128);
    b.kv_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-5);
    b.kv_f32(&format!("{arch}.rope.freq_base"), 10_000.0);

    // Minimal SentencePiece-style vocab for tests.
    let tok_names: Vec<String> = (0..vocab)
        .map(|i| match i {
            0 => "<unk>".into(),
            1 => "<s>".into(),
            2 => "</s>".into(),
            _ => format!("tok{i}"),
        })
        .collect();
    let tok_refs: Vec<&str> = tok_names.iter().map(String::as_str).collect();
    b.kv_str("tokenizer.ggml.model", "llama");
    b.kv_str_array("tokenizer.ggml.tokens", &tok_refs);
    b.kv_u32("tokenizer.ggml.bos_token_id", 1);
    b.kv_u32("tokenizer.ggml.eos_token_id", 2);
    b.kv_bool("tokenizer.ggml.add_bos_token", true);

    let encode_mat = |vals: &[f32]| match quant_weights {
        GgmlType::Q4_0 => encode_q4_0_bytes(vals),
        GgmlType::Q8_0 => encode_q8_0_bytes(vals),
        _ => encode_f32_bytes(vals),
    };

    b.add_tensor(
        "token_embd.weight",
        &[dim, vocab],
        quant_weights,
        encode_mat(&deterministic_f32(dim * vocab, 101)),
    );
    b.add_tensor(
        "output_norm.weight",
        &[dim],
        GgmlType::F32,
        encode_f32_bytes(&vec![1.0; dim]),
    );
    b.add_tensor(
        "output.weight",
        &[dim, vocab],
        quant_weights,
        encode_mat(&deterministic_f32(dim * vocab, 202)),
    );

    for l in 0..n_layers {
        let seed_base = (l as u64 + 1) * 1000;
        b.add_tensor(
            &format!("blk.{l}.attn_norm.weight"),
            &[dim],
            GgmlType::F32,
            encode_f32_bytes(&vec![1.0; dim]),
        );
        b.add_tensor(
            &format!("blk.{l}.attn_q.weight"),
            &[dim, dim],
            quant_weights,
            encode_mat(&deterministic_f32(dim * dim, seed_base + 1)),
        );
        b.add_tensor(
            &format!("blk.{l}.attn_k.weight"),
            &[dim, kv_dim],
            quant_weights,
            encode_mat(&deterministic_f32(dim * kv_dim, seed_base + 2)),
        );
        b.add_tensor(
            &format!("blk.{l}.attn_v.weight"),
            &[dim, kv_dim],
            quant_weights,
            encode_mat(&deterministic_f32(dim * kv_dim, seed_base + 3)),
        );
        if with_qkv_bias {
            b.add_tensor(
                &format!("blk.{l}.attn_q.bias"),
                &[dim],
                GgmlType::F32,
                encode_f32_bytes(&deterministic_f32(dim, seed_base + 4)),
            );
            b.add_tensor(
                &format!("blk.{l}.attn_k.bias"),
                &[kv_dim],
                GgmlType::F32,
                encode_f32_bytes(&deterministic_f32(kv_dim, seed_base + 5)),
            );
            b.add_tensor(
                &format!("blk.{l}.attn_v.bias"),
                &[kv_dim],
                GgmlType::F32,
                encode_f32_bytes(&deterministic_f32(kv_dim, seed_base + 6)),
            );
        }
        b.add_tensor(
            &format!("blk.{l}.attn_output.weight"),
            &[dim, dim],
            quant_weights,
            encode_mat(&deterministic_f32(dim * dim, seed_base + 7)),
        );
        b.add_tensor(
            &format!("blk.{l}.ffn_norm.weight"),
            &[dim],
            GgmlType::F32,
            encode_f32_bytes(&vec![1.0; dim]),
        );
        b.add_tensor(
            &format!("blk.{l}.ffn_gate.weight"),
            &[dim, hidden],
            quant_weights,
            encode_mat(&deterministic_f32(dim * hidden, seed_base + 8)),
        );
        b.add_tensor(
            &format!("blk.{l}.ffn_up.weight"),
            &[dim, hidden],
            quant_weights,
            encode_mat(&deterministic_f32(dim * hidden, seed_base + 9)),
        );
        b.add_tensor(
            &format!("blk.{l}.ffn_down.weight"),
            &[hidden, dim],
            quant_weights,
            encode_mat(&deterministic_f32(hidden * dim, seed_base + 10)),
        );
    }

    b.build()
}

#[test]
fn gguf_rejects_bad_magic_and_truncated_tensor() {
    assert!(GgufFile::from_bytes(b"NOPE1234".to_vec()).is_err());

    let mut valid = build_test_model_gguf("llama", false, GgmlType::F32);
    let parsed = GgufFile::from_bytes(valid.clone()).expect("valid GGUF should parse");
    assert_eq!(parsed.arch(), Some("llama"));

    valid.truncate(valid.len() - 64);
    assert!(GgufFile::from_bytes(valid).is_err());
}

#[test]
fn simd_kernels_match_scalar_and_dequant_reference() {
    let cols = 512usize;
    let x = deterministic_f32(cols, 777);
    let w_src = deterministic_f32(cols, 888);
    let backend = SimdBackend::detect();

    let mut x_q8_0 = vec![0u8; (cols / 32) * std::mem::size_of::<ActQ8_0>()];
    quantize_row_q8_0(&x, &mut x_q8_0);
    let mut x_q8_k = vec![0u8; (cols / 256) * std::mem::size_of::<ActQ8K>()];
    quantize_row_q8_k(&x, &mut x_q8_k);

    // 0. F16: SIMD fcvtl vs scalar
    let w_f16_bytes: Vec<u8> = w_src
        .iter()
        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
        .collect();
    let sc_f16 = scalar::dot_f16_f32(&w_f16_bytes, &x);
    let simd_f16 = dot_weight_row(backend, GgmlType::F16, &w_f16_bytes, &x, &[], cols);
    assert!((sc_f16 - simd_f16).abs() < 1e-4, "F16 scalar={sc_f16} simd={simd_f16}");

    // 1. Q8_0: SIMD vs scalar vs dequantize_row
    let w_q8 = encode_q8_0_bytes(&w_src);
    let mut w_q8_deq = vec![0.0f32; cols];
    dequantize_row(GgmlType::Q8_0, &w_q8, &mut w_q8_deq);
    let ref_q8 = scalar::dot_f32(&w_q8_deq, &x);
    let sc_q8 = scalar::dot_q8_0_q8_0(&w_q8, &x_q8_0, cols);
    let simd_q8 = dot_weight_row(backend, GgmlType::Q8_0, &w_q8, &x, &x_q8_0, cols);
    assert!((sc_q8 - simd_q8).abs() < 1e-4, "Q8_0 scalar={sc_q8} simd={simd_q8}");
    assert!((ref_q8 - simd_q8).abs() < 0.05, "Q8_0 ref={ref_q8} simd={simd_q8}");

    // 2. Q4_0: SIMD vs scalar vs dequantize_row
    let w_q4 = encode_q4_0_bytes(&w_src);
    let mut w_q4_deq = vec![0.0f32; cols];
    dequantize_row(GgmlType::Q4_0, &w_q4, &mut w_q4_deq);
    let ref_q4 = scalar::dot_f32(&w_q4_deq, &x);
    let sc_q4 = scalar::dot_q4_0_q8_0(&w_q4, &x_q8_0, cols);
    let simd_q4 = dot_weight_row(backend, GgmlType::Q4_0, &w_q4, &x, &x_q8_0, cols);
    assert!((sc_q4 - simd_q4).abs() < 1e-4, "Q4_0 scalar={sc_q4} simd={simd_q4}");
    assert!((ref_q4 - simd_q4).abs() < 0.05, "Q4_0 ref={ref_q4} simd={simd_q4}");

    // 3. Q4_K: SIMD vs scalar vs dequantize_row
    let n_k = cols / 256;
    let mut w_q4k = vec![0u8; n_k * 144];
    for blk in w_q4k.chunks_exact_mut(144) {
        blk[0..2].copy_from_slice(&f32_to_f16(0.03).to_le_bytes());
        blk[2..4].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
        for (i, b) in blk[4..16].iter_mut().enumerate() {
            *b = ((i as u8 * 7 + 3) & 0x3F) | 0x40;
        }
        for (i, b) in blk[16..144].iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
    }
    let mut w_q4k_deq = vec![0.0f32; cols];
    dequantize_row(GgmlType::Q4K, &w_q4k, &mut w_q4k_deq);
    let ref_q4k = scalar::dot_f32(&w_q4k_deq, &x);
    let sc_q4k = scalar::dot_q4_k_q8_k(&w_q4k, &x_q8_k, cols);
    let simd_q4k = dot_weight_row(backend, GgmlType::Q4K, &w_q4k, &x, &x_q8_k, cols);
    assert!((sc_q4k - simd_q4k).abs() < 1e-3, "Q4K scalar={sc_q4k} simd={simd_q4k}");
    assert!((ref_q4k - simd_q4k).abs() < 0.1, "Q4K ref={ref_q4k} simd={simd_q4k}");

    // 4. Q6_K: SIMD vs scalar vs dequantize_row
    let mut w_q6k = vec![0u8; n_k * 210];
    for blk in w_q6k.chunks_exact_mut(210) {
        for (i, b) in blk[0..128].iter_mut().enumerate() {
            *b = ((i * 17 + 11) & 0xFF) as u8;
        }
        for (i, b) in blk[128..192].iter_mut().enumerate() {
            *b = ((i * 29 + 7) & 0xFF) as u8;
        }
        for (i, b) in blk[192..208].iter_mut().enumerate() {
            *b = ((i as i8) - 8) as u8;
        }
        blk[208..210].copy_from_slice(&f32_to_f16(0.025).to_le_bytes());
    }
    let mut w_q6k_deq = vec![0.0f32; cols];
    dequantize_row(GgmlType::Q6K, &w_q6k, &mut w_q6k_deq);
    let ref_q6k = scalar::dot_f32(&w_q6k_deq, &x);
    let sc_q6k = scalar::dot_q6_k_q8_k(&w_q6k, &x_q8_k, cols);
    let simd_q6k = dot_weight_row(backend, GgmlType::Q6K, &w_q6k, &x, &x_q8_k, cols);
    assert!((sc_q6k - simd_q6k).abs() < 1e-3, "Q6K scalar={sc_q6k} simd={simd_q6k}");
    assert!((ref_q6k - simd_q6k).abs() < 0.1, "Q6K ref={ref_q6k} simd={simd_q6k}");
}

#[test]
fn kv_cache_matches_full_prefill_for_llama_and_qwen2() {
    for (arch, bias, w_dtype, kv_q) in [
        ("llama", false, GgmlType::F32, KvQuant::F16),
        ("qwen2", true, GgmlType::Q4_0, KvQuant::F16),
        ("qwen2", true, GgmlType::Q8_0, KvQuant::Q8_0),
    ] {
        let bytes = build_test_model_gguf(arch, bias, w_dtype);
        let gguf = GgufFile::from_bytes(bytes).unwrap();
        let model = Transformer::from_gguf(gguf).unwrap();
        let pool = ThreadPool::new(2);

        let seq = [1u32, 7, 13, 21, 5, 42, 19, 3, 11, 29];

        // Path A: full prefill over all 10 tokens.
        let mut kv_full = model.new_kv_cache(64, kv_q);
        let mut s_full = model.new_scratch(DEFAULT_PREFILL_BATCH);
        let logits_full = model.prefill(&seq, 0, &pool, &mut kv_full, &mut s_full).to_vec();

        // Path B: prefill first 5 tokens, then step-by-step decode_token for remaining 5.
        let mut kv_inc = model.new_kv_cache(64, kv_q);
        let mut s_inc = model.new_scratch(DEFAULT_PREFILL_BATCH);
        let _ = model.prefill(&seq[..5], 0, &pool, &mut kv_inc, &mut s_inc);
        let mut last_logits = Vec::new();
        for (pos, &tok_id) in seq.iter().enumerate().skip(5) {
            last_logits = model
                .decode_token(tok_id, pos, &pool, &mut kv_inc, &mut s_inc)
                .to_vec();
        }

        let max_diff = logits_full
            .iter()
            .zip(last_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "arch={arch} dtype={:?} kv={:?} max logit diff={max_diff}",
            w_dtype,
            kv_q
        );
    }
}

#[test]
fn tokenizer_gpt2_bpe_special_tokens_and_streaming_utf8() {
    let mut b = GgufBuilder::new();
    b.kv_str("general.architecture", "qwen2");
    b.kv_str("tokenizer.ggml.model", "gpt2");
    b.kv_str("tokenizer.ggml.pre", "qwen2");

    // Tokens include single-byte unicode proxies (`Ġ` for space, `ð`, `Ł`, `ļ`, `Ģ` for UTF-8 🚀),
    // merged words (`hi`, `Ġworld`), and special token `<|im_start|>`.
    let tokens = [
        "<|endoftext|>", // 0
        "<|im_start|>",  // 1
        "h",             // 2
        "i",             // 3
        "hi",            // 4
        "\u{0120}",      // 5 (' ' in GPT-2 byte table)
        "w",             // 6
        "o",             // 7
        "r",             // 8
        "l",             // 9
        "d",             // 10
        "\u{0120}w",     // 11
        "\u{0120}world", // 12 (shortcut direct match test via merges)
        "\u{00F0}",      // 13 (0xF0 byte of 🚀 U+1F680: F0 9F 9A 80)
        "\u{0141}",      // 14 (0x9F -> U+0141)
        "\u{013C}",      // 15 (0x9A -> U+013C)
        "\u{0122}",      // 16 (0x80 -> U+0122)
    ];
    let token_types = [3u32, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    b.kv_str_array("tokenizer.ggml.tokens", &tokens);
    b.kv_u32_array("tokenizer.ggml.token_type", &token_types);
    b.kv_str_array("tokenizer.ggml.merges", &["h i"]);
    b.kv_u32("tokenizer.ggml.eos_token_id", 0);

    let gguf = GgufFile::from_bytes(b.build()).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();

    let encoded = tok.encode("<|im_start|>hi", false);
    assert_eq!(encoded, vec![1, 4]);
    assert_eq!(tok.decode(&encoded), "<|im_start|>hi");

    // Verify multi-byte UTF-8 emoji split across 4 tokens streams cleanly without  replacement.
    let mut stream = Utf8Stream::new();
    assert_eq!(stream.push_token(&tok, 13), "");
    assert_eq!(stream.push_token(&tok, 14), "");
    assert_eq!(stream.push_token(&tok, 15), "");
    assert_eq!(stream.push_token(&tok, 16), "🚀");
}

#[test]
fn tokenizer_sentencepiece_roundtrip() {
    let mut b = GgufBuilder::new();
    b.kv_str("general.architecture", "llama");
    b.kv_str("tokenizer.ggml.model", "llama");
    let tokens = ["<unk>", "<s>", "</s>", "\u{2581}", "h", "i", "\u{2581}h", "\u{2581}hi"];
    let scores = [0.0f32, 0.0, 0.0, -1.0, -2.0, -2.0, -0.5, -0.1];
    b.kv_str_array("tokenizer.ggml.tokens", &tokens);
    b.kv_f32_array("tokenizer.ggml.scores", &scores);
    b.kv_u32("tokenizer.ggml.bos_token_id", 1);
    b.kv_u32("tokenizer.ggml.eos_token_id", 2);
    b.kv_bool("tokenizer.ggml.add_bos_token", true);

    let gguf = GgufFile::from_bytes(b.build()).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    let ids = tok.encode("hi", true);
    assert_eq!(ids, vec![1, 7]);
    assert_eq!(tok.decode(&[7]), " hi");
}
