//! Dedicated zero-allocation verification binary.
//!
//! Kept in its own integration test target so no unrelated parallel test threads
//! can trigger background allocations while `COUNTING` is enabled.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tetanus::arena::DEFAULT_PREFILL_BATCH;
use tetanus::f16::f32_to_f16;
use tetanus::gguf::{GgmlType, GgufFile};
use tetanus::kv_cache::KvQuant;
use tetanus::model::Transformer;
use tetanus::pool::ThreadPool;
use tetanus::sampler::{Sampler, SamplerConfig};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn minimal_q4_gguf() -> Vec<u8> {
    let vocab = 32usize;
    let dim = 64usize;
    let hidden = 128usize;
    let n_heads = 4usize;
    let n_kv_heads = 2usize;
    let kv_dim = 32usize;

    let mut out = Vec::new();
    let mut kvs: Vec<(&str, u32, Vec<u8>)> = Vec::new();
    let push_str = |kvs: &mut Vec<(&str, u32, Vec<u8>)>, k: &'static str, v: &str| {
        let mut b = Vec::new();
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
        kvs.push((k, 8, b));
    };
    let push_u32 = |kvs: &mut Vec<(&str, u32, Vec<u8>)>, k: &'static str, v: u32| {
        kvs.push((k, 4, v.to_le_bytes().to_vec()));
    };

    push_str(&mut kvs, "general.architecture", "qwen2");
    push_u32(&mut kvs, "qwen2.embedding_length", dim as u32);
    push_u32(&mut kvs, "qwen2.feed_forward_length", hidden as u32);
    push_u32(&mut kvs, "qwen2.block_count", 2);
    push_u32(&mut kvs, "qwen2.attention.head_count", n_heads as u32);
    push_u32(&mut kvs, "qwen2.attention.head_count_kv", n_kv_heads as u32);
    push_u32(&mut kvs, "qwen2.context_length", 128);

    let q4_mat = |rows: usize, cols: usize| -> Vec<u8> {
        let blocks = (cols / 32) * rows;
        let mut data = vec![0u8; blocks * 18];
        let d = f32_to_f16(0.02).to_le_bytes();
        for blk in data.chunks_exact_mut(18) {
            blk[0..2].copy_from_slice(&d);
            blk[2..18].fill(0x88);
        }
        data
    };
    let f32_ones = |len: usize| -> Vec<u8> {
        let mut data = Vec::with_capacity(len * 4);
        for _ in 0..len {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        data
    };

    let mut tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)> = Vec::new();
    tensors.push(("token_embd.weight".into(), vec![dim, vocab], 2, q4_mat(vocab, dim)));
    tensors.push(("output_norm.weight".into(), vec![dim], 0, f32_ones(dim)));
    for l in 0..2 {
        tensors.push((format!("blk.{l}.attn_norm.weight"), vec![dim], 0, f32_ones(dim)));
        tensors.push((format!("blk.{l}.attn_q.weight"), vec![dim, dim], 2, q4_mat(dim, dim)));
        tensors.push((format!("blk.{l}.attn_k.weight"), vec![dim, kv_dim], 2, q4_mat(kv_dim, dim)));
        tensors.push((format!("blk.{l}.attn_v.weight"), vec![dim, kv_dim], 2, q4_mat(kv_dim, dim)));
        tensors.push((format!("blk.{l}.attn_output.weight"), vec![dim, dim], 2, q4_mat(dim, dim)));
        tensors.push((format!("blk.{l}.ffn_norm.weight"), vec![dim], 0, f32_ones(dim)));
        tensors.push((format!("blk.{l}.ffn_gate.weight"), vec![dim, hidden], 2, q4_mat(hidden, dim)));
        tensors.push((format!("blk.{l}.ffn_up.weight"), vec![dim, hidden], 2, q4_mat(hidden, dim)));
        tensors.push((format!("blk.{l}.ffn_down.weight"), vec![hidden, dim], 2, q4_mat(dim, hidden)));
    }

    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, ty, v) in kvs {
        out.extend_from_slice(&(k.len() as u64).to_le_bytes());
        out.extend_from_slice(k.as_bytes());
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&v);
    }

    let mut rel = 0usize;
    for (name, dims, code, data) in &tensors {
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
        out.extend_from_slice(&code.to_le_bytes());
        out.extend_from_slice(&(rel as u64).to_le_bytes());
        rel += (data.len() + 31) & !31;
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for (_, _, _, data) in tensors {
        out.extend_from_slice(&data);
        while out.len() % 32 != 0 {
            out.push(0);
        }
    }
    let _ = GgmlType::Q4_0;
    out
}

#[test]
fn zero_heap_allocations_during_multithreaded_decode() {
    let gguf = GgufFile::from_bytes(minimal_q4_gguf()).unwrap();
    let model = Transformer::from_gguf(gguf).unwrap();
    let pool = ThreadPool::new(2);
    let mut kv = model.new_kv_cache(64, KvQuant::Q8_0);
    let mut scratch = model.new_scratch(DEFAULT_PREFILL_BATCH);
    let mut sampler = Sampler::new(
        model.config.vocab_size,
        SamplerConfig {
            temperature: 0.8,
            top_k: 16,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_window: 16,
            seed: 42,
        },
    );

    let prompt = [1u32, 2, 3, 4];
    let logits = model.prefill(&prompt, 0, &pool, &mut kv, &mut scratch);
    let mut history = Vec::with_capacity(64);
    history.extend_from_slice(&prompt);
    let mut tok = sampler.sample(logits, &history);

    // Warm up one decode step so worker threads are live and spinning.
    let logits = model.decode_token(tok, prompt.len(), &pool, &mut kv, &mut scratch);
    tok = sampler.sample(logits, &history);
    history.push(tok);

    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);

    for step in 1..16 {
        let pos = prompt.len() + step;
        let logits = model.decode_token(tok, pos, &pool, &mut kv, &mut scratch);
        tok = sampler.sample(logits, &history);
        history.push(tok);
    }

    COUNTING.store(false, Ordering::SeqCst);
    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    assert_eq!(
        allocs, 0,
        "decode + sampling must perform 0 heap allocations, observed {allocs} ({bytes} bytes)"
    );
}
