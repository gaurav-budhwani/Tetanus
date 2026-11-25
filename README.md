# Tetanus

A **zero-dependency** local Transformer inference engine written in pure Rust (`[dependencies]` is empty in `Cargo.toml`).

Tetanus loads standard **GGUF v2/v3** open-weights models (Qwen 2 / 2.5 / 3, Llama 2 / 3) directly via OS `mmap`, implements custom BPE tokenization (`gpt2` byte-level & `llama` SentencePiece), block quantization (`Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `F16`, `BF16`, `F32`), quantized KV-caching (`F16` and `Q8_0`), a persistent cache-line-aligned thread pool, and hand-written SIMD kernels (`aarch64` NEON/`sdot`/`fcvtl` + `x86_64` AVX2/FMA).

---

## Key Engineering Choices

1. **Zero external crates (`Cargo.toml` `[dependencies]` empty)**
   - **OS `mmap` wrapper (`src/mmap.rs`):** Direct `extern "C"` `mmap`/`munmap`/`madvise` (`MADV_SEQUENTIAL` during header parse, `MADV_WILLNEED` before inference) plus `getrusage` Peak RSS telemetry. Weights stay backed by the OS page cache without duplicating gigabytes onto the Rust heap.
   - **Persistent thread pool (`src/pool.rs`):** Avoids spawning OS threads inside every layer matmul (~170 matmuls/token). Uses a two-phase wait (short atomic spin-wait between layers for sub-microsecond wakeup, `Condvar` park between prompts) with cache-line-padded atomics (`128` bytes on Apple Silicon, `64` bytes on x86_64) to prevent false sharing. Defaults to hardware Performance cores (`hw.perflevel0.physicalcpu` on macOS).
   - **IEEE-754 `f16`/`bf16` (`src/f16.rs`):** Register bit-manipulation instead of a 256 KB lookup table so `L1D` (64 KB on M4) isn't thrashed while streaming quantized block scales. For full `F16` matmuls on ARM64, uses baseline ARMv8 NEON `fcvtl`/`fcvtl2` vector conversion + 4-way `vfmaq_f32`.

2. **Zero-allocation decode loop (`src/arena.rs`, `tests/zero_alloc.rs`)**
   - All activation slabs (`x`, `xb`, `q`, `k`, `v`, `hb`, `hb2`, `logits`) and the on-the-fly quantized activation buffer (`ActQ8_0` / `ActQ8K`) are allocated once at initialization.
   - Verified with a dedicated `#[global_allocator]` counter binary (`tests/zero_alloc.rs`) proving **0 heap allocations** during multi-threaded token decode + sampling.

3. **Integer-domain quantized matmuls (`src/quant.rs`, `src/kernels.rs`)**
   - Activations are quantized once per matmul into `ActQ8_0` or `ActQ8K` (including precomputed 16-element `bsums` so `Q4_K` min terms require zero inner-loop activation summation).
   - Because Rust stable (`1.95.0`) still gates `stdarch_neon_dotprod`, `kernels.rs` detects `+dotprod` at runtime (`is_aarch64_feature_detected!("dotprod")`) and emits `sdot` via `core::arch::asm!`, falling back cleanly to baseline NEON widening (`vmull_s8`/`vpaddlq_s16`) or portable scalar.
   - Batched prefill (`DEFAULT_PREFILL_BATCH = 32`) iterates weight rows on the outer loop and computes 4 prompt tokens per loaded weight row so weights stay hot in L1/registers instead of re-streaming from DRAM.

4. **Head-major KV cache (`src/kv_cache.rs`) & Tiled online-softmax attention (`src/model.rs`)**
   - KV cache layout is `[n_layers, n_kv_heads, max_seq_len, head_dim]`, keeping sequential token scans contiguous in memory; supports both `F16` and `Q8_0` KV storage.
   - Attention processes keys/values in `TILE_KV = 64` blocks using online softmax on stack buffers, eliminating `O(seq_len^2)` attention score allocations even at long context lengths.

5. **Real GGUF architecture & tokenizer quirks handled (`src/gguf.rs`, `src/model.rs`, `src/tokenizer.rs`)**
   - Respects GGUF innermost-first tensor dimensions (`dims[0] = in_features`, `dims[1] = out_features`).
   - Supports `Q4_0` GGML split-nibble layout (elements `0..16` in low nibbles, `16..32` in high nibbles), optional QKV biases (Qwen 2/2.5), optional per-head Q/K RMSNorm and explicit `key_length` (Qwen 3), tied embeddings, and Interleaved (`llama`) vs. Split-Half (`qwen2`/`qwen3`) RoPE.
   - Pure-Rust BPE tokenizer handles both GPT-2 byte-level BPE (`merges` + `tokenizer.ggml.pre` digit splitting rules + streaming multi-byte UTF-8 reconstruction) and SentencePiece (`scores` + `<0xXX>` byte fallback).

---

## Building & Testing

```bash
cargo build --release
cargo test --release
cargo clippy --all-targets -- -D warnings
```

The test suite verifies:
- Full 16-bit `f16` roundtrip accuracy (`0..=65535`) and `bf16` conversion
- SIMD (`sdot` / NEON / `fcvtl`) vs. scalar vs. dequantized reference parity across `F16`, `Q8_0`, `Q4_0`, `Q4_K`, and `Q6_K`
- Incremental KV-cache decode vs. full-sequence prefill logit equivalence for both Llama and Qwen2 (with `F16` and `Q8_0` KV caches)
- GPT-2 byte-level BPE, SentencePiece BPE, special tokens, and split-token UTF-8 streaming
- Strict **0 heap allocations** during multi-threaded decode + top-k/top-p sampling (`tests/zero_alloc.rs`)

---

## Measured Apple M4 Kernel Benchmarks

Measured on **Apple M4** (4 P-cores + 6 E-cores, 16 GB unified memory, `rustc 1.95.0`, `aarch64-neon-sdot`) via `cargo run --release -- bench` on `[4096 x 4096]` matrices (GEMV = single-token decode, GEMM32 = 32-token batched prefill):

| Dtype | Matrix Size | 1T GEMV | 4T (P-cores) GEMV | 8T (P+E) GEMV | 4T GEMM32 | 8T GEMM32 |
|---|---:|---:|---:|---:|---:|---:|
| `Q4_0` | 9.00 MiB | 990.6 µs (9.5 GB/s) | 272.6 µs (34.6 GB/s) | 240.6 µs (39.2 GB/s) | 3.81 ms (281.7 GFLOP/s) | 3.42 ms (314.1 GFLOP/s) |
| `Q4_K` | 9.00 MiB | 848.2 µs (11.1 GB/s) | **198.1 µs** (47.6 GB/s) | 229.1 µs (41.2 GB/s) | 7.21 ms (148.9 GFLOP/s) | 6.17 ms (174.1 GFLOP/s) |
| `Q6_K` | 13.12 MiB | 764.6 µs (18.0 GB/s) | 296.5 µs (46.4 GB/s) | 214.5 µs (64.2 GB/s) | 6.06 ms (177.2 GFLOP/s) | 4.96 ms (216.6 GFLOP/s) |
| `Q8_0` | 17.00 MiB | 863.5 µs (20.6 GB/s) | 274.0 µs (65.1 GB/s) | 254.4 µs (70.1 GB/s) | 3.60 ms (298.1 GFLOP/s) | 3.28 ms (327.0 GFLOP/s) |
| `F16` | 32.00 MiB | 1591.4 µs (21.1 GB/s) | 524.0 µs (64.0 GB/s) | 446.9 µs (75.1 GB/s) | 12.22 ms (87.9 GFLOP/s) | 7.53 ms (142.6 GFLOP/s) |
| `F32` | 64.00 MiB | 1739.1 µs (38.6 GB/s) | 956.3 µs (70.2 GB/s) | 880.4 µs (76.2 GB/s) | 10.17 ms (105.6 GFLOP/s) | 8.56 ms (125.4 GFLOP/s) |

**Roofline takeaways observed directly from the numbers:**
- **Decode (`batch=1`) is memory-bandwidth bound:** `F32` (64 MiB) and `Q8_0` (17 MiB) already reach ~65–70 GB/s at 4 P-cores, and `Q4_K` peaks at 4 P-cores (`198.1 µs`, slowing down at 8 threads due to E-core straggler/cache-coherence overhead).
- **4-bit quantization cuts decode latency ~4.4x vs `F32`** (`198–240 µs` vs `880 µs`) primarily by shrinking bytes read from DRAM (`9 MiB` vs `64 MiB`).
- **Hardware `fcvtl` NEON conversion matters:** Switching `F16` dot products from scalar bit-manipulation (`4518.4 µs`, `7.4 GB/s` at 4T) to ARMv8 NEON `fcvtl`/`fcvtl2` + `vfmaq_f32` (`524.0 µs`, `64.0 GB/s` at 4T) sped up `F16` GEMV by **8.6x** and `F16` GEMM32 by **13.4x** (`100.76 ms -> 7.53 ms` at 8T).

---

## Running on Real Open-Weights Models & Comparing Against `llama.cpp` / PyTorch

1. **Inspect any GGUF model:**
   ```bash
   ./target/release/tetanus inspect Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
   ```

2. **Generate text with streaming output & telemetry:**
   ```bash
   ./target/release/tetanus run Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
     --chat \
     --prompt "Explain virtual memory page tables in 3 concise sentences." \
     --kv-quant q8 \
     --max-tokens 128
   ```

3. **Benchmark end-to-end scaling on a GGUF file:**
   ```bash
   ./target/release/tetanus bench --model Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --threads 1,2,4,8 --kv-quant f16
   ./target/release/tetanus bench --model Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --threads 1,2,4,8 --kv-quant q8
   ```

4. **Side-by-side comparison vs. `llama.cpp` (CPU) and PyTorch (CPU):**
   ```bash
   python3 scripts/bench_compare.py \
     --gguf Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
     --hf-model Qwen/Qwen2.5-0.5B-Instruct \
     --threads 4 \
     --max-tokens 64
   ```
   The harness reports **Prefill tok/s**, **Decode tok/s**, **Peak RSS (MiB)** via `/usr/bin/time -l`, and **Top-1 logit ID parity**.
