//! CLI entry point for Tetanus (`inspect`, `run`, `bench`).

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::process;
use std::time::Instant;

use tetanus::arena::DEFAULT_PREFILL_BATCH;
use tetanus::f16::f32_to_f16;
use tetanus::gguf::{GgmlType, GgufFile};
use tetanus::kernels::{matmul_batch, SimdBackend};
use tetanus::kv_cache::KvQuant;
use tetanus::mmap::peak_rss_bytes;
use tetanus::model::{ModelConfig, Transformer};
use tetanus::pool::{default_perf_threads, ThreadPool};
use tetanus::sampler::{Sampler, SamplerConfig};
use tetanus::tokenizer::{Tokenizer, Utf8Stream};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_usage();
        return;
    }

    let res = match args[0].as_str() {
        "inspect" => cmd_inspect(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        other => Err(format!("unknown subcommand '{other}' (expected inspect, run, or bench)")),
    };

    if let Err(err) = res {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn print_usage() {
    println!(
        "\
tetanus — zero-dependency Rust GGUF inference engine

USAGE:
    tetanus inspect <model.gguf>
    tetanus run <model.gguf> --prompt <text> [options]
    tetanus bench [--model <model.gguf>] [options]

RUN OPTIONS:
    --prompt <text>         Input prompt (required; or use --chat)
    --chat                  Wrap prompt in model chat template (Qwen / Llama-3 / Llama-2)
    --max-tokens <N>        Maximum tokens to generate (default: 128)
    --ctx <N>               KV cache capacity (default: min(model_ctx, 4096))
    --threads <N>           Worker threads (default: hardware P-cores)
    --kv-quant <f16|q8>     KV cache storage precision (default: f16)
    --temp <F>              Sampling temperature, 0 = greedy (default: 0.0)
    --top-k <N>             Top-k filter (default: 40)
    --top-p <F>             Top-p nucleus filter (default: 0.9)
    --repeat-penalty <F>    Repetition penalty (default: 1.0)
    --seed <N>              Deterministic RNG seed (default: 42)
    --dump-logits           Print top-5 logit IDs/values after prefill (parity check)

BENCH OPTIONS:
    --model <model.gguf>    Benchmark a real GGUF model (runs synthetic kernel bench if omitted)
    --threads <list>        Comma-separated thread counts (default: 1,2,4)
    --prompt-len <N>        Prompt tokens for prefill benchmark (default: 64)
    --gen-len <N>           Decode steps for generation benchmark (default: 64)
    --kv-quant <f16|q8>             KV cache precision (default: f16)
    --dtype <q4_0|q8_0|q4_k|q6_k|f16|f32>
                                    Synthetic matrix dtype when --model is omitted (default: q4_0)"
    );
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .ok_or_else(|| "missing <model.gguf> path".to_string())?;
    let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
    let cfg = ModelConfig::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;

    let mut by_dtype: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut total_params = 0usize;
    let mut total_tensor_bytes = 0usize;
    for name in &gguf.tensor_order {
        let info = &gguf.tensors[name];
        total_params += info.n_elements;
        total_tensor_bytes += info.byte_size;
        let entry = by_dtype.entry(info.dtype.name()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += info.byte_size;
    }

    println!("File:            {path}");
    println!("GGUF version:    v{} (alignment {} B)", gguf.version, gguf.alignment);
    println!("File size:       {:.2} MiB", gguf.file_size() as f64 / (1024.0 * 1024.0));
    println!("SIMD backend:    {}", SimdBackend::detect().name());
    println!("Default threads: {}", default_perf_threads());
    println!("Architecture:    {} (RoPE: {:?})", cfg.arch, cfg.rope_style);
    println!(
        "Dimensions:      layers={}, dim={}, hidden={}, heads={}/{} (kv), head_dim={}",
        cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    );
    println!(
        "Vocab / Ctx:     vocab={}, max_ctx={}, tied_head={}",
        cfg.vocab_size, cfg.max_seq_len, cfg.tied_embeddings
    );
    println!(
        "Tokenizer:       {:?} (bos={:?}, eos={})",
        tok.kind(),
        tok.bos_id,
        tok.eos_id
    );
    println!(
        "Parameters:      {:.2} M ({:.2} MiB in tensor payload)",
        total_params as f64 / 1e6,
        total_tensor_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Tensor dtypes:");
    for (dtype, (count, bytes)) in by_dtype {
        println!(
            "  {dtype:<10} {count:>4} tensors   {:>8.2} MiB",
            bytes as f64 / (1024.0 * 1024.0)
        );
    }

    let kv_f16_4k = 2 * cfg.n_layers * cfg.n_kv_heads * 4096 * cfg.head_dim * 2;
    let kv_q8_4k = if cfg.head_dim % 32 == 0 {
        2 * cfg.n_layers * cfg.n_kv_heads * 4096 * (cfg.head_dim / 32) * 34
    } else {
        kv_f16_4k
    };
    println!(
        "KV @ 4096 ctx:   F16 = {:.2} MiB | Q8_0 = {:.2} MiB",
        kv_f16_4k as f64 / (1024.0 * 1024.0),
        kv_q8_4k as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let path = args
        .first()
        .ok_or_else(|| "missing <model.gguf> path".to_string())?;
    let mut prompt: Option<String> = None;
    let mut chat_mode = false;
    let mut max_tokens: usize = 128;
    let mut ctx_len: usize = 4096;
    let mut threads: usize = default_perf_threads();
    let mut kv_quant = KvQuant::F16;
    let mut sampler_cfg = SamplerConfig::default();
    let mut dump_logits = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                i += 1;
                prompt = Some(args.get(i).ok_or("--prompt requires a value")?.clone());
            }
            "--chat" => chat_mode = true,
            "--max-tokens" => {
                i += 1;
                max_tokens = parse_usize(args.get(i), "--max-tokens")?;
            }
            "--ctx" => {
                i += 1;
                ctx_len = parse_usize(args.get(i), "--ctx")?;
            }
            "--threads" => {
                i += 1;
                threads = parse_usize(args.get(i), "--threads")?;
            }
            "--kv-quant" => {
                i += 1;
                kv_quant = parse_kv_quant(args.get(i))?;
            }
            "--temp" => {
                i += 1;
                sampler_cfg.temperature = parse_f32(args.get(i), "--temp")?;
            }
            "--top-k" => {
                i += 1;
                sampler_cfg.top_k = parse_usize(args.get(i), "--top-k")?;
            }
            "--top-p" => {
                i += 1;
                sampler_cfg.top_p = parse_f32(args.get(i), "--top-p")?;
            }
            "--repeat-penalty" => {
                i += 1;
                sampler_cfg.repeat_penalty = parse_f32(args.get(i), "--repeat-penalty")?;
            }
            "--seed" => {
                i += 1;
                sampler_cfg.seed = parse_usize(args.get(i), "--seed")? as u64;
            }
            "--dump-logits" => dump_logits = true,
            other => return Err(format!("unknown flag '{other}'")),
        }
        i += 1;
    }

    let raw_prompt = prompt.ok_or_else(|| "missing --prompt <text>".to_string())?;
    let t_load = Instant::now();
    let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
    gguf.advise_willneed();
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let model = Transformer::from_gguf(gguf).map_err(|e| e.to_string())?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let formatted = if chat_mode {
        format_chat_prompt(&model.config.arch, &raw_prompt)
    } else {
        raw_prompt
    };
    let prepend_bos = !chat_mode && tok.add_bos;
    let prompt_ids = tok.encode(&formatted, prepend_bos);
    if prompt_ids.is_empty() {
        return Err("prompt encoded to 0 tokens".into());
    }

    let needed_ctx = prompt_ids.len() + max_tokens;
    let actual_ctx = ctx_len.max(needed_ctx).min(model.config.max_seq_len);
    if prompt_ids.len() >= actual_ctx {
        return Err(format!(
            "prompt length ({}) exceeds model context ({actual_ctx})",
            prompt_ids.len()
        ));
    }

    let pool = ThreadPool::new(threads);
    let mut kv = model.new_kv_cache(actual_ctx, kv_quant);
    let mut scratch = model.new_scratch(DEFAULT_PREFILL_BATCH);
    let mut sampler = Sampler::new(model.config.vocab_size, sampler_cfg);

    eprintln!(
        "[tetanus] arch={} backend={} threads={} kv={} ({:.1} MiB) scratch={:.1} MiB load={:.1} ms",
        model.config.arch,
        model.backend.name(),
        pool.num_threads(),
        kv.quant.name(),
        kv.byte_size() as f64 / (1024.0 * 1024.0),
        scratch.byte_size() as f64 / (1024.0 * 1024.0),
        load_ms
    );

    let t_prefill = Instant::now();
    let logits = model.prefill(&prompt_ids, 0, &pool, &mut kv, &mut scratch);
    let prefill_sec = t_prefill.elapsed().as_secs_f64();

    if dump_logits {
        print_top_logits(logits, &tok, 5);
    }

    let mut history = Vec::with_capacity(actual_ctx);
    history.extend_from_slice(&prompt_ids);
    let mut next_tok = sampler.sample(logits, &history);
    let mut stream = Utf8Stream::new();
    let mut generated = 0usize;
    let mut decode_calls = 0usize;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let t_decode = Instant::now();

    let gen_limit = max_tokens.min(actual_ctx - prompt_ids.len());
    while generated < gen_limit {
        if tok.is_eos(next_tok) {
            break;
        }
        let piece = stream.push_token(&tok, next_tok);
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();

        history.push(next_tok);
        generated += 1;
        if generated >= gen_limit {
            break;
        }

        let pos = prompt_ids.len() + generated - 1;
        let logits = model.decode_token(next_tok, pos, &pool, &mut kv, &mut scratch);
        decode_calls += 1;
        next_tok = sampler.sample(logits, &history);
    }
    let _ = out.write_all(b"\n");

    let decode_sec = t_decode.elapsed().as_secs_f64();
    let decode_tps = if decode_calls > 0 && decode_sec > 0.0 {
        decode_calls as f64 / decode_sec
    } else {
        0.0
    };
    let avg_ctx = prompt_ids.len() + generated / 2;
    let bytes_per_tok = model.bytes_per_token_decode(avg_ctx, kv.quant);
    let bw_gbps = (bytes_per_tok as f64 * decode_tps) / 1e9;
    let rss_mib = peak_rss_bytes()
        .map(|b| format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "n/a".into());

    eprintln!(
        "[stats] prefill: {} tok in {:.1} ms ({:.1} tok/s) | decode: {} tok in {:.2} s ({:.1} tok/s, {:.1} GB/s) | peak_rss={}",
        prompt_ids.len(),
        prefill_sec * 1e3,
        prompt_ids.len() as f64 / prefill_sec.max(1e-9),
        generated,
        decode_sec,
        decode_tps,
        bw_gbps,
        rss_mib
    );

    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let mut model_path: Option<String> = None;
    let mut thread_list = vec![1usize, 2, default_perf_threads()];
    thread_list.sort_unstable();
    thread_list.dedup();
    let mut prompt_len = 64usize;
    let mut gen_len = 64usize;
    let mut kv_quant = KvQuant::F16;
    let mut synth_dtype = GgmlType::Q4_0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(args.get(i).ok_or("--model requires a path")?.clone());
            }
            "--threads" => {
                i += 1;
                let s = args.get(i).ok_or("--threads requires a list")?;
                thread_list = s
                    .split(',')
                    .map(|p| p.trim().parse::<usize>().map_err(|_| format!("bad thread '{p}'")))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--prompt-len" => {
                i += 1;
                prompt_len = parse_usize(args.get(i), "--prompt-len")?;
            }
            "--gen-len" => {
                i += 1;
                gen_len = parse_usize(args.get(i), "--gen-len")?;
            }
            "--kv-quant" => {
                i += 1;
                kv_quant = parse_kv_quant(args.get(i))?;
            }
            "--dtype" => {
                i += 1;
                synth_dtype = match args.get(i).map(String::as_str) {
                    Some("q4_0") => GgmlType::Q4_0,
                    Some("q8_0") => GgmlType::Q8_0,
                    Some("q4_k") => GgmlType::Q4K,
                    Some("q6_k") => GgmlType::Q6K,
                    Some("f16") => GgmlType::F16,
                    Some("f32") => GgmlType::F32,
                    _ => return Err("--dtype must be one of q4_0, q8_0, q4_k, q6_k, f16, f32".into()),
                };
            }
            other => return Err(format!("unknown flag '{other}'")),
        }
        i += 1;
    }

    println!("SIMD backend: {}", SimdBackend::detect().name());

    if let Some(path) = model_path {
        let gguf = GgufFile::open(&path).map_err(|e| e.to_string())?;
        gguf.advise_willneed();
        let model = Transformer::from_gguf(gguf).map_err(|e| e.to_string())?;
        let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 100 + 1) as u32).collect();
        let ctx = (prompt_len + gen_len + 16).min(model.config.max_seq_len);

        println!(
            "Model: {} (layers={}, dim={}, hidden={}, kv={})",
            path,
            model.config.n_layers,
            model.config.dim,
            model.config.hidden_dim,
            kv_quant.name()
        );
        println!("threads | prefill_ms | prefill_tok/s | decode_tok/s | bw_GB/s | peak_rss_MiB");

        for &n_threads in &thread_list {
            let pool = ThreadPool::new(n_threads);
            let mut kv = model.new_kv_cache(ctx, kv_quant);
            let mut scratch = model.new_scratch(DEFAULT_PREFILL_BATCH);

            let t0 = Instant::now();
            let _ = model.prefill(&prompt, 0, &pool, &mut kv, &mut scratch);
            let prefill_s = t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            for step in 0..gen_len {
                let _ = model.decode_token(1, prompt_len + step, &pool, &mut kv, &mut scratch);
            }
            let decode_s = t1.elapsed().as_secs_f64();
            let decode_tps = gen_len as f64 / decode_s.max(1e-9);
            let bw_gbps = (model.bytes_per_token_decode(prompt_len + gen_len / 2, kv_quant) as f64
                * decode_tps)
                / 1e9;
            let rss_mib = peak_rss_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0);

            println!(
                "{n_threads:>7} | {:>10.2} | {:>13.1} | {:>12.1} | {:>7.1} | {:>12.1}",
                prefill_s * 1e3,
                prompt_len as f64 / prefill_s.max(1e-9),
                decode_tps,
                bw_gbps,
                rss_mib
            );
        }
    } else {
        // Synthetic GEMV / GEMM bandwidth micro-benchmark (default 4096 x 4096 = realistic MLP slice).
        let rows = 4096usize;
        let cols = 4096usize;
        let iters = 64usize;
        let bs = synth_dtype.block_size();
        let row_bytes = (cols / bs) * synth_dtype.type_size();
        let weights = build_synthetic_matrix(rows, cols, synth_dtype);
        let xs_decode: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let mut out_decode = vec![0.0f32; rows];
        let mut xq = vec![0u8; (cols / 32 + 8) * 40];

        let batch = 32usize;
        let xs_prefill = vec![0.05f32; batch * cols];
        let mut out_prefill = vec![0.0f32; batch * rows];
        let mut xq_prefill = vec![0u8; batch * ((cols / 32 + 8) * 40)];

        println!(
            "Synthetic matmul benchmark: [{rows}x{cols}] dtype={} (matrix={:.2} MiB)",
            synth_dtype.name(),
            weights.len() as f64 / (1024.0 * 1024.0)
        );
        println!("threads | gemv_us | gemv_GB/s | gemm32_ms | gemm32_GFLOP/s");

        let backend = SimdBackend::detect();
        let xq_len = xq.len();
        for &n_threads in &thread_list {
            let pool = ThreadPool::new(n_threads);
            // Warmup
            matmul_batch(
                &pool, backend, synth_dtype, &weights, rows, cols, &xs_decode, cols,
                &mut out_decode, rows, 1, &mut xq, xq_len,
            );

            let t0 = Instant::now();
            for _ in 0..iters {
                matmul_batch(
                    &pool, backend, synth_dtype, &weights, rows, cols, &xs_decode, cols,
                    &mut out_decode, rows, 1, &mut xq, xq_len,
                );
            }
            let gemv_s = t0.elapsed().as_secs_f64() / (iters as f64);
            let gemv_gbps = (weights.len() as f64 / gemv_s) / 1e9;

            let t1 = Instant::now();
            let gemm_iters = 8usize;
            for _ in 0..gemm_iters {
                matmul_batch(
                    &pool,
                    backend,
                    synth_dtype,
                    &weights,
                    rows,
                    cols,
                    &xs_prefill,
                    cols,
                    &mut out_prefill,
                    rows,
                    batch,
                    &mut xq_prefill,
                    (cols / 32 + 8) * 40,
                );
            }
            let gemm_s = t1.elapsed().as_secs_f64() / (gemm_iters as f64);
            let flops = 2.0 * (rows as f64) * (cols as f64) * (batch as f64);
            let gemm_gflops = (flops / gemm_s) / 1e9;

            println!(
                "{n_threads:>7} | {:>7.1} | {:>9.1} | {:>9.2} | {:>14.1}",
                gemv_s * 1e6,
                gemv_gbps,
                gemm_s * 1e3,
                gemm_gflops
            );
        }
        let _ = row_bytes;
    }

    Ok(())
}

fn build_synthetic_matrix(rows: usize, cols: usize, dtype: GgmlType) -> Vec<u8> {
    let bs = dtype.block_size();
    let row_bytes = (cols / bs) * dtype.type_size();
    let mut buf = vec![0u8; rows * row_bytes];
    let scale = f32_to_f16(0.02).to_le_bytes();
    match dtype {
        GgmlType::Q4_0 => {
            for blk in buf.chunks_exact_mut(18) {
                blk[0..2].copy_from_slice(&scale);
                for j in 0..16 {
                    blk[2 + j] = 0x97; // nibbles 7 & 9 (-1 and +1 after -8)
                }
            }
        }
        GgmlType::Q8_0 => {
            for blk in buf.chunks_exact_mut(34) {
                blk[0..2].copy_from_slice(&scale);
                for j in 0..32 {
                    blk[2 + j] = ((j as i8) - 16) as u8;
                }
            }
        }
        GgmlType::F16 => {
            for ch in buf.chunks_exact_mut(2) {
                ch.copy_from_slice(&scale);
            }
        }
        GgmlType::F32 => {
            let f = 0.02f32.to_le_bytes();
            for ch in buf.chunks_exact_mut(4) {
                ch.copy_from_slice(&f);
            }
        }
        GgmlType::Q4K => {
            for blk in buf.chunks_exact_mut(144) {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&scale);
                blk[4..16].fill(0x11);
                blk[16..144].fill(0x97);
            }
        }
        GgmlType::Q6K => {
            for blk in buf.chunks_exact_mut(210) {
                blk[0..128].fill(0x23);
                blk[128..192].fill(0x55);
                blk[192..208].fill(2);
                blk[208..210].copy_from_slice(&scale);
            }
        }
        _ => {}
    }
    buf
}

fn format_chat_prompt(arch: &str, user_msg: &str) -> String {
    if arch.starts_with("qwen") {
        format!("<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")
    } else if arch == "llama" {
        format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    } else {
        user_msg.to_owned()
    }
}

fn print_top_logits(logits: &[f32], tok: &Tokenizer, k: usize) {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let k = k.min(pairs.len());
    pairs.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    pairs[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("[logits top-{k}]");
    for &(id, val) in &pairs[..k] {
        let s = tok.token_str(id as u32).unwrap_or("?");
        eprintln!("  id={id:<7} logit={val:>9.4}  tok={s:?}");
    }
}

fn parse_usize(arg: Option<&String>, flag: &str) -> Result<usize, String> {
    arg.ok_or_else(|| format!("{flag} requires an integer"))?
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {flag}"))
}

fn parse_f32(arg: Option<&String>, flag: &str) -> Result<f32, String> {
    arg.ok_or_else(|| format!("{flag} requires a float"))?
        .parse::<f32>()
        .map_err(|_| format!("invalid float for {flag}"))
}

fn parse_kv_quant(arg: Option<&String>) -> Result<KvQuant, String> {
    match arg.map(String::as_str) {
        Some("f16") => Ok(KvQuant::F16),
        Some("q8") | Some("q8_0") => Ok(KvQuant::Q8_0),
        _ => Err("--kv-quant must be 'f16' or 'q8'".into()),
    }
}
