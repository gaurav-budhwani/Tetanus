#!/usr/bin/env python3
"""
Reproducible benchmark and logit-parity harness comparing Tetanus against
llama.cpp (`llama-cli` / `llama-bench`) and PyTorch CPU (`transformers`).

Usage:
    python3 scripts/bench_compare.py --gguf path/to/model.gguf \
        [--hf-model Qwen/Qwen2.5-0.5B-Instruct] \
        [--prompt "Explain virtual memory page tables in 3 sentences."] \
        [--max-tokens 128] [--threads 4]
"""

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_with_rss(cmd):
    """Runs `cmd` and returns (elapsed_s, stdout, stderr, peak_rss_mib)."""
    system = platform.system()
    if system == "Darwin":
        wrapped = ["/usr/bin/time", "-l", *cmd]
    elif system == "Linux":
        wrapped = ["/usr/bin/time", "-v", *cmd]
    else:
        wrapped = cmd

    t0 = time.perf_counter()
    proc = subprocess.run(wrapped, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    rss_mib = None
    if system == "Darwin":
        m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
        if m:
            rss_mib = int(m.group(1)) / (1024 * 1024)
    elif system == "Linux":
        m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", proc.stderr)
        if m:
            rss_mib = int(m.group(1)) / 1024

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return elapsed, proc.stdout, proc.stderr, rss_mib


def bench_tetanus(binary: str, gguf: str, prompt: str, max_tokens: int, threads: int, kv_quant: str):
    cmd = [
        binary,
        "run",
        gguf,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--threads",
        str(threads),
        "--kv-quant",
        kv_quant,
        "--temp",
        "0",
        "--dump-logits",
    ]
    _, stdout, stderr, rss_mib = run_with_rss(cmd)

    prefill_tps = decode_tps = bw_gbps = None
    m = re.search(
        r"prefill:\s+\d+\s+tok in [\d.]+\s+ms \(([\d.]+)\s+tok/s\)\s+\|\s+decode:\s+\d+\s+tok in [\d.]+\s+s \(([\d.]+)\s+tok/s,\s+([\d.]+)\s+GB/s\)",
        stderr,
    )
    if m:
        prefill_tps = float(m.group(1))
        decode_tps = float(m.group(2))
        bw_gbps = float(m.group(3))

    top_ids = [int(x) for x in re.findall(r"id=(\d+)\s+logit=", stderr)]
    return {
        "engine": f"Tetanus ({kv_quant})",
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "bw_gbps": bw_gbps,
        "rss_mib": rss_mib,
        "top_ids": top_ids,
        "text": stdout.strip(),
    }


def bench_llama_cpp(gguf: str, prompt: str, max_tokens: int, threads: int):
    llama_cli = shutil.which("llama-cli") or shutil.which("main")
    if not llama_cli:
        return None
    cmd = [
        llama_cli,
        "-m",
        gguf,
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        "-t",
        str(threads),
        "--temp",
        "0",
        "-ngl",
        "0",  # Force CPU-only apples-to-apples comparison
        "--no-display-prompt",
    ]
    _, stdout, stderr, rss_mib = run_with_rss(cmd)
    prefill_tps = decode_tps = None
    mp = re.search(r"prompt eval time\s*=.*?([\d.]+)\s+tokens per second", stderr)
    md = re.search(r"eval time\s*=.*?([\d.]+)\s+tokens per second", stderr)
    if mp:
        prefill_tps = float(mp.group(1))
    if md:
        decode_tps = float(md.group(1))
    return {
        "engine": "llama.cpp (CPU)",
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "bw_gbps": None,
        "rss_mib": rss_mib,
        "top_ids": [],
        "text": stdout.strip(),
    }


def bench_pytorch_cpu(hf_model: str, prompt: str, max_tokens: int, threads: int):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None

    # Measure in a clean child process so RSS captures PyTorch + weights accurately.
    snippet = """
import sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
hf_model, prompt, max_tokens, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(threads)
tok = AutoTokenizer.from_pretrained(hf_model)
model = AutoModelForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float32).eval()
inputs = tok(prompt, return_tensors="pt")
with torch.no_grad():
    t0 = time.perf_counter()
    out = model(**inputs, use_cache=True)
    prefill_s = time.perf_counter() - t0
    logits = out.logits[0, -1]
    top5 = torch.topk(logits, 5).indices.tolist()
    past = out.past_key_values
    next_tok = logits.argmax().view(1, 1)
    t1 = time.perf_counter()
    for _ in range(max_tokens - 1):
        step = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = step.past_key_values
        next_tok = step.logits[0, -1].argmax().view(1, 1)
    decode_s = time.perf_counter() - t1
print(f"PREFILL_TPS={inputs['input_ids'].shape[1] / prefill_s:.2f}")
print(f"DECODE_TPS={(max_tokens - 1) / decode_s:.2f}")
print(f"TOP5={top5}")
"""
    _, stdout, _, rss_mib = run_with_rss(
        [sys.executable, "-c", snippet, hf_model, prompt, str(max_tokens), str(threads)]
    )
    prefill_tps = float(re.search(r"PREFILL_TPS=([\d.]+)", stdout).group(1))
    decode_tps = float(re.search(r"DECODE_TPS=([\d.]+)", stdout).group(1))
    top_ids = eval(re.search(r"TOP5=(\[.*?\])", stdout).group(1))
    return {
        "engine": "PyTorch CPU (FP32)",
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "bw_gbps": None,
        "rss_mib": rss_mib,
        "top_ids": top_ids,
        "text": "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="Path to local GGUF model")
    p.add_argument("--hf-model", default=None, help="Optional HF model ID for PyTorch CPU comparison")
    p.add_argument("--bin", default="target/release/tetanus", help="Path to compiled tetanus binary")
    p.add_argument("--prompt", default="Explain virtual memory page tables in 3 concise sentences.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    if not Path(args.bin).exists():
        print(f"Binary {args.bin} not found. Run `cargo build --release` first.", file=sys.stderr)
        sys.exit(1)

    rows = [
        bench_tetanus(args.bin, args.gguf, args.prompt, args.max_tokens, args.threads, "f16"),
        bench_tetanus(args.bin, args.gguf, args.prompt, args.max_tokens, args.threads, "q8"),
    ]
    llama_res = bench_llama_cpp(args.gguf, args.prompt, args.max_tokens, args.threads)
    if llama_res:
        rows.append(llama_res)
    if args.hf_model:
        pt_res = bench_pytorch_cpu(args.hf_model, args.prompt, args.max_tokens, args.threads)
        if pt_res:
            rows.append(pt_res)

    print("\n| Engine | Prefill (tok/s) | Decode (tok/s) | Peak RSS (MiB) | Top-1 ID |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        pf = f"{r['prefill_tps']:.1f}" if r["prefill_tps"] is not None else "n/a"
        dc = f"{r['decode_tps']:.1f}" if r["decode_tps"] is not None else "n/a"
        rss = f"{r['rss_mib']:.1f}" if r["rss_mib"] is not None else "n/a"
        top1 = str(r["top_ids"][0]) if r["top_ids"] else "n/a"
        print(f"| {r['engine']} | {pf} | {dc} | {rss} | {top1} |")


if __name__ == "__main__":
    main()
