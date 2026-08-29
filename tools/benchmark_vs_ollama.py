#!/usr/bin/env python3
"""Reproducible, honest benchmark: Bad Apple vs Ollama on this exact Mac.

This does NOT compare identical model weights — that would be impossible,
since Bad Apple runs MLX-quantized weights and Ollama runs GGUF weights.
Instead it compares what a real user actually gets from each tool's default
local setup on the same hardware: same prompts, same output token budget,
each tool's own inference stack, each tool's own quantization.

Usage:
    .venv/bin/python tools/benchmark_vs_ollama.py
    .venv/bin/python tools/benchmark_vs_ollama.py --tokens 80 --ollama-model qwen2.5:7b-instruct-q4_0

Requires:
- Ollama running locally with the target model already pulled
  (`ollama pull <model>`).
- Bad Apple's release CLI built (`cargo build --release`).

Writes a timestamped Markdown + JSON report to benchmarks/results/.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Neutral prompts: no Bad Apple persona flavor, so this measures raw inference
# performance rather than personality/answer-style differences.
DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain how a car engine works in two sentences.",
    "What is 15 times 24?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class RunResult:
    tool: str
    model: str
    prompt: str
    tokens: int
    ttft_s: float
    decode_tps: float
    total_tps: float
    peak_mem_gb: float
    wall_s: float
    error: str = ""


def _run_shell(cmd: list[str], timeout: int = 300) -> tuple[str, str, int]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return proc.stdout, proc.stderr, proc.returncode


def _ollama_rss_gb(pid: int) -> float:
    try:
        out, _, _ = _run_shell(["ps", "-o", "rss=", "-p", str(pid)], timeout=5)
        return float(out.strip()) / (1024 * 1024)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def _find_ollama_pid() -> int | None:
    try:
        out, _, _ = _run_shell(["pgrep", "-f", "ollama serve"], timeout=5)
        pids = [int(p) for p in out.split() if p.strip()]
        return pids[0] if pids else None
    except (ValueError, subprocess.SubprocessError):
        return None


def run_ollama(prompt: str, model: str, max_tokens: int) -> RunResult:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})

    pid = _find_ollama_pid()
    peak_mem = _ollama_rss_gb(pid) if pid else 0.0
    t0 = time.time()
    try:
        # OLLAMA_URL is the hardcoded local http://localhost:11434 constant
        # above, not user input; no scheme injection is possible here.
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return RunResult("ollama", model, prompt, 0, 0, 0, 0, peak_mem, time.time() - t0, error=str(e))
    wall = time.time() - t0
    if pid:
        peak_mem = max(peak_mem, _ollama_rss_gb(pid))

    if "error" in body:
        return RunResult("ollama", model, prompt, 0, 0, 0, 0, peak_mem, wall, error=body["error"])

    eval_count = body.get("eval_count", 0)
    eval_duration_ns = body.get("eval_duration", 1) or 1
    prompt_eval_duration_ns = body.get("prompt_eval_duration", 0)
    total_duration_ns = body.get("total_duration", 1) or 1

    decode_tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
    total_tps = eval_count / (total_duration_ns / 1e9) if total_duration_ns else 0.0
    # Ollama does not report a true "first token" timestamp in non-streaming
    # mode; approximate TTFT as prompt-eval time (prefill), which is the
    # dominant component of time-to-first-token for a cold KV cache.
    ttft = prompt_eval_duration_ns / 1e9

    return RunResult("ollama", model, prompt, eval_count, ttft, decode_tps, total_tps, peak_mem, wall)


BENCH_ROW_RE = re.compile(
    r"^(?P<prompt>.{1,38})\s+(?P<tok>\d+)\s+(?P<ttft>[\d.]+)\s+(?P<decode>[\d.]+)\s+(?P<total>[\d.]+)\s+(?P<mem>[\d.]+)\s*$"
)


def run_bad_apple(prompt: str, max_tokens: int, badapple_bin: Path) -> RunResult:
    t0 = time.time()
    try:
        out, err, code = _run_shell([str(badapple_bin), "--benchmark", "-n", str(max_tokens), prompt], timeout=180)
    except subprocess.TimeoutExpired:
        return RunResult("bad_apple", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit", prompt, 0, 0, 0, 0, 0, time.time() - t0, error="timeout")
    wall = time.time() - t0
    if code != 0:
        return RunResult("bad_apple", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit", prompt, 0, 0, 0, 0, 0, wall, error=(err or out)[:200])

    for line in out.splitlines():
        m = BENCH_ROW_RE.match(line.rstrip())
        if m and not line.strip().startswith("prompt"):
            return RunResult(
                "bad_apple",
                "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
                prompt,
                int(m.group("tok")),
                float(m.group("ttft")),
                float(m.group("decode")),
                float(m.group("total")),
                float(m.group("mem")),
                wall,
            )
    return RunResult("bad_apple", "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit", prompt, 0, 0, 0, 0, 0, wall, error=f"could not parse output: {out[:200]!r}")


def _free_memory_percent() -> float:
    """Best-effort system-wide free memory percentage via memory_pressure."""
    try:
        out, _, _ = _run_shell(["memory_pressure"], timeout=5)
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else -1.0
    except subprocess.SubprocessError:
        return -1.0


def system_info() -> dict:
    try:
        chip, _, _ = _run_shell(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
    except subprocess.SubprocessError:
        chip = "unknown"
    try:
        mem_out, _, _ = _run_shell(["sysctl", "-n", "hw.memsize"], timeout=5)
        mem_gb = round(int(mem_out.strip()) / (1024 ** 3), 1)
    except (subprocess.SubprocessError, ValueError):
        mem_gb = 0.0
    return {
        "macos_version": platform.mac_ver()[0],
        "chip": chip.strip(),
        "memory_gb": mem_gb,
        "python": platform.python_version(),
        "free_memory_percent_at_start": _free_memory_percent(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=60, help="Max output tokens per prompt (default 60)")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct-q4_0", help="Ollama model tag to compare against")
    parser.add_argument("--badapple-bin", default=str(REPO_ROOT / "target/release/badapple"))
    parser.add_argument("--prompts-file", help="Optional file with one prompt per line, overrides defaults")
    args = parser.parse_args()

    badapple_bin = Path(args.badapple_bin)
    if not badapple_bin.is_file():
        print(f"error: badapple binary not found at {badapple_bin}. Run `cargo build --release` first.", file=sys.stderr)
        return 1

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if p.strip()]

    print(f"=== Bad Apple vs Ollama benchmark ({len(prompts)} prompts, max_tokens={args.tokens}) ===")
    sysinfo = system_info()
    print(f"Host: {sysinfo['chip']}, {sysinfo['memory_gb']} GB RAM, macOS {sysinfo['macos_version']}")
    print("Bad Apple model: caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit (MLX, 4-bit)")
    print(f"Ollama model:    {args.ollama_model} (GGUF, llama.cpp/Metal)")
    if 0 <= sysinfo["free_memory_percent_at_start"] < 40:
        print(f"WARNING: only {sysinfo['free_memory_percent_at_start']:.0f}% system memory free at start. "
              "Results will be pessimistic vs. an idle machine — close other apps for a fairer run.")
    print()

    # Warm up both engines once so the first timed prompt isn't paying cold-load cost.
    print("Warming up both engines...")
    run_bad_apple("Say hello.", 16, badapple_bin)
    run_ollama("Say hello.", args.ollama_model, 16)

    results: list[RunResult] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt!r}")
        ba = run_bad_apple(prompt, args.tokens, badapple_bin)
        ol = run_ollama(prompt, args.ollama_model, args.tokens)
        results.append(ba)
        results.append(ol)
        print(f"    bad_apple: {ba.tokens:3d} tok, ttft={ba.ttft_s:5.2f}s, decode={ba.decode_tps:6.1f} tok/s, mem={ba.peak_mem_gb:.2f} GB" + (f"  [ERROR: {ba.error}]" if ba.error else ""))
        print(f"    ollama:    {ol.tokens:3d} tok, ttft={ol.ttft_s:5.2f}s, decode={ol.decode_tps:6.1f} tok/s, mem={ol.peak_mem_gb:.2f} GB" + (f"  [ERROR: {ol.error}]" if ol.error else ""))

    # Aggregate.
    def _agg(tool: str, field: str) -> float:
        vals = [getattr(r, field) for r in results if r.tool == tool and not r.error]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "bad_apple": {
            "avg_ttft_s": round(_agg("bad_apple", "ttft_s"), 3),
            "avg_decode_tps": round(_agg("bad_apple", "decode_tps"), 2),
            "avg_total_tps": round(_agg("bad_apple", "total_tps"), 2),
            "max_mem_gb": round(max((r.peak_mem_gb for r in results if r.tool == "bad_apple"), default=0.0), 2),
            "errors": sum(1 for r in results if r.tool == "bad_apple" and r.error),
        },
        "ollama": {
            "avg_ttft_s": round(_agg("ollama", "ttft_s"), 3),
            "avg_decode_tps": round(_agg("ollama", "decode_tps"), 2),
            "avg_total_tps": round(_agg("ollama", "total_tps"), 2),
            "max_mem_gb": round(max((r.peak_mem_gb for r in results if r.tool == "ollama"), default=0.0), 2),
            "errors": sum(1 for r in results if r.tool == "ollama" and r.error),
        },
    }

    print("\n=== Summary ===")
    print(f"{'':20}{'avg TTFT (s)':>14}{'avg decode tok/s':>20}{'max mem (GB)':>16}")
    for tool, s in summary.items():
        print(f"{tool:20}{s['avg_ttft_s']:>14.2f}{s['avg_decode_tps']:>20.1f}{s['max_mem_gb']:>16.2f}")

    if summary["bad_apple"]["avg_decode_tps"] and summary["ollama"]["avg_decode_tps"]:
        ratio = summary["bad_apple"]["avg_decode_tps"] / summary["ollama"]["avg_decode_tps"]
        print(f"\nBad Apple decode throughput is {ratio:.2f}x Ollama's on this Mac.")
    if summary["bad_apple"]["avg_ttft_s"] and summary["ollama"]["avg_ttft_s"]:
        ratio = summary["ollama"]["avg_ttft_s"] / summary["bad_apple"]["avg_ttft_s"]
        print(f"Bad Apple time-to-first-token is {ratio:.2f}x faster than Ollama's on this Mac.")

    # Persist report.
    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"bad_apple_vs_ollama_{ts}.json"
    md_path = out_dir / f"bad_apple_vs_ollama_{ts}.md"

    json_path.write_text(json.dumps({
        "system": sysinfo,
        "bad_apple_model": "caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit",
        "ollama_model": args.ollama_model,
        "max_tokens": args.tokens,
        "results": [asdict(r) for r in results],
        "summary": summary,
    }, indent=2), encoding="utf-8")

    md_lines = [
        "# Bad Apple vs Ollama — reproducible local benchmark",
        "",
        f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Host: {sysinfo['chip']}, {sysinfo['memory_gb']} GB RAM, macOS {sysinfo['macos_version']}, "
        f"{sysinfo['free_memory_percent_at_start']:.0f}% memory free at start",
        "",
        "**Caveat 1 (models):** this does not compare identical model weights. Bad Apple runs "
        "`caiovicentino1/Qwen3.5-9B-HLWQ-MLX-4bit` (MLX). Ollama runs "
        f"`{args.ollama_model}` (GGUF via llama.cpp/Metal). Both are 4-bit-class quantized "
        "instruction models in a similar size class. This measures what a real user gets from "
        "each tool's default local setup on identical hardware, not raw model-vs-model quality.",
        "",
        "**Caveat 2 (system load):** this machine has 16 GB of unified memory, which both "
        "engines' models (~5 GB Bad Apple MLX + ~4.5 GB Ollama GGUF) compete for, along with "
        "whatever else is running. Numbers recorded with significant background load (an active "
        "IDE/agent session, browser, etc.) will be pessimistic versus an idle machine. Re-run "
        "with nothing else open for the most favorable, reproducible numbers.",
        "",
        "## Summary",
        "",
        "| Tool | Avg TTFT (s) | Avg decode tok/s | Max RSS/peak mem (GB) | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for tool, s in summary.items():
        md_lines.append(f"| {tool} | {s['avg_ttft_s']:.2f} | {s['avg_decode_tps']:.1f} | {s['max_mem_gb']:.2f} | {s['errors']} |")
    md_lines += ["", "## Per-prompt results", "", "| Tool | Prompt | Tokens | TTFT (s) | Decode tok/s | Mem (GB) |", "|---|---|---:|---:|---:|---:|"]
    for r in results:
        md_lines.append(f"| {r.tool} | {r.prompt[:40]} | {r.tokens} | {r.ttft_s:.2f} | {r.decode_tps:.1f} | {r.peak_mem_gb:.2f} |" + (f" ERROR: {r.error}" if r.error else ""))
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
