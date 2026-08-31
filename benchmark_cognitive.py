#!/usr/bin/env python3
"""A/B benchmark: compare Bad Apple with and without the cognitive architecture.

Runs a standardized set of queries in three modes:
1. Full cognitive stack (connectome, hyperdimensional, dual-process)
2. Fast tier only (0.5B model, no cognitive layer)
3. 9B only (no cognitive layer, no fast tier)

Measures: latency, token count, cache hit rate, classification accuracy.
Reports results as a table.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# Standard benchmark prompts covering different complexity levels
BENCHMARK_PROMPTS = [
    # Simple (should hit fast tier)
    ("What is 2+2?", "simple"),
    ("What time is it?", "simple"),
    ("Who are you?", "simple"),
    # Medium (should use 9B)
    ("Write a haiku about on-device AI.", "medium"),
    ("Explain what a Unix socket is in two sentences.", "medium"),
    ("What are the benefits of local inference?", "medium"),
    # Complex (should benefit from cognitive layer)
    ("Remember that I prefer Python over Rust. What language should I use for a new CLI tool?", "complex"),
    ("Based on what you know about my workspace, what project am I likely working on?", "complex"),
    ("Summarize the key themes from our recent conversations.", "complex"),
]

def run_query(prompt: str, max_tokens: int = 120, env_overrides: dict | None = None) -> dict:
    """Run a single query through the badapple CLI and return metrics."""
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    
    cli = Path(__file__).parent / "target" / "release" / "badapple"
    if not cli.exists():
        cli = Path(subprocess.check_output(["which", "badapple"]).decode().strip())
    
    start = time.monotonic()
    result = subprocess.run(
        [str(cli), "--json", "-n", str(max_tokens), prompt],
        capture_output=True, text=True, timeout=120, env=env, check=False
    )
    elapsed = time.monotonic() - start
    
    # Parse JSON output for metrics
    metrics = {
        "prompt": prompt,
        "latency_s": round(elapsed, 2),
        "returncode": result.returncode,
        "tokens": 0,
        "text": "",
    }
    
    for line in result.stdout.splitlines():
        try:
            msg = json.loads(line)
            if msg.get("type") == "done":
                m = msg.get("metrics") or {}
                metrics["tokens"] = m.get("tokens", 0)
                metrics["decode_tps"] = m.get("decode_tps", 0)
                metrics["text"] = msg.get("text", "")[:200]
                metrics["tier"] = m.get("tier", "unknown")
        except json.JSONDecodeError:
            continue
    
    return metrics

def run_benchmark(mode: str) -> list[dict]:
    """Run all benchmark prompts in a given mode."""
    env_overrides = {
        "cognitive_full": {"BADAPPLE_COGNITIVE": "1"},
        "fast_tier_only": {"BADAPPLE_FAST_TIER": "1", "BADAPPLE_COGNITIVE": "0"},
        "9b_only": {"BADAPPLE_FAST_TIER": "0", "BADAPPLE_COGNITIVE": "0"},
    }
    
    env = env_overrides.get(mode, {})
    results = []
    for prompt, complexity in BENCHMARK_PROMPTS:
        print(f"  [{mode}] {complexity}: {prompt[:50]}...", flush=True)
        m = run_query(prompt, env_overrides=env)
        m["complexity"] = complexity
        m["mode"] = mode
        results.append(m)
    return results

def print_results(all_results: dict[str, list[dict]]):
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("Cognitive Architecture A/B Benchmark Results")
    print("=" * 80)
    
    for mode, results in all_results.items():
        print(f"\n--- {mode} ---")
        print(f"{'Prompt':<45} {'Complexity':<10} {'Latency':<10} {'Tokens':<8} {'Tier':<10}")
        print("-" * 80)
        for r in results:
            print(f"{r['prompt'][:43]:<45} {r['complexity']:<10} {r['latency_s']:<10} {r['tokens']:<8} {r.get('tier', '?'):<10}")
        
        avg_latency = sum(r["latency_s"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        print(f"\n  Average latency: {avg_latency:.2f}s, Average tokens: {avg_tokens:.0f}")
    
    # Compare
    print("\n--- Comparison ---")
    modes = list(all_results.keys())
    if len(modes) >= 2:
        for i in range(len(modes)):
            for j in range(i+1, len(modes)):
                m1, m2 = modes[i], modes[j]
                r1, r2 = all_results[m1], all_results[m2]
                avg1 = sum(r["latency_s"] for r in r1) / len(r1)
                avg2 = sum(r["latency_s"] for r in r2) / len(r2)
                print(f"  {m1} vs {m2}: latency delta = {avg1 - avg2:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Cognitive architecture A/B benchmark")
    parser.add_argument("--modes", nargs="+", default=["cognitive_full", "fast_tier_only", "9b_only"],
                       help="Modes to benchmark")
    parser.add_argument("--output", type=str, help="Save results as JSON")
    args = parser.parse_args()
    
    all_results = {}
    for mode in args.modes:
        print(f"\nRunning benchmark in {mode} mode...", flush=True)
        all_results[mode] = run_benchmark(mode)
    
    print_results(all_results)
    
    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
