"""
MARS — Benchmark Runner
─────────────────────────────────────────────────────────────────────────────
Runs both the multi-agent pipeline and the single-agent baseline on the
full benchmark query set and saves results for analysis.

Usage:
    python evaluation/benchmark.py                  # run all queries
    python evaluation/benchmark.py --dry-run        # run first 3 queries only
"""

import sys
import json
import time
import argparse
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_queries() -> list[dict]:
    path = DATA_DIR / "benchmark_queries.json"
    with open(path) as f:
        return json.load(f)


def run_benchmark(dry_run: bool = False):
    from baseline.single_agent import run_baseline, save_result as save_baseline
    from pipeline.runner       import run_pipeline, save_result as save_multiagent

    queries = load_queries()
    if dry_run:
        queries = queries[:3]
        print(f"[benchmark] DRY RUN — running {len(queries)} queries only")
    else:
        print(f"[benchmark] Running full benchmark: {len(queries)} queries")

    summary = []

    for i, item in enumerate(queries, 1):
        q = item["query"]
        print(f"\n[benchmark] Query {i}/{len(queries)}: {q}")

        # Single-agent baseline
        t0     = time.time()
        base   = run_baseline(q, verbose=False)
        base_t = time.time() - t0
        save_baseline(base)

        # Multi-agent pipeline (Phase 3 — placeholder for now)
        # multi  = run_pipeline(q, verbose=False)
        # save_multiagent(multi, q)

        summary.append({
            "query":              q,
            "category":           item.get("category", "general"),
            "baseline_latency":   round(base_t, 2),
            "baseline_tokens":    base.token_count,
        })
        print(f"[benchmark] Baseline done in {base_t:.2f}s")

    out = RESULTS_DIR / "benchmark_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[benchmark] Summary saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_benchmark(dry_run=args.dry_run)
