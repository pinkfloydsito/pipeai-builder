#!/usr/bin/env python3
"""
Run benchmark on a single dataset.
Used by SLURM array jobs to parallelize across datasets.
"""

import sys
import argparse
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "benchmarks"))

from benchmark_automl_vs_llm import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run benchmark on single dataset")
    parser.add_argument("--dataset-id", type=int, required=True, help="OpenML dataset ID")
    parser.add_argument("--api-key", required=True, help="API key for LLM")
    parser.add_argument("--tpot-time", type=int, default=10, help="TPOT max time (minutes)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "openai"], help="LLM provider")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Running benchmark for dataset {args.dataset_id}")
    print(f"Output directory: {output_dir}")
    print(f"TPOT time limit: {args.tpot_time} minutes")
    print(f"LLM provider: {args.provider}")
    print(f"=" * 60)

    # Initialize runner
    runner = BenchmarkRunner(
        api_key=args.api_key,
        tpot_max_time_mins=args.tpot_time,
        provider=args.provider,
    )

    # Run on single dataset
    results_df = runner.run_benchmark(
        dataset_ids=[args.dataset_id],
        output_dir=str(output_dir),
    )

    # Save results for this dataset
    csv_path = output_dir / f"results_dataset_{args.dataset_id}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
