#!/usr/bin/env python3
"""
Merge results from all SLURM array jobs into a single CSV and summary.
Run this after all jobs complete.
"""

import sys
from pathlib import Path
import pandas as pd
import json

def main():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    # Find all result CSVs
    csv_files = list(results_dir.glob("results_dataset_*.csv"))

    if not csv_files:
        print("No result files found")
        return 1

    print(f"Found {len(csv_files)} result files")

    # Merge all CSVs
    dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} rows")

    merged_df = pd.concat(dfs, ignore_index=True)

    # Save merged results
    merged_path = results_dir / "benchmark_results_all.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"\nMerged results saved to: {merged_path}")
    print(f"Total rows: {len(merged_df)}")

    # Generate summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    # Performance by approach
    print("\nPerformance by Approach (Balanced Accuracy):")
    summary = merged_df.groupby("approach").agg({
        "test_balanced_accuracy": ["mean", "std", "min", "max"],
        "optimization_time": "mean",
    }).round(4)
    print(summary)

    # Win counts
    print("\n\nWin Counts (Best per Dataset):")
    win_counts = {"baseline": 0, "tpot": 0, "llm": 0}

    for dataset_name in merged_df["dataset_name"].unique():
        dataset_results = merged_df[merged_df["dataset_name"] == dataset_name]
        if len(dataset_results) > 0:
            winner_idx = dataset_results["test_balanced_accuracy"].idxmax()
            winner = dataset_results.loc[winner_idx]
            win_counts[winner["approach"]] = win_counts.get(winner["approach"], 0) + 1
            print(f"  {dataset_name:25s}: {winner['approach']:10s} ({winner['test_balanced_accuracy']:.4f})")

    print("\n\nOverall:")
    total = len(merged_df["dataset_name"].unique())
    for approach, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        print(f"  {approach:10s}: {count}/{total} ({pct:.1f}%)")

    # Save summary to JSON
    summary_data = {
        "total_datasets": total,
        "total_experiments": len(merged_df),
        "win_counts": win_counts,
        "mean_accuracy": merged_df.groupby("approach")["test_balanced_accuracy"].mean().to_dict(),
        "mean_time": merged_df.groupby("approach")["optimization_time"].mean().to_dict(),
    }

    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
