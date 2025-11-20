"""
Experiment runner for benchmarking LLM-AutoML.
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json

from sklearn.model_selection import train_test_split
import pandas as pd

from llm_automl import load_dataset, PipelineDesigner, PipelineExecutor, evaluate_pipeline
from .comparison import TPOTComparator


@dataclass
class ExperimentConfig:
    """Configuration for benchmark experiments."""

    dataset_ids: List[int]
    api_key: str
    llm_model: str = ""
    n_seeds: int = 3
    test_size: float = 0.2
    output_dir: str = "./results"
    run_tpot: bool = True
    tpot_generations: int = 50
    tpot_max_time_mins: int = 60


class BenchmarkRunner:
    """Run comprehensive benchmark experiments."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize benchmark runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("BenchmarkRunner")
        logger.setLevel(logging.INFO)

        # Console and file handlers
        ch = logging.StreamHandler()
        fh = logging.FileHandler(self.output_dir / "experiment.log")

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    def run_benchmark(self) -> pd.DataFrame:
        """
        Run complete benchmark on all datasets.

        Returns:
            DataFrame with all results
        """
        all_results = []

        for dataset_id in self.config.dataset_ids:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Dataset ID: {dataset_id}")
            self.logger.info(f"{'='*80}")

            try:
                results = self._run_dataset(dataset_id)
                all_results.extend(results)

                # Save intermediate
                self._save_results(all_results)

            except Exception as e:
                self.logger.error(f"Failed on dataset {dataset_id}: {e}")
                continue

        return pd.DataFrame(all_results)

    def _run_dataset(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Run experiments on single dataset with multiple seeds."""
        results = []

        # Load dataset
        X, y, dataset_info = load_dataset(dataset_id)
        self.logger.info(f"Dataset: {dataset_info.name}")

        # Run multiple seeds
        for seed in range(self.config.n_seeds):
            self.logger.info(f"Seed {seed + 1}/{self.config.n_seeds}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=seed, stratify=y
            )

            # Run LLM-AutoML
            llm_result = self._run_llm_automl(dataset_info, X_train, y_train, X_test, y_test)

            # Combine results
            result = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_info.name,
                "seed": seed,
                **{f"llm_{k}": v for k, v in llm_result.items()},
            }

            # Optionally run TPOT
            if self.config.run_tpot:
                tpot_result = self._run_tpot(X_train, y_train, X_test, y_test, seed)
                result.update({f"tpot_{k}": v for k, v in tpot_result.items()})

            results.append(result)
            self.logger.info(f"LLM Accuracy: {llm_result['balanced_accuracy']:.4f}")

        return results

    def _run_llm_automl(self, dataset_info, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Run LLM-AutoML pipeline."""
        designer = PipelineDesigner(api_key=self.config.api_key)
        executor = PipelineExecutor()

        # Design
        start_time = time.time()
        design = designer.design_pipeline(dataset_info)
        design_time = time.time() - start_time

        # Execute
        pipeline = executor.build_pipeline(design)
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate
        eval_results = evaluate_pipeline(
            pipeline, X_train, y_train, X_test, y_test, design, design_time + train_time
        )

        return {
            "balanced_accuracy": eval_results.balanced_accuracy,
            "f1_macro": eval_results.f1_macro,
            "f1_weighted": eval_results.f1_weighted,
            "runtime_seconds": eval_results.runtime_seconds,
            "complexity": eval_results.pipeline_complexity,
            "rationale": design.rationale,
        }

    def _run_tpot(self, X_train, y_train, X_test, y_test, seed: int) -> Dict[str, Any]:
        """Run TPOT baseline."""
        comparator = TPOTComparator(
            generations=self.config.tpot_generations, max_time_mins=self.config.tpot_max_time_mins
        )
        return comparator.run_baseline(X_train, y_train, X_test, y_test, seed)

    def _save_results(self, results: List[Dict]):
        """Save results to JSON."""
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Saved results to {output_file}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-AutoML benchmark")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--datasets", nargs="+", type=int, default=[61, 101])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--no-tpot", action="store_true", help="Skip TPOT comparison")

    args = parser.parse_args()

    config = ExperimentConfig(
        dataset_ids=args.datasets,
        api_key=args.api_key,
        n_seeds=args.seeds,
        output_dir=args.output_dir,
        run_tpot=not args.no_tpot,
    )

    runner = BenchmarkRunner(config)
    results_df = runner.run_benchmark()

    print(f"\nBenchmark complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
