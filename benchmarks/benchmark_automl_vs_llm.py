"""
Benchmark: AutoML (TPOT) vs LLM Designer
Compares full AutoML framework against LLM-driven pipeline design
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd
import time
import warnings
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier

from llm_automl.dataset import load_dataset, extract_dataset_info
from llm_automl.designer import PipelineDesigner
from llm_automl.executor import PipelineExecutor

warnings.filterwarnings("ignore")


@dataclass
class BenchmarkResult:
    """Results for a single approach on a dataset."""

    dataset_id: int
    dataset_name: str
    approach: str  # 'baseline', 'tpot', 'llm'
    model_type: str
    hyperparameters: Dict[str, Any]

    # Metrics
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_f1_macro: float
    test_f1_weighted: float

    # Performance
    optimization_time: float
    training_time: float

    # Pipeline design details
    feature_engineering: Optional[List[Dict[str, Any]]] = None
    rationale: Optional[str] = None
    pipeline_str: Optional[str] = None  # For TPOT

    def to_dict(self):
        """Convert to dictionary for DataFrame."""
        result = asdict(self)
        if self.hyperparameters:
            result["hyperparameters_json"] = json.dumps(self.hyperparameters)
        if self.feature_engineering:
            result["feature_engineering_json"] = json.dumps(self.feature_engineering)
        return result

    def to_pipeline_json(self) -> str:
        """Export pipeline design as JSON."""
        pipeline = {
            "dataset": {"id": self.dataset_id, "name": self.dataset_name},
            "approach": self.approach,
            "model": {"type": self.model_type, "hyperparameters": self.hyperparameters},
            "feature_engineering": self.feature_engineering or [],
            "rationale": self.rationale or "",
            "pipeline_str": self.pipeline_str or "",
            "performance": {
                "cv_accuracy": f"{self.cv_accuracy_mean:.4f} ± {self.cv_accuracy_std:.4f}",
                "test_accuracy": self.test_balanced_accuracy,
                "test_f1_macro": self.test_f1_macro,
            },
            "timing": {
                "optimization_seconds": self.optimization_time,
                "training_seconds": self.training_time,
            },
        }
        return json.dumps(pipeline, indent=2)


class BenchmarkRunner:
    """Run comprehensive benchmark across datasets."""

    def __init__(
        self,
        api_key: str,
        tpot_generations: int = 5,
        tpot_population_size: int = 20,
        tpot_max_time_mins: int = 5,
        provider: str = "deepseek",
    ):
        self.api_key = api_key
        self.tpot_generations = tpot_generations
        self.tpot_population_size = tpot_population_size
        self.tpot_max_time_mins = tpot_max_time_mins
        self.provider = provider
        self.results: List[BenchmarkResult] = []

    def run_baseline(
        self,
        dataset_id: int,
        dataset_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        """Run baseline with default RandomForest."""
        print(f"  Running baseline (RandomForest default)...")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        default_params = {"n_estimators": 100, "random_state": 42}

        # Cross-validation
        start_time = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1
        )

        # Train and evaluate
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        return BenchmarkResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            approach="baseline",
            model_type="RandomForestClassifier",
            hyperparameters=default_params,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            test_f1_macro=f1_score(y_test, y_pred, average="macro"),
            test_f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            optimization_time=0.0,
            training_time=train_time,
            rationale="Default RandomForest with 100 trees, no hyperparameter tuning",
        )

    def run_tpot(
        self,
        dataset_id: int,
        dataset_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        """Run TPOT AutoML."""
        print(f"  Running TPOT AutoML (max_time: {self.tpot_max_time_mins}min)...")

        # TPOT 1.1.0 API - use n_jobs=1 to avoid Dask distributed issues
        tpot = TPOTClassifier(
            search_space="linear-light",  # Faster search space
            scorers=["balanced_accuracy"],
            cv=5,
            max_time_mins=self.tpot_max_time_mins,
            max_eval_time_mins=1,
            random_state=42,
            verbose=0,
            n_jobs=1,  # Single-threaded to avoid Dask issues
        )

        start_time = time.time()
        tpot.fit(X_train, y_train)
        optim_time = time.time() - start_time

        # Get best pipeline info
        pipeline_str = str(tpot.fitted_pipeline_) if hasattr(tpot, 'fitted_pipeline_') else "N/A"

        # Try to extract model type from pipeline
        model_type = "TPOTPipeline"
        if hasattr(tpot, 'fitted_pipeline_') and tpot.fitted_pipeline_ is not None:
            try:
                # Get the last step (usually the classifier)
                if hasattr(tpot.fitted_pipeline_, 'steps'):
                    steps = tpot.fitted_pipeline_.steps
                    if steps:
                        model_type = type(steps[-1][1]).__name__
                else:
                    model_type = type(tpot.fitted_pipeline_).__name__
            except Exception:
                pass

        # Cross-validation scores
        cv_scores = cross_val_score(
            tpot.fitted_pipeline_, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1
        )

        # Evaluate
        start_time = time.time()
        y_pred = tpot.predict(X_test)
        train_time = time.time() - start_time

        return BenchmarkResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            approach="tpot",
            model_type=model_type,
            hyperparameters={"search_space": "linear-light", "max_time_mins": self.tpot_max_time_mins},
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            test_f1_macro=f1_score(y_test, y_pred, average="macro"),
            test_f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            optimization_time=optim_time,
            training_time=train_time,
            pipeline_str=pipeline_str,
            rationale=f"TPOT selected {model_type} after {optim_time:.1f}s evolutionary search",
        )

    def run_llm(
        self,
        dataset_id: int,
        dataset_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
    ) -> BenchmarkResult:
        """Run LLM Designer."""
        print(f"  Running LLM Designer...")

        # Get dataset info for LLM
        dataset_info = extract_dataset_info(dataset_id)

        # Design pipeline with LLM
        designer = PipelineDesigner(api_key=self.api_key, provider=self.provider)

        start_time = time.time()
        design = designer.design_pipeline(dataset_info, n_examples=0)
        design_time = time.time() - start_time

        # Build and train pipeline
        executor = PipelineExecutor()
        pipeline = executor.build_pipeline(design)

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1
        )

        # Train and evaluate
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)

        # Extract feature engineering steps
        feature_engineering = [
            {"name": step.name, "operation": step.operation, "parameters": step.parameters}
            for step in design.feature_engineering
        ]

        return BenchmarkResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            approach="llm",
            model_type=design.model.type,
            hyperparameters=design.model.hyperparameters,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            test_f1_macro=f1_score(y_test, y_pred, average="macro"),
            test_f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            optimization_time=design_time,
            training_time=train_time,
            feature_engineering=feature_engineering,
            rationale=design.rationale,
        )

    def run_on_dataset(self, dataset_id: int) -> List[BenchmarkResult]:
        """Run all approaches on a single dataset."""

        # Load dataset
        print(f"\nDataset {dataset_id}:")
        X, y, dataset_info = load_dataset(dataset_id)
        print(f"  Name: {dataset_info.name}")
        print(
            f"  Samples: {dataset_info.n_samples}, Features: {dataset_info.n_features}, Classes: {dataset_info.n_classes}"
        )

        # Encode labels to integers (needed for TPOT compatibility)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Standardized train/test split (same for all approaches)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        results = []

        # Baseline
        try:
            result = self.run_baseline(
                dataset_id, dataset_info.name, X_train, y_train, X_test, y_test
            )
            results.append(result)
            print(f"    ✓ Baseline: {result.test_balanced_accuracy:.4f}")
        except Exception as e:
            print(f"    ✗ Baseline failed: {e}")

        # TPOT AutoML
        try:
            result = self.run_tpot(
                dataset_id, dataset_info.name, X_train, y_train, X_test, y_test
            )
            results.append(result)
            print(
                f"    ✓ TPOT: {result.test_balanced_accuracy:.4f} ({result.model_type}, time: {result.optimization_time:.1f}s)"
            )
        except Exception as e:
            print(f"    ✗ TPOT failed: {e}")
            import traceback
            traceback.print_exc()

        # LLM Designer
        try:
            result = self.run_llm(
                dataset_id, dataset_info.name, X_train, y_train, X_test, y_test
            )
            results.append(result)
            print(
                f"    ✓ LLM: {result.test_balanced_accuracy:.4f} ({result.model_type}, time: {result.optimization_time:.1f}s)"
            )
        except Exception as e:
            print(f"    ✗ LLM failed: {e}")
            import traceback
            traceback.print_exc()

        return results

    def run_benchmark(
        self, dataset_ids: List[int], output_dir: str = "benchmark_results"
    ) -> pd.DataFrame:
        """Run benchmark on multiple datasets."""
        print("=" * 80)
        print("BENCHMARK: AutoML (TPOT) vs LLM Designer")
        print("=" * 80)
        print(f"\nTPOT config: generations={self.tpot_generations}, population={self.tpot_population_size}, max_time={self.tpot_max_time_mins}min")
        print(f"Datasets: {dataset_ids}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        pipelines_dir = output_path / "pipelines"
        pipelines_dir.mkdir(exist_ok=True)

        all_results = []

        for dataset_id in dataset_ids:
            try:
                results = self.run_on_dataset(dataset_id)
                all_results.extend(results)

                # Save pipeline designs to JSON
                for result in results:
                    self._save_pipeline_json(result, pipelines_dir)

            except Exception as e:
                print(f"  ✗ Dataset {dataset_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Store results
        self.results = all_results

        # Convert to DataFrame
        if all_results:
            df = pd.DataFrame([r.to_dict() for r in all_results])
        else:
            df = pd.DataFrame()

        # Print summary
        self._print_summary(df)

        # Print pipeline designs
        self._print_pipeline_designs(all_results)

        return df

    def _save_pipeline_json(self, result: BenchmarkResult, output_dir: Path):
        """Save pipeline design to JSON file."""
        filename = f"{result.dataset_name}_{result.approach}_{result.model_type}.json"
        # Clean filename
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            f.write(result.to_pipeline_json())

        print(f"    📄 Saved: {filename}")

    def _print_pipeline_designs(self, results: List[BenchmarkResult]):
        """Print pipeline designs."""
        print("\n" + "=" * 80)
        print("PIPELINE DESIGNS")
        print("=" * 80)

        for result in results:
            print(f"\n{result.dataset_name} - {result.approach.upper()}:")
            print(f"  Model: {result.model_type}")

            if result.approach == "tpot" and result.pipeline_str:
                print(f"  Full Pipeline: {result.pipeline_str[:200]}...")
            else:
                print(f"  Hyperparameters: {json.dumps(result.hyperparameters, indent=4)}")

            if result.feature_engineering:
                print(f"  Feature Engineering:")
                for step in result.feature_engineering:
                    print(f"    - {step['name']}: {step['operation']}")

            if result.rationale:
                rationale_preview = result.rationale[:200] + "..." if len(result.rationale) > 200 else result.rationale
                print(f"  Rationale: {rationale_preview}")

            print(f"  Performance: {result.test_balanced_accuracy:.4f}")

    def _print_summary(self, df: pd.DataFrame):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if df.empty:
            print("No results to summarize")
            return

        # Group by approach
        summary = (
            df.groupby("approach")
            .agg({
                "test_balanced_accuracy": ["mean", "std"],
                "test_f1_macro": "mean",
                "optimization_time": "mean",
                "training_time": "mean",
            })
            .round(4)
        )

        print("\nPerformance by Approach:")
        print(summary)

        # Win counts
        print("\n\nWin Counts (Best Balanced Accuracy per Dataset):")
        for dataset_name in df["dataset_name"].unique():
            dataset_results = df[df["dataset_name"] == dataset_name]
            winner = dataset_results.loc[dataset_results["test_balanced_accuracy"].idxmax()]
            print(
                f"  {dataset_name:20s}: {winner['approach']:10s} ({winner['test_balanced_accuracy']:.4f})"
            )

        # Overall winner
        win_counts = {"baseline": 0, "tpot": 0, "llm": 0}
        for dataset_name in df["dataset_name"].unique():
            dataset_results = df[df["dataset_name"] == dataset_name]
            winner = dataset_results.loc[dataset_results["test_balanced_accuracy"].idxmax(), "approach"]
            win_counts[winner] = win_counts.get(winner, 0) + 1

        print("\n\nOverall Win Counts:")
        total = len(df["dataset_name"].unique())
        for approach, count in win_counts.items():
            print(f"  {approach:10s}: {count}/{total}")

        # Time comparison
        print("\n\nTime Comparison (mean):")
        for approach in df["approach"].unique():
            approach_df = df[df["approach"] == approach]
            total_time = approach_df["optimization_time"].mean() + approach_df["training_time"].mean()
            print(
                f"  {approach:10s}: {total_time:.2f}s (opt: {approach_df['optimization_time'].mean():.2f}s, train: {approach_df['training_time'].mean():.2f}s)"
            )


def main():
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark AutoML (TPOT) vs LLM Designer")
    parser.add_argument("--api-key", required=True, help="API key for LLM")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=[61, 101],  # Iris, Vehicle
        help="OpenML dataset IDs",
    )
    parser.add_argument(
        "--tpot-generations",
        type=int,
        default=5,
        help="TPOT generations (default: 5)",
    )
    parser.add_argument(
        "--tpot-population",
        type=int,
        default=20,
        help="TPOT population size (default: 20)",
    )
    parser.add_argument(
        "--tpot-time",
        type=int,
        default=5,
        help="TPOT max time in minutes (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "openai"],
        help="LLM provider (default: deepseek)",
    )

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(
        api_key=args.api_key,
        tpot_generations=args.tpot_generations,
        tpot_population_size=args.tpot_population,
        tpot_max_time_mins=args.tpot_time,
        provider=args.provider,
    )

    results_df = runner.run_benchmark(args.datasets, output_dir=args.output_dir)

    # Save results CSV
    output_path = Path(args.output_dir) / "benchmark_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results CSV saved to {output_path}")
    print(f"✓ Pipeline JSONs saved to {Path(args.output_dir) / 'pipelines'}")


if __name__ == "__main__":
    main()
