"""
Benchmark: Optuna HPO vs LLM Designer
Compares hyperparameter optimization approaches on multiple datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import time
import warnings
import optuna
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from llm_automl.dataset import load_dataset, extract_dataset_info
from llm_automl.designer import PipelineDesigner
from llm_automl.executor import PipelineExecutor
from llm_automl.metrics import evaluate_pipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class BenchmarkResult:
    """Results for a single approach on a dataset."""

    dataset_id: int
    dataset_name: str
    approach: str  # 'baseline', 'optuna', 'llm'
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
    n_trials: int = 0  # Only for Optuna

    # Pipeline design (for LLM and detailed tracking)
    feature_engineering: Optional[List[Dict[str, Any]]] = None
    rationale: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for DataFrame."""
        result = asdict(self)
        # Convert complex types to JSON strings for CSV compatibility
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
            "performance": {
                "cv_accuracy": f"{self.cv_accuracy_mean:.4f} ± {self.cv_accuracy_std:.4f}",
                "test_accuracy": self.test_balanced_accuracy,
                "test_f1_macro": self.test_f1_macro,
            },
            "timing": {
                "optimization_seconds": self.optimization_time,
                "training_seconds": self.training_time,
                "n_trials": self.n_trials,
            },
        }
        return json.dumps(pipeline, indent=2)


class OptunaOptimizer:
    """Optimize hyperparameters using Optuna."""

    def __init__(self, model_type: str = "RandomForestClassifier", n_trials: int = 50):
        self.model_type = model_type
        self.n_trials = n_trials
        self.study = None

    def _get_search_space(self, trial, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter search space for different models."""

        if model_type == "RandomForestClassifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": 42,
            }
        elif model_type == "GradientBoostingClassifier":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_state": 42,
            }
        elif model_type == "LogisticRegression":
            return {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "solver": "saga",
                "max_iter": 1000,
                "random_state": 42,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def optimize(self, X_train, y_train) -> Dict[str, Any]:
        """Run Optuna optimization."""

        def objective(trial):
            params = self._get_search_space(trial, self.model_type)

            if self.model_type == "RandomForestClassifier":
                model = RandomForestClassifier(**params)
            elif self.model_type == "GradientBoostingClassifier":
                model = GradientBoostingClassifier(**params)
            elif self.model_type == "LogisticRegression":
                model = LogisticRegression(**params)

            # 5-fold cross-validation
            score = cross_val_score(
                model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1
            ).mean()
            return score

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        return self.study.best_params


class BenchmarkRunner:
    """Run comprehensive benchmark across datasets."""

    def __init__(self, api_key: str, n_optuna_trials: int = 50):
        self.api_key = api_key
        self.n_optuna_trials = n_optuna_trials
        self.results: List[BenchmarkResult] = []

    def run_baseline(
        self,
        dataset_id: int,
        dataset_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type: str = "RandomForestClassifier",
    ) -> BenchmarkResult:
        """Run baseline without HPO."""
        print(f"  Running baseline ({model_type})...")

        # Default hyperparameters
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=42)
            default_params = {"random_state": 42}
        elif model_type == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(random_state=42)
            default_params = {"random_state": 42}
        elif model_type == "LogisticRegression":
            model = LogisticRegression(max_iter=1000, random_state=42)
            default_params = {"max_iter": 1000, "random_state": 42}

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
            model_type=model_type,
            hyperparameters=default_params,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            test_f1_macro=f1_score(y_test, y_pred, average="macro"),
            test_f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            optimization_time=0.0,
            training_time=train_time,
            n_trials=0,
        )

    def run_optuna(
        self,
        dataset_id: int,
        dataset_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type: str = "RandomForestClassifier",
    ) -> BenchmarkResult:
        """Run Optuna HPO."""
        print(f"  Running Optuna HPO ({model_type})...")

        # Optimize hyperparameters
        optimizer = OptunaOptimizer(model_type=model_type, n_trials=self.n_optuna_trials)

        start_time = time.time()
        best_params = optimizer.optimize(X_train, y_train)
        optim_time = time.time() - start_time

        # Train with best params
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier(**best_params)
        elif model_type == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(**best_params)
        elif model_type == "LogisticRegression":
            model = LogisticRegression(**best_params)

        # Cross-validation with best params
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1
        )

        # Train and evaluate
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)

        return BenchmarkResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            approach="optuna",
            model_type=model_type,
            hyperparameters=best_params,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
            test_accuracy=accuracy_score(y_test, y_pred),
            test_balanced_accuracy=balanced_accuracy_score(y_test, y_pred),
            test_f1_macro=f1_score(y_test, y_pred, average="macro"),
            test_f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            optimization_time=optim_time,
            training_time=train_time,
            n_trials=self.n_optuna_trials,
        )

    def run_llm(
        self, dataset_id: int, dataset_name: str, X_train, y_train, X_test, y_test
    ) -> BenchmarkResult:
        """Run LLM Designer."""
        print(f"  Running LLM Designer...")

        # Get dataset info for LLM
        dataset_info = extract_dataset_info(dataset_id)

        # Design pipeline with LLM
        designer = PipelineDesigner(api_key=self.api_key)

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
            n_trials=1,  # LLM makes one decision
            feature_engineering=feature_engineering,
            rationale=design.rationale,
        )

    def run_on_dataset(
        self, dataset_id: int, model_type: str = "RandomForestClassifier"
    ) -> List[BenchmarkResult]:
        """Run all approaches on a single dataset."""

        # Load dataset
        print(f"\nDataset {dataset_id}:")
        X, y, dataset_info = load_dataset(dataset_id)
        print(f"  Name: {dataset_info.name}")
        print(
            f"  Samples: {dataset_info.n_samples}, Features: {dataset_info.n_features}, Classes: {dataset_info.n_classes}"
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = []

        # Baseline
        try:
            result = self.run_baseline(
                dataset_id,
                dataset_info.name,
                X_train,
                y_train,
                X_test,
                y_test,
                model_type=model_type,
            )
            results.append(result)
            print(f"    ✓ Baseline: {result.test_balanced_accuracy:.4f}")
        except Exception as e:
            print(f"    ✗ Baseline failed: {e}")

        # Optuna
        try:
            result = self.run_optuna(
                dataset_id,
                dataset_info.name,
                X_train,
                y_train,
                X_test,
                y_test,
                model_type=model_type,
            )
            results.append(result)
            print(
                f"    ✓ Optuna: {result.test_balanced_accuracy:.4f} (time: {result.optimization_time:.1f}s)"
            )
        except Exception as e:
            print(f"    ✗ Optuna failed: {e}")

        # LLM Designer
        try:
            result = self.run_llm(dataset_id, dataset_info.name, X_train, y_train, X_test, y_test)
            results.append(result)
            print(
                f"    ✓ LLM: {result.test_balanced_accuracy:.4f} (time: {result.optimization_time:.1f}s)"
            )
        except Exception as e:
            print(f"    ✗ LLM failed: {e}")

        return results

    def run_benchmark(
        self, dataset_ids: List[int], output_dir: str = "benchmark_results"
    ) -> pd.DataFrame:
        """Run benchmark on multiple datasets."""
        print("=" * 80)
        print("BENCHMARK: Optuna HPO vs LLM Designer")
        print("=" * 80)

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
        df = pd.DataFrame([r.to_dict() for r in all_results])

        # Print summary
        self._print_summary(df)

        # Print pipeline designs (for LLM approach)
        self._print_pipeline_designs(all_results)

        return df

    def _save_pipeline_json(self, result: BenchmarkResult, output_dir: Path):
        """Save pipeline design to JSON file."""
        filename = f"{result.dataset_name}_{result.approach}_{result.model_type}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            f.write(result.to_pipeline_json())

        print(f"    📄 Saved pipeline: {filename}")

    def _print_pipeline_designs(self, results: List[BenchmarkResult]):
        """Print pipeline designs for LLM approach."""
        print("\n" + "=" * 80)
        print("PIPELINE DESIGNS (LLM)")
        print("=" * 80)

        llm_results = [r for r in results if r.approach == "llm"]

        for result in llm_results:
            print(f"\n{result.dataset_name} ({result.dataset_id}):")
            print(f"  Model: {result.model_type}")
            print(f"  Hyperparameters: {json.dumps(result.hyperparameters, indent=4)}")

            if result.feature_engineering:
                print(f"  Feature Engineering:")
                for step in result.feature_engineering:
                    print(f"    - {step['name']}: {step['operation']}")
                    if step["parameters"]:
                        print(f"      Params: {step['parameters']}")
            else:
                print(f"  Feature Engineering: None")

            print(f"  Rationale: {result.rationale[:200]}...")
            print(f"  Performance: {result.test_balanced_accuracy:.4f}")

    def _print_summary(self, df: pd.DataFrame):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Group by approach
        summary = (
            df.groupby("approach")
            .agg(
                {
                    "test_balanced_accuracy": ["mean", "std"],
                    "test_f1_macro": "mean",
                    "optimization_time": "mean",
                    "training_time": "mean",
                }
            )
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
        win_counts = []
        for dataset_name in df["dataset_name"].unique():
            dataset_results = df[df["dataset_name"] == dataset_name]
            winner = dataset_results.loc[
                dataset_results["test_balanced_accuracy"].idxmax(), "approach"
            ]
            win_counts.append(winner)

        print("\n\nOverall Win Counts:")
        for approach in ["baseline", "optuna", "llm"]:
            count = win_counts.count(approach)
            print(f"  {approach:10s}: {count}/{len(win_counts)}")

        # Time comparison
        print("\n\nTime Comparison (mean):")
        for approach in df["approach"].unique():
            approach_df = df[df["approach"] == approach]
            total_time = (
                approach_df["optimization_time"].mean() + approach_df["training_time"].mean()
            )
            print(
                f"  {approach:10s}: {total_time:.2f}s (opt: {approach_df['optimization_time'].mean():.2f}s, train: {approach_df['training_time'].mean():.2f}s)"
            )


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Optuna vs LLM Designer")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=[61, 101, 40984],  # Iris, Vehicle, Digits
        help="OpenML dataset IDs",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results and pipeline JSONs",
    )

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(api_key=args.api_key, n_optuna_trials=args.n_trials)

    results_df = runner.run_benchmark(args.datasets, output_dir=args.output_dir)

    # Save results CSV
    output_path = Path(args.output_dir) / Path(args.output).name
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results CSV saved to {output_path}")
    print(f"✓ Pipeline JSONs saved to {Path(args.output_dir) / 'pipelines'}")


if __name__ == "__main__":
    main()
