"""
Hyperparameter optimization example.
Shows iterative improvement with LLM guidance.
"""

import os
import time
from sklearn.model_selection import train_test_split

from llm_automl import (
    load_dataset,
    PipelineDesigner,
    PipelineExecutor,
    evaluate_pipeline
)
from llm_automl.hpo import HyperparameterOptimizer


def main():
    """Run HPO example."""

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY environment variable")

    print("=" * 80)
    print("Hyperparameter Optimization with LLM")
    print("=" * 80)

    # 1. Load and split data
    print("\n1. Loading dataset...")
    X, y, dataset_info = load_dataset(dataset_id=61)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Get initial design
    print("\n2. Getting initial pipeline design...")
    designer = PipelineDesigner(api_key=api_key, provider="deepseek")
    initial_design = designer.design_pipeline(dataset_info)

    # 3. Evaluate initial design
    print("\n3. Evaluating initial design...")
    executor = PipelineExecutor()
    pipeline = executor.build_pipeline(initial_design)
    pipeline.fit(X_train, y_train)

    initial_results = evaluate_pipeline(
        pipeline, X_train, y_train, X_test, y_test,
        initial_design, 0.0
    )

    print(f"\n   Initial Performance:")
    print(f"   Balanced Accuracy: {initial_results.balanced_accuracy:.4f}")
    print(f"   F1 (macro): {initial_results.f1_macro:.4f}")

    # 4. Optimize hyperparameters
    print("\n4. Optimizing hyperparameters...")
    optimizer = HyperparameterOptimizer(api_key=api_key)

    improved_designs = optimizer.optimize(
        initial_design=initial_design,
        evaluation=initial_results,
        n_iterations=2  # Keep costs low for example
    )

    # 5. Evaluate improvements
    print("\n5. Evaluating improvements...")
    for i, design in enumerate(improved_designs[1:], 1):  # Skip initial
        print(f"\n   Iteration {i}:")
        pipeline = executor.build_pipeline(design)
        pipeline.fit(X_train, y_train)

        results = evaluate_pipeline(
            pipeline, X_train, y_train, X_test, y_test,
            design, 0.0
        )

        print(f"   Balanced Accuracy: {results.balanced_accuracy:.4f}")
        print(f"   F1 (macro): {results.f1_macro:.4f}")

        improvement = results.balanced_accuracy - initial_results.balanced_accuracy
        print(f"   Improvement: {improvement:+.4f}")

    print("\n" + "=" * 80)
    print("✓ HPO complete!")
    print("  LLM iteratively improved hyperparameters")
    print("=" * 80)


if __name__ == "__main__":
    main()
