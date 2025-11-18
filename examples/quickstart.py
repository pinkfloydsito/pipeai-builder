"""
Quickstart example for LLM-AutoML.
Shows basic usage with DeepSeek.
"""

import os
import time
from sklearn.model_selection import train_test_split

from llm_automl import load_dataset, PipelineDesigner, PipelineExecutor, evaluate_pipeline


def main():
    """Run quickstart example on Iris dataset."""

    # Get API key from environment
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY environment variable")

    print("=" * 80)
    print("LLM-AutoML Quickstart with DeepSeek")
    print("=" * 80)

    # 1. Load dataset
    print("\n1. Loading Iris dataset...")
    X, y, dataset_info = load_dataset(dataset_id=61)
    print(f"   Loaded: {dataset_info.name}")
    print(f"   Samples: {dataset_info.n_samples}, Features: {dataset_info.n_features}")

    # 2. Train/test split
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Design pipeline using DeepSeek
    print("\n3. Designing pipeline with DeepSeek...")
    designer = PipelineDesigner(api_key=api_key, provider="deepseek")

    start_time = time.time()
    design = designer.design_pipeline(dataset_info)
    design_time = time.time() - start_time

    print(f"   Design time: {design_time:.2f}s")
    print(f"   Model: {design.model.type}")
    print(f"   Preprocessing steps: {len(design.preprocessing_steps)}")
    print(f"\n   LLM's reasoning:")
    print(f"   {design.rationale[:200]}...")

    # 4. Execute pipeline
    print("\n4. Building and training pipeline...")
    executor = PipelineExecutor()
    pipeline = executor.build_pipeline(design)

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"   Train time: {train_time:.2f}s")

    # 5. Evaluate
    print("\n5. Evaluating...")
    results = evaluate_pipeline(
        pipeline, X_train, y_train, X_test, y_test, design, design_time + train_time
    )

    print(f"\n{results}")

    print("\n" + "=" * 80)
    print("✓ Quickstart complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
