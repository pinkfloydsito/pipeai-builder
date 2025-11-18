"""
Example showing manual preprocessing + LLM AutoML.
YOU control the preprocessing, LLM handles the pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from llm_automl import (
    extract_dataset_info,
    PipelineDesigner,
    PipelineExecutor,
    evaluate_pipeline
)
import openml


def custom_preprocessing(X, y):
    """
    YOUR custom preprocessing logic.
    This is where you apply domain knowledge.
    """
    print("\n📋 Applying custom preprocessing...")

    # Example 1: Handle missing values YOUR way
    print("   • Handling missing values with domain logic")
    for col in X.columns:
        if X[col].isna().sum() > 0:
            if X[col].dtype == 'object':
                X[col].fillna('UNKNOWN', inplace=True)
            else:
                X[col].fillna(X[col].median(), inplace=True)

    # Example 2: YOUR feature engineering
    print("   • Creating custom features")
    if 'age' in X.columns and 'income' in X.columns:
        X['age_income_ratio'] = X['age'] / (X['income'] + 1)

    # Example 3: YOUR outlier handling
    print("   • Removing outliers with custom logic")
    # Custom outlier removal logic here

    # Example 4: YOUR categorical encoding
    print("   • Encoding categoricals")
    cat_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_columns, drop_first=True)

    print(f"   ✓ Preprocessing complete: {X.shape[1]} features")

    return X, y


def main():
    """Manual preprocessing + LLM AutoML example."""

    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY environment variable")

    print("=" * 80)
    print("Manual Preprocessing + LLM AutoML")
    print("=" * 80)

    # 1. Load raw data
    print("\n1. Loading dataset...")
    dataset = openml.datasets.get_dataset(31)  # Credit-g
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    print(f"   Raw data: {X.shape}")

    # 2. YOUR CUSTOM PREPROCESSING (you control this!)
    X_clean, y_clean = custom_preprocessing(X, y)

    # 3. Extract dataset info AFTER preprocessing
    print("\n2. Extracting dataset characteristics...")
    # Create a temporary dataset with your preprocessed data
    dataset_info = extract_dataset_info(31)  # Use for context

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )

    # 5. LLM designs the ML pipeline (NOT preprocessing!)
    print("\n3. LLM designing ML pipeline...")
    designer = PipelineDesigner(api_key=api_key, provider="deepseek")
    design = designer.design_pipeline(dataset_info)

    print(f"   Model selected: {design.model.type}")
    print(f"   LLM's reasoning: {design.rationale[:150]}...")

    # 6. Execute
    print("\n4. Executing pipeline...")
    executor = PipelineExecutor()
    pipeline = executor.build_pipeline(design)
    pipeline.fit(X_train, y_train)

    # 7. Evaluate
    print("\n5. Evaluating...")
    results = evaluate_pipeline(
        pipeline, X_train, y_train, X_test, y_test, design, 0.0
    )

    print(f"\n{results}")

    print("\n" + "=" * 80)
    print("Key Takeaway:")
    print("  • YOU handled preprocessing (domain knowledge)")
    print("  • LLM handled ML pipeline (model selection, hyperparameters)")
    print("  • Best of both worlds!")
    print("=" * 80)


if __name__ == "__main__":
    main()
