# AutoML Experiments

This directory contains the experimental results and analysis notebooks for comparing different AutoML approaches on classification tasks using OpenML datasets.

## Overview

The experiments compare **4 different approaches** for automated machine learning pipeline generation:

1. **ChatGPT** - LLM-guided pipeline generation with hyperparameter optimization
2. **baseline** - Default RandomForest classifier (100 trees, no tuning)
3. **llm** - LLM-generated pipelines with custom feature engineering and model selection
4. **tpot** - TPOT evolutionary AutoML with 5-minute search budget

## Files

### `benchmark_results.csv`
Raw benchmark results containing performance metrics for all approaches across 10 OpenML datasets. Columns include:
- Dataset metadata (`dataset_id`, `dataset_name`)
- Approach identifier and model type
- Cross-validation metrics (`cv_accuracy_mean`, `cv_accuracy_std`)
- Test metrics (`test_accuracy`, `test_balanced_accuracy`, `test_f1_macro`, `test_f1_weighted`)
- Timing information (`optimization_time`, `training_time`)
- Hyperparameters and feature engineering details

### `results_combined.xlsx`
Combined results from all experiments in Excel format, used as input for statistical analysis.

### `gpt_generated_pipeline.ipynb`
Jupyter notebook implementing a complete AutoML pipeline with:
- **Data loading** from OpenML datasets
- **Preprocessing** with automatic handling of numerical/categorical features
- **Feature engineering** options (none or light log1p transformation)
- **Model search space** covering 9 classifiers (Logistic Regression, SVM variants, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, KNN, Naive Bayes)
- **Hyperparameter optimization** using random search with configurable budget
- **Evaluation** with stratified cross-validation and held-out test set

### `statistical_testing.ipynb`
Jupyter notebook for statistical comparison of the AutoML approaches:
- **Friedman test** to detect significant differences across approaches
- **Nemenyi post-hoc test** for pairwise comparisons (when Friedman is significant)
- **Visualizations** including rank heatmaps and mean rank bar plots

## Key Results

Based on the statistical analysis using `test_f1_weighted` as the metric:

| Approach | Mean Rank |
|----------|-----------|
| ChatGPT  | 2.15      |
| baseline | 2.35      |
| llm      | 2.75      |
| tpot     | 2.75      |

**Friedman Test Result**: p-value = 0.647 (not significant at α = 0.05)

The analysis indicates no statistically significant differences between the approaches across the 10 datasets tested.

## Datasets Used

The experiments were conducted on 10 OpenML classification datasets with varying characteristics:
- Sample sizes ranging from small to medium
- Mix of numerical and categorical features
- Different class imbalance ratios

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- openml
- scikit-posthocs (for Nemenyi test)
- matplotlib (for visualizations)
