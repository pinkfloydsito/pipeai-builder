# LLM-AutoML Benchmark

A research project comparing LLM-generated machine learning pipelines against traditional AutoML methods (TPOT) for classification tasks.

## Experiment Overview

This benchmark evaluates three approaches on 10 OpenML classification datasets:

1. **Baseline**: Default RandomForestClassifier with 100 estimators
2. **TPOT**: Genetic programming-based AutoML using the linear-light search space
3. **LLM**: Pipeline designed by a large language model (DeepSeek or OpenAI)

The experiment measures balanced accuracy, F1 scores, and optimization time for each approach.

## Installation

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install tpot
```

## Running the Benchmark

### Basic Usage

```bash
python benchmarks/benchmark_automl_vs_llm.py \
    --api-key YOUR_API_KEY \
    --datasets 37 44 61 1462 1464 1468 1475 1485 1487 1489 \
    --provider deepseek
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | Required | API key for the LLM provider |
| `--datasets` | 37 61 | Space-separated list of OpenML dataset IDs |
| `--provider` | deepseek | LLM provider: `deepseek` or `openai` |
| `--tpot-generations` | 5 | Number of TPOT generations |
| `--tpot-population` | 20 | TPOT population size |
| `--tpot-time` | 5 | TPOT maximum time in minutes |
| `--output-dir` | benchmark_results | Directory for output files |

### Full Replication

To replicate the full experiment with all 10 datasets:

```bash
python benchmarks/benchmark_automl_vs_llm.py \
    --api-key YOUR_API_KEY \
    --datasets 37 44 61 1462 1464 1468 1475 1485 1487 1489 \
    --provider deepseek \
    --tpot-time 5 \
    --output-dir benchmark_results
```

For OpenAI (ChatGPT-generated pipelines):

```bash
python benchmarks/benchmark_automl_vs_llm.py \
    --api-key YOUR_OPENAI_API_KEY \
    --datasets 37 44 61 1462 1464 1468 1475 1485 1487 1489 \
    --provider openai \
    --output-dir benchmark_results_openai
```

## Output

The benchmark generates the following outputs in the specified output directory:

- `benchmark_results.csv`: Complete results with all metrics for each dataset and approach
- `pipelines/`: Directory containing JSON files with pipeline configurations for each run

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploring and analyzing results:

- `gpt_generated_pipeline.ipynb`: Demonstrates the GPT-generated pipeline approach and includes detailed analysis of results
- `statistical_testing.ipynb`: Statistical tests comparing the different AutoML approaches

These notebooks provide an interactive way to explore the benchmark results and reproduce the analysis.

### Metrics

Each run records:
- Cross-validation accuracy (mean and standard deviation)
- Test accuracy
- Balanced accuracy
- F1 score (macro and weighted)
- Optimization time
- Training time

## Datasets

The benchmark uses the following OpenML dataset IDs:

| ID | Name |
|----|------|
| 37 | diabetes |
| 44 | spambase |
| 61 | iris |
| 1462 | banknote-authentication |
| 1464 | blood-transfusion |
| 1468 | cnae-9 |
| 1475 | first-order-theorem |
| 1485 | madelon |
| 1487 | ozone-level |
| 1489 | phoneme |

## Requirements

- Python 3.8+
- scikit-learn
- tpot
- openml
- pandas
- numpy
- API key for DeepSeek or OpenAI
