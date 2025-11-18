# Pipe-AI Builder: Research Project

An open-source project comparing LLM-driven AutoML against genetic programming (TPOT) for classification tasks.

## Overview

This project explores using Large Language Models (specifically ChatGPT 4) to design machine learning pipelines, with structured JSON outputs and comprehensive evaluation against the TPOT baseline.

### Key Features

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Basic Usage

```python
from llm_automl_core import (
    extract_dataset_info,
    PipelineDesigner,
    PipelineExecutor
)

# 1. Load dataset
dataset_info = extract_dataset_info(dataset_id=61)  # Iris

# 2. Design pipeline
designer = PipelineDesigner(api_key="your-anthropic-api-key")
design = designer.design_pipeline(dataset_info)

# 3. Execute pipeline
executor = PipelineExecutor()
pipeline = executor.build_sklearn_pipeline(design)

# 4. Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Run Benchmark

```bash
# Quick test on 2 datasets
python experiment_runner.py \
  --api-key YOUR_API_KEY \
  --datasets 61 101 \
  --seeds 3 \
  --quick

# Full benchmark
python experiment_runner.py \
  --api-key YOUR_API_KEY \
  --datasets 61 101 1590 40984 \
  --seeds 5 \
  --output-dir ./results
```

## Project Structure

```
.
├── llm_automl_guidance.md      # Complete project guidance
├── llm_automl_core.py          # Core implementation
├── experiment_runner.py         # Benchmark runner
└── README.md                    # This file
```

### Core Components

**llm_automl_core.py**:
- `PipelineDesigner`: LLM-based pipeline design with structured output
- `PipelineExecutor`: Safe pipeline execution
- `MetricsCalculator`: Comprehensive evaluation metrics
- `TPOTComparator`: Baseline comparison

**experiment_runner.py**:
- `BenchmarkRunner`: Full experimental protocol
- `ExperimentConfig`: Configuration management
- Statistical analysis and reporting

## Research Foundation

### Key Papers

1. **AutoML-Agent** (Trirat et al., 2024) - Multi-agent LLM framework
2. **IMPROVE** (2025) - Iterative refinement strategy
3. **TPOT** (Olson & Moore, 2016) - Genetic programming baseline

### Methodology

- **LLM Approach**: Few-shot learning with structured JSON schemas
- **Baseline**: TPOT with genetic programming
- **Datasets**: 20-30 diverse OpenML classification tasks
- **Metrics**: Balanced accuracy, F1, ROC-AUC, runtime, complexity
- **Statistics**: Wilcoxon signed-rank test, Cohen's d effect size

## Key Design Principles

### 1. Clean Code
```python
# Small, focused functions
def extract_dataset_info(dataset_id: int) -> DatasetInfo:
    """Single responsibility: extract dataset characteristics."""
    pass

# Type hints everywhere
def design_pipeline(
    dataset_info: DatasetInfo,
    n_examples: int = 3
) -> PipelineDesign:
    """Clear contracts via type hints."""
    pass
```

### 2. Structured Outputs
```python
class PipelineDesign(BaseModel):
    """Pydantic validation ensures reliability."""
    preprocessing_steps: List[PreprocessingStep]
    feature_engineering: List[FeatureEngineeringStep]
    model: ModelConfig
    rationale: str
```

### 3. Comprehensive Evaluation
```python
# Standard metrics
- Balanced Accuracy
- F1 Score (macro/weighted)
- ROC-AUC
- Cohen's Kappa

# LLM-specific metrics
- Pipeline coherence score
- Design-to-execution fidelity
- Reasoning quality
```

## Example Results Structure

```json
{
  "dataset_id": 61,
  "dataset_name": "iris",
  "llm_automl": {
    "balanced_accuracy": 0.96,
    "f1_macro": 0.95,
    "runtime_seconds": 12.3,
    "complexity": 4,
    "design_rationale": "StandardScaler + RandomForest..."
  },
  "tpot": {
    "balanced_accuracy": 0.94,
    "f1_macro": 0.93,
    "runtime_seconds": 180.5,
    "best_pipeline": "..."
  }
}
```

## Configuration Options

### LLM Settings
- `llm_model`: Model to use (default: claude-sonnet-4-20250514)
- `n_few_shot_examples`: Number of examples (default: 3)
- `temperature`: Creativity parameter (default: 0.7)

### TPOT Settings
- `generations`: GP generations (default: 50)
- `population_size`: Population per generation (default: 50)
- `max_time_mins`: Time limit (default: 60)
- `config_dict`: Operator set (default: 'TPOT light')

### Experiment Settings
- `n_seeds`: Repetitions per dataset (default: 3)
- `test_size`: Train/test split (default: 0.2)
- `cv_folds`: Cross-validation folds (default: 5)

## Extending the Project

### Add Custom Metrics
```python
class CustomMetrics:
    @staticmethod
    def your_metric(y_true, y_pred) -> float:
        """Implement custom evaluation metric."""
        pass
```

### Add Pipeline Components
```python
# Extend component map in PipelineExecutor
component_map = {
    'YourCustomTransformer': YourCustomTransformer,
    # ...
}
```

### Add Few-Shot Examples
```python
examples = [
    {
        "dataset_characteristics": {...},
        "successful_pipeline": {...},
        "performance": 0.95
    }
]
designer = PipelineDesigner(few_shot_examples=examples)
```

## Evaluation Protocol

### Benchmark Steps

1. **Dataset Selection**: 20-30 diverse OpenML datasets
2. **Multiple Seeds**: 3-5 runs per dataset
3. **Both Approaches**: LLM-AutoML and TPOT
4. **Statistical Testing**: Wilcoxon signed-rank, effect size
5. **Analysis**: Win/tie/loss, complementarity analysis

### Metrics Reported

**Primary**:
- Balanced Accuracy (main metric, handles imbalance)
- F1-Score (macro average)
- ROC-AUC (binary classification)
- Runtime (wall-clock time)
- Pipeline Complexity (number of components)

**Secondary**:
- Cohen's Kappa
- Matthews Correlation Coefficient
- Cross-validation scores (mean, std)

## Expected Outcomes

### Research Questions

1. **Performance**: When does LLM-AutoML match/exceed TPOT?
2. **Failure Modes**: What are common LLM errors?
3. **Few-Shot Impact**: How do examples affect performance?
4. **Complementarity**: Can we combine both approaches?
5. **Cost-Benefit**: API cost vs. computation time tradeoff

### Success Criteria

- ✅ Works reliably on 10+ datasets
- ✅ Produces valid pipelines >90% of time
- ✅ Competitive with TPOT on ≥30% of datasets
- ✅ Novel insights into LLM-driven AutoML
- ✅ Reproducible research artifact

## Troubleshooting

### Common Issues

**LLM returns invalid JSON**:
- Check prompt formatting
- Increase temperature if too deterministic
- Add more few-shot examples

**Pipeline execution fails**:
- Validate component compatibility
- Check for hallucinated components
- Review error logs

**TPOT timeout**:
- Reduce generations/population
- Use 'TPOT light' config
- Increase max_time_mins

## Contributing

This is an open-source research project. Contributions welcome:

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

### Areas for Contribution

- Additional evaluation metrics
- New dataset curation
- Prompt engineering improvements
- Visualization tools
- Documentation improvements

## Citation

If you use this work, please cite:

```bibtex
@software{llm_automl_2025,
  title={LLM-Driven AutoML: Comparing Language Models with Genetic Programming},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/llm-automl}
}
```

## References

### Key Papers

- **AutoML-Agent**: Trirat et al. (2024) - arXiv:2410.02958
- **IMPROVE**: 2025 - arXiv:2502.18530  
- **TPOT**: Olson & Moore (2016) - AutoML Book Chapter 8
- **AutoM3L**: Luo et al. (2024) - arXiv:2408.00665

### Resources

- [OpenML](https://www.openml.org/) - Dataset repository
- [TPOT Documentation](http://epistasislab.github.io/tpot/)
- [Anthropic API](https://docs.anthropic.com/)
- [AutoML.org](https://www.automl.org/)

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: @yourusername

---

**Status**: Research project in development  
**Version**: 1.0.0  
**Last Updated**: November 2025
