"""
LLM-AutoML: Automated Machine Learning with Language Models

A minimal, clean library for designing ML pipelines using LLMs.
Users handle preprocessing manually, while LLM handles pipeline design and HPO.
"""

__version__ = "0.1.0"

from .schemas import (
    FeatureEngineeringStep,
    ModelConfig,
    PipelineDesign
)
from .dataset import DatasetInfo, extract_dataset_info, load_dataset
from .designer import PipelineDesigner
from .executor import PipelineExecutor
from .metrics import EvaluationResults, evaluate_pipeline

__all__ = [
    # Schemas
    "FeatureEngineeringStep",
    "ModelConfig",
    "PipelineDesign",
    # Dataset
    "DatasetInfo",
    "extract_dataset_info",
    "load_dataset",
    # Designer
    "PipelineDesigner",
    # Executor
    "PipelineExecutor",
    # Metrics
    "EvaluationResults",
    "evaluate_pipeline",
]
