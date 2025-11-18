"""
Evaluation metrics and results tracking.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)

from .schemas import PipelineDesign


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""

    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    roc_auc: Optional[float]
    cohen_kappa: float
    matthews_corrcoef: float
    cv_scores: np.ndarray
    runtime_seconds: float
    pipeline_complexity: int

    def __str__(self) -> str:
        """Human-readable results."""
        return f"""Evaluation Results:
  Balanced Accuracy: {self.balanced_accuracy:.4f}
  F1 (macro): {self.f1_macro:.4f}
  F1 (weighted): {self.f1_weighted:.4f}
  Cohen's Kappa: {self.cohen_kappa:.4f}
  MCC: {self.matthews_corrcoef:.4f}
  CV: {self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}
  Runtime: {self.runtime_seconds:.2f}s
  Complexity: {self.pipeline_complexity}"""


def evaluate_pipeline(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    design: PipelineDesign,
    runtime: float
) -> EvaluationResults:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        pipeline: Fitted sklearn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        design: Pipeline design (for complexity)
        runtime: Total runtime in seconds

    Returns:
        EvaluationResults object
    """
    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities if available
    y_proba = None
    if hasattr(pipeline.named_steps['model'], 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    roc_auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring='balanced_accuracy'
    )

    return EvaluationResults(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        roc_auc=roc_auc,
        cohen_kappa=kappa,
        matthews_corrcoef=mcc,
        cv_scores=cv_scores,
        runtime_seconds=runtime,
        pipeline_complexity=design.get_complexity_score()
    )
