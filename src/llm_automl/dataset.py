"""
Dataset characterization and loading utilities.
Extracts meta-features for LLM context.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import openml


@dataclass
class DatasetInfo:
    """Dataset characteristics for LLM context."""

    id: int
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    feature_types: Dict[str, int]  # {categorical: X, numerical: Y}
    missing_value_pct: float
    class_imbalance_ratio: float
    domain: str
    description: str

    def to_llm_context(self) -> str:
        """Format dataset info for LLM consumption."""
        return f"""Dataset Characteristics:
- Name: {self.name}
- Samples: {self.n_samples:,} | Features: {self.n_features} | Classes: {self.n_classes}
- Feature Types: {self.feature_types['numerical']} numerical, {self.feature_types['categorical']} categorical
- Missing Values: {self.missing_value_pct:.1f}%
- Class Balance: {self.class_imbalance_ratio:.2f} (1.0 = perfect balance)
- Domain: {self.domain}
- Description: {self.description}"""


def extract_dataset_info(dataset_id: int) -> DatasetInfo:
    """
    Extract comprehensive dataset information from OpenML.

    Args:
        dataset_id: OpenML dataset ID

    Returns:
        DatasetInfo object with all characteristics
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )

    # Calculate characteristics
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    feature_types = {
        'categorical': sum(categorical_indicator),
        'numerical': n_features - sum(categorical_indicator)
    }

    missing_pct = (X.isna().sum().sum() / (n_samples * n_features)) * 100

    # Class imbalance (lower = more imbalanced)
    class_counts = np.bincount(y.astype(int))
    imbalance_ratio = class_counts.min() / class_counts.max()

    return DatasetInfo(
        id=dataset_id,
        name=dataset.name,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        feature_types=feature_types,
        missing_value_pct=missing_pct,
        class_imbalance_ratio=imbalance_ratio,
        domain=dataset.tag.split(',')[0] if dataset.tag else 'general',
        description=dataset.description[:200] if dataset.description else ""
    )


def load_dataset(dataset_id: int):
    """
    Load dataset from OpenML.

    Args:
        dataset_id: OpenML dataset ID

    Returns:
        Tuple of (X, y, dataset_info)
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    dataset_info = extract_dataset_info(dataset_id)

    return X, y, dataset_info
