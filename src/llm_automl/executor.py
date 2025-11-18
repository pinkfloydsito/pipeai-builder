"""
Pipeline executor - builds and runs sklearn pipelines from designs.
Uses a fixed default preprocessing pipeline for all models.
"""

import logging
from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .schemas import PipelineDesign


# Model registry
MODEL_MAP = {
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
}

# Feature engineering registry
FEATURE_ENG_MAP = {
    'PolynomialFeatures': PolynomialFeatures,
    'SelectKBest': SelectKBest,
}


class PipelineExecutor:
    """Execute designed pipelines with fixed preprocessing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_pipeline(self, design: PipelineDesign, categorical_features=None, numerical_features=None) -> Pipeline:
        """
        Build sklearn pipeline from design with default preprocessing.

        Args:
            design: Validated pipeline design
            categorical_features: List of categorical feature indices (optional)
            numerical_features: List of numerical feature indices (optional)

        Returns:
            sklearn Pipeline object with fixed preprocessing + LLM-selected feature eng + model
        """
        steps = []

        # Default preprocessing pipeline (same for all models)
        if categorical_features is not None and numerical_features is not None:
            # If we know the feature types, use ColumnTransformer
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            steps.append(('preprocessor', preprocessor))
        else:
            # If we don't know feature types, use simple numeric preprocessing
            steps.append(('imputer', SimpleImputer(strategy='mean')))
            steps.append(('scaler', StandardScaler()))

        # Add LLM-designed feature engineering steps
        for i, feat_step in enumerate(design.feature_engineering):
            component = self._create_feature_eng(feat_step.name, feat_step.parameters)
            steps.append((f"feat_{i}_{feat_step.name}", component))

        # Add LLM-designed model
        model = self._create_model(
            design.model.type,
            design.model.hyperparameters
        )
        steps.append(("model", model))

        self.logger.info(f"Built pipeline with fixed preprocessing + {len(design.feature_engineering)} feature eng steps + {design.model.type}")

        return Pipeline(steps)

    def _create_model(self, name: str, params: Dict[str, Any]):
        """
        Create sklearn model from name and parameters.

        Args:
            name: Model class name
            params: Initialization parameters

        Returns:
            Instantiated sklearn model

        Raises:
            ValueError: If model name is unknown
        """
        if name not in MODEL_MAP:
            raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_MAP.keys())}")

        return MODEL_MAP[name](**params)

    def _create_feature_eng(self, name: str, params: Dict[str, Any]):
        """
        Create sklearn feature engineering component from name and parameters.

        Args:
            name: Component class name
            params: Initialization parameters

        Returns:
            Instantiated sklearn component

        Raises:
            ValueError: If component name is unknown
        """
        if name not in FEATURE_ENG_MAP:
            raise ValueError(f"Unknown feature engineering: {name}. Available: {list(FEATURE_ENG_MAP.keys())}")

        return FEATURE_ENG_MAP[name](**params)


def add_model(name: str, model_class):
    """
    Register a new model type.

    Args:
        name: Model name
        model_class: sklearn-compatible class
    """
    MODEL_MAP[name] = model_class


def add_feature_engineering(name: str, component_class):
    """
    Register a new feature engineering component.

    Args:
        name: Component name
        component_class: sklearn-compatible class
    """
    FEATURE_ENG_MAP[name] = component_class
