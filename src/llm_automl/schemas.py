"""
Data schemas for LLM-driven AutoML pipeline design.
All structured outputs validated with Pydantic.
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field, validator


class FeatureEngineeringStep(BaseModel):
    """Feature engineering step specification."""

    name: str
    operation: str = Field(..., description="polynomial, interaction, selection")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Model configuration and hyperparameters."""

    type: str = Field(..., description="Model type: RandomForest, XGBoost, etc.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)

    @validator('type')
    def validate_model_type(cls, v):
        """Ensure model type is supported."""
        valid_types = {
            'RandomForestClassifier', 'GradientBoostingClassifier',
            'LogisticRegression', 'SVC', 'XGBClassifier',
            'LGBMClassifier', 'MLPClassifier'
        }
        if v not in valid_types:
            raise ValueError(f"Model type must be one of {valid_types}")
        return v


class PipelineDesign(BaseModel):
    """Complete ML pipeline design from LLM."""

    feature_engineering: List[FeatureEngineeringStep] = Field(default_factory=list)
    model: ModelConfig
    rationale: str = Field(..., description="Reasoning for design choices")

    def get_complexity_score(self) -> int:
        """Calculate pipeline complexity metric."""
        return (
            len(self.feature_engineering) * 2 +
            1  # Model itself, preprocessing is fixed
        )
