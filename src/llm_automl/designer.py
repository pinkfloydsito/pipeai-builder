"""
LLM-based pipeline designer.
Supports DeepSeek and OpenAI APIs.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Literal

from .schemas import PipelineDesign
from .dataset import DatasetInfo


class PipelineDesigner:
    """Design ML pipelines using LLM with structured output."""

    def __init__(
        self,
        api_key: str,
        provider: Literal["deepseek", "openai"] = "deepseek",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        few_shot_examples: Optional[List[Dict]] = None,
    ):
        """
        Initialize pipeline designer.

        Args:
            api_key: API key for the provider
            provider: LLM provider ("deepseek", "openai")
            model: Model name (auto-selected if None)
            base_url: Custom base URL (for DeepSeek or other providers)
            few_shot_examples: Optional list of example pipeline designs
        """
        from openai import OpenAI

        self.provider = provider
        self.few_shot_examples = few_shot_examples or []
        self.logger = logging.getLogger(__name__)

        # Setup client based on provider
        if provider == "deepseek":
            self.client = OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com")
            self.model = model or "deepseek-chat"
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4"
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: deepseek, openai")

    def design_pipeline(self, dataset_info: DatasetInfo, n_examples: int = 3) -> PipelineDesign:
        """
        Design ML pipeline for given dataset.

        Args:
            dataset_info: Dataset characteristics
            n_examples: Number of few-shot examples to use

        Returns:
            Validated pipeline design
        """
        prompt = self._build_prompt(dataset_info, n_examples)
        self.logger.debug(f"prompt: {prompt}")
        response = self._call_llm(prompt)

        self.logger.debug(f"JSON response: {response}")
        design = PipelineDesign(**response)

        self.logger.info(
            f"Designed pipeline with complexity {design.get_complexity_score()} "
            f"for {dataset_info.name}"
        )

        return design

    def _build_prompt(self, dataset_info: DatasetInfo, n_examples: int) -> str:
        """Construct prompt with system instructions, algorithm search space, and examples."""
        system_prompt = """You are an expert ML engineer designing AutoML pipelines.
    
    IMPORTANT: Preprocessing is handled already. You only design:
    1. Feature engineering (optional)
    2. Model selection
    3. Hyperparameters
    
    Default preprocessing (already applied):
    - Missing value imputation (mean for numeric, most_frequent for categorical)
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    
    AVAILABLE ALGORITHMS SEARCH SPACE:
    ----------------------------------
    For Classification tasks:
    - LogisticRegression: {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2", "elasticnet", "none"], "solver": ["lbfgs", "liblinear", "saga"]}
    - RandomForestClassifier: {"n_estimators": [50, 100, 200, 500], "max_depth": [None, 3, 5, 10, 20], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2", None]}
    - GradientBoostingClassifier: {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10], "subsample": [0.8, 0.9, 1.0]}
    - SVC: {"C": [0.1, 1, 10, 100], "kernel": ["linear", "poly", "rbf", "sigmoid"], "gamma": ["scale", "auto", 0.001, 0.01, 0.1]}
    - XGBClassifier: {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 8, 10], "subsample": [0.6, 0.8, 1.0]}
    
    
    AVAILABLE FEATURE ENGINEERING OPERATIONS:
    -----------------------------------------
    - polynomial: Creates polynomial and interaction features (parameters: {"degree": [2, 3]})
    - feature_selection: Selects most important features (parameters: {"method": ["univariate", "model_based"], "k_features": ["auto", 0.8, 0.9, 10, 20]})
    - pca: Dimensionality reduction (parameters: {"n_components": ["auto", 0.95, 0.99, 10, 50]})
    - binning: Converts numerical features to categorical (parameters: {"n_bins": [3, 5, 10], "strategy": ["uniform", "quantile", "kmeans"]})
    - interaction_terms: Creates interaction terms between features (parameters: {})
    - no_feature_engineering: Use empty list [] if no feature engineering is beneficial
    
    YOUR TASK:
    1. Based on the dataset characteristics, decide if feature engineering would be beneficial
    2. Select the most appropriate model type for the problem
    3. Choose sensible hyperparameters from the search space above
    4. Explain your reasoning considering dataset size, feature types, and problem complexity
    
    Respond with ONLY valid JSON:
    {
      "feature_engineering": [
        {"name": "PolynomialFeatures", "operation": "polynomial", "parameters": {"degree": 2}},
        {"name": "SelectKBest", "operation": "feature_selection", "parameters": {"method": "univariate", "k_features": 10}}
      ],
      "model": {
        "type": "RandomForestClassifier", 
        "hyperparameters": {
          "n_estimators": 100, 
          "max_depth": 10,
          "max_features": "sqrt"
        }
      },
      "rationale": "I chose polynomial features because the dataset shows non-linear relationships. Random Forest was selected as it handles the medium-sized dataset well and is robust to outliers. Feature selection helps reduce noise from the 50 input features."
    }
    
    Note: feature_engineering can be an empty list [] if not needed."""

        examples_text = ""
        if self.few_shot_examples and n_examples > 0:
            examples_text = "\n\nExamples:\n"
            for i, ex in enumerate(self.few_shot_examples[:n_examples], 1):
                examples_text += f"\n{i}. {json.dumps(ex, indent=2)}\n"

        user_prompt = (
            f"{dataset_info.to_llm_context()}\n{examples_text}\n\nDesign optimal ML pipeline:"
        )

        return system_prompt + "\n\n" + user_prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API and parse JSON response."""
        try:
            # OpenAI-compatible API (DeepSeek, OpenAI, etc.)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content.strip()

            # Clean markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
