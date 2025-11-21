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
        """Construct prompt with system instructions and examples."""
        system_prompt = """You are an expert ML engineer designing AutoML pipelines.

IMPORTANT: Preprocessing is handled already. You only design:
1. Feature engineering (optional)
2. Model selection
3. Hyperparameters

Default preprocessing (already applied):
- Missing value imputation (mean for numeric, most_frequent for categorical)
- StandardScaler for numeric features
- OneHotEncoder for categorical features

Your task:
1. Design feature engineering steps if beneficial (polynomial features, feature selection, etc.)
2. Select the best model for the dataset characteristics
3. Provide sensible hyperparameters
4. Explain your reasoning

Respond with ONLY valid JSON:
{
  "feature_engineering": [{"name": "PolynomialFeatures", "operation": "polynomial", "parameters": {"degree": 2}}],
  "model": {"type": "RandomForestClassifier", "hyperparameters": {"n_estimators": 100, "max_depth": 10}},
  "rationale": "Explanation of feature engineering and model choices..."
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
