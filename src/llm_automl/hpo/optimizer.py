"""
LLM-guided hyperparameter optimization.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import anthropic

from ..schemas import PipelineDesign, ModelConfig
from ..metrics import EvaluationResults


class HyperparameterOptimizer:
    """Optimize hyperparameters using LLM feedback loop."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize HPO optimizer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(__name__)

    def optimize(
        self,
        initial_design: PipelineDesign,
        evaluation: EvaluationResults,
        n_iterations: int = 3
    ) -> List[PipelineDesign]:
        """
        Iteratively improve hyperparameters based on results.

        Args:
            initial_design: Starting pipeline design
            evaluation: Initial evaluation results
            n_iterations: Number of optimization iterations

        Returns:
            List of improved pipeline designs
        """
        designs = [initial_design]
        results = [evaluation]

        for i in range(n_iterations):
            self.logger.info(f"HPO iteration {i + 1}/{n_iterations}")

            # Get improvement suggestions from LLM
            improved_design = self._suggest_improvement(
                designs[-1],
                results[-1]
            )

            designs.append(improved_design)

            # Note: Actual evaluation happens in user code
            # We just suggest improvements here

        return designs

    def _suggest_improvement(
        self,
        design: PipelineDesign,
        evaluation: EvaluationResults
    ) -> PipelineDesign:
        """
        Ask LLM to suggest hyperparameter improvements.

        Args:
            design: Current pipeline design
            evaluation: Current evaluation results

        Returns:
            Improved pipeline design
        """
        prompt = f"""You are optimizing ML pipeline hyperparameters.

Current design:
{json.dumps(design.dict(), indent=2)}

Current performance:
- Balanced Accuracy: {evaluation.balanced_accuracy:.4f}
- F1 (macro): {evaluation.f1_macro:.4f}
- CV: {evaluation.cv_scores.mean():.4f} ± {evaluation.cv_scores.std():.4f}

Suggest improved hyperparameters. Focus on the most impactful changes.
Respond with ONLY valid JSON matching the PipelineDesign schema."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()

            # Clean markdown if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            improved_dict = json.loads(response_text)
            return PipelineDesign(**improved_dict)

        except Exception as e:
            self.logger.error(f"HPO suggestion failed: {e}")
            return design  # Return original if improvement fails
