"""
TPOT baseline comparison utilities.
"""

import time
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import balanced_accuracy_score, f1_score


class TPOTComparator:
    """Compare LLM-AutoML against TPOT baseline."""

    def __init__(
        self,
        generations: int = 50,
        population_size: int = 50,
        max_time_mins: int = 60,
        cv: int = 5
    ):
        """
        Initialize TPOT comparator.

        Args:
            generations: Number of TPOT generations
            population_size: TPOT population size
            max_time_mins: Maximum optimization time
            cv: Cross-validation folds
        """
        self.config = {
            'generations': generations,
            'population_size': population_size,
            'cv': cv,
            'scoring': 'balanced_accuracy',
            'max_time_mins': max_time_mins,
            'random_state': 42,
            'config_dict': 'TPOT light',
            'n_jobs': -1,
            'verbosity': 1
        }
        self.logger = logging.getLogger(__name__)

    def run_baseline(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run TPOT baseline on dataset.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            seed: Random seed

        Returns:
            Dictionary with TPOT results
        """
        from tpot import TPOTClassifier

        config = self.config.copy()
        if seed is not None:
            config['random_state'] = seed

        self.logger.info("Running TPOT baseline...")
        start_time = time.time()

        tpot = TPOTClassifier(**config)
        tpot.fit(X_train, y_train)

        runtime = time.time() - start_time

        # Evaluate
        y_pred = tpot.predict(X_test)

        return {
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'runtime_seconds': runtime,
            'best_pipeline': str(tpot.fitted_pipeline_)
        }
