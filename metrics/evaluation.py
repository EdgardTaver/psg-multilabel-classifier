from typing import Any

import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate

from metrics.types import EvaluationPipelineResult


class EvaluationPipeline:
    model: Any
    n_folds: int

    def __init__(self, model: Any, n_folds: int = 5) -> None:
        self.model = model
        self.n_folds = n_folds

    def run(self, X: Any, y: Any) -> "EvaluationPipelineResult":
        accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)
        hamming_loss_scorer = metrics.make_scorer(
            metrics.hamming_loss, greater_is_better=False)

        f1_scorer = metrics.make_scorer(metrics.f1_score, average="macro", zero_division=0)
        # using "macro" average following what is described here:
        # """
        # Macro-averaging is to be preferred over micro-averaging in case of imbalanced classes
        # (which is almost always the case), because it weighs each of the classes equally and
        # isn’t influenced by the number of examples of each class.
        # """
        # source:
        # https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404


        scoring_set = {
            "accuracy": accuracy_scorer,
            "hamming_loss": hamming_loss_scorer,
            "f1": f1_scorer,
        }

        validate_result = cross_validate(
            self.model,
            X, y,
            cv=self.n_folds,
            scoring=scoring_set,
            return_train_score=True,
            verbose=10,
            n_jobs=-1)

        return EvaluationPipelineResult(validate_result)
