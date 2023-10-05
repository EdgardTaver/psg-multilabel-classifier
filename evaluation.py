from sklearn.model_selection import cross_validate
import sklearn.metrics as metrics
from typing import Any, Dict


class EvaluationPipelineResult:
    cross_validate_result: Dict[Any, Any]

    def __init__(self, cross_validate_result: Dict[Any, Any]) -> None:
        self.cross_validate_result = cross_validate_result

    def describe(self) -> None:
        print("Accuracy: {:.4f} ± {:.2f}".format(
            self.cross_validate_result["test_accuracy"].mean(),
            self.cross_validate_result["test_accuracy"].std()
        ))

        print("Hamming Loss: {:.4f} ± {:.2f}".format(
            self.cross_validate_result["test_hamming_loss"].mean(),
            self.cross_validate_result["test_hamming_loss"].std()
        ))

    def raw(self) -> Dict[Any, Any]:
        return self.cross_validate_result


class EvaluationPipeline:
    model: Any
    n_folds: int

    def __init__(self, model: Any, n_folds: int = 5) -> None:
        # TODO: establish the model type
        self.model = model
        self.n_folds = n_folds

    def run(self, X: Any, y: Any) -> EvaluationPipelineResult:
        accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)
        hamming_loss_scorer = metrics.make_scorer(
            metrics.hamming_loss, greater_is_better=False)

        scoring_set = {
            "accuracy": accuracy_scorer,
            "hamming_loss": hamming_loss_scorer,
        }

        validate_result = cross_validate(
            self.model,
            X, y,
            cv=self.n_folds,
            scoring=scoring_set,
            return_train_score=True)

        return EvaluationPipelineResult(validate_result)
