from typing import Dict

import numpy as np
from numpy.typing import NDArray

ModelName = str
DatasetName = str
RawEvaluationResults = Dict[ModelName, Dict[DatasetName, "EvaluationPipelineResult"]]

MetricName = str
CrossValidatedResults = Dict[MetricName, NDArray[np.float64]]


class EvaluationPipelineResult:
    cross_validated_results: CrossValidatedResults

    def __init__(self, cross_validate_result: CrossValidatedResults) -> None:
        self.cross_validated_results = cross_validate_result

    def describe(self) -> None:
        print("Accuracy: {:.4f} ± {:.2f}".format(
            self.cross_validated_results["test_accuracy"].mean(),
            self.cross_validated_results["test_accuracy"].std()
        ))

        print("Hamming Loss: {:.4f} ± {:.2f}".format(
            self.cross_validated_results["test_hamming_loss"].mean(),
            self.cross_validated_results["test_hamming_loss"].std()
        ))

        print("F1 score: {:.4f} ± {:.2f}".format(
            self.cross_validated_results["test_f1"].mean(),
            self.cross_validated_results["test_f1"].std()
        ))

    def raw(self) -> CrossValidatedResults:
        return self.cross_validated_results