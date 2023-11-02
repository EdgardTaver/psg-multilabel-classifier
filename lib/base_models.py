import copy
from typing import Any, List

import numpy as np
from skmultilearn.problem_transform import BinaryRelevance

from lib.types import MultiLabelClassifier


class StackedGeneralization(MultiLabelClassifier):
    first_layer_classifiers: BinaryRelevance
    second_layer_classifiers: BinaryRelevance

    def __init__(self, classifier: Any):
        super().__init__()

        first_base_classifier = copy.deepcopy(classifier)
        second_base_classifier = copy.deepcopy(classifier)

        self.first_layer_classifiers = BinaryRelevance(
            classifier=first_base_classifier,
            require_dense=[False, True]
        )

        self.second_layer_classifiers = BinaryRelevance(
            classifier=second_base_classifier,
            require_dense=[False, True]
        )
    
    def fit(self, X: Any, y: Any):
        print(f"FIT: X shape is {X.shape}")
        self.first_layer_classifiers.fit(X, y)

        first_layer_predictions = self.first_layer_classifiers.predict(X)
        formatted_first_layer_predictions = first_layer_predictions.todense()
        X_expanded = np.hstack([X.todense(), formatted_first_layer_predictions])

        first_layer_sum = np.sum(np.sum(formatted_first_layer_predictions, axis=1))
        print("FIT: summing the values (for first layer):", first_layer_sum)

        print(f"FIT: X_extended shape is {X_expanded.shape}")
        self.second_layer_classifiers.fit(X_expanded, y)
    
    def predict(self, X: Any) -> Any:
        print(f"PREDICT: X shape is {X.shape}")
        first_layer_predictions = self.first_layer_classifiers.predict(X)
        formatted_first_layer_predictions = first_layer_predictions.todense()

        X_expanded = np.hstack([X.todense(), formatted_first_layer_predictions])

        print("PREDICT: summing the values (for first layer):", np.sum(np.sum(formatted_first_layer_predictions, axis=1)))
        print(f"PREDICT: X_extended shape is {X_expanded.shape}")

        second_layer_predictions = self.second_layer_classifiers.predict(X_expanded)
        formatted_second_layer_predictions = second_layer_predictions.todense()

        print("PREDICT: summing the values (for second layer):", np.sum(np.sum(formatted_second_layer_predictions, axis=1)))

        return second_layer_predictions


class DependantBinaryRelevance(MultiLabelClassifier):
    first_layer_classifiers: BinaryRelevance
    second_layer_classifiers: List[Any]

    def __init__(self, classifier: Any):
        super().__init__()

        self.first_base_classifier = copy.deepcopy(classifier)
        self.second_base_classifier = copy.deepcopy(classifier)

        self.first_layer_classifiers = BinaryRelevance(
            classifier=self.first_base_classifier,
            require_dense=[False, True]
        )

        self.second_layer_classifiers = []
        self.labels_count = 0
    
    def fit(self, X: Any, y: Any):
        print(f"FIT: X shape is {X.shape}")
        self.first_layer_classifiers.fit(X, y)

        first_layer_predictions = self.first_layer_classifiers.predict(X)
        formatted_first_layer_predictions = first_layer_predictions.todense()
        X_expanded = np.hstack([X.todense(), formatted_first_layer_predictions])

        first_layer_sum = np.sum(np.sum(formatted_first_layer_predictions, axis=1))
        print("FIT: summing the values (for first layer):", first_layer_sum)

        self.labels_count = y.shape[1]

        for i in range(self.labels_count):
            labels_to_expand = [x for x in range(self.labels_count) if x != i]
            print(f"these are labels_to_expand: {labels_to_expand} | {i}")

            additional_input = formatted_first_layer_predictions[:, labels_to_expand]
            print(f"FIT: additional input shape is {additional_input.shape}")

            X_expanded = np.hstack([X.todense(), additional_input])
            X_expanded = np.asarray(X_expanded)

            print(f"FIT: X_extended shape, for label {i}, is {X_expanded.shape}")

            label_specific_y = y.todense()[:, i]
            label_specific_y = self.convert_matrix_to_vector(label_specific_y)

            label_specific_classifier = copy.deepcopy(self.second_base_classifier)
            label_specific_classifier.fit(X_expanded, label_specific_y)

            self.second_layer_classifiers.append(label_specific_classifier)

    def convert_matrix_to_vector(self, matrix: Any):
        return np.asarray(matrix).reshape(-1)

    
    def predict(self, X: Any) -> Any:
        if self.labels_count == 0:
            raise Exception("you must call `fit` before calling `predict`")

        print(f"PREDICT: X shape is {X.shape}")
        first_layer_predictions = self.first_layer_classifiers.predict(X)
        formatted_first_layer_predictions = first_layer_predictions.todense()

        second_layer_predictions = []
        for i in range(self.labels_count):
            labels_to_expand = [x for x in range(self.labels_count) if x != i]
            print(f"these are labels_to_expand: {labels_to_expand} | {i}")

            additional_input = formatted_first_layer_predictions[:, labels_to_expand]
            print(f"PREDICT: additional input shape is {additional_input.shape}")

            X_expanded = np.hstack([X.todense(), additional_input])
            X_expanded = np.asarray(X_expanded)

            print(f"PREDICT: X_extended shape, for label {i}, is {X_expanded.shape}")

            temp_preds = self.second_layer_classifiers[i].predict(X_expanded)
            print(f"temp_preds shape is {temp_preds.shape}")
            
            second_layer_predictions.append(temp_preds)

        reshaped_array = np.asarray(second_layer_predictions).T
        return reshaped_array