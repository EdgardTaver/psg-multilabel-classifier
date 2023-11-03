import copy
from typing import Any, List

import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance

from lib.types import MultiLabelClassifier
from lib.utils import has_duplicates


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

class ClassifierChain:
    """
    Works just like the `ClassifierChain` class from `skmultilearn`, but ensures
    that the output is in the same order as the labels in the dataset
    regardless of the custom order of the classifiers
    """

    def __init__(self, base_classifier: Any, order: List[int]) -> None:
        if has_duplicates(order):
            raise Exception("the order must not contain duplicated values")

        self.base_classifier = base_classifier
        self.order = order
    
    def fit(self, X: Any, y: Any):
        label_count = y.shape[1]
        if label_count != len(self.order):
            raise Exception("provided order does not match the label count")

        self.classifiers = {}
        # using a dict instead of a list as the index has to be the label following the custom order
        # a list would also work, but it is more readable this way

        self.metrics = []

        X_extended = np.asarray(X.todense())
        for label in self.order:
            y_label = y[:, label].todense()

            y_label_vector = np.asarray(y_label).reshape(-1)
            # convert_matrix_to_vector

            meta_classifier = copy.deepcopy(self.base_classifier)
            meta_classifier.fit(X_extended, y_label_vector)
            self.classifiers[label] = meta_classifier

            X_extended = np.hstack([X_extended, y_label])
            X_extended = np.asarray(X_extended)

            # the following section gathers metrics throughout the training
            # it has no impact in the actual model performance, but it might
            # be interesting to study how to model behaves as it is being trained
            metrics_X_train, metrics_X_test, metrics_y_train, metrics_y_test = train_test_split(
                X_extended, y_label_vector, test_size=0.33, random_state=42
            )

            meta_classifier_for_metrics = copy.deepcopy(self.base_classifier)
            meta_classifier_for_metrics.fit(metrics_X_train, metrics_y_train)
            metrics_predictions = meta_classifier_for_metrics.predict(metrics_X_test)

            hl = metrics.hamming_loss(metrics_y_test, metrics_predictions)
            f1 = metrics.f1_score(metrics_y_test, metrics_predictions, average="macro")

            self.metrics.append({
                "training_for_label": label,
                "hamming_loss": hl,
                "f1_score": f1
            })
        
    def predict(self, X: Any) -> Any:
        X_extended = np.asarray(X.todense())

        predicted_labels = {}
        for label in self.order:
            prediction = self.classifiers[label].predict(X_extended)
            predicted_labels[label] = prediction
            
            reshaped_prediction = np.asarray(prediction).reshape(-1, 1)
            X_extended = np.hstack([X_extended, reshaped_prediction])
            X_extended = np.asarray(X_extended)
        
        predictions_in_original_label_order = []
        original_label_order = np.arange(len(self.order))
        for label in original_label_order:
            predictions_in_original_label_order.append(predicted_labels[label])
        # this is **very important**, otherwise the output of the model will not
        # be comparable to the testing dataset 

        reshaped_array = np.asarray(predictions_in_original_label_order).T
        return reshaped_array