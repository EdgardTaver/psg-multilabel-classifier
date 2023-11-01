import copy
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pygad
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain

from lib.types import MultiLabelClassifier
from lib.utils import has_duplicates, has_negatives


class StackingWithFTests(MultiLabelClassifier):
    alpha: float
    use_first_layer_to_calculate_correlations: bool
    
    first_layer_classifiers: BinaryRelevance
    second_layer_classifiers: List[Any] # TODO should be any generic type of classifier
    labels_count: int

    def __init__(
        self,
        alpha: float = 0.5,
        use_first_layer_to_calculate_correlations: bool = False,
        classifier: Any = RandomForestClassifier()
    ):
        super().__init__()

        if alpha < 0.0 or alpha > 1.0:
            raise Exception("alpha must be >= 0.0 and <= 1.0")

        self.base_classifier = classifier

        self.alpha = alpha
        self.use_first_layer_to_calculate_correlations = use_first_layer_to_calculate_correlations
        
        self.first_layer_classifiers = BinaryRelevance(
            classifier=copy.deepcopy(self.base_classifier),
            require_dense=[False, True]
        )

        self.second_layer_classifiers = []
        self.correlated_labels_map = pd.DataFrame()
        self.labels_count = 0


    def fit(self, X: Any, y: Any):
        self.labels_count = y.shape[1]

        self.first_layer_classifiers.fit(X, y)
        
        label_classifications = y
        if self.use_first_layer_to_calculate_correlations:
            label_classifications = self.first_layer_classifiers.predict(X)

        f_tested_label_pairs = self.calculate_f_test_for_all_label_pairs(label_classifications)
        self.correlated_labels_map = self.get_map_of_correlated_labels(f_tested_label_pairs)

        for i in range(self.labels_count):
            mask = self.correlated_labels_map["for_label"] == i
            split_df = self.correlated_labels_map[mask].reset_index(drop=True)
            labels_to_expand = split_df["expand_this_label"].to_list()

            labels_to_expand.sort()

            additional_input = label_classifications.todense()[:, labels_to_expand]
            
            X_expanded = np.hstack([X.todense(), additional_input])
            X_expanded = np.asarray(X_expanded)

            print(f"FIT: X_extended shape, for label {i}, is {X_expanded.shape}")

            y_label_specific = y.todense()[:, i]
            y_label_specific = self.convert_matrix_to_vector(y_label_specific)

            meta_classifier = copy.deepcopy(self.base_classifier)
            meta_classifier.fit(X_expanded, y_label_specific)

            self.second_layer_classifiers.append(meta_classifier)
            print(f"finished training meta classifier for label {i}")
    
    def calculate_f_test_for_all_label_pairs(self, label_classifications: Any) -> List[Dict[str, Any]]:
        results = []

        for i in range(0, self.labels_count):
            for j in range(0, self.labels_count):
                if i == j:
                    continue

                X = label_classifications.todense()[:, i]
                base_label = self.convert_matrix_to_array(X)

                y = label_classifications.todense()[:, j]
                against_label = self.convert_matrix_to_vector(y)

                f_test_result = f_classif(base_label, against_label)[0]

                results.append({
                    "label_being_tested": i,
                    "against_label": j,
                    "f_test_result": float(f_test_result)
                })
        
        return results
    
    def convert_matrix_to_array(self, matrix: Any):
        return np.asarray(matrix).reshape(-1, 1)

    def convert_matrix_to_vector(self, matrix: Any):
        return np.asarray(matrix).reshape(-1)
    
    def get_map_of_correlated_labels(self, f_test_results: List[Dict[str, Any]]) -> pd.DataFrame:
        temp_df = pd.DataFrame(f_test_results)
        
        sorted_temp_df = temp_df.sort_values(
            by=["label_being_tested", "f_test_result"],
            ascending=[True, False])
        # ordering in descending order by the F-test result,
        # following what the main article describes

        selected_features = []

        for i in range(self.labels_count):
            mask = sorted_temp_df["label_being_tested"] == i
            split_df = sorted_temp_df[mask].reset_index(drop=True)

            big_f = split_df["f_test_result"].sum()
            max_cum_f = self.alpha * big_f

            cum_f = 0
            for _, row in split_df.iterrows():
                cum_f += row["f_test_result"]
                if cum_f > max_cum_f:
                    break

                selected_features.append({
                    "for_label": i,
                    "expand_this_label": int(row["against_label"]),
                    "f_test_result": float(row["f_test_result"]),
                })
        
        cols = ["for_label", "expand_this_label", "f_test_result"]
        return pd.DataFrame(selected_features, columns=cols)
    
    def predict(self, X: Any) -> np.ndarray[Any,Any]:
        if self.correlated_labels_map.columns.size == 0:
            raise Exception("model was not trained yet")

        predictions = self.first_layer_classifiers.predict(X)

        second_layer_predictions = []
        for i in range(self.labels_count):
            mask = self.correlated_labels_map["for_label"] == i
            split_df = self.correlated_labels_map[mask].reset_index(drop=True)
            labels_to_expand = split_df["expand_this_label"].to_list()

            labels_to_expand.sort()

            additional_input = predictions.todense()[:, labels_to_expand]

            X_expanded = np.hstack([X.todense(), additional_input])
            X_expanded = np.asarray(X_expanded)

            print(f"PREDICT: X_extended shape, for label {i}, is {X_expanded.shape}")

            temp_preds = self.second_layer_classifiers[i].predict(X_expanded)
            second_layer_predictions.append(temp_preds)

        reshaped_array = np.asarray(second_layer_predictions).T
        return reshaped_array

class ClassifierChainWithGeneticAlgorithm(MultiLabelClassifier):
    def __init__(self, base_classifier: Any, num_generations: int = 5, random_state: Optional[int] = None) -> None:
        self.base_classifier = base_classifier
        self.num_generations = num_generations

        if random_state is None:
            self.random_state = np.random.randint(0, 1000)
        else:
            self.random_state = random_state
    
    def fit(self, X: Any, y: Any):
        self.x = X
        self.y = y
        # this is the most practical way to pass the data to the fitness function

        label_count = self.y.shape[1]
        if label_count < 3:
            raise Exception("label count is too low, we need at least 3 labels")

        label_space = np.arange(label_count)
        solutions_per_population = math.ceil(label_count / 2)
        # to simplify the model, some heuristics are used

        ga_model = pygad.GA( #type:ignore
            gene_type=int,
            gene_space=label_space,
            random_seed=self.random_state,
            save_best_solutions=False,
            fitness_func=self.model_fitness_func,
            allow_duplicate_genes=False, # very important, otherwise we will have duplicate labels in the ordering
            num_genes=label_count,

            # set up
            num_generations=self.num_generations,
            sol_per_pop=solutions_per_population,

            # following what the article describes
            keep_elitism=1, # also following what the article describes, but we have to double check [TODO]
            parent_selection_type="rws", # following what the article describes
            # mutation_probability=0.005, # following what the article describes

            # the following settings are fixed
            # they were chosen for no particular reason
            # they are being kept as fixed to simplify the model
            num_parents_mating=2,
            crossover_type="scattered",
            mutation_type="random",
            mutation_by_replacement=True,
            mutation_num_genes=1,
        )

        ga_model.run()

        solution, _, _ = ga_model.best_solution()

        best_classifier = ClassifierChain(
            classifier=copy.deepcopy(self.base_classifier),
            require_dense=[False, True],
            order=solution,
        )

        best_classifier.fit(self.x, self.y)

        self.best_classifier = best_classifier
        
    def model_fitness_func(self, ga_instance: Any, solution: Any, solution_idx: Any) -> float:
        if has_duplicates(solution):
            print("solutions contains duplicated values, skipping")
            return 0
        
        if has_negatives(solution):
            print("solutions contains negative values, skipping")
            return 0

        hamming_loss = self.test_ordering(solution)
        hamming_loss = float(hamming_loss)
        return 1/hamming_loss
        # this will be the fitness function result, and we want to maximize it
        # therefore, we have to return the inverse of the hamming loss
    
    def test_ordering(self, solution: List[int]):
        print(f"testing order: {solution}")

        classifier = ClassifierChain(
            classifier=copy.deepcopy(self.base_classifier),
            require_dense=[False, True],
            order=solution,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=self.random_state
        )

        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)

        return metrics.hamming_loss(y_test, preds)


    def predict(self, X: Any) -> Any:
        if self.best_classifier is None:
            raise Exception("model was not trained yet")

        return self.best_classifier.predict(X)