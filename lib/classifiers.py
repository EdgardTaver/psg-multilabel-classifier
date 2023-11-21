import copy
import math
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import pygad
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance

from lib.types import MultiLabelClassifier, LOPMatrix
from lib.utils import has_duplicates, has_negatives
from lib.support import (
    CalculateLabelsCorrelationWithFTest,
    ConditionalEntropies,
    MutualInformation,
    build_chain_based_on_f_test,
)
from lib.base_models import PatchedClassifierChain, PartialClassifierChains


class StackingWithFTests(MultiLabelClassifier):
    alpha: float
    use_first_layer_to_calculate_correlations: bool

    first_layer_classifiers: BinaryRelevance
    second_layer_classifiers: List[Any]  # TODO should be any generic type of classifier
    labels_count: int

    def __init__(
        self,
        alpha: float = 0.5,
        use_first_layer_to_calculate_correlations: bool = False,
        base_classifier: Any = RandomForestClassifier(),
    ):
        if alpha < 0.0 or alpha > 1.0:
            raise Exception("alpha must be >= 0.0 and <= 1.0")

        super().__init__()
        self.copyable_attrs = [
            "base_classifier",
            "alpha",
            "use_first_layer_to_calculate_correlations"]

        self.base_classifier = base_classifier
        self.alpha = alpha
        self.use_first_layer_to_calculate_correlations = (
            use_first_layer_to_calculate_correlations
        )

        self.first_layer_classifiers = BinaryRelevance(
            classifier=copy.deepcopy(self.base_classifier), require_dense=[False, True]
        )

        self.second_layer_classifiers = []
        self.correlated_labels_map = pd.DataFrame()
        self.labels_count = 0

        self.ccf = CalculateLabelsCorrelationWithFTest(alpha=alpha)

    def fit(self, X: Any, y: Any):
        self.labels_count = y.shape[1]

        self.first_layer_classifiers.fit(X, y)

        label_classifications = y
        if self.use_first_layer_to_calculate_correlations:
            label_classifications = self.first_layer_classifiers.predict(X)

        self.correlated_labels_map = self.ccf.get(label_classifications)
        for i in range(self.labels_count):
            mask = self.correlated_labels_map["for_label"] == i
            split_df = self.correlated_labels_map[mask].reset_index(drop=True)
            labels_to_expand = split_df["expand_this_label"].to_list()
            labels_to_expand.sort()
            # some base classifiers, such as random forest, may be slightly influenced
            # by the order of the features
            # so we sort the labels to expand, to make sure the order is always the same

            additional_input = label_classifications.todense()[:, labels_to_expand]
            X_expanded = np.hstack([X.todense(), additional_input])
            X_expanded = np.asarray(X_expanded)

            y_label_specific = y.todense()[:, i]
            y_label_specific = self.convert_matrix_to_vector(y_label_specific)

            meta_classifier = copy.deepcopy(self.base_classifier)
            meta_classifier.fit(X_expanded, y_label_specific)

            self.second_layer_classifiers.append(meta_classifier)

    def convert_matrix_to_vector(self, matrix: Any):
        return np.asarray(matrix).reshape(-1)

    def predict(self, X: Any) -> Any:
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

            predictions_for_label = self.second_layer_classifiers[i].predict(X_expanded)
            second_layer_predictions.append(predictions_for_label)

        reshaped_array = np.asarray(second_layer_predictions).T
        return reshaped_array
    
class ClassifierChainWithFTestOrdering(MultiLabelClassifier):
    """
    Trains a ClassifierChain using the labels correlation calculated with F-Test
    to determine the order of the labels in the chain.

    Arguments:
        `ascending_chain`:
            - if True, the chain will be built in ascending order,
              from the least correlated label to the most correlated.
            - if False, the chain will be built in descending order,
              from the most correlated label to the least correlated.
            - default behavior is False.

        `base_classifier`: the classifier to be used as the base for the chain.
    """

    def __init__(
        self,
        ascending_chain: bool = False,
        base_classifier: Any = RandomForestClassifier(),
    ):
        super().__init__()
        self.copyable_attrs = [
            "base_classifier",
            "ascending_chain"]

        self.main_classifier = None
        self.ascending_chain = ascending_chain
        self.base_classifier = base_classifier
        self.calculator = CalculateLabelsCorrelationWithFTest(alpha=1)
        # `alpha=1` because we need to get a full chain ordering
        # therefore we need all the labels
    
    def fit(self, X: Any, y: Any):
        self.classes_ = np.arange(y.shape[1])
        # NOTE: this is required to run the evaluation pipeline
        # TODO: ideia -> this could be somehow abstracted to a mixin class
        
        f_test_ordering = build_chain_based_on_f_test(
            self.calculator.get(y), self.ascending_chain)
        
        self.main_classifier = PatchedClassifierChain(
            base_classifier=self.base_classifier,
            order=f_test_ordering,
        )

        self.main_classifier.fit(X, y)
    
    def predict(self, X: Any):
        if self.main_classifier is None:
            raise Exception("model was not trained yet")

        return self.main_classifier.predict(X)


class ClassifierChainWithGeneticAlgorithm(MultiLabelClassifier):
    def __init__(
        self,
        base_classifier: Any,
        num_generations: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.copyable_attrs = [
            "base_classifier",
            "num_generations",
            "random_state"]

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
        # also, it becomes too computationally expensive to try all possible combinations
        # the original paper, which inspires but does not describes this model
        # proposes 50 solutions per population

        ga_model = pygad.GA(  # type:ignore
            gene_type=int,
            gene_space=label_space,
            random_seed=self.random_state,
            save_best_solutions=False,
            fitness_func=self.model_fitness_func,
            allow_duplicate_genes=False,  # very important, otherwise we will have duplicate labels in the ordering
            num_genes=label_count,
            # set up
            num_generations=self.num_generations,
            sol_per_pop=solutions_per_population,
            # following what the article describes
            keep_elitism=1,
            parent_selection_type="rws",
            num_parents_mating=2,
            crossover_probability=0.9,
            crossover_type="two_points",
            mutation_type="swap",
            mutation_probability=0.01,
        )

        ga_model.run()

        solution, _, _ = ga_model.best_solution()
        best_classifier = PatchedClassifierChain(
            base_classifier=copy.deepcopy(self.base_classifier),
            order=solution,
        )

        best_classifier.fit(self.x, self.y)
        self.best_classifier = best_classifier

    def model_fitness_func(
        self, ga_instance: Any, solution: Any, solution_idx: Any
    ) -> float:
        if has_duplicates(solution):
            print("solutions contains duplicated values, skipping")
            return 0

        if has_negatives(solution):
            print("solutions contains negative values, skipping")
            return 0

        hamming_loss = self.test_ordering(solution)
        hamming_loss = float(hamming_loss)
        return 1 / hamming_loss
        # this will be the fitness function result, and we want to maximize it
        # therefore, we have to return the inverse of the hamming loss

    def test_ordering(self, solution: List[int]):
        print(f"testing order: {solution}")

        classifier = PatchedClassifierChain(
            base_classifier=copy.deepcopy(self.base_classifier),
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


class LOPSolver:
    # TODO: this could be organized as a Mixin class
    def model_fitness_func(
        self, ga_instance: Any, solution: Any, solution_idx: Any
    ) -> float:
        return self.test_solution(solution)

    def test_solution(self, label_order: List[int]) -> float:
        if self.conditional_entropies is None:
            raise Exception(
                "probabilities and entropies must be calculated before testing a solution"
            )

        lop_matrix = self.build_lop_matrix(label_order)
        return self.calculate_lop(lop_matrix)

    def build_lop_matrix(self, label_order: List[int]) -> LOPMatrix:
        if self.conditional_entropies is None:
            raise Exception(
                "probabilities and entropies must be calculated before testing a solution"
            )

        matrix = {}
        for row_i in label_order:
            matrix[row_i] = {}
            for row_j in label_order:
                conditional_entropy = self.conditional_entropies[row_i][row_j]
                matrix[row_i][row_j] = conditional_entropy

        return matrix

    def calculate_lop(self, lop_matrix: LOPMatrix) -> float:
        matrix_size_n = len(lop_matrix)
        lop_df = pd.DataFrame(lop_matrix)

        upper_triangle_sum = 0
        for row_position in range(matrix_size_n):
            for column_position in range(matrix_size_n):
                if column_position > row_position:
                    conditional_probability = lop_df.iloc[row_position, column_position]
                    upper_triangle_sum += cast(float, conditional_probability)
                    # the conversion to a data frame is not necessary
                    # but makes it easier to find the element we want
                    # by their order in the rows or columns
                    # instead of the actual column or row index

        return upper_triangle_sum


class ClassifierChainWithLOP(MultiLabelClassifier, LOPSolver):
    def __init__(
        self,
        base_classifier: Any,
        num_generations: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.copyable_attrs = [
            "base_classifier",
            "num_generations",
            "random_state"]
        
        self.base_classifier = base_classifier
        self.num_generations = num_generations

        if random_state is None:
            self.random_state = np.random.randint(0, 1000)
        else:
            self.random_state = random_state

        self.conditional_entropies = None
        self.best_classifier = None
        self.conditional_entropies_calculator = ConditionalEntropies()

    def fit(self, X: Any, y: Any):
        self.conditional_entropies = self.conditional_entropies_calculator.calculate(y)

        label_count = y.shape[1]
        label_space = np.arange(label_count)

        ga_model = pygad.GA(  # type:ignore
            gene_type=int,
            gene_space=label_space,
            random_seed=self.random_state,
            save_best_solutions=False,
            fitness_func=self.model_fitness_func,
            allow_duplicate_genes=False,  # very important, otherwise we will have duplicate labels in the ordering
            num_genes=label_count,
            # set up
            num_generations=self.num_generations,
            # following what the article describes
            sol_per_pop=50,
            keep_elitism=5,
            parent_selection_type="rws",
            num_parents_mating=2,
            crossover_probability=0.9,
            crossover_type="two_points",
            mutation_type="swap",
            mutation_probability=0.01,
        )

        ga_model.run()

        solution, _, _ = ga_model.best_solution()
        best_classifier = PatchedClassifierChain(
            base_classifier=copy.deepcopy(self.base_classifier),
            order=solution,
        )

        best_classifier.fit(X, y)
        self.best_classifier = best_classifier

    def predict(self, X: Any) -> Any:
        if self.best_classifier is None:
            raise Exception("model was not trained yet")

        return self.best_classifier.predict(X)


class PartialClassifierChainWithLOP(MultiLabelClassifier, LOPSolver):
    def __init__(
        self,
        base_classifier: Any,
        threshold: float = 0.01,
        num_generations: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.copyable_attrs = [
            "base_classifier",
            "num_generations",
            "random_state"]
        
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.num_generations = num_generations

        if random_state is None:
            self.random_state = np.random.randint(0, 1000)
        else:
            self.random_state = random_state

        self.best_classifier = None
        self.conditional_entropies = None
        self.mutual_information = None
        self.conditional_entropies_calculator = ConditionalEntropies()
        self.mutual_information_calculator = MutualInformation()

    def fit(self, X: Any, y: Any):
        self.conditional_entropies = self.conditional_entropies_calculator.calculate(y)
        self.mutual_information = self.mutual_information_calculator.calculate(y)

        label_count = y.shape[1]
        label_space = np.arange(label_count)

        ga_model = pygad.GA(  # type:ignore
            gene_type=int,
            gene_space=label_space,
            random_seed=self.random_state,
            save_best_solutions=False,
            fitness_func=self.model_fitness_func,
            allow_duplicate_genes=False,  # very important, otherwise we will have duplicate labels in the ordering
            num_genes=label_count,
            # set up
            num_generations=self.num_generations,
            # following what the article describes
            sol_per_pop=50,
            keep_elitism=5,
            parent_selection_type="rws",
            num_parents_mating=2,
            crossover_probability=0.9,
            crossover_type="two_points",
            mutation_type="swap",
            mutation_probability=0.01,
        )

        ga_model.run()

        solution, _, _ = ga_model.best_solution()
        partial_order = self.build_partial_orders(solution)

        best_classifier = PartialClassifierChains(
            base_classifier=copy.deepcopy(self.base_classifier),
            order=solution,
            partial_orders=partial_order,
        )

        best_classifier.fit(X, y)
        self.best_classifier = best_classifier

    def build_partial_orders(self, order: List[int]) -> Any:
        labels_to_expand = {}
        for pos_i in range(len(order)):
            label_i = order[pos_i]
            labels_to_expand[label_i] = []

            for label_j in order[:pos_i]:
                mi = self.mutual_information[label_i][label_j]
                if mi > self.threshold:
                    labels_to_expand[label_i].append(label_j)

        return labels_to_expand

    def predict(self, X: Any) -> Any:
        if self.best_classifier is None:
            raise Exception("model was not trained yet")

        return self.best_classifier.predict(X)
