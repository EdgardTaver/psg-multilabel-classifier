import logging
from typing import Any, Dict

from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import BinaryRelevance

from lib.base_models import (DependantBinaryRelevance, PatchedClassifierChain,
                             StackedGeneralization)
from lib.classifiers import (ClassifierChainWithFTestOrdering,
                             ClassifierChainWithLOP,
                             PartialClassifierChainWithLOP, StackingWithFTests)
from metrics.pipeline import (DatasetsLoader, MetricsPipeline,
                              MetricsPipelineRepository)


def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


def build_repository() -> MetricsPipelineRepository:
    PIPELINE_RESULTS_FILE = "./data/metrics.csv"
    return MetricsPipelineRepository(PIPELINE_RESULTS_FILE)


def build_dataset_loader() -> DatasetsLoader:
    return DatasetsLoader([
        # [done] fast datasets
        # "birds",
        # "emotions",
        # "scene",
        
        # [done] not so fast datasets
        # "yeast", 

        # slow datasets
        # "bibtex",
        # "delicious",
        # "enron", 
        # "genbase", 
        # "mediamill", 
        # "medical", 
        "tmc2007_500", 
    ])

def build_models_list() -> Dict[str, Any]:
    return {
        "baseline_binary_relevance_model": BinaryRelevance(
            classifier=KNeighborsClassifier(),
            require_dense=[False, True]
        ),
        "baseline_stacked_generalization": StackedGeneralization(
            base_classifier=KNeighborsClassifier(),
        ),
        "baseline_dependant_binary_relevance": DependantBinaryRelevance(
            base_classifier=KNeighborsClassifier(),
        ),
        "baseline_classifier_chain": PatchedClassifierChain(
            base_classifier=KNeighborsClassifier(),
        ),
        "stacking_with_f_tests-alpha=0.25": StackingWithFTests(
            base_classifier=KNeighborsClassifier(),
            alpha=0.25,
        ),
        "stacking_with_f_tests-alpha=0.50": StackingWithFTests(
            base_classifier=KNeighborsClassifier(),
            alpha=0.50,
        ),
        "stacking_with_f_tests-alpha=0.75": StackingWithFTests(
            base_classifier=KNeighborsClassifier(),
            alpha=0.75,
        ),
        "classifier_chain_with_f_test_ordering-ascending_chain=False": ClassifierChainWithFTestOrdering(
            base_classifier=KNeighborsClassifier(),
            ascending_chain=False,
        ),
        "classifier_chain_with_f_test_ordering-ascending_chain=True": ClassifierChainWithFTestOrdering(
            base_classifier=KNeighborsClassifier(),
            ascending_chain=True,
        ),
        "classifier_chain_with_lop-num_generations=10": ClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=10,
        ),
        "classifier_chain_with_lop-num_generations=25": ClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=25,
        ),
        "classifier_chain_with_lop-num_generations=50": ClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=50,
        ),
        "partial_classifier_chain_with_lop-num_generations=10": PartialClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=10,
        ),
        "partial_classifier_chain_with_lop-num_generations=25": PartialClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=25,
        ),
        "partial_classifier_chain_with_lop-num_generations=50": PartialClassifierChainWithLOP(
            base_classifier=KNeighborsClassifier(),
            num_generations=50,
        ),
    }


if __name__ == "__main__":
    setup_logging()

    repository = build_repository()
    loader = build_dataset_loader()
    models = build_models_list()

    pipe = MetricsPipeline(repository, loader, models)
    pipe.run()

    # TODO: still missing the most robust CC with Genetic Algorithm
