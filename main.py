import logging

from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lib.base_models import StackedGeneralization, DependantBinaryRelevance, PatchedClassifierChain

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
        "bibtex",
        "birds",
        "delicious",
        "emotions",
        "enron",
        "genbase",
        "mediamill",
        "medical",
        "scene",
        "tmc2007_500",
        "yeast",
    ])


if __name__ == "__main__":
    setup_logging()

    repository = build_repository()
    loader = build_dataset_loader()

    models = {
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
    }

    # pipe = MetricsPipeline(repository, loader, models)
    # pipe.run()
