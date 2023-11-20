import logging

from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from metrics.pipeline import (DatasetsLoader, MetricsPipeline,
                              MetricsPipelineRepository)


def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


PIPELINE_RESULTS_FILE = "./data/metrics.csv"

if __name__ == "__main__":
    setup_logging()

    repository = MetricsPipelineRepository(PIPELINE_RESULTS_FILE)
    loader = DatasetsLoader(["scene", "birds"])

    models = {
        "baseline_binary_relevance_model_svc_123": BinaryRelevance(
            classifier=SVC(random_state=123),
            require_dense=[False, True]
        ),
        "baseline_binary_relevance_model_svc_456": BinaryRelevance(
            classifier=SVC(random_state=456),
            require_dense=[False, True]
        ),
        "baseline_binary_relevance_model_rfc_123": BinaryRelevance(
            classifier=RandomForestClassifier(random_state=123),
            require_dense=[False, True]
        ),
        "baseline_binary_relevance_model_rfc_456": BinaryRelevance(
            classifier=RandomForestClassifier(random_state=456),
            require_dense=[False, True]
        ),
        "baseline_binary_relevance_model_knn": BinaryRelevance(
            classifier=KNeighborsClassifier(),
            require_dense=[False, True]
        ),
    }

    pipe = MetricsPipeline(repository, loader, models)
    pipe.run()
