import logging

from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance

from metrics.pipeline import (DatasetsLoader, MetricsPipeline,
                              MetricsPipelineRepository)


def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


PIPELINE_RESULTS_FILE = "./data/metrics.csv"

if __name__ == "__main__":
    setup_logging()

    repository = MetricsPipelineRepository(PIPELINE_RESULTS_FILE)
    loader = DatasetsLoader(["birds"])

    models = {
        "baseline_binary_relevance_model": BinaryRelevance(
           classifier=SVC(),
            require_dense=[False, True]
        ),
    }

    pipe = MetricsPipeline(repository, loader, models)
    pipe.run()
