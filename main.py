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

    repository = MetricsPipelineRepository()
    repository.load_from_file(PIPELINE_RESULTS_FILE)

    loader = DatasetsLoader(["scene"])
    loader.load()
    loader.describe()

    models = {
        "baseline_binary_relevance_model": BinaryRelevance(
           classifier=SVC(),
            require_dense=[False, True]
        ),
    }

    pipe = MetricsPipeline(repository, loader, models)
    pipe.run()

    repository.save_to_file(PIPELINE_RESULTS_FILE)
