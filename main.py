from metrics.pipeline import MetricsPipeline, MetricsPipelineRepository, DatasetsLoader
import logging

def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


PIPELINE_RESULTS_FILE = "./data/metrics.csv"

if __name__ == "__main__":
    setup_logging()

    repository = MetricsPipelineRepository()
    repository.load_from_file(PIPELINE_RESULTS_FILE)

    datasets = ["scene"]
    loader = DatasetsLoader(datasets)
    loader.load()
    loader.describe()

    # print(repository.raw_evaluation_results)

    # b = MetricsPipeline(repository)
    # b.run()

    # repository.save_to_file(PIPELINE_RESULTS_FILE)
    
