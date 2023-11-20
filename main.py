from metrics.pipeline import MetricsPipeline, MetricsPipelineRepository
import logging

def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


PIPELINE_RESULTS_FILE = "./data/metrics.csv"

if __name__ == "__main__":
    setup_logging()

    a = MetricsPipelineRepository()
    a.load_from_file(PIPELINE_RESULTS_FILE)

    print(a.raw_evaluation_results)

    b = MetricsPipeline(a)
    b.run()

    a.save_to_file(PIPELINE_RESULTS_FILE)
    
