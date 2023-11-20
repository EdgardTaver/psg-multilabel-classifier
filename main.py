from lib.metrics_pipeline import MetricsPipeline
import logging

def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


if __name__ == "__main__":
    setup_logging()
    MetricsPipeline().run()
