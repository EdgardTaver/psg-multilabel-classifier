import argparse
import logging
from typing import Any, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import BinaryRelevance

from lib.base_models import (DependantBinaryRelevance, PatchedClassifierChain,
                             StackedGeneralization)
from lib.classifiers import (ClassifierChainWithFTestOrdering,
                             ClassifierChainWithGeneticAlgorithm,
                             ClassifierChainWithLOP,
                             PartialClassifierChainWithLOP, StackingWithFTests)
from metrics.pipeline import (DatasetsLoader, MetricsPipeline,
                              MetricsPipelineRepository)


def setup_logging() -> None:
    LOGGING_FORMAT="%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


N_FOLDS = 10
BASE_FILE_NAME = f"rfc_n_folds={N_FOLDS}"
PIPELINE_RESULTS_FILE = f"./data/metrics_{BASE_FILE_NAME}.csv"
SUMMARIZED_RESULTS_FILE = f"./data/summarized_result_{BASE_FILE_NAME}.csv"

def build_repository() -> MetricsPipelineRepository:
    return MetricsPipelineRepository(PIPELINE_RESULTS_FILE)


def build_dataset_loader() -> DatasetsLoader:
    return DatasetsLoader([
        # [done] fast datasets
        "birds",
        "emotions",
        "scene",
        
        # [done] not so fast datasets
        "yeast",
        "enron",
        "genbase",
        "medical",

        # [done] slow datasets
        # "tmc2007_500",

        # impossibly slow datasets
        # "delicious",
        # "bibtex",
        # "mediamill",
    ])

def build_models_list() -> Dict[str, Any]:
    return {
        "baseline_binary_relevance_model": BinaryRelevance(
            classifier=RandomForestClassifier(random_state=42),
            require_dense=[False, True]
        ),
        "baseline_stacked_generalization": StackedGeneralization(
            base_classifier=RandomForestClassifier(random_state=42),
        ),
        "baseline_dependant_binary_relevance": DependantBinaryRelevance(
            base_classifier=RandomForestClassifier(random_state=42),
        ),
        "baseline_classifier_chain": PatchedClassifierChain(
            base_classifier=RandomForestClassifier(random_state=42),
        ),
        "stacking_with_f_tests-alpha=0.25": StackingWithFTests(
            base_classifier=RandomForestClassifier(random_state=42),
            alpha=0.25,
        ),
        "stacking_with_f_tests-alpha=0.50": StackingWithFTests(
            base_classifier=RandomForestClassifier(random_state=42),
            alpha=0.50,
        ),
        "stacking_with_f_tests-alpha=0.75": StackingWithFTests(
            base_classifier=RandomForestClassifier(random_state=42),
            alpha=0.75,
        ),
        "classifier_chain_with_f_test_ordering-ascending_chain=False": ClassifierChainWithFTestOrdering(
            base_classifier=RandomForestClassifier(random_state=42),
            ascending_chain=False,
        ),
        "classifier_chain_with_f_test_ordering-ascending_chain=True": ClassifierChainWithFTestOrdering(
            base_classifier=RandomForestClassifier(random_state=42),
            ascending_chain=True,
        ),
        "classifier_chain_with_lop-num_generations=10": ClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=10,
        ),
        "classifier_chain_with_lop-num_generations=25": ClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=25,
        ),
        "classifier_chain_with_lop-num_generations=50": ClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=50,
        ),
        "partial_classifier_chain_with_lop-num_generations=10": PartialClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=10,
        ),
        "partial_classifier_chain_with_lop-num_generations=25": PartialClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=25,
        ),
        "partial_classifier_chain_with_lop-num_generations=50": PartialClassifierChainWithLOP(
            base_classifier=RandomForestClassifier(random_state=42),
            num_generations=50,
        ),
        # "classifier_chain_with_genetic_algorithm-num_generations=5": ClassifierChainWithGeneticAlgorithm(
        #     base_classifier=RandomForestClassifier(random_state=42),
        #     num_generations=5,
        # ),
    }

DATASETS_INFO_TO_CSV = "datasets_info_to_csv"
DESCRIBE_DATASETS = "describe_datasets"
DESCRIBE_METRICS = "describe_metrics"
METRICS_TO_CSV = "metrics_to_csv"
RUN_MODELS = "run_models"

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        default=DESCRIBE_DATASETS,
        help="action to be executed",
        choices=[
            DATASETS_INFO_TO_CSV,
            DESCRIBE_DATASETS,
            DESCRIBE_METRICS,
            METRICS_TO_CSV,
            RUN_MODELS,
        ],
        action="store")
    
    args = parser.parse_args()

    repository = build_repository()
    loader = build_dataset_loader()
    models = build_models_list()

    if args.task == DESCRIBE_DATASETS:
        loader.load()
        loader.describe_log()
    
    if args.task == DATASETS_INFO_TO_CSV:
        loader.load()
        result = loader.describe_json()

        df = pd.DataFrame(result)
        df.to_csv("./data/datasets_info.csv", index=False)
    
    if args.task == DESCRIBE_METRICS:
        repository.load_from_file()
        repository.describe_log()
    
    if args.task == METRICS_TO_CSV:
        repository.load_from_file()
        result = repository.describe_dict()

        df = pd.DataFrame(result)
        df.to_csv(SUMMARIZED_RESULTS_FILE, index=False)

    if args.task == RUN_MODELS:
        pipe = MetricsPipeline(repository, loader, models, N_FOLDS)
        pipe.run()
