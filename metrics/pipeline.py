import logging
import os
from typing import Any, Dict, List

import pandas as pd
from skmultilearn.dataset import load_dataset

from metrics.evaluation import EvaluationPipeline, EvaluationPipelineResult
from metrics.support import (evaluation_results_to_flat_table,
                             flat_table_to_evaluation_results)
from metrics.types import RawEvaluationResults


class MetricsPipeline:
    def __init__(
        self,
        repository: "MetricsPipelineRepository",
        datasets_loader: "DatasetsLoader",
        models: Dict[str, Any],
    ) -> None:
        self.repository = repository
        self.datasets_loader = datasets_loader
        self.models = models

    def run(self):
        logging.info("pipeline: getting metrics for all the models")

        self.repository.load_from_file()
        self.datasets_loader.load()

        datasets = self.datasets_loader.list_datasets()
        
        total_datasets = len(datasets)
        total_models = len(self.models)

        total_evaluations = total_datasets * total_models
        evaluation_index = 0

        for model_name, model in self.models.items():
            n_folds = 10
            evaluation_pipeline = EvaluationPipeline(model, n_folds)

            for dataset_name, info in datasets.items():
                evaluation_index += 1

                log_fields = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "n_folds": n_folds,
                    "evaluation_index": evaluation_index,
                    "total_evaluations": total_evaluations,
                }

                logging.info(f"running evaluation | {log_fields}")

                if self.repository.result_already_exists(model_name, dataset_name):
                    logging.info(f"dataset already evaluated, skipping | {log_fields}")
                    continue

                result = evaluation_pipeline.run(info["X"], info["y"])
                self.repository.add_result(model_name, dataset_name, result)
                self.repository.save_to_file()

                logging.info(f"evaluation finished | {log_fields}")
        
        logging.info("pipeline: finished running all evaluations")


class DatasetsLoader:
    def __init__(self, dataset_names: List[str]) -> None:
        self.dataset_names = dataset_names
        self.loaded_datasets = {}
    
    def load(self) -> None:
        self.loaded_datasets = {}

        for dataset_name in self.dataset_names:
            print(f"getting dataset `{dataset_name}`")
            
            full_dataset = load_dataset(dataset_name, "undivided")
            if full_dataset is None:
                raise Exception(f"dataset `{dataset_name}` not found")
            X, y, _, _ = full_dataset

            # train_dataset = load_dataset(dataset_name, "train")
            # if train_dataset is None:
            #     raise Exception(f"dataset `{dataset_name}` not found")
            # X_train, y_train, _, _ = train_dataset

            # test_dataset = load_dataset(dataset_name, "test")
            # if test_dataset is None:
            #     raise Exception(f"dataset `{dataset_name}` not found")
            # X_test, y_test, _, _ = test_dataset

            self.loaded_datasets[dataset_name] = {
                "X": X,
                "y": y,
                # "X_train": X_train,
                # "y_train": y_train,
                # "X_test": X_test,
                # "y_test": y_test,
                "rows": X.shape[0],
                "features_count": X.shape[1],
                "labels_count": y.shape[1]
            }

        logging.info("finished getting all datasets")
    
    def list_datasets(self) -> Dict[str, Any]:
        if len(self.loaded_datasets) == 0:
            raise Exception("no datasets loaded")

        return self.loaded_datasets

    def describe(self) -> None:
        for name, info in self.loaded_datasets.items():
            structured_log = {
                "dataset": name,
                "rows": info["rows"],
                "features_count": info["features_count"],
                "labels_count": info["labels_count"],
            }

            logging.info(f"information for dataset: {structured_log}")


class MetricsPipelineRepository:
    """
    Wrapper for `RawEvaluationResults` with additional functionality.
    """

    raw_evaluation_results: RawEvaluationResults

    def __init__(self, csv_file_path: str) -> None:
        self.csv_file_path = csv_file_path
        self.raw_evaluation_results = {}
    
    def load_from_file(self) -> None:
        if not self.csv_file_path.endswith(".csv"):
            raise Exception("only CSV files are supported")
    
        if not os.path.exists(self.csv_file_path):
            logging.warn(f"file does not exist, creating new one: {self.csv_file_path}")
            return

        df = pd.read_csv(self.csv_file_path)
        self.raw_evaluation_results = flat_table_to_evaluation_results(df)
    
    def save_to_file(self) -> None:
        df = evaluation_results_to_flat_table(self.raw_evaluation_results)
        df.to_csv(self.csv_file_path, index=False)

    def add_result(self, model_name: str, dataset_name: str, result: EvaluationPipelineResult) -> None:
        if model_name not in self.raw_evaluation_results:
            self.raw_evaluation_results[model_name] = {}
        
        self.raw_evaluation_results[model_name][dataset_name] = result
    
    def result_already_exists(self, model_name: str, dataset_name: str) -> bool:
        if model_name not in self.raw_evaluation_results:
            return False
        
        if dataset_name not in self.raw_evaluation_results[model_name]:
            return False
        
        return True

    def describe_log(self):
        results = self.describe_dict()
        for result in results:
            logging.info(f"information for evaluation: {result}")
            
    def describe_dict(self) -> List[Dict[str, float]]:
        results = []

        for model_name, model_results in self.raw_evaluation_results.items():
            for dataset_name, result in model_results.items():
                json_result = result.describe_json()

                structured_information = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "accuracy": round(json_result["accuracy"], 4),
                    "accuracy_std": round(json_result["accuracy_std"], 4),
                    "hamming_loss": round(json_result["hamming_loss"], 4),
                    "hamming_loss_std": round(json_result["hamming_loss_std"], 4),
                    "f1": round(json_result["f1"], 4),
                    "f1_std": round(json_result["f1_std"], 4),
                }
                results.append(structured_information)

        return results