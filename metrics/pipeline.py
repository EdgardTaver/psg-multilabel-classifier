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

        datasets = self.datasets_loader.get()
        
        total_datasets = len(datasets)
        total_models = len(self.models)

        total_evaluations = total_datasets * total_models
        evaluation_index = 0

        for model_name, model in self.models.items():
            evaluation_index += 1

            n_folds = 2
            evaluation_pipeline = EvaluationPipeline(model, n_folds)

            for dataset_name, info in datasets.items():
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

            train_dataset = load_dataset(dataset_name, "train")
            if train_dataset is None:
                raise Exception(f"dataset `{dataset_name}` not found")
            X_train, y_train, _, _ = train_dataset

            test_dataset = load_dataset(dataset_name, "test")
            if test_dataset is None:
                raise Exception(f"dataset `{dataset_name}` not found")
            X_test, y_test, _, _ = test_dataset

            self.loaded_datasets[dataset_name] = {
                "X": X,
                "y": y,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "rows": X.shape[0],
                "labels_count": y.shape[1]
            }

        logging.info("finished getting all datasets")
    
    def get(self) -> Dict[str, Any]:
        if len(self.loaded_datasets) == 0:
            raise Exception("no datasets loaded")

        return self.loaded_datasets

    def describe(self) -> None:
        for name, info in self.loaded_datasets.items():
            structured_log = {
                "dataset": name,
                "rows": info["rows"],
                "labels_count": info["labels_count"],
            }

            logging.info(f"information for dataset: {structured_log}")



class MetricsPipelineRepository:
    """
    Wrapper for `RawEvaluationResults` with additional functionality.
    """

    raw_evaluation_results: RawEvaluationResults

    def __init__(self) -> None:
        self.raw_evaluation_results = {}
    
    def load_from_file(self, path:str) -> None:
        if not path.endswith(".csv"):
            raise Exception("only CSV files are supported")
    
        if not os.path.exists(path):
            raise Exception("file does not exist")

        df = pd.read_csv(path)
        self.raw_evaluation_results = flat_table_to_evaluation_results(df)
    
    def save_to_file(self, path:str) -> None:
        df = evaluation_results_to_flat_table(self.raw_evaluation_results)
        df.to_csv(path, index=False)

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
