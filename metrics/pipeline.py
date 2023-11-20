import os
import pandas as pd
from metrics.evaluation import EvaluationPipeline, EvaluationPipelineResult
from metrics.support import evaluation_results_to_flat_table, flat_table_to_evaluation_results
from metrics.types import RawEvaluationResults
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.dataset import load_dataset

import logging

class MetricsPipeline:
    def __init__(self, repository: "MetricsPipelineRepository"):
        self.repository = repository

    def run(self):
        # TODO: should be able to get info about each dataset
        # - rows; -labels
        
        # TODO: should run the models and get the evaluation results

        # desired_datasets = ["scene", "emotions", "birds"]
        desired_datasets = ["scene"]

        datasets = {}
        for dataset_name in desired_datasets:
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

            datasets[dataset_name] = {
                "X": X,
                "y": y,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "rows": X.shape[0],
                "labels_count": y.shape[1]
            }
        

        for name, info in datasets.items():
            logging.info("===")
            logging.info(f"information for dataset `{name}`")
            logging.info(f"rows: {info['rows']}, labels: {info['labels_count']}")


        logging.info("finished getting all datasets")

        baseline_binary_relevance_model = BinaryRelevance(
            classifier=SVC(),
            require_dense=[False, True]
        )

        models = {
            "baseline_binary_relevance_model": baseline_binary_relevance_model,
        }

        logging.info("will start getting metrics for all the models")

        for model_name, model in models.items():
            logging.info(f"# running model `{model_name}`")

            n_folds = 2
            evaluation_pipeline = EvaluationPipeline(model, n_folds)

            for dataset_name, info in datasets.items():
                logging.info(f"## running dataset `{dataset_name}`")

                if self.repository.result_already_exists(model_name, dataset_name):
                    logging.warn(f"## dataset `{dataset_name}` was already evaluated for model `{model_name}`")
                    continue

                result = evaluation_pipeline.run(info["X"], info["y"])
                self.repository.add_result(model_name, dataset_name, result)

                logging.info(f"results obtained:")
                result.describe()
        
        logging.info("finished getting metrics for all the models")




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
