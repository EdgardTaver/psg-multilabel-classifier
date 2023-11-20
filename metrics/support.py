import numpy as np
import pandas as pd
from metrics.types import RawEvaluationResults, EvaluationPipelineResult


def evaluation_results_to_flat_table(evaluation_results: RawEvaluationResults) -> pd.DataFrame:
    all_rows = []
    
    for model_name, model_results in evaluation_results.items():
        for dataset_name, dataset_results in model_results.items():
            flat_row = {}
            flat_row["model"] = model_name
            flat_row["dataset"] = dataset_name

            for metric_name, metric_results in dataset_results.raw().items():
                for fold_index, fold_result in enumerate(metric_results):
                    column_name = f"{metric_name}-{fold_index + 1}:{len(metric_results)}"
                    flat_row[column_name] = fold_result
            
            all_rows.append(flat_row)
    
    return pd.DataFrame(all_rows)

def flat_table_to_evaluation_results(flat_table: pd.DataFrame) -> RawEvaluationResults:
    raw_evaluation_results = {}

    for _, row in flat_table.iterrows():
        model_name = row["model"]
        dataset_name = row["dataset"]

        if model_name not in raw_evaluation_results:
            raw_evaluation_results[model_name] = {}
        
        if dataset_name not in raw_evaluation_results[model_name]:
            raw_evaluation_results[model_name][dataset_name] = EvaluationPipelineResult({})

        metrics = {}
        for column_name, column_value in row.items():
            if column_name == "model" or column_name == "dataset":
                continue

            metric_name, _ = column_name.split("-")
            if metric_name not in metrics.keys():
                metrics[metric_name] = np.array([])

            metrics[metric_name] = np.append(metrics[metric_name], column_value)
        
        raw_evaluation_results[model_name][dataset_name] = EvaluationPipelineResult(metrics)

    return raw_evaluation_results