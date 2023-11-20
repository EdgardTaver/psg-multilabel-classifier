import pandas as pd
from metrics.types import RawEvaluationResults


def evaluation_results_to_flat_table(evaluation_results: RawEvaluationResults) -> pd.DataFrame:
    all_rows = []
    
    for model_name, model_results in evaluation_results.items():
        for dataset_name, dataset_results in model_results.items():
            flat_row = {}
            flat_row["model"] = model_name
            flat_row["dataset"] = dataset_name

            for metric_name, metric_results in dataset_results.raw().items():
                for fold_index, fold_result in enumerate(metric_results):
                    column_name = f"{metric_name}_{fold_index + 1}:{len(metric_results)}"
                    flat_row[column_name] = fold_result
            
            all_rows.append(flat_row)
    
    return pd.DataFrame(all_rows)