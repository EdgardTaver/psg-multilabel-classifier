import pandas as pd

def analyse_summarized_metrics(summarized_metrics: pd.DataFrame) -> pd.DataFrame:
    analysis = summarized_metrics.copy()

    analysis["model_name"] = analysis["model"].apply(lambda x: x.split("-")[0])

    scores = [
        "accuracy",
        "hamming_loss",
        "f1"
    ]

    all_datasets = analysis["dataset"].unique()
    best_models = {}

    for score_name in scores:
        best_models[score_name] = []
        
        for model_name in analysis["model_name"].unique():
            for dataset_name in analysis["dataset"].unique():
                mask = analysis["model_name"] == model_name
                mask &= analysis["dataset"] == dataset_name

                if mask.sum() == 0:
                    best_models[score_name].append({
                        "model_name": model_name,
                        "model_and_params": "-",
                        "dataset": dataset_name,
                        score_name: 0,
                        f"{score_name}_std": 0,
                    })
                    continue

                analysis_slice = analysis[mask].copy()
                analysis_slice = analysis_slice.sort_values(by=score_name, ascending=False)

                best_models[score_name].append({
                    "model_name": model_name,
                    "model_and_params": analysis_slice["model"].iloc[0],
                    "dataset": dataset_name,
                    score_name: analysis_slice[score_name].iloc[0],
                    f"{score_name}_std": analysis_slice[f"{score_name}_std"].iloc[0],
                })
    
    for score_name in scores:
        best_acc = pd.DataFrame(best_models[score_name])

        count_of_datasets = len(all_datasets)
        current_counter = 0

        transformed_rows = []
        actual_row = {}

        for i, row in best_acc.iterrows():
            dataset_name = row["dataset"]

            if current_counter == count_of_datasets:
                transformed_rows.append(actual_row)
                current_counter = 0

                actual_row = {}
                actual_row["model"] = row["model_name"]
                actual_row[dataset_name] = row[score_name]
                actual_row[f"{dataset_name}_std"] = row[f"{score_name}_std"]
            
            else:
                actual_row["model"] = row["model_name"]
                actual_row[dataset_name] = row[score_name]
                actual_row[f"{dataset_name}_std"] = row[f"{score_name}_std"]
            
            current_counter += 1
    
        t = pd.DataFrame(transformed_rows)
        for dataset_name in all_datasets:
            max_for_dataset = t.max()[dataset_name]

            mask = t[dataset_name] == max_for_dataset

            t[f"best_{dataset_name}"] = 0
            t.loc[mask, f"best_{dataset_name}"] = 1

        best_cols = [col for col in t.columns if "best_" in col]
        t["best_awards"] = t[best_cols].sum(axis=1)

        t.to_csv(f"./data/best_for_{score_name}.csv", index=False)


