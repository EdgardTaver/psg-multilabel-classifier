import pandas as pd

ALL_DATASETS = [
    "birds",
    "emotions",
    "scene",
    "yeast",
    "enron",
    "genbase",
    "medical",
    "tmc2007_500",
]

def analyse_summarized_metrics(summarized_metrics: pd.DataFrame) -> pd.DataFrame:
    analysis = summarized_metrics.copy()

    analysis["model_name"] = analysis["model"].apply(lambda x: x.split("-")[0])

    scores = [
        "accuracy",
        "hamming_loss",
        "f1"
    ]

    best_models = {}

    # finds best model setup for each model and each dataset
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
    
    # will check the best model across all datasets
    # one check for each metric
    for score_name in scores:
        best_for_score = pd.DataFrame(best_models[score_name])

        count_of_datasets = len(ALL_DATASETS)
        current_counter = 0

        transformed_rows = []
        actual_row = {}

        for i, row in best_for_score.iterrows():
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
    
        transposed_for_score = pd.DataFrame(transformed_rows)
        best_for_score = calculate_best_awards(transposed_for_score)
        best_for_score.to_csv(f"./data/best_for_{score_name}.csv", index=False)

        ranked_best_for_score = calculate_ranking(best_for_score)
        ranked_best_for_score.to_csv(f"./data/ranked_best_for_{score_name}.csv", index=False)


def calculate_best_awards(models_to_datasets_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the best model for each dataset.
    When the model is the best, it gets a 1 in the column `best_{dataset_name}`.
    
    A final column is added to the data frame, `best_awards`,
    which is the sum of all `best_{dataset_name}` columns.
    """

    df = models_to_datasets_matrix.copy()

    for dataset_name in ALL_DATASETS:
        max_for_dataset = df.max()[dataset_name]

        mask = df[dataset_name] == max_for_dataset

        df[f"best_{dataset_name}"] = 0
        df.loc[mask, f"best_{dataset_name}"] = 1

    best_cols = [col for col in df.columns if "best_" in col]
    df["best_awards"] = df[best_cols].sum(axis=1)

    return df

def calculate_ranking(models_to_datasets_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a rank for each model, based on its performance in each dataset.
    The lower the value, the better the model.

    A final column is added to the data frame, `rank_average`,
    which is the average of all `rank_{dataset_name}` columns.
    """

    df = models_to_datasets_matrix.copy()

    for dataset_name in ALL_DATASETS:
        df[f"rank_{dataset_name}"] = df[dataset_name].rank(method="min", ascending=False)
    
    rank_cols = [col for col in df.columns if "rank_" in col]
    df["rank_average"] = df[rank_cols].mean(axis=1)

    df = df.sort_values(by="rank_average", ascending=True)
    return df