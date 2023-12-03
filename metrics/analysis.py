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


def analyse_summarized_metrics(summarized_metrics: pd.DataFrame, ranked_file_name: str) -> pd.DataFrame:
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
        ranked_for_score = calculate_ranking(best_for_score)
        ranked_for_score.to_csv(ranked_file_name.format(score_name=score_name), index=False)

        ranked_metrics_to_latex(ranked_for_score, score_name)


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

MODEL_NAME_MAP = {
    "baseline_binary_relevance_model": "(B)BR",
    "baseline_stacked_generalization": "(B)SG",
    "baseline_dependant_binary_relevance": "(B)DBR",
    "baseline_classifier_chain": "(B)CC",
    "stacking_with_f_tests": "S-F",
    "classifier_chain_with_f_test_ordering": "CC-F",
    "classifier_chain_with_lop": "CC-LOP",
    "partial_classifier_chain_with_lop":"PCC-LOP",
    "classifier_chain_with_genetic_algorithm":"CC-GA",
}

def ranked_metrics_to_latex(ranked_metrics: pd.DataFrame, score_name: str) -> str:
    datasets_for_first_part = [
        "emotions",
        "scene",
        "yeast",
        "birds",
        "tmc2007_500",
    ]

    datasets_for_second_part = [
        "genbase",
        "medical",
        "enron",
    ]

    def get_tuple(dataset_name: str) -> str:
        dataset_name_std = f"{dataset_name}_std"
        
        raw_value = str(row[dataset_name])
        brazilian_value = raw_value.replace(".", ",")

        raw_std = str(row[dataset_name_std])
        brazilian_std = raw_std.replace(".", ",")

        best_dataset_name = f"best_{dataset_name}"
        if row[best_dataset_name] == 1:
            return f"\\textbf{{{brazilian_value}}} \\newline ($\\sigma$ {brazilian_std})"
        else:
            return f"{brazilian_value} \\newline ($\\sigma$ {brazilian_std})"


    line = "{model_name} & {value_1} & {value_2} & {value_3} & {value_4} & {value_5} \\\\ \\\\"
    # the extra "\\" at the end is to ensure some space between the lines

    header = "\hline\n{model_col} & {data_1} & {data_2} & {data_3} & {data_4} & {data_5} \\\\ \n\hline"

    def italic(text: str) -> str:
        return f"\\textit{{{text}}}"
    
    def safe_underscore(text: str) -> str:
        return text.replace("_", "\\_")

    header_for_first_part = header.format(
        model_col="Modelo",
        data_1=italic(datasets_for_first_part[0]),
        data_2=italic(datasets_for_first_part[1]),
        data_3=italic(datasets_for_first_part[2]),
        data_4=italic(datasets_for_first_part[3]),
        data_5=italic(safe_underscore(datasets_for_first_part[4])),
    )

    lines_for_first_part = []
    for i, row in ranked_metrics.iterrows():
        f_line = line.format(
            model_name=MODEL_NAME_MAP[row["model"]],
            value_1=get_tuple(datasets_for_first_part[0]),
            value_2=get_tuple(datasets_for_first_part[1]),
            value_3=get_tuple(datasets_for_first_part[2]),
            value_4=get_tuple(datasets_for_first_part[3]),
            value_5=get_tuple(datasets_for_first_part[4]),
        )
        lines_for_first_part.append(f_line)
    
    with open("bla.tex", "w") as f:
        f.write(header_for_first_part)
        f.write("\n\n")
        f.write("\n".join(lines_for_first_part))



"""
\begin{table}[htbp]
	\centering
	\caption{Tabela para medida XXX (1)}
		\begin{tabular}
        {p{0.88in} p{0.88in} p{0.88in} p{0.88in} p{0.88in} p{0.88in}}
        
        \hline
        Model & birds & emotions & scene & yeast & enron \\
        \hline
        
        \textit{(B)CC} & 0,4933; 0,0655 & 0,3255; 0,0342 & 0,3255; 0,0342 & 0,3255; 0,0342 & 1;2 \\
        \textit{(B)BR} & 0,4933; 0,0655 & 0,3255; 0,0342 & 0,3255; 0,0342 & 0,3255; 0,0342 & 1;2 \\

        \vspace*{0.25cm} \\
        \hline
        Model & genbase & medical & tmc2007 & Tops & Rank \\
        \hline

        \textit{SF} & 0,4933; 0,0655 & 0,3255; 0,0342 & 0,3255; 0,0342 & 2 & 4 \\
        \textit{LOP} & 0,4933; 0,0655 & 0,3255; 0,0342 & 0,3255; 0,0342 & 3 & 5 \\

        
        \end{tabular}
	\label{tab:metricsForXXX}
  \source{Edgard Taver, 2023}
\end{table}
"""