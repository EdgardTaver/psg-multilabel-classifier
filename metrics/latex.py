import pandas as pd

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

SCORES_TRANSLATION = {
    "accuracy": "acurácia",
    "hamming_loss": "perda de \\textit{{Hamming}}",
    "f1": "f1",
}

SCORES_CAMEL_CASE = {
    "accuracy": "Accuracy",
    "hamming_loss": "HammingLoss",
    "f1": "F1",
}

TABLE_TEMPLATE = """
\\begin{{table}}[htbp]
	\\centering
	\\caption{{Resultados obtidos para {score_name} ({n})}}
		\\begin{{tabular}}
        {{ p{{0.88in}} p{{0.88in}} p{{0.88in}} p{{0.88in}} p{{0.88in}} p{{0.88in}} }}
        
        {content}

        \\end{{tabular}}
	\\label{{tab:metricsFor{score_name_camel_case}_{n}}}
  \\source{{Edgard Taver, 2023}}
\\end{{table}}
"""

def ranked_metrics_to_latex(ranked_metrics: pd.DataFrame, score_name: str) -> None:
    def get_tuple(dataset_name: str) -> str:
        dataset_name_std = f"{dataset_name}_std"
        
        raw_value = str(row[dataset_name])
        brazilian_value = raw_value.replace(".", ",").ljust(6, "0")

        raw_std = str(row[dataset_name_std])
        brazilian_std = raw_std.replace(".", ",").ljust(6, "0")

        if score_name == "hamming_loss":
            brazilian_value = brazilian_value.replace("-", "")
            brazilian_std = brazilian_std.replace("-", "")

        best_dataset_name = f"best_{dataset_name}"
        if row[best_dataset_name] == 1:
            return f"\\textbf{{{brazilian_value}}} \\newline ($\\sigma$ {brazilian_std})"
        else:
            return f"{brazilian_value} \\newline ($\\sigma$ {brazilian_std})"

    def italic(text: str) -> str:
        return f"\\textit{{{text}}}"
    
    def bold(text: str) -> str:
        return f"\\textbf{{{text}}}"
    
    def safe_underscore(text: str) -> str:
        return text.replace("_", "\\_")

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

    line = "{model_name} & {value_1} & {value_2} & {value_3} & {value_4} & {value_5} \\\\ \\\\"
    # the extra "\\" at the end is to ensure some space between the lines

    header = "\\hline\n{model_col} & {data_1} & {data_2} & {data_3} & {data_4} & {data_5} \\\\ \n\\hline \\\\"
    footer = "\\hline \\ \n {diff_coll} & {diff_1} & {diff_2} & {diff_3} & {diff_4} & {diff_5} \\\\ \n\\hline \\\\"

    header_for_first_part = header.format(
        model_col="Modelo",
        data_1=italic(safe_underscore(datasets_for_first_part[0])),
        data_2=italic(safe_underscore(datasets_for_first_part[1])),
        data_3=italic(safe_underscore(datasets_for_first_part[2])),
        data_4=italic(safe_underscore(datasets_for_first_part[3])),
        data_5=italic(safe_underscore(datasets_for_first_part[4])),
    )

    def get_diff(dataset: str):
        max_ = ranked_metrics[dataset].max()
        min_ = ranked_metrics[dataset].min()

        diff_ = str(round(max_ - min_, 4))
        return diff_.replace(".", ",").ljust(6, "0")

    footer_for_first_part = footer.format(
        diff_coll="Diferença",
        diff_1=get_diff(datasets_for_first_part[0]),
        diff_2=get_diff(datasets_for_first_part[1]),
        diff_3=get_diff(datasets_for_first_part[2]),
        diff_4=get_diff(datasets_for_first_part[3]),
        diff_5=get_diff(datasets_for_first_part[4]),
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
    
    header_for_second_part = header.format(
        model_col="Modelo",
        data_1=italic(safe_underscore(datasets_for_second_part[0])),
        data_2=italic(safe_underscore(datasets_for_second_part[1])),
        data_3=italic(safe_underscore(datasets_for_second_part[2])),
        data_4=bold("Vitórias"),
        data_5=bold("Rank"),
    )

    footer_for_second_part = footer.format(
        diff_coll="Diferença",
        diff_1=get_diff(datasets_for_second_part[0]),
        diff_2=get_diff(datasets_for_second_part[1]),
        diff_3=get_diff(datasets_for_second_part[2]),
        diff_4="",
        diff_5="",
    )

    liner_for_second_part = []
    for i, row in ranked_metrics.iterrows():
        f_line = line.format(
            model_name=MODEL_NAME_MAP[row["model"]],
            value_1=get_tuple(datasets_for_second_part[0]),
            value_2=get_tuple(datasets_for_second_part[1]),
            value_3=get_tuple(datasets_for_second_part[2]),
            value_4=row["best_awards"],
            value_5=str(row["rank_average"]).replace(".", ",").ljust(6, "0"),
        )
        liner_for_second_part.append(f_line)

    table_content_1 = header_for_first_part
    table_content_1 += "\n\n"
    table_content_1 += "\n".join(lines_for_first_part)
    table_content_1 += "\n\n"
    table_content_1 += footer_for_first_part
    
    table_content_2 = header_for_second_part
    table_content_2 += "\n\n"
    table_content_2 += "\n".join(liner_for_second_part)
    table_content_2 += "\n\n"
    table_content_2 += footer_for_second_part

    full_table_content_1 = TABLE_TEMPLATE.format(
        score_name=SCORES_TRANSLATION[score_name],
        score_name_camel_case=SCORES_CAMEL_CASE[score_name],
        content=table_content_1,
        n=1,
    )

    full_table_content_2 = TABLE_TEMPLATE.format(
        score_name=SCORES_TRANSLATION[score_name],
        score_name_camel_case=SCORES_CAMEL_CASE[score_name],
        content=table_content_2,
        n=2,
    )
    
    with open(f"./data/table_{score_name}.tex", "w", encoding="utf8") as f:
        f.write(full_table_content_1)
        f.write(full_table_content_2)
