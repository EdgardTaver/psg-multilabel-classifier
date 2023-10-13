from typing import Any
import numpy as np

import pandas as pd
from sklearn.feature_selection import f_classif


class CalculateLabelsCorrelationWithFTest:
    def __init__(
        self,
        alpha: float = 0.5,
    ):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be >= 0.0 and <= 1.0")

        self.alpha = alpha

    def get(self, labels: Any) -> pd.DataFrame:
        labels_count = labels.shape[1]

        f_tested_label_pairs = self.calculate_f_test_for_all_label_pairs(
            labels_count, labels
        )

        correlated_labels_map = self.get_map_of_correlated_labels(
            labels_count, f_tested_label_pairs
        )

        return correlated_labels_map

    def calculate_f_test_for_all_label_pairs(
        self, labels_count: int, labels: Any
    ) -> pd.DataFrame:
        results = []
        for i in range(0, labels_count):
            for j in range(0, labels_count):
                if i == j:
                    continue

                X = labels.todense()[:, i]
                base_label = self.convert_matrix_to_array(X)

                y = labels.todense()[:, j]
                against_label = self.convert_matrix_to_vector(y)

                f_test_result = f_classif(base_label, against_label)[0]

                results.append(
                    {
                        "label_being_tested": i,
                        "against_label": j,
                        "f_test_result": float(f_test_result),
                    }
                )

        return pd.DataFrame(results)

    def convert_matrix_to_array(self, matrix: Any):
        return np.asarray(matrix).reshape(-1, 1)

    def convert_matrix_to_vector(self, matrix: Any):
        return np.asarray(matrix).reshape(-1)

    def get_map_of_correlated_labels(
        self, labels_count: int, f_test_results: pd.DataFrame
    ) -> pd.DataFrame:
        sorted_f_test_results = f_test_results.sort_values(
            by=["label_being_tested", "f_test_result"], ascending=[True, False]
        )
        # ordering in descending order by the F-test result,
        # following what the main article describes

        selected_features = []
        for i in range(0, labels_count):
            mask = sorted_f_test_results["label_being_tested"] == i
            split_df = sorted_f_test_results[mask].reset_index(drop=True)

            big_f = split_df["f_test_result"].sum()
            max_cum_f = self.alpha * big_f

            cum_f = 0
            for _, row in split_df.iterrows():
                cum_f += row["f_test_result"]
                if cum_f > max_cum_f:
                    break

                selected_features.append(
                    {
                        "for_label": i,
                        "expand_this_label": int(row["against_label"]),
                        "f_test_result": float(row["f_test_result"]),
                    }
                )

        cols = ["for_label", "expand_this_label", "f_test_result"]
        return pd.DataFrame(selected_features, columns=cols)
