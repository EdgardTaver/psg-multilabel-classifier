from typing import Any, List

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.feature_selection import f_classif


class CalculateLabelsCorrelationWithFTest:
    def __init__(
        self,
        alpha: float,
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

                f_test_result = f_classif(base_label, against_label)[0][0]
                f_test_result = np.round(f_test_result, 8)

                results.append(
                    {
                        "label_being_tested": i,
                        "against_label": j,
                        "f_test_result": f_test_result,
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

            big_f = np.round(split_df["f_test_result"].sum(), 8)
            max_cum_f = np.round(self.alpha * big_f, 8)

            cum_f = 0
            for _, row in split_df.iterrows():
                cum_f += row["f_test_result"]
                cum_f = np.round(cum_f, 8)
                # these rounding operations are necessary because of the
                # floating point arithmetic

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


class ConditionalEntropies:
    def calculate(self, y: Any) -> List[List[float]]:
        dense_y = y.todense()

        label_count = dense_y.shape[1]

        results = []
        for label_x in range(label_count):
            results.append([])
            for label_y in range(label_count):
                y_label_specific_x = np.asarray(dense_y[:, label_x]).reshape(-1)
                y_label_specific_y = np.asarray(dense_y[:, label_y]).reshape(-1)
                
                conditional_entropy = drv.entropy_conditional(
                    y_label_specific_x.tolist(),
                    y_label_specific_y.tolist())
                
                results[label_x].append(float(conditional_entropy))
        
        return results

class MutualInformation:
    def calculate(self, y: Any) -> List[List[float]]:
        dense_y = y.todense()

        results = []
        for label_x in range(dense_y.shape[1]):
            results.append([])
            for label_y in range(dense_y.shape[1]):
                y_label_specific_x = np.asarray(dense_y[:, label_x]).reshape(-1)
                y_label_specific_y = np.asarray(dense_y[:, label_y]).reshape(-1)
                
                e = drv.information_mutual(y_label_specific_x.tolist(), y_label_specific_y.tolist())
                results[label_x].append(e)

        return results


def build_chain_based_on_f_test(f_test_results: pd.DataFrame, ascending_chain: bool) -> List[int]:
    chain = []
    sorted_res = f_test_results.sort_values(by=["f_test_result"], ascending=ascending_chain)
    
    element = int(sorted_res.iloc[0]["for_label"])
    chain.append(element)

    m = ~sorted_res["expand_this_label"].isin(chain)
    m &= sorted_res["for_label"] == element
    
    while m.sum() > 0:
        sliced_res = sorted_res[m]
        sorted_sliced_res = sliced_res.sort_values(by=["f_test_result"], ascending=ascending_chain)

        element = int(sorted_sliced_res.iloc[0]["expand_this_label"])
        chain.append(element)

        m = ~sorted_res["expand_this_label"].isin(chain)
        m &= sorted_res["for_label"] == element
    
    return chain