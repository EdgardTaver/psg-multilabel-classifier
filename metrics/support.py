import pandas as pd
from metrics.types import RawEvaluationResults


def evaluation_results_to_flat_table(evaluation_results: RawEvaluationResults) -> pd.DataFrame:
    flat_table = pd.DataFrame()
    return flat_table