from skmultilearn.dataset import load_dataset
from support import CalculateLabelsCorrelationWithFTest
import pandas as pd
from helper import expect_data_frames_to_be_equal
import os


def test_calculate_labels_correlation_with_f_test():
    train_data = load_dataset("scene", "train")
    _, y_train, _, _ = train_data

    ccf = CalculateLabelsCorrelationWithFTest(alpha=1)
    res = ccf.get(y_train)

    f = os.path.join(
        "test_data", "expected_calculate_labels_correlation_with_f_test.csv"
    )
    expected = pd.read_csv(f)

    expect_data_frames_to_be_equal(res, expected)
