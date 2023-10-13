from skmultilearn.dataset import load_dataset
from support import CalculateLabelsCorrelationWithFTest
import pandas as pd
from helper import expect_data_frames_to_be_equal


def test_bla():
    train_data = load_dataset("scene", "train")
    test_data = load_dataset("scene", "test")

    X_train, y_train, _, _ = train_data
    X_test, y_test, _, _ = test_data

    ccf = CalculateLabelsCorrelationWithFTest(alpha=1)
    res = ccf.fit(X_train, y_train)

    expected = pd.read_csv("bla.csv")

    expect_data_frames_to_be_equal(res, expected)
