import os

import pandas as pd
from skmultilearn.dataset import load_dataset

from lib.helper import expect_data_frames_to_be_equal
from lib.support import CalculateLabelsCorrelationWithFTest, ConditionalEntropies


def test_calculate_labels_correlation_with_f_test():
    train_data = load_dataset("scene", "train")
    _, y_train, _, _ = train_data

    ccf = CalculateLabelsCorrelationWithFTest(alpha=1)
    exercise = ccf.get(y_train)

    f = os.path.join(
        "test_data", "expected_calculate_labels_correlation_with_f_test.csv"
    )
    expected = pd.read_csv(f)

    expect_data_frames_to_be_equal(exercise, expected)

def test_calculate_conditional_entropies():
    train_data = load_dataset("scene", "train")
    _, y_train, _, _ = train_data

    ce = ConditionalEntropies()
    exercise = ce.calculate(y_train)

    expected = [
        [0.0, 0.6518570862058439, 0.642365165014003, 0.6426669132179721, 0.6756504131640805, 0.6720619435347841],
        [0.5300897781992291, 0.0, 0.5365421531744821, 0.5367536172846923, 0.5188401160987778, 0.5307446571028582],
        [0.58693399021182, 0.6028782863789139, 0.0, 0.6210497564116986, 0.5981259949778844, 0.5877310474274406],
        [0.5852801899881144, 0.6011342020614494, 0.6190942079840238, 0.0, 0.6315372189761469, 0.5920424966663211],
        [0.7553497170322889, 0.7203067279736011, 0.7332564736482761, 0.7686232460742131, 0.0, 0.7044473820784152],
        [0.6667909149959583, 0.6472409365706473, 0.6378911936907978, 0.6441581913573531, 0.619477049671381, 0.0]
    ]

    print(exercise)

    assert exercise == expected