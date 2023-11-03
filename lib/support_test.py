import os

import pandas as pd
from skmultilearn.dataset import load_dataset

from lib.helper import expect_data_frames_to_be_equal
from lib.support import CalculateLabelsCorrelationWithFTest, ConditionalEntropies, MutualInformation


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

    assert exercise == expected

def test_calculate_mutual_information():
    train_data = load_dataset("scene", "train")
    _, y_train, _, _ = train_data

    mi = MutualInformation()
    exercise = mi.calculate(y_train)

    expected = [
        [0.6961030672262447, 0.04424598102040078, 0.05373790221224173, 0.05343615400827262, 0.020452654062164233, 0.02404112369146061],
        [0.04424598102040078, 0.5743357592196299, 0.03779360604514781, 0.037582141934937585, 0.05549564312085209, 0.04359110211677164],
        [0.05373790221224173, 0.03779360604514781, 0.6406718924240618, 0.01962213601236318, 0.042545897446177317, 0.052940844996621106],
        [0.05343615400827262, 0.037582141934937585, 0.01962213601236318, 0.638716343996387, 0.007179125020240096, 0.046673847330065854],
        [0.020452654062164233, 0.05549564312085209, 0.042545897446177094, 0.007179125020240096, 0.7758023710944532, 0.07135498901603798],
        [0.02404112369146061, 0.04359110211677164, 0.052940844996621106, 0.046673847330065854, 0.07135498901603798, 0.6908320386874189]
    ]

    assert exercise == expected