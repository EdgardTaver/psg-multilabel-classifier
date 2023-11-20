import pandas as pd
from numpy import array

from metrics.evaluation import EvaluationPipelineResult
from metrics.support import (evaluation_results_to_flat_table,
                             flat_table_to_evaluation_results)
from metrics.test import expect_data_frames_to_be_equal
from metrics.types import RawEvaluationResults


def test_evaluation_results_to_flat_table():
    evaluation_results: RawEvaluationResults = {}

    evaluation_results["any_model_1"] = {}
    evaluation_results["any_model_1"]["any_dataset_1"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    evaluation_results["any_model_1"]["any_dataset_2"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    evaluation_results["any_model_2"] = {}
    evaluation_results["any_model_2"]["any_dataset_1"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    exercise = evaluation_results_to_flat_table(evaluation_results)

    expected = pd.DataFrame([
        {
            "model": "any_model_1",
            "dataset": "any_dataset_1",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
        {
            "model": "any_model_1",
            "dataset": "any_dataset_2",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
        {
            "model": "any_model_2",
            "dataset": "any_dataset_1",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
    ])

    expect_data_frames_to_be_equal(exercise, expected)

def test_flat_table_to_evaluation_results():
    flat_table = pd.DataFrame([
        {
            "model": "any_model_1",
            "dataset": "any_dataset_1",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
        {
            "model": "any_model_1",
            "dataset": "any_dataset_2",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
        {
            "model": "any_model_2",
            "dataset": "any_dataset_1",
            "fit_time-1:2": 2.47142076,
            "fit_time-2:2": 1.67912555,
            "score_time-1:2": 1.19149733,
            "score_time-2:2": 1.23108363,
            "test_accuracy-1:2": 0.60880399,
            "test_accuracy-2:2": 0.58686617,
            "train_accuracy-1:2": 0.719867,
            "train_accuracy-2:2": 0.71345515,
            "test_hamming_loss-1:2": -0.08388704,
            "test_hamming_loss-2:2": -0.08464949,
            "train_hamming_loss-1:2": -0.05444722,
            "train_hamming_loss-2:2": -0.05647841,
            "test_f1-1:2": 0.73406644,
            "test_f1-2:2": 0.72308818,
            "train_f1-1:2": 0.83046478,
            "train_f1-2:2": 0.83018986
        },
    ])
    
    expected_evaluation_results = {}
    expected_evaluation_results["any_model_1"] = {}
    expected_evaluation_results["any_model_1"]["any_dataset_1"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    expected_evaluation_results["any_model_1"]["any_dataset_2"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    expected_evaluation_results["any_model_2"] = {}
    expected_evaluation_results["any_model_2"]["any_dataset_1"] = EvaluationPipelineResult({
        "fit_time": array([2.47142076, 1.67912555]),
        "score_time": array([1.19149733, 1.23108363]),
        "test_accuracy": array([0.60880399, 0.58686617]),
        "train_accuracy": array([0.719867  , 0.71345515]),
        "test_hamming_loss": array([-0.08388704, -0.08464949]),
        "train_hamming_loss": array([-0.05444722, -0.05647841]),
        "test_f1": array([0.73406644, 0.72308818]),
        "train_f1": array([0.83046478, 0.83018986])
    })

    exercise = flat_table_to_evaluation_results(flat_table)
    print(exercise["any_model_1"]["any_dataset_1"].raw())
    print(expected_evaluation_results["any_model_1"]["any_dataset_1"].raw())

    assert exercise == expected_evaluation_results
