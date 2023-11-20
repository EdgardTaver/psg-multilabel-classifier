import pandas as pd
from metrics.types import RawEvaluationResults
from metrics.evaluation import EvaluationPipelineResult
from metrics.support import evaluation_results_to_flat_table
from metrics.test import expect_data_frames_to_be_equal

from numpy import array

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
            "fit_time_1:2": 2.47142076,
            "fit_time_2:2": 1.67912555,
            "score_time_1:2": 1.19149733,
            "score_time_2:2": 1.23108363,
            "test_accuracy_1:2": 0.60880399,
            "test_accuracy_2:2": 0.58686617,
            "train_accuracy_1:2": 0.719867,
            "train_accuracy_2:2": 0.71345515,
            "test_hamming_loss_1:2": -0.08388704,
            "test_hamming_loss_2:2": -0.08464949,
            "train_hamming_loss_1:2": -0.05444722,
            "train_hamming_loss_2:2": -0.05647841,
            "test_f1_1:2": 0.73406644,
            "test_f1_2:2": 0.72308818,
            "train_f1_1:2": 0.83046478,
            "train_f1_2:2": 0.83018986
        },
        {
            "model": "any_model_1",
            "dataset": "any_dataset_2",
            "fit_time_1:2": 2.47142076,
            "fit_time_2:2": 1.67912555,
            "score_time_1:2": 1.19149733,
            "score_time_2:2": 1.23108363,
            "test_accuracy_1:2": 0.60880399,
            "test_accuracy_2:2": 0.58686617,
            "train_accuracy_1:2": 0.719867,
            "train_accuracy_2:2": 0.71345515,
            "test_hamming_loss_1:2": -0.08388704,
            "test_hamming_loss_2:2": -0.08464949,
            "train_hamming_loss_1:2": -0.05444722,
            "train_hamming_loss_2:2": -0.05647841,
            "test_f1_1:2": 0.73406644,
            "test_f1_2:2": 0.72308818,
            "train_f1_1:2": 0.83046478,
            "train_f1_2:2": 0.83018986
        },
        {
            "model": "any_model_2",
            "dataset": "any_dataset_1",
            "fit_time_1:2": 2.47142076,
            "fit_time_2:2": 1.67912555,
            "score_time_1:2": 1.19149733,
            "score_time_2:2": 1.23108363,
            "test_accuracy_1:2": 0.60880399,
            "test_accuracy_2:2": 0.58686617,
            "train_accuracy_1:2": 0.719867,
            "train_accuracy_2:2": 0.71345515,
            "test_hamming_loss_1:2": -0.08388704,
            "test_hamming_loss_2:2": -0.08464949,
            "train_hamming_loss_1:2": -0.05444722,
            "train_hamming_loss_2:2": -0.05647841,
            "test_f1_1:2": 0.73406644,
            "test_f1_2:2": 0.72308818,
            "train_f1_1:2": 0.83046478,
            "train_f1_2:2": 0.83018986
        },
    ])

    expect_data_frames_to_be_equal(exercise, expected)
