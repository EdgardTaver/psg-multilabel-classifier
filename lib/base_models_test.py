from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.dataset import load_dataset
from sklearn.svm import SVC
from skmultilearn.base.problem_transformation import ProblemTransformationBase
from typing import List, Optional, Any, Tuple, Dict
import numpy as np
import sklearn.metrics as metrics
import json
import pandas as pd
from sklearn.feature_selection import f_classif
from evaluation import EvaluationPipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import copy
from lib.base_models import DependantBinaryRelevance

def test_dependant_binary_relevance():
    dataset_name = "scene"
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")

    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")

    X_test, y_test, _, _ = test_dataset

    model = DependantBinaryRelevance(
        classifier=RandomForestClassifier(random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    assert hamming_loss == 0.08737458193979933
    assert f1_score == 0.7065088014940012
