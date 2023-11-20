import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.dataset import load_dataset

from lib.classifiers import (ClassifierChainWithFTestOrdering,
                             ClassifierChainWithGeneticAlgorithm,
                             ClassifierChainWithLOP,
                             PartialClassifierChainWithLOP, StackingWithFTests)


def test_stacking_with_f_tests():
    dataset_name = "scene"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = StackingWithFTests(
        alpha=0.5,
        base_classifier=RandomForestClassifier(random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    assert hamming_loss == 0.0862597547380156
    assert f1_score == 0.7105885090069132

def test_classifier_chain_with_f_test_ordering_regular():
    """
    Uses the regular ordering of the chain, which is the descending order
    of the F-Test correlations.
    """

    dataset_name = "scene"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = ClassifierChainWithFTestOrdering(
        ascending_chain=False,
        base_classifier=RandomForestClassifier(random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    assert hamming_loss == 0.08876811594202899
    assert f1_score == 0.6952594727561733

def test_classifier_chain_with_f_test_ordering_ascending():
    """
    Uses the ascending ordering of the chain, which is the ascending order
    of the F-Test correlations.
    """

    dataset_name = "scene"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = ClassifierChainWithFTestOrdering(
        ascending_chain=True,
        base_classifier=RandomForestClassifier(random_state=42),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    assert hamming_loss == 0.08946488294314381
    assert f1_score == 0.6900349522611511

def test_classifier_chain_with_genetic_algorithm():
    dataset_name = "emotions"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = ClassifierChainWithGeneticAlgorithm(
        base_classifier=RandomForestClassifier(random_state=456),
        num_generations=1,
        random_state=123,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    assert hamming_loss == 0.2004950495049505
    assert f1_score == 0.6529424269090122

def test_classifier_chain_with_lop():
    dataset_name = "scene"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = ClassifierChainWithLOP(
        base_classifier=RandomForestClassifier(random_state=456),
        num_generations=20,
        random_state=123,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    print(hamming_loss)
    print(f1_score)

    assert hamming_loss == 0.09002229654403568
    assert f1_score == 0.6870162085376208

def test_partial_classifier_chain_with_lop():
    dataset_name = "scene"
    
    train_dataset = load_dataset(dataset_name, "train")
    if train_dataset is None:
        raise Exception("could not load dataset")
    X_train, y_train, _, _ = train_dataset

    test_dataset = load_dataset(dataset_name, "test")
    if test_dataset is None:
        raise Exception("could not load dataset")
    X_test, y_test, _, _ = test_dataset

    model = PartialClassifierChainWithLOP(
        base_classifier=RandomForestClassifier(random_state=456),
        num_generations=20,
        random_state=123,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hamming_loss = metrics.hamming_loss(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions, average="macro")

    print(hamming_loss)
    print(f1_score)

    assert hamming_loss == 0.0903010033444816
    assert f1_score == 0.6865867583024511