import pandas as pd

def expect_data_frames_to_be_equal(exercise: pd.DataFrame, expect: pd.DataFrame):
    exercise_data = exercise.fillna("")
    expect_data = expect.fillna("")

    msg = f"expected rows quantity: {expect_data.shape[0]}, got rows quantity: {exercise_data.shape[0]}"
    assert exercise_data.shape[0] == expect_data.shape[0], msg

    msg = f"expected columns not found in exercise: {set(expect_data.columns) - set(exercise_data.columns)}"
    assert set(expect_data.columns).issubset(set(exercise_data.columns)), msg
    
    msg = f"expected index: {expect_data.index.values}, got index: {exercise_data.index.values}"
    assert (exercise_data.index == expect_data.index).all(), msg

    cols = expect_data.columns
    diffs = exercise_data[cols].ne(expect_data[cols]).any(axis=1)
    msg = f"expected datasets to match, but they didn't: \n\ngot:\n{exercise_data[diffs]}\n---\nexpected:\n{expect_data[diffs]}"
    assert int(diffs.sum()) == 0, msg