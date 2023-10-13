import pandas as pd


def expect_data_frames_to_be_equal(exercise: pd.DataFrame, expect: pd.DataFrame):
    """
    Asserts that the exercise data frame has the same number or rows, at least the same columns,
    and the same values as the expected data frame.
    """

    exercise_data = exercise.fillna("")
    expect_data = expect.fillna("")

    msg = f"expected rows quantity: {expect_data.shape[0]}, got rows quantity: {exercise_data.shape[0]}"
    assert exercise_data.shape[0] == expect_data.shape[0], msg

    msg = f"expected columns not found in exercise: {set(expect_data.columns) - set(exercise_data.columns)}"
    assert set(expect_data.columns).issubset(set(exercise_data.columns)), msg

    msg = f"expected index: {expect_data.index.values}, got index: {exercise_data.index.values}"
    assert (exercise_data.index == expect_data.index).all(), msg

    cols = expect_data.columns
    rounded_exercise_data = exercise_data[cols].round(4)
    rounded_expect_data = expect_data[cols].round(4)

    diffs = rounded_exercise_data.ne(rounded_expect_data).any(axis=1)
    msg = f"expected datasets to match, but they didn't: \n\ngot:\n{rounded_exercise_data[diffs]}\n---\nexpected:\n{rounded_expect_data[diffs]}"
    assert int(diffs.sum()) == 0, msg
