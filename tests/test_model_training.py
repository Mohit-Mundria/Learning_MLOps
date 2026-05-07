# tests/test_model_training.py

import pytest
import pandas as pd
import os
from unittest.mock import patch

from src.model_training import (
    load_training_data,
    load_testing_data,
    objective,
)

# ══════════════════════════════════════════
# SECTION 1: Testing data loading functions
# ══════════════════════════════════════════

def test_load_training_data_returns_correct_shapes(tmp_path, sample_processed_df):
    """
    load_training_data should split the df into:
    - x_train: all columns except ROI_Score
    - y_train: only ROI_Score column
    """
    csv_path = tmp_path / "train.csv"
    sample_processed_df.to_csv(csv_path, index=False)

    x_train, y_train = load_training_data(str(csv_path))

    # ROI_Score must NOT be in features
    assert "ROI_Score" not in x_train.columns

    # y_train must be a Series with same length
    assert len(y_train) == len(sample_processed_df)


# def test_load_testing_data_returns_correct_shapes(tmp_path, sample_processed_df):
#     """Same logic applies for test data loading."""
#     csv_path = tmp_path / "test.csv"
#     sample_processed_df.to_csv(csv_path, index=False)

#     x_test, y_test = load_testing_data(str(csv_path))

#     assert "ROI_Score" not in x_test.columns
#     assert len(y_test) == len(sample_processed_df)


def test_load_training_data_x_and_y_same_length(tmp_path, sample_processed_df):
    """
    x_train and y_train must always have matching
    number of rows — otherwise model.fit() will crash.
    """
    csv_path = tmp_path / "train.csv"
    sample_processed_df.to_csv(csv_path, index=False)

    x_train, y_train = load_training_data(str(csv_path))

    assert len(x_train) == len(y_train)


# ══════════════════════════════════════════
# SECTION 2: Testing objective() function
# ══════════════════════════════════════════

# def test_objective_returns_float(sample_processed_df):
#     """
#     objective() is your Optuna function.
#     It must return a float (the R2 score).
#     We test it by creating a fake Optuna trial
#     using unittest.mock.
#     """
#     from unittest.mock import MagicMock

#     # MagicMock creates a fake object.
#     # We tell it exactly what to return for each method call
#     # so objective() runs without needing real Optuna.
#     mock_trial = MagicMock()
#     mock_trial.suggest_categorical.return_value = "RandomForestRegressor"
#     mock_trial.suggest_int.return_value = 10
#     mock_trial.suggest_float.return_value = 0.1

#     x_train = sample_processed_df.drop(columns=["ROI_Score"])
#     y_train = sample_processed_df["ROI_Score"]

#     # Use a bigger dataset so cross_val_score (cv=5) doesn't crash
#     # (needs at least 5 samples)
#     big_x = pd.concat([x_train] * 5, ignore_index=True)
#     big_y = pd.concat([y_train] * 5, ignore_index=True)

#     result = objective(mock_trial, big_x, big_y)

#     assert isinstance(result, float), f"Expected float, got {type(result)}"


def test_objective_score_is_valid_r2(sample_processed_df):
    """
    R2 score can theoretically be negative (very bad model)
    but for a sanity check on clean data, it should be
    a real number, not NaN or Infinity.
    """
    from unittest.mock import MagicMock
    import math

    mock_trial = MagicMock()
    mock_trial.suggest_categorical.return_value = "RandomForestRegressor"
    mock_trial.suggest_int.return_value = 10
    mock_trial.suggest_float.return_value = 0.1

    x_train = sample_processed_df.drop(columns=["ROI_Score"])
    y_train = sample_processed_df["ROI_Score"]

    big_x = pd.concat([x_train] * 5, ignore_index=True)
    big_y = pd.concat([y_train] * 5, ignore_index=True)

    result = objective(mock_trial, big_x, big_y)

    assert not math.isnan(result),  "R2 score is NaN — something is wrong"
    assert not math.isinf(result),  "R2 score is Infinite — something is wrong"