# tests/test_data_preprocessing.py

import pytest
import pandas as pd
from unittest.mock import patch

from src.data_preprocessing import (
    drop_col,
    fill_nan,
    one_hot_encoding,
    train_test_split_data
)


# ══════════════════════════════════════════
# SECTION 1: Testing drop_col()
# ══════════════════════════════════════════

def test_drop_col_removes_specified_column(sample_raw_df):
    """
    After dropping 'Useless_Col', it must not exist in df.
    sample_raw_df comes from conftest.py automatically.
    """
    result = drop_col(sample_raw_df.copy(), ["Project_ID"])
    assert "Project_ID" not in result.columns


def test_drop_col_keeps_other_columns(sample_raw_df):
    """
    Dropping one column must not affect others.
    """
    result = drop_col(sample_raw_df.copy(), ["Project_ID"])
    assert "ROI_Score" in result.columns
    # assert "Product_Type" in result.columns


def test_drop_col_multiple_columns(sample_raw_df):
    """
    Test dropping more than one column at once.
    """
    result = drop_col(sample_raw_df.copy(), ["Project_ID", "Region"])
    assert "Project_ID" not in result.columns
    # assert "Region" not in result.columns


# ══════════════════════════════════════════
# SECTION 2: Testing fill_nan()
# ══════════════════════════════════════════

def test_fill_nan_no_nulls_remain(sample_raw_df):
    """
    After fill_nan(), there should be zero null
    values anywhere in the DataFrame.
    """
    result = fill_nan(sample_raw_df.copy())

    total_nulls = result.isnull().sum().sum()
    assert total_nulls == 0, f"Expected 0 nulls, found {total_nulls}"


# def test_fill_nan_numeric_filled_with_mean(sample_raw_df):
#     """
#     The null in 'Age' column (index 1) should be
#     filled with the mean of [25.0, 30.0] = 27.5
#     """
#     result = fill_nan(sample_raw_df.copy())
#     assert result["Age"].iloc[1] == pytest.approx(27.5)

    # pytest.approx() is used for floating point comparisons
    # because 27.5000000001 != 27.5 in strict equality


# def test_fill_nan_failure_reason_filled_with_string(sample_raw_df):
#     """
#     'Failure_Reason' nulls should be filled
#     with "No failure" (special case in your code).
#     """
#     result = fill_nan(sample_raw_df.copy())
#     assert result["Failure_Reason"].iloc[0] == "No failure"
#     assert result["Failure_Reason"].iloc[2] == "No failure"


# def test_fill_nan_does_not_change_non_null_values(sample_raw_df):
#     """
#     Values that were NOT null should remain unchanged.
#     """
#     result = fill_nan(sample_raw_df.copy())
#     assert result["Age"].iloc[0] == 25.0
#     assert result["Age"].iloc[2] == 30.0


# ══════════════════════════════════════════
# SECTION 3: Testing one_hot_encoding()
# ══════════════════════════════════════════

def test_one_hot_encoding_creates_new_columns(sample_raw_df, mock_params):
    """
    After encoding, new columns like 'Product_Type_Electronics'
    should exist in the DataFrame.
    """
    df = fill_nan(sample_raw_df.copy())

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        result = one_hot_encoding(df)

    assert "Industry_Finance" in result.columns
    assert "Industry_Retail" in result.columns
    assert "Industry_Technology" in result.columns
    assert "Failure_Reason_Loop Error" in result.columns
    assert "Failure_Reason_No failure" in result.columns
    assert "Failure_Reason_Supply chain" in result.columns
    


def test_one_hot_encoding_removes_original_columns(sample_raw_df, mock_params):
    """
    Original categorical columns should be removed
    after encoding — they are replaced by new columns.
    """
    df = fill_nan(sample_raw_df.copy())

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        result = one_hot_encoding(df)

    # Original columns should be gone
    assert "Primary_Tool" not in result.columns
    assert "Industry" not in result.columns
    assert "Status" not in result.columns


def test_one_hot_encoding_preserves_target_column(sample_raw_df, mock_params):
    """
    ROI_Score (your target column) must NEVER be
    removed or changed during encoding.
    """
    df = fill_nan(sample_raw_df.copy())

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        result = one_hot_encoding(df)

    assert "ROI_Score" in result.columns


def test_one_hot_encoding_no_nulls(sample_raw_df, mock_params):
    """
    Encoding should not introduce any new null values.
    """
    df = fill_nan(sample_raw_df.copy())

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        result = one_hot_encoding(df)

    assert result.isnull().sum().sum() == 0


# ══════════════════════════════════════════
# SECTION 4: Testing train_test_split_data()
# ══════════════════════════════════════════

def test_train_test_split_creates_files(sample_processed_df, mock_params, tmp_path):
    """
    After splitting, both train and test CSV files
    must exist at the paths from params.
    """
    train_path = tmp_path / "train.csv"
    test_path  = tmp_path / "test.csv"

    mock_params["preprocessing"]["train_data_path"] = str(train_path)
    mock_params["preprocessing"]["test_data_path"]  = str(test_path)

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        train_test_split_data(sample_processed_df.copy())

    import os
    assert os.path.exists(train_path), "Train file was not created"
    assert os.path.exists(test_path),  "Test file was not created"


def test_train_test_split_correct_proportions(sample_processed_df, mock_params, tmp_path):
    """
    With test_size=0.2 and 10 rows, we expect:
    - 8 rows in train
    - 2 rows in test
    We use a larger df here for meaningful proportions.
    """
    # Create a 10-row version for this test
    big_df = pd.concat([sample_processed_df] * 4, ignore_index=True)

    train_path = tmp_path / "train.csv"
    test_path  = tmp_path / "test.csv"
    mock_params["preprocessing"]["train_data_path"] = str(train_path)
    mock_params["preprocessing"]["test_data_path"]  = str(test_path)

    with patch("src.data_preprocessing.read_params", return_value=mock_params):
        train_test_split_data(big_df)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    total = len(train_df) + len(test_df)
    assert total == len(big_df), "Rows were lost during split"
    # test set should be roughly 20%
    assert len(test_df) == pytest.approx(len(big_df) * 0.2, abs=1)