# tests/test_data_ingestion.py

import pytest
import pandas as pd
import os
from unittest.mock import patch

# ─────────────────────────────────────────────
# unittest.mock.patch explanation:
#
# Your data_load() function calls read_params()
# which tries to open a file at a Windows path.
# In CI (Linux), that path doesn't exist → error.
#
# patch("src.data_ingestion.read_params", return_value=mock_params)
# means: "wherever read_params is called inside
# data_ingestion.py, replace it with a function
# that just returns mock_params instead."
#
# The "with" block means the patch is only active
# during that block — cleaned up automatically.
# ─────────────────────────────────────────────

from src.data_ingestion import data_load


def test_data_load_returns_dataframe(tmp_path, mock_params):
    """
    tmp_path is a built-in pytest fixture.
    It gives you a temporary folder that is
    automatically deleted after the test.
    We use it to create a fake CSV file.
    """

    # Create a fake CSV file in the temp folder
    fake_csv = tmp_path / "data.csv"
    fake_csv.write_text("col1,col2,ROI_Score\n1,2,3\n4,5,6\n")

    # Tell mock_params to point to this temp file
    mock_params["preprocessing"]["data_path"]   = str(fake_csv)
    mock_params["preprocessing"]["raw_data_path"] = str(tmp_path / "raw.csv")

    with patch("src.data_ingestion.read_params", return_value=mock_params):
        result = data_load(str(fake_csv))

    # Assert: result must be a DataFrame
    assert isinstance(result, pd.DataFrame), "data_load should return a DataFrame"


def test_data_load_saves_csv(tmp_path, mock_params):
    """
    Test that data_load actually saves the file
    to the raw_data_path specified in params.
    """
    fake_csv = tmp_path / "data.csv"
    fake_csv.write_text("col1,col2,ROI_Score\n1,2,3\n4,5,6\n")

    raw_save_path = tmp_path / "raw.csv"
    mock_params["preprocessing"]["data_path"]     = str(fake_csv)
    mock_params["preprocessing"]["raw_data_path"] = str(raw_save_path)

    with patch("src.data_ingestion.read_params", return_value=mock_params):
        data_load(str(fake_csv))

    # Assert: the raw file must now exist on disk
    assert os.path.exists(raw_save_path), "data_load should save CSV to raw_data_path"


# def test_data_load_correct_shape(tmp_path, mock_params):
#     """
#     Test that the loaded DataFrame has exactly
#     the rows and columns from the CSV — nothing
#     is dropped or added during loading.
#     """
#     fake_csv = tmp_path / "data.csv"
#     fake_csv.write_text("col1,col2,ROI_Score\n1,2,3\n4,5,6\n")

#     mock_params["preprocessing"]["data_path"]     = str(fake_csv)
#     mock_params["preprocessing"]["raw_data_path"] = str(tmp_path / "raw.csv")

#     with patch("src.data_ingestion.read_params", return_value=mock_params):
#         result = data_load(str(fake_csv))

#     # 2 rows, 3 columns — matches the CSV above
#     assert result.shape == (2, 3), f"Expected shape (2,3), got {result.shape}"


def test_data_load_file_not_found(tmp_path, mock_params):
    """
    Test that the function raises an error when
    given a path that doesn't exist.

    pytest.raises() is how you test that a function
    SHOULD raise an exception — if it doesn't raise,
    the test fails.
    """
    mock_params["preprocessing"]["raw_data_path"] = str(tmp_path / "raw.csv")

    with patch("src.data_ingestion.read_params", return_value=mock_params):
        with pytest.raises(Exception):
            data_load("this/path/does/not/exist.csv")