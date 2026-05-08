# tests/conftest.py

import pytest
import pandas as pd

# ─────────────────────────────────────────────
# What is a fixture?
# A fixture is a reusable piece of setup code.
# Instead of creating a sample DataFrame in every
# single test file, you create it ONCE here and
# pytest automatically passes it to any test that
# needs it just by using its name as a parameter.
# ─────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """
    This is your raw dataset BEFORE any preprocessing.
    We simulate what your CSV would look like after data_ingestion.
    Mimics the kind of data your ROI project likely has.
    """
    return pd.DataFrame({
        "Product_ID":   ["AGP-0001", "AGP-0002", "AGP-0003"],
        "Industry":         ["Finance", "Retail", "Technology"],
        "Agent_Type":      ["Autonomous", "Multi-agent Swarm", "Autonomous"],
        "Primary_Tool":    ["AutoGPT 3.0", "LangGraph", "CrewAI"],
        "Status":          ["Production", "Pilot", None],
        "Failure_Reason": ["Loop Error", "Supply chain", None],
        "Human_in_the_Loop_Ratio": [0.5, 0.8, 0.3],
        "Tokens_per_Task": [100, 200, 150],
        "ROI_Score":      [1.2, 3.4, 5.6]
    })


@pytest.fixture
def sample_processed_df():
    """
    This is your dataset AFTER preprocessing —
    no nulls, no original categorical columns,
    one-hot encoded, ready for model training.
    """
    return pd.DataFrame({
        "Industry":         ["Finance", "Retail", "Technology"],
        "Agent_Type":      ["Autonomous", "Multi-agent Swarm", "Autonomous"],
        "Primary_Tool":    ["AutoGPT 3.0", "LangGraph", "CrewAI"],
        "Status":          ["Production", "Pilot", None],
        "Failure_Reason": ["Loop Error", "Supply chain", None],
        "Human_in_the_Loop_Ratio": [0.5, 0.8, 0.3],
        "Tokens_per_Task": [100, 200, 150],
        "ROI_Score":      [1.2, 3.4, 5.6]
    })


@pytest.fixture
def mock_params():
    """
    Your functions all call read_params() which reads
    from a hardcoded Windows path — that path does NOT
    exist in GitHub Actions (Linux environment).

    So we create a fake version of what params.yaml
    would return, and we'll inject this wherever
    read_params() is called in your actual code.
    """
    return {
        "preprocessing":{
            "missing_value_strategy": "mean",
            "categorical_encoding": "one_hot",
            "scaling_method": "standard",
            "data_path": "Download/agentic_reality.csv",
            "raw_data_path": "data_save/raw/raw_dataset.csv",

            "drop_columns" : ["Project_ID"],
            "one_hot_encode_columns": ["Industry", "Failure_Reason", "Agent_Type", "Primary_Tool", "Status"],
            "train_data_path": "data_save/processed/train.csv",
            "test_data_path": "data_save/processed/test.csv"

        },
        "model_training":{
            "test_size": 0.2,
            "random_state": 42,
            "model_path": "model_registry/model.pkl"
        }
    }