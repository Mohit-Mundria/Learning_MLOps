# tests/test_pipeline_integration.py
#
# ─────────────────────────────────────────────
# GATE 2 TESTS
#
# These tests don't mock much — they test the
# REAL pipeline working end-to-end.
# They run only on merge to main in GitHub Actions.
#
# They assume:
# - Real data exists (pulled via dvc pull)
# - Real params.yaml exists
# - reports/metrics.json was created by the pipeline
# ─────────────────────────────────────────────

import pytest
import os
import json
import pandas as pd


# ══════════════════════════════════════════
# SECTION 1: Check pipeline output files exist
# ══════════════════════════════════════════

def test_raw_data_file_exists():
    """
    After dvc repro, the raw data file must exist.
    If this fails: data_ingestion stage is broken.
    """
    assert os.path.exists("data/raw/raw_data.csv"), \
        "Raw data file missing — data_ingestion stage failed"


def test_train_data_file_exists():
    """
    After dvc repro, train split must exist.
    If this fails: data_preprocessing stage is broken.
    """
    assert os.path.exists("data/processed/train_data.csv"), \
        "Train data missing — data_preprocessing stage failed"


def test_test_data_file_exists():
    """Test split must also exist."""
    assert os.path.exists("data/processed/test_data.csv"), \
        "Test data missing — data_preprocessing stage failed"


def test_model_file_exists():
    """
    After dvc repro, the trained model file must exist.
    If this fails: model_training stage is broken.
    """
    assert os.path.exists("models/model.pkl"), \
        "Model file missing — model_training stage failed"


def test_metrics_file_exists():
    """
    metrics.json must be created by model_training.py.
    This is how Gate 2 reads the model's performance.
    """
    assert os.path.exists("reports/metrics.json"), \
        "metrics.json missing — was it created in model_training.py?"


# ══════════════════════════════════════════
# SECTION 2: Validate metrics content
# ══════════════════════════════════════════

@pytest.fixture
def metrics():
    """Load metrics.json once, reuse in multiple tests."""
    with open("reports/metrics.json") as f:
        return json.load(f)


def test_metrics_has_required_keys(metrics):
    """
    metrics.json must have both keys we defined.
    If a key is missing, model_training.py is incomplete.
    """
    assert "training_r2" in metrics, "training_r2 key missing from metrics.json"
    assert "testing_r2"  in metrics, "testing_r2 key missing from metrics.json"


def test_model_testing_score_above_threshold(metrics):
    """
    THE MOST IMPORTANT GATE 2 TEST.

    If the model's test R2 drops below 0.70,
    this test FAILS → GitHub Action turns RED →
    you get notified → bad model is never "approved".

    Change 0.70 to whatever makes sense for your data.
    """
    threshold = 65.0
    testing_r2 = metrics["testing_r2"]

    assert testing_r2 >= threshold, (
        f"Model quality too low! "
        f"Testing R2 = {testing_r2:.4f}, "
        f"required >= {threshold}"
    )


def test_model_is_not_overfitting(metrics):
    """
    If training score is much higher than testing score,
    the model is overfitting — memorizing training data
    instead of learning real patterns.

    We allow max 15% gap between train and test score.
    """
    gap = metrics["training_r2"] - metrics["testing_r2"]
    assert gap <= 0.15, (
        f"Model is overfitting! "
        f"Train R2 = {metrics['training_r2']:.4f}, "
        f"Test R2 = {metrics['testing_r2']:.4f}, "
        f"Gap = {gap:.4f} (max allowed: 0.15)"
    )


# ══════════════════════════════════════════
# SECTION 3: Validate processed data quality
# ══════════════════════════════════════════

def test_processed_train_data_has_no_nulls():
    """
    The training data fed to the model must have
    zero null values — nulls crash model training.
    """
    df = pd.read_csv("data/processed/train_data.csv")
    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"Train data has {nulls} null values"


def test_processed_data_has_target_column():
    """ROI_Score must exist in the processed data."""
    df = pd.read_csv("data/processed/train_data.csv")
    assert "ROI_Score" in df.columns, "Target column ROI_Score is missing"
