# Learning_MLOps

# 🚀 Learning MLOps — End-to-End Machine Learning Pipeline

> *From raw data to a production-grade, automatically retrained model — with experiment tracking, pipeline orchestration, automated testing, and CI/CD.*

![CI - Unit Tests](https://github.com/Mohit-Mundria/Learning_MLOps/actions/workflows/ci.yml/badge.svg)
![CD - ML Pipeline](https://github.com/Mohit-Mundria/Learning_MLOps/actions/workflows/cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Pipeline-purple)
![DagsHub](https://img.shields.io/badge/DagsHub-Remote-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What Is This Project?

This is my **first end-to-end MLOps project** — built not just to train a model, but to learn how real-world ML systems are built, versioned, tested, and deployed automatically.

The project predicts **ROI Score** for Agentic AI projects of Industry using a regression pipeline. But the *real* goal was never just the model — it was building the **infrastructure around the model**: reproducible pipelines, experiment tracking, automated testing, and a full CI/CD system using GitHub Actions.

> **The model is the output. The pipeline is the product.**

---

## 🗺️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SOURCE CODE                             │
│                                                                 │
│   data_ingestion.py  →  data_preprocessing.py  →  model_training.py  │
│         │                      │                        │       │
│    Load CSV &            Drop columns,            Optuna HPO,   │
│    save raw data         fill nulls,              XGB / RF,     │
│                          one-hot encode,          MLflow log    │
│                          train/test split                       │
└─────────────────────────────────────────────────────────────────┘
           │                      │                        │
           ▼                      ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DVC PIPELINE                             │
│                                                                 │
│    dvc.yaml defines stages → dvc repro runs only what changed  │
│    dvc push/pull syncs data & models with DagsHub remote       │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING                          │
│                                                                 │
│    MLflow + DagsHub → every run logs params, metrics, model    │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CI / CD PIPELINE                            │
│                                                                 │
│  Gate 1 (CI): pytest unit tests → runs on every git push       │
│  Gate 2 (CD): dvc repro + integration tests → runs on main     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| **Language** | Python 3.13 | Core development |
| **ML Libraries** | scikit-learn, XGBoost | Model training |
| **HPO** | Optuna | Automated hyperparameter tuning |
| **Pipeline** | DVC | Reproducible ML pipeline stages |
| **Experiment Tracking** | MLflow | Log params, metrics, models |
| **Remote Storage** | DagsHub | Data, models, and MLflow UI hosting |
| **Config Management** | params.yaml | Single source of truth for all parameters |
| **Testing** | pytest | Unit tests (Gate 1) + Integration tests (Gate 2) |
| **CI/CD** | GitHub Actions | Automated test and pipeline execution |
| **Version Control** | Git + GitHub | Code versioning |

---

## 📁 Project Structure

```
Learning_MLOps/
│
├── src/                          # Core pipeline source code
│   ├── __init__.py
│   ├── data_ingestion.py         # Stage 1: Load & save raw data
│   ├── data_preprocessing.py     # Stage 2: Clean, encode, split
│   └── model_training.py         # Stage 3: Optuna HPO + MLflow logging
│
├── utility_code/
│   └── utility_func.py           # read_params() and shared utilities
│
├── tests/                        # All test files
│   ├── conftest.py               # Shared fixtures (mock data, mock params)
│   ├── test_data_ingestion.py    # Gate 1 tests
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   └── test_pipeline_integration.py  # Gate 2 tests
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # Gate 1: Unit tests on every push
│       └── cd.yml                # Gate 2: Full pipeline on merge to main
│
├── data/
│   ├── raw/                      # Saved by data_ingestion.py (tracked by DVC)
│   └── processed/                # Train/test splits (tracked by DVC)
│
├── models/                       # Trained model artifacts (tracked by DVC)
├── reports/
│   └── metrics.json              # Model scores, read by Gate 2 quality check
│
├── dvc.yaml                      # Pipeline stage definitions
├── dvc.lock                      # Exact snapshot of last successful pipeline run
├── params.yaml                   # All hyperparameters and file paths
└── requirements.txt
```

---

## ⚙️ Pipeline Stages (DVC)

The entire pipeline is defined in `dvc.yaml` and run with a single command:

```bash
dvc repro
```

DVC is smart — it only re-runs stages whose **inputs have changed**. If you only change `model_training.py`, it skips ingestion and preprocessing entirely.

| Stage | Script | Input | Output |
|---|---|---|---|
| `data_ingestion` | `data_ingestion.py` | Original CSV path from params | `data/raw/raw_data.csv` |
| `data_preprocessing` | `data_preprocessing.py` | Raw data | `train_data.csv`, `test_data.csv` |
| `model_training` | `model_training.py` | Processed train/test data | `models/model.pkl`, `reports/metrics.json` |

---

## 🔬 Model Training — How It Works

Instead of manually trying hyperparameters, this project uses **Optuna** to automatically find the best model and hyperparameters:

```
Optuna runs 100 trials
    │
    ├── Trial: RandomForestRegressor with params (n_estimators, max_depth, ...)
    ├── Trial: XGBRegressor with params (learning_rate, gamma, ...)
    ├── ... (100 combinations tested)
    │
    └── Best trial found → train final model → log to MLflow → save model.pkl
```

Both `RandomForestRegressor` and `XGBRegressor` compete in the same Optuna study. The best model *and* the best hyperparameters for that model are selected automatically.

All results — parameters, training score, testing score, model artifact — are logged to **MLflow on DagsHub** for every run.

---

## 🧪 Testing Strategy — Two Gates

Testing an ML pipeline is different from testing a web app. There's no API endpoint to hit. So the testing strategy is split into two gates:

### Gate 1 — Unit Tests (CI)
Runs on **every git push** to any branch. Fast (~30 seconds). Uses **mock data** — no real files needed.

Tests cover:
- `data_ingestion`: Does it return a DataFrame? Save the file? Handle missing files?
- `data_preprocessing`: Does `drop_col` work? Does `fill_nan` eliminate all nulls? Does one-hot encoding produce correct columns and remove originals?
- `model_training`: Do data loaders return correct shapes? Does `objective()` return a valid float R² score?

### Gate 2 — Integration Tests (CD)
Runs **only on merge to main**. Uses real data (pulled via `dvc pull`). Tests the actual pipeline output.

Tests cover:
- Do all output files exist after `dvc repro`?
- Is `metrics.json` valid and complete?
- Is the model's test R² above the quality threshold?
- Is the model overfitting (train vs test gap > 15%)?
- Was any data lost during the train/test split?

> **If Gate 2 fails, the pipeline is red and you're notified immediately. A bad model never silently passes.**

---

## 🔄 CI/CD with GitHub Actions

Two separate workflow files handle the two gates:

```
Feature branch push
       │
       ▼
  ci.yml → pytest Gate 1 tests (fast, mocked)
       │
       ▼ (PR opened → merge to main)
  cd.yml → dvc pull → dvc repro → pytest Gate 2 → dvc push
       │
       ▼
  New experiment logged in DagsHub MLflow UI 🎉
```

**Why two separate files?**
- `ci.yml` runs on every branch — it's fast and needs no secrets
- `cd.yml` runs only on `main` — it uses DagsHub credentials, runs real training, takes minutes

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mohit-Mundria/Learning_MLOps.git
cd Learning_MLOps
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configure DagsHub remote (for DVC data)
```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

### 4. Pull the data
```bash
dvc pull
```

### 5. Run the full pipeline
```bash
dvc repro
```

### 6. Run tests
```bash
# Gate 1 — unit tests only
pytest tests/test_data_ingestion.py tests/test_data_preprocessing.py tests/test_model_training.py -v

# Gate 2 — integration tests (requires pipeline to have run first)
pytest tests/test_pipeline_integration.py -v

# All tests with coverage
pytest tests/ --cov=src --cov-report=term-missing -v
```

---

## 📊 Experiment Tracking

Every pipeline run automatically logs to **MLflow on DagsHub**:

- ✅ Model name (RandomForest or XGBoost — whichever Optuna picked)
- ✅ All hyperparameters
- ✅ Training R² score
- ✅ Testing R² score (cross-validated)
- ✅ Model artifact (`.pkl` file)

View all experiments at:
```
https://dagshub.com/mundriamohit100/Learning_MLOps/experiments
```

---

## ⚙️ Configuration — `params.yaml`

All paths, hyperparameter ranges, and settings live in one place: params.yaml

Change a value here → DVC detects the change → only affected stages re-run.

---

## 🧱 Problems I Faced & How I Solved Them

These are real challenges encountered while building this project — documented here because every MLOps learner will face them.

### 1. 🔴 Hardcoded Windows Paths Breaking CI
**Problem:** All source files had hardcoded paths like `D:\End to end project\Learning_MLOps\params.yaml`. GitHub Actions runs on Linux — these paths don't exist there, causing every CI run to immediately crash.

**Solution:** Refactored `read_params()` to accept a path parameter with a sensible default. Used `os.path` and relative paths everywhere. In tests, used `unittest.mock.patch` to replace `read_params` entirely with mock data.

**Lesson:** Never hardcode absolute paths in ML code. Always use relative paths or config-driven paths.

---

### 2. 🔴 "What Do I Even Test?" — Testing an ML Pipeline
**Problem:** My project doesn't serve predictions — it produces a trained model. I couldn't figure out what to test because there's no API, no endpoint, no user-facing output.

**Solution:** Realized that ML pipelines have plenty of testable units — just not the kind you're used to in web development. The strategy became: **test the functions, not the model**. Unit test each transformation function with mock data (Gate 1), then test the pipeline *outputs* after a real run (Gate 2 — do files exist? Is the R² above threshold? Was data lost?).

**Lesson:** In MLOps, you test *data quality*, *function correctness*, and *model quality thresholds* — not prediction endpoints.

---

### 3. 🔴 `params.yaml` Dependency in Every Function
**Problem:** Every single function called `read_params()` with the hardcoded Windows path. This made unit testing impossible — you can't test `one_hot_encoding()` in isolation if it immediately tries to open a file that doesn't exist in CI.

**Solution:** Used `unittest.mock.patch` to intercept the `read_params` call and return a `MOCK_PARAMS` dictionary instead. This lets each function be tested completely in isolation.

**Lesson:** Functions that depend on external resources (files, databases, APIs) must be mockable. This is why dependency injection and mocking are core software engineering skills for ML engineers.

---

### 4. 🔴 Gate 2 Has No Quality Enforcement By Default
**Problem:** `dvc repro` running successfully just means the pipeline didn't *crash* — it doesn't mean the model is *good*. A subtle preprocessing bug could silently produce a terrible model with R² = 0.2, and the CI would still show green.

**Solution:** Added `reports/metrics.json` output from `model_training.py`, then added a Gate 2 integration test that reads this file and explicitly fails if `testing_r2 < 0.70`. Also added an overfitting check (train/test gap > 15% fails the pipeline).

**Lesson:** A passing pipeline is not the same as a good model. Always add a **model quality gate** to your CD pipeline.

---

### 5. 🔴 Understanding the Difference Between CI and CD in MLOps
**Problem:** In web development, CI/CD is well-defined. In MLOps, it's less obvious — what does "deployment" even mean for a model that isn't served?

**Solution:** Redefined CD for this project as: *"on merge to main, reproduce the full pipeline with real data and enforce model quality."* The "deployment" is the new `model.pkl` and the new MLflow experiment. This is the MLOps equivalent of deploying a new version.

**Lesson:** CI/CD in MLOps means continuous training + continuous quality enforcement, not necessarily continuous serving.

---

## 📚 What I Learned From This Project

This project was deliberately built to learn — not just to ship a model. Here's what genuinely changed the way I think:

| Area | Key Learning |
|---|---|
| **DVC** | ML pipelines need version control too — not just code. `dvc repro` and `dvc.lock` make experiments 100% reproducible. |
| **MLflow** | Without experiment tracking, you're flying blind. Logging every run to DagsHub makes it trivial to compare 100 Optuna trials. |
| **Optuna** | Manual hyperparameter tuning is a waste of time. Optuna can search both model selection and hyperparameters simultaneously. |
| **pytest + mocking** | Testing ML code is different from testing web code, but equally important. `unittest.mock.patch` is essential for testing functions with external dependencies. |
| **GitHub Actions** | CI/CD is not magic — it's just your commands running on a Linux machine triggered by git events. Two files, clear triggers, separate concerns. |
| **Code Quality** | Hardcoded paths, no tests, and tightly coupled functions look fine locally but break immediately in any other environment. |
| **MLOps Mindset** | The model is a byproduct. The real product is a **reliable, reproducible, self-validating pipeline** that produces the model. |

---

## 👤 About

**Mohit Mundria** — Fresher AI/ML Engineer learning to build production-grade ML systems.

This project represents my first serious attempt at MLOps — integrating tools that real ML teams use in production. Every problem I hit, every bug I fixed, and every concept I finally understood is documented in this repo.

[![DagsHub](https://img.shields.io/badge/DagsHub-View%20Experiments-blue)](https://dagshub.com/mundriamohit100/Learning_MLOps)
[![GitHub](https://img.shields.io/badge/GitHub-Mohit--Mundria-black?logo=github)](https://github.com/Mohit-Mundria)


<div align="center">
  <sub>Built with 🔥 and a lot of debugging by Mohit Mundria</sub>
</div>