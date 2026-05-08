# What to Test in an ML Pipeline Project
Your pipeline has several distinct, testable units even though it doesn't serve predictions:
1. Data Ingestion Tests

Does data_load() return a DataFrame?
Does it actually save the CSV to the raw data path?
Does it handle a missing file gracefully?

2. Preprocessing Tests

Does drop_col() actually remove the specified columns?
Does fill_nan() leave no nulls behind?
Does one_hot_encoding() produce the expected new columns?
Does train_test_split_data() create files at the right paths?

3. Model Training Tests

Does load_training_data() return correctly shaped X and y?
Does objective() return a float (a valid R² score)?
Does mlflow_tracking() save a model file at the expected path?

4. Schema / Data Quality Tests

After preprocessing, do you have the expected number of columns?
Are there no unexpected nulls in the processed data?
Is the target column ROI_Score still present?



Code Push
    │
    ▼
┌─────────────────────────────┐
│  Gate 1: Unit Tests (CI)    │  ← pytest, runs in 30 seconds
│  - Do functions work?       │
│  - Is data schema correct?  │
│  - Does mock pipeline run?  │
└─────────────────────────────┘
    │ passes
    ▼
┌─────────────────────────────┐
│  Gate 2: DVC Pipeline (CD)  │  ← only on merge to main
│  - dvc repro                │
│  - Real training on real data│
│  - MLflow logs to DagsHub   │
└─────────────────────────────┘





# What is Gate 2?
Gate 1 (CI) only tests your code logic using mock data — it never actually runs your real ML pipeline. Gate 2 (CD) actually executes your DVC pipeline with real data, trains the model, and logs results to MLflow/DagsHub.
Think of it this way:
Gate 1 answers: "Is the code correct?"
Gate 2 answers: "Does the full pipeline actually work end-to-end?"

# Why Use Gate 2?
Your project has a unique challenge — the "output" isn't a web app going live, it's a trained model. So your CD step is:

"When code is merged to main, re-run the entire ML pipeline and produce a new model"

Without Gate 2, you could have:

Code that passes all unit tests ✅
But a DVC pipeline that is broken or produces a terrible model ❌

Gate 2 catches things like:

A preprocessing change that silently destroys model performance
A broken dvc.yaml stage
Data drift (if you pull fresh data in the pipeline)
Dependency version conflicts in the real environment





# Do You Need to Learn Pytest as an AI/ML Engineer?
Short answer: Yes, but not deeply. You need the basics.
Here's the honest breakdown:
As a Data Scientist — pytest is nice to have. Most DS roles focus on experimentation, notebooks, and model building. But even here, knowing basic pytest saves you from breaking things silently.
As an MLOps / ML Engineer — pytest is must have. You're building pipelines, not just notebooks. When your pipeline breaks at 2am in production, tests are what tell you exactly what broke and why.
Your project right now is MLOps-focused (DVC, MLflow, DagsHub, CI/CD) — so yes, you need it.
What you actually need to know (not the whole library):

assert statements
pytest.fixture — shared test data
pytest.raises — testing errors
unittest.mock.patch — mocking external dependencies like your params.yaml







# unittest.mock.patch explanation:
Your data_load() function calls read_params()
which tries to open a file at a Windows path.
In CI (Linux), that path doesn't exist → error.

patch("src.data_ingestion.read_params", return_value=mock_params)
means: "wherever read_params is called inside
data_ingestion.py, replace it with a function
that just returns mock_params instead."

The "with" block means the patch is only active
during that block — cleaned up automatically.

# tests/test_pipeline_integration.py
GATE 2 TESTS
These tests don't mock much — they test the
REAL pipeline working end-to-end.
They run only on merge to main in GitHub Actions.
They assume:
- Real data exists (pulled via dvc pull)
- Real params.yaml exists
- reports/metrics.json was created by the pipeline



# Quick Pytest Concept Summary
Concept                                     What it does                                                                                Where used
def test_xxx()                              Any function starting with test_ is auto-discovered by pytest                               Every file
assert                                      If condition is False, test fails with a clear message                                      Every test
@pytest.fixture                             Reusable setup — passed as parameter automatically                                          conftest.py
tmp_path                                    Built-in fixture — gives a temp folder, auto-deleted                                        Ingestion tests
patch(...)                                  Replaces a function with a fake during the test                                             Anywhere read_params is called
MagicMock()                                 Creates a fake object you can control completely                                            test_model_training.py
pytest.approx()                             Safe float comparison                                                                       Anywhere floats are compared
pytest.raises()                             Asserts that an exception IS raised                                                         File-not-found test



 #                        ci.yml                                  cd.yml
Triggers on             Every push, every branch                Only merge to main
Runs                    pytest (seconds)                        dvc repro (minutes)
Purpose                 Catch code bugs fast                    Run full ML pipeline
Needs real data?        No (uses mocks)                         Yes (dvc pull)
Needs DagsHub token?    No                                      Yes






# Important GitHub Actions Concepts 


# 1. Triggers (on:)
Controls when your workflow runs. Most important ones:
on:
  push:                        # on every git push
    branches: [main, dev]

  pull_request:                # when PR is opened/updated
    branches: [main]

  workflow_dispatch:           # adds a manual "Run" button in GitHub UI
                               # very useful for testing your workflow

  schedule:                    # run on a timer (like cron jobs)
    - cron: "0 2 * * *"        # runs every day at 2am UTC
                               # useful for daily model retraining



# 2. ${{ secrets.NAME }} vs ${{ env.NAME }}
secrets — stored encrypted in GitHub, never visible in logs
${{ secrets.DAGSHUB_TOKEN }}

# env — regular environment variables set in the workflow
${{ env.MY_VARIABLE }}

Set env variables for ALL steps below using $GITHUB_ENV
- run: echo "MODEL_VERSION=1.2" >> $GITHUB_ENV



# 3. uses: vs run:
uses: runs a pre-built action from GitHub Marketplace
- uses: actions/checkout@v3        # downloads your code
- uses: actions/setup-python@v4    # installs python

run: runs your own shell commands directly
- run: pip install -r requirements.txt
- run: pytest tests/ -v


# 4. Job Dependencies (needs:)
When you have multiple jobs and Job B should only run if Job A passes:
jobs:
  unit-tests:          # Job A — runs first
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -v

  run-pipeline:        # Job B — only runs if unit-tests passed
    needs: unit-tests  # ← this is the key line
    runs-on: ubuntu-latest
    steps:
      - run: dvc repro



# 5. Caching Dependencies
Without this, pip install runs fresh every single time — slow. With caching, dependencies are reused if requirements.txt hasn't changed:
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    # key changes only when requirements.txt changes
    # otherwise the cache is reused → much faster CI


# 6. Artifacts
Files you want to save from a workflow run and download later:
Save a file
- uses: actions/upload-artifact@v3
  with:
    name: my-model-metrics
    path: reports/metrics.json

These appear in GitHub Actions UI under each run
You can download them directly from the browser



# You write code on feature branch
        │
        ▼
git push → ci.yml triggers
           pytest runs (Gate 1)
           pass ✅ or fail ❌
        │
        ▼ (if pass)
Open Pull Request → ci.yml runs again
Code review by you (or team)
        │
        ▼
Merge to main → cd.yml triggers
                dvc pull
                dvc repro  (Gate 2)
                pytest integration tests
                dvc push
                metrics saved to DagsHub
        │
        ▼
New experiment visible in DagsHub MLflow UI 🎉