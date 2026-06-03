# Weekly Project Progress Tracker

This document maps the weekly Team Project Steps to the artifact(s) in this repository that satisfy them. Update after running the notebooks in SageMaker so the team and instructor can see the latest state.

## Module 3 Required Steps

| # | Required Step | Artifact | Status |
|---|---|---|---|
| 1 | Collect a raw data set and store it in an S3 Datalake | `notebooks/01_setup_S3_bucket.ipynb` creates `s3://yelp-sentiment-mlops-<account>/raw/` and uploads both `yelp-json.zip` and `reviews_raw.csv` | Completed |
| 2 | Set up Athena tables to enable cataloging and querying of your data | `athena_queries/01_Create_Athena_Database.ipynb` creates `yelp_db`. `athena_queries/02_Register_S3_With_Athena.ipynb` registers `reviews_raw`. `athena_queries/03_Convert_CSV_To_Parquet_With_Athena.ipynb` builds `reviews_parquet` for fast querying | Completed |
| 3 | Perform exploratory data analysis on your data in a SageMaker notebook | `notebooks/02_data_exploration_EDA.ipynb` produces star-distribution, class-balance, review-length, and top-words charts plus `reports/eda_summary.md` | Completed |
| 4 | Perform feature engineering on raw data and store it in a Feature Store | `notebooks/03_feature_engineering_feature_store.ipynb` creates a SageMaker Feature Group with `review_id` as record identifier and `event_time` as event time, then ingests engineered features | Completed |
| 5 | Split your feature data into training (~40%), test (~10%), validation (~10%) datasets | `notebooks/03_feature_engineering_feature_store.ipynb` assigns `split_type` via a stratified 40/10/10/40 split and writes per-split CSVs to `s3://<bucket>/processed/splits/` | Completed |
| 6 | Reserve some data for "production data" (~40%) | `split_type = 'production'` rows in the Feature Group and `processed/splits/production/production.csv` | Completed |

## Module 4 Required Steps

| # | Required Step | Artifact | Status |
|---|---|---|---|
| 1 | Set up a benchmark model in SageMaker | `notebooks/05_model_training_evaluation_deployment.ipynb` evaluates a majority-class benchmark against validation and test splits | Completed |
| 2 | Build, train, and debug your ML model in SageMaker | `src/train_sklearn.py` trains a TF-IDF + Logistic Regression classifier through a SageMaker SKLearn training job | Completed |
| 3 | Evaluate your model and compare it against your benchmark model | `notebooks/05_model_training_evaluation_deployment.ipynb` saves `reports/benchmark_vs_model_metrics.csv`, `reports/confusion_matrix.png`, and `reports/model_evaluation_summary.md` | Completed |
| 4 | Deploy your model to SageMaker | `notebooks/05_model_training_evaluation_deployment.ipynb` creates a SageMaker Batch Transform job using the reserved production split | Completed |

## Module 5 Required Steps

| # | Required Step | Artifact | Status |
|---|---|---|---|
| 1 | Implement model monitors on your ML system | `notebooks/06_model_data_infrastructure_monitoring.ipynb` checks macro F1 quality gate, benchmark comparison, batch prediction distribution drift, bias monitoring, and explainability proxy drift | Ready to run in SageMaker |
| 2 | Implement data monitors on your ML system | `notebooks/06_model_data_infrastructure_monitoring.ipynb` compares train baseline vs production split for missing text, review length drift, and class distribution drift | Ready to run in SageMaker |
| 3 | Implement infrastructure monitors on your ML system | `notebooks/06_model_data_infrastructure_monitoring.ipynb` checks SageMaker training job and latest Batch Transform job health, then publishes custom CloudWatch metrics | Ready to run in SageMaker |
| 4 | Create a monitoring dashboard for your ML endpoint/job on CloudWatch | `notebooks/06_model_data_infrastructure_monitoring.ipynb` creates the `yelp-sentiment-week5-monitoring` CloudWatch dashboard | Ready to run in SageMaker |
| 5 | Generate model and data reports on SageMaker | `reports/week5_monitoring_report.md`, model/data/infrastructure summaries, bias/explainability summaries, failure-mode notes, and dashboard plots | Ready to run in SageMaker |

## How to Run This Week's Notebooks

From SageMaker Studio (after cloning the repo and running `pip install -r requirements.txt`):

```text
notebooks/01_setup_S3_bucket.ipynb
athena_queries/01_Create_Athena_Database.ipynb
athena_queries/02_Register_S3_With_Athena.ipynb
athena_queries/03_Convert_CSV_To_Parquet_With_Athena.ipynb
notebooks/02_data_exploration_EDA.ipynb
notebooks/03_feature_engineering_feature_store.ipynb
notebooks/04_dataset_splits.ipynb
notebooks/05_model_training_evaluation_deployment.ipynb
notebooks/06_model_data_infrastructure_monitoring.ipynb
```

## Outputs To Capture For The Module 3 Tracker Update

After running the notebooks, capture and post in your tracker entry:

- The S3 bucket name (printed in notebook 01).
- The Athena database name and table list (notebooks 01–03 in `athena_queries/`).
- The split row counts and per-class counts table from `reports/dataset_splits_summary.md`.
- The feature group name (printed in notebook 03 and persisted with `%store feature_group_name`).
- Two or three EDA chart paths from `reports/`.

## Outputs To Capture For The Module 4 Tracker Update

After running `notebooks/05_model_training_evaluation_deployment.ipynb`, capture and post in your tracker entry:

- Benchmark model name and test metrics from `reports/benchmark_vs_model_metrics.csv`.
- Trained model name and test metrics from `reports/model_evaluation_summary.md`.
- SageMaker training job name and model artifact S3 path.
- SageMaker Batch Transform output S3 path.
- Confusion matrix plot path: `reports/confusion_matrix.png`.

Suggested tracker wording:

```text
Built a simple benchmark classifier, trained a first SageMaker TF-IDF + Logistic Regression sentiment model, compared model performance using accuracy/precision/recall/F1 and a confusion matrix, and deployed the trained model with SageMaker Batch Transform on the reserved production split.
```

## Outputs To Capture For The Module 5 Tracker Update

After running `notebooks/06_model_data_infrastructure_monitoring.ipynb`, capture and post in your tracker entry:

- CloudWatch dashboard name: `yelp-sentiment-week5-monitoring`.
- Model monitor status from `reports/model_monitoring_summary.json`.
- Data monitor status from `reports/data_monitoring_summary.json`.
- Infrastructure monitor status from `reports/infrastructure_monitoring_summary.json`.
- Bias/explainability monitor status from `reports/bias_explainability_monitoring_summary.json`.
- Failure modes and monitoring best practices from `reports/monitoring_failure_modes.md`.
- Week 5 monitoring report: `reports/week5_monitoring_report.md`.
- Dashboard plots: `reports/model_monitoring_dashboard.png`, `reports/data_monitoring_dashboard.png`, and `reports/bias_explainability_monitoring_dashboard.png`.

Suggested tracker wording:

```text
Implemented Week 5 monitoring for the Yelp sentiment batch ML system. Added model quality monitors, data drift/data quality monitors, bias and explainability proxy monitors, infrastructure job-health monitors, CloudWatch custom metrics/alarms, and a CloudWatch dashboard. Generated SageMaker monitoring reports and documented common failure modes/best practices.
```

## Still To Do In Later Modules

- SageMaker Processing job for repeatable preprocessing.
- Hugging Face transformer comparison or fine-tuning.
- SageMaker Model Registry with quality gate based on macro F1.
- SageMaker Pipeline (CI/CD DAG) with ConditionStep, success/failure runs.
- Optional real-time SageMaker Endpoint deployment.
- Production hardening for SageMaker Model Monitor schedules and CloudWatch alarms.
- ML Design Document final pass and 10–15 minute video demonstration.
