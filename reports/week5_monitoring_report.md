# Week 5 Monitoring Report

## Required Steps Covered

- Model monitors: macro F1 quality gate, prediction distribution drift, benchmark comparison, bias checks, and explainability proxy drift.
- Data monitors: missing text rate, review length drift, class distribution drift.
- Infrastructure monitors: SageMaker training job status and matching Batch Transform job status.
- CloudWatch dashboard: custom metric dashboard for model, data, bias/explainability, and infrastructure monitors.
- SageMaker reports: model, data, infrastructure, bias/explainability, and failure-mode artifacts saved under `reports/`.

## Model Monitoring Summary

- Benchmark test macro F1: 0.3333
- Trained model test macro F1: 0.9468
- Quality gate status: **PASS**
- Batch prediction count: 20,000
- Prediction drift status: **PASS**

## Data Monitoring Summary

- Production missing text rate: 0.0000
- Missing text status: **PASS**
- Average word count relative drift: 0.0112
- Word count drift status: **PASS**
- Positive-share drift status: **PASS**

## Bias and Explainability Monitoring Summary

- Post-model bias status: **PASS**
- Post-model positive-share drift: 0.0052
- Max term-frequency drift: 0.0003
- Top drift terms: burger, service, great, going, ordered

## Infrastructure Monitoring Summary

- Training job: `sagemaker-scikit-learn-2026-06-11-02-04-34-891` (Completed)
- Batch Transform job: `sagemaker-scikit-learn-2026-06-11-02-08-15-502` (Completed)
- CloudWatch dashboard: `yelp-sentiment-week5-monitoring`

## Artifacts

- `reports/model_monitoring_summary.json`
- `reports/data_monitoring_metrics.csv`
- `reports/data_monitoring_summary.json`
- `reports/infrastructure_monitoring_summary.json`
- `reports/cloudwatch_dashboard_body.json`
- `reports/model_monitoring_dashboard.png`
- `reports/data_monitoring_dashboard.png`
- `reports/bias_monitoring_metrics.csv`
- `reports/explainability_drift_metrics.csv`
- `reports/bias_explainability_monitoring_summary.json`
- `reports/bias_explainability_monitoring_dashboard.png`
- `reports/monitoring_failure_modes.md`
