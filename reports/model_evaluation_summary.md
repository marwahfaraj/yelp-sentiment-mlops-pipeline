# Week 4 Model Evaluation and Deployment Summary

## Required Steps

- Simple benchmark model: majority-class benchmark.
- First real model: SageMaker SKLearn TF-IDF + Logistic Regression training job.
- Evaluation: accuracy, macro precision, macro recall, macro F1, and confusion matrix.
- Deployment: SageMaker Batch Transform using the reserved production split.

## Test Metrics

- Benchmark macro F1: 0.3333
- Trained model macro F1: 0.9408
- Trained model accuracy: 0.9408

## Artifacts

- Model artifact: `s3://yelp-sentiment-mlops-045456814877/models/week4-sklearn-sentiment/artifacts/sagemaker-scikit-learn-2026-05-30-05-10-35-293/output/model.tar.gz`
- Batch input: `s3://yelp-sentiment-mlops-045456814877/batch/week4-sentiment/input/production_text.txt`
- Batch output: `s3://yelp-sentiment-mlops-045456814877/batch/week4-sentiment/output/`
- Metrics CSV: `reports/benchmark_vs_model_metrics.csv`
- Confusion matrix: `reports/confusion_matrix.png`
