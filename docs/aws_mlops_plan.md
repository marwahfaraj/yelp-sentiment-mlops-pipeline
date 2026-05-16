# AWS MLOps Implementation Plan

This document maps the local Yelp sentiment pipeline to an AWS/SageMaker workflow similar to the final project example.

## AWS Services

- **Amazon S3**: Store raw Yelp archives, processed train/validation/test datasets, batch inference inputs, reports, and model artifacts.
- **AWS Glue or Athena**: Catalog and query processed review data when the dataset is converted to Parquet.
- **Amazon SageMaker Processing**: Run data preparation and EDA jobs on the Yelp JSON review data.
- **Amazon SageMaker Feature Store**: Store processed text metadata and labels for reproducible training, validation, and batch inference datasets.
- **Amazon SageMaker Training**: Train the baseline model or fine-tune a transformer model.
- **Amazon SageMaker Experiments**: Track model runs, parameters, metrics, and artifacts.
- **Amazon SageMaker Model Registry**: Register approved models that pass the F1-score threshold.
- **Amazon SageMaker Batch Transform**: Run scheduled batch sentiment predictions on new review data.
- **Amazon SageMaker Endpoint**: Optional real-time sentiment API for review text.
- **SageMaker Model Monitor**: Monitor model quality and data quality using captured predictions and ground truth when available.
- **Amazon CloudWatch**: Track endpoint latency, CPU/memory utilization, errors, and model/data quality alarms.

## Target Data Layout In S3

```text
s3://<bucket>/yelp-sentiment/raw/
s3://<bucket>/yelp-sentiment/processed/train/
s3://<bucket>/yelp-sentiment/processed/validation/
s3://<bucket>/yelp-sentiment/processed/test/
s3://<bucket>/yelp-sentiment/batch/input/
s3://<bucket>/yelp-sentiment/batch/output/
s3://<bucket>/yelp-sentiment/models/
s3://<bucket>/yelp-sentiment/reports/
s3://<bucket>/yelp-sentiment/monitoring/
```

## Pipeline Checkpoints

1. **Data validation**: Confirm required fields are available from the Yelp review JSON data.
2. **Preprocessing**: Create sentiment labels, clean review text, remove neutral labels, and create train/validation/test/batch splits.
3. **EDA**: Generate data quality, class balance, rating distribution, and review length reports.
4. **Training**: Train baseline and transformer-based NLP models.
5. **Evaluation**: Calculate accuracy, precision, recall, F1-score, and confusion matrix.
6. **Model gate**: Approve the model only when macro F1-score meets the configured threshold.
7. **Registration**: Register approved model artifacts in SageMaker Model Registry.
8. **Deployment**: Deploy approved models to Batch Transform or a SageMaker Endpoint.
9. **Monitoring**: Track data quality, prediction distribution, review length drift, model quality, latency, and resource usage.

## Model Quality Gate

The local project uses `src/model_gate.py` to simulate the same idea as a SageMaker Pipeline condition step.

Default threshold:

```text
macro F1-score >= 0.80
```

If the model passes, the next step is model registration. If it fails, the pipeline should stop and require review of data quality, training parameters, or model selection.

## Monitoring Metrics

Model monitoring:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Prediction distribution

Data monitoring:

- Missing review text count
- Average review length
- Sentiment class balance
- Star rating distribution
- Input text length drift

Infrastructure monitoring:

- Endpoint latency
- Invocation errors
- CPU utilization
- Memory utilization
- Batch transform job failures

## First AWS Implementation Target

The first AWS version should focus on batch inference because review analysis does not require immediate real-time scoring. Real-time endpoint deployment can be added after the batch workflow is reliable.
