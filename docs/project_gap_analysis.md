# Project Gap Analysis Against Professor Example

The professor's example demonstrates a full MLOps lifecycle using AWS services. This project follows the same process pattern, but the current implementation is local-first with AWS integration documented as the next implementation layer.

## Already Covered

- Real data source using Yelp Open Dataset JSON reviews.
- Data preparation and sentiment label creation.
- Train, validation, test, and batch inference data splits.
- Baseline model training.
- Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix.
- Model quality gate based on F1-score threshold.
- Local monitoring for prediction distribution, missing text, and review length drift.
- Generated reports and reproducible run command.

## Still Needed For Full AWS Match

- Upload raw and processed data to Amazon S3.
- Convert processed data to Parquet for scalable querying.
- Add AWS Glue/Athena table definitions for processed Yelp review data.
- Add SageMaker Processing job for preprocessing and EDA.
- Add SageMaker Feature Store groups for training, validation, and batch inference.
- Add SageMaker Training or Hugging Face Estimator for model training.
- Add hyperparameter tuning for baseline or transformer models.
- Add SageMaker Model Registry for approved model artifacts.
- Add SageMaker Batch Transform or Endpoint deployment.
- Add SageMaker Model Monitor and CloudWatch alarms.
- Add a SageMaker Pipeline definition with a pass/fail model quality condition.

## Recommended Next Build Order

1. Strengthen local EDA reports and charts.
2. Add AWS S3 data layout and upload instructions.
3. Add SageMaker pipeline skeleton.
4. Add model registry and model approval documentation.
5. Add monitoring dashboard screenshots or generated monitoring reports.
6. Add final presentation diagrams and demo outputs.
