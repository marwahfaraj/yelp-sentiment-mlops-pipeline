# AWS Implementation

This folder contains planning material and skeleton code for mapping the local Yelp sentiment workflow to AWS/SageMaker.

The files do not include credentials, account IDs, bucket names, or deployment-specific secrets.

Recommended AWS build order:

1. Upload raw Yelp data to S3.
2. Run preprocessing as a SageMaker Processing job.
3. Store processed train, validation, test, and batch datasets in S3.
4. Optionally register processed features in SageMaker Feature Store.
5. Train the model with SageMaker Training or Hugging Face Estimator.
6. Evaluate metrics and apply the F1-score quality gate.
7. Register approved models in SageMaker Model Registry.
8. Deploy to SageMaker Batch Transform first.
9. Add SageMaker Endpoint only if real-time predictions are required.
10. Add SageMaker Model Monitor and CloudWatch alarms.
