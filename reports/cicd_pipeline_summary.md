# Week 6 CI/CD Pipeline Summary

Pipeline: `yelp-sentiment-cicd-pipeline`
Model Package Group: `yelp-sentiment-models`
Quality gate: macro F1 >= 0.80 on the test split (ConditionStep + FailStep)

## Checkpoints

- Data/system integration: `YelpPreprocess` validates schema and splits before training.
- Model code: training and inference scripts run inside pipeline containers from version-controlled `src/`.
- Model performance: `YelpEvaluate` scores the test split; `YelpF1Gate` blocks deployment below threshold.
- Deployment: on pass, the model is registered and Batch Transform scores the production reviews.

## Execution 1 - Baseline (max_features=50000, ngram<=2, C=1.0)

- ARN: `arn:aws:sagemaker:us-east-1:965705611982:pipeline/yelp-sentiment-cicd-pipeline/execution/mew2kx8cndwk`
- Test macro F1: 0.9468
- Test accuracy: 0.9468

| Step | Status |
|---|---|
| YelpPreprocess | Succeeded |
| YelpTrain | Succeeded |
| YelpEvaluate | Succeeded |
| YelpF1Gate | Succeeded |
| YelpRegisterModel-RegisterModel | Succeeded |
| YelpCreateModel-CreateModel | Succeeded |
| YelpBatchTransform | Succeeded |

## Execution 2 - Improved (max_features=100000, ngram<=3, C=2.0)

- ARN: `arn:aws:sagemaker:us-east-1:965705611982:pipeline/yelp-sentiment-cicd-pipeline/execution/agmosts2poa9`
- Test macro F1: 0.9482
- Test accuracy: 0.9482

| Step | Status |
|---|---|
| YelpPreprocess | Succeeded |
| YelpTrain | Succeeded |
| YelpEvaluate | Succeeded |
| YelpF1Gate | Succeeded |
| YelpRegisterModel-RegisterModel | Succeeded |
| YelpCreateModel-CreateModel | Succeeded |
| YelpBatchTransform | Succeeded |

## Model Registry

- `arn:aws:sagemaker:us-east-1:965705611982:model-package/yelp-sentiment-models/2` (PendingManualApproval)
- `arn:aws:sagemaker:us-east-1:965705611982:model-package/yelp-sentiment-models/1` (PendingManualApproval)

## Artifacts

- Batch Transform output: `s3://yelp-sentiment-mlops-965705611982/batch/cicd-pipeline/output/`
- Model artifacts: `s3://yelp-sentiment-mlops-965705611982/models/cicd-pipeline`
