# Yelp Review Sentiment MLOps Pipeline

![Yelp Sentiment Intelligence Project Overview](assets/yelp_sentiment_project_overview.png)

## Overview

This project builds an MLOps pipeline for analyzing Yelp customer reviews with natural language processing. The system classifies reviews as positive or negative, compares a baseline sentiment model with a transformer-based NLP model, evaluates model performance, applies a model quality gate, and monitors changes in review data and prediction behavior over time.

The project is designed for the AAI-540 final project and follows the same lifecycle pattern as the class example: data engineering, exploratory data analysis, feature preparation, model training, model evaluation, deployment planning, CI/CD checkpoints, and monitoring.

## Problem Statement

Businesses receive large volumes of customer reviews, but manually reading and interpreting every review is difficult to scale. A sentiment classification system can help summarize customer satisfaction, identify negative experiences earlier, and monitor how customer feedback changes over time.

The model objective is to predict customer sentiment from review text. This is a supervised NLP classification problem because Yelp star ratings can be converted into sentiment labels:

- 4- and 5-star reviews are labeled `positive`
- 1- and 2-star reviews are labeled `negative`
- 3-star reviews are excluded from the first version because they are more neutral or ambiguous

## Data Source

The main data source is the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/), which is intended for educational use and includes real-world Yelp data such as reviews, businesses, photos, check-ins, and business attributes. The review data provides the text and star ratings needed for sentiment classification.

The downloaded local files are:

```text
data/Yelp-JSON.zip
data/Yelp-Photos.zip
```

The sentiment pipeline uses `Yelp-JSON.zip`, which contains the Yelp review JSON data. `Yelp-Photos.zip` is not used in this project because the first version is focused on NLP text classification. The full Yelp files are large, so they are ignored by Git and should not be committed to GitHub.

Expected review fields:

- `review_text`: customer review text
- `stars`: Yelp star rating used to create the sentiment label

## Machine Learning Approach

The project will compare a simple baseline model against a more advanced NLP model:

- Baseline model: bag-of-words or TF-IDF style text features with a simple classifier
- Advanced model: Hugging Face transformer model such as DistilBERT or BERT
- Task type: supervised binary text classification
- Target labels: `positive` and `negative`

Evaluation metrics include accuracy, precision, recall, F1-score, and a confusion matrix. F1-score is especially important because the model should balance correctly identifying both positive and negative reviews.

## Exploratory Data Analysis

EDA is included because it helps determine whether the Yelp data is clean, balanced, and appropriate for modeling. Planned EDA includes:

- Review count and missing text checks
- Star rating distribution
- Positive vs negative class balance
- Average review length
- Common words and phrases by sentiment class
- Examples of ambiguous or difficult reviews

## MLOps Workflow

The local workflow mirrors the same process used in a cloud MLOps system:

1. Ingest Yelp review data
2. Clean and prepare review text
3. Create sentiment labels from star ratings
4. Split data into train, validation, test, and batch inference datasets
5. Train a baseline sentiment classifier
6. Evaluate or fine-tune a Hugging Face transformer model
7. Compare model performance
8. Apply a model quality gate using macro F1-score
9. Save model artifacts and reports
10. Run batch or API-based predictions
11. Monitor data quality, prediction distribution, review length drift, and model performance

## AWS MLOps Target Architecture

![Yelp Review Sentiment MLOps Architecture](assets/yelp_sentiment_mlops_architecture.png)

The local project is designed so it can be mapped to AWS services:

- Amazon S3 for raw Yelp data, processed datasets, batch inputs, batch outputs, reports, and model artifacts
- AWS Glue or Athena for querying processed Yelp review data
- SageMaker Processing for preprocessing and EDA jobs
- SageMaker Feature Store for training, validation, and batch inference datasets
- SageMaker Training or Hugging Face Estimator for model training
- SageMaker Experiments for tracking model runs and metrics
- SageMaker Model Registry for approved model versions
- SageMaker Batch Transform for scheduled review sentiment scoring
- SageMaker Endpoint for optional real-time sentiment prediction
- SageMaker Model Monitor and CloudWatch for data, model, and infrastructure monitoring

See `docs/aws_mlops_plan.md` for the AWS implementation plan.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── AAI_540_ML_Design_Document.md
├── assets/
│   ├── yelp_sentiment_project_overview.png
│   └── yelp_sentiment_mlops_architecture.png
├── data/
│   ├── README.md
│   ├── sample_yelp_reviews.csv
│   ├── Yelp-JSON.zip      # local only, ignored by Git
│   └── Yelp-Photos.zip    # local only, ignored by Git
├── notebooks/
│   ├── 01_setup_S3_bucket.ipynb
│   ├── 02_data_exploration_EDA.ipynb
│   ├── 03_feature_engineering_feature_store.ipynb
│   └── 04_dataset_splits.ipynb
├── athena_queries/
│   ├── 01_Create_Athena_Database.ipynb
│   ├── 02_Register_S3_With_Athena.ipynb
│   └── 03_Convert_CSV_To_Parquet_With_Athena.ipynb
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_transformer.py
│   ├── model_gate.py
│   ├── predict.py
│   └── monitor.py
├── docs/
│   ├── aws_mlops_plan.md
│   ├── project_gap_analysis.md
│   └── week_progress.md
├── aws/
│   ├── README.md
│   └── sagemaker_pipeline_skeleton.py
├── models/
└── reports/
```

## Running The Project

The project has two execution paths. Use the AWS path for the final deliverable; the local path is for fast iteration when you do not have SageMaker available.

### AWS / SageMaker path (final deliverable)

1. **One-time upload from your laptop** of the raw Yelp dataset to S3:

   ```bash
   aws s3 cp data/Yelp-JSON.zip s3://yelp-sentiment-mlops-<account-id>/raw/yelp-json.zip
   ```

2. **In SageMaker Studio**, open a terminal and clone this repo:

   ```bash
   git clone https://github.com/marwahfaraj/yelp-sentiment-mlops-pipeline.git
   cd yelp-sentiment-mlops-pipeline
   pip install -r requirements.txt
   ```

3. **Run the notebooks in order.** Each notebook persists its variables with `%store` so the next notebook can pick them up.

   | Step | Notebook                                                     | Deliverable                                  |
   |------|--------------------------------------------------------------|----------------------------------------------|
   | 1    | `notebooks/01_setup_S3_bucket.ipynb`                         | Raw Yelp data lake in S3                     |
   | 2    | `athena_queries/01_Create_Athena_Database.ipynb`             | `yelp_db` database                           |
   | 3    | `athena_queries/02_Register_S3_With_Athena.ipynb`            | `yelp_db.reviews_raw` external table         |
   | 4    | `athena_queries/03_Convert_CSV_To_Parquet_With_Athena.ipynb` | `yelp_db.reviews_parquet` Parquet table      |
   | 5    | `notebooks/02_data_exploration_EDA.ipynb`                    | EDA charts in `reports/`                     |
   | 6    | `notebooks/03_feature_engineering_feature_store.ipynb`       | SageMaker Feature Group + 40/10/10/40 splits |
   | 7    | `notebooks/04_dataset_splits.ipynb`                          | Verified splits + rubric check               |

   This sequence covers every Week module deliverable: S3 data lake, Athena catalog, EDA in a SageMaker notebook, Feature Store, and the 40/10/10/40 train/validation/test/production split.

### Local path (optional fast iteration)

```bash
python3 -m src.run_pipeline
python3 -m src.predict --text "The food was delicious and the service was fast."
```

This runs an end-to-end Naive Bayes baseline with EDA, evaluation, quality gate, and monitoring outputs against a small local sample. It does **not** touch AWS.

## Monitoring Plan

The monitoring component will track:

- Missing or empty review text
- Review length changes over time
- Sentiment prediction distribution changes
- Model quality changes when labeled data is available
- Latency and resource usage for future deployment

## CI/CD Quality Gate

The local project includes a model approval step in `src/model_gate.py`. This simulates the condition step used in an AWS SageMaker Pipeline.

Default approval rule:

```text
macro F1-score >= 0.80
```

If the model passes, it is ready for model registration. If it fails, the pipeline should stop and require review before deployment.

## Final Deliverables

The final project will include:

- Data preparation workflow
- EDA summary and visualizations
- Baseline model
- Transformer-based NLP model comparison
- Model evaluation results
- Prediction workflow
- Monitoring plan or implementation
- AWS MLOps architecture and CI/CD plan
- Final design document and presentation

## Data Notice

The full Yelp Open Dataset should not be committed to this repository because of its size. The downloaded files `data/Yelp-JSON.zip` and `data/Yelp-Photos.zip` are intentionally ignored in `.gitignore`.
