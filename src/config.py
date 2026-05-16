"""Shared project paths and constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "Yelp-JSON.zip"
SAMPLE_DATA_PATH = DATA_DIR / "sample_yelp_reviews.csv"
YELP_REVIEW_MEMBER_NAME = "yelp_academic_dataset_review.json"
MAX_RAW_REVIEWS = int(os.getenv("YELP_MAX_REVIEWS", "5000"))
TRAIN_DATA_PATH = DATA_DIR / "processed_train.csv"
VALIDATION_DATA_PATH = DATA_DIR / "processed_validation.csv"
TEST_DATA_PATH = DATA_DIR / "processed_test.csv"
BATCH_DATA_PATH = DATA_DIR / "batch_inference.csv"
BASELINE_STATS_PATH = DATA_DIR / "baseline_stats.json"
MODEL_APPROVAL_PATH = REPORTS_DIR / "model_approval.json"

BASELINE_MODEL_PATH = MODELS_DIR / "sentiment_baseline.json"

TEXT_COLUMN = "review_text"
RATING_COLUMN = "stars"
LABEL_COLUMN = "sentiment"
CLEAN_TEXT_COLUMN = "clean_text"
REVIEW_LENGTH_COLUMN = "review_length"

POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
MINIMUM_F1_SCORE = float(os.getenv("MINIMUM_F1_SCORE", "0.80"))
