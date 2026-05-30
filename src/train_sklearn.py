"""Train the Week 4 Yelp sentiment baseline model in SageMaker.

The training job expects SageMaker channels named ``train``, ``validation``,
and optionally ``test``. Each channel should contain the CSV files created by
the feature engineering notebook with ``clean_text`` and ``sentiment_label``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "sentiment_label"


def _read_channel_csv(channel_dir: str) -> pd.DataFrame:
    """Read and concatenate CSV files from a SageMaker input channel."""
    path = Path(channel_dir)
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {channel_dir}")
    return pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    missing = {TEXT_COLUMN, LABEL_COLUMN}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    clean_df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    clean_df[TEXT_COLUMN] = clean_df[TEXT_COLUMN].astype(str)
    clean_df[LABEL_COLUMN] = clean_df[LABEL_COLUMN].astype(int)
    return clean_df[TEXT_COLUMN], clean_df[LABEL_COLUMN]


def _evaluate(model: Pipeline, df: pd.DataFrame) -> dict:
    x, y_true = _prepare_xy(df)
    y_pred = model.predict(x)
    labels = [0, 1]
    return {
        "row_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("-c", "--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument(
        "--validation",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = _read_channel_csv(args.train)
    validation_df = _read_channel_csv(args.validation)
    test_df = _read_channel_csv(args.test) if args.test else None

    x_train, y_train = _prepare_xy(train_df)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    n_jobs=-1,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    metrics = {
        "model_type": "tfidf_logistic_regression",
        "hyperparameters": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "c": args.c,
            "max_iter": args.max_iter,
        },
        "train_rows": int(len(x_train)),
        "validation": _evaluate(model, validation_df),
    }
    if test_df is not None:
        metrics["test"] = _evaluate(model, test_df)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
