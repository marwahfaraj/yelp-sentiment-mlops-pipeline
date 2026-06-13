"""Evaluation step for the Yelp sentiment SageMaker Pipeline.

Loads the trained model artifact, scores the held-out test split, and writes
``evaluation.json``. The pipeline's ConditionStep reads ``classification_metrics
.f1_macro.value`` from that file to decide whether the model may be registered
and deployed (quality gate: macro F1 >= threshold).
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MODEL_DIR = Path("/opt/ml/processing/model")
TEST_DIR = Path("/opt/ml/processing/test")
OUTPUT_DIR = Path("/opt/ml/processing/evaluation")


def load_model():
    archive = MODEL_DIR / "model.tar.gz"
    extract_dir = MODEL_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    return joblib.load(extract_dir / "model.joblib")


def load_test_frame() -> pd.DataFrame:
    csv_files = sorted(TEST_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No test CSV files found in {TEST_DIR}")
    return pd.concat((pd.read_csv(path) for path in csv_files), ignore_index=True)


def main() -> None:
    model = load_model()
    test_df = load_test_frame()

    test_df = test_df.dropna(subset=["clean_text", "sentiment_label"])
    x_test = test_df["clean_text"].astype(str)
    y_true = test_df["sentiment_label"].astype(int)
    y_pred = model.predict(x_test)

    report = {
        "classification_metrics": {
            "f1_macro": {"value": float(f1_score(y_true, y_pred, average="macro", zero_division=0))},
            "accuracy": {"value": float(accuracy_score(y_true, y_pred))},
            "precision_macro": {"value": float(precision_score(y_true, y_pred, average="macro", zero_division=0))},
            "recall_macro": {"value": float(recall_score(y_true, y_pred, average="macro", zero_division=0))},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "test_rows": int(len(y_true)),
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
