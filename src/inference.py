"""Inference functions for SageMaker Batch Transform.

The model artifact is a scikit-learn Pipeline trained by ``train_sklearn.py``.
Batch input may be either raw text lines or CSV rows. If CSV data includes a
``clean_text`` or ``review_text`` column, that column is used for prediction.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd


TEXT_COLUMNS = ("clean_text", "review_text", "text")


def model_fn(model_dir: str):
    return joblib.load(Path(model_dir) / "model.joblib")


def input_fn(request_body: str, request_content_type: str):
    if request_content_type in ("text/csv", "application/csv"):
        data = pd.read_csv(io.StringIO(request_body))
        for column in TEXT_COLUMNS:
            if column in data.columns:
                return data[column].fillna("").astype(str)
        return data.iloc[:, 0].fillna("").astype(str)

    if request_content_type in ("text/plain", "application/jsonlines"):
        return pd.Series([line for line in request_body.splitlines() if line.strip()])

    if request_content_type == "application/json":
        payload = json.loads(request_body)
        if isinstance(payload, dict):
            records = payload.get("instances") or payload.get("reviews") or [payload]
        else:
            records = payload
        return pd.Series([record.get("clean_text", record.get("review_text", str(record))) for record in records])

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    labels = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return [
        {
            "predicted_label": int(label),
            "predicted_sentiment": "positive" if int(label) == 1 else "negative",
            "negative_probability": float(prob[0]),
            "positive_probability": float(prob[1]),
        }
        for label, prob in zip(labels, probabilities)
    ]


def output_fn(prediction, accept: str):
    if accept == "text/csv":
        return pd.DataFrame(prediction).to_csv(index=False), "text/csv"
    return json.dumps(prediction), "application/json"


if __name__ == "__main__":
    # Local smoke test helper for an extracted model directory.
    model_directory = os.environ.get("SM_MODEL_DIR", "model")
    model = model_fn(model_directory)
    sample = pd.Series(["great food and friendly service", "terrible wait and cold food"])
    print(output_fn(predict_fn(sample, model), "application/json")[0])
