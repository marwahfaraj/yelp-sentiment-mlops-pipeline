"""Evaluate an optional Hugging Face transformer sentiment model."""

import json

from src.config import (
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from src.data_utils import read_csv
from src.evaluate import calculate_metrics


HF_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def map_transformer_label(label: str) -> str:
    """Map Hugging Face sentiment labels to project labels."""
    normalized = label.lower()
    if "positive" in normalized:
        return POSITIVE_LABEL
    if "negative" in normalized:
        return NEGATIVE_LABEL
    raise ValueError(f"Unexpected transformer label: {label}")


def main() -> None:
    """Evaluate a pre-trained transformer model on the test set."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Install transformer dependencies with `pip install -r requirements.txt`."
        ) from exc

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = read_csv(TEST_DATA_PATH)
    classifier = pipeline("sentiment-analysis", model=HF_MODEL_NAME)

    review_texts = [row[CLEAN_TEXT_COLUMN] for row in test_rows]
    raw_predictions = classifier(review_texts, truncation=True)
    predictions = [
        map_transformer_label(prediction["label"]) for prediction in raw_predictions
    ]

    y_true = [row[LABEL_COLUMN] for row in test_rows]
    metrics = calculate_metrics(y_true, predictions, [NEGATIVE_LABEL, POSITIVE_LABEL])
    metrics = {
        "model_name": HF_MODEL_NAME,
        **metrics,
    }

    report_path = REPORTS_DIR / "transformer_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved transformer evaluation to {report_path}")


if __name__ == "__main__":
    main()
