"""Evaluate the baseline sentiment classifier."""

import json

from src.config import (
    BASELINE_MODEL_PATH,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from src.data_utils import read_csv, tokenize
from src.predict import predict_with_model


def calculate_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Calculate accuracy, precision, recall, and F1 without external packages."""
    total = len(y_true)
    correct = sum(1 for actual, predicted in zip(y_true, y_pred, strict=True) if actual == predicted)
    per_label = {}

    for label in labels:
        true_positive = sum(
            1
            for actual, predicted in zip(y_true, y_pred, strict=True)
            if actual == label and predicted == label
        )
        false_positive = sum(
            1
            for actual, predicted in zip(y_true, y_pred, strict=True)
            if actual != label and predicted == label
        )
        false_negative = sum(
            1
            for actual, predicted in zip(y_true, y_pred, strict=True)
            if actual == label and predicted != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for actual in y_true if actual == label),
        }

    return {
        "accuracy": round(correct / max(total, 1), 4),
        "precision_macro": round(
            sum(values["precision"] for values in per_label.values()) / len(labels),
            4,
        ),
        "recall_macro": round(
            sum(values["recall"] for values in per_label.values()) / len(labels),
            4,
        ),
        "f1_macro": round(
            sum(values["f1"] for values in per_label.values()) / len(labels),
            4,
        ),
        "per_label": per_label,
    }


def confusion_matrix_rows(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict]:
    """Create confusion matrix rows for CSV output."""
    rows = []
    for actual_label in labels:
        row = {"actual": actual_label}
        for predicted_label in labels:
            row[f"predicted_{predicted_label}"] = sum(
                1
                for actual, predicted in zip(y_true, y_pred, strict=True)
                if actual == actual_label and predicted == predicted_label
            )
        rows.append(row)
    return rows


def main() -> None:
    """Evaluate saved model and write reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = json.loads(BASELINE_MODEL_PATH.read_text(encoding="utf-8"))
    test_rows = read_csv(TEST_DATA_PATH)
    labels = [NEGATIVE_LABEL, POSITIVE_LABEL]

    y_true = [row[LABEL_COLUMN] for row in test_rows]
    y_pred = [
        predict_with_model(model, tokenize(row[CLEAN_TEXT_COLUMN]))["label"]
        for row in test_rows
    ]

    metrics = calculate_metrics(y_true, y_pred, labels)
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm_path = REPORTS_DIR / "confusion_matrix.csv"
    matrix_rows = confusion_matrix_rows(y_true, y_pred, labels)
    header = ["actual"] + [f"predicted_{label}" for label in labels]
    csv_lines = [",".join(header)]
    for row in matrix_rows:
        csv_lines.append(",".join(str(row[column]) for column in header))
    cm_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    summary_path = REPORTS_DIR / "evaluation_summary.md"
    summary_path.write_text(
        f"""# Baseline Model Evaluation

## Metrics

- Accuracy: {metrics["accuracy"]:.3f}
- Macro precision: {metrics["precision_macro"]:.3f}
- Macro recall: {metrics["recall_macro"]:.3f}
- Macro F1-score: {metrics["f1_macro"]:.3f}

## Confusion Matrix

Saved to `{cm_path}`.

## Interpretation

These scores are based on the configured Yelp review subset. For faster local
development, the loader can limit the number of raw Yelp reviews using the
`YELP_MAX_REVIEWS` environment variable. Larger training runs should use a
bigger review sample for more reliable final model conclusions.
""",
        encoding="utf-8",
    )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
