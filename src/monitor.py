"""Simple monitoring checks for review sentiment inputs and predictions."""

import argparse
import json

from src.config import (
    BASELINE_MODEL_PATH,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    REPORTS_DIR,
    TEST_DATA_PATH,
    TEXT_COLUMN,
)
from src.data_utils import clean_review_text, load_baseline_stats, read_csv, tokenize
from src.predict import predict_with_model


def distribution_gap(current_distribution: dict, baseline_distribution: dict) -> dict:
    """Calculate absolute label-distribution changes."""
    labels = set(current_distribution) | set(baseline_distribution)
    return {
        label: round(
            abs(current_distribution.get(label, 0.0) - baseline_distribution.get(label, 0.0)),
            4,
        )
        for label in sorted(labels)
    }


def monitor(input_path=TEST_DATA_PATH) -> dict:
    """Compare current prediction/input statistics against the training baseline."""
    model = json.loads(BASELINE_MODEL_PATH.read_text(encoding="utf-8"))
    baseline = load_baseline_stats()
    current_rows = read_csv(input_path)

    predicted_labels = []
    review_lengths = []
    missing_text_count = 0

    for row in current_rows:
        review_text = row.get(CLEAN_TEXT_COLUMN) or clean_review_text(
            row.get(TEXT_COLUMN, "")
        )
        if not row.get(TEXT_COLUMN):
            missing_text_count += 1

        tokens = tokenize(review_text)
        review_lengths.append(len(tokens))
        prediction = predict_with_model(model, tokens)
        predicted_labels.append(prediction["label"])

    prediction_counts: dict[str, int] = {}
    for label in predicted_labels:
        prediction_counts[label] = prediction_counts.get(label, 0) + 1
    total_predictions = max(len(predicted_labels), 1)
    prediction_distribution = {
        label: round(count / total_predictions, 4)
        for label, count in prediction_counts.items()
    }
    baseline_distribution = baseline.get("label_distribution", {})
    average_review_length = round(
        sum(review_lengths) / max(len(review_lengths), 1),
        4,
    )
    baseline_length = float(baseline.get("average_review_length", 0.0))

    prediction_shift = distribution_gap(
        prediction_distribution,
        baseline_distribution,
    )
    max_prediction_shift = max(prediction_shift.values()) if prediction_shift else 0.0
    length_shift = abs(average_review_length - baseline_length)

    alerts = []
    if max_prediction_shift > 0.30:
        alerts.append("Prediction distribution shifted by more than 30 percentage points.")
    if length_shift > 5:
        alerts.append("Average review length shifted by more than 5 words.")
    if missing_text_count > 0:
        alerts.append("Missing review text detected.")

    return {
        "input_path": str(input_path),
        "row_count": len(current_rows),
        "prediction_distribution": prediction_distribution,
        "baseline_label_distribution": baseline_distribution,
        "prediction_distribution_gap": prediction_shift,
        "average_review_length": average_review_length,
        "baseline_average_review_length": baseline_length,
        "missing_text_count": missing_text_count,
        "status": "alert" if alerts else "ok",
        "alerts": alerts,
    }


def main() -> None:
    """Run monitoring checks and save a report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(TEST_DATA_PATH),
        help="CSV file to monitor. Defaults to processed test data.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = monitor(args.input)
    report_path = REPORTS_DIR / "monitoring_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved monitoring report to {report_path}")


if __name__ == "__main__":
    main()
