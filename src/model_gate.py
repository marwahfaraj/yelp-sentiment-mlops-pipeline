"""Approve or reject a trained model based on evaluation thresholds."""

import json

from src.config import MINIMUM_F1_SCORE, MODEL_APPROVAL_PATH, REPORTS_DIR


def main() -> None:
    """Create a model approval report similar to a CI/CD quality gate."""
    metrics_path = REPORTS_DIR / "baseline_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    f1_score = float(metrics["f1_macro"])
    approved = f1_score >= MINIMUM_F1_SCORE

    approval_report = {
        "metric": "f1_macro",
        "metric_value": f1_score,
        "threshold": MINIMUM_F1_SCORE,
        "status": "approved" if approved else "rejected",
        "next_step": (
            "register_model"
            if approved
            else "stop_pipeline_and_review_training_data_or_parameters"
        ),
    }

    MODEL_APPROVAL_PATH.write_text(
        json.dumps(approval_report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(approval_report, indent=2))
    print(f"Saved model approval report to {MODEL_APPROVAL_PATH}")


if __name__ == "__main__":
    main()
