"""Prepare Yelp-style review data for sentiment modeling."""

import random

from src.config import (
    BASELINE_STATS_PATH,
    BATCH_DATA_PATH,
    CLEAN_TEXT_COLUMN,
    DATA_DIR,
    LABEL_COLUMN,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    VALIDATION_SIZE,
)
from src.data_utils import (
    build_baseline_stats,
    load_raw_data,
    label_distribution,
    prepare_reviews,
    save_json,
    write_csv,
)


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split rows into train/validation/test sets while preserving label balance."""
    random.seed(RANDOM_STATE)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[LABEL_COLUMN], []).append(row)

    train_rows = []
    validation_rows = []
    test_rows = []
    for label_rows in grouped.values():
        shuffled = label_rows[:]
        random.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * TEST_SIZE))
        validation_count = max(1, round(len(shuffled) * VALIDATION_SIZE))
        test_rows.extend(shuffled[:test_count])
        validation_rows.extend(shuffled[test_count : test_count + validation_count])
        train_rows.extend(shuffled[test_count + validation_count :])

    random.shuffle(train_rows)
    random.shuffle(validation_rows)
    random.shuffle(test_rows)
    return train_rows, validation_rows, test_rows


def build_batch_rows(test_rows: list[dict]) -> list[dict]:
    """Create unlabeled batch inference input from the test split."""
    batch_rows = []
    for row in test_rows:
        batch_rows.append(
            {
                "review_id": row.get("review_id", ""),
                CLEAN_TEXT_COLUMN: row[CLEAN_TEXT_COLUMN],
            }
        )
    return batch_rows


def write_eda_report(prepared_rows, train_rows, validation_rows, test_rows) -> None:
    """Write a small EDA summary for the project report folder."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "eda_summary.md"
    label_counts: dict[str, int] = {}
    review_lengths = []
    for row in prepared_rows:
        label_counts[row[LABEL_COLUMN]] = label_counts.get(row[LABEL_COLUMN], 0) + 1
        review_lengths.append(int(row["review_length"]))
    average_length = sum(review_lengths) / max(len(review_lengths), 1)

    report = f"""# EDA Summary

## Dataset

- Raw source: `{RAW_DATA_PATH}`
- Prepared rows: {len(prepared_rows)}
- Training rows: {len(train_rows)}
- Validation rows: {len(validation_rows)}
- Test rows: {len(test_rows)}

## Label Balance

- Counts: {label_counts}
- Distribution: {label_distribution(prepared_rows, LABEL_COLUMN)}

## Review Length

- Average review length: {average_length:.2f} words
- Minimum review length: {min(review_lengths)} words
- Maximum review length: {max(review_lengths)} words

## Data Split

- Training split supports model fitting.
- Validation split supports model selection, threshold checks, and future model registry approval.
- Test split supports final holdout evaluation.
- Batch inference file simulates unlabeled production reviews.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Create processed train/test datasets and monitoring baseline stats."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_data(RAW_DATA_PATH)
    prepared_rows = prepare_reviews(raw_rows)

    train_rows, validation_rows, test_rows = stratified_split(prepared_rows)
    batch_rows = build_batch_rows(test_rows)
    write_csv(train_rows, TRAIN_DATA_PATH)
    write_csv(validation_rows, VALIDATION_DATA_PATH)
    write_csv(test_rows, TEST_DATA_PATH)
    write_csv(batch_rows, BATCH_DATA_PATH)

    baseline_stats = build_baseline_stats(train_rows)
    save_json(baseline_stats, BASELINE_STATS_PATH)
    write_eda_report(prepared_rows, train_rows, validation_rows, test_rows)

    print(
        "Prepared "
        f"{len(train_rows)} training rows, "
        f"{len(validation_rows)} validation rows, and "
        f"{len(test_rows)} test rows."
    )
    print(f"Saved training data to {TRAIN_DATA_PATH}")
    print(f"Saved validation data to {VALIDATION_DATA_PATH}")
    print(f"Saved test data to {TEST_DATA_PATH}")
    print(f"Saved batch inference data to {BATCH_DATA_PATH}")
    print(f"Saved baseline stats to {BASELINE_STATS_PATH}")


if __name__ == "__main__":
    main()
