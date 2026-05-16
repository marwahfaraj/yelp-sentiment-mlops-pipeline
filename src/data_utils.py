"""Data loading, cleaning, labeling, and baseline statistic helpers."""

import csv
import json
import re
import tarfile
import zipfile
from pathlib import Path

from src.config import (
    BASELINE_STATS_PATH,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    MAX_RAW_REVIEWS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RATING_COLUMN,
    REVIEW_LENGTH_COLUMN,
    TEXT_COLUMN,
    YELP_REVIEW_MEMBER_NAME,
)


def tokenize(text: str) -> list[str]:
    """Split normalized review text into simple word tokens."""
    return re.findall(r"[a-z']+", text.lower())


def clean_review_text(text: str) -> str:
    """Normalize whitespace and casing while preserving sentiment words."""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def rating_to_sentiment(stars: int) -> str | None:
    """Convert Yelp-style star ratings into binary sentiment labels."""
    if stars >= 4:
        return POSITIVE_LABEL
    if stars <= 2:
        return NEGATIVE_LABEL
    return None


def load_csv_reviews(path: Path) -> list[dict]:
    """Load review data from a small CSV file."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {TEXT_COLUMN, RATING_COLUMN}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        return list(reader)


def normalize_yelp_review(review: dict) -> dict:
    """Map Yelp Open Dataset review fields into the project schema."""
    return {
        "review_id": review.get("review_id", ""),
        "business_id": review.get("business_id", ""),
        "user_id": review.get("user_id", ""),
        RATING_COLUMN: review.get("stars", ""),
        TEXT_COLUMN: review.get("text", ""),
        "date": review.get("date", ""),
    }


def load_yelp_reviews_from_zip(path: Path, max_reviews: int = MAX_RAW_REVIEWS) -> list[dict]:
    """Read Yelp review JSON records from the downloaded nested zip/tar archive."""
    rows = []
    with zipfile.ZipFile(path) as zip_file:
        tar_name = next(
            name
            for name in zip_file.namelist()
            if name.endswith(".tar") and not name.startswith("__MACOSX/")
        )
        with zip_file.open(tar_name) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode="r|gz") as tar_file:
                for member in tar_file:
                    if not member.name.endswith(YELP_REVIEW_MEMBER_NAME):
                        continue
                    extracted = tar_file.extractfile(member)
                    if extracted is None:
                        break
                    for line in extracted:
                        review = json.loads(line.decode("utf-8"))
                        rows.append(normalize_yelp_review(review))
                        if len(rows) >= max_reviews:
                            return rows
                    break
    return rows


def load_raw_data(path: Path) -> list[dict]:
    """Load review data from the real Yelp archive or a development CSV."""
    if path.suffix.lower() == ".csv":
        return load_csv_reviews(path)
    if path.suffix.lower() == ".zip":
        return load_yelp_reviews_from_zip(path)
    raise ValueError(f"Unsupported raw data format: {path}")


def prepare_reviews(rows: list[dict]) -> list[dict]:
    """Clean reviews, create sentiment labels, and remove neutral examples."""
    prepared = []
    for row in rows:
        review_text = row.get(TEXT_COLUMN, "")
        stars = row.get(RATING_COLUMN, "")
        if not review_text or not stars:
            continue
        sentiment = rating_to_sentiment(int(stars))
        if sentiment is None:
            continue
        clean_text = clean_review_text(review_text)
        if not clean_text:
            continue
        prepared_row = dict(row)
        prepared_row[LABEL_COLUMN] = sentiment
        prepared_row[CLEAN_TEXT_COLUMN] = clean_text
        prepared_row[REVIEW_LENGTH_COLUMN] = str(len(tokenize(clean_text)))
        prepared.append(prepared_row)
    return prepared


def write_csv(rows: list[dict], path: Path) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path | str) -> list[dict]:
    """Read dictionaries from CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def label_distribution(rows: list[dict], label_column: str) -> dict:
    """Calculate normalized label distribution."""
    counts: dict[str, int] = {}
    for row in rows:
        label = row[label_column]
        counts[label] = counts.get(label, 0) + 1
    total = max(sum(counts.values()), 1)
    return {label: round(count / total, 4) for label, count in counts.items()}


def build_baseline_stats(train_rows: list[dict]) -> dict:
    """Capture training-time statistics for later monitoring checks."""
    review_lengths = [int(row[REVIEW_LENGTH_COLUMN]) for row in train_rows]
    average_review_length = sum(review_lengths) / max(len(review_lengths), 1)
    return {
        "row_count": len(train_rows),
        "label_distribution": label_distribution(train_rows, LABEL_COLUMN),
        "average_review_length": round(average_review_length, 4),
        "missing_text_count": sum(1 for row in train_rows if not row.get(TEXT_COLUMN)),
    }


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_baseline_stats(path: Path = BASELINE_STATS_PATH) -> dict:
    """Load saved baseline monitoring statistics."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
