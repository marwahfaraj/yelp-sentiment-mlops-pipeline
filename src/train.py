"""Train a standard-library baseline Yelp review sentiment classifier."""

import json
import math

from src.config import (
    BASELINE_MODEL_PATH,
    CLEAN_TEXT_COLUMN,
    LABEL_COLUMN,
    MODELS_DIR,
    TRAIN_DATA_PATH,
)
from src.data_utils import read_csv, tokenize


def train_naive_bayes(rows: list[dict]) -> dict:
    """Train a multinomial Naive Bayes model on review tokens."""
    class_counts: dict[str, int] = {}
    token_counts: dict[str, dict[str, int]] = {}
    total_tokens: dict[str, int] = {}
    vocabulary = set()

    for row in rows:
        label = row[LABEL_COLUMN]
        class_counts[label] = class_counts.get(label, 0) + 1
        token_counts.setdefault(label, {})
        total_tokens.setdefault(label, 0)

        for token in tokenize(row[CLEAN_TEXT_COLUMN]):
            vocabulary.add(token)
            token_counts[label][token] = token_counts[label].get(token, 0) + 1
            total_tokens[label] += 1

    total_documents = sum(class_counts.values())
    class_log_priors = {
        label: math.log(count / total_documents)
        for label, count in class_counts.items()
    }

    return {
        "model_type": "multinomial_naive_bayes",
        "class_counts": class_counts,
        "class_log_priors": class_log_priors,
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "vocabulary": sorted(vocabulary),
    }


def main() -> None:
    """Train and save the baseline model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = read_csv(TRAIN_DATA_PATH)

    model = train_naive_bayes(train_rows)
    BASELINE_MODEL_PATH.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved baseline model to {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()
