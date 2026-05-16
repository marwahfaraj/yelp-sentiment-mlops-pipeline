"""Predict sentiment for a new customer review."""

import argparse
import json
import math

from src.config import BASELINE_MODEL_PATH
from src.data_utils import clean_review_text, tokenize


POSITIVE_KEYWORDS = {
    "amazing",
    "best",
    "clean",
    "delicious",
    "excellent",
    "fast",
    "fresh",
    "friendly",
    "great",
    "helpful",
    "loved",
    "perfect",
    "quick",
    "welcome",
    "wonderful",
}

NEGATIVE_KEYWORDS = {
    "bad",
    "bland",
    "cold",
    "dirty",
    "dry",
    "late",
    "missing",
    "overpriced",
    "slow",
    "undercooked",
    "worst",
}


def predict_with_model(model: dict, tokens: list[str]) -> dict:
    """Predict with a saved multinomial Naive Bayes model."""
    vocabulary_size = len(model["vocabulary"])
    scores = {}

    for label, log_prior in model["class_log_priors"].items():
        score = log_prior
        label_token_counts = model["token_counts"][label]
        label_total_tokens = model["total_tokens"][label]
        denominator = label_total_tokens + vocabulary_size

        for token in tokens:
            count = label_token_counts.get(token, 0)
            score += math.log((count + 1) / denominator)
        scores[label] = score

    # Small domain lexicon boost helps the tiny sample model handle clear reviews.
    positive_hits = sum(1 for token in tokens if token in POSITIVE_KEYWORDS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_KEYWORDS)
    if "positive" in scores:
        scores["positive"] += 0.9 * positive_hits
    if "negative" in scores:
        scores["negative"] += 0.9 * negative_hits

    best_label = max(scores, key=scores.get)
    max_score = max(scores.values())
    exp_scores = {
        label: math.exp(score - max_score) for label, score in scores.items()
    }
    total_exp_score = sum(exp_scores.values())
    probabilities = {
        label: round(value / total_exp_score, 4)
        for label, value in exp_scores.items()
    }
    return {"label": best_label, "probabilities": probabilities}


def predict_sentiment(review_text: str) -> dict:
    """Return predicted sentiment and class probabilities."""
    model = json.loads(BASELINE_MODEL_PATH.read_text(encoding="utf-8"))
    clean_text = clean_review_text(review_text)
    prediction = predict_with_model(model, tokenize(clean_text))
    return {
        "review_text": review_text,
        "predicted_sentiment": prediction["label"],
        "class_probabilities": prediction["probabilities"],
    }


def main() -> None:
    """Parse a review string and print a sentiment prediction."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        required=True,
        help="Review text to classify.",
    )
    args = parser.parse_args()
    result = predict_sentiment(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
