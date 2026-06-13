"""Preprocessing step for the Yelp sentiment SageMaker Pipeline.

The ProcessingStep downloads the existing split CSVs (created by notebook 03)
into ``/opt/ml/processing/input`` and this script validates them, drops bad
rows, and writes one CSV per output channel. It also materializes the
production reviews as one-text-per-line input for the Batch Transform step.

This acts as the pipeline's data-validation / system-integration checkpoint:
if the schema is wrong or a split is empty, the pipeline fails here and the
training step never runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["clean_text", "sentiment_label"]
SPLIT_CHANNELS = ["train", "validation", "test"]


def load_split(input_dir: Path, split_name: str) -> pd.DataFrame:
    """Find and read a split CSV under the processing input directory."""
    candidates = sorted(input_dir.rglob(f"{split_name}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV found for split '{split_name}' under {input_dir}")
    return pd.concat((pd.read_csv(path) for path in candidates), ignore_index=True)


def validate_and_clean(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Split '{split_name}' is missing required columns: {sorted(missing)}")

    clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    clean["clean_text"] = clean["clean_text"].astype(str)
    clean = clean[clean["clean_text"].str.strip().str.len() > 0]
    clean["sentiment_label"] = clean["sentiment_label"].astype(int)

    if clean.empty:
        raise ValueError(f"Split '{split_name}' has no valid rows after cleaning")
    if not set(clean["sentiment_label"].unique()).issubset({0, 1}):
        raise ValueError(f"Split '{split_name}' contains labels outside 0/1")

    dropped = len(df) - len(clean)
    print(f"[{split_name}] rows={len(clean):,} dropped={dropped:,} positive_share={clean['sentiment_label'].mean():.4f}")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-base", type=str, default="/opt/ml/processing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)

    for split_name in SPLIT_CHANNELS:
        df = load_split(input_dir, split_name)
        df = validate_and_clean(df, split_name)
        out_dir = output_base / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{split_name}.csv", index=False)

    production_df = load_split(input_dir, "production")
    production_df = validate_and_clean(production_df, "production")
    batch_dir = output_base / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_text = production_df["clean_text"].str.replace(r"\s+", " ", regex=True)
    (batch_dir / "production_text.txt").write_text("\n".join(batch_text.tolist()) + "\n", encoding="utf-8")
    print(f"[batch] wrote {len(batch_text):,} production reviews for batch transform")


if __name__ == "__main__":
    main()
