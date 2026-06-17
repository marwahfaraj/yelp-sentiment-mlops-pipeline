# EDA Summary

## Dataset

- Source table: `yelp_db.reviews_parquet` (Athena, Parquet-backed)
- Total reviews: 299,991
- Reviews after removing 3-star neutral class: 266,005

## Class Balance

- Positive (stars >= 4): 208,445
- Negative (stars <= 2): 57,560
- Positive share: 78.36%
- Negative share: 21.64%

## Review Length (10% sample)

- Average characters per review: 548.1
- Median characters per review: 390.0
- 95th percentile: 1499.0

## Data Quality

- Missing review text: 9
- Missing star rating: 0

## Charts

- `reports/eda_star_distribution.png`
- `reports/eda_class_balance.png`
- `reports/eda_review_length.png`
- `reports/eda_top_words.png`
