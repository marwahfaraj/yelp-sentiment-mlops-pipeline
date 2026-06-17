# Dataset Splits Summary

Split policy: train 40% | validation 10% | test 10% | production 40%

| split_type   |   row_count |   share_of_total |   positive_count |   negative_count |   positive_share |   avg_review_word_count |
|:-------------|------------:|-----------------:|-----------------:|-----------------:|-----------------:|------------------------:|
| train        |       20000 |              0.4 |            10000 |            10000 |              0.5 |                   109.5 |
| validation   |        5000 |              0.1 |             2500 |             2500 |              0.5 |                   108.2 |
| test         |        5000 |              0.1 |             2500 |             2500 |              0.5 |                   110.6 |
| production   |       20000 |              0.4 |            10000 |            10000 |              0.5 |                   108.3 |

Training records per class meet >=10,000 rubric requirement: **PASS**

Feature Group: `yelp-sentiment-feature-group-11-01-06-06`
S3 prefix:    `s3://yelp-sentiment-mlops-965705611982/processed/splits/`