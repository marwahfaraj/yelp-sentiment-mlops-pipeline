# Data

This folder stores local data for the Yelp review sentiment MLOps project.

## Current Data Source

The project uses the Yelp Open Dataset downloaded from:

<https://business.yelp.com/data/resources/open-dataset/>

Local downloaded files:

- `Yelp-JSON.zip`: contains the Yelp JSON archive, including `yelp_academic_dataset_review.json`
- `Yelp-Photos.zip`: contains Yelp photo data
- `sample_yelp_reviews.csv`: small fallback sample dataset for quick development tests

The NLP sentiment model uses the review JSON data from `Yelp-JSON.zip`. The photo zip is not used in the first version because this project focuses on text classification, but it may support future multimodal analysis.

## Important Git Note

The Yelp zip files are large and should not be committed to GitHub. They are intentionally listed in `.gitignore`.

## Expected Review Fields

Expected columns:

- `review_text`: customer review text
- `stars`: Yelp star rating

The original Yelp review JSON uses `text` for review text. The project loader maps that field to `review_text`.

## EDA

The final project includes exploratory data analysis before modeling. Planned EDA includes:

- Missing review text checks
- Star rating distribution
- Positive vs negative class balance
- Average review length
- Common words and phrases by sentiment class
- Examples of ambiguous reviews

EDA is useful because it shows whether the data is balanced, clean, and appropriate for model training.
