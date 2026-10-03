# Customer Voice Intelligence Platform

An end-to-end project for exploring customer reviews with sentiment analysis, SQL, topic analysis, and semantic search.

## What it does

- Analyzes Amazon Fine Food reviews using SQLite and SQL.
- Classifies reviews as negative, neutral, or positive using TF-IDF and Logistic Regression.
- Groups negative reviews into topics and explores common complaint themes.
- Finds similar reviews using semantic search.
- Provides local MCP tools for product feedback lookup and review search.

## Model evaluation

The model was tested on later-dated reviews:

- Accuracy: 79.6%
- Macro F1: 0.632
- Majority-baseline macro F1: 0.289

The model performed best on positive reviews. Neutral reviews were harder to classify because a 3-star rating does not always match the sentiment expressed in the review text.

## Dataset

This project uses the Amazon Fine Foods Reviews dataset from [Stanford SNAP](https://snap.stanford.edu/data/web-FineFoods.html). Download the dataset separately; do not upload the dataset or generated database and embedding files to GitHub.

## How to run

1. Download the dataset from Stanford SNAP.
2. Open `Customer_Voice_Intelligence_Sentiment_RAG.ipynb` in Jupyter.
3. Install any Python packages that the notebook says are missing.
4. Run the notebook cells from top to bottom.
5. Update file paths if your dataset is saved in a different folder.

The optional OpenAI-powered answer function requires an API key and available API credits. Never put your API key in the notebook or GitHub.

## Limitations

- Sentiment labels come from star ratings, which can disagree with the review wording.
- Topic and complaint categories are exploratory and may need manual review.
- Results on this historical dataset may not represent current customer reviews.
