from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from mcp.server import MCPServer

PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "customer_reviews.db"
REVIEWS_PATH = PROJECT_DIR / "rag_reviews.pkl"
EMBEDDINGS_PATH = PROJECT_DIR / "review_embeddings.npy"

rag_reviews = pd.read_pickle(REVIEWS_PATH)
review_embeddings = np.load(EMBEDDINGS_PATH)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

mcp = MCPServer("Customer Voice Intelligence")


@mcp.tool()
def get_product_feedback(product_id: str) -> str:
    """Get review count, average rating, and low-rating share for a product."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*),
                ROUND(AVG(rating), 2),
                ROUND(
                    100.0 * SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END)
                    / COUNT(*),
                    1
                )
            FROM reviews
            WHERE product_id = ?
            """,
            (product_id.strip(),),
        ).fetchone()

    count, average_rating, low_rating_percent = row
    if count == 0:
        return f"No reviews found for product {product_id}."

    return (
        f"Product: {product_id} | Reviews: {count} | "
        f"Average rating: {average_rating}/5 | "
        f"Low ratings (1–2 stars): {low_rating_percent}%"
    )


@mcp.tool()
def search_reviews(
    query: str,
    top_k: int = 3,
    sentiment: str = "negative",
) -> str:
    """Retrieve relevant customer reviews using semantic search."""
    sentiment = sentiment.lower().strip()

    if sentiment not in {"negative", "neutral", "positive", "all"}:
        return "Choose sentiment: negative, neutral, positive, or all."

    top_k = max(1, min(int(top_k), 5))

    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = review_embeddings @ query_vector

    if sentiment == "all":
        allowed = np.ones(len(rag_reviews), dtype=bool)
    else:
        allowed = rag_reviews["sentiment"].eq(sentiment).to_numpy()

    candidates = np.flatnonzero(allowed)
    ranked = candidates[
        np.argsort(scores[candidates])[::-1][:top_k]
    ]

    matches = []
    for index in ranked:
        review = rag_reviews.iloc[index]
        matches.append(
            f"Review ID: {review['review_id']} | "
            f"Product: {review['product_id']} | "
            f"Rating: {review['rating']} | "
            f"Similarity: {scores[index]:.3f}\n"
            f"{str(review['review_text'])[:700]}"
        )

    return "\n\n".join(matches) if matches else "No matching reviews found."
