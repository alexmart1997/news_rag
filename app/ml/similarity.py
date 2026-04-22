from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import RUSSIAN_STOPWORDS


def find_similar_articles(
    articles_df: pd.DataFrame,
    selected_index: int,
    top_k: int = 5,
) -> pd.DataFrame:
    if articles_df.empty or selected_index < 0 or selected_index >= len(articles_df):
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    model_df = articles_df.reset_index(drop=True).copy()
    text_series = (
        model_df["title"].fillna("")
        + " "
        + model_df["overview"].fillna("")
        + " "
        + model_df["text"].fillna("")
    ).str.strip()

    if (text_series.str.len() > 0).sum() < 2:
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(RUSSIAN_STOPWORDS),
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(text_series)
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    similarities = cosine_similarity(matrix[selected_index], matrix).flatten()
    model_df["similarity"] = similarities
    model_df = model_df.drop(index=selected_index)
    model_df = model_df.sort_values("similarity", ascending=False, ignore_index=True)
    model_df["similarity"] = model_df["similarity"].round(3)

    return model_df[["id", "title", "topic", "published_at", "similarity", "url"]].head(top_k)
