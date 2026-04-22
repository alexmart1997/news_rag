from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import ALL_STOPWORDS, build_analysis_text


def find_similar_articles(
    articles_df: pd.DataFrame,
    selected_index: int,
    top_k: int = 5,
) -> pd.DataFrame:
    if articles_df.empty or selected_index < 0 or selected_index >= len(articles_df):
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    model_df = articles_df.reset_index(drop=True).copy()
    text_series = build_analysis_text(model_df)

    if (text_series.str.len() > 0).sum() < 2:
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ALL_STOPWORDS),
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=4000,
    )
    word_matrix = word_vectorizer.fit_transform(text_series)
    char_matrix = char_vectorizer.fit_transform(text_series)
    if word_matrix.shape[1] == 0 or char_matrix.shape[1] == 0:
        return pd.DataFrame(columns=["id", "title", "topic", "published_at", "similarity", "url"])

    word_similarities = cosine_similarity(word_matrix[selected_index], word_matrix).flatten()
    char_similarities = cosine_similarity(char_matrix[selected_index], char_matrix).flatten()
    similarities = 0.75 * word_similarities + 0.25 * char_similarities

    if "topic" in model_df.columns:
        selected_topic = model_df.iloc[selected_index]["topic"]
        topic_bonus = (model_df["topic"] == selected_topic).astype(float) * 0.08
        similarities = similarities + topic_bonus.to_numpy()

    model_df["similarity"] = similarities.clip(0, 1)
    model_df = model_df.drop(index=selected_index)
    model_df = model_df.sort_values("similarity", ascending=False, ignore_index=True)
    model_df["similarity"] = model_df["similarity"].round(3)

    return model_df[["id", "title", "topic", "published_at", "similarity", "url"]].head(top_k)
