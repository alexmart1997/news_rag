from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.news_analytics import RUSSIAN_STOPWORDS


@dataclass(slots=True)
class ClusteringResult:
    articles_df: pd.DataFrame
    topic_summary_df: pd.DataFrame


def cluster_articles(
    articles_df: pd.DataFrame,
    n_clusters: int = 5,
    top_terms: int = 5,
) -> ClusteringResult:
    if articles_df.empty:
        return ClusteringResult(
            articles_df=articles_df.copy(),
            topic_summary_df=pd.DataFrame(columns=["topic", "size", "keywords"]),
        )

    model_df = articles_df.copy().reset_index(drop=True)
    model_df["topic"] = "Topic 1"

    text_series = (
        model_df["title"].fillna("")
        + " "
        + model_df["overview"].fillna("")
        + " "
        + model_df["text"].fillna("")
    ).str.strip()

    non_empty_mask = text_series.str.len() > 0
    usable_count = int(non_empty_mask.sum())
    if usable_count < 2:
        summary_df = pd.DataFrame(
            [{"topic": "Topic 1", "size": len(model_df), "keywords": "not enough text"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    effective_clusters = max(1, min(n_clusters, usable_count))
    if effective_clusters == 1:
        summary_df = pd.DataFrame(
            [{"topic": "Topic 1", "size": len(model_df), "keywords": "single cluster"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    usable_texts = text_series[non_empty_mask]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(RUSSIAN_STOPWORDS),
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(usable_texts)

    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        summary_df = pd.DataFrame(
            [{"topic": "Topic 1", "size": len(model_df), "keywords": "not enough features"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)

    model_df.loc[non_empty_mask, "topic"] = [f"Topic {label + 1}" for label in labels]

    feature_names = vectorizer.get_feature_names_out()
    topic_rows: list[dict[str, object]] = []
    for topic_index in range(effective_clusters):
        center = kmeans.cluster_centers_[topic_index]
        top_indices = center.argsort()[-top_terms:][::-1]
        keywords = ", ".join(feature_names[index] for index in top_indices)
        topic_name = f"Topic {topic_index + 1}"
        topic_size = int((model_df["topic"] == topic_name).sum())
        topic_rows.append({"topic": topic_name, "size": topic_size, "keywords": keywords})

    summary_df = pd.DataFrame(topic_rows).sort_values("size", ascending=False, ignore_index=True)
    return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)
