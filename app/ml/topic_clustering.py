from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.news_analytics import ALL_STOPWORDS, build_analysis_text


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
    model_df["topic"] = "\u041e\u0431\u0449\u0430\u044f \u0442\u0435\u043c\u0430"

    text_series = build_analysis_text(model_df)

    non_empty_mask = text_series.str.len() > 0
    usable_count = int(non_empty_mask.sum())
    if usable_count < 2:
        summary_df = pd.DataFrame(
            [{"topic": "\u041e\u0431\u0449\u0430\u044f \u0442\u0435\u043c\u0430", "size": len(model_df), "keywords": "\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0442\u0435\u043a\u0441\u0442\u0430"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    effective_clusters = max(1, min(n_clusters, usable_count))
    if effective_clusters == 1:
        summary_df = pd.DataFrame(
            [{"topic": "\u041e\u0431\u0449\u0430\u044f \u0442\u0435\u043c\u0430", "size": len(model_df), "keywords": "\u043e\u0434\u043d\u0430 \u0433\u0440\u0443\u043f\u043f\u0430"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    usable_texts = text_series[non_empty_mask]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ALL_STOPWORDS),
        max_features=3000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(usable_texts)

    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        summary_df = pd.DataFrame(
            [{"topic": "\u041e\u0431\u0449\u0430\u044f \u0442\u0435\u043c\u0430", "size": len(model_df), "keywords": "\u043d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u043e\u0432"}]
        )
        return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)

    kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_rows: list[dict[str, object]] = []
    label_map: dict[int, str] = {}
    for topic_index in range(effective_clusters):
        center = kmeans.cluster_centers_[topic_index]
        top_indices = center.argsort()[-top_terms:][::-1]
        top_keywords_list = [feature_names[index] for index in top_indices]
        keywords = ", ".join(top_keywords_list)
        topic_name = _build_topic_name(top_keywords_list, topic_index)
        label_map[topic_index] = topic_name
        topic_size = int((labels == topic_index).sum())
        topic_rows.append({"topic": topic_name, "size": topic_size, "keywords": keywords})

    model_df.loc[non_empty_mask, "topic"] = [label_map[label] for label in labels]
    summary_df = pd.DataFrame(topic_rows).sort_values("size", ascending=False, ignore_index=True)
    return ClusteringResult(articles_df=model_df, topic_summary_df=summary_df)


def _build_topic_name(keywords: list[str], topic_index: int) -> str:
    cleaned_keywords = [keyword.replace("_", " ").strip() for keyword in keywords if keyword.strip()]
    if not cleaned_keywords:
        return f"\u0422\u0435\u043c\u0430 {topic_index + 1}"
    label_keywords = cleaned_keywords[:3]
    return " / ".join(label_keywords)
