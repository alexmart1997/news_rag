from __future__ import annotations

from collections import Counter
import re

import pandas as pd


RUSSIAN_STOPWORDS = {
    "\u0438",
    "\u0432",
    "\u0432\u043e",
    "\u043d\u0430",
    "\u043d\u043e",
    "\u043f\u043e",
    "\u0441",
    "\u0441\u043e",
    "\u0434\u043b\u044f",
    "\u0447\u0442\u043e",
    "\u044d\u0442\u043e",
    "\u043a\u0430\u043a",
    "\u0442\u0430\u043a",
    "\u0442\u043e\u0436\u0435",
    "\u043f\u043e\u0441\u043b\u0435",
    "\u043f\u0440\u0438",
    "\u0438\u0437",
    "\u043e\u0442",
    "\u0434\u043e",
    "\u043e\u0431",
    "\u043e",
    "\u0443",
    "\u0430",
    "\u0438\u043b\u0438",
    "\u0436\u0435",
    "\u043b\u0438",
    "\u0431\u044b",
    "\u0431\u044b\u043b",
    "\u0431\u044b\u043b\u0430",
    "\u0431\u044b\u043b\u0438",
    "\u0431\u0443\u0434\u0435\u0442",
    "\u0431\u0443\u0434\u0443\u0442",
    "\u044d\u0442\u0438",
    "\u044d\u0442\u043e\u0442",
    "\u044d\u0442\u043e\u0439",
    "\u0442\u043e\u043b\u044c\u043a\u043e",
    "\u0443\u0436\u0435",
    "\u0435\u0449\u0435",
    "\u043c\u0435\u0436\u0434\u0443",
    "\u0431\u0435\u0437",
    "\u0438\u0445",
    "\u0435\u0433\u043e",
    "\u0435\u0435",
    "\u0438\u043c",
    "\u0438\u043c\u0438",
    "\u043c\u044b",
    "\u043e\u043d",
    "\u043e\u043d\u0430",
    "\u043e\u043d\u0438",
    "\u0432\u0441\u0435",
    "\u0432\u0441\u0435\u0445",
}


def build_daily_counts(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty or "published_at" not in articles_df.columns:
        return pd.DataFrame(columns=["date", "articles"])

    daily_df = articles_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["published_at"], errors="coerce").dt.date
    daily_df = daily_df.dropna(subset=["date"])
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "articles"])

    return (
        daily_df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "articles"})
        .sort_values("date", ignore_index=True)
    )


def build_top_keywords(
    articles_df: pd.DataFrame,
    top_n: int = 15,
    min_word_length: int = 4,
) -> pd.DataFrame:
    if articles_df.empty:
        return pd.DataFrame(columns=["keyword", "count"])

    text_series = articles_df["title"].fillna("") + " " + articles_df["overview"].fillna("")
    combined_text = " ".join(text_series.tolist()).lower()
    tokens = re.findall(r"[^\W\d_]+", combined_text, flags=re.UNICODE)
    filtered_tokens = [
        token
        for token in tokens
        if len(token) >= min_word_length and token not in RUSSIAN_STOPWORDS
    ]
    top_keywords = Counter(filtered_tokens).most_common(top_n)
    return pd.DataFrame(top_keywords, columns=["keyword", "count"])


def build_topic_dynamics(articles_df: pd.DataFrame) -> pd.DataFrame:
    if articles_df.empty or "published_at" not in articles_df.columns or "topic" not in articles_df.columns:
        return pd.DataFrame()

    dynamics_df = articles_df.copy()
    dynamics_df["date"] = pd.to_datetime(dynamics_df["published_at"], errors="coerce").dt.date
    dynamics_df = dynamics_df.dropna(subset=["date", "topic"])
    if dynamics_df.empty:
        return pd.DataFrame()

    pivot_df = (
        dynamics_df.groupby(["date", "topic"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return pivot_df
