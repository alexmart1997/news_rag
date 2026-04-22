from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import RUSSIAN_STOPWORDS


@dataclass(slots=True)
class QAResult:
    answer: str
    hits_df: pd.DataFrame


def answer_question(
    articles_df: pd.DataFrame,
    question: str,
    top_k: int = 3,
) -> QAResult:
    empty_hits = pd.DataFrame(columns=["id", "title", "topic", "published_at", "score", "url", "snippet"])
    if articles_df.empty or not question.strip():
        return QAResult(answer="Enter a question to search the news collection.", hits_df=empty_hits)

    model_df = articles_df.reset_index(drop=True).copy()
    text_series = (
        model_df["title"].fillna("")
        + " "
        + model_df["overview"].fillna("")
        + " "
        + model_df["text"].fillna("")
    ).str.strip()

    if (text_series.str.len() > 0).sum() < 1:
        return QAResult(answer="Not enough text data to answer the question.", hits_df=empty_hits)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(RUSSIAN_STOPWORDS),
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(text_series.tolist() + [question.strip()])
    if matrix.shape[1] == 0:
        return QAResult(answer="Could not extract enough meaningful terms from the question.", hits_df=empty_hits)

    article_matrix = matrix[:-1]
    question_vector = matrix[-1]
    scores = cosine_similarity(question_vector, article_matrix).flatten()
    model_df["score"] = scores
    model_df = model_df.sort_values("score", ascending=False, ignore_index=True)

    hits_df = model_df.head(top_k).copy()
    if hits_df.empty or float(hits_df.iloc[0]["score"]) <= 0:
        return QAResult(answer="I could not find relevant news for this question in the current filtered dataset.", hits_df=empty_hits)

    hits_df["score"] = hits_df["score"].round(3)
    hits_df["snippet"] = hits_df.apply(_build_snippet, axis=1)

    answer_lines = []
    for _, row in hits_df.iterrows():
        summary = row["overview"] or row["snippet"] or row["title"]
        answer_lines.append(f"- {row['title']}: {summary}")

    answer = "Top relevant news for your question:\n" + "\n".join(answer_lines)
    return QAResult(
        answer=answer,
        hits_df=hits_df[["id", "title", "topic", "published_at", "score", "url", "snippet"]],
    )


def _build_snippet(row: pd.Series, max_sentences: int = 2) -> str:
    text = row.get("text") or row.get("overview") or row.get("title") or ""
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not cleaned:
        return ""
    return " ".join(cleaned[:max_sentences])[:400]
