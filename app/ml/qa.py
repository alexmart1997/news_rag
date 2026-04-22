from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import ALL_STOPWORDS, build_analysis_text


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
        return QAResult(answer="\u0412\u0432\u0435\u0434\u0438\u0442\u0435 \u0432\u043e\u043f\u0440\u043e\u0441, \u0447\u0442\u043e\u0431\u044b \u0438\u0441\u043a\u0430\u0442\u044c \u043f\u043e \u043d\u043e\u0432\u043e\u0441\u0442\u044f\u043c.", hits_df=empty_hits)

    model_df = articles_df.reset_index(drop=True).copy()
    text_series = build_analysis_text(model_df)

    if (text_series.str.len() > 0).sum() < 1:
        return QAResult(answer="\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0442\u0435\u043a\u0441\u0442\u043e\u0432, \u0447\u0442\u043e\u0431\u044b \u043e\u0442\u0432\u0435\u0442\u0438\u0442\u044c \u043d\u0430 \u0432\u043e\u043f\u0440\u043e\u0441.", hits_df=empty_hits)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ALL_STOPWORDS),
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(text_series.tolist() + [question.strip()])
    if matrix.shape[1] == 0:
        return QAResult(answer="\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0432\u044b\u0434\u0435\u043b\u0438\u0442\u044c \u0438\u0437 \u0432\u043e\u043f\u0440\u043e\u0441\u0430 \u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0441\u043c\u044b\u0441\u043b\u043e\u0432\u044b\u0445 \u0442\u0435\u0440\u043c\u0438\u043d\u043e\u0432.", hits_df=empty_hits)

    article_matrix = matrix[:-1]
    question_vector = matrix[-1]
    scores = cosine_similarity(question_vector, article_matrix).flatten()
    model_df["score"] = scores
    model_df = model_df.sort_values("score", ascending=False, ignore_index=True)

    hits_df = model_df.head(top_k).copy()
    if hits_df.empty or float(hits_df.iloc[0]["score"]) <= 0:
        return QAResult(answer="\u041f\u043e \u0442\u0435\u043a\u0443\u0449\u0435\u0439 \u0432\u044b\u0431\u043e\u0440\u043a\u0435 \u043d\u0435 \u043d\u0430\u0448\u043b\u043e\u0441\u044c \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u044b\u0445 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439.", hits_df=empty_hits)

    hits_df["score"] = hits_df["score"].round(3)
    hits_df["snippet"] = hits_df.apply(_build_snippet, axis=1)

    answer_lines = []
    for _, row in hits_df.iterrows():
        summary = row["overview"] or row["snippet"] or row["title"]
        answer_lines.append(f"- {row['title']}: {summary}")

    answer = "\u041d\u0430\u0438\u0431\u043e\u043b\u0435\u0435 \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u044b\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u043f\u043e \u0432\u0430\u0448\u0435\u043c\u0443 \u0432\u043e\u043f\u0440\u043e\u0441\u0443:\n" + "\n".join(answer_lines)
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
