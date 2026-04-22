from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from app.ml.news_analytics import ALL_STOPWORDS, build_analysis_text, tokenize_for_analysis


NARRATIVE_PATTERNS: dict[str, dict[str, object]] = {
    "growth": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0440\u043e\u0441\u0442\u0430",
        "template": "\u041e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f \u0440\u043e\u0441\u0442: {focus}",
        "terms": {
            "\u0440\u043e\u0441\u0442", "\u0432\u044b\u0440\u0430\u0441\u0442\u0435\u0442", "\u0440\u0430\u0441\u0442\u0435\u0442", "\u0443\u0432\u0435\u043b\u0438\u0447\u0438\u0442\u0441\u044f", "\u0443\u0441\u0438\u043b\u0438\u0442\u0441\u044f", "\u0443\u0441\u043a\u043e\u0440\u0438\u0442\u0441\u044f",
            "\u043f\u043e\u0434\u043e\u0440\u043e\u0436\u0430\u0435\u0442", "\u043f\u043e\u0434\u043d\u0438\u043c\u0435\u0442\u0441\u044f", "\u0440\u0430\u0441\u0448\u0438\u0440\u0438\u0442\u0441\u044f", "\u0443\u0432\u0435\u043b\u0438\u0447\u0435\u043d\u0438\u0435",
        },
    },
    "decline": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0441\u043d\u0438\u0436\u0435\u043d\u0438\u044f",
        "template": "\u041e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f \u0441\u043d\u0438\u0436\u0435\u043d\u0438\u0435: {focus}",
        "terms": {
            "\u0441\u043d\u0438\u0436\u0435\u043d\u0438\u0435", "\u0441\u043d\u0438\u0437\u0438\u0442\u0441\u044f", "\u0443\u043f\u0430\u0434\u0435\u0442", "\u043f\u0430\u0434\u0435\u043d\u0438\u0435", "\u0441\u043e\u043a\u0440\u0430\u0442\u0438\u0442\u0441\u044f", "\u0437\u0430\u043c\u0435\u0434\u043b\u0438\u0442\u0441\u044f",
            "\u043e\u0441\u043b\u0430\u0431\u043d\u0435\u0442", "\u0441\u043f\u0430\u0434", "\u0434\u0435\u0448\u0435\u0432\u0435\u0435\u0442", "\u0441\u043e\u043a\u0440\u0430\u0449\u0435\u043d\u0438\u0435",
        },
    },
    "risk": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0440\u0438\u0441\u043a\u0430",
        "template": "\u0424\u043e\u0440\u043c\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u0440\u0438\u0441\u043a: {focus}",
        "terms": {
            "\u0440\u0438\u0441\u043a", "\u0443\u0433\u0440\u043e\u0437\u0430", "\u043a\u0440\u0438\u0437\u0438\u0441", "\u0434\u0435\u0444\u0438\u0446\u0438\u0442", "\u0434\u0430\u0432\u043b\u0435\u043d\u0438\u0435", "\u043f\u0440\u043e\u0431\u043b\u0435\u043c\u0430", "\u0443\u0434\u0430\u0440",
            "\u0441\u0430\u043d\u043a\u0446\u0438\u0438", "\u043e\u0431\u0432\u0430\u043b", "\u043d\u0435\u0445\u0432\u0430\u0442\u043a\u0430", "\u043f\u043e\u0442\u0435\u0440\u0438", "\u0443\u0445\u0443\u0434\u0448\u0435\u043d\u0438\u0435",
        },
    },
    "replacement": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0437\u0430\u043c\u0435\u0449\u0435\u043d\u0438\u044f",
        "template": "\u0424\u043e\u0440\u043c\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u043c\u043d\u0435\u043d\u0438\u0435 \u043e \u0437\u0430\u043c\u0435\u0449\u0435\u043d\u0438\u0438: {focus}",
        "terms": {
            "\u0437\u0430\u043c\u0435\u043d\u0438\u0442", "\u0437\u0430\u043c\u0435\u043d\u044f\u0442", "\u0432\u044b\u0442\u0435\u0441\u043d\u0438\u0442", "\u0432\u044b\u0442\u0435\u0441\u043d\u044f\u0442", "\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0437\u0430\u0446\u0438\u044f",
            "\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0437\u0438\u0440\u0443\u0435\u0442", "\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0437\u0438\u0440\u0443\u044e\u0442", "\u043d\u0435\u0439\u0440\u043e\u0441\u0435\u0442\u0438", "\u0438\u0438", "\u0440\u043e\u0431\u043e\u0442\u044b",
        },
    },
    "regulation": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f",
        "template": "\u041e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f \u0443\u0441\u0438\u043b\u0435\u043d\u0438\u0435 \u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f: {focus}",
        "terms": {
            "\u0437\u0430\u043f\u0440\u0435\u0442", "\u0437\u0430\u043f\u0440\u0435\u0442\u044f\u0442", "\u043e\u0433\u0440\u0430\u043d\u0438\u0447\u0430\u0442", "\u0443\u0436\u0435\u0441\u0442\u043e\u0447\u0430\u0442", "\u0440\u0435\u0433\u0443\u043b\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435",
            "\u043a\u043e\u043d\u0442\u0440\u043e\u043b\u044c", "\u0442\u0440\u0435\u0431\u043e\u0432\u0430\u043d\u0438\u044f", "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430", "\u043d\u0430\u0434\u0437\u043e\u0440", "\u0437\u0430\u043a\u043e\u043d\u043e\u043f\u0440\u043e\u0435\u043a\u0442",
        },
    },
    "stability": {
        "label": "\u0421\u0438\u0433\u043d\u0430\u043b \u0441\u0442\u0430\u0431\u0438\u043b\u0438\u0437\u0430\u0446\u0438\u0438",
        "template": "\u0424\u043e\u0440\u043c\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u043e\u0436\u0438\u0434\u0430\u043d\u0438\u0435 \u0441\u0442\u0430\u0431\u0438\u043b\u0438\u0437\u0430\u0446\u0438\u0438: {focus}",
        "terms": {
            "\u0441\u0442\u0430\u0431\u0438\u043b\u0438\u0437\u0430\u0446\u0438\u044f", "\u0432\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435", "\u0432\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0442\u0441\u044f", "\u0441\u0442\u0430\u0431\u0438\u043b\u0438\u0437\u0438\u0440\u0443\u0435\u0442\u0441\u044f",
            "\u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u0435", "\u043f\u043e\u0434\u0434\u0435\u0440\u0436\u043a\u0430", "\u0441\u043c\u044f\u0433\u0447\u0435\u043d\u0438\u0435", "\u043d\u043e\u0440\u043c\u0430\u043b\u0438\u0437\u0430\u0446\u0438\u044f",
        },
    },
}

FUTURE_CUES = {
    "\u0431\u0443\u0434\u0435\u0442", "\u0431\u0443\u0434\u0443\u0442", "\u043c\u043e\u0436\u0435\u0442", "\u043c\u043e\u0433\u0443\u0442", "\u043e\u0436\u0438\u0434\u0430\u0435\u0442\u0441\u044f", "\u043e\u0436\u0438\u0434\u0430\u044e\u0442", "\u043f\u0440\u043e\u0433\u043d\u043e\u0437",
    "\u043f\u0440\u043e\u0433\u043d\u043e\u0437\u0438\u0440\u0443\u044e\u0442", "\u0432\u0435\u0440\u043e\u044f\u0442\u043d\u043e", "\u0432\u043e\u0437\u043c\u043e\u0436\u043d\u043e", "\u0433\u0440\u044f\u0434\u0435\u0442", "\u0441\u043a\u043e\u0440\u043e",
}

GENERIC_FOCUS_TOKENS = ALL_STOPWORDS | FUTURE_CUES | {
    "\u0440\u0443\u0431\u043b\u044c",
    "\u0440\u0443\u0431\u043b\u044f",
    "\u0440\u0443\u0431\u043b\u0435\u0439",
    "\u0434\u043e\u043b\u043b\u0430\u0440",
    "\u0434\u043e\u043b\u043b\u0430\u0440\u0430",
    "\u0434\u043e\u043b\u043b\u0430\u0440\u043e\u0432",
    "\u0435\u0432\u0440\u043e",
    "\u043f\u0440\u043e\u0446\u0435\u043d\u0442",
    "\u043f\u0440\u043e\u0446\u0435\u043d\u0442\u0430",
    "\u043f\u0440\u043e\u0446\u0435\u043d\u0442\u043e\u0432",
    "\u043c\u0438\u043b\u043b\u0438\u043e\u043d",
    "\u043c\u0438\u043b\u043b\u0438\u043e\u043d\u0430",
    "\u043c\u0438\u043b\u043b\u0438\u043e\u043d\u043e\u0432",
    "\u043c\u0438\u043b\u043b\u0438\u0430\u0440\u0434",
    "\u043c\u0438\u043b\u043b\u0438\u0430\u0440\u0434\u0430",
    "\u043c\u0438\u043b\u043b\u0438\u0430\u0440\u0434\u043e\u0432",
    "\u0442\u044b\u0441\u044f\u0447",
    "\u0442\u044b\u0441\u044f\u0447\u0430",
    "\u0442\u044b\u0441\u044f\u0447\u0438",
    "\u0446\u0435\u043d\u0430",
    "\u0446\u0435\u043d\u044b",
    "\u043a\u0443\u0440\u0441",
}


@dataclass(slots=True)
class NarrativeResult:
    summary_df: pd.DataFrame
    details_df: pd.DataFrame


def detect_narratives(articles_df: pd.DataFrame, top_n: int = 8) -> NarrativeResult:
    empty_summary = pd.DataFrame(
        columns=[
            "narrative",
            "topic",
            "pattern",
            "strength",
            "momentum",
            "confidence",
            "articles",
            "keywords",
        ]
    )
    empty_details = pd.DataFrame(
        columns=["narrative", "title", "published_at", "topic", "url", "overview"]
    )

    required_columns = {"topic", "title", "overview", "text", "published_at", "url"}
    if articles_df.empty or not required_columns.issubset(set(articles_df.columns)):
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    work_df = articles_df.copy().reset_index(drop=True)
    work_df["analysis_text"] = build_analysis_text(work_df)
    work_df["date"] = pd.to_datetime(work_df["published_at"], errors="coerce").dt.date
    work_df["tokens"] = work_df["analysis_text"].apply(tokenize_for_analysis)
    work_df["future_cues"] = work_df["analysis_text"].str.lower().apply(_count_future_cues)

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for topic_name, topic_df in work_df.groupby("topic"):
        topic_tokens = [token for tokens in topic_df["tokens"] for token in tokens]
        if not topic_tokens:
            continue

        pattern_counts = _count_patterns(topic_tokens)
        dominant_pattern, pattern_score = _pick_dominant_pattern(pattern_counts)
        if dominant_pattern is None:
            continue

        recent_count, previous_count = _recent_vs_previous(topic_df)
        momentum = recent_count - previous_count
        articles_count = len(topic_df)
        future_signal = int(topic_df["future_cues"].sum())
        confidence = round(min(1.0, (pattern_score + future_signal) / max(3, articles_count)), 3)

        top_keywords = _top_keywords_from_tokens(topic_tokens, top_n=4)
        narrative_name = _build_narrative_name(
            topic_df=topic_df,
            pattern_key=dominant_pattern,
            keywords=top_keywords,
        )

        strength = round(pattern_score * confidence * max(1.0, articles_count / 8), 2)

        summary_rows.append(
            {
                "narrative": narrative_name,
                "topic": topic_name,
                "pattern": NARRATIVE_PATTERNS[dominant_pattern]["label"],
                "strength": strength,
                "momentum": momentum,
                "confidence": confidence,
                "articles": articles_count,
                "keywords": ", ".join(top_keywords),
            }
        )

        top_examples = topic_df.sort_values(
            ["future_cues", "published_at"],
            ascending=[False, False],
            na_position="last",
        ).head(3)
        if not top_examples.empty:
            detail_frame = top_examples[["title", "published_at", "topic", "url", "overview"]].copy()
            detail_frame.insert(0, "narrative", narrative_name)
            detail_frames.append(detail_frame)

    if not summary_rows:
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["strength", "momentum", "articles"],
        ascending=[False, False, False],
        ignore_index=True,
    ).head(top_n)

    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else empty_details
    details_df = details_df[details_df["narrative"].isin(summary_df["narrative"])]

    return NarrativeResult(summary_df=summary_df, details_df=details_df)


def _count_future_cues(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in FUTURE_CUES)


def _count_patterns(tokens: list[str]) -> dict[str, int]:
    token_set = set(tokens)
    counts: dict[str, int] = {}
    for key, config in NARRATIVE_PATTERNS.items():
        terms = config["terms"]
        counts[key] = sum(1 for token in tokens if token in terms) + sum(1 for term in terms if term in token_set)
    return counts


def _pick_dominant_pattern(pattern_counts: dict[str, int]) -> tuple[str | None, int]:
    if not pattern_counts:
        return None, 0
    dominant_key = max(pattern_counts, key=pattern_counts.get)
    dominant_score = pattern_counts[dominant_key]
    if dominant_score <= 0:
        return None, 0
    return dominant_key, dominant_score


def _recent_vs_previous(topic_df: pd.DataFrame) -> tuple[int, int]:
    dated_df = topic_df.dropna(subset=["date"]).sort_values("date")
    if dated_df.empty:
        return len(topic_df), 0

    max_date = dated_df["date"].max()
    recent_border = max_date - pd.Timedelta(days=6)
    previous_border = max_date - pd.Timedelta(days=13)

    recent_count = int((dated_df["date"] >= recent_border).sum())
    previous_count = int(((dated_df["date"] >= previous_border) & (dated_df["date"] < recent_border)).sum())
    return recent_count, previous_count


def _top_keywords_from_tokens(tokens: list[str], top_n: int = 4) -> list[str]:
    counts = pd.Series(tokens).value_counts()
    return counts.head(top_n).index.tolist()


def _build_narrative_name(topic_df: pd.DataFrame, pattern_key: str, keywords: list[str]) -> str:
    claim = _extract_claim_from_examples(topic_df, pattern_key)
    if claim:
        return claim

    focus = _build_focus_phrase(topic_df, keywords)
    if pattern_key in NARRATIVE_PATTERNS:
        template = str(NARRATIVE_PATTERNS[pattern_key]["template"])
        return template.format(focus=focus)
    return focus


def _extract_claim_from_examples(topic_df: pd.DataFrame, pattern_key: str) -> str | None:
    pattern_terms = NARRATIVE_PATTERNS.get(pattern_key, {}).get("terms", set())
    if not pattern_terms:
        return None

    ranked_df = topic_df.sort_values(
        ["future_cues", "published_at"],
        ascending=[False, False],
        na_position="last",
    )
    for _, row in ranked_df.head(5).iterrows():
        candidate_parts = [row.get("title"), row.get("overview")]
        for part in candidate_parts:
            cleaned = _clean_claim_text(part)
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if any(term in lowered for term in FUTURE_CUES) or any(term in lowered for term in pattern_terms):
                return cleaned
    return None


def _build_focus_phrase(topic_df: pd.DataFrame, keywords: list[str]) -> str:
    phrase = _extract_focus_ngram(topic_df)
    if phrase:
        return phrase

    filtered_keywords = [keyword for keyword in keywords if keyword not in GENERIC_FOCUS_TOKENS]
    if filtered_keywords:
        return ", ".join(filtered_keywords[:3])
    return "\u0432 \u044d\u0442\u043e\u0439 \u0442\u0435\u043c\u0435"


def _clean_claim_text(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.strip(" -,:;")
    if not cleaned:
        return None
    if len(cleaned) > 140:
        cleaned = cleaned[:137].rsplit(" ", 1)[0] + "..."
    return cleaned[:1].upper() + cleaned[1:]


def _extract_focus_ngram(topic_df: pd.DataFrame) -> str | None:
    text_parts = []
    ranked_df = topic_df.sort_values(
        ["future_cues", "published_at"],
        ascending=[False, False],
        na_position="last",
    )
    for _, row in ranked_df.head(12).iterrows():
        for column in ("title", "overview"):
            value = row.get(column)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.lower())

    if not text_parts:
        return None

    phrase_counts: dict[str, int] = {}
    for text in text_parts:
        tokens = [
            token
            for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
            if len(token) >= 4 and token not in GENERIC_FOCUS_TOKENS
        ]
        for size in (3, 2):
            for index in range(len(tokens) - size + 1):
                gram_tokens = tokens[index:index + size]
                if len(set(gram_tokens)) < len(gram_tokens):
                    continue
                phrase = " ".join(gram_tokens)
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    if not phrase_counts:
        token_counts = pd.Series(
            [
                token
                for text in text_parts
                for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
                if len(token) >= 4 and token not in GENERIC_FOCUS_TOKENS
            ]
        ).value_counts()
        if token_counts.empty:
            return None
        return ", ".join(token_counts.head(3).index.tolist())

    best_phrase = sorted(
        phrase_counts.items(),
        key=lambda item: (item[1], len(item[0].split()), len(item[0])),
        reverse=True,
    )[0][0]
    return best_phrase
