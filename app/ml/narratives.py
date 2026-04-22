from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import ALL_STOPWORDS, build_analysis_text, tokenize_for_analysis


NARRATIVE_PATTERNS: dict[str, dict[str, object]] = {
    "growth": {
        "label": "Сигнал роста",
        "terms": {
            "рост", "вырастет", "растет", "увеличится", "усилится", "ускорится",
            "подорожает", "поднимется", "расширится", "увеличение",
        },
    },
    "decline": {
        "label": "Сигнал снижения",
        "terms": {
            "снижение", "снизится", "упадет", "падение", "сократится", "замедлится",
            "ослабнет", "спад", "сокращение",
        },
    },
    "risk": {
        "label": "Сигнал риска",
        "terms": {
            "риск", "угроза", "кризис", "дефицит", "давление", "проблема", "удар",
            "санкции", "обвал", "нехватка", "потери", "ухудшение", "энергокризис",
        },
    },
    "replacement": {
        "label": "Сигнал замещения",
        "terms": {
            "заменит", "заменят", "вытеснит", "вытеснят", "автоматизация",
            "автоматизирует", "автоматизируют", "нейросети", "ии", "роботы",
        },
    },
    "regulation": {
        "label": "Сигнал регулирования",
        "terms": {
            "запрет", "запретят", "ограничат", "ужесточат", "регулирование",
            "контроль", "требования", "проверка", "надзор", "законопроект",
        },
    },
    "stability": {
        "label": "Сигнал стабилизации",
        "terms": {
            "стабилизация", "восстановление", "восстановится", "стабилизируется",
            "улучшение", "поддержка", "смягчение", "нормализация",
        },
    },
}

FUTURE_CUES = {
    "будет", "будут", "может", "могут", "ожидается", "ожидают", "прогноз",
    "прогнозируют", "спрогнозировали", "вероятно", "возможно", "грядет", "скоро",
}

NOISY_PREFIXES = ("фото:", "видео:", "репортаж:")
NOISY_MARKERS = {"риа", "тасс", "новости", "фото", "видео", "корреспондент", "фоторепортаж"}
GENERIC_FOCUS_TOKENS = ALL_STOPWORDS | FUTURE_CUES | {
    "рубль", "рубля", "рублей", "доллар", "доллара", "долларов", "евро",
    "процент", "процента", "процентов", "миллион", "миллиона", "миллионов",
    "миллиард", "миллиарда", "миллиардов", "тысяч", "тысяча", "тысячи",
    "цена", "цены", "курс", "мир", "страна", "стран", "рынок",
}


@dataclass(slots=True)
class NarrativeResult:
    summary_df: pd.DataFrame
    details_df: pd.DataFrame


def detect_narratives(articles_df: pd.DataFrame, top_n: int = 8) -> NarrativeResult:
    empty_summary = pd.DataFrame(
        columns=["narrative", "topic", "pattern", "strength", "momentum", "confidence", "articles", "keywords"]
    )
    empty_details = pd.DataFrame(columns=["narrative", "title", "published_at", "topic", "url", "overview"])

    required_columns = {"title", "overview", "text", "published_at", "url", "topic"}
    if articles_df.empty or not required_columns.issubset(set(articles_df.columns)):
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    work_df = articles_df.copy().reset_index(drop=True)
    work_df["analysis_text"] = build_analysis_text(work_df)
    work_df["published_at"] = pd.to_datetime(work_df["published_at"], errors="coerce")

    candidate_df = _extract_candidates(work_df)
    if candidate_df.empty:
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    grouped_df = _assign_narrative_groups(candidate_df)
    if grouped_df.empty:
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []

    for _, group_df in grouped_df.groupby("group_id"):
        group_df = group_df.sort_values(["score", "published_at"], ascending=[False, False], na_position="last")
        representative = group_df.iloc[0]

        recent_count, previous_count = _recent_vs_previous(group_df)
        momentum = recent_count - previous_count
        articles_count = len(group_df)
        avg_score = float(group_df["score"].mean())
        avg_similarity = float(group_df["similarity_to_anchor"].mean()) if "similarity_to_anchor" in group_df else 0.5
        confidence = round(min(1.0, 0.45 + avg_similarity / 2 + min(0.2, articles_count / 50)), 3)
        strength = round(avg_score * confidence * max(1.0, articles_count / 3), 2)

        keywords = _top_keywords_from_texts(group_df["candidate_text"].tolist(), top_n=4)
        topic_name = str(group_df["topic"].mode().iloc[0]) if not group_df["topic"].mode().empty else "Смешанная тема"
        narrative_name = _format_narrative_label(str(representative["candidate_text"]), str(representative["pattern_key"]))

        summary_rows.append(
            {
                "narrative": narrative_name,
                "topic": topic_name,
                "pattern": NARRATIVE_PATTERNS[str(representative["pattern_key"])]["label"],
                "strength": strength,
                "momentum": momentum,
                "confidence": confidence,
                "articles": articles_count,
                "keywords": ", ".join(keywords),
            }
        )

        detail_frame = group_df[["title", "published_at", "topic", "url", "overview"]].head(5).copy()
        detail_frame.insert(0, "narrative", narrative_name)
        detail_frames.append(detail_frame)

    if not summary_rows:
        return NarrativeResult(summary_df=empty_summary, details_df=empty_details)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["strength", "momentum", "articles"], ascending=[False, False, False], ignore_index=True)
    summary_df = summary_df.head(top_n)

    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else empty_details
    details_df = details_df[details_df["narrative"].isin(summary_df["narrative"])]

    return NarrativeResult(summary_df=summary_df, details_df=details_df)


def _extract_candidates(articles_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in articles_df.iterrows():
        best_candidate: dict[str, object] | None = None
        for source_text in (row.get("title"), row.get("overview")):
            for fragment in _split_candidate_fragments(source_text):
                cleaned = _clean_claim_text(fragment)
                if not cleaned or _looks_like_noise(cleaned.lower()):
                    continue

                tokens = tokenize_for_analysis(cleaned)
                if len(tokens) < 2:
                    continue

                pattern_counts = _count_patterns(tokens)
                pattern_key, pattern_score = _pick_dominant_pattern(pattern_counts)
                future_score = _count_future_cues(cleaned)
                score = pattern_score * 2 + future_score
                if pattern_key is None or score < 2:
                    continue

                candidate = {
                    "title": row.get("title"),
                    "overview": row.get("overview"),
                    "url": row.get("url"),
                    "topic": row.get("topic"),
                    "published_at": row.get("published_at"),
                    "candidate_text": cleaned,
                    "pattern_key": pattern_key,
                    "pattern_score": pattern_score,
                    "future_score": future_score,
                    "score": score,
                    "tokens": tokens,
                }
                if best_candidate is None or score > float(best_candidate["score"]):
                    best_candidate = candidate

        if best_candidate is not None:
            rows.append(best_candidate)

    return pd.DataFrame(rows)


def _assign_narrative_groups(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    grouped_parts: list[pd.DataFrame] = []
    next_group_id = 0

    for pattern_key, pattern_df in candidate_df.groupby("pattern_key"):
        if len(pattern_df) == 1:
            single_df = pattern_df.copy()
            single_df["group_id"] = next_group_id
            single_df["similarity_to_anchor"] = 1.0
            grouped_parts.append(single_df)
            next_group_id += 1
            continue

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(GENERIC_FOCUS_TOKENS),
            ngram_range=(1, 2),
            min_df=1,
            max_features=1500,
        )
        matrix = vectorizer.fit_transform(pattern_df["candidate_text"].tolist())
        similarity = cosine_similarity(matrix)

        pattern_df = pattern_df.reset_index(drop=True).copy()
        assigned = [-1] * len(pattern_df)

        for anchor_idx in pattern_df.sort_values(["score", "published_at"], ascending=[False, False]).index.tolist():
            if assigned[anchor_idx] != -1:
                continue

            assigned[anchor_idx] = next_group_id
            anchor_tokens = set(pattern_df.loc[anchor_idx, "tokens"])

            for idx in range(len(pattern_df)):
                if assigned[idx] != -1:
                    continue

                token_overlap = _token_overlap(anchor_tokens, set(pattern_df.loc[idx, "tokens"]))
                sim_score = float(similarity[anchor_idx, idx])
                if sim_score >= 0.34 or (sim_score >= 0.22 and token_overlap >= 0.34):
                    assigned[idx] = next_group_id

            next_group_id += 1

        pattern_df["group_id"] = assigned
        pattern_df["similarity_to_anchor"] = 0.0

        for group_id, group_df in pattern_df.groupby("group_id"):
            anchor_idx = group_df.sort_values(["score", "published_at"], ascending=[False, False]).index[0]
            pattern_df.loc[group_df.index, "similarity_to_anchor"] = [float(similarity[anchor_idx, idx]) for idx in group_df.index]

        grouped_parts.append(pattern_df)

    return pd.concat(grouped_parts, ignore_index=True)


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


def _recent_vs_previous(group_df: pd.DataFrame) -> tuple[int, int]:
    dated_df = group_df.dropna(subset=["published_at"]).copy()
    if dated_df.empty:
        return len(group_df), 0

    dated_df["date"] = dated_df["published_at"].dt.date
    max_date = dated_df["date"].max()
    recent_border = max_date - pd.Timedelta(days=6)
    previous_border = max_date - pd.Timedelta(days=13)

    recent_count = int((dated_df["date"] >= recent_border).sum())
    previous_count = int(((dated_df["date"] >= previous_border) & (dated_df["date"] < recent_border)).sum())
    return recent_count, previous_count


def _split_candidate_fragments(text: object) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    for prefix in NOISY_PREFIXES:
        if lowered.startswith(prefix):
            normalized = re.sub(r"^[^.!?]+[.!?]?\s*", "", normalized, count=1)
            break

    fragments = re.split(r"[.!?;]\s+", normalized)
    return [fragment.strip(" -,:;\"'«»") for fragment in fragments if fragment.strip()]


def _clean_claim_text(text: object) -> str | None:
    if not isinstance(text, str):
        return None

    cleaned = " ".join(text.strip().split())
    cleaned = re.sub(r"^\s*(?:фото|video|video:|photo)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*/\s*(?:риа(?:\s+новости)?|тасс)\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,:;")
    if not cleaned or len(cleaned.split()) < 3:
        return None
    if len(cleaned) > 140:
        cleaned = cleaned[:137].rsplit(" ", 1)[0] + "..."
    return cleaned[:1].upper() + cleaned[1:]


def _looks_like_noise(text: str) -> bool:
    if not text:
        return True

    marker_hits = sum(1 for marker in NOISY_MARKERS if marker in text)
    if marker_hits >= 2:
        return True

    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    meaningful_words = [word for word in words if len(word) >= 4 and word.lower() not in GENERIC_FOCUS_TOKENS]
    return len(meaningful_words) < 2


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _top_keywords_from_texts(texts: list[str], top_n: int = 4) -> list[str]:
    tokens = []
    for text in texts:
        tokens.extend(tokenize_for_analysis(text))
    if not tokens:
        return []
    counts = pd.Series(tokens).value_counts()
    return counts.head(top_n).index.tolist()


def _format_narrative_label(text: str, pattern_key: str) -> str:
    cleaned = _clean_claim_text(text)
    if cleaned and not _looks_like_bad_label(cleaned, pattern_key):
        return cleaned

    focus = _extract_focus_from_text(cleaned or text, pattern_key)
    if focus:
        if pattern_key == "growth":
            return f"Ожидается рост {focus}"
        if pattern_key == "decline":
            return f"Продолжается снижение {focus}"
        if pattern_key == "risk":
            return f"Формируется риск {focus}"
        if pattern_key == "replacement":
            return f"Формируется замещение {focus}"
        if pattern_key == "regulation":
            return f"Ожидается усиление регулирования {focus}"
        return f"Формируется стабилизация {focus}"

    if pattern_key == "growth":
        return "Ожидается рост в этой теме"
    if pattern_key == "decline":
        return "Ожидается снижение в этой теме"
    if pattern_key == "risk":
        return "Формируется риск в этой теме"
    if pattern_key == "replacement":
        return "Формируется мнение о замещении в этой теме"
    if pattern_key == "regulation":
        return "Ожидается усиление регулирования в этой теме"
    return "Формируется ожидание стабилизации в этой теме"


def _looks_like_bad_label(text: str, pattern_key: str) -> bool:
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered, flags=re.UNICODE)
    if len(words) < 3:
        return True

    if text.endswith("..."):
        return True

    broken_starts = {"российских", "мировым", "мировых", "крупнейших", "ведущих", "банках", "странах", "компаниях"}
    if words[0] in broken_starts:
        return True

    adjective_like_endings = ("ых", "их", "ого", "ему", "ому", "ими", "ыми", "ой", "ей")
    if words[0].endswith(adjective_like_endings) and words[0] not in {"рост", "снижение", "падение", "угроза", "риск"}:
        return True

    pattern_terms = NARRATIVE_PATTERNS.get(pattern_key, {}).get("terms", set())
    if not any(term in lowered for term in FUTURE_CUES) and not any(term in lowered for term in pattern_terms):
        return True

    return False


def _extract_focus_from_text(text: str, pattern_key: str) -> str | None:
    if not isinstance(text, str):
        return None

    pattern_terms = set(NARRATIVE_PATTERNS.get(pattern_key, {}).get("terms", set()))
    tokens = [
        token
        for token in re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 4 and token not in GENERIC_FOCUS_TOKENS and token not in pattern_terms
    ]
    if not tokens:
        return None

    unique_tokens: list[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    return " ".join(unique_tokens[:3])
