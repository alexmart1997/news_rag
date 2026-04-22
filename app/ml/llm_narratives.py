from __future__ import annotations

from dataclasses import dataclass
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ml.news_analytics import ALL_STOPWORDS, tokenize_for_analysis
from app.ml.ollama_client import generate_text, is_ollama_available


@dataclass(slots=True)
class NarrativeResult:
    summary_df: pd.DataFrame
    details_df: pd.DataFrame
    signals_df: pd.DataFrame


def detect_llm_narratives(
    articles_df: pd.DataFrame,
    top_n: int = 8,
    batch_size: int = 30,
    max_articles: int = 240,
    min_coverage: int = 3,
) -> NarrativeResult:
    empty_summary = pd.DataFrame(
        columns=["narrative", "topic", "pattern", "coverage", "share", "weekly_change", "cohesion", "kind", "keywords"]
    )
    empty_details = pd.DataFrame(columns=["narrative", "title", "published_at", "topic", "url", "overview"])
    empty_signals = pd.DataFrame(columns=["signal", "topic", "pattern", "published_at", "keywords", "url"])

    required_columns = {"id", "title", "overview", "published_at", "url", "topic"}
    if articles_df.empty or not required_columns.issubset(set(articles_df.columns)) or not is_ollama_available():
        return NarrativeResult(empty_summary, empty_details, empty_signals)

    corpus_df = (
        articles_df.copy()
        .dropna(subset=["id", "title"])
        .sort_values("published_at", ascending=False, na_position="last")
        .head(max_articles)
        .reset_index(drop=True)
    )
    if len(corpus_df) < 10:
        return NarrativeResult(empty_summary, empty_details, empty_signals)

    corpus_df["published_at"] = pd.to_datetime(corpus_df["published_at"], errors="coerce")
    corpus_df["date"] = corpus_df["published_at"].dt.date.astype("string").fillna("")
    lookup = corpus_df.set_index("id").to_dict("index")

    extracted_rows: list[dict[str, object]] = []
    for start in range(0, len(corpus_df), batch_size):
        batch_df = corpus_df.iloc[start:start + batch_size].copy()
        batch_rows = _extract_batch_narratives(batch_df)
        extracted_rows.extend(batch_rows)

    if not extracted_rows:
        return NarrativeResult(empty_summary, empty_details, empty_signals)

    extracted_df = pd.DataFrame(extracted_rows)
    merged_df = _merge_narratives(extracted_df)
    if merged_df.empty:
        return NarrativeResult(empty_summary, empty_details, empty_signals)

    summary_rows: list[dict[str, object]] = []
    details_frames: list[pd.DataFrame] = []
    signal_rows: list[dict[str, object]] = []
    total_articles = max(1, len(corpus_df))

    for _, row in merged_df.iterrows():
        article_ids = [article_id for article_id in row["article_ids"] if article_id in lookup]
        if not article_ids:
            continue

        evidence_df = pd.DataFrame([{"id": aid, **lookup[aid]} for aid in article_ids])
        evidence_df = evidence_df.sort_values("published_at", ascending=False, na_position="last")

        coverage = len(article_ids)
        share = round(coverage / total_articles, 3)
        weekly_change = _weekly_change(evidence_df)
        cohesion = round(float(row["cohesion"]), 3)
        keywords = _keywords_from_rows(evidence_df)
        topic_name = _dominant_topic(evidence_df)
        pattern = str(row["pattern"])
        label = str(row["narrative"])

        if coverage >= min_coverage:
            summary_rows.append(
                {
                    "narrative": label,
                    "topic": topic_name,
                    "pattern": pattern,
                    "coverage": coverage,
                    "share": share,
                    "weekly_change": weekly_change,
                    "cohesion": cohesion,
                    "kind": "narrative",
                    "keywords": keywords,
                }
            )
            detail_df = evidence_df[["title", "published_at", "topic", "url", "overview"]].head(5).copy()
            detail_df.insert(0, "narrative", label)
            details_frames.append(detail_df)
        else:
            top_row = evidence_df.iloc[0]
            signal_rows.append(
                {
                    "signal": label,
                    "topic": topic_name,
                    "pattern": pattern,
                    "published_at": top_row.get("published_at"),
                    "keywords": keywords,
                    "url": top_row.get("url"),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["coverage", "weekly_change", "cohesion"], ascending=[False, False, False], ignore_index=True).head(top_n)

    details_df = pd.concat(details_frames, ignore_index=True) if details_frames else empty_details
    if not details_df.empty and not summary_df.empty:
        details_df = details_df[details_df["narrative"].isin(summary_df["narrative"])]

    signals_df = pd.DataFrame(signal_rows)
    if not signals_df.empty:
        signals_df = signals_df.sort_values("published_at", ascending=False, ignore_index=True).head(top_n)

    return NarrativeResult(
        summary_df=summary_df if not summary_df.empty else empty_summary,
        details_df=details_df if not details_df.empty else empty_details,
        signals_df=signals_df if not signals_df.empty else empty_signals,
    )


def _extract_batch_narratives(batch_df: pd.DataFrame) -> list[dict[str, object]]:
    lines = []
    valid_ids: set[int] = set()
    for row in batch_df.itertuples(index=False):
        valid_ids.add(int(row.id))
        overview = str(row.overview or "").replace("\n", " ").strip()
        if len(overview) > 180:
            overview = overview[:177] + "..."
        lines.append(f"ID={row.id} | DATE={row.date} | TITLE={row.title} | OVERVIEW={overview}")

    prompt = (
        "Проанализируй набор новостей и найди именно повторяющиеся медийные нарративы.\n"
        "Нарратив — это повторяющийся тезис или интерпретация, а не просто тема.\n"
        "Нужно вернуть только те нарративы, которые встречаются минимум в 2 новостях из списка.\n\n"
        "Верни строгий JSON-объект формата:\n"
        '{\n'
        '  "narratives": [\n'
        '    {\n'
        '      "narrative": "краткое обобщенное утверждение",\n'
        '      "pattern": "рост|снижение|риск|замещение|регулирование|стабилизация|другое",\n'
        '      "article_ids": [1, 2],\n'
        '      "keywords": ["слово1", "слово2", "слово3"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Правила:\n"
        "1. Не придумывай то, чего нет в новостях.\n"
        "2. Не используй статьи, которые не повторяют тезис.\n"
        "3. Не возвращай одиночные сигналы.\n"
        "4. Пиши нарратив кратко, ясно, по-русски.\n"
        "5. Используй только ID из списка.\n\n"
        f"Новости:\n{chr(10).join(lines)}"
    )
    system = "Ты аналитик медийных нарративов. Отвечай строго JSON без пояснений."
    response = generate_text(prompt=prompt, system=system, temperature=0.1)
    if not response:
        return []

    payload = _parse_json_payload(response)
    if not payload:
        return []

    narratives = payload.get("narratives", [])
    if not isinstance(narratives, list):
        return []

    rows: list[dict[str, object]] = []
    for item in narratives:
        if not isinstance(item, dict):
            continue
        article_ids = [int(x) for x in item.get("article_ids", []) if str(x).isdigit() and int(x) in valid_ids]
        article_ids = sorted(set(article_ids))
        if len(article_ids) < 2:
            continue
        narrative = str(item.get("narrative", "")).strip()
        if not narrative:
            continue
        pattern = str(item.get("pattern", "другое")).strip() or "другое"
        keywords = item.get("keywords", [])
        if isinstance(keywords, list):
            keywords = [str(keyword).strip() for keyword in keywords[:4] if str(keyword).strip()]
        else:
            keywords = []
        rows.append(
            {
                "narrative": narrative,
                "pattern": pattern,
                "article_ids": article_ids,
                "keywords": keywords,
            }
        )
    return rows


def _parse_json_payload(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _merge_narratives(extracted_df: pd.DataFrame) -> pd.DataFrame:
    if extracted_df.empty:
        return extracted_df

    vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(ALL_STOPWORDS), ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(extracted_df["narrative"].tolist())
    similarity = cosine_similarity(matrix)

    assigned = [-1] * len(extracted_df)
    group_id = 0
    for anchor_idx in range(len(extracted_df)):
        if assigned[anchor_idx] != -1:
            continue
        assigned[anchor_idx] = group_id
        anchor_ids = set(extracted_df.iloc[anchor_idx]["article_ids"])
        for idx in range(len(extracted_df)):
            if assigned[idx] != -1:
                continue
            shared_articles = len(anchor_ids & set(extracted_df.iloc[idx]["article_ids"]))
            sim_score = float(similarity[anchor_idx, idx])
            if sim_score >= 0.45 or (sim_score >= 0.32 and shared_articles >= 1):
                assigned[idx] = group_id
        group_id += 1

    extracted_df = extracted_df.copy()
    extracted_df["group_id"] = assigned

    rows: list[dict[str, object]] = []
    for _, group_df in extracted_df.groupby("group_id"):
        article_ids: set[int] = set()
        keyword_pool: list[str] = []
        for ids in group_df["article_ids"]:
            article_ids.update(ids)
        for keywords in group_df["keywords"]:
            keyword_pool.extend(keywords)

        representative = group_df.iloc[0]
        rows.append(
            {
                "narrative": _pick_representative_narrative(group_df["narrative"].tolist()),
                "pattern": _pick_representative_pattern(group_df["pattern"].tolist()),
                "article_ids": sorted(article_ids),
                "cohesion": _average_pairwise_similarity(group_df["narrative"].tolist()),
                "keywords": ", ".join(pd.Series(keyword_pool).value_counts().head(4).index.tolist()) if keyword_pool else "",
            }
        )
    return pd.DataFrame(rows)


def _pick_representative_narrative(narratives: list[str]) -> str:
    if len(narratives) == 1:
        return narratives[0]
    lengths = [(text, abs(len(text) - 60)) for text in narratives]
    return sorted(lengths, key=lambda item: item[1])[0][0]


def _pick_representative_pattern(patterns: list[str]) -> str:
    if not patterns:
        return "другое"
    return pd.Series(patterns).value_counts().index[0]


def _average_pairwise_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=list(ALL_STOPWORDS), ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    mask = ~pd.DataFrame(similarity).eq(1.0)
    values = pd.DataFrame(similarity).where(mask).stack()
    if values.empty:
        return 1.0
    return float(values.mean())


def _weekly_change(evidence_df: pd.DataFrame) -> int:
    dated_df = evidence_df.dropna(subset=["published_at"]).copy()
    if dated_df.empty:
        return len(evidence_df)
    dated_df["date"] = dated_df["published_at"].dt.date
    max_date = dated_df["date"].max()
    recent_border = max_date - pd.Timedelta(days=6)
    previous_border = max_date - pd.Timedelta(days=13)
    recent_count = int((dated_df["date"] >= recent_border).sum())
    previous_count = int(((dated_df["date"] >= previous_border) & (dated_df["date"] < recent_border)).sum())
    return recent_count - previous_count


def _keywords_from_rows(evidence_df: pd.DataFrame) -> str:
    texts = (evidence_df["title"].fillna("") + " " + evidence_df["overview"].fillna("")).tolist()
    tokens = []
    for text in texts:
        tokens.extend(tokenize_for_analysis(text))
    if not tokens:
        return ""
    return ", ".join(pd.Series(tokens).value_counts().head(4).index.tolist())


def _dominant_topic(evidence_df: pd.DataFrame) -> str:
    if evidence_df.empty or evidence_df["topic"].dropna().empty:
        return "Смешанная тема"
    return str(evidence_df["topic"].mode().iloc[0])
