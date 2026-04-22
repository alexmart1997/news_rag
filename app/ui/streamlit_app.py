from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import Select, func, or_, select
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.ml.narratives import detect_narratives
from app.ml.news_analytics import build_daily_counts, build_top_keywords, build_topic_dynamics
from app.ml.qa import answer_question
from app.ml.similarity import find_similar_articles
from app.ml.topic_clustering import cluster_articles
from app.models import Article, Source


def _to_date(value: datetime | None) -> date | None:
    if value is None:
        return None
    return value.date()


def load_sources() -> list[str]:
    with SessionLocal() as session:
        return list(session.scalars(select(Source.name).order_by(Source.name)).all())


def load_article_stats() -> tuple[int, date | None, date | None]:
    with SessionLocal() as session:
        total_articles = session.scalar(select(func.count()).select_from(Article)) or 0
        min_date = _to_date(session.scalar(select(func.min(Article.published_at))))
        max_date = _to_date(session.scalar(select(func.max(Article.published_at))))
    return total_articles, min_date, max_date


def load_articles(
    limit: int,
    source_name: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    search_text: str = "",
) -> pd.DataFrame:
    stmt: Select = (
        select(
            Article.id,
            Source.name.label("source"),
            Article.title,
            Article.overview,
            Article.text,
            Article.url,
            Article.published_at,
        )
        .select_from(Article)
        .join(Source, Source.id == Article.source_id, isouter=True)
        .order_by(Article.published_at.desc().nullslast(), Article.id.desc())
        .limit(limit)
    )

    if source_name and source_name != "All":
        stmt = stmt.where(Source.name == source_name)
    if date_from:
        stmt = stmt.where(Article.published_at >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        stmt = stmt.where(Article.published_at <= datetime.combine(date_to, datetime.max.time()))
    if search_text.strip():
        pattern = f"%{search_text.strip()}%"
        stmt = stmt.where(
            or_(
                Article.title.ilike(pattern),
                Article.overview.ilike(pattern),
                Article.text.ilike(pattern),
            )
        )

    with SessionLocal() as session:
        rows = session.execute(stmt).mappings().all()

    return pd.DataFrame(rows)


def load_narrative_corpus(limit: int = 1500) -> pd.DataFrame:
    stmt: Select = (
        select(
            Article.id,
            Source.name.label("source"),
            Article.title,
            Article.overview,
            Article.text,
            Article.url,
            Article.published_at,
        )
        .select_from(Article)
        .join(Source, Source.id == Article.source_id, isouter=True)
        .order_by(Article.published_at.desc().nullslast(), Article.id.desc())
        .limit(limit)
    )

    with SessionLocal() as session:
        rows = session.execute(stmt).mappings().all()

    return pd.DataFrame(rows)


def render_sidebar(
    total_articles: int,
    min_date: date | None,
    max_date: date | None,
) -> tuple[str, date | None, date | None, str, int, int]:
    st.sidebar.header("\u0424\u0438\u043b\u044c\u0442\u0440\u044b")
    sources = ["\u0412\u0441\u0435", *load_sources()]
    selected_source = st.sidebar.selectbox("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a", options=sources, index=0)
    search_text = st.sidebar.text_input("\u041f\u043e\u0438\u0441\u043a \u043f\u043e \u043a\u043b\u044e\u0447\u0435\u0432\u043e\u043c\u0443 \u0441\u043b\u043e\u0432\u0443", value="")
    date_from = st.sidebar.date_input("\u0414\u0430\u0442\u0430 \u0441", value=min_date, min_value=min_date, max_value=max_date)
    date_to = st.sidebar.date_input("\u0414\u0430\u0442\u0430 \u043f\u043e", value=max_date, min_value=min_date, max_value=max_date)
    limit = st.sidebar.slider("\u0427\u0438\u0441\u043b\u043e \u0441\u0442\u0440\u043e\u043a", min_value=10, max_value=500, value=120, step=10)
    topic_count = st.sidebar.slider("\u0427\u0438\u0441\u043b\u043e \u0442\u0435\u043c", min_value=2, max_value=10, value=5, step=1)

    st.sidebar.markdown("### \u0414\u0430\u043d\u043d\u044b\u0435")
    st.sidebar.write(f"\u041d\u043e\u0432\u043e\u0441\u0442\u0435\u0439: {total_articles}")
    st.sidebar.write(f"\u041f\u0435\u0440\u0438\u043e\u0434: {min_date} - {max_date}")

    return selected_source, date_from, date_to, search_text, limit, topic_count


def main() -> None:
    st.set_page_config(page_title="News RAG", layout="wide")
    st.title("News RAG")
    st.caption("\u0421\u0438\u0441\u0442\u0435\u043c\u0430 \u0441\u0431\u043e\u0440\u0430, \u0430\u043d\u0430\u043b\u0438\u0437\u0430 \u0438 \u043f\u043e\u0438\u0441\u043a\u0430 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439.")

    total_articles, min_date, max_date = load_article_stats()
    if total_articles == 0:
        st.info("\u0411\u0430\u0437\u0430 \u043f\u043e\u043a\u0430 \u043f\u0443\u0441\u0442\u0430. \u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 ingestion \u0438 \u0437\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438.")
        return

    selected_source, date_from, date_to, search_text, limit, topic_count = render_sidebar(
        total_articles=total_articles,
        min_date=min_date,
        max_date=max_date,
    )

    articles_df = load_articles(
        limit=limit,
        source_name=selected_source,
        date_from=date_from,
        date_to=date_to,
        search_text=search_text,
    )

    if articles_df.empty:
        st.warning("\u041f\u043e \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u044b\u043c \u0444\u0438\u043b\u044c\u0442\u0440\u0430\u043c \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e.")
        return

    clustering_result = cluster_articles(articles_df, n_clusters=topic_count)
    articles_df = clustering_result.articles_df
    topic_summary_df = clustering_result.topic_summary_df

    narrative_corpus_df = load_narrative_corpus(limit=1500)
    if not narrative_corpus_df.empty:
        narrative_corpus_df = cluster_articles(narrative_corpus_df, n_clusters=topic_count).articles_df
    narrative_result = detect_narratives(narrative_corpus_df, top_n=6)
    narrative_summary_df = narrative_result.summary_df
    narrative_details_df = narrative_result.details_df
    narrative_signals_df = narrative_result.signals_df

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("\u041d\u043e\u0432\u043e\u0441\u0442\u0438")
        table_df = articles_df[["id", "source", "published_at", "topic", "title", "url"]].copy()
        table_df = table_df.rename(
            columns={
                "id": "ID",
                "source": "\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a",
                "published_at": "\u0414\u0430\u0442\u0430",
                "topic": "\u0422\u0435\u043c\u0430",
                "title": "\u0417\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
                "url": "URL",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("\u041a\u0440\u0430\u0442\u043a\u0430\u044f \u0441\u0442\u0430\u0442\u0438\u0441\u0442\u0438\u043a\u0430")
        st.metric("\u0412\u0438\u0434\u0438\u043c\u044b\u0445 \u0441\u0442\u0440\u043e\u043a", len(articles_df))
        st.metric("\u0423\u043d\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0445 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u043e\u0432", int(articles_df["source"].fillna("\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e").nunique()))
        st.metric("\u041d\u043e\u0432\u043e\u0441\u0442\u0435\u0439 \u0441 \u0442\u0435\u043a\u0441\u0442\u043e\u043c", int(articles_df["text"].notna().sum()))

    analytics_col1, analytics_col2 = st.columns(2)
    with analytics_col1:
        st.subheader("\u0414\u0438\u043d\u0430\u043c\u0438\u043a\u0430 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439")
        daily_counts_df = build_daily_counts(articles_df)
        if daily_counts_df.empty:
            st.info("\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0434\u0430\u043d\u043d\u044b\u0445 \u043e \u0434\u0430\u0442\u0430\u0445.")
        else:
            st.line_chart(daily_counts_df.set_index("date")["articles"], use_container_width=True)

    with analytics_col2:
        st.subheader("\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430")
        keywords_df = build_top_keywords(articles_df, top_n=15)
        if keywords_df.empty:
            st.info("\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0442\u0435\u043a\u0441\u0442\u0430 \u0434\u043b\u044f \u0432\u044b\u0434\u0435\u043b\u0435\u043d\u0438\u044f \u043a\u043b\u044e\u0447\u0435\u0432\u044b\u0445 \u0441\u043b\u043e\u0432.")
        else:
            st.bar_chart(keywords_df.set_index("keyword")["count"], use_container_width=True)

    st.subheader("\u0422\u0435\u043c\u044b")
    if topic_summary_df.empty:
        st.info("\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0442\u0435\u043a\u0441\u0442\u0430 \u0434\u043b\u044f \u0432\u044b\u0434\u0435\u043b\u0435\u043d\u0438\u044f \u0442\u0435\u043c.")
    else:
        cluster_col1, cluster_col2 = st.columns([1, 2])
        with cluster_col1:
            st.dataframe(
                topic_summary_df.rename(
                    columns={
                        "topic": "\u0422\u0435\u043c\u0430",
                        "size": "\u0420\u0430\u0437\u043c\u0435\u0440",
                        "keywords": "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with cluster_col2:
            st.bar_chart(topic_summary_df.set_index("topic")["size"], use_container_width=True)

    st.subheader("\u0414\u0438\u043d\u0430\u043c\u0438\u043a\u0430 \u0442\u0435\u043c")
    topic_dynamics_df = build_topic_dynamics(articles_df)
    if topic_dynamics_df.empty:
        st.info("\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0434\u0430\u043d\u043d\u044b\u0445 \u0434\u043b\u044f \u0434\u0438\u043d\u0430\u043c\u0438\u043a\u0438 \u0442\u0435\u043c.")
    else:
        st.area_chart(topic_dynamics_df, use_container_width=True)

    st.subheader("\u0420\u0430\u0434\u0430\u0440 \u043d\u0430\u0440\u0440\u0430\u0442\u0438\u0432\u043e\u0432")
    if narrative_summary_df.empty:
        st.info("\u041f\u043e\u043a\u0430 \u043d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0432\u044b\u0434\u0435\u043b\u0438\u0442\u044c \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u044e\u0449\u0438\u0435\u0441\u044f \u043d\u0430\u0440\u0440\u0430\u0442\u0438\u0432\u044b.")
    else:
        narrative_col1, narrative_col2 = st.columns([1.2, 1.8])
        with narrative_col1:
            st.dataframe(
                narrative_summary_df.rename(
                    columns={
                        "narrative": "\u041d\u0430\u0440\u0440\u0430\u0442\u0438\u0432",
                        "topic": "\u0422\u0435\u043c\u0430",
                        "pattern": "\u0422\u0438\u043f \u0441\u0438\u0433\u043d\u0430\u043b\u0430",
                        "coverage": "\u041e\u0445\u0432\u0430\u0442",
                        "share": "\u0414\u043e\u043b\u044f",
                        "weekly_change": "\u0420\u043e\u0441\u0442 \u0437\u0430 \u043d\u0435\u0434\u0435\u043b\u044e",
                        "cohesion": "\u0421\u043e\u0433\u043b\u0430\u0441\u043e\u0432\u0430\u043d\u043d\u043e\u0441\u0442\u044c",
                        "articles": "\u0421\u0442\u0430\u0442\u0435\u0439",
                        "keywords": "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with narrative_col2:
            st.bar_chart(
                narrative_summary_df.set_index("narrative")["coverage"],
                use_container_width=True,
            )

        selected_narrative = st.selectbox(
            "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043d\u0430\u0440\u0440\u0430\u0442\u0438\u0432",
            options=narrative_summary_df["narrative"].tolist(),
        )
        selected_narrative_row = narrative_summary_df[
            narrative_summary_df["narrative"] == selected_narrative
        ].iloc[0]
        st.markdown(f"**\u041d\u0430\u0440\u0440\u0430\u0442\u0438\u0432:** {selected_narrative_row['narrative']}")
        st.markdown(
            f"**\u041e\u0445\u0432\u0430\u0442:** {selected_narrative_row['coverage']} \u0441\u0442\u0430\u0442\u0435\u0439 | "
            f"**\u0414\u043e\u043b\u044f \u0432 \u043a\u043e\u0440\u043f\u0443\u0441\u0435:** {selected_narrative_row['share']:.1%} | "
            f"**\u0420\u043e\u0441\u0442 \u0437\u0430 \u043d\u0435\u0434\u0435\u043b\u044e:** {selected_narrative_row['weekly_change']}"
        )
        st.markdown(
            f"**\u0421\u043e\u0433\u043b\u0430\u0441\u043e\u0432\u0430\u043d\u043d\u043e\u0441\u0442\u044c:** {selected_narrative_row['cohesion']} | "
            f"**\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u0438\u0433\u043d\u0430\u043b\u044b:** {selected_narrative_row['keywords']}"
        )
        selected_details_df = narrative_details_df[
            narrative_details_df["narrative"] == selected_narrative
        ]
        if not selected_details_df.empty:
            st.dataframe(
                selected_details_df.rename(
                    columns={
                        "narrative": "\u041d\u0430\u0440\u0440\u0430\u0442\u0438\u0432",
                        "title": "\u0417\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
                        "published_at": "\u0414\u0430\u0442\u0430",
                        "topic": "\u0422\u0435\u043c\u0430",
                        "url": "URL",
                        "overview": "\u0410\u043d\u043e\u043d\u0441",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("\u0421\u0438\u0433\u043d\u0430\u043b\u044b")
    if narrative_signals_df.empty:
        st.info("\u041e\u0434\u0438\u043d\u043e\u0447\u043d\u044b\u0445 \u0441\u0438\u0433\u043d\u0430\u043b\u043e\u0432 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e.")
    else:
        st.caption("\u042d\u0442\u043e \u0435\u0449\u0435 \u043d\u0435 \u043d\u0430\u0440\u0440\u0430\u0442\u0438\u0432\u044b, \u0430 \u043e\u0434\u0438\u043d\u043e\u0447\u043d\u044b\u0435 \u0438\u043b\u0438 \u0441\u043b\u0430\u0431\u043e \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u044e\u0449\u0438\u0435\u0441\u044f \u0442\u0435\u0437\u0438\u0441\u044b.")
        st.dataframe(
            narrative_signals_df.rename(
                columns={
                    "signal": "\u0421\u0438\u0433\u043d\u0430\u043b",
                    "topic": "\u0422\u0435\u043c\u0430",
                    "pattern": "\u0422\u0438\u043f \u0441\u0438\u0433\u043d\u0430\u043b\u0430",
                    "published_at": "\u0414\u0430\u0442\u0430",
                    "keywords": "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430",
                    "url": "URL",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("\u0412\u043e\u043f\u0440\u043e\u0441 \u043f\u043e \u043d\u043e\u0432\u043e\u0441\u0442\u044f\u043c")
    question = st.text_input(
        "\u0417\u0430\u0434\u0430\u0439\u0442\u0435 \u0432\u043e\u043f\u0440\u043e\u0441 \u043f\u043e \u0442\u0435\u043a\u0443\u0449\u0435\u0439 \u0432\u044b\u0431\u043e\u0440\u043a\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439",
        value="",
        placeholder="\u041d\u0430\u043f\u0440\u0438\u043c\u0435\u0440: \u0427\u0442\u043e \u043f\u0440\u043e\u0438\u0441\u0445\u043e\u0434\u0438\u043b\u043e \u0432 \u044d\u043a\u043e\u043d\u043e\u043c\u0438\u043a\u0435 \u0432 \u044d\u0442\u043e\u0442 \u043f\u0435\u0440\u0438\u043e\u0434?",
    )
    if question.strip():
        qa_result = answer_question(articles_df, question=question, top_k=3)
        st.write(qa_result.answer)
        if not qa_result.hits_df.empty:
            st.dataframe(
                qa_result.hits_df.rename(
                    columns={
                        "id": "ID",
                        "title": "\u0417\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
                        "topic": "\u0422\u0435\u043c\u0430",
                        "published_at": "\u0414\u0430\u0442\u0430",
                        "score": "\u0421\u0445\u043e\u0436\u0435\u0441\u0442\u044c",
                        "url": "URL",
                        "snippet": "\u0424\u0440\u0430\u0433\u043c\u0435\u043d\u0442",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("\u0414\u0435\u0442\u0430\u043b\u0438 \u043d\u043e\u0432\u043e\u0441\u0442\u0438")
    indexed_df = articles_df.reset_index(drop=True)
    article_options = {f"{row.id}: {row.title[:100]}": index for index, row in indexed_df.iterrows()}
    selected_label = st.selectbox("\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u044c", options=list(article_options.keys()))
    selected_row = indexed_df.iloc[article_options[selected_label]]

    st.markdown(f"**\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a:** {selected_row['source'] or '\u041d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e'}")
    st.markdown(f"**\u0414\u0430\u0442\u0430:** {selected_row['published_at']}")
    st.markdown(f"**URL:** [\u041e\u0442\u043a\u0440\u044b\u0442\u044c \u043d\u043e\u0432\u043e\u0441\u0442\u044c]({selected_row['url']})")
    if selected_row["overview"]:
        st.markdown("**\u0410\u043d\u043e\u043d\u0441**")
        st.write(selected_row["overview"])
    if selected_row["text"]:
        st.markdown("**\u0422\u0435\u043a\u0441\u0442**")
        st.write(selected_row["text"])

    st.subheader("\u041f\u043e\u0445\u043e\u0436\u0438\u0435 \u0441\u0442\u0430\u0442\u044c\u0438")
    similar_df = find_similar_articles(indexed_df, selected_index=article_options[selected_label], top_k=5)
    if similar_df.empty:
        st.info("\u041d\u0435\u0434\u043e\u0441\u0442\u0430\u0442\u043e\u0447\u043d\u043e \u0442\u0435\u043a\u0441\u0442\u0430 \u0434\u043b\u044f \u043f\u043e\u0438\u0441\u043a\u0430 \u043f\u043e\u0445\u043e\u0436\u0438\u0445 \u043d\u043e\u0432\u043e\u0441\u0442\u0435\u0439.")
    else:
        st.dataframe(
            similar_df.rename(
                columns={
                    "id": "ID",
                    "title": "\u0417\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
                    "topic": "\u0422\u0435\u043c\u0430",
                    "published_at": "\u0414\u0430\u0442\u0430",
                    "similarity": "\u0421\u0445\u043e\u0436\u0435\u0441\u0442\u044c",
                    "url": "URL",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
