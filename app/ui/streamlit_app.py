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
from app.ml.news_analytics import build_daily_counts, build_top_keywords
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


def render_sidebar(
    total_articles: int,
    min_date: date | None,
    max_date: date | None,
) -> tuple[str, date | None, date | None, str, int]:
    st.sidebar.header("Filters")
    sources = ["All", *load_sources()]
    selected_source = st.sidebar.selectbox("Source", options=sources, index=0)
    search_text = st.sidebar.text_input("Keyword search", value="")
    date_from = st.sidebar.date_input("Date from", value=min_date, min_value=min_date, max_value=max_date)
    date_to = st.sidebar.date_input("Date to", value=max_date, min_value=min_date, max_value=max_date)
    limit = st.sidebar.slider("Rows", min_value=10, max_value=200, value=50, step=10)

    st.sidebar.markdown("### Dataset")
    st.sidebar.write(f"Articles: {total_articles}")
    st.sidebar.write(f"Date range: {min_date} - {max_date}")

    return selected_source, date_from, date_to, search_text, limit


def main() -> None:
    st.set_page_config(page_title="News RAG", layout="wide")
    st.title("News RAG")
    st.caption("Minimal interface for browsing collected news articles.")

    total_articles, min_date, max_date = load_article_stats()
    if total_articles == 0:
        st.info("The database is empty. Run ingestion first to load articles.")
        return

    selected_source, date_from, date_to, search_text, limit = render_sidebar(
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
        st.warning("No articles found for the selected filters.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Articles")
        table_df = articles_df[["id", "source", "published_at", "title", "url"]].copy()
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Quick stats")
        st.metric("Visible rows", len(articles_df))
        st.metric("Unique sources", int(articles_df["source"].fillna("Unknown").nunique()))
        st.metric("Rows with text", int(articles_df["text"].notna().sum()))

    analytics_col1, analytics_col2 = st.columns(2)
    with analytics_col1:
        st.subheader("News Dynamics")
        daily_counts_df = build_daily_counts(articles_df)
        if daily_counts_df.empty:
            st.info("Not enough date information for dynamics.")
        else:
            st.line_chart(daily_counts_df.set_index("date")["articles"], use_container_width=True)

    with analytics_col2:
        st.subheader("Top Keywords")
        keywords_df = build_top_keywords(articles_df, top_n=15)
        if keywords_df.empty:
            st.info("Not enough text information for keywords.")
        else:
            st.bar_chart(keywords_df.set_index("keyword")["count"], use_container_width=True)

    st.subheader("Article details")
    indexed_df = articles_df.reset_index(drop=True)
    article_options = {f"{row.id}: {row.title[:100]}": index for index, row in indexed_df.iterrows()}
    selected_label = st.selectbox("Select article", options=list(article_options.keys()))
    selected_row = indexed_df.iloc[article_options[selected_label]]

    st.markdown(f"**Source:** {selected_row['source'] or 'Unknown'}")
    st.markdown(f"**Published:** {selected_row['published_at']}")
    st.markdown(f"**URL:** [Open article]({selected_row['url']})")
    if selected_row["overview"]:
        st.markdown("**Overview**")
        st.write(selected_row["overview"])
    if selected_row["text"]:
        st.markdown("**Text**")
        st.write(selected_row["text"])


if __name__ == "__main__":
    main()
