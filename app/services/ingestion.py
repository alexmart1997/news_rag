from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models import Article, Source
from app.parsers.lenta import ParsedArticle


@dataclass(slots=True)
class IngestionResult:
    fetched: int
    inserted: int
    updated: int


class NewsIngestionService:
    """Coordinates parsing and persistence for news sources."""

    def __init__(self, parser: Any):
        self.parser = parser

    def run(
        self,
        search_params: Any,
        include_text: bool = True,
        **fetch_kwargs: Any,
    ) -> IngestionResult:
        parsed_articles = self.parser.fetch(
            search_params=search_params,
            include_text=include_text,
            **fetch_kwargs,
        )

        inserted = 0
        updated = 0

        with SessionLocal() as session:
            source = self._get_or_create_source(session)

            for parsed_article in parsed_articles:
                article, created = self._upsert_article(
                    session=session,
                    source=source,
                    parsed_article=parsed_article,
                )
                inserted += int(created)
                updated += int(not created and article is not None)

            session.commit()

        return IngestionResult(
            fetched=len(parsed_articles),
            inserted=inserted,
            updated=updated,
        )

    def _get_or_create_source(self, session: Session) -> Source:
        source = session.scalar(select(Source).where(Source.name == self.parser.SOURCE_NAME))
        if source is not None:
            return source

        source = Source(name=self.parser.SOURCE_NAME, base_url=self.parser.SOURCE_URL)
        session.add(source)
        session.flush()
        return source

    def _upsert_article(
        self,
        session: Session,
        source: Source,
        parsed_article: ParsedArticle,
    ) -> tuple[Article | None, bool]:
        existing_article = session.scalar(select(Article).where(Article.url == parsed_article.url))
        if existing_article is None:
            article = Article(
                source_id=source.id,
                url=parsed_article.url,
                title=parsed_article.title,
                overview=parsed_article.overview,
                text=parsed_article.text,
                published_at=parsed_article.published_at,
            )
            session.add(article)
            return article, True

        self._merge_article(existing_article, parsed_article)
        existing_article.parsed_at = datetime.utcnow()
        return existing_article, False

    @staticmethod
    def _merge_article(existing_article: Article, parsed_article: ParsedArticle) -> None:
        existing_article.title = parsed_article.title or existing_article.title
        existing_article.overview = parsed_article.overview or existing_article.overview
        existing_article.text = parsed_article.text or existing_article.text
        existing_article.published_at = parsed_article.published_at or existing_article.published_at
