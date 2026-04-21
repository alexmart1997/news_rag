from __future__ import annotations

import argparse

from sqlalchemy import desc, select

from app.db.session import SessionLocal
from app.models import Article


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show recent articles stored in the database.")
    parser.add_argument("--limit", type=int, default=10, help="How many articles to display")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with SessionLocal() as session:
        articles = session.scalars(
            select(Article).order_by(desc(Article.published_at), desc(Article.id)).limit(args.limit)
        ).all()

    if not articles:
        print("No articles found in the database.")
        return

    for article in articles:
        print("-" * 80)
        print(f"id: {article.id}")
        print(f"title: {article.title}")
        print(f"published_at: {article.published_at}")
        print(f"url: {article.url}")
        if article.overview:
            print(f"overview: {article.overview[:200]}")
        if article.text:
            print(f"text: {article.text[:300]}")


if __name__ == "__main__":
    main()
