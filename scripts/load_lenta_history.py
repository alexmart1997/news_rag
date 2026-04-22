from __future__ import annotations

import argparse

from app.parsers.lenta import LentaParser, LentaSearchParams
from app.services.ingestion import NewsIngestionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a larger historical Lenta.ru dataset into the database.")
    parser.add_argument("--date-from", default="2024-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--date-to", default="2024-12-31", help="End date in YYYY-MM-DD")
    parser.add_argument("--query", default="", help="Optional search query")
    parser.add_argument("--bloc", default="4", help="Section filter, economics=4")
    parser.add_argument("--size", type=int, default=1000, help="Articles per chunk")
    parser.add_argument("--chunk-days", type=int, default=30, help="Chunk size in days")
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Also download article texts. Without this flag only metadata is loaded.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = NewsIngestionService(parser=LentaParser())
    result = service.run(
        search_params=LentaSearchParams(
            query=args.query,
            date_from=args.date_from,
            date_to=args.date_to,
            size=args.size,
            bloc=args.bloc,
        ),
        include_text=args.with_text,
        chunk_days=args.chunk_days,
    )
    print(
        "Historical ingestion finished. "
        f"Fetched={result.fetched}, inserted={result.inserted}, updated={result.updated}."
    )


if __name__ == "__main__":
    main()
