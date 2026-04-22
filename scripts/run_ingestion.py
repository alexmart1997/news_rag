import argparse

from app.parsers.lenta import LentaParser, LentaSearchParams
from app.services.ingestion import NewsIngestionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Lenta.ru articles and save them to the database.")
    parser.add_argument("--query", default="", help="Search query. Empty string means broad search.")
    parser.add_argument("--date-from", required=True, help="Start date in YYYY-MM-DD or DD.MM.YYYY")
    parser.add_argument("--date-to", required=True, help="End date in YYYY-MM-DD or DD.MM.YYYY")
    parser.add_argument("--offset", type=int, default=0, help="Search offset")
    parser.add_argument("--size", type=int, default=500, help="Articles per search chunk")
    parser.add_argument("--sort", default="3", help="Lenta sort mode")
    parser.add_argument("--title-only", default="0", help="Search only in title")
    parser.add_argument("--domain", default="1", help="Lenta domain parameter")
    parser.add_argument("--material", default="0", help="Material type filter")
    parser.add_argument("--bloc", default="4", help="Section filter, economics=4")
    parser.add_argument("--chunk-days", type=int, default=37, help="Chunk size for date range splitting")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip article page parsing and save only search metadata",
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
            offset=args.offset,
            size=args.size,
            sort=args.sort,
            title_only=args.title_only,
            domain=args.domain,
            material=args.material,
            bloc=args.bloc,
        ),
        include_text=not args.no_text,
        chunk_days=args.chunk_days,
    )
    print(
        "Ingestion finished. "
        f"Fetched={result.fetched}, inserted={result.inserted}, updated={result.updated}."
    )


if __name__ == "__main__":
    main()
