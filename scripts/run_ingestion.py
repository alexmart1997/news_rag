import argparse

from app.parsers.rbc import RBCParser, RBCSearchParams
from app.services.ingestion import NewsIngestionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch RBC articles and save them to the database.")
    parser.add_argument("--query", required=True, help="Search query, for example: РБК")
    parser.add_argument("--date-from", required=True, help="Start date in YYYY-MM-DD or DD.MM.YYYY")
    parser.add_argument("--date-to", required=True, help="End date in YYYY-MM-DD or DD.MM.YYYY")
    parser.add_argument("--category", default="", help="RBC category, for example: TopRbcRu_economics")
    parser.add_argument("--project", default="rbcnews", help="RBC project name")
    parser.add_argument("--material", default="", help="RBC material filter")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional page limit for debugging")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip article page parsing and save only search metadata",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    service = NewsIngestionService(parser=RBCParser())
    result = service.run(
        search_params=RBCSearchParams(
            query=args.query,
            date_from=args.date_from,
            date_to=args.date_to,
            category=args.category,
            project=args.project,
            material=args.material,
        ),
        include_text=not args.no_text,
        max_pages=args.max_pages,
    )
    print(
        "Ingestion finished. "
        f"Fetched={result.fetched}, inserted={result.inserted}, updated={result.updated}."
    )


if __name__ == "__main__":
    main()
