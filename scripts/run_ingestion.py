from app.parsers.rbc import RBCParser
from app.services.ingestion import NewsIngestionService


def main() -> None:
    service = NewsIngestionService(parser=RBCParser())
    articles = service.run()
    print(f"Fetched {len(articles)} articles.")


if __name__ == "__main__":
    main()
