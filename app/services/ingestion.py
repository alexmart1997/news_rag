from app.parsers.rbc import RBCParser


class NewsIngestionService:
    """Coordinates parsing and persistence for news sources."""

    def __init__(self, parser: RBCParser):
        self.parser = parser

    def run(self) -> list[dict]:
        return self.parser.fetch()
