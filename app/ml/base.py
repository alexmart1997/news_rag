class BaseAnalyzer:
    """Base interface for interchangeable analytics modules."""

    def analyze(self, articles: list[dict]) -> dict:
        raise NotImplementedError
