from dataclasses import dataclass


@dataclass(slots=True)
class RBCSearchParams:
    query: str
    project: str = "rbcnews"
    category: str = ""
    date_from: str = ""
    date_to: str = ""
    page: int = 0
    material: str = ""


class RBCParser:
    """Placeholder for the RBC parser implementation."""

    def fetch(self) -> list[dict]:
        return []
