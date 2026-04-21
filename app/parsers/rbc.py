from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

from app.config import settings


@dataclass(slots=True)
class RBCSearchParams:
    query: str
    date_from: str
    date_to: str
    project: str = "rbcnews"
    category: str = "TopRbcRu_economics"
    page: int = 0
    material: str = ""

    def to_param_dict(self, page: int | None = None) -> dict[str, str]:
        return {
            "query": self.query,
            "project": self.project,
            "category": self.category,
            "dateFrom": _normalize_rbc_date(self.date_from),
            "dateTo": _normalize_rbc_date(self.date_to),
            "page": str(self.page if page is None else page),
            "material": self.material,
        }


@dataclass(slots=True)
class ParsedArticle:
    title: str
    url: str
    overview: str | None
    text: str | None
    published_at: datetime | None


def _normalize_rbc_date(value: str) -> str:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def _parse_published_at(raw_value: str | int | None) -> datetime | None:
    if raw_value in (None, ""):
        return None

    try:
        return datetime.fromtimestamp(int(raw_value))
    except (TypeError, ValueError, OSError):
        return None


class RBCParser:
    SOURCE_NAME = "RBC"
    SOURCE_URL = "https://www.rbc.ru/"

    def __init__(self, timeout: int | None = None, delay: float | None = None):
        self.timeout = timeout or settings.rbc_request_timeout
        self.delay = delay if delay is not None else settings.rbc_request_delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )

    def _get_url(self, param_dict: dict[str, str]) -> str:
        return (
            "https://www.rbc.ru/search/ajax/?"
            f"project={param_dict['project']}&"
            f"category={param_dict['category']}&"
            f"dateFrom={param_dict['dateFrom']}&"
            f"dateTo={param_dict['dateTo']}&"
            f"page={param_dict['page']}&"
            f"query={param_dict['query']}&"
            f"material={param_dict['material']}"
        )

    def _get_search_table(
        self,
        param_dict: dict[str, str],
        include_text: bool = True,
    ) -> pd.DataFrame:
        url = self._get_url(param_dict)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()
        search_table = pd.DataFrame(payload.get("items", []))
        if include_text and not search_table.empty:
            get_text = lambda row: self._get_article_data(row["fronturl"])
            search_table[["overview", "text"]] = search_table.apply(get_text, axis=1).tolist()

        if "publish_date_t" in search_table.columns:
            search_table = search_table.sort_values("publish_date_t", ignore_index=True)

        return search_table

    def _iterable_load_by_page(
        self,
        param_dict: dict[str, str],
        include_text: bool = True,
        max_pages: int | None = None,
    ) -> pd.DataFrame:
        param_copy = param_dict.copy()
        results: list[pd.DataFrame] = []
        pages_loaded = 0

        result = self._get_search_table(param_copy, include_text=include_text)
        while not result.empty:
            results.append(result)
            pages_loaded += 1
            if max_pages is not None and pages_loaded >= max_pages:
                break

            param_copy["page"] = str(int(param_copy["page"]) + 1)
            time.sleep(self.delay)
            result = self._get_search_table(param_copy, include_text=include_text)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, axis=0, ignore_index=True)

    def _get_article_data(self, url: str) -> tuple[str | None, str | None]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, features="lxml")
        div_overview = soup.find("div", {"class": "article__text__overview"})
        if div_overview:
            overview = div_overview.text.replace("<br />", "\n").strip()
        else:
            overview = None

        article_root = (
            soup.find("div", {"class": "article__text"})
            or soup.find("div", {"class": "article__content"})
            or soup
        )
        p_text = article_root.find_all("p")
        if p_text:
            text = " ".join(
                map(
                    lambda x: x.text.replace("<br />", "\n").strip(),
                    p_text,
                )
            )
        else:
            text = None

        time.sleep(self.delay)
        return overview, text

    def fetch(
        self,
        search_params: RBCSearchParams,
        include_text: bool = True,
        max_pages: int | None = None,
    ) -> list[ParsedArticle]:
        table = self._iterable_load_by_page(
            search_params.to_param_dict(),
            include_text=include_text,
            max_pages=max_pages,
        )
        if table.empty:
            return []

        table = table.drop_duplicates(subset=["fronturl"], keep="first").reset_index(drop=True)

        articles: list[ParsedArticle] = []
        for item in table.to_dict(orient="records"):
            title = item.get("title")
            url = item.get("fronturl")
            if not title or not url:
                continue

            overview = item.get("overview") or item.get("announce")
            text = item.get("text")

            articles.append(
                ParsedArticle(
                    title=title.strip(),
                    url=url.strip(),
                    overview=overview.strip() if isinstance(overview, str) and overview.strip() else None,
                    text=text.strip() if isinstance(text, str) and text.strip() else None,
                    published_at=_parse_published_at(item.get("publish_date_t")),
                )
            )

        return articles
