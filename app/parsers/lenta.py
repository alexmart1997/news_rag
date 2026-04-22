from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

from app.config import settings


@dataclass(slots=True)
class ParsedArticle:
    title: str
    url: str
    overview: str | None
    text: str | None
    published_at: datetime | None


@dataclass(slots=True)
class LentaSearchParams:
    query: str
    date_from: str
    date_to: str
    offset: int = 0
    size: int = 500
    sort: str = "3"
    title_only: str = "0"
    domain: str = "1"
    material: str = "0"
    bloc: str = "4"

    def to_param_dict(self, offset: int | None = None) -> dict[str, str]:
        return {
            "query": self.query,
            "from": str(self.offset if offset is None else offset),
            "size": str(self.size),
            "dateFrom": _normalize_lenta_date(self.date_from),
            "dateTo": _normalize_lenta_date(self.date_to),
            "sort": self.sort,
            "title_only": self.title_only,
            "type": self.material,
            "bloc": self.bloc,
            "domain": self.domain,
        }


def _normalize_lenta_date(value: str) -> str:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def _parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class LentaParser:
    SOURCE_NAME = "Lenta.ru"
    SOURCE_URL = "https://lenta.ru/"

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
        has_type = int(param_dict["type"]) != 0
        has_bloc = int(param_dict["bloc"]) != 0

        return (
            "https://lenta.ru/search/v2/process?"
            f"from={param_dict['from']}&"
            f"size={param_dict['size']}&"
            f"sort={param_dict['sort']}&"
            f"title_only={param_dict['title_only']}&"
            f"domain={param_dict['domain']}&"
            "modified%2Cformat=yyyy-MM-dd&"
            f"{'type=' + param_dict['type'] + '&' if has_type else ''}"
            f"{'bloc=' + param_dict['bloc'] + '&' if has_bloc else ''}"
            f"modified%2Cfrom={param_dict['dateFrom']}&"
            f"modified%2Cto={param_dict['dateTo']}&"
            f"query={param_dict['query']}"
        )

    def _get_search_table(self, param_dict: dict[str, str]) -> pd.DataFrame:
        response = self.session.get(self._get_url(param_dict), timeout=self.timeout)
        response.raise_for_status()
        return pd.DataFrame(response.json().get("matches", []))

    def _get_article_data(self, url: str) -> tuple[str | None, str | None]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        overview = None
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag and description_tag.get("content"):
            overview = description_tag["content"].strip()

        article_root = soup.find("article") or soup.find("main") or soup
        paragraphs = article_root.find_all("p")
        text_parts = [p.get_text(" ", strip=True) for p in paragraphs if p.get_text(" ", strip=True)]
        text = " ".join(text_parts) if text_parts else None

        time.sleep(self.delay)
        return overview, text

    def fetch(
        self,
        search_params: LentaSearchParams,
        include_text: bool = True,
        chunk_days: int = 37,
    ) -> list[ParsedArticle]:
        param_copy = search_params.to_param_dict()
        step = timedelta(days=chunk_days)
        date_from = datetime.strptime(param_copy["dateFrom"], "%Y-%m-%d")
        date_to = datetime.strptime(param_copy["dateTo"], "%Y-%m-%d")
        if date_from > date_to:
            raise ValueError("date_from should be less than or equal to date_to")

        frames: list[pd.DataFrame] = []
        current = date_from
        while current <= date_to:
            chunk_end = min(current + step, date_to)
            chunk_params = param_copy.copy()
            chunk_params["dateFrom"] = current.strftime("%Y-%m-%d")
            chunk_params["dateTo"] = chunk_end.strftime("%Y-%m-%d")
            search_table = self._get_search_table(chunk_params)
            if include_text and not search_table.empty and "url" in search_table.columns:
                get_text = lambda row: self._get_article_data(row["url"])
                search_table[["overview", "text"]] = search_table.apply(get_text, axis=1).tolist()
            frames.append(search_table)
            current = chunk_end + timedelta(days=1)
            time.sleep(self.delay)

        if not frames:
            return []

        table = pd.concat(frames, axis=0, ignore_index=True)
        if table.empty:
            return []

        if "url" in table.columns:
            table = table.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

        articles: list[ParsedArticle] = []
        for item in table.to_dict(orient="records"):
            title = item.get("title")
            url = item.get("url")
            if not title or not url:
                continue

            overview = item.get("overview") or item.get("snippet")
            text = item.get("text")
            published_at = _parse_published_at(item.get("modified") or item.get("date") or item.get("pubdate"))
            articles.append(
                ParsedArticle(
                    title=title.strip(),
                    url=url.strip(),
                    overview=overview.strip() if isinstance(overview, str) and overview.strip() else None,
                    text=text.strip() if isinstance(text, str) and text.strip() else None,
                    published_at=published_at,
                )
            )

        return articles
