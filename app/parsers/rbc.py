from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import settings


@dataclass(slots=True)
class RBCSearchParams:
    query: str
    date_from: str
    date_to: str
    project: str = "rbcnews"
    category: str = ""
    page: int = 0
    material: str = ""

    def to_request_params(self, page: int) -> dict[str, str]:
        return {
            "query": self.query,
            "project": self.project,
            "category": self.category,
            "dateFrom": _normalize_rbc_date(self.date_from),
            "dateTo": _normalize_rbc_date(self.date_to),
            "page": str(page),
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


def _parse_published_at(item: dict[str, Any]) -> datetime | None:
    publish_timestamp = item.get("publish_date_t")
    if publish_timestamp:
        try:
            return datetime.fromtimestamp(int(publish_timestamp))
        except (TypeError, ValueError, OSError):
            pass

    for key in ("publish_date", "publish_date_rfc", "date"):
        value = item.get(key)
        if not value:
            continue

        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    return None


class RBCParser:
    BASE_URL = "https://www.rbc.ru/search/ajax/"
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
                ),
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.rbc.ru/search/",
                "Origin": "https://www.rbc.ru",
                "X-Requested-With": "XMLHttpRequest",
            }
        )

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._is_warmed_up = False

    def fetch(
        self,
        search_params: RBCSearchParams,
        include_text: bool = True,
        max_pages: int | None = None,
    ) -> list[ParsedArticle]:
        articles: list[ParsedArticle] = []
        page = search_params.page

        while True:
            if max_pages is not None and page >= search_params.page + max_pages:
                break

            payload = self._get_search_page(search_params, page)
            items = payload.get("items", [])
            if not items:
                break

            for item in items:
                article = self._parse_search_item(item, include_text=include_text)
                if article is not None:
                    articles.append(article)

            page += 1
            time.sleep(self.delay)

        return articles

    def _get_search_page(self, search_params: RBCSearchParams, page: int) -> dict[str, Any]:
        self._warm_up_session()
        response = self.session.get(
            self.BASE_URL,
            params=search_params.to_request_params(page),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _warm_up_session(self) -> None:
        if self._is_warmed_up:
            return

        # RBC may require cookies from a regular page visit before the AJAX
        # search endpoint becomes available.
        self.session.get(self.SOURCE_URL, timeout=self.timeout)
        time.sleep(self.delay)
        self.session.get("https://www.rbc.ru/search/", timeout=self.timeout)
        time.sleep(self.delay)
        self._is_warmed_up = True

    def _parse_search_item(
        self,
        item: dict[str, Any],
        include_text: bool,
    ) -> ParsedArticle | None:
        url = item.get("fronturl") or item.get("url")
        title = item.get("title")
        if not url or not title:
            return None

        overview = item.get("announce") or item.get("overview")
        text = None

        if include_text:
            page_overview, page_text = self._fetch_article_details(url)
            overview = page_overview or overview
            text = page_text

        return ParsedArticle(
            title=title.strip(),
            url=url.strip(),
            overview=overview.strip() if isinstance(overview, str) and overview.strip() else None,
            text=text,
            published_at=_parse_published_at(item),
        )

    def _fetch_article_details(self, url: str) -> tuple[str | None, str | None]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None, None

        soup = BeautifulSoup(response.text, "lxml")

        overview_tag = soup.find("div", class_="article__text__overview")
        overview = overview_tag.get_text(" ", strip=True) if overview_tag else None

        article_root = (
            soup.find("div", class_="article__text")
            or soup.find("div", class_="article__content")
            or soup.find("article")
        )

        if article_root is None:
            return overview, None

        paragraphs = article_root.find_all("p")
        text_parts = [p.get_text(" ", strip=True) for p in paragraphs if p.get_text(" ", strip=True)]
        text = " ".join(text_parts) if text_parts else None

        time.sleep(self.delay)
        return overview, text
