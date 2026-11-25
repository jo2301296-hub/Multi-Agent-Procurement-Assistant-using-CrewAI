"""Facts collected by the tools during one run.

Guardrails compare what the agents *say* with what the tools *saw*, so a made-up
URL or price never reaches the report.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from .config import Company, Limits, ProcurementRequest
from .schemas import Product, RankedProduct, Requirement


def normalise_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, "")
    )


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def spread_over_websites(urls: list[str]) -> list[str]:
    """Reorder so that every website gets a turn before any gets a second one."""
    seen: dict[str, int] = {}
    turn: dict[str, int] = {}
    for url in urls:
        domain = domain_of(url)
        turn[url] = seen[domain] = seen.get(domain, 0) + 1
    return sorted(urls, key=lambda url: turn[url])


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
_NUMBER = re.compile(r"\d[\d.,]*")
_SPACED_THOUSANDS = re.compile(r"(?<=\d)[   ](?=\d{3}(?!\d))")


def numbers_in(text: str) -> set[float]:
    """Every number in the text, reading both 12,999.50 and 12.999,50."""
    text = text.translate(_ARABIC_DIGITS)
    found: set[float] = set()
    # "12 999" may be one number or two, so both readings are kept.
    for variant in (text, _SPACED_THOUSANDS.sub("", text)):
        for token in _NUMBER.findall(variant):
            token = token.rstrip(".,")
            readings = (token.replace(",", ""), token.replace(".", "").replace(",", "."))
            for reading in readings:
                try:
                    found.add(float(reading))
                except ValueError:
                    pass
    return found


@dataclass
class ScrapedPage:
    url: str
    tool: str
    text: str

    @property
    def extracted(self) -> dict | None:
        """The fields ScrapeGraph extracted, or None for plain text."""
        try:
            data = json.loads(self.text)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    @property
    def usable(self) -> bool:
        data = self.extracted
        return data is None or bool(data.get("price"))

    def shows_price(self, price: float) -> bool:
        # ScrapeGraph returns a price field; "1 TB" must not pass as a price of 1.
        data = self.extracted
        if data and isinstance(data.get("price"), (int, float)):
            return abs(data["price"] - price) < 0.01
        return any(abs(n - price) < 0.01 for n in numbers_in(self.text))


@dataclass
class RunState:
    company: Company
    request: ProcurementRequest
    demo: bool = False
    limits: Limits = Limits()
    queries: list[str] = field(default_factory=list)
    found_urls: dict[str, str] = field(default_factory=dict)  # normalised -> original
    candidates: list[str] = field(default_factory=list)  # normalised
    pages: dict[str, ScrapedPage] = field(default_factory=dict)
    failed_pages: dict[str, str] = field(default_factory=dict)
    requirements: list[Requirement] = field(default_factory=list)
    products: list[Product] = field(default_factory=list)
    ranking: list[RankedProduct] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def products_missing(self) -> int:
        usable = sum(page.usable for page in self.pages.values())
        return max(0, self.request.max_products - usable)

    @property
    def recommended(self) -> RankedProduct | None:
        return next((r for r in self.ranking if r.eligible), None)
