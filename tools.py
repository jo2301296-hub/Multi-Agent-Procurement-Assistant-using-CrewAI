"""CrewAI tools: Tavily search, ScrapeGraph scraping and the ranking lookup."""

from __future__ import annotations

import json
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from .state import RunState, ScrapedPage, domain_of, normalise_url

MAX_PAGE_CHARS = 100_000
MAX_FALLBACK_CHARS = 40_000

NOT_A_PRODUCT = "not a page of one product"

EXTRACTION_PROMPT = (
    "Extract the main product offered for sale on this page: title, current unit price "
    "(not the crossed-out or instalment price), ISO currency code, availability, seller, "
    "warranty, and every technical specification. Set is_single_product to false if the page "
    "is a category, search result, article or error page. Use null for anything not on the page."
)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_single_product": {"type": "boolean"},
        "title": {"type": "string"},
        "price": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "in_stock": {"type": ["boolean", "null"]},
        "seller": {"type": ["string", "null"]},
        "warranty": {"type": ["string", "null"]},
        "specs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                "required": ["name", "value"],
            },
        },
    },
    "required": ["is_single_product", "title", "specs"],
}


def shorten(text: str, limit: int) -> str:
    """Fit a page into the limit; for extracted JSON drop the last specs, never the price."""
    if len(text) <= limit:
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return text[:limit]
    if not isinstance(data, dict):
        return text[:limit]
    specs = data.get("specs")
    if isinstance(specs, list):
        data["specs"] = specs = list(specs)
        while specs and len(json.dumps(data, ensure_ascii=False)) > limit:
            specs.pop()
    return json.dumps(data, ensure_ascii=False)


class StatefulTool(BaseTool):
    """A tool that records what it saw in the shared RunState."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    state: Any = Field(exclude=True)
    # CrewAI must not serve a cached answer: the tools keep their own record.
    cache_function: Any = lambda _args, _result: False

    @property
    def run_state(self) -> RunState:
        return self.state


class SearchInput(BaseModel):
    query: str = Field(description="The web search query")


class TavilySearchTool(StatefulTool):
    name: str = "tavily_search"
    description: str = (
        "Search the web for product pages. Returns a JSON list of {title, url, snippet}."
    )
    args_schema: Type[BaseModel] = SearchInput
    client: Any = Field(exclude=True)

    def _run(self, query: str) -> str:
        state, company = self.run_state, self.run_state.company
        options: dict[str, Any] = {"search_depth": "advanced", "max_results": state.limits.search_results}
        if company.search_country:
            options["country"] = company.search_country
        if company.preferred_websites:
            options["include_domains"] = company.preferred_websites
        if company.excluded_websites:
            options["exclude_domains"] = company.excluded_websites
        try:
            response = self.client.search(query, **options)
        except Exception as exc:  # let the agent see the error and adapt
            state.warn(f"Search failed for '{query}': {exc}")
            return f"Search failed: {exc}"
        state.queries.append(query)
        rows = []
        for item in response.get("results", []):
            url = item.get("url", "")
            if not url.startswith(("http://", "https://")):
                continue
            state.found_urls.setdefault(normalise_url(url), url)
            rows.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": (item.get("content") or "")[: state.limits.snippet_chars],
                }
            )
        return json.dumps(rows, ensure_ascii=False) if rows else "No results for this query."


class ScrapeInput(BaseModel):
    url: str = Field(description="A product page URL from the candidate list")


class ScrapeGraphTool(StatefulTool):
    name: str = "scrape_product_page"
    description: str = (
        "Scrape one product page with ScrapeGraph and return its title, price, currency, "
        "availability, seller, warranty and specifications."
    )
    args_schema: Type[BaseModel] = ScrapeInput
    client: Any = Field(exclude=True)
    fallback: Any = Field(default=None, exclude=True)  # Tavily client

    def _run(self, url: str) -> str:
        state = self.run_state
        key = normalise_url(url)
        if key not in state.candidates:
            return "Refused: this URL is not in the candidate list."
        if key in state.pages:  # never pay twice for the same page
            return shorten(state.pages[key].text, state.limits.page_chars)
        if key in state.failed_pages:
            return f"This page already failed: {state.failed_pages[key]}"
        if not state.products_missing:
            return "Enough pages were scraped. Give the final answer now."

        tool = "ScrapeGraph"
        text, error = self._scrapegraph(url=url)
        if text is None and self.fallback is not None and error != NOT_A_PRODUCT:
            tool = "Tavily Extract"
            text, fallback_error = self._read_with_tavily(url)
            if text is None:
                error = f"{error}; Tavily Extract: {fallback_error}"
            else:
                state.warn(
                    f"{domain_of(url)} was read with Tavily Extract "
                    f"because ScrapeGraph could not open it: {error}"
                )
        if text is None:
            state.failed_pages[key] = error or "unknown error"
            state.warn(f"Could not scrape {domain_of(url)}: {error}")
            return f"Scraping failed: {error}"
        state.pages[key] = ScrapedPage(url=url, tool=tool, text=text[:MAX_PAGE_CHARS])
        return shorten(state.pages[key].text, state.limits.page_chars)

    def _read_with_tavily(self, url: str) -> tuple[str | None, str | None]:
        """Let Tavily fetch the page and ScrapeGraph extract the fields from its text.

        Raw text starts with menus, so it is only used as it is when all of it fits.
        """
        text, error = self._tavily_extract(url)
        if text is None:
            return None, error
        data, error = self._scrapegraph(markdown=text[:MAX_FALLBACK_CHARS])
        if data is None and error != NOT_A_PRODUCT:
            if len(text) <= self.run_state.limits.page_chars:
                return text, None
        return data, error

    def _scrapegraph(self, **source: str) -> tuple[str | None, str | None]:
        try:
            result = self.client.extract(EXTRACTION_PROMPT, schema=EXTRACTION_SCHEMA, **source)
        except Exception as exc:
            return None, str(exc)
        if result.status != "success" or result.data is None:
            return None, result.error or "no data"
        if result.data.json_data:
            if result.data.json_data.get("is_single_product") is False:
                return None, NOT_A_PRODUCT
            return json.dumps(result.data.json_data, ensure_ascii=False), None
        return (result.data.raw, None) if result.data.raw else (None, "empty page")

    def _tavily_extract(self, url: str) -> tuple[str | None, str | None]:
        try:
            response = self.fallback.extract(url, format="text")
        except Exception as exc:
            return None, str(exc)
        results = response.get("results") or []
        content = results[0].get("raw_content") if results else None
        return (content, None) if content else (None, "empty page")


class NoInput(BaseModel):
    pass


class RankingTool(StatefulTool):
    name: str = "get_product_ranking"
    description: str = (
        "Return the official ranking: scores, eligibility, issues and the recommended product."
    )
    args_schema: Type[BaseModel] = NoInput

    def _run(self) -> str:
        return ranking_json(self.run_state)


def ranking_json(state: RunState) -> str:
    best = state.recommended
    quantity = state.request.quantity
    return json.dumps(
        {
            "currency": state.company.currency,
            "quantity": quantity,
            "recommended_product_id": best.product.id if best else None,
            "products": [
                {
                    "id": r.product.id,
                    "rank": r.rank,
                    "eligible": r.eligible,
                    "title": r.product.title,
                    "website": domain_of(r.product.url),
                    "unit_price": r.unit_price,
                    "order_total": r.unit_price * quantity if r.unit_price else None,
                    "warranty": r.product.warranty,
                    "scores": {
                        "price": round(r.price_score, 1),
                        "specs": round(r.spec_score, 1),
                        "value": round(r.value_score, 1),
                        "total": round(r.total_score, 1),
                    },
                    "issues": r.issues,
                    "pros": r.assessment.pros,
                    "cons": r.assessment.cons,
                }
                for r in state.ranking
            ],
        },
        ensure_ascii=False,
    )
