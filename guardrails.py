"""Checks that run on each task's output before the next agent sees it.

A guardrail returns (True, cleaned result) or (False, feedback); on False CrewAI
sends the feedback to the agent and lets it try again.
"""

# No "from __future__ import annotations" here: CrewAI inspects the real return types.
from typing import Any, TypeVar

from crewai.tasks.task_output import TaskOutput
from pydantic import BaseModel, ValidationError

from .ranking import rank_products
from .schemas import (
    Assessment,
    AssessmentList,
    Candidate,
    CandidateList,
    Product,
    ProductList,
    ReportContent,
    Requirement,
    RequirementCheck,
    SearchPlan,
)
from .state import RunState, domain_of, normalise_url, spread_over_websites

T = TypeVar("T", bound=BaseModel)


def parse_output(output: TaskOutput, model: type[T]) -> T:
    """Read the agent's answer, tolerating text or code fences around the JSON."""
    if isinstance(output.pydantic, model):
        return output.pydantic
    raw = output.raw
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise ValueError("The answer contains no JSON object.")
    try:
        return model.model_validate_json(raw[start : end + 1])
    except ValidationError as exc:
        raise ValueError(f"The JSON does not match the required schema: {exc}") from exc


class Guardrails:
    def __init__(self, state: RunState):
        self.state = state
        self._asked: set[str] = set()

    def _ask_once(self, topic: str) -> bool:
        """True the first time only: worth one retry, not worth failing the run."""
        if topic in self._asked:
            return False
        self._asked.add(topic)
        return True

    # --- Task 1 --------------------------------------------------------------
    def plan(self, output: TaskOutput) -> tuple[bool, Any]:
        try:
            plan = parse_output(output, SearchPlan)
        except ValueError as exc:
            return False, str(exc)
        queries = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))
        queries = queries[: self.state.limits.queries]
        if not queries:
            return False, "Provide at least 3 search queries."

        # What the user asked for always comes first and cannot be rewritten.
        request = self.state.request
        merged: list[tuple[str, bool]] = [(t, True) for t in request.must_have]
        merged += [(t, False) for t in request.nice_to_have]
        known = {text.casefold() for text, _ in merged}
        for req in plan.requirements:
            if req.text.casefold() not in known:
                known.add(req.text.casefold())
                # With explicit must-haves from the user, extras are only nice to have.
                merged.append((req.text, req.must_have and not request.must_have))
        if not merged:
            return False, "Provide at least 3 checkable requirements."
        limit = max(self.state.limits.requirements, len(request.must_have) + len(request.nice_to_have))
        requirements = [
            Requirement(id=f"R{i}", text=text, must_have=must)
            for i, (text, must) in enumerate(merged[:limit], start=1)
        ]
        self.state.requirements = requirements
        return True, SearchPlan(requirements=requirements, queries=queries)

    # --- Task 2 --------------------------------------------------------------
    def candidates(self, output: TaskOutput) -> tuple[bool, Any]:
        try:
            answer = parse_output(output, CandidateList)
        except ValueError as exc:
            return False, str(exc)
        state = self.state
        kept: dict[str, Candidate] = {}
        for candidate in answer.candidates:
            key = normalise_url(candidate.url)
            if key not in state.found_urls:
                state.warn(f"Ignored a URL that no search returned: {candidate.url}")
                continue
            kept.setdefault(
                key, candidate.model_copy(update={"url": state.found_urls[key]})
            )
        selected = dict(list(kept.items())[: state.request.max_candidates])
        if not selected:
            return False, (
                "None of these URLs was returned by tavily_search. "
                "Run the searches and copy the URLs exactly."
            )
        websites = {domain_of(key) for key in selected}
        available = {domain_of(key) for key in state.found_urls}
        if len(websites) < 2 <= len(available) and self._ask_once("websites"):
            return False, "Choose product pages from at least two different websites."
        wanted = min(state.request.max_candidates, len(state.found_urls))
        if len(selected) < wanted and self._ask_once("count"):
            return False, (
                f"Only {len(selected)} pages. Some will fail to load, so choose up to {wanted}; "
                "run another search if you need more single-product pages."
            )
        state.candidates = spread_over_websites(list(selected))
        return True, CandidateList(candidates=[selected[key] for key in state.candidates])

    # --- Task 3 --------------------------------------------------------------
    def products(self, output: TaskOutput) -> tuple[bool, Any]:
        try:
            answer = parse_output(output, ProductList)
        except ValueError as exc:
            return False, str(exc)
        state = self.state
        not_tried = [
            state.found_urls[c]
            for c in state.candidates
            if c not in state.pages and c not in state.failed_pages
        ]
        missing = state.products_missing
        if missing and not_tried and self._ask_once("not_tried"):
            return False, (
                f"{missing} more products are needed. Also scrape: " + ", ".join(not_tried)
            )
        if missing:
            for url in not_tried:
                state.warn(f"{domain_of(url)} was selected but never scraped.")

        kept: dict[str, Product] = {}
        for product in answer.products:
            key = normalise_url(product.url)
            page = state.pages.get(key)
            if page is None:
                state.warn(f"Dropped '{product.title}': its page was not scraped.")
                continue
            if key in kept:
                continue
            update: dict[str, Any] = {"id": f"P{len(kept) + 1}", "url": state.found_urls[key]}
            if product.price is not None and not page.shows_price(product.price):
                state.warn(
                    f"Removed the price of '{product.title}': "
                    f"{product.price:,.2f} does not appear on its page."
                )
                update["price"] = None
            kept[key] = product.model_copy(update=update)
        if not kept:
            if state.candidates and all(c in state.failed_pages for c in state.candidates):
                errors = "; ".join(sorted(set(state.failed_pages.values())))
                raise RuntimeError(f"No candidate page could be scraped: {errors}")
            return False, "Return one product for every page you scraped, with its exact URL."
        state.products = list(kept.values())
        return True, ProductList(products=state.products)

    # --- Task 4 --------------------------------------------------------------
    def assessments(self, output: TaskOutput) -> tuple[bool, Any]:
        try:
            answer = parse_output(output, AssessmentList)
        except ValueError as exc:
            return False, str(exc)
        state = self.state
        requirement_ids = [r.id for r in state.requirements]
        by_id = {a.product_id: a for a in answer.assessments}
        gaps: list[str] = []
        complete: list[Assessment] = []
        for product in state.products:
            found = by_id.get(product.id)
            checks = {c.requirement_id: c for c in found.checks} if found else {}
            missing = [r for r in requirement_ids if r not in checks]
            if missing:
                gaps.append(f"{product.id}: {', '.join(missing)}")
            complete.append(
                Assessment(
                    product_id=product.id,
                    checks=[
                        checks.get(r)
                        or RequirementCheck(
                            requirement_id=r, status="unknown", evidence="Not assessed"
                        )
                        for r in requirement_ids
                    ],
                    pros=found.pros if found else [],
                    cons=found.cons if found else [],
                )
            )
        if gaps and self._ask_once("gaps"):
            return False, "These requirement checks are missing: " + "; ".join(gaps)
        state.ranking = rank_products(
            state.products, complete, state.requirements, state.request, state.company
        )
        return True, AssessmentList(assessments=complete)

    # --- Task 5 --------------------------------------------------------------
    def report(self, output: TaskOutput) -> tuple[bool, Any]:
        try:
            content = parse_output(output, ReportContent)
        except ValueError as exc:
            return False, str(exc)
        best = self.state.recommended
        expected = best.product.id if best else None
        if (content.recommended_product_id or None) != expected:
            if expected is None:
                return False, (
                    "No product qualifies. Set recommended_product_id to null and say "
                    "clearly that no offer can be recommended."
                )
            return False, f"The ranking recommends {expected}. Recommend that product."
        return True, content
