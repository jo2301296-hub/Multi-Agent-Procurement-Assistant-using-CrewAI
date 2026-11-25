"""The five tasks, executed one after another."""

from __future__ import annotations

from crewai import Task

from .agents import ProcurementAgents
from .guardrails import Guardrails
from .schemas import AssessmentList, CandidateList, ProductList, ReportContent, SearchPlan
from .state import RunState


def build_tasks(state: RunState, agents: ProcurementAgents) -> list[Task]:
    request = state.request.as_prompt(state.company.currency)
    guard = Guardrails(state)
    limits = state.limits
    queries = "3" if limits.queries <= 3 else f"3 to {limits.queries}"

    plan = Task(
        name="plan",
        description=(
            f"Plan this purchase.\n\n{request}\n\n"
            "1. List the requirements as checkable specifications. Keep the user's must-have "
            "and nice-to-have items word for word and add only what a professional buyer "
            f"would check for this kind of product ({limits.requirements} in total at most). Never restate one of the user's items in "
            "other words. Price and stock are checked separately, so leave them out.\n"
            f"2. Write {queries} web search queries that lead to pages selling such products "
            "in the company's country. Use plain words, no operators such as site:. Vary brands, wording and, where the local market "
            "does so, language."
        ),
        expected_output="Requirements and search queries as JSON.",
        agent=agents.planner,
        output_pydantic=SearchPlan,
        guardrail=guard.plan,
    )
    search = Task(
        name="search",
        description=(
            f"Find product pages for this purchase.\n\n{request}\n\n"
            "Run every query from the plan with tavily_search. Then choose "
            f"{state.request.max_candidates} pages, best first, that each sell ONE matching "
            "product (some will fail to load, so spares are included). "
            "Skip category, search and listing pages (URLs such as /laptops, /search, /slp/ or "
            "/category/), reviews and articles, spread the choice over several "
            "websites, and copy each URL exactly as the tool returned it. Write the final answer "
            "as JSON text, not as a tool call."
        ),
        expected_output="The chosen product pages as JSON.",
        agent=agents.searcher,
        context=[plan],
        output_pydantic=CandidateList,
        guardrail=guard.candidates,
    )
    scrape = Task(
        name="scrape",
        description=(
            "Call scrape_product_page for the candidate URLs, one at a time and in order, until "
            f"{state.request.max_products} pages succeeded or none is left. Return one product per "
            "page that was scraped successfully. Copy values exactly: price is the current "
            "unit price as a plain number, currency is its ISO code, and anything the page "
            "does not state is null. Leave out pages that failed. Write the "
            "final answer as JSON text, not as a tool call."
        ),
        expected_output="The scraped products as JSON.",
        agent=agents.scraper,
        context=[search],
        output_pydantic=ProductList,
        guardrail=guard.products,
    )
    analyse = Task(
        name="analyse",
        description=(
            "Compare the scraped products. For EVERY product, give EVERY requirement of the "
            "plan a status: met, partial, unmet, or unknown when the page does not say, "
            "with the scraped value as evidence. Add 2 to 4 short pros and cons per product, "
            "taking the company's policy into account. Use the product and requirement ids. "
            "Do not compute scores; they are calculated automatically."
        ),
        expected_output="One assessment per product as JSON.",
        agent=agents.analyst,
        context=[plan, scrape],
        output_pydantic=AssessmentList,
        guardrail=guard.assessments,
    )
    report = Task(
        name="report",
        description=(
            f"Write the procurement report for this request.\n\n{request}\n\n"
            "First call get_product_ranking. Recommend exactly its recommended_product_id; "
            "if that is null, say plainly that no offer qualifies and what to change. Use "
            "only numbers from the ranking, mention unconfirmed points as risks, and keep "
            "every text short and concrete. Write the final answer as JSON text, not as a "
            "tool call."
        ),
        expected_output="The report content as JSON.",
        agent=agents.reporter,
        # The ranking tool already holds what the report needs.
        context=[plan, scrape, analyse] if limits.full_report_context else [],
        output_pydantic=ReportContent,
        guardrail=guard.report,
    )
    return [plan, search, scrape, analyse, report]
