"""Puts agents and tasks together and runs them sequentially."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from crewai import Crew, Process

from .agents import build_agents
from .config import Company, ConfigError, Limits, ProcurementRequest, build_llm
from .schemas import ReportContent
from .state import RunState
from .tasks import build_tasks


@dataclass
class Services:
    """The outside world: swapped for offline stand-ins by the demo and the tests."""

    llm: Any
    tavily: Any
    scrapegraph: Any

    @classmethod
    def live(cls) -> "Services":
        from scrapegraph_py import ScrapeGraphAI
        from tavily import TavilyClient

        return cls(
            llm=build_llm(),
            tavily=TavilyClient(api_key=os.environ["TAVILY_API_KEY"]),
            scrapegraph=ScrapeGraphAI(api_key=os.environ["SGAI_API_KEY"]),
        )


def run_procurement(
    request: ProcurementRequest,
    company: Company,
    services: Services,
    *,
    demo: bool = False,
    limits: Limits = Limits(),
    verbose: bool = True,
) -> tuple[RunState, ReportContent]:
    company.weights.normalised()  # fail before spending credits
    if request.max_products > limits.products:
        request = request.model_copy(update={"max_products": limits.products})
    state = RunState(company=company, request=request, demo=demo, limits=limits)
    agents = build_agents(
        state, services.llm, services.tavily, services.scrapegraph, verbose=verbose
    )
    tasks = build_tasks(state, agents)
    crew = Crew(
        agents=[t.agent for t in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=verbose,
        tracing=False,
    )
    result = crew.kickoff()
    content = result.pydantic
    if not isinstance(content, ReportContent):
        raise ConfigError("The report agent did not return valid report content.")
    return state, content
