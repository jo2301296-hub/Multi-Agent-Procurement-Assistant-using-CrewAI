"""Offline stand-ins so the real crew can run without API keys.

The agents, tasks, tools, guardrails, ranking and report are the real ones; only the
three outside services are scripted. All products and websites are invented (the
.example domain is reserved for that).
"""

from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

from .config import ProcurementRequest
from .crew import Services

DEMO_REQUEST = ProcurementRequest(
    product="Business laptop for developers",
    quantity=10,
    max_unit_price=45000,
    must_have=["16 GB RAM or more", "512 GB SSD or more"],
    nice_to_have=["Full HD screen or better"],
    max_products=5,
)

# title, url, price, in_stock, warranty, RAM, storage, screen
_OFFERS = [
    ("Orion WorkBook 14", "https://store-a.example/orion-workbook-14", 38500, True, "2 years", "16 GB", "512 GB SSD", "14 inch 1920x1200"),
    ("Lyra ProBook 15", "https://store-b.example/lyra-probook-15", 42900, True, "1 year", "32 GB", "1 TB SSD", "15.6 inch 2560x1440"),
    ("Vega Slim 13", "https://store-c.example/vega-slim-13", 31900, True, "1 year", "8 GB", "512 GB SSD", "13.3 inch 1920x1080"),
    ("Orion WorkBook 14", "https://store-c.example/orion-workbook-14", 39990, False, "2 years", "16 GB", "512 GB SSD", "14 inch 1920x1200"),
    ("Altair Studio 16", "https://store-d.example/altair-studio-16", 61500, True, "3 years", "32 GB", "1 TB SSD", "16 inch 3840x2400"),
]


def _page(offer: tuple) -> dict[str, Any]:
    title, url, price, stock, warranty, ram, ssd, screen = offer
    return {
        "title": title,
        "price": price,
        "currency": "EGP",
        "in_stock": stock,
        "seller": url.split("/")[2],
        "warranty": warranty,
        "specs": [
            {"name": "RAM", "value": ram},
            {"name": "Storage", "value": ssd},
            {"name": "Screen", "value": screen},
        ],
    }


class DemoTavily:
    def search(self, query: str, **_: Any) -> dict[str, Any]:
        return {
            "results": [
                {"title": o[0], "url": o[1], "content": f"{o[0]} for sale"} for o in _OFFERS
            ]
        }

    def extract(self, url: str, **_: Any) -> dict[str, Any]:
        return {"results": []}


class DemoScrapeGraph:
    def extract(self, prompt: str, *, url: str = "", **_: Any) -> SimpleNamespace:
        for offer in _OFFERS:
            if offer[1] == url:
                data = SimpleNamespace(json_data=_page(offer), raw=None)
                return SimpleNamespace(status="success", data=data, error=None)
        return SimpleNamespace(status="error", data=None, error="HTTP 404")


def _action(tool: str, **arguments: Any) -> str:
    return (
        f"Thought: I need {tool}.\nAction: {tool}\nAction Input: {json.dumps(arguments)}"
    )


def _answer(data: dict[str, Any]) -> str:
    return f"Thought: I now know the final answer.\nFinal Answer: {json.dumps(data)}"


def _check(requirement: str, ok: bool, evidence: str) -> dict[str, str]:
    return {
        "requirement_id": requirement,
        "status": "met" if ok else "unmet",
        "evidence": evidence,
    }


def demo_script() -> dict[str, list[str]]:
    """What the scripted LLM says, per task name."""
    products = [{"id": f"P{i}", "url": o[1], **_page(o)} for i, o in enumerate(_OFFERS, 1)]
    assessments = []
    for i, (title, _, _, _, warranty, ram, ssd, screen) in enumerate(_OFFERS, 1):
        enough_ram = int(ram.split()[0]) >= 16
        long_warranty = int(warranty.split()[0]) >= 2
        assessments.append(
            {
                "product_id": f"P{i}",
                "checks": [
                    _check("R1", enough_ram, ram),
                    _check("R2", True, ssd),
                    _check("R3", True, screen),
                    _check("R4", long_warranty, warranty),
                ],
                "pros": [f"{ssd} storage", f"{warranty} warranty"],
                "cons": [] if enough_ram else [f"Only {ram} RAM"],
            }
        )
    return {
        "plan": [
            _answer(
                {
                    "requirements": [
                        {"id": "R1", "text": "Warranty of 2 years or more", "must_have": False}
                    ],
                    "queries": [
                        "business laptop 16GB RAM 512GB SSD price Egypt",
                        "buy developer laptop Egypt EGP",
                        "لاب توب للمبرمجين سعر مصر",
                    ],
                }
            )
        ],
        "search": [
            _action("tavily_search", query="business laptop 16GB RAM 512GB SSD price Egypt"),
            _action("tavily_search", query="buy developer laptop Egypt EGP"),
            _action("tavily_search", query="لاب توب للمبرمجين سعر مصر"),
            _answer(
                {
                    "candidates": [
                        {"title": o[0], "url": o[1], "reason": "Single product page"}
                        for o in _OFFERS
                    ]
                }
            ),
        ],
        "scrape": [_action("scrape_product_page", url=o[1]) for o in _OFFERS]
        + [_answer({"products": products})],
        "analyse": [_answer({"assessments": assessments})],
        "report": [
            _action("get_product_ranking"),
            _answer(
                {
                    "executive_summary": (
                        "Five offers from four websites were compared. Two of them meet every "
                        "must-have, are in stock and fit the budget."
                    ),
                    "market_overview": (
                        "In-budget laptops with 16 GB RAM or more cost between 38,500 and "
                        "42,900 EGP; the cheaper model has only 8 GB RAM."
                    ),
                    "comparison_notes": [
                        "Orion WorkBook 14 is cheaper and has a 2-year warranty.",
                        "Lyra ProBook 15 doubles the RAM and storage for 4,400 EGP more.",
                        "The same Orion costs more on store-c.example and is out of stock there.",
                    ],
                    "recommended_product_id": "P1",
                    "recommendation": (
                        "Buy the Orion WorkBook 14 from store-a.example: it meets every "
                        "requirement at the lowest eligible price."
                    ),
                    "risks": ["Stock for the full quantity has not been confirmed."],
                    "next_steps": [
                        "Ask store-a.example for a written quote for 10 units.",
                        "Confirm delivery time and warranty terms.",
                    ],
                }
            ),
        ],
    }


class ScriptedLLM(BaseLLM):
    """Plays back prepared answers; uses the text (ReAct) tool protocol."""

    model: str = "scripted-demo"
    script: dict[str, list[str]] = {}
    _position: dict[str, int] = PrivateAttr(default_factory=lambda: defaultdict(int))

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        name = from_task.name
        lines = self.script[name]
        index = min(self._position[name], len(lines) - 1)  # repeat the last answer
        self._position[name] += 1
        return lines[index]

    async def acall(self, *args: Any, **kwargs: Any) -> str:
        return self.call(*args, **kwargs)

    def get_input_tokens_count(self, *args: Any, **kwargs: Any) -> int:
        return 0  # nothing is sent anywhere

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False


def demo_services(script: dict[str, list[str]] | None = None) -> Services:
    return Services(
        llm=ScriptedLLM(model="scripted-demo", provider="demo", script=script or demo_script()),
        tavily=DemoTavily(),
        scrapegraph=DemoScrapeGraph(),
    )
