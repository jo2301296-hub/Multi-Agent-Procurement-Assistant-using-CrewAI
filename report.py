"""Renders the final HTML report."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from .schemas import ReportContent
from .state import RunState, domain_of, normalise_url


def _money(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def render_report(state: RunState, content: ReportContent) -> str:
    env = Environment(
        loader=PackageLoader("procurement", "templates"),
        autoescape=True,  # scraped text is untrusted
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["money"] = _money
    env.filters["domain"] = domain_of
    priced = [r for r in state.ranking if r.unit_price]
    w_price, w_specs, w_value = state.company.weights.normalised()
    return env.get_template("report.html.j2").render(
        state=state,
        company=state.company,
        request=state.request,
        content=content,
        best=state.recommended,
        ranking=state.ranking,
        max_price=max((r.unit_price for r in priced), default=0),
        chart=sorted(priced, key=lambda r: r.unit_price),
        websites=sorted({domain_of(r.product.url) for r in state.ranking}),
        weights={"price": w_price, "specs": w_specs, "value": w_value},
        page_of=lambda product: state.pages[normalise_url(product.url)],
        generated=datetime.now().strftime("%d %B %Y, %H:%M"),
    )


def save_report(state: RunState, content: ReportContent, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w]+", "-", state.request.product.lower()).strip("-")[:40] or "report"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{'demo-' if state.demo else ''}{slug}-{stamp}.html"
    path.write_text(render_report(state, content), encoding="utf-8")
    return path
