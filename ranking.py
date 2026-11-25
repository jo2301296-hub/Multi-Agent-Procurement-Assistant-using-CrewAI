"""Deterministic scoring.

The analyst agent judges which requirements a product meets; the arithmetic is
done here so that the ranking is reproducible and cannot be hallucinated.

    price score = cheapest eligible price / product price (100 at most)
    spec score  = weighted share of requirements met (must-haves count double)
    value score = specs per unit of money, relative to the best eligible offer
    total       = weighted average, using the weights in company.yaml
"""

from __future__ import annotations

from .config import Company, ProcurementRequest
from .schemas import Assessment, Product, RankedProduct, Requirement

STATUS_POINTS = {"met": 1.0, "partial": 0.5, "unmet": 0.0, "unknown": 0.0}


def spec_score(assessment: Assessment, requirements: list[Requirement]) -> float:
    if not requirements:
        return 0.0
    status = {c.requirement_id: c.status for c in assessment.checks}
    earned = total = 0.0
    for req in requirements:
        weight = 2.0 if req.must_have else 1.0
        total += weight
        earned += weight * STATUS_POINTS[status.get(req.id, "unknown")]
    return 100 * earned / total


def to_company_currency(
    product: Product, company: Company
) -> tuple[float | None, str | None]:
    """Return (unit price in the company currency, problem)."""
    if product.price is None or product.price <= 0:
        return None, "No verified price on the page"
    currency = (product.currency or "").upper()
    if not currency:
        return None, "Currency not stated on the page"
    if currency == company.currency.upper():
        return product.price, None
    rates = {k.upper(): v for k, v in company.exchange_rates.items()}
    if currency in rates:
        return product.price * rates[currency], None
    return None, f"Priced in {currency}; no exchange rate configured"


def _issues(
    product: Product,
    assessment: Assessment,
    price: float | None,
    requirements: list[Requirement],
    request: ProcurementRequest,
) -> tuple[list[str], list[str]]:
    """Return (blocking issues, things to confirm before ordering)."""
    blocking: list[str] = []
    confirm: list[str] = []
    status = {c.requirement_id: c.status for c in assessment.checks}
    for req in requirements:
        if not req.must_have:
            continue
        state = status.get(req.id, "unknown")
        if state == "unmet":
            blocking.append(f"Must-have not met: {req.text}")
        elif state != "met":
            confirm.append(f"Confirm with seller: {req.text}")
    if product.in_stock is False:
        blocking.append("Out of stock")
    if price and request.max_unit_price and price > request.max_unit_price:
        blocking.append("Over budget")
    return blocking, confirm


def rank_products(
    products: list[Product],
    assessments: list[Assessment],
    requirements: list[Requirement],
    request: ProcurementRequest,
    company: Company,
) -> list[RankedProduct]:
    by_id = {a.product_id: a for a in assessments}
    ranked: list[RankedProduct] = []
    for product in products:
        assessment = by_id.get(product.id) or Assessment(
            product_id=product.id, checks=[], pros=[], cons=[]
        )
        price, price_problem = to_company_currency(product, company)
        blocking, confirm = _issues(product, assessment, price, requirements, request)
        if price_problem:
            blocking.insert(0, price_problem)
        ranked.append(
            RankedProduct(
                product=product,
                assessment=assessment,
                unit_price=price,
                spec_score=spec_score(assessment, requirements),
                eligible=not blocking,
                issues=blocking + confirm,
            )
        )

    priced = [r for r in ranked if r.unit_price]
    if priced:
        w_price, w_specs, w_value = company.weights.normalised()
        # Eligible offers set the scale: a cheap accessory scraped by mistake must not.
        reference = [r for r in priced if r.eligible] or priced
        cheapest = min(r.unit_price for r in reference)
        best_value = max(r.spec_score / r.unit_price for r in reference)
        for r in priced:
            r.price_score = min(100.0, 100 * cheapest / r.unit_price)
            if best_value:
                r.value_score = min(100.0, 100 * r.spec_score / r.unit_price / best_value)
            r.total_score = (
                w_price * r.price_score
                + w_specs * r.spec_score
                + w_value * r.value_score
            )

    # Eligible offers first, then priced but disqualified ones, then unpriced ones.
    ranked.sort(
        key=lambda r: (not r.eligible, r.unit_price is None, -r.total_score, r.product.id)
    )
    for position, r in enumerate((r for r in ranked if r.eligible), start=1):
        r.rank = position
    return ranked
