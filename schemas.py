"""Structured data exchanged between the agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Task 1: search plan -----------------------------------------------------
class Requirement(BaseModel):
    id: str = Field(description="Short id such as R1, R2")
    text: str = Field(description="One checkable specification, e.g. '16 GB RAM or more'")
    must_have: bool = Field(description="True if a product without it is unacceptable")


class SearchPlan(BaseModel):
    requirements: list[Requirement] = Field(description="Checkable product requirements")
    queries: list[str] = Field(description="Web search queries")


# --- Task 2: search ----------------------------------------------------------
class Candidate(BaseModel):
    title: str
    url: str = Field(description="URL exactly as returned by the search tool")
    reason: str = Field(description="Why this page is worth scraping")


class CandidateList(BaseModel):
    candidates: list[Candidate]


# --- Task 3: scraping --------------------------------------------------------
class Spec(BaseModel):
    name: str
    value: str


class Product(BaseModel):
    id: str = Field(description="Short id such as P1, P2")
    title: str
    url: str = Field(description="URL of the scraped page")
    price: float | None = Field(
        description="Current unit price as a plain number, null if the page shows none"
    )
    currency: str | None = Field(description="ISO 4217 code such as EGP, null if unknown")
    in_stock: bool | None = Field(description="null when the page does not say")
    seller: str | None
    warranty: str | None
    specs: list[Spec]


class ProductList(BaseModel):
    products: list[Product]


# --- Task 4: analysis --------------------------------------------------------
class RequirementCheck(BaseModel):
    requirement_id: str
    status: Literal["met", "partial", "unmet", "unknown"]
    evidence: str = Field(description="The scraped value that justifies the status")


class Assessment(BaseModel):
    product_id: str
    checks: list[RequirementCheck]
    pros: list[str]
    cons: list[str]


class AssessmentList(BaseModel):
    assessments: list[Assessment]


# --- Computed in Python from tasks 3 and 4 -----------------------------------
class RankedProduct(BaseModel):
    product: Product
    assessment: Assessment
    rank: int | None = None
    eligible: bool = False
    issues: list[str] = []
    unit_price: float | None = None  # in the company currency
    price_score: float = 0.0
    spec_score: float = 0.0
    value_score: float = 0.0
    total_score: float = 0.0


# --- Task 5: report ----------------------------------------------------------
class ReportContent(BaseModel):
    executive_summary: str = Field(description="One short paragraph for management")
    market_overview: str = Field(description="What the scraped offers look like as a group")
    comparison_notes: list[str] = Field(description="Key trade-offs between the products")
    recommended_product_id: str | None = Field(
        description="Id of the ranking's recommended product, null if it has none"
    )
    recommendation: str = Field(description="What to buy and why, or why nothing qualifies")
    risks: list[str]
    next_steps: list[str]
