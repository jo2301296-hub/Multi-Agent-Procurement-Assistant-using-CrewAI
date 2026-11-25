"""Company context, procurement request and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPANY_FILE = ROOT / "config" / "company.yaml"
DEFAULT_MODEL = "groq/openai/gpt-oss-120b"

# Provider prefix -> environment variables, any one of which is enough.
PROVIDER_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


class ConfigError(RuntimeError):
    """The user has to fix something in the configuration."""


class Weights(BaseModel):
    price: float = Field(0.35, ge=0)
    specs: float = Field(0.40, ge=0)
    value: float = Field(0.25, ge=0)

    def normalised(self) -> tuple[float, float, float]:
        total = self.price + self.specs + self.value
        if total <= 0:
            raise ConfigError("At least one weight in company.yaml must be above 0.")
        return self.price / total, self.specs / total, self.value / total


class Company(BaseModel):
    name: str
    industry: str = ""
    country: str = ""
    search_country: str = ""
    size: str = ""
    currency: str = "USD"
    about: str = ""
    procurement_policy: list[str] = []
    preferred_websites: list[str] = []
    excluded_websites: list[str] = []
    weights: Weights = Weights()
    exchange_rates: dict[str, float] = {}

    @classmethod
    def load(cls, path: Path = DEFAULT_COMPANY_FILE) -> "Company":
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise ConfigError(f"Company file not found: {path}") from None
        except (yaml.YAMLError, ValueError) as exc:
            raise ConfigError(f"Invalid company file {path}: {exc}") from exc

    def as_prompt(self) -> str:
        policy = "\n".join(f"  - {rule}" for rule in self.procurement_policy)
        return (
            f"Company: {self.name} ({self.industry}, {self.size}) in {self.country}.\n"
            f"About: {self.about.strip()}\n"
            f"Currency: {self.currency}\n"
            f"Procurement policy:\n{policy}"
        )


# Extra pages to try when a site blocks scraping or a page turns out to be a listing.
SPARE_PAGES = 4


class ProcurementRequest(BaseModel):
    product: str = Field(min_length=2)
    quantity: int = Field(1, ge=1)
    max_unit_price: float | None = Field(None, gt=0)
    must_have: list[str] = []
    nice_to_have: list[str] = []
    max_products: int = Field(6, ge=2, le=20)

    @property
    def max_candidates(self) -> int:
        return self.max_products + SPARE_PAGES

    def as_prompt(self, currency: str) -> str:
        budget = (
            f"{self.max_unit_price:,.2f} {currency} per unit"
            if self.max_unit_price
            else "no limit given"
        )
        return (
            f"Product: {self.product}\nQuantity: {self.quantity}\nBudget: {budget}\n"
            f"Must have: {'; '.join(self.must_have) or 'not specified'}\n"
            f"Nice to have: {'; '.join(self.nice_to_have) or 'not specified'}"
        )


@dataclass(frozen=True)
class Limits:
    """How much text is put in front of the LLM."""

    products: int = 20
    queries: int = 5
    requirements: int = 12
    search_results: int = 8
    snippet_chars: int = 400
    page_chars: int = 12_000
    output_tokens: int = 16_000  # the Anthropic SDK insists on streaming above ~21k
    full_report_context: bool = True


# Fits a limit of 8,000 tokens per minute, the base limit Groq lists for gpt-oss.
COMPACT = Limits(
    products=4,
    queries=3,
    requirements=6,
    search_results=6,
    snippet_chars=150,
    page_chars=1_200,
    output_tokens=3_000,
    full_report_context=False,
)


def limits_for(model: str) -> Limits:
    """Groq gets the compact limits unless COMPACT=false; others on COMPACT=true."""
    setting = os.getenv("COMPACT", "").strip().lower()
    if setting in ("true", "1", "yes"):
        return COMPACT
    if setting in ("false", "0", "no"):
        return Limits()
    return COMPACT if model.startswith("groq/") else Limits()


def missing_keys(model: str) -> list[str]:
    """Names of the environment variables a live run still needs."""
    missing = [k for k in ("TAVILY_API_KEY", "SGAI_API_KEY") if not os.getenv(k)]
    options = PROVIDER_KEYS.get(model.split("/")[0], ())
    if options and not any(os.getenv(k) for k in options):
        missing.insert(0, " or ".join(options))
    return missing


def current_model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULT_MODEL


def build_llm():
    """The LLM shared by all agents, chosen by the LLM_MODEL variable."""
    model = current_model()
    output_tokens = limits_for(model).output_tokens
    provider, _, name = model.partition("/")
    if provider == "groq":
        from .groq_llm import build_groq_llm

        key = os.getenv("GROQ_API_KEY")
        if not key or not name:
            raise ConfigError("Groq needs GROQ_API_KEY and LLM_MODEL=groq/<model-name>.")
        return build_groq_llm(name, key, output_tokens)

    from crewai import LLM

    # No temperature: newer Claude models reject sampling parameters.
    return LLM(model=model, max_tokens=output_tokens)
