# Multi-Agent Procurement Assistant (CrewAI)

Five AI agents search for a product, scrape several stores, compare price and
specifications, and write a procurement report in HTML with a recommendation.

**Demo:** [video](demo_video/procurement-demo.mp4) |
[report from a real run](reports/business-laptop-20261125-062606.html) |
[report with invented demo data](reports/demo-report.html) (download the HTML files to view them)

## How it works

The agents run one after another (`Process.sequential`). Each answer is validated by a
guardrail before the next agent sees it.

| # | Agent | Tool | Produces | Guardrail |
|---|-------|------|----------|-----------|
| 1 | Procurement Planner | none | Requirements and search queries | Keeps the user's requirements unchanged |
| 2 | Product Search Specialist | Tavily Search | Product page URLs | URL must come from a search; at least 2 websites when available |
| 3 | Web Scraping Specialist | ScrapeGraph (falls back to Tavily Extract) | Prices and specs | Page must have been scraped; price must match it |
| 4 | Product Comparison Analyst | none | met / partial / unmet / unknown per requirement | Every product against every requirement |
| 5 | Procurement Report Writer | Ranking lookup | Report text | Must recommend the top-ranked offer |

Scores are calculated in Python ([ranking.py](procurement/ranking.py)), not by the LLM, so
the same data always gives the same ranking:

- **Price** = cheapest eligible price / this price
- **Specs** = share of requirements met (must-haves count double)
- **Value** = specs per unit of money, relative to the best eligible offer
- **Total** = weighted average (weights in `config/company.yaml`)

An offer is not eligible if it is over budget, out of stock, misses a must-have, or has no
verified price in a known currency.

## Setup

Needs Python 3.12 or 3.13.

```powershell
py -3.13 -m venv .venv
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
Copy-Item .env.example .env
```

Put your keys in `.env`: one LLM provider, [Tavily](https://app.tavily.com) and
[ScrapeGraph](https://scrapegraphai.com/dashboard). `LLM_MODEL` selects the provider.

### Groq

`.env.example` is set up for Groq with `groq/openai/gpt-oss-120b`. Use a model that supports
*strict structured outputs* (`openai/gpt-oss-120b` or `openai/gpt-oss-20b`); the Llama models
on Groq do not. Groq's base limit for these is 8,000 tokens per minute, so runs use a compact
mode (at most 4 products, 3 queries, shortened pages) and wait whenever Groq says so. If your
account has higher limits, set `COMPACT=false`.

## Run

Without keys, using invented data (real crew, tools, guardrails and report; scripted
services):

```powershell
.venv\Scripts\python.exe main.py --demo --open
```

For real:

```powershell
.venv\Scripts\python.exe main.py "business laptop" --quantity 10 --budget 45000 --must "16 GB RAM or more" --must "512 GB SSD or more" --nice "Full HD screen" --open
```

`--max-products` sets how many products are compared. Up to 4 spare pages are tried when a
site blocks scraping; every attempt can cost ScrapeGraph credits.
`--quiet` hides the agents' log. Reports are saved in `reports/`.

## Company context

Edit [config/company.yaml](config/company.yaml): company profile, currency, procurement
policy, preferred or excluded websites, score weights and exchange rates.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

They run offline and include full crew runs with hallucinated prices and products, a wrong
recommendation, and failing scrapes.

## Structure

```text
main.py                 command line
config/company.yaml     company context
procurement/
  agents.py  tasks.py  crew.py    the workflow
  tools.py                        Tavily, ScrapeGraph, ranking lookup
  guardrails.py  state.py         validation against what the tools saw
  schemas.py  ranking.py          data models and scoring
  report.py  templates/           HTML report
  groq_llm.py                     Groq connection
  demo.py                         offline stand-ins
tests/
```
