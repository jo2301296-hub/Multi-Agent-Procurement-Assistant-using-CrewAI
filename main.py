"""Command line entry point.

    python main.py --demo
    python main.py "office laptop" --quantity 10 --budget 45000 --must "16 GB RAM or more"
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import procurement  # noqa: F401  (sets the privacy variables before crewai loads)
from dotenv import load_dotenv
from pydantic import ValidationError

from procurement.config import (
    DEFAULT_COMPANY_FILE,
    ROOT,
    Company,
    ConfigError,
    Limits,
    ProcurementRequest,
    current_model,
    limits_for,
    missing_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent procurement assistant")
    parser.add_argument("product", nargs="?", help="what to buy")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--budget", type=float, help="maximum unit price, company currency")
    parser.add_argument("--must", action="append", default=[], help="must-have (repeatable)")
    parser.add_argument("--nice", action="append", default=[], help="nice-to-have (repeatable)")
    parser.add_argument("--max-products", type=int, default=6, help="products to compare (2-20)")
    parser.add_argument("--company", type=Path, default=DEFAULT_COMPANY_FILE)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--demo", action="store_true", help="run offline with invented data")
    parser.add_argument("--open", action="store_true", help="open the report when done")
    parser.add_argument("--quiet", action="store_true", help="hide the agents' log")
    args = parser.parse_args()
    if not args.demo and not args.product:
        parser.error("say what to buy, or use --demo")
    return args


def run(args: argparse.Namespace) -> Path:
    company = Company.load(args.company)

    limits = Limits() if args.demo else limits_for(current_model())
    if args.demo:
        request = None
    else:
        try:
            request = ProcurementRequest(
                product=args.product,
                quantity=args.quantity,
                max_unit_price=args.budget,
                must_have=args.must,
                nice_to_have=args.nice,
                max_products=args.max_products,
            )
        except ValidationError as exc:
            problems = [f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()]
            raise ConfigError("; ".join(problems)) from exc
        missing = missing_keys(current_model())
        if missing:
            raise ConfigError(
                "Missing in .env: " + ", ".join(missing) + " (see .env.example). "
                "To try the system without keys: python main.py --demo"
            )

        if request.max_products > limits.products:
            print(f"Compact mode: comparing at most {limits.products} products.")

    # Imported late: crewai is slow to load and not needed for the checks above.
    from procurement.crew import Services, run_procurement
    from procurement.report import save_report

    if args.demo:
        from procurement.demo import DEMO_REQUEST, demo_services

        request, services = DEMO_REQUEST, demo_services()
    else:
        services = Services.live()

    state, content = run_procurement(
        request, company, services, demo=args.demo, limits=limits, verbose=not args.quiet
    )
    return save_report(state, content, args.out)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
        stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env")
    args = parse_args()
    try:
        path = run(args)
    except ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"The run failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nReport: {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
