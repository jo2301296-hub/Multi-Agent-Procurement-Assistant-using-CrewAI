"""Multi-agent procurement assistant built with CrewAI."""

import os

# Must be set before crewai is imported: keep runs private and non-interactive.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
