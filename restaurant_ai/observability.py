"""
Observability with Arize Phoenix (open source, runs 100% locally).

Start the Phoenix UI in a separate terminal:   make phoenix
Then open http://localhost:6006 to see every agent step, LLM call and
MCP tool call as a trace (inputs, outputs, latency, token counts).

Calling `setup_tracing()` once at program start is all that is needed:
OpenInference auto-instruments LangChain/LangGraph, so we don't have to add
any tracing code inside the agents themselves.
"""

import urllib.request
import urllib.error
from contextlib import contextmanager

from opentelemetry import context as otel_context
from opentelemetry import trace

from restaurant_ai import config

_already_setup = False


def _phoenix_is_running() -> bool:
    try:
        with urllib.request.urlopen(config.PHOENIX_URL, timeout=1):
            pass
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def setup_tracing() -> bool:
    """Send traces to Phoenix if it is enabled and running. Returns True if tracing is on."""
    global _already_setup
    if _already_setup:
        return True
    if not config.TRACING_ENABLED:
        return False
    if not _phoenix_is_running():
        print(f"[tracing] Phoenix not reachable at {config.PHOENIX_URL} - tracing disabled. "
              "Run `make phoenix` in another terminal to enable it.")
        return False

    from phoenix.otel import register

    register(
        project_name=config.PHOENIX_PROJECT,
        endpoint=f"{config.PHOENIX_URL}/v1/traces",
        auto_instrument=True,  # picks up the LangChain instrumentor automatically
        batch=True,
        verbose=False,
    )
    _already_setup = True
    print(f"[tracing] Sending traces to Phoenix -> {config.PHOENIX_URL}")
    return True


@contextmanager
def new_trace(name: str, user_input: str = ""):
    """Start a brand-new Phoenix trace for one user action (one button click).

    Why this is needed: Gradio runs every click inside one long-lived web
    connection, and FastAPI (under Gradio) traces that connection. Without a
    fresh root span, all clicks would pile up inside ONE trace in Phoenix.
    `context=Context()` means "ignore any parent span - start a new trace".

    Usage:
        with new_trace("concierge-chat", message) as span:
            answer = ...
            span.set_attribute("output.value", answer)

    If tracing is off, OpenTelemetry returns a no-op span, so this is always safe.
    """
    tracer = trace.get_tracer("restaurant_ai")
    attributes = {"openinference.span.kind": "CHAIN", "input.value": user_input}
    with tracer.start_as_current_span(name, context=otel_context.Context(), attributes=attributes) as span:
        yield span
