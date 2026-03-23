"""Formatting templates for injecting SQL metric results into LLM prompts."""

from .metric_store import MetricResult

SQL_CONTEXT_TEMPLATE = """## Structured Metrics (from your tracking databases)

The following are EXACT numbers from your personal databases — these are facts, not estimates.

{metrics}
"""


def format_metric_results(results: list[MetricResult]) -> str:
    """Format a list of MetricResults into the SQL context block for prompt injection."""
    if not results:
        return ""

    sections = []
    for r in results:
        if r.summary:
            sections.append(f"### {r.description} [{r.time_range}]\n{r.summary}")

    if not sections:
        return ""

    metrics_text = "\n\n".join(sections)
    return SQL_CONTEXT_TEMPLATE.format(metrics=metrics_text)
