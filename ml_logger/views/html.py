"""Dependency-free HTML reports generated from persisted run events."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..events import EventKind, RunEvent
from ..storage import RunContext
from .selection import flatten_values, matches_metric

CHART_WIDTH = 760
CHART_HEIGHT = 240
CHART_PADDING = 36


class HtmlReportView:
    """Create a portable summary when a run reaches a terminal state."""

    def __init__(self, context: RunContext, settings: dict[str, Any]):
        self.context = context
        self.settings = settings
        filename = Path(settings.get("filename", "run_report.html")).name
        self.output_path = context.run_dir / "reports" / filename

    def on_event(self, event: RunEvent) -> None:
        """Generate the report on successful or failed completion."""
        if event.kind in {EventKind.RUN_COMPLETED, EventKind.RUN_FAILED}:
            self.generate()

    def generate(self) -> Path:
        """Render the current manifest and selected metric histories."""
        histories = self._metric_histories()
        latest_telemetry = self._latest_telemetry()
        document = _html_document(
            self.context.manifest,
            histories,
            latest_telemetry,
            self.settings,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(document, encoding="utf-8")
        return self.output_path

    def _metric_histories(self) -> dict[str, list[tuple[int, float]]]:
        """Load selected numeric metric histories from canonical events."""
        histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
        filters = self.settings.get("metrics", {})
        events = self.context.catalog.list_events(
            self.context.run_id,
            EventKind.METRICS,
        )
        for event_index, event in enumerate(events):
            step = event.step if event.step is not None else event_index
            for name, value in flatten_values(event.payload).items():
                if _is_number(value) and matches_metric(name, filters):
                    histories[name].append((step, float(value)))
        return dict(histories)

    def _latest_telemetry(self) -> dict[str, Any]:
        events = self.context.catalog.list_events(
            self.context.run_id,
            EventKind.TELEMETRY,
        )
        return flatten_values(events[-1].payload) if events else {}


def _html_document(
    manifest: dict[str, Any],
    histories: dict[str, list[tuple[int, float]]],
    telemetry: dict[str, Any],
    settings: dict[str, Any],
) -> str:
    """Assemble the complete portable run report document."""
    title = html.escape(f"{manifest['name']} · {manifest['run_id']}")
    summary = _summary_table(manifest, histories)
    visualization = settings.get("visualization", "auto")
    charts = _chart_section(histories, settings) if visualization != "table" else ""
    telemetry_table = _mapping_table("Latest telemetry", telemetry)
    manifest_json = html.escape(json.dumps(manifest, indent=2, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{_styles()}
</head>
<body>
<main>
<header><p class="eyebrow">ML Logger run report</p><h1>{title}</h1></header>
{summary}
{charts}
{telemetry_table}
<section><h2>Manifest</h2><pre>{manifest_json}</pre></section>
</main>
</body>
</html>
"""


def _summary_table(
    manifest: dict[str, Any],
    histories: dict[str, list[tuple[int, float]]],
) -> str:
    """Render lifecycle metadata and the latest value of each metric."""
    rows = {
        "Status": manifest["status"],
        "Started": manifest["started_at"],
        "Ended": manifest.get("ended_at") or "running",
        "Run type": manifest["run_type"],
        "Metrics": len(histories),
    }
    rows.update(
        {
            f"Latest · {name}": _format_number(points[-1][1])
            for name, points in histories.items()
            if points
        }
    )
    return _mapping_table("Summary", rows)


def _chart_section(
    histories: dict[str, list[tuple[int, float]]],
    settings: dict[str, Any],
) -> str:
    """Render up to the configured number of metric history charts."""
    max_charts = int(settings.get("max_charts", 20))
    charts = [
        _line_chart(name, points)
        for name, points in list(histories.items())[:max_charts]
        if len(points) > 1
    ]
    if not charts:
        return ""
    return "<section><h2>Metric history</h2>" + "".join(charts) + "</section>"


def _line_chart(name: str, points: list[tuple[int, float]]) -> str:
    """Render one metric history as an inline SVG polyline."""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    coordinates = [
        (
            _scale(x, min(x_values), max(x_values), CHART_WIDTH),
            _scale(y, min(y_values), max(y_values), CHART_HEIGHT, invert=True),
        )
        for x, y in points
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    label = html.escape(name)
    latest = _format_number(y_values[-1])
    return f"""<article class="chart">
<div class="chart-title"><h3>{label}</h3><span>{latest}</span></div>
<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" aria-label="{label}">
<line class="axis" x1="{CHART_PADDING}" y1="{CHART_PADDING}"
 x2="{CHART_PADDING}" y2="{CHART_HEIGHT - CHART_PADDING}"/>
<line class="axis" x1="{CHART_PADDING}" y1="{CHART_HEIGHT - CHART_PADDING}"
 x2="{CHART_WIDTH - CHART_PADDING}" y2="{CHART_HEIGHT - CHART_PADDING}"/>
<polyline class="series" points="{polyline}"/>
</svg></article>"""


def _scale(
    value: float,
    minimum: float,
    maximum: float,
    extent: int,
    invert: bool = False,
) -> float:
    usable = extent - 2 * CHART_PADDING
    ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    if invert:
        ratio = 1 - ratio
    return CHART_PADDING + ratio * usable


def _mapping_table(title: str, values: dict[str, Any]) -> str:
    if not values:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<section><h2>{html.escape(title)}</h2><table>{rows}</table></section>"


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _styles() -> str:
    """Return the self-contained report stylesheet."""
    return """<style>
:root { color-scheme: dark; --bg:#0d1020; --card:#171b31; --ink:#eef1ff;
--muted:#9ba5c9; --cyan:#59d9e8; --magenta:#cf7cff; --line:#343b61; }
* { box-sizing:border-box; }
body { margin:0; background:radial-gradient(circle at top,#202750,var(--bg) 42%);
color:var(--ink); font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; }
main { width:min(1100px,calc(100% - 32px)); margin:48px auto 96px; }
header { margin-bottom:28px; } h1 { margin:.2rem 0; font-size:clamp(1.7rem,4vw,3rem); }
h2 { margin-top:0; } .eyebrow { color:var(--cyan); text-transform:uppercase;
letter-spacing:.16em; font-weight:700; }
section,.chart { background:color-mix(in srgb,var(--card) 92%,transparent);
border:1px solid var(--line); border-radius:16px; padding:20px; margin:16px 0; }
table { width:100%; border-collapse:collapse; } th,td { padding:8px 10px;
border-bottom:1px solid var(--line); text-align:left; } th { color:var(--muted); }
.chart-title { display:flex; justify-content:space-between; align-items:center; }
.chart-title h3 { margin:0; } .chart-title span { color:var(--cyan); font-weight:700; }
svg { width:100%; height:auto; overflow:visible; }
.axis { stroke:var(--line); stroke-width:1; }
.series { fill:none; stroke:var(--magenta); stroke-width:3;
stroke-linejoin:round; stroke-linecap:round; }
pre { overflow:auto; color:#d8defa; background:#0a0d19; padding:16px; border-radius:10px; }
</style>"""
