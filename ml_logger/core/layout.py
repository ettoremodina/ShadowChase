"""Rich renderables composing the command-center dashboard."""

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LogStreamPanel:
    """Render the newest wrapped log rows inside the available panel height."""

    def __init__(self, logs: list[Text]):
        self.logs = logs

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        inner_width = max(1, options.max_width - 2)
        panel_height = options.height or console.height
        inner_height = max(1, panel_height - 2)
        content = Text("\n").join(self.logs)
        wrapped_lines = content.wrap(console, inner_width)
        visible_lines = wrapped_lines[-inner_height:]
        yield Panel(
            Text("\n").join(visible_lines),
            title="Log Stream",
            border_style="white",
        )


def create_command_center_layout() -> Layout:
    """Create the command-center layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="main_split"),
        Layout(name="footer", size=3, visible=False),
    )
    layout["main_split"].split_row(
        Layout(name="logs", ratio=2),
        Layout(name="sidebar", ratio=1),
    )
    layout["main_split"]["sidebar"].split_column(
        Layout(name="metrics", ratio=3),
        Layout(name="hardware", ratio=2),
    )
    return layout


def render_metrics_table(metrics: dict) -> Table:
    """Render a flat metric mapping into a Rich table."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right")
    for name, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(name, formatted)
    return table


def render_hardware_table(telemetry: dict) -> Table:
    """Render selected structured telemetry values."""
    table = Table(box=box.SIMPLE, expand=True, show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Usage", justify="right", style="green")
    for name, value in telemetry.items():
        table.add_row(_telemetry_label(name), _telemetry_value(name, value))
    return table


def render_logs_panel(log_queue: list[Text]) -> LogStreamPanel:
    """Create a height-aware panel that follows the tail of the log stream."""
    return LogStreamPanel(log_queue)


def _telemetry_label(name: str) -> str:
    return name.replace("_", " ").replace("/", " · ")


def _telemetry_value(name: str, value) -> str:
    """Format numeric telemetry using the unit encoded in its name."""
    if not isinstance(value, (int, float)):
        return str(value)
    if name.endswith("_percent"):
        return f"{value:.1f}%"
    if name.endswith("_mb"):
        return f"{value:.1f} MB"
    if name.endswith("_c"):
        return f"{value:.1f} °C"
    return f"{value:.2f}" if isinstance(value, float) else str(value)
