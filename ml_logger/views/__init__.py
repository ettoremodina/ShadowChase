"""Built-in event views."""

from .html import HtmlReportView
from .terminal import LiveDashboard, resolve_dashboard_mode

__all__ = ["HtmlReportView", "LiveDashboard", "resolve_dashboard_mode"]
