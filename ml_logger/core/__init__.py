"""Internal Rich components.

``DashboardLogger`` remains available lazily to avoid importing the public
runtime while its logging handlers are being initialized.
"""

__all__ = ["DashboardLogger"]


def __getattr__(name):
    if name == "DashboardLogger":
        from .logger import DashboardLogger

        return DashboardLogger
    raise AttributeError(name)
