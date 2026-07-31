"""Extension hooks for optional dashboard integrations."""

from typing import Any, Dict

class LoggerPlugin:
    """Base class for ML Logger extensions that hook into the event loop."""
    
    def on_startup(self, config: Dict[str, Any]):
        """Called when the logger starts."""
        pass
        
    def on_metric_update(self, category: str, name: str, value: Any):
        """Called whenever a metric is logged."""
        pass
        
    def on_shutdown(self):
        """Called when the logger stops."""
        pass
