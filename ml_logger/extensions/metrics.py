"""Small stateful metric transforms for dashboard extensions."""


class ExponentialMovingAverage:
    """An extension for calculating EMA of live metrics."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._ema = None

    def update(self, value: float) -> float:
        """Add a sample and return the updated exponential moving average."""
        if self._ema is None:
            self._ema = value
        else:
            self._ema = (self.alpha * value) + ((1.0 - self.alpha) * self._ema)
        return self._ema
