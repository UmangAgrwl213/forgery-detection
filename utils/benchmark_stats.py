"""Uncertainty for image-level benchmark proportions."""

import math


def binomial_summary(positive, total):
    """Return a count and a two-sided 95% Wilson score interval."""
    if not 0 <= positive <= total or total == 0:
        raise ValueError("Expected 0 <= positive <= total and a nonempty sample")
    z = 1.959963984540054
    rate = positive / total
    adjustment = z * z / total
    center = (rate + adjustment / 2) / (1 + adjustment)
    radius = z * math.sqrt((rate * (1 - rate) + adjustment / 4) / total) / (1 + adjustment)
    return {"count": positive, "total": total,
            "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)]}
