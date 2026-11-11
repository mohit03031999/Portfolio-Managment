"""Build the T × N daily simple-return matrix."""

from __future__ import annotations

import pandas as pd

ALIGN_INTERSECTION = "intersection"
ALIGN_UNION = "union"
_ALIGNMENTS = {ALIGN_INTERSECTION, ALIGN_UNION}


def daily_returns(prices: pd.DataFrame, align: str = ALIGN_INTERSECTION) -> pd.DataFrame:
    """Convert adjusted closes into daily percentage changes.

    ``intersection`` keeps a date only when every ticker has a return that day.
    That is the dense ``T × N`` matrix most cross-sectional work expects.
    ``union`` keeps a date when at least one ticker traded, and leaves the
    other cells missing. Use it when the basket mixes exchanges whose holidays
    do not line up.
    """
    if align not in _ALIGNMENTS:
        choices = ", ".join(sorted(_ALIGNMENTS))
        raise ValueError(f"align must be one of: {choices}")
    if prices.empty:
        raise ValueError("Price matrix is empty.")

    returns = prices.sort_index().pct_change()
    returns = returns.dropna(axis=1, how="all")
    if returns.empty:
        raise ValueError(
            "No daily returns could be computed. Each symbol needs at least two prices."
        )
    how = "any" if align == ALIGN_INTERSECTION else "all"
    returns = returns.dropna(how=how)
    returns.index.name = "Date"
    if returns.empty:
        raise ValueError(
            "The return matrix is empty after dropping missing days. "
            "Widen the date range, or use align='union' when exchanges do not share a calendar."
        )
    return returns
