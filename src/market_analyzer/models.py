from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MarketData:
    """Prices and the aligned daily-return matrix for one user-defined basket."""

    name: str
    start: str
    end: str | None
    requested_tickers: list[str]
    missing_tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(T, N)`` for the return matrix."""
        return self.returns.shape
