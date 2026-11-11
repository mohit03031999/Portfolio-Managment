"""Download adjusted close prices from Yahoo Finance."""

from __future__ import annotations

import warnings

import pandas as pd


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Download adjusted closes for ``tickers``.

    ``auto_adjust`` is off so the frame still contains the ``Adj Close`` field
    from the original workflow. Current yfinance builds adjust ``Close`` in
    place and omit ``Adj Close`` when that flag is left on.

    The returned frame is indexed by date and has one column per ticker that
    actually came back. The second value lists requested symbols with no prices.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. From the project root run: pip install -e ."
        ) from exc

    kwargs: dict[str, object] = {
        "start": start,
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "group_by": "column",
        "threads": True,
        "interval": "1d",
    }
    if end:
        kwargs["end"] = end

    try:
        # yfinance still sorts and renames with inplace=True, which pandas 3 warns about.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The inplace keyword .* is deprecated")
            raw = yf.download(tickers, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Yahoo Finance download failed "
            f"({exc}). Check the symbols, the date range, and the network."
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no prices. Check each symbol on finance.yahoo.com."
        )
    return extract_adjusted_close(raw, tickers)


def extract_adjusted_close(
    raw: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Pull a date × ticker adjusted-close frame out of a yfinance download."""
    if not tickers:
        raise ValueError("No tickers provided.")
    prices = _close_frame(raw, tickers)
    prices.columns = [str(column) for column in prices.columns]
    by_upper = {column.upper(): column for column in prices.columns}

    found: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in tickers:
        column = by_upper.get(ticker.upper())
        if column is None:
            missing.append(ticker)
            continue
        series = prices[column]
        if series.isna().all():
            missing.append(ticker)
            continue
        found[ticker] = series

    if not found:
        raise RuntimeError(
            "None of the requested symbols returned prices: " + ", ".join(tickers)
        )

    frame = pd.DataFrame(found)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame.index.name = "Date"
    return frame, missing


def _close_frame(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return a frame whose columns are ticker symbols, not OHLC field names."""
    field = _preferred_field(_column_names(raw))
    if isinstance(raw.columns, pd.MultiIndex):
        selected = _select_field(raw, field)
        if isinstance(selected, pd.Series):
            label = str(selected.name) if selected.name else tickers[0]
            if label == field:
                label = tickers[0]
            return selected.to_frame(name=label)
        return selected.copy()

    names = {str(column) for column in raw.columns}
    # A single symbol comes back as Open/High/Low/Close/Adj Close/Volume.
    if len(tickers) == 1 and field in names:
        prices = raw[[field]].copy()
        prices.columns = [tickers[0]]
        return prices
    return raw.copy()


def _column_names(raw: pd.DataFrame) -> set[str]:
    if isinstance(raw.columns, pd.MultiIndex):
        names: set[str] = set()
        for level in range(raw.columns.nlevels):
            names.update(str(name) for name in raw.columns.get_level_values(level))
        return names
    return {str(name) for name in raw.columns}


def _preferred_field(names: set[str]) -> str:
    if "Adj Close" in names:
        return "Adj Close"
    if "Close" in names:
        return "Close"
    available = ", ".join(sorted(names))
    raise RuntimeError(f"Downloaded data has no close column. Columns: {available}")


def _select_field(raw: pd.DataFrame, field: str) -> pd.DataFrame | pd.Series:
    level0 = {str(name) for name in raw.columns.get_level_values(0)}
    if field in level0:
        return raw[field]
    level1 = raw.columns.get_level_values(1)
    if field in {str(name) for name in level1}:
        subset = raw.loc[:, [str(name) == field for name in level1]]
        subset.columns = [str(name) for name in subset.columns.get_level_values(0)]
        return subset
    available = ", ".join(sorted(_column_names(raw)))
    raise RuntimeError(f"Downloaded data has no close column. Columns: {available}")
