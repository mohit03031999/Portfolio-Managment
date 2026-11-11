"""Tie a user ticker list to prices, returns, and saved tables."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from market_analyzer.analyzer import summarize
from market_analyzer.matrices import CorrelationModel
from market_analyzer.models import MarketData
from market_analyzer.prices import download_prices
from market_analyzer.returns import ALIGN_INTERSECTION, daily_returns
from market_analyzer.universe import UniverseSpec, clean_tickers, load_universe

DEFAULT_START = "2020-01-01"
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def build_market_data(
    *,
    tickers: list[str] | None = None,
    universe_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    name: str | None = None,
    align: str = ALIGN_INTERSECTION,
) -> MarketData:
    """Download one basket and build its daily return matrix.

    Pass ``tickers`` for an explicit list, or ``universe_path`` for a YAML file.
    When both are set, ``tickers`` is the list that is downloaded, and the file
    still supplies the name and dates you do not override.
    """
    if tickers is None and universe_path is None:
        raise ValueError("Provide tickers or a universe file.")

    spec = load_universe(universe_path) if universe_path is not None else None
    if tickers is not None:
        chosen = clean_tickers(tickers)
    elif spec is not None:
        chosen = spec.tickers
    else:
        raise ValueError("Provide tickers or a universe file.")
    start_date = _as_date("start", start or _spec_value(spec, "start") or DEFAULT_START)
    end_raw = end if end is not None else _spec_value(spec, "end")
    end_date = _as_date("end", end_raw) if end_raw else None
    if end_date is not None and pd.Timestamp(end_date) <= pd.Timestamp(start_date):
        raise ValueError(
            "end must be after start. Yahoo Finance treats the end date as exclusive."
        )

    label = _safe_name(name or (spec.name if spec is not None else "custom"))
    prices, missing = download_prices(chosen, start_date, end_date)
    returns = daily_returns(prices, align=align)
    without_returns = [ticker for ticker in prices.columns if ticker not in returns.columns]
    if without_returns:
        missing = [*missing, *without_returns]
        prices = prices.loc[:, returns.columns]
    notes = _notes(prices, returns, missing, align)
    return MarketData(
        name=label,
        start=start_date,
        end=end_date,
        requested_tickers=chosen,
        missing_tickers=missing,
        prices=prices,
        returns=returns,
        notes=notes,
    )


def save_market_data(
    data: MarketData,
    directory: str | Path,
    periods_per_year: int = 252,
) -> dict[str, Path]:
    """Write prices, returns, the summary, and the risk matrices.

    When the return panel is dense, this also writes the eigenvalue spectrum
    and the correlation matrix after Marchenko-Pastur clipping. A one-line
    description of that spectrum is appended to ``data.notes``.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "prices": folder / "prices.csv",
        "returns": folder / "returns.csv",
        "summary": folder / "summary.csv",
    }
    data.prices.to_csv(paths["prices"])
    data.returns.to_csv(paths["returns"])
    summarize(data.returns, periods_per_year).to_csv(paths["summary"])
    paths.update(_write_risk_matrices(data, folder))
    return paths


def _write_risk_matrices(data: MarketData, folder: Path) -> dict[str, Path]:
    try:
        model = CorrelationModel.from_returns(data.returns)
    except ValueError as exc:
        data.notes.append(str(exc))
        paths = {
            "covariance": folder / "covariance.csv",
            "correlation": folder / "correlation.csv",
        }
        data.returns.cov().to_csv(paths["covariance"])
        data.returns.corr().to_csv(paths["correlation"])
        return paths

    data.notes.append(model.note())
    return model.save(folder)


def _spec_value(spec: UniverseSpec | None, field: str) -> str | None:
    if spec is None:
        return None
    return getattr(spec, field)


def _as_date(label: str, value: str) -> str:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a date like YYYY-MM-DD, got {value!r}.") from exc
    if pd.isna(stamp):
        raise ValueError(f"{label} must be a date like YYYY-MM-DD, got {value!r}.")
    return str(stamp.date())


def _safe_name(name: str) -> str:
    cleaned = _NAME_RE.sub("-", name).strip("-._")
    if not cleaned:
        raise ValueError(f"Basket name {name!r} cannot be used as a folder name.")
    return cleaned


def _notes(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    missing: list[str],
    align: str,
) -> list[str]:
    notes: list[str] = []
    if missing:
        notes.append("No usable price history for: " + ", ".join(missing))
    if align == ALIGN_INTERSECTION and prices.shape[1] > 1:
        candidate_rows = int(prices.pct_change().dropna(how="all").shape[0])
        kept = len(returns)
        if candidate_rows and kept < 0.8 * candidate_rows:
            notes.append(
                f"{kept} of {candidate_rows} return dates are shared by every ticker. "
                "Baskets that mix exchanges often need --align union."
            )
    return notes
