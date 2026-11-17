"""Scoring return signals across stocks: does a ranking today predict next month?

The lab asks one question of each signal: on each decision date, if we rank
every stock by the signal, how well does that ranking line up with the stocks'
returns over the following month? A single portfolio's track record over a
few years cannot answer this (its Sharpe ratio has an error bar of about
±0.4), but a cross-section of thousands of stocks, repeated every month,
can detect an edge as small as a rank correlation of 0.03.

Per signal and date the lab computes, across the stocks with both a signal
and an outcome:

* the **information coefficient**: Spearman rank correlation between the
  signal and the forward return;
* the **decile returns**: equal-weight forward return of each tenth of the
  ranking, and the **spread** (top tenth minus bottom tenth), a long-short
  portfolio whose monthly series gives a Sharpe ratio with an error bar;
* the **calibration slope**: ordinary-least-squares slope of the forward
  relative return on the standardised signal, in return per standard
  deviation of signal — the size of edge the optimiser could act on; for a
  signal already in return units the raw slope is the fraction of the
  forecast that survived, i.e. the data's own trust setting;
* **decay**: the information coefficient against the return of the k-th
  month ahead, which says how often the ranking would need refreshing;
* **turnover** of the top decile and the **coverage** (stocks scored).

Everything is per decision date first, then summarised (Fama–MacBeth style):
the mean across dates and a t-statistic from the spread of the per-date
values, which handles the cross-sectional correlation of stocks on the same
day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import MONTH, forward_return

PERIODS_PER_YEAR = 12


# --------------------------------------------------------------------------- dates


def decision_dates(index: pd.DatetimeIndex, frequency: str = "M") -> pd.DatetimeIndex:
    """The last trading session of each month (or ``frequency`` period) in ``index``."""
    stamps = pd.Series(index, index=index)
    period = stamps.dt.to_period(frequency)
    return pd.DatetimeIndex(stamps.groupby(period).last().to_numpy())


# --------------------------------------------------------------------------- row statistics


def _aligned(x: pd.DataFrame, y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Float arrays on the common dates and tickers, with a joint validity mask."""
    y = y.reindex(index=x.index, columns=x.columns)
    a = x.to_numpy(dtype=float)
    b = y.to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    return a, b, valid


def _row_moments(a: np.ndarray, b: np.ndarray, valid: np.ndarray, min_count: int):
    n = valid.sum(axis=1)
    enough = n >= min_count
    safe = np.where(enough, n, 1)
    a = np.where(valid, a, 0.0)
    b = np.where(valid, b, 0.0)
    da = (a - (a.sum(axis=1) / safe)[:, None]) * valid
    db = (b - (b.sum(axis=1) / safe)[:, None]) * valid
    cov = (da * db).sum(axis=1)
    var_a = (da * da).sum(axis=1)
    var_b = (db * db).sum(axis=1)
    return n, enough, cov, var_a, var_b


def rank_correlation(x: pd.DataFrame, y: pd.DataFrame, min_count: int = 20) -> pd.Series:
    """Per-date Spearman correlation across tickers between two panels."""
    a, b, valid = _aligned(x, y)
    ranked_x = pd.DataFrame(np.where(valid, a, np.nan)).rank(axis=1).to_numpy()
    ranked_y = pd.DataFrame(np.where(valid, b, np.nan)).rank(axis=1).to_numpy()
    n, enough, cov, var_a, var_b = _row_moments(ranked_x, ranked_y, valid, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(enough, cov / np.sqrt(var_a * var_b), np.nan)
    return pd.Series(rho, index=x.index, name="ic")


def cross_sectional_slope(
    x: pd.DataFrame, y: pd.DataFrame, standardise: bool = True, min_count: int = 20
) -> pd.Series:
    """Per-date OLS slope of ``y`` on ``x`` across tickers.

    With ``standardise`` the slope is per standard deviation of ``x`` (the
    regressor is z-scored each date); without it the slope is in ``y`` units
    per unit of ``x``. Demeaning both sides per date removes the market's
    move from the outcome, so only relative calls are scored.
    """
    a, b, valid = _aligned(x, y)
    n, enough, cov, var_a, _ = _row_moments(a, b, valid, min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov / np.sqrt(var_a * (n - 1)) if standardise else cov / var_a
    slope = np.where(enough, slope, np.nan)
    return pd.Series(slope, index=x.index, name="slope")


def coverage(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    _, _, valid = _aligned(x, y)
    return pd.Series(valid.sum(axis=1), index=x.index, name="coverage")


# --------------------------------------------------------------------------- deciles


def decile_returns(
    x: pd.DataFrame, y: pd.DataFrame, n_deciles: int = 10, min_count: int = 20
) -> pd.DataFrame:
    """Equal-weight forward return of each ranking bucket, per date (columns 1..n)."""
    a, b, valid = _aligned(x, y)
    out = np.full((a.shape[0], n_deciles), np.nan)
    for i in range(a.shape[0]):
        mask = valid[i]
        count = int(mask.sum())
        if count < min_count:
            continue
        order = np.argsort(a[i, mask], kind="stable")
        buckets = np.floor(np.arange(count) * n_deciles / count).astype(int)
        values = b[i, mask][order]
        out[i] = np.bincount(buckets, weights=values, minlength=n_deciles) / np.bincount(
            buckets, minlength=n_deciles
        )
    return pd.DataFrame(out, index=x.index, columns=range(1, n_deciles + 1))


def top_decile_turnover(x: pd.DataFrame, n_deciles: int = 10) -> pd.Series:
    """Share of the top bucket's names that were not in it on the previous date."""
    ranks = x.rank(axis=1, pct=True)
    top = ranks > 1.0 - 1.0 / n_deciles
    previous = top.shift(1).fillna(False).astype(bool)
    size = top.sum(axis=1)
    stayed = (top & previous).sum(axis=1)
    turnover = 1.0 - stayed / size.where(size > 0)
    turnover.iloc[0] = np.nan
    return turnover.rename("turnover")


# --------------------------------------------------------------------------- summaries


def t_statistic(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.mean() / (clean.std(ddof=1) / np.sqrt(len(clean))))


def sharpe_with_error(series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> tuple[float, float]:
    """Annualised Sharpe ratio and its standard error (Lo 2002, iid approximation)."""
    clean = series.dropna()
    if len(clean) < 2 or clean.std(ddof=1) == 0:
        return float("nan"), float("nan")
    per_period = float(clean.mean() / clean.std(ddof=1))
    error = np.sqrt((1.0 + per_period**2 / 2.0) / len(clean))
    root = np.sqrt(periods_per_year)
    return per_period * root, float(error * root)


@dataclass
class SignalScore:
    key: str
    label: str
    ic: pd.Series
    slope: pd.Series
    deciles: pd.DataFrame
    coverage: pd.Series
    turnover: pd.Series
    decay: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    raw_slope: pd.Series | None = None

    @property
    def spread(self) -> pd.Series:
        top = self.deciles.columns.max()
        return (self.deciles[top] - self.deciles[1]).rename("spread")

    def ic_by_year(self) -> pd.Series:
        return self.ic.groupby(self.ic.index.year).mean()

    def summary(self) -> dict[str, float | str]:
        sharpe, sharpe_se = sharpe_with_error(self.spread)
        halves = np.array_split(self.ic.dropna(), 2)
        row: dict[str, float | str] = {
            "signal": self.label,
            "ic_mean": float(self.ic.mean()),
            "ic_std": float(self.ic.std(ddof=1)),
            "ic_t": t_statistic(self.ic),
            "ic_positive_share": float((self.ic.dropna() > 0).mean()),
            "ic_first_half": float(halves[0].mean()) if len(halves[0]) else float("nan"),
            "ic_second_half": float(halves[1].mean()) if len(halves[1]) else float("nan"),
            "spread_annual_return": float(self.spread.mean() * PERIODS_PER_YEAR),
            "spread_annual_vol": float(self.spread.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)),
            "spread_sharpe": sharpe,
            "spread_sharpe_se": sharpe_se,
            "slope_per_sd": float(self.slope.mean()),
            "slope_t": t_statistic(self.slope),
            "top_decile_turnover": float(self.turnover.mean()),
            "coverage": float(self.coverage.mean()),
            "months": int(self.ic.notna().sum()),
        }
        if self.raw_slope is not None:
            row["trust_fraction"] = float(self.raw_slope.mean())
            row["trust_t"] = t_statistic(self.raw_slope)
        for k, value in self.decay.items():
            row[f"ic_month_{k}"] = float(value)
        return row


def score_signal(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    key: str = "signal",
    label: str | None = None,
    horizon: int = MONTH,
    decay_months: tuple[int, ...] = (1, 2, 3, 6, 12),
    n_deciles: int = 10,
    tickers: list[str] | pd.Index | None = None,
    mask: pd.DataFrame | None = None,
    in_return_units: bool = False,
    shuffle_seed: int | None = None,
) -> SignalScore:
    """Score one oriented signal panel on the given decision dates.

    ``tickers`` restricts the cross-section to a fixed list; ``mask`` (a
    boolean dates × tickers panel) restricts it date by date, for example to
    the most liquid stocks as of each date; ``in_return_units`` adds the raw
    calibration slope, the trust fraction; ``shuffle_seed`` permutes each
    date's outcomes among its stocks, the control that must score zero.
    """
    dates = pd.DatetimeIndex(dates)
    x = signal.reindex(index=dates)
    if tickers is not None:
        x = x.reindex(columns=pd.Index(tickers).intersection(x.columns))
    if mask is not None:
        x = x.where(mask.reindex(index=x.index, columns=x.columns).fillna(False).astype(bool))
    y = forward_return(returns, horizon).reindex(index=dates, columns=x.columns)
    if shuffle_seed is not None:
        y = _shuffle_rows(y, shuffle_seed)
    score = SignalScore(
        key=key,
        label=label or key,
        ic=rank_correlation(x, y),
        slope=cross_sectional_slope(x, y, standardise=True),
        deciles=decile_returns(x, y, n_deciles),
        coverage=coverage(x, y),
        turnover=top_decile_turnover(x, n_deciles),
        raw_slope=cross_sectional_slope(x, y, standardise=False) if in_return_units else None,
    )
    decay = {}
    for k in decay_months:
        later = forward_return(returns, horizon, offset=(k - 1) * horizon)
        later = later.reindex(index=dates, columns=x.columns)
        if shuffle_seed is not None:
            later = _shuffle_rows(later, shuffle_seed + k)
        decay[k] = float(rank_correlation(x, later).mean())
    score.decay = pd.Series(decay, name="ic").rename_axis("month_ahead")
    return score


def _shuffle_rows(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute each row's finite values among its finite positions."""
    rng = np.random.default_rng(seed)
    values = panel.to_numpy(dtype=float).copy()
    for i in range(values.shape[0]):
        mask = np.isfinite(values[i])
        values[i, mask] = rng.permutation(values[i, mask])
    return pd.DataFrame(values, index=panel.index, columns=panel.columns)


def summary_table(scores: list[SignalScore]) -> pd.DataFrame:
    return pd.DataFrame([score.summary() for score in scores]).set_index("signal")
