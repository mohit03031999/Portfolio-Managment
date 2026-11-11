"""What the web page computes for one basket of stocks.

Two questions, both answered from the market pack without any return
forecast:

:func:`analyse_basket`
    Over the last ``window`` sessions, how much would an equal split of the
    basket have swung, how much does the minimum-variance mix promise to
    swing, how is the money spread, and why does each stock get the weight
    it gets.
:func:`backtest_basket`
    What would have happened over the last few years if the basket had been
    re-optimised every month: realised risk against the forecast (the
    honesty ratio), drawdowns, and the growth of 100 dollars.

The rules are the ones studied on the S&P 500: long only, a gross-exposure
cap of two, and no limits. Each carries the honesty band measured there,
so the page can say how far the forecast was trusted in that test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest
from market_analyzer.market_pack import NOTE_YOUNG, MarketPack
from market_analyzer.optimizer import (
    ESTIMATORS,
    equal_weights,
    gross_capped_minimum_variance_weights,
    portfolio_variance,
    weight_statistics,
)

PERIODS_PER_YEAR = 252
EQUAL_WEIGHT = "equal_weight"


@dataclass(frozen=True)
class Rule:
    """A portfolio rule as the page offers it."""

    key: str
    label: str
    cap: float
    prefix: str
    honesty: tuple[float, float]
    note: str


RULES: dict[str, Rule] = {
    "long_only": Rule(
        "long_only",
        "Long only: no shorting",
        1.0,
        "longonly_",
        (0.85, 1.02),
        "Every weight is zero or positive and they add to one. The mix a normal brokerage account can hold.",
    ),
    "cap2": Rule(
        "cap2",
        "Small shorts allowed: gross exposure capped at 2",
        2.0,
        "gross2_",
        (0.95, 1.10),
        "Longs plus shorts may add to at most twice the money, so at most half a dollar short per dollar invested.",
    ),
    "unconstrained": Rule(
        "unconstrained",
        "No limits (research only)",
        math.inf,
        "minvar_",
        (1.25, 1.35),
        "The textbook answer. It shorts freely and borrows several times over; the forecast runs optimistic.",
    ),
}

CLEANINGS: dict[str, str] = {
    "rie": "Cleaned correlations (RIE)",
    "clipped": "Clipped correlations (Marchenko-Pastur)",
    "sample": "Plain sample covariance",
}


def strategy_name(rule: str, cleaning: str) -> str:
    """The backtest strategy name for a rule and a cleaning, e.g. ``longonly_rie``."""
    if rule not in RULES:
        raise ValueError(f"Unknown rule {rule!r}; choose one of {', '.join(RULES)}.")
    if cleaning not in CLEANINGS:
        raise ValueError(f"Unknown cleaning {cleaning!r}; choose one of {', '.join(CLEANINGS)}.")
    return RULES[rule].prefix + cleaning


@dataclass
class BasketAnalysis:
    """The risk picture of one basket over one window."""

    tickers: list[str]
    excluded: dict[str, str]
    rule: str
    cleaning: str
    window: int
    start: pd.Timestamp
    end: pd.Timestamp
    equal_vol: float
    optimised_vol: float
    equal_realised_vol: float
    weights: pd.Series
    equal: pd.Series
    table: pd.DataFrame
    stats: dict[str, float]
    info: dict[str, float]

    @property
    def reduction(self) -> float:
        """Fraction of the equal-split risk the optimiser's mix removes."""
        return 1.0 - self.optimised_vol / self.equal_vol if self.equal_vol > 0 else 0.0

    @property
    def honesty(self) -> tuple[float, float]:
        return RULES[self.rule].honesty


def analyse_basket(
    pack: MarketPack,
    tickers: list[str],
    rule: str = "long_only",
    cleaning: str = "rie",
    window: int = 1000,
) -> BasketAnalysis:
    """Risk of an equal split against the minimum-variance mix over the last ``window`` sessions."""
    name = strategy_name(rule, cleaning)
    wanted = _unique([str(t) for t in tickers])
    missing = [t for t in wanted if t not in pack.returns.columns]
    if missing:
        raise ValueError("Not in the market pack: " + ", ".join(missing))
    panel = pack.returns.loc[:, wanted].iloc[-window:].astype("float64")
    complete = panel.notna().all(axis=0)
    moving = panel.std(ddof=1) > 0
    usable = [t for t in wanted if complete[t] and moving[t]]
    excluded = {}
    for ticker in wanted:
        if ticker in usable:
            continue
        young = bool(pack.meta.loc[ticker, "young"]) if ticker in pack.meta.index and "young" in pack.meta else False
        excluded[ticker] = NOTE_YOUNG if young else "missing or flat prices inside the window"
    if len(usable) < 2:
        raise ValueError("Need at least two stocks with a complete price history over the window.")

    panel = panel.loc[:, usable]
    estimate = ESTIMATORS[cleaning](panel)
    covariance = estimate.covariance
    weights = gross_capped_minimum_variance_weights(covariance, RULES[rule].cap).reindex(usable)
    equal = equal_weights(usable)
    annualise = math.sqrt(PERIODS_PER_YEAR)
    equal_vol = math.sqrt(max(portfolio_variance(equal, covariance), 0.0)) * annualise
    optimised_vol = math.sqrt(max(portfolio_variance(weights, covariance), 0.0)) * annualise
    equal_realised_vol = float(panel.mean(axis=1).std(ddof=1)) * annualise

    table = position_table(pack, weights, equal, covariance)
    stats = weight_statistics(weights)
    return BasketAnalysis(
        tickers=usable,
        excluded=excluded,
        rule=rule,
        cleaning=cleaning,
        window=int(len(panel)),
        start=pd.Timestamp(panel.index[0]),
        end=pd.Timestamp(panel.index[-1]),
        equal_vol=equal_vol,
        optimised_vol=optimised_vol,
        equal_realised_vol=equal_realised_vol,
        weights=weights,
        equal=equal,
        table=table,
        stats={key: float(value) for key, value in stats.items()},
        info={key: float(value) for key, value in estimate.info.items()},
    )


def position_table(pack: MarketPack, weights: pd.Series, equal: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """One row per stock: weight, equal weight, own volatility, average correlation, risk share, reason."""
    tickers = [str(t) for t in weights.index]
    matrix = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    deviations = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = matrix / np.outer(deviations, deviations)
    np.fill_diagonal(correlation, np.nan)
    avg_corr = np.nanmean(correlation, axis=1) if len(tickers) > 1 else np.zeros(len(tickers))
    own_vol = deviations * math.sqrt(PERIODS_PER_YEAR)
    vector = weights.to_numpy(dtype=float)
    marginal = matrix @ vector
    total = float(vector @ marginal)
    risk_share = vector * marginal / total if total > 0 else np.zeros(len(tickers))
    vol_rank = pd.Series(own_vol, index=tickers).rank(pct=True)
    corr_rank = pd.Series(avg_corr, index=tickers).rank(pct=True)

    rows = []
    for i, ticker in enumerate(tickers):
        facts = pack.meta.loc[ticker] if ticker in pack.meta.index else pd.Series(dtype=object)
        rows.append(
            {
                "ticker": ticker,
                "name": str(facts.get("name", ticker)),
                "sector": str(facts.get("sector", "Unknown")),
                "weight": float(vector[i]),
                "equal_weight": float(equal.get(ticker, 0.0)),
                "own_vol": float(own_vol[i]),
                "avg_corr": float(avg_corr[i]),
                "risk_share": float(risk_share[i]),
                "reason": describe_position(
                    float(vector[i]), float(equal.get(ticker, 0.0)), float(vol_rank[ticker]), float(corr_rank[ticker])
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)
    return table


def describe_position(weight: float, equal: float, vol_rank: float, corr_rank: float) -> str:
    """A plain-words reason for a weight, from the stock's own volatility and correlation ranks."""
    calm = vol_rank <= 0.34
    jumpy = vol_rank >= 0.67
    loner = corr_rank <= 0.34
    follower = corr_rank >= 0.67
    if weight < -1e-6:
        return "shorted: it moves so closely with others that betting against it offsets their swings"
    if abs(weight) <= 1e-6:
        if jumpy and follower:
            return "left out: jumpy and moves with the rest, so it only adds risk"
        if follower:
            return "left out: moves with the rest, so the others already cover it"
        if jumpy:
            return "left out: jumpier than the stocks that play the same role"
        return "left out: the others combine into a calmer mix without it"
    if weight >= 1.5 * equal:
        if calm and loner:
            return "big weight: calm and moves on its own"
        if calm:
            return "big weight: among the calmest in the basket"
        if loner:
            return "big weight: moves on its own, so it offsets the rest"
        return "big weight: a useful counterweight to the others"
    if weight <= 0.5 * equal:
        if jumpy and follower:
            return "trimmed: jumpy and moves with the rest"
        if jumpy:
            return "trimmed: jumpier than most of the basket"
        if follower:
            return "trimmed: moves with the rest"
        return "trimmed: others do its job with less risk"
    return "near an equal share: nothing unusual about its risk"


@dataclass
class BasketBacktest:
    """Monthly re-optimisation of a basket over the last few years."""

    strategy: str
    result: BacktestResult
    summary: pd.DataFrame
    growth: pd.DataFrame
    worst: dict[str, object]

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.result.daily_returns.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.result.daily_returns.index[-1])


def backtest_basket(
    pack: MarketPack,
    tickers: list[str],
    rule: str = "long_only",
    cleaning: str = "rie",
    window: int = 500,
    years: float = 3.0,
    holding: int = 21,
) -> BasketBacktest:
    """Re-optimise every ``holding`` sessions over the last ``years`` years, estimating on ``window`` sessions."""
    name = strategy_name(rule, cleaning)
    wanted = _unique([str(t) for t in tickers])
    rows = int(window + round(years * PERIODS_PER_YEAR))
    panel = pack.returns.loc[:, wanted].iloc[-rows:].astype("float64")
    config = BacktestConfig(window=window, holding=holding, strategies=(EQUAL_WEIGHT, name), min_assets=2)
    result = run_backtest(panel, config)
    summary = result.summary()
    growth = (1.0 + result.daily_returns).cumprod() * 100.0
    periods = result.periods[result.periods["strategy"] == name]
    worst_row = periods.loc[periods["vol_ratio"].idxmax()] if periods["vol_ratio"].notna().any() else None
    worst = (
        {
            "hold_start": pd.Timestamp(worst_row["hold_start"]),
            "predicted_vol": float(worst_row["predicted_vol"]),
            "realised_vol": float(worst_row["realised_vol"]),
            "vol_ratio": float(worst_row["vol_ratio"]),
        }
        if worst_row is not None
        else {}
    )
    return BasketBacktest(strategy=name, result=result, summary=summary, growth=growth, worst=worst)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
