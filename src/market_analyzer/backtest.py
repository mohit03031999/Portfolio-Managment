"""Rolling out-of-sample test of covariance estimators through minimum-variance portfolios.

On each decision date the engine looks back ``window`` sessions, keeps every
ticker with a complete return history over that window, builds one
covariance matrix per estimator, turns each into unconstrained
minimum-variance weights, and holds those weights fixed for the next
``holding`` sessions. Nothing from the holding period touches the estimate.
The engine then slides forward by ``holding`` sessions and repeats.

For every period and strategy it records what the model said the
portfolio's volatility would be, what it turned out to be, and how far the
weights moved. The headline number is ``vol_ratio``, realised over
predicted volatility. An honest risk model lands near one. A covariance
matrix full of estimation noise lets the optimizer find directions that
look riskless and are not, and its ratio sits well above one.

Strategy names are ``minvar_<estimator>`` for every key in
:data:`market_analyzer.optimizer.ESTIMATORS`, ``longonly_<estimator>`` for
the same matrices with every weight held at or above zero (no shorting, so
gross leverage is exactly one), ``gross<cap>_<estimator>`` for a cap on
total long plus short exposure (``gross2_rie`` is a 150/50 book; ``gross1_``
is long-only and a cap no portfolio reaches is the unconstrained optimiser),
plus ``equal_weight``. Equal weight has no
risk model of its own, so its predicted volatility is taken from the sample
covariance. That makes it a useful control: the sample
covariance is accurate for weights that were not chosen by looking at it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from market_analyzer.analyzer import summarize
from market_analyzer.optimizer import (
    ESTIMATORS,
    CovarianceEstimate,
    Estimator,
    equal_weights,
    gross_capped_minimum_variance_weights,
    portfolio_variance,
    turnover,
    weight_statistics,
)

EQUAL_WEIGHT = "equal_weight"
MINVAR_PREFIX = "minvar_"
LONG_ONLY_PREFIX = "longonly_"
GROSS_PATTERN = re.compile(r"^gross(\d+(?:\.\d+)?)_(.+)$")
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "minvar_sample",
    "minvar_clipped",
    "minvar_clipped_unit",
    "minvar_diagonal",
    "minvar_ledoit_wolf",
    EQUAL_WEIGHT,
)
_SUMMARY_COLUMNS = (
    "n_periods",
    "ann_volatility",
    "median_vol_ratio",
    "mean_turnover",
    "mean_gross_leverage",
    "mean_effective_n",
    "ann_return",
    "sharpe",
    "max_drawdown",
)


@dataclass(frozen=True)
class BacktestConfig:
    """Everything the rolling test needs to know besides the returns."""

    window: int = 500
    holding: int = 21
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    periods_per_year: int = 252
    min_assets: int = 2
    cost_bps: float = 0.0

    def validate(self) -> None:
        if self.window < 3:
            raise ValueError("window must be at least 3 sessions.")
        if self.holding < 2:
            raise ValueError("holding must be at least 2 sessions so realised volatility is defined.")
        if not self.strategies:
            raise ValueError("Choose at least one strategy.")
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("Strategy names must be unique.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")
        if self.min_assets < 2:
            raise ValueError("min_assets must be at least 2.")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative.")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["strategies"] = list(self.strategies)
        return data


@dataclass
class BacktestResult:
    """Per-period records, the stitched daily returns, and the weights that produced them."""

    config: BacktestConfig
    periods: pd.DataFrame
    daily_returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    notes: list[str] = field(default_factory=list)

    @property
    def strategies(self) -> list[str]:
        return list(self.config.strategies)

    def summary(self) -> pd.DataFrame:
        """One row per strategy: realised risk, honesty ratio, turnover, concentration, returns."""
        ppy = self.config.periods_per_year
        stats = summarize(self.daily_returns, ppy)
        grouped = self.periods.groupby("strategy")
        steady = self.periods.loc[~self.periods["first_period"]].groupby("strategy")
        table = pd.DataFrame(
            {
                "n_periods": grouped.size(),
                "ann_return": stats["ann_return"],
                "ann_volatility": stats["ann_volatility"],
                "sharpe": stats["mean_daily"] * ppy / stats["ann_volatility"],
                "max_drawdown": stats["max_drawdown"],
                "mean_predicted_vol": grouped["predicted_vol"].mean(),
                "mean_realised_vol": grouped["realised_vol"].mean(),
                "median_vol_ratio": grouped["vol_ratio"].median(),
                "vol_ratio_q25": grouped["vol_ratio"].quantile(0.25),
                "vol_ratio_q75": grouped["vol_ratio"].quantile(0.75),
                "mean_turnover": steady["turnover"].mean(),
                "mean_gross_leverage": grouped["gross_leverage"].mean(),
                "mean_max_weight": grouped["max_weight"].mean(),
                "mean_n_short": grouped["n_short"].mean(),
                "mean_n_positions": grouped["n_positions"].mean() if "n_positions" in self.periods else np.nan,
                "mean_effective_n": grouped["effective_n"].mean(),
                "share_singular": grouped["singular"].mean(),
            }
        )
        table = table.reindex([name for name in self.strategies if name in table.index])
        table.index.name = "strategy"
        return table

    def report(self) -> str:
        """Config line plus the summary columns that decide the comparison."""
        config = self.config
        first = self.periods["hold_start"].min()
        last = self.periods["hold_end"].max()
        lines = [
            f"Rolling backtest: window {config.window} sessions, hold {config.holding} sessions, "
            f"{self.periods['hold_start'].nunique()} periods from {_date(first)} to {_date(last)}.",
            "vol_ratio is realised over predicted volatility per period. Honest risk models sit near 1.",
            "",
            self.summary()
            .loc[:, list(_SUMMARY_COLUMNS)]
            .to_string(float_format=lambda value: f"{value:.4f}"),
        ]
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)

    def restrict(self, start=None, end=None) -> BacktestResult:
        """Keep the holding periods that start on or after ``start`` and end on or before ``end``.

        Use it to compare runs with different windows over the dates they
        share. Turnover flags are kept as they were, so a period that followed
        a dropped one still counts as a steady-state rebalance.
        """
        periods = self.periods
        mask = pd.Series(True, index=periods.index)
        if start is not None:
            mask &= periods["hold_start"] >= pd.Timestamp(start)
        if end is not None:
            mask &= periods["hold_end"] <= pd.Timestamp(end)
        kept = periods.loc[mask].reset_index(drop=True)
        if kept.empty:
            raise ValueError("No holding period falls inside the requested dates.")
        low, high = kept["hold_start"].min(), kept["hold_end"].max()
        daily = self.daily_returns.loc[(self.daily_returns.index >= low) & (self.daily_returns.index <= high)]
        weights = {}
        for name, frame in self.weights.items():
            starts = kept.loc[kept["strategy"] == name, "hold_start"]
            weights[name] = frame.loc[frame.index.isin(starts)]
        return BacktestResult(config=self.config, periods=kept, daily_returns=daily, weights=weights, notes=list(self.notes))

    @classmethod
    def load(cls, directory: str | Path) -> BacktestResult:
        """Read back what :meth:`save` wrote."""
        folder = Path(directory)
        config_data = json.loads((folder / "config.json").read_text())
        config_data["strategies"] = tuple(config_data["strategies"])
        config = BacktestConfig(**config_data)
        periods = pd.read_csv(folder / "periods.csv")
        for column in ("estimation_start", "estimation_end", "hold_start", "hold_end"):
            periods[column] = _maybe_dates(periods[column])
        daily = pd.read_csv(folder / "daily_returns.csv", index_col=0)
        daily.index = _maybe_dates(daily.index)
        weights = {}
        for name in config.strategies:
            path = folder / f"weights_{name}.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path, index_col=0)
            frame.index = _maybe_dates(frame.index)
            frame.index.name = "hold_start"
            frame.columns.name = "ticker"
            weights[name] = frame
        notes_path = folder / "notes.txt"
        notes = notes_path.read_text().splitlines() if notes_path.is_file() else []
        return cls(config=config, periods=periods, daily_returns=daily, weights=weights, notes=notes)

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write periods, daily returns, the summary, one weight file per strategy, and the config."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        paths = {
            "periods": folder / "periods.csv",
            "daily_returns": folder / "daily_returns.csv",
            "summary": folder / "summary.csv",
            "config": folder / "config.json",
        }
        self.periods.to_csv(paths["periods"], index=False)
        self.daily_returns.to_csv(paths["daily_returns"])
        self.summary().to_csv(paths["summary"])
        paths["config"].write_text(json.dumps(self.config.to_dict(), indent=2) + "\n")
        for name, frame in self.weights.items():
            path = folder / f"weights_{name}.csv"
            frame.to_csv(path)
            paths[f"weights_{name}"] = path
        if self.notes:
            paths["notes"] = folder / "notes.txt"
            paths["notes"].write_text("\n".join(self.notes) + "\n")
        return paths


def run_backtest(
    returns: pd.DataFrame,
    config: BacktestConfig | None = None,
    estimators: dict[str, Estimator] | None = None,
) -> BacktestResult:
    """Run the rolling test on a date × ticker return panel.

    The panel may contain missing values. A ticker joins a decision only
    when its returns are complete and not constant over that estimation
    window. A missing return inside a holding period counts as zero, which
    is what a position that stopped trading would earn.

    ``estimators`` adds or overrides covariance estimators by key. A key
    ``"oracle"`` makes ``"minvar_oracle"`` a valid strategy name.
    """
    config = config or BacktestConfig()
    config.validate()
    panel = _prepare(returns)
    registry: dict[str, Estimator] = {**ESTIMATORS, **(estimators or {})}
    plan = [_resolve(name, registry) for name in config.strategies]
    window, holding = config.window, config.holding
    if len(panel) < window + holding:
        raise ValueError(
            f"Need at least {window + holding} return dates for one estimation window "
            f"plus one holding period, found {len(panel)}."
        )

    rows: list[dict[str, object]] = []
    daily: dict[str, list[pd.Series]] = {name: [] for name, _, _ in plan}
    history: dict[str, list[pd.Series]] = {name: [] for name, _, _ in plan}
    previous: dict[str, pd.Series | None] = {name: None for name, _, _ in plan}
    notes: list[str] = []
    annualize = math.sqrt(config.periods_per_year)

    for start in range(window, len(panel) - holding + 1, holding):
        estimation = panel.iloc[start - window : start]
        complete = estimation.notna().all()
        moving = estimation.std(ddof=1) > 0
        eligible = estimation.columns[(complete & moving).to_numpy()]
        if len(eligible) < config.min_assets:
            notes.append(
                f"Skipped the decision at {_date(estimation.index[-1])}: "
                f"only {len(eligible)} tickers have a complete window."
            )
            continue
        estimation = estimation.loc[:, eligible]
        hold = panel.iloc[start : start + holding].loc[:, eligible]
        filled = int(hold.isna().to_numpy().sum())
        if filled:
            hold = hold.fillna(0.0)
        cache: dict[str, CovarianceEstimate] = {}

        for name, key, cap in plan:
            if key is None:
                weights = equal_weights(eligible)
                estimate = _estimate(cache, registry, "sample", estimation)
                info: dict[str, float] = {}
            else:
                estimate = _estimate(cache, registry, key, estimation)
                weights = gross_capped_minimum_variance_weights(estimate.covariance, cap)
                info = estimate.info
            weights = weights.reindex(eligible)
            realised = pd.Series(
                hold.to_numpy(dtype=float) @ weights.to_numpy(dtype=float),
                index=hold.index,
                name=name,
            )
            move = turnover(weights, previous[name])
            if config.cost_bps:
                realised.iloc[0] -= config.cost_bps / 1e4 * move
            predicted_vol = math.sqrt(max(portfolio_variance(weights, estimate.covariance), 0.0)) * annualize
            realised_vol = float(realised.std(ddof=1)) * annualize
            rows.append(
                {
                    "strategy": name,
                    "estimation_start": estimation.index[0],
                    "estimation_end": estimation.index[-1],
                    "hold_start": hold.index[0],
                    "hold_end": hold.index[-1],
                    "n_assets": int(len(eligible)),
                    "first_period": previous[name] is None,
                    "predicted_vol": predicted_vol,
                    "realised_vol": realised_vol,
                    "vol_ratio": realised_vol / predicted_vol if predicted_vol > 0 else float("nan"),
                    "period_return": float((1.0 + realised).prod() - 1.0),
                    "turnover": move,
                    **weight_statistics(weights),
                    "singular": bool(weights.attrs.get("singular", False)),
                    "condition_number": float(weights.attrs.get("condition_number", float("nan"))),
                    "filled_returns": filled,
                    **info,
                }
            )
            daily[name].append(realised)
            history[name].append(weights.rename(hold.index[0]))
            previous[name] = weights

    if not rows:
        raise ValueError("No decision date had enough tickers with a complete window. Lower min_assets or window.")

    periods = pd.DataFrame(rows)
    daily_returns = pd.DataFrame({name: pd.concat(series) for name, series in daily.items() if series})
    daily_returns.index.name = panel.index.name or "Date"
    weights = {}
    for name, series in history.items():
        if not series:
            continue
        frame = pd.DataFrame(series)
        frame = frame.loc[:, sorted(frame.columns)]
        frame.index.name = "hold_start"
        frame.columns.name = "ticker"
        weights[name] = frame
    return BacktestResult(config=config, periods=periods, daily_returns=daily_returns, weights=weights, notes=notes)


def strategy_names(estimators: dict[str, Estimator] | None = None) -> list[str]:
    """Every strategy name the engine accepts for the given estimator registry."""
    registry = {**ESTIMATORS, **(estimators or {})}
    return [f"{MINVAR_PREFIX}{key}" for key in registry] + [f"{LONG_ONLY_PREFIX}{key}" for key in registry] + [EQUAL_WEIGHT]


def _estimate(
    cache: dict[str, CovarianceEstimate],
    registry: dict[str, Estimator],
    key: str,
    estimation: pd.DataFrame,
) -> CovarianceEstimate:
    if key not in cache:
        cache[key] = registry[key](estimation)
    return cache[key]


def _resolve(name: str, registry: dict[str, Estimator]) -> tuple[str, str | None, float]:
    """Strategy name → (name, estimator key or None for equal weight, gross-exposure cap: inf, 1, or the number in ``gross<cap>_``)."""
    if name == EQUAL_WEIGHT:
        return name, None, float("inf")
    if name.startswith(MINVAR_PREFIX) and name[len(MINVAR_PREFIX) :] in registry:
        return name, name[len(MINVAR_PREFIX) :], float("inf")
    if name.startswith(LONG_ONLY_PREFIX) and name[len(LONG_ONLY_PREFIX) :] in registry:
        return name, name[len(LONG_ONLY_PREFIX) :], 1.0
    match = GROSS_PATTERN.match(name)
    if match and match.group(2) in registry:
        cap = float(match.group(1))
        if cap < 1.0:
            raise ValueError(f"Strategy {name!r}: the gross-exposure cap must be at least 1, since the weights sum to one.")
        return name, match.group(2), cap
    choices = ", ".join(strategy_names(registry))
    raise ValueError(f"Unknown strategy {name!r}. Choose from: {choices}, or gross<cap>_<estimator> such as gross2_rie.")


def _prepare(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame with one column per ticker.")
    panel = returns.copy()
    panel.columns = [str(column) for column in panel.columns]
    if panel.columns.duplicated().any():
        raise ValueError("Return columns must be unique tickers.")
    panel = panel.sort_index()
    panel = panel.dropna(axis=1, how="all")
    if panel.shape[1] < 2:
        raise ValueError("Need at least two tickers with returns.")
    values = panel.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("The return panel has infinite values.")
    return panel


def _date(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def _maybe_dates(values):
    """Parse to timestamps when the values look like dates, otherwise return them unchanged."""
    try:
        return pd.to_datetime(values)
    except (ValueError, TypeError):
        return values
