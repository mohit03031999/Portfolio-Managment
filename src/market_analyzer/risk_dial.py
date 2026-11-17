"""The customer's risk dial: split the money between a stock basket and cash to hit a volatility target.

The optimiser answers "which stocks". The dial answers "how much". A customer
names a target volatility, the yearly swing they accept. The dial forecasts
how much the basket swings and puts

    exposure = target volatility / forecast volatility

of the money in the basket, capped at ``max_exposure`` (1 means never
borrow). The rest sits in cash. Scaling a basket by ``e`` scales its
volatility by ``e``, so the split delivers the target exactly when the
forecast is right. The whole question is therefore the forecast, and there
are three on offer:

``"model"``
    The risk model's own number, ``sqrt(wᵀ Σ w)`` from the covariance that
    chose the weights. Slow: it looks back over the whole estimation window.
``"recent"``
    How the basket in force would have behaved over the last ``lookback``
    sessions, the root mean square of ``R w``. With ``halflife`` set, recent
    days count more: a day ``halflife`` sessions old gets half the weight of
    yesterday. Fast: it sees a storm within days or weeks.
``"higher"``
    The larger of the two, so the dial only ever leans to the cautious side.

The dial can be reset more often than the stocks are reshuffled
(``dial_every``). Moving the split is one trade in the whole basket, not a
new set of weights.

Everything is point in time. A split that applies from session ``d`` uses
returns strictly before ``d``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

CASH = "CASH"
FORECASTS = ("model", "recent", "higher")


def exposure_for_target(target_vol, forecast_vol, max_exposure: float = 1.0):
    """Share of the money to put in the basket: ``target / forecast``, capped at ``max_exposure``.

    Works on numbers, arrays and Series. A forecast that is zero, negative
    or missing gives the cap, since nothing argues for holding less.
    """
    if target_vol < 0:
        raise ValueError("target_vol cannot be negative.")
    if max_exposure <= 0:
        raise ValueError("max_exposure must be positive.")
    forecast = np.asarray(forecast_vol, dtype=float)
    usable = np.isfinite(forecast) & (forecast > 0)
    raw = np.divide(target_vol, forecast, out=np.full(forecast.shape, np.inf), where=usable)
    exposure = np.minimum(raw, max_exposure)
    if not np.isfinite(exposure).all():
        raise ValueError("The forecast volatility is not positive and max_exposure is unbounded, so the split is undefined.")
    if isinstance(forecast_vol, pd.Series):
        return pd.Series(exposure, index=forecast_vol.index, name="exposure")
    return float(exposure) if exposure.ndim == 0 else exposure


def split_with_cash(weights: pd.Series, exposure: float) -> pd.Series:
    """Basket weights scaled by ``exposure``, plus a ``CASH`` line holding the rest.

    The result sums to one. With ``exposure`` above one the cash line is
    negative, which is borrowing.
    """
    if CASH in weights.index:
        raise ValueError(f"The basket already has a {CASH} line.")
    scaled = weights.astype(float) * float(exposure)
    out = pd.concat([scaled, pd.Series({CASH: 1.0 - float(scaled.sum())})])
    out.name = "weight"
    return out


def recent_volatility(path, halflife: float | None = None, periods_per_year: int = 252) -> float:
    """Annual volatility of a path of daily returns, oldest first, with the average daily return taken as zero.

    Over a few weeks the measured average is mostly noise, and subtracting it
    would hide part of a steady fall. With ``halflife`` the squared returns
    are weighted by ``0.5 ** (age / halflife)``, so the newest days dominate.
    """
    squares = np.square(np.asarray(path, dtype=float))
    if squares.size < 2:
        raise ValueError("Need at least 2 returns.")
    if halflife is None:
        mean_square = float(squares.mean())
    else:
        if halflife <= 0:
            raise ValueError("halflife must be positive.")
        decay = 0.5 ** (np.arange(squares.size - 1, -1, -1) / halflife)
        mean_square = float(decay @ squares / decay.sum())
    return math.sqrt(mean_square * periods_per_year)


def basket_volatility(
    weights: pd.Series,
    returns: pd.DataFrame,
    lookback: int = 63,
    halflife: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Annual volatility this basket would have shown over the last ``lookback`` sessions of ``returns``.

    This is the ``"recent"`` forecast for a basket chosen today. A missing
    return counts as zero.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2 sessions.")
    held = weights[weights != 0]
    missing = held.index.difference(returns.columns)
    if len(missing):
        raise ValueError(f"No returns for {len(missing)} held tickers, for example {missing[0]}.")
    recent = returns.loc[:, held.index].tail(lookback)
    if len(recent) < lookback:
        raise ValueError(f"Need {lookback} sessions of returns, found {len(recent)}.")
    path = recent.fillna(0.0).to_numpy(dtype=float) @ held.to_numpy(dtype=float)
    return recent_volatility(path, halflife, periods_per_year)


@dataclass(frozen=True)
class DialConfig:
    """What the customer asked for and how the dial forecasts the basket's risk.

    The defaults are the fast settings from ``examples/sp500_risk_dial_study.py``:
    recent-weighted behaviour (half-life 20 sessions over the last 252), reset
    every 5 sessions. Where the risk model's forecast is at hand, ``"higher"``
    was the forecast that kept the promise best. ``halflife=None`` weights the
    lookback days equally and ``dial_every=None`` moves the dial only when the
    basket is rebalanced.
    """

    target_vol: float
    forecast: str = "recent"
    lookback: int = 252
    halflife: float | None = 20.0
    dial_every: int | None = 5
    max_exposure: float = 1.0
    cash_rate: float = 0.0
    periods_per_year: int = 252

    def validate(self) -> None:
        if not self.target_vol > 0:
            raise ValueError("target_vol must be positive, for example 0.08 for 8% a year.")
        if self.forecast not in FORECASTS:
            raise ValueError(f"Unknown forecast {self.forecast!r}. Choose from: {', '.join(FORECASTS)}.")
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2 sessions.")
        if self.halflife is not None and not self.halflife > 0:
            raise ValueError("halflife must be positive, or None to weight the lookback days equally.")
        if self.dial_every is not None and self.dial_every < 1:
            raise ValueError("dial_every must be at least 1 session, or None to move only with the basket.")
        if self.max_exposure <= 0:
            raise ValueError("max_exposure must be positive.")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")


@dataclass
class DialResult:
    """Day by day: what the basket earned, how much money was in it, and what the customer got.

    ``daily`` has one row per session with ``basket_return``,
    ``forecast_vol`` (annual, the number the split in force was set from),
    ``exposure`` and ``return`` (the dialled portfolio).
    """

    config: DialConfig
    daily: pd.DataFrame

    def delivered_vol(self) -> float:
        """Annual volatility of the dialled portfolio over the whole track."""
        return float(self.daily["return"].std(ddof=1)) * math.sqrt(self.config.periods_per_year)

    def by_year(self, min_sessions: int = 60) -> pd.DataFrame:
        """Delivered volatility and average exposure per calendar year with at least ``min_sessions`` sessions."""
        annualize = math.sqrt(self.config.periods_per_year)
        grouped = self.daily.groupby(self.daily.index.year)
        table = pd.DataFrame(
            {
                "sessions": grouped.size(),
                "delivered_vol": grouped["return"].std(ddof=1) * annualize,
                "basket_vol": grouped["basket_return"].std(ddof=1) * annualize,
                "mean_exposure": grouped["exposure"].mean(),
            }
        )
        table = table.loc[table["sessions"] >= min_sessions]
        table["delivered_over_target"] = table["delivered_vol"] / self.config.target_vol
        table.index.name = "year"
        return table

    def summary(self) -> dict[str, float]:
        """The promise in numbers: asked, delivered, how much sat in the basket, how hard the dial traded."""
        config = self.config
        daily = self.daily
        years = len(daily) / config.periods_per_year
        delivered = self.delivered_vol()
        yearly = self.by_year()
        growth = (1.0 + daily["return"]).cumprod()
        capped = np.isclose(daily["exposure"].to_numpy(), config.max_exposure)
        moves = daily["exposure"].diff().abs().sum()
        return {
            "target_vol": config.target_vol,
            "delivered_vol": delivered,
            "delivered_over_target": delivered / config.target_vol,
            "basket_vol": float(daily["basket_return"].std(ddof=1)) * math.sqrt(config.periods_per_year),
            "calmest_year_vol": float(yearly["delivered_vol"].min()) if len(yearly) else float("nan"),
            "wildest_year_vol": float(yearly["delivered_vol"].max()) if len(yearly) else float("nan"),
            "mean_exposure": float(daily["exposure"].mean()),
            "min_exposure": float(daily["exposure"].min()),
            "share_at_cap": float(capped.mean()),
            "dial_turnover_per_year": float(moves / years) if years > 0 else float("nan"),
            "worst_day": float(daily["return"].min()),
            "max_drawdown": float((growth / growth.cummax() - 1.0).min()),
        }


def run_dial(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: DialConfig,
    model_vol: pd.Series | None = None,
    end=None,
) -> DialResult:
    """Replay the dial over a sequence of baskets.

    ``weights`` has one row per rebalance date and one column per ticker.
    A row is the basket held from that date until the next row, or until
    ``end`` (the last date of ``returns`` by default). ``returns`` is the
    date × ticker panel of daily returns. It must reach back ``lookback``
    sessions before the first rebalance when the forecast uses recent
    behaviour. ``model_vol`` is the risk model's annual volatility forecast
    per rebalance date, needed by ``"model"`` and ``"higher"``.

    A missing weight or return counts as zero, the same convention as the
    backtest engine.
    """
    config.validate()
    if weights.empty:
        raise ValueError("Need at least one basket.")
    baskets = weights.sort_index()
    panel = returns.sort_index()
    if panel.index.has_duplicates or baskets.index.has_duplicates:
        raise ValueError("Dates must be unique.")
    missing = [ticker for ticker in baskets.columns if ticker not in panel.columns and (baskets[ticker].fillna(0.0) != 0).any()]
    if missing:
        raise ValueError(f"No returns for {len(missing)} held tickers, for example {missing[0]}.")
    use_model = config.forecast in ("model", "higher")
    use_recent = config.forecast in ("recent", "higher")
    if use_model:
        if model_vol is None:
            raise ValueError(f"The {config.forecast!r} forecast needs model_vol, the risk model's volatility per rebalance date.")
        model = model_vol.reindex(baskets.index).to_numpy(dtype=float)
        if not (np.isfinite(model).all() and (model > 0).all()):
            raise ValueError("model_vol must hold a positive number for every rebalance date.")

    positions = panel.index.get_indexer(baskets.index)
    if (positions < 0).any():
        raise ValueError("Every rebalance date must be a date of the return panel.")
    last = len(panel) - 1 if end is None else int(panel.index.get_indexer([pd.Timestamp(end)])[0])
    if last < positions[-1]:
        raise ValueError("end must be a date of the return panel on or after the last rebalance date.")
    if use_recent and positions[0] < config.lookback:
        raise ValueError(
            f"The recent-behaviour forecast needs {config.lookback} sessions before the first basket, found {positions[0]}."
        )

    values = panel.reindex(columns=baskets.columns).fillna(0.0).to_numpy(dtype=float)
    held = baskets.fillna(0.0).to_numpy(dtype=float)
    first = int(positions[0])
    size = last - first + 1
    basket_return = np.empty(size)
    forecast = np.empty(size)
    for k, start in enumerate(positions):
        stop = int(positions[k + 1]) if k + 1 < len(positions) else last + 1
        w = held[k]
        basket_return[start - first : stop - first] = values[start:stop] @ w
        step = config.dial_every or (stop - start)
        for point in range(start, stop, step):
            options = []
            if use_model:
                options.append(model[k])
            if use_recent:
                history = values[point - config.lookback : point] @ w
                options.append(recent_volatility(history, config.halflife, config.periods_per_year))
            forecast[point - first : min(point + step, stop) - first] = max(options)

    exposure = exposure_for_target(config.target_vol, forecast, config.max_exposure)
    cash = config.cash_rate / config.periods_per_year
    daily = pd.DataFrame(
        {
            "basket_return": basket_return,
            "forecast_vol": forecast,
            "exposure": exposure,
            "return": exposure * basket_return + (1.0 - exposure) * cash,
        },
        index=panel.index[first : last + 1],
    )
    daily.index.name = panel.index.name or "Date"
    return DialResult(config=config, daily=daily)


def dial_backtest(result, strategy: str, returns: pd.DataFrame, config: DialConfig) -> DialResult:
    """Put the dial on top of one strategy of a finished :class:`~market_analyzer.backtest.BacktestResult`.

    The baskets and the model's forecasts come from the backtest. ``returns``
    is the panel the backtest ran on.
    """
    if strategy not in result.weights:
        raise ValueError(f"The backtest has no weights for {strategy!r}.")
    rows = result.periods.loc[result.periods["strategy"] == strategy]
    model_vol = pd.Series(rows["predicted_vol"].to_numpy(dtype=float), index=pd.Index(rows["hold_start"]))
    return run_dial(result.weights[strategy], returns, config, model_vol=model_vol, end=rows["hold_end"].max())
