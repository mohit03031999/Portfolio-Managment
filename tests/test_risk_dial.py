import math

import numpy as np
import pandas as pd
import pytest

from market_analyzer.backtest import BacktestConfig, run_backtest
from market_analyzer.risk_dial import (
    CASH,
    DialConfig,
    basket_volatility,
    dial_backtest,
    exposure_for_target,
    recent_volatility,
    run_dial,
    split_with_cash,
)

ANNUAL = math.sqrt(252)


def _panel(seed: int, n_dates: int, daily_vol=0.01, n_assets: int = 4) -> pd.DataFrame:
    """Independent Gaussian stocks. ``daily_vol`` may be one number or one per date."""
    rng = np.random.default_rng(seed)
    scale = np.broadcast_to(np.asarray(daily_vol, dtype=float).reshape(-1, 1), (n_dates, 1))
    values = rng.normal(size=(n_dates, n_assets)) * scale
    index = pd.bdate_range("2015-01-01", periods=n_dates, name="Date")
    return pd.DataFrame(values, index=index, columns=[f"S{i}" for i in range(n_assets)])


def _baskets(panel: pd.DataFrame, first: int, every: int) -> pd.DataFrame:
    """Equal-weight baskets rebalanced every ``every`` sessions from position ``first``."""
    dates = panel.index[first::every]
    return pd.DataFrame(1.0 / panel.shape[1], index=dates, columns=panel.columns)


def test_exposure_is_target_over_forecast_up_to_the_cap():
    assert exposure_for_target(0.08, 0.16) == pytest.approx(0.5)
    assert exposure_for_target(0.08, 0.04) == pytest.approx(1.0)
    assert exposure_for_target(0.08, 0.04, max_exposure=3.0) == pytest.approx(2.0)
    assert exposure_for_target(0.08, 0.0) == pytest.approx(1.0)
    series = exposure_for_target(0.10, pd.Series([0.2, 0.05, np.nan], index=list("abc")))
    assert list(series.index) == list("abc")
    assert series.to_numpy() == pytest.approx([0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="negative"):
        exposure_for_target(-0.1, 0.2)
    with pytest.raises(ValueError, match="undefined"):
        exposure_for_target(0.1, 0.0, max_exposure=float("inf"))


def test_split_with_cash_sums_to_one_and_borrows_above_one():
    basket = pd.Series({"A": 0.6, "B": 0.4})
    split = split_with_cash(basket, 0.25)
    assert split.sum() == pytest.approx(1.0)
    assert split["A"] == pytest.approx(0.15)
    assert split[CASH] == pytest.approx(0.75)
    assert split_with_cash(basket, 1.5)[CASH] == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="already"):
        split_with_cash(pd.Series({"A": 0.5, CASH: 0.5}), 0.5)


def test_recent_volatility_units_and_the_pull_of_the_newest_days():
    flat = np.tile([0.01, -0.01], 30)
    assert recent_volatility(flat) == pytest.approx(0.01 * ANNUAL)
    assert recent_volatility(flat, halflife=5) == pytest.approx(0.01 * ANNUAL)
    calm_then_storm = np.concatenate([np.tile([0.01, -0.01], 30), np.tile([0.03, -0.03], 5)])
    assert recent_volatility(calm_then_storm, halflife=5) > recent_volatility(calm_then_storm) > 0.01 * ANNUAL
    assert recent_volatility(calm_then_storm, halflife=5) < 0.03 * ANNUAL


def test_basket_volatility_matches_a_hand_computation():
    panel = _panel(0, 300)
    weights = pd.Series({"S0": 0.7, "S1": 0.3, "S2": 0.0})
    path = panel[["S0", "S1"]].tail(63).to_numpy() @ np.array([0.7, 0.3])
    assert basket_volatility(weights, panel, lookback=63) == pytest.approx(math.sqrt((path**2).mean()) * ANNUAL)
    with pytest.raises(ValueError, match="No returns"):
        basket_volatility(pd.Series({"S0": 0.5, "ZZZ": 0.5}), panel)
    with pytest.raises(ValueError, match="Need 63 sessions"):
        basket_volatility(weights, panel.tail(10))


def test_a_correct_forecast_delivers_the_target():
    panel = _panel(1, 4000)
    weights = _baskets(panel, 100, 21)
    true_vol = 0.01 / math.sqrt(panel.shape[1]) * ANNUAL
    model = pd.Series(true_vol, index=weights.index)
    result = run_dial(weights, panel, DialConfig(0.03, forecast="model"), model_vol=model)
    daily = result.daily
    assert daily.index[0] == weights.index[0] and daily.index[-1] == panel.index[-1]
    assert daily["exposure"].to_numpy() == pytest.approx(0.03 / true_vol)
    assert daily["return"].to_numpy() == pytest.approx((daily["exposure"] * daily["basket_return"]).to_numpy())
    summary = result.summary()
    assert summary["delivered_over_target"] == pytest.approx(1.0, abs=0.04)
    assert summary["dial_turnover_per_year"] == pytest.approx(0.0)
    assert summary["share_at_cap"] == pytest.approx(0.0)


def test_the_cap_stops_the_dial_from_borrowing():
    panel = _panel(2, 600)
    weights = _baskets(panel, 100, 21)
    result = run_dial(weights, panel, DialConfig(0.50, lookback=63))
    assert (result.daily["exposure"] == 1.0).all()
    assert result.summary()["share_at_cap"] == pytest.approx(1.0)
    assert result.delivered_vol() == pytest.approx(result.summary()["basket_vol"])
    levered = run_dial(weights, panel, DialConfig(0.50, forecast="recent", lookback=63, max_exposure=20.0))
    assert (levered.daily["exposure"] > 1.0).all()


def test_cash_earns_its_rate_on_the_part_outside_the_basket():
    panel = _panel(3, 300) * 0.0
    weights = _baskets(panel, 100, 21)
    model = pd.Series(0.20, index=weights.index)
    result = run_dial(weights, panel, DialConfig(0.05, forecast="model", cash_rate=0.04), model_vol=model)
    assert result.daily["exposure"].to_numpy() == pytest.approx(0.25)
    assert result.daily["return"].to_numpy() == pytest.approx(0.75 * 0.04 / 252)


def test_the_split_uses_only_returns_before_the_day_it_applies():
    panel = _panel(4, 500)
    weights = _baskets(panel, 150, 21)
    config = DialConfig(0.03, forecast="recent", lookback=40, halflife=10, dial_every=5)
    before = run_dial(weights, panel, config).daily
    cut = 300
    shocked = panel.copy()
    shocked.iloc[cut:] *= 7.0
    after = run_dial(weights, shocked, config).daily
    day = panel.index[cut]
    assert after.loc[:day, "exposure"].to_numpy() == pytest.approx(before.loc[:day, "exposure"].to_numpy())
    assert after["exposure"].iloc[-1] < 0.5 * before["exposure"].iloc[-1]


def test_the_dial_moves_only_on_its_own_schedule():
    panel = _panel(5, 400)
    weights = _baskets(panel, 100, 20)
    monthly = run_dial(weights, panel, DialConfig(0.03, lookback=30, dial_every=None)).daily["exposure"]
    changed = monthly.index[monthly.diff().abs() > 0]
    assert set(changed) <= set(weights.index)
    weekly = run_dial(weights, panel, DialConfig(0.03, lookback=30, dial_every=5)).daily["exposure"]
    positions = panel.index.get_indexer(weekly.index[weekly.diff().abs() > 0]) - 100
    assert len(positions) > len(changed)
    assert (positions % 5 == 0).all()


def test_a_fast_dial_keeps_the_promise_through_a_storm_and_a_slow_one_does_not():
    n = 3000
    vol = np.where((np.arange(n) // 250) % 2 == 0, 0.006, 0.018)
    panel = _panel(6, n, daily_vol=vol)
    weights = _baskets(panel, 500, 21)
    stale = pd.Series(0.006 / 2 * ANNUAL, index=weights.index)
    slow = run_dial(weights, panel, DialConfig(0.03, forecast="model", max_exposure=10), model_vol=stale)
    fast = run_dial(
        weights, panel, DialConfig(0.03, forecast="recent", lookback=126, halflife=10, dial_every=5, max_exposure=10)
    )
    assert slow.summary()["delivered_over_target"] > 1.8
    assert fast.summary()["delivered_over_target"] == pytest.approx(1.0, abs=0.12)
    assert fast.by_year()["delivered_over_target"].max() < slow.by_year()["delivered_over_target"].max()
    cautious = run_dial(
        weights,
        panel,
        DialConfig(0.03, forecast="higher", lookback=126, halflife=10, dial_every=5, max_exposure=10),
        model_vol=stale,
    )
    assert (cautious.daily["exposure"] <= fast.daily["exposure"] + 1e-12).all()
    assert (cautious.daily["exposure"] <= slow.daily["exposure"] + 1e-12).all()


def test_dial_backtest_reuses_the_baskets_of_a_finished_backtest():
    rng = np.random.default_rng(7)
    n_dates, n_assets = 250 + 21 * 12, 20
    market = rng.normal(0.0, 0.01, size=n_dates)
    values = market[:, None] * rng.uniform(0.6, 1.4, size=n_assets) + rng.normal(size=(n_dates, n_assets)) * 0.012
    returns = pd.DataFrame(
        values, columns=[f"S{i:02d}" for i in range(n_assets)], index=pd.bdate_range("2015-01-01", periods=n_dates, name="Date")
    )
    strategies = ("longonly_clipped", "equal_weight")
    backtest = run_backtest(returns, BacktestConfig(window=250, holding=21, strategies=strategies))
    for name in strategies:
        dialled = dial_backtest(backtest, name, returns, DialConfig(0.05, forecast="higher", lookback=63))
        assert dialled.daily["basket_return"].to_numpy() == pytest.approx(backtest.daily_returns[name].to_numpy())
        assert list(dialled.daily.index) == list(backtest.daily_returns.index)
        summary = dialled.summary()
        assert 0.0 < summary["mean_exposure"] <= 1.0
        assert {"delivered_vol", "delivered_over_target", "wildest_year_vol", "max_drawdown"} <= set(summary)
        assert dialled.by_year()["sessions"].sum() <= len(dialled.daily)
    with pytest.raises(ValueError, match="no weights"):
        dial_backtest(backtest, "minvar_rie", returns, DialConfig(0.05))


def test_bad_inputs_are_refused_with_a_reason():
    panel = _panel(8, 300)
    weights = _baskets(panel, 100, 21)
    with pytest.raises(ValueError, match="Unknown forecast"):
        run_dial(weights, panel, DialConfig(0.05, forecast="crystal_ball"))
    with pytest.raises(ValueError, match="target_vol must be positive"):
        run_dial(weights, panel, DialConfig(0.0))
    with pytest.raises(ValueError, match="needs model_vol"):
        run_dial(weights, panel, DialConfig(0.05, forecast="model"))
    with pytest.raises(ValueError, match="needs 252 sessions"):
        run_dial(weights, panel, DialConfig(0.05))
    off_calendar = weights.copy()
    off_calendar.index = off_calendar.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="must be a date of the return panel"):
        run_dial(off_calendar, panel, DialConfig(0.05, lookback=63))
    with pytest.raises(ValueError, match="No returns"):
        run_dial(weights.assign(ZZZ=0.1), panel, DialConfig(0.05))
