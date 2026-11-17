"""Point-in-time signals and the cross-sectional scoring of the signal lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analyzer.signal_lab import (
    cross_sectional_slope,
    decile_returns,
    decision_dates,
    rank_correlation,
    score_signal,
    sharpe_with_error,
    summary_table,
    top_decile_turnover,
)
from market_analyzer.signals import (
    MONTH,
    SIGNALS,
    YEAR,
    compute_signals,
    forward_return,
    historical_mean,
    market_beta,
    momentum,
    trailing_volatility,
)


def _panel(days: int = 900, stocks: int = 40, seed: int = 0, scale: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-02", periods=days)
    columns = [f"S{i:02d}" for i in range(stocks)]
    return pd.DataFrame(rng.normal(0.0, scale, (days, stocks)), index=index, columns=columns)


def test_forward_return_matches_hand_calculation():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03, 0.00]}, index=pd.bdate_range("2020-01-01", periods=5))
    ahead = forward_return(returns, horizon=2)
    assert ahead["A"].iloc[0] == pytest.approx(1.02 * 0.99 - 1)
    assert ahead["A"].iloc[2] == pytest.approx(1.03 * 1.00 - 1)
    assert np.isnan(ahead["A"].iloc[3]) and np.isnan(ahead["A"].iloc[4])
    later = forward_return(returns, horizon=2, offset=1)
    assert later["A"].iloc[0] == pytest.approx(0.99 * 1.03 - 1)


def test_momentum_of_constant_returns_is_the_compounded_window():
    returns = pd.DataFrame({"A": [0.001] * 400}, index=pd.bdate_range("2018-01-01", periods=400))
    signal = momentum(returns, lookback=YEAR, skip=MONTH, min_fraction=1.0)
    assert signal["A"].iloc[-1] == pytest.approx(1.001**YEAR - 1)
    assert np.isnan(signal["A"].iloc[YEAR + MONTH - 2])
    assert np.isfinite(signal["A"].iloc[YEAR + MONTH - 1])


def test_signals_use_past_data_only():
    returns = _panel()
    cutoff = returns.index[600]
    before = compute_signals(returns)
    altered = returns.copy()
    altered.loc[altered.index > cutoff] = 0.05
    after = compute_signals(altered)
    for key in SIGNALS:
        pd.testing.assert_frame_equal(before[key].loc[:cutoff], after[key].loc[:cutoff])
        assert not before[key].loc[cutoff:].iloc[1:].equals(after[key].loc[cutoff:].iloc[1:])


def test_volatility_and_mean_have_the_stated_units():
    returns = pd.DataFrame({"A": [0.01, -0.01] * 150}, index=pd.bdate_range("2018-01-01", periods=300))
    vol = trailing_volatility(returns, lookback=YEAR)["A"].iloc[-1]
    assert vol == pytest.approx(returns["A"].iloc[-YEAR:].std() * np.sqrt(YEAR))
    mean = historical_mean(pd.DataFrame({"A": [0.002] * 300}, index=returns.index))["A"].iloc[-1]
    assert mean == pytest.approx(0.002 * MONTH)


def test_beta_of_a_stock_equal_to_the_market_is_one():
    base = _panel(days=400, stocks=5)
    base["AVG"] = base.mean(axis=1)
    beta = market_beta(base, lookback=YEAR)
    assert beta["AVG"].iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_decision_dates_are_the_last_session_of_each_month():
    index = pd.bdate_range("2020-01-01", "2020-04-15")
    dates = decision_dates(index)
    assert list(dates.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-15"]


def test_rank_correlation_and_slope_recover_a_planted_edge():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=60)
    columns = [f"S{i}" for i in range(500)]
    z = pd.DataFrame(rng.normal(size=(60, 500)), index=dates, columns=columns)
    y = 0.02 * z + pd.DataFrame(rng.normal(0, 0.10, (60, 500)), index=dates, columns=columns)
    z.iloc[:, :10] = np.nan
    ic = rank_correlation(z, y)
    assert 0.1 < ic.mean() < 0.3
    slope = cross_sectional_slope(z, y, standardise=True)
    assert slope.mean() == pytest.approx(0.02, abs=0.004)
    raw = cross_sectional_slope(2 * z, y, standardise=False)
    assert raw.mean() == pytest.approx(0.01, abs=0.002)
    deciles = decile_returns(z, y)
    assert deciles.shape == (60, 10)
    assert deciles.mean().is_monotonic_increasing


def test_shuffled_outcomes_score_zero():
    returns = _panel(days=700, stocks=200, seed=3)
    dates = decision_dates(returns.index)[14:-2]
    signal = compute_signals(returns, ["momentum"])["momentum"]
    control = score_signal(signal, returns, dates, key="control", shuffle_seed=0, decay_months=(1, 2))
    assert abs(control.ic.mean()) < 0.03
    assert control.coverage.min() == 200
    assert set(control.decay.index) == {1, 2}


def test_score_signal_end_to_end_and_summary_columns():
    returns = _panel(days=700, stocks=120, seed=5)
    dates = decision_dates(returns.index)[14:-2]
    signal = compute_signals(returns, ["mean_1y"])["mean_1y"]
    score = score_signal(signal, returns, dates, key="mean_1y", label="Mean", in_return_units=True, decay_months=(1,))
    table = summary_table([score])
    assert table.index.tolist() == ["Mean"]
    for column in ("ic_mean", "ic_t", "spread_sharpe", "spread_sharpe_se", "slope_per_sd", "trust_fraction", "ic_month_1"):
        assert column in table.columns
    assert table.loc["Mean", "months"] == len(dates)
    assert score.spread.index.equals(dates)
    assert 0 <= score.turnover.dropna().min() <= score.turnover.dropna().max() <= 1


def test_top_decile_turnover_is_zero_for_a_frozen_ranking():
    frozen = pd.DataFrame(np.tile(np.arange(50.0), (4, 1)), index=pd.bdate_range("2020-01-01", periods=4))
    turnover = top_decile_turnover(frozen)
    assert np.isnan(turnover.iloc[0]) and turnover.iloc[1:].eq(0).all()


def test_sharpe_error_bar_shrinks_with_the_square_root_of_time():
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0.01, 0.04, 36))
    long = pd.Series(rng.normal(0.01, 0.04, 144))
    _, se_short = sharpe_with_error(short)
    _, se_long = sharpe_with_error(long)
    assert se_short / se_long == pytest.approx(2.0, rel=0.15)


def test_mask_restricts_the_cross_section_date_by_date():
    returns = _panel(days=700, stocks=100, seed=7)
    dates = decision_dates(returns.index)[14:-2]
    signal = compute_signals(returns, ["momentum"])["momentum"]
    mask = pd.DataFrame(False, index=dates, columns=returns.columns)
    mask.iloc[:, :40] = True
    mask.iloc[0, 40:70] = True
    score = score_signal(signal, returns, dates, mask=mask, decay_months=(1,))
    assert score.coverage.iloc[0] == 70 and score.coverage.iloc[1:].eq(40).all()
