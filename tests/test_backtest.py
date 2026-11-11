from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_analyzer.backtest import (
    DEFAULT_STRATEGIES,
    BacktestConfig,
    BacktestResult,
    run_backtest,
    strategy_names,
)
from market_analyzer.cli import main
from market_analyzer.optimizer import CovarianceEstimate


def _factor_market(seed: int, n_dates: int, n_assets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-factor returns and the covariance that generated them."""
    rng = np.random.default_rng(seed)
    market_vol = 0.01
    market = rng.normal(0.0, market_vol, size=n_dates)
    betas = rng.uniform(0.6, 1.4, size=n_assets)
    idio = rng.uniform(0.008, 0.02, size=n_assets)
    returns = market[:, None] * betas[None, :] + rng.normal(size=(n_dates, n_assets)) * idio
    tickers = [f"S{i:03d}" for i in range(n_assets)]
    frame = pd.DataFrame(returns, columns=tickers, index=pd.bdate_range("2015-01-01", periods=n_dates))
    frame.index.name = "Date"
    true_cov = market_vol**2 * np.outer(betas, betas) + np.diag(idio**2)
    return frame, pd.DataFrame(true_cov, index=tickers, columns=tickers)


def test_periods_line_up_and_equal_weight_is_one_over_n():
    window, holding, n_periods = 250, 21, 10
    returns, _ = _factor_market(0, window + holding * n_periods, 30)
    result = run_backtest(returns, BacktestConfig(window=window, holding=holding))

    assert set(result.periods["strategy"]) == set(DEFAULT_STRATEGIES)
    assert result.periods["hold_start"].nunique() == n_periods
    assert len(result.daily_returns) == holding * n_periods
    assert list(result.daily_returns.columns) == list(DEFAULT_STRATEGIES)
    assert not result.daily_returns.isna().any().any()
    first = result.periods.iloc[0]
    assert first["estimation_end"] < first["hold_start"]
    assert first["hold_start"] == returns.index[window]
    equal = result.weights["equal_weight"]
    assert equal.shape == (n_periods, 30)
    assert equal.to_numpy() == pytest.approx(np.full((n_periods, 30), 1 / 30))
    summary = result.summary()
    assert list(summary.index) == list(DEFAULT_STRATEGIES)
    assert summary.loc["equal_weight", "mean_turnover"] == pytest.approx(0.0)
    assert (summary["n_periods"] == n_periods).all()
    assert "vol_ratio" in result.report()


def test_cleaning_beats_the_sample_covariance_when_q_is_near_one():
    window, holding, n_periods = 120, 21, 25
    returns, _ = _factor_market(1, window + holding * n_periods, 100)
    result = run_backtest(returns, BacktestConfig(window=window, holding=holding))
    summary = result.summary()

    sample = summary.loc["minvar_sample"]
    clipped = summary.loc["minvar_clipped"]
    assert clipped["ann_volatility"] < sample["ann_volatility"]
    assert abs(clipped["median_vol_ratio"] - 1) < abs(sample["median_vol_ratio"] - 1)
    assert sample["median_vol_ratio"] > 1.5
    assert clipped["mean_gross_leverage"] < sample["mean_gross_leverage"]
    assert clipped["mean_turnover"] < sample["mean_turnover"]
    assert (result.periods.loc[result.periods["strategy"] == "minvar_clipped", "n_signal"] >= 1).all()
    assert not result.periods["singular"].any()


def test_the_true_covariance_is_an_honest_risk_model():
    window, holding, n_periods = 120, 21, 25
    returns, true_cov = _factor_market(2, window + holding * n_periods, 60)

    def oracle(panel: pd.DataFrame) -> CovarianceEstimate:
        return CovarianceEstimate(true_cov.loc[panel.columns, panel.columns])

    config = BacktestConfig(window=window, holding=holding, strategies=("minvar_oracle", "minvar_sample"))
    result = run_backtest(returns, config, estimators={"oracle": oracle})
    summary = result.summary()
    assert 0.8 < summary.loc["minvar_oracle", "median_vol_ratio"] < 1.25
    assert summary.loc["minvar_oracle", "ann_volatility"] < summary.loc["minvar_sample", "ann_volatility"]
    assert summary.loc["minvar_oracle", "mean_turnover"] == pytest.approx(0.0, abs=1e-9)


def test_membership_follows_the_estimation_window():
    window, holding, n_periods = 100, 10, 30
    returns, _ = _factor_market(3, window + holding * n_periods, 12)
    late = returns.columns[0]
    returns.loc[returns.index[: window + holding * 3], late] = np.nan  # first trades at index 130
    gap = returns.columns[1]
    returns.loc[returns.index[window + 2], gap] = np.nan  # one missing day inside the first hold

    result = run_backtest(returns, BacktestConfig(window=window, holding=holding))
    counts = result.periods.loc[result.periods["strategy"] == "equal_weight"].set_index("hold_start")["n_assets"]
    assert len(counts) == n_periods
    # Period 0: only the late ticker is missing. The gap then sits inside the
    # estimation window until the decision at index 210 (period 11), and the
    # late ticker needs a full window after index 130, so it joins at 230
    # (period 13).
    assert counts.iloc[0] == 11
    assert counts.iloc[1:11].tolist() == [10] * 10
    assert counts.iloc[11:13].tolist() == [11, 11]
    assert counts.iloc[13:].tolist() == [12] * (n_periods - 13)
    weights = result.weights["equal_weight"]
    assert weights[late].isna().iloc[0]
    assert weights[late].notna().iloc[-1]
    assert weights[gap].notna().iloc[0]
    assert weights[gap].isna().iloc[1]
    first = result.periods.loc[result.periods["hold_start"] == returns.index[window]]
    assert (first["filled_returns"] == 1).all()
    assert not result.daily_returns.isna().any().any()


def test_costs_are_charged_on_the_first_day_of_each_hold():
    window, holding, n_periods = 60, 10, 4
    returns, _ = _factor_market(4, window + holding * n_periods, 8)
    base = BacktestConfig(window=window, holding=holding, strategies=("minvar_sample",))
    free = run_backtest(returns, base)
    paid = run_backtest(returns, BacktestConfig(window=window, holding=holding, strategies=("minvar_sample",), cost_bps=10.0))
    difference = free.daily_returns["minvar_sample"] - paid.daily_returns["minvar_sample"]
    turnover = free.periods.set_index("hold_start")["turnover"]
    charged = difference[difference.abs() > 1e-15]
    assert len(charged) == n_periods
    assert charged.to_numpy() == pytest.approx((turnover * 10.0 / 1e4).to_numpy())


def test_bad_inputs_are_rejected():
    returns, _ = _factor_market(5, 80, 5)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(returns, BacktestConfig(window=30, holding=10, strategies=("minvar_magic",)))
    with pytest.raises(ValueError, match="Need at least"):
        run_backtest(returns, BacktestConfig(window=70, holding=21))
    with pytest.raises(ValueError, match="holding"):
        BacktestConfig(holding=1).validate()
    assert "minvar_clipped" in strategy_names()
    assert "longonly_clipped" in strategy_names()
    names = strategy_names({"oracle": None})
    assert names[-1] == "equal_weight"
    assert "minvar_oracle" in names and "longonly_oracle" in names
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(returns, BacktestConfig(window=30, holding=10, strategies=("longonly_magic",)))


def test_long_only_strategies_hold_no_shorts_and_predict_no_less_risk_than_unconstrained():
    returns, _ = _factor_market(11, 260, 8)
    config = BacktestConfig(window=120, holding=20, strategies=("minvar_sample", "longonly_sample", "minvar_clipped", "longonly_clipped"))
    result = run_backtest(returns, config)
    periods = result.periods
    for estimator in ("sample", "clipped"):
        short = periods.loc[periods["strategy"] == f"minvar_{estimator}"].set_index("hold_start")
        long = periods.loc[periods["strategy"] == f"longonly_{estimator}"].set_index("hold_start")
        assert (long["n_short"] == 0).all()
        assert long["gross_leverage"].to_numpy() == pytest.approx(1.0)
        assert (long["n_positions"] <= short["n_positions"]).all()
        # Same matrix, tighter constraints: the in-sample (predicted) variance cannot be lower.
        assert (long["predicted_vol"] >= short["predicted_vol"] * (1 - 1e-9)).all()
        weights = result.weights[f"longonly_{estimator}"]
        assert (weights.fillna(0.0) >= 0).all().all()
        assert weights.sum(axis=1).to_numpy() == pytest.approx(1.0)
    summary = result.summary()
    assert summary.loc["longonly_sample", "mean_n_short"] == 0
    assert summary.loc["longonly_sample", "mean_n_positions"] <= summary.loc["minvar_sample", "mean_n_positions"]


def test_gross_cap_strategies_respect_the_cap_and_sit_between_long_only_and_unconstrained():
    returns, _ = _factor_market(12, 260, 8)
    config = BacktestConfig(window=120, holding=20, strategies=("minvar_sample", "gross1.5_sample", "longonly_sample"))
    result = run_backtest(returns, config)
    periods = result.periods
    capped = periods.loc[periods["strategy"] == "gross1.5_sample"].set_index("hold_start")
    free = periods.loc[periods["strategy"] == "minvar_sample"].set_index("hold_start")
    long = periods.loc[periods["strategy"] == "longonly_sample"].set_index("hold_start")
    assert (capped["gross_leverage"] <= 1.5 + 1e-9).all()
    assert (capped["predicted_vol"] >= free["predicted_vol"] * (1 - 1e-9)).all()
    assert (capped["predicted_vol"] <= long["predicted_vol"] * (1 + 1e-9)).all()
    assert result.weights["gross1.5_sample"].sum(axis=1).to_numpy() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least 1"):
        run_backtest(returns, BacktestConfig(window=30, holding=10, strategies=("gross0.5_sample",)))
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(returns, BacktestConfig(window=30, holding=10, strategies=("gross2_magic",)))


def test_backtest_command_writes_the_result_files(tmp_path: Path, capsys):
    returns, _ = _factor_market(6, 60 + 10 * 5, 8)
    source = tmp_path / "returns.csv"
    returns.to_csv(source)
    out = tmp_path / "bt"
    code = main(["backtest", str(source), "--window", "60", "--holding", "10", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "vol_ratio" in printed
    assert (out / "summary.csv").is_file()
    assert (out / "periods.csv").is_file()
    assert (out / "daily_returns.csv").is_file()
    assert (out / "weights_minvar_clipped.csv").is_file()
    assert (out / "config.json").is_file()
    summary = pd.read_csv(out / "summary.csv", index_col=0)
    assert set(summary.index) == set(DEFAULT_STRATEGIES)
    daily = pd.read_csv(out / "daily_returns.csv", index_col=0, parse_dates=True)
    assert len(daily) == 50


def test_backtest_command_accepts_prices(tmp_path: Path):
    returns, _ = _factor_market(7, 60 + 10 * 3, 6)
    prices = 100 * (1 + returns).cumprod()
    prices.loc[prices.index[:5], prices.columns[0]] = np.nan
    source = tmp_path / "prices.csv"
    prices.to_csv(source)
    out = tmp_path / "bt"
    code = main(["backtest", str(source), "--from-prices", "--window", "60", "--holding", "10", "--out", str(out)])
    assert code == 0
    periods = pd.read_csv(out / "periods.csv")
    assert periods["n_assets"].max() == 6


def test_restrict_and_load_round_trip(tmp_path: Path):
    returns, _ = _factor_market(8, 60 + 10 * 6, 8)
    result = run_backtest(returns, BacktestConfig(window=60, holding=10))
    starts = sorted(result.periods["hold_start"].unique())
    cut = result.restrict(start=starts[2], end=result.periods["hold_end"].max())
    assert cut.periods["hold_start"].nunique() == 4
    assert len(cut.daily_returns) == 40
    assert all(len(frame) == 4 for frame in cut.weights.values())
    assert cut.summary()["n_periods"].eq(4).all()
    with pytest.raises(ValueError, match="No holding period"):
        result.restrict(start="2100-01-01")

    result.save(tmp_path)
    loaded = BacktestResult.load(tmp_path)
    assert loaded.config == result.config
    assert loaded.periods["hold_start"].dtype.kind == "M"
    assert isinstance(loaded.daily_returns.index, pd.DatetimeIndex)
    pd.testing.assert_frame_equal(loaded.daily_returns, result.daily_returns, check_freq=False)
    pd.testing.assert_frame_equal(loaded.summary(), result.summary())
    pd.testing.assert_frame_equal(
        loaded.weights["minvar_clipped"], result.weights["minvar_clipped"], check_freq=False
    )
