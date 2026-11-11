import pandas as pd
import pytest

from market_analyzer.analyzer import summarize


def test_summary_compounds_the_return_path():
    returns = pd.DataFrame({"A": [0.10, -0.20, 0.05]})
    summary = summarize(returns, periods_per_year=252)

    assert summary.loc["A", "observations"] == 3
    assert summary.loc["A", "total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    # Wealth path: 1 → 1.1 → 0.88 → 0.924. The trough is 0.88 / 1.1 - 1.
    assert summary.loc["A", "max_drawdown"] == pytest.approx(-0.2)


def test_constant_daily_return_annualizes_geometrically():
    returns = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01]})
    summary = summarize(returns, periods_per_year=252)

    assert summary.loc["A", "ann_return"] == pytest.approx((1.01**4) ** (252 / 4) - 1)
    assert summary.loc["A", "ann_volatility"] == pytest.approx(0.0)
    assert summary.loc["A", "vol_daily"] == pytest.approx(0.0)


def test_periods_per_year_must_be_positive():
    with pytest.raises(ValueError, match="periods_per_year"):
        summarize(pd.DataFrame({"A": [0.01]}), periods_per_year=0)
