import pandas as pd
import pytest

from market_analyzer.returns import daily_returns


def _prices() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 121.0],
            "B": [50.0, 50.0, None, 60.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    frame.index.name = "Date"
    return frame


def test_intersection_keeps_only_complete_dates():
    returns = daily_returns(_prices(), align="intersection")
    assert list(returns.index) == [pd.Timestamp("2024-01-03")]
    assert returns.loc[pd.Timestamp("2024-01-03"), "A"] == pytest.approx(0.1)
    assert returns.loc[pd.Timestamp("2024-01-03"), "B"] == pytest.approx(0.0)


def test_union_keeps_partial_dates():
    returns = daily_returns(_prices(), align="union")
    assert len(returns) == 3
    assert pd.isna(returns.loc[pd.Timestamp("2024-01-04"), "B"])
    assert returns.loc[pd.Timestamp("2024-01-04"), "A"] == pytest.approx(0.1)


def test_symbol_with_one_price_is_removed():
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [5.0, None, None],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    returns = daily_returns(prices, align="intersection")
    assert list(returns.columns) == ["A"]
    assert len(returns) == 2
    assert returns.iloc[0, 0] == pytest.approx(0.1)


def test_unknown_alignment_is_rejected():
    with pytest.raises(ValueError, match="align"):
        daily_returns(_prices(), align="outer")
