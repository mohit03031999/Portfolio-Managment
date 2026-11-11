"""The basket engine on a synthetic market pack."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analyzer.basket import (
    RULES,
    analyse_basket,
    backtest_basket,
    describe_position,
    strategy_name,
)
from market_analyzer.market_pack import NOTE_YOUNG, MarketPack


def _pack(seed: int = 0, n_days: int = 1600) -> MarketPack:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-02", periods=n_days)
    tickers = ["CALM", "JUMPY", "ALPHA", "BETA", "GAMMA", "DELTA", "TWIN", "YOUNG"]
    market = rng.normal(0.0, 0.01, n_days)
    sector = rng.normal(0.0, 0.006, n_days)
    columns = {
        "CALM": 0.4 * market + rng.normal(0.0, 0.004, n_days),
        "JUMPY": 1.4 * market + rng.normal(0.0, 0.025, n_days),
        "ALPHA": market + sector + rng.normal(0.0, 0.008, n_days),
        "BETA": market + sector + rng.normal(0.0, 0.008, n_days),
        "GAMMA": market - sector + rng.normal(0.0, 0.008, n_days),
        "DELTA": 0.8 * market + rng.normal(0.0, 0.009, n_days),
    }
    columns["TWIN"] = columns["ALPHA"] + rng.normal(0.0, 0.001, n_days)
    young = np.full(n_days, np.nan)
    young[-200:] = market[-200:] + rng.normal(0.0, 0.01, 200)
    columns["YOUNG"] = young
    returns = pd.DataFrame(columns, index=index, columns=tickers).astype("float32")
    returns.index.name = "Date"
    meta = pd.DataFrame(
        {
            "name": [f"{t.title()} Inc." for t in tickers],
            "sector": ["Utilities", "Technology", "Finance", "Finance", "Finance", "Industrials", "Finance", "Technology"],
            "eligible": [True] * 7 + [False],
            "young": [False] * 7 + [True],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return MarketPack(returns=returns, meta=meta, manifest={"n_tickers": len(tickers)})


def test_strategy_names_follow_the_backtest_convention():
    assert strategy_name("long_only", "rie") == "longonly_rie"
    assert strategy_name("cap2", "sample") == "gross2_sample"
    assert strategy_name("unconstrained", "clipped") == "minvar_clipped"
    with pytest.raises(ValueError):
        strategy_name("magic", "rie")
    with pytest.raises(ValueError):
        strategy_name("long_only", "magic")


def test_long_only_analysis_is_a_feasible_improvement_on_equal_weight():
    pack = _pack()
    analysis = analyse_basket(pack, ["CALM", "JUMPY", "ALPHA", "BETA", "GAMMA", "DELTA", "TWIN"], "long_only", "rie", 1000)
    assert analysis.window == 1000
    assert analysis.tickers == ["CALM", "JUMPY", "ALPHA", "BETA", "GAMMA", "DELTA", "TWIN"]
    assert analysis.weights.sum() == pytest.approx(1.0)
    assert (analysis.weights >= -1e-9).all()
    assert analysis.optimised_vol <= analysis.equal_vol + 1e-9
    assert 0.0 <= analysis.reduction < 1.0
    assert analysis.equal_realised_vol == pytest.approx(analysis.equal_vol, rel=0.25)
    assert analysis.weights["CALM"] > analysis.weights["JUMPY"]
    assert analysis.stats["gross_leverage"] == pytest.approx(1.0)
    assert analysis.honesty == RULES["long_only"].honesty
    table = analysis.table
    assert list(table.columns) == ["ticker", "name", "sector", "weight", "equal_weight", "own_vol", "avg_corr", "risk_share", "reason"]
    assert table["weight"].is_monotonic_decreasing
    assert table["risk_share"].sum() == pytest.approx(1.0)
    assert table["reason"].str.len().gt(0).all()
    assert table.loc[table["ticker"] == "CALM", "reason"].iloc[0].startswith("big weight")


def test_young_and_duplicate_tickers_are_reported():
    pack = _pack()
    analysis = analyse_basket(pack, ["CALM", "ALPHA", "ALPHA", "YOUNG"], "long_only", "sample", 500)
    assert analysis.tickers == ["CALM", "ALPHA"]
    assert analysis.excluded == {"YOUNG": NOTE_YOUNG}
    with pytest.raises(ValueError, match="at least two"):
        analyse_basket(pack, ["CALM", "YOUNG"], "long_only", "rie", 500)
    with pytest.raises(ValueError, match="Not in the market pack"):
        analyse_basket(pack, ["CALM", "NOPE"], "long_only", "rie", 500)


def test_rules_nest_as_in_the_study():
    pack = _pack()
    basket = ["CALM", "JUMPY", "ALPHA", "BETA", "GAMMA", "DELTA", "TWIN"]
    long_only = analyse_basket(pack, basket, "long_only", "rie", 1000)
    capped = analyse_basket(pack, basket, "cap2", "rie", 1000)
    free = analyse_basket(pack, basket, "unconstrained", "rie", 1000)
    assert free.optimised_vol <= capped.optimised_vol + 1e-9 <= long_only.optimised_vol + 2e-9
    assert capped.stats["gross_leverage"] <= 2.0 + 1e-6
    assert free.stats["n_short"] >= 1


def test_describe_position_covers_every_branch():
    assert describe_position(-0.1, 0.1, 0.5, 0.9).startswith("shorted")
    assert describe_position(0.0, 0.1, 0.9, 0.9).startswith("left out: jumpy")
    assert describe_position(0.0, 0.1, 0.2, 0.9).startswith("left out: moves")
    assert describe_position(0.0, 0.1, 0.9, 0.2).startswith("left out: jumpier")
    assert describe_position(0.0, 0.1, 0.5, 0.5).startswith("left out: the others")
    assert describe_position(0.3, 0.1, 0.1, 0.1) == "big weight: calm and moves on its own"
    assert describe_position(0.3, 0.1, 0.1, 0.9).startswith("big weight: among")
    assert describe_position(0.3, 0.1, 0.9, 0.1).startswith("big weight: moves on its own")
    assert describe_position(0.3, 0.1, 0.5, 0.5).startswith("big weight: a useful")
    assert describe_position(0.03, 0.1, 0.9, 0.9).startswith("trimmed: jumpy")
    assert describe_position(0.03, 0.1, 0.9, 0.1).startswith("trimmed: jumpier")
    assert describe_position(0.03, 0.1, 0.1, 0.9).startswith("trimmed: moves")
    assert describe_position(0.03, 0.1, 0.5, 0.5).startswith("trimmed: others")
    assert describe_position(0.1, 0.1, 0.5, 0.5).startswith("near an equal")


def test_backtest_basket_reports_both_books_and_the_worst_month():
    pack = _pack()
    basket = ["CALM", "JUMPY", "ALPHA", "BETA", "GAMMA", "DELTA", "TWIN", "YOUNG"]
    test = backtest_basket(pack, basket, "long_only", "rie", window=500, years=2.0, holding=21)
    assert test.strategy == "longonly_rie"
    assert set(test.summary.index) == {"equal_weight", "longonly_rie"}
    assert test.summary.loc["longonly_rie", "ann_volatility"] < test.summary.loc["equal_weight", "ann_volatility"]
    assert 0.5 < test.summary.loc["longonly_rie", "median_vol_ratio"] < 1.5
    assert list(test.growth.columns) == ["equal_weight", "longonly_rie"]
    assert test.growth.iloc[0].between(95, 105).all()
    assert test.worst["vol_ratio"] >= test.summary.loc["longonly_rie", "median_vol_ratio"]
    assert test.end > test.start
    # the young ticker joins once its window is complete, so late decisions see more names
    assert test.result.periods["n_assets"].max() >= 7
