"""The history checks and the pack files, on synthetic prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_analyzer.market_pack import (
    NOTE_YOUNG,
    REASON_ANOMALY,
    REASON_GAPS,
    REASON_ILLIQUID,
    REASON_NO_PRICES,
    REASON_PRICE,
    REASON_STALE,
    QualityRules,
    _field_frame,
    assess_history,
    build_manifest,
    daily_returns_panel,
    fetch_pack,
    format_report,
    load_pack,
    write_pack,
)


def _prices(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=1500)
    n = len(index)

    def walk(scale: float = 0.01, start: float = 50.0) -> np.ndarray:
        return start * np.exp(np.cumsum(rng.normal(0.0, scale, n)))

    close = pd.DataFrame(index=index)
    volume = pd.DataFrame(index=index)
    close["GOOD"] = walk()
    volume["GOOD"] = 1e6
    close["YOUNG"] = walk()
    close.loc[index[: n - 300], "YOUNG"] = np.nan
    volume["YOUNG"] = 1e6
    close["STALE"] = walk()
    close.loc[index[n - 40 :], "STALE"] = np.nan
    volume["STALE"] = 1e6
    close["PENNY"] = walk(start=0.5)
    volume["PENNY"] = 1e8
    close["THIN"] = walk()
    volume["THIN"] = 1e3
    close["GAPPY"] = walk()
    gaps = rng.choice(np.arange(n - 504, n), size=60, replace=False)
    close.iloc[gaps, close.columns.get_loc("GAPPY")] = np.nan
    volume["GAPPY"] = 1e6
    close["SPIKE"] = walk()
    close.iloc[n - 100, close.columns.get_loc("SPIKE")] *= 10.0
    volume["SPIKE"] = 1e6
    close["EMPTY"] = np.nan
    volume["EMPTY"] = np.nan
    return close, volume


def test_each_rule_catches_its_example():
    close, volume = _prices()
    table = assess_history(close, volume)
    reasons = table["reason"]
    assert reasons["GOOD"] == ""
    assert table.loc["GOOD", "eligible"] and not table.loc["GOOD", "young"]
    assert reasons["YOUNG"] == ""
    assert table.loc["YOUNG", "young"] and not table.loc["YOUNG", "eligible"]
    assert reasons["STALE"] == REASON_STALE
    assert reasons["PENNY"] == REASON_PRICE
    assert reasons["THIN"] == REASON_ILLIQUID
    assert reasons["GAPPY"] == REASON_GAPS
    assert reasons["SPIKE"] == REASON_ANOMALY
    assert reasons["EMPTY"] == REASON_NO_PRICES
    assert table.loc["GOOD", "history_years"] == pytest.approx(1499 / 252 * 252 / 365.25 * 365.25 / 252, rel=0.2)
    assert 0.1 < table.loc["GAPPY", "missing_recent"] < 0.15
    assert table.loc["YOUNG", "missing_recent"] == 0.0


def test_rules_are_adjustable():
    close, volume = _prices()
    loose = QualityRules(min_median_dollar_volume=1.0, min_last_price=0.0, max_missing_fraction=0.5, max_abs_return=100.0, stale_sessions=100, min_history_years=1.0)
    table = assess_history(close, volume, loose)
    assert (table.loc[["STALE", "PENNY", "THIN", "GAPPY", "SPIKE"], "reason"] == "").all()
    assert table.loc["YOUNG", "eligible"]


def test_daily_returns_panel_keeps_gaps_and_is_float32():
    close, _ = _prices()
    returns = daily_returns_panel(close[["GOOD", "GAPPY"]])
    assert returns.dtypes.eq("float32").all()
    assert len(returns) == len(close) - 1
    expected = close["GOOD"].iloc[1] / close["GOOD"].iloc[0] - 1.0
    assert returns["GOOD"].iloc[0] == pytest.approx(expected, rel=1e-5)
    assert returns["GAPPY"].isna().sum() >= 60


def test_field_frame_handles_the_yfinance_column_shapes():
    index = pd.bdate_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["AAA", "BBB"]])
    raw = pd.DataFrame(np.arange(12.0).reshape(3, 4), index=index, columns=columns)
    close = _field_frame(raw, "Adj Close", ["AAA", "BBB"])
    assert list(close.columns) == ["AAA", "BBB"]
    assert close.iloc[0, 0] == 0.0
    flat = pd.DataFrame({"Adj Close": [1.0, 2.0, 3.0], "Volume": [1, 1, 1]}, index=index)
    single = _field_frame(flat, "Adj Close", ["AAA"])
    assert list(single.columns) == ["AAA"]


def test_pack_round_trip_and_report(tmp_path):
    close, volume = _prices()
    checks = assess_history(close, volume)
    kept = checks.index[checks["reason"] == ""]
    returns = daily_returns_panel(close.loc[:, kept])
    facts = pd.DataFrame(
        {"name": ["Good Co", "Young Co"], "exchange": ["NYSE", "NASDAQ"], "sector": ["Technology", "Health Care"]},
        index=pd.Index(["GOOD", "YOUNG"], name="ticker"),
    )
    meta = facts.join(checks, how="right").loc[kept]
    manifest = build_manifest(
        "2019-01-01", returns, meta, checks, QualityRules(), {"listed": 10, "candidates": 8, "exchange-traded fund": 2}, 1e8, {"prices": "test"}, ["DEAD"]
    )
    report = format_report(manifest, checks.join(facts, how="left"))
    folder = write_pack(tmp_path / "pack", returns, meta, manifest, report)
    pack = load_pack(folder)
    assert pack.tickers == ["GOOD", "YOUNG"]
    assert pack.eligible == ["GOOD"]
    assert pack.returns.dtypes.eq("float32").all()
    assert pack.manifest["n_tickers"] == 2 and pack.manifest["n_young"] == 1
    assert pack.last_date == returns.index[-1]
    text = (folder / "report.txt").read_text()
    assert "exchange-traded fund" in text and NOTE_YOUNG in text and "Technology" in text
    assert pack.manifest["history_counts"]["stopped trading"] == 1
    assert "stopped trading" in text


def test_fetch_pack_downloads_once_per_build(tmp_path):
    close, volume = _prices()
    checks = assess_history(close, volume)
    kept = checks.index[checks["reason"] == ""]
    returns = daily_returns_panel(close.loc[:, kept])
    meta = checks.loc[kept]
    manifest = build_manifest("2019-01-01", returns, meta, checks, QualityRules(), {}, 0.0, {}, [])
    published = write_pack(tmp_path / "published", returns, meta, manifest, "report")
    cache = tmp_path / "cache"
    folder = fetch_pack(published.as_uri(), cache)
    pack = load_pack(folder)
    assert pack.tickers == list(kept)
    stamp = (cache / "returns.parquet").stat().st_mtime_ns
    fetch_pack(published.as_uri(), cache)
    assert (cache / "returns.parquet").stat().st_mtime_ns == stamp
    manifest["built_at"] = "2030-01-01T00:00:00+00:00"
    write_pack(published, returns, meta, manifest, "report")
    fetch_pack(published.as_uri(), cache)
    assert load_pack(cache).manifest["built_at"] == "2030-01-01T00:00:00+00:00"
