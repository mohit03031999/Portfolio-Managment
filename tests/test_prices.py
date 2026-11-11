import sys
import types

import pandas as pd
import pytest

from market_analyzer.prices import download_prices, extract_adjusted_close


def _index(count: int = 3) -> pd.DatetimeIndex:
    return pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"][:count])


def test_extract_column_grouped_multiindex():
    columns = pd.MultiIndex.from_product([["Adj Close", "Close"], ["AAPL", "MSFT"]])
    raw = pd.DataFrame(
        [
            [100.0, 50.0, 101.0, 51.0],
            [110.0, 50.0, 111.0, 51.0],
            [121.0, 55.0, 122.0, 56.0],
        ],
        index=_index(),
        columns=columns,
    )
    prices, missing = extract_adjusted_close(raw, ["MSFT", "AAPL", "NOPE"])
    assert missing == ["NOPE"]
    assert list(prices.columns) == ["MSFT", "AAPL"]
    assert prices.iloc[1]["AAPL"] == pytest.approx(110.0)


def test_extract_ticker_grouped_multiindex():
    columns = pd.MultiIndex.from_tuples(
        [("AAPL", "Adj Close"), ("AAPL", "Close"), ("MSFT", "Adj Close")]
    )
    raw = pd.DataFrame(
        [[10.0, 11.0, 20.0], [12.0, 13.0, 22.0]],
        index=_index(2),
        columns=columns,
    )
    prices, missing = extract_adjusted_close(raw, ["AAPL", "MSFT"])
    assert missing == []
    assert prices.iloc[0].tolist() == pytest.approx([10.0, 20.0])


def test_extract_single_ticker_ohlc_frame():
    raw = pd.DataFrame(
        {"Open": [1.0, 1.0], "Close": [9.0, 9.0], "Adj Close": [100.0, 110.0]},
        index=_index(2),
    )
    prices, missing = extract_adjusted_close(raw, ["RELIANCE.NS"])
    assert missing == []
    assert list(prices.columns) == ["RELIANCE.NS"]
    assert prices.iloc[1, 0] == pytest.approx(110.0)


def test_extract_drops_all_nan_column():
    columns = pd.MultiIndex.from_product([["Adj Close"], ["AAPL", "DEAD"]])
    raw = pd.DataFrame(
        [[100.0, None], [110.0, None]],
        index=_index(2),
        columns=columns,
    )
    prices, missing = extract_adjusted_close(raw, ["AAPL", "DEAD"])
    assert missing == ["DEAD"]
    assert list(prices.columns) == ["AAPL"]


def test_download_requests_adjusted_close(monkeypatch):
    calls: dict[str, object] = {}

    def download(tickers, **kwargs):
        calls["tickers"] = tickers
        calls["kwargs"] = kwargs
        columns = pd.MultiIndex.from_product([["Adj Close"], ["AAPL"]])
        return pd.DataFrame([[100.0], [110.0]], index=_index(2), columns=columns)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))
    prices, missing = download_prices(["AAPL"], "2024-01-01", "2024-02-01")

    assert calls["tickers"] == ["AAPL"]
    assert calls["kwargs"]["auto_adjust"] is False
    assert calls["kwargs"]["end"] == "2024-02-01"
    assert missing == []
    assert prices.iloc[1, 0] == pytest.approx(110.0)
