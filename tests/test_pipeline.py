from pathlib import Path

import pandas as pd
import pytest

from market_analyzer.pipeline import build_market_data


def test_build_records_missing_symbols_and_shared_calendar_note(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=10)

    def download(tickers, start, end):
        frame = pd.DataFrame(index=index, columns=tickers, dtype=float)
        frame["A"] = [float(value) for value in range(100, 110)]
        frame["B"] = [None] * 8 + [50.0, 55.0]
        frame = frame.drop(columns=[ticker for ticker in tickers if ticker == "GONE"], errors="ignore")
        frame.index.name = "Date"
        missing = [ticker for ticker in tickers if ticker not in frame.columns]
        return frame, missing

    monkeypatch.setattr("market_analyzer.pipeline.download_prices", download)
    data = build_market_data(
        tickers=["A", "B", "GONE"],
        start="2024-01-01",
        end="2024-03-01",
        name="mixed calendar",
    )

    assert data.name == "mixed-calendar"
    assert data.missing_tickers == ["GONE"]
    assert data.shape[1] == 2
    assert data.shape[0] < 8
    assert any("shared by every ticker" in note for note in data.notes)


def test_end_must_follow_start():
    with pytest.raises(ValueError, match="end must be after start"):
        build_market_data(tickers=["AAPL"], start="2024-06-01", end="2024-06-01")


def test_universe_dates_apply_when_the_file_is_used(tmp_path: Path, monkeypatch):
    path = tmp_path / "basket.yaml"
    path.write_text(
        """
        name: file-basket
        start: "2023-01-01"
        end: "2024-01-01"
        tickers: ["INFY.NS"]
        """,
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def download(tickers, start, end):
        seen["tickers"] = tickers
        seen["start"] = start
        seen["end"] = end
        index = pd.to_datetime(["2023-01-03", "2023-01-04"])
        frame = pd.DataFrame({"INFY.NS": [100.0, 101.0]}, index=index)
        frame.index.name = "Date"
        return frame, []

    monkeypatch.setattr("market_analyzer.pipeline.download_prices", download)
    data = build_market_data(universe_path=path)
    assert seen == {"tickers": ["INFY.NS"], "start": "2023-01-01", "end": "2024-01-01"}
    assert data.returns.iloc[0, 0] == pytest.approx(0.01)
