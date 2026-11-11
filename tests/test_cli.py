from pathlib import Path

import pandas as pd

from market_analyzer.cli import main


def _install_download(monkeypatch, prices: dict[str, list[float]], index: pd.DatetimeIndex):
    def download(tickers, start, end):
        frame = pd.DataFrame({ticker: prices[ticker] for ticker in tickers if ticker in prices}, index=index)
        frame.index.name = "Date"
        missing = [ticker for ticker in tickers if ticker not in prices]
        return frame, missing

    monkeypatch.setattr("market_analyzer.pipeline.download_prices", download)


def test_cli_requires_one_ticker_source(capsys):
    assert main(["pull"]) == 1
    assert "exactly one" in capsys.readouterr().err


def test_cli_writes_the_return_matrix(tmp_path: Path, monkeypatch, capsys):
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    _install_download(
        monkeypatch,
        {"AAPL": [100.0, 110.0, 121.0], "MSFT": [50.0, 55.0, 55.0]},
        index,
    )
    code = main(
        [
            "pull",
            "--tickers",
            "AAPL",
            "MSFT",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--name",
            "my basket",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    returns = pd.read_csv(tmp_path / "my-basket" / "returns.csv", index_col=0)
    assert returns.shape == (2, 2)
    assert "Return matrix: 2 dates × 2 tickers" in capsys.readouterr().out
    assert (tmp_path / "my-basket" / "prices.csv").is_file()
    assert (tmp_path / "my-basket" / "summary.csv").is_file()
    assert (tmp_path / "my-basket" / "correlation.csv").is_file()


def test_strict_mode_stops_on_a_missing_symbol(tmp_path: Path, monkeypatch, capsys):
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    _install_download(monkeypatch, {"AAPL": [100.0, 110.0]}, index)
    code = main(
        [
            "pull",
            "--tickers",
            "AAPL",
            "ZZZZ",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--out",
            str(tmp_path),
            "--strict",
        ]
    )
    assert code == 1
    assert "ZZZZ" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
