from pathlib import Path

import pytest

from market_analyzer.universe import load_universe, tickers_from_html_table, wikipedia_yahoo_symbol

ROOT = Path(__file__).resolve().parents[1]


def test_example_universe_keeps_exchange_suffixes():
    spec = load_universe(ROOT / "config" / "universe.example.yaml")
    assert spec.name == "global-sample"
    assert spec.start == "2020-01-01"
    assert spec.end is None
    assert spec.tickers[0] == "AAPL"
    assert "RELIANCE.NS" in spec.tickers
    assert "BRK-B" in spec.tickers
    assert "7203.T" in spec.tickers


def test_unquoted_boolean_ticker_is_rejected(tmp_path: Path):
    path = tmp_path / "on.yaml"
    path.write_text('name: semis\ntickers:\n  - ON\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Quote"):
        load_universe(path)


def test_quoted_on_ticker_is_kept(tmp_path: Path):
    path = tmp_path / "on.yaml"
    path.write_text('name: semis\ntickers:\n  - "ON"\n  - "AAPL"\n', encoding="utf-8")
    spec = load_universe(path)
    assert spec.tickers == ["ON", "AAPL"]


def test_wikipedia_class_share_maps_to_yahoo():
    assert wikipedia_yahoo_symbol("BRK.B") == "BRK-B"
    assert wikipedia_yahoo_symbol("RELIANCE.NS") == "RELIANCE-NS"


def test_html_table_can_map_class_shares(tmp_path: Path):
    page = tmp_path / "constituents.html"
    page.write_text(
        """
        <table>
          <tr><th>Symbol</th><th>Security</th></tr>
          <tr><td>BRK.B</td><td>Berkshire</td></tr>
          <tr><td>AAPL</td><td>Apple</td></tr>
          <tr><td>AAPL</td><td>Apple again</td></tr>
        </table>
        """,
        encoding="utf-8",
    )
    mapped = tickers_from_html_table(str(page), column="symbol", wikipedia_class_shares=True)
    raw = tickers_from_html_table(str(page), column="Symbol", wikipedia_class_shares=False)
    assert mapped == ["BRK-B", "AAPL"]
    assert raw == ["BRK.B", "AAPL"]


def test_universe_file_accepts_a_local_table(tmp_path: Path):
    page = tmp_path / "constituents.html"
    page.write_text(
        """
        <table>
          <tr><th>Ticker</th></tr>
          <tr><td>TCS.NS</td></tr>
          <tr><td>INFY.NS</td></tr>
        </table>
        """,
        encoding="utf-8",
    )
    config = tmp_path / "universe.yaml"
    config.write_text(
        f"""
        name: india
        start: "2024-01-01"
        source:
          type: html_table
          url: "{page}"
          column: Ticker
          wikipedia_class_shares: false
        """,
        encoding="utf-8",
    )
    spec = load_universe(config)
    assert spec.tickers == ["TCS.NS", "INFY.NS"]


def test_universe_rejects_both_tickers_and_source(tmp_path: Path):
    path = tmp_path / "both.yaml"
    path.write_text(
        """
        name: both
        tickers: ["AAPL"]
        source:
          type: html_table
          url: "https://example.com"
          column: Symbol
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_universe(path)
