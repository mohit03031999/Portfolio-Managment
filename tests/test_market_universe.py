"""The listing rules, run on small hand-made symbol files."""

from __future__ import annotations

import pandas as pd

from market_analyzer.market_universe import (
    REASON_CAP,
    REASON_ETF,
    REASON_FUND,
    REASON_INDUSTRY,
    REASON_NAME,
    REASON_NO_CAP,
    REASON_STATUS,
    REASON_SYMBOL,
    REASON_TEST,
    candidate_table,
    display_name,
    parse_screener,
    parse_symbol_directory,
    yahoo_symbol,
)

NASDAQ_TEXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZT|Zapata Test Issue - Common Stock|G|Y|N|100|N|N
BADD|Bad Debt Corp. - Common Stock|S|N|D|100|N|N
ABCDW|Alpha Blank Corp. - Warrant|G|N|N|100|N|N
SPAC|Shiny Acquisition Corp. - Class A Ordinary Shares|G|N|N|100|N|N
HBANP|Huntington Bancshares - Depositary Shares 4.5% Preferred|G|N|N|100|N|N
NTRS|Northern Trust Corporation - Common Stock|Q|N|N|100|N|N
ARCC|Ares Capital Corporation - Closed End Fund|Q|N|N|100|N|N
TINY|Tiny Micro Cap Inc. - Common Stock|S|N|N|100|N|N
File Creation Time: 1111202617:01|||||||
"""

OTHER_TEXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK/B|N|100|N|BRK.B
AAC.U|Ares Acquisition Corporation III Units|N|AAC.U|N|100|N|AAC=
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
CBOE|Cboe Global Markets, Inc. Common Stock|Z|CBOE|N|100|N|CBOE
BDJ|Blackrock Enhanced Equity Dividend Trust Common Shares of Beneficial Interest|N|BDJ|N|100|N|BDJ
AKR|Acadia Realty Trust Common Stock|N|AKR|N|100|N|AKR
GIW|GigCapital8 Corp. Common Stock|N|GIW|N|100|N|GIW
GJT|Synthetic Fixed-Income Securities, Inc. Floating Rate STRATS Certificates|N|GJT|N|100|N|GJT
NOCAP|No Market Value Corp. Common Stock|N|NOCAP|N|100|N|NOCAP
HQH|abrdn Healthcare Investors Common Shares of Beneficial Interest|N|HQH|N|100|N|HQH
File Creation Time: 1111202617:01||||||
"""


def _screener() -> pd.DataFrame:
    rows = [
        ("AAPL", "Apple Inc. Common Stock", "Technology", "Computer Manufacturing", "3000000000000", "United States", "1980"),
        ("A", "Agilent Technologies Inc.", "Industrials", "Biotechnology: Laboratory Analytical Instruments", "50000000000", "United States", "1999"),
        ("BRK/B", "Berkshire Hathaway Inc.", "", "", "1000000000000", "United States", ""),
        ("CBOE", "Cboe Global Markets", "Finance", "Investment Bankers/Brokers/Service", "20000000000", "United States", "2010"),
        ("BDJ", "Blackrock Enhanced Equity Dividend Trust", "Finance", "Finance Companies", "1700000000", "United States", ""),
        ("AKR", "Acadia Realty Trust", "Real Estate", "Real Estate Investment Trusts", "2700000000", "United States", ""),
        ("NTRS", "Northern Trust Corporation", "Finance", "Major Banks", "32000000000", "United States", ""),
        ("GIW", "GigCapital8 Corp.", "Finance", "Blank Checks", "0.00", "United States", "2025"),
        ("TINY", "Tiny Micro Cap Inc.", "Health Care", "Biotechnology", "40000000", "United States", "2021"),
        ("NOCAP", "No Market Value Corp.", "Industrials", "Industrial Machinery", "0.00", "United States", ""),
        ("HQH", "abrdn Healthcare Investors", "Finance", "Finance Companies", "1000000000", "United States", ""),
        ("ABR^D", "Arbor Realty Trust preferred", "Finance", "Real Estate", "100", "United States", ""),
    ]
    return parse_screener(
        pd.DataFrame(rows, columns=["symbol", "name", "sector", "industry", "marketCap", "country", "ipoyear"])
    )


def _table(**kwargs):
    nasdaq = parse_symbol_directory(NASDAQ_TEXT)
    other = parse_symbol_directory(OTHER_TEXT)
    return candidate_table(nasdaq, other, _screener(), **kwargs)


def test_parse_drops_the_trailer_line_and_keeps_strings():
    nasdaq = parse_symbol_directory(NASDAQ_TEXT)
    assert len(nasdaq) == 10
    assert not nasdaq["Symbol"].str.startswith("File Creation").any()
    assert pd.api.types.is_string_dtype(nasdaq["Round Lot Size"])


def test_screener_symbols_use_yahoo_form_and_drop_preferreds():
    screener = _screener()
    assert "BRK-B" in set(screener["ticker"])
    assert not screener["ticker"].str.contains(r"\^").any()
    assert screener.loc[screener["ticker"] == "GIW", "market_cap"].iloc[0] == 0.0


def test_common_stocks_survive_with_their_facts():
    table = _table()
    kept = table.candidates.set_index("ticker")
    for ticker in ("AAPL", "A", "BRK-B", "CBOE", "NTRS", "AKR"):
        assert ticker in kept.index, ticker
    assert kept.loc["AAPL", "sector"] == "Technology"
    assert kept.loc["AAPL", "exchange"] == "NASDAQ"
    assert kept.loc["A", "exchange"] == "NYSE"
    assert kept.loc["CBOE", "exchange"] == "Cboe BZX"
    assert kept.loc["BRK-B", "sector"] == "Unknown"
    assert kept.loc["BRK-B", "market_cap"] == 1e12
    assert kept.loc["AAPL", "name"] == "Apple Inc."


def test_each_rule_removes_its_example():
    table = _table()
    reasons = table.excluded.set_index("ticker")["reason"]
    expected = {
        "QQQ": REASON_ETF,
        "SPY": REASON_ETF,
        "ZZZT": REASON_TEST,
        "BADD": REASON_STATUS,
        "ABCDW": REASON_SYMBOL,
        "AAC-U": REASON_SYMBOL,
        "SPAC": REASON_NAME,
        "HBANP": REASON_NAME,
        "ARCC": REASON_NAME,
        "GJT": REASON_NAME,
        "GIW": REASON_INDUSTRY,
        "BDJ": REASON_FUND,
        "HQH": REASON_FUND,
        "NOCAP": REASON_NO_CAP,
    }
    for ticker, reason in expected.items():
        assert reasons.get(ticker) == reason, (ticker, reasons.get(ticker))
    assert "TINY" in set(table.candidates["ticker"])


def test_market_cap_floor_applies_only_to_known_caps():
    table = _table(min_market_cap=1e8)
    reasons = table.excluded.set_index("ticker")["reason"]
    assert reasons["TINY"] == REASON_CAP
    assert "BRK-B" in set(table.candidates["ticker"])


def test_counts_add_up():
    table = _table(min_market_cap=1e8)
    counts = table.counts
    assert counts["listed"] == len(table.candidates) + len(table.excluded)
    assert counts["candidates"] == len(table.candidates)
    assert sum(value for key, value in counts.items() if key not in {"listed", "candidates"}) == len(table.excluded)


def test_symbol_and_name_helpers():
    assert yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo_symbol("AAPL") == "AAPL"
    assert display_name("Agilent Technologies, Inc. Common Stock") == "Agilent Technologies, Inc."
    assert display_name("Apple Inc. - Common Stock") == "Apple Inc."
    assert display_name("Berkshire Hathaway Inc. Class B Common Stock") == "Berkshire Hathaway Inc."
    assert display_name("Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares") == "Taiwan Semiconductor Manufacturing Company Ltd."
