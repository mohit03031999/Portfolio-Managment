"""The list of US common stocks the web page can offer.

Two public files from NASDAQ Trader list every security on a US exchange:
``nasdaqlisted.txt`` for NASDAQ and ``otherlisted.txt`` for the other
exchanges. They carry no sector or size, so the NASDAQ stock screener,
one JSON download, supplies sector, industry, market capitalisation,
country and IPO year.

Only operating companies belong in a risk tool, so the candidate table
drops exchange-traded funds and notes, closed-end funds, warrants, units,
rights, preferred shares, blank-cheque companies and test issues. Each
excluded row keeps the rule that removed it, so the build report can say
how many names each rule cost. Symbols are written the way Yahoo Finance
writes them: class shares as ``BRK-B`` rather than ``BRK.B``.

The network functions are thin; everything that decides is a pure
function of data frames, so it can be tested without a connection.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd

SYMBOL_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/{name}"
SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
USER_AGENT = "market-analyzer/0.1 (research; symbol directory)"
SCREENER_HEADERS = {
    # The screener answers browsers only, so the request looks like one.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

OTHER_EXCHANGES = {"N": "NYSE", "A": "NYSE American", "P": "NYSE Arca", "Z": "Cboe BZX"}
NASDAQ_TIERS = {"Q": "NASDAQ Global Select", "G": "NASDAQ Global", "S": "NASDAQ Capital"}
NASDAQ_EXCHANGE = "NASDAQ"

# Security names that are not common stock of an operating company.
EXCLUDED_NAME = re.compile(
    r"\b(warrants?|units?|rights?|preferred|preference|notes?|debentures?|subordinated"
    r"|acquisition|closed[- ]end|income fund|funds?|etfs?|etns?|trust preferred"
    r"|certificates?|repackaged|synthetic fixed[- ]income|strats|capital securities|trust securities)\b|%",
    re.IGNORECASE,
)
# Screener industries that never hold an operating company.
EXCLUDED_INDUSTRIES = {"Blank Checks", "Trusts Except Educational Religious and Charitable"}
# Closed-end funds sit in the Finance sector under these industries, named
# like funds. Banks, insurers and asset managers never use these words.
FUND_SECTOR = "Finance"
FUND_INDUSTRIES = {
    "Finance Companies",
    "Investment Managers",
    "Finance/Investors Services",
    "Finance: Consumer Services",
    "Investment Bankers/Brokers/Service",
    "Unknown",
}
FUND_VOCABULARY = re.compile(
    r"\b(income|municipal\w*|bond|dividend|term trust|allocation|opportunit\w*|floating rate|senior loan"
    r"|senior floating|convertible|tax[- ]?(advantaged?|exempt|free|managed)|strategic|equity trust"
    r"|multi-?sector|total return|premier|premium|enhanced|high yield|credit (income|opportunit\w*|allocation"
    r"|strateg\w*|fund|trust)|health sciences|science (and|&) technology|energy and resources|resources trust"
    r"|utility|investors trust|securities corporation|diversified|core plus|core bond|target term"
    r"|global (equity|dividend|income|value|growth)|value trust|growth trust|capital trust|equity fund"
    r"|investors$|healthcare investors|participation investors|healthcare & wellness|gold trust|silver trust"
    r"|physical|(micro|small|mid)-cap trust|investment grade|tech100|all-star|tri-continental|source capital"
    r"|general american investors)\b",
    re.IGNORECASE,
)
# NYSE symbol suffixes for units, warrants, rights, preferreds and the like.
EXCLUDED_SUFFIXES = {"U", "W", "WS", "R", "P", "Z"}
VALID_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

REASON_ETF = "exchange-traded fund"
REASON_TEST = "test issue"
REASON_STATUS = "delinquent or deficient"
REASON_EXCHANGE = "exchange not NYSE, NYSE American or NASDAQ"
REASON_NAME = "not common stock (warrant, unit, preferred, note, fund, SPAC)"
REASON_SYMBOL = "symbol suffix or characters"
REASON_SECTOR = "no sector in the screener"
REASON_INDUSTRY = "blank-cheque company or trust"
REASON_FUND = "closed-end fund"
REASON_NO_CAP = "no market value in the screener"
REASON_CAP = "market cap below the floor"

META_COLUMNS = [
    "ticker",
    "name",
    "exchange",
    "sector",
    "industry",
    "market_cap",
    "country",
    "ipo_year",
    "directory_name",
]


@dataclass
class CandidateTable:
    """Survivors of the listing rules plus a count of what each rule removed."""

    candidates: pd.DataFrame
    excluded: pd.DataFrame
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return [str(ticker) for ticker in self.candidates["ticker"]]


# --------------------------------------------------------------------------- network


def fetch_symbol_directory(timeout: float = 60.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and parse ``nasdaqlisted.txt`` and ``otherlisted.txt``."""
    import requests

    frames = []
    for name in ("nasdaqlisted.txt", "otherlisted.txt"):
        response = requests.get(
            SYMBOL_DIRECTORY_URL.format(name=name), headers={"User-Agent": USER_AGENT}, timeout=timeout
        )
        response.raise_for_status()
        frames.append(parse_symbol_directory(response.text))
    return frames[0], frames[1]


def fetch_screener(timeout: float = 60.0) -> pd.DataFrame:
    """Download the NASDAQ stock screener as a frame with Yahoo-style symbols."""
    import requests

    response = requests.get(
        SCREENER_URL,
        params={"tableonly": "true", "limit": "25", "offset": "0", "download": "true"},
        headers=SCREENER_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("rows")
    if not rows:
        raise RuntimeError("The NASDAQ screener returned no rows.")
    return parse_screener(pd.DataFrame(rows))


# --------------------------------------------------------------------------- parsing


def parse_symbol_directory(text: str) -> pd.DataFrame:
    """Parse a pipe-separated NASDAQ Trader file, dropping the trailer line."""
    frame = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    first = frame.columns[0]
    frame = frame[~frame[first].str.startswith("File Creation Time")]
    return frame.reset_index(drop=True)


def parse_screener(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the screener columns the page needs, with numbers as numbers."""
    frame = pd.DataFrame(
        {
            "ticker": raw["symbol"].astype(str).str.strip().str.replace("/", "-", regex=False),
            "screener_name": raw["name"].astype(str).str.strip(),
            "sector": raw["sector"].astype(str).str.strip(),
            "industry": raw["industry"].astype(str).str.strip(),
            "market_cap": pd.to_numeric(raw["marketCap"], errors="coerce"),
            "country": raw["country"].astype(str).str.strip(),
            "ipo_year": pd.to_numeric(raw["ipoyear"], errors="coerce"),
        }
    )
    frame = frame[~frame["ticker"].str.contains(r"\^", regex=True)]
    return frame.drop_duplicates("ticker").reset_index(drop=True)


# --------------------------------------------------------------------------- rules


def listing_rows(nasdaq: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """One row per listed security with the columns the rules look at."""
    nasdaq_rows = pd.DataFrame(
        {
            "symbol": nasdaq["Symbol"].astype(str).str.strip(),
            "directory_name": nasdaq["Security Name"].astype(str).str.strip(),
            "exchange": NASDAQ_EXCHANGE,
            "tier": nasdaq["Market Category"].map(NASDAQ_TIERS).fillna(""),
            "etf": nasdaq["ETF"].astype(str).str.upper().eq("Y"),
            "test_issue": nasdaq["Test Issue"].astype(str).str.upper().eq("Y"),
            "financial_status": nasdaq["Financial Status"].astype(str).str.upper(),
            "exchange_code": "Q",
        }
    )
    other_rows = pd.DataFrame(
        {
            "symbol": other["ACT Symbol"].astype(str).str.strip(),
            "directory_name": other["Security Name"].astype(str).str.strip(),
            "exchange": other["Exchange"].map(OTHER_EXCHANGES).fillna(""),
            "tier": "",
            "etf": other["ETF"].astype(str).str.upper().eq("Y"),
            "test_issue": other["Test Issue"].astype(str).str.upper().eq("Y"),
            "financial_status": "N",
            "exchange_code": other["Exchange"].astype(str).str.upper(),
        }
    )
    return pd.concat([nasdaq_rows, other_rows], ignore_index=True)


def yahoo_symbol(symbol: str) -> str:
    """``BRK.B`` becomes ``BRK-B``; everything else is unchanged."""
    return symbol.replace(".", "-")


def exclusion_reason(row: pd.Series) -> str:
    """The first rule that removes a listing row, or an empty string."""
    if row["etf"]:
        return REASON_ETF
    if row["test_issue"]:
        return REASON_TEST
    if row["financial_status"] != "N":
        return REASON_STATUS
    if not row["exchange"]:
        return REASON_EXCHANGE
    symbol = str(row["symbol"])
    if not VALID_SYMBOL.match(symbol):
        return REASON_SYMBOL
    if "." in symbol and symbol.split(".", 1)[1] in EXCLUDED_SUFFIXES:
        return REASON_SYMBOL
    if row["exchange"] == NASDAQ_EXCHANGE and len(symbol) == 5 and symbol[-1] in {"W", "U", "R"}:
        return REASON_SYMBOL
    if EXCLUDED_NAME.search(str(row["directory_name"])):
        return REASON_NAME
    return ""


def candidate_table(
    nasdaq: pd.DataFrame,
    other: pd.DataFrame,
    screener: pd.DataFrame | None = None,
    min_market_cap: float = 0.0,
    require_sector: bool = False,
) -> CandidateTable:
    """Apply the listing rules and join the screener.

    ``min_market_cap`` (dollars) drops names below the floor. Names the
    screener does not know keep a missing market cap and stay in, so the
    liquidity check on real prices decides them later; a market cap of
    zero is a blank-cheque or structured product and goes. Setting
    ``require_sector`` also drops names without a screener sector.
    """
    rows = listing_rows(nasdaq, other)
    rows["reason"] = rows.apply(exclusion_reason, axis=1)
    rows["ticker"] = rows["symbol"].map(yahoo_symbol)
    rows["name"] = rows["directory_name"].map(display_name)

    if screener is not None:
        rows = rows.merge(screener, on="ticker", how="left")
    else:
        for column in ("screener_name", "sector", "industry", "market_cap", "country", "ipo_year"):
            rows[column] = pd.NA
    rows["sector"] = rows["sector"].fillna("").replace("", "Unknown")
    rows["industry"] = rows["industry"].fillna("").replace("", "Unknown")
    rows["country"] = rows["country"].fillna("").replace("", "Unknown")
    rows["market_cap"] = pd.to_numeric(rows["market_cap"], errors="coerce")
    rows["ipo_year"] = pd.to_numeric(rows["ipo_year"], errors="coerce")

    unresolved = rows["reason"] == ""
    industry_hit = unresolved & rows["industry"].isin(EXCLUDED_INDUSTRIES)
    rows.loc[industry_hit, "reason"] = REASON_INDUSTRY
    unresolved = rows["reason"] == ""
    fund_hit = (
        unresolved
        & rows["sector"].isin({FUND_SECTOR, "Unknown"})
        & rows["industry"].isin(FUND_INDUSTRIES)
        & rows["name"].map(lambda name: FUND_VOCABULARY.search(name) is not None)
    )
    rows.loc[fund_hit, "reason"] = REASON_FUND
    unresolved = rows["reason"] == ""
    if require_sector:
        missing = unresolved & rows["sector"].eq("Unknown")
        rows.loc[missing, "reason"] = REASON_SECTOR
        unresolved = rows["reason"] == ""
    zero = unresolved & rows["market_cap"].notna() & (rows["market_cap"] <= 0)
    rows.loc[zero, "reason"] = REASON_NO_CAP
    unresolved = rows["reason"] == ""
    if min_market_cap > 0:
        small = unresolved & rows["market_cap"].notna() & (rows["market_cap"] < min_market_cap)
        rows.loc[small, "reason"] = REASON_CAP

    rows = rows.drop_duplicates("ticker", keep="first")
    counts = rows["reason"].value_counts().to_dict()
    counts = {str(key): int(value) for key, value in counts.items() if key}
    counts["listed"] = int(len(rows))
    counts["candidates"] = int((rows["reason"] == "").sum())

    survivors = rows.loc[rows["reason"] == "", META_COLUMNS].sort_values("ticker").reset_index(drop=True)
    excluded = rows.loc[rows["reason"] != "", ["ticker", "directory_name", "exchange", "reason"]].reset_index(drop=True)
    return CandidateTable(candidates=survivors, excluded=excluded, counts=counts)


_NAME_TAILS = re.compile(
    r"\s*[-,]?\s*(class [a-z] )?(common stock|common shares|ordinary shares|american depositary shares"
    r"|american depositary receipts|depositary shares|shares of beneficial interest)\b.*$",
    re.IGNORECASE,
)


def display_name(directory_name: str) -> str:
    """Strip the share-class tail: ``Agilent Technologies, Inc. Common Stock`` → ``Agilent Technologies, Inc.``"""
    cleaned = _NAME_TAILS.sub("", str(directory_name)).strip().rstrip(",-").strip()
    return cleaned or str(directory_name)
