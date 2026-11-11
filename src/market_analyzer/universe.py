"""Resolve a user-defined ticker list from a YAML file or an HTML table."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import urllib.request

import pandas as pd
import yaml

_USER_AGENT = "market-analyzer/0.1 (research; ticker-list fetch)"


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    tickers: list[str]
    start: str | None = None
    end: str | None = None


def clean_tickers(symbols: list[str] | tuple[str, ...]) -> list[str]:
    """Drop blanks and duplicates, and keep the caller's order and spelling."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        if any(character.isspace() for character in symbol):
            raise ValueError(f"Ticker {symbol!r} contains whitespace.")
        if symbol not in seen:
            seen.add(symbol)
            cleaned.append(symbol)
    if not cleaned:
        raise ValueError("No tickers provided.")
    return cleaned


def wikipedia_yahoo_symbol(symbol: str) -> str:
    """Map a Wikipedia class share such as ``BRK.B`` to Yahoo's ``BRK-B``.

    Do not use this on symbols you already copied from Yahoo Finance.
    ``RELIANCE.NS`` is already a Yahoo symbol; replacing the dot would break it.
    """
    return symbol.strip().replace(".", "-")


def tickers_from_html_table(
    source: str,
    column: str,
    table_index: int = 0,
    wikipedia_class_shares: bool = False,
) -> list[str]:
    """Read one column from an HTML table on disk or on the web."""
    frame = _read_html_table(source, table_index)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " ".join(str(part) for part in col if str(part) != "nan").strip()
            for col in frame.columns
        ]
    lookup = {str(name).strip().lower(): name for name in frame.columns}
    key = column.strip().lower()
    if key not in lookup:
        available = ", ".join(str(name) for name in frame.columns)
        raise ValueError(f"Column {column!r} was not found. Available columns: {available}")

    series = frame[lookup[key]].dropna().astype(str)
    if wikipedia_class_shares:
        series = series.map(wikipedia_yahoo_symbol)
    return clean_tickers(series.tolist())


def load_universe(path: str | Path) -> UniverseSpec:
    """Load tickers, and optional dates, from a YAML universe file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Universe file not found: {file_path}")

    with file_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{file_path} must contain a YAML mapping.")

    has_tickers = "tickers" in raw and raw["tickers"] is not None
    has_source = "source" in raw and raw["source"] is not None
    if has_tickers == has_source:
        raise ValueError(
            f"{file_path} must set exactly one of 'tickers' or 'source'."
        )

    if has_tickers:
        tickers = clean_tickers(_string_tickers(raw["tickers"]))
    else:
        tickers = _tickers_from_source(raw["source"])

    name = str(raw.get("name") or file_path.stem).strip()
    if not name:
        raise ValueError("Universe name cannot be empty.")

    return UniverseSpec(
        name=name,
        tickers=tickers,
        start=_optional_text(raw.get("start")),
        end=_optional_text(raw.get("end")),
    )


def _tickers_from_source(source: object) -> list[str]:
    if not isinstance(source, dict):
        raise ValueError("'source' must be a mapping.")
    source_type = str(source.get("type", "")).strip()
    if source_type != "html_table":
        raise ValueError("The only source type supported today is 'html_table'.")
    url = source.get("url")
    column = source.get("column")
    if not url or not column:
        raise ValueError("An html_table source needs 'url' and 'column'.")
    table_index = source.get("table_index", 0)
    if not isinstance(table_index, int):
        raise ValueError("'table_index' must be an integer.")
    return tickers_from_html_table(
        source=str(url),
        column=str(column),
        table_index=table_index,
        wikipedia_class_shares=bool(source.get("wikipedia_class_shares", False)),
    )


def _string_tickers(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("'tickers' must be a list of Yahoo Finance symbols.")
    symbols: list[str] = []
    for item in value:
        # YAML 1.1 reads the unquoted words ON, NO, YES, and OFF as booleans.
        # ON is a real US ticker, so a bare ON would silently become True.
        if isinstance(item, bool) or not isinstance(item, str):
            raise ValueError(
                f"Ticker {item!r} is not a string. Quote every symbol in the YAML file, "
                'for example "ON" or "7203.T".'
            )
        symbols.append(item)
    return symbols


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_html_table(source: str, table_index: int) -> pd.DataFrame:
    path = Path(source)
    if path.is_file():
        tables = pd.read_html(path)
    else:
        request = urllib.request.Request(source, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read()
        except Exception as exc:
            raise ValueError(f"Could not read the ticker table at {source}.") from exc
        tables = pd.read_html(io.BytesIO(html))

    if table_index < 0 or table_index >= len(tables):
        raise ValueError(
            f"table_index {table_index} is out of range. The page has {len(tables)} tables."
        )
    return tables[table_index]
