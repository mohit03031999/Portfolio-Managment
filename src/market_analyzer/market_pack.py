"""Build and load the market data pack the web page runs on.

The pack is a folder with four files:

``returns.parquet``
    Daily simple returns, one column per ticker, one row per NYSE session,
    stored as float32 with missing days left missing.
``meta.parquet``
    One row per ticker: name, exchange, sector, industry, market cap and the
    history checks below, so the page can label a stock and say why it may
    not be optimised.
``manifest.json``
    When and how the pack was built, the rules used, and the counts.
``report.txt``
    The build report in plain words.

A ticker enters the pack when its price history passes the
:class:`QualityRules`: it still trades, its price and dollar volume are not
tiny, its recent history has few gaps, and no single day moves by more than
the limit (250% by default): such a day is either a data error or a one-off
event that would dominate any covariance estimate. A ticker with under three years of history
stays in the pack but is marked ``young``: the risk model needs a long
window, so the page lists it but cannot optimise it.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PACK_VERSION = 1
RETURNS_FILE = "returns.parquet"
META_FILE = "meta.parquet"
MANIFEST_FILE = "manifest.json"
REPORT_FILE = "report.txt"
RAW_CLOSE_FILE = "adj_close.parquet"
RAW_VOLUME_FILE = "volume.parquet"
PACK_FILES = (RETURNS_FILE, META_FILE, MANIFEST_FILE, REPORT_FILE)

REASON_NO_PRICES = "no prices"
REASON_STALE = "stopped trading"
REASON_PRICE = "price below the floor"
REASON_ILLIQUID = "too thinly traded"
REASON_GAPS = "too many missing days"
REASON_ANOMALY = "a single-day move beyond the limit"
NOTE_YOUNG = "under three years of history"

META_CHECK_COLUMNS = [
    "first_date",
    "last_date",
    "sessions",
    "history_years",
    "missing_recent",
    "median_dollar_volume",
    "last_price",
    "max_abs_return",
    "reason",
    "eligible",
    "young",
]


@dataclass(frozen=True)
class QualityRules:
    """What a price history must look like to enter the pack."""

    min_history_years: float = 3.0
    max_missing_fraction: float = 0.05
    min_median_dollar_volume: float = 1e6
    min_last_price: float = 1.0
    max_abs_return: float = 2.5
    recent_sessions: int = 504
    liquidity_sessions: int = 126
    stale_sessions: int = 10

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class MarketPack:
    """The loaded pack: returns, per-ticker facts, and how it was built."""

    returns: pd.DataFrame
    meta: pd.DataFrame
    manifest: dict[str, object]

    @property
    def tickers(self) -> list[str]:
        return [str(ticker) for ticker in self.returns.columns]

    @property
    def eligible(self) -> list[str]:
        flags = self.meta["eligible"].astype(bool)
        return [str(ticker) for ticker in self.meta.index[flags.to_numpy()]]

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns.index[-1])


# --------------------------------------------------------------------------- download


def download_history(
    tickers: list[str],
    start: str,
    end: str | None = None,
    batch_size: int = 100,
    retries: int = 3,
    pause: float = 1.0,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Adjusted closes and volumes for ``tickers``, downloaded in batches.

    Yahoo treats ``end`` as exclusive; left unset it is today, so the last
    row is the last completed session. Symbols with no prices are listed in
    the third value rather than raising, since a few thousand requests always
    include a handful of dead ones.
    """
    import yfinance as yf

    say = log or (lambda message: None)
    end = end or dt.date.today().isoformat()
    closes: list[pd.DataFrame] = []
    volumes: list[pd.DataFrame] = []
    failed: list[str] = []
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    for number, batch in enumerate(batches, start=1):
        raw = None
        for attempt in range(1, retries + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw = yf.download(
                        batch,
                        start=start,
                        end=end,
                        auto_adjust=False,
                        actions=False,
                        progress=False,
                        group_by="column",
                        threads=True,
                        interval="1d",
                    )
                break
            except Exception as exc:  # network hiccups and rate limits
                say(f"batch {number}/{len(batches)} attempt {attempt} failed: {exc}")
                time.sleep(pause * 5 * attempt)
        if raw is None or raw.empty:
            failed.extend(batch)
            say(f"batch {number}/{len(batches)}: no data")
            continue
        close = _field_frame(raw, "Adj Close", batch)
        volume = _field_frame(raw, "Volume", batch)
        got = [ticker for ticker in batch if ticker in close.columns and close[ticker].notna().any()]
        failed.extend(ticker for ticker in batch if ticker not in got)
        closes.append(close.loc[:, got])
        volumes.append(volume.reindex(columns=got))
        say(f"batch {number}/{len(batches)}: {len(got)}/{len(batch)} tickers")
        time.sleep(pause)
    if not closes:
        raise RuntimeError("No ticker returned any prices.")
    adj_close = _tidy(pd.concat(closes, axis=1))
    volume = _tidy(pd.concat(volumes, axis=1)).reindex(index=adj_close.index, columns=adj_close.columns)
    return adj_close, volume, failed


def _field_frame(raw: pd.DataFrame, field: str, batch: list[str]) -> pd.DataFrame:
    """One field (``Adj Close``, ``Volume``) as a date × ticker frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = {str(name) for name in raw.columns.get_level_values(0)}
        if field in level0:
            frame = raw[field].copy()
        else:
            level1 = raw.columns.get_level_values(1)
            frame = raw.loc[:, [str(name) == field for name in level1]].copy()
            frame.columns = [str(name) for name in frame.columns.get_level_values(0)]
    else:
        if field not in raw.columns:
            raise RuntimeError(f"Downloaded data has no {field} column.")
        frame = raw[[field]].copy()
        frame.columns = [batch[0]]
    frame.columns = [str(column) for column in frame.columns]
    return frame


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.sort_index()
    frame.index.name = "Date"
    frame = frame.loc[:, sorted(frame.columns)]
    return frame.astype("float64")


# --------------------------------------------------------------------------- checks


def assess_history(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    rules: QualityRules | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run the :class:`QualityRules` on every ticker; one row per ticker.

    ``reason`` is empty for a ticker that passes. ``eligible`` means it
    passes and has enough history for the risk model; ``young`` means it
    passes but does not.
    """
    rules = rules or QualityRules()
    adj_close = adj_close.sort_index()
    volume = volume.reindex(index=adj_close.index, columns=adj_close.columns)
    index = adj_close.index
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(index[-1])
    valid = adj_close.notna()
    sessions = valid.sum()
    first_date = adj_close.apply(lambda series: series.first_valid_index())
    last_date = adj_close.apply(lambda series: series.last_valid_index())
    position = pd.Series(np.arange(len(index)), index=index)
    last_position = last_date.map(lambda stamp: position.get(stamp, -1) if stamp is not None else -1)
    sessions_since_last = len(index) - 1 - last_position
    last_price = adj_close.ffill().iloc[-1]
    dollar = (adj_close * volume).iloc[-rules.liquidity_sessions :]
    median_dollar_volume = dollar.median(axis=0)

    recent = adj_close.iloc[-rules.recent_sessions :]
    starts = first_date.map(lambda stamp: stamp.value if stamp is not None else np.iinfo(np.int64).max)
    after_start = recent.index.values.astype("datetime64[ns]").astype(np.int64)[:, None] >= starts.to_numpy()[None, :]
    missing_cells = recent.isna().to_numpy() & after_start
    counted = after_start.sum(axis=0)
    missing_recent = pd.Series(
        np.where(counted > 0, missing_cells.sum(axis=0) / np.maximum(counted, 1), 0.0), index=adj_close.columns
    )

    returns = adj_close / adj_close.shift(1) - 1.0
    max_abs_return = returns.abs().max(axis=0)
    history_years = pd.Series(
        [((as_of - stamp).days / 365.25) if stamp is not None else 0.0 for stamp in first_date],
        index=adj_close.columns,
    )

    reason = np.select(
        [
            sessions.to_numpy() == 0,
            sessions_since_last.to_numpy() > rules.stale_sessions,
            last_price.to_numpy() < rules.min_last_price,
            ~(median_dollar_volume.to_numpy() >= rules.min_median_dollar_volume),
            missing_recent.to_numpy() > rules.max_missing_fraction,
            max_abs_return.to_numpy() > rules.max_abs_return,
        ],
        [REASON_NO_PRICES, REASON_STALE, REASON_PRICE, REASON_ILLIQUID, REASON_GAPS, REASON_ANOMALY],
        default="",
    )
    passes = reason == ""
    enough = history_years.to_numpy() >= rules.min_history_years
    table = pd.DataFrame(
        {
            "first_date": first_date,
            "last_date": last_date,
            "sessions": sessions.astype(int),
            "history_years": history_years.round(2),
            "missing_recent": missing_recent.round(4),
            "median_dollar_volume": median_dollar_volume,
            "last_price": last_price,
            "max_abs_return": max_abs_return,
            "reason": reason,
            "eligible": passes & enough,
            "young": passes & ~enough,
        },
        index=adj_close.columns,
    )
    table.index.name = "ticker"
    return table


def daily_returns_panel(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns as float32, missing where either close is missing."""
    adj_close = adj_close.sort_index()
    returns = (adj_close / adj_close.shift(1) - 1.0).iloc[1:]
    returns = returns.loc[:, sorted(returns.columns)].astype("float32")
    returns.index.name = "Date"
    return returns


# --------------------------------------------------------------------------- files


def build_manifest(
    start: str,
    returns: pd.DataFrame,
    meta: pd.DataFrame,
    checks: pd.DataFrame,
    rules: QualityRules,
    universe_counts: dict[str, int],
    min_market_cap: float,
    sources: dict[str, str],
    failed: list[str],
) -> dict[str, object]:
    """``meta`` holds the tickers in the pack; ``checks`` every ticker that was checked."""
    counts = checks["reason"].value_counts().to_dict()
    return {
        "pack_version": PACK_VERSION,
        "built_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "start": start,
        "first_date": str(pd.Timestamp(returns.index[0]).date()),
        "last_date": str(pd.Timestamp(returns.index[-1]).date()),
        "n_sessions": int(len(returns)),
        "n_tickers": int(returns.shape[1]),
        "n_eligible": int(meta["eligible"].sum()),
        "n_young": int(meta["young"].sum()),
        "min_market_cap": float(min_market_cap),
        "rules": rules.to_dict(),
        "universe_counts": {str(key): int(value) for key, value in universe_counts.items()},
        "history_counts": {str(key): int(value) for key, value in counts.items() if key},
        "download_failed": int(len(failed)),
        "sources": sources,
    }


def write_pack(
    folder: str | Path,
    returns: pd.DataFrame,
    meta: pd.DataFrame,
    manifest: dict[str, object],
    report: str | None = None,
) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    returns.astype("float32").to_parquet(folder / RETURNS_FILE, compression="zstd")
    meta.to_parquet(folder / META_FILE)
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if report is not None:
        (folder / REPORT_FILE).write_text(report)
    return folder


def load_pack(folder: str | Path) -> MarketPack:
    folder = Path(folder)
    returns = pd.read_parquet(folder / RETURNS_FILE)
    returns.index = pd.to_datetime(returns.index)
    meta = pd.read_parquet(folder / META_FILE)
    manifest = json.loads((folder / MANIFEST_FILE).read_text())
    return MarketPack(returns=returns, meta=meta, manifest=manifest)


def fetch_pack(url: str, folder: str | Path, timeout: float = 60.0) -> Path:
    """Download the pack files published at ``url`` into ``folder`` unless it already holds that build.

    ``url`` is the folder-like address the four files sit under, such as a
    GitHub release's download path. The manifest is read first; when its
    ``built_at`` matches the cached one and every file is present, nothing
    is downloaded. Files land under a temporary name and are renamed last,
    so a failed download never leaves a half-written pack.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = url.rstrip("/")
    with urllib.request.urlopen(f"{base}/{MANIFEST_FILE}", timeout=timeout) as response:
        remote = json.loads(response.read().decode("utf-8"))
    cached = folder / MANIFEST_FILE
    if cached.exists() and all((folder / name).exists() for name in PACK_FILES):
        try:
            if json.loads(cached.read_text()).get("built_at") == remote.get("built_at"):
                return folder
        except (OSError, ValueError):
            pass
    for name in PACK_FILES:
        if name == MANIFEST_FILE:
            continue
        target = folder / name
        partial = folder / (name + ".part")
        with urllib.request.urlopen(f"{base}/{name}", timeout=timeout) as response, open(partial, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        partial.replace(target)
    cached.write_text(json.dumps(remote, indent=2) + "\n")
    return folder


def format_report(manifest: dict[str, object], meta: pd.DataFrame) -> str:
    """The build in plain words: what was listed, what each rule removed, what is left."""
    lines = [
        f"Market pack built {manifest['built_at']}",
        f"Sessions {manifest['first_date']} to {manifest['last_date']} ({manifest['n_sessions']} days)",
        "",
        "Listing rules (NASDAQ Trader symbol files + NASDAQ screener):",
    ]
    universe = dict(manifest.get("universe_counts", {}))
    listed = universe.pop("listed", 0)
    candidates = universe.pop("candidates", 0)
    lines.append(f"  {'listed securities':<57s} {listed:>6}")
    for key, value in sorted(universe.items(), key=lambda item: -item[1]):
        if key:
            lines.append(f"  removed: {key:<48s} {value:>6}")
    lines.append(f"  {'candidates':<57s} {candidates:>6}")
    lines.append(f"  {'no prices from Yahoo':<57s} {manifest.get('download_failed', 0):>6}")
    lines.append("")
    lines.append("History rules (Yahoo Finance daily prices):")
    for key, value in sorted(dict(manifest.get("history_counts", {})).items(), key=lambda item: -item[1]):
        if key:
            lines.append(f"  removed: {key:<48s} {value:>6}")
    lines.append(f"  {'in the pack':<57s} {manifest['n_tickers']:>6}")
    lines.append(f"    {'eligible for the risk model':<55s} {manifest['n_eligible']:>6}")
    lines.append(f"    {'young (' + NOTE_YOUNG + ')':<55s} {manifest['n_young']:>6}")
    lines.append("")
    kept = meta[meta["reason"] == ""]
    if "exchange" in kept.columns:
        lines.append("By exchange: " + ", ".join(f"{k} {v}" for k, v in kept["exchange"].value_counts().items()))
    if "sector" in kept.columns:
        lines.append("By sector:")
        for sector, count in kept["sector"].value_counts().items():
            lines.append(f"  {sector:<28s} {count:>6}")
    return "\n".join(lines) + "\n"
