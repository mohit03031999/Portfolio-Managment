"""Build the market data pack the web page runs on.

Steps: fetch the NASDAQ Trader symbol files and the NASDAQ screener, keep
US common stocks of operating companies, download their daily prices from
Yahoo Finance in batches, run the history checks, and write
``data/pack/`` (returns, meta, manifest, report). Raw downloads are kept in
``data/raw/`` so the checks can be re-run without downloading again.

    .venv/bin/python examples/build_market_pack.py                 # full build
    .venv/bin/python examples/build_market_pack.py --limit 50      # quick trial
    .venv/bin/python examples/build_market_pack.py --skip-download # re-run the checks only
"""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from market_analyzer.market_pack import (  # noqa: E402
    RAW_CLOSE_FILE,
    RAW_VOLUME_FILE,
    QualityRules,
    assess_history,
    build_manifest,
    daily_returns_panel,
    download_history,
    format_report,
    write_pack,
)
from market_analyzer.market_universe import (  # noqa: E402
    META_COLUMNS,
    SCREENER_URL,
    SYMBOL_DIRECTORY_URL,
    candidate_table,
    fetch_screener,
    fetch_symbol_directory,
)


def say(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2018-01-01", help="first price date (default 2018-01-01)")
    parser.add_argument("--end", default=None, help="exclusive end date; default today, so the pack ends at the last completed session")
    parser.add_argument("--out", default=str(ROOT / "data" / "pack"), help="pack folder")
    parser.add_argument("--raw", default=str(ROOT / "data" / "raw"), help="folder for raw downloads")
    parser.add_argument("--min-market-cap", type=float, default=1e8, help="market cap floor in dollars (default 100 million)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between batches")
    parser.add_argument("--limit", type=int, default=0, help="only the first N candidates (a trial run)")
    parser.add_argument("--skip-universe", action="store_true", help="reuse raw/candidates.csv")
    parser.add_argument("--skip-download", action="store_true", help="reuse raw/adj_close.parquet and raw/volume.parquet")
    return parser.parse_args()


def build_universe(raw: Path, min_market_cap: float, reuse: bool) -> tuple[pd.DataFrame, dict[str, int]]:
    candidates_file = raw / "candidates.csv"
    counts_file = raw / "universe_counts.json"
    if reuse and candidates_file.exists() and counts_file.exists():
        say(f"reusing {candidates_file}")
        return pd.read_csv(candidates_file), json.loads(counts_file.read_text())
    say("fetching the NASDAQ Trader symbol files")
    nasdaq, other = fetch_symbol_directory()
    say(f"  {len(nasdaq)} NASDAQ rows, {len(other)} other-exchange rows")
    say("fetching the NASDAQ screener")
    screener = fetch_screener()
    say(f"  {len(screener)} screener rows")
    table = candidate_table(nasdaq, other, screener, min_market_cap=min_market_cap)
    raw.mkdir(parents=True, exist_ok=True)
    nasdaq.to_csv(raw / "nasdaqlisted.csv", index=False)
    other.to_csv(raw / "otherlisted.csv", index=False)
    screener.to_csv(raw / "screener.csv", index=False)
    table.candidates.to_csv(candidates_file, index=False)
    table.excluded.to_csv(raw / "excluded.csv", index=False)
    counts_file.write_text(json.dumps(table.counts, indent=2) + "\n")
    say(f"  {table.counts['candidates']} candidates of {table.counts['listed']} listed")
    return table.candidates, table.counts


def download(raw: Path, tickers: list[str], start: str, end: str | None, batch_size: int, pause: float, reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    close_file, volume_file, failed_file = raw / RAW_CLOSE_FILE, raw / RAW_VOLUME_FILE, raw / "download_failed.txt"
    if reuse and close_file.exists() and volume_file.exists():
        say(f"reusing {close_file}")
        failed = failed_file.read_text().split() if failed_file.exists() else []
        return pd.read_parquet(close_file), pd.read_parquet(volume_file), failed
    say(f"downloading {len(tickers)} tickers from {start} in batches of {batch_size}")
    started = time.time()
    adj_close, volume, failed = download_history(tickers, start, end, batch_size=batch_size, pause=pause, log=say)
    say(f"  done in {(time.time() - started) / 60:.1f} min; {adj_close.shape[1]} tickers, {len(failed)} without prices")
    raw.mkdir(parents=True, exist_ok=True)
    adj_close.to_parquet(close_file)
    volume.to_parquet(volume_file)
    failed_file.write_text("\n".join(failed) + ("\n" if failed else ""))
    return adj_close, volume, failed


def main() -> None:
    args = parse_args()
    raw, out = Path(args.raw), Path(args.out)
    rules = QualityRules()

    candidates, universe_counts = build_universe(raw, args.min_market_cap, args.skip_universe)
    tickers = [str(ticker) for ticker in candidates["ticker"]]
    if args.limit:
        tickers = tickers[: args.limit]
        say(f"trial run: first {len(tickers)} candidates")

    adj_close, volume, failed = download(raw, tickers, args.start, args.end, args.batch_size, args.pause, args.skip_download)
    adj_close = adj_close.loc[:, [t for t in adj_close.columns if t in set(tickers)]]
    volume = volume.reindex(columns=adj_close.columns)

    say("running the history checks")
    checks = assess_history(adj_close, volume, rules)
    kept = checks.index[checks["reason"] == ""]
    say(f"  {len(kept)} tickers pass; {int(checks['eligible'].sum())} eligible, {int(checks['young'].sum())} young")

    returns = daily_returns_panel(adj_close.loc[:, kept])
    facts = candidates.set_index("ticker").loc[:, [c for c in META_COLUMNS if c != "ticker"]]
    meta = facts.join(checks, how="right").loc[kept]
    meta.index.name = "ticker"
    meta = meta.drop(columns=["directory_name"], errors="ignore")

    sources = {"symbol_directory": SYMBOL_DIRECTORY_URL, "screener": SCREENER_URL, "prices": "Yahoo Finance via yfinance"}
    manifest = build_manifest(args.start, returns, meta, checks, rules, universe_counts, args.min_market_cap, sources, failed)
    report = format_report(manifest, checks.join(facts, how="left"))
    write_pack(out, returns, meta, manifest, report)
    say(f"pack written to {out}")
    print()
    print(report)


if __name__ == "__main__":
    main()
