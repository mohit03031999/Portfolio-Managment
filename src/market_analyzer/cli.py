"""Command line entry point for user-defined market pulls."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from market_analyzer.analyzer import summarize
from market_analyzer.backtest import DEFAULT_STRATEGIES, BacktestConfig, run_backtest, strategy_names
from market_analyzer.matrices import CorrelationModel
from market_analyzer.pipeline import build_market_data, save_market_data
from market_analyzer.returns import ALIGN_INTERSECTION, ALIGN_UNION


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "matrices":
            return _matrices(args)
        if args.command == "backtest":
            return _backtest(args)
        return _pull(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-analyzer",
        description="Download adjusted closes and daily returns for any tickers you choose.",
        epilog="How to find a symbol for any company: docs/getting-stock-data.md",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pull = commands.add_parser(
        "pull",
        help="Download adjusted closes and build the daily return matrix.",
    )
    pull.add_argument("--universe", type=Path, help="YAML file with your tickers or a table source.")
    pull.add_argument(
        "--tickers",
        nargs="+",
        help="Yahoo Finance symbols, for example AAPL RELIANCE.NS HSBA.L",
    )
    pull.add_argument("--start", help="Inclusive start date, YYYY-MM-DD. Default: 2020-01-01.")
    pull.add_argument(
        "--end",
        help="Exclusive end date, YYYY-MM-DD. Omit it to pull through the latest session.",
    )
    pull.add_argument("--name", help="Folder name under the output directory.")
    pull.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Directory that will contain <name>/prices.csv and returns.csv. Default: data",
    )
    pull.add_argument(
        "--align",
        choices=(ALIGN_INTERSECTION, ALIGN_UNION),
        default=ALIGN_INTERSECTION,
        help="intersection keeps dates where every ticker traded. union keeps partial days.",
    )
    pull.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Trading days used to annualize volatility and return. Default: 252.",
    )
    pull.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any symbol has no usable price history.",
    )
    matrices = commands.add_parser(
        "matrices",
        help="Build correlation, covariance, and the denoised correlation from returns.csv.",
    )
    matrices.add_argument("returns", type=Path, help="Date × ticker return CSV from pull.")
    matrices.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: the directory that contains the CSV.",
    )
    matrices.add_argument(
        "--sigma2",
        default="fitted",
        help="Noise variance for the Marchenko-Pastur edge: fitted, market, or a number such as 1.0. Default: fitted.",
    )
    backtest = commands.add_parser(
        "backtest",
        help="Roll minimum-variance portfolios forward and compare raw against cleaned covariance.",
    )
    backtest.add_argument("returns", type=Path, help="Date × ticker return CSV. Missing cells are allowed.")
    backtest.add_argument(
        "--from-prices",
        action="store_true",
        help="Treat the CSV as adjusted closes and compute daily simple returns first.",
    )
    backtest.add_argument("--window", type=int, default=500, help="Estimation window in sessions. Default: 500.")
    backtest.add_argument("--holding", type=int, default=21, help="Holding period in sessions. Default: 21.")
    backtest.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        help="Strategies to run. Choices: " + ", ".join(strategy_names()) + ", or gross<cap>_<estimator> for a gross-exposure cap, e.g. gross2_rie.",
    )
    backtest.add_argument(
        "--cost-bps",
        type=float,
        default=0.0,
        help="Cost in basis points per unit of turnover, charged on the first day of each hold. Default: 0.",
    )
    backtest.add_argument(
        "--min-assets",
        type=int,
        default=2,
        help="Skip a decision date with fewer complete tickers than this. Default: 2.",
    )
    backtest.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Sessions used to annualize volatility and return. Default: 252.",
    )
    backtest.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: a backtest folder next to the CSV.",
    )
    return parser


def _pull(args: argparse.Namespace) -> int:
    if (args.universe is None) == (args.tickers is None):
        raise ValueError("Pass exactly one of --universe or --tickers.")

    if args.periods_per_year <= 0:
        raise ValueError("--periods-per-year must be positive.")

    data = build_market_data(
        tickers=args.tickers,
        universe_path=args.universe,
        start=args.start,
        end=args.end,
        name=args.name,
        align=args.align,
    )
    if args.strict and data.missing_tickers:
        raise RuntimeError("No usable price history for: " + ", ".join(data.missing_tickers))

    folder = args.out / data.name
    paths = save_market_data(data, folder, args.periods_per_year)
    _print_report(data, paths, args.periods_per_year)
    return 0


def _matrices(args: argparse.Namespace) -> int:
    if not args.returns.is_file():
        raise ValueError(f"Return matrix not found: {args.returns}")
    returns = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    model = CorrelationModel.from_returns(returns, sigma2=args.sigma2)
    folder = args.out or args.returns.parent
    paths = model.save(folder)
    print(model.note())
    print()
    preview = model.eigenvalues.head(10)
    print(preview.to_string(float_format=lambda value: f"{value:.4f}"))
    hidden = len(model.eigenvalues) - len(preview)
    if hidden:
        print(f"... {hidden} more modes in {paths['eigenvalues']}")
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


def _backtest(args: argparse.Namespace) -> int:
    if not args.returns.is_file():
        raise ValueError(f"Return matrix not found: {args.returns}")
    panel = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    if args.from_prices:
        panel = panel.sort_index().pct_change(fill_method=None).iloc[1:]
    config = BacktestConfig(
        window=args.window,
        holding=args.holding,
        strategies=tuple(args.strategies),
        periods_per_year=args.periods_per_year,
        min_assets=args.min_assets,
        cost_bps=args.cost_bps,
    )
    result = run_backtest(panel, config)
    folder = args.out or args.returns.parent / "backtest"
    paths = result.save(folder)
    print(result.report())
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


def _print_report(data, paths: dict[str, Path], periods_per_year: int) -> None:
    rows, columns = data.shape
    end = data.end or "latest"
    print(f"Basket: {data.name}")
    print(f"Window: {data.start} → {end} (end is exclusive when you set it)")
    print(f"Requested: {len(data.requested_tickers)}   with prices: {columns}")
    ticker_word = "ticker" if columns == 1 else "tickers"
    print(f"Return matrix: {rows} dates × {columns} {ticker_word}")
    if len(data.returns):
        print(f"First return date: {data.returns.index[0].date()}")
        print(f"Last return date:  {data.returns.index[-1].date()}")
    for note in data.notes:
        print(f"note: {note}")
    print()
    print(
        summarize(data.returns, periods_per_year).to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
