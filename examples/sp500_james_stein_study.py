"""Does James-Stein on the market eigenvector help the RIE, and does it help more on short windows?

``james_stein_rie_covariance`` cleans the correlation with the RIE, then
shrinks each stock's market loading toward the average loading by the
James-Stein factor ``c = 1 − s²/V̂`` (noise variance of one loading over
their measured spread). Theory says ``c`` is close to 1 at T = 1000 and
smaller on short windows, where the market direction is noisier.

Two experiments:

* ``--part exact``: worlds where the truth is known exactly (Gaussian days
  from the raw full-sample covariance; real days shuffled, truth = the
  covariance of the days outside the window). Windows of 250, 500 and 1000
  days, decision dates every 21 days. For each window: the RIE, the RIE with
  the data-chosen ``c``, fixed ``c`` values, and the RIE handed the true
  market direction (the ceiling for any market-direction fix). Also the
  overlap of the market direction with the truth before and after, and the
  ``c`` that would have been best. Realised risk is the exact ``wᵀΣw``.
* ``--part real``: the rolling S&P 500 backtest (21-day holds) with the RIE
  and the James-Stein RIE side by side at each window.

    .venv/bin/python examples/sp500_james_stein_study.py --part exact --panel gaussian_raw --seed 1
    .venv/bin/python examples/sp500_james_stein_study.py --part exact --panel shuffled --seed 1
    .venv/bin/python examples/sp500_james_stein_study.py --part real --window 250     # also 500, 1000
    .venv/bin/python examples/sp500_james_stein_study.py --report
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sp500_eigenvector_error as error  # noqa: E402
import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from market_analyzer.optimizer import james_stein_rie_covariance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sp500_james_stein"
WINDOWS = (25, 50, 100, 250, 500, 1000)
COMMON_START = "2018-12-24"  # first hold of the 1000-day window: every window is compared from here on
HOLDING = 21
ANNUAL = 252
FIXED_C = (0.5, 0.8, 0.9)
PANEL_LABELS = {"gaussian_raw": "Gaussian days", "shuffled": "real days, shuffled", "real": "real market"}
VARIANTS = ("rie", "js", "c=0.5", "c=0.8", "c=0.9", "true market direction", "truth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("exact", "real"), default="exact")
    parser.add_argument("--panel", choices=("gaussian_raw", "shuffled"), default="gaussian_raw")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, choices=WINDOWS, default=1000)
    parser.add_argument("--windows", default="250,500,1000", help="exact part: comma-separated windows")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    returns = simulated.load_returns(simulated.PRICES)
    if args.part == "exact":
        run_exact(returns, args.panel, args.seed, tuple(int(w) for w in args.windows.split(",")))
    else:
        run_real(returns, args.window)


def _minvar(matrix: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(matrix, np.ones(matrix.shape[0]))
    return raw / raw.sum()


def _top_direction(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    _, vectors = np.linalg.eigh(cov / np.outer(vol, vol))
    top = vectors[:, -1]
    return top * np.sign(top.sum())


def run_exact(returns: pd.DataFrame, panel: str, seed: int, windows: tuple[int, ...]) -> None:
    if panel == "gaussian_raw":
        data = shuffle.gaussian_raw(returns, seed).to_numpy(dtype=float)
        fixed = returns.cov(ddof=1).to_numpy(dtype=float)
    else:
        data = shuffle.shuffled(returns, seed).to_numpy(dtype=float)
        fixed = None
    n_days = data.shape[0]
    rows = []
    for window in windows:
        starts = list(range(window, n_days - HOLDING + 1, HOLDING))
        basis = error._basis(fixed) if fixed is not None else None
        for i, start in enumerate(starts):
            sample = data[start - window : start]
            if fixed is None:
                sigma = np.cov(np.concatenate([data[: start - window], data[start:]]), rowvar=False)
                basis = error._basis(sigma)
            else:
                sigma = fixed
            frame = pd.DataFrame(sample)
            js = james_stein_rie_covariance(frame)
            estimates = {
                "rie": james_stein_rie_covariance(frame, shrink=1.0).covariance.to_numpy(),
                "js": js.covariance.to_numpy(),
            }
            for c in FIXED_C:
                estimates[f"c={c}"] = james_stein_rie_covariance(frame, shrink=c).covariance.to_numpy()
            top = basis["vectors"][:, :1]
            estimates["true market direction"] = error._swap_block(estimates["rie"], top, np.diag(basis["values"][:1]))
            estimates["truth"] = sigma
            true_corr_top = basis["corr_vectors"][:, 0] * np.sign(basis["corr_vectors"][:, 0].sum())
            for name, matrix in estimates.items():
                w = _minvar(matrix)
                row = {"panel": panel, "seed": seed, "window": window, "date": i, "variant": name, "realised": float(w @ sigma @ w), "predicted": float(w @ matrix @ w)}
                if name in ("rie", "js"):
                    row["overlap"] = float((_top_direction(matrix) @ true_corr_top) ** 2)
                if name == "js":
                    row["c"] = js.info["c_data"]
                    row["turn_degrees"] = js.info["turn_degrees"]
                rows.append(row)
        print(f"{panel} seed {seed}: window {window} done ({len(starts)} dates)", flush=True)
    folder = OUT / "exact"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{panel}_s{seed}.csv" if windows == (250, 500, 1000) else f"{panel}_s{seed}_T{'-'.join(map(str, windows))}.csv"
    pd.DataFrame(rows).to_csv(folder / name, index=False)


def run_real(returns: pd.DataFrame, window: int) -> None:
    folder = OUT / f"real_T{window}"
    config = BacktestConfig(window=window, holding=HOLDING, strategies=("minvar_rie", "minvar_rie_js"))
    result = run_backtest(returns, config)
    result.save(folder)
    summary = result.summary()
    for name in config.strategies:
        print(f"T = {window} {name:16s} ratio {summary.loc[name, 'median_vol_ratio']:.3f}  vol {summary.loc[name, 'ann_volatility']:.2%}  turnover {summary.loc[name, 'mean_turnover']:.3f}")


def report() -> None:
    lines = ["James-Stein on the market eigenvector, on top of the RIE.", ""]
    exact_files = sorted((OUT / "exact").glob("*.csv")) if (OUT / "exact").exists() else []
    curves = {}
    if exact_files:
        data = pd.concat([pd.read_csv(f) for f in exact_files])
        data["ratio"] = np.sqrt(data["realised"] / data["predicted"])
        for panel, block in data.groupby("panel", sort=False):
            seeds = sorted(block["seed"].unique())
            lines.append(f"=== {PANEL_LABELS[panel]} (exact truth, seeds {seeds}) ===")
            present = tuple(sorted(block["window"].unique()))
            lines.append(f"{'variant':<24}" + "".join(f"{'T = ' + str(w):>22}" for w in present))
            vol = block.groupby(["window", "variant"])["realised"].mean().pipe(lambda s: np.sqrt(ANNUAL * s))
            hon = block.groupby(["window", "variant"])["ratio"].median()
            for variant in VARIANTS:
                cells = "".join(f"{vol[(w, variant)]:>13.2%} ({hon[(w, variant)]:.3f})" for w in present)
                lines.append(f"{variant:<24}{cells}")
            lines.append("  cells: realised volatility (median honesty)")
            for w in present:
                rie, js, ceiling = vol[(w, "rie")], vol[(w, "js")], vol[(w, "true market direction")]
                share = (rie - js) / (rie - ceiling) if rie > ceiling else float("nan")
                sub = block[block["window"] == w]
                c = sub.loc[sub["variant"] == "js", "c"]
                turn = sub.loc[sub["variant"] == "js", "turn_degrees"]
                ov_r = sub.loc[sub["variant"] == "rie", "overlap"].mean()
                ov_j = sub.loc[sub["variant"] == "js", "overlap"].mean()
                grid = {1.0: vol[(w, "rie")], **{cc: vol[(w, f"c={cc}")] for cc in FIXED_C}}
                best = min(grid, key=grid.get)
                lines.append(
                    f"  T = {w}: James-Stein {js / rie - 1:+.2%} vs RIE; captures {share:.0%} of what the true market direction gives ({ceiling / rie - 1:+.2%}); "
                    f"c median {c.median():.3f} (range {c.min():.2f}–{c.max():.2f}), turn {turn.median():.2f}°; "
                    f"market overlap {ov_r:.4f} → {ov_j:.4f}; best fixed c of 0.5/0.8/0.9/1: {best}"
                )
                curves.setdefault(panel, {})[w] = (rie, js, ceiling)
            lines.append("")
    real = []
    for w in WINDOWS:
        folder = OUT / f"real_T{w}"
        if (folder / "periods.csv").is_file():
            real.append((w, _common_period(BacktestResult.load(folder))))
    if real:
        lines.append(f"=== real market, rolling backtest, 21-day holds, every window scored on the same days ({COMMON_START} on) ===")
        for w, stats in real:
            r, j = stats["minvar_rie"], stats["minvar_rie_js"]
            lines.append(
                f"T = {w:>4}: RIE vol {r['vol']:.2%} honesty {r['honesty']:.3f} turnover {r['turnover']:.3f}  |  "
                f"James-Stein vol {j['vol']:.2%} honesty {j['honesty']:.3f} turnover {j['turnover']:.3f} c {j['c']:.3f}  ({j['vol'] / r['vol'] - 1:+.2%})"
            )
            curves.setdefault("real", {})[w] = (r["vol"], j["vol"], None)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    if curves:
        plot(curves, OUT / "james_stein.png")


def _common_period(result: BacktestResult) -> dict[str, dict[str, float]]:
    """Volatility, median honesty, turnover and median c over the holds that start on or after COMMON_START."""
    start = pd.Timestamp(COMMON_START)
    daily = result.daily_returns.loc[pd.to_datetime(result.daily_returns.index) >= start]
    periods = result.periods.copy()
    periods = periods.loc[pd.to_datetime(periods["hold_start"]) >= start]
    out = {}
    for name in ("minvar_rie", "minvar_rie_js"):
        rows = periods.loc[periods["strategy"] == name]
        out[name] = {
            "vol": float(daily[name].std(ddof=1) * np.sqrt(ANNUAL)),
            "honesty": float(rows["vol_ratio"].median()),
            "turnover": float(rows["turnover"].mean()),
            "c": float(rows["c_data"].median()) if "c_data" in rows else float("nan"),
        }
    return out


def plot(curves: dict, path: Path) -> None:
    simulated._style()
    panels = [p for p in ("gaussian_raw", "shuffled", "real") if p in curves]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        windows = sorted(curves[panel])
        x = np.arange(len(windows))
        rie = [curves[panel][w][0] for w in windows]
        js = [curves[panel][w][1] for w in windows]
        ax.bar(x - 0.2, rie, width=0.38, color="#9a9a9a", label="RIE")
        ax.bar(x + 0.2, js, width=0.38, color="#2a78d6", label="RIE + James-Stein")
        ceiling = [curves[panel][w][2] for w in windows]
        if all(v is not None for v in ceiling):
            ax.scatter(x + 0.2, ceiling, marker="_", s=400, color="#1f1f1f", linewidths=2, label="true market direction", zorder=3)
        for xi, a, b in zip(x, rie, js):
            ax.text(xi + 0.2, b, f"{b / a - 1:+.1%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"T = {w}" for w in windows])
        low = min(min(js), min(c for c in ceiling if c is not None) if any(c is not None for c in ceiling) else min(js))
        ax.set_ylim(low * 0.9, max(rie) * 1.04)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.set_title(PANEL_LABELS[panel])
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("realised volatility, annualised")
    axes[0][0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("James-Stein on the market direction, on top of the RIE", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
