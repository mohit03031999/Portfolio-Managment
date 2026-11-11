"""Design A, "directions from the past, sizes from the present": the two-speed estimator.

The idea (Mohit's, after the memory-length study): take the directions from a
long history, because the top room survives a long window and the quiet
directions get genuinely quieter with more data; take the sizes along those
directions from recent days, because a size measured on days the direction was
not fitted to carries over honestly; take the volatilities from recent days,
because they are weather.

Recipe, per decision date:

1. directions U = eigenvectors of the correlation matrix of the last L_dir days
   ("all" = every day before the date, or a fixed 2000);
2. sizes xi_i = u_iᵀ C_recent u_i, the variance of the last L_size days along
   each direction (correlation units), forced to be decreasing in the old rank by
   isotonic regression (pool-adjacent-violators) to tame the noise;
3. correlation = U diag(xi) Uᵀ, renormalised to ones on the diagonal;
4. covariance = that correlation scaled by the sample volatilities of the last
   L_size days.

No random-matrix formula is needed: the sizes are out of sample by construction.

Baselines: the 1000-day RIE and the 250-day RIE (our best plain window). Worlds:
``real`` (realised risk from the next 21 days; turnover and leverage recorded)
and ``gaussian_raw`` (truth = raw full-sample covariance; nothing drifts, so more
data can only help: a machinery check).

Outputs in output/sp500_two_speed/: {panel}[_s{seed}]/risk.csv (+ returns.csv,
weights.csv for real), spectrum.csv; summary.txt; two_speed.png.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.optimizer import rie_covariance  # noqa: E402

OUT = ROOT / "output" / "sp500_two_speed"
WINDOW, HOLDING, ANNUAL = 1000, 21, 252
SIZE_LENGTHS = (250, 500, 1000)
LONG = 2000
PANEL_LABELS = {"real": "real market", "gaussian_raw": "Gaussian days"}
RIE, TWO, LONGC, MUTED, INK = "#2a6fdb", "#e07b00", "#1a9e5c", "#777777", "#222222"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", choices=tuple(PANEL_LABELS), default="real")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", action="store_true", help="aggregate saved runs and draw the figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    returns = simulated.load_returns(simulated.PRICES)
    run(returns, args.panel, args.seed)


# ----------------------------------------------------------------------------- the estimator


def correlation(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean(axis=0)
    vol = np.sqrt((centred**2).sum(axis=0) / (len(x) - 1))
    vol = np.where(vol > 0, vol, 1.0)
    z = centred / vol
    return z.T @ z / (len(x) - 1)


def top_vectors(c: np.ndarray) -> np.ndarray:
    _, vectors = np.linalg.eigh(c)
    return vectors[:, ::-1]


def pav_decreasing(y: np.ndarray) -> np.ndarray:
    """Best non-increasing fit to ``y`` in least squares (pool-adjacent-violators)."""
    values, counts = [], []
    for v in y[::-1]:
        values.append(float(v))
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            total = counts[-2] + counts[-1]
            merged = (values[-2] * counts[-2] + values[-1] * counts[-1]) / total
            values.pop()
            counts.pop()
            values[-1], counts[-1] = merged, total
    return np.repeat(values, counts)[::-1]


def two_speed(history: np.ndarray, recent: np.ndarray, isotonic: bool = True, floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Covariance with directions from ``history`` and sizes and volatilities from ``recent``. Returns (covariance, sizes)."""
    vectors = top_vectors(correlation(history))
    c_recent = correlation(recent)
    sizes = np.einsum("ij,ik,kj->j", vectors, c_recent, vectors)
    if isotonic:
        sizes = pav_decreasing(sizes)
    sizes = np.maximum(sizes, floor)
    corr = (vectors * sizes) @ vectors.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vol = (recent - recent.mean(axis=0)).std(axis=0, ddof=1)
    return corr * np.outer(vol, vol), sizes


def _minvar(matrix: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(matrix, np.ones(matrix.shape[0]))
    return raw / raw.sum()


def _revol(cov: np.ndarray, recent: np.ndarray) -> np.ndarray:
    """The same correlation structure, rescaled by the sample volatilities of ``recent``."""
    sd = np.sqrt(np.diag(cov))
    vol = (recent - recent.mean(axis=0)).std(axis=0, ddof=1)
    return cov / np.outer(sd, sd) * np.outer(vol, vol)


def estimates(data: np.ndarray, start: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All estimators at one decision date. Also returns spectra for the figure."""
    window = data[start - WINDOW : start]
    last_250 = data[start - 250 : start]
    out = {"rie_1000": rie_covariance(pd.DataFrame(window)).covariance.to_numpy(dtype=float)}
    out["rie_250"] = rie_covariance(pd.DataFrame(last_250)).covariance.to_numpy(dtype=float)
    # weather controls: the same correlation structure with the other window's volatilities
    out["rie_1000_vol_250"] = _revol(out["rie_1000"], last_250)
    out["rie_250_vol_1000"] = _revol(out["rie_250"], window)
    spectra = {}
    history = data[:start]  # includes the sizing window: the sizes are partly in-sample
    for size in SIZE_LENGTHS:
        out[f"two_all_{size}"], spectra[f"two_all_{size}"] = two_speed(history, data[start - size : start])
    out["two_all_250_raw"], spectra["two_all_250_raw"] = two_speed(history, last_250, isotonic=False)
    for size in (250, 500):  # disjoint: directions from the days before the sizing window only
        out[f"two_disjoint_{size}"], spectra[f"two_disjoint_{size}"] = two_speed(data[: start - size], data[start - size : start])
    if start >= LONG:
        for size in (250, 500):
            out[f"two_{LONG}_{size}"], _ = two_speed(data[start - LONG : start], data[start - size : start])
        out[f"two_{LONG}_disjoint_250"], _ = two_speed(data[start - LONG : start - 250], last_250)
    return out, spectra


# ----------------------------------------------------------------------------- the run


def run(returns: pd.DataFrame, panel: str, seed: int) -> None:
    if panel == "gaussian_raw":
        data = shuffle.gaussian_raw(returns, seed).to_numpy(dtype=float)
        truth = returns.cov(ddof=1).to_numpy(dtype=float)
        w_star = _minvar(truth)
        best = float(w_star @ truth @ w_star)
    else:
        data = returns.to_numpy(dtype=float)
        truth = None
    n_days, n = data.shape
    starts = list(range(WINDOW, n_days - HOLDING + 1, HOLDING))
    risk_rows, return_rows, weight_rows = [], [], []
    previous: dict[str, np.ndarray] = {}
    clock = time.perf_counter()
    for i, start in enumerate(starts):
        matrices, spectra = estimates(data, start)
        if i == len(starts) - 1:
            sample_values = np.sort(np.linalg.eigvalsh(correlation(data[start - WINDOW : start])))[::-1]
            rie_corr = matrices["rie_1000"] / np.outer(*(2 * [np.sqrt(np.diag(matrices["rie_1000"]))]))
            rie_values = np.sort(np.linalg.eigvalsh(rie_corr))[::-1]
            table = {"rank": np.arange(1, n + 1), "sample_1000": sample_values, "rie_1000": rie_values, **spectra}
            pd.DataFrame(table).to_csv(OUT / f"spectrum_{panel}.csv", index=False)
        for name, matrix in matrices.items():
            w = _minvar(matrix)
            predicted = float(w @ matrix @ w)
            if truth is None:
                daily = data[start : start + HOLDING] @ w
                realised = float(np.mean(daily**2))
                return_rows += [{"estimator": name, "date": i, "day": d, "ret": float(r)} for d, r in enumerate(daily)]
            else:
                realised = float(w @ truth @ w)
            turnover = float(np.abs(w - previous[name]).sum()) if name in previous else np.nan
            previous[name] = w
            risk_rows.append({"estimator": name, "date": i, "realised": realised, "predicted": predicted, "turnover": turnover, "leverage": float(np.abs(w).sum()), "long_history": start >= LONG})
        if truth is not None:
            risk_rows.append({"estimator": "oracle", "date": i, "realised": best, "predicted": best, "turnover": 0.0, "leverage": float(np.abs(w_star).sum()), "long_history": start >= LONG})
        if i % 10 == 0:
            print(f"{panel}: date {i + 1} of {len(starts)}, {time.perf_counter() - clock:.0f}s", flush=True)
    folder = OUT / (f"{panel}_s{seed}" if panel != "real" else "real")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(risk_rows).to_csv(folder / "risk.csv", index=False)
    if return_rows:
        pd.DataFrame(return_rows).to_csv(folder / "returns.csv", index=False)
    print(f"saved {folder}")


# ----------------------------------------------------------------------------- the report

LABELS = {
    "rie_1000": "RIE, 1000 days (baseline)",
    "rie_1000_vol_250": "  control: RIE 1000-day correlation, 250-day vols",
    "rie_250": "RIE, 250 days (best plain window)",
    "rie_250_vol_1000": "  control: RIE 250-day correlation, 1000-day vols",
    "two_all_1000": "A: dirs all history (incl. window), sizes & vols 1000 d",
    "two_all_500": "A: dirs all history (incl. window), sizes & vols 500 d",
    "two_all_250": "A: dirs all history (incl. window), sizes & vols 250 d",
    "two_all_250_raw": "A: same as above, without isotonic smoothing",
    "two_disjoint_500": "A: dirs from days before the window, sizes & vols 500 d",
    "two_disjoint_250": "A: dirs from days before the window, sizes & vols 250 d",
    f"two_{LONG}_250": f"A: dirs {LONG} d (incl. window), sizes & vols 250 d",
    f"two_{LONG}_500": f"A: dirs {LONG} d (incl. window), sizes & vols 500 d",
    f"two_{LONG}_disjoint_250": f"A: dirs {LONG - 250} d before the window, sizes & vols 250 d",
}


def _table(risk: pd.DataFrame) -> pd.DataFrame:
    risk = risk.copy()
    risk["ratio"] = np.sqrt(risk["realised"] / risk["predicted"])
    table = risk.groupby("estimator").agg(realised=("realised", "mean"), honesty=("ratio", "median"), turnover=("turnover", "mean"), leverage=("leverage", "mean"))
    table["volatility"] = np.sqrt(ANNUAL * table["realised"])
    return table


def report() -> None:
    lines = ["Design A, directions from the past and sizes from the present. Holds of 21 days.", ""]
    results: dict[str, pd.DataFrame] = {}
    for panel in PANEL_LABELS:
        folders = sorted(path for path in OUT.glob(f"{panel}_s*" if panel != "real" else "real") if (path / "risk.csv").exists())
        if not folders:
            continue
        risk = pd.concat([pd.read_csv(f / "risk.csv").assign(run=f.name) for f in folders])
        lines.append(f"=== {PANEL_LABELS[panel]} ({', '.join(f.name for f in folders)}; {risk['date'].nunique()} dates) ===")
        full = risk[~risk["estimator"].str.startswith(f"two_{LONG}")]
        table = _table(full)
        if panel == "real":
            daily = pd.concat([pd.read_csv(f / "returns.csv") for f in folders])
            overall = daily[~daily["estimator"].str.startswith(f"two_{LONG}")].groupby("estimator")["ret"].std() * np.sqrt(ANNUAL)
            table["volatility"] = overall
        base = float(table.loc["rie_1000", "volatility"])
        oracle = float(table.loc["oracle", "volatility"]) if "oracle" in table.index else None
        header = "  estimator                                          volatility   vs RIE 1000" + ("   gap closed" if oracle else "") + "   honesty   turnover   leverage"
        lines.append(header)
        for name in [k for k in LABELS if k in table.index]:
            row = table.loc[name]
            extra = f"   {(base - row['volatility']) / (base - oracle):+8.0%}" if oracle else ""
            lines.append(f"  {LABELS[name]:<50} {row['volatility']:7.2%}   {row['volatility'] / base - 1:+8.1%}{extra}   {row['honesty']:7.2f}   {row['turnover']:8.2f}   {row['leverage']:8.2f}")
        if oracle:
            lines.append(f"  truth: {oracle:.2%}")
        results[panel] = table
        subset = risk[risk["long_history"]]
        if subset["estimator"].str.startswith(f"two_{LONG}").any():
            sub_table = _table(subset)
            if panel == "real":
                dates = set(subset["date"])
                sub_daily = daily[daily["date"].isin(dates)]
                sub_table["volatility"] = sub_daily.groupby("estimator")["ret"].std() * np.sqrt(ANNUAL)
            lines.append(f"  --- dates with {LONG} days of history ({subset['date'].nunique()} dates) ---")
            sub_base = float(sub_table.loc["rie_1000", "volatility"])
            for name in ("rie_1000", "rie_1000_vol_250", "rie_250", "two_all_250", "two_disjoint_250", f"two_{LONG}_250", f"two_{LONG}_500", f"two_{LONG}_disjoint_250"):
                if name in sub_table.index:
                    row = sub_table.loc[name]
                    lines.append(f"  {LABELS[name]:<50} {row['volatility']:7.2%}   {row['volatility'] / sub_base - 1:+8.1%}   honesty {row['honesty']:.2f}   turnover {row['turnover']:.2f}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    plot(results, OUT / "two_speed.png")


def plot(results: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    names = ["rie_1000", "rie_1000_vol_250", "rie_250", "rie_250_vol_1000", "two_all_1000", "two_all_500", "two_disjoint_500", "two_all_250", "two_disjoint_250"]
    short = ["RIE\n1000 d", "RIE 1000\n+ vols 250", "RIE\n250 d", "RIE 250\n+ vols 1000", "A: all dirs\nsizes 1000", "A: all dirs\nsizes 500", "A: disjoint\nsizes 500", "A: all dirs\nsizes 250", "A: disjoint\nsizes 250"]
    pale = "#8fb4ec"
    for ax, panel in zip(axes[:2], ("real", "gaussian_raw")):
        if panel not in results:
            ax.set_visible(False)
            continue
        table = results[panel]
        vols = [table.loc[n, "volatility"] * 100 for n in names]
        colors = [RIE, pale, RIE, pale] + [TWO] * 5
        bars = ax.bar(range(len(names)), vols, color=colors, width=0.62)
        for b, n in zip(bars, names):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{table.loc[n, 'volatility']:.1%}\nh {table.loc[n, 'honesty']:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        if "oracle" in table.index:
            ax.axhline(table.loc["oracle", "volatility"] * 100, color=MUTED, lw=1, ls=":")
            ax.text(len(names) - 0.5, table.loc["oracle", "volatility"] * 100 + 0.05, "truth", fontsize=8, color=MUTED, ha="right")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=7.5)
        ax.set_ylim(min(vols) - 1.6, max(vols) + 1.0)
        ax.set_ylabel("realised volatility, % per year")
        ax.set_title(f"{'1' if panel == 'real' else '2'}. {PANEL_LABELS[panel]} (h = honesty)", loc="left", fontsize=11)
    ax = axes[2]
    spectrum = OUT / "spectrum_real.csv"
    if spectrum.exists():
        s = pd.read_csv(spectrum)
        ax.plot(s["rank"], s["sample_1000"], color=RIE, lw=1.5, ls="--", label="sample, 1000 days")
        ax.plot(s["rank"], s["rie_1000"], color=RIE, lw=2, label="RIE, 1000 days")
        ax.plot(s["rank"], s["two_all_250_raw"], color=TWO, lw=1, alpha=0.5, label="A: sizes on 250 days, raw")
        ax.plot(s["rank"], s["two_all_250"], color=TWO, lw=2, label="A: sizes on 250 days, isotonic (dirs incl. window)")
        ax.plot(s["rank"], s["two_disjoint_250"], color=LONGC, lw=2, label="A: sizes on 250 days, isotonic (dirs before window)")
        ax.plot(s["rank"], s["two_all_1000"], color=MUTED, lw=1.5, label="A: sizes on 1000 days, isotonic")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("direction rank")
        ax.set_ylabel("size (correlation units, log)")
        ax.set_title("3. The sizes each method assigns (last real window)", loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
