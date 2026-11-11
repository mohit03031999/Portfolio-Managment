"""Which directions does the RIE get wrong, and which of them cost risk?

The RIE gives each sample direction the best possible variance, yet its
minimum-variance portfolio stays riskier than the one built from the truth.
What is left is error in the directions themselves. This script asks where
that error sits, in two worlds where the truth is known exactly:

* ``gaussian_raw``: Gaussian days drawn from the raw full-sample covariance
  of the 456 S&P 500 stocks. Pure estimation noise.
* ``shuffled``: the real days in random order. Fat tails and jumps, no time
  order. The truth for a window is the covariance of the days outside it,
  because the hold days are drawn from those.

For every window of T = 1000 days (the same 93 decision dates as the
backtest) it measures four things.

1. **Where the extra risk lands.** With budget-only weights the extra
   variance of any portfolio over the best one is exactly
   ``(w − w*)ᵀ Σ (w − w*) = Σ_j λ_j (v_jᵀ (w − w*))²`` over the true
   eigenvectors ``v_j`` with variances ``λ_j``, because ``Σ w*`` is
   proportional to the vector of ones. That splits the extra variance by
   true direction: the market, the next few factors, and so on.
2. **Hand the RIE some true directions.** Replace the RIE's matrix inside
   the span of the true top-``k`` directions by the truth there, keep the
   RIE in the rest: ``V_k Λ_k V_kᵀ + P Σ̂ P`` with ``P = I − V_k V_kᵀ``.
   The same with the true bottom-``k`` directions (the quietest ones, where
   minimum variance puts its weight). ``k = 0`` is the RIE, ``k = 456`` the
   truth. A second version gives only the true top-``k`` *subspace* and
   keeps the RIE's own numbers inside it: ``V_k (V_kᵀ Σ̂ V_k) V_kᵀ + P Σ̂ P``.
3. **How well each sample direction lines up with the true one**: the
   squared overlap of the sample and true correlation eigenvectors for the
   top 20 modes, next to the textbook prediction for a single spike on a
   flat noise floor.
4. **The dispersion bias** of Goldberg, Papanicolaou and Shkolnik: the
   cross-sectional spread of the sample market eigenvector's entries
   relative to their mean, divided by the same for the true one.

Realised variance is the exact expected variance ``wᵀ Σ w`` under the
date's truth, not a simulated hold.

    .venv/bin/python examples/sp500_eigenvector_error.py --panel gaussian_raw --seed 1   # runs can go in parallel
    .venv/bin/python examples/sp500_eigenvector_error.py --panel gaussian_raw --seed 2
    .venv/bin/python examples/sp500_eigenvector_error.py --panel shuffled --seed 1
    .venv/bin/python examples/sp500_eigenvector_error.py --panel shuffled --seed 2
    .venv/bin/python examples/sp500_eigenvector_error.py --report
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

import sp500_shuffle_test as shuffle  # noqa: E402
import sp500_simulated_backtest as simulated  # noqa: E402
from market_analyzer.optimizer import rie_covariance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "sp500_eigenvector_error"
WINDOW = 1000
HOLDING = 21
ANNUAL = 252
COUNTS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 456)
GROUPS = (("market (1)", 1, 1), ("factors 2–10", 2, 10), ("modes 11–50", 11, 50), ("modes 51–200", 51, 200), ("quietest 201–456", 201, 456))
N_OVERLAP = 20
PANEL_LABELS = {"gaussian_raw": "Gaussian days", "shuffled": "real days, shuffled"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", choices=tuple(PANEL_LABELS), default="gaussian_raw")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", action="store_true", help="aggregate saved runs and draw the figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return
    returns = simulated.load_returns(simulated.PRICES)
    run(returns, args.panel, args.seed)


def run(returns: pd.DataFrame, panel: str, seed: int) -> None:
    if panel == "gaussian_raw":
        data = shuffle.gaussian_raw(returns, seed).to_numpy(dtype=float)
        fixed_truth = returns.cov(ddof=1).to_numpy(dtype=float)
    else:
        data = shuffle.shuffled(returns, seed).to_numpy(dtype=float)
        fixed_truth = None
    n_days, n_assets = data.shape
    starts = list(range(WINDOW, n_days - HOLDING + 1, HOLDING))
    truth_basis = _basis(fixed_truth) if fixed_truth is not None else None
    risk_rows, excess_rows, overlap_rows = [], [], []
    for i, start in enumerate(starts):
        window = data[start - WINDOW : start]
        if fixed_truth is None:
            outside = np.concatenate([data[: start - WINDOW], data[start:]])
            sigma = np.cov(outside, rowvar=False)
            basis = _basis(sigma)
        else:
            sigma, basis = fixed_truth, truth_basis
        risk, excess, overlap = analyse(window, sigma, basis)
        for row in risk:
            row["date"] = i
        for row in excess:
            row["date"] = i
        overlap["date"] = i
        risk_rows += risk
        excess_rows += excess
        overlap_rows.append(overlap)
        if i % 10 == 0:
            print(f"{panel} seed {seed}: window {i + 1} of {len(starts)}", flush=True)
    folder = OUT / f"{panel}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(risk_rows).to_csv(folder / "risk.csv", index=False)
    pd.DataFrame(excess_rows).to_csv(folder / "excess.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(folder / "overlap.csv", index=False)
    print(f"saved {folder}")


def _basis(sigma: np.ndarray) -> dict[str, np.ndarray]:
    values, vectors = np.linalg.eigh(sigma)
    vol = np.sqrt(np.diag(sigma))
    corr_values, corr_vectors = np.linalg.eigh(sigma / np.outer(vol, vol))
    return {
        "values": values[::-1],
        "vectors": vectors[:, ::-1],
        "corr_values": corr_values[::-1],
        "corr_vectors": corr_vectors[:, ::-1],
        "w_star": _minvar(sigma),
    }


def _minvar(matrix: np.ndarray) -> np.ndarray:
    raw = np.linalg.solve(matrix, np.ones(matrix.shape[0]))
    return raw / raw.sum()


def _swap_block(estimate: np.ndarray, vectors: np.ndarray, block: np.ndarray) -> np.ndarray:
    """``vectors · block · vectorsᵀ + P · estimate · P`` with ``P = I − vectors vectorsᵀ``."""
    a = vectors.T @ estimate
    inner = a @ vectors
    rest = estimate - vectors @ a - a.T @ vectors.T + vectors @ inner @ vectors.T
    return vectors @ block @ vectors.T + rest


def analyse(window: np.ndarray, sigma: np.ndarray, basis: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], dict]:
    n = sigma.shape[0]
    values, vectors, w_star = basis["values"], basis["vectors"], basis["w_star"]
    best = float(w_star @ sigma @ w_star)
    rie = rie_covariance(pd.DataFrame(window)).covariance.to_numpy(dtype=float)
    sample = np.cov(window, rowvar=False)

    risk = [{"estimator": "oracle", "count": n, "version": "truth", "realised": best, "predicted": best}]

    def record(estimator: str, count: int, version: str, matrix: np.ndarray) -> np.ndarray:
        w = _minvar(matrix)
        risk.append({"estimator": estimator, "count": count, "version": version, "realised": float(w @ sigma @ w), "predicted": float(w @ matrix @ w)})
        return w

    excess = []
    for name, matrix in (("rie", rie), ("sample", sample)):
        w = record(name, 0, "as is", matrix)
        parts = values * (vectors.T @ (w - w_star)) ** 2
        for label, lo, hi in GROUPS:
            excess.append({"estimator": name, "group": label, "extra": float(parts[lo - 1 : hi].sum())})
    for k in COUNTS[1:-1]:
        top = vectors[:, :k]
        record("rie", k, "true top block", _swap_block(rie, top, np.diag(values[:k])))
        record("rie", k, "true top subspace", _swap_block(rie, top, top.T @ rie @ top))
        bottom = vectors[:, n - k :]
        record("rie", k, "true bottom block", _swap_block(rie, bottom, np.diag(values[n - k :])))

    corr_vectors, corr_values = basis["corr_vectors"], basis["corr_values"]
    centred = window - window.mean(axis=0)
    vol = np.sqrt(np.sum(centred**2, axis=0) / (len(window) - 1))
    sample_corr = (centred / vol).T @ (centred / vol) / (len(window) - 1)
    _, sample_vectors = np.linalg.eigh(sample_corr)
    sample_vectors = sample_vectors[:, ::-1]
    overlap = {f"overlap_{k + 1}": float((sample_vectors[:, k] @ corr_vectors[:, k]) ** 2) for k in range(N_OVERLAP)}
    overlap["dispersion_true"] = _dispersion(corr_vectors[:, 0])
    overlap["dispersion_sample"] = _dispersion(sample_vectors[:, 0])
    q = n / (len(window) - 1)
    floor = float(corr_values[N_OVERLAP:].mean())
    for k in range(N_OVERLAP):
        overlap[f"theory_{k + 1}"] = _spike_overlap(corr_values[k] / floor, q)
    return risk, excess, overlap


def _dispersion(vector: np.ndarray) -> float:
    vector = vector if vector.mean() > 0 else -vector
    return float(vector.std() / vector.mean())


def _spike_overlap(ratio: float, q: float) -> float:
    """Squared overlap of a sample spike with its true direction: one spike ``ratio`` times a flat noise floor (Paul 2007)."""
    excess = ratio - 1.0
    if excess <= np.sqrt(q):
        return 0.0
    return float((1.0 - q / excess**2) / (1.0 + q / excess))


def _runs() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for panel in PANEL_LABELS:
        found[panel] = sorted(path for path in OUT.glob(f"{panel}_s*") if (path / "risk.csv").exists())
    return found


def report() -> None:
    runs = _runs()
    lines = ["Where the RIE's direction error costs risk. Window T = 1000, 93 decision dates per run, exact truth.", ""]
    curves, shares, overlaps = {}, {}, {}
    for panel, folders in runs.items():
        if not folders:
            continue
        risk = pd.concat([pd.read_csv(f / "risk.csv").assign(run=f.name) for f in folders])
        excess = pd.concat([pd.read_csv(f / "excess.csv").assign(run=f.name) for f in folders])
        overlap = pd.concat([pd.read_csv(f / "overlap.csv").assign(run=f.name) for f in folders])
        label = PANEL_LABELS[panel]
        lines.append(f"=== {label} ({', '.join(f.name for f in folders)}) ===")

        table = risk.groupby(["estimator", "version", "count"]).agg(realised=("realised", "mean"), predicted=("predicted", "mean"))
        risk["ratio"] = np.sqrt(risk["realised"] / risk["predicted"])
        table["honesty"] = risk.groupby(["estimator", "version", "count"])["ratio"].median()
        table["volatility"] = np.sqrt(ANNUAL * table["realised"])
        oracle = float(table.loc[("oracle", "truth", 456), "volatility"])
        base = float(table.loc[("rie", "as is", 0), "volatility"])
        sample = float(table.loc[("sample", "as is", 0), "volatility"])
        lines.append(f"Volatility: sample {sample:.2%}, RIE {base:.2%}, truth {oracle:.2%}; the RIE is {base / oracle - 1:+.1%} above the truth.")
        lines.append("")
        lines.append("Give the RIE the truth on some true directions (realised volatility; share of the RIE-to-truth gap closed; median honesty):")
        curve = {}
        for version in ("true top block", "true bottom block", "true top subspace"):
            rows = [("RIE as is", base, 0.0, float(table.loc[("rie", "as is", 0), "honesty"]))]
            points = [(0, base)]
            for k in COUNTS[1:-1]:
                vol = float(table.loc[("rie", version, k), "volatility"])
                rows.append((f"k = {k}", vol, (base - vol) / (base - oracle), float(table.loc[("rie", version, k), "honesty"])))
                points.append((k, vol))
            if version != "true top subspace":
                points.append((456, oracle))
            curve[version] = points
            lines.append(f"  {version}:")
            lines += [f"    {name:<10} {vol:7.2%}   gap closed {closed:6.0%}   honesty {hon:.3f}" for name, vol, closed, hon in rows]
        curves[panel] = (curve, base, oracle)
        lines.append("")

        lines.append("Where the extra variance lands (share of the extra variance over the truth's portfolio, by true direction):")
        totals = excess.groupby(["estimator", "run", "date"])["extra"].transform("sum")
        excess["share"] = excess["extra"] / totals
        mean_extra = excess.groupby(["estimator", "group"], sort=False)["extra"].mean()
        share = mean_extra / mean_extra.groupby(level=0).transform("sum")
        shares[panel] = share
        for estimator in ("rie", "sample"):
            parts = "  ".join(f"{group} {share[(estimator, group)]:.0%}" for group, _, _ in GROUPS)
            lines.append(f"  {estimator:<7} {parts}")
        lines.append("")

        mean = overlap.mean(numeric_only=True)
        overlaps[panel] = mean
        lines.append("Squared overlap of each sample correlation eigenvector with the true one (1 = same direction, ~0.002 = random):")
        lines.append("  mode      " + " ".join(f"{k:>5}" for k in range(1, N_OVERLAP + 1)))
        lines.append("  measured  " + " ".join(f"{mean[f'overlap_{k}']:5.2f}" for k in range(1, N_OVERLAP + 1)))
        lines.append("  theory    " + " ".join(f"{mean[f'theory_{k}']:5.2f}" for k in range(1, N_OVERLAP + 1)))
        ratio = overlap["dispersion_sample"] / overlap["dispersion_true"]
        lines.append(
            f"Dispersion of the market eigenvector (std / mean of its entries): true {mean['dispersion_true']:.3f}, "
            f"sample {mean['dispersion_sample']:.3f}; ratio median {ratio.median():.2f} (range {ratio.min():.2f}–{ratio.max():.2f})."
        )
        lines.append("")
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n")
    print(text)
    if curves:
        plot(curves, shares, OUT / "eigenvector_error.png")


def plot(curves: dict, shares: dict, path: Path) -> None:
    simulated._style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    colors = {"true top block": "#2a78d6", "true bottom block": "#eb6834"}
    names = {"true top block": "largest directions first", "true bottom block": "quietest directions first"}
    styles = {"gaussian_raw": "-", "shuffled": "--"}
    positions = {k: i for i, k in enumerate(COUNTS)}
    for panel, (curve, base, oracle) in curves.items():
        for version in ("true top block", "true bottom block"):
            xs = [positions[k] for k, _ in curve[version]]
            ys = [v for _, v in curve[version]]
            left.plot(xs, ys, styles[panel], color=colors[version], linewidth=2, marker="o", markersize=4, label=f"{names[version]}, {PANEL_LABELS[panel]}")
    left.set_xticks(range(len(COUNTS)))
    left.set_xticklabels([str(k) for k in COUNTS])
    left.set_xlabel("number of true directions handed to the RIE (0 = RIE, 456 = the truth)")
    left.set_ylabel("realised volatility, annualised")
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    left.set_title("Which true directions would close the gap?")
    left.legend(fontsize=9, loc="lower left")

    groups = [g for g, _, _ in GROUPS]
    shades = ("#0b3d91", "#2a78d6", "#7fb0ea", "#bcd5f3", "#e3edf9")
    bars = []
    for panel in shares:
        for estimator, name in (("rie", "RIE"), ("sample", "sample")):
            bars.append((f"{name}\n{PANEL_LABELS[panel]}", [float(shares[panel][(estimator, g)]) for g in groups]))
    for i, (label, values) in enumerate(bars):
        left_edge = 0.0
        for j, value in enumerate(values):
            right.barh(i, value, left=left_edge, color=shades[j], edgecolor="white", linewidth=2, label=groups[j] if i == 0 else None)
            if value >= 0.07:
                right.text(left_edge + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=9, color="white" if j < 2 else "#1f1f1f")
            left_edge += value
    right.set_yticks(range(len(bars)))
    right.set_yticklabels([label for label, _ in bars], fontsize=9)
    right.invert_yaxis()
    right.set_xlim(0, 1)
    right.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    right.grid(axis="y", visible=False)
    right.set_title("Where the extra variance lands, by true direction")
    right.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
