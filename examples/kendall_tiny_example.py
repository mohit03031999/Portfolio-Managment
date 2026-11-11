"""Kendall's tau by hand, on real S&P 500 returns: a teaching example.

1. Five days of two stocks: every pair of days casts one vote (did both stocks
   move the same way between the two days?), and tau = (agree − disagree) / pairs.
2. One company-news day (a stock moving more than 15% on a calm market day):
   how much one day moves Pearson's correlation and how much it moves Kendall's.
3. The Gaussian dictionary tau = (2/pi) arcsin(rho) and the noise bulks of the
   two matrices at our N/T.

Prints to the terminal; no files are written.
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from sp500_simulated_backtest import PRICES, load_returns  # noqa: E402


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def kendall(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    i, j = np.triu_indices(len(a), 1)
    sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    return float((sa @ sb) / np.sqrt((sa @ sa) * (sb @ sb))), int((sa * sb > 0).sum()), int((sa * sb < 0).sum())


def main() -> None:
    r = load_returns(PRICES)

    pair = ("AAPL", "XOM") if {"AAPL", "XOM"} <= set(r.columns) else tuple(r.columns[:2])
    days = r.loc["2024-01-03":, list(pair)].iloc[:5]
    print("=== 1. Five days, two stocks (returns in %) ===")
    print((days * 100).round(2).to_string())
    a, b = days.iloc[:, 0].to_numpy(), days.iloc[:, 1].to_numpy()
    print(f"\nday pair   {pair[0]:>6}   {pair[1]:>6}   vote")
    for s, t in zip(*np.triu_indices(5, 1)):
        sa, sb = np.sign(a[s] - a[t]), np.sign(b[s] - b[t])
        print(f"  ({s + 1},{t + 1})    {'up' if sa > 0 else 'down':>6}   {'up' if sb > 0 else 'down':>6}   {'agree' if sa * sb > 0 else 'disagree'}")
    tau, agree, disagree = kendall(a, b)
    print(f"\nagree {agree}, disagree {disagree} of 10 pairs → Kendall tau = ({agree} − {disagree}) / 10 = {tau:.2f}")
    print(f"Pearson correlation of the same 5 days = {pearson(a, b):.2f}")

    print("\n=== 2. A huge single-stock day on a calm market day (60 days around it) ===")
    market = r.mean(axis=1)
    corr = r.corr()
    candidates = []
    for col in r.columns:
        for d in r.index[(r[col].abs() > 0.15) & (market.abs() < 0.003)]:
            if "2015" <= str(d)[:4] <= "2025":
                candidates.append((col, d, float(r.loc[d, col]), float(market.loc[d])))
    candidates.sort(key=lambda x: -abs(x[2]))
    shown = 0
    for col, d, ret, mkt in candidates:
        loc = r.index.get_loc(d)
        if loc < 30 or loc + 30 >= len(r):
            continue
        partner = corr[col].drop(col).idxmax()
        stretch = r.iloc[loc - 30 : loc + 30][[col, partner]]
        without = stretch.drop(index=d)
        p_with, p_without = pearson(*stretch.to_numpy().T), pearson(*without.to_numpy().T)
        k_with, k_without = kendall(*stretch.to_numpy().T)[0], kendall(*without.to_numpy().T)[0]
        print(f"{col} on {str(d)[:10]}: {ret:+.1%} while the market moved {mkt:+.2%}; partner {partner} ({r.loc[d, partner]:+.1%} that day)")
        print(f"   Pearson {p_with:+.2f} with the day, {p_without:+.2f} without  |  Kendall {k_with:+.2f} with, {k_without:+.2f} without")
        shown += 1
        if shown == 3:
            break

    print("\n=== 3. If returns were Gaussian: tau = (2/pi) arcsin(rho) ===")
    for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        print(f"  rho {rho:.2f} → tau {2 / np.pi * np.arcsin(rho):.2f}")
    q = r.shape[1] / 999
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    print(f"\nNoise bulk at q = {q:.3f} (T = 1000): Pearson eigenvalues in [{lo:.3f}, {hi:.3f}]; Kendall eigenvalues in [{2 / 3 * lo + 1 / 3:.3f}, {2 / 3 * hi + 1 / 3:.3f}]")


if __name__ == "__main__":
    main()
