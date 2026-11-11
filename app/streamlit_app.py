"""Portfolio Risk Lab: pick stocks, see the risk, see the mix, see what would have happened.

Run locally with

    .venv/bin/streamlit run app/streamlit_app.py

The page reads the market pack in ``data/pack`` (or the folder named by the
``MARKET_PACK_DIR`` environment variable) and calls
:mod:`market_analyzer.basket` for every number it shows. Nothing here
forecasts returns.
"""

from __future__ import annotations

import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from market_analyzer.basket import (  # noqa: E402
    CLEANINGS,
    RULES,
    BasketAnalysis,
    BasketBacktest,
    analyse_basket,
    backtest_basket,
)
from market_analyzer.market_pack import MarketPack, fetch_pack, load_pack  # noqa: E402

PACK_DIR = Path(os.environ.get("MARKET_PACK_DIR", str(ROOT / "data" / "pack")))
PACK_CACHE = Path(os.environ.get("MARKET_PACK_CACHE", str(Path.home() / ".cache" / "market_pack")))
PACK_REFRESH_SECONDS = 6 * 3600
MAX_BASKET = 60
WINDOWS = {"4 years": 1000, "2 years": 500}
BACKTEST_WINDOW = 500
BACKTEST_YEARS = 3.0
PRESETS = {
    "Big tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "ADBE"],
    "Blue chips": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "JPM", "XOM", "HD", "UNH", "V", "BRK-B", "COST", "PFE", "CVX"],
    "Mixed 20": ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "NEE", "CAT", "DIS", "NKE", "AMGN", "UNP", "T", "LMT", "SBUX", "MMM", "SPGI", "GS", "DE", "ADP"],
    "Banks and energy": ["JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "SLB", "EOG", "OXY"],
}
OPTIMISED_LABEL = "Optimised mix"
EQUAL_LABEL = "Equal split"
COLOURS = {OPTIMISED_LABEL: "#2563EB", EQUAL_LABEL: "#9CA3AF"}


# --------------------------------------------------------------------------- data


def pack_url() -> str:
    """Where a fresher pack may be published: the environment first, then Streamlit secrets."""
    url = os.environ.get("MARKET_PACK_URL", "")
    if not url:
        try:
            url = str(st.secrets.get("MARKET_PACK_URL", ""))
        except Exception:  # no secrets file is the normal local case
            url = ""
    return url.strip()


@st.cache_resource(ttl=PACK_REFRESH_SECONDS, show_spinner="Loading the market data…")
def get_pack() -> MarketPack:
    """The committed pack, or a fresher one from ``MARKET_PACK_URL`` when that works."""
    url = pack_url()
    if url:
        try:
            return load_pack(fetch_pack(url, PACK_CACHE))
        except Exception as exc:  # the committed pack is the fallback
            print(f"market pack: could not fetch {url} ({exc}); using {PACK_DIR}", flush=True)
    return load_pack(PACK_DIR)


@st.cache_data(show_spinner=False)
def run_analysis(tickers: tuple[str, ...], rule: str, cleaning: str, window: int) -> BasketAnalysis:
    return analyse_basket(get_pack(), list(tickers), rule, cleaning, window)


@st.cache_data(show_spinner=False)
def run_history(tickers: tuple[str, ...], rule: str, cleaning: str) -> BasketBacktest:
    return backtest_basket(get_pack(), list(tickers), rule, cleaning, BACKTEST_WINDOW, BACKTEST_YEARS)


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def day(stamp: pd.Timestamp) -> str:
    return pd.Timestamp(stamp).strftime("%d %b %Y")


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Lab", page_icon="📉", layout="wide")
    pack = get_pack()
    labels = ticker_labels(pack)
    options = sorted(labels)

    if "basket" not in st.session_state:
        from_url = [t.strip().upper() for t in str(st.query_params.get("basket", "")).split(",") if t.strip()]
        initial = [t for t in from_url if t in labels] or [t for t in PRESETS["Big tech"] if t in labels]
        st.session_state["basket"] = initial

    rule, cleaning, window = sidebar(pack)

    st.title("Portfolio Risk Lab")
    st.write(
        "Pick some stocks. The page shows how much an equal split of them would swing, how much a "
        "minimum-variance mix is forecast to swing, how that mix spreads the money, and what would have "
        "happened over the last three years. It makes no forecast of returns."
    )

    st.subheader("1. Pick your stocks")
    columns = st.columns(len(PRESETS) + 1)
    for column, (name, tickers) in zip(columns, PRESETS.items()):
        if column.button(name, help="Replace the basket with this preset"):
            st.session_state["basket"] = [t for t in tickers if t in labels]
            st.rerun()
    if columns[-1].button("Clear"):
        st.session_state["basket"] = []
        st.rerun()
    selected = st.multiselect(
        "Search by ticker or company name",
        options,
        format_func=lambda ticker: labels.get(ticker, ticker),
        key="basket",
        max_selections=MAX_BASKET,
        placeholder="Type a ticker or a name",
    )
    st.query_params["basket"] = ",".join(selected)
    if len(selected) < 2:
        st.info("Pick at least two stocks to see a risk picture.")
        return

    try:
        analysis = run_analysis(tuple(selected), rule, cleaning, window)
    except ValueError as exc:
        st.warning(str(exc))
        return

    section_risk(analysis)
    section_mix(analysis)
    section_history(analysis.tickers, rule, cleaning)
    section_limits()


def ticker_labels(pack: MarketPack) -> dict[str, str]:
    meta = pack.meta
    labels: dict[str, str] = {}
    for ticker in pack.tickers:
        if ticker in meta.index:
            row = meta.loc[ticker]
            text = f"{ticker} · {row.get('name', ticker)}"
            sector = str(row.get("sector", ""))
            if sector and sector != "Unknown":
                text += f" · {sector}"
            if bool(row.get("young", False)):
                text += " · young"
        else:
            text = ticker
        labels[ticker] = text
    return labels


def sidebar(pack: MarketPack) -> tuple[str, str, int]:
    with st.sidebar:
        st.header("Settings")
        rule = st.radio(
            "Portfolio rule",
            list(RULES),
            format_func=lambda key: RULES[key].label,
            help="What the optimiser may do with the weights.",
        )
        st.caption(RULES[rule].note)
        cleaning = st.selectbox(
            "Correlation cleaning",
            list(CLEANINGS),
            format_func=lambda key: CLEANINGS[key],
            help="How the noise is removed from the correlation matrix before optimising. "
            "With a handful of stocks it hardly matters; with hundreds it matters a lot.",
        )
        window_label = st.radio("History used for the risk model", list(WINDOWS), horizontal=True)
        window = WINDOWS[window_label]
        st.divider()
        manifest = pack.manifest
        st.caption(
            f"Market data: {manifest.get('n_tickers', len(pack.tickers))} US stocks, daily prices through "
            f"{manifest.get('last_date', day(pack.last_date))}. {manifest.get('n_young', 0)} are too young to optimise."
        )
        with st.expander("What this page is"):
            st.write(
                "A minimum-variance optimiser built on random-matrix cleaning of the correlation matrix, "
                "tested on the S&P 500 from 2018 to 2026. It answers one question: how to spread money across "
                "the stocks you chose so the portfolio swings as little as possible. It does not predict returns, "
                "and it cannot see crashes coming."
            )
    return rule, cleaning, window


def section_risk(analysis: BasketAnalysis) -> None:
    st.subheader("2. How much this basket swings")
    low, high = analysis.honesty
    left, middle, right = st.columns(3)
    left.metric(
        "Equal split, forecast swings per year",
        pct(analysis.equal_vol),
        help="Annualised volatility the risk model expects for an equal-money split.",
    )
    middle.metric(
        "Optimised mix, forecast swings per year",
        pct(analysis.optimised_vol),
        delta=f"{-analysis.reduction * 100:.0f}% vs equal split",
        delta_color="inverse",
        help="Annualised volatility the risk model expects for the minimum-variance mix.",
    )
    right.metric(
        "Effective number of stocks",
        f"{analysis.stats['effective_n']:.1f}",
        help="One over the sum of squared weights: how many stocks the mix really leans on.",
    )
    st.write(
        f"Between {day(analysis.start)} and {day(analysis.end)}, an equal split of these {len(analysis.tickers)} "
        f"stocks actually swung **{pct(analysis.equal_realised_vol)}** a year. The risk model, built on the same "
        f"{analysis.window} sessions, forecasts **{pct(analysis.equal_vol)}** for the equal split and "
        f"**{pct(analysis.optimised_vol)}** for its own mix. In our S&P 500 tests with this rule, real swings "
        f"came in at **{low:.0%} to {high:.0%}** of the forecast; section 4 measures it for this basket."
    )
    if analysis.excluded:
        st.caption(
            "Not optimised: " + "; ".join(f"{ticker} ({reason})" for ticker, reason in analysis.excluded.items())
        )


def section_mix(analysis: BasketAnalysis) -> None:
    st.subheader("3. How the optimiser spreads the money")
    table = analysis.table
    long_form = pd.concat(
        [
            pd.DataFrame({"ticker": table["ticker"], "series": OPTIMISED_LABEL, "weight": table["weight"]}),
            pd.DataFrame({"ticker": table["ticker"], "series": EQUAL_LABEL, "weight": table["equal_weight"]}),
        ]
    )
    order = table["ticker"].tolist()
    chart = (
        alt.Chart(long_form)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("ticker:N", sort=order, title=None),
            x=alt.X("weight:Q", axis=alt.Axis(format="%"), title="Share of the money"),
            yOffset=alt.YOffset("series:N", sort=list(COLOURS)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("ticker:N", title="Stock"),
                alt.Tooltip("series:N", title="Mix"),
                alt.Tooltip("weight:Q", title="Weight", format=".1%"),
            ],
        )
        .properties(height=max(220, 26 * len(order)))
    )
    st.altair_chart(chart, width="stretch")

    stats = analysis.stats
    facts = [f"{int(stats['n_positions'])} of {len(analysis.tickers)} stocks held", f"largest weight {pct(stats['max_weight'])}"]
    if stats["n_short"] > 0:
        facts.append(f"{int(stats['n_short'])} shorted, gross exposure {stats['gross_leverage']:.2f}")
    st.caption(" · ".join(facts))

    shown = pd.DataFrame(
        {
            "Stock": table["ticker"],
            "Name": table["name"],
            "Sector": table["sector"],
            "Weight": table["weight"].map(pct),
            "Equal": table["equal_weight"].map(pct),
            "Own swings": table["own_vol"].map(pct),
            "Correlation with the rest": table["avg_corr"].map(lambda value: f"{value:.2f}"),
            "Share of the risk": table["risk_share"].map(lambda value: pct(value, 0)),
            "Why": table["reason"],
        }
    )
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(
        "Own swings: the stock's own annualised volatility. Correlation with the rest: its average cleaned "
        "correlation with the other stocks. Share of the risk: how much of the mix's variance it accounts for."
    )


def section_history(tickers: list[str], rule: str, cleaning: str) -> None:
    st.subheader("4. What would have happened")
    st.write(
        f"Every 21 sessions over the last {BACKTEST_YEARS:.0f} years the mix is re-optimised on the previous "
        f"{BACKTEST_WINDOW} sessions and held. The forecast made on each decision day is then compared with "
        "the swings that followed."
    )
    try:
        with st.spinner("Re-running the last three years…"):
            history = run_history(tuple(tickers), rule, cleaning)
    except ValueError as exc:
        st.info(f"Not enough history for the backtest: {exc}")
        return
    summary = history.summary
    equal = summary.loc["equal_weight"]
    mix = summary.loc[history.strategy]
    honesty = float(mix["median_vol_ratio"])

    first, second, third, fourth = st.columns(4)
    first.metric("Equal split, realised swings", pct(float(equal["ann_volatility"])))
    second.metric(
        "Optimised mix, realised swings",
        pct(float(mix["ann_volatility"])),
        delta=f"{(float(mix['ann_volatility']) / float(equal['ann_volatility']) - 1) * 100:.0f}% vs equal split",
        delta_color="inverse",
    )
    third.metric(
        "Honesty of the forecast",
        f"{honesty:.2f}",
        help="Median of realised over forecast swings across the holding periods. 1.00 means the forecast was right on average; above 1 means it ran optimistic.",
    )
    fourth.metric(
        "Worst fall from a peak",
        f"{pct(float(mix['max_drawdown']))} vs {pct(float(equal['max_drawdown']))}",
        help="Optimised mix against the equal split.",
    )

    growth = history.growth.rename(columns={"equal_weight": EQUAL_LABEL, history.strategy: OPTIMISED_LABEL})
    long_form = growth.reset_index().melt(id_vars=growth.index.name or "Date", var_name="series", value_name="value")
    long_form.columns = ["date", "series", "value"]
    chart = (
        alt.Chart(long_form)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title="Growth of $100", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Mix"),
                alt.Tooltip("value:Q", title="$100 became", format=".1f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

    lines = [
        f"From {day(history.start)} to {day(history.end)} the equal split returned {pct(float(equal['ann_return']))} "
        f"a year and the optimised mix {pct(float(mix['ann_return']))} a year. Past returns say little about "
        "future ones; the swings are the part the model is built to get right."
    ]
    if history.worst:
        worst = history.worst
        lines.append(
            f"The worst forecast miss was the period starting {day(worst['hold_start'])}: forecast "
            f"{pct(worst['predicted_vol'])}, realised {pct(worst['realised_vol'])}, "
            f"{worst['vol_ratio']:.1f} times the forecast. No risk model sees a shock coming."
        )
    lines.append(
        f"Average turnover per re-optimisation was {float(mix['mean_turnover']):.2f} of the portfolio; "
        "trading costs are not included."
    )
    st.write(" ".join(lines))


def section_limits() -> None:
    st.subheader("5. Read this before using it")
    st.markdown(
        """
- **No view on returns.** The optimiser only knows how stocks have moved together. A stock with a
  low weight is not a bad investment; it just adds swings to this particular basket.
- **The past is the only input.** Correlations drift over the years and break in shocks. The S&P 500
  tests show the forecast holding up in calm years and missing by a factor of two to twelve in crashes.
- **Few stocks, little cleaning.** With a handful of names the random-matrix cleaning changes little;
  the rule (long only or not) matters more. The cleaning earns its keep on hundreds of stocks.
- **No costs, no taxes, no borrowing.** The backtest trades for free and the "no limits" rule assumes
  shorting and borrowing at no cost, which no retail account gets.
- **Educational.** Prices come from Yahoo Finance and can contain errors. Nothing here is investment advice.
"""
    )


if __name__ == "__main__":
    main()
