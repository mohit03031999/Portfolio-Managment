# The web page: Portfolio Risk Lab

## 0. In one paragraph

A Streamlit page where someone picks stocks from the US market and sees, for that basket, how much an equal split would swing, how much the minimum-variance mix is forecast to swing, how that mix spreads the money and why, and what would have happened over the last three years with monthly re-optimisation. Every number comes from the same `market_analyzer` code that produced the S&P 500 studies; the page adds no maths of its own. It makes no forecast of returns.

## 1. What the page shows

1. **Pick your stocks.** A search box over 3,368 US common stocks (NYSE, NYSE American, NASDAQ, Cboe BZX), four preset baskets, and a limit of sixty names. The basket is written into the page address, so a link shares a basket with no database.
2. **How much this basket swings.** Three tiles: the forecast annual volatility of an equal split, the forecast for the optimised mix with its reduction, and the effective number of stocks the mix leans on. One paragraph gives the window, what the equal split actually did over it, and the honesty band measured on the S&P 500 for the chosen rule.
3. **How the optimiser spreads the money.** A grouped bar chart of optimised against equal weights, a line of concentration facts, and a table with each stock's own volatility, its average cleaned correlation with the rest, its share of the mix's risk, and a plain-words reason for its weight.
4. **What would have happened.** The rolling backtest from `market_analyzer.backtest` on the basket: every 21 sessions over the last three years the mix is re-optimised on the previous 500 sessions and held. Four tiles (realised volatility of each book, the honesty ratio, worst drawdowns), the growth of 100 dollars, the returns, the worst forecast miss, and the turnover.
5. **Read this before using it.** No view on returns, the past is the only input, few stocks means little cleaning, no costs, educational.

The sidebar holds the three settings: the portfolio rule (long only; gross exposure capped at 2; no limits), the correlation cleaning (RIE; clipping; plain sample), and the history window for the risk model (four or two years).

## 2. Architecture

| piece | where | what it does |
|---|---|---|
| universe | `src/market_analyzer/market_universe.py` | symbol files and screener in, candidate table out, with the count each rule removed |
| pack builder | `src/market_analyzer/market_pack.py` | batched Yahoo download, history checks, pack files, build report, pack fetch |
| build script | `examples/build_market_pack.py` | runs the two above into `data/pack/`; raw downloads cached in `data/raw/` |
| engine | `src/market_analyzer/basket.py` | `analyse_basket` and `backtest_basket`: the numbers the page shows |
| page | `app/streamlit_app.py` | layout, settings, charts, text; caches the pack and every result |
| refresh | `.github/workflows/refresh-pack.yml` | nightly rebuild, published as the `pack-latest` GitHub release |

**Why Streamlit and not a static page.** A static page needs the optimiser rewritten in JavaScript, and two implementations of the same maths drift apart. Streamlit runs the tested Python directly, including the backtest engine and the honesty numbers, and each click costs a few hundred milliseconds. The price is a server process: Streamlit Community Cloud runs it free and puts it to sleep after days without visitors, so the first visitor then waits about half a minute.

**Caching.** The pack loads once per process (`st.cache_resource`, refreshed every six hours when a pack URL is set). Each analysis and each backtest is cached by basket, rule, cleaning and window (`st.cache_data`), so flipping a setting back is instant.

## 3. The market pack

**Sources.** Two public NASDAQ Trader files list every security on a US exchange (`nasdaqlisted.txt`, `otherlisted.txt`); the NASDAQ stock screener gives sector, industry, market cap, country and IPO year in one download; Yahoo Finance gives daily adjusted closes and volumes through `yfinance`.

**Listing rules**, applied in order, with the counts of the first build (11 November 2026):

| rule | removed |
|---|---|
| listed securities | 13,282 |
| exchange-traded fund or note | 5,790 |
| symbol suffix or characters (units, warrants, rights, preferreds) | 1,162 |
| market cap below 100 million dollars | 1,068 |
| name says warrant, unit, preferred, note, fund, SPAC, structured product | 875 |
| delinquent, deficient or halted | 347 |
| blank-cheque company or trust industry | 63 |
| closed-end fund (Finance sector, fund vocabulary in the name) | 56 |
| test issue | 35 |
| zero market value in the screener | 31 |
| exchange other than NYSE, NYSE American, NASDAQ, NYSE Arca, Cboe BZX | 12 |
| **candidates** | **3,843** |

Class shares are written the Yahoo way (`BRK-B`). A name the screener does not know keeps a missing market cap and stays in, so the liquidity check decides it. Business development companies that NASDAQ labels "Closed End Fund" (Ares Capital, for one) follow that label and are out. Every S&P 500 member in our October 2025 list survives except Kenvue, which is no longer listed.

**History rules** on the downloaded prices (`QualityRules`):

| rule | default | removed |
|---|---|---|
| no prices from Yahoo | | 6 |
| stopped trading | last price more than 10 sessions before the end | 0 |
| price below the floor | 1 dollar | 35 |
| too thinly traded | median dollar volume over the last 126 sessions below 1 million | 375 |
| too many missing days | more than 5% of the last 504 sessions | 0 |
| a single-day move beyond the limit | 250% | 59 |
| **in the pack** | | **3,368** |
| eligible for the risk model | at least three years of history | 3,040 |
| young | under three years; listed but not optimised | 328 |

The single-day limit catches data errors and one-off events alike (a biotech that quintupled on trial news goes too); a day like that would dominate any covariance estimate.

**Files.** `returns.parquet` (2,226 sessions from 3 January 2018, float32 simple returns, 34 MB), `meta.parquet` (facts and checks per ticker), `manifest.json` (build time, rules, counts, sources), `report.txt` (the table above in words). The committed pack is the fallback the page always has; a fresher one is fetched from `MARKET_PACK_URL` when set.

**Rebuilding.** `python examples/build_market_pack.py` does the whole thing in about five minutes (3,843 tickers in 39 batches of 100). `--limit 50` is a trial, `--skip-download` re-runs the checks on the cached raw files, `--min-market-cap` moves the floor.

## 4. The engine

`analyse_basket(pack, tickers, rule, cleaning, window)` takes the last `window` sessions, keeps the tickers with a complete, non-constant history there (the rest are reported with a reason, "under three years of history" for young names), estimates the covariance with the chosen cleaning from `ESTIMATORS`, solves the chosen rule with `gross_capped_minimum_variance_weights` (cap 1 is long only, cap 2 the capped book, infinity unconstrained), and reports both books' forecast volatility, the equal split's realised volatility over the window, the weights, the concentration statistics, and a per-stock table.

The **reason** for each weight comes from two ranks inside the basket, the stock's own volatility and its average cleaned correlation with the others: a big weight is "calm", "moves on its own", or both; a zero weight is "jumpy and moves with the rest", "moves with the rest, so the others already cover it", or "jumpier than the stocks that play the same role"; a short is "it moves so closely with others that betting against it offsets their swings".

The **honesty band** per rule is the one measured on the S&P 500 at 1,000 and 250 days: long only 0.85 to 1.02, cap 2 about 0.95 to 1.10, unconstrained 1.25 to 1.35. It is a prior; section 4 of the page measures the ratio for the basket itself.

`backtest_basket(pack, tickers, rule, cleaning, window=500, years=3, holding=21)` runs `run_backtest` with the equal-weight book and the chosen strategy (`longonly_rie`, `gross2_rie`, `minvar_rie`, or the same with `clipped` or `sample`) on the last `window + 3 × 252` sessions, and returns the summary, the growth of 100 dollars, and the holding period with the worst forecast miss. A young name joins the backtest as soon as its window is complete.

## 5. Running and testing locally

```
uv pip install --python .venv/bin/python -e ".[app]"
.venv/bin/streamlit run app/streamlit_app.py
```

`MARKET_PACK_DIR` points the page at another pack folder. The tests cover the listing rules on hand-made symbol files, the history checks on synthetic prices, the pack round trip and fetch, and the engine on a synthetic factor market (`tests/test_market_universe.py`, `tests/test_market_pack.py`, `tests/test_basket.py`). The page itself is exercised headlessly with Streamlit's `AppTest`; the whole journey renders in under two seconds, and each preset, rule, cleaning and window change was checked that way before deployment.

## 6. Deployment

1. Push the repository to GitHub (public). `data/pack` is committed; `data/raw`, `output` and the venv are ignored.
2. On share.streamlit.io, sign in with GitHub, choose the repository, branch `main`, main file `app/streamlit_app.py`, Python 3.12. `requirements.txt` already lists Streamlit, pyarrow and requests next to the package's own needs.
3. Nightly data needs no further setup. The workflow runs after each US session (02:30 UTC, Tuesday to Saturday) and whenever its own file changes, and publishes the pack as the `pack-latest` release of the repository. The page looks for that release by default (`DEFAULT_PACK_URL` in the app), fetches it at most every six hours, and falls back to the committed pack when the fetch fails. `MARKET_PACK_URL`, as an environment variable or a Streamlit secret, points the page elsewhere; the value `none` keeps it on the committed pack.

**A caveat on the refresh.** Yahoo Finance sometimes rate-limits requests from cloud runners. The workflow retries each batch and the page keeps working from the committed pack regardless, but a failed nightly run only shows in the Actions tab. If that proves frequent, the fallback is to run the build script on a machine of ours and upload with `gh release upload pack-latest data/pack/* --clobber`.

## 7. What the presets showed on the first build

Risk model on four years, long only, RIE, prices through 10 November 2026; backtest over the last three years.

| basket | stocks | equal split forecast | optimised forecast | held | realised: equal / optimised | honesty | worst drawdown: optimised / equal |
|---|---|---|---|---|---|---|---|
| Big tech | 10 | 25.7% | 20.9% | 10 | 25.7% / 21.8% | 0.86 | −26.9% / −28.8% |
| Blue chips | 15 | 11.2% | 11.0% | 14 | 11.4% / 11.6% | 0.90 | −8.8% / −9.2% |
| Mixed 20 | 20 | 13.0% | 10.7% | 14 | 12.9% / 11.3% | 0.84 | −12.8% / −16.2% |
| Banks and energy | 12 | 19.7% | 17.9% | 6 | 19.4% / 18.1% | 0.83 | −18.2% / −22.5% |

Three things the small baskets teach that the S&P 500 did not. The optimiser's gain over an equal split depends on how different the stocks are: fifteen blue chips of similar risk gain nothing, twenty mixed names gain 13 to 18%. The cleaning changes nothing at this size (sample and RIE agree to the decimal on twelve stocks), so the rule is the only dial that matters. And the forecasts ran slightly pessimistic (honesty 0.81 to 0.90) because 2024 to 2026 was calmer than the 2022 bear market inside the estimation windows.

## 8. Limits and what comes next

Not in version 1: picking a basket from the whole market (the risk model needs far fewer stocks than sessions, so a pre-selection step comes first), sector exposure, saved portfolios beyond the link, volatility targeting, trading costs. The page shows what the research found and nothing it did not test: it has no return view, it cannot see shocks, and with a handful of stocks the random-matrix cleaning is almost idle.
