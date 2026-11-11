# Getting stock data for any company

This project downloads daily prices for whatever symbols you give it and turns those prices into a return matrix. The symbol decides the company. A US large cap, an Indian single stock, a London listing, an ETF, and an index are the same kind of request.

Yahoo Finance is the data source, through the `yfinance` library. It is an unofficial interface: it is the right tool for research and a market analyzer, and it is not a brokerage feed. Prices can be revised, and a symbol that does not match Yahoo's spelling comes back empty.

## 1. Find the Yahoo symbol

1. Open [finance.yahoo.com](https://finance.yahoo.com).
2. Search the company name, for example `Reliance Industries` or `HSBC`.
3. Open the quote page. The symbol in the page title and in the URL is the one to copy.

`https://finance.yahoo.com/quote/RELIANCE.NS` means the symbol is `RELIANCE.NS`. Copy it exactly, including the suffix after the dot or hyphen.

Search by company name when you are unsure of the suffix. Guessing `RELIANCE` without `.NS` often hits a different listing, or no listing.

## 2. Exchange suffixes

Yahoo encodes the exchange in the symbol. US listings usually have no suffix. Everywhere else, the suffix is part of the identifier.

| Market | What to add | Example |
| --- | --- | --- |
| United States | nothing | `AAPL`, `MSFT`, `SPY` |
| US class share | a hyphen, not a dot | `BRK-B`, `BF-B` |
| India, NSE | `.NS` | `RELIANCE.NS`, `TCS.NS` |
| India, BSE | `.BO` | `RELIANCE.BO` |
| London | `.L` | `HSBA.L`, `SHEL.L` |
| Toronto | `.TO` | `RY.TO` |
| Tokyo | `.T` | `7203.T` |
| Hong Kong | `.HK` | `0700.HK` |
| Australia | `.AX` | `BHP.AX` |
| Germany, Xetra | `.DE` | `SAP.DE` |
| Paris | `.PA` | `MC.PA` |
| Index | a leading `^` | `^GSPC`, `^NSEI`, `^FTSE` |
| FX pair | `=X` | `EURUSD=X` |
| Crypto | `-USD` | `BTC-USD` |

An ETF uses the same form as a stock on that exchange: `SPY` in the US, `NIFTYBEES.NS` in India.

Two spellings that are easy to mix up:

- `BRK.B` is how Wikipedia writes Berkshire Hathaway Class B. Yahoo's symbol is `BRK-B`.
- `RELIANCE.NS` already is a Yahoo symbol. The dot is the exchange suffix. Do not change it to a hyphen.

## 3. Request the prices

You need a symbol, a start date, and optionally an end date.

- `start` is the first session you want, inclusive.
- `end` is exclusive. `end="2025-01-01"` keeps data through 2024-12-31.
- Omit `end` when the basket should run through the latest session Yahoo has published.

### From the command line

One or more companies:

```bash
market-analyzer pull \
  --tickers AAPL RELIANCE.NS HSBA.L \
  --start 2020-01-01 \
  --end 2025-01-01 \
  --name global-sample
```

A list you keep in a file: copy `config/universe.example.yaml`, change the symbols, then run:

```bash
market-analyzer pull --universe config/my-universe.yaml
```

```yaml
name: my-universe
start: "2020-01-01"
# end: "2025-01-01"   # omit this line to stay current

tickers:
  - "AAPL"
  - "RELIANCE.NS"
  - "HSBA.L"
```

Quote every symbol. In YAML the bare words `ON`, `NO`, `YES`, and `OFF` are booleans, and `ON` is a real US ticker (ON Semiconductor). Write `"ON"`, not `ON`.

### From Python

```python
from market_analyzer import build_market_data

data = build_market_data(
    tickers=["AAPL", "RELIANCE.NS", "HSBA.L"],
    start="2020-01-01",
    end="2025-01-01",
)

data.prices    # adjusted close, date × ticker
data.returns   # daily simple returns, shape (T, N)
data.missing_tickers
```

`examples/pull_custom_basket.py` is the same call for a short window, written out as a script.

### What the library asks Yahoo for

The three steps below are the whole download. The package runs them for whatever list you pass in; the S&P 500 page is not involved unless you choose that page yourself (see below).

```python
import yfinance as yf

tickers = ["AAPL", "RELIANCE.NS", "HSBA.L"]  # your list

raw = yf.download(
    tickers,
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False,   # keep the Adj Close column
    progress=False,
)
prices = raw["Adj Close"]
returns = prices.pct_change().dropna(how="any")
```

`auto_adjust=False` matters. Current `yfinance` turns adjustment on by default, folds dividends and splits into `Close`, and then omits `Adj Close`. This project turns that default off and reads `Adj Close`, which is the split- and dividend-adjusted price.

A single symbol comes back with ordinary columns (`Open`, `High`, `Close`, `Adj Close`, …) rather than a table with one column per company. The package turns that into a one-column price frame so the rest of the pipeline stays the same. The snippet above assumes more than one symbol.

## 4. Adjusted close

Adjusted close is the price series to use for returns. A split or a dividend changes the raw close even when the shareholder's economic position did not change by that amount. Yahoo's adjusted close rewrites the history so a return across a split is the market move, not the split ratio.

Use the raw close only when you explicitly want the printed price, for example to compare with an old report. For a return matrix, stay with adjusted close.

## 5. The return matrix

Daily simple return on date *t* for company *i* is:

```text
R[t, i] = Price[t, i] / Price[t - 1, i] - 1
```

`returns.csv` is that matrix.

- Each row is a trading date. There are **T** rows.
- Each column is one company. There are **N** columns.
- The first price date has no return, because there is no previous close. It is dropped.
- Values are simple returns, not log returns. `0.01` means up 1 percent that day.

Log returns, if a later study wants them, are `numpy.log1p(data.returns)`. Build them from this matrix rather than downloading again.

### Which dates survive

Companies on different exchanges do not share a calendar. A US holiday can be a normal session in India, and the reverse.

`--align intersection` (the default) keeps a date only when **every** symbol has a return that day. The matrix is dense: no empty cells. That is the `T × N` block most cross-sectional work wants. The cost is that a global basket loses every date that was a holiday anywhere in the basket.

`--align union` keeps a date when **at least one** symbol traded. Other cells stay empty. Use this when you want each company's own history and you are ready to handle missing values.

The command prints a note when intersection throws away more than about 20 percent of the dates. That usually means the basket mixes calendars.

A symbol with fewer than two prices cannot form a return. It is reported as missing and left out of the matrix. `--strict` turns that into an error so a typo does not silently shrink the basket.

## 6. The summary files

`summary.csv` is computed on the same dates as `returns.csv`.

| Column | Meaning |
| --- | --- |
| `observations` | Number of daily returns used |
| `total_return` | Compounded simple return over those dates |
| `ann_return` | Geometric annualized return, using `--periods-per-year` (default 252) |
| `ann_volatility` | Sample standard deviation of daily returns, times `sqrt(252)` |
| `mean_daily` | Arithmetic mean of daily returns |
| `vol_daily` | Sample standard deviation of daily returns |
| `max_drawdown` | Worst peak-to-trough loss on the compounded path, inside this sample |

`correlation.csv` is the Pearson correlation of the columns. `covariance.csv` is the matching sample covariance. With one company the correlation is a 1×1 table, and the noise-cleaning step is skipped because it needs at least two stocks.

When the return matrix has no missing cells and every column varies, the same folder also gets `eigenvalues.csv`, `cleaned_correlation.csv`, `cleaned_covariance.csv`, and `signal_loadings.csv`. Those come from clipping correlation eigenvalues that sit inside the Marchenko-Pastur noise band. Use `cleaned_covariance.csv` as the risk matrix for later portfolio weights. A return panel with gaps (`--align union`) still gets the raw correlation and covariance, and the command says why the cleaned files were skipped.

Annualized return and volatility scale a short sample up to a year. A few sessions of ordinary moves can print a very large `ann_return`. For a window of days or weeks, read `total_return` and `vol_daily`. Use the annualized columns once the sample covers many months.

252 is the equity convention for the number of sessions in a year. It is a knob, not a fact about every market. Pass `--periods-per-year` when you want a different scale. The daily returns themselves do not change.

## 7. Optional: symbols from an HTML table

Typing the symbols is the reliable path, because you can check each one on Yahoo. When you do want a published constituent list, any HTML table that `pandas.read_html` can parse works the same way. Point `source` at the page and name the column that holds the ticker.

```yaml
name: from-html-table
start: "2020-01-01"

source:
  type: html_table
  url: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
  table_index: 0
  column: Symbol
  wikipedia_class_shares: true
```

`config/html_table.example.yaml` is that file. Swap the URL for another list, for example the Wikipedia page of an index you actually want, and set `column` to whatever that page calls the ticker. `wikipedia_class_shares: true` rewrites `BRK.B` to `BRK-B`. Turn it off when the table already contains Yahoo symbols such as `RELIANCE.NS`.

Constituent pages describe **today's** membership. Pulling those symbols back to 2020 does not recreate the historical index: companies that left the index are absent, and companies that joined recently appear for the whole window. For a study of specific companies, type those symbols. For a historical index, you need a membership list as of each date, which this downloader does not build.

After a table pull, open `missing_tickers` in the output (or the note printed by the command). Index pages often omit the Yahoo suffix, and those rows will be empty until you fix the spelling.

## 8. When a symbol comes back empty

Check these in order:

1. Open the quote page and compare the symbol character for character.
2. Confirm the suffix (`.NS`, `.L`, `.TO`, …).
3. Confirm the company was listed during your window. A start date before the IPO produces a shorter series, which is fine; a symbol that never traded produces nothing.
4. Class shares use a hyphen on Yahoo (`BRK-B`).
5. Delisted names often disappear from Yahoo. There is no repair for that in this project.

The other symbols are still saved. `--strict` refuses to write a partial basket.

## 9. A worked basket

This is a complete, non-index request: one US name, one Indian name, one UK name, from 2024-01-01 through 2024-03-31.

```bash
market-analyzer pull \
  --tickers AAPL RELIANCE.NS HSBA.L \
  --start 2024-01-01 \
  --end 2024-04-01 \
  --name worked-example
```

You should get:

- `data/worked-example/prices.csv` with three price columns
- `data/worked-example/returns.csv` with three return columns
- a printed shape `T dates × 3 tickers`
- a note about shared dates if the three calendars barely overlap, in which case rerun with `--align union` or keep the intersection on purpose

That return matrix is the input to whatever you build next: correlation, volatility, or a later cross-sectional study. The downloader stops once the matrix and the summary tables are on disk.
