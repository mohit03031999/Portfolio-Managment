"""Pull a basket you name yourself.

Run from the repository root after installing the package:

    python examples/pull_custom_basket.py
"""

from market_analyzer import build_market_data
from market_analyzer.analyzer import summarize
from market_analyzer.pipeline import save_market_data


def main() -> None:
    data = build_market_data(
        tickers=["AAPL", "MSFT", "RELIANCE.NS"],
        start="2024-01-01",
        end="2024-04-01",
        name="three-names",
    )
    print(f"Return matrix shape (T x N): {data.shape}")
    print(summarize(data.returns).to_string(float_format=lambda value: f"{value:.4f}"))
    save_market_data(data, "data/three-names")


if __name__ == "__main__":
    main()
