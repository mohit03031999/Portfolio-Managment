"""The web page end to end, run headlessly on the committed market pack."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "streamlit_app.py"
PACK = ROOT / "data" / "pack"

pytestmark = pytest.mark.skipif(not (PACK / "returns.parquet").is_file(), reason="the market pack is not in data/pack")


def _metrics(page: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in page.metric}


def _percent(text: str) -> float:
    return float(text.rstrip("%")) / 100.0


def test_the_page_renders_and_the_dial_follows_the_slider(monkeypatch):
    monkeypatch.setenv("MARKET_PACK_URL", "none")
    monkeypatch.setenv("MARKET_PACK_DIR", str(PACK))
    page = AppTest.from_file(str(PAGE), default_timeout=300).run()
    assert not page.exception
    titles = [heading.value for heading in page.subheader]
    assert [title.split(".")[0] for title in titles] == ["1", "2", "3", "4", "5", "6"]
    assert "How much of your money goes in" in titles[3]

    slider = page.slider[0]
    assert slider.label == "Yearly swings you accept"
    first = _metrics(page)
    in_mix, in_cash = _percent(first["In the stock mix"]), _percent(first["In cash"])
    assert 0.0 < in_mix <= 1.0 and in_mix + in_cash == pytest.approx(1.0, abs=0.011)
    assert first["Swings you asked for"] == f"{slider.value}%"
    assert _percent(first["Swings you would have got"]) > 0
    split = page.dataframe[1].value
    assert split.iloc[0]["Holding"] == "Cash"
    assert len(split) >= 3

    slider.set_value(slider.min).run()
    assert not page.exception
    cautious = _metrics(page)
    assert _percent(cautious["In the stock mix"]) < in_mix
    assert _percent(cautious["Swings you would have got"]) < _percent(first["Swings you would have got"])

    slider.set_value(slider.max).run()
    assert not page.exception
    bold = _metrics(page)
    assert _percent(bold["In the stock mix"]) >= in_mix
    assert _percent(bold["Swings to expect per year"]) <= slider.max / 100.0 + 1e-9
