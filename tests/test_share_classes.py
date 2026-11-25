"""One share class per company: the list, the drop, the hints, and the pack build step."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_analyzer.market_pack import (
    REASON_ILLIQUID,
    REASON_SHARE_CLASS,
    QualityRules,
    build_manifest,
    drop_second_share_classes,
    format_report,
)
from market_analyzer.share_classes import (
    NOT_SHARE_CLASSES,
    SHARE_CLASS_GROUPS,
    one_class_per_company,
    second_share_classes,
    share_class_candidates,
)


def test_the_list_has_no_ticker_twice():
    tickers = [ticker for group in SHARE_CLASS_GROUPS + NOT_SHARE_CLASSES for ticker in group]
    assert len(tickers) == len(set(tickers))
    assert all(len(group) >= 2 for group in SHARE_CLASS_GROUPS)


def test_first_present_class_is_kept():
    assert second_share_classes(["GOOG", "GOOGL", "AAPL", "NWS", "NWSA"]) == {"GOOG": "GOOGL", "NWS": "NWSA"}
    # The preferred class is missing, so the other one stays and nothing is dropped.
    assert second_share_classes(["GOOG", "AAPL"]) == {}
    assert second_share_classes(["AAPL", "MSFT"]) == {}


def test_one_class_per_company_drops_columns_only():
    frame = pd.DataFrame(np.ones((3, 4)), columns=["GOOGL", "GOOG", "AAPL", "BRK-A"])
    cleaned = one_class_per_company(frame)
    assert list(cleaned.columns) == ["GOOGL", "AAPL", "BRK-A"]
    assert cleaned.shape[0] == 3
    untouched = pd.DataFrame(np.ones((2, 2)), columns=["AAPL", "MSFT"])
    assert one_class_per_company(untouched) is untouched


def test_candidates_flag_new_name_matches_and_skip_known_ones():
    names = pd.Series(
        {
            "ACMA": "Acme Widgets, Inc. - Class A Common Stock",
            "ACMB": "Acme Widgets Inc. Class B",
            "XYZ-A": "Zed Corp",
            "XYZ-B": "Something Else Holdings",
            "GOOGL": "Alphabet Inc.",
            "GOOG": "Alphabet Inc. - Class C Capital Stock",
            "IBCP": "Independent Bank Corporation",
            "INDB": "Independent Bank Corp.",
            "AAPL": "Apple Inc.",
        }
    )
    found = share_class_candidates(names)
    assert ("ACMA", "ACMB") in found  # same name once class words go
    assert ("XYZ-A", "XYZ-B") in found  # same root before the hyphen
    assert all("GOOG" not in group for group in found)  # already on the list
    assert all("IBCP" not in group for group in found)  # checked: different banks
    assert all("AAPL" not in group for group in found)


def _checks(reasons: dict[str, str]) -> pd.DataFrame:
    table = pd.DataFrame({"reason": pd.Series(reasons)})
    table["eligible"] = table["reason"] == ""
    table["young"] = False
    table.index.name = "ticker"
    return table


def test_pack_step_drops_the_extra_class_among_passing_tickers():
    checks, extra = drop_second_share_classes(_checks({"GOOGL": "", "GOOG": "", "AAPL": "", "NWSA": REASON_ILLIQUID, "NWS": ""}))
    assert extra == {"GOOG": "GOOGL"}  # NWSA failed a check, so NWS stays as News Corp's class
    assert checks.loc["GOOG", "reason"] == REASON_SHARE_CLASS
    assert not checks.loc["GOOG", "eligible"] and not checks.loc["GOOG", "young"]
    assert checks.loc["NWS", "reason"] == "" and checks.loc["AAPL", "reason"] == ""
    assert checks.loc["NWSA", "reason"] == REASON_ILLIQUID


def test_manifest_and_report_name_the_dropped_classes():
    checks, extra = drop_second_share_classes(_checks({"GOOGL": "", "GOOG": "", "AAPL": ""}))
    kept = checks.index[checks["reason"] == ""]
    returns = pd.DataFrame(np.zeros((2, len(kept))), index=pd.date_range("2024-01-02", periods=2), columns=kept)
    meta = checks.loc[kept]
    manifest = build_manifest("2024-01-01", returns, meta, checks, QualityRules(), {}, 0.0, {}, [], extra, [("ACMA", "ACMB")])
    assert manifest["share_classes"] == {"GOOG": "GOOGL"}
    assert manifest["share_class_candidates"] == [["ACMA", "ACMB"]]
    assert manifest["history_counts"] == {REASON_SHARE_CLASS: 1}
    report = format_report(manifest, checks)
    assert "GOOGL    kept, GOOG dropped" in report
    assert "ACMA, ACMB" in report
    assert REASON_SHARE_CLASS in report
