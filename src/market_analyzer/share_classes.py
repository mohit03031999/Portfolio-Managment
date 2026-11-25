"""One share class per company.

Some companies list two or more classes of the same stock, such as GOOGL
and GOOG for Alphabet. The classes differ in voting rights, not in the
business, so their prices move almost exactly together. Keeping both:

* counts one company twice, so a basket looks more spread out than it is;
* hands the optimiser a "buy one class, sell the other" bet that looks
  nearly riskless but only adds trading and borrowing;
* puts an eigenvalue near zero into the correlation spectrum. The RIE's
  inverse-Wishart lift (:mod:`market_analyzer.rie`) calibrates itself on
  the single smallest eigenvalue, so one such pair stops it from lifting
  the other quiet directions. On the S&P 500 at 1000 days, dropping GOOG
  and NWS took the lifted directions from about 20 to 119, turnover of the
  minimum-variance portfolio from 1.34 to 0.97 and leverage from 6.9 to 6.1.

``SHARE_CLASS_GROUPS`` lists every multi-class company found in the data
pack on 2026-11-25 by matching company names and checking that the classes
move together. The first ticker of each group is the one kept: the class
with the larger median dollar volume that day. A fixed choice keeps a
customer's saved basket stable from one nightly build to the next.
:func:`share_class_candidates` flags new name matches for a person to check.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

SHARE_CLASS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("GOOGL", "GOOG"),  # Alphabet
    ("Z", "ZG"),  # Zillow Group
    ("FOXA", "FOX"),  # Fox
    ("LLYVK", "LLYVA"),  # Liberty Live
    ("LEN", "LEN-B"),  # Lennar
    ("UAA", "UA"),  # Under Armour
    ("LBTYA", "LBTYK"),  # Liberty Global
    ("NWSA", "NWS"),  # News Corp
    ("CENTA", "CENT"),  # Central Garden & Pet
    ("BRK-B", "BRK-A"),  # Berkshire Hathaway
    ("FWONK", "FWONA"),  # Liberty Media, Formula One
    ("BATRK", "BATRA"),  # Atlanta Braves Holdings
    ("LILAK", "LILA"),  # Liberty Latin America
    ("GLIBK", "GLIBA"),  # Liberty Capital, GCI Group
    ("HEI", "HEI-A"),  # Heico
    ("BF-B", "BF-A"),  # Brown-Forman
    ("UHAL-B", "UHAL"),  # U-Haul Holding
    ("RUSHA", "RUSHB"),  # Rush Enterprises
    ("GEF", "GEF-B"),  # Greif
    ("BELFB", "BELFA"),  # Bel Fuse
    ("BH-A", "BH"),  # Biglari Holdings
)

# Name matches that are different companies, checked by hand on 2026-11-25.
NOT_SHARE_CLASSES: tuple[tuple[str, ...], ...] = (
    ("IBCP", "INDB"),  # Independent Bank Corporation (Michigan) and Independent Bank Corp. (Massachusetts)
    ("OBDC", "OWL"),  # Blue Owl Capital Corporation (a lending fund) and Blue Owl Capital Inc. (the manager)
    ("CBNK", "TBBK"),  # Capital Bancorp and The Bancorp
    ("GHC", "GHM"),  # Graham Holdings and Graham Corporation
    ("FBNC", "FBP"),  # First Bancorp (North Carolina) and First BanCorp. (Puerto Rico)
)

_CLASS_WORDS = re.compile(
    r"\b(class|series|common|capital|stock|shares?|ordinary|non ?voting|voting|limited|subordinate|sub|vot|"
    r"inc|incorporated|corp|corporation|co|company|holdings?|ltd|plc|llc|lp|the|nv|sa|new|[a-c]|1|2|3)\b"
)


def second_share_classes(tickers: Iterable[str], groups: tuple[tuple[str, ...], ...] = SHARE_CLASS_GROUPS) -> dict[str, str]:
    """Map every extra share class in ``tickers`` to the class kept for its company.

    The kept class is the first ticker of its group that is present, so a
    company whose preferred class is missing keeps whichever class is there.
    """
    present = set(tickers)
    extra: dict[str, str] = {}
    for group in groups:
        found = [ticker for ticker in group if ticker in present]
        for ticker in found[1:]:
            extra[ticker] = found[0]
    return extra


def one_class_per_company(frame: pd.DataFrame, groups: tuple[tuple[str, ...], ...] = SHARE_CLASS_GROUPS) -> pd.DataFrame:
    """``frame`` without the extra share classes among its columns."""
    extra = second_share_classes(frame.columns, groups)
    return frame.drop(columns=list(extra)) if extra else frame


def _company_key(name: str) -> str:
    text = re.sub(r"[^a-z0-9 ]", " ", str(name).lower().replace("&", " and "))
    return " ".join(_CLASS_WORDS.sub(" ", text).split())


def share_class_candidates(
    names: pd.Series,
    groups: tuple[tuple[str, ...], ...] = SHARE_CLASS_GROUPS,
    known_different: tuple[tuple[str, ...], ...] = NOT_SHARE_CLASSES,
) -> list[tuple[str, ...]]:
    """Tickers whose company names match once class words are removed and that no list covers yet.

    ``names`` maps ticker to company name. Tickers that share a root before a
    hyphen, such as ``BRK-A`` and ``BRK-B``, also count as a match. Each
    result is a hint for a person to check, not a verdict: two different
    companies can share a name.
    """
    covered = {frozenset(group) for group in groups} | {frozenset(group) for group in known_different}
    known = {ticker for group in groups for ticker in group}
    keys = pd.Series({ticker: _company_key(name) for ticker, name in names.items()})
    roots = pd.Series({ticker: str(ticker).split("-")[0] for ticker in names.index})
    found: set[tuple[str, ...]] = set()
    for series in (keys[keys != ""], roots):
        for _, members in series.groupby(series):
            tickers = tuple(sorted(members.index))
            if len(tickers) < 2 or frozenset(tickers) in covered or set(tickers) <= known:
                continue
            found.add(tickers)
    return sorted(found)
