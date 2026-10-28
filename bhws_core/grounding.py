"""Keep AI-written text tied to the numbers it was given.

The narrative and news-colour prompts in backend.py send figures to an LLM and
publish what comes back. Nothing checks the reply. This module extracts every
number in a reply and checks it against the facts supplied in the prompt.
Unsupported numbers, probability or win-rate language without a supplied
statistic, and price levels that were never provided are flagged. `sanitize`
drops the offending sentences rather than editing them, so what remains is
the model's own wording minus the unsupported claims.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

_NUM = re.compile(r"(?<![A-Za-z])[-+−]?\$?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:%|bps|bp|k|K|m|bn|pp)\b|%)?")
_PROB = re.compile(r"\b(win rate|hit rate|probabilit\w*|odds of|chance of|likely to (?:rise|fall)|"
                   r"\d+\s?% (?:chance|probability))\b", re.I)
_YEARISH = re.compile(r"^(19|20)\d\d$")


def _to_float(tok: str):
    t = tok.replace("−", "-").replace("$", "").replace(",", "").strip()
    t = re.sub(r"\s?(%|bps|bp|k|K|m|bn|pp)$", "", t)
    try:
        return float(t)
    except ValueError:
        return None


def _flatten(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for m in _NUM.finditer(obj):
            v = _to_float(m.group(0))
            if v is not None:
                yield v
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _flatten(v)


def _supported(v: float, allowed: Set[float]) -> bool:
    for a in allowed:
        if a == v or abs(a - v) <= max(0.051, 0.005 * abs(a)):
            return True
        # allow sign-free restatement ("down 0.4%" for -0.4)
        if abs(abs(a) - abs(v)) <= max(0.051, 0.005 * abs(a)):
            return True
    return False


def check(text: str, facts: Dict[str, Any], *, allow_small_integers: bool = True,
          allow_probability_language: bool = False) -> dict:
    allowed = set(_flatten(facts))
    issues: List[dict] = []
    for m in _NUM.finditer(text or ""):
        tok = m.group(0).strip()
        v = _to_float(tok)
        if v is None:
            continue
        if _YEARISH.match(tok.lstrip("+-−$")):
            continue
        if allow_small_integers and float(v).is_integer() and abs(v) <= 12 and "%" not in tok:
            continue  # counts like "two of three", "6 weeks", ordinals
        if not _supported(v, allowed):
            issues.append({"type": "unsupported_number", "token": tok})
    if not allow_probability_language:
        for m in _PROB.finditer(text or ""):
            issues.append({"type": "probability_language", "token": m.group(0)})
    return {"grounded": not issues, "issues": issues, "n_facts": len(allowed)}


def sanitize(text: str, facts: Dict[str, Any], **kw) -> dict:
    """Drop sentences that carry unsupported claims. Returns text + log."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept, dropped = [], []
    for s in sentences:
        if not s:
            continue
        (kept if check(s, facts, **kw)["grounded"] else dropped).append(s)
    return {"text": " ".join(kept), "dropped": dropped, "grounded": not dropped}
