#!/usr/bin/env python3
"""
inject_ff_macro.py
──────────────────
Fetches ForexFactory actual-vs-forecast surprise data for non-USD currencies
from the sandbox (where FF is accessible), computes EMS-style economy scores,
and POSTs them to the BH Weather System backend via /api/inject-ff-macro.

This replaces the FRED trailing-average fallback with real consensus-based
surprise scores for EUR, GBP, JPY, AUD, CAD, CHF, NZD.

Run from cron or daily backup script.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Optional

import requests

BACKEND = "https://bhweathersystem-backend.onrender.com"
INJECT_URL = f"{BACKEND}/api/inject-ff-macro"

# ── ForexFactory indicator maps ──────────────────────────────────────────────
# (key, higher_is_good) for each currency's major events
FF_CURRENCY_INDICATOR_MAP = {
    "EUR": {
        "German Ifo Business Climate":    ("growth",    True),
        "German ZEW Economic Sentiment":  ("growth",    True),
        "Flash Manufacturing PMI":         ("mfg_pmi",  True),
        "Flash Services PMI":              ("svc_pmi",  True),
        "CPI y/y":                         ("inflation", False),
        "Core CPI y/y":                    ("inflation", False),
        "CPI Flash Estimate y/y":          ("inflation", False),
        "Core CPI Flash Estimate y/y":     ("inflation", False),
        "Unemployment Rate":               ("jobs",      False),
        "GDP q/q":                         ("growth",    True),
        "Retail Sales m/m":                ("growth",    True),
    },
    "GBP": {
        "GDP m/m":                         ("growth",    True),
        "GDP q/q":                         ("growth",    True),
        "CPI y/y":                         ("inflation", False),
        "Core CPI y/y":                    ("inflation", False),
        "Claimant Count Change":           ("jobs",      False),
        "Unemployment Rate":               ("jobs",      False),
        "Manufacturing PMI":               ("mfg_pmi",  True),
        "Services PMI":                    ("svc_pmi",  True),
        "Retail Sales m/m":                ("growth",    True),
        "Average Earnings Index 3m/y":     ("jobs",      True),
        "BOE Credit Conditions Survey":    ("growth",    True),
    },
    "JPY": {
        "Tankan Large Manufacturers Index":("growth",   True),
        "GDP q/q":                         ("growth",   True),
        "CPI y/y":                         ("inflation", False),
        "Tokyo Core CPI y/y":              ("inflation", False),
        "Unemployment Rate":               ("jobs",     False),
        "Manufacturing PMI":               ("mfg_pmi", True),
        "Services PMI":                    ("svc_pmi", True),
        "Industrial Production m/m":       ("growth",  True),
        "Retail Sales y/y":                ("growth",  True),
    },
    "AUD": {
        "Employment Change":               ("jobs",     True),
        "Unemployment Rate":               ("jobs",     False),
        "CPI q/q":                         ("inflation", False),
        "CPI y/y":                         ("inflation", False),
        "Trimmed Mean CPI q/q":            ("inflation", False),
        "GDP q/q":                         ("growth",   True),
        "Manufacturing PMI":               ("mfg_pmi", True),
        "Services PMI":                    ("svc_pmi", True),
        "Retail Sales m/m":                ("growth",  True),
        "Trade Balance":                   ("growth",  True),
    },
    "CAD": {
        "Employment Change":               ("jobs",     True),
        "Unemployment Rate":               ("jobs",     False),
        "CPI m/m":                         ("inflation", False),
        "CPI y/y":                         ("inflation", False),
        "Median CPI y/y":                  ("inflation", False),
        "GDP m/m":                         ("growth",   True),
        "Manufacturing PMI":               ("mfg_pmi", True),
        "Retail Sales m/m":                ("growth",  True),
        "Trade Balance":                   ("growth",  True),
    },
    "CHF": {
        "CPI m/m":                         ("inflation", False),
        "CPI y/y":                         ("inflation", False),
        "GDP q/q":                         ("growth",   True),
        "Manufacturing PMI":               ("mfg_pmi", True),
        "Unemployment Rate":               ("jobs",     False),
        "Retail Sales y/y":                ("growth",  True),
        "KOF Economic Barometer":          ("growth",  True),
    },
    "NZD": {
        "GDP q/q":                         ("growth",   True),
        "CPI q/q":                         ("inflation", False),
        "Employment Change q/q":           ("jobs",     True),
        "Unemployment Rate":               ("jobs",     False),
        "Manufacturing PMI":               ("mfg_pmi", True),
        "Retail Sales q/q":                ("growth",  True),
        "Trade Balance":                   ("growth",  True),
    },
}

# Currency codes as they appear in the FF calendar
FF_CURRENCY_CODE = {
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
    "AUD": "AUD",
    "CAD": "CAD",
    "CHF": "CHF",
    "NZD": "NZD",
}

# ── FF fetch helpers ─────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.forexfactory.com/",
}

def _get_week_strings(n_weeks: int = 16) -> list:
    today = date.today()
    day_of_week = today.weekday()
    days_since_sunday = (day_of_week + 1) % 7
    current_sunday = today - timedelta(days=days_since_sunday)
    months_short = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    week_strings = []
    for i in range(n_weeks):
        sunday = current_sunday - timedelta(weeks=i)
        mon = months_short[sunday.month - 1]
        week_strings.append(f"{mon}{sunday.day}.{sunday.year}")
    return week_strings


def _fetch_ff_week(week_str: str) -> list:
    url = f"https://www.forexfactory.com/calendar?week={week_str}"
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        if r.status_code != 200:
            return []
        html = r.text
        pattern = r'\{"id":\d+,"ebaseId":\d+,"name":"[^"]+.*?\}(?=,\{"id"|\])'
        blobs = re.findall(pattern, html, re.DOTALL)
        events = []
        for blob in blobs:
            try:
                obj = json.loads(blob)
                events.append({
                    "name":        obj.get("name", ""),
                    "actual":      obj.get("actual", "") or "",
                    "forecast":    obj.get("forecast", "") or "",
                    "previous":    obj.get("previous", "") or "",
                    "currency":    obj.get("currency", ""),
                    "dateline":    obj.get("dateline") or obj.get("date"),
                    "impactClass": obj.get("impactClass", ""),
                })
            except Exception:
                pass
        return events
    except Exception as e:
        print(f"  [FF FETCH] week={week_str} error: {e}")
        return []


def _parse_ff_value(v) -> Optional[float]:
    if v is None or v == "" or v == "—":
        return None
    s = str(v).strip().replace(",", "")
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    try:
        for suffix, mult in multipliers.items():
            if s.upper().endswith(suffix):
                return float(s[:-1]) * mult
        return float(s.replace("%", ""))
    except Exception:
        return None


def fetch_all_events(n_weeks: int = 16) -> list:
    week_strings = _get_week_strings(n_weeks)
    all_events = []
    print(f"  Fetching {n_weeks} weeks of FF calendar...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_fetch_ff_week, ws): ws for ws in week_strings}
        for fut in as_completed(futs):
            try:
                events = fut.result()
                all_events.extend(events)
            except Exception:
                pass
    print(f"  Got {len(all_events)} total events")
    return all_events


# ── Surprise scoring ─────────────────────────────────────────────────────────

def _ems_score_from_releases(releases: list) -> Optional[float]:
    """Given a list of {beat: bool} releases (oldest→newest), compute 0-10 EMS score."""
    recent = releases[-8:]
    if not recent:
        return None
    hits = [1 if r["beat"] else -1 for r in recent]
    weights = [1.0] * len(hits)
    for i in range(max(0, len(hits) - 4), len(hits)):
        weights[i] = 1.5
    raw = sum(h * w for h, w in zip(hits, weights)) / sum(weights)
    return round(max(0.0, min(10.0, raw * 3 + 5)), 1)


def compute_ff_economy_score(currency: str, all_events: list) -> dict:
    """
    From a flat list of all FF events, filter to this currency and compute
    an economy score using actual-vs-forecast surprise method.
    """
    ff_code = FF_CURRENCY_CODE[currency]
    indicator_map = FF_CURRENCY_INDICATOR_MAP[currency]

    # Collect releases per category
    cat_releases: dict = {}
    cat_details: dict  = {}

    for ev in all_events:
        if ev.get("currency") != ff_code:
            continue
        name         = ev.get("name", "")
        actual_str   = ev.get("actual", "")
        forecast_str = ev.get("forecast", "")

        # Skip if no actual/forecast
        if not actual_str or not forecast_str or actual_str in ("", "—") or forecast_str in ("", "—"):
            continue

        for event_name, (category, higher_is_good) in indicator_map.items():
            # Fuzzy match: event name starts with or equals indicator key
            if name == event_name or name.startswith(event_name.split(" ")[0]):
                # Stricter: must share first two words
                ev_words = name.lower().split()
                map_words = event_name.lower().split()
                if not (len(ev_words) >= 1 and len(map_words) >= 1 and ev_words[0] == map_words[0]):
                    continue
                if len(map_words) >= 2 and len(ev_words) >= 2 and ev_words[1] != map_words[1]:
                    continue

                actual_raw   = _parse_ff_value(actual_str)
                forecast_raw = _parse_ff_value(forecast_str)
                if actual_raw is None or forecast_raw is None:
                    break

                surprise_raw = actual_raw - forecast_raw
                if higher_is_good is False:
                    # Lower actual than forecast = positive surprise (e.g. lower unemployment)
                    beat = actual_raw < forecast_raw
                else:
                    beat = actual_raw > forecast_raw

                # Format display
                try:
                    actual_disp   = actual_str.strip()
                    forecast_disp = forecast_str.strip()
                    surprise_disp = round(actual_raw - forecast_raw, 3)
                except Exception:
                    actual_disp = forecast_disp = ""
                    surprise_disp = 0

                if category not in cat_releases:
                    cat_releases[category] = []
                    cat_details[category]  = []

                cat_releases[category].append({
                    "dateline": ev.get("dateline"),
                    "beat":     beat,
                })
                cat_details[category].append({
                    "name":     event_name,
                    "actual":   actual_disp,
                    "forecast": forecast_disp,
                    "score":    1 if beat else -1,
                })
                break

    if not cat_releases:
        return {
            "score":       5.0,
            "label":       f"{currency} Macro Neutral",
            "currency":    currency,
            "cats":        {},
            "cat_details": {},
            "source":      "ff_injected",
            "n_releases":  0,
        }

    # Sort each category's releases chronologically
    for cat in cat_releases:
        cat_releases[cat].sort(key=lambda x: x.get("dateline") or 0)

    # Score each category
    cat_scores = {}
    for cat, releases in cat_releases.items():
        ems = _ems_score_from_releases(releases)
        if ems is not None:
            cat_scores[cat] = ems

    if not cat_scores:
        return {
            "score":       5.0,
            "label":       f"{currency} Macro Neutral",
            "currency":    currency,
            "cats":        {},
            "cat_details": {},
            "source":      "ff_injected",
            "n_releases":  0,
        }

    # Composite: average of category EMS scores (0-10)
    composite = sum(cat_scores.values()) / len(cat_scores)
    composite = round(max(0.0, min(10.0, composite)), 1)

    # Confidence dampening: fewer categories → pull toward neutral (5.0)
    n_cats = len(cat_scores)
    confidence = min(1.0, n_cats / 4.0)  # full at 4+ categories
    composite = round(5.0 + (composite - 5.0) * confidence, 1)

    # Label
    if composite >= 7.5:    label = f"{currency} Macro Strong"
    elif composite >= 6.2:  label = f"{currency} Macro Improving"
    elif composite <= 2.5:  label = f"{currency} Macro Weak"
    elif composite <= 3.8:  label = f"{currency} Macro Deteriorating"
    else:                   label = f"{currency} Macro Neutral"

    # Convert 0-10 cat scores to -2..+2 for cats dict (matches FRED schema)
    cats_normalised = {cat: round((s - 5.0) / 2.5, 3) for cat, s in cat_scores.items()}

    total_releases = sum(len(v) for v in cat_releases.values())

    return {
        "score":       composite,
        "label":       label,
        "currency":    currency,
        "cats":        cats_normalised,
        "cat_details": cat_details,
        "source":      "ff_injected",
        "n_releases":  total_releases,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[FF MACRO INJECT] Starting non-USD ForexFactory economy score injection")
    print(f"  Backend: {BACKEND}")

    # Fetch all events once (shared across all currencies)
    all_events = fetch_all_events(n_weeks=16)

    if not all_events:
        print("  ERROR: No events fetched — FF may be down or blocked. Aborting.")
        return

    currencies = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    results = {}

    for ccy in currencies:
        print(f"\n  Processing {ccy}...")
        score_data = compute_ff_economy_score(ccy, all_events)
        n = score_data.get("n_releases", 0)
        cats_found = list(score_data.get("cats", {}).keys())
        print(f"    Score: {score_data['score']:.1f} | Label: {score_data['label']}")
        print(f"    Categories: {cats_found} | Releases found: {n}")

        if n == 0:
            print(f"    WARNING: 0 releases found for {ccy} — skipping inject (keeping FRED fallback)")
            results[ccy] = {"ok": False, "reason": "0 releases", "score": score_data["score"]}
            continue

        # POST to backend
        try:
            r = requests.post(INJECT_URL, json=score_data, timeout=15)
            resp = r.json()
            if resp.get("ok"):
                print(f"    Injected OK → backend score={resp['score']:.1f}")
                results[ccy] = {"ok": True, "score": resp["score"]}
            else:
                print(f"    Inject FAILED: {resp}")
                results[ccy] = {"ok": False, "reason": str(resp)}
        except Exception as e:
            print(f"    Inject ERROR: {e}")
            results[ccy] = {"ok": False, "reason": str(e)}

    # Summary
    print("\n[FF MACRO INJECT] Summary:")
    for ccy, res in results.items():
        status = "OK" if res.get("ok") else f"SKIP ({res.get('reason','')})"
        score_str = f" score={res['score']:.1f}" if res.get("score") is not None else ""
        print(f"  {ccy}: {status}{score_str}")

    ok_count = sum(1 for r in results.values() if r.get("ok"))
    print(f"\n  {ok_count}/{len(currencies)} currencies injected successfully.")


if __name__ == "__main__":
    main()
