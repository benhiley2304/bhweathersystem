#!/usr/bin/env python3
"""
inject_ff_macro.py  —  v2 "surprise z-decay" (2026-09-04)
────────────────────────────────────────────────────────────
Fetches ForexFactory actual-vs-forecast data for the G8 currencies
(USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD) from the sandbox (FF is blocked on
Render), converts every release into a standardised surprise z-score, decays
it by age, aggregates into growth / jobs / inflation, and POSTs one payload per
currency to /api/inject-ff-macro on the BH Weather System backend.

Method (chosen from a 2022-2026 backtest across 26 FX pairs, see
bh_fx_macro_audit_report.md):

  z_i     = clip((actual - forecast) * sign / sd_indicator, -3, +3)
            sign = -1 for lower-is-better prints (unemployment, claims)
            sd_indicator = historical sd of that indicator's raw surprise
            (static table in ff_surprise_sd.json, built from 2022-2026 FF data)
            NOTE: hot inflation surprise = currency-BULLISH (hawkish CB)
  w_i     = 0.5 ** (age_days / 20)   over a 90-day window
  cat_z   = sum(w_i z_i) / sum(w_i)  for growth (incl. PMIs, retail, GDP,
            sentiment), jobs, inflation
  ccy_z   = 0.40*growth + 0.35*jobs + 0.25*inflation (renormalised if a
            category is missing). Central-bank decisions are NOT scored
            (tested: no directional edge) — shown as context only.
  display = 5 + clip(ccy_z / 0.30, -2, +2) * 1.25        (0-10)

The backend maps a pair as raw = clip((z_base - z_quote)/0.30, -2, 2).

Run from the nightly backup task and the FF release watcher.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

BACKEND = "https://bhweathersystem-backend.onrender.com"
INJECT_URL = f"{BACKEND}/api/inject-ff-macro"

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]

# ── Method parameters (keep in sync with the backend pair mapping) ─────────────
HALF_LIFE_D = 20.0
WINDOW_D    = 90
Z_CLIP      = 3.0
Z_SCALE     = 0.30          # 1 unit of display raw (= 1.25 score pts) per 0.30 z
CAT_WEIGHTS = {"growth": 0.40, "jobs": 0.35, "inflation": 0.25}
CAT_FOLD    = {"mfg_pmi": "growth", "svc_pmi": "growth"}
MIN_RELEASES_FULL_CONF = 3   # fewer than this in-window -> damp toward 0

# Central-bank decision event names (context only, weight 0)
RATE_EVENT = {"EUR": "Main Refinancing Rate", "GBP": "Official Bank Rate",
              "JPY": "BOJ Policy Rate", "AUD": "Cash Rate", "CAD": "Overnight Rate",
              "CHF": "SNB Policy Rate", "NZD": "Official Cash Rate",
              "USD": "Federal Funds Rate"}

_SD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ff_surprise_sd.json")
try:
    SD_TABLE = json.load(open(_SD_PATH))
except Exception as _e:  # pragma: no cover
    SD_TABLE = {}
    print(f"  ERROR: could not load {_SD_PATH}: {_e}")

# ── FF fetch helpers ─────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
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
        # Cloudflare TLS-fingerprints python-requests (403 challenge) — curl passes.
        import subprocess as _sp
        _cp = _sp.run(["curl", "-s", "--max-time", "20",
                       "-A", HEADERS["User-Agent"],
                       "-H", f"Accept: {HEADERS['Accept']}",
                       "-H", f"Accept-Language: {HEADERS['Accept-Language']}",
                       "-H", f"Referer: {HEADERS['Referer']}",
                       url], capture_output=True, text=True)
        if _cp.returncode != 0 or not _cp.stdout:
            return []
        html = _cp.stdout
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


# ── Surprise z-decay scoring ─────────────────────────────────────────────────

def _release_date(dateline) -> Optional[date]:
    """FF dateline (unix UTC) -> trading date. Prints after 21:00 UTC belong to
    the next session (matches the backtest convention)."""
    try:
        ts = datetime.fromtimestamp(int(dateline), tz=timezone.utc)
    except Exception:
        return None
    d = ts.date()
    if ts.hour >= 21:
        d = d + timedelta(days=1)
    return d


def _fmt_z(z: float) -> str:
    return f"{z:+.2f}"


def _ccy_label(currency: str, z: float) -> str:
    if z >= 0.60:   return f"{currency} Macro Strong"
    if z >= 0.24:   return f"{currency} Macro Improving"
    if z <= -0.60:  return f"{currency} Macro Weak"
    if z <= -0.24:  return f"{currency} Macro Deteriorating"
    return f"{currency} Macro Neutral"


def display_score_from_z(z: float) -> float:
    raw = max(-2.0, min(2.0, z / Z_SCALE))
    return round(5.0 + raw * 1.25, 2)


def compute_ff_economy_score(currency: str, all_events: list, as_of: Optional[date] = None) -> dict:
    """Surprise z-decay economy score for one currency from a flat FF event list."""
    as_of = as_of or date.today()
    table = SD_TABLE.get(currency, {})
    seen: set = set()
    releases: list = []      # scored releases
    cb_context: Optional[dict] = None

    for ev in all_events:
        if ev.get("currency") != currency:
            continue
        name = ev.get("name", "")
        a_str, f_str, p_str = (ev.get("actual") or "").strip(), (ev.get("forecast") or "").strip(), (ev.get("previous") or "").strip()
        d = _release_date(ev.get("dateline"))
        if d is None or d > as_of:
            continue

        # Central-bank decision: context only
        if name == RATE_EVENT.get(currency) and a_str:
            if cb_context is None or d > date.fromisoformat(cb_context["date"]):
                cb_context = {"name": name, "date": d.isoformat(), "actual": a_str,
                              "forecast": f_str or "—", "previous": p_str or "—"}
            continue

        info = table.get(name)
        if not info or not a_str or not f_str:
            continue
        key = (name, ev.get("dateline"), a_str)
        if key in seen:
            continue
        seen.add(key)

        a, f = _parse_ff_value(a_str), _parse_ff_value(f_str)
        if a is None or f is None:
            continue
        age = (as_of - d).days
        if age < 0 or age >= WINDOW_D:
            continue
        sd = float(info["sd"])
        if sd <= 0:
            continue
        z = max(-Z_CLIP, min(Z_CLIP, (a - f) * info["sign"] / sd))
        w = 0.5 ** (age / HALF_LIFE_D)
        cat = CAT_FOLD.get(info["cat"], info["cat"])
        releases.append({
            "name": name, "date": d.isoformat(), "age_d": age,
            "actual": a_str, "forecast": f_str, "previous": p_str or "—",
            "raw_surprise": round((a - f) * info["sign"], 4), "sd": sd,
            "z": round(z, 3), "weight": round(w, 3), "cat": cat,
            # legacy sign field kept for older UI code (BEAT / MISS / LINE)
            "score": 1 if z > 0.05 else (-1 if z < -0.05 else 0),
        })

    if not releases:
        return {"score": 5.0, "label": f"{currency} Macro Neutral", "currency": currency,
                "z": 0.0, "cats": {}, "cat_details": {}, "source": "ff_injected",
                "method": "zdecay_v2", "n_releases": 0, "as_of": as_of.isoformat(),
                "context": {"cb": cb_context}}

    # Category aggregation: decay-weighted mean z
    cat_z: dict = {}
    cat_details: dict = {}
    for cat in ("growth", "jobs", "inflation"):
        items = [r for r in releases if r["cat"] == cat]
        if not items:
            continue
        sw = sum(r["weight"] for r in items)
        cat_z[cat] = round(sum(r["z"] * r["weight"] for r in items) / sw, 3)
        cat_details[cat] = sorted(items, key=lambda r: r["date"], reverse=True)

    wsum = sum(CAT_WEIGHTS[c] for c in cat_z)
    ccy_z = sum(cat_z[c] * CAT_WEIGHTS[c] for c in cat_z) / wsum if wsum else 0.0
    n = len(releases)
    conf = min(1.0, n / MIN_RELEASES_FULL_CONF)
    ccy_z = round(ccy_z * conf, 3)

    return {
        "score":       display_score_from_z(ccy_z),
        "label":       _ccy_label(currency, ccy_z),
        "currency":    currency,
        "z":           ccy_z,
        "cats":        cat_z,               # units: sd of typical surprise (decay-weighted)
        "cat_details": cat_details,
        "source":      "ff_injected",
        "method":      "zdecay_v2",
        "params":      {"half_life_d": HALF_LIFE_D, "window_d": WINDOW_D,
                        "weights": CAT_WEIGHTS, "z_scale": Z_SCALE, "z_clip": Z_CLIP},
        "n_releases":  n,
        "confidence":  round(conf, 2),
        "as_of":       as_of.isoformat(),
        "context":     {"cb": cb_context},
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[FF MACRO INJECT v2] Surprise z-decay economy scores (G8 incl. USD)")
    print(f"  Backend: {BACKEND} | HL={HALF_LIFE_D:.0f}d window={WINDOW_D}d weights={CAT_WEIGHTS}")
    if not SD_TABLE:
        print("  ERROR: surprise sd table missing — aborting.")
        raise SystemExit(2)

    all_events = fetch_all_events(n_weeks=16)
    if not all_events:
        print("  ERROR: No events fetched — FF may be down or blocked. Aborting.")
        raise SystemExit(1)

    results = {}
    for ccy in CURRENCIES:
        sd = compute_ff_economy_score(ccy, all_events)
        n = sd.get("n_releases", 0)
        print(f"\n  {ccy}: z={_fmt_z(sd['z'])} score={sd['score']:.2f} {sd['label']} | "
              f"cats={ {k: round(v, 2) for k, v in sd['cats'].items()} } | n={n}")
        if n == 0:
            print(f"    WARNING: 0 scored releases for {ccy} — skipping inject (backend keeps fallback)")
            results[ccy] = {"ok": False, "reason": "0 releases", "score": sd["score"]}
            continue
        try:
            r = requests.post(INJECT_URL, json=sd, timeout=20)
            resp = r.json()
            if resp.get("ok"):
                print(f"    Injected OK -> backend score={float(resp['score']):.2f}")
                results[ccy] = {"ok": True, "score": float(resp["score"])}
            else:
                print(f"    Inject FAILED: {resp}")
                results[ccy] = {"ok": False, "reason": str(resp)}
        except Exception as e:
            print(f"    Inject ERROR: {e}")
            results[ccy] = {"ok": False, "reason": str(e)}
        time.sleep(0.5)

    print("\n[FF MACRO INJECT v2] Summary:")
    for ccy, res in results.items():
        status = "OK" if res.get("ok") else f"SKIP ({res.get('reason','')[:80]})"
        print(f"  {ccy}: {status}" + (f" score={res['score']:.2f}" if res.get("score") is not None else ""))
    ok_count = sum(1 for r in results.values() if r.get("ok"))
    print(f"\n  {ok_count}/{len(CURRENCIES)} currencies injected successfully.")


if __name__ == "__main__":
    main()
