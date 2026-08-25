"""
COT Weather Station — Backend v2
FastAPI server: CFTC COT (all 3 groups), FRED macro (surprise-based),
Yahoo Finance prices, cross-asset regime. Returns structured JSON.
"""

import asyncio
import json
import math
import time
import threading
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
import orjson
import gc, glob, os, pathlib, re
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _memory_mb() -> float:
    """Return current process RSS in MB. Returns 0 if psutil not available."""
    try:
        import psutil as _ps
        return _ps.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def _gc_if_heavy(label: str = ""):
    """Run gc.collect() and log memory before/after for heavy operations."""
    before = _memory_mb()
    gc.collect()
    after = _memory_mb()
    if before > 0:
        print(f"[GC] {label}: {before:.0f}MB → {after:.0f}MB (freed {before-after:.0f}MB)")

# Custom JSON encoder that replaces NaN/Inf with None so the response never crashes
class _SafeJSONResponse(JSONResponse):
    """
    JSON response using orjson for robust numpy/NaN/Inf handling.
    orjson natively serialises numpy int64/float64/bool/ndarray.
    NaN and Inf are converted to null via custom default.
    """
    media_type = "application/json"

    def render(self, content) -> bytes:
        import math as _math
        import numpy as _np

        def _default(obj):
            # numpy types that orjson might not catch in all versions
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                v = float(obj)
                return None if (_math.isnan(v) or _math.isinf(v)) else v
            if isinstance(obj, _np.bool_):
                return bool(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return orjson.dumps(content, default=_default, option=orjson.OPT_NON_STR_KEYS)


app = FastAPI(title="COT Weather Station v2", default_response_class=_SafeJSONResponse)

# Dedicated thread pool — large enough to avoid deadlocks when heavy sync functions
# (compute_macro_all, compute_risk_regime, _fetch_ff_months_parallel etc.) run concurrently.
import concurrent.futures as _cf
# FIX: Reduced from 8→5 workers. On a 2GB Render instance, 8 concurrent
# sync threads (each loading COT/yfinance/FRED data) easily exceeds memory.
# 5 is enough for the 3 main score functions + 2 slack.
_APP_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=5, thread_name_prefix="bh-worker")
# Dedicated low-priority executor for score_history heavy yfinance prefetch.
# Capped at 2 workers so concurrent history requests cannot OOM by competing
# with the main scores/FRED/COT executor on the 2GB Render instance.
_SH_EXECUTOR  = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bh-sh")  # single worker: walk-forward is memory-heavy (see _SH_GLOBAL_MAX)
# Per-market in-progress guard: prevents concurrent score_history prefetches doubling memory
_SH_MARKET_LOCKS: dict = {}
_photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
if os.path.isdir(_photos_dir):
    app.mount("/photos", StaticFiles(directory=_photos_dir), name="photos")

# GZip: compress responses >1KB — reduces /api/scores from ~216KB to ~29KB over the wire
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler: ensures score_history in-progress locks are always
# released even when an unhandled exception escapes the endpoint.
from fastapi import Request
from fastapi.responses import JSONResponse as _FJR
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    # Release any score_history lock that may be stuck
    path = request.url.path
    if "/api/score_history" in path:
        mkt_param = request.query_params.get("market", "").upper()
        if mkt_param:
            _SH_MARKET_LOCKS.pop(mkt_param, None)
    import traceback as _tb
    _full = _tb.format_exc()
    print(f"[GLOBAL ERROR] {path}: {type(exc).__name__}: {exc}\n{_full}", flush=True)
    return _FJR({"error": "Internal server error", "detail": str(exc), "traceback": _full[-2000:]}, status_code=500)

# ============================================================
# MARKET DEFINITIONS
# ============================================================
MARKETS = [
    # Equity Indices
    {"id": "ES",  "name": "S&P 500",       "ticker": "ES1!",  "yf": "^GSPC",    "category": "equity",    "cftc_code": "13874A", "cftc_name": "E-MINI S&P 500"},
    {"id": "NQ",  "name": "NASDAQ",        "ticker": "NQ1!",  "yf": "^NDX",     "category": "equity",    "cftc_code": "209742", "cftc_name": "E-MINI NASDAQ-100"},
    {"id": "YM",  "name": "Dow Jones",     "ticker": "YM1!",  "yf": "^DJI",     "category": "equity",    "cftc_code": "124603", "cftc_name": "DJIA x $5"},
    # FX
    {"id": "6E",  "name": "EUR/USD",       "ticker": "6E1!",  "yf": "EURUSD=X", "category": "fx",        "cftc_code": "099741", "cftc_name": "EURO FX"},
    {"id": "6J",  "name": "JPY/USD",   "ticker": "6J1!",  "yf": "JPYUSD=X",    "category": "fx",        "cftc_code": "097741", "cftc_name": "JAPANESE YEN",
     "cot_note": "Long JPY futures = long yen / short USD. Bullish score = bullish JPY (bearish USD/JPY)."},
    {"id": "6B",  "name": "GBP/USD",       "ticker": "6B1!",  "yf": "GBPUSD=X", "category": "fx",        "cftc_code": "096742", "cftc_name": "BRITISH POUND"},
    {"id": "6A",  "name": "AUD/USD",       "ticker": "6A1!",  "yf": "AUDUSD=X", "category": "fx",        "cftc_code": "232741", "cftc_name": "AUSTRALIAN DOLLAR"},
    {"id": "DX",  "name": "Dollar Index",  "ticker": "DX1!",  "yf": "DX-Y.NYB", "category": "fx",        "cftc_code": "098662", "cftc_name": "U.S. DOLLAR INDEX"},
    # Commodities
    {"id": "GC",  "name": "Gold",          "ticker": "GC1!",  "yf": "GC=F",     "category": "commodity", "cftc_code": "088691", "cftc_name": "GOLD"},
    {"id": "SI",  "name": "Silver",        "ticker": "SI1!",  "yf": "SI=F",     "category": "commodity", "cftc_code": "084691", "cftc_name": "SILVER"},
    {"id": "CL",  "name": "Crude Oil",     "ticker": "CL1!",  "yf": "CL=F",     "category": "commodity", "cftc_code": "067651", "cftc_name": "CRUDE OIL, LIGHT SWEET"},
    {"id": "HG",  "name": "Copper",        "ticker": "HG1!",  "yf": "HG=F",     "category": "commodity", "cftc_code": "085692", "cftc_name": "COPPER-GRADE #1"},
    {"id": "PL",  "name": "Platinum",      "ticker": "PL1!",  "yf": "PL=F",     "category": "commodity", "cftc_code": "076651", "cftc_name": "PLATINUM"},
    {"id": "PA",  "name": "Palladium",     "ticker": "PA1!",  "yf": "PA=F",     "category": "commodity", "cftc_code": "075651", "cftc_name": "PALLADIUM"},
    {"id": "KC",  "name": "Coffee",        "ticker": "KC1!",  "yf": "KC=F",     "category": "commodity", "cftc_code": "083731", "cftc_name": "COFFEE C",
     "cot_note": "Pure CFTC Arabica COT data (NY, ~196k OI). Arabica and Robusta are structurally different commodities with separate supply chains, participant profiles, and commercial bases — blending their COT data adds noise rather than signal. KC scores are therefore based solely on Arabica positioning."},
    {"id": "SB",  "name": "Sugar",         "ticker": "SB1!",  "yf": "SB=F",     "category": "commodity", "cftc_code": "080732", "cftc_name": "SUGAR NO. 11"},
    {"id": "ZC",  "name": "Corn",          "ticker": "ZC1!",  "yf": "ZC=F",     "category": "commodity", "cftc_code": "002602", "cftc_name": "CORN"},
    {"id": "ZS",  "name": "Soybeans",      "ticker": "ZS1!",  "yf": "ZS=F",     "category": "commodity", "cftc_code": "005602", "cftc_name": "SOYBEANS"},
    {"id": "ZW",  "name": "Wheat",         "ticker": "ZW1!",  "yf": "ZW=F",     "category": "commodity", "cftc_code": "001602", "cftc_name": "WHEAT"},
    # Fixed Income
    {"id": "ZB",  "name": "T-Bonds",       "ticker": "ZB1!",  "yf": "ZB=F",     "category": "bond",      "cftc_code": "020601", "cftc_name": "U.S. TREASURY BONDS"},
    {"id": "ZN",  "name": "10Y T-Notes",   "ticker": "ZN1!",  "yf": "ZN=F",     "category": "bond",      "cftc_code": "043602", "cftc_name": "10-YEAR U.S. TREASURY NOTES"},
    {"id": "ZF",  "name": "5Y T-Notes",    "ticker": "ZF1!",  "yf": "ZF=F",     "category": "bond",      "cftc_code": "044601", "cftc_name": "UST 5Y NOTE"},
    {"id": "ZT",  "name": "2Y T-Notes",    "ticker": "ZT1!",  "yf": "ZT=F",     "category": "bond",      "cftc_code": "042601", "cftc_name": "UST 2Y NOTE"},

    {"id": "6C",  "name": "CAD/USD",        "ticker": "6C1!",  "yf": "6C=F",     "category": "fx",        "cftc_code": "090741", "cftc_name": "CANADIAN DOLLAR"},
    {"id": "6N",  "name": "NZD/USD",        "ticker": "6N1!",  "yf": "6N=F",     "category": "fx",        "cftc_code": "112741", "cftc_name": "NZ DOLLAR"},
    {"id": "6S",  "name": "CHF/USD",        "ticker": "6S1!",  "yf": "6S=F",     "category": "fx",        "cftc_code": "092741", "cftc_name": "SWISS FRANC"},
    {"id": "6M",  "name": "MXN/USD",        "ticker": "6M1!",  "yf": "6M=F",     "category": "fx",        "cftc_code": "095741", "cftc_name": "MEXICAN PESO"},

    {"id": "RTY", "name": "Russell 2000",   "ticker": "RTY1!", "yf": "RTY=F",    "category": "equity",   "cftc_code": "239742", "cftc_name": "RUSSELL E-MINI"},

    {"id": "NG",  "name": "Natural Gas",    "ticker": "NG1!",  "yf": "NG=F",     "category": "commodity", "cftc_code": "023651", "cftc_name": "NAT GAS NYME"},
    {"id": "RB",  "name": "RBOB Gasoline",  "ticker": "RB1!",  "yf": "RB=F",     "category": "commodity", "cftc_code": "111659", "cftc_name": "GASOLINE RBOB"},
    {"id": "HO",  "name": "Heating Oil",    "ticker": "HO1!",  "yf": "HO=F",     "category": "commodity", "cftc_code": "022651", "cftc_name": "NY HARBOR ULSD"},
    {"id": "CC",  "name": "Cocoa",          "ticker": "CC1!",  "yf": "CC=F",     "category": "commodity", "cftc_code": "073732", "cftc_name": "COCOA",
     "cot_note": "COT blends CFTC NY Cocoa (60%) + ICE London Cocoa (40%) via z-score normalization. Raw blending is invalid: London is GBP-denominated with ~100k OI vs NY ~200k, and London has higher commercial concentration (65-73% of OI vs NY 48-50%). Each exchange's signals are normalized within their own history before blending, making them comparable on a unit-free basis. London adds the European physical demand signal (cash buyers, grinding industry); NY captures the financial/macro speculative overlay."},
    {"id": "CT",  "name": "Cotton",         "ticker": "CT1!",  "yf": "CT=F",     "category": "commodity", "cftc_code": "033661", "cftc_name": "COTTON NO. 2"},
    {"id": "LE",  "name": "Live Cattle",    "ticker": "LE1!",  "yf": "LE=F",     "category": "commodity", "cftc_code": "057642", "cftc_name": "LIVE CATTLE"},
    {"id": "HE",  "name": "Lean Hogs",      "ticker": "HE1!",  "yf": "HE=F",     "category": "commodity", "cftc_code": "054642", "cftc_name": "LEAN HOGS"},
    {"id": "GF",  "name": "Feeder Cattle",  "ticker": "GF1!",  "yf": "GF=F",     "category": "commodity", "cftc_code": "061641", "cftc_name": "FEEDER CATTLE"},
    # ── Crypto ────────────────────────────────────────────────────────────────
    # CME Bitcoin futures: Fund Managers are primary signal (vs commercials for commodities)
    # COT note: Lspec (Fund Managers/Large Specs) net positioning is the credible signal for crypto
    {"id": "BTC", "name": "Bitcoin",   "ticker": "BTC1!", "yf": "BTC-USD", "category": "crypto",
     "cftc_code": "133741", "cftc_name": "BITCOIN",
     "cot_note": "CME Bitcoin futures. Large Specs (fund managers) are the primary signal — they are trend-followers whose extreme positioning reliably marks turns. Commercials (miners/hedgers) in crypto behave differently from traditional commodities.",
     "crypto_cot_mode": True},  # Flag: use lspec as primary COT signal
    {"id": "ETH", "name": "Ethereum",  "ticker": "ETH1!", "yf": "ETH-USD", "category": "crypto",
     "cftc_code": "146021", "cftc_name": "ETHER CASH SETTLED",
     "cot_note": "CME Ether futures. Large Specs (fund managers) are the primary signal — extreme long positioning has historically marked local tops; extreme shorts have marked bottoms.",
     "crypto_cot_mode": True},

    # ── FX Cross Pairs (derived COT from base/quote leg Briese differential) ──
    {"id": "EURJPY", "name": "EUR/JPY", "ticker": "EURJPY", "yf": "EURJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6E vs 6J legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over JPY."},
    {"id": "EURGBP", "name": "EUR/GBP", "ticker": "EURGBP", "yf": "EURGBP=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6B",
     "cot_note": "COT: 3-category net spread of 6E vs 6B legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over GBP."},
    {"id": "EURAUD", "name": "EUR/AUD", "ticker": "EURAUD", "yf": "EURAUD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6A",
     "cot_note": "COT: 3-category net spread of 6E vs 6A legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over AUD."},
    {"id": "EURCAD", "name": "EUR/CAD", "ticker": "EURCAD", "yf": "EURCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6C",
     "cot_note": "COT: 3-category net spread of 6E vs 6C legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over CAD."},
    {"id": "EURNZD", "name": "EUR/NZD", "ticker": "EURNZD", "yf": "EURNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6N",
     "cot_note": "COT: 3-category net spread of 6E vs 6N legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over NZD."},
    {"id": "EURCHF", "name": "EUR/CHF", "ticker": "EURCHF", "yf": "EURCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6S",
     "cot_note": "COT: 3-category net spread of 6E vs 6S legs (commercials / large specs / small specs, OI-normalised). Measures EUR positioning advantage over CHF."},
    {"id": "GBPJPY", "name": "GBP/JPY", "ticker": "GBPJPY", "yf": "GBPJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6B vs 6J legs (commercials / large specs / small specs, OI-normalised). Measures GBP positioning advantage over JPY."},
    {"id": "GBPAUD", "name": "GBP/AUD", "ticker": "GBPAUD", "yf": "GBPAUD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6A",
     "cot_note": "COT: 3-category net spread of 6B vs 6A legs (commercials / large specs / small specs, OI-normalised). Measures GBP positioning advantage over AUD."},
    {"id": "GBPCAD", "name": "GBP/CAD", "ticker": "GBPCAD", "yf": "GBPCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6C",
     "cot_note": "COT: 3-category net spread of 6B vs 6C legs (commercials / large specs / small specs, OI-normalised). Measures GBP positioning advantage over CAD."},
    {"id": "GBPNZD", "name": "GBP/NZD", "ticker": "GBPNZD", "yf": "GBPNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6N",
     "cot_note": "COT: 3-category net spread of 6B vs 6N legs (commercials / large specs / small specs, OI-normalised). Measures GBP positioning advantage over NZD."},
    {"id": "GBPCHF", "name": "GBP/CHF", "ticker": "GBPCHF", "yf": "GBPCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6S",
     "cot_note": "COT: 3-category net spread of 6B vs 6S legs (commercials / large specs / small specs, OI-normalised). Measures GBP positioning advantage over CHF."},
    {"id": "AUDJPY", "name": "AUD/JPY", "ticker": "AUDJPY", "yf": "AUDJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6A vs 6J legs (commercials / large specs / small specs, OI-normalised). Classic risk barometer — bullish = risk-on."},
    {"id": "AUDNZD", "name": "AUD/NZD", "ticker": "AUDNZD", "yf": "AUDNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6N",
     "cot_note": "COT: 3-category net spread of 6A vs 6N legs (commercials / large specs / small specs, OI-normalised). Measures AUD positioning advantage over NZD."},
    {"id": "AUDCAD", "name": "AUD/CAD", "ticker": "AUDCAD", "yf": "AUDCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6C",
     "cot_note": "COT: 3-category net spread of 6A vs 6C legs (commercials / large specs / small specs, OI-normalised). Both commodity currencies — spread captures relative commodity exposure."},
    {"id": "NZDJPY", "name": "NZD/JPY", "ticker": "NZDJPY", "yf": "NZDJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6N", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6N vs 6J legs (commercials / large specs / small specs, OI-normalised). Risk barometer — bullish = risk-on."},
    {"id": "NZDCAD", "name": "NZD/CAD", "ticker": "NZDCAD", "yf": "NZDCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6N", "quote_leg": "6C",
     "cot_note": "COT: 3-category net spread of 6N vs 6C legs (commercials / large specs / small specs, OI-normalised). Commodity currency spread."},
    {"id": "CADJPY", "name": "CAD/JPY", "ticker": "CADJPY", "yf": "CADJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6C", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6C vs 6J legs (commercials / large specs / small specs, OI-normalised). Oil-linked risk barometer."},
    {"id": "CHFJPY", "name": "CHF/JPY", "ticker": "CHFJPY", "yf": "CHFJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6S", "quote_leg": "6J",
     "cot_note": "COT: 3-category net spread of 6S vs 6J legs (commercials / large specs / small specs, OI-normalised). Dual safe-haven pair — risk-off = bearish (JPY strengthens more)."},
    {"id": "AUDCHF", "name": "AUD/CHF", "ticker": "AUDCHF", "yf": "AUDCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6S",
     "cot_note": "COT: 3-category net spread of 6A vs 6S legs (commercials / large specs / small specs, OI-normalised). Risk appetite gauge — AUD vs safe-haven CHF."},

    # ── ICE Europe markets: REMOVED 2026-07-12 at Ben's request ──
    # (Brent B, LS Gas Oil GO, Robusta RC, FTSE 100 Z, Long Gilt R — different COT
    # layout, never traded. ICE fetch/inject plumbing left dormant elsewhere.)
]

_REMOVED_ICE_MARKETS = [
    # COT sourced from ICE Europe (not CFTC). Use ice_code field instead of cftc_code.
    # Disaggregated markets (energy/softs): same scoring pipeline as CFTC disagg.
    # Financial markets (FTSE/Gilt): TFF format, AM mapped to comm_net, LevFund to lspec_net.
    # History depth: energy/softs 2020-present (~329w); FTSE ~73w; Long Gilt ~57w.
    # Limited history on financial markets flagged in UI — percentiles less reliable.

    # Brent Crude — ICE Europe global benchmark
    # Spec correlation with CFTC CL = 0.12 (genuinely independent signal from WTI)
    # Commercial correlation with CL = 0.54 (moderate divergence — different delivery/grade)
    {"id": "B",  "name": "Brent Crude",  "ticker": "CB1!",  "yf": "BZ=F",    "category": "commodity",
     "ice_code": "B", "cot_format": "disagg",
     "cot_note": "ICE Europe. Brent commercial positioning diverges meaningfully from CFTC WTI (spec correlation 0.12) — genuinely independent signal reflecting North Sea/European physical market."},

    # Low Sulphur Gasoil — European diesel benchmark (ARA region), ICE ticker G
    # TradingView: ULS1! (rebranded from QS1! when ICE renamed contract ~2019)
    # 329 weeks of data. Commercials currently 99th %ile — historically extreme.
    {"id": "GO", "name": "LS Gas Oil",   "ticker": "ULS1!", "yf": "HO=F",    "category": "commodity",  # HO=F = NYMEX Heating Oil, best YF proxy for LS Gas Oil
     "ice_code": "G", "cot_format": "disagg",
     "cot_note": "ICE Europe. GO commercials are buyer-dominated (airlines, distributors, European petroleum buyers) — not producer-hedgers. Extreme commercial longs = buyers aggressively locking in forward supply = scarcity signal, NOT overvalued hedging. Confirmed by: ARA diesel stocks at 18-month lows, crack spreads 80% above pre-war levels, curve in backwardation."},

    # Robusta Coffee — companion to CFTC Arabica (KC)
    # Robusta = lower-grade, used in instant coffee/espresso blends; grown in Vietnam/Indonesia
    # Different supply chain from Arabica (Brazil-dominated) — meaningful divergences possible
    {"id": "RC", "name": "Robusta Coffee", "ticker": "DF1!", "yf": "KC=F",    "category": "commodity",  # KC=F = CFTC Arabica, best YF proxy for Robusta seasonality/momentum
     "ice_code": "RC", "cot_format": "disagg", "ice_limited_history": True,
     "cot_note": "ICE Europe. Robusta (Vietnam ~42%, Indonesia) vs Arabica (Brazil ~40%). ICO research: open interest (not net managed-money positions) is the more reliable RC predictor — spec positioning has weaker signal quality than KC. RC has higher supply-shock volatility from ENSO impacts. Price proxy KC=F is Arabica — divergences of 50-70% possible in supply-shock years."},

    # FTSE 100 Index Futures — ICE Europe financial TFF format
    # Asset Managers = institutional longs (pension funds, SWFs) → mapped to comm_net
    # Leveraged Funds = hedge funds, structurally short U2192 mapped to lspec_net
    # History: Dec 2024 — present (~73 weeks). Percentiles flagged as limited in UI.
    {"id": "Z",  "name": "FTSE 100",     "ticker": "Z1!",   "yf": "^FTSE",  "category": "equity",
     "ice_code": "Z", "cot_format": "tff", "ice_limited_history": True,
     "ice_fin": True,  # TFF format — AM/HF scoring, not commercial
     "cot_note": "ICE Europe TFF format. 73w history (Dec 2024-present) — below the 156w Briese index minimum. COT weight reduced to 12% and score dampened 35% toward neutral. AM at 87th %ile = structural institutional support (lagging confirming signal, not a leading fade). HF at 70th %ile = elevated but not contrarian territory (>85th %ile needed). Treat directionally. Full weight restored when history reaches Dec 2027."},

    # Long Gilt — UK government bond futures (equivalent of CFTC ZB for UK rates)
    # 1.18M OI — highly liquid, major institutional market
    # HF very short (19th %ile) currently — significant positioning signal for UK rates
    # History: Mar 2025 — present (~57 weeks). Percentiles flagged as limited.
    {"id": "R",  "name": "Long Gilt",    "ticker": "G1!",   "yf": "IGLT.L", "category": "bond",
     "ice_code": "R", "cot_format": "tff", "ice_limited_history": True,
     "ice_fin": True,
     "cot_note": "ICE Europe TFF format. UK government bond futures. Asset Managers long = structural institutional demand. HF at 19th %ile (heavily short) — BUT ~70-80% of HF shorts in bond futures are basis trades (long cash/short futures), not directional. Research: HF extremes in bonds are contrarian only when a macro regime-shift catalyst is present. Use directionally; treat with caution. 57w history — well below 156w Briese threshold."},
]

# ============================================================
# SEASONALITY ENGINE
# ============================================================

SEASONAL_WINDOWS = {
    "GC":  {"bull": [(8,10), (6,7)],  "bear": [(1,3)]},
    "SI":  {"bull": [(8,10), (4,5)],  "bear": [(1,3)]},
    "CL":  {"bull": [(2,4), (9,10)],  "bear": [(5,7)]},
    "ES":  {"bull": [(10,12), (1,1)], "bear": [(8,10)]},
    "NQ":  {"bull": [(10,12), (1,1)], "bear": [(8,10)]},
    "YM":  {"bull": [(10,12), (1,1)], "bear": [(8,10)]},
    "6E":  {"bull": [(4,6)],          "bear": [(1,2), (10,12)]},
    "6J":  {"bull": [(3,5)],          "bear": [(7,10)]},
    "6B":  {"bull": [(4,6)],          "bear": [(10,12)]},
    "6A":  {"bull": [(4,6)],          "bear": [(10,12)]},
    "DX":  {"bull": [(1,2), (9,12)],  "bear": [(4,7)]},
    "KC":  {"bull": [(5,9)],          "bear": [(11,1)]},
    "SB":  {"bull": [(1,4)],          "bear": [(9,11)]},
    "HG":  {"bull": [(1,4)],          "bear": [(7,9)]},
    "ZC":  {"bull": [(5,7)],          "bear": [(9,11)]},
    "ZS":  {"bull": [(5,7)],          "bear": [(8,11)]},
    "ZW":  {"bull": [(3,5)],          "bear": [(7,9)]},
    "ZB":  {"bull": [(1,3)],          "bear": [(4,8)]},
    "ZN":  {"bull": [(1,3)],          "bear": [(4,8)]},
    "PL":  {"bull": [(7,10)],         "bear": [(2,5)]},
    "PA":  {"bull": [(9,11)],         "bear": [(3,6)]},
    # New FX
    "6C":  {"bull": [(4,6)],          "bear": [(10,12)]},  # CAD: spring commodity strength
    "6N":  {"bull": [(4,6)],          "bear": [(10,12)]},  # NZD: mirrors AUD
    "6S":  {"bull": [(3,5)],          "bear": [(8,10)]},   # CHF: safe-haven spring
    "6M":  {"bull": [(2,5)],          "bear": [(8,11)]},   # MXN: carry trade season
    # New equity
    "RTY": {"bull": [(10,12), (1,1)], "bear": [(8,10)]},   # Russell 2000: same as large-caps
}

# ============================================================
# RELATIVE VALUE CONFIG
# ============================================================
REL_VAL_CONFIG = {
    # Crypto: Q4 bull (Oct-Dec) strongest, Q1 often strong, summer doldrums
    # Also sensitive to halving cycle and election-year liquidity
    "BTC": {
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",     "label": "vs T-Bond",  "color": "#5c9eff",
             "bt_wr": 75.0, "bt_n": 8,
             "logic": "BTC cheap vs bonds = liquidity-driven repricing; hard assets oversold vs safe-haven"},
            {"id": "NQ",  "yf": "NQ=F",     "label": "vs NASDAQ",  "color": "#38bdf8",
             "bt_wr": None, "bt_n": None,
             "logic": "BTC/NQ captures tech-adjacent risk appetite; cheap BTC vs NQ = crypto lagging"},
            {"id": "GC",  "yf": "GC=F",     "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": None, "bt_n": None,
             "logic": "BTC/Gold ratio captures digital gold narrative strength or weakness"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "BTC/ZB: 75% WR (n=8) over 10yr backtest. Ensemble Z-score model improves timing further.",
    },
    "GC": {
        "peers": [
            {"id": "SI",  "yf": "SI=F",      "label": "vs Silver",   "color": "#94a3b8",
             "logic": "Gold/Silver ratio captures precious metals relative value"},
            {"id": "ZB",  "yf": "ZB=F",      "label": "vs T-Bonds",  "color": "#5c9eff",
             "bt_wr": 88.9, "bt_n": 9,
             "logic": "Gold/Bond ratio: cheap gold vs bonds = real rates falling, macro repricing"},
            {"id": "DX",  "yf": "DX-Y.NYB",  "label": "vs DXY",     "color": "#a78bfa",
             "logic": "Gold cheap vs DXY = dollar weakness + real rate signal"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "GC/ZB: 88.9% WR (n=9). Gold cheap vs bonds = macro regime shift.",
    },
    "SI": {
        "peers": [
            {"id": "GC",  "yf": "GC=F",      "label": "vs Gold",     "color": "#f5c842"},
            {"id": "HG",  "yf": "HG=F",      "label": "vs Copper",   "color": "#f97316"},
            {"id": "ZB",  "yf": "ZB=F",      "label": "vs T-Bonds",  "color": "#5c9eff"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   80,
    },
    "CL": {
        "peers": [
            {"id": "NG",  "yf": "NG=F",      "label": "vs Nat Gas",  "color": "#34d399"},
            {"id": "RB",  "yf": "RB=F",      "label": "vs RBOB Gas", "color": "#fb923c"},
            {"id": "ZB",  "yf": "ZB=F",      "label": "vs T-Bonds",  "color": "#5c9eff",
             "logic": "Crude cheap vs bonds = growth scare/demand collapse signal"},
            {"id": "DX",  "yf": "DX-Y.NYB",  "label": "vs DXY",     "color": "#a78bfa",
             "logic": "Crude priced in USD: cheap crude vs DXY = double undervaluation"},
        ],
        "periods": [10, 26],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "CL/ZN: 94.4% WR (n=18) over 10yr backtest.",
    },
    "HG": {
        "peers": [
            {"id": "GC",  "yf": "GC=F",      "label": "vs Gold",    "color": "#fbbf24"},
            {"id": "CL",  "yf": "CL=F",      "label": "vs Crude",  "color": "#34d399"},
            {"id": "DX",  "yf": "DX-Y.NYB",  "label": "vs DXY",   "color": "#a78bfa"},
        ],
        "periods": [10, 30],
    },
    "PL": {
        "peers": [
            {"id": "PA",  "yf": "PA=F",       "label": "vs Palladium", "color": "#5b6ef5"},
            {"id": "GC",  "yf": "GC=F",       "label": "vs Gold",      "color": "#fbbf24"},
        ],
        "periods": [10, 30],
    },
    "PA": {
        "peers": [
            {"id": "PL",  "yf": "PL=F",       "label": "vs Platinum", "color": "#e2e8f0"},
            {"id": "GC",  "yf": "GC=F",       "label": "vs Gold",     "color": "#fbbf24"},
        ],
        "periods": [10, 30],
    },
    # ── Grains ───────────────────────────────────────────────────────────────────
    "ZC": {
        "peers": [
            {"id": "ZW",  "yf": "ZW=F",       "label": "vs Wheat",    "color": "#f59e0b"},
            {"id": "ZS",  "yf": "ZS=F",       "label": "vs Soybeans", "color": "#84cc16"},
        ],
        "periods": [10, 30],
    },
    "ZW": {
        "peers": [
            {"id": "ZC",  "yf": "ZC=F",       "label": "vs Corn",     "color": "#fde68a"},
            {"id": "ZS",  "yf": "ZS=F",       "label": "vs Soybeans", "color": "#84cc16"},
        ],
        "periods": [10, 30],
    },
    "ZS": {
        "peers": [
            {"id": "ZC",  "yf": "ZC=F",       "label": "vs Corn",  "color": "#fde68a"},
            {"id": "ZW",  "yf": "ZW=F",       "label": "vs Wheat", "color": "#f59e0b"},
        ],
        "periods": [10, 30],
    },
    # ── Softs ────────────────────────────────────────────────────────────────────
    "KC": {
        "peers": [
            {"id": "SB",  "yf": "SB=F",       "label": "vs Sugar", "color": "#fb7185"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",  "color": "#a78bfa"},
        ],
        "periods": [10, 30],
    },
    "SB": {
        "peers": [
            {"id": "KC",  "yf": "KC=F",       "label": "vs Coffee", "color": "#92400e"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",   "color": "#a78bfa"},
        ],
        "periods": [10, 30],
    },
    # ── FX ───────────────────────────────────────────────────────────────────────
    "6E": {
        "peers": [
            {"id": "ZN",  "yf": "ZN=F",      "label": "vs 10Y Note", "color": "#5c9eff",
             "bt_wr": 100.0, "bt_n": 15, "bt_hold": 3,
             "logic": "EUR/10Y captures USD rate differential; EUR cheap vs ZN = oversold vs rate spread"},
            {"id": "DX",  "yf": "DX-Y.NYB",  "label": "vs DXY",     "color": "#a78bfa",
             "bt_wr": 72.7, "bt_n": 11,
             "logic": "DXY strength drives EUR weakness; expensive EUR vs DXY = bearish signal"},
            {"id": "GC",  "yf": "GC=F",      "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": None, "bt_n": None,
             "logic": "EUR/Gold captures global risk appetite and USD debasement narrative"},
        ],
        "periods": [20, 52],
        "cheap_thr": 15,
        "exp_thr":   75,
        "signal_notes": "6E/ZN: 100% win rate (n=15) over 10yr backtest. EUR oversold vs rate differential = strong pullback long. 6E/DX: 72.7% for short signals.",
    },
    "6B": {
        "peers": [
            {"id": "GC",  "yf": "GC=F",      "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": 78.6, "bt_n": 14,
             "logic": "GBP/Gold ratio captures UK macro risk premium; cheap GBP vs gold = crisis-driven oversell"},
            {"id": "ZB",  "yf": "ZB=F",      "label": "vs T-Bond",  "color": "#5c9eff",
             "bt_wr": None, "bt_n": None,
             "logic": "GBP cheap vs US bonds = UK rate disadvantage priced in; reversion candidate"},
            {"id": "6E",  "yf": "EURUSD=X",  "label": "vs EUR/USD", "color": "#818cf8",
             "bt_wr": None, "bt_n": None,
             "logic": "EUR/GBP spread: GBP cheap vs EUR = post-Brexit discount potentially excessive"},
        ],
        "periods": [13, 26],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "GBP/Gold: 78.6% WR (n=14) over 10yr backtest. GBP pullbacks vs gold resolve to upside in trending environments.",
    },
    "6A": {
        "peers": [
            {"id": "6E",  "yf": "EURUSD=X",   "label": "vs EUR",  "color": "#818cf8"},
            {"id": "6B",  "yf": "GBPUSD=X",   "label": "vs GBP",  "color": "#60a5fa"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",  "color": "#a78bfa"},
            {"id": "ZB",  "yf": "ZB=F",        "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [10, 30],
    },
    "6J": {
        "peers": [
            {"id": "6E",  "yf": "EURUSD=X",  "label": "vs EUR/USD", "color": "#818cf8",
             "bt_wr": 81.8, "bt_n": 11,
             "logic": "JPY cheap vs EUR = yen oversold relative to EUR-denominated risk appetite"},
            {"id": "ZB",  "yf": "ZB=F",      "label": "vs T-Bond",  "color": "#5c9eff",
             "bt_wr": None, "bt_n": None,
             "logic": "Classic JPY/bond correlation: cheap yen vs bonds = carry unwind not yet priced"},
            {"id": "DX",  "yf": "DX-Y.NYB",  "label": "vs DXY",    "color": "#a78bfa",
             "bt_wr": None, "bt_n": None,
             "logic": "JPY/DXY ratio captures broad dollar strength vs yen weakness"},
        ],
        "periods": [10, 26],
        "cheap_thr": 15,
        "exp_thr":   80,
        "signal_notes": "6J/6E: 81.8% WR (n=11). Yen cheap vs EUR best identifies JPY pullback entries in carry-driven trends.",
    },
    "DX": {
        "peers": [
            {"id": "6E",  "yf": "EURUSD=X",   "label": "vs EUR",     "color": "#818cf8"},
            {"id": "GC",  "yf": "GC=F",        "label": "vs Gold",    "color": "#fbbf24"},
            {"id": "ZB",  "yf": "ZB=F",        "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [10, 30],
    },
    # ── Equities — quarterly cadence (13/26w) ────────────────────────────────────
    "ES": {
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",          "label": "vs T-Bond",  "color": "#5c9eff",
             "bt_wr": 88.9, "bt_n": 9,
             "logic": "ES cheap vs bonds = flight to safety = equity pullback, mean-reversion opportunity"},
            {"id": "GC",  "yf": "GC=F",          "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": 80.0, "bt_n": 5,
             "logic": "Equities cheap vs gold = risk-off regime, contrarian long setup"},
            {"id": "DX",  "yf": "DX-Y.NYB",      "label": "vs DXY",     "color": "#a78bfa",
             "bt_wr": None, "bt_n": None,
             "logic": "Dollar strength weighs on equities; ES/DX cheapness = oversold"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "ES/ZB ensemble: 88.9% win rate over 10yr backtest. Use as pullback long confirmation in secular uptrends.",
    },
    "NQ": {
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",          "label": "vs T-Bond",  "color": "#5c9eff",
             "bt_wr": 100.0, "bt_n": 5,  "bt_hold": 2,
             "logic": "NQ cheap vs bonds = bonds bid = risk-off selloff = pullback long opportunity"},
            {"id": "GC",  "yf": "GC=F",          "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": None, "bt_n": None,
             "logic": "NQ/Gold captures risk appetite regime"},
            {"id": "DX",  "yf": "DX-Y.NYB",      "label": "vs DXY",     "color": "#a78bfa",
             "bt_wr": None, "bt_n": None,
             "logic": "Dollar strength weighs on risk assets; cheap NQ vs USD = oversold"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   75,
        "signal_notes": "NQ/ZB is the primary pullback signal (100% win rate, 10yr backtest). Periods 13+39w catch both near-term and structural cheapness.",
    },
    "YM": {
        "peers": [
            {"id": "ES",  "yf": "^GSPC",         "label": "vs S&P 500",  "color": "#34d399"},
            {"id": "NQ",  "yf": "^NDX",           "label": "vs NASDAQ",   "color": "#38bdf8"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds",  "color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    # ── Bonds ────────────────────────────────────────────────────────────────────
    "ZB": {
        "peers": [
            {"id": "6E",  "yf": "EURUSD=X",       "label": "vs EUR/USD",  "color": "#818cf8",
             "bt_wr": 70.0, "bt_n": 20,
             "logic": "ZB cheap vs EUR = dollar-denominated bonds oversold vs FX; captures rate differential pricing"},
            {"id": "GC",  "yf": "GC=F",            "label": "vs Gold",    "color": "#f5c842",
             "bt_wr": None, "bt_n": None,
             "logic": "ZB/Gold ratio: bonds cheap vs gold = safe haven rotation under-priced"},
            {"id": "ZN",  "yf": "ZN=F",            "label": "vs 10Y Note","color": "#6ee7b7",
             "bt_wr": None, "bt_n": None,
             "logic": "Yield curve spread proxy: ZB/ZN captures 30y vs 10y relative value"},
        ],
        "periods": [13, 39],
        "cheap_thr": 20,
        "exp_thr":   80,
        "signal_notes": "ZB/6E has strong short-signal edge (83.3% WR via ensemble). ZB relative value best used for duration-adjusted entries.",
    },
    "ZN": {
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",            "label": "vs 30Y Bonds", "color": "#a5f3fc"},
            {"id": "ES",  "yf": "^GSPC",            "label": "vs S&P 500",  "color": "#34d399"},
        ],
        "periods": [10, 30],
    },
    # ── New Bond Tenors ────────────────────────────────────────────────────
    "ZF": {
        "peers": [
            {"id": "ZN",  "yf": "ZN=F",            "label": "vs 10Y Notes", "color": "#6ee7b7"},
            {"id": "ZT",  "yf": "ZT=F",            "label": "vs 2Y Notes",  "color": "#93c5fd"},
        ],
        "periods": [10, 30],
    },
    "ZT": {
        "peers": [
            {"id": "ZF",  "yf": "ZF=F",            "label": "vs 5Y Notes",  "color": "#6ee7b7"},
            {"id": "ZN",  "yf": "ZN=F",            "label": "vs 10Y Notes", "color": "#a5f3fc"},
        ],
        "periods": [8, 20],
    },
    # ── New FX Pairs ─────────────────────────────────────────────────────────
    "6C": {
        "peers": [
            {"id": "6A",  "yf": "AUDUSD=X",       "label": "vs AUD",    "color": "#4ade80"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",   "color": "#a78bfa"},
            {"id": "CL",  "yf": "CL=F",           "label": "vs Crude", "color": "#f97316"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    "6N": {
        "peers": [
            {"id": "6A",  "yf": "AUDUSD=X",       "label": "vs AUD",   "color": "#4ade80"},
            {"id": "6C",  "yf": "6C=F",           "label": "vs CAD",   "color": "#fb923c"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",  "color": "#a78bfa"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    "6S": {
        "peers": [
            {"id": "6J",  "yf": "JPYUSD=X",       "label": "vs JPY",   "color": "#f9a8d4"},
            {"id": "GC",  "yf": "GC=F",           "label": "vs Gold",  "color": "#fbbf24"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",  "color": "#a78bfa"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    "6M": {
        "peers": [
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",   "color": "#a78bfa"},
            {"id": "6A",  "yf": "AUDUSD=X",       "label": "vs AUD",   "color": "#4ade80"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds","color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    # ── Russell 2000 ────────────────────────────────────────────────────────────
    "RTY": {
        "peers": [
            {"id": "ES",  "yf": "^GSPC",          "label": "vs S&P 500",  "color": "#34d399"},
            {"id": "NQ",  "yf": "^NDX",           "label": "vs NASDAQ",   "color": "#38bdf8"},
            {"id": "ZB",  "yf": "ZB=F",           "label": "vs T-Bonds",  "color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    # ── Energy ──────────────────────────────────────────────────────────────────
    "NG": {
        # NG relval: compare vs macro anchors (bonds + dollar) rather than energy siblings.
        # NG/CL and NG/HO are spread trades driven by weather/storage vs OPEC — not genuine
        # fair-value pairs. NG cheap vs ZB = commodity undervalued vs safe-haven (same logic
        # as CL/ZB, 94% WR). NG cheap vs DX = double undervaluation (USD-priced, demand-driven).
        # HO retained as secondary intra-energy check (NG/HO spread = heating vs power demand).
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",       "label": "vs T-Bond",   "color": "#5c9eff",
             "logic": "NG cheap vs bonds = real asset undervalued vs safe-haven; demand signal"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",     "color": "#a78bfa",
             "logic": "NG priced in USD: cheap NG vs strong DXY = double undervaluation signal"},
            {"id": "HO",  "yf": "HO=F",       "label": "vs Heat Oil", "color": "#fb923c",
             "logic": "Intra-energy check: NG/HO captures relative heating vs power demand"},
        ],
        "periods": [13, 26],
        "cheap_thr": 20,
        "exp_thr":   80,
    },
    "RB": {
        "peers": [
            {"id": "CL",  "yf": "CL=F",           "label": "vs Crude",    "color": "#34d399"},
            {"id": "HO",  "yf": "HO=F",           "label": "vs Heat Oil",  "color": "#f97316"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",     "color": "#a78bfa"},
        ],
        "periods": [13, 26],
    },
    "HO": {
        "peers": [
            {"id": "CL",  "yf": "CL=F",           "label": "vs Crude",    "color": "#34d399"},
            {"id": "NG",  "yf": "NG=F",           "label": "vs Nat Gas",  "color": "#60a5fa"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",     "color": "#a78bfa"},
        ],
        "periods": [13, 26],
    },
    # ── Softs ──────────────────────────────────────────────────────────────────────
    "CC": {
        "peers": [
            {"id": "KC",  "yf": "KC=F",           "label": "vs Coffee", "color": "#92400e"},
            {"id": "SB",  "yf": "SB=F",           "label": "vs Sugar",  "color": "#fb7185"},
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",   "color": "#a78bfa"},
        ],
        "periods": [13, 26],
    },
    "CT": {
        "peers": [
            {"id": "DX",  "yf": "DX-Y.NYB",       "label": "vs DXY",      "color": "#a78bfa"},
            {"id": "ZC",  "yf": "ZC=F",           "label": "vs Corn",     "color": "#fde68a"},
            {"id": "SB",  "yf": "SB=F",           "label": "vs Sugar",    "color": "#fb7185"},
        ],
        "periods": [13, 26],
    },
    # ── Livestock ───────────────────────────────────────────────────────────────────
    "LE": {
        "peers": [
            {"id": "GF",  "yf": "GF=F",           "label": "vs Feeder",   "color": "#d97706"},
            {"id": "HE",  "yf": "HE=F",           "label": "vs Lean Hogs","color": "#f472b6"},
            {"id": "ZC",  "yf": "ZC=F",           "label": "vs Corn",     "color": "#fde68a"},
        ],
        "periods": [13, 26],
    },
    "HE": {
        # HE relval: HE/LE (hog-cattle spread) and HE/ZC (hog-corn = feed cost ratio) are
        # intra-sector spreads, not macro valuation signals. Replacing with ZB (bonds as macro
        # anchor) + ZC (feed cost as a cost-of-production signal, retained as secondary check).
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",       "label": "vs T-Bond",   "color": "#5c9eff",
             "logic": "Lean Hogs cheap vs bonds = commodity undervalued vs safe-haven"},
            {"id": "ZC",  "yf": "ZC=F",       "label": "vs Corn",     "color": "#fde68a",
             "logic": "HE/ZC ratio captures profit margin proxy (hog price vs feed cost)"},
        ],
        "periods": [13, 26],
        "cheap_thr": 20,
        "exp_thr":   80,
    },
    "GF": {
        "peers": [
            {"id": "LE",  "yf": "LE=F",           "label": "vs Live Cattle","color": "#d97706"},
            {"id": "ZC",  "yf": "ZC=F",           "label": "vs Corn",      "color": "#fde68a"},
        ],
        "periods": [13, 26],
    },
    # ── FX Cross Pairs ────────────────────────────────────────────────────────────
    # Peers = sibling crosses sharing the same base or quote currency.
    # e.g. EURJPY peers are other JPY crosses (GBPJPY, AUDJPY) + EURUSD.
    "EURJPY": {
        "peers": [
            {"id": "GBPJPY", "yf": "GBPJPY=X", "label": "vs GBP/JPY", "color": "#60a5fa"},
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
            {"id": "6E",     "yf": "EURUSD=X", "label": "vs EUR/USD", "color": "#818cf8"},
        ],
        "periods": [10, 26],
    },
    "EURGBP": {
        "peers": [
            {"id": "6E",     "yf": "EURUSD=X", "label": "vs EUR/USD", "color": "#818cf8"},
            {"id": "6B",     "yf": "GBPUSD=X", "label": "vs GBP/USD", "color": "#60a5fa"},
        ],
        "periods": [10, 26],
    },
    "EURAUD": {
        "peers": [
            {"id": "GBPAUD", "yf": "GBPAUD=X", "label": "vs GBP/AUD", "color": "#60a5fa"},
            {"id": "6E",     "yf": "EURUSD=X", "label": "vs EUR/USD", "color": "#818cf8"},
            {"id": "6A",     "yf": "AUDUSD=X", "label": "vs AUD/USD", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "EURCAD": {
        "peers": [
            {"id": "GBPCAD", "yf": "GBPCAD=X", "label": "vs GBP/CAD", "color": "#60a5fa"},
            {"id": "6E",     "yf": "EURUSD=X", "label": "vs EUR/USD", "color": "#818cf8"},
        ],
        "periods": [10, 26],
    },
    "EURNZD": {
        "peers": [
            {"id": "GBPNZD", "yf": "GBPNZD=X", "label": "vs GBP/NZD", "color": "#60a5fa"},
            {"id": "AUDNZD", "yf": "AUDNZD=X", "label": "vs AUD/NZD", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "EURCHF": {
        "peers": [
            {"id": "GBPCHF", "yf": "GBPCHF=X", "label": "vs GBP/CHF", "color": "#60a5fa"},
            {"id": "AUDCHF", "yf": "AUDCHF=X", "label": "vs AUD/CHF", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "GBPJPY": {
        "peers": [
            {"id": "EURJPY", "yf": "EURJPY=X", "label": "vs EUR/JPY", "color": "#818cf8"},
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
            {"id": "6B",     "yf": "GBPUSD=X", "label": "vs GBP/USD", "color": "#60a5fa"},
        ],
        "periods": [10, 26],
    },
    "GBPAUD": {
        "peers": [
            {"id": "EURAUD", "yf": "EURAUD=X", "label": "vs EUR/AUD", "color": "#818cf8"},
            {"id": "AUDNZD", "yf": "AUDNZD=X", "label": "vs AUD/NZD", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "GBPCAD": {
        "peers": [
            {"id": "EURCAD", "yf": "EURCAD=X", "label": "vs EUR/CAD", "color": "#818cf8"},
            {"id": "AUDCAD", "yf": "AUDCAD=X", "label": "vs AUD/CAD", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "GBPNZD": {
        "peers": [
            {"id": "EURNZD", "yf": "EURNZD=X", "label": "vs EUR/NZD", "color": "#818cf8"},
            {"id": "AUDNZD", "yf": "AUDNZD=X", "label": "vs AUD/NZD", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "GBPCHF": {
        "peers": [
            {"id": "EURCHF", "yf": "EURCHF=X", "label": "vs EUR/CHF", "color": "#818cf8"},
            {"id": "AUDCHF", "yf": "AUDCHF=X", "label": "vs AUD/CHF", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    "AUDJPY": {
        "peers": [
            {"id": "GBPJPY", "yf": "GBPJPY=X", "label": "vs GBP/JPY", "color": "#60a5fa"},
            {"id": "EURJPY", "yf": "EURJPY=X", "label": "vs EUR/JPY", "color": "#818cf8"},
            {"id": "NZDJPY", "yf": "NZDJPY=X", "label": "vs NZD/JPY", "color": "#34d399"},
        ],
        "periods": [10, 26],
    },
    "AUDNZD": {
        "peers": [
            {"id": "GBPNZD", "yf": "GBPNZD=X", "label": "vs GBP/NZD", "color": "#60a5fa"},
            {"id": "EURNZD", "yf": "EURNZD=X", "label": "vs EUR/NZD", "color": "#818cf8"},
        ],
        "periods": [10, 26],
    },
    "AUDCAD": {
        "peers": [
            {"id": "GBPCAD", "yf": "GBPCAD=X", "label": "vs GBP/CAD", "color": "#60a5fa"},
            {"id": "NZDCAD", "yf": "NZDCAD=X", "label": "vs NZD/CAD", "color": "#34d399"},
        ],
        "periods": [10, 26],
    },
    "NZDJPY": {
        "peers": [
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
            {"id": "CADJPY", "yf": "CADJPY=X", "label": "vs CAD/JPY", "color": "#fb923c"},
        ],
        "periods": [10, 26],
    },
    "NZDCAD": {
        "peers": [
            {"id": "AUDCAD", "yf": "AUDCAD=X", "label": "vs AUD/CAD", "color": "#4ade80"},
            {"id": "GBPCAD", "yf": "GBPCAD=X", "label": "vs GBP/CAD", "color": "#60a5fa"},
        ],
        "periods": [10, 26],
    },
    "CADJPY": {
        "peers": [
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
            {"id": "NZDJPY", "yf": "NZDJPY=X", "label": "vs NZD/JPY", "color": "#34d399"},
            {"id": "GBPJPY", "yf": "GBPJPY=X", "label": "vs GBP/JPY", "color": "#60a5fa"},
        ],
        "periods": [10, 26],
    },
    "CHFJPY": {
        "peers": [
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
            {"id": "EURCHF", "yf": "EURCHF=X", "label": "vs EUR/CHF", "color": "#818cf8"},
        ],
        "periods": [10, 26],
    },
    "AUDCHF": {
        "peers": [
            {"id": "EURCHF", "yf": "EURCHF=X", "label": "vs EUR/CHF", "color": "#818cf8"},
            {"id": "GBPCHF", "yf": "GBPCHF=X", "label": "vs GBP/CHF", "color": "#60a5fa"},
            {"id": "AUDJPY", "yf": "AUDJPY=X", "label": "vs AUD/JPY", "color": "#4ade80"},
        ],
        "periods": [10, 26],
    },
    # ── ICE Europe markets ─────────────────────────────────────────────────
    # B (Brent): compare vs WTI (CL) to show Brent premium/discount, and DXY
    "B": {
        "peers": [
            {"id": "CL",  "yf": "CL=F",       "label": "vs WTI",      "color": "#34d399"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",     "color": "#a78bfa"},
            {"id": "GC",  "yf": "GC=F",        "label": "vs Gold",    "color": "#fbbf24"},
        ],
        "periods": [10, 30],
    },
    # GO (Gas Oil): compare vs WTI and Heating Oil (closest CFTC equivalent)
    "GO": {
        "peers": [
            {"id": "CL",  "yf": "CL=F",       "label": "vs WTI",       "color": "#34d399"},
            {"id": "HO",  "yf": "HO=F",       "label": "vs Heat Oil",  "color": "#fb923c"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",      "color": "#a78bfa"},
        ],
        "periods": [10, 30],
    },
    # RC (Robusta Coffee): compare vs Arabica (KC) — the most important spread
    "RC": {
        "peers": [
            {"id": "KC",  "yf": "KC=F",       "label": "vs Arabica",  "color": "#92400e"},
            {"id": "DX",  "yf": "DX-Y.NYB",   "label": "vs DXY",     "color": "#a78bfa"},
        ],
        "periods": [10, 30],
    },
    # Z (FTSE 100): compare vs S&P 500, NASDAQ, and T-Bonds — classic equity relative value
    "Z": {
        "peers": [
            {"id": "ES",  "yf": "^GSPC",      "label": "vs S&P 500",  "color": "#34d399"},
            {"id": "NQ",  "yf": "^NDX",       "label": "vs NASDAQ",   "color": "#38bdf8"},
            {"id": "ZB",  "yf": "ZB=F",        "label": "vs T-Bonds", "color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    # R (Long Gilt): compare vs US T-Bond (ZB) and 10Y Note — duration relative value
    "R": {
        "peers": [
            {"id": "ZB",  "yf": "ZB=F",        "label": "vs 30Y Bond",  "color": "#a5f3fc"},
            {"id": "ZN",  "yf": "ZN=F",        "label": "vs 10Y Note",  "color": "#6ee7b7"},
        ],
        "periods": [10, 30],
    },
    # ── Crypto ────────────────────────────────────────────────────────────────────
    # Compare each crypto vs: (1) the other crypto, (2) NASDAQ as risk-proxy,
    # (3) T-Bonds as risk-off/liquidity anchor.
    # Periods: 13/26 weeks (quarterly cadence, same as equity indices).
    "BTC": {
        "peers": [
            {"id": "ETH",  "yf": "ETH-USD",  "label": "vs ETH",     "color": "#818cf8"},
            {"id": "NQ",   "yf": "^NDX",     "label": "vs NASDAQ",  "color": "#38bdf8"},
            {"id": "ZB",   "yf": "ZB=F",     "label": "vs T-Bonds", "color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
    "ETH": {
        "peers": [
            {"id": "BTC",  "yf": "BTC-USD",  "label": "vs BTC",     "color": "#f97316"},
            {"id": "NQ",   "yf": "^NDX",     "label": "vs NASDAQ",  "color": "#38bdf8"},
            {"id": "ZB",   "yf": "ZB=F",     "label": "vs T-Bonds", "color": "#f472b6"},
        ],
        "periods": [13, 26],
    },
}


def compute_rel_val_score(market_id: str) -> dict:
    """
    Relative Valuation stochastic oscillator — mirrors TZv-WVal (TradingView).

    For each peer in REL_VAL_CONFIG[market_id], compute the stochastic
    (0–100) of the price ratio over each configured period.

    A value of 0 means this market is historically cheapest vs that peer
    (ratio at its lowest); 100 means historically most expensive.

    Score aggregation:
      average_stoch = mean of all peer×period lines (current bar only)
      avg < 20  → score +2 (very cheap)
      avg < 35  → score +1 (cheap)
      avg 35-65 → score  0 (neutral)
      avg > 65  → score -1 (expensive)
      avg > 80  → score -2 (very expensive)

    Also returns per-peer time-series (last 104 bars) for charting.
    """
    cfg = REL_VAL_CONFIG.get(market_id)
    if not cfg:
        return {"score": 0, "label": "No peers defined",
                "avg_stoch": None, "lines": [], "periods": []}

    mkt = next((m for m in MARKETS if m["id"] == market_id), None)
    if not mkt:
        return {"score": 0, "label": "Market not found",
                "avg_stoch": None, "lines": [], "periods": []}

    df_self = fetch_price_data(mkt["yf"])
    if df_self is None or df_self.empty:
        return {"score": 0, "label": "Price data unavailable",
                "avg_stoch": None, "lines": [], "periods": []}

    # Normalise self index
    self_close = df_self["Close"].copy()
    self_close.index = pd.to_datetime(self_close.index).tz_localize(None).normalize()

    peers    = cfg["peers"]
    periods  = cfg["periods"]
    # Per-asset ML-calibrated thresholds (fallback to classic 20/80 if not specified)
    CHEAP_THR  = cfg.get("cheap_thr", 20)
    EXP_THR    = cfg.get("exp_thr",   80)
    SIGNAL_NOTES = cfg.get("signal_notes", "")
    HIST_LEN = 104  # weeks of history to return for charting

    all_current_stochs: list[float] = []  # one value per peer×period for scoring
    all_zscores: list[float] = []  # z-score per peer for composite
    lines: list[dict] = []  # per-peer chart series

    for peer in peers:
        df_peer = fetch_price_data(peer["yf"])
        if df_peer is None or df_peer.empty:
            continue

        peer_close = df_peer["Close"].copy()
        peer_close.index = pd.to_datetime(peer_close.index).tz_localize(None).normalize()

        # Align the two series on common dates
        combined = pd.concat(
            [self_close.rename("self"), peer_close.rename("peer")], axis=1
        ).dropna()
        if len(combined) < max(periods) + 5:
            continue

        ratio = combined["self"] / combined["peer"]

        # Compute stochastic for each period, then average them for the chart line
        period_stochs_at_each_bar: list[pd.Series] = []
        for w in periods:
            if len(ratio) < w:
                continue
            roll_min = ratio.rolling(w).min()
            roll_max = ratio.rolling(w).max()
            denom    = roll_max - roll_min
            stoch_w  = pd.Series(
                np.where(denom > 0, (ratio - roll_min) / denom * 100, 50.0),
                index=ratio.index
            ).round(1)
            period_stochs_at_each_bar.append(stoch_w)
            # Record current-bar value for scoring
            last_val = stoch_w.dropna().iloc[-1] if not stoch_w.dropna().empty else None
            if last_val is not None:
                all_current_stochs.append(float(last_val))

        if not period_stochs_at_each_bar:
            continue

        # Average across periods → single composite line per peer
        stacked = pd.concat(period_stochs_at_each_bar, axis=1).dropna()
        avg_line = stacked.mean(axis=1).round(1)

        # Trim to last HIST_LEN bars for the chart
        hist = avg_line.iloc[-HIST_LEN:]
        dates = [str(d.date()) for d in hist.index]
        vals  = [None if np.isnan(v) else float(v) for v in hist.values]

        # ── Z-score of ratio vs rolling 52w mean/std ─────────────────────
        # How many standard deviations is the current ratio from "fair value"?
        _z_window = 52  # 52-week rolling window
        _ratio_mean = ratio.rolling(_z_window, min_periods=26).mean()
        _ratio_std  = ratio.rolling(_z_window, min_periods=26).std()
        _denom_z    = _ratio_std.replace(0, np.nan)
        zscore_series = ((ratio - _ratio_mean) / _denom_z).round(3)

        # Normalised ratio: ratio as % deviation from rolling 52w mean (for chart display)
        # 0 = at mean, +10 = 10% above mean, -10 = 10% below mean
        norm_ratio_series = ((ratio / _ratio_mean.replace(0, np.nan) - 1) * 100).round(2)

        # Trim to HIST_LEN for charting
        _hist_ratio  = norm_ratio_series.iloc[-HIST_LEN:]
        _hist_zscore = zscore_series.iloc[-HIST_LEN:]

        ratio_vals  = [None if (np.isnan(v) or not np.isfinite(v)) else float(v) for v in _hist_ratio.values]
        zscore_vals = [None if (np.isnan(v) or not np.isfinite(v)) else float(v) for v in _hist_zscore.values]
        zscore_curr = zscore_vals[-1] if zscore_vals else None
        ratio_mean_pct = ratio_vals[-1] if ratio_vals else None

        # Record current zscore for composite
        if zscore_curr is not None:
            all_zscores.append(zscore_curr)

        lines.append({
            "peer_id":      peer["id"],
            "label":        peer["label"],
            "color":        peer["color"],
            "dates":        dates,
            "values":       vals,
            "current":      vals[-1] if vals else None,
            "ratio_values": ratio_vals,
            "zscore_values": zscore_vals,
            "zscore_current": zscore_curr,
            "ratio_mean_pct": ratio_mean_pct,
            # Backtest metadata from config
            "bt_wr":   peer.get("bt_wr"),
            "bt_n":    peer.get("bt_n"),
            "bt_hold": peer.get("bt_hold"),
            "logic":   peer.get("logic", ""),
        })

    # ── Score from average of all current stochastics ────────────────────────
    if not all_current_stochs:
        return {"score": 0, "label": "Insufficient data",
                "avg_stoch": None, "lines": lines, "periods": periods}

    peer_avg_stoch = round(sum(all_current_stochs) / len(all_current_stochs), 1)

    # Self 52w range stoch (40% weight): where is price in its own 52w range?
    # Prevents DX (dropped 110->98) appearing 'fairly valued' on peer ratios alone.
    _self_vals = self_close.values.astype(float)
    _self_52w  = _self_vals[-52:] if len(_self_vals) >= 52 else _self_vals
    _self_hi   = float(np.nanmax(_self_52w))
    _self_lo   = float(np.nanmin(_self_52w))
    _self_curr = float(_self_vals[-1])
    self_stoch_52w = round((_self_curr - _self_lo) / (_self_hi - _self_lo) * 100, 1) if _self_hi > _self_lo else 50.0

    # 40% self-range + 60% peer-ratios
    avg_stoch = round(self_stoch_52w * 0.40 + peer_avg_stoch * 0.60, 1)


    # ── Trend gate: price vs SMA200 ───────────────────────────────────────
    # Determines whether valuation signal is actionable.
    # Bernd: "Do not short an undervalued market; do not long an overvalued market."
    # Additionally, trend must confirm valuation before we take a directional view.
    def _sma_series(arr, n):
        import pandas as pd
        return pd.Series(arr.astype(float)).rolling(n, min_periods=n).mean().values

    closes_arr = self_close.values.astype(float)
    curr_price = float(closes_arr[-1]) if len(closes_arr) > 0 else None

    if curr_price is not None and len(closes_arr) >= 200:
        sma200_arr = _sma_series(closes_arr, 200)
        sma200 = float(sma200_arr[-1]) if not np.isnan(sma200_arr[-1]) else None
    elif curr_price is not None and len(closes_arr) >= 50:
        # Fallback to EMA50 if insufficient history for SMA200
        sma200_arr = pd.Series(closes_arr).ewm(span=50, adjust=False).mean().values
        sma200 = float(sma200_arr[-1])
    else:
        sma200 = None


    # -- Trend gate: SMA200 + EMA50 composite ----------------------------------
    # EMA50 is more responsive than SMA200. A market above its lagging SMA200
    # but below its EMA50 is actively downtrending (e.g. PA: SMA200 still
    # lagging from prior crash, EMA50 correctly reflects recent fall).
    # BULL: above SMA200 >=1.5% AND above EMA50 (both MAs confirm trend)
    # BEAR: below EMA50 >=1.5% OR below SMA200 >=1.5% (rejected by either MA)
    import pandas as _pdtg
    # span=50 (EMA50) is intentional — span=10 flips direction on 1-2 bad weeks,
    # constantly toggling the trend gate and corrupting the relval signal with noise.
    _ema_short_tg = _pdtg.Series(closes_arr.astype(float)).ewm(span=50, adjust=False).mean().values
    ema_short_rv = float(_ema_short_tg[-1]) if len(_ema_short_tg) > 0 and not _pdtg.isna(_ema_short_tg[-1]) else None

    if curr_price is not None and sma200 is not None and sma200 > 0:
        pct_vs_200   = (curr_price - sma200)  / sma200  * 100
        pct_vs_ema50 = (curr_price - ema_short_rv) / ema_short_rv * 100 if ema_short_rv and ema_short_rv > 0 else 0.0
        if pct_vs_200 >= 1.5 and pct_vs_ema50 >= 0:
            trend_gate = "bull"   # above both MAs
        elif pct_vs_ema50 <= -1.5 or pct_vs_200 <= -1.5:
            trend_gate = "bear"   # below EMA50 or SMA200 significantly
        else:
            trend_gate = "neutral"
    else:
        trend_gate = "neutral"

    # ── Valuation label — relative to per-asset thresholds ─────────────────
    _cheap_mid   = CHEAP_THR                    # e.g. 20
    _cheap_deep  = CHEAP_THR / 2                # e.g. 10
    _exp_mid     = EXP_THR                      # e.g. 75
    _exp_deep    = EXP_THR + (100 - EXP_THR)/2  # e.g. 87.5
    _neutral_lo  = CHEAP_THR + (EXP_THR - CHEAP_THR) * 0.25
    _neutral_hi  = CHEAP_THR + (EXP_THR - CHEAP_THR) * 0.75

    if avg_stoch <= _cheap_deep:
        val_label = "Very Cheap"
    elif avg_stoch <= _cheap_mid:
        val_label = "Cheap"
    elif avg_stoch <= _neutral_lo:
        val_label = "Mildly Cheap"
    elif avg_stoch <= _neutral_hi:
        val_label = "Fairly Valued"
    elif avg_stoch <= _exp_mid:
        val_label = "Mildly Expensive"
    elif avg_stoch <= _exp_deep:
        val_label = "Expensive"
    else:
        val_label = "Very Expensive"

    # Market category for equities exception
    cat = mkt.get("category", "")
    is_equity = (cat == "equity")

    # ── ML-calibrated confluence count (using per-asset thresholds) ───────
    bull_count = sum(1 for v in all_current_stochs if v <= CHEAP_THR)
    bear_count = sum(1 for v in all_current_stochs if v >= EXP_THR)
    total_lines = len(all_current_stochs)
    confluence_peers = bull_count >= 2 or bear_count >= 2

    # ── Trend-gated scoring matrix (thresholds from ML backtest) ─────────
    if avg_stoch <= _cheap_mid:
        # Cheap zone — pullback long opportunity
        if trend_gate == "bull":
            score = 8.5 if avg_stoch <= _cheap_deep else 8.0
        elif trend_gate == "bear":
            # KEY INSIGHT: cheap + downtrend = PULLBACK LONG (price pulled back to value)
            score = 7.5 if avg_stoch <= _cheap_deep else 7.0
        else:
            score = 6.5 if is_equity else 6.0
    elif avg_stoch <= _neutral_lo:
        # Mildly cheap zone
        if trend_gate == "bull":
            score = 7.0
        elif trend_gate == "bear":
            score = 6.5  # mild pullback long
        else:
            score = 6.0 if is_equity else 5.5
    elif avg_stoch <= _neutral_hi:
        # Fair value zone — no signal
        score = 5.0
    elif avg_stoch <= _exp_mid:
        # Mildly expensive zone
        if trend_gate == "bear":
            score = 3.5
        elif trend_gate == "bull":
            # Expensive + uptrend = PULLBACK SHORT opportunity
            score = 3.0
        else:
            score = 4.0
    else:
        # Very expensive zone — pullback short opportunity
        if trend_gate == "bull":
            score = 2.0 if avg_stoch >= _exp_deep else 2.5
        elif trend_gate == "bear":
            score = 1.5 if avg_stoch >= _exp_deep else 2.0
        else:
            score = 3.0

    # Confluence nudge (±0.3 when multiple peers confirm same extreme)
    if confluence_peers:
        if bull_count >= 2:
            score = min(10.0, score + 0.3)
        if bear_count >= 2:
            score = max(0.0, score - 0.3)

    score = round(score, 1)

    # Build descriptive label
    trend_word = {"bull": "Uptrend", "bear": "Downtrend", "neutral": "Sideways"}[trend_gate]
    if avg_stoch <= _neutral_lo:
        label = f"{val_label} + {trend_word}"
    elif avg_stoch >= _neutral_hi:
        label = f"{val_label} + {trend_word}"
    else:
        label = val_label  # "Fairly Valued" — trend irrelevant

    confluence_note = ""
    if bull_count >= 2:
        confluence_note = f" ({bull_count}/{total_lines} peers confirm cheap)"
    elif bear_count >= 2:
        confluence_note = f" ({bear_count}/{total_lines} peers confirm expensive)"

    # ── Pullback signal classification (ML-calibrated) ───────────────────
    # PULLBACK LONG:  cheap vs peers + price has pulled back (downtrend or neutral)
    # PULLBACK SHORT: expensive vs peers + price extended (uptrend or neutral)
    is_cheap     = avg_stoch <= _neutral_lo
    is_expensive = avg_stoch >= _neutral_hi
    zscore_composite = round(float(np.mean(all_zscores)), 2) if all_zscores else None

    if is_cheap and trend_gate == "bear":
        signal_type = "pullback_long"
        signal_strength = "strong" if avg_stoch <= _cheap_mid else "moderate"
    elif is_cheap and (trend_gate == "neutral" or trend_gate == "bull"):
        signal_type = "pullback_long"
        signal_strength = "moderate" if (trend_gate == "bull" and avg_stoch <= _cheap_mid) else "weak"
    elif is_expensive and trend_gate == "bull":
        signal_type = "pullback_short"
        signal_strength = "strong" if avg_stoch >= _exp_mid else "moderate"
    elif is_expensive and (trend_gate == "neutral" or trend_gate == "bear"):
        signal_type = "pullback_short"
        signal_strength = "moderate" if (trend_gate == "bear" and avg_stoch >= _exp_mid) else "weak"
    else:
        signal_type = "none"
        signal_strength = "none"

    return {
        "score":            score,
        "label":            label + confluence_note,
        "avg_stoch":        avg_stoch,
        "peer_avg_stoch":   peer_avg_stoch,
        "self_stoch_52w":   self_stoch_52w,
        "trend_gate":       trend_gate,
        "bull_count":       bull_count,
        "bear_count":       bear_count,
        "total_lines":      total_lines,
        "lines":            lines,
        "periods":          periods,
        "signal_type":      signal_type,
        "signal_strength":  signal_strength,
        "zscore_composite": zscore_composite,
        "signal_notes":     SIGNAL_NOTES,
        "cheap_thr":        CHEAP_THR,
        "exp_thr":          EXP_THR,
    }


# ============================================================
# PUT/CALL RATIO — Contrarian Sentiment for Equity Markets
# ============================================================
# Uses CBOE Equity Put/Call ratio (20-day MA, percentile-based scoring)
# High P/C = fear/put buying = contrarian BULLISH signal
# ── PCR Market Configuration ─────────────────────────────────────────────────
# Backtest findings:
#   Equities/Metals: High PCR (fear) = contrarian BULL signal. No reliable bear edge.
#   Oil: Bidirectional — both fear and greed scored.
#   Crypto (Deribit): Different norms — raw PCR thresholds calibrated separately.
#   Bonds/FX/Ag: No statistically significant edge found — excluded.
#
# PCR weight tiers — evidence-based liquidity hierarchy:
#   TIER-1 (10%): ES, NQ, RTY, YM            — deepest equity options markets, strong backtest edge
#   TIER-2 ( 5%): GC, CL                     — GLD/USO ETF options meaningful but thinner than equity
#   TIER-3 ( 3%): BTC, ETH                   — good depth on Deribit, unique norms
#   TIER-4 ( 4%): SI                         — SLV ETF options thinner than GLD; dual industrial/precious char

PCR_EQUITY_SYMBOLS = {"ES", "NQ", "YM", "RTY"}  # keep for legacy weight-switching

# All assets with PCR scoring
PCR_ALL_SYMBOLS = {"ES", "NQ", "YM", "RTY", "GC", "SI", "CL", "BTC", "ETH"}

# CBOE ticker proxies (for non-equity assets)
PCR_CBOE_PROXY = {
    "GC":  "GLD",    # Gold ETF — 5M OI, 8k strikes
    "SI":  "SLV",    # Silver ETF — 8M OI, 5.5k strikes
    "CL":  "USO",    # Oil ETF — 1.7M OI, 4.7k strikes
}

# Signal tiers control weight and whether bear signal is active
# fmt: {mkt: {"tier": int, "source": "cboe"|"deribit"}}
PCR_TIERS = {
    "ES":  {"tier": 1, "source": "cboe_equity"},
    "NQ":  {"tier": 1, "source": "cboe_equity"},
    "YM":  {"tier": 1, "source": "cboe_equity"},
    "RTY": {"tier": 1, "source": "cboe_equity"},
    "GC":  {"tier": 2, "source": "cboe_etf"},   # GLD ETF options — deep but secondary signal; 5% weight
    "SI":  {"tier": 4, "source": "cboe_etf"},   # SLV ETF options — thinner than GLD; 4% weight
    "CL":  {"tier": 2, "source": "cboe_etf"},
    "BTC": {"tier": 3, "source": "deribit"},
    "ETH": {"tier": 3, "source": "deribit"},
}

# Cache for per-ticker CBOE ETF snapshots (refreshed hourly)
PCR_ETF_CACHE: dict = {}
PCR_ETF_CACHE_TTL = 3600

PCR_CACHE: dict = {"data": None, "time": 0}
PCR_CACHE_TTL = 3600  # 1 hour — daily data changes once per day

# Deribit cache
PCR_DERIBIT_CACHE: dict = {}
PCR_DERIBIT_CACHE_TTL = 1800


def fetch_pcr_history() -> Optional[pd.DataFrame]:
    """
    Fetch CBOE Equity Put/Call ratio history.
    Sources:
    1. CBOE CDN CSV: Nov 2006 – Oct 2019
    2. CBOE daily JSON API: Oct 2019 – present (parallel fetched)
    Returns DataFrame with columns: DATE (index), equity_pc, pc_ma10, pc_ma20
    """
    now = time.time()
    if PCR_CACHE["data"] is not None and (now - PCR_CACHE["time"]) < PCR_CACHE_TTL:
        return PCR_CACHE["data"]

    try:
        # --- Part 1: Historical CSV (2006-Oct 2019) ---
        cboe_headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cboe.com/us/options/market_statistics/historical_data/"
        }
        r = requests.get(
            "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
            headers=cboe_headers, timeout=15
        )
        if r.status_code != 200:
            raise ValueError(f"CBOE CSV returned {r.status_code}")

        import io as _io
        df_old = pd.read_csv(_io.StringIO(r.text), skiprows=2)
        df_old.columns = df_old.columns.str.strip()
        df_old["DATE"] = pd.to_datetime(df_old["DATE"], format="%m/%d/%Y", errors="coerce")
        df_old = df_old.dropna(subset=["DATE"])[["DATE", "P/C Ratio"]].rename(
            columns={"P/C Ratio": "equity_pc"}
        )
        df_old = df_old.sort_values("DATE").reset_index(drop=True)

        # --- Part 2: Daily API (Oct 2019 – present) ---
        csv_end = pd.Timestamp("2019-10-04")
        start_new = csv_end + pd.Timedelta(days=1)
        end_new = pd.Timestamp.today().normalize()
        all_dates = list(pd.bdate_range(start_new, end_new))

        daily_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cboe.com/"}

        from concurrent.futures import ThreadPoolExecutor as _TPE

        def _fetch_day(d):
            date_str = d.strftime("%Y-%m-%d")
            url = f"https://cdn.cboe.com/data/us/options/market_statistics/daily/{date_str}_daily_options"
            # Retry a couple of times — the CBOE CDN intermittently 403s at the
            # edge even for days that exist. A short retry avoids silently
            # dropping the freshest print (which matters for a daily indicator).
            for attempt in range(3):
                try:
                    resp = requests.get(url, headers=daily_headers, timeout=6)
                    if resp.status_code == 200:
                        j = resp.json()
                        equity = next(
                            (float(x["value"]) for x in j.get("ratios", [])
                             if "EQUITY PUT" in x.get("name", "")),
                            None
                        )
                        if equity is not None:
                            return {"DATE": d, "equity_pc": equity}
                        return None  # day exists but no equity ratio — don't retry
                    if resp.status_code == 404:
                        return None  # day genuinely not published — don't retry
                    # 403/5xx — transient edge block, retry after a brief pause
                except Exception:
                    pass
                time.sleep(0.4 * (attempt + 1))
            return None

        with _TPE(max_workers=4) as ex:  # FIX: reduced from 8
            fetch_results = list(ex.map(_fetch_day, all_dates))

        new_rows = [row for row in fetch_results if row is not None]
        if new_rows:
            df_new = pd.DataFrame(new_rows).sort_values("DATE").reset_index(drop=True)
        else:
            df_new = pd.DataFrame(columns=["DATE", "equity_pc"])

        # --- Merge ---
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.sort_values("DATE").drop_duplicates("DATE").reset_index(drop=True)
        df_all = df_all.set_index("DATE")
        df_all["pc_ma5"]  = df_all["equity_pc"].rolling(5).mean()
        df_all["pc_ma10"] = df_all["equity_pc"].rolling(10).mean()
        df_all["pc_ma20"] = df_all["equity_pc"].rolling(20).mean()

        PCR_CACHE["data"] = df_all
        PCR_CACHE["time"] = time.time()
        return df_all

    except Exception as e:
        print(f"[PCR] fetch_pcr_history error: {e}")
        return None


def fetch_cboe_etf_pcr(ticker: str) -> Optional[dict]:
    """
    Fetch current PCR from CBOE's delayed quote API for a given ETF ticker.
    Returns {pcr_oi, pcr_vol, total_oi, n_strikes} or None.
    Cached per-ticker for PCR_ETF_CACHE_TTL seconds.
    """
    now = time.time()
    cached = PCR_ETF_CACHE.get(ticker)
    if cached and (now - cached["time"]) < PCR_ETF_CACHE_TTL:
        return cached["data"]

    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
    hdrs = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cboe.com/",
            "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs, timeout=10)
        if r.status_code != 200:
            return None
        options = r.json().get("data", {}).get("options", [])
        if not options:
            return None

        p_oi = c_oi = p_vol = c_vol = 0
        for o in options:
            sym = o.get("option", "")
            if len(sym) < 9:
                continue
            flag = sym[-9]
            oi  = o.get("open_interest", 0) or 0
            vol = o.get("volume", 0) or 0
            if flag == "P":
                p_oi += oi; p_vol += vol
            elif flag == "C":
                c_oi += oi; c_vol += vol

        if c_oi == 0:
            return None
        result = {
            "pcr_oi":  round(p_oi / c_oi,  3),
            "pcr_vol": round(p_vol / c_vol, 3) if c_vol > 0 else None,
            "total_oi": int(p_oi + c_oi),
            "n_strikes": len(options),
        }
        PCR_ETF_CACHE[ticker] = {"data": result, "time": now}
        return result
    except Exception as e:
        print(f"[PCR-ETF] {ticker} error: {e}")
        return None


def fetch_deribit_pcr(currency: str) -> Optional[dict]:
    """
    Fetch current PCR from Deribit for BTC or ETH.
    Returns {pcr_oi, pcr_vol, total_oi} or None.
    """
    now = time.time()
    cached = PCR_DERIBIT_CACHE.get(currency)
    if cached and (now - cached["time"]) < PCR_DERIBIT_CACHE_TTL:
        return cached["data"]

    try:
        BASE = "https://www.deribit.com/api/v2"
        r = requests.get(f"{BASE}/public/get_book_summary_by_currency",
                         params={"currency": currency, "kind": "option"}, timeout=12)
        data = r.json().get("result", [])
        p_oi  = sum(x.get("open_interest", 0) for x in data if x.get("instrument_name","").endswith("-P"))
        c_oi  = sum(x.get("open_interest", 0) for x in data if x.get("instrument_name","").endswith("-C"))
        p_vol = sum(x.get("volume", 0)        for x in data if x.get("instrument_name","").endswith("-P"))
        c_vol = sum(x.get("volume", 0)        for x in data if x.get("instrument_name","").endswith("-C"))
        if c_oi == 0:
            return None
        result = {
            "pcr_oi":  round(p_oi / c_oi,  3),
            "pcr_vol": round(p_vol / c_vol, 3) if c_vol > 0 else None,
            "total_oi": int(p_oi + c_oi),
        }
        PCR_DERIBIT_CACHE[currency] = {"data": result, "time": now}
        return result
    except Exception as e:
        print(f"[PCR-Deribit] {currency} error: {e}")
        return None


def score_pcr(market_id: str) -> dict:
    """
    Score the put/call ratio for all supported markets.
    
    Sources:
      Equities (ES/NQ/YM/RTY): CBOE aggregate equity PCR (daily, with 20-day MA)
      Metals (GC/SI) & Oil (CL): CBOE delayed ETF option chain (GLD/SLV/USO)
      Crypto (BTC/ETH): Deribit public API
    
    Scoring approach (from backtest):
      - FEAR extreme (high PCR/high percentile) = contrarian BULL
      - GREED extreme (low PCR/low percentile) = contrarian BEAR (equities only — 
        all markets score both fear and greed signals bidirectionally
      - Tiers control weight contribution (see PCR_TIERS)
    
    Returns dict with score (0-10), label, detail, tier.
    Unsupported markets return score=5 (neutral).
    """
    if market_id not in PCR_ALL_SYMBOLS:
        return {"score": 5.0, "label": "N/A", "tier": 0,
                "detail": {"reason": "PCR not available for this market"}}

    tier_cfg = PCR_TIERS.get(market_id, {})
    tier = tier_cfg.get("tier", 0)
    source = tier_cfg.get("source", "")

    # ── EQUITY: use LATEST DAILY CBOE aggregate equity P/C ratio ────────────
    # FAST-REACTING redesign: score off the newest daily print (no MA smoothing)
    # so a sharp same-day swing into fear/greed registers immediately. Extremes
    # are calibrated against the trailing ~1yr distribution of DAILY prints, and
    # a z-score flags how far the current print sits from its recent normal —
    # so a big one-day jump reads as extreme even before percentile saturates.
    if source == "cboe_equity":
        df = fetch_pcr_history()
        if df is None or df.empty:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": "Could not fetch P/C ratio data"}}

        df_clean = df.dropna(subset=["equity_pc"])
        if df_clean.empty:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": "No daily P/C values"}}

        latest        = df_clean.iloc[-1]
        current_daily = float(latest["equity_pc"])
        # prior print for a same-day delta / "today's move" readout
        prev_daily    = float(df_clean["equity_pc"].iloc[-2]) if len(df_clean) >= 2 else current_daily
        daily_change  = current_daily - prev_daily
        # short context MA (display only — NOT used for scoring)
        ma5_series    = df_clean["equity_pc"].rolling(5).mean()
        current_ma5   = float(ma5_series.iloc[-1]) if not pd.isna(ma5_series.iloc[-1]) else current_daily
        latest_date   = str(df_clean.index[-1].date())

        # Regime window: trailing ~1yr of DAILY prints (calibrate extremes to
        # the current environment, not the 2008/2020 tails).
        daily_all  = df_clean["equity_pc"].values
        window     = daily_all[-252:] if len(daily_all) >= 60 else daily_all
        # Percentile of the latest DAILY print vs the trailing window
        percentile = float(np.mean(window < current_daily))
        # z-score: how far today's print is from its recent normal (fast flag)
        w_mean = float(np.mean(window))
        w_std  = float(np.std(window)) or 1e-6
        zscore = (current_daily - w_mean) / w_std

        # Score blends percentile position (0-10) with a z-score nudge so a
        # sharp same-day spike pushes toward the extreme faster than percentile
        # alone. Clamped to 0-10.
        base_score = percentile * 10.0
        z_nudge    = max(-1.5, min(1.5, zscore)) * 0.8  # up to ±1.2 pts
        score      = round(max(0.0, min(10.0, base_score + z_nudge)), 1)

        # Labels driven by BOTH percentile and z-score, so a big absolute jump
        # earns an "extreme" tag even if the 1yr percentile is still climbing.
        hot_fear  = (percentile >= 0.90) or (zscore >= 1.5)
        hot_greed = (percentile <= 0.10) or (zscore <= -1.5)
        if   hot_fear:            label = "Extreme Fear"
        elif percentile >= 0.75:  label = "High Fear"
        elif percentile >= 0.60:  label = "Mild Fear"
        elif percentile >= 0.40:  label = "Neutral"
        elif percentile >= 0.25:  label = "Mild Greed"
        elif not hot_greed:       label = "High Greed"
        else:                     label = "Extreme Greed"

        if   percentile >= 0.75 or hot_fear:   signal = "Contrarian Bullish"
        elif percentile >= 0.60:               signal = "Lean Bullish"
        elif percentile <= 0.25 or hot_greed:  signal = "Contrarian Bearish"
        elif percentile <= 0.40:               signal = "Lean Bearish"
        else:                                  signal = "Neutral"

        return {
            "score": score, "label": label, "tier": tier,
            "detail": {
                "current_daily": round(current_daily, 3),
                "prev_daily":    round(prev_daily, 3),
                "daily_change":  round(daily_change, 3),
                "ma5":  round(current_ma5,  3),  # display context only
                "percentile": round(percentile * 100, 1),
                "zscore": round(zscore, 2),
                "signal": signal, "label": label,
                "latest_date": latest_date,
                "source": "CBOE Aggregate Equity P/C",
                "scoring_basis": "latest daily print (1yr percentile + z-score)",
                "thresholds": {
                    "extreme_greed":  round(float(np.percentile(window, 10)), 3),
                    "moderate_greed": round(float(np.percentile(window, 25)), 3),
                    "neutral_low":    round(float(np.percentile(window, 40)), 3),
                    "neutral_high":   round(float(np.percentile(window, 60)), 3),
                    "moderate_fear":  round(float(np.percentile(window, 75)), 3),
                    "extreme_fear":   round(float(np.percentile(window, 90)), 3),
                }
            }
        }

    # ── ETF-BASED PCR (GC→GLD, SI→SLV, CL→USO) ──────────────────────────
    elif source == "cboe_etf":
        proxy_ticker = PCR_CBOE_PROXY.get(market_id)
        if not proxy_ticker:
            return {"score": 5.0, "label": "N/A", "tier": tier,
                    "detail": {"reason": "No CBOE proxy configured"}}

        snap = fetch_cboe_etf_pcr(proxy_ticker)
        if snap is None:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": f"Could not fetch {proxy_ticker} option chain"}}

        pcr = snap["pcr_oi"]

        # ── Regime-relative scoring (aligned with the P/C history chart) ────
        # Static absolute thresholds mis-fire when an option market is
        # structurally call- or put-heavy for long stretches (e.g. GLD in a
        # gold bull run never prints PCR > 0.8, so "fear" was unreachable and
        # the factor skewed permanently bearish while the chart showed puts at
        # the top of their 1yr range). Score off the percentile + z-score of
        # the live PCR vs the market's OWN trailing-year distribution — the
        # exact series the chart plots — and only fall back to the legacy
        # absolute thresholds if the history is too thin.
        window = []
        try:
            _series, _sr = _build_etf_pcr_series(market_id.upper(), refresh=False)
            # If no direct ticker snapshot is cached yet (e.g. right after a
            # deploy wiped the disk cache), the series is unscaled ETP proxy on
            # a different level — refresh once to anchor the scale.
            if _sr == 1.0:
                _series2, _sr2 = _build_etf_pcr_series(market_id.upper(), refresh=True)
                if _series2:
                    _series, _sr = _series2, _sr2
            window = [v for _, v in _series][-252:]
        except Exception as e:
            print(f"[PCR] {market_id} relative-scoring history load failed: {e}")

        # Scale-compatibility guard: if the live PCR sits wildly off the
        # window's level, the backfill is unanchored — use absolute fallback.
        if len(window) >= 60:
            _med = float(np.median(window))
            if _med <= 0 or not (0.4 <= pcr / _med <= 2.5):
                print(f"[PCR] {market_id} window scale mismatch (pcr={pcr}, med={_med:.3f}) — absolute fallback")
                window = []

        if len(window) >= 60:
            w = np.array(window, dtype=float)
            percentile = float(np.mean(w < pcr))
            w_mean = float(w.mean())
            w_std  = float(w.std()) or 1e-6
            zscore = (pcr - w_mean) / w_std

            base_score = percentile * 10.0
            z_nudge    = max(-1.5, min(1.5, zscore)) * 0.8  # up to ±1.2 pts
            score      = round(max(0.0, min(10.0, base_score + z_nudge)), 1)

            hot_fear  = (percentile >= 0.90) or (zscore >= 1.5)
            hot_greed = (percentile <= 0.10) or (zscore <= -1.5)
            if   hot_fear:            label = "Extreme Fear"
            elif percentile >= 0.75:  label = "High Fear"
            elif percentile >= 0.60:  label = "Mild Fear"
            elif percentile >= 0.40:  label = "Neutral"
            elif percentile >= 0.25:  label = "Mild Greed"
            elif not hot_greed:       label = "High Greed"
            else:                     label = "Extreme Greed"

            if   percentile >= 0.75 or hot_fear:   signal = "Contrarian Bullish"
            elif percentile >= 0.60:               signal = "Lean Bullish"
            elif percentile <= 0.25 or hot_greed:  signal = "Contrarian Bearish"
            elif percentile <= 0.40:               signal = "Lean Bearish"
            else:                                  signal = "Neutral"

            return {
                "score": score, "label": label, "tier": tier,
                "detail": {
                    "pcr_oi": pcr,
                    "pcr_vol": snap.get("pcr_vol"),
                    "total_oi": snap["total_oi"],
                    "n_strikes": snap.get("n_strikes"),
                    "percentile": round(percentile * 100, 1),
                    "zscore": round(zscore, 2),
                    "signal": signal, "label": label,
                    "proxy_ticker": proxy_ticker,
                    "source": f"CBOE {proxy_ticker} Options",
                    "scoring_basis": "live OI PCR vs own 1yr distribution (percentile + z-score)",
                    "thresholds": {
                        "extreme_greed":  round(float(np.percentile(w, 10)), 3),
                        "moderate_greed": round(float(np.percentile(w, 25)), 3),
                        "neutral_low":    round(float(np.percentile(w, 40)), 3),
                        "neutral_high":   round(float(np.percentile(w, 60)), 3),
                        "moderate_fear":  round(float(np.percentile(w, 75)), 3),
                        "extreme_fear":   round(float(np.percentile(w, 90)), 3),
                    },
                }
            }

        # ── Fallback: legacy calibrated absolute thresholds (thin history) ─
        THRESHOLDS = {
            "GLD": {"xfear": 1.20, "hfear": 1.00, "mfear": 0.80,
                    "mgreed": 0.55, "hgreed": 0.45, "xgreed": 0.35},
            "SLV": {"xfear": 1.30, "hfear": 1.10, "mfear": 0.90,
                    "mgreed": 0.50, "hgreed": 0.40, "xgreed": 0.30},
            "USO": {"xfear": 2.00, "hfear": 1.70, "mfear": 1.40,
                    "mgreed": 1.00, "hgreed": 0.80, "xgreed": 0.60},
        }
        th = THRESHOLDS.get(proxy_ticker, THRESHOLDS["GLD"])

        score_bear_ok = True
        if   pcr >= th["xfear"]:  label = "Extreme Fear";  score = 9.5
        elif pcr >= th["hfear"]:  label = "High Fear";     score = 8.0
        elif pcr >= th["mfear"]:  label = "Mild Fear";     score = 6.5
        elif pcr >= th["mgreed"]: label = "Neutral";       score = 5.0
        elif pcr >= th["hgreed"]: label = "Mild Greed";    score = 3.5
        elif pcr >= th["xgreed"]: label = "High Greed";    score = 2.0
        else:                     label = "Extreme Greed"; score = 0.5


        if   score >= 8.0: signal = "Contrarian Bullish"
        elif score >= 6.5: signal = "Lean Bullish"
        elif score <= 2.0: signal = "Contrarian Bearish"
        elif score <= 3.5: signal = "Lean Bearish"
        else:              signal = "Neutral"

        return {
            "score": round(score, 1), "label": label, "tier": tier,
            "detail": {
                "pcr_oi": pcr,
                "pcr_vol": snap.get("pcr_vol"),
                "total_oi": snap["total_oi"],
                "n_strikes": snap.get("n_strikes"),
                "signal": signal,
                "proxy_ticker": proxy_ticker,
                "source": f"CBOE {proxy_ticker} Options",
                "thresholds": th,
            }
        }

    # ── CRYPTO PCR (BTC/ETH via Deribit) ─────────────────────────────────
    elif source == "deribit":
        currency = "BTC" if market_id == "BTC" else "ETH"
        snap = fetch_deribit_pcr(currency)
        if snap is None:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": f"Could not fetch {currency} Deribit data"}}

        pcr = snap["pcr_oi"]

        # Crypto options are structurally CALL-heavy (speculation bias)
        # Normal BTC PCR_OI ~0.40-0.80. Fear >1.0. Greed <0.35.
        # ETH is even more call-heavy — lower thresholds.
        if currency == "BTC":
            th = {"xfear": 1.00, "hfear": 0.85, "mfear": 0.70,
                  "mgreed": 0.55, "hgreed": 0.45, "xgreed": 0.35}
        else:
            th = {"xfear": 0.80, "hfear": 0.65, "mfear": 0.55,
                  "mgreed": 0.42, "hgreed": 0.35, "xgreed": 0.28}

        if   pcr >= th["xfear"]:  label = "Extreme Fear";  score = 9.5
        elif pcr >= th["hfear"]:  label = "High Fear";     score = 8.0
        elif pcr >= th["mfear"]:  label = "Mild Fear";     score = 6.5
        elif pcr >= th["mgreed"]: label = "Neutral";       score = 5.0
        elif pcr >= th["hgreed"]: label = "Mild Greed";    score = 3.5
        elif pcr >= th["xgreed"]: label = "High Greed";    score = 2.0
        else:                     label = "Extreme Greed"; score = 0.5

        if   score >= 8.0: signal = "Contrarian Bullish"
        elif score >= 6.5: signal = "Lean Bullish"
        elif score <= 2.0: signal = "Contrarian Bearish"
        elif score <= 3.5: signal = "Lean Bearish"
        else:              signal = "Neutral"

        return {
            "score": round(score, 1), "label": label, "tier": tier,
            "detail": {
                "pcr_oi": pcr,
                "pcr_vol": snap.get("pcr_vol"),
                "total_oi": snap["total_oi"],
                "signal": signal,
                "exchange": "Deribit",
                "source": f"Deribit {currency} Options",
                "thresholds": th,
                "note": "Retail contrarian. Crypto options are structurally call-heavy.",
            }
        }

    return {"score": 5.0, "label": "N/A", "tier": tier,
            "detail": {"reason": "Unknown PCR source"}}


# ============================================================
# ICE EUROPE COT DATA
# ============================================================

# Store ICE cache alongside the app in DATA_DIR — persists across container
# restarts (unlike /tmp which is ephemeral). On full redeploys the disk is wiped
# but the startup event will re-fetch, and the Friday cron re-injects weekly.
_ICE_DISK_CACHE_DIR = os.path.join(DATA_DIR, "ice_cot_cache")
os.makedirs(_ICE_DISK_CACHE_DIR, exist_ok=True)

_ICE_MEM_CACHE: dict = {}
_ICE_MEM_CACHE_TTL = 3600 * 6  # 6h


def _ice_disk_cache_path(ice_market_code: str) -> str:
    return os.path.join(_ICE_DISK_CACHE_DIR, f"{ice_market_code}.pkl")


def _save_ice_to_disk(ice_market_code: str, df) -> None:
    try:
        import pickle
        with open(_ice_disk_cache_path(ice_market_code), "wb") as fh:
            pickle.dump(df, fh)
    except Exception:
        pass


def _load_ice_from_disk(ice_market_code: str):
    try:
        import pickle
        p = _ice_disk_cache_path(ice_market_code)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    return None


_ICE_DISK_MAX_AGE_DAYS = 12  # disk-cached data older than this triggers a live refetch attempt


def _ice_df_age_days(df) -> float:
    """Days since the newest COT row in the frame (9999 on any error)."""
    try:
        return float((pd.Timestamp.now() - pd.to_datetime(df["date"]).max()).days)
    except Exception:
        return 9999.0


def _ice_merge_frames(old_df, new_df):
    """Union two ICE COT frames by date. On duplicate dates the FRESHER frame
    (later max date) wins. Never lets a stale fetch shrink or roll back history."""
    if old_df is None or getattr(old_df, "empty", True):
        return new_df
    if new_df is None or getattr(new_df, "empty", True):
        return old_df
    try:
        if pd.to_datetime(new_df["date"]).max() >= pd.to_datetime(old_df["date"]).max():
            frames = [old_df, new_df]   # keep='last' -> new wins on duplicate dates
        else:
            frames = [new_df, old_df]   # old frame is fresher -> old wins
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        return merged
    except Exception:
        return new_df if len(new_df) >= len(old_df) else old_df


def _ice_store_merged(ice_market_code: str, df):
    """Merge df with whatever is currently cached (mem, then disk) and store the
    union to BOTH mem and disk. Closes the race where a slow boot-time live fetch
    lands after an injection and would otherwise overwrite fresher data."""
    cur = None
    c = _ICE_MEM_CACHE.get(ice_market_code)
    if c is not None:
        cur = c.get("df")
    if cur is None or getattr(cur, "empty", True):
        cur = _load_ice_from_disk(ice_market_code)
    merged = _ice_merge_frames(cur, df)
    _ICE_MEM_CACHE[ice_market_code] = {"df": merged, "ts": time.time()}
    _save_ice_to_disk(ice_market_code, merged)
    return merged


def _ice_disk_mtime(ice_market_code: str) -> float:
    """mtime of the shared disk cache file (0.0 if missing)."""
    try:
        return os.path.getmtime(_ice_disk_cache_path(ice_market_code))
    except OSError:
        return 0.0


def _ice_mem_lookup(ice_market_code: str, now: float):
    """Return the mem-cached frame if still valid. Render runs multiple worker
    processes sharing ONE disk: if another worker wrote a newer disk file since
    this worker cached in memory (e.g. a cron injection), merge the disk data
    in so every worker converges on the freshest history within seconds."""
    cached = _ICE_MEM_CACHE.get(ice_market_code)
    if not cached or (now - cached["ts"]) >= _ICE_MEM_CACHE_TTL:
        return None
    if _ice_disk_mtime(ice_market_code) > cached["ts"] + 1.0:
        disk_df = _load_ice_from_disk(ice_market_code)
        if disk_df is not None and not disk_df.empty:
            merged = _ice_merge_frames(cached["df"], disk_df)
            _ICE_MEM_CACHE[ice_market_code] = {"df": merged, "ts": now}
            return merged
    return cached["df"]


def _fetch_ice_fin_cot_raw(ice_market_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch ICE Europe TFF (Traders in Financial Futures) COT data for Z (FTSE 100)
    and R (Long Gilt). Uses EUFINCOTHist{year}.csv annual files.

    TFF group mapping (ice_fin=True markets):
      comm_net  = Leveraged Fund net  (HF / fast money — primary directional signal)
      lspec_net = Asset Manager net   (institutional — structural / confirming signal)
      sspec_net = Non-Reportable net  (retail)
    NB: Dealer Intermediary is excluded (mostly hedging / balance sheet).

    History: Z from Dec 2024, R from Mar 2025 — thin but usable with dampened scoring.
    """
    import csv as _csv
    import io as _io

    now = time.time()
    _mem_df = _ice_mem_lookup(ice_market_code, now)
    if _mem_df is not None:
        return _mem_df

    disk_df = _load_ice_from_disk(ice_market_code)
    if disk_df is not None and not disk_df.empty and _ice_df_age_days(disk_df) <= _ICE_DISK_MAX_AGE_DAYS:
        _ICE_MEM_CACHE[ice_market_code] = {"df": disk_df, "ts": now}
        return disk_df
    # Disk data missing or stale (> _ICE_DISK_MAX_AGE_DAYS old) -> attempt live refetch;
    # stale disk_df is retained as merge base / fallback.

    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.ice.com/report/122",
    }

    all_rows = []
    current_year = date.today().year
    START_YEAR = 2024  # Z available from Dec 2024, R from Mar 2025
    consecutive_failures = 0

    for year in range(START_YEAR, current_year + 1):
        url = f"https://www.ice.com/publicdocs/futures/EUFINCOTHist{year}.csv"
        try:
            time.sleep(0.4)
            _req = requests.get(url, timeout=30, headers=_headers)
            if _req.status_code == 429:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    print(f"[ICE FIN COT] {ice_market_code}: rate limited, aborting")
                    break
                time.sleep(10)
                _req = requests.get(url, timeout=30, headers=_headers)
                if _req.status_code != 200:
                    break
            if _req.status_code != 200:
                print(f"[ICE FIN COT] {ice_market_code}: HTTP {_req.status_code} for {year}")
                consecutive_failures += 1
                continue
            consecutive_failures = 0
            content = _req.content.decode("utf-8-sig")
            if "<!doctype" in content[:200].lower():
                consecutive_failures += 1
                continue

            reader = _csv.DictReader(_io.StringIO(content))
            rows = list(reader)
            # Filter to target market code, FutOnly only
            _g  = [r for r in rows if r.get("CFTC_Commodity_Code", "").strip() == ice_market_code]
            _gs = [r for r in _g  if r.get("FutOnly_or_Combined", "") == "FutOnly"]
            if not _gs and _g:
                _gs = _g
            all_rows.extend(_gs)
            print(f"[ICE FIN COT] {ice_market_code} {year}: {len(_gs)} rows")
        except Exception as e:
            print(f"[ICE FIN COT] {ice_market_code} {year}: {e}")
            continue

    if not all_rows:
        disk_fallback = _load_ice_from_disk(ice_market_code)
        if disk_fallback is not None and not disk_fallback.empty:
            _ICE_MEM_CACHE[ice_market_code] = {"df": disk_fallback, "ts": now}
            return disk_fallback
        print(f"[ICE FIN COT] {ice_market_code}: no data found")
        return None

    try:
        df = pd.DataFrame(all_rows)

        # Parse date
        df["date"] = pd.to_datetime(df.get("As_of_Date_Form_MM/DD/YYYY", pd.Series(dtype=str)), errors="coerce")
        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "As_of_Date_In_Form_YYMMDD"], format="%y%m%d", errors="coerce"
            )
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # Open interest
        df["open_interest_all"] = pd.to_numeric(
            df.get("Open_Interest_All", pd.Series(dtype=float)), errors="coerce"
        )

        # TFF group mapping:
        #   comm_net  = Leveraged Fund (fast money — directional signal for fin markets)
        #   lspec_net = Asset Manager  (institutional / structural)
        #   sspec_net = Non-Reportable (retail)
        lf_long  = pd.to_numeric(df.get("Leveraged_Fund_Long_All",  pd.Series(dtype=float)), errors="coerce")
        lf_short = pd.to_numeric(df.get("Leveraged_Fund_Short_All", pd.Series(dtype=float)), errors="coerce")
        am_long  = pd.to_numeric(df.get("Asset_Manager_Long_All",   pd.Series(dtype=float)), errors="coerce")
        am_short = pd.to_numeric(df.get("Asset_Manager_Short_All",  pd.Series(dtype=float)), errors="coerce")
        nr_long  = pd.to_numeric(df.get("NonRept_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce")
        nr_short = pd.to_numeric(df.get("NonRept_Positions_Short_All", pd.Series(dtype=float)), errors="coerce")

        # Store raw columns for completeness
        df["comm_positions_long_all"]    = lf_long
        df["comm_positions_short_all"]   = lf_short
        df["noncomm_positions_long_all"] = am_long
        df["noncomm_positions_short_all"]= am_short
        df["nonrept_positions_long_all"] = nr_long
        df["nonrept_positions_short_all"]= nr_short

        df["comm_net"]  = lf_long  - lf_short   # Leveraged Fund net
        df["lspec_net"] = am_long  - am_short    # Asset Manager net
        df["sspec_net"] = nr_long  - nr_short    # Non-Reportable net
        df["lspec_chg"] = df["lspec_net"].diff().fillna(0)

        df = df[["date","comm_net","lspec_net","sspec_net","lspec_chg",
                 "comm_positions_long_all","comm_positions_short_all",
                 "noncomm_positions_long_all","noncomm_positions_short_all",
                 "nonrept_positions_long_all","nonrept_positions_short_all",
                 "open_interest_all"]].dropna(subset=["comm_net"])

        df = df.sort_values("date").reset_index(drop=True)

        # Deduplicate (keep latest per date in case of combined/futonly overlap)
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        print(f"[ICE FIN COT] {ice_market_code}: {len(df)} rows fetched, "
              f"date range {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

        # Merge with any existing cached history (mem/disk) so a stale or partial
        # live fetch can never roll back or shrink previously known data.
        df = _ice_store_merged(ice_market_code, df)
        return df

    except Exception as e:
        print(f"[ICE FIN COT] parse error {ice_market_code}: {e}")
        return None


# ── Shared annual file cache ──────────────────────────────────────────────────
# Each COTHist{year}.csv contains ALL ICE markets. One download per year
# serves all markets — eliminates redundant 400KB downloads per market.
_ICE_ANNUAL_ROW_CACHE: dict = {}
_ICE_ANNUAL_CACHE_LOCK = threading.Lock()


def _fetch_ice_annual_rows(year: int) -> list:
    """Fetch and parse one year's COTHist CSV. Results cached in memory (shared across markets)."""
    import csv as _csv, io as _io
    with _ICE_ANNUAL_CACHE_LOCK:
        if year in _ICE_ANNUAL_ROW_CACHE:
            return _ICE_ANNUAL_ROW_CACHE[year]

    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.ice.com/report/122",
    }
    url = f"https://www.ice.com/publicdocs/futures/COTHist{year}.csv"
    try:
        resp = requests.get(url, timeout=30, headers=_headers)
        if resp.status_code != 200:
            print(f"[ICE ANNUAL] {year}: HTTP {resp.status_code}")
            return []
        raw = resp.content.decode("utf-8-sig")
        if "<!doctype" in raw[:200].lower():
            print(f"[ICE ANNUAL] {year}: HTML/blocked response")
            return []
        reader = _csv.DictReader(_io.StringIO(raw))
        all_rows = list(reader)
        # Keep FutOnly; fall back to all rows if FutOnly subset is absent
        fut_rows = [r for r in all_rows if r.get("FutOnly_or_Combined", "") == "FutOnly"]
        result = fut_rows if fut_rows else all_rows
        print(f"[ICE ANNUAL] {year}: {len(result)} FutOnly rows ({len(all_rows)} total)")
        with _ICE_ANNUAL_CACHE_LOCK:
            _ICE_ANNUAL_ROW_CACHE[year] = result
        return result
    except Exception as e:
        print(f"[ICE ANNUAL] {year}: {e}")
        return []


def _fetch_ice_cot_raw(ice_market_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch ICE Europe COT data for the given market code.
    Downloads each year's COTHist CSV once (shared across all ICE markets via
    _ICE_ANNUAL_ROW_CACHE) using parallel ThreadPoolExecutor fetches, then
    filters to the target CFTC_Commodity_Code.
    Z / R (TFF markets) are routed to _fetch_ice_fin_cot_raw.
    """
    import csv as _csv
    import io as _io

    now = time.time()
    _mem_df = _ice_mem_lookup(ice_market_code, now)
    if _mem_df is not None:
        return _mem_df

    # Disk cache (survives warm restarts) — only trusted while reasonably fresh.
    disk_df = _load_ice_from_disk(ice_market_code)
    if disk_df is not None and not disk_df.empty and _ice_df_age_days(disk_df) <= _ICE_DISK_MAX_AGE_DAYS:
        _ICE_MEM_CACHE[ice_market_code] = {"df": disk_df, "ts": now}
        return disk_df
    # Disk data missing or stale -> fall through to live fetch (stale disk kept as fallback).

    # TFF markets (FTSE100, Long Gilt) use the financial futures series
    if ice_market_code in ("Z", "R"):
        return _fetch_ice_fin_cot_raw(ice_market_code)

    current_year = date.today().year
    START_YEAR = 2011  # ICE disagg available from ~2011

    # Parallel fetch — all years downloaded concurrently, each file fetched once
    years = list(range(START_YEAR, current_year + 1))
    all_rows: list = []
    # Inline pool: short-lived, fetches ICE annual CSVs in parallel.
    # Capped at 3 to avoid thread spike when called for multiple markets simultaneously.
    with _cf.ThreadPoolExecutor(max_workers=3) as _pool:
        year_results = list(_pool.map(_fetch_ice_annual_rows, years))
    for year_rows in year_results:
        matched = [r for r in year_rows if r.get("CFTC_Commodity_Code", "").strip() == ice_market_code]
        all_rows.extend(matched)

    if not all_rows:
        # Fall back to stale disk cache on complete fetch failure
        disk_fallback = _load_ice_from_disk(ice_market_code)
        if disk_fallback is not None and not disk_fallback.empty:
            print(f"[ICE COT] {ice_market_code}: using stale disk cache ({len(disk_fallback)} rows)")
            _ICE_MEM_CACHE[ice_market_code] = {"df": disk_fallback, "ts": now}
            return disk_fallback
        print(f"[ICE COT] {ice_market_code}: no rows found across all years")
        return None

    # ── Convert rows → DataFrame ──────────────────────────────────────────────
    try:
        df = pd.DataFrame(all_rows)

        # Parse date (prefer MM/DD/YYYY, fall back to YYMMDD)
        df["date"] = pd.to_datetime(
            df.get("As_of_Date_Form_MM/DD/YYYY", pd.Series(dtype=str)), errors="coerce"
        )
        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "As_of_Date_In_Form_YYMMDD"], format="%y%m%d", errors="coerce"
            )
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        df["open_interest_all"] = pd.to_numeric(
            df.get("Open_Interest_All", pd.Series(dtype=float)), errors="coerce"
        )

        # ICE disagg group mapping:
        #   comm_net  = Prod/Merc + Swap (hedgers — primary commercial signal)
        #   lspec_net = Managed Money    (hedge funds — directional signal)
        #   sspec_net = Non-Reportable   (retail)
        pm_long  = pd.to_numeric(df.get("Prod_Merc_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce").fillna(0)
        pm_short = pd.to_numeric(df.get("Prod_Merc_Positions_Short_All", pd.Series(dtype=float)), errors="coerce").fillna(0)
        sw_long  = pd.to_numeric(df.get("Swap_Positions_Long_All",       pd.Series(dtype=float)), errors="coerce").fillna(0)
        sw_short = pd.to_numeric(df.get("Swap_Positions_Short_All",      pd.Series(dtype=float)), errors="coerce").fillna(0)
        mm_long  = pd.to_numeric(df.get("M_Money_Positions_Long_All",    pd.Series(dtype=float)), errors="coerce").fillna(0)
        mm_short = pd.to_numeric(df.get("M_Money_Positions_Short_All",   pd.Series(dtype=float)), errors="coerce").fillna(0)
        nr_long  = pd.to_numeric(df.get("NonRept_Positions_Long_All",    pd.Series(dtype=float)), errors="coerce").fillna(0)
        nr_short = pd.to_numeric(df.get("NonRept_Positions_Short_All",   pd.Series(dtype=float)), errors="coerce").fillna(0)

        df["comm_positions_long_all"]    = pm_long + sw_long
        df["comm_positions_short_all"]   = pm_short + sw_short
        df["noncomm_positions_long_all"] = mm_long
        df["noncomm_positions_short_all"]= mm_short
        df["nonrept_positions_long_all"] = nr_long
        df["nonrept_positions_short_all"]= nr_short

        df["comm_net"]  = (pm_long + sw_long)  - (pm_short + sw_short)
        df["lspec_net"] = mm_long - mm_short
        df["sspec_net"] = nr_long - nr_short
        df["lspec_chg"] = df["lspec_net"].diff().fillna(0)

        df = df[["date","comm_net","lspec_net","sspec_net","lspec_chg",
                 "comm_positions_long_all","comm_positions_short_all",
                 "noncomm_positions_long_all","noncomm_positions_short_all",
                 "nonrept_positions_long_all","nonrept_positions_short_all",
                 "open_interest_all"]].dropna(subset=["comm_net"])

        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

        print(f"[ICE COT] {ice_market_code}: {len(df)} rows, "
              f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

        # Merge with any existing cached history (mem/disk) so a stale or partial
        # live fetch can never roll back or shrink previously known data.
        df = _ice_store_merged(ice_market_code, df)
        return df

    except Exception as e:
        print(f"[ICE COT] parse error {ice_market_code}: {e}")
        return None

async def fetch_ice_cot_history(ice_code: str) -> Optional[pd.DataFrame]:
    """Async wrapper for _fetch_ice_cot_raw — matches signature of fetch_cot_history()."""
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    return await loop.run_in_executor(_APP_EXECUTOR, _fetch_ice_cot_raw, ice_code)


# ============================================================
# COT DATA FETCHING
# ============================================================

COT_CACHE = {}
COT_CACHE_TIME = {}
COT_CACHE_TTL = 3600 * 6  # 6 hours

CFTC_API = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"


async def fetch_cot_history(cftc_code: str, name_hint: str = "") -> Optional[pd.DataFrame]:
    """
    Fetch full COT history from CFTC public API back to 2008 (~950 rows, 18+ years).
    Uses pagination to bypass the 500-row API limit.
    Returns all 3 trader groups: Commercials, Large Specs (Non-Commercial), Small Specs.
    """
    cache_key = cftc_code
    now = time.time()
    if cache_key in COT_CACHE and (now - COT_CACHE_TIME.get(cache_key, 0)) < COT_CACHE_TTL:
        return COT_CACHE[cache_key]

    COT_CUTOFF = "2008-01-01T00:00:00"
    _SELECT = (
        "report_date_as_yyyy_mm_dd,"
        "comm_positions_long_all,comm_positions_short_all,"
        "noncomm_positions_long_all,noncomm_positions_short_all,"
        "nonrept_positions_long_all,nonrept_positions_short_all,"
        "open_interest_all,"
        "change_in_noncomm_long_all,change_in_noncomm_short_all"
    )
    _WHERE  = f"cftc_contract_market_code='{cftc_code}' AND report_date_as_yyyy_mm_dd >= '{COT_CUTOFF}'"
    PAGE    = 500
    all_data = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            offset = 0
            while True:
                params = {
                    "$select": _SELECT,
                    "$where":  _WHERE,
                    "$order":  "report_date_as_yyyy_mm_dd ASC",
                    "$limit":  str(PAGE),
                    "$offset": str(offset),
                }
                r = await client.get(CFTC_API, params=params)
                r.raise_for_status()
                batch = r.json()
                if not batch:
                    break
                all_data.extend(batch)
                if len(batch) < PAGE:
                    break
                offset += PAGE

        if not all_data:
            return None

        df = pd.DataFrame(all_data)
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
        for col in [
            "comm_positions_long_all", "comm_positions_short_all",
            "noncomm_positions_long_all", "noncomm_positions_short_all",
            "nonrept_positions_long_all", "nonrept_positions_short_all",
            "open_interest_all",
            "change_in_noncomm_long_all", "change_in_noncomm_short_all",
        ]:
            df[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce")

        df["comm_net"]  = df["comm_positions_long_all"]  - df["comm_positions_short_all"]
        df["lspec_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["sspec_net"] = df["nonrept_positions_long_all"] - df["nonrept_positions_short_all"]
        df["lspec_chg"] = df["change_in_noncomm_long_all"] - df["change_in_noncomm_short_all"]

        df = df.dropna(subset=["comm_net"]).sort_values("date").reset_index(drop=True)
        COT_CACHE[cache_key]      = df
        COT_CACHE_TIME[cache_key] = now
        return df
    except Exception as e:
        print(f"COT fetch error for {cftc_code}: {e}")
        return None


# ============================================================
# COT SCORING
# ============================================================

def compute_cot_score(df: Optional[pd.DataFrame], market_id: str = "") -> dict:
    """
    COT scoring — EdgeFinder-style with backtested fund manager divergence signals.

    Architecture (5 signal layers):
    ──────────────────────────────────────────────────────────────────────────────
    LAYER 1 — Commercial Briese Index (primary bias, side with smart money)
    LAYER 2 — Fund Manager Divergence (price at 8-week high/low, managers opposite)
    LAYER 3 — Manager Exhaustion (managers at extreme AND reversing)
    LAYER 4 — Spec vs Commercial Alignment (all three groups converging)
    LAYER 5 — Briese momentum / normalise signals
    ──────────────────────────────────────────────────────────────────────────────
    """
    EMPTY = {
        "score": 5.0, "label": "No Data", "detail": {},
        "comm_index": None, "lspec_index": None, "sspec_index": None,
        "comm_net": None, "lspec_net": None, "sspec_net": None,
        "turning": None, "lspec_chg_3w": None, "lspec_chg_pct": None,
        "alignment": None, "signal_detail": "Insufficient data",
        "divergence": None, "exhaustion": None, "flip": None, "oi_signal": None,
    }
    if df is None or len(df) < 10:
        return EMPTY

    comm_net  = df["comm_net"].values.astype(float)
    lspec_net = df["lspec_net"].values.astype(float)
    sspec_net = df["sspec_net"].values.astype(float)
    window = min(520, len(df))  # 10-year Briese window — matches chart display

    def briese_index(arr, win=None):
        if win is None: win = window
        effective_win = min(win, len(arr))
        if effective_win < 2: return 50.0
        recent = arr[-effective_win:]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return round((arr[-1] - lo) / (hi - lo) * 100, 1)

    def briese_at(arr, idx, win):
        end = len(arr) - idx
        if end < 2: return None
        effective_win = min(win, end)
        recent = arr[max(0, end - effective_win): end]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return (arr[end - 1] - lo) / (hi - lo) * 100

    # Primary: 10-year Briese (LT) blended with 3-year (ST) for recency
    # LT = 520w (10yr) — dominant, matches chart display
    # ST = 104w (2yr) — recency tilt
    comm_idx_lt  = briese_index(comm_net, 520)
    comm_idx_st  = briese_index(comm_net, 104)
    comm_idx     = comm_idx_lt * 0.75 + comm_idx_st * 0.25

    lspec_idx_lt = briese_index(lspec_net, 520)
    lspec_idx_st = briese_index(lspec_net, 104)
    lspec_idx    = lspec_idx_lt * 0.75 + lspec_idx_st * 0.25

    _lspec_currently_low  = lspec_idx < 30
    _lspec_currently_high = lspec_idx > 70

    sspec_idx_lt = briese_index(sspec_net, 520)
    sspec_idx_st = briese_index(sspec_net, 104)
    sspec_idx    = sspec_idx_lt * 0.75 + sspec_idx_st * 0.25

    # ── Layer 1: Base score from Commercial Briese ────────────────────────────
    # Adaptive — score reflects the degree of extreme positioning
    if comm_idx >= 85: base = 8.5
    elif comm_idx >= 75: base = 7.5
    elif comm_idx >= 60: base = 6.5
    elif comm_idx >= 40: base = 5.0
    elif comm_idx >= 25: base = 3.5
    elif comm_idx >= 15: base = 2.5
    else: base = 1.5

    score = base

    # ── Layer 2: Fund Manager Divergence ─────────────────────────────────────
    # Price at recent high/low while managers positioned opposite = early smart money signal
    divergence = None
    price_8w_high = False
    price_8w_low  = False
    if len(df) >= 12:
        price_col = None
        for col in ["offset", "return", "close", "price"]:
            if col in df.columns:
                price_col = col
                break
        if price_col is not None:
            px = df[price_col].values.astype(float)
            if not np.isnan(px[-1]):
                px_hi8 = np.nanmax(px[-9:-1]) if len(px) > 9 else np.nanmax(px[:-1])
                px_lo8 = np.nanmin(px[-9:-1]) if len(px) > 9 else np.nanmin(px[:-1])
                at_hi  = px[-1] >= px_hi8 * 0.985
                at_lo  = px[-1] <= px_lo8 * 1.015
                oi = df.get("open_interest_all", pd.Series(dtype=float)).values.astype(float)
                oi_last = oi[-1] if len(oi) else 0
                oi_chg  = oi[-1] - oi[-4] if len(oi) >= 4 else 0
                oi_pct  = (oi_chg / oi[-4] * 100) if (len(oi) >= 4 and oi[-4] != 0) else 0

                if at_lo and comm_idx >= 70 and lspec_idx <= 35:
                    # Price at 8w low, commercials buying, managers still short
                    oi_str = f"OI {oi_pct:+.1f}%" if abs(oi_pct) > 2 else ""
                    strength = "strong" if (comm_idx >= 80 and lspec_idx <= 25) else "moderate"
                    divergence = {
                        "type": "bull", "strength": strength,
                        "label": f"Price at 8w low, commercials buying (idx={comm_idx:.0f}), managers still short (idx={lspec_idx:.0f}){'. '+oi_str if oi_str else ''}",
                    }
                    price_8w_low = True
                    score = min(10.0, score + (2.0 if strength == "strong" else 1.5))
                elif at_hi and comm_idx <= 30 and lspec_idx >= 65:
                    # Price at 8w high, commercials selling, managers piling long
                    oi_str = f"OI {oi_pct:+.1f}%" if abs(oi_pct) > 2 else ""
                    strength = "strong" if (comm_idx <= 20 and lspec_idx >= 75) else "moderate"
                    divergence = {
                        "type": "bear", "strength": strength,
                        "label": f"Price at 8w high, commercials distributing (idx={comm_idx:.0f}), managers crowded (idx={lspec_idx:.0f}){'. '+oi_str if oi_str else ''}",
                    }
                    price_8w_high = True
                    score = max(0.0, score - (2.0 if strength == "strong" else 1.5))

    # ── Layer 3: Exhaustion (managers at extreme AND reversing) ───────────────
    exhaustion = None
    lspec_chg_3w = None
    lspec_chg_pct = None
    if len(df) >= 4 and "lspec_chg" in df.columns:
        chg = df["lspec_chg"].values.astype(float)
        chg3 = np.nansum(chg[-3:])
        lspec_chg_3w = int(chg3)
        lspec_net_last = lspec_net[-1]
        lspec_chg_pct = round(chg3 / abs(lspec_net_last) * 100, 1) if abs(lspec_net_last) > 50000 else 0

        if lspec_idx >= 75 and chg3 < -500:
            exhaustion = {
                "type": "bear",
                "label": f"Extreme longs reversing — managers at {lspec_idx:.0f}/100, 3w change: {lspec_chg_3w:+,}",
            }
            score = max(0.0, score - 1.0)
        elif lspec_idx <= 25 and chg3 > 500:
            exhaustion = {
                "type": "bull",
                "label": f"Extreme shorts covering — managers at {lspec_idx:.0f}/100, 3w change: {lspec_chg_3w:+,}",
            }
            score = min(10.0, score + 1.0)

    # ── Layer 4 & 5: Alignment + COT phase ───────────────────────────────────
    alignment = None
    convergence_signal = False
    if comm_idx >= 60 and lspec_idx <= 40 and sspec_idx <= 50:
        alignment = "bull"
        convergence_signal = True
    elif comm_idx <= 40 and lspec_idx >= 60 and sspec_idx >= 50:
        alignment = "bear"
        convergence_signal = True

    # COT Phase classification (4-phase cycle)
    cot_phase, cot_phase_dir, cot_phase_label, cot_phase_desc = _classify_cot_phase(
        comm_idx, lspec_idx, sspec_idx)

    # Normalise signal: commercials at extreme AND starting to unwind
    normalise_signal = False
    flatten_signal   = False
    comm_momentum_signal = None
    if len(df) >= 4:
        comm_recent = comm_net[-4:]
        if comm_idx >= 70 and comm_recent[-1] > comm_recent[-2]:
            comm_momentum_signal = {"type": "bull", "detail": "Commercials still accumulating"}
        elif comm_idx <= 30 and comm_recent[-1] < comm_recent[-2]:
            comm_momentum_signal = {"type": "bear", "detail": "Commercials still distributing"}

    flip = None
    if len(df) >= 8:
        prev_comm_idx = briese_at(comm_net, 4, window)
        if prev_comm_idx is not None:
            if prev_comm_idx < 50 and comm_idx >= 50:
                flip = {"type": "bull", "label": f"Commercial net flipped bullish (now {comm_idx:.0f}/100, was {prev_comm_idx:.0f})"}
                score = min(10.0, score + 0.5)
            elif prev_comm_idx > 50 and comm_idx <= 50:
                flip = {"type": "bear", "label": f"Commercial net flipped bearish (now {comm_idx:.0f}/100, was {prev_comm_idx:.0f})"}
                score = max(0.0, score - 0.5)

    # OI signal
    oi_signal = None
    willco_signal = None
    sspec_signal  = None
    oi_regime_signal = None
    if "open_interest_all" in df.columns and len(df) >= 4:
        oi = df["open_interest_all"].values.astype(float)
        oi_chg4 = oi[-1] - oi[-4] if len(oi) >= 4 else 0
        oi_pct4 = (oi_chg4 / oi[-4] * 100) if (len(oi) >= 4 and oi[-4] != 0) else 0
        if oi_pct4 > 9 and comm_idx >= 60:
            oi_signal = {"type": "bull", "name": "OI Confluence — Bull", "label": f"Rising OI ({oi_pct4:+.1f}% in 4w) with commercials bullish ({comm_idx:.0f}/100) — new money entering on the bull side"}
        elif oi_pct4 < -9 and comm_idx <= 40:
            oi_signal = {"type": "bear", "name": "OI Confluence — Bear", "label": f"Falling OI ({oi_pct4:+.1f}% in 4w) with commercials bearish ({comm_idx:.0f}/100) — longs exiting, bear pressure building"}
        # OI Regime: hot/cold market signal (Williams)
        oi_avg_26 = float(np.nanmean(oi[-26:])) if len(oi) >= 26 else float(np.nanmean(oi))
        oi_vs_avg = (oi[-1] / oi_avg_26 - 1) * 100 if oi_avg_26 > 0 else 0
        if oi_vs_avg < -15 and comm_idx >= 65:
            oi_regime_signal = {"type": "bull", "label": f"Cold market (OI {oi_vs_avg:.0f}% below 26w avg) + commercials loading ({comm_idx:.0f}/100) — Williams OI regime bull"}
        elif oi_vs_avg > 15 and comm_idx <= 35:
            oi_regime_signal = {"type": "bear", "label": f"Hot market (OI {oi_vs_avg:.0f}% above 26w avg) + commercials exiting ({comm_idx:.0f}/100) — Williams OI regime bear"}
    # WILLCO (OI-normalised commercial index)
    if "open_interest_all" in df.columns and len(df) >= 156:
        oi = df["open_interest_all"].values.astype(float)
        if len(oi) >= 156 and not np.isnan(oi[-1]):
            comm_oi_ratio = comm_net / np.where(oi > 0, oi, np.nan)  # comm net / OI
            valid = ~np.isnan(comm_oi_ratio)
            if valid.sum() >= 52:
                ratio_series = comm_oi_ratio[valid]
                recent_156 = ratio_series[-min(156, len(ratio_series)):]
                lo_w, hi_w = recent_156.min(), recent_156.max()
                if hi_w != lo_w:
                    willco = (comm_oi_ratio[-1] - lo_w) / (hi_w - lo_w) * 100
                    if willco >= 80:
                        willco_signal = {"type": "bull", "label": f"WILLCO {willco:.0f}/100 — commercial conviction very high relative to market size"}
                    elif willco <= 20:
                        willco_signal = {"type": "bear", "label": f"WILLCO {willco:.0f}/100 — commercial conviction bearish relative to market size"}
    # Small spec dual-extreme (Skorupinski)
    if sspec_idx >= 80 and comm_idx >= 60:
        sspec_signal = {"type": "bull", "label": f"Retail extremes confirm: small specs at {sspec_idx:.0f}/100 (contrarian bull) with commercials at {comm_idx:.0f}/100"}
    elif sspec_idx <= 20 and comm_idx <= 40:
        sspec_signal = {"type": "bear", "label": f"Retail extremes confirm: small specs at {sspec_idx:.0f}/100 (contrarian bear) with commercials at {comm_idx:.0f}/100"}

    score = round(max(0.0, min(10.0, score)), 1)

    # Label
    if score >= 7.5: label = "Strong Bull COT"
    elif score >= 6.0: label = "Mild Bull COT"
    elif score >= 4.5: label = "Neutral COT"
    elif score >= 3.0: label = "Mild Bear COT"
    else: label = "Strong Bear COT"

    turning = (divergence is not None) or (exhaustion is not None) or (flip is not None)

    # Turning label for spec-reversal signal (used in frontend badge)
    turning_label = None
    if lspec_chg_3w is not None:
        if lspec_idx >= 70 and lspec_chg_3w < -50000:
            turning_label = f"Specs Cutting Longs — {abs(lspec_chg_3w):,} contracts in 3w"
        elif lspec_idx >= 75 and lspec_chg_3w < -20000:
            turning_label = f"Large Specs Reducing — {abs(lspec_chg_3w):,} contracts in 3w"
        elif lspec_idx <= 30 and lspec_chg_3w > 50000:
            turning_label = f"Specs Covering Shorts — {lspec_chg_3w:,} contracts in 3w"
        elif lspec_idx <= 25 and lspec_chg_3w > 20000:
            turning_label = f"Large Specs Adding Longs — {lspec_chg_3w:,} contracts in 3w"

    signal_detail = []
    if divergence:    signal_detail.append(divergence.get("label", "Divergence"))
    if exhaustion:    signal_detail.append(exhaustion.get("label", "Exhaustion"))
    if alignment:     signal_detail.append(f"Full {alignment.upper()} alignment")
    if comm_momentum_signal: signal_detail.append(comm_momentum_signal.get("detail", "Comm momentum"))

    return {
        "score":     score,
        "label":     label,
        "detail": {
            "comm_index":        round(comm_idx, 1),
            "comm_index_lt":     round(comm_idx_lt, 1),
            "comm_index_st":     round(comm_idx_st, 1),
            "lspec_index":       round(lspec_idx, 1),
            "lspec_index_lt":    round(lspec_idx_lt, 1),
            "lspec_index_st":    round(lspec_idx_st, 1),
            "sspec_index":       round(sspec_idx, 1),
            "sspec_index_lt":    round(sspec_idx_lt, 1),
            "sspec_index_st":    round(sspec_idx_st, 1),
            "comm_net":          int(comm_net[-1]),
            "lspec_net":         int(lspec_net[-1]),
            "sspec_net":         int(sspec_net[-1]),
            "turning":           turning,
            "turning_label":     turning_label,
            "lspec_chg_3w":      lspec_chg_3w,
            "lspec_chg_pct":     lspec_chg_pct,
            "alignment":         alignment,
            "signal_detail":     " | ".join(signal_detail) if signal_detail else "COT only",
            "divergence":        divergence,
            "exhaustion":        exhaustion,
            "flip":              flip,
            "oi_signal":         oi_signal,
            "willco_signal":     willco_signal,
            "sspec_signal":      sspec_signal,
            "oi_regime_signal":  oi_regime_signal,
            "price_8w_high":     price_8w_high,
            "price_8w_low":      price_8w_low,
            "convergence_signal":convergence_signal,
            "normalise_signal":  normalise_signal,
            "flatten_signal":    flatten_signal,
            "comm_momentum_signal": comm_momentum_signal,
            "cot_phase":         cot_phase,
            "cot_phase_dir":     cot_phase_dir,
            "cot_phase_label":   cot_phase_label,
            "cot_phase_desc":    cot_phase_desc,
        },
    }


def compute_cot_score_v2(df: Optional[pd.DataFrame], market_id: str = "") -> dict:
    """
    COT Scoring v2 — Phase/direction-based architecture.

    Philosophy:
    -----------
    The market-moving story in COT is about SEQUENCE and DIRECTION, not just
    where positioning sits on a historical scale. The ideal signal is:
      1. Commercials consistently move one way (accumulating/distributing)
      2. Fund managers crowd in the opposite direction (following price)
      3. Fund managers form a local peak/trough and begin to reverse
      4. The sequence is coherent — comm turn and spec turn close in time

    The Briese level acts as a CONVICTION MULTIPLIER, not the primary signal.
    A setup at a historically extreme level scores higher than an identical-shaped
    setup at a mid-range level — but the shape is what drives the base score.

    This handles slow (20w+) and fast (4-6w) setups because it looks at
    directional consistency across multiple windows, not fixed-lookback deltas.

    Layers
    ------
    L1  Directional alignment      — are comm and lspec moving oppositely?
    L2  Direction consistency       — how clean/sustained is the trend in each?
    L3  Spec local peak/trough      — has the spec line actually turned?
    L4  Commercial turn coherence   — did comm also turn, and are they in sync?
    L5  Briese level multiplier     — how historically extreme was the setup?
    L6  Price alignment             — are managers visibly following price?
    """
    EMPTY = {
        "score": 5.0, "label": "No Data", "detail": {},
        "comm_index": None, "lspec_index": None, "sspec_index": None,
        "comm_net": None, "lspec_net": None, "sspec_net": None,
        "turning": None, "lspec_chg_3w": None, "lspec_chg_pct": None,
        "alignment": None, "signal_detail": "Insufficient data",
        "divergence": None, "exhaustion": None, "flip": None, "oi_signal": None,
        "comm_momentum_signal": None,
    }
    if df is None or len(df) < 12:
        return EMPTY

    comm_net  = df["comm_net"].values.astype(float)
    lspec_net = df["lspec_net"].values.astype(float)
    sspec_net = df["sspec_net"].values.astype(float)

    n = len(comm_net)
    i = n - 1

    # ── Helpers ────────────────────────────────────────────────────────────────

    def briese(arr, win=520):
        win = min(win, n)
        recent = arr[-win:]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return (arr[-1] - lo) / (hi - lo) * 100

    def briese_at(arr, idx, win=520):
        sub = arr[:idx+1]
        win = min(win, len(sub))
        recent = sub[-win:]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return (sub[-1] - lo) / (hi - lo) * 100

    def direction_consistency(arr, end_i, window):
        """
        Fraction of weeks moving in the majority direction over `window` weeks.
        Returns (consistency_pct 0-100, dominant_direction +1/-1).
        Handles variable-length setups — consistent over 8w AND 16w is stronger
        than consistent over only one window.
        """
        w = min(window, end_i)
        if w < 2: return 0.0, 0
        changes = np.diff(arr[end_i - w: end_i + 1])
        if len(changes) == 0: return 0.0, 0
        pos = float(np.sum(changes > 0))
        neg = float(np.sum(changes < 0))
        if pos >= neg:
            return pos / len(changes) * 100, 1
        else:
            return neg / len(changes) * 100, -1

    def find_local_turn(arr, end_i, min_run=3, max_lookback=52):
        """
        Scan backwards to find the most recent local peak or trough.
        Uses linear regression over before/after windows to detect slope reversal.
        More robust than simple high/low — handles noisy, slow-moving series.

        Returns: (turned: bool, direction: int, weeks_since: int, briese_at_turn: float)
          direction: -1 = bear turn (series topped), +1 = bull turn (series bottomed)
        """
        if end_i < min_run * 2: return False, 0, 0, 50.0
        lookback = min(max_lookback, end_i - min_run)
        for lag in range(1, lookback):
            before = arr[end_i - lag - min_run: end_i - lag]
            after  = arr[end_i - lag:           end_i + 1]
            if len(before) < 2 or len(after) < 2: continue
            b_slope = float(np.polyfit(range(len(before)), before, 1)[0])
            a_slope = float(np.polyfit(range(len(after)),  after,  1)[0])
            # Peak: was rising, now falling
            if b_slope > 0 and a_slope < 0:
                return True, -1, lag, briese_at(arr, end_i - lag, 520)
            # Trough: was falling, now rising
            if b_slope < 0 and a_slope > 0:
                return True, +1, lag, briese_at(arr, end_i - lag, 520)
        return False, 0, 0, 50.0

    # ── Compute Briese indices ─────────────────────────────────────────────────
    ci_lt = briese(comm_net,  520)
    ci_st = briese(comm_net,  104)
    ci    = ci_lt * 0.75 + ci_st * 0.25

    li_lt = briese(lspec_net, 520)
    li_st = briese(lspec_net, 104)
    li    = li_lt * 0.75 + li_st * 0.25

    si_lt = briese(sspec_net, 520)
    si_st = briese(sspec_net, 104)
    si    = si_lt * 0.75 + si_st * 0.25

    ci_display = ci
    li_display = li
    si_display = si

    # ── Direction consistency at multiple windows ──────────────────────────────
    c_cons_8,  c_dir_8  = direction_consistency(comm_net,  i, 8)
    c_cons_16, c_dir_16 = direction_consistency(comm_net,  i, 16)
    l_cons_8,  l_dir_8  = direction_consistency(lspec_net, i, 8)
    l_cons_16, l_dir_16 = direction_consistency(lspec_net, i, 16)

    # Best commercial window: whichever is more consistent
    c_best_cons = max(c_cons_8, c_cons_16)
    c_best_dir  = c_dir_8 if c_cons_8 >= c_cons_16 else c_dir_16

    # ── Spec peak/trough detection ─────────────────────────────────────────────
    # min_run=5: require at least 5 weeks of prior trend before declaring a turn
    # This avoids detecting minor 1-2 week noise blips as meaningful peaks/troughs
    spec_turned, spec_turn_dir, spec_weeks_since, spec_briese_at_turn = find_local_turn(lspec_net, i, min_run=5)
    comm_turned, comm_turn_dir, comm_weeks_since, comm_briese_at_turn = find_local_turn(comm_net,  i, min_run=5)

    # ── L1: Signal direction — LEVEL-DOMINANT, direction as modifier ──────────
    # Framework:
    #   1. COMMERCIALS are smart money. Their ABSOLUTE POSITIONING level is the
    #      primary signal. ci=80 means comms are at the 80th percentile of their
    #      historical net long — that IS a bullish stance regardless of 8w trend.
    #   2. The 8-week direction MODIFIES the level anchor — it doesn't override it.
    #      Comms at ci=64 distributing for 8 weeks are still historically long;
    #      the score should anchor to ~6 (bull) not collapse to ~3 (bear).
    #   3. NON-COMMERCIALS (large specs) are trend-followers / fuel.
    #      We enter when specs START TO AGREE with the commercial direction.
    #      Best entry = comms already loaded, specs just beginning to turn.
    #
    # Level-dominant base score:
    #   level_base = maps ci 0→1, 50→5, 100→9 (linear through midpoint)
    #   ci=20  → base=2.2  (strongly bear)
    #   ci=35  → base=3.7  (mild bear)
    #   ci=50  → base=5.0  (neutral)
    #   ci=64  → base=6.1  (mild bull)
    #   ci=77  → base=7.2  (strong bull)
    #   ci=89  → base=8.1  (very strong bull)
    level_base = 1.0 + (ci / 100.0) * 8.0  # range: 1.0 → 9.0

    # Direction signal is still COMMERCIALS primary, specs fallback
    dirs_opposite = (c_best_dir != 0 and l_dir_8 != 0 and c_best_dir != l_dir_8)
    signal_dir = 0  # +1 = bull, -1 = bear
    if c_best_dir != 0:
        signal_dir = c_best_dir   # comms define the direction
    elif l_dir_8 != 0:
        signal_dir = l_dir_8      # fallback: specs only, lower confidence

    # ── L2: Consistency score ─────────────────────────────────────────────────
    # Commercial directional consistency is the backbone.
    # Bonus when specs are also moving in the SAME direction (confirming).
    specs_confirming = (l_dir_8 != 0 and l_dir_8 == signal_dir)
    if c_best_cons > 0:
        if specs_confirming:
            consistency_score = (c_best_cons * 0.65 + l_cons_8 * 0.35) / 100
        else:
            consistency_score = c_best_cons / 100 * 0.55  # comms only, no spec confirm
    else:
        consistency_score = 0.0

    # ── L3: Spec turn confirmation (entry timing) ─────────────────────────────
    # Has the spec line TURNED in the same direction as signal_dir?
    # Bull: spec_turn_dir=+1 (specs bottoming, starting to go long)
    # Bear: spec_turn_dir=-1 (specs topping, starting to go short)
    spec_turn_confirmed = False
    spec_turn_strength  = 0.0
    spec_turn_label     = ""
    if spec_turned and signal_dir != 0 and spec_turn_dir == signal_dir:
        spec_turn_confirmed = True
        if spec_weeks_since <= 3:
            spec_turn_strength = 1.0
        elif spec_weeks_since <= 8:
            spec_turn_strength = 0.75
        elif spec_weeks_since <= 16:
            spec_turn_strength = 0.45
        else:
            spec_turn_strength = 0.2
        dir_word = "bottoming" if spec_turn_dir == 1 else "topping"
        spec_turn_label = "Specs %s %dw ago (idx=%.0f/100 at turn)" % (
            dir_word, spec_weeks_since, spec_briese_at_turn)

    # ── L4: Comm/spec turn coherence ─────────────────────────────────────────
    # Ideal: commercials turn first (smart money leads), specs confirm after.
    # P1 setup = comm has turned but spec hasn't yet — score this well.
    phase_coherence = 0.0
    phase_label = ""
    if comm_turned and spec_turned:
        turns_aligned = (comm_turn_dir == signal_dir and spec_turn_dir == signal_dir)
        if turns_aligned:
            timing_gap = comm_weeks_since - spec_weeks_since  # +ve = comm turned earlier
            if 0 <= timing_gap <= 8:
                phase_coherence = 1.0
                phase_label = "Comm led, spec confirmed %dw later" % timing_gap
            elif timing_gap > 8:
                phase_coherence = 0.7
                phase_label = "Comm turned %dw before spec — spec late to confirm" % timing_gap
            else:
                phase_coherence = 0.5
                phase_label = "Spec-led move — comm not yet confirmed"
        else:
            phase_coherence = 0.1
    elif comm_turned and not spec_turned:
        # P1 setup: comm turned, spec hasn't confirmed yet — valid early entry
        if comm_turn_dir == signal_dir:
            phase_coherence = 0.55
            phase_label = "Comm turned %dw ago, spec not yet confirmed — P1 setup" % comm_weeks_since
    elif spec_turned and not comm_turned:
        phase_coherence = 0.2
        phase_label = "Spec turned %dw ago, comm turn not detected" % spec_weeks_since

    # ── L5: Briese level multiplier ──────────────────────────────────────────
    # How extreme is the COMMERCIAL positioning? High extreme = high conviction.
    # Bull setup: comms at HIGH Briese (heavy net long accumulation)
    # Bear setup: comms at LOW Briese (heavy net short distribution)
    if signal_dir == 1:   # bull: want comms high
        if   ci >= 80: level_mult = 1.4
        elif ci >= 65: level_mult = 1.2
        elif ci >= 50: level_mult = 1.0
        elif ci >= 35: level_mult = 0.8
        else:          level_mult = 0.6
    elif signal_dir == -1:  # bear: want comms low
        if   ci <= 20: level_mult = 1.4
        elif ci <= 35: level_mult = 1.2
        elif ci <= 50: level_mult = 1.0
        elif ci <= 65: level_mult = 0.8
        else:          level_mult = 0.6
    else:
        level_mult = 0.7

    # Spec extremity at turn confirms crowd was wrong-footed
    # Bull: specs at LOW Briese when they turned = oversold crowd = more conviction
    # Bear: specs at HIGH Briese when they turned = crowded longs = more conviction
    if spec_turn_confirmed:
        if signal_dir == 1:
            if   spec_briese_at_turn <= 20: spec_level_mult = 1.3
            elif spec_briese_at_turn <= 35: spec_level_mult = 1.15
            elif spec_briese_at_turn <= 50: spec_level_mult = 1.0
            else:                           spec_level_mult = 0.85
        else:
            if   spec_briese_at_turn >= 80: spec_level_mult = 1.3
            elif spec_briese_at_turn >= 65: spec_level_mult = 1.15
            elif spec_briese_at_turn >= 50: spec_level_mult = 1.0
            else:                           spec_level_mult = 0.85
        blended_mult = level_mult * 0.6 + spec_level_mult * 0.4
    else:
        blended_mult = level_mult

    # ── L6: Price alignment ───────────────────────────────────────────────────
    # Are managers visibly following price in the expected direction before the turn?
    price_alignment_bonus = 0.0
    price_col = None
    for col in ["offset", "return", "close", "price"]:
        if col in df.columns:
            price_col = col
            break
    if price_col is not None and spec_turn_confirmed and spec_weeks_since >= 3:
        px = df[price_col].values.astype(float)
        # Did price trend in the spec direction BEFORE the turn?
        before_turn_i = max(0, i - spec_weeks_since)
        run_start     = max(0, before_turn_i - 12)
        if run_start < before_turn_i:
            px_run   = px[run_start:before_turn_i + 1]
            px_valid = px_run[~np.isnan(px_run)]
            if len(px_valid) >= 4:
                px_slope = float(np.polyfit(range(len(px_valid)), px_valid, 1)[0])
                # Bear setup: price should have been rising (managers followed it up)
                if signal_dir == -1 and px_slope > 0:
                    price_alignment_bonus = 0.3
                elif signal_dir == 1 and px_slope < 0:
                    price_alignment_bonus = 0.3

    # ── Composite score ───────────────────────────────────────────────────────
    # LEVEL-DOMINANT approach:
    #   1. Start from level_base (ci maps linearly to 1–9)
    #   2. Direction consistency shifts ±up to 1.5 pts from that anchor
    #   3. Spec turn + phase coherence shifts ±up to 1.5 pts further
    #   4. Total max shift from level_base: ±3.0 pts (clamped to 1–9)
    #
    # This means ci=64 distributing scores ~6.1 - 0.8 = ~5.3+ (not 3.4)
    # And ci=89 distributing scores ~8.1 - 1.5 = ~6.6+ (not 5.0)
    # And ci=77 accumulating scores ~7.2 + 1.0 = ~8.2 (not 6.2)

    if signal_dir == 0:
        # No directional signal — anchor purely to level
        # ci=50→5.0, ci=60→5.8, ci=40→4.2 etc.
        final_score = round(max(1.0, min(9.0, level_base)), 1)
        label = "No clear COT directional signal — level anchor only"

        # Raw/shift for debug continuity
        raw_signal = 0.0
        shift = 0.0
    else:
        # ── Direction adjustment: how strongly are comms moving, confirmed by specs? ──
        # Ranges from -1.5 (strong counter-level signal) to +1.5 (strong with-level)
        # "with-level" = direction agrees with where comms are (bull if ci>50, bear if ci<50)
        level_side = 1 if level_base >= 5.0 else -1  # which side is the level on?
        dir_agrees_with_level = (signal_dir == level_side)  # True = direction confirms level

        # Consistency contribution
        # When direction AGREES with level side, it can push up to +1.2.
        # When direction OPPOSES level side (e.g. ci=64 but distributing), it
        # can only pull down a maximum of 0.6 — level takes priority.
        if c_best_cons > 0:
            if specs_confirming:
                cons_strength = (c_best_cons * 0.65 + l_cons_8 * 0.35) / 100  # 0–1
            else:
                cons_strength = c_best_cons / 100 * 0.55  # comms only
        else:
            cons_strength = 0.0

        level_side = 1 if level_base >= 5.0 else -1
        dir_agrees_with_level = (signal_dir == level_side)
        # Max dir weight: 1.2 if agrees with level, 0.5 if opposes level
        dir_weight = 1.2 if dir_agrees_with_level else 0.5
        dir_adj = cons_strength * dir_weight * signal_dir

        # Spec turn + phase contribution
        # Also dampened when direction opposes level
        turn_weight = 1.0 if dir_agrees_with_level else 0.4
        turn_phase = (
            (spec_turn_strength if spec_turn_confirmed else 0.0) * 0.6 +
            phase_coherence * 0.3 +
            price_alignment_bonus * 0.1
        ) * turn_weight * signal_dir

        # Combined directional adjustment
        # Agrees with level: up to +2.2 (push strongly)
        # Opposes level:  up to -0.9 (modest pullback only)
        directional_adj = dir_adj + turn_phase

        # For debug fields (keep naming consistent with old approach)
        raw_signal = abs(directional_adj)
        shift = abs(directional_adj)

        # Final score: level anchor + directional adjustment
        pre_clamp = level_base + directional_adj
        final_score = round(max(1.0, min(9.0, pre_clamp)), 1)

        # Labels based on final score
        if final_score >= 7.5:
            label = "Strong bullish COT — commercials heavily loaded, spec turn confirmed"
        elif final_score >= 6.5:
            label = "Moderately bullish COT — commercials accumulating, specs starting to confirm"
        elif final_score >= 5.5:
            label = "Mildly bullish COT — commercial lean, level elevated"
        elif final_score >= 4.5:
            label = "Neutral COT — mixed signals"
        elif final_score >= 3.5:
            label = "Mildly bearish COT — commercial lean, level depressed"
        elif final_score >= 2.5:
            label = "Moderately bearish COT — commercials distributing, specs starting to confirm"
        else:
            label = "Strong bearish COT — commercials heavily short, spec turn confirmed"

    # ── Level floor/cap — safety rails only (level_base already anchors) ───────
    # With level-dominant scoring, floor/cap are narrow safety rails.
    # The level_base already encodes ci into the score; these just prevent
    # extreme directional adjustments from going fully off-rails.
    # e.g. ci=90 but somehow scores <6.0 after big neg adj → floor to 5.5
    if   ci >= 85: level_floor = 6.8
    elif ci >= 75: level_floor = 6.2
    elif ci >= 62: level_floor = 5.8
    elif ci >= 55: level_floor = 5.3
    elif ci >= 50: level_floor = 5.0
    else:          level_floor = 1.0

    if   ci <= 15: level_cap = 3.2
    elif ci <= 25: level_cap = 3.8
    elif ci <= 35: level_cap = 4.2
    elif ci <= 42: level_cap = 4.6
    elif ci <= 48: level_cap = 5.0
    else:          level_cap = 9.0

    # Apply unconditionally — level_base anchors already handle direction conflicts
    final_score = max(final_score, level_floor)
    final_score = min(final_score, level_cap)
    final_score = round(final_score, 1)

    # ── COT phase classification (reuse v1 helper) ────────────────────────────
    cot_phase, cot_phase_dir, cot_phase_label, cot_phase_desc = _classify_cot_phase(
        ci, li, si)

    # ── Build signal detail dict ──────────────────────────────────────────────
    signal_parts = []
    if c_best_dir != 0:
        spec_status = "confirming" if specs_confirming else "not yet agreeing"
        signal_parts.append("Comm %s (%.0f%% consistency, %dw) — specs %s" % (
            "accumulating" if c_best_dir > 0 else "distributing",
            c_best_cons,
            8 if c_cons_8 >= c_cons_16 else 16,
            spec_status,
        ))
    if spec_turn_label:
        signal_parts.append(spec_turn_label)
    if phase_label:
        signal_parts.append(phase_label)
    if price_alignment_bonus > 0:
        signal_parts.append("Price aligned with spec run pre-turn")

    signal_detail = " | ".join(signal_parts) if signal_parts else "Insufficient directional signal"

    # Build comm_momentum_signal for compatibility with existing front-end rendering
    comm_momentum_signal = None
    if c_best_dir > 0 and c_best_cons >= 60:
        comm_momentum_signal = {"type": "bull", "detail": "Commercials consistently buying (%.0f%% of weeks)" % c_best_cons}
    elif c_best_dir < 0 and c_best_cons >= 60:
        comm_momentum_signal = {"type": "bear", "detail": "Commercials consistently selling/distributing (%.0f%% of weeks)" % c_best_cons}

    # Exhaustion signal for UI compatibility
    exhaustion = None
    if spec_turn_confirmed and spec_turn_strength >= 0.7:
        exhaustion = {
            "type": "bear" if spec_turn_dir == -1 else "bull",
            "label": spec_turn_label
        }

    # Divergence signal (replaces old Layer 2)
    divergence = None
    if spec_turn_confirmed and phase_coherence >= 0.5:
        divergence = {
            "type": "bear" if signal_dir == -1 else "bull",
            "strength": "strong" if blended_mult >= 1.2 else "moderate",
            "label": "Phase transition: %s" % phase_label
        }

    # OI signal passthrough (keep existing logic for UI)
    oi_signal = None
    if "open_interest_all" in df.columns and len(df) >= 4:
        oi = df["open_interest_all"].values.astype(float)
        oi_chg4 = oi[-1] - oi[-4] if len(oi) >= 4 else 0
        oi_pct4 = (oi_chg4 / oi[-4] * 100) if (len(oi) >= 4 and oi[-4] != 0) else 0
        if oi_pct4 > 9 and ci >= 60:
            oi_signal = {"type": "bull", "name": "OI Confluence", "label": "Rising OI (%.1f%% in 4w) with commercials buying" % oi_pct4}
        elif oi_pct4 < -9 and ci <= 40:
            oi_signal = {"type": "bear", "name": "OI Confluence", "label": "Falling OI (%.1f%% in 4w) with commercials distributing" % oi_pct4}

    return {
        "score":              final_score,
        "label":              label,
        "comm_index":         round(ci_display, 1),
        "lspec_index":        round(li_display, 1),
        "sspec_index":        round(si_display, 1),
        "comm_net":           int(comm_net[-1]),
        "lspec_net":          int(lspec_net[-1]),
        "sspec_net":          int(sspec_net[-1]),
        "turning":            spec_turn_confirmed,
        "lspec_chg_3w":       int(np.nansum(df["lspec_chg"].values[-3:])) if "lspec_chg" in df.columns else None,
        "lspec_chg_pct":      None,
        "alignment":          "bull" if signal_dir == 1 else ("bear" if signal_dir == -1 else None),
        "signal_detail":      signal_detail,
        "divergence":         divergence,
        "exhaustion":         exhaustion,
        "flip":               None,
        "oi_signal":          oi_signal,
        "comm_momentum_signal": comm_momentum_signal,
        # v2-specific debug fields
        "v2_signal_dir":         signal_dir,
        "v2_c_best_dir":         c_best_dir,
        "v2_c_best_cons":        round(c_best_cons, 1),
        "v2_l_dir_8":            l_dir_8,
        "v2_l_cons_8":           round(l_cons_8, 1),
        "v2_spec_turn_confirmed": spec_turn_confirmed,
        "v2_spec_turn_dir":      spec_turn_dir if spec_turned else 0,
        "v2_spec_weeks_since":   spec_weeks_since if spec_turned else 0,
        "v2_comm_turn_dir":      comm_turn_dir if comm_turned else 0,
        "v2_comm_weeks_since":   comm_weeks_since if comm_turned else 0,
        "v2_consistency":        round(cons_strength if signal_dir != 0 else (c_best_cons / 100.0 if c_best_cons > 0 else 0.0), 3),
        "v2_spec_turn_strength": round(spec_turn_strength, 3),
        "v2_phase_coherence":    round(phase_coherence, 3),
        "v2_level_mult":         round(blended_mult, 3),
        "v2_raw_signal":         round(raw_signal if signal_dir != 0 else 0, 3),
        "v2_shift":              round(shift if signal_dir != 0 else 0, 3),
        "detail": {
            "comm_index":       round(ci_display, 1),
            "lspec_index":      round(li_display, 1),
            "sspec_index":      round(si_display, 1),
            "cot_phase":        cot_phase,
            "cot_phase_dir":    cot_phase_dir,
            "cot_phase_label":  cot_phase_label,
            "cot_phase_desc":   cot_phase_desc,
        },
    }


def _classify_cot_phase(comm_idx: float, lspec_idx: float, sspec_idx: float):
    """Classify COT into 4-phase cycle for both bull and bear directions."""
    # Bull cycle phases
    if comm_idx >= 70 and lspec_idx <= 35:
        return 1, "bull", "Bull P1: Prime Entry", "Commercials loaded, managers still offside"
    elif comm_idx >= 45 and lspec_idx <= 58:
        return 2, "bull", "Bull P2: Momentum", "Trend confirmed, both smart and fund money aligned"
    elif comm_idx <= 58 and lspec_idx >= 50:
        return 3, "bull", "Bull P3: Crowded", "Managers crowded long, diminishing returns"
    elif comm_idx <= 40 and lspec_idx >= 60:
        return 4, "bull", "Bull P4: Overstretched", "Commercials out, managers at peak — mirror image of bear P1"
    # Bear cycle phases
    elif comm_idx <= 30 and lspec_idx >= 65:
        return 1, "bear", "Bear P1: Prime Entry", "Commercials short, managers still long — best bear entry"
    elif comm_idx <= 55 and lspec_idx >= 42:
        return 2, "bear", "Bear P2: Momentum", "Bear trend confirmed"
    elif comm_idx >= 42 and lspec_idx <= 50:
        return 3, "bear", "Bear P3: Extended", "Bear move maturing"
    elif comm_idx >= 60 and lspec_idx <= 40:
        return 4, "bear", "Bear P4: Overstretched", "Commercials covering, managers trapped short"
    else:
        return 0, "neutral", "Transitioning", "Positioning mid-range — no dominant phase signal"


def compute_consensus_fade(cot_data: dict, news_sentiment: Optional[float],
                           market_name: str = "", category: str = "") -> dict:
    """
    Consensus-fade / crowded-trade detector.

    Ben's edge: the asymmetry isn't just extreme positioning — it's extreme
    positioning that AGREES with a one-sided consensus narrative, where fading
    the crowd has minimal downside (already priced in) and large upside (a
    surprise the other way leaves everyone offside).

    Blends two live inputs:
      1. COT positioning extreme — managed-money (large spec) Briese percentile.
         >72 = crowd crammed long; <28 = crowd crammed short. Weighted by whether
         the crowd is STILL piling in (specs not yet turned) per Ben's framework
         ("follow commercials, enter when non-commercials start to turn").
      2. News/narrative one-sidedness — narrative_scores 0-10 (>6.5 bullish crowd,
         <3.5 bearish crowd). Confirms the story everyone is trading.

    The fade fires hardest when BOTH agree in the same direction and the crowd
    hasn't turned yet.

    Returns a dict:
      fade_score      float 0-10   (magnitude of the crowded-trade / fade edge; 0 = no edge)
      fade_dir        str          contrarian direction: 'long' | 'short' | None
                                   (= the side to take AGAINST the crowd)
      crowd_side      str          'long' | 'short' | None (what the crowd is doing)
      spec_pctile     float|None   large-spec Briese percentile (0-100)
      spec_adding     bool|None    True if crowd still piling in (not yet turned)
      news_score      float|None   0-10 narrative sentiment
      confirms        bool         True if positioning + narrative agree (both one-sided same way)
      asymmetry       str          'what would put the crowd offside' note
      inputs          dict         raw components for transparency
    """
    EMPTY = {
        "fade_score": 0.0, "fade_dir": None, "crowd_side": None,
        "spec_pctile": None, "spec_adding": None, "news_score": None,
        "confirms": False, "asymmetry": "", "inputs": {},
    }
    if not cot_data:
        return EMPTY

    spec_idx = cot_data.get("lspec_index")
    turning  = cot_data.get("turning")          # True once specs have TURNED (fade is then late)
    chg3w    = cot_data.get("lspec_chg_3w")      # net spec change over 3w (>0 = still adding longs)
    detail   = cot_data.get("detail", {}) or {}
    phase_dir   = detail.get("cot_phase_dir")    # 'bull'/'bear'/'neutral'
    phase_label = detail.get("cot_phase_label", "")

    if spec_idx is None:
        return EMPTY

    # ── 1. Positioning extreme strength (0-1) ─────────────────────────────────
    # Distance of large-spec percentile from the 50 midpoint, normalised so that
    # the extreme zones (≥72 or ≤28) start to score and 90/10 ≈ full strength.
    HI, LO = 72.0, 28.0
    if spec_idx >= HI:
        crowd_side = "long"          # managed money crammed net long
        pos_strength = min(1.0, (spec_idx - HI) / (95.0 - HI))
    elif spec_idx <= LO:
        crowd_side = "short"         # managed money crammed net short
        pos_strength = min(1.0, (LO - spec_idx) / (LO - 5.0))
    else:
        crowd_side = None
        pos_strength = 0.0

    # ── 2. "Still piling in" multiplier (Ben's timing rule) ───────────────────
    # Best fade = crowd extreme AND hasn't turned yet (commercials still opposite).
    # If specs have already turned (cot_data['turning']=True) the crowd is unwinding
    # → the fade window is closing, so we discount it heavily.
    still_adding = None
    timing_mult = 1.0
    if crowd_side is not None:
        if turning:
            timing_mult = 0.45           # crowd already reversing — late
            still_adding = False
        elif chg3w is not None:
            # crowd adding in the SAME direction as its extreme = strongest fade
            adding_long  = crowd_side == "long"  and chg3w > 0
            adding_short = crowd_side == "short" and chg3w < 0
            still_adding = bool(adding_long or adding_short)
            timing_mult = 1.0 if still_adding else 0.8
        else:
            timing_mult = 0.9

    # ── 3. Narrative one-sidedness (0-1) + agreement with positioning ─────────
    news_side = None
    news_strength = 0.0
    if news_sentiment is not None:
        if news_sentiment >= 6.5:
            news_side = "long"
            news_strength = min(1.0, (news_sentiment - 6.5) / (9.5 - 6.5))
        elif news_sentiment <= 3.5:
            news_side = "short"
            news_strength = min(1.0, (3.5 - news_sentiment) / (3.5 - 0.5))

    confirms = bool(crowd_side is not None and news_side is not None and crowd_side == news_side)

    # ── 4. Blend into a 0-10 fade score ───────────────────────────────────────
    # Positioning is the hard core (70%); narrative confirmation is the amplifier (30%).
    # When narrative CONFIRMS the crowd, apply a confluence boost; when it CONTRADICTS,
    # damp it (the story is fighting the positioning — weaker one-sided consensus).
    base = pos_strength * timing_mult
    if confirms:
        blended = 0.70 * base + 0.30 * news_strength
        blended = min(1.0, blended + 0.12 * base * news_strength)   # confluence boost
    elif crowd_side is not None and news_side is not None and crowd_side != news_side:
        blended = base * 0.75            # narrative fights positioning — softer edge
    else:
        blended = 0.70 * base            # positioning only, no narrative read

    fade_score = round(blended * 10.0, 1)
    fade_dir = None
    if crowd_side == "long":
        fade_dir = "short"               # fade the crowded long → take the short
    elif crowd_side == "short":
        fade_dir = "long"                # fade the crowded short → take the long

    # ── 5. Asymmetry note — 'what would put the crowd offside' ────────────────
    asym = ""
    if crowd_side == "long":
        story = " and news flow one-sided bullish" if confirms else ""
        asym = (f"Crowd crammed long (specs {spec_idx:.0f}/100{story}). "
                f"Upside largely priced in — a bearish surprise leaves longs offside; "
                f"fade risk/reward favours the short if price and your zones confirm.")
    elif crowd_side == "short":
        story = " and news flow one-sided bearish" if confirms else ""
        asym = (f"Crowd crammed short (specs {spec_idx:.0f}/100{story}). "
                f"Downside largely priced in — a bullish surprise squeezes shorts; "
                f"fade risk/reward favours the long if price and your zones confirm.")

    return {
        "fade_score":  fade_score,
        "fade_dir":    fade_dir,
        "crowd_side":  crowd_side,
        "spec_pctile": round(float(spec_idx), 1),
        "spec_adding": still_adding,
        "news_score":  round(float(news_sentiment), 1) if news_sentiment is not None else None,
        "confirms":    confirms,
        "asymmetry":   asym,
        "inputs": {
            "pos_strength": round(pos_strength, 3),
            "timing_mult":  round(timing_mult, 3),
            "news_strength": round(news_strength, 3),
            "news_side":    news_side,
            "phase_label":  phase_label,
            "phase_dir":    phase_dir,
            "turning":      bool(turning) if turning is not None else None,
            "lspec_chg_3w": chg3w,
        },
    }


def compute_crypto_cot_score(df: Optional[pd.DataFrame], market_id: str = "") -> dict:
    """
    Crypto-specific COT scoring — simplified trend-following approach.

    Large Specs (non-commercials / fund managers) are the primary signal.
    Logic: follow the TREND of large spec net positioning.
    - Bullish while large specs are trending higher or holding elevated levels.
    - Only turn bearish on a SUSTAINED, MATERIAL reversal (not a single-week dip).
    - No contrarian penalty for elevated readings — in crypto, high spec longs
      during a bull trend is normal and should not be faded.
    - Commercials ignored for direction (structurally short via basis trades / ETF hedging).
    """
    EMPTY = {"score": 5.0, "label": "No data", "detail": {}}
    if df is None or len(df) < 10:
        return EMPTY

    lspec_long  = pd.to_numeric(df.get("noncomm_positions_long_all",  pd.Series(dtype=float)), errors="coerce").values.astype(float)
    lspec_short = pd.to_numeric(df.get("noncomm_positions_short_all", pd.Series(dtype=float)), errors="coerce").values.astype(float)
    comm_long   = pd.to_numeric(df.get("comm_positions_long_all",  pd.Series(dtype=float)), errors="coerce").values.astype(float)
    comm_short  = pd.to_numeric(df.get("comm_positions_short_all", pd.Series(dtype=float)), errors="coerce").values.astype(float)

    def briese_index(arr, win=520):
        effective_win = min(win, len(arr))
        if effective_win < 2: return 50.0
        recent = arr[-effective_win:]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return round((arr[-1] - lo) / (hi - lo) * 100, 1)

    sspec_long  = pd.to_numeric(df.get("nonrept_positions_long_all",  pd.Series(dtype=float)), errors="coerce").values.astype(float)
    sspec_short = pd.to_numeric(df.get("nonrept_positions_short_all", pd.Series(dtype=float)), errors="coerce").values.astype(float)

    lspec_net = lspec_long - lspec_short
    comm_net  = comm_long  - comm_short
    sspec_net = sspec_long - sspec_short
    lspec_briese = briese_index(lspec_net)
    comm_briese  = briese_index(comm_net)
    sspec_briese = briese_index(sspec_net)

    lspec_range  = float(lspec_net.max() - lspec_net.min()) or 1.0

    score = 5.0
    detail_parts = []

    # ── 1. TREND: 12-week slope of large spec net positioning ─────────────
    # Bullish while trend is up or flat at elevated levels.
    # Only bearish signal when trend is DOWN and level has also fallen materially.
    n_trend = min(12, len(lspec_net))
    trend_slope = np.polyfit(np.arange(n_trend), lspec_net[-n_trend:], 1)[0] if n_trend >= 3 else 0
    trend_slope = round(float(trend_slope), 1)

    # Normalise slope as % of full historical range per week, then scale to score points
    # e.g. gaining 1% of range per week over 12w = meaningful trend = +0.75 pts
    _slope_pct_per_wk = (trend_slope / lspec_range) * 100.0  # % of range per week
    if trend_slope > 0:
        score += min(1.5, _slope_pct_per_wk * 0.75)
        detail_parts.append(f"Large spec trend rising (+{trend_slope:.0f}/wk, +{_slope_pct_per_wk:.1f}%/wk) — bullish")
    else:
        # Only penalise if briese has also fallen materially below 50 (sustained reversal)
        if lspec_briese < 45:
            score += max(-1.5, _slope_pct_per_wk * 0.75)
            detail_parts.append(f"Large spec trend falling, briese {lspec_briese:.0f} — bearish reversal")
        elif lspec_briese < 65:
            score -= 0.3
            detail_parts.append(f"Large spec trend softening (briese {lspec_briese:.0f}) — mild caution")
        else:
            detail_parts.append(f"Large spec normalising from highs (briese {lspec_briese:.0f}) — still constructive")

    # ── 2. LEVEL: where are specs positioned in their historical range ─────
    # High level = trend in place = constructive. Low = bearish until reversal.
    if lspec_briese >= 65:
        score += 1.0
        detail_parts.append(f"Spec net long — elevated ({lspec_briese:.0f}/100)")
    elif lspec_briese >= 45:
        score += 0.3
        detail_parts.append(f"Spec net long — moderate ({lspec_briese:.0f}/100)")
    elif lspec_briese >= 25:
        score -= 0.5
        detail_parts.append(f"Spec net light ({lspec_briese:.0f}/100) — cautious")
    else:
        score -= 1.5
        detail_parts.append(f"Spec net short/depressed ({lspec_briese:.0f}/100) — bearish")

    # ── 3. MATERIAL REVERSAL CHECK: 4-week momentum ───────────────────────
    # Only flag a reversal if the last 4 weeks show a sharp drop from a high base.
    n_short = min(4, len(lspec_net))
    short_slope = np.polyfit(np.arange(n_short), lspec_net[-n_short:], 1)[0] if n_short >= 3 else 0
    short_slope = round(float(short_slope), 1)
    lspec_4w_chg = float(lspec_net[-1] - lspec_net[-min(4, len(lspec_net))])
    reversal_pct = abs(lspec_4w_chg) / lspec_range  # fraction of full range moved in 4w

    if short_slope < 0 and reversal_pct > 0.35 and lspec_briese > 60:
        # Sharp 4-week drop from elevated = material reversal warning
        score -= 1.0
        detail_parts.append(f"Sharp 4w unwind ({lspec_4w_chg:+.0f} contracts, {reversal_pct*100:.0f}% of range) — reversal watch")
    elif short_slope < 0 and lspec_briese < 45:
        # Falling from already-low base = sustained bearish
        score -= 0.5
        detail_parts.append(f"Continued de-risking from low base — bearish")

    # ── Stance label ──────────────────────────────────────────────────────
    if lspec_briese >= 70:    stance = "elevated & trending — bullish"
    elif lspec_briese >= 50:  stance = "constructive — above neutral"
    elif lspec_briese >= 30:  stance = "below neutral — cautious"
    else:                     stance = "depressed — bearish"

    score = round(max(0.0, min(10.0, score)), 1)
    label = "Bullish Crypto COT" if score >= 6 else "Bearish Crypto COT" if score <= 4 else "Neutral Crypto COT"

    # Alignment label for frontend
    if score >= 7:   alignment = "Bullish Positioning"
    elif score >= 6: alignment = "Mild Bull Positioning"
    elif score <= 3: alignment = "Bearish Positioning"
    elif score <= 4: alignment = "Mild Bear Positioning"
    else:            alignment = "Neutral Positioning"

    signal_str = " | ".join(detail_parts) if detail_parts else f"Fund Mgr Briese {lspec_briese:.0f}/100 — {stance}"

    return {
        "score": score,
        "label": label,
        # Standard field names expected by frontend cotTab()
        "lspec_index": round(lspec_briese, 1),
        "comm_index":  round(comm_briese,  1),
        "sspec_index": round(sspec_briese, 1),
        "lspec_net":   round(float(lspec_net[-1]), 0),
        "comm_net":    round(float(comm_net[-1]),  0),
        "sspec_net":   round(float(sspec_net[-1]), 0) if len(sspec_net) > 0 else None,
        "alignment":   alignment,
        "signal_detail": signal_str,
        "detail": {
            # Standard index fields read by cotTab() frontend
            "comm_index":  round(comm_briese,  1),
            "lspec_index": round(lspec_briese, 1),
            "sspec_index": round(sspec_briese, 1),
            "comm_net":    round(float(comm_net[-1]),  0),
            "lspec_net":   round(float(lspec_net[-1]), 0),
            "sspec_net":   round(float(sspec_net[-1]), 0) if len(sspec_net) > 0 else None,
            "alignment":   alignment,
            "signal_detail": signal_str,
            # Extra crypto-specific fields
            "lspec_briese": round(lspec_briese, 1),
            "comm_briese":  round(comm_briese,  1),
            "sspec_briese": round(sspec_briese, 1),
            "lspec_momentum": round(float(trend_slope), 1),
            "cot_phase": 0, "cot_phase_dir": "neutral",
            "cot_phase_label": "Crypto COT", "cot_phase_desc": "",
        },
    }


def build_cross_cot_df(base_leg: str, quote_leg: str, cot_cache: dict):
    """Construct a SYNTHETIC 3-category COT DataFrame for an FX cross from its two
    USD-denominated futures legs, so a cross gets the SAME full v2 treatment as an
    outright market (all three legacy categories + divergence/exhaustion/turning/
    phase) instead of the old single-number commercial-Briese differential.

    Construction logic
    ------------------
    The cross PRICE is literally base_future / quote_future (e.g. GBPJPY = 6B / 6J),
    so holding 1 unit of the cross ≈ LONG the base leg + SHORT the quote leg.
    Therefore, for EACH legacy category C (commercials, large specs / non-commercials,
    small specs / non-reportables) the cross net position is:

        cross_net_C = net_C(base) − net_C(quote)

    Raw contract counts are NOT comparable across two differently-sized currency books
    (one market can have 5x the open interest of the other), so we normalise each leg's
    net by its OWN open interest first, difference the two, then re-express the spread
    in contract-equivalent units using the average OI of the legs:

        cross_net_C = ( net_C(base)/OI(base) − net_C(quote)/OI(quote) ) * (OI(base)+OI(quote))/2

    This preserves all three categories (no Briese collapse), keeps the magnitudes in a
    sane contract range for the v2 thresholds, and is sign-correct: cross_comm_net > 0
    means commercials are net-long the cross (long base / short quote).
    """
    df_b = cot_cache.get(base_leg)
    df_q = cot_cache.get(quote_leg)
    need = ["date", "comm_net", "lspec_net", "sspec_net", "open_interest_all"]
    if (df_b is None or df_q is None or len(df_b) < 12 or len(df_q) < 12 or
            any(c not in df_b.columns for c in need) or any(c not in df_q.columns for c in need)):
        return None
    try:
        b = df_b[need].copy()
        q = df_q[need].copy()
        b["date"] = pd.to_datetime(b["date"]).dt.tz_localize(None).dt.normalize()
        q["date"] = pd.to_datetime(q["date"]).dt.tz_localize(None).dt.normalize()
        m = pd.merge_asof(b.sort_values("date"), q.sort_values("date"), on="date",
                          direction="nearest", tolerance=pd.Timedelta(days=10),
                          suffixes=("_b", "_q")).dropna()
        if len(m) < 12:
            return None
        oi_b = m["open_interest_all_b"].replace(0, np.nan)
        oi_q = m["open_interest_all_q"].replace(0, np.nan)
        ref_oi = (oi_b + oi_q) / 2.0
        out = pd.DataFrame({"date": m["date"].values})
        for cat in ("comm_net", "lspec_net", "sspec_net"):
            base_pct  = m[f"{cat}_b"] / oi_b
            quote_pct = m[f"{cat}_q"] / oi_q
            out[cat] = ((base_pct - quote_pct) * ref_oi).round()
        out["open_interest_all"] = ref_oi.round()
        out = out.dropna().reset_index(drop=True)
        if len(out) < 12:
            return None
        out["lspec_chg"] = out["lspec_net"].diff().fillna(0)
        return out
    except Exception as _e:
        print(f"[build_cross_cot_df] {base_leg}/{quote_leg}: {_e}", flush=True)
        return None


def compute_cross_cot_score(market_id: str, base_leg: str, quote_leg: str, cot_cache: dict) -> dict:
    """
    DEPRECATED (kept for reference / fallback): single-number commercial-Briese
    differential. Superseded by build_cross_cot_df() + compute_cot_score_v2(), which
    constructs all three legacy categories for the cross. See build_cross_cot_df.

    Derive COT score for a cross pair from two USD-denominated futures legs.
    cot_cache: dict mapping market_id -> pd.DataFrame (already fetched in main loop)
    Returns same dict structure as compute_cot_score().
    """
    NO_DATA = {
        "score": 5.0, "label": "No Data",
        "comm_index": None, "lspec_index": None, "sspec_index": None,
        "comm_net": None, "lspec_net": None, "sspec_net": None,
        "turning": None, "alignment": None,
        "signal_detail": f"{market_id} COT data unavailable",
        "divergence": None, "exhaustion": None, "flip": None, "oi_signal": None,
        "detail": {
            "cot_phase": 0, "cot_phase_dir": "neutral",
            "cot_phase_label": "Cross Pair", "cot_phase_desc": "",
            "cross": True, "base_leg": base_leg, "quote_leg": quote_leg,
        },
    }
    df_base  = cot_cache.get(base_leg)
    df_quote = cot_cache.get(quote_leg)
    if df_base is None or len(df_base) < 30 or df_quote is None or len(df_quote) < 30:
        return NO_DATA

    def briese_index(arr, win=520):
        effective_win = min(win, len(arr))
        if effective_win < 2: return 50.0
        recent = arr[-effective_win:]
        lo, hi = recent.min(), recent.max()
        if hi == lo: return 50.0
        return round((arr[-1] - lo) / (hi - lo) * 100, 1)

    def briese_series(arr, win=156):
        out = []
        for i in range(len(arr)):
            end = i + 1
            effective_win = min(win, end)
            if effective_win < 2: out.append(50.0); continue
            recent = arr[max(0, end-effective_win):end]
            lo, hi = recent.min(), recent.max()
            out.append(50.0 if hi == lo else (arr[i] - lo) / (hi - lo) * 100)
        return np.array(out)

    n = min(len(df_base), len(df_quote))
    df_base  = df_base.tail(n).reset_index(drop=True)
    df_quote = df_quote.tail(n).reset_index(drop=True)

    base_comm  = df_base["comm_net"].values.astype(float)
    quote_comm = df_quote["comm_net"].values.astype(float)

    base_briese  = briese_index(base_comm)
    quote_briese = briese_index(quote_comm)
    differential = round(base_briese - quote_briese, 1)

    # Score: differential mapped to 0-10 (0 diff = 5.0)
    score = round(max(0.0, min(10.0, (differential / 100.0) * 5.0 + 5.0)), 1)

    # Cross-specific signals
    divergence  = None
    exhaustion  = None
    flip        = None
    turning     = False
    prev_n_base  = briese_series(base_comm)
    prev_base    = prev_n_base[-4] if len(prev_n_base) >= 4 else base_briese
    prev_n_quote = briese_series(quote_comm)
    prev_quote   = prev_n_quote[-4] if len(prev_n_quote) >= 4 else quote_briese

    prev_diff = prev_base - prev_quote
    if prev_diff < 0 and differential > 0:
        flip = "bull_flip"
        turning = True
    elif prev_diff > 0 and differential < 0:
        flip = "bear_flip"
        turning = True

    if abs(differential) > 40 and abs(differential) > abs(prev_diff) * 1.2:
        divergence = f"{base_leg} Briese: {base_briese:.0f} vs {quote_leg}: {quote_briese:.0f}"

    cot_phase, cot_phase_dir, cot_phase_label, cot_phase_desc = (
        (1, "bull", "Cross Bull P1", f"{base_leg} smart money long vs {quote_leg}") if differential >= 40
        else (1, "bear", "Cross Bear P1", f"{quote_leg} smart money long vs {base_leg}") if differential <= -40
        else (0, "neutral", "Cross Neutral", "Differential mid-range")
    )

    return {
        "score":       score,
        "label":       f"Cross COT {differential:+.0f}",
        "comm_index":  round(differential, 1),
        "lspec_index": round(base_briese, 1),
        "sspec_index": round(quote_briese, 1),
        "comm_net":    int(base_comm[-1]),
        "lspec_net":   int(quote_comm[-1]),
        "sspec_net":   None,
        "turning":     turning,
        "alignment":   "bull" if differential > 20 else "bear" if differential < -20 else None,
        "signal_detail": f"Differential {differential:+.0f} | {base_leg}: {base_briese:.0f} | {quote_leg}: {quote_briese:.0f}",
        "divergence":  divergence,
        "exhaustion":  exhaustion,
        "flip":        flip,
        "oi_signal":   None,
        "normalise_signal": False,
        "flatten_signal":   False,
        "convergence_signal": abs(differential) > 30,
        "comm_momentum_signal": None,
        "detail": {
            "comm_index":    round(differential, 1),
            "lspec_index":   round(base_briese, 1),
            "sspec_index":   round(quote_briese, 1),
            "base_briese":   round(base_briese, 1),
            "quote_briese":  round(quote_briese, 1),
            "differential":  differential,
            "base_leg":      base_leg,
            "quote_leg":     quote_leg,
            "cross":         True,
            "cot_phase":     cot_phase,
            "cot_phase_dir": cot_phase_dir,
            "cot_phase_label": cot_phase_label,
            "cot_phase_desc": cot_phase_desc,
            "turning":       turning,
            "signal_detail": f"Differential {differential:+.0f} | {base_leg}: {base_briese:.0f} | {quote_leg}: {quote_briese:.0f}",
            "divergence":    divergence,
            "exhaustion":    exhaustion,
            "flip":          flip,
            "oi_signal":     None,
            "comm_net":      int(base_comm[-1]),
            "lspec_net":     int(quote_comm[-1]),
            "sspec_net":     None,
            "comm_index_lt": round(base_briese, 1),
            "comm_index_st": round(base_briese, 1),
            "lspec_index_lt":round(quote_briese, 1),
            "lspec_index_st":round(quote_briese, 1),
            "sspec_index_lt":50.0,
            "sspec_index_st":50.0,
            "alignment":     "bull" if differential > 20 else "bear" if differential < -20 else None,
            "convergence_signal": abs(differential) > 30,
            "normalise_signal": False,
            "flatten_signal": False,
            "comm_momentum_signal": None,
            "lspec_chg_3w": None,
            "lspec_chg_pct": None,
        },
    }


# ============================================================
# PRICE / MOMENTUM
# ============================================================

PRICE_CACHE     = {}
PRICE_CACHE_TTL = 3600 * 2  # 2h
PRICE_CACHE_MAX = 80         # max entries (each ~1-5MB DataFrame) — evict oldest on overflow

def _price_cache_evict():
    """Evict oldest half of PRICE_CACHE entries if over the max limit."""
    data_keys = [k for k in PRICE_CACHE if not k.endswith('_t')]
    if len(data_keys) > PRICE_CACHE_MAX:
        # Sort by timestamp, remove oldest half
        oldest = sorted(data_keys, key=lambda k: PRICE_CACHE.get(k + '_t', 0))
        for k in oldest[:len(oldest)//2]:
            PRICE_CACHE.pop(k, None)
            PRICE_CACHE.pop(k + '_t', None)
        print(f"[PRICE_CACHE] evicted {len(oldest)//2} entries, {len(data_keys)-len(oldest)//2} remain")


_YF_TIMEOUT = 20  # seconds — hard cap per yfinance call to prevent scoring loop hangs

def _yf_with_timeout(fn, *args, timeout=_YF_TIMEOUT, label="yf", **kwargs):
    """Run a yfinance callable in a thread with a hard timeout.
    Returns None and logs a warning if it exceeds the timeout."""
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
        _fut = _ex.submit(fn, *args, **kwargs)
        try:
            return _fut.result(timeout=timeout)
        except _cf.TimeoutError:
            print(f"[yf_timeout] {label} exceeded {timeout}s — skipping", flush=True)
            return None
        except Exception as _e:
            print(f"[yf_error] {label}: {_e}", flush=True)
            return None

def fetch_price_data(yf_ticker: str) -> Optional[pd.DataFrame]:
    now = time.time()
    if yf_ticker in PRICE_CACHE and (now - PRICE_CACHE.get(yf_ticker + "_t", 0)) < PRICE_CACHE_TTL:
        return PRICE_CACHE[yf_ticker]
    try:
        tk = yf.Ticker(yf_ticker)
        df = _yf_with_timeout(tk.history, period="1y", interval="1d", label=yf_ticker)
        if df is None:
            return None
        _price_cache_evict()
        PRICE_CACHE[yf_ticker]          = df
        PRICE_CACHE[yf_ticker + "_t"]   = now
        return df
    except Exception as _e:
        print(f"[fetch_price_data] {yf_ticker}: {_e}")
        return None


def fetch_price_data_long(yf_ticker: str) -> Optional[pd.DataFrame]:
    """Fetch up to 5 years of daily price data."""
    cache_key = yf_ticker + "_long"
    now = time.time()
    if cache_key in PRICE_CACHE and (now - PRICE_CACHE.get(cache_key + "_t", 0)) < PRICE_CACHE_TTL:
        return PRICE_CACHE[cache_key]
    try:
        tk = yf.Ticker(yf_ticker)
        df = _yf_with_timeout(tk.history, period="5y", interval="1d", label=yf_ticker+"_long")
        if df is None:
            return None
        _price_cache_evict()
        PRICE_CACHE[cache_key]          = df
        PRICE_CACHE[cache_key + "_t"]   = now
        return df
    except Exception as _e:
        print(f"[fetch_price_data_long] {yf_ticker}: {_e}")
        return None


def score_momentum(yf_ticker: str) -> dict:
    """
    Momentum scoring — multi-timeframe EMA/SMA stack with trend confirmation.
    Short-term momentum weighted more heavily per user instruction.
    """
    df = fetch_price_data(yf_ticker)
    if df is None or df.empty or len(df) < 20:
        return {"score": 5.0, "label": "No Data", "detail": {}}

    closes = df["Close"].values.astype(float)
    curr   = closes[-1]
    hi52   = np.nanmax(closes[-252:]) if len(closes) >= 252 else np.nanmax(closes)
    lo52   = np.nanmin(closes[-252:]) if len(closes) >= 252 else np.nanmin(closes)
    pct_range = round((curr - lo52) / (hi52 - lo52) * 100, 1) if (hi52 - lo52) > 0 else 50.0

    def _ema(arr, n):
        s = pd.Series(arr)
        return s.ewm(span=n, adjust=False).mean().values

    def _sma(arr, n):
        return pd.Series(arr).rolling(n, min_periods=1).mean().values

    ema8    = _ema(closes, 8)[-1]
    ema20   = _ema(closes, 20)[-1]
    ema21   = _ema(closes, 21)[-1]
    ema50   = _ema(closes, 50)[-1]
    sma200  = _sma(closes, 200)[-1] if len(closes) >= 200 else _sma(closes, len(closes))[-1]

    sma200_above    = curr > sma200
    sma200_pct_diff = round((curr - sma200) / sma200 * 100, 2) if sma200 > 0 else 0

    # Slope as %/week (annualised short-term momentum)
    ema8_arr  = _ema(closes, 8)
    ema20_arr = _ema(closes, 20)
    ema_st_slope_pct = round((ema8_arr[-1] - ema8_arr[-5]) / ema8_arr[-5] * 100, 2) if (len(ema8_arr) >= 5 and ema8_arr[-5] > 0) else 0
    ema_slope_pct    = round((ema20_arr[-1] - ema20_arr[-5]) / ema20_arr[-5] * 100, 2) if (len(ema20_arr) >= 5 and ema20_arr[-5] > 0) else 0

    roc1w  = round((closes[-1] / closes[-2]  - 1) * 100, 2) if len(closes) >= 2  else 0
    roc4w  = round((closes[-1] / closes[-5]  - 1) * 100, 2) if len(closes) >= 5  else 0
    roc13w = round((closes[-1] / closes[-14] - 1) * 100, 2) if len(closes) >= 14 else 0
    roc26w = round((closes[-1] / closes[-27] - 1) * 100, 2) if len(closes) >= 27 else roc13w

    # ── Regime / trend inputs for the scoring engine, on a WEEKLY series ───────
    # The factor ROCs above run on daily bars (their thresholds are calibrated that
    # way). The ENGINE's regime gate and multi-timeframe trend need true weekly
    # horizons, so resample to weekly here. Winsorise weekly returns first to absorb
    # front-month roll gaps that would otherwise spike the efficiency ratio.
    try:
        _wk = df["Close"].resample("W-FRI").last().dropna()
        wk_closes = _wk.values.astype(float)
    except Exception:
        wk_closes = closes[::5]   # fallback: every 5th daily bar ≈ weekly
    if len(wk_closes) >= 6:
        _wk_ret = np.diff(wk_closes) / wk_closes[:-1]
        _cap = np.nanpercentile(np.abs(_wk_ret), 98) if len(_wk_ret) >= 10 else np.inf
        _wk_ret = np.clip(_wk_ret, -_cap, _cap)                 # winsorise roll gaps
        wk_clean = wk_closes[0] * np.concatenate([[1.0], np.cumprod(1 + _wk_ret)])
    else:
        wk_clean = wk_closes
    # Kaufman Efficiency Ratio over ~26 WEEKS: |net move| / sum(|week-to-week moves|).
    # 0 = pure chop, 1 = clean trend. Regime-gate input (validated, Rounds 10-12).
    _ern = 26 if len(wk_clean) >= 27 else max(4, len(wk_clean) - 1)
    _er_seg = wk_clean[-(_ern + 1):]
    _net = abs(_er_seg[-1] - _er_seg[0])
    _path = float(np.sum(np.abs(np.diff(_er_seg))))
    efficiency_ratio = round(_net / _path, 4) if _path > 0 else 0.0
    # True weekly trend horizons for the engine's multi-timeframe read
    roc_lt_pct = round((wk_clean[-1] / wk_clean[-27] - 1) * 100, 2) if len(wk_clean) >= 27 else \
                 (round((wk_clean[-1] / wk_clean[0] - 1) * 100, 2) if len(wk_clean) >= 2 else 0.0)
    roc_st_pct = round((wk_clean[-1] / wk_clean[-5] - 1) * 100, 2) if len(wk_clean) >= 5 else 0.0

    # Sub-scores: each −2 to +2
    # WEIGHT RATIONALE (v2 — evidence-led):
    # Academic evidence (Moskowitz et al., AQR) shows medium-term momentum (8-26w)
    # has the strongest predictive power for 3-6w forward returns (SR 1.0-1.8).
    # Short-term (1-4w) IC shows REVERSAL at weekly horizons, not continuation.
    # Therefore: tilt toward 13w and 26w; reduce 1w and EMA-8 (noise at this horizon).
    # The 200 SMA serves as a long-term trend filter / regime confirmation.
    sub_scores = {}

    # 26w ROC (medium-long trend: 6 months) — highest weight per academic evidence
    if roc26w > 18:  sub_scores["roc26w"] = 2
    elif roc26w > 7: sub_scores["roc26w"] = 1
    elif roc26w < -18:sub_scores["roc26w"] = -2
    elif roc26w < -7: sub_scores["roc26w"] = -1
    else:              sub_scores["roc26w"] = 0

    # 13w ROC (medium term: 3 months) — second highest weight
    if roc13w > 13:  sub_scores["roc13w"] = 2
    elif roc13w > 5: sub_scores["roc13w"] = 1
    elif roc13w < -13:sub_scores["roc13w"] = -2
    elif roc13w < -5: sub_scores["roc13w"] = -1
    else:              sub_scores["roc13w"] = 0

    # 4w ROC (short-medium: 1 month) — moderate weight
    if roc4w > 10:   sub_scores["roc4w"] = 2
    elif roc4w > 3:  sub_scores["roc4w"] = 1
    elif roc4w < -10:sub_scores["roc4w"] = -2
    elif roc4w < -3: sub_scores["roc4w"] = -1
    else:             sub_scores["roc4w"] = 0

    # 200 SMA position (long-term trend filter)
    if sma200_pct_diff > 10:    sub_scores["sma200"] = 2
    elif sma200_pct_diff > 3:   sub_scores["sma200"] = 1
    elif sma200_pct_diff < -10: sub_scores["sma200"] = -2
    elif sma200_pct_diff < -3:  sub_scores["sma200"] = -1
    else:                        sub_scores["sma200"] = 0

    # 1w ROC — removed: weekly IC shows reversal tendency at 3-6w forward horizon
    # (Moskowitz et al. / AQR: short-term IC negative at 3-6w; omitted from composite)

    # Weighted sum: tilted toward medium-term (13-26w) per academic evidence
    # Previous: roc26w=0.35, roc13w=0.30, roc4w=0.20, sma200=0.10, roc1w=0.05
    # v3: roc26w=0.37, roc13w=0.33, roc4w=0.20, sma200=0.10 (roc1w removed, weight redistributed)
    weights = {"roc26w": 0.37, "roc13w": 0.33, "roc4w": 0.20, "sma200": 0.10}
    raw = sum(sub_scores.get(k, 0) * w for k, w in weights.items())
    # Map -2..+2 to 0..10
    score = round(max(0.0, min(10.0, raw * 2.5 + 5.0)), 1)

    # ── MULTI-TIMEFRAME SUB-SCORES (r15) ─────────────────────────────────
    # The composite score above blends all horizons into one number, which
    # loses shape information: a strong LT uptrend with a stalling ST leg
    # gets scored the same as a market that's ripping across all horizons.
    # Expose three timeframe reads (0-10 each) and a majority-rule vote so
    # the composite engine can weight confirmation correctly.
    #
    # ST  (1-4 weeks):  EMA8-vs-EMA21 slope + 4w ROC (roc4w)
    # MT  (4-13 weeks): EMA20-vs-EMA50 slope + 13w ROC
    # LT  (13-26 weeks): 26w ROC + price vs SMA200
    def _clip10(x):
        return round(max(0.0, min(10.0, x)), 1)
    # ST: normalise slope and 4w ROC to ±1 each, sum → -2..+2 → 0..10
    _st_slope_n = max(-1.0, min(1.0, ema_st_slope_pct / 2.5))   # ±2.5% = full
    _st_roc_n   = max(-1.0, min(1.0, roc4w / 6.0))              # ±6% = full
    mom_st_score = _clip10(5.0 + (_st_slope_n + _st_roc_n) * 2.5)
    # MT: EMA20-vs-EMA50 slope + 13w ROC
    _mt_slope_n = max(-1.0, min(1.0, ema_slope_pct / 4.0))       # ±4% = full
    _mt_roc_n   = max(-1.0, min(1.0, roc13w / 10.0))             # ±10% = full
    mom_mt_score = _clip10(5.0 + (_mt_slope_n + _mt_roc_n) * 2.5)
    # LT: 26w ROC + SMA200 %-diff
    _lt_roc_n   = max(-1.0, min(1.0, roc26w / 18.0))             # ±18% = full
    _lt_sma_n   = max(-1.0, min(1.0, sma200_pct_diff / 10.0))    # ±10% = full
    mom_lt_score = _clip10(5.0 + (_lt_roc_n + _lt_sma_n) * 2.5)

    # Each horizon votes ±1 / 0 with a 0.5-point deadband
    def _tf_sign(s):
        d = s - 5.0
        return 1 if d > 0.5 else -1 if d < -0.5 else 0
    st_sign = _tf_sign(mom_st_score)
    mt_sign = _tf_sign(mom_mt_score)
    lt_sign = _tf_sign(mom_lt_score)
    # Majority-rule momentum vote:
    #   3 agree → ±1.0 (full trend confirmation)
    #   2 agree → ±0.5 (partial confirmation)
    #   split or all-neutral → 0 (abstain)
    _up   = sum(1 for s in (st_sign, mt_sign, lt_sign) if s > 0)
    _down = sum(1 for s in (st_sign, mt_sign, lt_sign) if s < 0)
    if _up == 3:      mtf_vote =  1.0
    elif _down == 3:  mtf_vote = -1.0
    elif _up == 2 and _down == 0:  mtf_vote =  0.5
    elif _down == 2 and _up == 0:  mtf_vote = -0.5
    else:                          mtf_vote =  0.0

    if score >= 7.5:  label = "Strong Uptrend"
    elif score >= 6.0:label = "Mild Uptrend"
    elif score >= 4.5:label = "Neutral"
    elif score >= 3.0:label = "Mild Downtrend"
    else:              label = "Strong Downtrend"

    # Add MTF colour to the label when horizons disagree (visual signal)
    if abs(mtf_vote) == 0.5:
        label += " (mixed)"
    elif mtf_vote == 0.0 and (_up + _down) >= 2:
        label += " (whipsaw)"

    return {
        "score": score,
        "label": label,
        "detail": {
            "price": round(float(curr), 4),
            "hi52":  round(float(hi52), 4),
            "lo52":  round(float(lo52), 4),
            "pct_range": float(pct_range),
            "ema8": round(float(ema8), 4),
            "ema20": round(float(ema20), 4),
            "ema21": round(float(ema21), 4),
            "ema50": round(float(ema50), 4),
            "sma200": round(float(sma200), 4),
            "sma200_above":    bool(sma200_above),
            "sma200_pct_diff": float(sma200_pct_diff),
            "ema_st_slope_pct": float(ema_st_slope_pct),
            "ema_slope_pct":    float(ema_slope_pct),
            "roc1w_pct":  float(roc1w),
            "roc4w_pct":  float(roc4w),
            "roc13w_pct": float(roc13w),
            "roc26w_pct": float(roc26w),
            "efficiency_ratio": float(efficiency_ratio),
            "roc_lt_pct": float(roc_lt_pct),   # true ~26-week trend (engine MTF)
            "roc_st_pct": float(roc_st_pct),   # true ~4-week trend  (engine MTF)
            "sub_scores": sub_scores,
            # r15 multi-timeframe breakdown
            "mom_st_score": mom_st_score,
            "mom_mt_score": mom_mt_score,
            "mom_lt_score": mom_lt_score,
            "mtf_st_sign":  st_sign,
            "mtf_mt_sign":  mt_sign,
            "mtf_lt_sign":  lt_sign,
            "mtf_vote":     mtf_vote,   # -1.0, -0.5, 0, 0.5, 1.0
        },
    }


# ============================================================
# FOREX FACTORY — FF-BASED MACRO ENGINE
# ============================================================

FF_CACHE: dict = {"data": None, "time": 0}
FF_CACHE_TTL = 3600 * 3  # 3h

FF_MACRO_CACHE: dict = {"data": None, "time": 0}
FF_MACRO_CACHE_TTL = 3600  # 1 hour — keeps macro data fresh within one cache cycle

US_MACRO_CACHE: dict = {"data": None, "time": 0}
US_MACRO_TTL = 3600  # 1 hour — aligns with main scores cache TTL

_FF_MONTH_CACHE: dict = {}

# FRED fallback caches
FRED_CACHE = {}
FRED_CACHE_TIME_MAP = {}
FRED_CACHE_TTL = 3600 * 6
# Per-currency FRED economy-score cache (was used but never initialised at module
# scope — defining here removes a latent NameError on the first scoring call).
_FRED_CCY_CACHE: dict = {}
_FRED_CCY_TTL = 3600          # 1h; ff_injected entries use a 24h TTL (checked inline)

# Per-currency FRED indicator config (was referenced but never defined — so every
# non-USD currency silently fell back to a flat 5.0 macro score). These are long-
# standing OECD/Eurostat series on FRED. Each entry:
#   key: (fred_id, transform, higher_is_good, label, category)
# transform: 'yoy' (index → year-over-year %), 'level' (use level vs trailing avg).
# If any series id fails to resolve in prod it is simply skipped (graceful), and the
# live ForexFactory surprise tilt still drives the currency macro.
_FRED_CCY_SERIES: dict = {
    "EUR": {
        "cpi":   ("CP0000EZ19M086NEST", "yoy",   False, "Euro Area HICP",          "inflation"),
        "unemp": ("LRHUTTTTEZM156S",     "level", False, "Euro Area Unemployment",  "jobs"),
        "ip":    ("EA19PRINTO01GYSAM",   "level", True,  "Euro Area Industrial Prod","growth"),
    },
    "GBP": {
        "cpi":   ("GBRCPIALLMINMEI", "yoy",   False, "UK CPI",          "inflation"),
        "unemp": ("LRHUTTTTGBM156S", "level", False, "UK Unemployment", "jobs"),
        "ip":    ("GBRPROINDMISMEI", "yoy",   True,  "UK Industrial Prod","growth"),
    },
    "JPY": {
        "cpi":   ("JPNCPIALLMINMEI", "yoy",   False, "Japan CPI",          "inflation"),
        "unemp": ("LRHUTTTTJPM156S", "level", False, "Japan Unemployment", "jobs"),
        "ip":    ("JPNPROINDMISMEI", "yoy",   True,  "Japan Industrial Prod","growth"),
    },
    "AUD": {
        "cpi":   ("AUSCPIALLQINMEI", "yoy",   False, "Australia CPI",          "inflation"),
        "unemp": ("LRHUTTTTAUM156S", "level", False, "Australia Unemployment", "jobs"),
    },
    "CAD": {
        "cpi":   ("CANCPIALLMINMEI", "yoy",   False, "Canada CPI",          "inflation"),
        "unemp": ("LRHUTTTTCAM156S", "level", False, "Canada Unemployment", "jobs"),
        "ip":    ("CANPROINDMISMEI", "yoy",   True,  "Canada Industrial Prod","growth"),
    },
    "CHF": {
        "cpi":   ("CHECPIALLMINMEI", "yoy",   False, "Swiss CPI",          "inflation"),
        "unemp": ("LRHUTTTTCHM156S", "level", False, "Swiss Unemployment", "jobs"),
    },
    "NZD": {
        "cpi":   ("NZLCPIALLQINMEI", "yoy",   False, "NZ CPI",          "inflation"),
        "unemp": ("LRHUTTTTNZQ156S", "level", False, "NZ Unemployment", "jobs"),
    },
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


# FF indicator map for USD — maps event name substrings to (category, higher_is_good)
US_MACRO_INDICATOR_MAP = {
    # Jobs
    "Non-Farm Employment Change": ("JOBS",     True),
    "ADP Non-Farm Employment":    ("JOBS",     True),
    "Unemployment Claims":        ("CLAIMS",   False),
    "Unemployment Rate":          ("UNEMP",    False),
    "JOLTS Job Openings":         ("JOLTS",    True),
    "Average Hourly Earnings":    ("WAGES",    True),
    # Growth
    "GDP":                        ("GDP",      True),
    "ISM Manufacturing PMI":      ("MFG_PMI",  True),
    "ISM Services PMI":           ("SVC_PMI",  True),
    "Manufacturing PMI":          ("MFG_PMI",  True),
    "Services PMI":               ("SVC_PMI",  True),
    "Core Retail Sales":          ("RETAIL",   True),
    "Retail Sales":               ("RETAIL",   True),
    "Industrial Production":      ("MFG_PMI",  True),
    # Inflation
    "CPI":                        ("CPI",      False),  # higher CPI = bearish for most
    "Core CPI":                   ("CORE_CPI", False),
    "PPI":                        ("PPI",      False),
    "Core PCE Price Index":       ("PCE",      False),
    "PCE Price Index":            ("PCE",      False),
}

# FF indicator maps for non-USD currencies
FF_CURRENCY_INDICATOR_MAP = {
    "EUR": {
        "German Ifo Business Climate": ("growth",    True),
        "German ZEW Economic Sentiment":("growth",   True),
        "Flash Manufacturing PMI":      ("MFG_PMI",  True),
        "Flash Services PMI":           ("SVC_PMI",  True),
        "CPI":                          ("CPI",      False),
        "Core CPI":                     ("CPI",      False),
        "Unemployment Rate":            ("UNEMP",    False),
        "GDP":                          ("GDP",      True),
        "Retail Sales":                 ("RETAIL",   True),
        "ECB":                          ("rates",    None),
    },
    "GBP": {
        "GDP":                          ("GDP",      True),
        "CPI":                          ("CPI",      False),
        "Core CPI":                     ("CPI",      False),
        "Claimant Count Change":        ("CLAIMS",   False),
        "Unemployment Rate":            ("UNEMP",    False),
        "Manufacturing PMI":            ("MFG_PMI",  True),
        "Services PMI":                 ("SVC_PMI",  True),
        "Retail Sales":                 ("RETAIL",   True),
        "Average Earnings Index":       ("WAGES",    True),
        "BOE":                          ("rates",    None),
    },
    "JPY": {
        "Tankan":                        ("growth",  True),
        "GDP":                           ("GDP",     True),
        "CPI":                           ("CPI",     False),
        "Tokyo Core CPI":                ("CPI",     False),
        "Unemployment Rate":             ("UNEMP",   False),
        "Manufacturing PMI":             ("MFG_PMI", True),
        "Services PMI":                  ("SVC_PMI", True),
        "Industrial Production":         ("MFG_PMI", True),
        "Retail Sales":                  ("RETAIL",  True),
        "BOJ":                           ("rates",   None),
    },
    "AUD": {
        "Employment Change":             ("JOBS",    True),
        "Unemployment Rate":             ("UNEMP",   False),
        "CPI":                           ("CPI",     False),
        "GDP":                           ("GDP",     True),
        "Manufacturing PMI":             ("MFG_PMI", True),
        "Services PMI":                  ("SVC_PMI", True),
        "Retail Sales":                  ("RETAIL",  True),
        "Trade Balance":                 ("RETAIL",  True),
        "RBA":                           ("rates",   None),
    },
    "CAD": {
        "Employment Change":             ("JOBS",    True),
        "Unemployment Rate":             ("UNEMP",   False),
        "CPI":                           ("CPI",     False),
        "GDP":                           ("GDP",     True),
        "Manufacturing PMI":             ("MFG_PMI", True),
        "Retail Sales":                  ("RETAIL",  True),
        "Trade Balance":                 ("RETAIL",  True),
        "BOC":                           ("rates",   None),
    },
    "CHF": {
        "CPI":                           ("CPI",     False),
        "GDP":                           ("GDP",     True),
        "Manufacturing PMI":             ("MFG_PMI", True),
        "Unemployment Rate":             ("UNEMP",   False),
        "Retail Sales":                  ("RETAIL",  True),
        "SNB":                           ("rates",   None),
    },
    "NZD": {
        "GDP":                           ("GDP",     True),
        "CPI":                           ("CPI",     False),
        "Employment Change":             ("JOBS",    True),
        "Unemployment Rate":             ("UNEMP",   False),
        "Manufacturing PMI":             ("MFG_PMI", True),
        "Retail Sales":                  ("RETAIL",  True),
        "RBNZ":                          ("rates",   None),
    },
}

# Score scales per indicator type (for normalising FF surprise magnitude)
_FF_INDICATOR_SCALES = {
    "Non-Farm Employment Change": 80000.0,
    "ADP Non-Farm Employment":    30000.0,
    "Unemployment Claims":         15000.0,
    "Unemployment Rate":           0.15,
    "Average Hourly Earnings":     0.1,
    "JOLTS Job Openings":         200000.0,
    "GDP":                         0.3,
    "CPI":                         0.2,
    "Core CPI":                    0.15,
    "PPI":                         0.2,
    "PCE Price Index":             0.15,
    "Core PCE Price Index":        0.15,
    "ISM Manufacturing PMI":       1.5,
    "ISM Services PMI":            1.5,
    "Manufacturing PMI":           1.5,
    "Services PMI":                1.5,
    "Core Retail Sales":           0.4,
    "Retail Sales":                0.4,
    "Industrial Production":       0.3,
    "Employment Change":           5000.0,
    "Average Earnings Index":      0.1,
    "Claimant Count Change":       5000.0,
}


def _parse_ff_value(v) -> Optional[float]:
    """Parse FF value string like '3.2%', '178K', '-0.5M', '2.71T' -> float."""
    if v is None or v == "" or v == "—":
        return None
    s = str(v).strip().replace(",", "")
    multipliers = {"K": 1000.0, "M": 1000000.0, "B": 1000000000.0, "T": 1000000000000.0}
    try:
        for suffix, mult in multipliers.items():
            if s.upper().endswith(suffix):
                return float(s[:-1]) * mult
        return float(s.replace("%", ""))
    except Exception:
        return None


def _fetch_ff_month(year: int, month: int) -> list:
    """
    Fetch one month of Forex Factory calendar events.
    FF calendar is Cloudflare-blocked server-side in this environment.
    Returns an empty list immediately — callers fall back to FRED/COT/regime data.
    """
    return []


def _fetch_ff_months_parallel(year_month_pairs: list) -> list:
    """
    Fetch multiple months in parallel and return a flat list of event dicts,
    each with: ts, currency, name, actual, forecast, previous, impactClass.
    Uses the shared app executor to avoid thread pool deadlocks.
    """
    flat_events = []
    _ex_months = _cf.ThreadPoolExecutor(max_workers=3)  # capped: FF month fetches, each makes HTTP calls
    try:
        futs = {_ex_months.submit(_fetch_ff_month, y, m): (y, m) for y, m in year_month_pairs}
        done, pending = _cf.wait(futs, timeout=45)  # hard 45s wall-clock cap
        for fut in pending:
            fut.cancel()
        for fut in done:
            try:
                days = fut.result()  # list of day-dicts from new _fetch_ff_month
                for day in days:
                    if not isinstance(day, dict):
                        continue
                    for ev in day.get('events', []):
                        if not isinstance(ev, dict):
                            continue
                        dl = ev.get('dateline') or ev.get('ts')
                        if not dl:
                            continue
                        flat_events.append({
                            'ts':          float(dl),
                            'currency':    ev.get('currency', ''),
                            'name':        ev.get('name', ''),
                            'actual':      ev.get('actual', '') or '',
                            'forecast':    ev.get('forecast', '') or '',
                            'previous':    ev.get('previous', '') or '',
                            'impactClass': ev.get('impactClass', ''),
                            'dateline':    float(dl),
                        })
            except Exception:
                pass
    finally:
        _ex_months.shutdown(wait=False)
    return flat_events


# ── ForexFactory Labour Surprise Cache ───────────────────────────────────────
# Fetches last 12 weeks of FF calendar pages from sandbox (no Cloudflare block)
# and extracts actual vs forecast for key US labour market events.
# Cache TTL 4h — refreshed automatically on each get_all_scores() cycle.

_FF_LABOUR_CACHE: dict = {"data": None, "time": 0}
_FF_LABOUR_CACHE_TTL = 3600  # 1 hour — aligns with main scores cache TTL

# Key labour event names as they appear on ForexFactory
# Maps FF event name → internal key. Includes BOTH the old HTML scrape names
# (forexfactory.com) AND the faireconomy.media JSON feed title variants so that
# whichever source populates the store, events are always matched.
_FF_LABOUR_EVENTS = {
    # ── NFP — HTML name vs JSON title ────────────────────────────────────────
    "Non-Farm Employment Change":        {"key": "nfp",    "unit": "K",  "higher_is_good": True},
    "Non-Farm Payrolls":                 {"key": "nfp",    "unit": "K",  "higher_is_good": True},  # faireconomy title
    "Nonfarm Payrolls":                  {"key": "nfp",    "unit": "K",  "higher_is_good": True},  # alternate spelling
    # ── ADP ──────────────────────────────────────────────────────────────────
    "ADP Non-Farm Employment Change":    {"key": "adp",    "unit": "K",  "higher_is_good": True},
    "ADP Non-Farm Employment":           {"key": "adp",    "unit": "K",  "higher_is_good": True},  # faireconomy title
    "ADP Nonfarm Employment":            {"key": "adp",    "unit": "K",  "higher_is_good": True},
    # ── Unemployment ─────────────────────────────────────────────────────────
    "Unemployment Rate":                 {"key": "unrate", "unit": "%",  "higher_is_good": False},
    # ── Claims ───────────────────────────────────────────────────────────────
    "Unemployment Claims":               {"key": "claims", "unit": "K",  "higher_is_good": False},
    "Initial Jobless Claims":            {"key": "claims", "unit": "K",  "higher_is_good": False},  # faireconomy title
    # ── JOLTS ─────────────────────────────────────────────────────────────────
    "JOLTS Job Openings":                {"key": "jolts",  "unit": "M",  "higher_is_good": True},
    "JOLTs Job Openings":                {"key": "jolts",  "unit": "M",  "higher_is_good": True},  # capitalisation variant
    # ── Wages ─────────────────────────────────────────────────────────────────
    "Average Hourly Earnings m/m":       {"key": "wages",  "unit": "%",  "higher_is_good": True},
    "Avg. Hourly Earnings m/m":          {"key": "wages",  "unit": "%",  "higher_is_good": True},  # faireconomy abbrev
}


def _fetch_ff_week_html(week_str: str) -> list:
    """
    Fetch one week of FF calendar events from HTML (JSON blobs embedded).
    week_str: e.g. 'may3.2026', 'apr26.2026'
    Returns flat list of event dicts with keys: name, actual, forecast,
    previous, currency, dateline, impactClass
    """
    import re as _re
    url = f"https://www.forexfactory.com/calendar?week={week_str}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": "https://www.forexfactory.com/",
    }
    try:
        r = requests.get(url, timeout=8, headers=headers)
        if r.status_code != 200:
            return []
        html = r.text
        # Extract JSON event blobs embedded in the HTML
        pattern = r'\{"id":\d+,"ebaseId":\d+,"name":"[^"]+.*?\}(?=,\{"id"|\])'
        blobs = _re.findall(pattern, html, _re.DOTALL)
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
    except Exception as _e:
        print(f"[FF Labour] week={week_str} fetch error: {_e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# FAIR ECONOMY JSON CALENDAR FEED  (reliable replacement for the forexfactory.com
# HTML scrape, which Cloudflare-blocks datacenter IPs in production)
# ────────────────────────────────────────────────────────────────────────────
# forexfactory.com publishes its calendar as JSON via the Fair Economy CDN. That
# CDN does NOT IP-block, so it works from Render/cloud. The feed is current-week
# only, so we MERGE released prints into a small on-disk store to build the rolling
# history the labour-EMS / surprise scoring needs.
_FF_JSON_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    # Note: ff_calendar_lastweek.json returns 404 — faireconomy only serves thisweek/nextweek.
    # Historical USD labour/inflation events are injected via inject_ff_labour.py (daily cron)
    # which uses the ForexFactory HTML scrape from the sandbox (accessible there, not on Render).
)
_FF_STORE_PATH   = os.path.join(DATA_DIR, "ff_event_store.json")
_FF_STORE_MAX_DAYS = 180
_FF_JSON_CACHE   = {"data": None, "time": 0}
_FF_JSON_TTL     = 1800  # 30 min

def _ff_impact_norm(s) -> str:
    s = (s or "").lower()
    if "high" in s:   return "high"
    if "med" in s:    return "medium"
    if "low" in s:    return "low"
    return ""

def fetch_ff_calendar_json(force: bool = False) -> list:
    """This-week + last-week FF calendar from the Fair Economy JSON feed, merged and
    deduped, mapped to the same event dict shape the rest of the pipeline expects
    (name/actual/forecast/previous/currency/dateline/impactClass).
    Fetches ALL URLs (not just the first successful one) so released events at
    the week boundary are never missed."""
    now = time.time()
    if not force and _FF_JSON_CACHE["data"] is not None and (now - _FF_JSON_CACHE["time"]) < _FF_JSON_TTL:
        return _FF_JSON_CACHE["data"]
    out = []
    seen_keys: set = set()
    for url in _FF_JSON_URLS:
        try:
            r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15, follow_redirects=True)
            if r.status_code != 200:
                continue
            raw = r.json()
            for e in raw:
                iso = e.get("date", "") or ""
                try:
                    ts = datetime.fromisoformat(iso).timestamp()
                except Exception:
                    ts = None
                ev = {
                    "name":        e.get("title", "") or "",
                    "actual":      e.get("actual", "") or "",
                    "forecast":    e.get("forecast", "") or "",
                    "previous":    e.get("previous", "") or "",
                    "currency":    e.get("country", "") or "",   # FF 'country' holds the currency code
                    "dateline":    ts,
                    "impactClass": _ff_impact_norm(e.get("impact", "")),
                }
                # Deduplicate by currency|name|day so merging two weeks never double-counts
                _day = iso[:10] if iso else "na"
                _key = f"{ev['currency']}|{ev['name']}|{_day}"
                if _key not in seen_keys:
                    seen_keys.add(_key)
                    out.append(ev)
        except Exception as _e:
            print(f"[FF JSON] fetch error {url}: {_e}", flush=True)
    _FF_JSON_CACHE["data"] = out
    _FF_JSON_CACHE["time"] = now
    return out

def _ff_store_load() -> dict:
    try:
        with open(_FF_STORE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}

def _ff_store_save(store: dict) -> None:
    try:
        with open(_FF_STORE_PATH, "w") as f:
            json.dump(store, f)
    except Exception as _e:
        print(f"[FF store] save error: {_e}", flush=True)

def refresh_ff_event_store() -> list:
    """Merge newly-released events (those with an 'actual') from the live feed into the
    persistent store, keyed by currency|name|day so re-fetches are idempotent. Prunes to
    _FF_STORE_MAX_DAYS. Returns the full list of stored event dicts."""
    store = _ff_store_load()
    for ev in fetch_ff_calendar_json():
        if not ev.get("actual"):          # keep only RELEASED prints (have an actual)
            continue
        ts = ev.get("dateline")
        day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") if ts else "na"
        store[f"{ev.get('currency','')}|{ev.get('name','')}|{day}"] = ev
    cutoff = time.time() - _FF_STORE_MAX_DAYS * 86400
    store = {k: v for k, v in store.items() if (v.get("dateline") or 0) >= cutoff}
    _ff_store_save(store)
    return list(store.values())

# Per-currency macro-surprise tilt — name-substring → (category, higher_is_good)
_FF_SURPRISE_MAP = [
    ("Non-Farm Employment",   ("jobs",   True)),
    ("Employment Change",     ("jobs",   True)),
    ("ADP",                   ("jobs",   True)),
    ("Unemployment Rate",     ("jobs",   False)),
    ("Unemployment Claims",   ("jobs",   False)),
    ("Claimant Count",        ("jobs",   False)),
    ("Average Hourly Earnings",("wages", True)),
    ("Average Earnings",      ("wages",  True)),
    ("GDP",                   ("growth", True)),
    ("Retail Sales",          ("growth", True)),
    ("Manufacturing PMI",     ("growth", True)),
    ("Services PMI",          ("growth", True)),
    ("ISM",                   ("growth", True)),
    ("Industrial Production", ("growth", True)),
    ("Ifo Business Climate",  ("growth", True)),
    ("ZEW Economic Sentiment",("growth", True)),
    ("Tankan",                ("growth", True)),
    ("CPI",                   ("inflation", True)),   # hot inflation = hawkish = currency-supportive
    ("PPI",                   ("inflation", True)),
    ("PCE",                   ("inflation", True)),
]
_FF_SURPRISE_HALFLIFE_DAYS = 21   # ~3 weeks: a surprise fades to half-weight after 3wk

def compute_ff_surprise_tilt(currency: str, store_events: list = None) -> dict:
    """Recency-decayed macro-surprise tilt for a currency, in [-1, +1], from the FF
    store's released prints. Each surprise is signed (actual vs forecast, oriented by
    higher_is_good), weighted by impact (high=1.0, medium=0.5) and an exponential
    half-life (~3wk). Returns {tilt, n, detail}. tilt 0 = no recent surprise signal."""
    if store_events is None:
        store_events = _ff_store_load().values()
    now = time.time()
    num = 0.0; den = 0.0; n = 0; contribs = []
    for ev in store_events:
        if ev.get("currency") != currency:
            continue
        a = _parse_ff_value(ev.get("actual", ""))
        f = _parse_ff_value(ev.get("forecast", ""))
        if a is None or f is None:
            continue
        name = ev.get("name", "")
        cat_info = next((ci for sub, ci in _FF_SURPRISE_MAP if sub in name), None)
        if cat_info is None:
            continue
        _cat, higher_is_good = cat_info
        denom = max(abs(f), 1e-9)
        rel = (a - f) / denom                       # relative surprise
        signed = rel if higher_is_good else -rel
        signed = max(-1.0, min(1.0, signed * 4.0))  # scale & clamp (25% beat ~ full)
        impact_w = 1.0 if ev.get("impactClass") == "high" else 0.5 if ev.get("impactClass") == "medium" else 0.2
        ts = ev.get("dateline") or now
        age_days = max(0.0, (now - ts) / 86400.0)
        decay = 0.5 ** (age_days / _FF_SURPRISE_HALFLIFE_DAYS)
        w = impact_w * decay
        num += signed * w; den += w; n += 1
        contribs.append((name, round(signed, 2), round(w, 2)))
    tilt = round(num / den, 3) if den > 0 else 0.0
    contribs.sort(key=lambda x: -x[2])
    return {"tilt": tilt, "n": n, "detail": contribs[:6]}



def _get_week_strings(n_weeks: int = 12) -> list:
    """
    Generate last n_weeks FF week URL strings (week starts Sunday).
    FF format: 'may3.2026', 'apr26.2026'
    """
    import calendar as _cal
    from datetime import date, timedelta
    today = date.today()
    # Walk back to the most recent Sunday
    day_of_week = today.weekday()  # Mon=0, Sun=6
    days_since_sunday = (day_of_week + 1) % 7
    current_sunday = today - timedelta(days=days_since_sunday)
    months_short = ["jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec"]
    week_strings = []
    for i in range(n_weeks):
        sunday = current_sunday - timedelta(weeks=i)
        mon = months_short[sunday.month - 1]
        week_strings.append(f"{mon}{sunday.day}.{sunday.year}")
    return week_strings


def _fetch_ff_labour_surprises(force: bool = False) -> dict:
    """
    Fetch last 12 weeks of ForexFactory calendar (HTML) and extract
    actual vs forecast surprise data for key US labour market events.
    Returns a dict with per-event lists of releases and aggregate surprise scores.
    Cached for _FF_LABOUR_CACHE_TTL (4h) — auto-refreshes on each score cycle.
    """
    global _FF_LABOUR_CACHE
    now = time.time()
    if not force and _FF_LABOUR_CACHE["data"] and (now - _FF_LABOUR_CACHE["time"]) < _FF_LABOUR_CACHE_TTL:
        return _FF_LABOUR_CACHE["data"]

    # Source: refresh the persistent FF event store from the Fair Economy JSON feed
    # (reliable; the old per-week forexfactory.com scrape is Cloudflare-blocked in prod),
    # then take the last 14 weeks of released prints from the store.
    try:
        store_events = refresh_ff_event_store()
    except Exception as _se:
        print(f"[FF Labour] store refresh error: {_se}", flush=True)
        store_events = list(_ff_store_load().values())
    _cutoff = now - 14 * 7 * 86400
    all_events = [e for e in store_events if (e.get("dateline") or 0) >= _cutoff]

    # Filter: USD only, has actual AND forecast, is a key labour event
    releases: dict = {key_info["key"]: [] for key_info in _FF_LABOUR_EVENTS.values()}

    for ev in all_events:
        if ev.get("currency") != "USD":
            continue
        name = ev.get("name", "")
        actual_str   = ev.get("actual", "")
        forecast_str = ev.get("forecast", "")
        if not actual_str or not forecast_str or actual_str in ("", "—") or forecast_str in ("", "—"):
            continue
        for event_name, meta in _FF_LABOUR_EVENTS.items():
            if name == event_name:
                actual_raw   = _parse_ff_value(actual_str)
                forecast_raw = _parse_ff_value(forecast_str)
                previous_raw = _parse_ff_value(ev.get("previous", ""))
                if actual_raw is None or forecast_raw is None:
                    break
                surprise_raw = actual_raw - forecast_raw
                # Normalise for display (convert to same units as forecast)
                unit = meta["unit"]
                if unit == "K":
                    # NFP/ADP/Claims: values already in persons from _parse_ff_value (multiplied by 1000)
                    # Display in thousands for readability
                    actual_disp   = round(actual_raw / 1000, 1) if abs(actual_raw) > 500 else round(actual_raw, 2)
                    forecast_disp = round(forecast_raw / 1000, 1) if abs(forecast_raw) > 500 else round(forecast_raw, 2)
                    surprise_disp = round(surprise_raw / 1000, 1) if abs(surprise_raw) > 500 else round(surprise_raw, 2)
                elif unit == "M":
                    actual_disp   = round(actual_raw / 1e6, 3)
                    forecast_disp = round(forecast_raw / 1e6, 3)
                    surprise_disp = round(surprise_raw / 1e6, 3)
                else:  # % etc
                    actual_disp   = round(actual_raw, 2)
                    forecast_disp = round(forecast_raw, 2)
                    surprise_disp = round(surprise_raw, 2)

                # Direction: positive surprise = actual beat expectation
                beat = surprise_raw > 0 if meta["higher_is_good"] else surprise_raw < 0
                releases[meta["key"]].append({
                    "dateline":    ev.get("dateline"),
                    "actual":      actual_disp,
                    "forecast":    forecast_disp,
                    "previous":    round(previous_raw / 1000, 1) if (previous_raw and unit == "K" and abs(previous_raw) > 500) else (
                                   round(previous_raw / 1e6, 3) if (previous_raw and unit == "M") else
                                   (round(previous_raw, 2) if previous_raw else None)),
                    "surprise":    surprise_disp,
                    "beat":        beat,
                    "unit":        unit,
                })
                break

    # Sort each event's releases chronologically (oldest → newest)
    for key in releases:
        releases[key].sort(key=lambda x: x["dateline"] or 0)

    # Compute rolling EMS-style surprise scores per event (last 6 releases)
    # Score per release: +1 beat, -1 miss (weighted by magnitude quartile)
    def _ems_score(rel_list: list) -> Optional[float]:
        recent = rel_list[-8:]  # last 8 releases
        if not recent:
            return None
        hits = [1 if r["beat"] else -1 for r in recent]
        # Weight: most recent 4 get 1.5x weight
        weights = [1.0] * len(hits)
        for i in range(max(0, len(hits) - 4), len(hits)):
            weights[i] = 1.5
        raw = sum(h * w for h, w in zip(hits, weights)) / sum(weights)
        # Convert -1..+1 → 0..10
        return round(max(0, min(10, raw * 3 + 5)), 1)

    scores = {}
    latest = {}  # Most recent release for each metric (for display)
    for key, rel_list in releases.items():
        if rel_list:
            scores[key] = _ems_score(rel_list)
            latest[key] = rel_list[-1]  # most recent

    # Composite EMS score (average of available metric scores)
    valid_scores = [s for s in scores.values() if s is not None]
    composite_ems = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

    result = {
        "releases":      releases,
        "scores":        scores,
        "latest":        latest,
        "composite_ems": composite_ems,
        "fetched_at":    now,
        "n_events_found": sum(len(v) for v in releases.values()),
    }

    _FF_LABOUR_CACHE["data"] = result
    _FF_LABOUR_CACHE["time"] = now
    print(f"[FF Labour] Refreshed — {result['n_events_found']} releases across {len(releases)} metrics, composite EMS: {composite_ems}")
    return result


# ── ForexFactory Inflation Surprises ─────────────────────────────────────────
# Mirrors the labour pattern exactly. FF provides real consensus + actual for
# CPI m/m, CPI y/y, Core CPI m/m, PPI m/m, Core PPI m/m, Core PCE m/m.
# These replace the FRED rolling-average "expected" with real market forecasts.

_FF_INFL_CACHE: dict = {"data": None, "time": 0}
_FF_INFL_CACHE_TTL = 3600  # 1 hour

# Exact event names as they appear on ForexFactory (USD events only)
# For inflation lower=more hawkish, but for the SURPRISE direction:
#   higher_is_good=False means a beat (actual > forecast) is BEARISH (hotter than expected)
_FF_INFL_EVENTS = {
    "CPI m/m":               {"key": "cpi_mom",     "unit": "%", "higher_is_good": False},
    "CPI y/y":               {"key": "cpi_yoy",     "unit": "%", "higher_is_good": False},
    "Core CPI m/m":          {"key": "core_cpi_mom","unit": "%", "higher_is_good": False},
    "Core CPI y/y":          {"key": "core_cpi_yoy","unit": "%", "higher_is_good": False},
    "PPI m/m":               {"key": "ppi_mom",     "unit": "%", "higher_is_good": False},
    "PPI y/y":               {"key": "ppi_yoy",     "unit": "%", "higher_is_good": False},
    "Core PPI m/m":          {"key": "core_ppi_mom","unit": "%", "higher_is_good": False},
    "Core PCE Price Index m/m": {"key": "core_pce_mom", "unit": "%", "higher_is_good": False},
}


def _fetch_ff_inflation_surprises(force: bool = False) -> dict:
    """
    Fetch last 14 weeks of ForexFactory calendar and extract actual vs forecast
    for key US inflation events. Mirrors _fetch_ff_labour_surprises exactly.
    Returns per-event release lists + latest + composite hot/cool score.
    """
    global _FF_INFL_CACHE
    now = time.time()
    if not force and _FF_INFL_CACHE["data"] and (now - _FF_INFL_CACHE["time"]) < _FF_INFL_CACHE_TTL:
        return _FF_INFL_CACHE["data"]

    # Source: the persistent FF event store (Fair Economy feed + inject cron).
    # FIX (2026-07-19): the old per-week forexfactory.com HTML scrape is
    # Cloudflare-blocked on Render, so this fetcher silently returned 0 events in
    # production and the inflation FF overlay never fired. Use the store, same
    # as _fetch_ff_labour_surprises.
    try:
        store_events = refresh_ff_event_store()
    except Exception as _se:
        print(f"[FF Inflation] store refresh error: {_se}", flush=True)
        store_events = list(_ff_store_load().values())
    _cutoff = now - 16 * 7 * 86400  # 16 weeks ≈ 4 months (CPI/PCE monthly releases)
    all_events = [e for e in store_events if (e.get("dateline") or 0) >= _cutoff]

    # Deduplicate (store is keyed, but belt-and-braces)
    seen = set()
    unique_events = []
    for ev in all_events:
        dedup_key = (ev.get("name"), ev.get("dateline"), ev.get("actual"))
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_events.append(ev)

    releases: dict = {meta["key"]: [] for meta in _FF_INFL_EVENTS.values()}

    for ev in unique_events:
        if ev.get("currency") != "USD":
            continue
        name = ev.get("name", "")
        actual_str   = ev.get("actual", "")
        forecast_str = ev.get("forecast", "")
        if not actual_str or not forecast_str or actual_str in ("", "—") or forecast_str in ("", "—"):
            continue
        for event_name, meta in _FF_INFL_EVENTS.items():
            if name == event_name:
                actual_raw   = _parse_ff_value(actual_str)
                forecast_raw = _parse_ff_value(forecast_str)
                previous_raw = _parse_ff_value(ev.get("previous", ""))
                if actual_raw is None or forecast_raw is None:
                    break
                surprise_raw  = actual_raw - forecast_raw
                actual_disp   = round(actual_raw, 2)
                forecast_disp = round(forecast_raw, 2)
                surprise_disp = round(surprise_raw, 3)
                # For inflation: beat = actual > forecast = HOTTER than expected
                beat = actual_raw > forecast_raw  # True = hot surprise
                releases[meta["key"]].append({
                    "dateline": ev.get("dateline"),
                    "actual":   actual_disp,
                    "forecast": forecast_disp,
                    "previous": round(previous_raw, 2) if previous_raw is not None else None,
                    "surprise": surprise_disp,
                    "beat":     beat,   # True = hotter than forecast
                    "unit":     meta["unit"],
                })
                break

    # Sort chronologically
    for key in releases:
        releases[key].sort(key=lambda x: x["dateline"] or 0)

    # Most recent release per metric
    latest = {key: rel_list[-1] for key, rel_list in releases.items() if rel_list}

    # Hot/Cool score per metric: +1 if hot beat, -1 if cool miss, weighted recent
    def _hot_score(rel_list: list) -> Optional[float]:
        recent = rel_list[-6:]
        if not recent:
            return None
        hits = [1 if r["beat"] else -1 for r in recent]
        weights = [1.0] * len(hits)
        for i in range(max(0, len(hits) - 3), len(hits)):
            weights[i] = 1.5
        raw = sum(h * w for h, w in zip(hits, weights)) / sum(weights)
        return round(max(0, min(10, raw * 3 + 5)), 1)  # 0=cool, 5=neutral, 10=hot

    scores = {}
    for key, rel_list in releases.items():
        if rel_list:
            scores[key] = _hot_score(rel_list)

    # Composite heat score
    valid_scores = [s for s in scores.values() if s is not None]
    composite_heat = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

    result = {
        "releases":       releases,
        "scores":         scores,
        "latest":         latest,
        "composite_heat": composite_heat,
        "fetched_at":     now,
        "n_events_found": sum(len(v) for v in releases.values()),
    }

    _FF_INFL_CACHE["data"] = result
    _FF_INFL_CACHE["time"] = now
    print(f"[FF Inflation] Refreshed — {result['n_events_found']} releases, composite heat: {composite_heat}")
    return result


# ── ForexFactory Growth Surprises ────────────────────────────────────────────
# Mirrors the labour/inflation pattern. FF provides real consensus + actual for
# GDP q/q (advance/prelim/final), ISM Mfg/Services PMI, Retail Sales m/m,
# Core Retail Sales m/m and CB Consumer Confidence.

_FF_GROWTH_CACHE: dict = {"data": None, "time": 0}
_FF_GROWTH_CACHE_TTL = 3600  # 1 hour

# Exact event names as they appear on ForexFactory (USD events only).
# All growth events: higher_is_good=True — a beat (actual > forecast) is BULLISH.
_FF_GROWTH_EVENTS = {
    "Advance GDP q/q":        {"key": "gdp",         "unit": "%",   "higher_is_good": True},
    "Prelim GDP q/q":         {"key": "gdp",         "unit": "%",   "higher_is_good": True},
    "Final GDP q/q":          {"key": "gdp",         "unit": "%",   "higher_is_good": True},
    "ISM Manufacturing PMI":  {"key": "ism_mfg",     "unit": "idx", "higher_is_good": True},
    "ISM Services PMI":       {"key": "ism_svc",     "unit": "idx", "higher_is_good": True},
    "ISM Non-Manufacturing PMI": {"key": "ism_svc",  "unit": "idx", "higher_is_good": True},
    "Retail Sales m/m":       {"key": "retail",      "unit": "%",   "higher_is_good": True},
    "Core Retail Sales m/m":  {"key": "core_retail", "unit": "%",   "higher_is_good": True},
    "CB Consumer Confidence": {"key": "conf_cb",     "unit": "idx", "higher_is_good": True},
}


def _fetch_ff_growth_surprises(force: bool = False) -> dict:
    """
    Fetch last 26 weeks of ForexFactory calendar and extract actual vs forecast
    for key US growth events. Mirrors _fetch_ff_inflation_surprises exactly.
    Returns per-event release lists + latest + composite growth score.
    """
    global _FF_GROWTH_CACHE
    now = time.time()
    if not force and _FF_GROWTH_CACHE["data"] and (now - _FF_GROWTH_CACHE["time"]) < _FF_GROWTH_CACHE_TTL:
        return _FF_GROWTH_CACHE["data"]

    # Source: the persistent FF event store (Fair Economy feed + inject cron).
    # The old per-week forexfactory.com HTML scrape is Cloudflare-blocked on
    # Render — the store is the production-reliable path (same as labour).
    try:
        store_events = refresh_ff_event_store()
    except Exception as _se:
        print(f"[FF Growth] store refresh error: {_se}", flush=True)
        store_events = list(_ff_store_load().values())
    _cutoff = now - 26 * 7 * 86400  # 26 weeks ≈ 6 months (ensures ≥2 quarterly GDP prints)
    all_events = [e for e in store_events if (e.get("dateline") or 0) >= _cutoff]

    # Deduplicate (store is keyed, but belt-and-braces)
    seen = set()
    unique_events = []
    for ev in all_events:
        dedup_key = (ev.get("name"), ev.get("dateline"), ev.get("actual"))
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_events.append(ev)

    releases: dict = {meta["key"]: [] for meta in _FF_GROWTH_EVENTS.values()}

    for ev in unique_events:
        if ev.get("currency") != "USD":
            continue
        name = ev.get("name", "")
        actual_str   = ev.get("actual", "")
        forecast_str = ev.get("forecast", "")
        if not actual_str or not forecast_str or actual_str in ("", "—") or forecast_str in ("", "—"):
            continue
        for event_name, meta in _FF_GROWTH_EVENTS.items():
            if name == event_name:
                actual_raw   = _parse_ff_value(actual_str)
                forecast_raw = _parse_ff_value(forecast_str)
                previous_raw = _parse_ff_value(ev.get("previous", ""))
                if actual_raw is None or forecast_raw is None:
                    break
                surprise_raw  = actual_raw - forecast_raw
                # Growth: beat = stronger than expected = BULLISH; respects
                # higher_is_good so future lower-is-better events wire in correctly
                beat = surprise_raw > 0 if meta.get("higher_is_good", True) else surprise_raw < 0
                releases[meta["key"]].append({
                    "dateline": ev.get("dateline"),
                    "actual":   round(actual_raw, 2),
                    "forecast": round(forecast_raw, 2),
                    "previous": round(previous_raw, 2) if previous_raw is not None else None,
                    "surprise": round(surprise_raw, 3),
                    "beat":     beat,
                    "unit":     meta["unit"],
                    "event":    event_name,
                })
                break

    # Sort chronologically
    for key in releases:
        releases[key].sort(key=lambda x: x["dateline"] or 0)

    # Most recent release per metric
    latest = {key: rel_list[-1] for key, rel_list in releases.items() if rel_list}

    # Growth momentum score per metric: +1 beat, -1 miss, recent releases weighted 1.5x
    def _growth_score(rel_list: list) -> Optional[float]:
        recent = rel_list[-6:]
        if not recent:
            return None
        hits = [1 if r["beat"] else -1 for r in recent]
        weights = [1.0] * len(hits)
        for i in range(max(0, len(hits) - 3), len(hits)):
            weights[i] = 1.5
        raw = sum(h * w for h, w in zip(hits, weights)) / sum(weights)
        return round(max(0, min(10, raw * 3 + 5)), 1)  # 0=contracting, 5=neutral, 10=expanding

    scores = {}
    for key, rel_list in releases.items():
        if rel_list:
            scores[key] = _growth_score(rel_list)

    valid_scores = [s for s in scores.values() if s is not None]
    composite_growth = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else None

    result = {
        "releases":         releases,
        "scores":           scores,
        "latest":           latest,
        "composite_growth": composite_growth,
        "fetched_at":       now,
        "n_events_found":   sum(len(v) for v in releases.values()),
    }

    _FF_GROWTH_CACHE["data"] = result
    _FF_GROWTH_CACHE["time"] = now
    print(f"[FF Growth] Refreshed — {result['n_events_found']} releases, composite growth: {composite_growth}")
    return result


def compute_fred_economy_score(currency: str) -> dict:
    """
    FRED-based economy score for a non-USD currency.
    Uses trailing-average surprise method: actual vs 3-period trailing average.
    Returns same schema as compute_ff_economy_score for frontend compatibility.
    """
    now = time.time()
    cached = _FRED_CCY_CACHE.get(currency)
    if cached:
        # FF-injected entries use 24h TTL; FRED fallback uses standard 1h TTL
        ttl = 86400 if cached.get("data", {}).get("source") == "ff_injected" else _FRED_CCY_TTL
        if (now - cached["time"]) < ttl:
            return cached["data"]

    series_map = _FRED_CCY_SERIES.get(currency)
    if not series_map:
        result = {"score": 5.0, "label": f"{currency} Macro Neutral", "currency": currency,
                  "cats": {}, "cat_details": []}
        _FRED_CCY_CACHE[currency] = {"data": result, "time": now}
        return result

    cat_scores: dict = {}
    cat_details: dict = {}

    for key, (fred_id, transform, higher_is_good, label, category) in series_map.items():
        try:
            periods = 24 if transform == "qoq" else 18
            raw = fetch_fred_series(fred_id, periods)
            if not raw or len(raw) < 4:
                continue

            vals = [x["value"] for x in raw if x.get("value") is not None]
            if len(vals) < 4:
                continue

            # Compute the statistic
            if transform == "level":
                actual = vals[-1]
                hist   = vals[-4:-1]
            elif transform == "mom":
                # Month-over-month % change
                if len(vals) < 2:
                    continue
                actual = (vals[-1] / vals[-2] - 1.0) * 100 if vals[-2] != 0 else 0
                chgs   = [(vals[i]/vals[i-1]-1.0)*100 for i in range(max(1,len(vals)-4), len(vals)-1) if vals[i-1] != 0]
                hist   = chgs if chgs else [0]
            elif transform == "qoq":
                if len(vals) < 2:
                    continue
                actual = (vals[-1] / vals[-2] - 1.0) * 100 if vals[-2] != 0 else 0
                chgs   = [(vals[i]/vals[i-1]-1.0)*100 for i in range(max(1,len(vals)-4), len(vals)-1) if vals[i-1] != 0]
                hist   = chgs if chgs else [0]
            elif transform == "yoy":
                # Infer sampling frequency from the series dates so YoY uses the
                # correct lag: monthly->12, quarterly->4, annual->1. (Bug fix:
                # AUS/NZ CPI are quarterly; a hardcoded 12-period lag compared
                # points ~3yr apart and produced absurd readings like +13% CPI.)
                lag = 12
                try:
                    dts = [datetime.strptime(x["date"], "%Y-%m-%d")
                           for x in raw if x.get("value") is not None and x.get("date")]
                    if len(dts) >= 3:
                        gaps = sorted((dts[i] - dts[i-1]).days for i in range(1, len(dts)))
                        mgap = gaps[len(gaps)//2]
                        if mgap >= 250:   lag = 1     # annual
                        elif mgap >= 75:  lag = 4     # quarterly
                        else:             lag = 12    # monthly
                except Exception:
                    lag = 12
                if len(vals) < lag + 1:
                    continue
                actual = (vals[-1] / vals[-1-lag] - 1.0) * 100 if vals[-1-lag] != 0 else 0
                yoys   = [(vals[i]/vals[i-lag]-1.0)*100
                          for i in range(max(lag, len(vals)-4), len(vals)-1)
                          if i-lag >= 0 and vals[i-lag] != 0]
                hist   = yoys if yoys else [0]
            else:
                continue

            if not hist:
                continue

            expected = sum(hist) / len(hist)
            surprise = actual - expected
            scale    = max(abs(expected) * 0.3, 0.1)

            if not higher_is_good:
                surprise = -surprise

            # Score: -2 to +2
            s = int(round(max(-2.0, min(2.0, surprise / scale))))

            # Format display values
            if transform in ("mom", "qoq", "yoy"):
                actual_disp   = f"{actual:+.2f}%"
                forecast_disp = f"{expected:+.2f}%"
            elif transform == "level":
                if "unemployment" in label.lower() or "%" in label:
                    actual_disp   = f"{actual:.1f}%"
                    forecast_disp = f"{expected:.2f}%"
                else:
                    actual_disp   = str(round(actual, 2))
                    forecast_disp = str(round(expected, 2))
            else:
                actual_disp   = str(round(actual, 3))
                forecast_disp = str(round(expected, 3))

            if category not in cat_scores:
                cat_scores[category] = []
                cat_details[category] = []

            cat_scores[category].append(s)
            cat_details[category].append({
                "name":     label,
                "actual":   actual_disp,
                "forecast": forecast_disp,
                "score":    s,
            })

        except Exception as _e:
            continue

    if not cat_scores:
        result = {"score": 5.0, "label": f"{currency} Macro Neutral", "currency": currency,
                  "cats": {}, "cat_details": {}}
        _FRED_CCY_CACHE[currency] = {"data": result, "time": now}
        return result

    cat_avgs = {cat: sum(v) / len(v) for cat, v in cat_scores.items()}
    raw = sum(cat_avgs.values()) / max(1, len(cat_avgs))
    raw = max(-2.0, min(2.0, raw))
    # Confidence dampening: with fewer data points, pull toward neutral (5.0)
    # 1 data point → 50% weight toward neutral; 2 → 67%; 3+ → 80%+; 5+ → full weight
    total_indicators = sum(len(v) for v in cat_scores.values())
    confidence = min(1.0, total_indicators / 5.0)  # full confidence at 5+ indicators
    raw = raw * confidence  # dampen toward 0 (which maps to 5.0 on 0-10 scale)
    score = round((raw + 2.0) / 4.0 * 10.0, 1)
    score = max(0.0, min(10.0, score))

    if raw >= 1.0:    label = f"{currency} Macro Strong"
    elif raw >= 0.3:  label = f"{currency} Macro Improving"
    elif raw <= -1.0: label = f"{currency} Macro Weak"
    elif raw <= -0.3: label = f"{currency} Macro Deteriorating"
    else:              label = f"{currency} Macro Neutral"

    result = {"score": score, "label": label, "currency": currency,
              "cats": cat_avgs, "cat_details": cat_details}
    _FRED_CCY_CACHE[currency] = {"data": result, "time": now}
    return result


def compute_ff_economy_score(events: list, currency: str) -> dict:
    """FRED-based macro score, overlaid with the live ForexFactory surprise tilt
    (recency-decayed, from the Fair Economy feed store). FRED remains the base and the
    fallback: when the FF store has no recent surprises for the currency, tilt = 0 and
    the score is pure FRED."""
    try:
        base = compute_fred_economy_score(currency)
    except Exception as _fe:
        print(f"[FRED macro] {currency} failed ({_fe}) — using neutral base + FF surprise", flush=True)
        base = {"score": 5.0, "label": f"{currency} Macro Neutral", "currency": currency, "cats": {}, "cat_details": []}
    try:
        st = compute_ff_surprise_tilt(currency)
        tilt = st.get("tilt", 0.0)
        if tilt:
            base = dict(base)
            base_score = base.get("score", 5.0)
            # A full +1 surprise composite nudges the macro read by 1.5 points (bounded).
            base["score"] = round(max(0.0, min(10.0, base_score + tilt * 1.5)), 1)
            base["label"] = _macro_label(base["score"] - 5.0) if "_macro_label" in globals() else base.get("label")
            base["ff_surprise_tilt"] = tilt
            base["ff_surprise_n"]    = st.get("n", 0)
            base["ff_surprise_detail"] = st.get("detail", [])
    except Exception as _te:
        print(f"[FF surprise] {currency}: {_te}", flush=True)
    return base


def compute_all_ff_macro() -> dict:
    """
    Returns economy scores for all major currencies — all computed from
    the same Forex Factory calendar data.
    USD score is sourced from compute_macro_all() for consistency.
    """
    now = time.time()
    if FF_MACRO_CACHE["data"] and (now - FF_MACRO_CACHE["time"]) < FF_MACRO_CACHE_TTL:
        return FF_MACRO_CACHE["data"]

    # Use FRED-based economy scores (FF calendar is blocked in this environment)
    # Fetch all 7 currencies in parallel — reduces cold-start from 7*5*12s=420s to ~12s
    result = {}
    CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    _ccy_ex = _cf.ThreadPoolExecutor(max_workers=7)
    try:
        _ccy_futs = {_ccy_ex.submit(compute_fred_economy_score, curr): curr for curr in CURRENCIES}
        done_ccy, _ = _cf.wait(_ccy_futs, timeout=30)  # 30s hard cap
        for fut in done_ccy:
            try:
                ccy_result = fut.result()
                result[ccy_result["currency"]] = ccy_result
            except Exception:
                pass
        # Fill in any that timed out with neutral placeholders
        for curr in CURRENCIES:
            if curr not in result:
                result[curr] = {"score": 5.0, "label": f"{curr} Macro Neutral", "currency": curr, "cats": {}}
    finally:
        _ccy_ex.shutdown(wait=False)

    # USD from compute_macro_all
    us_macro = compute_macro_all()
    cats = us_macro.get("category_scores", {})

    # Blend FF EMS jobs score into USD category score if available
    # This ensures FX pair macro differentials use real Bloomberg-consensus jobs surprise
    _ff_lab_data = _FF_LABOUR_CACHE.get("data")
    _jobs_cat_raw = cats.get("jobs", 0)  # -2..+2 FRED trailing avg
    if _ff_lab_data and _ff_lab_data.get("composite_ems") is not None:
        # FF EMS is 0-10; convert to -2..+2 for blending
        _ff_jobs_raw = (_ff_lab_data["composite_ems"] - 5.0) / 2.5
        # 60% FF consensus, 40% FRED trailing
        _jobs_cat_raw = round(_ff_jobs_raw * 0.60 + _jobs_cat_raw * 0.40, 3)
        cats = {**cats, "jobs": _jobs_cat_raw}  # non-destructive copy

    # Normalise US cat scores (-2..+2) to 0-10
    _s = sum(cats.values()) / max(1, len(cats)) if cats else 0
    usd_raw   = max(-2.0, min(2.0, _s))
    usd_score = round((usd_raw + 2.0) / 4.0 * 10.0, 1)
    if usd_raw >= 1.0:    usd_label = "USD Macro Strong"
    elif usd_raw >= 0.3:  usd_label = "USD Macro Improving"
    elif usd_raw <= -1.0: usd_label = "USD Macro Weak"
    elif usd_raw <= -0.3: usd_label = "USD Macro Deteriorating"
    else:                  usd_label = "USD Macro Neutral"
    result["USD"] = {
        "score": usd_score, "label": usd_label, "currency": "USD",
        "cats": {
            "UNEMP":   cats.get("jobs", 0),
            "CLAIMS":  cats.get("jobs", 0),
            "JOLTS":   cats.get("jobs", 0),
            "WAGES":   cats.get("jobs", 0),
            "MFG_PMI": cats.get("growth", 0),
            "SVC_PMI": cats.get("growth", 0),
            "RETAIL":  cats.get("growth", 0),
            "DGS2":    cats.get("rates", 0),
        },
        "cat_details": us_macro.get("components", {}),
    }

    FF_MACRO_CACHE["data"] = result
    FF_MACRO_CACHE["time"] = now
    return result


# ============================================================
# FRED-BASED MACRO (US fallback / supplementary)
# ============================================================

FRED_SERIES = {
    "GDP":      "A191RL1Q225SBEA",
    "INDPRO":   "INDPRO",
    "CPI":      "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "PPI":      "PPIFIS",   # PPI: Final Demand (SA) — BLS headline figure. PPIACO (All Commodities) was incorrectly used before — it's a raw commodity spot index, not the market-reported PPI.
    "PCE":      "PCEPI",
    "CORE_PCE": "PCEPILFE",
    "CFNAI":    "CFNAI",
    "NFP":      "PAYEMS",
    "UNEMP":    "UNRATE",
    "CLAIMS":   "ICSA",
    "ADP":      "ADPMNUSNERSA",   # ADP National Employment Report — total private (level, persons)
    "WAGES":    "CES0500000003",   # Avg Hourly Earnings, Total Private ($/hr) — m/m % computed
    "DGS2":     "DGS2",
    "DGS10":    "DGS10",
    "YLDCRV":   "T10Y2Y",
    "T10Y3M":   "T10Y3M",
    "DFII10":   "DFII10",
    "FEDFUNDS": "FEDFUNDS",
    # CAPE not available via FRED — computed via yfinance fallback in compute_stock_climate
    "WALCL":    "WALCL",
    # Credit spreads (ICE BofA)
    "HYOAS":    "BAMLH0A0HYM2",   # US HY OAS (bps)
    "NFCI":     "NFCI",             # Chicago Fed Financial Conditions Index (weekly, -=loose)
    "STLFSI4":  "STLFSI4",          # St. Louis Fed Stress Index (weekly, +=stress)
    "IGOAS":    "BAMLC0A0CM",     # US IG OAS (bps)
    # Labour
    "JOLTS":    "JTSJOL",         # JOLTS job openings (thousands)
    # Growth / sentiment
    "UMCSENT":  "UMCSENT",        # UoM Consumer Sentiment (FRED fallback for CB Confidence)
}


def fetch_fred_series(series_id: str, periods: int = 24) -> Optional[list]:
    resolved_id = FRED_SERIES.get(series_id, series_id)
    cache_key = resolved_id
    now = time.time()
    # CACHE FIX (2026-07-19): cache stores the FULL parsed series and each caller
    # gets its own slice. Previously the cache stored only the first caller's
    # truncated slice (keyed by series id alone), so e.g. the startup prefetch
    # requesting HYOAS with periods=24 poisoned the credit dashboard's
    # periods=780 request — 3M/6M spread deltas were always None and the
    # "3-year percentile" was computed over 24 days.
    if cache_key in FRED_CACHE and (now - FRED_CACHE_TIME_MAP.get(cache_key, 0)) < FRED_CACHE_TTL:
        _full = FRED_CACHE[cache_key]
        return _full[-periods:] if len(_full) >= periods else _full
    try:
        url = FRED_BASE + resolved_id
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        data = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() != ".":
                try:
                    data.append({"date": parts[0].strip(), "value": float(parts[1].strip())})
                except Exception:
                    pass
        recent = data[-periods:] if len(data) >= periods else data
        # FIX: Cap FRED_CACHE at 60 entries — evict oldest to prevent unbounded growth
        if len(FRED_CACHE) >= 60:
            oldest_key = min(FRED_CACHE_TIME_MAP, key=FRED_CACHE_TIME_MAP.get)
            FRED_CACHE.pop(oldest_key, None)
            FRED_CACHE_TIME_MAP.pop(oldest_key, None)
        FRED_CACHE[cache_key]              = data      # store FULL series, slice per request
        FRED_CACHE_TIME_MAP[cache_key]     = now
        return recent
    except Exception as e:
        print(f"FRED error {series_id} (resolved: {resolved_id}): {e}")
        return None


# ── FRED full-history cache (used by walk-forward score history) ─────────────
# Stores complete sorted [{date, value}] for each series — no period truncation.
_FRED_FULL_CACHE: dict = {}           # {resolved_id: [{date:str, value:float}, ...]}
_FRED_FULL_CACHE_TIME: dict = {}       # {resolved_id: float timestamp}
_FRED_FULL_CACHE_TTL = 3600 * 6       # 6h — same as FRED_CACHE_TTL

def fetch_fred_series_full(series_id: str) -> Optional[list]:
    """
    Fetch the COMPLETE history for a FRED series (no period truncation).
    Returns a sorted list of {date: str (YYYY-MM-DD), value: float}.
    Used exclusively by the walk-forward score history engine.
    Results cached for 6h to avoid redundant HTTP calls during multi-market runs.
    """
    resolved_id = FRED_SERIES.get(series_id, series_id)
    now = time.time()
    cached = _FRED_FULL_CACHE.get(resolved_id)
    if cached is not None and (now - _FRED_FULL_CACHE_TIME.get(resolved_id, 0)) < _FRED_FULL_CACHE_TTL:
        return cached
    try:
        url = FRED_BASE + resolved_id
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        data = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                try:
                    data.append({"date": parts[0].strip(), "value": float(parts[1].strip())})
                except Exception:
                    pass
        data.sort(key=lambda x: x["date"])
        _FRED_FULL_CACHE[resolved_id] = data
        _FRED_FULL_CACHE_TIME[resolved_id] = now
        return data
    except Exception as e:
        print(f"FRED full-history error {series_id} (resolved: {resolved_id}): {e}")
        return None


# ── yfinance yield cache (fallback when FRED times out) ────────────────────
_YF_YIELD_CACHE: dict = {}
_YF_YIELD_CACHE_TIME: dict = {}
_YF_YIELD_CACHE_TTL = 3600  # 1h

def _fetch_yf_yield_series(ticker: str, periods: int = 270) -> Optional[list]:
    """
    Fetch Treasury yield history from yfinance as a FRED-compatible list.
    Tickers: ^IRX (3M, value/10=%), ^FVX (5Y, /10), ^TNX (10Y, /10), ^TYX (30Y, /10)
    Returns [{date: str, value: float}] with yields already in % (e.g. 4.35).
    """
    cache_key = ticker + "_yld"
    now = time.time()
    if cache_key in _YF_YIELD_CACHE and (now - _YF_YIELD_CACHE_TIME.get(cache_key, 0)) < _YF_YIELD_CACHE_TTL:
        data = _YF_YIELD_CACHE[cache_key]
        return data[-periods:] if len(data) >= periods else data
    try:
        tk = yf.Ticker(ticker)
        df = _yf_with_timeout(tk.history, period="2y", interval="1d", label=ticker+"_yld")
        if df is None or df.empty:
            return None
        data = []
        for ts, row in df.iterrows():
            try:
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, 'strftime') else str(ts)[:10]
                # yfinance ^TNX/^IRX/^FVX/^TYX are already in % (e.g. 4.54 = 4.54%)
                val = float(row["Close"])
                data.append({"date": date_str, "value": round(val, 3)})
            except Exception:
                pass
        data.sort(key=lambda x: x["date"])
        _YF_YIELD_CACHE[cache_key] = data
        _YF_YIELD_CACHE_TIME[cache_key] = now
        return data[-periods:] if len(data) >= periods else data
    except Exception as e:
        print(f"[_fetch_yf_yield_series] {ticker}: {e}")
        return None


def _compute_surprise_at_date(
    series: list,           # full sorted [{date, value}]
    bar_date_str: str,      # YYYY-MM-DD — data visible on this bar date (lag-adjusted)
    transform: str,         # 'level' | 'yoy' | 'mom'
    higher_is_good: bool,
    scale: float,
    fred_id: str = "",      # FRED series ID — used to look up publication lag
) -> int:
    """
    Compute a surprise score (-2..+2) for a FRED series at a given bar date.
    Applies publication lag: FRED dates series by observation-period START, not release.
    E.g. NFP for January is "2026-01-01" in FRED but released Feb 7. The lag corrects
    this so a bar_date of Jan 31 only sees November NFP data — what the market actually knew.
    Surprise = actual vs 3-period trailing average of prior readings.
    """
    if not series:
        return 0
    from datetime import date as _d2, timedelta as _td2
    pub_lag = _SH_FRED_PUB_LAGS.get(fred_id, 0)
    if pub_lag > 0:
        _cutoff = (_d2.fromisoformat(bar_date_str) - _td2(days=pub_lag)).isoformat()
    else:
        _cutoff = bar_date_str
    # Slice to data actually available on bar_date (after applying publication lag)
    avail = [pt for pt in series if pt["date"] <= _cutoff]
    if not avail:
        return 0

    if transform == "yoy":
        # Build YoY series from available raw data
        if len(avail) < 13:
            return 0
        yoy = []
        for i in range(12, len(avail)):
            pct = (avail[i]["value"] / avail[i - 12]["value"] - 1) * 100 \
                  if avail[i - 12]["value"] != 0 else 0.0
            yoy.append({"date": avail[i]["date"], "value": round(pct, 3)})
        if len(yoy) < 4:
            return 0
        actual = yoy[-1]["value"]
        prior  = [y["value"] for y in yoy[-4:-1]]
        expectation = sum(prior) / len(prior)
        surprise = actual - expectation

    elif transform == "mom":
        if len(avail) < 2:
            return 0
        mom = []
        for i in range(1, len(avail)):
            mom.append({"date": avail[i]["date"], "value": avail[i]["value"] - avail[i-1]["value"]})
        if len(mom) < 4:
            return 0
        actual = mom[-1]["value"]
        prior  = [m["value"] for m in mom[-4:-1]]
        expectation = sum(prior) / len(prior)
        surprise = actual - expectation

    elif transform == "qoq":
        # Quarter-over-quarter: same as mom but for quarterly series
        if len(avail) < 2:
            return 0
        qoq = []
        for i in range(1, len(avail)):
            prev = avail[i-1]["value"]
            qoq.append({"date": avail[i]["date"],
                         "value": (avail[i]["value"] / prev - 1) * 100 if prev != 0 else 0.0})
        if len(qoq) < 4:
            return 0
        actual = qoq[-1]["value"]
        prior  = [q["value"] for q in qoq[-4:-1]]
        expectation = sum(prior) / len(prior)
        surprise = actual - expectation

    else:  # level
        if len(avail) < 4:
            return 0
        actual = avail[-1]["value"]
        prior  = [p["value"] for p in avail[-4:-1]]
        expectation = sum(prior) / len(prior)
        surprise = actual - expectation

    norm = surprise / scale if scale != 0 else 0.0
    if norm > 1.5:    raw = 2
    elif norm > 0.4:  raw = 1
    elif norm < -1.5: raw = -2
    elif norm < -0.4: raw = -1
    else:             raw = 0
    return raw if higher_is_good else -raw


# US FRED series to fetch for walk-forward macro history
# ── FRED publication lags (calendar days from observation period START to public release) ──
# FRED dates series by observation-period START, not release date.
# E.g. NFP for January is in FRED as "2026-01-01" but released Feb 7.
# Adding these lags ensures a bar_date only "sees" FRED data the market actually knew.
_SH_FRED_PUB_LAGS: dict = {
    # Monthly (dated at month START, released ~15-37 days after month END):
    "PAYEMS":              37,   # NFP: 1st Friday of following month
    "UNRATE":              37,   # Unemployment: same release as NFP
    "CPIAUCSL":            45,   # CPI: ~14th of following month
    "CPILFESL":            45,   # Core CPI: same day as CPI
    "PPIFIS":              43,   # PPI: ~12th of following month
    "PCEPI":               58,   # PCE: ~4 weeks after month end
    "PCEPILFE":            58,   # Core PCE: same day as PCE
    "RSAFS":               46,   # Retail Sales: ~16th of following month
    "INDPRO":              47,   # Industrial Production: ~16th of following month
    "CFNAI":               55,   # Chicago Fed: 4th week of following month
    # Quarterly (dated at quarter START, advance estimate ~4 weeks after quarter end):
    "A191RL1Q225SBEA":    120,   # GDP advance: ~28 days after quarter end + ~90 day quarter
    # Weekly:
    "ICSA":                 7,   # Initial Claims: released Thursday for prior week
    "WALCL":                5,   # Fed BS: released Thursday for prior week
    # FX country series (all monthly, approximate):
    "HICP_EA": 45, "HICP_DE": 45, "HICP_FR": 45,
    "UNRATE_EA": 37, "UNRATE_DE": 37,
    "GDP_EA": 120, "INDPRO_EA": 47,
    "CPIUK": 45, "UNRATE_UK": 37, "GDP_UK": 120,
    "CPIJAPAN": 45, "UNRATE_JP": 37, "INDPRO_JP": 47,
    "CPIAUS": 45, "UNRATE_AU": 37, "GDP_AU": 120,
    "CPICAN": 45, "UNRATE_CA": 37, "GDP_CA": 120,
    "CPICHE": 45, "UNRATE_CH": 37,
    "CPINZ": 45, "UNRATE_NZ": 37,
}

_SH_FRED_US_SERIES: dict = {
    # key → (fred_series_id, transform, higher_is_good, scale, category)
    "GDP":       ("A191RL1Q225SBEA", "level",  True,  0.5,    "growth"),
    "INDPRO":    ("INDPRO",          "mom",    True,  0.3,    "growth"),
    "CFNAI":     ("CFNAI",           "level",  True,  0.2,    "growth"),
    "RETAIL":    ("RSAFS",           "mom",    True,  4000.0, "growth"),
    "CPI":       ("CPIAUCSL",        "yoy",    True,  0.3,    "inflation"),
    "CORE_CPI":  ("CPILFESL",        "yoy",    True,  0.3,    "inflation"),
    "PPI":       ("PPIFIS",          "yoy",    True,  0.4,    "inflation"),
    "PCE":       ("PCEPI",           "yoy",    True,  0.3,    "inflation"),
    "CORE_PCE":  ("PCEPILFE",        "yoy",    True,  0.3,    "inflation"),
    "NFP":       ("PAYEMS",          "mom",    True,  150000, "jobs"),  # mom change in thousands of persons; scale=150000 = ~1-sigma
    "UNEMP":     ("UNRATE",          "level",  False, 0.2,    "jobs"),  # higher unemp = bad; level surprise vs 3-period trailing avg
    "CLAIMS":    ("ICSA",            "mom",    False, 30000,  "jobs"),  # weekly initial claims — rising is bad
}

# FX currency → FRED series for walk-forward macro
_SH_FRED_FX_SERIES: dict = {
    "EUR": [
        ("DEUCPIALLMINMEI", "mom",  True,  0.2,  "inflation"),
        ("LRHUTTTTEZM156S", "level",False, 0.3,  "jobs"),
        ("NAEXKP01EZQ661S", "qoq",  True,  0.5,  "growth"),
    ],
    "GBP": [
        ("GBRCPIALLMINMEI", "mom",  True,  0.2,  "inflation"),
        ("LRHUTTTTGBM156S", "level",False, 0.3,  "jobs"),
        ("NAEXKP01GBQ661S", "qoq",  True,  0.5,  "growth"),
    ],
    "JPY": [
        ("JPNCPIALLMINMEI", "mom",  True,  0.2,  "inflation"),
        ("LRUN64TTJPM156S", "level",False, 0.3,  "jobs"),
        ("JPNPROINDMISMEI", "mom",  True,  0.3,  "growth"),
    ],
    "AUD": [
        ("AUSCPIALLQINMEI", "qoq",  True,  0.3,  "inflation"),
        ("LRHUTTTTAUM156S", "level",False, 0.3,  "jobs"),
    ],
    "CAD": [
        ("CANCPIALLMINMEI", "mom",  True,  0.2,  "inflation"),
        ("LRHUTTTTCAM156S", "level",False, 0.3,  "jobs"),
    ],
    "CHF": [
        ("CHECPIALLMINMEI", "mom",  True,  0.2,  "inflation"),
    ],
    "NZD": [
        ("NZLCPIALLQINMEI", "qoq",  True,  0.3,  "inflation"),
        ("LRUN64TTNZQ156S", "level",False, 0.3,  "jobs"),
    ],
}


def _score_macro_at_fred(
    market_id: str,
    bar_date_str: str,          # YYYY-MM-DD
    fred_us_full: dict,         # {key: full_series_list} pre-fetched
    fred_fx_full: dict,         # {currency: {series_id: full_series_list}} pre-fetched
) -> float:
    """
    Walk-forward macro score for a historical bar using FRED data only.
    Zero lookahead — uses only FRED data with date <= bar_date_str.
    Replaces the broken FF-based _score_macro_at (ForexFactory is blocked server-side).
    """
    FX_CCY_MAP = {"6E":"EUR","6B":"GBP","6A":"AUD","6J":"JPY",
                  "6C":"CAD","6N":"NZD","6S":"CHF","6M":"MXN","DX":"USD"}

    m = market_id.upper()
    mkt_obj = next((x for x in MARKETS if x["id"] == m), None)
    cat = mkt_obj.get("category", "") if mkt_obj else ""

    # ── Compute US macro component scores ─────────────────────────────────────
    us_cat_sums: dict = {}   # {category: [scores...]}
    us_comps: dict = {}      # {key: score_int}

    for key, (fred_id, transform, higher_is_good, scale, category) in _SH_FRED_US_SERIES.items():
        series = fred_us_full.get(key)
        if not series:
            continue
        sc = _compute_surprise_at_date(series, bar_date_str, transform, higher_is_good, scale,
                                       fred_id=fred_id)
        us_comps[key] = sc
        if category not in us_cat_sums:
            us_cat_sums[category] = []
        us_cat_sums[category].append(sc)

    us_cat_avg: dict = {cat_k: sum(v)/len(v) for cat_k, v in us_cat_sums.items() if v}
    growth_s    = us_cat_avg.get("growth",    0.0)
    jobs_s      = us_cat_avg.get("jobs",      0.0)
    inflation_s = us_cat_avg.get("inflation", 0.0)

    # Individual series
    cpi_s  = float(us_comps.get("CPI",      0))
    pce_s  = float(us_comps.get("PCE",      0))
    gdp_s  = float(us_comps.get("GDP",      0))
    infl_avg = (cpi_s + pce_s) / 2 if (cpi_s != 0 or pce_s != 0) else inflation_s
    # pmi_avg: use CFNAI + INDPRO as growth activity proxies
    cfnai_s  = float(us_comps.get("CFNAI",  0))
    indpro_s = float(us_comps.get("INDPRO", 0))
    pmi_avg  = (cfnai_s + indpro_s) / 2 if (cfnai_s != 0 or indpro_s != 0) else growth_s
    growth_s2 = (growth_s + pmi_avg) / 2 if pmi_avg else growth_s

    # dgs2_s: no direct FRED series for 2yr yield surprise in walk-forward mode.
    # Use 0 as neutral — the regime factor already captures rate path.
    dgs2_s = 0.0

    # ── Compute FX currency-specific scores ───────────────────────────────────
    ff_macro_snap: dict = {}  # Mimics ff_macro dict for get_macro_score_for_market

    def _fx_score_for_ccy(ccy: str) -> dict:
        # fred_fx_full[ccy] = {fred_id: series_list} — fetched during _do_prefetch
        # _SH_FRED_FX_SERIES[ccy] = [(fred_id, transform, higher_is_good, scale, cat_k), ...]
        ccy_series_map = fred_fx_full.get(ccy, {})
        ccy_cfg = _SH_FRED_FX_SERIES.get(ccy, [])
        if not ccy_series_map or not ccy_cfg:
            return {"score": 0.0, "cats": {}}
        cat_sums: dict = {}
        for (fred_id, transform, higher_is_good, scale, cat_k) in ccy_cfg:
            series = ccy_series_map.get(fred_id)
            if not series:
                continue
            sc = _compute_surprise_at_date(series, bar_date_str, transform, higher_is_good, scale,
                                           fred_id=fred_id)
            if cat_k not in cat_sums:
                cat_sums[cat_k] = []
            cat_sums[cat_k].append(sc)
        cat_avg = {k: sum(v)/len(v) for k, v in cat_sums.items() if v}
        agg = sum(cat_avg.values()) / len(cat_avg) if cat_avg else 0.0
        return {"score": round(agg, 2), "cats": cat_avg}

    # For FX and cross pairs, compute relevant currency scores
    if cat in ("fx", "fx_cross") or m == "DX":
        for ccy, fred_id_map in fred_fx_full.items():
            ff_macro_snap[ccy] = _fx_score_for_ccy(ccy)
    # USD always needed
    if "USD" not in ff_macro_snap:
        usd_agg = sum(us_cat_avg.values()) / len(us_cat_avg) if us_cat_avg else 0.0
        ff_macro_snap["USD"] = {"score": max(-2.0, min(2.0, usd_agg)), "cats": us_cat_avg}

    # ── Build macro dict in get_macro_score_for_market expected format ─────────
    macro_snap = {
        "category_scores": {
            "growth":    growth_s,
            "jobs":      jobs_s,
            "inflation": inflation_s,
            "rates":     0.0,   # neutral — rates captured by regime factor
            "MFG_PMI":   cfnai_s,
            "SVC_PMI":   indpro_s,
        },
        "components": {
            "GDP":      {"score": gdp_s},
            "CPI":      {"score": cpi_s},
            "PCE":      {"score": pce_s},
            "DGS2":     {"score": dgs2_s},
            "JOBS":     {"score": jobs_s},
            "MFG_PMI":  {"score": cfnai_s},
            "SVC_PMI":  {"score": indpro_s},
        },
    }

    result = get_macro_score_for_market(m, macro_snap, ff_macro=ff_macro_snap)
    return round(max(0.0, min(10.0, result.get("score", 5.0))), 1)


def surprise_score(actual: float, history: list, higher_is_good: bool = True, scale: float = 1.0) -> int:
    """
    Score a data surprise: compare actual to rolling 3-period moving average of prior values.
    Returns -2, -1, 0, +1, +2.
    higher_is_good: True if a beat (above expectation) is positive (e.g. GDP, PMI, NFP)
                   False if a beat is negative (e.g. unemployment, claims, inflation for equities)
    scale: normalisation factor for the surprise magnitude.
    """
    if not history or len(history) < 3:
        return 0
    prior = [h["value"] for h in history[-4:-1]]
    if not prior:
        return 0
    expectation = sum(prior) / len(prior)
    surprise    = actual - expectation
    norm = surprise / scale if scale != 0 else 0
    if norm > 1.5:    raw = 2
    elif norm > 0.4:  raw = 1
    elif norm < -1.5: raw = -2
    elif norm < -0.4: raw = -1
    else:              raw = 0
    return raw if higher_is_good else -raw


def _surprise_label(surprise: float, scale: float) -> str:
    norm = surprise / scale if scale != 0 else 0
    if norm > 1.5:    return "Strong Beat"
    elif norm > 0.4:  return "Beat"
    elif norm < -1.5: return "Strong Miss"
    elif norm < -0.4: return "Miss"
    else:              return "In Line"

def compute_macro_surprise(series_list: list, higher_is_good: bool, transform: str = "level",
                            scale: float = 1.0) -> dict:
    """
    Compute a surprise score for a FRED series list.
    transform: 'level' | 'yoy' | 'mom'
    Returns {score, actual, expected, surprise, label}
    """
    if not series_list or len(series_list) < 4:
        return {"score": 0, "actual": None, "expected": None, "surprise": None, "label": "No Data"}

    vals = series_list

    if transform == "yoy":
        if len(vals) < 13:
            return {"score": 0, "actual": None, "expected": None, "surprise": None, "label": "Insufficient data"}
        yoy_series = []
        for i in range(12, len(vals)):
            yoy = (vals[i]["value"] / vals[i - 12]["value"] - 1) * 100
            yoy_series.append({"date": vals[i]["date"], "value": round(yoy, 2)})
        if len(yoy_series) < 4:
            return {"score": 0, "actual": None, "expected": None, "surprise": None, "label": "Insufficient data"}
        actual = yoy_series[-1]["value"]
        expected = sum(v["value"] for v in yoy_series[-4:-1]) / 3
        surprise = actual - expected
        score = surprise_score(actual, yoy_series[:-1], higher_is_good, scale)
        return {"score": score, "actual": round(actual, 2), "expected": round(expected, 2),
                "surprise": round(surprise, 2), "label": _surprise_label(surprise, scale)}

    elif transform == "mom":
        if len(vals) < 3:
            return {"score": 0, "actual": None, "expected": None, "surprise": None, "label": "No Data"}
        mom_series = []
        for i in range(1, len(vals)):
            chg = vals[i]["value"] - vals[i-1]["value"]
            mom_series.append({"date": vals[i]["date"], "value": round(chg, 3)})
        if len(mom_series) < 4:
            return {"score": 0, "actual": None, "expected": None, "surprise": None, "label": "Insufficient data"}
        actual = mom_series[-1]["value"]
        expected = sum(v["value"] for v in mom_series[-4:-1]) / 3
        surprise = actual - expected
        score = surprise_score(actual, mom_series[:-1], higher_is_good, scale)
        return {"score": score, "actual": round(actual, 3), "expected": round(expected, 3),
                "surprise": round(surprise, 3), "label": _surprise_label(surprise, scale)}

    else:  # level
        actual = vals[-1]["value"]
        expected = sum(v["value"] for v in vals[-4:-1]) / 3
        surprise = actual - expected
        score = surprise_score(actual, vals[:-1], higher_is_good, scale)
        return {"score": score, "actual": round(actual, 2), "expected": round(expected, 2),
                "surprise": round(surprise, 2), "label": _surprise_label(surprise, scale)}


def compute_macro_all() -> dict:
    """
    Compute US macro surprise scores from FRED data (pure FRED — no FF scraping needed).
    Method: actual vs 3-period trailing average = surprise direction.
    Returns {components, category_scores, equity_overall}
    """
    now = time.time()
    if US_MACRO_CACHE["data"] is not None and (now - US_MACRO_CACHE["time"]) < US_MACRO_TTL:
        return US_MACRO_CACHE["data"]

    # ── Parallel FRED pre-fetch ─────────────────────────────────────────────
    # On cold start, fetching 15+ FRED series sequentially takes up to 180s.
    # Pre-fetch all needed series in parallel so total wait ~ 1 series (12s max).
    _FRED_PREFETCH_LIST = [
        # US macro
        ("GDP", 24), ("INDPRO", 12), ("CFNAI", 12), ("RSAFS", 12), ("UMCSENT", 14),
        ("CPI", 24), ("CORE_CPI", 24), ("PPI", 24), ("PCE", 24), ("CORE_PCE", 24),
        ("NFP", 12), ("UNEMP", 12), ("CLAIMS", 12), ("JOLTS", 12),
        ("DGS2", 30), ("YLDCRV", 30), ("WALCL", 160), ("HYOAS", 24),
        ("NFCI", 24), ("STLFSI4", 24), ("IGOAS", 24),
        ("DGS10", 30), ("T10Y3M", 30), ("DFII10", 24), ("FEDFUNDS", 36),
        # CB policy rates for intl regime
        ("IUDSOIA", 400), ("ECBDFR", 400), ("IR3TIB01JPM156N", 36),
        ("IR3TIB01AUM156N", 36), ("IR3TIB01CAM156N", 36),
        ("IR3TIB01NZM156N", 36), ("IR3TIB01CHM156N", 36), ("IR3TIB01MXM156N", 36),
        # Non-USD macro (for compute_fred_economy_score)
        ("DEUCPIALLMINMEI", 24), ("LRHUTTTTEZM156S", 18), ("NAEXKP01EZQ661S", 24),
        ("GBRCPIALLMINMEI", 24), ("LRHUTTTTGBM156S", 18), ("NAEXKP01GBQ661S", 24),
        ("JPNCPIALLMINMEI", 24), ("LRUN64TTJPM156S", 18), ("JPNPROINDMISMEI", 18),
        ("AUSCPIALLQINMEI", 24), ("LRHUTTTTAUM156S", 18),
        ("CANCPIALLMINMEI", 24), ("LRHUTTTTCAM156S", 18),
        ("CHECPIALLMINMEI", 24),
        ("NZLCPIALLQINMEI", 24), ("LRUN64TTNZQ156S", 18),
    ]
    _pf_ex = _cf.ThreadPoolExecutor(max_workers=20)
    try:
        _pf_futs = [_pf_ex.submit(fetch_fred_series, sid, periods) for sid, periods in _FRED_PREFETCH_LIST]
        _cf.wait(_pf_futs, timeout=20)  # 20s hard cap — all fetch in parallel
    finally:
        _pf_ex.shutdown(wait=False)
    # All fetched series now in FRED_CACHE — subsequent calls are instant

    components = {}

    # ── GROWTH ────────────────────────────────────────────────────────────────
    gdp_data = fetch_fred_series("GDP", 24)
    if gdp_data:
        r = compute_macro_surprise(gdp_data, higher_is_good=True, transform="level", scale=0.5)
        components["GDP"] = {**r, "title": "GDP Growth QoQ", "category": "growth",
                             "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    indpro_data = fetch_fred_series("INDPRO", 12)
    if indpro_data:
        r = compute_macro_surprise(indpro_data, higher_is_good=True, transform="mom", scale=0.3)
        components["MFG_PMI"] = {**r, "title": "Mfg PMI / Industrial Production", "category": "growth",
                                 "display": f"{r['actual']:+.2f}" if r['actual'] is not None else "—"}

    cfnai_data = fetch_fred_series("CFNAI", 12)
    if cfnai_data:
        r = compute_macro_surprise(cfnai_data, higher_is_good=True, transform="level", scale=0.2)
        components["SVC_PMI"] = {**r, "title": "Services PMI / Economic Activity", "category": "growth",
                                 "display": f"{r['actual']:+.2f}" if r['actual'] is not None else "—"}

    retail_data = fetch_fred_series("RSAFS", 12)
    if retail_data:
        r = compute_macro_surprise(retail_data, higher_is_good=True, transform="mom", scale=4000.0)
        if r["actual"] is not None:
            r["display"] = f"{r['actual']:+.0f}M"
        components["RETAIL"] = {**r, "title": "Retail Sales", "category": "growth",
                                "display": r.get("display", "—")}

    # CONF (FRED fallback) — UoM Consumer Sentiment level. FF overlays CB Consumer
    # Confidence on top when the FF event store is populated (different survey but
    # both index-level sentiment reads; FF-first display wins in the frontend).
    conf_data = fetch_fred_series("UMCSENT", 14)
    if conf_data and len(conf_data) >= 5:
        r = compute_macro_surprise(conf_data, higher_is_good=True, transform="level", scale=4.0)
        if r.get("expected") is not None:
            r["expected"] = round(r["expected"], 1)
        components["CONF"] = {**r, "title": "Consumer Confidence", "category": "growth",
                              "display": f"{r['actual']:.1f}" if r['actual'] is not None else "—"}

    # ── INFLATION ─────────────────────────────────────────────────────────────────────────
    cpi_data = fetch_fred_series("CPI", 24)
    if cpi_data:
        r = compute_macro_surprise(cpi_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["CPI"] = {**r, "title": "CPI YoY", "category": "inflation",
                              "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    core_cpi_data = fetch_fred_series("CORE_CPI", 24)
    if core_cpi_data:
        r = compute_macro_surprise(core_cpi_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["CORE_CPI"] = {**r, "title": "Core CPI YoY", "category": "inflation",
                                   "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    ppi_data = fetch_fred_series("PPI", 24)
    if ppi_data:
        r = compute_macro_surprise(ppi_data, higher_is_good=True, transform="yoy", scale=0.4)
        components["PPI"] = {**r, "title": "PPI Final Demand", "category": "inflation",
                              "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    pce_data = fetch_fred_series("PCE", 24)
    if pce_data:
        r = compute_macro_surprise(pce_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["PCE"] = {**r, "title": "PCE YoY", "category": "inflation",
                              "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    core_pce_data = fetch_fred_series("CORE_PCE", 24)
    if core_pce_data:
        r = compute_macro_surprise(core_pce_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["CORE_PCE"] = {**r, "title": "Core PCE YoY", "category": "inflation",
                                   "display": f"{r['actual']}%" if r['actual'] is not None else "—"}


    # ── JOBS ──────────────────────────────────────────────────────────────────

    # ── Overlay real ForexFactory consensus + actual onto inflation components ──
    # FF gives us real market forecasts (vs FRED rolling avg) for print-day events.
    try:
        _ff_infl = _fetch_ff_inflation_surprises()
        if _ff_infl and _ff_infl.get("n_events_found", 0) > 0:
            _ff_latest  = _ff_infl.get("latest", {})
            _ff_scores  = _ff_infl.get("scores", {})
            _ff_rels    = _ff_infl.get("releases", {})

            # CPI: inject FF CPI m/m latest + y/y
            if "CPI" in components:
                _cpi_l = _ff_latest.get("cpi_mom", {})
                if _cpi_l:
                    components["CPI"]["actual_ff"]   = _cpi_l.get("actual")
                    components["CPI"]["forecast_ff"] = _cpi_l.get("forecast")
                    components["CPI"]["surprise_ff"] = _cpi_l.get("surprise")
                    components["CPI"]["beat_ff"]     = _cpi_l.get("beat")
                    components["CPI"]["ff_score"]    = _ff_scores.get("cpi_mom")
                    components["CPI"]["ff_releases"] = _ff_rels.get("cpi_mom", [])
                _cpi_yoy_l = _ff_latest.get("cpi_yoy", {})
                if _cpi_yoy_l:
                    components["CPI"]["actual_ff_yoy"]   = _cpi_yoy_l.get("actual")
                    components["CPI"]["forecast_ff_yoy"] = _cpi_yoy_l.get("forecast")
                    components["CPI"]["surprise_ff_yoy"] = _cpi_yoy_l.get("surprise")
                    components["CPI"]["beat_ff_yoy"]     = _cpi_yoy_l.get("beat")

            # Core CPI
            if "CORE_CPI" in components:
                _ccpi_l = _ff_latest.get("core_cpi_mom", {})
                if _ccpi_l:
                    components["CORE_CPI"]["actual_ff"]   = _ccpi_l.get("actual")
                    components["CORE_CPI"]["forecast_ff"] = _ccpi_l.get("forecast")
                    components["CORE_CPI"]["surprise_ff"] = _ccpi_l.get("surprise")
                    components["CORE_CPI"]["beat_ff"]     = _ccpi_l.get("beat")
                    components["CORE_CPI"]["ff_score"]    = _ff_scores.get("core_cpi_mom")
                    components["CORE_CPI"]["ff_releases"] = _ff_rels.get("core_cpi_mom", [])

            # PPI: FF PPI m/m + Core PPI m/m
            if "PPI" in components:
                _ppi_l = _ff_latest.get("ppi_mom", {})
                if _ppi_l:
                    components["PPI"]["actual_ff"]   = _ppi_l.get("actual")
                    components["PPI"]["forecast_ff"] = _ppi_l.get("forecast")
                    components["PPI"]["surprise_ff"] = _ppi_l.get("surprise")
                    components["PPI"]["beat_ff"]     = _ppi_l.get("beat")
                    components["PPI"]["ff_score"]    = _ff_scores.get("ppi_mom")
                    components["PPI"]["ff_releases"] = _ff_rels.get("ppi_mom", [])
                _cppi_l = _ff_latest.get("core_ppi_mom", {})
                if _cppi_l:
                    components["PPI"]["core_actual_ff"]   = _cppi_l.get("actual")
                    components["PPI"]["core_forecast_ff"] = _cppi_l.get("forecast")
                    components["PPI"]["core_surprise_ff"] = _cppi_l.get("surprise")
                    components["PPI"]["core_beat_ff"]     = _cppi_l.get("beat")

            # Core PCE
            if "CORE_PCE" in components:
                _cpce_l = _ff_latest.get("core_pce_mom", {})
                if _cpce_l:
                    components["CORE_PCE"]["actual_ff"]   = _cpce_l.get("actual")
                    components["CORE_PCE"]["forecast_ff"] = _cpce_l.get("forecast")
                    components["CORE_PCE"]["surprise_ff"] = _cpce_l.get("surprise")
                    components["CORE_PCE"]["beat_ff"]     = _cpce_l.get("beat")
                    components["CORE_PCE"]["ff_score"]    = _ff_scores.get("core_pce_mom")
                    components["CORE_PCE"]["ff_releases"] = _ff_rels.get("core_pce_mom", [])

            # ── Re-score inflation components using FF m/m surprises ──────
            # BUG FIX (r15d): previously CPI/PCE score was computed from FRED
            # YoY vs trailing-3-YoY average, which does NOT reflect what markets
            # actually reacted to on release day. Homepage cards show FF m/m
            # (real consensus surprise) while category_scores.inflation used
            # the stale FRED-YoY score — producing sign mismatches for Gold and
            # other assets. Mirror the JOBS/growth pattern: overwrite `score`
            # and `label` from the FF m/m surprise. Sigma per series calibrated
            # from historical release day |surprise| distributions.
            def _score_from_surp(surp, sigma):
                if surp is None or isinstance(surp, str) or sigma <= 0:
                    return None
                n = surp / sigma
                return (2 if n > 1.25 else 1 if n > 0.4 else
                        -2 if n < -1.25 else -1 if n < -0.4 else 0)
            def _lbl(s):
                return ("Strong Beat" if s == 2 else "Beat" if s == 1 else
                        "Strong Miss" if s == -2 else "Miss" if s == -1 else "In Line")
            # Sigma ≈ typical release-day |surprise| in pp for m/m readings
            _infl_sigma = {"CPI": 0.15, "CORE_CPI": 0.10, "PPI": 0.35,
                           "PCE": 0.10, "CORE_PCE": 0.10}
            for _ikey in ("CPI", "CORE_CPI", "PPI", "PCE", "CORE_PCE"):
                _cc = components.get(_ikey)
                if not _cc:
                    continue
                # Preserve the pre-FF FRED-derived YoY value BEFORE any overwrite happens.
                # For CPI/CORE_CPI/PPI/PCE/CORE_PCE, `_cc['actual']` at this point is the
                # FRED YoY figure (see compute_macro_surprise transform="yoy" above).
                _fred_yoy_actual = None
                _av = _cc.get("actual")
                if isinstance(_av, (int, float)):
                    _fred_yoy_actual = float(_av)
                elif isinstance(_av, str):
                    try:
                        _fred_yoy_actual = float(_av.replace('%','').strip())
                    except Exception:
                        _fred_yoy_actual = None
                _surp_ff = _cc.get("surprise_ff")
                if _surp_ff is None:
                    continue
                # SCORING uses m/m surprise vs forecast — that's what beats/misses are measured on.
                _s_new = _score_from_surp(_surp_ff, _infl_sigma.get(_ikey, 0.15))
                if _s_new is not None:
                    _cc["score"] = _s_new
                    _cc["label"] = _lbl(_s_new)
                # DISPLAY: for CPI/CORE_CPI, prefer y/y — that's what the panel is labelled
                # "CPI YoY" and compared to the 2% Fed target. Using m/m here made
                # 0.10% look like it was "1.90pp below 2% target" when YoY is actually ~3.4%
                # (i.e. 1.4pp ABOVE target). Fix 2026-08-13.
                _af_yoy = _cc.get("actual_ff_yoy") if _ikey in ("CPI", "CORE_CPI") else None
                _ff_yoy = _cc.get("forecast_ff_yoy") if _ikey in ("CPI", "CORE_CPI") else None
                # Fallback to FRED YoY when FF y/y isn't available (e.g. FF doesn't publish
                # Core CPI y/y separately). This is the safety net that keeps the panel honest.
                if _af_yoy is None and _ikey in ("CPI", "CORE_CPI") and _fred_yoy_actual is not None:
                    _af_yoy = _fred_yoy_actual
                    _cc["actual_ff_yoy"] = _fred_yoy_actual  # expose for frontend
                    _cc["yoy_source"] = "FRED"
                elif _af_yoy is not None:
                    _cc["yoy_source"] = "FF"
                _af = _cc.get("actual_ff")
                _ff = _cc.get("forecast_ff")
                # Prefer y/y for CPI/CORE_CPI display; fall back to m/m for others (PPI, PCE, CORE_PCE)
                # or when y/y is missing.
                _af_disp = _af_yoy if _af_yoy is not None else _af
                _ff_disp = _ff_yoy if _ff_yoy is not None else _ff
                if _af_disp is not None:
                    _cc["actual"] = f"{_af_disp:.2f}%"
                    _cc["display"] = _cc["actual"]
                if _ff_disp is not None:
                    _cc["expected"] = _ff_disp
                    _cc["forecast"] = f"{_ff_disp:.2f}%"
                # Keep m/m accessible in dedicated fields so the frontend can show the m/m
                # print alongside the y/y headline without ambiguity.
                if _af is not None:
                    _cc["actual_mom"]   = f"{_af:.2f}%"
                    _cc["actual_mom_ff"] = _af
                if _ff is not None:
                    _cc["forecast_mom"]    = f"{_ff:.2f}%"
                    _cc["forecast_mom_ff"] = _ff

            # Composite heat score at top level (for P2 badge)
            components["_ff_infl_heat"] = {
                "composite": _ff_infl.get("composite_heat"),
                "scores":    _ff_scores,
                "n_events":  _ff_infl.get("n_events_found", 0),
            }
    except Exception as _ff_ie:
        print(f"[FF Inflation] Injection error (non-fatal): {_ff_ie}")

    # ── Overlay real ForexFactory consensus + actual onto GROWTH components ──
    # GDP q/q, ISM Mfg/Svc PMI, Retail Sales m/m, CB Consumer Confidence.
    # FF-first: replaces FRED-proxy expectations with real market consensus and
    # re-scores each component from the genuine surprise (JOBS re-score pattern).
    try:
        _ff_gro = _fetch_ff_growth_surprises()
        if _ff_gro and _ff_gro.get("n_events_found", 0) > 0:
            _fg_latest = _ff_gro.get("latest", {})
            _fg_scores = _ff_gro.get("scores", {})
            _fg_rels   = _ff_gro.get("releases", {})

            def _surp_score(surp, sigma):
                """Normalise a surprise into a -2..+2 score (JOBS pattern thresholds)."""
                if surp is None or isinstance(surp, str):
                    return None
                n = surp / sigma
                return (2 if n > 1.25 else 1 if n > 0.4 else
                        -2 if n < -1.25 else -1 if n < -0.4 else 0)

            def _score_label(s):
                return ("Strong Beat" if s == 2 else "Beat" if s == 1 else
                        "Strong Miss" if s == -2 else "Miss" if s == -1 else "In Line")

            # GDP q/q (annualised) — sigma ~0.4pp for GDP surprises
            _g_lat = _fg_latest.get("gdp")
            if _g_lat and "GDP" in components:
                _g_act, _g_fc = _g_lat.get("actual"), _g_lat.get("forecast")
                components["GDP"]["actual_ff"]   = _g_act
                components["GDP"]["forecast_ff"] = _g_fc
                components["GDP"]["surprise_ff"] = _g_lat.get("surprise")
                components["GDP"]["beat_ff"]     = _g_lat.get("beat")
                components["GDP"]["ff_score"]    = _fg_scores.get("gdp")
                components["GDP"]["ff_releases"] = _fg_rels.get("gdp", [])
                components["GDP"]["ff_event"]    = _g_lat.get("event")
                _s = _surp_score(_g_lat.get("surprise"), 0.4)
                if _s is not None:
                    components["GDP"]["score"] = _s
                    components["GDP"]["label"] = _score_label(_s)
                if _g_act is not None:
                    components["GDP"]["actual"]  = f"{_g_act:.1f}%"
                    components["GDP"]["display"] = f"{_g_act:.1f}%"
                if _g_fc is not None:
                    components["GDP"]["expected"] = _g_fc
                    components["GDP"]["forecast"] = f"{_g_fc:.1f}%"

            # ISM PMIs — surprise sigma ~1.2pts; blend in absolute level context
            # (a PMI beat below 47.5 is still contraction; a miss above 52.5 is
            #  still expansion) so the category score reflects reality.
            for _pmi_key, _comp_key, _title in (("ism_mfg", "MFG_PMI", "ISM Manufacturing PMI"),
                                                ("ism_svc", "SVC_PMI", "ISM Services PMI")):
                _p_lat = _fg_latest.get(_pmi_key)
                if not _p_lat or _comp_key not in components:
                    continue
                _p_act, _p_fc = _p_lat.get("actual"), _p_lat.get("forecast")
                c = components[_comp_key]
                c["actual_ff"]   = _p_act
                c["forecast_ff"] = _p_fc
                c["surprise_ff"] = _p_lat.get("surprise")
                c["beat_ff"]     = _p_lat.get("beat")
                c["ff_score"]    = _fg_scores.get(_pmi_key)
                c["ff_releases"] = _fg_rels.get(_pmi_key, [])
                c["title"]       = _title
                _s = _surp_score(_p_lat.get("surprise"), 1.2)
                if _s is not None and _p_act is not None:
                    _lvl = 1 if _p_act > 52.5 else -1 if _p_act < 47.5 else 0
                    _s = max(-2, min(2, _s + _lvl))
                    c["score"] = _s
                    c["label"] = _score_label(_s) if _lvl == 0 else (
                        "Expansion" if _lvl == 1 else "Contraction")
                if _p_act is not None:
                    c["actual"]  = f"{_p_act:.1f}"
                    c["display"] = f"{_p_act:.1f}"
                if _p_fc is not None:
                    c["expected"] = _p_fc
                    c["forecast"] = f"{_p_fc:.1f}"

            # Retail Sales m/m — sigma ~0.3pp; core retail rides as extra fields
            _r_lat = _fg_latest.get("retail")
            if _r_lat and "RETAIL" in components:
                _r_act, _r_fc = _r_lat.get("actual"), _r_lat.get("forecast")
                c = components["RETAIL"]
                c["actual_ff"]   = _r_act
                c["forecast_ff"] = _r_fc
                c["surprise_ff"] = _r_lat.get("surprise")
                c["beat_ff"]     = _r_lat.get("beat")
                c["ff_score"]    = _fg_scores.get("retail")
                c["ff_releases"] = _fg_rels.get("retail", [])
                c["title"]       = "Retail Sales m/m"
                _s = _surp_score(_r_lat.get("surprise"), 0.3)
                if _s is not None:
                    c["score"] = _s
                    c["label"] = _score_label(_s)
                if _r_act is not None:
                    c["actual"]  = f"{_r_act:+.1f}%"
                    c["display"] = f"{_r_act:+.1f}%"
                if _r_fc is not None:
                    c["expected"] = _r_fc
                    c["forecast"] = f"{_r_fc:+.1f}%"
                _cr_lat = _fg_latest.get("core_retail")
                if _cr_lat:
                    c["core_actual_ff"]   = _cr_lat.get("actual")
                    c["core_forecast_ff"] = _cr_lat.get("forecast")
                    c["core_surprise_ff"] = _cr_lat.get("surprise")
                    c["core_beat_ff"]     = _cr_lat.get("beat")

            # CB Consumer Confidence — sigma ~3.0 index pts; overlays UoM fallback
            _cf_lat = _fg_latest.get("conf_cb")
            if _cf_lat:
                _c_act, _c_fc = _cf_lat.get("actual"), _cf_lat.get("forecast")
                _s = _surp_score(_cf_lat.get("surprise"), 3.0)
                components["CONF"] = {
                    **components.get("CONF", {}),
                    "actual":      f"{_c_act:.1f}" if _c_act is not None else "—",
                    "forecast":    f"{_c_fc:.1f}" if _c_fc is not None else "—",
                    "expected":    _c_fc,
                    "actual_ff":   _c_act,
                    "forecast_ff": _c_fc,
                    "surprise_ff": _cf_lat.get("surprise"),
                    "beat_ff":     _cf_lat.get("beat"),
                    "ff_score":    _fg_scores.get("conf_cb"),
                    "ff_releases": _fg_rels.get("conf_cb", []),
                    "score":       _s if _s is not None else components.get("CONF", {}).get("score", 0),
                    "label":       _score_label(_s) if _s is not None else components.get("CONF", {}).get("label", ""),
                    "title":       "CB Consumer Confidence",
                    "category":    "growth",
                    "display":     f"{_c_act:.1f}" if _c_act is not None else "—",
                }

            # Composite growth momentum at top level (for pane badge)
            components["_ff_growth"] = {
                "composite": _ff_gro.get("composite_growth"),
                "scores":    _fg_scores,
                "n_events":  _ff_gro.get("n_events_found", 0),
            }
            print(f"[FF Growth] Injected into components: GDP={'actual_ff' in components.get('GDP',{})}, "
                  f"MFG={'actual_ff' in components.get('MFG_PMI',{})}, SVC={'actual_ff' in components.get('SVC_PMI',{})}, "
                  f"RETAIL={'actual_ff' in components.get('RETAIL',{})}, CONF={'actual_ff' in components.get('CONF',{})}")
    except Exception as _ff_ge:
        print(f"[FF Growth] Injection error (non-fatal): {_ff_ge}")

    nfp_data = fetch_fred_series("NFP", 12) or fetch_fred_series("NFP", 12)
    if nfp_data:
        r = compute_macro_surprise(nfp_data, higher_is_good=True, transform="mom", scale=80)
        if r["actual"] is not None:
            r["actual"]   = round(r["actual"])
            r["expected"] = round(r["expected"])
            r["surprise"] = round(r["surprise"])
            r["display"]  = f"{r['actual']:+.0f}K"
        components["JOBS"] = {**r, "title": "Non-Farm Payrolls", "category": "jobs",
                              "display": r.get("display", "—")}

    unemp_data = fetch_fred_series("UNEMP", 12) or fetch_fred_series("UNEMP", 12)
    if unemp_data:
        r = compute_macro_surprise(unemp_data, higher_is_good=False, transform="level", scale=0.15)
        components["UNEMP"] = {**r, "title": "Unemployment Rate", "category": "jobs",
                                "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    claims_data = fetch_fred_series("CLAIMS", 26) or fetch_fred_series("CLAIMS", 26)  # 26 weeks = 6m for stable baseline
    if claims_data:
        r = compute_macro_surprise(claims_data, higher_is_good=False, transform="level", scale=15000)
        components["CLAIMS"] = {**r, "title": "Initial Claims", "category": "jobs",
                                 "display": f"{int(r['actual']):,}" if r['actual'] is not None else "—"}

    # ADP (FRED fallback) — ADP National Employment Report level (persons). Compute the
    # m/m change in thousands so it reads like NFP. FF data overlays this when available.
    adp_data = fetch_fred_series("ADP", 14) or fetch_fred_series("ADP", 14)
    if adp_data and len(adp_data) >= 5:
        _chg = [(adp_data[i]["value"] - adp_data[i-1]["value"]) / 1000.0 for i in range(1, len(adp_data))]
        if len(_chg) >= 4:
            _a = round(_chg[-1]); _e = round(sum(_chg[-4:-1]) / 3); _s = _a - _e
            components["ADP"] = {"actual": _a, "expected": _e, "surprise": _s,
                                  "higher_is_good": True, "category": "jobs",
                                  "score": 1 if _s > 25 else -1 if _s < -25 else 0,
                                  "title": "ADP Non-Farm Employment",
                                  "display": f"{_a:+d}K"}

    # WAGES (FRED fallback) — Avg Hourly Earnings level ($); compute m/m % change.
    wages_data = fetch_fred_series("WAGES", 14) or fetch_fred_series("WAGES", 14)
    if wages_data and len(wages_data) >= 5:
        _pct = [(wages_data[i]["value"] / wages_data[i-1]["value"] - 1.0) * 100
                for i in range(1, len(wages_data)) if wages_data[i-1]["value"]]
        if len(_pct) >= 4:
            _a = round(_pct[-1], 2); _e = round(sum(_pct[-4:-1]) / 3, 2); _s = round(_a - _e, 2)
            components["WAGES"] = {"actual": _a, "expected": _e, "surprise": _s,
                                    "higher_is_good": True, "category": "jobs",
                                    "score": 1 if _s > 0.05 else -1 if _s < -0.05 else 0,
                                    "title": "Avg Hourly Earnings m/m",
                                    "display": f"{_a:.2f}%"}

    # ── Inject ForexFactory labour data onto JOBS/UNEMP/CLAIMS/WAGES/ADP ─────────
    # FF provides real market consensus vs actual for NFP, ADP, Unemployment,
    # Claims, Wages m/m, and JOLTS. These replace FRED rolling-avg expectations
    # with real-time market forecasts, making surprise signals genuinely useful.
    try:
        _ff_lab = _fetch_ff_labour_surprises()
        _ff_lab_latest  = _ff_lab.get("latest", {})
        _ff_lab_releases = _ff_lab.get("releases", {})
        _ff_lab_scores  = _ff_lab.get("scores", {})

        # NFP → JOBS
        _nfp_lat = _ff_lab_latest.get("nfp")
        if _nfp_lat and "JOBS" in components:
            _nfp_act  = _nfp_lat.get("actual")    # K
            _nfp_fc   = _nfp_lat.get("forecast")  # K
            _nfp_surp = _nfp_lat.get("surprise")  # K
            _nfp_beat = _nfp_lat.get("beat")      # bool
            components["JOBS"]["actual_ff"]   = _nfp_act
            components["JOBS"]["forecast_ff"] = _nfp_fc
            components["JOBS"]["surprise_ff"] = _nfp_surp
            components["JOBS"]["beat_ff"]     = _nfp_beat
            components["JOBS"]["ff_releases"] = _ff_lab_releases.get("nfp", [])
            components["JOBS"]["ff_score"]    = _ff_lab_scores.get("nfp")
            # Re-score JOBS using real FF consensus surprise (K), not FRED rolling avg
            # Scale: 80K = 1 sigma for NFP surprises historically
            if _nfp_surp is not None:
                _nfp_norm = (_nfp_surp if not isinstance(_nfp_surp, str) else 0) / 60.0
                _nfp_score_ff = (2 if _nfp_norm > 1.25 else
                                 1 if _nfp_norm > 0.4 else
                                -2 if _nfp_norm < -1.25 else
                                -1 if _nfp_norm < -0.4 else 0)
                components["JOBS"]["score"]    = _nfp_score_ff
                components["JOBS"]["label"]    = (
                    "Strong Beat" if _nfp_score_ff == 2 else
                    "Beat"        if _nfp_score_ff == 1 else
                    "Strong Miss" if _nfp_score_ff == -2 else
                    "Miss"        if _nfp_score_ff == -1 else "In Line"
                )
            # Override display fields with FF consensus values
            if _nfp_act is not None:
                components["JOBS"]["actual"]   = f"{round(_nfp_act):+d}K" if isinstance(_nfp_act, (int, float)) else str(_nfp_act)
                components["JOBS"]["display"]  = components["JOBS"]["actual"]
            if _nfp_fc is not None:
                components["JOBS"]["expected"] = round(_nfp_fc) if isinstance(_nfp_fc, (int, float)) else _nfp_fc
                components["JOBS"]["forecast"] = f"{round(_nfp_fc):+d}K" if isinstance(_nfp_fc, (int, float)) else str(_nfp_fc)
            if _nfp_surp is not None:
                components["JOBS"]["surprise"] = round(_nfp_surp) if isinstance(_nfp_surp, (int, float)) else _nfp_surp

        # Unemployment Rate → UNEMP
        _un_lat = _ff_lab_latest.get("unrate")
        if _un_lat and "UNEMP" in components:
            components["UNEMP"]["actual_ff"]   = _un_lat.get("actual")   # %
            components["UNEMP"]["forecast_ff"] = _un_lat.get("forecast") # %
            components["UNEMP"]["surprise_ff"] = _un_lat.get("surprise") # pp
            components["UNEMP"]["beat_ff"]     = _un_lat.get("beat")     # beat = lower than expected
            components["UNEMP"]["ff_releases"] = _ff_lab_releases.get("unrate", [])
            components["UNEMP"]["ff_score"]    = _ff_lab_scores.get("unrate")

        # Initial Claims → CLAIMS
        _cl_lat = _ff_lab_latest.get("claims")
        if _cl_lat and "CLAIMS" in components:
            components["CLAIMS"]["actual_ff"]   = _cl_lat.get("actual")   # K
            components["CLAIMS"]["forecast_ff"] = _cl_lat.get("forecast") # K
            components["CLAIMS"]["surprise_ff"] = _cl_lat.get("surprise") # K
            components["CLAIMS"]["beat_ff"]     = _cl_lat.get("beat")     # beat = lower than expected
            components["CLAIMS"]["ff_releases"] = _ff_lab_releases.get("claims", [])
            components["CLAIMS"]["ff_score"]    = _ff_lab_scores.get("claims")

        # Average Hourly Earnings m/m → WAGES (new component)
        _wg_lat = _ff_lab_latest.get("wages")
        if _wg_lat:
            _wg_surp = _wg_lat.get("surprise", 0)
            _wg_beat = _wg_lat.get("beat", False)
            # Score: beat=hot (wages rising above expectation) = potentially inflationary
            # From a market perspective: wage beat = labour market tighter than expected = positive USD
            _wg_score_raw = _ff_lab_scores.get("wages")
            components["WAGES"] = {
                **components.get("WAGES", {}),   # keep FRED fallback fields; FF overlays below
                "actual":      f"{_wg_lat.get('actual')}%" if _wg_lat.get('actual') is not None else "—",
                "forecast":    f"{_wg_lat.get('forecast')}%" if _wg_lat.get('forecast') is not None else "—",
                "actual_ff":   _wg_lat.get("actual"),
                "forecast_ff": _wg_lat.get("forecast"),
                "surprise_ff": _wg_lat.get("surprise"),
                "beat_ff":     _wg_beat,
                "ff_releases": _ff_lab_releases.get("wages", []),
                "ff_score":    _wg_score_raw,
                "score":       1 if _wg_beat else -1 if _wg_beat is False else 0,
                "title":       "Avg Hourly Earnings m/m",
                "category":    "jobs",
                "display":     f"{_wg_lat.get('actual')}%" if _wg_lat.get('actual') is not None else "—",
            }

        # ADP Non-Farm → ADP (new component for extra context)
        _adp_lat = _ff_lab_latest.get("adp")
        if _adp_lat:
            _adp_beat = _adp_lat.get("beat", False)
            components["ADP"] = {
                **components.get("ADP", {}),   # keep FRED fallback fields; FF overlays below
                "actual":      f"{_adp_lat.get('actual')}K" if _adp_lat.get('actual') is not None else "—",
                "forecast":    f"{_adp_lat.get('forecast')}K" if _adp_lat.get('forecast') is not None else "—",
                "actual_ff":   _adp_lat.get("actual"),
                "forecast_ff": _adp_lat.get("forecast"),
                "surprise_ff": _adp_lat.get("surprise"),
                "beat_ff":     _adp_beat,
                "ff_releases": _ff_lab_releases.get("adp", []),
                "ff_score":    _ff_lab_scores.get("adp"),
                "score":       1 if _adp_beat else -1 if _adp_beat is False else 0,
                "title":       "ADP Non-Farm Employment",
                "category":    "jobs",
                "display":     f"{_adp_lat.get('actual')}K" if _adp_lat.get('actual') is not None else "—",
            }

        print(f"[FF Labour] Injected into components: NFP={'JOBS' in components and 'actual_ff' in components.get('JOBS',{})}, "
              f"UNEMP={'actual_ff' in components.get('UNEMP',{})}, CLAIMS={'actual_ff' in components.get('CLAIMS',{})}, "
              f"WAGES={'WAGES' in components}, ADP={'ADP' in components}")
    except Exception as _ff_le:
        print(f"[FF Labour] Injection error (non-fatal): {_ff_le}")

    # ── RATES ─────────────────────────────────────────────────────────────────
    dgs2_data = fetch_fred_series("DGS2", 30)
    if dgs2_data:
        r = compute_macro_surprise(dgs2_data, higher_is_good=True, transform="level", scale=0.2)
        components["DGS2"] = {**r, "title": "2Y Treasury Yield", "category": "rates",
                               "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    yldcrv_data = fetch_fred_series("YLDCRV", 30)
    if yldcrv_data:
        r = compute_macro_surprise(yldcrv_data, higher_is_good=True, transform="level", scale=0.15)
        components["YLDCRV"] = {**r, "title": "Yield Curve (10Y-2Y)", "category": "rates",
                                  "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    # ── Category aggregation ──────────────────────────────────────────────────
    growth_scores    = [components[k]["score"] for k in ["GDP", "MFG_PMI", "SVC_PMI", "RETAIL", "CONF"] if k in components]
    inflation_scores = [components[k]["score"] for k in ["CPI", "CORE_CPI", "PPI", "PCE", "CORE_PCE"] if k in components]
    jobs_scores      = [components[k]["score"] for k in ["JOBS", "UNEMP", "CLAIMS", "ADP", "WAGES"] if k in components]
    rates_scores     = [components[k]["score"] for k in ["DGS2", "YLDCRV"] if k in components]

    def avg_score(lst): return sum(lst) / len(lst) if lst else 0

    growth_avg    = avg_score(growth_scores)
    inflation_avg = avg_score(inflation_scores)
    jobs_avg      = avg_score(jobs_scores)
    rates_avg     = avg_score(rates_scores)

    equity_base = growth_avg * 0.35 + jobs_avg * 0.35 + inflation_avg * (-0.15) + rates_avg * 0.15
    equity_overall = (2 if equity_base > 0.8 else 1 if equity_base > 0.25
                      else -2 if equity_base < -0.8 else -1 if equity_base < -0.25 else 0)

    # ── Normalise component fields for frontend compatibility ────────────────
    # Frontend reads c.actual (formatted string) and c.forecast (formatted expected)
    for key, comp in components.items():
        disp = comp.get("display", "")
        exp_raw = comp.get("expected")
        # Use display as the formatted actual string
        if disp and disp != "—":
            comp["actual"] = disp
        # Add forecast alias: format expected the same way as actual
        if exp_raw is not None:
            # Determine format from display suffix
            d = disp or ""
            if d.endswith("%"):
                comp["forecast"] = f"{exp_raw:.2f}%"
            elif d.endswith("K"):
                comp["forecast"] = f"{exp_raw:+.0f}K"
            elif d.endswith("M"):
                comp["forecast"] = f"{exp_raw:+.0f}M"
            elif "," in d:  # claims: 214,000
                try:
                    comp["forecast"] = f"{int(exp_raw):,}"
                except Exception:
                    comp["forecast"] = str(round(exp_raw, 2))
            else:
                comp["forecast"] = str(round(exp_raw, 4))
        else:
            comp["forecast"] = "—"

    result = {
        "components": components,
        "category_scores": {
            "growth":    round(growth_avg, 2),
            "inflation": round(inflation_avg, 2),
            "jobs":      round(jobs_avg, 2),
            "rates":     round(rates_avg, 2),
        },
        "equity_overall": equity_overall,
    }
    US_MACRO_CACHE["data"] = result
    US_MACRO_CACHE["time"] = now
    return result
def compute_eia_inventory_signal() -> dict:
    """
    Derive EIA weekly crude inventory surprise proxy from WTI Wednesday price reactions.
    Returns a score in [-2, +2] range suitable for macro sub-factor blending.
    """
    now = time.time()
    try:
        import yfinance as _yf
        cl_raw = _yf.Ticker("CL=F")
        cl = cl_raw.history(period="3mo", interval="1d", auto_adjust=True)
        if cl.empty or len(cl) < 10:
            return {"score": 0, "label": "Neutral EIA signal"}
        cl_df = cl.copy()
        cl_df["daily_ret"] = cl_df["Close"].pct_change()
        # Wednesdays (weekday=2)
        wed = cl_df[cl_df.index.weekday == 2]["daily_ret"].dropna()
        if len(wed) < 3:
            return {"score": 0, "label": "Neutral EIA signal"}
        # Use thresholds from pyc: p20=-0.0181, p80=0.0179
        p20_thresh = -0.0181
        p80_thresh = 0.0179
        last_ret  = wed.iloc[-1]
        prev_ret  = wed.iloc[-2] if len(wed) >= 2 else 0

        last_drop   = last_ret < p20_thresh
        last_rally  = last_ret > p80_thresh
        prev_drop   = prev_ret < p20_thresh
        prev_rally  = prev_ret > p80_thresh

        consecutive_drops   = last_drop and prev_drop
        consecutive_rallies = last_rally and prev_rally

        if consecutive_drops:
            raw_score = 1.5; label = "Consecutive bearish EIA days — contrarian bullish"
        elif last_drop:
            raw_score = 0.8; label = "Bearish EIA day — lean bullish"
        elif consecutive_rallies:
            raw_score = -1.2; label = "Consecutive bullish EIA days — continuation bearish"
        elif last_rally:
            raw_score = -0.5; label = "Bullish EIA day — lean bearish (continuation)"
        else:
            raw_score = 0.0; label = "Neutral EIA signal"

        return {"score": raw_score, "label": label}
    except Exception as e:
        return {"score": 0, "label": f"EIA signal error: {e}"}


def compute_ng_storage_signal() -> dict:
    """
    Derive EIA Natural Gas Storage (Thursday) surprise proxy from NG Thursday price reactions.
    """
    now = time.time()
    try:
        import yfinance as _yf
        ng_raw = _yf.Ticker("NG=F")
        ng = ng_raw.history(period="3mo", interval="1d", auto_adjust=True)
        if ng.empty or len(ng) < 10:
            return {"score": 0, "label": "Neutral NG storage signal"}
        ng_df = ng.copy()
        ng_df["daily_ret"] = ng_df["Close"].pct_change()
        # Thursdays (weekday=3)
        thu = ng_df[ng_df.index.weekday == 3]["daily_ret"].dropna()
        if len(thu) < 3:
            return {"score": 0, "label": "Neutral NG storage signal"}
        p20_thresh = -0.0181
        p80_thresh = 0.0179
        last_ret  = thu.iloc[-1]
        prev_ret  = thu.iloc[-2] if len(thu) >= 2 else 0

        last_drop   = last_ret < p20_thresh
        last_rally  = last_ret > p80_thresh
        prev_drop   = prev_ret < p20_thresh
        prev_rally  = prev_ret > p80_thresh

        consecutive_drops   = last_drop and prev_drop
        consecutive_rallies = last_rally and prev_rally

        if consecutive_drops:
            raw_score = 1.5; label = "Consecutive bearish storage days — contrarian bullish NG"
        elif last_drop:
            raw_score = 0.8; label = "Bearish storage day — lean bullish NG"
        elif consecutive_rallies:
            raw_score = -1.2; label = "Consecutive bullish storage days — lean bearish NG"
        elif last_rally:
            raw_score = -0.5; label = "Bullish storage day — lean bearish NG"
        else:
            raw_score = 0.0; label = "Neutral NG storage signal"

        return {"score": raw_score, "label": label}
    except Exception as e:
        return {"score": 0, "label": f"NG signal error: {e}"}


def get_macro_score_for_market(market_id: str, macro: dict, ff_macro: dict = None) -> dict:
    """
    EdgeFinder-style asset-specific macro scoring.

    Built from deep research into how each macro indicator affects each market:
    - Correct per-indicator polarities (CPI is bearish equities but bullish gold)
    - Per-asset custom weighting (bonds are 100% rates/inflation, equities care more about jobs/growth)
    - FF-based economy scores used for FX cross pairs
    Returns {score: 0-10, label, reason, ...}
    """
    if ff_macro is None:
        ff_macro = {}

    comps       = macro.get("components", {})
    cat_scores  = macro.get("category_scores", {})

    growth_s    = cat_scores.get("growth",    0)
    jobs_s      = cat_scores.get("jobs",      0)
    inflation_s = cat_scores.get("inflation", 0)
    rates_s     = cat_scores.get("rates",     0)

    # Individual series for fine-grained control
    dgs2_s   = comps.get("DGS2",   {}).get("score", 0)
    cpi_s    = comps.get("CPI",    {}).get("score", 0)
    pce_s    = comps.get("PCE",    {}).get("score", 0)
    gdp_s    = comps.get("GDP",    {}).get("score", 0)
    jobs_d   = comps.get("JOBS",   {}).get("score", jobs_s)
    infl_avg = (cpi_s + pce_s) / 2 if (cpi_s != 0 or pce_s != 0) else inflation_s

    pmi_avg    = (cat_scores.get("MFG_PMI", 0) + cat_scores.get("SVC_PMI", 0)) / 2
    growth_s2  = (growth_s + pmi_avg) / 2 if pmi_avg else growth_s

    def score_to_010(raw, scale=2.0):
        return round(max(0.0, min(10.0, (raw / scale) * 2.5 + 5.0)), 1)

    m = market_id.upper()

    # ── Equity Indices ─────────────────────────────────────────────────────
    if m in ("ES", "NQ", "YM", "RTY", "RUT"):
        raw = growth_s2 * 0.40 + jobs_s * 0.35 - infl_avg * 0.15 - dgs2_s * 0.10
        reason = f"Growth: {growth_s:+.1f}, Jobs: {jobs_s:+.1f}, CPI: {-infl_avg:+.1f}"

    # ── FTSE 100 (Z) — international equity, UK macro blend ─────────────────
    elif m == "Z":
        # FTSE 100: global risk/growth dominant (70% of revenues are non-UK).
        # US macro used as global proxy + UK-specific GBP macro where available.
        # Higher UK CPI = BoE hawkish = headwind for FTSE (EPS translation drag).
        _uk_d = ff_macro.get("GBP", {})
        uk_infl  = _uk_d.get("cats", {}).get("inflation", 0) if _uk_d else 0
        uk_jobs  = _uk_d.get("cats", {}).get("jobs", 0)      if _uk_d else 0
        # Global proxy score (US macro 60%) + UK-specific (40%)
        us_raw  = growth_s2 * 0.40 + jobs_s * 0.35 - infl_avg * 0.15 - dgs2_s * 0.10
        uk_raw  = uk_jobs * 0.35 - uk_infl * 0.40 - dgs2_s * 0.25  # BoE hawkish = headwind
        raw = us_raw * 0.60 + uk_raw * 0.40
        reason = f"Growth: {growth_s:+.1f}, UK macro blend"

    # ── Dollar Index ────────────────────────────────────────────────────────
    elif m == "DX":
        raw = jobs_s * 0.30 + growth_s * 0.25 + infl_avg * 0.30 + dgs2_s * 0.15
        reason = f"Jobs: {jobs_s:+.1f}, Growth: {growth_s:+.1f}, CPI: {infl_avg:+.1f}"

    # ── FX Pairs (base currency vs USD) ────────────────────────────────────
    elif m in ("6E", "6B", "6A", "6C", "6N", "6S", "6M"):
        # Determine base currency
        ccy_map = {"6E": "EUR", "6B": "GBP", "6A": "AUD", "6C": "CAD",
                   "6N": "NZD", "6S": "CHF", "6M": "MXN"}
        base_ccy = ccy_map.get(m, "EUR")
        base_ff  = ff_macro.get(base_ccy, {})
        usd_ff   = ff_macro.get("USD", {})
        base_score_ff  = base_ff.get("score", 5.0)  # already 0-10
        usd_score_ff   = usd_ff.get("score",  5.0)
        # Differential: positive = base ccy stronger than USD = bullish pair
        diff = base_score_ff - usd_score_ff  # range -10..+10
        raw  = diff / 4.0  # normalise to ~-2..+2
        reason = f"{base_ccy} macro: {base_score_ff:.1f}/10 vs USD: {usd_score_ff:.1f}/10"
        scr = score_to_010(raw)
        return {
            "score": scr, "label": _macro_label(scr),
            "reason": reason, "growth_s": growth_s, "inflation_s": inflation_s,
            "jobs_s": jobs_s, "rates_s": rates_s,
            "base_ff_score": base_score_ff, "usd_ff_score": usd_score_ff,
            "fx_detail": {
                "foreign": {
                    "currency": base_ccy,
                    "score": base_score_ff - 5.0,  # centre on 0 for differential bar
                    "cats": base_ff.get("cats", {}),
                    "cat_details": base_ff.get("cat_details", {}),
                },
                "usd": {
                    "currency": "USD",
                    "score": usd_score_ff - 5.0,  # centre on 0
                    "cats": usd_ff.get("cats", {}),
                    "cat_details": usd_ff.get("cat_details", {}),
                },
            },
        }

    # ── Japanese Yen ────────────────────────────────────────────────────────
    elif m == "6J":
        jpy_ff  = ff_macro.get("JPY", {})
        usd_ff  = ff_macro.get("USD", {})
        jpy_sc  = jpy_ff.get("score", 5.0)
        usd_sc  = usd_ff.get("score", 5.0)
        # 6J = JPY/USD futures: bullish when JPY strengthens (weak USD or strong JPY)
        diff = jpy_sc - usd_sc
        raw  = diff / 4.0
        reason = f"JPY macro: {jpy_sc:.1f}/10 vs USD: {usd_sc:.1f}/10"
        scr = score_to_010(raw)
        return {
            "score": scr, "label": _macro_label(scr),
            "reason": reason, "growth_s": growth_s, "inflation_s": inflation_s,
            "jobs_s": jobs_s, "rates_s": rates_s,
            "fx_detail": {
                "foreign": {
                    "currency": "JPY",
                    "score": jpy_sc - 5.0,
                    "cats": jpy_ff.get("cats", {}),
                    "cat_details": jpy_ff.get("cat_details", {}),
                },
                "usd": {
                    "currency": "USD",
                    "score": usd_sc - 5.0,
                    "cats": usd_ff.get("cats", {}),
                    "cat_details": usd_ff.get("cat_details", {}),
                },
            },
        }

    # ── Gold ────────────────────────────────────────────────────────────────
    # Hot CPI → Fed hikes → nominal yields rise faster than breakevens → real yields up → bearish gold
    # (inflation-hedge narrative is secondary; real-yield mechanism dominates in the short term)
    elif m == "GC":
        raw = -infl_avg * 0.55 - dgs2_s * 0.30 - growth_s * 0.08 - jobs_s * 0.07
        reason = f"CPI/PCE: {-infl_avg:+.1f}, 2Y yield: {-dgs2_s:+.1f}"

    # ── Silver ─────────────────────────────────────────────────────────────
    # Precious leg (41%): real-yield mechanism same as gold — hot CPI bearish via higher real yields
    elif m == "SI":
        raw = -infl_avg * 0.35 + growth_s2 * 0.22 - dgs2_s * 0.25 - jobs_s * 0.18
        reason = f"CPI: {-infl_avg:+.1f}, Growth: {growth_s2:+.1f}"

    # ── Bonds (ZB, ZN, ZF, ZT) ────────────────────────────────────────────
    elif m in ("ZB", "ZN", "ZF", "ZT", "GBL", "R"):
        raw = -(infl_avg * 0.35) - (jobs_s * 0.30) - (growth_s * 0.20) - (dgs2_s * 0.15)
        reason = f"Infl: {-infl_avg:+.1f}, Jobs: {-jobs_s:+.1f} (inverted)"
        # UK bonds (R = Long Gilt): blend in UK macro if available
        if m == "R":
            _uk_data = ff_macro.get("GBP", {})
            if _uk_data.get("score") is not None:
                uk_cpi_raw   = _uk_data.get("cats", {}).get("inflation", 0)
                uk_unemp_raw = _uk_data.get("cats", {}).get("jobs", 0)
                # UK Gilt score: blends UK CPI/jobs inverse with US rate/inflation backdrop
                raw = raw * 0.55 + (-(uk_cpi_raw * 0.45) - (uk_unemp_raw * 0.1)) * 0.45
            reason = f"UK/US Macro blend (Gilt inverse)"

    # ── Oil family (CL, B, GO, HO, RB) ─────────────────────────────────────
    elif m in ("CL", "B", "GO", "HO", "RB"):
        eia = compute_eia_inventory_signal()
        eia_s = eia.get("score", 0)
        raw = growth_s2 * 0.35 + infl_avg * 0.15 + jobs_s * 0.15 - dgs2_s * 0.15 + eia_s * 0.20
        reason = f"Growth: {growth_s2:+.1f}, EIA: {eia_s:+.1f}"

    # ── Natural Gas (NG) ───────────────────────────────────────────────────
    elif m == "NG":
        ng_sig = compute_ng_storage_signal()
        ng_s   = ng_sig.get("score", 0)
        raw = growth_s2 * 0.20 + infl_avg * 0.10 + ng_s * 0.50 - dgs2_s * 0.20
        reason = f"Storage: {ng_s:+.1f}, Growth: {growth_s2:+.1f}"

    # ── Copper (HG) ────────────────────────────────────────────────────────
    # Hot CPI → Fed hikes → USD stronger → copper (USD-priced) headwind.
    # Growth is the dominant driver (50%). Both inflation and rates are bearish via USD channel.
    elif m == "HG":
        raw = growth_s2 * 0.50 + jobs_s * 0.25 - infl_avg * 0.10 - dgs2_s * 0.15
        reason = f"Growth: {growth_s2:+.1f}, CPI: {-infl_avg:+.1f}, 2Y: {-dgs2_s:+.1f}"

    # ── Soft Commodities / Agri ───────────────────────────────────────────
    elif m in ("ZC", "ZS", "ZW", "KC", "SB", "CT", "CC", "RC"):
        raw = (infl_avg * 0.30 + growth_s * 0.20 - dgs2_s * 0.20 + jobs_s * 0.10) / 0.80
        reason = f"Infl: {infl_avg:+.1f}, Growth: {growth_s:+.1f}"

    # ── Livestock ─────────────────────────────────────────────────────────
    # Hot CPI → consumer squeeze → reduced protein demand (demand destruction > supply squeeze
    # for near-term contracts). USD strength also hurts export competitiveness.
    # Net: modest bearish for hot CPI. Growth + jobs are the primary bullish drivers.
    elif m in ("LE", "HE", "GF"):
        raw = growth_s * 0.35 + jobs_s * 0.30 - infl_avg * 0.10 - dgs2_s * 0.25
        reason = f"Growth: {growth_s:+.1f}, Jobs: {jobs_s:+.1f}, CPI: {-infl_avg:+.1f}"

    # ── Platinum, Palladium ───────────────────────────────────────────────
    # Hot CPI → USD hawkish → commodity headwind (USD-priced industrial metals).
    # PL/PA have minor precious metal hedge component but industrial demand dominates.
    elif m in ("PL", "PA"):
        raw = growth_s2 * 0.40 - infl_avg * 0.15 + jobs_s * 0.25 - dgs2_s * 0.20
        reason = f"Growth: {growth_s2:+.1f}, Jobs: {jobs_s:+.1f}, CPI: {-infl_avg:+.1f}, 2Y: {-dgs2_s:+.1f}"

    # ── Crypto ────────────────────────────────────────────────────────────
    # Risk-on assets: growth + jobs bullish.
    # Hot CPI → Fed hikes → risk-off → crypto headwind (bearish via rate path).
    # Rising 2Y → tighter financial conditions → risk asset headwind.
    elif m in ("BTC", "ETH"):
        raw = (growth_s * 0.30 + jobs_s * 0.20 - dgs2_s * 0.15 - infl_avg * 0.10) / 0.75
        reason = f"Growth: {growth_s:+.1f}, Rates: {-dgs2_s:+.1f}, CPI: {-infl_avg:+.1f}"

    # ── FX Cross Pairs — use ff_macro leg differential ──────────────────────
    elif m in ("EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
               "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
               "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
               "AUDCAD", "AUDNZD", "AUDCHF", "NZDCAD"):
        # Map each cross to its base and quote currency
        _cross_map = {
            "EURJPY": ("EUR", "JPY"), "GBPJPY": ("GBP", "JPY"),
            "AUDJPY": ("AUD", "JPY"), "NZDJPY": ("NZD", "JPY"),
            "CADJPY": ("CAD", "JPY"), "CHFJPY": ("CHF", "JPY"),
            "EURGBP": ("EUR", "GBP"), "EURAUD": ("EUR", "AUD"),
            "EURCAD": ("EUR", "CAD"), "EURNZD": ("EUR", "NZD"),
            "EURCHF": ("EUR", "CHF"), "GBPAUD": ("GBP", "AUD"),
            "GBPCAD": ("GBP", "CAD"), "GBPNZD": ("GBP", "NZD"),
            "GBPCHF": ("GBP", "CHF"), "AUDCAD": ("AUD", "CAD"),
            "AUDNZD": ("AUD", "NZD"), "AUDCHF": ("AUD", "CHF"),
            "NZDCAD": ("NZD", "CAD"),
        }
        base_ccy, quote_ccy = _cross_map.get(m, ("EUR", "USD"))
        base_sc  = (ff_macro or {}).get(base_ccy,  {}).get("score", 5.0)
        quote_sc = (ff_macro or {}).get(quote_ccy, {}).get("score", 5.0)
        diff = base_sc - quote_sc   # positive = base stronger = pair bullish
        raw  = diff / 4.0           # normalise to ~-2..+2
        reason = f"{base_ccy}: {base_sc:.1f}/10 vs {quote_ccy}: {quote_sc:.1f}/10"
        scr = score_to_010(raw)
        return {
            "score": scr, "label": _macro_label(scr),
            "reason": reason, "growth_s": growth_s, "inflation_s": inflation_s,
            "jobs_s": jobs_s, "rates_s": rates_s,
            "base_ff_score": base_sc, "quote_ff_score": quote_sc,
            "fx_detail": {
                "foreign": {
                    "currency": base_ccy,
                    "score": base_sc - 5.0,
                    "cats": (ff_macro or {}).get(base_ccy,  {}).get("cats", {}),
                    "cat_details": (ff_macro or {}).get(base_ccy,  {}).get("cat_details", {}),
                },
                "usd": {
                    "currency": quote_ccy,
                    "score": quote_sc - 5.0,
                    "cats": (ff_macro or {}).get(quote_ccy, {}).get("cats", {}),
                    "cat_details": (ff_macro or {}).get(quote_ccy, {}).get("cat_details", {}),
                },
            },
        }

    else:
        raw = growth_s * 0.40 + jobs_s * 0.35 - infl_avg * 0.15 + dgs2_s * 0.10
        reason = f"Default: Growth {growth_s:+.1f}, Jobs {jobs_s:+.1f}"

    return {
        "score":      score_to_010(raw),
        "label":      _macro_label(score_to_010(raw)),
        "reason":     reason,
        "growth_s":   growth_s,
        "inflation_s": inflation_s,
        "jobs_s":     jobs_s,
        "rates_s":    rates_s,
    }


def _macro_label(score: float) -> str:
    if score >= 7.5:  return "Macro Bullish"
    elif score >= 6.0:return "Mild Macro Bull"
    elif score >= 4.5:return "Neutral"
    elif score >= 3.0:return "Mild Macro Bear"
    else:              return "Macro Bearish"


# ============================================================
# RISK REGIME
# ============================================================

RISK_ASSETS = {
    "SPX":    "^GSPC",
    "NDX":    "^NDX",
    "RUT":    "^RUT",
    "VIX":    "^VIX",
    "VIX3M":  "^VIX3M",
    "HYG":    "HYG",
    "LQD":    "LQD",
    "GLD":    "GC=F",
    "TLT":    "ZB=F",
    "DXY":    "DX-Y.NYB",
    "OIL":    "CL=F",
    "COPPER": "HG=F",
    "USDJPY": "JPY=X",
    # Bitcoin — speculative-appetite leg of the Growth & Crypto pillar (BTC/gold ratio)
    "BTC":    "BTC-USD",
    # TIPS ETF: used as real-yield proxy in historical backtest
    # TIP modified duration ~7.5y → price change inversely tracks real yield changes
    "TIP":    "TIP",
    # Treasury yields — needed for yield curve signal in historical regime scoring
    "TNX":    "^TNX",   # 10-year Treasury yield (term spread numerator)
    "IRX":    "^IRX",   # 13-week T-bill (term spread denominator + rate path proxy)
    # Equal-weight breadth
    "SPY":    "SPY",    # S&P 500 cap-weight
    "RSP":    "RSP",    # S&P 500 equal-weight
}

RISK_REGIME_CACHE: dict = {"data": None, "time": 0}
RISK_REGIME_CACHE_TTL = 3600  # 1h — aligns with main scores cache TTL


def _regime_core_score(rets: dict,
                       vix: float = None, vix3m: float = None,
                       hy_oas_bps: float = None, hy_delta_4w: float = None,
                       hyg_1m: float = None, lqd_1m: float = None,
                       geo_tension: float = None) -> dict:
    """
    Shared holistic risk-climate core — the SINGLE source of truth used by the
    live scorer (compute_risk_regime), the historical backtest (_score_regime_at)
    and the 52-week score-history endpoint. Graded (continuous) signals, no
    hard threshold flips.

    rets: {name: {"1w": pct, "1m": pct}} for any of
          SPX, NDX, RUT, VIX, GLD, COPPER, BTC, DXY, USDJPY (missing → 0)
    hy_oas_bps / hy_delta_4w: FRED BAML HY OAS level (bps) + 4-week change (bps).
          When unavailable (backtest), falls back to HYG−LQD 1m return spread.
    geo_tension: 0..1 geopolitical stress index from news scan, or None = no data.

    Pillars (total clamped to ±4):
      1. Equity trend        ±1.2   SPX 50% / RUT 25% / NDX 25%, 3m+1m+1w blend
                                     (3m weighted for trend context, so a healthy
                                     pullback inside a strong quarter reads as
                                     consolidation — not a downtrend)
      2. Volatility          ±1.2   VIX level (graded) + 1w momentum (capped,
                                     only active when VIX>18) + term structure
      3. Credit              ±1.0   HY OAS 4w trend + absolute level grade
      4. Havens & FX         ±0.8   gold, USD/JPY carry, DXY
      5. Growth & crypto     ±0.6   copper + BTC/gold ratio (speculative appetite)
      6. Geopolitics    −0.15..+0.05 news scan, multi-factor corroborated (tiny
                                     weight by design — headlines are noisy;
                                     drag scaled down when market shrugs it off)

    Returns {"score": -4..+4, "components": {pillar: contrib}, "detail": {...}}
    """
    def _g(name, k):
        try:
            v = rets.get(name, {}).get(k)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _cl(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    # ── 1. Equity trend ±1.2 — breadth-weighted, trend + pullback aware ─────
    # SPX carries half; RUT (breadth/cyclicals) + NDX (duration/growth) quarter each.
    # Blend across 3m/1m/1w = 0.45/0.40/0.15 so a healthy pullback inside a
    # strong quarter reads as consolidation, not a downtrend. When 3m data is
    # missing (backtest/history), the blend falls back to 1m/1w = 0.75/0.25.
    eq_blend = 0.0
    _has_3m = any((_g(_n, "3m") != 0.0) for _n in ("SPX", "NDX", "RUT"))
    for _nm, _w in (("SPX", 0.50), ("RUT", 0.25), ("NDX", 0.25)):
        if _has_3m:
            eq_blend += _w * (0.45 * _g(_nm, "3m") + 0.40 * _g(_nm, "1m") + 0.15 * _g(_nm, "1w"))
        else:
            eq_blend += _w * (0.75 * _g(_nm, "1m") + 0.25 * _g(_nm, "1w"))
    # Divisor lifted 2.0 → 3.0 with 3m data (a full-trend ±4.5% quarterly move
    # is the ±1.2 anchor). Backtest keeps the 2.0 divisor for continuity.
    eq_pillar = _cl(eq_blend / (3.0 if _has_3m else 2.0)) * 1.2

    # ── 2. Volatility ±1.2 — graded level + momentum + term structure ───────
    # 1w VIX momentum divisor 15 → 25 and capped at ±0.20 so a mechanical spike
    # off a low base (e.g. 15 → 19 = +25%) can't dominate the pillar. Also
    # gated: the momentum sub-component only activates when VIX is genuinely
    # elevated (>18) — below that, 1w changes are noise, not signal.
    vol_pillar = 0.0
    vix_1w = _g("VIX", "1w")
    vix_calm = (vix is not None and vix > 0 and vix < 20.0)
    if vix and vix > 0:
        _lvl = _cl((19.0 - vix) / 5.0, -1.5, 1.2) * 0.50      # graded around 19
        _chg_raw = _cl(-vix_1w / 25.0) * 0.35 if vix > 18.0 else 0.0
        _chg = max(-0.20, min(0.20, _chg_raw))                 # cap ±0.20
        _ts = 0.0
        if vix3m and vix3m > 0:
            _ts = _cl((vix3m / vix - 1.0) * 10.0) * 0.35       # contango/inversion
        vol_pillar = _cl(_lvl + _chg + _ts, -1.2, 1.2)

    # ── 3. Credit ±1.0 — OAS trend + level; HYG−LQD fallback ────────────────
    credit_pillar = 0.0
    credit_widening = False
    if hy_oas_bps is not None:
        _d = _cl(-(hy_delta_4w or 0.0) / 30.0) * 0.6           # 4w spread trend
        _lvlg = (0.4 if hy_oas_bps < 250 else
                 0.2 if hy_oas_bps < 300 else
                 0.0 if hy_oas_bps < 450 else
                 -0.4 if hy_oas_bps < 550 else -0.6)           # absolute level
        credit_pillar = _cl(_d + _lvlg)
        credit_widening = (hy_delta_4w or 0.0) > 15.0          # >+15bp 4w = meaningful stress
    elif hyg_1m is not None and lqd_1m is not None:
        # Duration-matched-ish price fallback for history (no OAS series)
        credit_pillar = _cl((hyg_1m - lqd_1m) / 3.0) * 0.6
        credit_widening = (hyg_1m - lqd_1m) < -1.5             # HYG underperforming LQD

    # ── 4. Havens & FX ±0.8 — gold bid, JPY carry, dollar ────────────────────
    hav = (_cl(-_g("GLD", "1m") / 6.0) * 0.30 +
           _cl(_g("USDJPY", "1m") / 4.0) * 0.25 +
           _cl(-_g("DXY", "1m") / 3.0) * 0.30)
    hav_pillar = _cl(hav, -0.8, 0.8)
    # Havens bid signal for geo corroboration — gold catching a real bid
    havens_bid = _g("GLD", "1m") > 2.0

    # ── 5. Growth commodities & crypto ±0.6 ─────────────────────────────────
    grw = _cl(_g("COPPER", "1m") / 5.0) * 0.30
    btc_rel = None
    if "BTC" in rets:
        btc_rel = _g("BTC", "1m") - _g("GLD", "1m")           # BTC/gold ratio proxy
        grw += _cl(btc_rel / 12.0) * 0.30
    grw_pillar = _cl(grw, -0.6, 0.6)

    # ── 6. Geopolitics −0.15..+0.05 — noisy, multi-factor corroborated ───────
    # News keyword scans are inconsistent and hard-to-quantify. Weight is tiny
    # by design (max ±0.15 out of a ±4 raw score = ~3.75%). Additionally
    # requires market corroboration: if VIX is calm AND credit isn't widening
    # AND havens aren't bid, the drag is scaled to 20% (news noise the tape
    # is shrugging off). With any one confirmation, half weight. With two or
    # more, full (still small) weight.
    geo_pillar = 0.0
    _geo_confirms = None
    _geo_scale = None
    if geo_tension is not None:
        if geo_tension <= 0:
            geo_pillar = 0.05
        else:
            # Threshold: below 0.3 tension, ignore entirely (routine news flow)
            _t = max(0.0, (_cl(geo_tension, 0.0, 1.0) - 0.30) / 0.70)
            _raw_drag = -0.15 * _t
            _confirms = sum([(not vix_calm), credit_widening, havens_bid])
            _scale = 0.20 if _confirms == 0 else 0.5 if _confirms == 1 else 1.0
            geo_pillar = _raw_drag * _scale
            _geo_confirms = _confirms
            _geo_scale = _scale

    score = _cl(eq_pillar + vol_pillar + credit_pillar + hav_pillar + grw_pillar + geo_pillar,
                -4.0, 4.0)
    return {
        "score": round(score, 2),
        "components": {
            "equity":     round(eq_pillar, 2),
            "volatility": round(vol_pillar, 2),
            "credit":     round(credit_pillar, 2),
            "havens":     round(hav_pillar, 2),
            "growth":     round(grw_pillar, 2),
            "geo":        round(geo_pillar, 2),
        },
        "detail": {"eq_blend": round(eq_blend, 2), "btc_rel": (round(btc_rel, 2) if btc_rel is not None else None),
                   "geo_confirms": _geo_confirms, "geo_scale": _geo_scale},
    }

STOCK_CLIMATE_CACHE: dict = {"data": None, "time": 0}
STOCK_CLIMATE_CACHE_TTL = 3600  # 1h



# ── International CB rates cache ─────────────────────────────────────────────
_INTL_RATES_CACHE: dict = {"data": None, "time": 0}
_INTL_RATES_TTL = 3600 * 6  # 6h

# FRED series for central bank overnight/policy rates
_CB_RATE_SERIES = {
    "BOE":    "IUDSOIA",           # BoE SONIA daily
    "ECB":    "ECBDFR",            # ECB deposit facility rate
    "BOJ":    "IR3TIB01JPM156N",   # Bank of Japan 3m interbank (IRSTCB01 stale as of 2023)
    "RBA":    "IR3TIB01AUM156N",   # RBA 3m interbank (policy rate proxy, current)
    "BOC":    "IR3TIB01CAM156N",   # Bank of Canada 3m interbank
    "RBNZ":   "IR3TIB01NZM156N",   # RBNZ 3m interbank (no direct policy rate on FRED)
    "SNB":    "IR3TIB01CHM156N",   # SNB 3m interbank (policy rate proxy)
    "BANXICO":"IR3TIB01MXM156N",   # Banxico 3m interbank
    "US":     "FEDFUNDS",          # US Fed Funds (key must match frontend CB_ORDER 'US')
}

# ── Hardcoded CB policy rate fallback layer ───────────────────────────────────
# Anchor rates as of the last confirmed decision + PUBLISHED meeting calendars.
# The 'rate' is only a safety anchor: for true policy-rate FRED series
# (FEDFUNDS/SONIA/ECBDFR) fresh FRED data always wins; for interbank proxies the
# rate auto-tracks FRED moves after the anchor date (see _compute_intl_rates).
# 'meetings' = published decision dates (ISO). next_meeting is computed
# dynamically at request time — never hardcode a single date.
# Format: { CB_KEY: { "rate": float, "date": "YYYY-MM-DD", "meetings": [...],
#                     "prev_rate": float, "cycle_peak": float, "cycle_trough": float } }
_CB_POLICY_FALLBACK = {
    # US: target 3.50-3.75% (held 17 Jun 2026, first meeting under Warsh). EFFR ~3.63.
    # FOMC decision days: federalreserve.gov/monetarypolicy/fomccalendars.htm
    "US":    {"rate": 3.63,  "date": "2026-06-17",
              "meetings": ["2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
                           "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
                           "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08"],
              "prev_rate": 4.33, "cycle_peak": 5.33, "cycle_trough": 0.08,
              "target_low": 3.50, "target_high": 3.75},
    # BoE: Bank Rate 3.75% (held 18 Jun 2026, 7-2). bankofengland.co.uk MPC dates
    "BOE":   {"rate": 3.75,  "date": "2026-06-18",
              "meetings": ["2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17",
                           "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-17",
                           "2027-07-29", "2027-09-16", "2027-11-04", "2027-12-16"],
              "prev_rate": 4.25, "cycle_peak": 5.25, "cycle_trough": 0.10},
    # ECB: Deposit facility rate 2.25% (HIKED +25bp 11 Jun 2026, effective 17 Jun)
    "ECB":   {"rate": 2.25,  "date": "2026-06-11",
              "meetings": ["2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
                           "2027-02-04", "2027-03-18", "2027-04-29", "2027-06-10",
                           "2027-07-22", "2027-09-09", "2027-10-28", "2027-12-16"],
              "prev_rate": 2.00, "cycle_peak": 4.00, "cycle_trough": -0.50},
    # BoJ: Policy rate 1.00% (HIKED +25bp 16 Jun 2026). 2027 dates not yet published.
    "BOJ":   {"rate": 1.00,  "date": "2026-06-16",
              "meetings": ["2026-07-31", "2026-09-18", "2026-10-30", "2026-12-18"],
              "prev_rate": 0.50, "cycle_peak": 1.00, "cycle_trough": -0.10},
    # RBA: Cash rate 4.35% (hiked 6 May 2026, held 17 Jun). Decision = day 2 of meeting.
    "RBA":   {"rate": 4.35,  "date": "2026-05-06",
              "meetings": ["2026-08-11", "2026-09-29", "2026-11-03", "2026-12-08"],
              "prev_rate": 3.85, "cycle_peak": 4.35, "cycle_trough": 0.10},
    # BoC: Policy rate 2.25% (held 10 Jun 2026, 5th consecutive hold)
    "BOC":   {"rate": 2.25,  "date": "2026-06-10",
              "meetings": ["2026-07-15", "2026-09-02", "2026-10-28", "2026-12-09"],
              "prev_rate": 2.75, "cycle_peak": 5.00, "cycle_trough": 0.25},
    # RBNZ: OCR 2.50% (HIKED +25bp 8 Jul 2026, tightening bias)
    "RBNZ":  {"rate": 2.50,  "date": "2026-07-08",
              "meetings": ["2026-09-02", "2026-10-28", "2026-12-09", "2027-02-10"],
              "prev_rate": 3.25, "cycle_peak": 5.50, "cycle_trough": 0.25},
    # SNB: Policy rate 0.00% (held 18 Jun 2026). Quarterly assessments.
    "SNB":   {"rate": 0.00,  "date": "2026-06-18",
              "meetings": ["2026-09-24", "2026-12-17"],
              "prev_rate": 0.00, "cycle_peak": 1.75, "cycle_trough": -0.75},
}

def _next_cb_meeting(fallback: dict):
    """First not-yet-past decision date from the CB's published calendar.

    Computed at request time so the value rolls forward automatically as
    meetings pass. Returns None once the calendar is exhausted (frontend
    renders a null meeting gracefully) — refresh the 'meetings' lists when
    central banks publish their next-year schedules.
    """
    from datetime import datetime as _dtm
    today = _dtm.utcnow().strftime("%Y-%m-%d")
    for d in fallback.get("meetings") or []:
        if d >= today:
            return d
    legacy = fallback.get("next_meeting")
    if legacy and legacy >= today:
        return legacy
    return None

def _compute_intl_rates() -> dict:
    """Fetch central bank policy rates from FRED, apply hardcoded fallback for accuracy.

    Strategy:
    - Use FRED series for TREND signals (3m/6m direction) — FRED is reliable for this
    - Override the 'rate' field with _CB_POLICY_FALLBACK when FRED diverges >30bp or is stale
    - Always attach next_meeting, cycle_peak, cycle_trough, change_from_peak from fallback
    - This gives accurate rates + real trend signals from continuous data
    """
    now = time.time()
    if _INTL_RATES_CACHE["data"] and (now - _INTL_RATES_CACHE["time"]) < _INTL_RATES_TTL:
        return _INTL_RATES_CACHE["data"]

    from datetime import datetime as _dt
    result = {}
    STALE_CUTOFF_DAYS = 548  # 18 months
    FALLBACK_DIVERGE_THRESHOLD = 0.30  # 30bp divergence triggers override
    FALLBACK_STALE_DAYS = 45   # override if FRED data >45 days old

    for cb, fred_id in _CB_RATE_SERIES.items():
        fallback = _CB_POLICY_FALLBACK.get(cb, {})
        try:
            _DAILY_CBS = {"BOE", "ECB"}
            periods = 400 if cb in _DAILY_CBS else 36
            raw = fetch_fred_series(fred_id, periods)
            if not raw or len(raw) < 3:
                raise ValueError("insufficient FRED data")
            vals = [x["value"] for x in raw if x.get("value") is not None]
            dates = [x["date"]  for x in raw if x.get("value") is not None]
            if len(vals) < 3:
                raise ValueError("insufficient FRED data")

            # Staleness check
            last_date = _dt.strptime(dates[-1], "%Y-%m-%d")
            days_old  = (datetime.utcnow() - last_date).days
            if days_old > STALE_CUTOFF_DAYS:
                raise ValueError("FRED data too stale")

            fred_rate = vals[-1]

            # Build trend using the FRED time-series (even if rate is overridden)
            if cb in _DAILY_CBS:
                v_3m = vals[-65]  if len(vals) >= 65  else vals[0]
                v_6m = vals[-130] if len(vals) >= 130 else vals[0]
                v_12m = vals[-250] if len(vals) >= 250 else vals[0]
            else:
                v_3m  = vals[-3]  if len(vals) >= 3  else vals[0]
                v_6m  = vals[-6]  if len(vals) >= 6  else vals[0]
                v_12m = vals[-12] if len(vals) >= 12 else vals[0]

            trend_3m  = round(fred_rate - v_3m,  3)
            trend_6m  = round(fred_rate - v_6m,  3)
            trend_12m = round(fred_rate - v_12m, 3)
            bias = 1 if trend_6m > 0.1 else -1 if trend_6m < -0.1 else 0

            # ── Rate determination (future-proof) ──────────────────────────
            # Policy-rate series (FEDFUNDS, SONIA, ECBDFR) track the true policy
            # rate on FRED → trust FRED whenever it is fresh; the hardcoded
            # fallback is only a safety net for stale/broken feeds. (The old
            # logic did the opposite — a >30bp divergence made the stale
            # hardcoded rate override fresh data, so real hikes/cuts were hidden.)
            # Interbank proxies (IR3TIB01*): start from the hand-verified anchor
            # rate and auto-track FRED moves AFTER the anchor, snapped to 25bp.
            # Baseline = first observation >=35 days post-anchor, so the move
            # that prompted the anchor update is never double-counted.
            _is_policy_series = fred_id in ("FEDFUNDS", "IUDSOIA", "ECBDFR")
            fallback_rate = fallback.get("rate")
            rate_source = "fred"
            actual_rate = fred_rate
            if _is_policy_series:
                # Monthly series (FEDFUNDS) publish with ~1 month lag — allow 75d
                _stale_days = FALLBACK_STALE_DAYS if cb in _DAILY_CBS else 75
                if days_old > _stale_days and fallback_rate is not None:
                    actual_rate = fallback_rate
                    rate_source = "fallback"
            elif fallback_rate is not None:
                actual_rate = fallback_rate
                rate_source = "fallback"
                try:
                    _anchor = fallback.get("date")
                    if _anchor:
                        _anchor_dt = _dt.strptime(_anchor, "%Y-%m-%d")
                        _baseline = None
                        for _d, _v in zip(dates, vals):
                            if (_dt.strptime(_d, "%Y-%m-%d") - _anchor_dt).days >= 35:
                                _baseline = _v
                                break
                        if _baseline is not None:
                            _step = round((fred_rate - _baseline) / 0.25) * 0.25
                            if abs(_step) >= 0.25:
                                actual_rate = fallback_rate + _step
                                rate_source = "fallback+delta"
                except Exception:
                    pass
            use_fallback_rate = rate_source != "fred"

            # --- Label logic ---
            # For CB policy rates (FEDFUNDS, IUDSOIA, ECBDFR): use t3 as primary
            # For interbank proxies (IR3TIB01*): use t6 as primary, t3 as confirmation
            # Additionally use fallback data to confirm: if peak-to-now delta is large,
            # label reflects the cycle even if recent FRED noise is flat
            _IS_POLICY_RATE = fred_id in ("FEDFUNDS", "IUDSOIA", "ECBDFR")
            _t3_flat = abs(trend_3m) < 0.05
            _t6_flat = abs(trend_6m) < 0.1

            # Fallback-assisted cycle label: compare actual rate vs prev_rate over 12m
            if fallback:
                fb_prev  = fallback.get("prev_rate", actual_rate)
                fb_peak  = fallback.get("cycle_peak", actual_rate)
                fb_trough= fallback.get("cycle_trough", actual_rate)
                fb_delta_12m = round(actual_rate - fb_prev, 3)  # change since ~12M ago
                at_peak  = abs(actual_rate - fb_peak) < 0.26
                at_trough= abs(actual_rate - fb_trough) < 0.26
                actively_cutting = fb_delta_12m < -0.30
                actively_hiking  = fb_delta_12m > 0.30
            else:
                fb_delta_12m = trend_12m
                at_peak = at_trough = False
                actively_cutting = fb_delta_12m < -0.30
                actively_hiking  = fb_delta_12m > 0.30

            if _IS_POLICY_RATE:
                if _t3_flat:
                    if abs(trend_6m) > 0.3 or abs(fb_delta_12m) > 0.3:
                        _label = "Paused"
                    else:
                        _label = "Flat"
                elif trend_3m > 0.5:
                    _label = "Hiking"
                elif trend_3m > 0.1:
                    _label = "Tightening"
                elif trend_3m < -0.5:
                    _label = "Cutting"
                elif trend_3m < -0.1:
                    _label = "Easing"
                else:
                    _label = "Flat"
            else:
                # Interbank proxy — use t6 + fallback confirmation
                if (trend_6m > 0.5 and trend_3m > 0.05) or actively_hiking:
                    _label = "Hiking"
                elif trend_6m > 0.1 and trend_3m > -0.05:
                    _label = "Tightening"
                elif (trend_6m < -0.5 and trend_3m < -0.05) or actively_cutting:
                    _label = "Cutting"
                elif trend_6m < -0.1 and trend_3m < 0.05:
                    _label = "Easing"
                elif _t6_flat and not actively_cutting and not actively_hiking:
                    _label = "Flat"
                else:
                    _label = "Paused" if (abs(trend_6m) > 0.3 or abs(fb_delta_12m) > 0.3) else "Flat"

            # Dynamic Fed target band: derive from the live effective rate so
            # the band moves automatically with future hikes/cuts (25bp grid).
            _tgt_low  = fallback.get("target_low")
            _tgt_high = fallback.get("target_high")
            if cb == "US" and actual_rate is not None:
                _tgt_low  = int(actual_rate / 0.25) * 0.25
                _tgt_high = round(_tgt_low + 0.25, 2)

            result[cb] = {
                "rate":           round(actual_rate, 2),
                "rate_source":    rate_source,
                "fred_rate":      round(fred_rate, 3),
                "trend_3m":       trend_3m,
                "trend_6m":       trend_6m,
                "trend_12m":      trend_12m,
                "change_12m":     round(fb_delta_12m, 2),
                "bias":           bias,
                "data_date":      dates[-1],
                "label":          _label,
                "next_meeting":   _next_cb_meeting(fallback),
                "cycle_peak":     fallback.get("cycle_peak"),
                "cycle_trough":   fallback.get("cycle_trough"),
                "change_from_peak": round(actual_rate - fallback.get("cycle_peak", actual_rate), 2) if fallback.get("cycle_peak") is not None else None,
                "target_low":     _tgt_low,
                "target_high":    _tgt_high,
            }
        except Exception:
            # FRED fetch failed entirely — use fallback-only if available
            if fallback and fallback.get("rate") is not None:
                fb_rate  = fallback["rate"]
                fb_prev  = fallback.get("prev_rate", fb_rate)
                fb_delta = round(fb_rate - fb_prev, 2)
                if fb_delta > 0.30:
                    _label = "Hiking"
                elif fb_delta < -0.30:
                    _label = "Cutting"
                elif abs(fb_delta) > 0.05:
                    _label = "Paused"
                else:
                    _label = "Flat"
                result[cb] = {
                    "rate":           round(fb_rate, 2),
                    "rate_source":    "fallback_only",
                    "fred_rate":      None,
                    "trend_3m":       None,
                    "trend_6m":       fb_delta,
                    "trend_12m":      fb_delta,
                    "change_12m":     fb_delta,
                    "bias":           1 if fb_delta > 0.1 else -1 if fb_delta < -0.1 else 0,
                    "data_date":      fallback.get("date"),
                    "label":          _label,
                    "next_meeting":   _next_cb_meeting(fallback),
                    "cycle_peak":     fallback.get("cycle_peak"),
                    "cycle_trough":   fallback.get("cycle_trough"),
                    "change_from_peak": round(fb_rate - fallback.get("cycle_peak", fb_rate), 2) if fallback.get("cycle_peak") is not None else None,
                    "target_low":     fallback.get("target_low"),
                    "target_high":    fallback.get("target_high"),
                }
            continue

    _INTL_RATES_CACHE["data"] = result
    _INTL_RATES_CACHE["time"] = now
    return result


# ============================================================
# STOCK MARKET CLIMATE
# ============================================================

def compute_stock_climate() -> dict:
    """
    Compute a Stock Market Climate panel with four pillars:
      VIX           — fear / volatility regime
      Forward PE    — valuation (Shiller CAPE via FRED)
      SPY/RSP       — breadth (equal-weight vs cap-weight divergence)
      S&P Momentum  — 3m price momentum on SPX
    Returns a dict ready for the frontend macro panel.
    """
    now = time.time()
    if STOCK_CLIMATE_CACHE["data"] and (now - STOCK_CLIMATE_CACHE["time"]) < STOCK_CLIMATE_CACHE_TTL:
        return STOCK_CLIMATE_CACHE["data"]

    result = {}
    signals = {}

    try:
        # ── 1. VIX ──────────────────────────────────────────────────────────
        try:
            _vix_ticker = yf.Ticker("^VIX")
            vix_tk = _yf_with_timeout(_vix_ticker.history, period="5d", interval="1d", label="VIX")
            vix_level = float(vix_tk["Close"].dropna().iloc[-1]) if vix_tk is not None and not vix_tk.empty else None
        except Exception:
            vix_level = None

        if vix_level is not None:
            if vix_level < 13:
                vix_score, vix_label = 2, "Very Low"
            elif vix_level < 17:
                vix_score, vix_label = 1, "Low"
            elif vix_level < 21:
                vix_score, vix_label = 0, "Moderate"
            elif vix_level < 27:
                vix_score, vix_label = -1, "Elevated"
            else:
                vix_score, vix_label = -2, "High Stress"
            signals["VIX"] = {
                "title": "VIX",
                "value": f"{vix_level:.1f}",
                "label": vix_label,
                "score": vix_score,
                "category": "volatility",
            }

        # ── 2. (Slot reserved — FORWARD_PE signal removed; FWD_PE signal covers valuation) ──
        pass

        # ── 3. SPY/RSP Breadth ──────────────────────────────────────────────
        try:
            prices = {}
            for tk_key, tk_sym in [("SPY", "SPY"), ("RSP", "RSP")]:
                _tk = yf.Ticker(tk_sym)
                d = _yf_with_timeout(_tk.history, period="6mo", interval="1wk", label=tk_sym)
                if d is not None and not d.empty:
                    prices[tk_key] = d["Close"].dropna()

            if "SPY" in prices and "RSP" in prices:
                # Align on common dates
                spy = prices["SPY"]
                rsp = prices["RSP"]
                common = spy.index.intersection(rsp.index)
                if len(common) >= 12:
                    spy_c = spy.loc[common]
                    rsp_c = rsp.loc[common]
                    ratio_now  = rsp_c.iloc[-1] / spy_c.iloc[-1]
                    ratio_12w  = rsp_c.iloc[-12] / spy_c.iloc[-12]
                    breadth_chg = (ratio_now - ratio_12w) / ratio_12w  # fractional change

                    if breadth_chg > 0.04:
                        br_score, br_label = 2, "Broad Rally"
                    elif breadth_chg > 0.015:
                        br_score, br_label = 1, "Improving"
                    elif breadth_chg > -0.015:
                        br_score, br_label = 0, "Neutral"
                    elif breadth_chg > -0.05:
                        br_score, br_label = -1, "Narrowing"
                    else:
                        br_score, br_label = -2, "Thin Breadth"

                    signals["BREADTH"] = {
                        "title": "SPY/RSP Breadth",
                        "value": f"{breadth_chg*100:+.1f}%",
                        "label": br_label,
                        "score": br_score,
                        "category": "breadth",
                    }
        except Exception:
            pass

        # ── 4. S&P 500 Momentum (3m) ─────────────────────────────────────────
        try:
            _spx_tk2 = yf.Ticker("^GSPC")
            spx = _yf_with_timeout(_spx_tk2.history, period="6mo", interval="1wk", label="SPX_MOM")
            if spx is not None and not spx.empty:
                closes = spx["Close"].dropna()
                if len(closes) >= 13:
                    mom_3m = (closes.iloc[-1] / closes.iloc[-13] - 1) * 100
                    if mom_3m > 8:
                        mom_score, mom_label = 2, "Strong Uptrend"
                    elif mom_3m > 3:
                        mom_score, mom_label = 1, "Positive"
                    elif mom_3m > -3:
                        mom_score, mom_label = 0, "Neutral"
                    elif mom_3m > -8:
                        mom_score, mom_label = -1, "Negative"
                    else:
                        mom_score, mom_label = -2, "Downtrend"
                    signals["SPX_MOM"] = {
                        "title": "S&P 3m Momentum",
                        "value": f"{mom_3m:+.1f}%",
                        "label": mom_label,
                        "score": mom_score,
                        "category": "momentum",
                    }
                    # Store last 26 weekly closes for frontend sparkline
                    n_spark = min(26, len(closes))
                    signals["_spx_closes"] = [round(float(v), 2) for v in closes.iloc[-n_spark:].tolist()]
        except Exception:
            pass

        # ── 5. VIX Term Structure (VIX3M/VIX ratio) ─────────────────────────
        try:
            _vix3m_tk = yf.Ticker("^VIX3M")
            vix3m_hist = _yf_with_timeout(_vix3m_tk.history, period="5d", interval="1d", label="VIX3M_SC")
            vix3m_level = float(vix3m_hist["Close"].dropna().iloc[-1]) if vix3m_hist is not None and not vix3m_hist.empty else None
            if vix3m_level is not None and vix_level is not None and vix_level > 0:
                ts_ratio = vix3m_level / vix_level
                if ts_ratio > 1.10:
                    ts_score, ts_label = 2, "Steep Contango"
                elif ts_ratio > 1.02:
                    ts_score, ts_label = 1, "Contango"
                elif ts_ratio > 0.95:
                    ts_score, ts_label = 0, "Flat"
                elif ts_ratio > 0.88:
                    ts_score, ts_label = -1, "Backwardation"
                else:
                    ts_score, ts_label = -2, "Deep Backwardation"
                signals["VIX_TS"] = {
                    "title": "VIX Term Structure",
                    "value": f"{ts_ratio:.3f}",
                    "label": ts_label,
                    "score": ts_score,
                    "category": "volatility",
                    "vix3m": round(vix3m_level, 1),
                }
        except Exception:
            pass

        # ── 6. NFCI (Chicago Fed Financial Conditions) ───────────────────────
        try:
            nfci_raw = fetch_fred_series("NFCI", 8)  # 8 weekly obs
            if nfci_raw and len(nfci_raw) >= 1:
                nfci_vals = [r["value"] for r in nfci_raw if r.get("value") is not None]
                if nfci_vals:
                    nfci_now = nfci_vals[-1]
                    nfci_4w  = nfci_vals[-4] if len(nfci_vals) >= 4 else nfci_now
                    nfci_delta = round(nfci_now - nfci_4w, 3)
                    # Negative = loose (good), positive = tight (bad)
                    if nfci_now < -0.5:
                        nfci_score, nfci_label = 2, "Very Loose"
                    elif nfci_now < -0.1:
                        nfci_score, nfci_label = 1, "Loose"
                    elif nfci_now < 0.2:
                        nfci_score, nfci_label = 0, "Neutral"
                    elif nfci_now < 0.6:
                        nfci_score, nfci_label = -1, "Tightening"
                    else:
                        nfci_score, nfci_label = -2, "Stressed"
                    signals["NFCI"] = {
                        "title": "Fin. Conditions (NFCI)",
                        "value": f"{nfci_now:+.2f}",
                        "label": nfci_label,
                        "score": nfci_score,
                        "category": "conditions",
                        "delta_4w": round(nfci_delta, 3),
                    }
        except Exception:
            pass

        # ── 7. Equity Risk Premium (ERP = earnings yield − 10Y yield) ────────
        # ERP = (1/TrailingPE * 100) − DGS10. Negative = stocks expensive vs bonds.
        # NOTE: SPY trailing PE is fetched later in section 10 (FWD_PE).
        # ERP computation is deferred to section 10b below to reuse that value.
        pass  # see section 10b

        # ── 8. HY Spread Quadrant (level + direction) ────────────────────────
        # Use the already-fetched HYOAS data from FRED for a secondary signal
        try:
            hy_raw = fetch_fred_series("HYOAS", 130)  # 6mo for direction
            if hy_raw and len(hy_raw) >= 20:
                hy_vals = [r["value"] for r in hy_raw if r.get("value") is not None]
                if hy_vals and len(hy_vals) >= 20:
                    hy_now  = hy_vals[-1] * 100   # convert % → bps
                    hy_4w   = hy_vals[-20] * 100
                    hy_dir  = hy_now - hy_4w       # positive=widening, negative=tightening
                    hy_wide = hy_now > 400          # wide = above 400bps
                    # Verdad quadrant: wide+falling=Recovery, wide+rising=Recession,
                    #                  narrow+falling=Growth, narrow+rising=Overheating
                    if not hy_wide and hy_dir < -10:
                        hy_quad, hy_qs, hy_ql = "Growth", 2, "Narrow & Tightening"
                    elif not hy_wide and hy_dir > 10:
                        # Overheating: spreads still tight but widening = late-cycle warning.
                        # Verdad research: weakest subsequent equity returns of the four quadrants.
                        # Correctly scored 0 (neutral warning), not +1 (bullish).
                        hy_quad, hy_qs, hy_ql = "Overheating", 0, "Narrow & Widening"
                    elif not hy_wide:
                        hy_quad, hy_qs, hy_ql = "Stable", 1, "Narrow & Stable"
                    elif hy_wide and hy_dir < -15:
                        hy_quad, hy_qs, hy_ql = "Recovery", 1, "Wide & Tightening"
                    else:
                        hy_quad, hy_qs, hy_ql = "Stress", -2, "Wide & Widening"
                    signals["HY_QUAD"] = {
                        "title": "HY Credit Regime",
                        "value": f"{round(hy_now)}bp",
                        "label": hy_ql,
                        "score": hy_qs,
                        "category": "credit",
                        "quadrant": hy_quad,
                        "dir_4w": round(hy_dir),
                    }
        except Exception:
            pass

        # ── 9. CBOE SKEW Index ───────────────────────────────────────────────
        # SKEW measures tail risk (left-tail demand from OTM puts).
        # Range typically 100–170; >135 = elevated tail hedging, <120 = complacent.
        # Useful as a divergence signal: Low VIX + High SKEW = stealth worry.
        try:
            _skew_tk = yf.Ticker("^SKEW")
            skew_hist = _yf_with_timeout(_skew_tk.history, period="5d", interval="1d", label="SKEW")
            skew_level = float(skew_hist["Close"].dropna().iloc[-1]) if skew_hist is not None and not skew_hist.empty else None
            if skew_level is not None:
                # SKEW is a contrarian signal: HIGH skew = protection demand = mildly bearish sentiment
                if skew_level < 110:
                    skew_score, skew_label = 2, "Complacent"
                elif skew_level < 125:
                    skew_score, skew_label = 1, "Low Hedging"
                elif skew_level < 135:
                    skew_score, skew_label = 0, "Neutral"
                elif skew_level < 145:
                    skew_score, skew_label = -1, "Elevated"
                else:
                    skew_score, skew_label = -2, "Tail Risk"
                # VIX/SKEW divergence: low VIX + high SKEW = institutional stealth worry
                vix_skew_div = (
                    vix_level is not None and vix_level < 18 and skew_level > 135
                )
                signals["SKEW"] = {
                    "title": "CBOE SKEW",
                    "value": f"{skew_level:.0f}",
                    "label": skew_label,
                    "score": skew_score,
                    "category": "volatility",
                    "divergence": vix_skew_div,
                }
        except Exception:
            pass

        # ── 10. S&P 500 Trailing P/E via yfinance ───────────────────────────
        # Uses SPY trailingPE as the best freely available S&P 500 valuation proxy.
        # Note: yfinance does not return forwardPE for ETFs like SPY — trailingPE is
        # the most consistent free signal. Labelled clearly as "Trailing P/E".
        # Historical range (Shiller, modern era): ~9x trough / ~45x dotcom peak.
        # 5yr avg (post-2019) ~21x; LT avg (post-1990) ~17-18x.
        try:
            _fpe_tk = yf.Ticker("SPY")
            _fpe_info = _fpe_tk.info
            fwd_pe = _fpe_info.get("trailingPE")
            if fwd_pe is None:
                fwd_pe = _fpe_info.get("forwardPE")
            if fwd_pe:
                fwd_pe = float(fwd_pe)
                # Scoring: LT fair value ~17x; recent era 5yr avg ~21x; extreme >35x
                if fwd_pe < 14:
                    fpe_score, fpe_label = 2, "Very Cheap"
                elif fwd_pe < 18:
                    fpe_score, fpe_label = 1, "Fair Value"
                elif fwd_pe < 23:
                    fpe_score, fpe_label = 0, "Elevated"
                elif fwd_pe < 30:
                    fpe_score, fpe_label = -1, "Expensive"
                else:
                    fpe_score, fpe_label = -2, "Very Expensive"
                signals["FWD_PE"] = {
                    "title": "S&P 500 Trailing P/E",
                    "value": f"{fwd_pe:.1f}x",
                    "label": fpe_label,
                    "score": fpe_score,
                    "category": "valuation",
                    "raw": round(fwd_pe, 2),
                    "avg5yr": 21.4,   # ~5yr post-COVID average
                    "avg_lt": 17.5,   # long-term average (post-1990)
                }
        except Exception:
            pass

        # ── 10b. ERP using PE from section 10 ───────────────────────────────
        # Reuses FWD_PE["raw"] to avoid a second yfinance call.
        try:
            dgs10_raw = fetch_fred_series("DGS10", 5)
            dgs10_now = None
            if dgs10_raw:
                dgs10_vals = [r["value"] for r in dgs10_raw if r.get("value") is not None]
                if dgs10_vals:
                    dgs10_now = dgs10_vals[-1]
            spy_pe_float = signals.get("FWD_PE", {}).get("raw") if signals.get("FWD_PE") else None
            if spy_pe_float and spy_pe_float > 0 and dgs10_now:
                earnings_yield = round(1.0 / spy_pe_float * 100, 2)  # %
                erp = round(earnings_yield - dgs10_now, 2)
                if erp > 1.5:
                    erp_score, erp_label = 2, "Cheap vs Bonds"
                elif erp > 0.0:
                    erp_score, erp_label = 1, "Fair vs Bonds"
                elif erp > -1.5:
                    erp_score, erp_label = -1, "Expensive vs Bonds"
                else:
                    erp_score, erp_label = -2, "Very Expensive"
                signals["ERP"] = {
                    "title": "Equity Risk Premium",
                    "value": f"{erp:+.2f}%",
                    "label": erp_label,
                    "score": erp_score,
                    "category": "valuation",
                    "erp_raw": erp,
                    "dgs10": round(dgs10_now, 2),
                    "ey": earnings_yield,
                }
        except Exception:
            pass

        # ── 11. CBOE Equity Put/Call Ratio ───────────────────────────────────
        # Uses the same fetch_pcr_history() already used by the PCR tab.
        # Equity-only strips out index/ETF hedging bias — much cleaner sentiment.
        # CBOE equity P/C thresholds (historical range 0.35–1.20):
        #   < 0.45 → extreme greed / complacency
        #   0.45–0.60 → greed
        #   0.60–0.75 → neutral
        #   0.75–0.90 → mild fear / defensive
        #   > 0.90 → elevated fear (contrarian buy signal)
        try:
            _pcr_df = fetch_pcr_history()
            if _pcr_df is not None and not _pcr_df.empty:
                _pcr_ma5_series = _pcr_df["pc_ma5"].dropna() if "pc_ma5" in _pcr_df.columns else None
                _pcr_daily = _pcr_df["equity_pc"].dropna()
                if _pcr_ma5_series is not None and len(_pcr_ma5_series) > 0:
                    pcr_now  = float(_pcr_daily.iloc[-1])          # raw daily — SCORING basis (fast)
                    pcr_prev = float(_pcr_daily.iloc[-2]) if len(_pcr_daily) >= 2 else pcr_now
                    pcr_ma5  = float(_pcr_ma5_series.iloc[-1])     # 5d MA — kept as context only
                    pcr_ma20 = float(_pcr_df["pc_ma20"].dropna().iloc[-1]) if "pc_ma20" in _pcr_df.columns else None
                    pcr_date = str(_pcr_daily.index[-1].date())
                    # FAST-REACTING: score off the latest DAILY print, not the MA,
                    # so a sharp same-day swing into fear/greed shows immediately.
                    # Thresholds calibrated to CBOE equity PCR distribution 2006-present
                    if pcr_now > 0.85:
                        pcr_score, pcr_label = 2, "Elevated Fear"
                    elif pcr_now > 0.68:
                        pcr_score, pcr_label = 1, "Defensive"
                    elif pcr_now > 0.55:
                        pcr_score, pcr_label = 0, "Neutral"
                    elif pcr_now > 0.45:
                        pcr_score, pcr_label = -1, "Greed"
                    else:
                        pcr_score, pcr_label = -2, "Extreme Greed"
                    signals["PUT_CALL"] = {
                        "title": "CBOE Equity Put/Call",
                        "value": f"{pcr_now:.2f}",  # latest daily print as headline
                        "label": pcr_label,
                        "score": pcr_score,
                        "category": "sentiment",
                        "raw":   round(pcr_now,  3),  # daily print used by thermometer
                        "daily": round(pcr_now,  3),
                        "change": round(pcr_now - pcr_prev, 3),  # today's move
                        "ma5":   round(pcr_ma5,  3),  # context only
                        "ma20":  round(pcr_ma20, 3) if pcr_ma20 else None,
                        "date":  pcr_date,
                    }
        except Exception:
            pass

        # ── Composite score ──────────────────────────────────────────────────
        weights = {"VIX": 0.20, "VIX_TS": 0.15, "BREADTH": 0.15, "SPX_MOM": 0.20,
                   "HY_QUAD": 0.15, "NFCI": 0.10, "ERP": 0.05}
        composite = 0.0
        total_w = 0.0
        for k, w in weights.items():
            if k in signals:
                composite += signals[k]["score"] * w
                total_w += w
        composite = composite / total_w if total_w > 0 else 0

        overall = (2 if composite > 1.2 else 1 if composite > 0.4 else
                   -2 if composite < -1.2 else -1 if composite < -0.4 else 0)

        result = {
            "signals": signals,
            "composite": round(composite, 2),
            "overall": overall,
        }

    except Exception as e:
        result = {"signals": {}, "composite": 0, "overall": 0, "error": str(e)}

    STOCK_CLIMATE_CACHE["data"] = result
    STOCK_CLIMATE_CACHE["time"] = now
    return result


def compute_risk_regime() -> dict:
    now = time.time()
    if RISK_REGIME_CACHE["data"] and (now - RISK_REGIME_CACHE["time"]) < RISK_REGIME_CACHE_TTL:
        return RISK_REGIME_CACHE["data"]

    returns: dict = {}
    levels:  dict = {}

    # Fetch RISK_ASSETS using fetch_price_data (uses PRICE_CACHE + _yf_with_timeout)
    # PRICE_CACHE is pre-warmed before compute_risk_regime() is called, so these
    # are typically instant cache hits — no raw yfinance calls needed at this point.
    def _fetch_one_risk_asset(name_ticker):
        name, ticker = name_ticker
        try:
            # Use fetch_price_data: respects PRICE_CACHE + _yf_with_timeout (20s)
            df = fetch_price_data(ticker)
            if df is not None and not df.empty and len(df) >= 5:
                # Resample daily → weekly (Friday close)
                weekly = df["Close"].resample("W-FRI").last().dropna()
                close = weekly.values.astype(float)
                if len(close) >= 4:
                    ret_1w = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
                    ret_1m = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0
                    # 3m ≈ 13 weekly bars; fall back to earliest available
                    idx_3m = max(0, len(close) - 13)
                    ret_3m = (close[-1] / close[idx_3m] - 1) * 100
                    return name, {"return_1w": round(ret_1w, 2), "return_1m": round(ret_1m, 2), "return_3m": round(ret_3m, 2)}, round(float(close[-1]), 4)
        except Exception as _e:
            print(f"[regime_fetch] {name}/{ticker}: {_e}", flush=True)
        return name, {"return_1w": 0, "return_1m": 0, "return_3m": 0}, None

    _ra_ex = _cf.ThreadPoolExecutor(max_workers=8)
    try:
        _ra_futs = {_ra_ex.submit(_fetch_one_risk_asset, item): item[0] for item in RISK_ASSETS.items()}
        done_ra, _ = _cf.wait(_ra_futs, timeout=60)  # 60s — each call uses _yf_with_timeout(20s)
        for fut in done_ra:
            try:
                name, ret_dict, level = fut.result()
                returns[name] = ret_dict
                if level is not None:
                    levels[name] = level
            except Exception:
                pass
    finally:
        _ra_ex.shutdown(wait=False)

    regime_signals: dict = {}  # structured dict: key -> {signal, value, label}

    def _clamp1(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    # ── Raw inputs for the shared scoring core ──────────────────────────────
    spx_1m = returns.get("SPX", {}).get("return_1m", 0)
    spx_1w = returns.get("SPX", {}).get("return_1w", 0)
    ndx_1m = returns.get("NDX", {}).get("return_1m", 0)
    ndx_1w = returns.get("NDX", {}).get("return_1w", 0)
    rut_1m = returns.get("RUT", {}).get("return_1m", 0)
    rut_1w = returns.get("RUT", {}).get("return_1w", 0)
    gld_1m = returns.get("GLD", {}).get("return_1m", 0)
    copper_1m = returns.get("COPPER", {}).get("return_1m", 0)
    btc_1m = returns.get("BTC", {}).get("return_1m", 0)
    dxy_1m = returns.get("DXY", {}).get("return_1m", 0)
    usdjpy_1m = returns.get("USDJPY", {}).get("return_1m", 0)
    vix_1w_chg = returns.get("VIX", {}).get("return_1w", 0)
    vix_level   = levels.get("VIX",  20.0)
    vix3m_level = levels.get("VIX3M", 0.0)
    ts_ratio = (vix3m_level / vix_level) if (vix_level > 0 and vix3m_level > 0) else None

    # ── Credit raw inputs (HYG/LQD price behaviour + FRED BAML OAS) ─────────
    # NOTE: scoring now happens inside _regime_core_score — this block only
    # gathers raw credit inputs + display enrichment (percentiles, deltas).
    hyg_1m = returns.get("HYG", {}).get("return_1m", 0)
    lqd_1m = returns.get("LQD", {}).get("return_1m", 0)
    spread_sig = hyg_1m - lqd_1m  # positive = HY outperforming = risk-on
    credit_trend = "Tightening" if spread_sig > 0.3 else "Widening" if spread_sig < -0.3 else "Neutral"

    # Fetch actual OAS levels from FRED (BAML indices) — daily, in basis points
    hy_oas_bps = None
    ig_oas_bps = None
    hy_oas_score = 0.0
    hy_delta_4w = None
    hy_delta_3m = None
    hy_delta_6m = None
    hy_ig_ratio = None
    ig_delta_4w = None
    ig_delta_3m = None
    ig_delta_6m = None
    hy_pct = None; hy_pct_min = None; hy_pct_max = None
    ig_pct = None; ig_pct_min = None; ig_pct_max = None
    try:
        hy_data = fetch_fred_series("HYOAS", 780)   # ~3 years of daily data for percentile
        if hy_data and len(hy_data) >= 4:
            vals = [row["value"] for row in hy_data if row.get("value") is not None]
            if vals:
                # BAMLH0A0HYM2 is in % (e.g. 2.85 = 285 bps) — convert to bps
                hy_oas_bps = round(vals[-1] * 100, 0)
                # Score: tight=bullish (+1), wide=bearish (-1)
                hy_oas_score = (1.0 if hy_oas_bps < 250 else
                                0.5 if hy_oas_bps < 300 else
                               -0.5 if hy_oas_bps < 450 else
                               -1.0 if hy_oas_bps >= 450 else 0.0)
                if len(vals) >= 20:
                    hy_delta_4w = round((vals[-1] - vals[-20]) * 100, 0)
                if len(vals) >= 65:
                    hy_delta_3m = round((vals[-1] - vals[-65]) * 100, 0)
                if len(vals) >= 130:
                    hy_delta_6m = round((vals[-1] - vals[-130]) * 100, 0)
                # Percentile vs 3yr history (tight end = low % = bullish)
                if len(vals) >= 20:
                    sorted_vals = sorted(vals)
                    cur = vals[-1]
                    below = sum(1 for v in sorted_vals if v <= cur)
                    hy_pct = round(below / len(sorted_vals) * 100, 0)
                    hy_pct_min = round(sorted_vals[0] * 100, 0)
                    hy_pct_max = round(sorted_vals[-1] * 100, 0)
                    del sorted_vals  # free large list
    except Exception as _e: print(f"[credit_signal] {_e}")
    try:
        ig_data = fetch_fred_series("IGOAS", 780)   # ~3 years
        if ig_data and len(ig_data) >= 2:
            ig_vals = [row["value"] for row in ig_data if row.get("value") is not None]
            if ig_vals:
                ig_oas_bps = round(ig_vals[-1] * 100, 0)  # % -> bps
                if hy_oas_bps is not None and ig_oas_bps > 0:
                    hy_ig_ratio = round(hy_oas_bps / ig_oas_bps, 2)
                if len(ig_vals) >= 20:
                    ig_delta_4w = round((ig_vals[-1] - ig_vals[-20]) * 100, 0)
                if len(ig_vals) >= 65:
                    ig_delta_3m = round((ig_vals[-1] - ig_vals[-65]) * 100, 0)
                if len(ig_vals) >= 130:
                    ig_delta_6m = round((ig_vals[-1] - ig_vals[-130]) * 100, 0)
                # Percentile vs 3yr history
                if len(ig_vals) >= 20:
                    sorted_ig = sorted(ig_vals)
                    cur_ig = ig_vals[-1]
                    below_ig = sum(1 for v in sorted_ig if v <= cur_ig)
                    ig_pct = round(below_ig / len(sorted_ig) * 100, 0)
                    ig_pct_min = round(sorted_ig[0] * 100, 0)
                    ig_pct_max = round(sorted_ig[-1] * 100, 0)
                    del sorted_ig  # free large list
    except Exception as _e: print(f"[credit_percentile] {_e}")

    # ── Geopolitical tension scan — FF breaking news, 48h window ────────────
    geo_hits = 0.0
    geo_top = ""
    geo_n_items = 0
    try:
        _geo_items = fetch_ff_news(hours_back=48) or []
        geo_n_items = len(_geo_items)
        _GEO_KW = ("war", "missile", "airstrike", "air strike", "strikes on", "attack",
                   "invasion", "invade", "escalat", "sanction", "nuclear", "conflict",
                   "troops", "airspace", "drone", "retaliat", "blockade", "embargo",
                   "coup", "hostage", "tariff", "geopolit", "warship",
                   "mobilis", "mobiliz", "martial law")
        for _gi in _geo_items[:80]:
            _gt = ((_gi.get("title") or "") + " " + (_gi.get("preview") or "")).lower()
            if any(_kw in _gt for _kw in _GEO_KW):
                geo_hits += 1.5 if (_gi.get("impact") == "high") else 1.0
                if not geo_top:
                    geo_top = (_gi.get("title") or "")[:90]
    except Exception as _e:
        print(f"[geo_signal] {_e}", flush=True)
    geo_tension = (min(1.0, geo_hits / 8.0) if geo_n_items > 0 else None)

    # ── Shared holistic core — single source of truth for the score ─────────
    _core_rets = {}
    for _nm in ("SPX", "NDX", "RUT", "VIX", "GLD", "COPPER", "BTC", "DXY", "USDJPY"):
        if _nm in returns:
            _core_rets[_nm] = {"1w": returns[_nm].get("return_1w", 0),
                               "1m": returns[_nm].get("return_1m", 0),
                               "3m": returns[_nm].get("return_3m", 0)}
    _core = _regime_core_score(
        _core_rets,
        vix=vix_level, vix3m=vix3m_level,
        hy_oas_bps=hy_oas_bps, hy_delta_4w=hy_delta_4w,
        hyg_1m=hyg_1m, lqd_1m=lqd_1m,
        geo_tension=geo_tension,
    )
    regime_score = _core["score"]
    _comp = _core["components"]
    _btc_rel = _core["detail"].get("btc_rel")

    # ── Legacy normalised vars still consumed by market-env composites ──────
    eq_raw = round(_comp["equity"] / 1.2 * 1.5, 2)          # ±1.5 scale
    credit_sig_norm = round(_clamp1(_comp["credit"]), 2)
    dxy_sig_norm = round(_clamp1(-dxy_1m / 3.0), 2)

    # ── Signals for the climate card (graded per-input reads) ───────────────
    def _updown(v, up, down, pos="Bullish", neg="Bearish", mid="Neutral"):
        return pos if v > up else neg if v < down else mid

    regime_signals["SPX"] = {
        "signal": round(_clamp1((0.6 * spx_1m + 0.4 * spx_1w) / 2.0), 2),
        "value": f"{spx_1m:+.1f}%",
        "label": _updown(spx_1m, 1.0, -1.0),
    }
    regime_signals["NDX"] = {
        "signal": round(_clamp1((0.6 * ndx_1m + 0.4 * ndx_1w) / 2.0), 2),
        "value": f"{ndx_1m:+.1f}%",
        "label": _updown(ndx_1m, 1.0, -1.0),
    }
    regime_signals["RTY"] = {
        "signal": round(_clamp1((0.6 * rut_1m + 0.4 * rut_1w) / 2.0), 2),
        "value": f"{rut_1m:+.1f}%",
        "label": _updown(rut_1m, 1.0, -1.0),
    }
    ts_label = ("Contango" if (ts_ratio or 1.0) > 1.02 else "Inverted" if (ts_ratio or 1.0) < 0.98 else "Flat")
    regime_signals["VIX"] = {
        "signal": round(_clamp1(_comp["volatility"] / 1.2), 2),
        "value": f"{vix_level:.1f}",
        "label": f"{ts_label} — {'Low' if vix_level < 16 else 'Elevated' if vix_level > 25 else 'Moderate'} vol"
                 + (f", 1w {vix_1w_chg:+.0f}%" if abs(vix_1w_chg) >= 5 else ""),
    }
    regime_signals["Credit"] = {
        "signal": credit_sig_norm,
        "value": (f"{hy_oas_bps:.0f}bp" if hy_oas_bps is not None else f"{spread_sig:+.1f}%"),
        "label": (f"HY OAS {'+' if (hy_delta_4w or 0) >= 0 else ''}{(hy_delta_4w or 0):.0f}bp 4w — {credit_trend}"
                  if hy_oas_bps is not None else
                  f"{credit_trend} (HYG {hyg_1m:+.1f}% / LQD {lqd_1m:+.1f}%)"),
    }
    regime_signals["Gold"] = {
        "signal": round(_clamp1(-gld_1m / 6.0), 2),
        "value": f"{gld_1m:+.1f}%",
        "label": "Safe-haven bid" if gld_1m > 2 else "Risk-on" if gld_1m < -1 else "Neutral",
    }
    regime_signals["Copper"] = {
        "signal": round(_clamp1(copper_1m / 5.0), 2),
        "value": f"{copper_1m:+.1f}%",
        "label": "Industrial demand" if copper_1m > 2 else "Demand weakness" if copper_1m < -2 else "Neutral",
    }
    if _btc_rel is not None:
        regime_signals["BTC/Gold"] = {
            "signal": round(_clamp1(_btc_rel / 12.0), 2),
            "value": f"{_btc_rel:+.1f}%",
            "label": ("Speculative appetite" if _btc_rel > 4 else
                      "Defensive rotation" if _btc_rel < -4 else "Neutral")
                     + f" — BTC {btc_1m:+.1f}% vs gold {gld_1m:+.1f}%",
        }
    regime_signals["DXY"] = {
        "signal": dxy_sig_norm,
        "value": f"{dxy_1m:+.1f}%",
        "label": "Strengthening" if dxy_1m > 0.5 else "Weakening" if dxy_1m < -0.5 else "Flat",
    }
    usdjpy_sig = round(_clamp1(usdjpy_1m / 4.0), 2)
    regime_signals["USD/JPY"] = {
        "signal": usdjpy_sig,
        "value": f"{usdjpy_1m:+.1f}%",
        "label": "JPY weakening (risk-on)" if usdjpy_1m > 1 else "JPY strengthening (risk-off)" if usdjpy_1m < -1 else "Neutral",
    }
    if geo_tension is not None:
        _geo_lbl = ("Low" if geo_tension < 0.15 else "Moderate" if geo_tension < 0.45 else
                    "Elevated" if geo_tension < 0.75 else "Severe")
        # Corroboration-aware display: headline tension only matters if the
        # market confirms it (VIX stressed / credit widening / havens bid).
        _geo_conf = (_core.get("detail", {}) or {}).get("geo_confirms")
        if geo_tension >= 0.30 and _geo_conf is not None:
            if _geo_conf == 0:
                _geo_lbl += " headlines \u00b7 markets calm"
            elif _geo_conf == 1:
                _geo_lbl += " \u00b7 partly confirmed"
            else:
                _geo_lbl += " \u00b7 market-confirmed"
        regime_signals["Geo Risk"] = {
            "signal": round(_comp["geo"] / 0.15, 2),
            "value": _geo_lbl,
            "label": (geo_top if geo_top else "No major geopolitical stress in 48h news flow"),
        }

    # ── Pillar composition (for the climate card UI) ────────────────────────
    regime_pillars = [
        {"key": "equity",     "label": "Equity Trend",    "contrib": _comp["equity"],     "max": 1.2},
        {"key": "volatility", "label": "Volatility",      "contrib": _comp["volatility"], "max": 1.2},
        {"key": "credit",     "label": "Credit",          "contrib": _comp["credit"],     "max": 1.0},
        {"key": "havens",     "label": "Havens & FX",     "contrib": _comp["havens"],     "max": 0.8},
        {"key": "growth",     "label": "Growth & Crypto", "contrib": _comp["growth"],     "max": 0.6},
        {"key": "geo",        "label": "Geopolitics",     "contrib": _comp["geo"],        "max": 0.15},
    ]

    # Clamp to -4..+4
    regime_score = round(max(-4.0, min(4.0, regime_score)), 1)

    # Normalise regime score -4..+4 to readable label (7-band scale for nuance)
    if regime_score >= 3.0:     regime_name = "Strong Risk-On";  regime_label = "Unambiguous risk appetite — equities, credit, commodities all aligned"
    elif regime_score >= 1.8:   regime_name = "Risk-On";         regime_label = "Broad risk appetite — equities, credit, commodities favoured"
    elif regime_score >= 0.7:   regime_name = "Lean Risk-On";    regime_label = "Mild risk appetite — equities and carry performing with some mixed signals"
    elif regime_score <= -3.0:  regime_name = "Strong Risk-Off"; regime_label = "Unambiguous de-risking — bonds, gold, USD, JPY all in demand"
    elif regime_score <= -1.8:  regime_name = "Risk-Off";        regime_label = "Broad de-risking — bonds, gold, USD favoured"
    elif regime_score <= -0.7:  regime_name = "Lean Risk-Off";   regime_label = "Mild risk aversion — defensive positioning building"
    else:                        regime_name = "Neutral";         regime_label = "No clear risk trend — mixed signals across assets"

    vix_level   = levels.get("VIX",  None)
    vix3m_level = levels.get("VIX3M", None)
    # Discrete term-structure read for the frontend (legacy contract:
    # >=1 Contango, <=-2 Inverted, else Flat)
    vix_ts      = 1 if (ts_ratio or 1.0) > 1.02 else -2 if (ts_ratio or 1.0) < 0.98 else 0

    # ── MACRO DASHBOARD: FRED enrichment ─────────────────────────────────
    macro_dashboard = {}
    rate_signal = {}
    rate_label = ""
    try:
        # Yield curve: T10Y2Y + T10Y3M (daily series — 130 days covers 6m)
        # FRED fallback: if FRED times out, compute spread from yfinance ^TNX (10Y) and derive 2Y
        yc_data   = fetch_fred_series("YLDCRV",  130)
        yc3m_data = fetch_fred_series("T10Y3M",  130)
        # ── yfinance fallback if FRED yields empty ────────────────────────────
        if not yc_data or len(yc_data) < 2:
            print("[yield_curve] FRED T10Y2Y empty — trying yfinance ^TNX/^IRX fallback")
            _tnx = _fetch_yf_yield_series("^TNX", 270)   # 10Y
            _irx = _fetch_yf_yield_series("^IRX", 270)   # 3M (x0.1 already applied)
            if _tnx and len(_tnx) >= 2 and _irx and len(_irx) >= 2:
                # Align by date and compute 10Y-3M spread as proxy for 10Y-2Y
                _irx_map = {x["date"]: x["value"] for x in _irx}
                yc_data = [{"date": x["date"], "value": round(x["value"] - _irx_map[x["date"]], 3)}
                           for x in _tnx if x["date"] in _irx_map]
                if not yc3m_data or len(yc3m_data) < 2:
                    yc3m_data = yc_data  # same spread since we used 3M already
                print(f"[yield_curve] yfinance fallback: {len(yc_data)} rows of 10Y-3M spread")
        if yc_data and len(yc_data) >= 2:
            yc_vals = [x["value"] for x in yc_data  if x.get("value") is not None]
            t3m_vals= [x["value"] for x in yc3m_data if x.get("value") is not None] if yc3m_data else []
            t10y2y  = yc_vals[-1]
            t10y3m  = t3m_vals[-1] if t3m_vals else None
            # 3m ≈ 65 trading days; 6m ≈ 130 trading days
            t10y2y_3m = yc_vals[-65]  if len(yc_vals)  >= 65  else yc_vals[0]
            t10y2y_6m = yc_vals[-130] if len(yc_vals)  >= 130 else yc_vals[0]
            t10y3m_3m = t3m_vals[-65] if len(t3m_vals) >= 65  else (t3m_vals[0] if t3m_vals else None)
            steepening_3m   = t10y2y - t10y2y_3m
            steepening_6m   = t10y2y - t10y2y_6m
            t10y3m_3m_chg   = round(t10y3m - t10y3m_3m, 3) if (t10y3m is not None and t10y3m_3m is not None) else None
            t10y2y_3m_chg   = round(steepening_3m, 3)
            # Primary spread for regime classification: prefer 10Y-3M
            primary = t10y3m if t10y3m is not None else t10y2y
            if primary >= 0.5:
                curve_regime = "Normal"
            elif primary >= 0:
                curve_regime = "Flat"
            elif primary > -0.5:
                curve_regime = "Slightly Inverted"
            else:
                curve_regime = "Inverted"
            if steepening_3m > 0.25:  curve_regime = "Steepening"
            # Bull/Bear steepener/flattener: determined by which end moves more
            # Bull: long end falls more (or short end rises less) → rates generally down
            # Bear: short end falls more (or long end rises more) → rates generally up
            yc_move_type = "Neutral"
            if abs(steepening_3m) > 0.05:
                # Use 3M T-bill as proxy for short end if available
                short_end_chg = t10y3m_3m_chg if t10y3m_3m_chg is not None else 0
                long_end_chg  = macro_dashboard.get("dgs10", {}).get("chg_3m", 0) or 0
                is_steepening = steepening_3m > 0
                if is_steepening:
                    # Steepening: long end rising more than short end = Bear Steepener
                    #             short end falling more = Bull Steepener
                    yc_move_type = "Bear Steepener" if long_end_chg > 0 and long_end_chg > (short_end_chg or 0) else "Bull Steepener"
                else:
                    # Flattening: short end rising more = Bear Flattener
                    #             long end falling more = Bull Flattener
                    yc_move_type = "Bear Flattener" if (short_end_chg or 0) > long_end_chg else "Bull Flattener"
            macro_dashboard["yield_curve"] = {
                "t10y2y":         round(t10y2y, 3),
                "t10y3m":         round(t10y3m, 3) if t10y3m is not None else None,
                "curve_regime":   curve_regime,
                "move_type":      yc_move_type,
                "steepening_3m":  round(steepening_3m, 3),
                "steepening_6m":  round(steepening_6m, 3),
                "t10y2y_3m_chg":  t10y2y_3m_chg,
                "t10y3m_3m_chg":  t10y3m_3m_chg,
            }
            regime_signals["Yield Curve"] = {
                "signal": round(min(1, max(-1, t10y2y / 2.0)), 2),
                "value": f"{t10y2y:+.2f}%",
                "label": curve_regime,
            }
    except Exception as _e: print(f"[macro_dashboard] {_e}")

    try:
        # Individual tenor yields for yield curve visualisation + DGS10 level/chg_3m.
        # IMPORTANT: fetch 270 periods FIRST so the cache is primed with full history.
        # Do NOT call fetch_fred_series("DGS10", 16) separately — it would cache only 16
        # rows and block the 270-period snapshot fetches that follow.
        # 270 trading days ≈ 13 calendar months, sufficient for 3M/6M/12M snapshots.
        dgs2_raw  = fetch_fred_series("DGS2",  270)
        dgs5_raw  = fetch_fred_series("DGS5",  270)
        dgs10_raw = fetch_fred_series("DGS10", 270)
        dgs30_raw = fetch_fred_series("DGS30", 270)
        dtb3_raw  = fetch_fred_series("DGS3MO", 270)  # 3-Month CMT — consistent with yield-curve-history
        # ── yfinance fallback for any empty FRED tenor series ──────────────────────
        if not dgs10_raw or len(dgs10_raw) < 2:
            dgs10_raw = _fetch_yf_yield_series("^TNX", 270)
            if dgs10_raw: print(f"[tenors] DGS10 → yfinance ^TNX ({len(dgs10_raw)} rows)")
        if not dgs5_raw or len(dgs5_raw) < 2:
            dgs5_raw = _fetch_yf_yield_series("^FVX", 270)
            if dgs5_raw: print(f"[tenors] DGS5 → yfinance ^FVX ({len(dgs5_raw)} rows)")
        if not dgs30_raw or len(dgs30_raw) < 2:
            dgs30_raw = _fetch_yf_yield_series("^TYX", 270)
            if dgs30_raw: print(f"[tenors] DGS30 → yfinance ^TYX ({len(dgs30_raw)} rows)")
        if not dtb3_raw or len(dtb3_raw) < 2:
            dtb3_raw = _fetch_yf_yield_series("^IRX", 270)
            if dtb3_raw: print(f"[tenors] DGS3MO → yfinance ^IRX ({len(dtb3_raw)} rows)")
        # 2Y: no direct yfinance symbol — estimate as (3M + 10Y) / 2 if FRED unavailable
        if (not dgs2_raw or len(dgs2_raw) < 2) and dtb3_raw and dgs10_raw:
            _irx_map = {x["date"]: x["value"] for x in (dtb3_raw or [])}
            dgs2_raw = [{"date": x["date"], "value": round((x["value"] + _irx_map[x["date"]]) / 2, 3)}
                       for x in dgs10_raw if x["date"] in _irx_map]
            if dgs2_raw: print(f"[tenors] DGS2 → estimated from (10Y+3M)/2 ({len(dgs2_raw)} rows)")

        def _tenor_snapshot(series, idx):
            """Return the yield at trading-day offset idx from end.
            idx=-1 → latest, -65 → ~3M, -130 → ~6M, -260 → ~12M.
            Falls back to the oldest available value if the series is too short."""
            vals = [x for x in series if x.get("value") is not None] if series else []
            if not vals: return None
            try: return round(vals[idx]["value"], 3)
            except IndexError: return round(vals[0]["value"], 3)

        tenors = {}

        # ── DGS10: level + 3-month change (replaces old 16-period block) ─────────
        if dgs10_raw and len(dgs10_raw) >= 2:
            dgs10_now = dgs10_raw[-1]["value"]
            dgs10_3m  = dgs10_raw[-65]["value"] if len(dgs10_raw) >= 65 else dgs10_raw[0]["value"]
            macro_dashboard["dgs10"] = {
                "level":  round(dgs10_now, 3),
                "chg_3m": round(dgs10_now - dgs10_3m, 3),
            }

        # ── 2-Year snapshots ──────────────────────────────────────────────────────
        if dgs2_raw:
            tenors["t2y"]     = _tenor_snapshot(dgs2_raw, -1)
            tenors["t2y_3m"]  = _tenor_snapshot(dgs2_raw, -65)
            tenors["t2y_6m"]  = _tenor_snapshot(dgs2_raw, -130)
            tenors["t2y_12m"] = _tenor_snapshot(dgs2_raw, -260)

        # ── 5-Year snapshots ──────────────────────────────────────────────────────
        if dgs5_raw:
            tenors["t5y"]     = _tenor_snapshot(dgs5_raw, -1)
            tenors["t5y_3m"]  = _tenor_snapshot(dgs5_raw, -65)
            tenors["t5y_6m"]  = _tenor_snapshot(dgs5_raw, -130)
            tenors["t5y_12m"] = _tenor_snapshot(dgs5_raw, -260)

        # ── 10-Year snapshots ─────────────────────────────────────────────────────
        if dgs10_raw:
            tenors["t10y_3m"]  = _tenor_snapshot(dgs10_raw, -65)
            tenors["t10y_6m"]  = _tenor_snapshot(dgs10_raw, -130)
            tenors["t10y_12m"] = _tenor_snapshot(dgs10_raw, -260)

        # ── 30-Year snapshots ─────────────────────────────────────────────────────
        if dgs30_raw:
            tenors["t30y"]     = _tenor_snapshot(dgs30_raw, -1)
            tenors["t30y_3m"]  = _tenor_snapshot(dgs30_raw, -65)
            tenors["t30y_6m"]  = _tenor_snapshot(dgs30_raw, -130)
            tenors["t30y_12m"] = _tenor_snapshot(dgs30_raw, -260)

        # ── 3-Month T-bill (short-end anchor for historical YC curves) ────────────
        if dtb3_raw:
            tenors["t3m"]     = _tenor_snapshot(dtb3_raw, -1)
            tenors["t3m_3m"]  = _tenor_snapshot(dtb3_raw, -65)
            tenors["t3m_6m"]  = _tenor_snapshot(dtb3_raw, -130)
            tenors["t3m_12m"] = _tenor_snapshot(dtb3_raw, -260)

        if tenors:
            macro_dashboard["yield_curve"] = {**macro_dashboard.get("yield_curve", {}), **tenors}
    except Exception as _te:
        print(f"[macro_dashboard tenor snapshots] {_te}")
        pass

    try:
        # Real yield: DFII10 (FRED 10Y TIPS) — with yfinance fallback since FRED blocked on Render
        ry_val = None
        ry_source = "fred"
        try:
            ry_data = fetch_fred_series("DFII10", 6)
            if ry_data and len(ry_data) >= 1:
                ry_val = ry_data[-1]["value"]
                ry_source = "fred"
        except Exception:
            ry_val = None
        # yfinance fallback: estimate real yield = 10Y nominal - CPI YoY
        if ry_val is None:
            try:
                import yfinance as yf
                # Use cached yfinance yield if available (already fetched for yield curve)
                t10_nominal = None
                try:
                    _tnx_cached = _YF_YIELD_CACHE.get("^TNX", {})
                    if _tnx_cached and not _tnx_cached.get("series", {}).empty:
                        t10_nominal = float(_tnx_cached["series"].iloc[-1])
                except Exception:
                    pass
                if t10_nominal is None:
                    t10_tick = yf.Ticker("^TNX")
                    t10_hist = t10_tick.history(period="5d")
                    if not t10_hist.empty:
                        t10_nominal = float(t10_hist["Close"].iloc[-1])
                # CPI YoY: read directly from FRED cache (already fetched in macro_all prefetch)
                # Use macro_dashboard dgs10 level as alternative 10Y source
                if t10_nominal is None:
                    _dgs10_cached = macro_dashboard.get("dgs10", {})
                    if _dgs10_cached and _dgs10_cached.get("level"):
                        t10_nominal = float(_dgs10_cached["level"])
                # CPI from macro_all scores cache
                cpi_yoy = None
                try:
                    _cpi_raw = fetch_fred_series("CPIAUCSL", 15)
                    if _cpi_raw and len(_cpi_raw) >= 13:
                        _cpi_now = _cpi_raw[-1]["value"]
                        _cpi_12m = _cpi_raw[-13]["value"]
                        if _cpi_12m and _cpi_12m > 0:
                            cpi_yoy = round((_cpi_now - _cpi_12m) / _cpi_12m * 100, 2)
                except Exception:
                    cpi_yoy = None
                if t10_nominal is not None and cpi_yoy is not None:
                    ry_val = round(t10_nominal - cpi_yoy, 3)
                    ry_source = "yfinance_est"
            except Exception:
                ry_val = None
        if ry_val is not None:
            ry_regime = "Restrictive" if ry_val > 2.0 else "Elevated" if ry_val > 1.0 else "Neutral" if ry_val > 0 else "Accommodative"
            macro_dashboard["real_yield"] = {"value": round(ry_val, 3), "regime": ry_regime, "source": ry_source}
    except Exception: pass

    try:
        # CPI
        cpi_data = fetch_fred_series("CPI", 15)
        if cpi_data and len(cpi_data) >= 13:
            cpi_now  = cpi_data[-1]["value"]
            cpi_prev = cpi_data[-13]["value"]
            cpi_yoy  = round((cpi_now / cpi_prev - 1) * 100, 2)
            cpi_mom  = round((cpi_now / cpi_data[-2]["value"] - 1) * 100, 3) if len(cpi_data) >= 2 else 0
            # trend: compare last 3m average vs prior 3m
            if len(cpi_data) >= 6:
                r1 = (cpi_data[-1]["value"] / cpi_data[-4]["value"] - 1) * 100
                r2 = (cpi_data[-4]["value"] / cpi_data[-7]["value"] - 1) * 100 if len(cpi_data) >= 7 else r1
                cpi_trend = "Rising" if r1 > r2 + 0.05 else "Falling" if r1 < r2 - 0.05 else "Stable"
            else:
                cpi_trend = "Stable"
            macro_dashboard["inflation"] = {
                "cpi_yoy": cpi_yoy,
                "cpi_mom": cpi_mom,
                "trend":   cpi_trend,
            }
    except Exception: pass

    try:
        # Fed balance sheet: WALCL (trillions) — fetch 160 weeks (~3 years) for chart + 12m changes
        walcl_data = fetch_fred_series("WALCL", 160)
        if walcl_data and len(walcl_data) >= 4:
            bs_vals = [x["value"] / 1e6 for x in walcl_data if x.get("value") is not None]
            bs_now  = bs_vals[-1]
            # 3m ≈ 13 weekly; 6m ≈ 26; 12m ≈ 52
            bs_3m   = bs_vals[-13] if len(bs_vals) >= 13 else bs_vals[0]
            bs_6m   = bs_vals[-26] if len(bs_vals) >= 26 else bs_vals[0]
            bs_12m  = bs_vals[-52] if len(bs_vals) >= 52 else bs_vals[0]
            bs_trend = "Expanding" if bs_now > bs_3m * 1.005 else "QT (Contracting)" if bs_now < bs_3m * 0.995 else "Flat / Stable"
            chg_3m     = round(bs_now - bs_3m, 2)
            chg_3m_pct = round((bs_now / bs_3m - 1.0) * 100, 2) if bs_3m else 0
            chg_6m_pct = round((bs_now / bs_6m - 1.0) * 100, 2) if bs_6m else 0
            chg_12m    = round(bs_now - bs_12m, 2)
            chg_12m_pct= round((bs_now / bs_12m - 1.0) * 100, 2) if bs_12m else 0
            chg_6m = round(bs_now - bs_6m, 2)
            # Build history array for frontend chart: date + value (T), sampled weekly
            bs_history = [
                {"d": x["date"], "v": round(x["value"] / 1e6, 3)}
                for x in walcl_data if x.get("value") is not None and x.get("date")
            ]
            macro_dashboard["fed_balance"] = {
                "level":       round(bs_now, 2),
                "trend":       bs_trend,
                "pct":         round((bs_now / 9.0) * 100, 1),  # % of QE peak $9T
                "chg_3m":      chg_3m,
                "chg_3m_pct":  chg_3m_pct,
                "chg_6m":      chg_6m,
                "chg_6m_pct":  chg_6m_pct,
                "chg_12m":     chg_12m,
                "chg_12m_pct": chg_12m_pct,
                "history":     bs_history,
            }
    except Exception: pass

    try:
        # Credit: use HYG/LQD spread + level
        hyg_level = levels.get("HYG", None)
        lqd_level = levels.get("LQD", None)
        macro_dashboard["credit"] = {
            "hy_trend": credit_trend,
            "hy_spread_sig": round(spread_sig, 2),
            "hyg_1m": round(hyg_1m, 2),
            "lqd_1m": round(lqd_1m, 2),
            # FRED BAML OAS levels — needed by frontend credit panel
            "hy_oas": hy_oas_bps,          # non-null triggers OAS panel in frontend
            "hy_oas_bps": hy_oas_bps,
            "ig_oas_bps": ig_oas_bps,
            "hy_score": round(hy_oas_score, 2),
            "hy_delta_4w": hy_delta_4w,
            "hy_delta_3m": hy_delta_3m,
            "hy_delta_6m": hy_delta_6m,
            "ig_delta_4w": ig_delta_4w,
            "ig_delta_3m": ig_delta_3m,
            "ig_delta_6m": ig_delta_6m,
            "hy_ig_ratio": hy_ig_ratio,
            "hy_pct": hy_pct,
            "hy_pct_min": hy_pct_min,
            "hy_pct_max": hy_pct_max,
            "ig_pct": ig_pct,
            "ig_pct_min": ig_pct_min,
            "ig_pct_max": ig_pct_max,
        }
    except Exception: pass

    try:
        # Labour market dashboard — NFP, UNRATE, ICSA, JOLTS
        # All raw scores are -2..+2 (surprise_score output); convert to 0-10 via *1.25+5
        _lab = {}
        # Fetch US macro components once (cached — no extra HTTP cost)
        # compute_macro_all() is always called before compute_risk_regime() in get_all_scores()
        _us_macro = compute_macro_all()
        _us_comps = _us_macro.get("components", {})

        # NFP (PAYEMS): monthly level in thousands → compute MoM changes
        # Fetch 16 months for a stable 6m baseline (avoids BLS revision distortion)
        _nfp_raw = fetch_fred_series("NFP", 16)
        if _nfp_raw and len(_nfp_raw) >= 5:
            _nfp_vals = [x["value"] for x in _nfp_raw if x.get("value") is not None]
            _nfp_mom  = [_nfp_vals[i] - _nfp_vals[i-1] for i in range(1, len(_nfp_vals))]
            if len(_nfp_mom) >= 4:
                _lab["nfp_mom"]        = round(_nfp_mom[-1], 0)           # latest MoM gain (K)
                # 6m avg for stable expectation (resist BLS revision noise)
                _nfp_window = _nfp_mom[-7:-1]
                _lab["nfp_6m_avg"]     = round(sum(_nfp_window) / len(_nfp_window), 0) if _nfp_window else None
                # 3m avg for recent trend
                _lab["nfp_3m_avg"]     = round(sum(_nfp_mom[-4:-1]) / 3, 0)
                # Surprise vs 3m avg (closer to Wall Street consensus than 6m avg,
                # which gets diluted by stale winter prints after BLS revisions).
                # FF overlay (below) replaces this with the real market consensus when available.
                _nfp_exp = _lab["nfp_3m_avg"] or _lab["nfp_6m_avg"] or 0
                _nfp_surp = _nfp_mom[-1] - _nfp_exp
                _lab["nfp_surprise"]   = round(_nfp_surp, 0)
                _lab["nfp_surprise_label"] = (
                    "Strong Beat" if _nfp_surp > 40  else
                    "Beat"        if _nfp_surp > 15  else
                    "Strong Miss" if _nfp_surp < -40 else
                    "Miss"        if _nfp_surp < -15 else "In Line"
                )
                # Raw score from compute_macro_all components (JOBS is in category 'jobs')
                _us_macro = compute_macro_all()
                _us_comps = _us_macro.get("components", {})
                _raw_nfp = _us_comps.get("JOBS", {}).get("score", 0)
                _lab["nfp_score_10"]   = round(min(10, max(0, _raw_nfp * 1.25 + 5)), 1)
                _lab["nfp_date"]       = _nfp_raw[-1].get("date", "")[:7]

        # UNRATE: monthly level (%)
        _ur_raw = fetch_fred_series("UNEMP", 14)
        if _ur_raw and len(_ur_raw) >= 4:
            _ur_vals = [x["value"] for x in _ur_raw if x.get("value") is not None]
            if _ur_vals:
                _lab["unrate"]         = round(_ur_vals[-1], 2)
                _lab["unrate_prev"]    = round(_ur_vals[-2], 2) if len(_ur_vals) >= 2 else None
                _lab["unrate_3m_chg"]  = round(_ur_vals[-1] - _ur_vals[-4], 2) if len(_ur_vals) >= 4 else None
                # Trend: rising/falling/stable over last 3 months
                _ur_3 = _ur_vals[-4:-1]
                if _ur_3:
                    _ur_slope = _ur_vals[-1] - _ur_3[0]
                    _lab["unrate_trend"] = "Rising" if _ur_slope > 0.15 else "Falling" if _ur_slope < -0.15 else "Stable"
                _raw_ur = _us_comps.get("UNEMP", {}).get("score", 0) if '_us_comps' in dir() else 0
                _lab["unrate_score_10"] = round(min(10, max(0, _raw_ur * 1.25 + 5)), 1)
                _lab["unrate_date"]     = _ur_raw[-1].get("date", "")[:7]

        # ICSA: weekly initial claims (absolute level, not thousands — ICSA is already in persons)
        # Fetch 26 weeks (6m) for robust baseline
        _cl_raw = fetch_fred_series("CLAIMS", 26)
        if _cl_raw and len(_cl_raw) >= 6:
            _cl_vals = [x["value"] for x in _cl_raw if x.get("value") is not None]
            if _cl_vals:
                _lab["claims"]          = round(_cl_vals[-1])              # latest weekly (persons)
                _lab["claims_4w_avg"]   = round(sum(_cl_vals[-5:-1]) / 4) if len(_cl_vals) >= 5 else None
                _lab["claims_52w_avg"]  = round(sum(_cl_vals) / len(_cl_vals)) if _cl_vals else None
                _lab["claims_chg_4w"]   = round(_cl_vals[-1] - _cl_vals[-5]) if len(_cl_vals) >= 5 else None
                _raw_cl = _us_comps.get("CLAIMS", {}).get("score", 0) if '_us_comps' in dir() else 0
                _lab["claims_score_10"] = round(min(10, max(0, _raw_cl * 1.25 + 5)), 1)
                _lab["claims_date"]     = _cl_raw[-1].get("date", "")

        # JOLTS job openings (JTSJOL via alias "JOLTS") — monthly, thousands
        try:
            _jolts_raw = fetch_fred_series("JOLTS", 8)
            if _jolts_raw and len(_jolts_raw) >= 2:
                _j_vals = [x["value"] for x in _jolts_raw if x.get("value") is not None]
                if _j_vals:
                    _lab["jolts"]        = round(_j_vals[-1] / 1000, 2)   # millions
                    _lab["jolts_prev"]   = round(_j_vals[-2] / 1000, 2) if len(_j_vals) >= 2 else None
                    _lab["jolts_chg"]    = round((_j_vals[-1] - _j_vals[-2]) / 1000, 2) if len(_j_vals) >= 2 else None
                    _lab["jolts_6m_avg"] = round(sum(_j_vals[-7:-1]) / 6 / 1000, 2) if len(_j_vals) >= 7 else None
                    _lab["jolts_date"]   = _jolts_raw[-1].get("date", "")[:7]
        except Exception:
            pass

        # Composite jobs score (0-10) — average of 3 converted scores
        _composite_parts = [
            _lab.get("nfp_score_10"),
            _lab.get("unrate_score_10"),
            _lab.get("claims_score_10"),
        ]
        _parts_valid = [p for p in _composite_parts if p is not None]
        if _parts_valid:
            _lab["jobs_composite_10"] = round(sum(_parts_valid) / len(_parts_valid), 1)

        # Also store the raw jobs_avg (-2..+2) for internal use
        try:
            _lab["jobs_avg_raw"] = round(jobs_avg, 2)
        except Exception:
            pass

        # ── ForexFactory actual vs consensus surprise injection ──────────────────────
        # Fetch is cached for 4h; zero network cost on subsequent calls.
        # If FF is unreachable, we gracefully fall through and the frontend
        # shows FRED-based scores only.
        try:
            _ff_labour = _fetch_ff_labour_surprises()
            if _ff_labour and _ff_labour.get("n_events_found", 0) > 0:
                _lab["ff_labour"] = {
                    "scores":        _ff_labour.get("scores", {}),
                    "latest":        _ff_labour.get("latest", {}),
                    "releases":      _ff_labour.get("releases", {}),
                    "composite_ems": _ff_labour.get("composite_ems"),
                    "fetched_at":    _ff_labour.get("fetched_at"),
                    "n_events":      _ff_labour.get("n_events_found", 0),
                }
                # Override jobs_composite_10 with FF EMS score when available
                # Blend: 60% FF EMS + 40% FRED trailing (best of both worlds)
                _ff_ems = _ff_labour.get("composite_ems")
                if _ff_ems is not None and _lab.get("jobs_composite_10") is not None:
                    _lab["jobs_composite_10"] = round(
                        _ff_ems * 0.6 + _lab["jobs_composite_10"] * 0.4, 1
                    )
                elif _ff_ems is not None:
                    _lab["jobs_composite_10"] = _ff_ems

                # Inject per-metric FF scores to replace FRED trailing scores where available
                _ff_scores = _ff_labour.get("scores", {})
                _ff_latest = _ff_labour.get("latest", {})
                if "nfp" in _ff_scores and _ff_scores["nfp"] is not None:
                    _lab["nfp_score_10"] = _ff_scores["nfp"]
                    # Replace nfp_surprise with real FF consensus surprise
                    _nfp_lat = _ff_latest.get("nfp", {})
                    if _nfp_lat:
                        _lab["nfp_surprise"]       = _nfp_lat.get("surprise")
                        _lab["nfp_consensus"]      = _nfp_lat.get("forecast")
                        _lab["nfp_actual_ff"]      = _nfp_lat.get("actual")
                        _beat = _nfp_lat.get("beat")
                        _surp_val = _nfp_lat.get("surprise", 0) or 0
                        _lab["nfp_surprise_label"] = (
                            "Strong Beat" if _beat and abs(_surp_val) > 40 else
                            "Beat"        if _beat and abs(_surp_val) > 15 else
                            "Strong Miss" if not _beat and abs(_surp_val) > 40 else
                            "Miss"        if not _beat and abs(_surp_val) > 15 else "In Line"
                        )
                if "unrate" in _ff_scores and _ff_scores["unrate"] is not None:
                    _lab["unrate_score_10"] = _ff_scores["unrate"]
                    _ur_lat = _ff_latest.get("unrate", {})
                    if _ur_lat:
                        _lab["unrate_consensus"] = _ur_lat.get("forecast")
                        _lab["unrate_actual_ff"] = _ur_lat.get("actual")
                        _lab["unrate_surprise"]  = _ur_lat.get("surprise")
                        _lab["unrate_beat"]      = _ur_lat.get("beat")
                if "claims" in _ff_scores and _ff_scores["claims"] is not None:
                    _lab["claims_score_10"] = _ff_scores["claims"]
                    _cl_lat = _ff_latest.get("claims", {})
                    if _cl_lat:
                        _lab["claims_consensus"] = _cl_lat.get("forecast")
                        _lab["claims_actual_ff"] = _cl_lat.get("actual")
                        _lab["claims_surprise"]  = _cl_lat.get("surprise")
                        _lab["claims_beat"]      = _cl_lat.get("beat")
                if "jolts" in _ff_scores and _ff_scores["jolts"] is not None:
                    _jolts_lat = _ff_latest.get("jolts", {})
                    if _jolts_lat:
                        _lab["jolts_consensus"] = _jolts_lat.get("forecast")
                        _lab["jolts_actual_ff"] = _jolts_lat.get("actual")
                        _lab["jolts_surprise"]  = _jolts_lat.get("surprise")
                        _lab["jolts_beat"]      = _jolts_lat.get("beat")
                if "adp" in _ff_scores and _ff_scores["adp"] is not None:
                    _lab["adp_score_10"]  = _ff_scores["adp"]
                    _adp_lat = _ff_latest.get("adp", {})
                    if _adp_lat:
                        _lab["adp_consensus"] = _adp_lat.get("forecast")
                        _lab["adp_actual_ff"] = _adp_lat.get("actual")
                        _lab["adp_surprise"]  = _adp_lat.get("surprise")
                        _lab["adp_beat"]      = _adp_lat.get("beat")
                if "wages" in _ff_scores and _ff_scores["wages"] is not None:
                    _lab["wages_score_10"] = _ff_scores["wages"]
                    _wg_lat = _ff_latest.get("wages", {})
                    if _wg_lat:
                        _lab["wages_consensus"] = _wg_lat.get("forecast")
                        _lab["wages_actual_ff"] = _wg_lat.get("actual")
                        _lab["wages_surprise"]  = _wg_lat.get("surprise")
                        _lab["wages_beat"]      = _wg_lat.get("beat")
        except Exception as _ff_e:
            print(f"[FF Labour] Injection error (non-fatal): {_ff_e}")

        if _lab:
            macro_dashboard["labour"] = _lab

    except Exception as e:
        print(f"Labour dashboard error: {e}")

    try:
        # Macro composites: equity + bond + commodity etc.
        # Each asset-class dict stores:
        #   composite_10 : 0-10 overall macro score for this asset class
        #   credit       : signed contribution from credit spreads (+ = tailwind)
        #   yield_curve  : signed contribution from yield curve shape
        #   fed_bs       : signed contribution from Fed balance sheet trend
        #   inflation    : signed contribution from CPI / inflation level
        #   real_yield   : signed contribution from TIPS real yield
        # All contributions are asset-class-oriented: positive = tailwind for that asset.
        # Contributions are used by the frontend Macro Dashboard waterfall bar chart.
        #
        # Signal ranges (for normalisation reference):
        #   credit_sig_norm : -1..+1  (+1 = tight spreads / risk-on)
        #   dxy_sig_norm    : -1..+1  (+1 = USD weakened = risk-on for risk assets)
        #   t10y2y          : ~-2..+2 (+ve = normal/steepening, -ve = inverted)
        #   bs_chg3m_pct    : ~-3..+3 (+ve = QE/expanding, -ve = QT/contracting)
        #   cpi_yoy         : ~0..8   (neutral anchor = 2%)
        #   ry_val (DFII10) : ~-1..+3 (neutral anchor = 1%; positive = restrictive)

        # Extract macro sub-signals safely (all can be None if FRED fetch failed)
        _mc_ry_val    = (macro_dashboard.get("real_yield")  or {}).get("value", None)
        _mc_bs_pct    = (macro_dashboard.get("fed_balance") or {}).get("chg_3m_pct", None)
        _mc_t10y2y    = (macro_dashboard.get("yield_curve") or {}).get("t10y2y", None)
        _mc_cpi       = (macro_dashboard.get("inflation")   or {}).get("cpi_yoy", None)

        # Use 0.0 when data unavailable — frontend filters out zero contributions naturally
        _s_ry   = float(_mc_ry_val) if _mc_ry_val is not None else 0.0   # real yield level
        _s_bs   = float(_mc_bs_pct) if _mc_bs_pct is not None else 0.0   # BS 3m % chg
        _s_yc   = float(_mc_t10y2y) if _mc_t10y2y is not None else 0.0   # 10Y-2Y spread
        _s_cpi  = float(_mc_cpi)    if _mc_cpi    is not None else 0.0   # CPI YoY
        _s_cr   = float(credit_sig_norm)                                   # credit signal

        # Normalised intermediate values
        _n_ry  = max(-1.5, min(1.5, (_s_ry - 1.0) / 1.5))   # centred at 1%; +ve = restrictive
        _n_bs  = max(-1.0, min(1.0, _s_bs / 3.0))            # +ve = expanding (QE)
        _n_yc  = max(-1.0, min(1.0, _s_yc / 2.0))            # +ve = normal/steepening
        _n_cpi = max(-1.0, min(1.0, (_s_cpi - 2.0) / 4.0))  # centred at 2%; +ve = elevated

        eq_comp = round(5 + eq_raw * 1.5, 1)

        # ── EQUITY: risk-on positive; rate headwind when real yield elevated ──
        _eq_cr  = round(_s_cr  *  0.25, 3)   # tight spreads = tailwind
        _eq_yc  = round(_n_yc  *  0.20, 3)   # normal/steep curve = growth = tailwind
        _eq_bs  = round(_n_bs  *  0.20, 3)   # QE/expanding = tailwind; QT = headwind
        _eq_inf = round(-max(0.0, _n_cpi) * 0.15, 3)  # only penalise above-neutral inflation
        _eq_ry  = round(-_n_ry *  0.20, 3)   # high real yield = discount rate headwind

        # ── BOND (price-oriented): risk-off / curve inversion / QE = price tailwind ──
        # All signals inverted vs equity: risk-on environment = bond price headwind
        _bo_cr  = round(-_s_cr *  0.20, 3)   # tight credit = risk-on = bond price headwind
        _bo_yc  = round(-_n_yc *  0.25, 3)   # steepening = rising long yields = price headwind
        _bo_bs  = round(_n_bs  *  0.20, 3)   # QE buys treasuries = price tailwind
        _bo_inf = round(-_n_cpi * 0.25, 3)   # inflation erodes real return + forces hikes
        _bo_ry  = round(-_n_ry *  0.10, 3)   # high real yield = duration compression = headwind

        # ── GOLD: real yield dominant driver; safe-haven bid when risk-off ──
        _gc_cr  = round(-_s_cr *  0.08, 3)   # tight credit = risk-on = mild safe-haven headwind
        _gc_yc  = round(-_n_yc *  0.05, 3)   # near-irrelevant; captured by real yield
        _gc_bs  = round(_n_bs  *  0.20, 3)   # QE = debasement narrative = gold tailwind
        _gc_inf = 0.0                          # near-zero direct (fully captured by real yield)
        _gc_ry  = round(-_n_ry *  0.40, 3)   # DOMINANT: high real yield = gold headwind

        # ── COMMODITY: growth-driven; mild inflation tailwind; DXY channel ──
        _co_cr  = round(_s_cr  *  0.20, 3)   # tight credit = growth demand = tailwind
        _co_yc  = round(_n_yc  *  0.15, 3)   # steepening = growth = tailwind
        _co_bs  = round(_n_bs  *  0.10, 3)   # liquidity mild tailwind
        _co_inf = round(_n_cpi *  0.15, 3)   # commodities drive CPI = mild tailwind when elevated
        _co_ry  = round(-(_s_ry / 4.0) * 0.10, 3)  # USD channel: high real yield = stronger USD = mild headwind

        # ── FX_FOREIGN (non-USD G10 pairs vs USD): carry + risk-on positive ──
        # FIX: was "5 - dxy_sig_norm * 3" which was INVERTED — dxy_sig_norm > 0 means USD weakened
        # = bullish for foreign FX, so composite must be 5 + dxy_sig_norm * 3
        _fx_cr  = round(_s_cr  *  0.20, 3)   # risk-on = carry demand = tailwind
        _fx_yc  = round(-_n_yc *  0.15, 3)   # steepening US curve = US rate advantage = headwind for FX
        _fx_bs  = round(_n_bs  *  0.15, 3)   # QE = weaker USD = FX tailwind
        _fx_inf = round(-_n_cpi * 0.10, 3)   # high US CPI = Fed hikes = USD strength = headwind
        _fx_ry  = round(-_n_ry *  0.20, 3)   # high US real yield = capital to USD = FX headwind

        # ── CRYPTO: liquidity dominant; real yield competition; risk appetite ──
        _cr_cr  = round(_s_cr  *  0.20, 3)   # risk appetite = tailwind
        _cr_yc  = round(_n_yc  *  0.10, 3)   # mild growth signal
        _cr_bs  = round(_n_bs  *  0.40, 3)   # DOMINANT: liquidity = speculative demand
        _cr_inf = round(-max(0.0, _n_cpi) * 0.10, 3)  # high inflation = Fed tightening = headwind
        _cr_ry  = round(-_n_ry *  0.30, 3)   # high real yield = competition for speculative capital

        # ── FX_USD (Dollar Index): inverse of fx_foreign ──
        _dx_cr  = round(-_s_cr *  0.20, 3)   # risk-on = capital leaves USD safe haven = headwind
        _dx_yc  = round(_n_yc  *  0.15, 3)   # steepening = higher US long yields = USD mild tailwind
        _dx_bs  = round(-_n_bs *  0.25, 3)   # QE = dollar supply dilution = headwind
        _dx_inf = round(_n_cpi *  0.20, 3)   # high US CPI = Fed hawkish = USD carry tailwind
        _dx_ry  = round(_n_ry  *  0.20, 3)   # high US real yield = capital inflow = USD tailwind

        def _c10(v): return max(0, min(10, v))

        macro_dashboard["macro_composites"] = {
            "equity": {
                "composite_10": _c10(eq_comp),
                "credit":       _eq_cr,
                "yield_curve":  _eq_yc,
                "fed_bs":       _eq_bs,
                "inflation":    _eq_inf,
                "real_yield":   _eq_ry,
            },
            "bond": {
                "composite_10": _c10(round(5 - eq_raw * 1.2, 1)),
                "credit":       _bo_cr,
                "yield_curve":  _bo_yc,
                "fed_bs":       _bo_bs,
                "inflation":    _bo_inf,
                "real_yield":   _bo_ry,
            },
            "gold": {
                "composite_10": _c10(round(5 - regime_score * 0.8, 1)),
                "credit":       _gc_cr,
                "yield_curve":  _gc_yc,
                "fed_bs":       _gc_bs,
                "inflation":    _gc_inf,
                "real_yield":   _gc_ry,
            },
            "commodity": {
                "composite_10": _c10(round(5 + regime_score * 0.6, 1)),
                "credit":       _co_cr,
                "yield_curve":  _co_yc,
                "fed_bs":       _co_bs,
                "inflation":    _co_inf,
                "real_yield":   _co_ry,
            },
            "fx_foreign": {
                # composite_10 corrected: dxy_sig_norm > 0 means USD weakened = bullish FX foreign
                "composite_10": _c10(round(5 + dxy_sig_norm * 3, 1)),
                "credit":       _fx_cr,
                "yield_curve":  _fx_yc,
                "fed_bs":       _fx_bs,
                "inflation":    _fx_inf,
                "real_yield":   _fx_ry,
            },
            "fx_usd": {
                "composite_10": _c10(round(5 - regime_score * 0.9, 1)),
                "credit":       _dx_cr,
                "yield_curve":  _dx_yc,
                "fed_bs":       _dx_bs,
                "inflation":    _dx_inf,
                "real_yield":   _dx_ry,
            },
            "crypto": {
                "composite_10": _c10(round(5 + regime_score * 1.0, 1)),
                "credit":       _cr_cr,
                "yield_curve":  _cr_yc,
                "fed_bs":       _cr_bs,
                "inflation":    _cr_inf,
                "real_yield":   _cr_ry,
            },
            # Legacy top-level scalars retained for any other consumers
            "credit":      round(credit_sig_norm, 2),
            "yield_curve": round((macro_dashboard.get("yield_curve", {}).get("t10y2y", 0) or 0) / 2, 2),
        }
    except Exception as _e: print(f"[macro_composites] {_e}")

    try:
        # ── Rate signal: Fed Funds Futures (CME ZQ contracts) ────────────────
        # Use market-implied path, not lagging FEDFUNDS historical series.
        # ZQ{month_code}{yr2}.CBT: price = 100 - implied EFFR for that month.
        # Cuts implied = (spot_implied - forward_implied) / 0.25  (in 25bp increments)
        _fff_months_code = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',
                            7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
        _today_d = date.today()
        _fff_results = {}  # (year, month) -> implied_rate
        for _i in range(0, 20):  # current month + 19 months out
            _m = (_today_d.month - 1 + _i) % 12 + 1
            _y = _today_d.year + ((_today_d.month - 1 + _i) // 12)
            _tkr = f"ZQ{_fff_months_code[_m]}{str(_y)[-2:]}.CBT"
            try:
                _h = yf.Ticker(_tkr).history(period="5d")
                if not _h.empty:
                    _fff_results[(_y, _m)] = round(100.0 - float(_h["Close"].iloc[-1]), 4)
            except Exception:
                pass

        _fff_keys = sorted(_fff_results.keys())
        if len(_fff_keys) >= 2:
            _effr_spot    = _fff_results[_fff_keys[0]]   # current month implied
            _effr_12m     = _fff_results[_fff_keys[min(12, len(_fff_keys)-1)]]
            _effr_18m     = _fff_results[_fff_keys[min(18, len(_fff_keys)-1)]]
            # Cuts implied (positive = cuts, negative = hikes)
            cuts_12m      = round((_effr_spot - _effr_12m) / 0.25, 2)
            cuts_18m      = round((_effr_spot - _effr_18m) / 0.25, 2)
            # rate_norm: 0 = tight/hiking, 1 = loose/cutting
            # 4+ cuts priced = fully easing; -2+ hikes = fully tightening
            rate_norm_val = round(min(1.0, max(0.0, 0.5 + cuts_12m * 0.10)), 2)
            # Label: based on FORWARD-LOOKING implied path
            if cuts_12m >= 1.5:      # ≥ 1.5 cuts (37.5bp) priced in 12m
                rate_label = "Easing"
            elif cuts_12m <= -1.5:   # ≥ 1.5 hikes priced
                rate_label = "Tightening"
            else:
                rate_label = "On Hold"
            # Fallback spot EFFR from FEDFUNDS FRED if futures spot looks wrong
            _effr_fred = None
            try:
                _fred_ff = fetch_fred_series("FEDFUNDS", 3)
                if _fred_ff:
                    _effr_fred = _fred_ff[-1]["value"]
            except Exception:
                pass
            effr_val = _effr_fred if _effr_fred is not None else _effr_spot
            # Build full monthly path array for frontend step-chart
            _path_monthly = []
            for _pk in _fff_keys[:18]:
                _py, _pm = _pk
                _implied = _fff_results[_pk]
                # Change vs spot (positive = hike, negative = cut)
                _delta_bp = round((_implied - _effr_spot) * 100, 0)
                _path_monthly.append({
                    "year": _py, "month": _pm,
                    "rate": round(_implied, 4),
                    "delta_bp": _delta_bp,
                })

            # Hike probability at next FOMC meeting
            # Use month 1 vs month 2 implied rates; each 25bp = one hike
            # Probability = fractional 25bp move above current rate
            _hike_prob_next = 0.0
            _cut_prob_next  = 0.0
            if len(_fff_keys) >= 2:
                _next_delta = _fff_results[_fff_keys[1]] - _effr_spot
                if _next_delta < 0:
                    # Cut expected
                    _cut_prob_next = round(min(100, abs(_next_delta) / 0.25 * 100), 1)
                elif _next_delta > 0:
                    # Hike expected
                    _hike_prob_next = round(min(100, _next_delta / 0.25 * 100), 1)

            # Total bp change across 12 months (negative = cuts)
            _total_bp_12m = round((_effr_12m - _effr_spot) * 100, 1)

            rate_signal = {
                "effr":           round(effr_val, 3),
                "effr_spot":      round(_effr_spot, 3),
                "effr_12m":       round(_effr_12m, 3),
                "effr_18m":       round(_effr_18m, 3),
                "cuts_12m":       cuts_12m,
                "cuts_18m":       cuts_18m,
                "rate_norm":      rate_norm_val,
                "source":         "fff",  # fed funds futures
                "path_monthly":   _path_monthly,
                "hike_prob_next": _hike_prob_next,
                "cut_prob_next":  _cut_prob_next,
                "total_bp_12m":   _total_bp_12m,
            }
            print(f"[rate_signal] FFF: spot={_effr_spot}% 12m={_effr_12m}% "
                  f"cuts_12m={cuts_12m} hike_prob={_hike_prob_next}% cut_prob={_cut_prob_next}% "
                  f"total_bp_12m={_total_bp_12m}bp label={rate_label}")
        else:
            # Fallback: FEDFUNDS historical (lagging but better than nothing)
            _fred_ff = fetch_fred_series("FEDFUNDS", 6)
            if _fred_ff and len(_fred_ff) >= 1:
                effr_val      = _fred_ff[-1]["value"]
                effr_3m       = _fred_ff[-3]["value"] if len(_fred_ff) >= 3 else effr_val
                effr_chg_3m   = effr_val - effr_3m
                cuts_12m      = round(max(-3.0, min(3.0, -effr_chg_3m * 4)), 2)
                rate_norm_val = round(min(1.0, max(0.0, 0.5 - effr_chg_3m * 0.5)), 2)
                rate_label    = ("Easing" if effr_chg_3m < -0.1 else
                                 "Tightening" if effr_chg_3m > 0.1 else "On Hold")
                rate_signal   = {"effr": round(effr_val,3), "cuts_12m": cuts_12m,
                                 "cuts_18m": round(cuts_12m*1.3,2), "rate_norm": rate_norm_val,
                                 "source": "fred_fallback"}
            else:
                rate_signal = {"effr": None, "cuts_12m": 0, "cuts_18m": 0, "rate_norm": 0.5}
                rate_label  = "On Hold"
    except Exception as _e:
        print(f"[rate_signal] error: {_e}")
        rate_signal = {"effr": None, "cuts_12m": 0, "cuts_18m": 0, "rate_norm": 0.5}
        rate_label  = "On Hold"

    _score_10 = round(max(0.0, min(10.0, (regime_score + 4.0) / 8.0 * 10.0)), 1)
    result = {
        "score":         regime_score,
        "score_10":      _score_10,
        "regime":        regime_name,
        "regime_label":  regime_label,
        "raw_score":     regime_score,
        "vix_level":     vix_level,
        "vix3m_level":   vix3m_level,
        "vix_ts":        vix_ts,
        "signals":       regime_signals,
        "pillars":       regime_pillars,
        "geo_tension":   (round(geo_tension, 2) if geo_tension is not None else None),
        "returns":       returns,
        "levels":        levels,
        "rate_signal":   rate_signal,
        "rate_label":    rate_label,
        "intl_rates":    _compute_intl_rates(),
        "macro_dashboard": macro_dashboard,
    }
    RISK_REGIME_CACHE["data"] = result
    RISK_REGIME_CACHE["time"] = now
    return result


def get_regime_score_for_market(market_id: str, regime: dict, news_sentiment: float = None) -> dict:
    """
    Per-asset regime/climate scoring.

    Inputs
    ------
    raw_score   : global risk-on/off composite, range -4..+4
                  positive = risk-on, negative = risk-off
    us_rate_adj : Fed rate-path signal, range -2..+2
                  positive = hiking/tight (bearish for rate-sensitive assets)
                  negative = cutting/easing (bullish for rate-sensitive assets)
                  derived from EFFR trend + bias flag in rate_signal

    All scores are mapped to 0-10 (5.0 = neutral).

    Architecture notes
    ------------------
    - Rate path now feeds bonds, equities, gold, DX directly (was missing before)
    - Bond multipliers are duration-graduated (ZT < ZF < ZN < ZB)
    - 6J (JPY) and 6S (CHF) risk polarity is INVERTED — both are safe-haven currencies
    - NG separated from oil group: near-zero risk-regime correlation
    - PA multiplier reduced: supply-driven, low macro beta
    - FX pair normalization extended to full [0, 10]
    - Neutral dead zone added to all label thresholds
    - Crypto score clamped to [0, 10]
    - news_sentiment (0-10) blended in at 30% as soft overlay
    """
    raw_score = regime.get("score", 0.0)
    m = market_id.upper()

    # ── US rate-path signal ─────────────────────────────────────────────
    # us_rate_adj > 0 → Fed hiking/tight → bearish for bonds, equities, gold; bullish for USD
    # us_rate_adj < 0 → Fed cutting/easy → bullish for bonds, equities, gold; bearish for USD
    #
    # PRIMARY: Fed Funds Futures implied path (cuts_12m from rate_signal).
    # cuts_12m > 0 → market expects cuts → us_rate_adj < 0 (easing)
    # cuts_12m < 0 → market expects hikes → us_rate_adj > 0 (tightening)
    # Scale: 4 cuts (100bp) ≈ rate_adj -2.0; 4 hikes ≈ +2.0
    #
    # SECONDARY: backward-looking intl_rates trend (captures what the Fed HAS done,
    # which still matters for rate-differential pricing in FX / bonds).
    # Blend: 60% futures-implied (forward-looking) + 40% historical trend.
    rate_signal = regime.get("rate_signal", {})
    intl_rates  = regime.get("intl_rates",  {})
    us_ir       = intl_rates.get("US", {})

    # Forward-looking: futures-implied cuts/hikes (range -2..+2)
    _cuts_12m   = rate_signal.get("cuts_12m", 0) or 0
    _fff_adj    = max(-2.0, min(2.0, -_cuts_12m * 0.5))  # 4 cuts → -2.0, 4 hikes → +2.0

    # Backward-looking: historical EFFR trend
    _us_t6   = us_ir.get("trend_6m") or 0.0
    _us_t3   = us_ir.get("trend_3m") or 0.0
    _us_bias = us_ir.get("bias") or 0
    _us_raw  = (_us_t6 or 0.0) * 0.65 + (_us_t3 or 0.0) * 0.35 + (_us_bias or 0) * 0.25
    _hist_adj = max(-2.0, min(2.0, _us_raw / 0.3))

    # Blend: 60% forward (futures) + 40% backward (history)
    if rate_signal.get("source") == "fff":
        us_rate_adj = round(0.60 * _fff_adj + 0.40 * _hist_adj, 3)
    else:
        # No futures data — fall back to historical only
        us_rate_adj = _hist_adj
    us_rate_adj = max(-2.0, min(2.0, us_rate_adj))

    # ── Cross-asset signals from live regime data (no new API calls) ───────
    # DXY 1m return: USD strengthening = headwind for gold/copper/grains/crypto.
    # Normalised: 5% monthly DXY move = ±1.0; typical range ±0.3.
    _returns    = regime.get("returns", {})
    _macro_dash = regime.get("macro_dashboard", {})
    _dxy_1m     = (_returns.get("DXY") or {}).get("return_1m", 0.0) or 0.0
    _dxy_sig    = max(-1.0, min(1.0, _dxy_1m / 5.0))
    # _dxy_sig < 0 = USD weakening = bullish for gold, copper, grains

    # DFII10 10Y TIPS real yield (already fetched from FRED in compute_risk_regime).
    # _ry_adj centred at 1% (historical gold neutral level).
    # _ry_adj > 0 = real rates below neutral = gold tailwind
    # _ry_adj < 0 = real rates above neutral = gold headwind
    _ry_val  = (_macro_dash.get("real_yield") or {}).get("value", None)
    _ry_adj  = max(-2.0, min(2.0, -(float(_ry_val) - 1.0) / 1.5)) if _ry_val is not None else 0.0

    # Fed balance sheet (WALCL) 3m % change: expanding = liquidity tailwind for crypto.
    _bs_chg3m  = (_macro_dash.get("fed_balance") or {}).get("chg_3m_pct", 0.0) or 0.0
    _walcl_sig = max(-1.0, min(1.0, _bs_chg3m / 3.0))
    # _walcl_sig > 0 = QE/expanding = crypto tailwind; < 0 = QT = headwind

    # ── Helper: clamp to [0, 10] ──────────────────────────────────
    def _sc(v):
        return round(max(0.0, min(10.0, v)), 1)

    # Return values (set per-branch)
    rate_score = 0.0
    rate_label = ""

    # ════════════════════════════════════════════════════════════════════
    # BONDS
    # Risk-off → safe-haven bid (price up, yield down) — INVERSE of equity
    # Rate path is critical: hiking cycle (2022) can produce risk-off WITH bond selloff
    # Duration graduated: ZT most rate-sensitive; ZB most regime-sensitive
    # ════════════════════════════════════════════════════════════════════
    if m in ("ZB", "ZN", "ZF", "ZT", "R"):
        # Duration-graduated risk sensitivity + rate-path adjustment
        # us_rate_adj > 0 (hiking) → headwind for bond prices (yields rising)
        # Risk component: risk-off (raw_score < 0) → bond bullish
        if m == "ZT":
            # 2Y: dominated by rate path, weakest safe-haven characteristic
            score_raw = -raw_score * 0.6 + us_rate_adj * (-1.00) + 5.0
            rate_label = "Rate-path dominant"
        elif m == "ZF":
            score_raw = -raw_score * 0.9 + us_rate_adj * (-0.80) + 5.0
            rate_label = "Belly: rate + risk balanced"
        elif m == "ZN":
            score_raw = -raw_score * 1.25 + us_rate_adj * (-0.60) + 5.0
            rate_label = "10Y benchmark"
        elif m == "ZB":
            # 30Y: highest duration, most regime-sensitive
            score_raw = -raw_score * 1.50 + us_rate_adj * (-0.50) + 5.0
            rate_label = "Long duration: regime dominant"
        elif m == "R":
            # Long Gilt: BoE-driven; use BoE rate signal if available
            boe = intl_rates.get("BOE", {})
            _boe_t6 = boe.get("trend_6m") or 0.0
            _boe_t3 = boe.get("trend_3m") or 0.0
            _boe_b  = boe.get("bias") or 0
            _boe_adj = max(-2.0, min(2.0, ((_boe_t6 or 0.0) * 0.65 + (_boe_t3 or 0.0) * 0.35 + (_boe_b or 0) * 0.25) / 0.3))
            score_raw = -raw_score * 1.25 + _boe_adj * (-0.55) + 5.0
            rate_label = "BoE rate path"
            rate_score = round(_boe_adj, 2)
        normalized = _sc(score_raw)
        _live = regime.get("rate_label", "")
        _easing_now = us_rate_adj < -0.3 and _live == "Easing"
        _hiking_now = us_rate_adj > 0.5  or _live == "Tightening"
        if raw_score < -0.5 and not _hiking_now:
            label = "Risk-off + Easing (Bond Bullish)" if _easing_now else "Risk-off (Bond Bullish)"
        elif raw_score < -0.5 and _hiking_now:
            label = "Risk-off but Hiking (Mixed)"
        elif raw_score > 0.5 and _hiking_now:
            label = "Risk-on + Hiking (Bond Bearish)"
        elif raw_score > 0.5:
            label = "Risk-on (Bond Bearish)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # EQUITIES
    # Risk-on = bullish; rate hikes compress multiples even in risk-on
    # NQ (high-duration growth) penalised more by hikes than YM (value)
    # ════════════════════════════════════════════════════════════════════
    elif m in ("Z", "ES", "NQ", "YM", "RTY", "NKD"):
        if m == "NQ":
            # High-duration: richest multiples, most rate-sensitive.
            # Empirical equity duration: NQ -7.94 vs SPX -4.54 (ratio 1.75x).
            # Rate coeff raised -0.70 -> -0.80 (= ES -0.45 * 1.75 rounded).
            # 2022: NQ -33% vs SPX -20% vs YM -9% confirms this differentiation.
            score_raw = raw_score * 1.10 + us_rate_adj * (-0.80) + 5.0
        elif m == "YM":
            # Value/dividend: lower duration, less rate-sensitive
            score_raw = raw_score * 1.30 + us_rate_adj * (-0.30) + 5.0
        elif m == "RTY":
            # Small caps: high floating-rate debt (38-45% of total vs 6-9% for SPX).
            # The floating-rate damage operates with a 12-24m lag, not immediate shock.
            # 2022: RTY fell same as ES (-20.5% vs -19.4%) — lag confirmed.
            # Rate sensitivity reduced -0.60 -> -0.45 to reflect this timing.
            # Rate cuts still provide a dual tailwind (sentiment + eventual debt relief).
            score_raw = raw_score * 1.20 + us_rate_adj * (-0.45) + 5.0
        else:
            # ES, Z (FTSE), NKD: standard equity
            score_raw = raw_score * 1.20 + us_rate_adj * (-0.45) + 5.0
        normalized = _sc(score_raw)
        _live = regime.get("rate_label", "")
        _easing_now = us_rate_adj < -0.3 and _live == "Easing"
        _hiking_now = us_rate_adj > 0.5  or _live == "Tightening"
        if raw_score > 0.5 and _easing_now:
            label = "Risk-on + Easing (Equity Bullish)"
        elif raw_score > 0.5 and _hiking_now:
            label = "Risk-on but Rate Headwind"
        elif raw_score > 1.8:
            label = "Risk-on (Equity Bullish)"
        elif raw_score > 0.5:
            label = "Lean Risk-on (Mild Equity Tailwind)"
        elif raw_score < -1.8:
            label = "Risk-off (Defensive)"
        elif raw_score < -0.5:
            label = "Lean Risk-off (Caution)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # GOLD
    # Primary drivers: (1) real rates — TIPS yield, (2) risk-off safe-haven
    # Real rate = nominal rate - inflation expectations
    # us_rate_adj captures the nominal rate path; used as proxy for real rate direction
    # Gold failed to rally in 2022 risk-off because real rates spiked +250bp
    # ════════════════════════════════════════════════════════════════════
    elif m == "GC":
        # Gold scoring uses three inputs:
        #   1. Risk regime (dampened -0.45): risk-on is bearish for gold (safe-haven reversal)
        #      but gold is NOT purely a safe-haven — dampened so other channels dominate.
        #   2. TIPS 10Y real yield (_ry_adj, 0.65 weight): primary gold driver.
        #      100bps real yield rise -> ~18% gold fall (PIMCO empirical). Replaces the
        #      nominal rate proxy (us_rate_adj) which mixed real rates + inflation expectations.
        #      _ry_adj < 0 = real rates elevated/rising = headwind; > 0 = accommodative.
        #   3. DXY 1m return (_dxy_sig, -0.25 weight): USD strength is bearish gold.
        #      WGC two-factor model (TIPS + DXY) achieves R2=0.85 (2007-2020).
        #      Partially orthogonal to TIPS: captures de-dollarisation / CB demand flows.
        # Fallback: if _ry_adj unavailable (FRED outage), blend in us_rate_adj at half weight.
        if _ry_adj != 0.0:
            score_raw = -raw_score * 0.45 + _ry_adj * 0.65 + _dxy_sig * (-0.25) + 5.0
        else:
            # TIPS data unavailable: fall back to nominal rate proxy at reduced weight
            score_raw = -raw_score * 0.50 + us_rate_adj * (-0.55) + _dxy_sig * (-0.25) + 5.0
        normalized = _sc(score_raw)
        # Labels use real yield signal when available, else nominal rate
        _ry_restrictive = (_ry_adj < -0.3) if _ry_adj != 0.0 else (us_rate_adj > 0.5)
        _ry_accommodative = (_ry_adj > 0.3) if _ry_adj != 0.0 else (us_rate_adj < -0.3 and regime.get("rate_label", "") == "Easing")
        _usd_weak = _dxy_sig < -0.1
        if _ry_accommodative and raw_score < -0.3:
            label = "Risk-off + Real Rate Tailwind (Gold Optimal)"
        elif _ry_accommodative and _usd_weak:
            label = "Real Rate + USD Tailwind (Gold Bullish)"
        elif _ry_accommodative:
            label = "Real Rate Tailwind (Gold Bullish)"
        elif raw_score < -0.5 and not _ry_restrictive:
            label = "Risk-off (Safe-Haven Bid)"
        elif _ry_restrictive and _dxy_sig > 0.1:
            label = "Real Rates Elevated + USD Strong (Gold Headwind)"
        elif _ry_restrictive:
            label = "Real Rates Elevated (Gold Headwind)"
        elif raw_score > 0.5:
            label = "Risk-on (Gold Bearish)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # SILVER
    # ~61% industrial (solar PV, EVs, grid; up from 55% a decade ago) + ~39% precious.
    # Net: mild positive risk-on, heavily dampened. Rate sensitivity ~1.22x gold
    # but partially buffered by industrial demand during expansion.
    # Real yield added at 0.20 weight (vs 0.65 for gold): silver's ~39% precious
    # metal component means real yield matters but is dampened by industrial leg.
    # _ry_adj > 0 = real rates below 1% neutral = precious metal tailwind.
    # ════════════════════════════════════════════════════════════════════
    elif m == "SI":
        score_raw = raw_score * 0.45 + us_rate_adj * (-0.30) + _ry_adj * 0.20 + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5:
            label = "Risk-on (Industrial Demand)"
        elif raw_score < -0.5:
            label = "Risk-off (Industrial Drag > Haven Bid)"
        else:
            label = "Mixed (Industrial vs Haven)"

    # ════════════════════════════════════════════════════════════════════
    # DOLLAR INDEX (DX)
    # Risk-off: mild USD safe-haven bid (not always — depends on crisis origin)
    # Rate path: Fed hiking relative to peers = dominant medium-term driver
    # 2022: DXY +20% purely on Fed vs ECB/BoJ rate differential — risk neutral
    # ════════════════════════════════════════════════════════════════════
    elif m == "DX":
        # Risk-off: mild safe-haven (negative raw_score → USD up)
        # Hiking: positive rate_adj → USD up (yield differential)
        score_raw = -raw_score * 0.60 + us_rate_adj * 0.70 + 5.0
        normalized = _sc(score_raw)
        # Use live rate_label (FEDFUNDS-derived) to anchor the label correctly.
        # us_rate_adj can lag by 3-6m during Fed transitions (e.g. cut cycle → pause).
        _live_rate_lbl = regime.get("rate_label", "")  # "Easing", "On Hold", "Tightening"
        _fed_hiking  = us_rate_adj > 0.5  or _live_rate_lbl == "Tightening"
        _fed_easing  = us_rate_adj < -0.5 and _live_rate_lbl == "Easing"
        _fed_on_hold = _live_rate_lbl == "On Hold" or (not _fed_hiking and not _fed_easing)
        if _fed_hiking:
            label = "Fed Hiking (USD Bullish)"
        elif raw_score < -0.5 and _fed_on_hold:
            label = "Risk-off (Safe-Haven Bid)"
        elif _fed_easing:
            label = "Fed Easing (USD Headwind)"
        elif raw_score > 0.5:
            label = "Risk-on (USD Headwind)"
        elif _fed_on_hold:
            label = "Fed On Hold"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # FX PAIRS vs USD
    # Two-dimensional: global risk appetite + foreign CB rate differential
    # CRITICAL FIX: 6J (JPY) and 6S (CHF) are SAFE-HAVEN currencies —
    #   risk-on weakens them vs USD; their risk polarity is INVERTED
    # All other pairs: risk-on = foreign ccy strengthens vs USD
    # FX normalization extended to full [0, 10]
    # ════════════════════════════════════════════════════════════════════
    elif m in ("6E", "6B", "6A", "6J", "6C", "6N", "6S", "6M"):
        ccy_map  = {"6E": "ECB", "6B": "BOE", "6A": "RBA", "6C": "BOC",
                    "6J": "BOJ", "6N": "RBNZ", "6S": "SNB", "6M": "BANXICO"}
        cb_names = {"ECB": "ECB", "BOE": "BoE", "RBA": "RBA", "BOC": "BoC",
                    "BOJ": "BoJ", "RBNZ": "RBNZ", "SNB": "SNB", "BANXICO": "Banxico"}

        # Safe-haven currencies: risk-on WEAKENS them vs USD (inverted polarity)
        HAVEN_FX = {"6J", "6S"}

        foreign_cb = ccy_map.get(m)
        rate_score = 0.0
        rate_label = ""
        if foreign_cb and intl_rates.get(foreign_cb):
            cb_data = intl_rates[foreign_cb]
            cb_name = cb_names.get(foreign_cb, foreign_cb)
            t6   = cb_data.get("trend_6m") or 0.0
            t3   = cb_data.get("trend_3m") or 0.0
            bias = cb_data.get("bias") or 0
            raw_cb   = (t6 or 0.0) * 0.65 + (t3 or 0.0) * 0.35
            raw_cb_b = bias * 0.5
            SENSITIVITY = 0.25
            _t3_flat = abs(t3) < 0.05
            if _t3_flat:
                rate_label = f"{cb_name} Paused" if abs(t6) > 0.3 else f"{cb_name} Flat"
            elif t3 > 1.5:  rate_label = f"{cb_name} Tightening Cycle"
            elif t3 > 0.5:  rate_label = f"{cb_name} Hiking"
            elif t3 > 0.1:  rate_label = f"{cb_name} Tightening"
            elif t3 < -1.5: rate_label = f"{cb_name} Easing Cycle"
            elif t3 < -0.5: rate_label = f"{cb_name} Cutting"
            elif t3 < -0.1: rate_label = f"{cb_name} Easing"
            else:           rate_label = f"{cb_name} Flat"
            rate_score = max(-2.0, min(2.0, (raw_cb + raw_cb_b) / SENSITIVITY * 0.125))

        # Risk direction — INVERTED for JPY and CHF (safe-haven)
        if m in HAVEN_FX:
            risk_dir = -raw_score * 0.5   # risk-on → score down (ccy weakens)
        else:
            risk_dir = raw_score * 0.5    # risk-on → score up (ccy strengthens)

        rate_dir     = rate_score
        risk_contrib = risk_dir * 0.65 + rate_dir * 0.35

        # Normalize to [0, 10] using theoretical max of ±2.0
        normalized = _sc((risk_contrib / 2.0) * 5.0 + 5.0)
        label = rate_label or ("Risk-on" if raw_score > 0 else "Risk-off")

    # ════════════════════════════════════════════════════════════════════
    # ENERGY: CL, HO, RB (oil complex)
    # Risk-on = demand growth = bullish
    # High sensitivity: oil has strong global growth beta
    # ════════════════════════════════════════════════════════════════════
    elif m in ("CL", "B", "GO", "HO", "RB"):
        score_raw  = raw_score * 1.25 + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5:
            label = "Risk-on (Energy Demand)"
        elif raw_score < -0.5:
            label = "Risk-off (Demand Risk)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # NATURAL GAS (NG) — separated from oil
    # Near-zero correlation with risk-on/off regime
    # Driven by: weather/storage, LNG exports, Henry Hub dynamics
    # 2022: NG +200% in risk-off; 2023: NG -70% in neutral/risk-on
    # ════════════════════════════════════════════════════════════════════
    elif m == "NG":
        score_raw  = raw_score * 0.25 + 5.0
        normalized = _sc(score_raw)
        # Label is deliberately minimal — regime barely matters for NG
        label = "Low Regime Sensitivity (Weather/Storage Driven)"

    # ════════════════════════════════════════════════════════════════════
    # COPPER (HG) — industrial metal, highest macro beta in metals complex
    # Strong global growth / China construction beta
    # ════════════════════════════════════════════════════════════════════
    elif m == "HG":
        # Copper: highest macro beta in industrial metals + strong DXY sensitivity.
        # BIS study (444 months): copper DXY beta = -0.08*** (strongest among metals).
        # risk 1.10: global growth demand signal; DXY -0.40: USD-denominated commodity.
        # 2022: -30% driven by joint DXY surge + rate hikes + China PMI weakness.
        score_raw  = raw_score * 1.10 + _dxy_sig * (-0.40) + 5.0
        normalized = _sc(score_raw)
        _usd_headwind = _dxy_sig > 0.1
        _usd_tailwind = _dxy_sig < -0.1
        if raw_score > 0.5 and _usd_tailwind:
            label = "Risk-on + Weak USD (Copper Bullish)"
        elif raw_score > 0.5 and _usd_headwind:
            label = "Risk-on but USD Headwind"
        elif raw_score > 0.5:
            label = "Risk-on (Industrial Demand)"
        elif raw_score < -0.5 and _usd_headwind:
            label = "Risk-off + Strong USD (Copper Bearish)"
        elif raw_score < -0.5:
            label = "Risk-off (Growth Concern)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # PLATINUM (PL) — industrial + jewelry; moderate macro beta
    # Lower risk-on correlation than copper; fuel cell/hydrogen demand adds idiosyncratic risk
    # ════════════════════════════════════════════════════════════════════
    elif m == "PL":
        # PL: broader industrial demand than PA (jewelry + hydrogen/fuel cell + diesel auto).
        # Raised 0.65 -> 0.70 to reflect wider demand base vs PA.
        score_raw  = raw_score * 0.70 + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5:
            label = "Risk-on (Industrial / Jewelry / Fuel Cell)"
        elif raw_score < -0.5:
            label = "Risk-off (Demand Concern)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # PALLADIUM (PA) — auto catalytic (gasoline engines), Russia supply risk
    # Near-zero macro beta; supply shocks dominate
    # Regime score here is a very weak signal — primarily decorative
    # ════════════════════════════════════════════════════════════════════
    elif m == "PA":
        score_raw  = raw_score * 0.40 + 5.0
        normalized = _sc(score_raw)
        label = "Low Regime Sensitivity (Supply/Auto Driven)"

    # ════════════════════════════════════════════════════════════════════
    # GRAINS: ZC (corn), ZS (soybeans), ZW (wheat)
    # Mild risk-on correlation; USD direction is the dominant macro input
    # Strong USD → grain headwind (USD-denominated, EM buyer cost rises)
    # Rate hiking → stronger USD → grain headwind
    # ════════════════════════════════════════════════════════════════════
    elif m in ("ZC", "ZS", "ZW"):
        # Grains: mild risk-on beta + strong USD sensitivity.
        # REPLACED: rate-path proxy (us_rate_adj * -0.40) was logically sound but
        # empirically unreliable (2022: rate hikes AND grain prices rose together).
        # Direct DXY 1m return is more responsive and captures non-rate USD moves.
        # BIS data: soybeans have the strongest grain-DXY link (beta = -0.05***).
        # Note: ZS has additional China demand driver not modelled here (60% of
        # global soybean imports) — monitor as a future enhancement.
        score_raw  = raw_score * 0.35 + _dxy_sig * (-0.50) + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5 and _dxy_sig < -0.1:
            label = "Risk-on + Weak USD (Grain Supportive)"
        elif _dxy_sig > 0.15:
            label = "Strong USD (Grain Headwind)"
        elif _dxy_sig < -0.15:
            label = "Weak USD (Grain Tailwind)"
        elif raw_score > 0.5:
            label = "Mild Risk-on Tailwind"
        elif raw_score < -0.5:
            label = "Risk-off Headwind"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # SOFTS: KC (coffee), SB (sugar), CC (cocoa), CT (cotton)
    # Primarily supply/weather driven; macro background only
    # Mild risk-on + USD headwind (same logic as grains, lower sensitivity)
    # ════════════════════════════════════════════════════════════════════
    elif m in ("KC", "SB", "CC", "CT", "RC"):
        # Softs are overwhelmingly weather/supply driven. BIS data: coffee DXY beta
        # is statistically indistinguishable from zero. Rate term removed entirely.
        # Macro regime provides only a weak demand background signal.
        score_raw  = raw_score * 0.30 + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5:
            label = "Risk-on (Soft Demand)"
        elif raw_score < -0.5:
            label = "Risk-off (Demand Headwind)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # LIVESTOCK: LE (live cattle), HE (lean hogs), GF (feeder cattle)
    # Consumer demand signal; modest correlation with risk-on
    # Feed cost (corn) is a key input — not captured here
    # ════════════════════════════════════════════════════════════════════
    elif m in ("LE", "HE", "GF"):
        # Empirically: livestock has near-zero macro R2 improvement from adding macro vars.
        # Record highs in 2026 during elevated uncertainty confirm supply-cycle dominance.
        # Reduced from 0.70 to 0.45; regime provides weak directional background only.
        score_raw  = raw_score * 0.45 + 5.0
        normalized = _sc(score_raw)
        if raw_score > 0.5:
            label = "Risk-on (Mild Consumer Demand)"
        elif raw_score < -0.5:
            label = "Risk-off (Mild Demand Concern)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # CRYPTO: BTC, ETH
    # High beta to risk appetite; rate sensitivity secondary
    # BTC increasingly store-of-value; ETH more tech/risk-on correlated
    # Score clamped to [0, 10] — multiplier would otherwise breach bounds
    # ════════════════════════════════════════════════════════════════════
    elif m in ("BTC", "ETH"):
        if m == "BTC":
            # BTC: risk-on dominant. Empirical beta ~1.5x SPX (CoinMetrics, post-ETF).
            # Rate coeff reduced to -0.15: digital gold narrative empirically weak;
            # BTC-software correlation (0.68-0.78) dominates BTC-gold correlation (0.12-0.31).
            # WALCL +0.15: Fed balance sheet expansion = liquidity tailwind for BTC
            # (global M2 correlation R2=0.71-0.90 at 6-24m horizon, Macro Alf data).
            score_raw = raw_score * 1.50 + us_rate_adj * (-0.15) + _walcl_sig * 0.15 + 5.0
        else:
            # ETH: higher tech/growth beta (10-15% deeper drawdowns vs BTC confirmed).
            # Raised 1.50 -> 1.60; rate -0.30 supported by DeFi/growth channel.
            # WALCL +0.12: similar liquidity sensitivity to BTC but slightly lower.
            score_raw = raw_score * 1.60 + us_rate_adj * (-0.30) + _walcl_sig * 0.12 + 5.0
        normalized = _sc(score_raw)  # clamp handles 1.5*4+5=11 → 10
        if raw_score > 0.5:
            label = "Risk-on (Crypto Bullish)"
        elif raw_score < -0.5:
            label = "Risk-off (Crypto Bearish)"
        else:
            label = "Neutral"

    # ════════════════════════════════════════════════════════════════════
    # FX CROSS PAIRS (e.g. EURJPY, EURGBP, AUDJPY)
    # Direction depends on base/quote character
    # IMPORTANT: if EITHER leg is JPY or CHF, the risk polarity of that leg is inverted
    #   EURJPY: EUR risk-on (+), JPY safe-haven (risk-on WEAKENS JPY) → double risk-on
    #   EURGBP: both non-haven → undefined risk direction, near-zero sensitivity
    #   CHFJPY: both haven → undefined; use rate differential only
    # ════════════════════════════════════════════════════════════════════
    elif len(m) == 6 and m.isalpha():
        base_is_haven  = m[:3] in ("JPY", "CHF")
        quote_is_haven = m[3:] in ("JPY", "CHF")

        if base_is_haven and quote_is_haven:
            # Both haven (e.g. CHFJPY): risk regime undefined — use flat 5.0
            # Rate differential is the actual driver but not modeled per-cross here
            normalized = 5.0
            label = "Haven Cross (Regime Undefined — Rate Differential Driven)"
        elif base_is_haven:
            # e.g. JPYEUR: JPY base = risk-off → higher score when safe-haven bid
            # Score: risk-off (raw_score < 0) → positive for haven base
            score_raw  = -raw_score * 0.70 + 5.0
            normalized = _sc(score_raw)
            label = "Risk-off (Haven Base Bid)" if raw_score < -0.5 else "Risk-on (Haven Base Weakens)" if raw_score > 0.5 else "Neutral"
        elif quote_is_haven:
            # e.g. EURJPY: EUR (risk-on base) + JPY (haven quote weakens in risk-on)
            # Both legs amplify the risk-on signal → use higher multiplier
            score_raw  = raw_score * 1.00 + 5.0
            normalized = _sc(score_raw)
            label = "Risk-on (Carry)" if raw_score > 0.5 else "Risk-off (Safe-Haven Bid)" if raw_score < -0.5 else "Neutral"
        else:
            # Both non-haven (e.g. EURGBP, EURCAD): risk-on/off effect is minimal
            score_raw  = raw_score * 0.30 + 5.0
            normalized = _sc(score_raw)
            label = "Low Regime Sensitivity (Rate Differential Driven)" if abs(raw_score) < 1.0 else ("Mild Risk-on" if raw_score > 0 else "Mild Risk-off")

    # ════════════════════════════════════════════════════════════════════
    # DEFAULT fallback
    # ════════════════════════════════════════════════════════════════════
    else:
        score_raw  = raw_score * 0.75 + 5.0
        normalized = _sc(score_raw)
        label = "Risk-on" if raw_score > 0.5 else "Risk-off" if raw_score < -0.5 else "Neutral"

    # ── News sentiment overlay (30% within climate) ─────────────────────
    # Sonar-derived per-asset sentiment blended in as a soft overlay.
    # 70% mechanical signal, 30% news. Max influence ±1.5 pts on 0-10 scale.
    news_sentiment_score = None
    if news_sentiment is not None:
        raw_blended = 0.70 * normalized + 0.30 * float(news_sentiment)
        news_sentiment_score = round(float(news_sentiment), 1)
        normalized = _sc(raw_blended)

    return {
        "score":          normalized,
        "label":          label,
        "raw_regime":     raw_score,
        "us_rate_adj":    round(us_rate_adj, 2),
        "rate_score":     round(rate_score, 2),
        "rate_label":     rate_label,
        "news_sentiment": news_sentiment_score,
    }


# ============================================================
# NEWS CONTEXT
# ============================================================

FF_NEWS_CACHE: dict = {"data": None, "time": 0}
FF_NEWS_TTL = 3600 * 2   # 2 hours — news doesn't change that fast

# Keep NEWS_CACHE as alias so /api/scores path still works
NEWS_CACHE = FF_NEWS_CACHE
NEWS_CACHE_TTL = FF_NEWS_TTL

# Per-asset narrative cache (separate from news item cache)
NARR_CACHE: dict = {"data": None, "time": 0}
NARR_CACHE_TTL = 3600 * 2  # 2 hours, same as news

# ── Weekly consensus-outlook cache ────────────────────────────────────────────
# Reads the week's bank/broker outlooks, CTA/trend-fund commentary and news flow
# via a web-search-enabled Sonar call, and distils the CROWD VIEW per market:
# what the consensus believes, how one-sided it is, and the fade angle. This is
# the qualitative "read the outlooks" layer that feeds the Consensus-Fade panel.
# Refreshed weekly (fund managers publish weekly), cached to disk so a Render
# redeploy doesn't wipe it.
CONSENSUS_CACHE: dict = {"data": None, "time": 0}
CONSENSUS_CACHE_TTL = 3600 * 24 * 7          # 7 days
CONSENSUS_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consensus_cache.json")


def generate_asset_narratives(news_items: list) -> dict:
    """
    Given a list of FF news headlines, make one Sonar call that returns
    a per-asset narrative dict: {assetId: "1-2 sentence narrative"}.

    The narrative explains how the current macro/news backdrop relates
    to each asset — even if the asset isn't mentioned in the headlines.
    E.g. war headlines → GC narrative about geopolitical safe-haven bid.

    Returns {} on any error (graceful degradation).
    """
    import json as _json
    if not news_items:
        return {}

    # Build compact headline digest
    headlines = []
    for n in news_items[:15]:  # top 15 newest
        title = n.get("title", "")
        preview = n.get("preview", "")
        impact = n.get("impact", "medium").upper()
        if title:
            line = f"[{impact}] {title}"
            if preview:
                line += f" — {preview[:120]}"
            headlines.append(line)

    ASSET_LIST = [
        # Equities
        ("ES",  "S&P 500 futures (US large-cap equity index)"),
        ("NQ",  "NASDAQ 100 futures (US tech-heavy equity index)"),
        ("YM",  "Dow Jones futures (US blue-chip equity index)"),
        ("RTY", "Russell 2000 futures (US small-cap equity index)"),
        ("Z",   "FTSE 100 futures (UK equity index; ~70% revenues international)"),
        # Metals
        ("GC",  "Gold futures"),
        ("SI",  "Silver futures"),
        ("HG",  "Copper futures (global growth proxy)"),
        ("PL",  "Platinum futures (auto catalyst, hydrogen economy play)"),
        ("PA",  "Palladium futures (autocatalyst, EV transition headwind)"),
        # Energy
        ("CL",  "Crude Oil WTI futures"),
        ("B",   "Brent Crude Oil futures (ICE Europe; global benchmark)"),
        ("NG",  "Natural Gas futures (Henry Hub; weather/LNG sensitive)"),
        ("RB",  "RBOB Gasoline futures (driving season, crack spread)"),
        ("HO",  "Heating Oil futures (distillate, diesel demand proxy)"),
        ("G",   "Gas Oil futures (ICE Europe; European diesel/heating benchmark)"),
        # Bonds
        ("ZB",  "US 30Y T-Bond futures (most duration-sensitive)"),
        ("ZN",  "US 10Y T-Note futures (global rate benchmark)"),
        ("ZF",  "US 5Y T-Note futures (belly of curve; most Fed-path sensitive)"),
        ("ZT",  "US 2Y T-Note futures (purest Fed policy pricing instrument)"),
        ("R",   "UK Long Gilt futures (ICE Europe; BoE policy sensitive)"),
        # FX Majors
        ("6E",  "EUR/USD futures"),
        ("6J",  "Japanese Yen futures (USD/JPY inverse; safe-haven)"),
        ("6B",  "GBP/USD futures (British Pound; BoE sensitive)"),
        ("6A",  "AUD/USD futures (Australian Dollar; China/commodities proxy)"),
        ("6C",  "CAD/USD futures (Canadian Dollar; oil-linked)"),
        ("6N",  "NZD/USD futures (New Zealand Dollar; dairy/China proxy)"),
        ("6S",  "CHF/USD futures (Swiss Franc; safe-haven, SNB)"),
        ("6M",  "MXN/USD futures (Mexican Peso; nearshoring, Banxico, carry)"),
        ("DX",  "US Dollar Index futures"),
        # FX Crosses — note these are derived from major futures above
        ("EURJPY",  "EUR/JPY (risk-on carry: ECB vs BoJ divergence)"),
        ("EURGBP",  "EUR/GBP (ECB vs BoE; eurozone vs UK divergence)"),
        ("EURAUD",  "EUR/AUD (EU manufacturing vs Australian commodities)"),
        ("EURCAD",  "EUR/CAD (ECB vs BoC; EU growth vs oil)"),
        ("EURCHF",  "EUR/CHF (EU stress indicator; SNB floor history)"),
        ("EURNZD",  "EUR/NZD (ECB vs RBNZ; EU vs NZ rate differential)"),
        ("GBPJPY",  "GBP/JPY (high-beta carry; BoE vs BoJ)"),
        ("GBPAUD",  "GBP/AUD (UK services vs Australian commodities; risk appetite)"),
        ("GBPCAD",  "GBP/CAD (BoE vs BoC; UK vs Canadian growth)"),
        ("GBPCHF",  "GBP/CHF (risk-on vs safe-haven; BoE vs SNB)"),
        ("GBPNZD",  "GBP/NZD (BoE vs RBNZ; UK vs NZ rate differential)"),
        ("AUDJPY",  "AUD/JPY (quintessential risk-on/off cross; China/commodities vs yen)"),
        ("AUDCAD",  "AUD/CAD (iron ore vs oil; RBA vs BoC; twin commodity currencies)"),
        ("AUDCHF",  "AUD/CHF (pure risk barometer: carry vs safe-haven)"),
        ("AUDNZD",  "AUD/NZD (RBA vs RBNZ; iron ore vs dairy; trans-Tasman spread)"),
        ("CADJPY",  "CAD/JPY (oil vs yen safe-haven; BoC vs BoJ)"),
        ("CHFJPY",  "CHF/JPY (twin safe-havens; SNB vs BoJ ultra-loose policy)"),
        ("NZDCAD",  "NZD/CAD (dairy vs oil; RBNZ vs BoC)"),
        ("NZDJPY",  "NZD/JPY (carry trade; risk sentiment; China dairy linkage)"),
        # Agriculturals
        ("ZS",  "Soybean futures (China demand, La Nina/El Nino weather)"),
        ("ZC",  "Corn futures (ethanol demand, feed use, weather)"),
        ("ZW",  "Wheat futures (Black Sea supply, food security)"),
        ("CC",  "Cocoa futures (West Africa supply; El Nino weather)"),
        ("KC",  "Coffee Arabica futures (Brazil weather, demand)"),
        ("RC",  "Robusta Coffee futures (ICE; Vietnam/Indonesia supply)"),
        ("SB",  "Sugar No.11 futures (Brazil ethanol/sugar split, India)"),
        ("CT",  "Cotton No.2 futures (US acreage, China demand)"),
        # Livestock
        ("LE",  "Live Cattle futures (herd cycle, beef demand, feed costs)"),
        ("HE",  "Lean Hogs futures (China ASF, US pork exports)"),
        ("GF",  "Feeder Cattle futures (corn price linkage, drought, herd expansion)"),
        # Crypto
        ("BTC", "Bitcoin (halving cycle, institutional adoption, macro liquidity)"),
        ("ETH", "Ethereum (DeFi, staking yield, EIP-4844, institutional flows)"),
    ]

    asset_str = "\n".join(f"{aid}: {aname}" for aid, aname in ASSET_LIST)
    headline_str = "\n".join(headlines)

    prompt = (
        "You are the markets analyst for the BH Weather System, a swing-trading bias tool for "
        "2-4 week futures positions. Your commentary must speak the system's methodology, which is:\n"
        "  • Directional bias = a CONFLUENCE of the factors with real edge (risk regime, "
        "macro surprise, seasonality) — NOT a simple average. Factors with no clear view abstain.\n"
        "  • COT positioning votes at its natural sign: extreme commercial buying is a bull vote, "
        "extreme spec crowding is a bear vote, regardless of price trend. Extreme positioning that "
        "disagrees with price trend is a mean-reversion warning, not noise. Also frames the R/R setup: "
        "side WITH commercials against a crowded-spec extreme, tight stop, let winners run.\n"

        "  • Conviction is gated by REGIME: act in clean trends, stand aside in chop.\n"
        "  • Multi-timeframe: the long-term trend sets the stable bias; a short-term counter-move is "
        "an entry-watch zone, not a reversal.\n\n"
        "Below are the latest high/medium-impact financial news headlines from the last 48 hours.\n\n"
        f"HEADLINES:\n{headline_str}\n\n"
        "For each instrument below, produce TWO things:\n"
        "1. A SHORT 1-2 sentence analyst comment that adds NEWS / MACRO COLOR consistent with the "
        "methodology above. Explain how the current backdrop supports or challenges the read, and "
        "flag positioning setups in risk/reward terms (e.g. 'commercials absorbing the spec short — "
        "a contrarian long setup if price confirms'). Do NOT issue a blunt standalone buy/sell call "
        "that competes with the engine; CONTEXTUALISE. Use direct market language: 'bid', 'offered', "
        "'headwind', 'tailwind', 'under pressure', 'supported', 'crowded', 'squeeze risk'. Infer the "
        "effect even if the asset isn't named in the headlines.\n"
        "2. A news sentiment SCORE from -1.0 (most bearish) to +1.0 (most bullish), reflecting how the "
        "current NEWS backdrop alone affects that instrument. 0.0 = neutral/no clear news effect.\n\n"
        "Return ONLY valid JSON — NO markdown fences, NO code blocks, NO extra text before or after. "
        "Each key is the instrument ID, each value is an object with 'text' (string) and 'score' (float -1.0 to +1.0). "
        "Start your response with { and end with }. Example:\n"
        '{"ES": {"text": "...", "score": -0.4}, "GC": {"text": "...", "score": 0.8}}\n\n'
        f"INSTRUMENTS:\n{asset_str}"
    )

    try:
        api_key = os.environ.get("PPLX_API_KEY", "")
        if not api_key:
            return {}
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 6000,
                "temperature": 0.3,
            },
            timeout=45.0,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present (belt-and-braces)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0].strip()
        # Trim any stray text before first { or after last }
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end+1]
        result = _json.loads(raw)
        if not isinstance(result, dict):
            return {}
        # Normalise: handle both {k: "text"} (old) and {k: {text, score}} (new)
        out = {}
        for k, v in result.items():
            if isinstance(v, dict):
                text  = str(v.get("text", ""))
                score = v.get("score", None)
                if score is not None:
                    try:
                        score = max(-1.0, min(1.0, float(score)))
                    except (TypeError, ValueError):
                        score = None
                # Convert -1..+1 score to 0..10 scale for regime blending
                score_10 = round((score + 1.0) * 5.0, 1) if score is not None else None
                out[k] = {"text": text, "score": score, "score_10": score_10}
            else:
                out[k] = {"text": str(v), "score": None, "score_10": None}
        return out
    except Exception as e:
        print(f"[narr] generate_asset_narratives error: {e}", flush=True)
        return {}


def fetch_ff_news(hours_back: int = 48) -> list:
    """
    Fetch ForexFactory /news page, parse structured data-items JSON
    embedded in <news-block-component> tags. Returns high+medium impact
    items from the last N hours, sorted newest first.
    Uses httpx (already imported) — robust in threaded executor context.
    """
    import re as _re_news
    from html import unescape as _unescape
    now_ts = time.time()
    if FF_NEWS_CACHE["data"] is not None and (now_ts - FF_NEWS_CACHE["time"]) < FF_NEWS_TTL:
        return FF_NEWS_CACHE["data"]

    import datetime as _dt
    cutoff = now_ts - (hours_back * 3600)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.forexfactory.com/",
    }
    try:
        resp = httpx.get("https://www.forexfactory.com/news", headers=headers, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return []
        html = resp.text

        comps = _re_news.findall(r'<news-block-component[^>]+>', html, _re_news.DOTALL)
        seen_ids: set = set()
        results = []

        for comp in comps:
            # Skip non-market editorial sections
            title_m = _re_news.search(r'data-title="([^"]+)"', comp)
            comp_title = title_m.group(1) if title_m else ""
            if any(x in comp_title for x in ("Entertainment", "Educational", "Industry", "Technical", "Sponsored")):
                continue

            items_m = _re_news.search(r'data-items="(\[(?:[^"]|&quot;)*\])"', comp)
            if not items_m:
                continue
            try:
                items = json.loads(_unescape(items_m.group(1)))
            except Exception:
                continue

            for item in items:
                item_id = item.get("id")
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Accept any impact — FF doesn't tag all stories; untagged = include
                impact_raw = (item.get("impact") or "").strip().lower()
                impact = impact_raw if impact_raw in ("high", "medium") else "medium"

                dateline = item.get("dateline") or 0
                if dateline < cutoff:
                    continue

                preview_raw = item.get("preview") or ""
                preview = _re_news.sub(r'<[^>]+>', '', preview_raw).strip()
                if len(preview) > 200:
                    preview = preview[:200].rsplit(' ', 1)[0] + "\u2026"
                preview = _unescape(preview)
                # Normalise ALL CAPS preview text from some FF sources — convert to sentence case
                if preview and preview == preview.upper() and len(preview) > 10:
                    preview = preview.capitalize()

                # Human-friendly date/time labels
                pub_dt = _dt.datetime.fromtimestamp(dateline, _dt.timezone.utc)
                today_utc = _dt.datetime.now(_dt.timezone.utc).date()
                if pub_dt.date() == today_utc:
                    date_label = "Today"
                elif pub_dt.date() == today_utc - _dt.timedelta(days=1):
                    date_label = "Yesterday"
                else:
                    date_label = pub_dt.strftime("%a %b %d")
                time_label = pub_dt.strftime("%H:%M UTC")

                results.append({
                    "id":         item_id,
                    "title":      _unescape(item.get("title") or ""),
                    "preview":    preview,
                    "impact":     impact,
                    "source":     (item.get("source") or "").lstrip("@"),
                    "dateline":   dateline,
                    "date_label": date_label,
                    "time_label": time_label,
                    "url":        "https://www.forexfactory.com" + (item.get("url") or ""),
                    # frontend schema compat
                    "result":      "",
                    "is_upcoming": False,
                })

        results.sort(key=lambda x: x["dateline"], reverse=True)
        # Deduplicate by title (same story may appear in multiple components)
        seen_titles: set = set()
        deduped = []
        for r in results:
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                deduped.append(r)

        FF_NEWS_CACHE["data"] = deduped
        FF_NEWS_CACHE["time"] = now_ts
        return deduped
    except Exception as e:
        print(f"[news] fetch_ff_news error: {e}", flush=True)
        return []


# Global narrative cache (separate TTL — regen every 2h with fresh macro data)
GLOBAL_NARR_CACHE: dict = {"data": None, "time": 0}
GLOBAL_NARR_CACHE_TTL = 3600 * 2  # 2 hours


def generate_global_narrative(regime_data: dict, news_items: list) -> str | None:
    """
    Generate a 3–4 sentence global market narrative using Sonar.
    Has sight of: regime, macro dashboard (yield curve, credit, labour,
    inflation, rate path) and the latest news headlines.
    Returns the narrative string or None on failure.
    """
    import json as _json
    api_key = os.environ.get("PPLX_API_KEY", "")
    if not api_key:
        return None

    # ── Pull key macro fields from regime_data ──────────────────────────
    regime_name  = regime_data.get("regime", "Unknown")
    score_10     = regime_data.get("score_10", 5.0)
    md           = regime_data.get("macro_dashboard", {})
    yc           = md.get("yield_curve", {})
    cr           = md.get("credit", {})
    lb           = md.get("labour", {})
    fb           = md.get("fed_balance", {})
    ry           = md.get("real_yield", {})
    rs           = regime_data.get("rate_signal", {})
    mc           = regime_data.get("macro_composites", {})

    # Yield curve
    yc_spread    = yc.get("t10y2y")
    yc_regime    = yc.get("curve_regime", "Unknown")
    t2y          = yc.get("t2y")
    t10y         = yc.get("t10y")
    # Credit
    hy_oas       = cr.get("hy_oas_bps")
    ig_oas       = cr.get("ig_oas_bps")
    hy_trend     = cr.get("hy_trend", "")
    # Labour
    jobs_comp    = lb.get("jobs_composite_10")
    nfp_mom      = lb.get("nfp_mom")
    nfp_surp     = lb.get("nfp_surprise_label", "")
    unrate       = lb.get("unrate")
    # Fed balance / rates
    fb_level     = fb.get("level")
    fb_trend     = fb.get("trend", "")
    effr         = rs.get("effr")
    cuts_12m     = rs.get("cuts_12m", 0)
    rate_label   = regime_data.get("rate_label", "On Hold")
    # Real yield
    real_yield   = ry.get("value") if isinstance(ry, dict) else None
    # Macro composites by asset class
    eq_score     = mc.get("equity",  {}).get("composite_10") if isinstance(mc.get("equity"), dict)  else None
    bond_score   = mc.get("bond",    {}).get("composite_10") if isinstance(mc.get("bond"),   dict)  else None
    comm_score   = mc.get("commodity",{}).get("composite_10") if isinstance(mc.get("commodity"),dict) else None
    gold_score   = mc.get("gold",   {}).get("composite_10") if isinstance(mc.get("gold"),    dict)  else None

    # ── Build the context block ──────────────────────────────────────────
    lines = [f"REGIME: {regime_name} (climate score {score_10}/10)"]
    if yc_spread is not None:
        lines.append(f"YIELD CURVE: {yc_regime} | 10Y-2Y spread {yc_spread:+.2f}% | 2Y {t2y:.2f}% | 10Y {t10y:.2f}%" if t2y and t10y else f"YIELD CURVE: {yc_regime} | 10Y-2Y {yc_spread:+.2f}%")
    if hy_oas is not None:
        lines.append(f"CREDIT SPREADS: HY OAS {hy_oas}bp ({hy_trend}) | IG OAS {ig_oas}bp" if ig_oas else f"CREDIT: HY OAS {hy_oas}bp ({hy_trend})")
    if jobs_comp is not None:
        nfp_str = f" | NFP {nfp_mom:+.0f}K ({nfp_surp})" if nfp_mom is not None else ""
        ur_str  = f" | Unemployment {unrate}%" if unrate is not None else ""
        lines.append(f"LABOUR: composite {jobs_comp:.1f}/10{nfp_str}{ur_str}")
    if effr is not None:
        if cuts_12m:
            _path_bp = round(-cuts_12m * 25)  # positive = hike, negative = cut
            _direction = "hike" if _path_bp > 2 else "cut" if _path_bp < -2 else "hold"
            cuts_str = f" | market pricing {_path_bp:+d}bp ({_direction}) over 12m"
        else:
            cuts_str = ""
        lines.append(f"FED POLICY: EFFR {effr:.2f}% | stance {rate_label}{cuts_str}")
    if fb_level is not None:
        lines.append(f"FED BALANCE SHEET: ${fb_level:.2f}T ({fb_trend})")
    if real_yield is not None:
        lines.append(f"REAL YIELD (10Y TIPS): {real_yield:+.2f}%")
    if eq_score is not None:  lines.append(f"MACRO COMPOSITE — Equities: {eq_score:.1f}/10")
    if bond_score is not None: lines.append(f"MACRO COMPOSITE — Bonds: {bond_score:.1f}/10")
    if comm_score is not None: lines.append(f"MACRO COMPOSITE — Commodities: {comm_score:.1f}/10")
    if gold_score is not None: lines.append(f"MACRO COMPOSITE — Gold: {gold_score:.1f}/10")

    macro_block = "\n".join(lines)

    # ── Top headlines ────────────────────────────────────────────────────
    headlines = []
    for n in news_items[:10]:
        title   = n.get("title", "")
        impact  = n.get("impact", "medium").upper()
        preview = n.get("preview", "")
        if title:
            line = f"[{impact}] {title}"
            if preview:
                line += f" — {preview[:100]}"
            headlines.append(line)
    headline_block = "\n".join(headlines) if headlines else "No significant headlines."

    prompt = (
        "You are a senior macro analyst writing a concise market brief for a professional trader. "
        "Based on the macro data and headlines below, write a 3–4 sentence global market narrative. "
        "Rules: (1) Reference specific data points where relevant — e.g. the yield curve spread, "
        "credit spread levels, NFP figure, or EFFR. (2) Explain what the regime means for cross-asset "
        "positioning — which asset classes benefit or face headwinds. (3) Flag any notable macro tensions "
        "or contradictions (e.g. risk-on regime but labour softening). (4) Be direct and data-driven. "
        "No bullet points, no headers — flowing prose only. No markdown. 3–4 sentences maximum.\n\n"
        f"MACRO CONTEXT:\n{macro_block}\n\n"
        f"RECENT HEADLINES (last 48h):\n{headline_block}"
    )

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.4,
            },
            timeout=30.0,
        )
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip any stray markdown
        text = text.replace("**", "").replace("##", "").replace("\n\n", " ").strip()
        return text if len(text) > 40 else None
    except Exception as e:
        print(f"[global_narr] error: {e}", flush=True)
        return None


# ============================================================
# WEEKLY CONSENSUS OUTLOOK  (read the fund-manager / bank outlooks)
# ============================================================


def _load_consensus_from_disk() -> None:
    """Warm CONSENSUS_CACHE from disk on boot so a Render redeploy keeps the
    last weekly read until the next scheduled refresh."""
    try:
        if os.path.exists(CONSENSUS_CACHE_FILE):
            with open(CONSENSUS_CACHE_FILE, "r") as f:
                blob = json.load(f)
            if isinstance(blob, dict) and blob.get("data"):
                CONSENSUS_CACHE["data"] = blob["data"]
                CONSENSUS_CACHE["time"] = float(blob.get("time", 0))
                print(f"[consensus] warmed {len(blob['data'].get('markets', {}))} markets from disk", flush=True)
    except Exception as e:
        print(f"[consensus] disk warm failed: {e}", flush=True)


def _save_consensus_to_disk() -> None:
    try:
        with open(CONSENSUS_CACHE_FILE, "w") as f:
            json.dump({"data": CONSENSUS_CACHE["data"], "time": CONSENSUS_CACHE["time"]}, f)
    except Exception as e:
        print(f"[consensus] disk save failed: {e}", flush=True)


# Map COT market ids/names to the consensus 'trade' language a strategist speaks.
# dir_if_long = what going long the FUTURE means in crowd terms.
_COT_TRADE_MAP = {
    "DX":  "long US dollar",       "6E": "long EUR / short USD",  "6B": "long GBP / short USD",
    "6J":  "long JPY / short USD", "6A": "long AUD / short USD",  "6C": "long CAD / short USD",
    "6S":  "long CHF / short USD", "6N": "long NZD / short USD",
    "ES":  "long US equities (S&P)", "NQ": "long US tech (Nasdaq)", "YM": "long US large-cap (Dow)",
    "RTY": "long US small-caps",   "Z":  "long UK equities (FTSE)",
    "GC":  "long gold", "SI": "long silver", "HG": "long copper", "PA": "long palladium", "PL": "long platinum",
    "CL":  "long crude oil", "NG": "long natural gas", "RB": "long gasoline", "HO": "long heating oil",
    "ZB":  "long long-bond / duration", "ZN": "long 10y Treasuries", "ZT": "long 2y Treasuries",
    "ZF":  "long 5y Treasuries",
    "ZC":  "long corn", "ZW": "long wheat", "ZS": "long soybeans",
    "CC":  "long cocoa", "KC": "long coffee", "SB": "long sugar", "CT": "long cotton",
    "LE":  "long live cattle", "HE": "long lean hogs",
    "BTC": "long bitcoin", "ETH": "long ether",
}


def _cot_match_keywords(name: str) -> list[str]:
    """Lower-case tokens used to detect if an offside card references this market."""
    n = (name or "").lower()
    kw = set()
    # base tokens from the market name
    for tok in n.replace("/", " ").replace("-", " ").split():
        if len(tok) >= 3:
            kw.add(tok)
    # asset-class synonyms so narrative phrasing matches
    syn = {
        "bitcoin": ["bitcoin", "btc", "crypto"],
        "ether":   ["ether", "ethereum", "eth", "crypto"],
        "gbp":     ["gbp", "sterling", "pound", "cable"],
        "eur":     ["eur", "euro"],
        "jpy":     ["jpy", "yen"],
        "aud":     ["aud", "aussie"],
        "cad":     ["cad", "loonie", "canada", "canadian"],
        "nzd":     ["nzd", "kiwi"],
        "chf":     ["chf", "franc", "swiss"],
        "crude":   ["crude", "oil", "wti", "energy"],
        "gold":    ["gold", "bullion", "xau"],
        "copper":  ["copper"],
        "silver":  ["silver"],
        "gas":     ["gas", "natgas"],
    }
    for key, words in syn.items():
        if key in n or any(w in n for w in words):
            kw.update(words)
    return sorted(kw)


def _assemble_offside_brief() -> dict:
    """Assemble the app's OWN hard data into a compact brief the consensus prompt
    can reason against — so Sonar isn't guessing where the crowd is offside.

    Three dynamic inputs:
      1. COT positioning EXTREMES  — large-spec percentile >=85 (crowd very long)
         or <=15 (crowd very short), plus whether specs are starting to TURN
         (the early-reversal tell). Pulled live from the scores cache.
      2. Upcoming CATALYSTS        — high/medium-impact FF calendar events still
         ahead of now this week (data releases, central-bank decisions, speakers),
         with forecast vs previous so the prompt can see the setup.
      3. News / real-world flow    — latest headline framing.

    Returns {"positioning": str, "catalysts": str, "news": str} — all plain text.
    """
    positioning_lines: list[str] = []
    catalysts_lines: list[str] = []
    news_lines: list[str] = []
    extremes: list[dict] = []

    # ---- 1. COT extremes from the live scores cache -------------------------
    try:
        blob = ALL_DATA_CACHE.get("data") or {}
        markets = blob.get("markets", []) or []
        print(f"[consensus] assemble: {len(markets)} markets in scores cache", flush=True)
        rows = []
        for m in markets:
            cot = (m.get("scores") or {}).get("cot") or {}
            li = cot.get("lspec_index")
            ci = cot.get("comm_index")
            if li is None:
                continue
            if li >= 85 or li <= 15:
                mid = (m.get("id") or "").upper()
                base_trade = _COT_TRADE_MAP.get(mid, f"long {m.get('name','?')}")
                # Large specs = trend-following crowd. High index = crowd very LONG the
                # base trade; low index = crowd very SHORT it, so invert the phrasing.
                crowd_side = "very long" if li >= 85 else "very short"
                if li <= 15:
                    # Flip "long A / short B" -> "short A / long B"; else prepend short.
                    if " / " in base_trade:
                        legs = []
                        for leg in base_trade.split(" / "):
                            leg = leg.strip()
                            if leg.startswith("long "):
                                legs.append("short " + leg[5:])
                            elif leg.startswith("short "):
                                legs.append("long " + leg[6:])
                            else:
                                legs.append(leg)
                        trade = " / ".join(legs)
                    elif base_trade.startswith("long "):
                        trade = "short " + base_trade[5:]
                    else:
                        trade = "short " + base_trade
                else:
                    trade = base_trade
                # Are specs starting to turn against their extreme? (early reversal)
                turn_dir = cot.get("v2_spec_turn_dir")
                turn_conf = cot.get("v2_spec_turn_confirmed")
                turn_txt = ""
                if turn_conf and turn_dir in (1, -1):
                    turn_txt = " — specs CONFIRMED turning" + (" up" if turn_dir == 1 else " down")
                elif turn_dir in (1, -1):
                    turn_txt = " — specs tentatively turning" + (" up" if turn_dir == 1 else " down")
                extremity = abs(li - 50)
                rows.append((extremity, m.get("name", "?"), trade, crowd_side,
                             round(li, 0), round(ci, 0) if ci is not None else None, turn_txt))
        rows.sort(key=lambda r: -r[0])
        for _, name, trade, side, li, ci, turn in rows[:12]:
            cpart = f", commercials {int(ci)}th pct" if ci is not None else ""
            positioning_lines.append(
                f"- {name}: large specs {side} ({int(li)}th pct{cpart}) => crowd is {trade}{turn}"
            )
            # Structured record for deterministic offside-card positioning fill.
            _crowd_dir = "long" if side == "very long" else "short"
            extremes.append({
                "name": name,
                "keywords": _cot_match_keywords(name),
                "crowd_dir": _crowd_dir,
                "text": f"Crowd is {'very ' if abs(li-50)>=40 else ''}{_crowd_dir} {trade} "
                        f"({int(li)}th pct COT) — parallels the consensus view",
            })
    except Exception as e:
        print(f"[consensus] positioning assemble failed: {e}", flush=True)

    # ---- 2. Upcoming catalysts from the FF calendar -------------------------
    try:
        cal = fetch_ff_calendar_json() or []
        now_ts = time.time()
        up = []
        for e in cal:
            dl = e.get("dateline")
            imp = (e.get("impactClass") or "").lower()
            if dl and dl > now_ts and imp in ("high", "medium"):
                up.append((dl, imp, e))
        up.sort(key=lambda x: x[0])
        for dl, imp, e in up[:14]:
            when = datetime.utcfromtimestamp(dl).strftime("%a %d %b %H:%M")
            cur = e.get("currency", "")
            nm = e.get("name", "")
            fc = e.get("forecast", ""); pv = e.get("previous", "")
            fcpart = f" (fc {fc} vs prev {pv})" if (fc or pv) else ""
            tag = "HIGH" if imp == "high" else "med"
            catalysts_lines.append(f"- {when} UTC [{tag}] {cur} {nm}{fcpart}")
    except Exception as e:
        print(f"[consensus] catalysts assemble failed: {e}", flush=True)

    # ---- 3. News / real-world flow -----------------------------------------
    try:
        news = fetch_ff_news(hours_back=48) or []
        for n in news[:10]:
            t = (n.get("title") or "").strip()
            if t:
                news_lines.append(f"- {t[:130]}")
    except Exception as e:
        print(f"[consensus] news assemble failed: {e}", flush=True)

    return {
        "positioning": "\n".join(positioning_lines) if positioning_lines else "(no extreme COT positioning detected)",
        "catalysts":   "\n".join(catalysts_lines) if catalysts_lines else "(no high/medium-impact events remaining this week)",
        "news":        "\n".join(news_lines) if news_lines else "(no recent headlines)",
        "extremes":    extremes,
    }


def generate_consensus_outlook() -> dict:
    """
    Build the CONSENSUS & OFFSIDE-RISK read: a robust synthesis of what the crowd
    (bank desks / CTAs / fund surveys / news flow) collectively believes right
    now, woven together with WHERE THAT CONSENSUS IS MOST LIKELY TO BE CAUGHT
    OFFSIDE — i.e. a widely-held belief that an upcoming catalyst, real-world
    event, performance shift, or stretched positioning could violently unwind.

    The purpose is NOT to list crowded trades (COT already does that). It is to
    identify asymmetry: where the narrative and the positioning point the same
    way AND a catalyst sits ahead that could break it.

    Sonar (web-search) gathers this week's commentary; we hand it the app's OWN
    hard data via _assemble_offside_brief() — live COT positioning extremes,
    upcoming FF catalysts, and news flow — so the offside cross-reference is
    grounded in real numbers, not guessed.

    Returns:
      {
        "outlook":   "woven consensus narrative (what the crowd believes + why)",
        "offside":   [ {"belief": "...", "catalyst": "...", "positioning": "...",
                        "risk": "long|short|two-way", "note": "..."}, ... ],
        "as_of":     "13 July 2026",
        "citations": [ ...urls... ]
      }
    Returns {} on any error (graceful degradation — the block just hides).
    """
    api_key = os.environ.get("PPLX_API_KEY", "")
    if not api_key:
        print("[consensus] no PPLX_API_KEY — skipping", flush=True)
        return {}

    today = datetime.utcnow().strftime("%d %B %Y")
    brief = _assemble_offside_brief()

    prompt = (
        f"Today is {today}. You are the head strategist for a macro swing-trading desk. "
        "Your job this week is to (1) capture the CONSENSUS view across global markets, and "
        "(2) identify WHERE THAT CONSENSUS IS MOST LIKELY TO BE CAUGHT OFFSIDE.\n\n"
        "Read THIS WEEK's commentary the way you would if you'd read a stack of fund-manager "
        "outlooks. Draw from the LATEST (last 7-10 days):\n"
        "  • Sell-side bank & broker weekly outlooks (JPMorgan, Goldman Sachs, Morgan Stanley, "
        "Citi, Barclays, BofA, UBS, Deutsche, Nomura, Scotiabank, Commerzbank, ING).\n"
        "  • CTA / trend-following & macro fund positioning notes and surveys (BofA CTA monitor, "
        "SocGen CTA index, BofA Global Fund Manager Survey, 'most crowded trade' polls).\n"
        "  • Financial news flow framing (Reuters / Bloomberg / FT-style).\n\n"
        "You have been given the desk's OWN hard data below. USE IT to ground your offside calls "
        "— when the crowd's talked-about view lines up with a stretched positioning reading AND a "
        "catalyst sits ahead, that is a high-value offside setup. Catalysts can be economic data "
        "releases, central-bank decisions/speakers, real-world/geopolitical events, or performance "
        "itself (a move that's gone too far, too fast).\n\n"
        "=== DESK POSITIONING DATA (live COT extremes — large specs = trend-following crowd) ===\n"
        f"{brief['positioning']}\n\n"
        "=== UPCOMING CATALYSTS (high/medium-impact events still ahead this week) ===\n"
        f"{brief['catalysts']}\n\n"
        "=== RECENT NEWS / REAL-WORLD FLOW ===\n"
        f"{brief['news']}\n\n"
        "Now produce the read. Return ONLY valid JSON — no markdown, no text before/after. "
        "Do NOT use asterisks, underscores, or bracketed [n] citation markers inside any string. Shape:\n"
        "{\n"
        '  "outlook": "4-6 sentences: the prevailing cross-asset consensus this week and why, '
        'naming the desks/surveys driving it. This is the robust narrative of what the crowd '
        'believes across USD/G10 FX, equities, rates, metals, energy and crypto.",\n'
        '  "offside": [ {\n'
        '     "belief": "the consensus belief at risk, stated crisply in under 15 words (e.g. \'crowd firmly long USD on soft-landing\')",\n'
        '     "catalyst": "the event/data/move that could break it, with timing if known",\n'
        '     "positioning": "how stretched/aligned positioning is (cite the COT reading above if it parallels), or \'n/a\' if positioning is not a factor",\n'
        '     "risk": "long|short|two-way",\n'
        '     "note": "1 crisp sentence on the asymmetry / what unwinds if the belief is wrong"\n'
        '  } ]\n'
        "}\n"
        "Give 3-5 offside setups, ranked most-asymmetric first. CRITICAL RULE ON POSITIONING: the "
        "DESK POSITIONING DATA above lists the markets where the trend-following crowd is ALREADY at "
        "a COT extreme. These are your highest-value offside candidates because the narrative AND the "
        "positioning point the same way. You MUST build AT LEAST TWO of your setups directly around "
        "the specific markets named in that positioning data (e.g. if it shows 'crowd very long "
        "Bitcoin 96th pct' and 'crowd very short GBP 8th pct', make setups about crowded crypto longs "
        "and crowded GBP shorts). In each such setup, NAME that market in the 'belief' text and quote "
        "the exact percentile in the 'positioning' field (e.g. 'crowd is very long Bitcoin, 96th pct "
        "— matches the bullish crypto narrative'). Only use 'n/a' for positioning on setups that are "
        "genuinely macro/rates themes with no single-market COT parallel. Do NOT make every setup a "
        "broad macro theme while ignoring the concrete positioning extremes handed to you. "
        "'risk' = the direction of the crowd's exposure that is vulnerable (crowd long -> risk='long'; "
        "crowd short -> risk='short'). Start with { and end with }."
    )

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar-pro",           # web-search enabled — actually reads the outlooks
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3500,
                "temperature": 0.2,
                "search_recency_filter": "week",
            },
            timeout=110.0,
        )
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        citations = data.get("citations", []) or []
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0].strip()
        s = raw.find("{"); e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e + 1]
        parsed = json.loads(raw)
        def _clip(txt, n):
            """Clip to n chars on a word boundary (avoid mid-word cuts)."""
            t = str(txt or "").strip()
            if len(t) <= n:
                return t
            cut = t[:n]
            sp = cut.rfind(" ")
            return (cut[:sp] if sp > n * 0.6 else cut).rstrip(" ,;:-") + "\u2026"
        offside_in = parsed.get("offside", []) if isinstance(parsed, dict) else []
        extremes = brief.get("extremes", []) or []
        def _fill_positioning(pos_txt, belief, note, risk):
            """Deterministically attach a COT reading when a card references a market
            that is at a positioning extreme and the crowd's side matches the
            vulnerable direction — even if the model left positioning as 'n/a'."""
            existing = str(pos_txt or "").strip()
            if existing and existing.lower() not in ("n/a", "na", "none", "-"):
                return existing
            hay = f"{belief} {note}".lower()
            for ex in extremes:
                kws = ex.get("keywords") or []
                if not any(k in hay for k in kws):
                    continue
                # only attach if the crowd's COT side matches the card's risk side
                # (long-crowd card ↔ crowd long; short-crowd card ↔ crowd short)
                if risk in ("long", "short") and ex.get("crowd_dir") != risk:
                    continue
                # two-way cards accept either side (positioning still informative)
                return ex.get("text", "")
            return ""
        offside = []
        for o in offside_in:
            if not isinstance(o, dict):
                continue
            rk = str(o.get("risk", "")).lower().strip()
            if rk not in ("long", "short", "two-way"):
                rk = "two-way"
            belief = _clip(o.get("belief", ""), 210)
            if not belief:
                continue
            pos_final = _fill_positioning(o.get("positioning", ""),
                                          o.get("belief", ""), o.get("note", ""), rk)
            offside.append({
                "belief":      belief,
                "catalyst":    _clip(o.get("catalyst", ""), 240),
                "positioning": _clip(pos_final, 240),
                "risk":        rk,
                "note":        _clip(o.get("note", ""), 280),
            })
        outlook_raw = str(parsed.get("outlook", "")).strip() if isinstance(parsed, dict) else ""
        # Clamp without cutting mid-sentence: if too long, trim back to the last
        # sentence-ending punctuation within the limit.
        if len(outlook_raw) > 1600:
            cut = outlook_raw[:1600]
            last_end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
            outlook_raw = (cut[:last_end + 1] if last_end > 200 else cut.rstrip()).strip()
        result = {
            "outlook":   outlook_raw,
            "offside":   offside[:6],
            "as_of":     today,
            "citations": [str(c)[:300] for c in citations][:20],
        }
        print(f"[consensus] read: {len(result['outlook'])} chars outlook, {len(offside)} offside setups", flush=True)
        return result
    except Exception as e:
        print(f"[consensus] error: {e}", flush=True)
        return {}


def compute_consensus_outlook(force: bool = False) -> dict:
    """Cached wrapper around generate_consensus_outlook (weekly TTL + disk)."""
    now = time.time()
    if (not force and CONSENSUS_CACHE["data"] is not None
            and (now - CONSENSUS_CACHE["time"]) < CONSENSUS_CACHE_TTL):
        return CONSENSUS_CACHE["data"]
    # Guard: never regenerate against a cold scores cache — the offside brief
    # would carry NO COT positioning, producing positioning-less cards that
    # overwrite a good cached read. Only allow a cold-cache regen when there is
    # no existing consensus at all (first populate). An explicit force still
    # regenerates (the caller is expected to warm /api/scores first).
    _markets = ((ALL_DATA_CACHE.get("data") or {}).get("markets") or [])
    if not force and not _markets and CONSENSUS_CACHE["data"] is not None:
        print("[consensus] scores cache cold — keeping existing consensus, skip regen", flush=True)
        return CONSENSUS_CACHE["data"]
    # How many COT extremes are available right now? If >=2, we expect the
    # offside read to surface at least 2 positioning parallels. sonar-pro is
    # non-deterministic, so retry once if the first read ignored positioning.
    _n_extremes = len(_assemble_offside_brief().get("extremes", []))
    def _pos_count(d):
        return sum(1 for o in (d.get("offside") or []) if (o.get("positioning") or "").strip())
    fresh = generate_consensus_outlook()
    if (fresh and fresh.get("outlook") and _n_extremes >= 2
            and _pos_count(fresh) < 2):
        print(f"[consensus] only {_pos_count(fresh)} positioning parallels vs "
              f"{_n_extremes} extremes available — regenerating once", flush=True)
        retry = generate_consensus_outlook()
        if retry and retry.get("outlook") and _pos_count(retry) > _pos_count(fresh):
            fresh = retry
    if fresh and fresh.get("outlook"):
        CONSENSUS_CACHE["data"] = fresh
        CONSENSUS_CACHE["time"] = now
        _save_consensus_to_disk()
        return fresh
    # generation failed — serve any stale cache rather than nothing
    return CONSENSUS_CACHE["data"] or {"outlook": "", "offside": [], "as_of": "", "citations": []}


def compute_news_context(force: bool = False) -> dict:
    """
    News feed: qualitative financial headlines from ForexFactory /news page,
    plus per-asset AI narratives generated from those headlines via Sonar.
    Caches: news 2h, narratives 2h (independent).
    """
    now = time.time()

    # ── 1. Fetch FF news (uses its own cache internally) ────────────────
    news_cache_hit = (
        not force
        and FF_NEWS_CACHE["data"] is not None
        and (now - FF_NEWS_CACHE["time"]) < FF_NEWS_TTL
    )
    if news_cache_hit:
        news_items = FF_NEWS_CACHE["data"]
        news_ts    = FF_NEWS_CACHE["time"]
    else:
        news_items = fetch_ff_news(hours_back=48)
        news_ts    = now
        # fetch_ff_news already updates FF_NEWS_CACHE internally

    # ── 2. Per-asset narratives (Sonar) — independent cache ────────────
    narr_hit = (
        not force
        and NARR_CACHE["data"] is not None
        and (now - NARR_CACHE["time"]) < NARR_CACHE_TTL
    )
    if narr_hit:
        narratives = NARR_CACHE["data"]
    else:
        # Generate fresh narratives from current headlines
        print("[narr] Generating per-asset narratives from FF headlines…", flush=True)
        narratives = generate_asset_narratives(news_items)
        NARR_CACHE["data"] = narratives
        NARR_CACHE["time"] = now
        if narratives:
            print(f"[narr] Generated narratives for {len(narratives)} assets", flush=True)
        else:
            print("[narr] No narratives generated (empty or error)", flush=True)

    # Separate text strings (for frontend display) from scores (for regime blending)
    narratives_text   = {k: v["text"]     for k, v in narratives.items() if isinstance(v, dict)}
    narratives_scores = {k: v["score_10"] for k, v in narratives.items() if isinstance(v, dict) and v.get("score_10") is not None}

    # ── 3. Global narrative (Sonar) — uses macro + news context ────────────
    gnl_hit = (
        not force
        and GLOBAL_NARR_CACHE["data"] is not None
        and (now - GLOBAL_NARR_CACHE["time"]) < GLOBAL_NARR_CACHE_TTL
    )
    if gnl_hit:
        global_narrative = GLOBAL_NARR_CACHE["data"]
    else:
        regime_data = RISK_REGIME_CACHE.get("data") or {}
        print("[global_narr] Generating global market narrative…", flush=True)
        global_narrative = generate_global_narrative(regime_data, news_items)
        GLOBAL_NARR_CACHE["data"] = global_narrative
        GLOBAL_NARR_CACHE["time"] = now
        if global_narrative:
            print(f"[global_narr] Generated: {global_narrative[:80]}…", flush=True)
        else:
            print("[global_narr] Generation failed — will use fallback", flush=True)

    # ── 4. Weekly consensus outlook (Sonar web-search) — own 7-day cache ────
    # Read the week's fund-manager / bank-desk / CTA outlooks and distil the
    # cross-asset crowd view. Sits alongside the Market Narrative on the front
    # end. Weekly TTL so it doesn't re-run on every 2h news refresh. `force`
    # here would over-run the weekly cadence, so we only force it when there is
    # no data at all (first populate); otherwise honour its own TTL.
    try:
        consensus_outlook = compute_consensus_outlook(
            force=(force and CONSENSUS_CACHE["data"] is None)
        )
    except Exception as _ce:
        print(f"[consensus] compute failed: {_ce}", flush=True)
        consensus_outlook = CONSENSUS_CACHE["data"] or {}
    consensus_ts = CONSENSUS_CACHE["time"] or None

    return {
        "narratives":        narratives_text,    # {assetId: "text string"} — frontend display
        "narrative_scores":  narratives_scores,  # {assetId: 0-10 float} — regime blending
        "news_items":        news_items[:20],
        "global_narrative":  global_narrative,
        "consensus_outlook": consensus_outlook,   # weekly cross-asset crowd view
        "consensus_updated_at": consensus_ts,
        "price_context":     {},
        "updated_at":        news_ts,
        "ff_event_count":    len(news_items),
    }


# ============================================================
# WEIGHTED SCORE + BIAS LABEL
# ============================================================

# ── WEIGHT RATIONALE (v2 — evidence-led, May 2026) ─────────────────────────────
# COT: 25% standard (down from 30%). Briese commercials index is more nuanced than
#   raw net positioning tested in academic literature (Fernandez-Perez SR ~0.5).
#   Still the single largest weight in physical commodities where commercial hedgers
#   have genuine information edge. Reduced due to post-2008 structural decay in
#   financial futures (equities, FX) where SR drops to ~0.37.
# Momentum: 20% (up from 15%). Strong cross-asset evidence (Moskowitz et al. SR 1.0-1.8
#   diversified). Now tilted toward 8-26 week lookback per academic evidence; short-term
#   weekly IC shows reversal not continuation at 1-4 week horizon.
# Macro: 15% (unchanged). Leading indicators (CLI/BCI) show OOS R² 5-8% for commodities;
#   NFP/CPI surprises have 3-6 week rate-path impact.
# Seasonal: 13% (down from 15%). Accounts for decay in pure calendar seasonality post-2004
#   but preserved above academic evidence due to election cycle overlay — presidential cycle
#   has well-documented effects on DXY, indices, crude that raw studies miss.
# Rel. Value: 12% (down from 15%). Asset-specific mean-reversion evidence is solid
#   (gold-silver SR 0.71; energy spreads SR 1.0-1.2) but pair-specific and regime-dependent.
# Climate & Rates: 15% (up from 5%). KC Fed: 1 SD risk-off shock drives 78bps equity
#   return with 150+ day persistence; RORO outperforms VIX and EBP as predictor of
#   high-magnitude moves. Tripled from 5% to reflect its dominance in extreme market moves.
# PCR: 0% standard (unchanged — active only for equity/metals markets via per-asset tiers).
# ─────────────────────────────────────────────────────────────────────────────────────────
#
# ══ v3 WEIGHT SYSTEM — Research-backed per-asset tiers (May 2026) ══════════════════════
#
# Replaced the 6 flat-class tiers with 14 empirically-derived per-asset tiers.
# Key research findings driving the changes:
#
# COT reliability by class (UMBC/Briese/Sanders et al.):
#   FX: HIGHEST (R²~0.48 for EUR, 8/10) — counterintuitively ABOVE physical commodities.
#   Gold/Coffee: High (7-7.5/10). Crude/Base metals: Medium (6.5/10).
#   Grains: Medium-low (5-6/10) — Sanders et al. 2009: traders FOLLOW price, not the reverse.
#   NG/Bonds: Medium-low (5-5.5/10) — storage cycle / FOMC structural distortion.
#   Livestock: LOW (5/10) — meatpackers/feedlots are MECHANICAL seasonal hedgers with
#     kill-schedule-driven positioning that contains NO directional price information.
#   Cocoa: LOW (4.5/10) — govt marketing boards (COCOBOD/CCC) corrupt commercial signal.
#   Equity index: VERY LOW (3/10) — CXO Advisory R²=0.02. Portfolio hedgers have zero edge.
#
# Macro sensitivity (Chicago Fed, St Gallen, IntechOpen):
#   Bonds/FX/Equity: Very high (40/38/35% of variance). Gold: High (28%, real-yield -13.1%/100bp).
#   Crude: Medium (35% post-2008 financialisation). Grains: Low (weather dominates, ~12%).
#   Softs: Very low (~8%, 90-95% physical). Livestock: Ultra-low (~6%, >95% physical).
#   NG: Near-zero macro sensitivity (coal explains 73% of variance; macro = 0.14-0.40%).
#
# Climate/regime sensitivity (Tang & Xiong 2012, Hamilton & Wu 2014, KC Fed):
#   Bonds/Equity/FX: Very high (35-38%). Gold: High (safe-haven, RORO, real yield).
#   Crypto: High (BTC tracks global M2 83% of time, 0.94 correlation, Lyn Alden 2024).
#   Grains/Softs: Very low — financialisation raised cross-commodity correlation but does
#     NOT reliably predict price levels OOS (Hamilton & Wu adj R² frequently negative).
#   Livestock: Ultra-low (~5%) — supply cycles are biological, not financial.
#
# Seasonality (EIA, CME livestock, MRCI, Univ. of Wisconsin):
#   NG: TIER 1 (EIA 10:1 heating ratio, Bank of England WP 591 confirmed stochastic).
#   Lean Hogs: TIER 1 (biological lock — sow farrowing cycle mechanistically reliable).
#   Grains: High (Wisconsin: 9/10 years hit harvest low).
#   Softs: LOW — tropical weather/disease events fully override annual calendar seasonals.
#
# ─────────────────────────────────────────────────────────────────────────────────────────

# ── STANDARD fallback (rare edge-cases not covered below) ───────────────────────────────
WEIGHTS = {
    "cot":      0.25,
    "seasonal": 0.13,
    "momentum": 0.18,
    "macro":    0.18,
    "regime":   0.18,
    "relval":   0.08,
    "pcr":      0.00,
}

# ── HG COPPER ────────────────────────────────────────────────────────────────────────────
# COT strong (7/10): genuine informed industrial hedgers; macro/regime high (growth barometer).
# Macro 24%: copper is one of the most macro-sensitive base metals (China PMI, global IP).
# Regime 22%: copper is the canonical risk-on proxy / global growth signal.
WEIGHTS_HG = {
    "cot":      0.22,
    "seasonal": 0.08,
    "momentum": 0.20,
    "macro":    0.24,  # China PMI / global IP are primary copper price drivers
    "regime":   0.22,  # Risk-on proxy; copper = global growth barometer
    "relval":   0.04,
    "pcr":      0.00,
}

# ── PA PALLADIUM ─────────────────────────────────────────────────────────────────────────
# COT moderate-low (5/10): dominated by spec momentum, auto-catalyst demand structural.
# Momentum high (25%): autocatalyst supply squeezes produce fast spec-driven moves.
# PCR 5%: PALL options exist with moderate signal.
WEIGHTS_PA = {
    "cot":      0.14,
    "seasonal": 0.06,
    "momentum": 0.30,  # Highest in system: autocatalyst supply squeezes drive fast spec moves
    "macro":    0.22,  # Auto production cycles and EV transition matter
    "regime":   0.20,
    "relval":   0.08,  # Palladium/platinum spread relationship meaningful
    "pcr":      0.00,  # Removed: PA options illiquid, PCR readings unreliable
}

# ── PL PLATINUM ──────────────────────────────────────────────────────────────────────────
# COT moderate (6/10): genuine fabricator hedging. Relval high vs gold/palladium.
WEIGHTS_PL = {
    "cot":      0.20,
    "seasonal": 0.08,
    "momentum": 0.20,
    "macro":    0.20,
    "regime":   0.22,
    "relval":   0.10,  # Pt/Pd/Au spread relationships are meaningful
    "pcr":      0.00,
}

# ── FX MAJORS (6E, 6J, 6B, 6A, 6C, 6N, 6S, 6M, DX) ────────────────────────────────────
# COT is the STRONGEST signal class for FX per UMBC study (R²~0.48 EUR net positions). 25%.
# Macro very high (rate differentials are THE primary FX driver). 27%.
# Seasonal trimmed to 8% — academic evidence weak for FX seasonals vs equities.
WEIGHTS_FX = {
    "cot":      0.25,  # Strong: FX COT R²~0.48; leveraged fund extremes signal key turns
    "seasonal": 0.08,
    "momentum": 0.20,
    "macro":    0.27,  # Rate differentials dominate FX pricing (highest weight)
    "regime":   0.20,  # RORO and carry unwind are primary FX regime signals
    "relval":   0.00,
    "pcr":      0.00,
}

# ── FX CROSSES (computed pairs: EURJPY, EURGBP, GBPJPY etc.) ─────────────────────────────
# Crosses have proxied COT (lower reliability) → COT cut to 15%.
# Macro higher at 30%: cross pricing is driven by relative rate differential between
#   the two constituent currencies, amplifying macro sensitivity vs the majors.
WEIGHTS_FX_CROSSES = {
    "cot":      0.15,  # Lower: proxied from two majors; less direct than CME futures COT
    "seasonal": 0.08,
    "momentum": 0.22,
    "macro":    0.30,  # Highest: cross = relative macro between two currencies
    "regime":   0.20,
    "relval":   0.05,
    "pcr":      0.00,
}

# ── EQUITY INDICES (ES, NQ, YM, RTY) ────────────────────────────────────────────────────
# COT very low (3/10, R²=0.02) — portfolio hedgers have zero directional edge. 10%.
# Macro+regime each 25% — Fed/EPS/credit and RORO dominate equity price discovery.
# PCR 8% (slightly reduced from 10% to accommodate regime/macro uplift).
WEIGHTS_EQUITY = {
    "cot":      0.10,  # Very low: R²=0.02, portfolio hedgers have no directional edge
    "seasonal": 0.10,
    "momentum": 0.22,
    "macro":    0.25,  # Fed/EPS cycle/credit spreads explain ~35% of variance
    "regime":   0.25,  # RORO is dominant equity futures predictor (KC Fed research)
    "relval":   0.00,
    "pcr":      0.08,  # Deep options markets; ES/NQ have genuine sentiment signal
}

# ── GOLD (GC) ────────────────────────────────────────────────────────────────────────────
# COT high (7.5/10) — 50-year track record, Briese commercials right ~2/3 of time. 22%.
# Macro 30%: real-yield sensitivity is the most precisely quantified macro relationship in
#   commodity finance — +100bp real yield → -13.1% real gold; +1pp inflation expectations
#   → +37% (WGC research). Boosted 22%→30%.
# Regime 22%: gold is THE safe-haven asset; RORO and real yield move together.
# Momentum trimmed to 15% (macro/regime dominate; momentum is confirming signal).
WEIGHTS_GOLD = {
    "cot":      0.22,
    "seasonal": 0.08,
    "momentum": 0.18,  # Gold trends well; raised from 0.15
    "macro":    0.26,  # Real-yield anchor; trimmed from 0.30 — macro+regime were correlated at 52%
    "regime":   0.20,  # Safe-haven demand = RORO derivative; trimmed from 0.22
    "relval":   0.03,  # Au/Ag ratio signal; small but non-zero
    "pcr":      0.03,  # GLD options deeply liquid — PCR has genuine contrarian sentiment signal
}

# ── SILVER (SI) ──────────────────────────────────────────────────────────────────────────
# COT 6.5/10 (structurally always net short commercial — different mechanics to gold). 22%.
# Dual industrial/monetary nature: macro 20%, regime 20%.
# PCR 3% (SLV options thinner than GLD).
WEIGHTS_SILVER = {
    "cot":      0.22,
    "seasonal": 0.09,
    "momentum": 0.20,
    "macro":    0.20,
    "regime":   0.20,
    "relval":   0.06,
    "pcr":      0.03,
}

# ── CRUDE OIL (CL) ────────────────────────────────────────────────────────────────────────
# COT 6.5/10 (OPEC structural interference post-2016, spec crowding). 22%.
# Post-2008 financialisation: macro now explains 35% of CL return variance (up from 11%).
# Seasonal high (refinery/demand seasonal well-established). 18%.
# PCR 5% (USO options moderate liquidity).
WEIGHTS_CRUDE = {
    "cot":      0.22,  # OPEC distortion, spec crowding reduces reliability
    "seasonal": 0.18,  # Refinery/demand seasonal well-established
    "momentum": 0.20,
    "macro":    0.20,  # 35% of return variance post-2008 financialisation
    "regime":   0.12,  # Risk appetite is a supporting driver
    "relval":   0.03,
    "pcr":      0.05,
}

# ── NATURAL GAS (NG) ─────────────────────────────────────────────────────────────────────
# COT 5.5/10 (storage injection cycle mechanical). 18%.
# MACRO NEAR-ZERO: structural VAR shows coal explains 73% of NG variance; interest
#   rates explain only 0.14-0.40% (Domfeh, IntechOpen 2023). 6%.
# CLIMATE ULTRA-LOW: weather and storage dominate; RORO has negligible NG impact. 4%.
# SEASONAL TIER 1: EIA 10:1 heating ratio; Bank of England WP591 confirms stochastic
#   seasonality. Boosted 22%→30% — strongest physical seasonal in all futures.
# Relval 20%: energy calendar spread (NGQ vs NGZ) is a core signal for NG.
WEIGHTS_NATGAS = {
    "cot":      0.18,  # Storage cycle mechanical hedging reduces reliability
    "seasonal": 0.30,  # TIER 1 HIGHEST: physically anchored heating demand cycle
    "momentum": 0.22,  # Storage injection reactions drive fast moves
    "macro":    0.06,  # Near-zero: coal/weather explain 73%+; macro = 0.14-0.40% of variance
    "regime":   0.04,  # Ultra-low: weather and storage dominate; RORO negligible
    "relval":   0.20,  # Energy spread dynamics (NGQ vs NGZ) meaningful
    "pcr":      0.00,
}

# ── AGRICULTURAL GRAINS (ZC corn, ZS soybeans, ZW wheat) ────────────────────────────────
# COT 5-6/10: Sanders et al. 2009 Granger causality — traders RESPOND to prices,
#   prices do NOT respond to traders. CIT index-fund distortion severe. Boosted 20%→25%.
# SEASONAL HIGH: Wisconsin Univ documents 9/10 years hit harvest low. Boosted 20%→22%.
# MACRO LOW: weather and supply fundamentals dominate. 8%.
# CLIMATE VERY LOW: Tang/Xiong financialisation does NOT hold OOS for agri contracts
#   (Hamilton & Wu 2014, adj R² frequently negative). 5%.
# Relval 20%: old/new-crop spreads are one of the most robust grain signals.
WEIGHTS_GRAINS = {
    "cot":      0.25,  # Raised: despite CIT distortion, extremes still signal major turns
    "seasonal": 0.22,  # Harvest calendar: highly consistent timing (9/10 years)
    "momentum": 0.20,
    "macro":    0.08,  # Low: weather/supply fundamentals dominate macro signals
    "regime":   0.05,  # Very low: financialisation effect not significant OOS
    "relval":   0.20,  # Old/new-crop spreads and cross-grain relval meaningful
    "pcr":      0.00,
}

# ── SOFT COMMODITIES (SB sugar, CT cotton) ─────────────────────────────────────────────
# Seasonal LOW: weather/disease shocks routinely override annual calendar patterns.
# Macro VERY LOW: 90-95% of soft commodity price moves are physically driven.
# Climate VERY LOW: 7% literature estimate; financialisation not reliable OOS for softs.
# Relval HIGH but capped at 26%: inter-soft mean-reversion and spread relationships.
# CC (cocoa) separated into WEIGHTS_COCOA due to COCOBOD/CCC govt board corruption.
WEIGHTS_SOFTS = {
    "cot":      0.12,  # SB/CT: moderate; reduced from 20% — govt boards distort SB too
    "seasonal": 0.12,  # LOW: weather shocks routinely fully override annual seasonals
    "momentum": 0.22,  # Price action dominates in supply-shock-driven markets
    "macro":    0.18,  # Somewhat raised vs pure softs — SB/CT have financial component
    "regime":   0.10,  # Low but non-zero: financialisation partial for SB/CT
    "relval":   0.26,  # HIGH but capped: inter-soft mean-reversion
    "pcr":      0.00,
}

# ── COCOA (CC) ──────────────────────────────────────────────────────────────────────────
# COT near-zero (0.05): COCOBOD (Ghana) and CCC (Ivory Coast) pre-sell 70-80% of each
#   harvest via government marketing boards, making the commercial signal structurally
#   near-meaningless for directional positioning. Validated by multiple crop-finance studies.
# Macro and regime raised: financial buyers dominate cocoa (ICCO data).
# Seasonal 15%: West African main crop / mid-crop cycle is physically anchored.
# Relval 23%: inter-soft spread (CC/SB) and processing margin (grindings) meaningful.
WEIGHTS_COCOA = {
    "cot":      0.12,  # Aligned with SB: commercial signal degraded by COCOBOD/CCC
                       # pre-selling, but large spec extremes remain predictive
    "seasonal": 0.12,
    "momentum": 0.22,
    "macro":    0.18,
    "regime":   0.10,
    "relval":   0.26,  # CC/SB spread and grindings data meaningful
    "pcr":      0.00,
}

# ── COFFEE (KC / RC Robusta) ─────────────────────────────────────────────────────────────
# KC is the outlier soft: roasters and trading houses are semi-informed hedgers.
# COT 25%: reliable for major trend change detection (Briese: 7/10).
# Macro raised to 15%: global coffee demand has a genuine income elasticity component.
# Relval trimmed to 19%: KC/SB spread dynamics less reliable than previously weighted.
WEIGHTS_COFFEE = {
    "cot":      0.25,  # High: roasters/trading houses are genuinely semi-informed hedgers
    "seasonal": 0.10,
    "momentum": 0.23,
    "macro":    0.15,  # Income elasticity and EM consumer demand matter
    "regime":   0.08,
    "relval":   0.19,  # KC/SB and arabica/robusta spread dynamics
    "pcr":      0.00,
}

# ── LIVESTOCK (HE lean hogs, LE live cattle, GF feeder cattle) ──────────────────────────
# COT LOW (5/10): meatpackers and feedlots are MECHANICAL SEASONAL HEDGERS.
#   Kill-schedule-driven positioning contains NO directional price information.
#   Managed money extremes are the only useful COT signal here (as crowding indicator). 13%.
# SEASONAL TIER 1 (HE): sow farrowing produces market-ready hogs by Nov-Dec — the most
#   mechanistically reliable seasonal in all of futures. Boosted 22%→27%.
# MACRO ULTRA LOW: >95% physical price determination. Consumer spending is marginal. 8%.
# CLIMATE ULTRA LOW: ~5% literature estimate. Supply cycles are biological, not financial. 5%.
# Relval trimmed 27%→12%: cattle/hog spread less reliable than previously weighted.
# PCR 10%: lean hog options have genuine cyclical sentiment signal.
WEIGHTS_LIVESTOCK = {
    "cot":      0.13,  # LOW: meatpackers/feedlots = mechanical seasonal hedgers, no edge
    "seasonal": 0.27,  # TIER 1 RAISED: biological lock makes this the most reliable seasonal
    "momentum": 0.25,  # Supply cycle moves are fast; price action follows biology
    "macro":    0.08,  # Ultra-low: >95% physical; consumer spending marginal
    "regime":   0.05,  # Ultra-low: biological supply cycles, not financial cycles
    "relval":   0.12,  # Trimmed: cattle/hog spread less robust than previously estimated
    "pcr":      0.10,  # Lean hog options have cyclical sentiment signal (HE options active)
}

# ── BOND FUTURES (ZB 30yr, ZN 10yr, ZF 5yr, ZT 2yr) ────────────────────────────────────
# COT 5.5/10 (FOMC policy announcements create large non-informational position
#   changes; bank duration hedging overlaps with liability management). Trim 18%→13%.
# MACRO 30%: macro explains the largest share of bond returns of any asset class
#   (~40% of variance per literature). Boosted 25%→30%.
# REGIME 30%: flight-to-safety is the primary non-macro bond driver. Boosted 25%→30%.
# Seasonal 8%: trimmed — ZB seasonal well-documented but subordinate to macro/regime.
WEIGHTS_BONDS = {
    "cot":      0.13,  # FOMC non-informational distortion; bank duration mechanical
    "seasonal": 0.08,  # ZB: 87% MRCI win-rate but subordinate to macro/regime
    "momentum": 0.15,
    "macro":    0.30,  # HIGHEST: ~40% of bond return variance is macro-driven
    "regime":   0.30,  # HIGHEST: flight-to-safety is primary non-macro bond driver
    "relval":   0.04,
    "pcr":      0.00,
}

# ── CRYPTO (BTC, ETH) ───────────────────────────────────────────────────────────────────
# COT 4.5/10 (<10yr history, no genuine commercials, leveraged fund timing exists but
#   unreliable). Cut to 8%.
# REGIME 35%: BTC tracks global M2 liquidity 83% of time over 12m periods, with 0.94
#   correlation over full 2013-2024 dataset (Lyn Alden/Sam Callahan 2024). DOMINANT factor.
#   Boosted from 22%→35% — regime/climate is the primary BTC price driver.
# MACRO 20%: liquidity/M2 channel is macro-adjacent (~20% variance).
# SEASONAL LOW 8%: limited history, patterns not yet reliable.
# PCR 2% (crypto ETF options present but thinner than equity markets).
WEIGHTS_CRYPTO = {
    "cot":      0.08,  # Very low: no genuine commercials; <10yr history; slashed
    "seasonal": 0.08,  # LOW: insufficient history for reliable seasonal patterns
    "momentum": 0.22,
    "macro":    0.20,  # M2/liquidity channel is real and macro-adjacent
    "regime":   0.35,  # DOMINANT: BTC tracks global M2 83% of time (0.94 corr, Alden 2024)
    "relval":   0.05,
    "pcr":      0.02,
}

# ── ICE EUROPE THIN-DATA (Z FTSE100, R Long Gilt) ───────────────────────────────────────
# COT cut to 8%: <156w history makes Briese percentile unreliable; halved from prior 12%.
# Z and R are equity/bond-like in character → macro/regime each boosted to 27%.
# Seasonal trimmed to 10% (insufficient history to validate calendar patterns).
WEIGHTS_ICE_THIN = {
    "cot":      0.08,  # Low: thin data (<156w) makes Briese percentile unreliable
    "seasonal": 0.10,  # Trimmed: insufficient history to validate calendar patterns
    "momentum": 0.22,
    "macro":    0.27,  # Bond/equity-like: macro is the primary driver
    "regime":   0.27,  # Flight-to-safety / RORO critical for both Z and R
    "relval":   0.06,
    "pcr":      0.00,
}

BIAS_LABELS = {
    (1.3, 2.0):   ("Very Bullish",    "#22c55e"),
    (0.7, 1.3):   ("Bullish",         "#4ade80"),
    (0.25, 0.7):  ("Mildly Bullish",  "#86efac"),
    (-0.25, 0.25):("Neutral",         "#94a3b8"),
    (-0.7, -0.25):("Mildly Bearish",  "#fca5a5"),
    (-1.3, -0.7): ("Bearish",         "#f87171"),
    (-2.0, -1.3): ("Very Bearish",    "#ef4444"),
}

# ── Module-level COT v2 signal keys tuple ─────────────────────────────────────
# Defined once here and referenced in the main scoring loop + DX feedback loop
# to avoid duplication and guarantee both sites always use the same key set.
_COT_V2_SIGNAL_KEYS: tuple = (
    "divergence", "exhaustion", "comm_momentum_signal", "oi_signal",
    "alignment", "signal_detail", "turning", "lspec_chg_3w",
    "normalise_signal", "convergence_signal", "flatten_signal",
    "v2_signal_dir", "v2_c_best_dir", "v2_c_best_cons", "v2_l_dir_8", "v2_l_cons_8",
    "v2_spec_turn_confirmed", "v2_spec_turn_dir", "v2_spec_weeks_since",
    "v2_comm_turn_dir", "v2_comm_weeks_since",
    "v2_consistency", "v2_spec_turn_strength",
    "v2_phase_coherence", "v2_level_mult", "v2_raw_signal", "v2_shift",
)

# ── Weight-map router — single source of truth ─────────────────────────────────
# Called from compute_engine_bias, the main scoring loop, the DX feedback loop,
# and score_history. Previously the 30-line if/elif chain was copy-pasted four
# times; any future weight tier change now only needs editing here.
_FX_MARKET_IDS = frozenset({
    "6E", "6J", "6B", "6A", "6C", "6N", "6S", "6M", "DX",
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "AUDJPY", "AUDCAD", "AUDNZD", "AUDCHF",
    "CADJPY", "NZDJPY", "NZDCAD", "CHFJPY",
})

# Computed cross-pairs: proxied COT, elevated macro sensitivity
_FX_CROSS_IDS = frozenset({
    "EURJPY", "EURGBP", "EURAUD", "EURCAD", "EURNZD", "EURCHF",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF",
    "AUDJPY", "AUDCAD", "AUDNZD", "AUDCHF",
    "CADJPY", "NZDJPY", "NZDCAD", "CHFJPY",
})

def _get_weight_map(market_id: str) -> dict:
    """
    Return the correct weight map dict for a given market_id.
    Priority: ICE thin → per-asset specific → FX crosses → FX majors → fallback.
    """
    mid = market_id.upper()
    if mid in {"Z", "R"}:                       return WEIGHTS_ICE_THIN
    if mid in {"ES", "NQ", "YM", "RTY"}:        return WEIGHTS_EQUITY
    if mid == "GC":                              return WEIGHTS_GOLD
    if mid == "SI":                              return WEIGHTS_SILVER
    if mid == "HG":                              return WEIGHTS_HG
    if mid == "PA":                              return WEIGHTS_PA
    if mid == "PL":                              return WEIGHTS_PL
    if mid == "CL":                              return WEIGHTS_CRUDE
    if mid == "NG":                              return WEIGHTS_NATGAS
    if mid in {"ZB", "ZN", "ZF", "ZT"}:         return WEIGHTS_BONDS
    if mid in {"ZC", "ZS", "ZW"}:               return WEIGHTS_GRAINS
    if mid == "KC":                              return WEIGHTS_COFFEE
    if mid == "CC":                              return WEIGHTS_COCOA
    if mid in {"SB", "CT"}:                     return WEIGHTS_SOFTS
    if mid in {"HE", "LE", "GF"}:               return WEIGHTS_LIVESTOCK
    if mid in {"BTC", "ETH"}:                   return WEIGHTS_CRYPTO
    if mid in {"B", "GO", "HO", "RB"}:          return WEIGHTS_CRUDE
    if mid == "RC":                              return WEIGHTS_COFFEE
    if mid in _FX_CROSS_IDS:                    return WEIGHTS_FX_CROSSES
    if mid in _FX_MARKET_IDS:                   return WEIGHTS_FX
    return WEIGHTS  # Fallback


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATED SCORING ENGINE (v3) — trend-gated confluence, regime gate, MTF, tiers
# ───────────────────────────────────────────────────────────────────────────
# Replaces the linear weighted-average (which collapsed everything to Neutral and
# could not surface shorts). Validated out-of-sample on ~20yr futures data:
# IC +0.066 vs +0.035 for the weighted average; expectancy harness 48% win,
# PF 1.62, +0.21R/trade. See BH_Backtest_Findings_R1.md and
# BH_Weather_System_FINAL_Scoring_and_Logic.md.
#
# Core ideas:
#   1) DIRECTIONAL BACKDROP = trend-gated confluence of the factors that actually
#      carry directional edge (regime, macro, seasonality). Factors with no clear
#      view ABSTAIN — they do not vote Neutral (the bug that flat-lined the old tool).
#   2) CONVICTION = net count of trend-aligned factors agreeing, x regime gate.
#   3) REGIME GATE = Kaufman efficiency ratio — act big in trends, stand aside in chop.
#   4) MULTI-TIMEFRAME = long-term trend sets the stable bias; short-term
#      counter-move is an entry-watch zone (for the user's supply/demand entries).
#   5) SETUP / R-R = COT positioning + relval + PCR/VIX define a setup-QUALITY
#      tier, NOT a directional vote. COT is judged by risk/reward, not hit-rate.
# ═══════════════════════════════════════════════════════════════════════════
_ENG_ER_CHOPPY, _ENG_ER_TREND = 0.10, 0.35   # efficiency-ratio gate bounds
_ENG_GATE_FLOOR = 0.30                        # min conviction multiplier in chop
_ENG_TIER_CUTS = {"Strong": 3.0, "Setup": 1.5, "Watch": 0.5}   # |conviction| cuts
_ENG_DEADBAND = 0.5    # |score-5| below this ⇒ factor abstains (no directional view)

# Which factor families are trusted per market group (Rounds 7-9):
#   commodity pairs (metal/energy) mean-revert on relval BOTH sides; everything
#   else only counts relval when it is trend-aligned. PCR only for equities.
_ENG_MARKET_GROUP = {
    "ES":"equity","NQ":"equity","YM":"equity","RTY":"equity","Z":"equity",
    "GC":"metal","SI":"metal","HG":"metal","PL":"metal","PA":"metal",
    "CL":"energy","B":"energy","NG":"energy","RB":"energy","HO":"energy","GO":"energy",
    "ZB":"rates","ZN":"rates","ZF":"rates","ZT":"rates","R":"rates",
    "6E":"fx","6J":"fx","6B":"fx","6A":"fx","6C":"fx","6N":"fx","6S":"fx","6M":"fx","DX":"fx",
    "ZS":"ag","ZC":"ag","ZW":"ag","CC":"ag","KC":"ag","RC":"ag","SB":"ag","CT":"ag",
    "LE":"ag","HE":"ag","GF":"ag",
}

def _eng_signed(score, deadband: float = _ENG_DEADBAND):
    """0-10 factor score → signed directional value, or None to ABSTAIN.
    Near-neutral scores abstain rather than dragging the composite to Neutral."""
    if score is None:
        return None
    try:
        d = float(score) - 5.0
    except (TypeError, ValueError):
        return None
    return None if abs(d) < deadband else d

def _eng_sign(x) -> int:
    if x is None:
        return 0
    try:
        if x > 0: return 1
        if x < 0: return -1
    except TypeError:
        return 0
    return 0

def _eng_regime_gate(er: float) -> float:
    return float(max(_ENG_GATE_FLOOR, min(1.0,
        (er - _ENG_ER_CHOPPY) / (_ENG_ER_TREND - _ENG_ER_CHOPPY))))

def _eng_regime_label(er: float) -> str:
    if er >= _ENG_ER_TREND:  return "Trending"
    if er <= _ENG_ER_CHOPPY: return "Choppy"
    return "Mixed"

def _eng_trend_state(lt: int, st: int) -> str:
    if lt > 0 and st < 0: return "Uptrend · short-term pullback (entry-watch for longs)"
    if lt > 0 and st > 0: return "Uptrend · momentum aligned (continuation)"
    if lt > 0:            return "Uptrend · short-term flat"
    if lt < 0 and st > 0: return "Downtrend · short-term bounce (entry-watch for shorts)"
    if lt < 0 and st < 0: return "Downtrend · momentum aligned (continuation)"
    if lt < 0:            return "Downtrend · short-term flat"
    return "Rangebound · no clear trend"

def _eng_setup_quality(grp: str, spec_idx, comm_idx, spec_chg, price_confirm, cot_sign):
    """COT setup-quality / risk-reward read. The COT setup has its OWN direction
    (side WITH commercials / fade the crowd) which can differ from the macro
    backdrop — so we surface it explicitly. This is the user's primary R/R engine:
    judged by risk/reward (tight stop, let winners run), NOT hit-rate.
    Returns (tier_label, setup_dir, list_of_driver_strings)."""
    drv = []
    if spec_idx is None or comm_idx is None:
        return "n/a", 0, drv
    spec_extreme = abs(spec_idx - 50) >= 30     # large specs crowded one side
    comm_extreme = abs(comm_idx - 50) >= 30     # commercials heavily loaded one side
    # An extreme on EITHER side of the market counts — the user's framework follows
    # the commercials, so comms heavily loaded (e.g. CHF comms 83 net long) is a valid
    # setup even when the spec index sits just inside the crowd threshold.
    if not (spec_extreme or comm_extreme):
        return "no positioning extreme", 0, drv
    # side with commercials; if cot score is exactly neutral at an extreme,
    # fall back to fading the crowd (crowded long → bearish setup, & vice-versa)
    setup_dir = cot_sign
    if setup_dir == 0:
        if spec_extreme: setup_dir = -1 if spec_idx >= 70 else 1
        else:            setup_dir = 1 if comm_idx >= 50 else -1
    dword = "bullish" if setup_dir > 0 else "bearish" if setup_dir < 0 else ""
    if spec_extreme:
        crowd = "specs crowded long" if spec_idx >= 70 else "specs crowded short"
    else:
        crowd = "commercials heavily net long" if comm_idx >= 50 else "commercials heavily net short"
    diverging = (spec_extreme and spec_chg is not None and
                 ((spec_idx >= 70 and spec_chg < 0) or (spec_idx <= 30 and spec_chg > 0)))
    if not price_confirm:
        drv.append(f"COT {dword} setup ({crowd}) — price not yet confirming, wait (don't fight price)")
        return "ripe, unconfirmed", setup_dir, drv
    if diverging:
        drv.append(f"COT {dword} setup: {crowd} + crowd unwinding + price confirming — high-quality R/R")
        return "high-quality", setup_dir, drv
    drv.append(f"COT {dword} setup ({crowd}) + price confirming")
    return "confirmed", setup_dir, drv


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT-COMPOSITE — GOLDILOCKS ENTRY-TIMING TILT ON THE SEASONAL FACTOR
# ───────────────────────────────────────────────────────────────────────────
# Ben's Read B: "if all other fundamentals are strong AND we're close to a good
# seasonal entry, I still want a positive-leaning score, but with a clear caveat
# saying almost ready seasonally."
#
# The engine only ever sees the SIGN of the seasonal score (|score-5| must clear
# _ENG_DEADBAND = 0.5 or the factor abstains). So a market sitting at a flat 5.0
# because its goldilocks entry is 8 trading days away contributes NOTHING to the
# composite — the single most actionable seasonal state in Ben's workflow is the
# one the composite is blind to. This tilt fixes that, and its mirror image:
# a goldilocks entry we MISSED by 1-3 weeks is a decaying edge, so the seasonal
# factor is pulled back toward neutral rather than left at full strength.
#
# Guard rails (all deliberate, all tested):
#   * BOOST only fires when goldilocks_dir MATCHES the lean of the OTHER factors
#     (regime + macro + COT + momentum + relval + PCR). It confirms, never leads.
#   * The other-factor lean is computed BEFORE the tilt, so seasonality can never
#     self-confirm.
#   * Magnitude capped at _SEAS_TILT_BOOST_CAP (0.75) and scaled by proximity,
#     goldilocks peak strength, and the market's own reliability dampen — a
#     grade-D mirage barely moves.
#   * NEVER INVERTS: the tilt can never carry the seasonal score across 5.0. If it
#     would, the score is clamped AT 5.0 (i.e. the factor abstains) — honest, and
#     it preserves Ben's "never invert the composite score" rule at the factor level.
#   * Result is clipped to [0, 10].
#
# HOW THE TILT REACHES THE COMPOSITE — and why it is NOT a vote flip
# ─────────────────────────────────────────────────────────────────────
# The engine reads factors through _eng_signed(), which ABSTAINS below a 0.5
# deadband and otherwise casts a full ±1 vote. Feeding the tilted score straight
# back into that produces a STEP FUNCTION: a +0.45 tilt does literally nothing
# (5.45 still abstains) while a +0.51 tilt buys a whole extra edge vote, worth
# ~+0.8 on the headline. Neither is "a positive lean with a caveat" — one is
# invisible, the other over-drives.
#
# So the tilt is transmitted as a small CONVICTION CREDIT in bias-aligned units
# instead: continuous, capped, and it cannot manufacture or flip a direction
# (conviction = bias x raw_conv x gate, so a zero bias stays zero). The tilted
# score itself is still published as `seasonal_score_adj` for the UI/audit trail,
# but the FACTOR VOTE stays anchored on the untilted score — timing informs
# conviction, it never invents a directional view.
# ═══════════════════════════════════════════════════════════════════════════
_SEAS_TILT_BOOST_CAP = 0.75    # max points added toward goldilocks_dir
_SEAS_TILT_DECAY_CAP = 0.60    # max points pulled back toward neutral when late
_SEAS_TILT_MIN_TD    = 3       # inside 3 TD we are effectively AT the entry
_SEAS_TILT_MAX_TD    = 10      # beyond 10 TD it is context, not a lean
_SEAS_DECAY_MIN_TD   = 3       # missed by <3 TD = still fine, no decay
_SEAS_DECAY_MAX_TD   = 15      # missed by >15 TD = fully decayed
_SEAS_TILT_CONV_K    = 0.80    # tilt points → raw-conviction points
#   → max boost credit 0.75 x 0.80 = +0.60 raw_conv (≤ +0.6 on the headline after
#     the regime gate); max decay credit 0.60 x 0.80 = -0.48. Deliberately smaller
#     than one edge vote (1.0) so a timing caveat can never outrank a real factor.


def _eng_other_lean(regime_z, macro_z, cot_z, rv_z, pcr_z, mom_z,
                    momentum_detail: dict = None) -> int:
    """Directional lean of every factor EXCEPT seasonality, as -1 / 0 / +1.
    Edge factors (regime, macro) carry full weight; the confirmation factors
    (COT, momentum, relval, PCR) carry half — same hierarchy the conviction calc
    uses, so the gate agrees with the engine's own view of what matters."""
    momentum_detail = momentum_detail or {}
    _mtf = momentum_detail.get("mtf_vote")
    if _mtf is None:
        _mtf = float(_eng_sign(mom_z))
    net = (1.0 * _eng_sign(regime_z) + 1.0 * _eng_sign(macro_z)
           + 0.5 * _eng_sign(cot_z) + 0.5 * float(_mtf)
           + 0.5 * _eng_sign(rv_z) + 0.5 * _eng_sign(pcr_z))
    # Require a real lean, not a 0.5 whisper, before we let it confirm anything.
    if net >= 1.0:  return 1
    if net <= -1.0: return -1
    return 0


def _seas_timing_tilt(seas_score, seas_detail: dict, other_dir: int):
    """Apply the goldilocks entry-timing tilt to the seasonal factor score.
    Returns (adjusted_score, hint_string, meta_dict). Never inverts; clips 0-10."""
    meta = {"tilt": 0.0, "mode": None, "other_dir": int(other_dir or 0)}
    if seas_score is None or not isinstance(seas_detail, dict) or not seas_detail:
        return seas_score, None, meta
    try:
        base = float(seas_score)
    except (TypeError, ValueError):
        return seas_score, None, meta

    timing = seas_detail.get("entry_timing")
    dtg    = seas_detail.get("days_to_goldilocks")
    gdir   = seas_detail.get("goldilocks_dir")
    lean   = seas_detail.get("goldilocks_lean")
    dampen = seas_detail.get("dampen_factor")
    dampen = 1.0 if dampen is None else max(0.0, min(1.0, float(dampen)))
    meta.update({"entry_timing": timing, "days_to_goldilocks": dtg, "goldilocks_dir": gdir})

    if timing in (None, "none") or dtg is None:
        return round(base, 2), None, meta

    _dirword = "long" if (gdir or 0) > 0 else "short" if (gdir or 0) < 0 else "directionless"
    tilt = 0.0
    hint = None

    # ── APPROACHING: goldilocks entry 1-10 TD out, other factors agree ─────
    if gdir in (1, -1) and 0 <= dtg <= _SEAS_TILT_MAX_TD:
        prox     = max(0.1, min(1.0, (_SEAS_TILT_MAX_TD + 1 - dtg) / float(_SEAS_TILT_MAX_TD)))
        strength = min(1.0, abs(float(lean)) / 3.0) if lean else 0.5
        strength = max(0.30, strength)
        if other_dir != 0 and other_dir == gdir:
            tilt = gdir * min(_SEAS_TILT_BOOST_CAP,
                              _SEAS_TILT_BOOST_CAP * prox * strength * max(0.40, dampen))
            meta["mode"] = "approaching_confirmed"
            hint = (f"Seasonal goldilocks in {dtg} TD (~{dtg/5.0:.1f}wk) — "
                    f"{_dirword} direction confirms other factors")
        else:
            meta["mode"] = "approaching_unconfirmed"
            hint = (f"Seasonal goldilocks in {dtg} TD (~{dtg/5.0:.1f}wk), {_dirword} — "
                    f"other factors not aligned, no composite tilt applied")

    # ── PAST: entry missed by 3-15 TD — the edge is decaying ───────────────
    elif dtg <= -_SEAS_DECAY_MIN_TD:
        span  = float(_SEAS_DECAY_MAX_TD - _SEAS_DECAY_MIN_TD)
        decay = max(0.0, min(1.0, (abs(dtg) - _SEAS_DECAY_MIN_TD) / span))
        pull  = _SEAS_TILT_DECAY_CAP * decay
        _sgn  = 1 if base > 5.0 else -1 if base < 5.0 else 0
        if _sgn != 0 and pull > 0:
            tilt = -_sgn * min(pull, abs(base - 5.0))
            meta["mode"] = "past_decayed"
        else:
            meta["mode"] = "past_flat"
        hint = (f"Past optimal seasonal entry — {abs(dtg)} TD ago "
                f"(~{abs(dtg)/5.0:.1f}wk), {_dirword}; edge decaying")

    # ── EARLY / NOW / JUST_PASSED: caveat only, no tilt ────────────────────
    elif timing == "early" and gdir in (1, -1):
        meta["mode"] = "early"
        hint = (f"Seasonal goldilocks {dtg} TD out (~{dtg/5.0:.1f}wk), {_dirword} — "
                f"too early to lean on")
    elif timing == "now":
        meta["mode"] = "now"
        hint = f"Seasonal goldilocks {_dirword} entry is NOW"
    elif timing == "just_passed":
        meta["mode"] = "just_passed"
        hint = f"Seasonal goldilocks {_dirword} entry {abs(dtg)} TD ago — still live"

    adj = base + tilt
    # NEVER INVERT: the tilt may not carry the factor across neutral.
    if base > 5.0 and adj < 5.0:   adj = 5.0
    elif base < 5.0 and adj > 5.0: adj = 5.0
    adj = max(0.0, min(10.0, adj))
    meta["tilt"] = round(adj - base, 2)
    return round(adj, 2), hint, meta


def compute_engine_bias(scores: dict, market_id: str = "",
                        cot_detail: dict = None, momentum_detail: dict = None,
                        seas_detail: dict = None) -> dict:
    """Validated trend-gated confluence engine. Drop-in replacement for
    compute_weighted_bias — returns a SUPERSET of its keys (weighted/bias/color)
    plus the engine state (direction/conviction/tier/regime/trend_state/
    setup_quality/drivers/factor_votes) consumed by the redesigned UI."""
    cot_detail = cot_detail or {}
    momentum_detail = momentum_detail or {}
    mid = (market_id or "").upper()
    grp = _ENG_MARKET_GROUP.get(mid, "other")

    # ── signed factor values (None = abstain) ──────────────────────────────
    seas_z   = _eng_signed(scores.get("seasonal"))
    regime_z = _eng_signed(scores.get("regime"))
    macro_z  = _eng_signed(scores.get("macro"))
    mom_z    = _eng_signed(scores.get("momentum"))
    cot_z    = _eng_signed(scores.get("cot"))
    rv_z     = _eng_signed(scores.get("relval"))
    pcr_z    = _eng_signed(scores.get("pcr")) if ("pcr" in scores) else None

    # ── trend signs (multi-timeframe) from momentum detail ─────────────────
    # Prefer the true weekly horizons (roc_lt_pct ~26wk, roc_st_pct ~4wk); the daily
    # roc26w/roc4w are factor-calibrated misnomers, so fall back to them only if needed.
    roc26 = momentum_detail.get("roc_lt_pct", momentum_detail.get("roc26w_pct"))
    roc4  = momentum_detail.get("roc_st_pct", momentum_detail.get("roc4w_pct"))
    trend_lt = (1 if (roc26 is not None and roc26 > 2.0) else
                -1 if (roc26 is not None and roc26 < -2.0) else 0)
    if trend_lt == 0 and momentum_detail.get("sma200_above") is not None:
        spd = momentum_detail.get("sma200_pct_diff", 0) or 0
        trend_lt = 1 if spd > 3 else -1 if spd < -3 else 0
    trend_st = (1 if (roc4 is not None and roc4 > 1.0) else
                -1 if (roc4 is not None and roc4 < -1.0) else 0)
    er = momentum_detail.get("efficiency_ratio")
    if er is None:
        er = 0.20

    # ── AUDIT-COMPOSITE: goldilocks entry-timing tilt on the seasonal factor ──
    # other_dir is computed from the NON-seasonal factors only, and BEFORE the
    # tilt is applied, so seasonality can never confirm itself. See the block
    # comment above _seas_timing_tilt for the full rationale + guard rails.
    _other_dir = _eng_other_lean(regime_z, macro_z, cot_z, rv_z, pcr_z, mom_z,
                                 momentum_detail)
    _seas_raw_score = scores.get("seasonal")
    _seas_adj_score, seasonal_hint, _seas_tilt_meta = _seas_timing_tilt(
        _seas_raw_score, seas_detail, _other_dir)
    # NOTE: seas_z is deliberately NOT recomputed from the tilted score — see the
    # "HOW THE TILT REACHES THE COMPOSITE" note above. The tilt lands as a capped
    # conviction credit further down, so it can never flip a factor vote.
    _seas_tilt_pts = float(_seas_tilt_meta.get("tilt") or 0.0)
    _seas_tilt_dir = _eng_sign(_seas_tilt_pts)

    # ── 1) DIRECTIONAL BACKDROP — only edge-bearing factors, abstain-aware ──
    bd_vals = [v for v in (seas_z, regime_z, macro_z) if v is not None]
    backdrop = (sum(bd_vals) / len(bd_vals)) if bd_vals else 0.0
    bias = _eng_sign(backdrop)

    # ── 2) CONFLUENCE (COT ungated as of r14) ──────────────────────────────
    # COT no longer gated by trend — walk-forward on 10yr showed the gate
    # muted the loudest factor exactly when it mattered most (Cotton IC=-0.27,
    # Sugar -0.29, Coffee -0.23, Copper -0.09..-0.18 at 4wk fwd). COT votes at
    # its natural sign now, period. User directive: "just listen to cot".
    cot_vote = _eng_sign(cot_z)
    if grp in ("metal", "energy"):
        rv_vote = _eng_sign(rv_z)
    else:
        rv_vote = _eng_sign(rv_z) if (_eng_sign(rv_z) != 0 and _eng_sign(rv_z) == trend_lt) else 0
    pcr_vote = _eng_sign(pcr_z) if grp == "equity" else 0

    # ── r15: momentum uses multi-timeframe majority-rule vote ──────────────
    # mtf_vote from score_momentum() is ±1.0 (all 3 TFs agree), ±0.5 (2 of 3),
    # or 0 (split / whipsaw). We keep the raw ±1 sign for the votes dict
    # (used by agree/disagree UI counters) but use the fractional value below
    # in the conf_net conviction calc, so a mixed-timeframe momentum reads as
    # partial confirmation rather than a full vote in either direction.
    _mtf_vote_raw = momentum_detail.get("mtf_vote")
    if _mtf_vote_raw is None:
        _mtf_vote_raw = float(_eng_sign(mom_z))   # fallback if detail missing
    _mom_vote_int = 1 if _mtf_vote_raw > 0.25 else -1 if _mtf_vote_raw < -0.25 else 0

    votes = {
        "seasonal": _eng_sign(seas_z), "regime": _eng_sign(regime_z), "macro": _eng_sign(macro_z),
        "momentum": _mom_vote_int, "cot": cot_vote, "relval": rv_vote, "pcr": pcr_vote,
    }
    agree    = sum(1 for v in votes.values() if bias != 0 and v == bias)
    disagree = sum(1 for v in votes.values() if bias != 0 and v == -bias)
    # CONVICTION is driven by the EDGE-bearing factors (seasonal / regime / macro — the
    # only factors with validated standalone IC: +0.034 / +0.046 / +0.036 on a 14-market
    # walk-forward). The remaining factors (momentum / cot / relval / pcr — ~0 IC alone)
    # act as HALF-WEIGHT CONFIRMATION: they can reinforce a genuine edge-factor signal but
    # cannot manufacture a high-conviction tier on their own.
    # WHY: with all-equal votes the "Strong" tier was the WEAKEST cohort (+0.17% fwd,
    # because zero-IC factors inflated conviction); edge-weighted conviction makes Strong
    # the BEST cohort (+1.18% fwd, 57% hit) — i.e. the tiers now actually rank edge.
    _EDGE_F = ("seasonal", "regime", "macro")
    _CONF_F_NON_MOM = ("cot", "relval", "pcr")
    edge_net = (sum(1 for f in _EDGE_F if bias != 0 and votes[f] == bias)
                - sum(1 for f in _EDGE_F if bias != 0 and votes[f] == -bias))
    conf_net_non_mom = (sum(1 for f in _CONF_F_NON_MOM if bias != 0 and votes[f] == bias)
                        - sum(1 for f in _CONF_F_NON_MOM if bias != 0 and votes[f] == -bias))
    # Momentum contributes its full fractional MTF vote (-1.0 / -0.5 / 0 / +0.5 / +1.0)
    # aligned with the current bias direction. When bias is bullish (=+1), a +0.5 MTF
    # vote contributes +0.5, a -0.5 contributes -0.5, etc.
    mom_conf = _mtf_vote_raw * bias if bias != 0 else 0.0
    conf_net = conf_net_non_mom + mom_conf
    raw_conv = edge_net + 0.5 * conf_net

    # ── AUDIT-COMPOSITE: goldilocks timing credit (capped, cannot cross zero) ──
    # Positive when the seasonal tilt points the same way as the composite bias
    # ("almost ready seasonally, and it agrees"), negative when a goldilocks entry
    # that USED to support the bias has decayed past its sweet spot. Clamped so it
    # can never carry raw_conv across zero — i.e. a timing caveat can soften or
    # firm up a lean, but it can NEVER invert the composite. Ben's rule holds.
    seas_timing_conv = 0.0
    if bias != 0 and _seas_tilt_dir != 0:
        seas_timing_conv = (_SEAS_TILT_CONV_K * abs(_seas_tilt_pts)
                            * (1.0 if _seas_tilt_dir == bias else -1.0))
        _rc_before = raw_conv
        if _rc_before >= 0:
            raw_conv = max(0.0, _rc_before + seas_timing_conv)
        else:
            raw_conv = min(0.0, _rc_before + seas_timing_conv)
        seas_timing_conv = round(raw_conv - _rc_before, 3)

    # ── 3) REGIME GATE — scale conviction by trend quality ─────────────────
    gate = _eng_regime_gate(er)
    conviction = round(bias * raw_conv * gate, 2)
    a = abs(conviction)
    tier = ("Strong" if a >= _ENG_TIER_CUTS["Strong"] else
            "Setup"  if a >= _ENG_TIER_CUTS["Setup"]  else
            "Watch"  if a >= _ENG_TIER_CUTS["Watch"]  else "Neutral")
    direction = "Bullish" if bias > 0 else "Bearish" if bias < 0 else "Neutral"

    # ── 4) MULTI-TIMEFRAME trend state ─────────────────────────────────────
    trend_state = _eng_trend_state(trend_lt, trend_st)

    # ── 5) SETUP QUALITY (R-R) from positioning + price confirmation ───────
    spec_idx = cot_detail.get("lspec_index")
    comm_idx = cot_detail.get("comm_index")
    spec_chg = cot_detail.get("lspec_chg_3w")
    # COT direction uses the raw score sign (no deadband) — at an extreme the score
    # is decisive, and we want the setup's own direction even if the backdrop abstains.
    _cot_raw = scores.get("cot")
    cot_sign = _eng_sign((_cot_raw - 5.0)) if _cot_raw is not None else 0
    price_confirm = (trend_st != 0 and trend_st == cot_sign)
    setup_quality, setup_dir, setup_drivers = _eng_setup_quality(
        grp, spec_idx, comm_idx, spec_chg, price_confirm, cot_sign)
    if setup_dir == 0:      setup_vs_backdrop = "none"
    elif bias == 0:         setup_vs_backdrop = "standalone"   # backdrop neutral → COT is the signal
    elif setup_dir == bias: setup_vs_backdrop = "aligned"      # COT confirms backdrop → strongest
    else:                   setup_vs_backdrop = "counter"      # COT fades backdrop → mean-reversion watch
    setup_direction = "Bullish" if setup_dir > 0 else "Bearish" if setup_dir < 0 else "Neutral"

    # ── human-readable drivers ─────────────────────────────────────────────
    _names = {"seasonal":"Seasonality","regime":"Risk regime","macro":"Macro",
              "momentum":"Momentum","cot":"COT (trend-aligned)","relval":"Relative value",
              "pcr":"Sentiment"}
    drivers = []
    for k, v in votes.items():
        if v > 0:   drivers.append(f"{_names[k]}: bullish")
        elif v < 0: drivers.append(f"{_names[k]}: bearish")
    drivers += setup_drivers
    # AUDIT-COMPOSITE: surface the seasonal timing caveat in the driver list too,
    # so it shows up in the existing UI driver rail without a frontend change.
    if seasonal_hint:
        drivers.append(seasonal_hint)

    # ── climate score (0-10) for the gauge — derived from bias x conviction ─
    weighted = round(max(0.2, min(9.8, 5.0 + conviction)), 2)

    bias_lbl = "Neutral"; color = "#94a3b8"
    if   weighted >= 8.0:  bias_lbl = "Very Bullish";    color = "#22c55e"
    elif weighted >= 7.0:  bias_lbl = "Bullish";         color = "#4ade80"
    elif weighted >= 6.2:  bias_lbl = "Mildly Bullish";  color = "#86efac"
    elif weighted >= 5.5:  bias_lbl = "Lean Bullish";    color = "#a7f3d0"
    elif weighted >= 4.5:  bias_lbl = "Neutral";         color = "#94a3b8"
    elif weighted >= 3.8:  bias_lbl = "Lean Bearish";    color = "#fde68a"
    elif weighted >= 3.0:  bias_lbl = "Mildly Bearish";  color = "#fca5a5"
    elif weighted >= 2.0:  bias_lbl = "Bearish";         color = "#f87171"
    else:                  bias_lbl = "Very Bearish";    color = "#ef4444"

    return {
        "weighted":          weighted,
        "bias":              bias_lbl,
        "color":             color,
        "confluence_bonus":  0.0,
        "confluence_reason": None,
        "direction":         direction,
        "bias_sign":         bias,
        "conviction":        conviction,
        "tier":              tier,
        "regime":            _eng_regime_label(er),
        "efficiency_ratio":  round(float(er), 3),
        "regime_gate":       round(gate, 3),
        "trend_lt":          trend_lt,
        "trend_st":          trend_st,
        "trend_state":       trend_state,
        "setup_quality":     setup_quality,
        "setup_direction":   setup_direction,
        "setup_vs_backdrop": setup_vs_backdrop,
        "agree":             agree,
        "disagree":          disagree,
        "factor_votes":      votes,
        "drivers":           drivers,
        # ── AUDIT-COMPOSITE: seasonal goldilocks entry-timing exposure ────────
        # seasonal_hint is the human-readable caveat Ben asked for ("almost ready
        # seasonally"). The rest lets the UI render a chip and lets anyone audit
        # exactly how much the tilt moved the seasonal factor.
        "seasonal_hint":            seasonal_hint,
        "seasonal_score_raw":       (round(float(_seas_raw_score), 2)
                                     if _seas_raw_score is not None else None),
        "seasonal_score_adj":       _seas_adj_score,
        "seasonal_timing_tilt":     _seas_tilt_meta.get("tilt", 0.0),
        "seasonal_timing_mode":     _seas_tilt_meta.get("mode"),
        "seasonal_other_lean":      _other_dir,
        # how many raw-conviction points the timing credit actually contributed
        "seasonal_timing_conv":     seas_timing_conv,
        "entry_timing":             (seas_detail or {}).get("entry_timing"),
        "days_to_goldilocks":       (seas_detail or {}).get("days_to_goldilocks"),
        "goldilocks_dir":           (seas_detail or {}).get("goldilocks_dir"),
        "entry_note":               (seas_detail or {}).get("entry_note"),
        # 10-TD "very near term" read. Deliberately NOT fed into the headline
        # composite (the scoring agent anchors the headline on the 20-TD swing
        # window) — published so the UI can show a very-near-term chip.
        "immediate_score":          (seas_detail or {}).get("imm_score"),
    }


# ============================================================
# MAIN API ENDPOINTS
# ============================================================

# ── Weekly Score Snapshots ────────────────────────────────────────────────────
# Disk-persisted daily snapshots so score deltas survive server restarts.
# Snapshot = {market_id: weighted_score}, stored as JSON in DATA_DIR.
def _save_scores_snapshot(scores_map: dict) -> None:
    """Persist today's {market_id: weighted_score} map to disk."""
    try:
        fname = os.path.join(DATA_DIR, f"scores_snapshot_{date.today().isoformat()}.json")
        with open(fname, "w") as fh:
            json.dump({"saved_at": time.time(), "scores": scores_map}, fh)
        # Auto-prune snapshots older than 14 days
        cutoff = time.time() - 14 * 86400
        for old_f in glob.glob(os.path.join(DATA_DIR, "scores_snapshot_*.json")):
            if os.path.getmtime(old_f) < cutoff:
                try: os.remove(old_f)
                except Exception: pass
    except Exception as _e:
        print(f"[snapshot] save failed: {_e}")

# ── Full-payload scores snapshot (cold-start instant render) ──────────────────
# The full /api/scores payload is persisted to disk after every successful
# refresh. On boot (before the warm-up finishes) it is loaded back so the very
# first visitor gets last-known data INSTANTLY instead of a 202 "warming" blank.
# Survives soft/periodic instance restarts; a full redeploy wipes it but the
# first post-deploy warm-up repopulates it.
_FULL_SNAPSHOT_PATH = os.path.join(DATA_DIR, "scores_full_snapshot.json")
_FULL_SNAPSHOT_MAX_AGE = 24 * 3600  # don't serve a snapshot older than 24h

def _save_full_scores_snapshot(payload: dict) -> None:
    """Persist the complete /api/scores payload to disk (atomic write)."""
    try:
        tmp = _FULL_SNAPSHOT_PATH + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(orjson.dumps({"saved_at": time.time(), "payload": payload},
                                  default=str, option=orjson.OPT_NON_STR_KEYS))
        os.replace(tmp, _FULL_SNAPSHOT_PATH)  # atomic
    except Exception as _e:
        print(f"[snapshot] full save failed: {_e}")

def _load_full_scores_snapshot() -> bool:
    """Load the disk snapshot into ALL_DATA_CACHE if present and fresh enough.
    Marks it STALE (time older than TTL) so a background refresh still fires on
    the first real request. Returns True if a snapshot was loaded."""
    try:
        if not os.path.exists(_FULL_SNAPSHOT_PATH):
            return False
        with open(_FULL_SNAPSHOT_PATH, "rb") as fh:
            blob = orjson.loads(fh.read())
        saved_at = float(blob.get("saved_at", 0))
        payload  = blob.get("payload")
        if not payload:
            return False
        age = time.time() - saved_at
        if age > _FULL_SNAPSHOT_MAX_AGE:
            print(f"[snapshot] full snapshot too old ({age/3600:.1f}h) — skipping", flush=True)
            return False
        ALL_DATA_CACHE["data"] = payload
        # Backdate the timestamp so the cache reads as STALE and the next request
        # triggers a background refresh (stale-while-revalidate).
        ALL_DATA_CACHE["time"] = time.time() - ALL_DATA_TTL - 1
        print(f"[snapshot] full snapshot loaded ({age/60:.0f}min old) — instant cold-start render enabled", flush=True)
        return True
    except Exception as _e:
        print(f"[snapshot] full load failed: {_e}")
        return False

# ------------------------------------------------------------------
# Scores cache + concurrency guard (restored — these module globals
# were inadvertently dropped during an earlier dead-code cleanup)
# ------------------------------------------------------------------
ALL_DATA_CACHE = {"data": None, "time": 0}
ALL_DATA_TTL = 3600  # 60 min — data sources (COT, macro, prices) change at most hourly
# Single async lock prevents cache stampede — when N concurrent requests all
# find an expired cache simultaneously, only ONE recomputes; the rest wait and
# then get the freshly cached result.
_SCORES_LOCK = asyncio.Lock()
_SCORES_BG_RUNNING = False  # True while a background refresh is in flight

async def _refresh_scores_background():
    """Run the full score refresh in the background without blocking any request."""
    global _SCORES_BG_RUNNING
    if _SCORES_BG_RUNNING:
        return  # Already running — don't double-up
    _SCORES_BG_RUNNING = True
    try:
        # Use wait_for on lock acquisition too — don't get stuck behind a deadlock
        try:
            await asyncio.wait_for(_SCORES_LOCK.acquire(), timeout=180)
        except asyncio.TimeoutError:
            print("[scores] BG refresh: lock timeout — skipping", flush=True)
            return
        try:
            # Re-check inside lock — another coroutine may have just refreshed
            if ALL_DATA_CACHE["data"] and (time.time() - ALL_DATA_CACHE["time"]) < ALL_DATA_TTL:
                return
            await asyncio.wait_for(_do_scores_refresh(), timeout=600)
        except asyncio.TimeoutError:
            print("[scores] BG refresh: _do_scores_refresh timed out after 600s", flush=True)
        finally:
            _SCORES_LOCK.release()
    except Exception as _e:
        print(f"[scores] Background refresh error: {_e}", flush=True)
    finally:
        _SCORES_BG_RUNNING = False

@app.head("/api/scores")
async def head_all_scores():
    """HEAD handler for uptime monitors (e.g. UptimeRobot free tier, which only
    sends HEAD). Returns 200 instantly with no body, and fires a NON-BLOCKING
    background refresh if the cache is stale/cold — so the ping actually warms
    the data cache without making the monitor wait for the full build."""
    from fastapi.responses import Response as _Resp
    try:
        now = time.time()
        stale = (not ALL_DATA_CACHE["data"]) or (now - ALL_DATA_CACHE["time"]) >= ALL_DATA_TTL
        if stale and not _SCORES_BG_RUNNING:
            asyncio.ensure_future(_refresh_scores_background())
    except Exception as _e:
        print(f"[scores] HEAD warm trigger error (non-fatal): {_e}", flush=True)
    return _Resp(status_code=200)

# AUDIT-COMPOSITE: `market=` returns the single-market DETAIL payload (same object
# the market list carries, plus the global regime/weights context) instead of all
# 60+ markets. The full payload is ~600KB; this is ~15KB. Purely additive — with
# no `market` param the response is byte-for-byte what it always was, so the
# existing frontend is untouched.
def _scores_market_view(payload: dict, market: str) -> dict:
    mid = (market or "").strip().upper()
    mkts = (payload or {}).get("markets") or []
    hit = next((m for m in mkts if str(m.get("id", "")).upper() == mid), None)
    if hit is None:
        return {"error": f"Unknown market '{market}'",
                "available": [m.get("id") for m in mkts]}
    return {
        "updated_at": (payload or {}).get("updated_at"),
        "market":     hit,
        "regime":     (payload or {}).get("regime"),
        "weights":    (payload or {}).get("weights"),
    }


@app.get("/api/scores")
async def get_all_scores(force: bool = False, market: str = None):
    now = time.time()
    cache_age = now - ALL_DATA_CACHE["time"]
    cache_exists = bool(ALL_DATA_CACHE["data"])

    # AUDIT-COMPOSITE: single exit point so the optional `market=` slice is applied
    # on every return path (fresh cache, stale-while-revalidate, cold compute).
    def _out(_payload):
        return _SafeJSONResponse(_scores_market_view(_payload, market) if market else _payload)

    # Fast path: fresh cache — return immediately
    if not force and cache_exists and cache_age < ALL_DATA_TTL:
        return _out(ALL_DATA_CACHE["data"])

    # Stale-while-revalidate: if we have ANY cached data (even expired), serve it
    # immediately and kick off a background refresh so the user never waits
    if not force and cache_exists:
        print(f"[scores] Cache stale ({cache_age:.0f}s old) — serving stale, refreshing in background", flush=True)
        asyncio.ensure_future(_refresh_scores_background())
        return _out(ALL_DATA_CACHE["data"])

    # If cache is cold and still warming up, return 202 so frontend can poll
    if _WARMING["started"] and not _WARMING["done"] and not force:
        from fastapi.responses import JSONResponse as _JR
        return _JR({"status": "warming", "message": "Cache warming — retry in 10s"}, status_code=202)

    # True cold start (no data at all) — must wait for first result
    print(f"[scores] Cold start — must compute synchronously (no stale data available)", flush=True)
    # Acquire lock with a timeout — if a previous compute is deadlocked, give up
    # after 90s and return whatever stale data we have (or a 503)
    try:
        acquired = await asyncio.wait_for(_SCORES_LOCK.acquire(), timeout=180)
    except asyncio.TimeoutError:
        print("[scores] Lock acquisition timed out (deadlock?) — returning stale/empty", flush=True)
        if ALL_DATA_CACHE["data"]:
            return _out(ALL_DATA_CACHE["data"])
        from fastapi.responses import JSONResponse as _JR
        return _JR({"error": "Server busy — retry in 30s"}, status_code=503)
    try:
        # Re-check cache inside lock: a previous waiter may have already recomputed
        now = time.time()
        if not force and ALL_DATA_CACHE["data"] and (now - ALL_DATA_CACHE["time"]) < ALL_DATA_TTL:
            return _out(ALL_DATA_CACHE["data"])
        if force:
            ALL_DATA_CACHE["data"] = None
            FF_CACHE["data"] = None
            FF_MACRO_CACHE["data"] = None
        # Wrap the refresh with a hard 600s timeout — prevents indefinite hangs on cold start
        try:
            await asyncio.wait_for(_do_scores_refresh(), timeout=600)
        except asyncio.TimeoutError:
            print("[scores] _do_scores_refresh timed out after 600s — serving stale", flush=True)
    finally:
        _SCORES_LOCK.release()
    return _out(ALL_DATA_CACHE["data"])

async def _do_scores_refresh(force: bool = False):
    """Core refresh logic — must be called with _SCORES_LOCK already held."""
    _refresh_start = time.time()
    now = time.time()
    print(f"[scores] Cache refresh START", flush=True)

    # ── Parallel price pre-warm ────────────────────────────────────────────────────────────────────────
    # On cold start, 62 markets × fetch_price_data (20s timeout each) = 1240s sequential.
    # Pre-warm all market price caches in parallel before the scoring loop.
    _price_tickers = [m["yf"] for m in MARKETS if m.get("yf")]
    # Also pre-warm RISK_ASSETS tickers so compute_risk_regime() can use cache
    _price_tickers += list(RISK_ASSETS.values())
    _price_tickers_uniq = list(dict.fromkeys(_price_tickers))  # deduplicate preserving order
    def _prewarm_prices():
        _px_ex = _cf.ThreadPoolExecutor(max_workers=15)
        try:
            _px_futs = [_px_ex.submit(fetch_price_data, t) for t in _price_tickers_uniq]
            # Also pre-warm yfinance yield series in parallel (fallback for FRED)
            _yld_tickers = ["^TNX", "^IRX", "^FVX", "^TYX"]
            _px_futs += [_px_ex.submit(_fetch_yf_yield_series, t, 270) for t in _yld_tickers]
            _cf.wait(_px_futs, timeout=40)  # 40s hard cap — run all in parallel
        finally:
            _px_ex.shutdown(wait=False)
        print(f"[scores] Price pre-warm done ({len(_price_tickers_uniq)} tickers + 4 yield tickers)", flush=True)

    _loop = asyncio.get_event_loop()
    # Fire price pre-warm in parallel with compute_macro_all (which fetches FRED)
    # Both complete in ~20s vs 1400s+ sequential
    _prewarm_task = _loop.run_in_executor(_APP_EXECUTOR, _prewarm_prices)

    # Run all sync blocking data-fetch functions in thread executors
    # so the async event loop (and /api/health) remain responsive
    # Run sequentially (not concurrently) to avoid OOM on 2GB instance.
    # Each function is cache-backed (TTL 2h) so the sequential cost is trivial
    # on warm hits. On a cold start only the first call is expensive per function.
    macro    = await _loop.run_in_executor(_APP_EXECUTOR, compute_macro_all)
    await _prewarm_task  # wait for price pre-warm to finish before scoring loop
    _gc_if_heavy("post-macro-all")
    regime   = await _loop.run_in_executor(_APP_EXECUTOR, compute_risk_regime)
    _gc_if_heavy("post-risk-regime")
    stock_climate = await _loop.run_in_executor(_APP_EXECUTOR, compute_stock_climate)
    _gc_if_heavy("post-stock-climate")
    ff_macro = await _loop.run_in_executor(_APP_EXECUTOR, compute_all_ff_macro)
    _gc_if_heavy("post-ff-macro")
    # News context — always use whatever is in cache right now (may be stale/empty).
    # Narrative generation (40 Sonar AI calls, ~22s) is fired in a background thread
    # BEFORE the market scoring loop so it warms in parallel. It never blocks
    # the scores lock — clients get scores immediately after market computation.
    _news_now = time.time()
    _news_cold = not NEWS_CACHE["data"] or (_news_now - NEWS_CACHE["time"]) >= NEWS_CACHE_TTL
    _narr_cold = not NARR_CACHE["data"] or (_news_now - NARR_CACHE["time"]) >= NARR_CACHE_TTL
    if _news_cold or _narr_cold:
        # Fire-and-forget in background — does NOT block market scoring below
        _bg_narr_thread = threading.Thread(
            target=compute_news_context, daemon=True, name="narr-bg-refresh"
        )
        _bg_narr_thread.start()
        print("[narr] Background narrative refresh started (non-blocking)", flush=True)
    # Serve whatever is already cached — frontend fetches /api/news-context separately
    _cached_items  = NEWS_CACHE["data"] if (NEWS_CACHE["data"] and (_news_now - NEWS_CACHE["time"]) < NEWS_CACHE_TTL) else []
    _raw_narrs     = NARR_CACHE["data"] if (NARR_CACHE["data"] and (_news_now - NARR_CACHE["time"]) < NARR_CACHE_TTL) else {}
    _narr_text     = {k: v["text"]     for k, v in _raw_narrs.items() if isinstance(v, dict)}
    _narr_scores   = {k: v["score_10"] for k, v in _raw_narrs.items() if isinstance(v, dict) and v.get("score_10") is not None}
    news_ctx = {
        "narratives":       _narr_text,
        "narrative_scores": _narr_scores,
        "news_items":       _cached_items[:20],
        "global_narrative": None,
        "price_context":    {},
        "updated_at":       NEWS_CACHE["time"] if _cached_items else _news_now,
        "ff_event_count":   len(_cached_items),
    }
    # Separate regular markets from cross pairs
    regular_markets = [m for m in MARKETS if not m.get("cross")]
    cross_markets   = [m for m in MARKETS if m.get("cross")]
    
    # Fetch COT only for regular markets (cross pairs derive from legs)
    # ICE markets use fetch_ice_cot_history; CFTC markets use fetch_cot_history
    async def _fetch_cot_for_market(m):
        if m.get("ice_code"):
            return await fetch_ice_cot_history(m["ice_code"])
        else:
            return await fetch_cot_history(m["cftc_code"], m["name"])
    
    # Also pre-fetch supplementary ICE datasets for cross-market COT blending:
    # CC (NY Cocoa) <- supplemented by ICE London Cocoa ("Cocoa")
    # KC (Arabica)  <- supplemented by ICE Robusta RC
    async def _fetch_ice_london_cocoa():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_APP_EXECUTOR, _fetch_ice_cot_raw, "Cocoa")
    
    async def _fetch_ice_robusta():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_APP_EXECUTOR, _fetch_ice_cot_raw, "RC")
    
    async with httpx.AsyncClient(timeout=20):
        cot_results, ice_london_cocoa_df, ice_robusta_df = await asyncio.gather(
            asyncio.gather(*[_fetch_cot_for_market(m) for m in regular_markets], return_exceptions=True),
            _fetch_ice_london_cocoa(),
            _fetch_ice_robusta(),
        )
    
    def _zscore_blend_cot(primary_df: pd.DataFrame, secondary_df: pd.DataFrame,
                          primary_weight: float = 0.60) -> pd.DataFrame:
        """
        Z-score normalized blend of two COT DataFrames.
    
        Raw contract blending is INVALID when the two exchanges have different:
          - OI scales (NY Cocoa ~200k lots vs London ~100k lots)
          - Participant composition (London 65-73% commercial vs NY 48-50%)
          - Currency denomination (USD vs GBP)
    
        Correct approach (per research):
          1. For each exchange, compute z-scores of comm_net and lspec_net
             within their own full history (rolling 260w / 5yr window).
          2. Blend the NORMALIZED scores (not raw contracts).
          3. Back-convert to synthetic contract counts so downstream code is unaffected.
    
        primary_weight: fraction assigned to primary exchange (e.g. 0.60 for NY Cocoa).
        """
        if secondary_df is None or secondary_df.empty:
            return primary_df
        if primary_df is None or primary_df.empty:
            return secondary_df
        try:
            p = primary_df.copy()
            s = secondary_df.copy()
            p["date"] = pd.to_datetime(p["date"])
            s["date"] = pd.to_datetime(s["date"])
            sw = 1.0 - primary_weight
    
            # ── Z-score each series within its own history ──────────────────
            def rolling_zscore(series: pd.Series, window: int = 260) -> pd.Series:
                """Rolling z-score; min_periods=52 (1yr) for stability."""
                mu  = series.rolling(window, min_periods=52).mean()
                sig = series.rolling(window, min_periods=52).std()
                return (series - mu) / sig.replace(0, 1)
    
            p = p.sort_values("date").reset_index(drop=True)
            s = s.sort_values("date").reset_index(drop=True)
    
            p["comm_z"]  = rolling_zscore(p["comm_net"].astype(float))
            p["lspec_z"] = rolling_zscore(p["lspec_net"].astype(float))
            p["sspec_z"] = rolling_zscore(p["sspec_net"].astype(float))
            s["comm_z"]  = rolling_zscore(s["comm_net"].astype(float))
            s["lspec_z"] = rolling_zscore(s["lspec_net"].astype(float))
            s["sspec_z"] = rolling_zscore(s["sspec_net"].astype(float))
    
            # ── Merge on matching dates ─────────────────────────────────────
            merged = pd.merge(
                p[["date","comm_z","lspec_z","sspec_z","comm_net","open_interest_all"]],
                s[["date","comm_z","lspec_z","sspec_z"]],
                on="date", suffixes=("_p","_s"), how="inner"
            )
            if len(merged) < 52:
                print(f"[COT ZBLEND] only {len(merged)} overlapping dates — skipping blend")
                return primary_df
    
            # ── Blend z-scores ──────────────────────────────────────────────
            merged["comm_z_blended"]  = merged["comm_z_p"]  * primary_weight + merged["comm_z_s"]  * sw
            merged["lspec_z_blended"] = merged["lspec_z_p"] * primary_weight + merged["lspec_z_s"] * sw
            merged["sspec_z_blended"] = merged["sspec_z_p"] * primary_weight + merged["sspec_z_s"] * sw
    
            # ── Back-convert to synthetic contract counts ───────────────────
            # Use primary's own rolling stats so downstream percentile logic is unaffected.
            # blended_z * primary_std + primary_mean ≈ "what primary would look like"
            # if it had the blended signal embedded.
            def zscore_to_contracts(z_blend: pd.Series, raw_primary: pd.Series) -> pd.Series:
                window = 260
                mu  = raw_primary.rolling(window, min_periods=52).mean()
                sig = raw_primary.rolling(window, min_periods=52).std().replace(0, 1)
                result = (z_blend * sig + mu).round(0)
                # Replace any NaN/inf with the raw primary value (safe fallback)
                fallback = raw_primary.round(0)
                result = result.where(result.notna() & np.isfinite(result), other=fallback)
                return result.fillna(0).astype(int)
    
            # Align indices for rolling stats (use primary df aligned to merged dates)
            p_aligned = p[p["date"].isin(merged["date"])].reset_index(drop=True)
            merged = merged.reset_index(drop=True)
    
            merged["comm_net_blended"]  = zscore_to_contracts(merged["comm_z_blended"],  p_aligned["comm_net"].astype(float))
            merged["lspec_net_blended"] = zscore_to_contracts(merged["lspec_z_blended"], p_aligned["lspec_net"].astype(float))
            merged["sspec_net_blended"] = zscore_to_contracts(merged["sspec_z_blended"], p_aligned["sspec_net"].astype(float))
    
            # ── Write blended values back into primary df ───────────────────
            # Vectorised merge — O(n) vs O(n²) iterrows approach.
            out = primary_df.copy()
            out["date"] = pd.to_datetime(out["date"])
            blend_patch = merged[["date", "comm_net_blended", "lspec_net_blended", "sspec_net_blended"]].copy()
            blend_patch["date"] = pd.to_datetime(blend_patch["date"])
            out = out.merge(blend_patch, on="date", how="left", suffixes=("", "_new"))
            for col in ("comm_net", "lspec_net", "sspec_net"):
                new_col = col + "_new"
                if new_col in out.columns:
                    # Only overwrite rows that matched (non-NaN in the patch column)
                    mask = out[new_col].notna()
                    out.loc[mask, col] = out.loc[mask, new_col]
                    out.drop(columns=[new_col], inplace=True)
    
            print(f"[COT ZBLEND] {len(merged)} dates blended via z-score normalization "
                  f"({primary_weight:.0%} primary / {sw:.0%} secondary)")
            return out
        except Exception as _be:
            print(f"[COT ZBLEND] blend failed: {_be}")
            return primary_df
    
    # Build COT cache dict: market_id -> DataFrame (for cross pair derivation)
    cot_df_cache: dict = {}
    for i, market in enumerate(regular_markets):
        df = cot_results[i] if not isinstance(cot_results[i], Exception) else None
        mid = market["id"]
    
        # ── Cross-market COT blending ──────────────────────────────────────
        # CC (NY Cocoa): z-score normalized blend — 60% CFTC NY + 40% ICE London.
        # Raw contract blending is INVALID: London is GBP-denominated, different OI scale
        # (~100k lots vs NY ~200k), different participant composition (London 65-73%
        # commercial vs NY 48-50%). Z-score normalization makes each exchange's signal
        # comparable on a unit-free basis before blending.
        if mid == "CC" and ice_london_cocoa_df is not None:
            df = _zscore_blend_cot(df, ice_london_cocoa_df, primary_weight=0.60)
    
        # KC (Arabica Coffee): NO blending with Robusta.
        # Arabica and Robusta are structurally different commodities with separate commercial
        # bases, supply chains, and participant profiles. Blending raw or even z-scored
        # positions conflates independent supply/demand signals and adds noise, not signal.
        # KC uses pure CFTC Arabica data only.
        # (ice_robusta_df is fetched above and used only for the standalone RC market)
    
        cot_df_cache[market["id"]] = df
    
    def _merge_price_into_cot(cot_df, yf_ticker, mid):
        """Merge weekly price closes into COT df for divergence signals."""
        if cot_df is None or len(cot_df) < 10:
            return cot_df
        try:
            px_df = fetch_price_data(yf_ticker)
            if px_df is None or px_df.empty:
                return cot_df
            px_idx = pd.to_datetime(px_df.index).tz_localize(None).normalize().astype("datetime64[us]")
            px_close = px_df["Close"].values.astype(float)
            price_lookup = pd.DataFrame({"_cot_date": px_idx, "close": px_close})
            price_lookup = price_lookup.sort_values("_cot_date").reset_index(drop=True)
            if "date" in cot_df.columns:
                cot_idx = pd.to_datetime(cot_df["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[us]")
            else:
                cot_idx = pd.to_datetime(cot_df.index).tz_localize(None).normalize().astype("datetime64[us]")
            cot_df = cot_df.copy()
            cot_df["_cot_date"] = cot_idx.values
            merged = pd.merge_asof(
                cot_df.sort_values("_cot_date"),
                price_lookup,
                on="_cot_date",
                direction="nearest",
                tolerance=pd.Timedelta(days=7),
            )
            if "close" in merged.columns:
                return merged.drop(columns=["_cot_date"])
            else:
                return cot_df.drop(columns=["_cot_date"])
        except Exception as _px_err:
            print(f"Price merge warning for {mid}: {_px_err}")
            return cot_df
    
    # Run the synchronous per-market scoring loop in a thread executor so it
    # doesn't block the async event loop. Each market calls yfinance (momentum,
    # relval, price merge) which are synchronous I/O — cannot run in the event loop.
    def _compute_all_market_scores():
      _results = []
      for market in MARKETS:
        mid      = market["id"]
        is_cross = market.get("cross", False)
    
        # ── COT scoring ──────────────────────────────────────────────────────────
        if is_cross:
            # Build a synthetic 3-category cross COT (commercials / large specs /
            # small specs) from the two USD legs, then run it through the SAME v2
            # scorer the outright markets use — full parity, all three categories.
            cross_df = build_cross_cot_df(market["base_leg"], market["quote_leg"], cot_df_cache)
            if cross_df is not None and len(cross_df) >= 12:
                cross_df = _merge_price_into_cot(cross_df, market["yf"], mid)
                cot_data = compute_cot_score_v2(cross_df, market_id=mid)
                # tag cross metadata so the UI can label legs / methodology
                if isinstance(cot_data, dict):
                    _cd = cot_data.setdefault("detail", {}) or {}
                    _cd["cross"]     = True
                    _cd["base_leg"]  = market["base_leg"]
                    _cd["quote_leg"] = market["quote_leg"]
                    _cd["cross_method"] = "3-category net spread (base − quote, OI-normalised)"
                    cot_data["detail"] = _cd
            else:
                # Fallback: legacy commercial-Briese differential if legs unavailable
                cot_data = compute_cross_cot_score(
                    mid, market["base_leg"], market["quote_leg"], cot_df_cache)
        elif market.get("crypto_cot_mode"):
            cot_df = cot_df_cache.get(mid)
            cot_df = _merge_price_into_cot(cot_df, market["yf"], mid)
            cot_data = compute_crypto_cot_score(cot_df, market_id=mid)
        else:
            cot_df = cot_df_cache.get(mid)
            cot_df = _merge_price_into_cot(cot_df, market["yf"], mid)
            cot_data = compute_cot_score_v2(cot_df, market_id=mid)
        # ────────────────────────────────────────────────────────────────────────
    
        seasonal_data = score_seasonality(mid)
        momentum_data = score_momentum(market["yf"])
        gc.collect()  # release yfinance buffers between markets
        macro_data    = get_macro_score_for_market(mid, macro, ff_macro=ff_macro)
        _news_sent    = news_ctx.get("narrative_scores", {}).get(mid)
        regime_data   = get_regime_score_for_market(mid, regime, news_sentiment=_news_sent)
        fade_data     = compute_consensus_fade(cot_data, _news_sent,
                                               market_name=market["name"], category=market["category"])
        relval_data   = compute_rel_val_score(mid)
        pcr_data      = score_pcr(mid)  # returns neutral for unsupported markets
    
        scores = {
            "cot":      cot_data["score"],
            "seasonal": seasonal_data["score"],
            "momentum": momentum_data["score"],
            "macro":    macro_data["score"],
            "regime":   regime_data["score"],
            "relval":   relval_data["score"],
        }
        # PCR only active for equity index markets
        if mid in PCR_EQUITY_SYMBOLS:
            scores["pcr"] = pcr_data["score"]
    
        # Build a merged cot_detail dict for compute_engine_bias.
        # compute_cot_score_v2 returns v2 signal keys at the TOP LEVEL of cot_data, NOT inside
        # cot_data["detail"]. We must merge both so the COT signals in compute_engine_bias can
        # see them. Without this the confluence bonus never fires.
        # _COT_V2_SIGNAL_KEYS is the module-level tuple defined above the weight maps.
        _cot_detail_inner  = cot_data.get("detail", {}) or {}
        _cot_v2_signals    = {k: cot_data[k] for k in _COT_V2_SIGNAL_KEYS if k in cot_data}
        _cot_detail_merged = {**_cot_detail_inner, **_cot_v2_signals, "score": cot_data.get("score", 5.0)}
        # AUDIT-COMPOSITE: pass the seasonal detail so the engine can apply the
        # goldilocks entry-timing tilt + publish seasonal_hint.
        bias = compute_engine_bias(scores, market_id=mid, cot_detail=_cot_detail_merged,
                                   momentum_detail=momentum_data.get("detail", {}),
                                   seas_detail=seasonal_data.get("detail", {}) or {})
    
        # Include v2 debug fields alongside the cot entry so they
        # are visible in /api/scores for client-side debugging.
        _cot_v2_out = {k: cot_data.get(k) for k in _COT_V2_SIGNAL_KEYS}
        _cot_top_level = {
            k: cot_data.get(k)
            for k in ("comm_index", "lspec_index", "sspec_index",
                       "comm_net", "lspec_net", "sspec_net",
                       "turning", "lspec_chg_3w")
        }
        scores_out = {
            "cot": {
                "score":  cot_data["score"],
                "label":  cot_data["label"],
                "detail": cot_data.get("detail", cot_data),
                **_cot_top_level,
                **_cot_v2_out,
            },
            "seasonal": {"score": seasonal_data["score"], "label": seasonal_data["label"], "detail": seasonal_data.get("detail", seasonal_data)},
            "momentum": {"score": momentum_data["score"], "label": momentum_data["label"], "detail": momentum_data.get("detail", {})},
            "macro":    {"score": macro_data["score"],    "label": macro_data["label"],    "detail": macro_data},
            "regime":   {"score": regime_data["score"],   "label": regime_data["label"],   "detail": regime_data},
            "relval":   {"score": relval_data["score"],   "label": relval_data["label"],   "detail": {k: v for k, v in relval_data.items() if k != "lines"}},
        }
        if mid in PCR_ALL_SYMBOLS:
            scores_out["pcr"] = {
                "score": pcr_data["score"],
                "label": pcr_data["label"],
                "tier":  pcr_data.get("tier", 0),
                "detail": pcr_data.get("detail", {}),
            }
    
        # Determine the actual weight map used for this market (legacy per-market weight mini-bars)
        # This is exposed per-market so the frontend can render the correct weight mini-bars
        mkt_weights = _get_weight_map(mid)
        # Only expose weights for factors actually present in this market's scores
        active_weights = {k: v for k, v in mkt_weights.items() if k in scores_out}
    
        _results.append({
            "id":                mid,
            "name":              market["name"],
            "ticker":            market["ticker"],
            "category":          market["category"],
            "cross":             is_cross,
            "base_leg":          market.get("base_leg"),
            "quote_leg":         market.get("quote_leg"),
            "cot_note":          market.get("cot_note", None),
            "ice_source":        bool(market.get("ice_code")),
            "ice_limited_history": bool(market.get("ice_limited_history", False)),
            "cot_format":        market.get("cot_format", "legacy"),
            "bias":              bias["bias"],
            "weighted_score":    bias["weighted"],
            "color":             bias["color"],
            "confluence_bonus":  bias.get("confluence_bonus", 0.0),
            "confluence_reason": bias.get("confluence_reason"),
            # ── validated engine state (primary signal for the UI) ─────────────
            "direction":         bias.get("direction", "Neutral"),
            "bias_sign":         bias.get("bias_sign", 0),
            "conviction":        bias.get("conviction", 0.0),
            "tier":              bias.get("tier", "Neutral"),
            "regime":            bias.get("regime", "Mixed"),
            "efficiency_ratio":  bias.get("efficiency_ratio"),
            "regime_gate":       bias.get("regime_gate"),
            "trend_lt":          bias.get("trend_lt", 0),
            "trend_st":          bias.get("trend_st", 0),
            "trend_state":       bias.get("trend_state", ""),
            "setup_quality":     bias.get("setup_quality", "n/a"),
            "setup_direction":   bias.get("setup_direction", "Neutral"),
            "setup_vs_backdrop": bias.get("setup_vs_backdrop", "none"),
            "agree":             bias.get("agree", 0),
            "disagree":          bias.get("disagree", 0),
            "factor_votes":      bias.get("factor_votes", {}),
            "drivers":           bias.get("drivers", []),
            # ── AUDIT-COMPOSITE: seasonal goldilocks entry-timing on the SUMMARY ──
            # These are duplicated at market top level (not just inside
            # scores.seasonal.detail) so the market-list cards can render an
            # "almost ready seasonally" chip without reaching into the detail blob.
            "seasonal_hint":       bias.get("seasonal_hint"),
            "seasonal_score_raw":  bias.get("seasonal_score_raw"),
            "seasonal_score_adj":  bias.get("seasonal_score_adj"),
            "seasonal_timing_tilt": bias.get("seasonal_timing_tilt", 0.0),
            "seasonal_timing_mode": bias.get("seasonal_timing_mode"),
            "seasonal_timing_conv": bias.get("seasonal_timing_conv", 0.0),
            "seasonal_other_lean": bias.get("seasonal_other_lean", 0),
            "entry_timing":        bias.get("entry_timing"),
            "days_to_goldilocks":  bias.get("days_to_goldilocks"),
            "goldilocks_dir":      bias.get("goldilocks_dir"),
            "entry_note":          bias.get("entry_note"),
            "immediate_score":     bias.get("immediate_score"),
            "scores":            scores_out,
            "weights":           active_weights,  # Per-market weight map (varies by data quality + PCR tier)
            "fade":              fade_data,        # consensus-fade / crowded-trade detector
        })
      return _results
    # end _compute_all_market_scores
    
    # Run the loop in a thread — it contains sync yfinance calls (momentum, relval, price merge)
    _loop = asyncio.get_event_loop()
    results = await _loop.run_in_executor(_APP_EXECUTOR, _compute_all_market_scores)
    print(f"[scores] Market scoring done in {time.time()-_refresh_start:.1f}s", flush=True)
    
    # ── DX REGIME FEEDBACK LOOP ───────────────────────────────────────────────────
    # When DX (US Dollar Index) has a strong composite signal, apply a
    # calibrated cross-asset tilt to correlated markets via their regime score.
    #
    # Logic:
    #   DX score ≥ 7.0 (bullish dollar) → bearish tilt on: GC, SI, CL, HG, 6E, 6B, 6A, 6C, 6J
    #   DX score ≤ 3.0 (bearish dollar) → bullish tilt on: same set
    #
    # Magnitude:
    #   Tilt applied to the *regime* component score only (keeps other factors clean)
    #   Max tilt: ±0.4 on a 0-10 scale (modest — dollar is one factor among many)
    #   Scaled by how far DX is from 5.0: a 7.0 DX applies less tilt than a 9.0 DX
    #   FX pairs: full tilt. Commodities: 70% (supply factors dilute dollar effect).
    #   Yen (6J): inverted — strong dollar = yen weakness IS the signal, already in COT
    #
    # Rationale:
    #   The dollar’s inverse relationship with commodities and non-USD FX is well-established
    #   (DXY vs GC 1Y correlation ~-0.75, vs CL ~-0.45, vs 6E ~-0.90).
    #   This is not double-counting: the individual FX regime score uses CB differentials,
    #   not the DX composite. The DX composite score incorporates COT + seasonality +
    #   momentum + macro, giving a richer signal than rates alone.
    # ────────────────────────────────────────────────────────────────
    dx_market = next((r for r in results if r["id"] == "DX"), None)
    dx_score  = dx_market["weighted_score"] if dx_market else None
    
    # Only apply feedback when DX signal is clear (outside neutral zone 4.0–6.0)
    if dx_score is not None and (dx_score >= 6.5 or dx_score <= 3.5):
        dx_deviation = dx_score - 5.0   # +ve = dollar bullish, -ve = dollar bearish
        # Scale tilt: each full point beyond neutral = 0.08 tilt (capped at 0.40)
        base_tilt = round(max(-0.40, min(0.40, dx_deviation * 0.08)), 3)
    
        # Markets affected + their tilt multiplier
        # Sign of tilt: inverse to DX direction
        #   DX bullish (positive deviation) → bearish tilt (negative) on these assets
        DX_CORRELATED = {
            # Precious metals — strong DX inverse
            "GC":  -0.95,   # gold: strongest correlation
            "SI":  -0.85,   # silver: strong but diluted by industrial
            # Base metals — moderate DX inverse (growth channel dominates)
            "HG":  -0.60,
            # Energy — moderate DX inverse
            "CL":  -0.55,
            "RB":  -0.50,
            "HO":  -0.50,
            # FX (non-USD) — near-perfect inverse of DX
            "6E":  -0.90,
            "6B":  -0.80,
            "6A":  -0.75,
            "6C":  -0.70,
            "6N":  -0.70,
            "6S":  -0.75,
            "6M":  -0.65,
            # Yen: usually inverse DX, but COT/macro already captures BoJ dynamic well
            # Apply a reduced weight to avoid double-counting
            "6J":  -0.45,
            # Crypto: mild dollar inverse (especially at extremes)
            "BTC": -0.40,
            "ETH": -0.40,
            # ICE Europe: Brent and Gas Oil follow crude inverse-dollar pattern
            "B":   -0.50,   # Brent: ~-0.50 inverse with DXY (slightly less than WTI)
            "GO":  -0.45,   # Gas Oil: European diesel, moderate dollar inverse
            # FTSE 100: USD strength = GBP weakness = higher FTSE EPS in GBP terms
            # So FTSE has a POSITIVE correlation with strong USD (FX translation tailwind)
            # This partially offsets risk-off pressure. Net: mild positive DX correlation
            "Z":   +0.30,   # FTSE 100: weak pound = overseas earnings boost
        }
    
        for mkt in results:
            mid = mkt["id"]
            if mid not in DX_CORRELATED or mid == "DX":
                continue
            corr_mult = DX_CORRELATED[mid]
            # Tilt direction: base_tilt is signed by DX direction
            # corr_mult is negative (inverse relationship) so:
            # DX bullish (+ve base_tilt) * corr_mult (-ve) = negative tilt on correlated asset
            raw_tilt = base_tilt * corr_mult   # e.g. DX=8 → base_tilt=+0.24, GC: 0.24*-0.95=-0.228
            regime_detail = mkt["scores"].get("regime", {})
            old_regime_score = regime_detail.get("score", 5.0)
            new_regime_score = round(max(0.0, min(10.0, old_regime_score + raw_tilt)), 2)
    
            # Recompute weighted score with adjusted regime score
            # Guard against None scores (None-safe — compute_engine_bias handles None filtering)
            factor_scores = {
                k: mkt["scores"][k]["score"]
                for k in mkt["scores"]
                if "score" in mkt["scores"][k] and mkt["scores"][k]["score"] is not None
            }
            factor_scores["regime"] = new_regime_score
            # Must include top-level v2 signals so confluence bonus can fire after DX tilt.
            # Uses module-level _COT_V2_SIGNAL_KEYS — same tuple as the main scoring loop.
            _dx_cot_raw = mkt["scores"].get("cot", {})
            _dx_cot_inner = _dx_cot_raw.get("detail", {}) or {}
            _dx_v2_sigs = {k: _dx_cot_raw[k] for k in _COT_V2_SIGNAL_KEYS if k in _dx_cot_raw}
            cot_detail_for_mid = {**_dx_cot_inner, **_dx_v2_sigs, "score": _dx_cot_raw.get("score", 5.0)}
            _dx_mom_detail = mkt["scores"].get("momentum", {}).get("detail", {}) or {}
            # AUDIT-COMPOSITE: the DX feedback loop must re-flow the seasonal
            # timing tilt too, otherwise the tilt (and seasonal_hint) silently
            # vanished for every DX-correlated market — which is most of the book.
            _dx_seas_detail = mkt["scores"].get("seasonal", {}).get("detail", {}) or {}
            new_bias = compute_engine_bias(factor_scores, market_id=mid, cot_detail=cot_detail_for_mid,
                                           momentum_detail=_dx_mom_detail,
                                           seas_detail=_dx_seas_detail)
    
            # Update the result in place — re-flow the FULL engine state after the tilt
            mkt["scores"]["regime"]["score"]  = new_regime_score
            mkt["scores"]["regime"]["dx_tilt"] = round(raw_tilt, 3)
            mkt["scores"]["regime"]["dx_tilt_source"] = f"DX {dx_score:.1f}/10 → {'bull' if dx_deviation > 0 else 'bear'} dollar feedback"
            mkt["weighted_score"]  = new_bias["weighted"]
            mkt["bias"]            = new_bias["bias"]
            mkt["color"]           = new_bias["color"]
            mkt["confluence_bonus"]= new_bias.get("confluence_bonus", 0.0)
            for _k in ("direction","bias_sign","conviction","tier","regime","efficiency_ratio",
                       "regime_gate","trend_lt","trend_st","trend_state","setup_quality",
                       "setup_direction","setup_vs_backdrop","agree","disagree","factor_votes","drivers",
                       # AUDIT-COMPOSITE: keep the seasonal timing block in sync
                       # after the DX regime tilt re-flows the engine.
                       "seasonal_hint","seasonal_score_raw","seasonal_score_adj",
                       "seasonal_timing_tilt","seasonal_timing_mode","seasonal_timing_conv",
                       "seasonal_other_lean",
                       "entry_timing","days_to_goldilocks","goldilocks_dir",
                       "entry_note","immediate_score"):
                mkt[_k] = new_bias.get(_k, mkt.get(_k))
    
    results.sort(key=lambda x: x["weighted_score"], reverse=True)
    
    # Strip nulls from cot.detail for every market — saves ~15KB from the payload
    for mkt in results:
        cot_detail = mkt.get("scores", {}).get("cot", {}).get("detail")
        if isinstance(cot_detail, dict):
            mkt["scores"]["cot"]["detail"] = {k: v for k, v in cot_detail.items() if v is not None}

    # ── Consensus-fade aggregate — ranked crowded-trade candidates for homepage ──
    # Surface the markets where the crowd is most one-sided AND still offside, so
    # Ben can see the best asymmetric fade setups at a glance. Threshold at 3.5 to
    # keep only meaningful extremes; sort by fade_score desc.
    _FADE_MIN = 3.5
    _fade_candidates = []
    for mkt in results:
        fd = mkt.get("fade") or {}
        fs = fd.get("fade_score") or 0.0
        if fs >= _FADE_MIN and fd.get("fade_dir"):
            _fade_candidates.append({
                "id":          mkt["id"],
                "name":        mkt["name"],
                "category":    mkt["category"],
                "fade_score":  fs,
                "fade_dir":    fd.get("fade_dir"),
                "crowd_side":  fd.get("crowd_side"),
                "spec_pctile": fd.get("spec_pctile"),
                "spec_adding": fd.get("spec_adding"),
                "news_score":  fd.get("news_score"),
                "confirms":    fd.get("confirms"),
                "asymmetry":   fd.get("asymmetry"),
                # engine's own directional read, so the panel can flag when the
                # composite engine ALREADY agrees with the fade (extra confluence)
                "engine_direction": mkt.get("direction", "Neutral"),
                "engine_conviction": mkt.get("conviction", 0.0),
                "weighted_score": mkt.get("weighted_score"),
            })
    _fade_candidates.sort(key=lambda x: x["fade_score"], reverse=True)
    consensus_fade = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "count":      len(_fade_candidates),
        "candidates": _fade_candidates[:12],
    }

    output = {
        "updated_at":      datetime.utcnow().isoformat() + "Z",
        "regime":          regime,
        "macro_all":       macro,
        "stock_climate":   stock_climate,
        "ff_macro":        ff_macro,  # per-currency FF economy scores
        "consensus_fade":  consensus_fade,  # ranked crowded-trade / fade candidates
        "markets":       results,
        "weights":               WEIGHTS,               # Fallback
        "weights_hg":            WEIGHTS_HG,
        "weights_pa":            WEIGHTS_PA,
        "weights_pl":            WEIGHTS_PL,
        "weights_equity":        WEIGHTS_EQUITY,
        "weights_fx":            WEIGHTS_FX,
        "weights_fx_crosses":    WEIGHTS_FX_CROSSES,
        "weights_gold":          WEIGHTS_GOLD,
        "weights_silver":        WEIGHTS_SILVER,
        "weights_crude":         WEIGHTS_CRUDE,
        "weights_natgas":        WEIGHTS_NATGAS,
        "weights_bonds":         WEIGHTS_BONDS,
        "weights_grains":        WEIGHTS_GRAINS,
        "weights_softs":         WEIGHTS_SOFTS,
        "weights_cocoa":         WEIGHTS_COCOA,
        "weights_coffee":        WEIGHTS_COFFEE,
        "weights_livestock":     WEIGHTS_LIVESTOCK,
        "weights_crypto":        WEIGHTS_CRYPTO,
        "weights_ice_thin":      WEIGHTS_ICE_THIN,      # Z (FTSE100) and R (Long Gilt)
        # news_context intentionally excluded — frontend fetches /api/news-context separately
    }
    # Always store with full TTL — narratives have their own endpoint and cache
    ALL_DATA_CACHE["data"] = output
    ALL_DATA_CACHE["time"] = time.time()
    # Save daily snapshot for weekly score delta calculation (survives restarts)
    try:
        _snap_map = {m["id"]: m["weighted_score"] for m in results if "id" in m and "weighted_score" in m}
        _save_scores_snapshot(_snap_map)
    except Exception as _snap_e:
        print(f"[snapshot] save error: {_snap_e}")
    # Persist the FULL payload to disk so the next cold start renders instantly
    try:
        _save_full_scores_snapshot(output)
    except Exception as _fs_e:
        print(f"[snapshot] full save error: {_fs_e}")
    print(f"[scores] Cache populated — total {time.time()-_refresh_start:.1f}s", flush=True)

# ============================================================
# NEWS CONTEXT ENDPOINT
# ============================================================

@app.get("/api/news-context")
async def get_news_context(force: bool = False):
    """
    Returns FF calendar events (last 48h high+medium impact) + AI narrative.
    Runs in a thread executor so it never blocks the event loop.
    Query param: force=true to bust the cache.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        ctx = await loop.run_in_executor(_APP_EXECUTOR, lambda: compute_news_context(force=force))
        return _SafeJSONResponse(ctx)
    except Exception as e:
        return {"narratives": {}, "news_items": [], "price_context": {}, "error": str(e), "updated_at": time.time()}


@app.get("/api/consensus-outlook")
async def get_consensus_outlook(force: bool = False):
    """
    Weekly cross-asset consensus outlook — the synthesised "what the crowd /
    trend-following funds believe" read from this week's bank & CTA commentary.
    Cached 7 days. force=true triggers a fresh Sonar web-search read (used by
    the weekly pre-warm cron).
    """
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        out = await loop.run_in_executor(_APP_EXECUTOR, lambda: compute_consensus_outlook(force=force))
        return _SafeJSONResponse({
            "consensus_outlook": out,
            "updated_at": CONSENSUS_CACHE["time"] or None,
        })
    except Exception as e:
        return {"consensus_outlook": {}, "error": str(e), "updated_at": None}

# ============================================================
# SEASONALITY ENDPOINT — serves pre-computed curves for all 21 markets
# ============================================================
import os as _os

# Seasonality is now built dynamically (see _build_seasonality_from_closes / _load_dyn_seas_file).
# The old static 29MB seasonality_all21.json is no longer read.
_SEASONALITY_CACHE = {"data": None}

# Cache for current-year actual price returns (refreshed every 30 min)
_CY_ACTUAL_CACHE: dict = {}
_CY_ACTUAL_TIME: dict = {}
_CY_ACTUAL_TTL = 1800  # 30 min

def _get_current_year_actual(market_id: str) -> list:
    """
    Fetch YTD price data for the given market and return cumulative % return
    from trading day 1 of the current year, as [[td, pct], ...] pairs.

    TD mapping: each trading day in the current year is assigned a sequential
    index 1..N, then linearly scaled to 1..252 so the trace aligns with the
    historical seasonality curves (which are also built on a 252-point scale).
    Duplicate TDs are avoided because the index is strictly sequential.

    Returns [] on any error (frontend will simply not render the trace).
    """
    import datetime as _dt
    now = time.time()
    if market_id in _CY_ACTUAL_CACHE and (now - _CY_ACTUAL_TIME.get(market_id, 0)) < _CY_ACTUAL_TTL:
        return _CY_ACTUAL_CACHE[market_id]
    try:
        mkt = next((x for x in MARKETS if x["id"] == market_id), None)
        if not mkt or not mkt.get("yf"):
            return []
        yf_sym = mkt["yf"]
        cur_year = _dt.date.today().year
        start_date = f"{cur_year}-01-01"
        tk_obj = yf.Ticker(yf_sym)
        df = tk_obj.history(start=start_date, interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return []
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return []
        base = float(closes.iloc[0])
        result = []
        for idx, price in closes.items():
            # Calendar-DOY based td — MUST match the formula used by current_td and
            # by _score_seasonality_at() so that the CYA line and the TODAY marker
            # always sit on the same coordinate.  The previous linear-index formula
            # mapped the LAST data point to td=252 (year-end) even in May, which
            # caused buildSeasChart to clip the visible trace at ~30/86 days.
            _d   = idx.date() if hasattr(idx, 'date') else idx
            _doy = _d.timetuple().tm_yday
            td   = max(1, min(252, round((_doy / 365) * 252)))
            pct  = round((float(price) / base - 1.0) * 100.0, 4)
            result.append([td, pct])
        # Deduplicate: keep last value for any duplicate TDs (adjacent trading days
        # can share the same DOY-based td when the calendar fraction rounds the same)
        seen: dict = {}
        for pair in result:
            seen[pair[0]] = pair[1]
        result = [[t, v] for t, v in sorted(seen.items())]
        _CY_ACTUAL_CACHE[market_id] = result
        _CY_ACTUAL_TIME[market_id] = now
        return result
    except Exception as e:
        print(f"[cy_actual] {market_id}: {e}")
        return []

@app.get("/api/seasonality")
async def get_seasonality(market: str = None):
    # Dynamic seasonality: build the requested market on demand (cached/weekly),
    # then serve from the dynamic cache. No static 29MB file.
    if market:
        try:
            await asyncio.get_event_loop().run_in_executor(
                _APP_EXECUTOR, _ensure_market_seas, market.upper())
        except Exception as _e:
            print(f"[seas api] ensure {market}: {_e}", flush=True)
    data = _load_seas_data()
    if market:
        m = market.upper()
        if m not in data or data[m].get("v") != 2 or not data[m].get("curve"):
            return {"error": f"Seasonality for '{m}' is still building or unavailable"}
        ent = data[m]
        cy_actual = _get_current_year_actual(m)
        import datetime as _seas_dt
        _today = _seas_dt.date.today()
        _doy   = _today.timetuple().tm_yday
        _current_td = max(1, min(252, round((_doy / 365) * 252)))
        _months_axis = []
        for _mo in range(1, 13):
            _md = _seas_dt.date(_today.year, _mo, 1)
            _mtd = max(1, min(252, round((_md.timetuple().tm_yday / 365) * 252)))
            _months_axis.append({"td": _mtd, "label": _md.strftime("%b")})
        cycles = ent.get("cycles") or {}
        stats = _seas_window_stats(m, _today)
        return {
            "market": m,
            "all":    ent.get("curve", []),
            "band":   ent.get("band", []),
            "mt":     cycles.get("midterm", []),
            "post_election": cycles.get("post_election", []),
            "midterm":       cycles.get("midterm", []),
            "pre_election":  cycles.get("pre_election", []),
            "election":      cycles.get("election", []),
            "months": _months_axis,
            "current_td": _current_td,
            "current_year_actual": cy_actual,
            "n_years": ent.get("n_years"),
            "years_span": ent.get("years_span"),
            "turns": ent.get("turns", []),
            "window": stats,
        }
    return _SafeJSONResponse(data)

# ============================================================
# Seasonality Lab v2 — asset-class default blends
# ============================================================
# Election-cycle sensitivity varies by asset class:
#  - Stock indices, major FX, US rates → cycle-sensitive (Fed/political calendar)
#  - Commodities (metals, energy, ags, softs, livestock), crypto → all-years
#    (supply-cycle driven, not politics)
#  - FX crosses → mild cycle sensitivity (carry-dominated)
_SEAS_BLEND_PROFILES = {
    "cycle_heavy":  {"cycle_w": 2.5, "halflife": 15, "detrend": False, "tag": "Cycle-sensitive"},
    "fx_cross":     {"cycle_w": 1.5, "halflife": 15, "detrend": False, "tag": "Mild cycle"},
    "all_years":    {"cycle_w": 1.0, "halflife": 15, "detrend": False, "tag": "Supply-cycle"},
}

def _seas_asset_class(market_id: str) -> str:
    """Return 'cycle_heavy' | 'fx_cross' | 'all_years' for the given market."""
    m = market_id.upper()
    # Cycle-heavy: equity indices + major FX legs + US rates
    _CYCLE_HEAVY = {
        "ES", "NQ", "YM", "RTY",                                 # equity indices
        "6E", "6B", "6J", "6A", "6C", "6S", "6N", "6M", "DX",   # major FX + DX
        "ZB", "ZN", "ZF", "ZT",                                 # US rates
    }
    if m in _CYCLE_HEAVY:
        return "cycle_heavy"
    # FX crosses (carry-dominated but some cycle imprint via legs)
    mkt = next((x for x in MARKETS if x["id"] == m), None)
    if mkt and mkt.get("category") == "fx_cross":
        return "fx_cross"
    # Everything else (commodities, crypto) → all-years
    return "all_years"

def _seas_default_blend(market_id: str) -> dict:
    prof = _seas_asset_class(market_id)
    d = dict(_SEAS_BLEND_PROFILES[prof])
    d["profile"] = prof
    return d

def _seas_year_weights(years_list, cycle_w: float, halflife: float,
                       asof_year: int, cycle_key: str) -> dict:
    """Build per-year weight dict from cycle_w (cycle overweight) + halflife
    (recency exponential decay). Returns {year_str: weight_float}.
    halflife = 0 → flat recency; else weight = 0.5 ** ((asof - y) / halflife).
    cycle_w applied to years matching current cycle_key."""
    weights = {}
    for ys in years_list:
        y = int(ys)
        # Recency
        if halflife and halflife > 0:
            age = max(0, asof_year - y)
            w_rec = 0.5 ** (age / float(halflife))
        else:
            w_rec = 1.0
        # Cycle overweight
        w_cyc = cycle_w if _cycle_key_for_year(y) == cycle_key else 1.0
        weights[ys] = w_rec * w_cyc
    return weights

def _seas_weighted_stats(years_dict: dict, weights: dict,
                         current_year: int = None,
                         current_td: int = None) -> dict:
    """Compute weighted median + p25/p75 + weighted mean per TD across the
    given per-year paths. Returns dict with lists of length 252.

    ── SEAS-R2 (Issue 2) ROOT-CAUSE FIX: CURRENT-YEAR PAD POLLUTION ──────────
    Ben selected a late-Sep -> early-Oct window on NQ where the plotted blend
    line visibly DROPPED, yet the window stats reported a 48% short win rate.
    Three numbers on one screen disagreed. Diagnosis:

      * the stored current-year path is FORWARD-FILLED past today: for NQ,
        years['2026'] holds the constant 15.88 for every TD from 164 (today)
        through 251 — 88 fabricated points, not data.
      * _seas_year_weights gives the current year age=0 AND a cycle match, so it
        carries the single LARGEST weight of all 42 years (2.5).
      * this is a weighted median OF LEVELS. So past today the flat pad sat at
        the top of the weight stack and simply *was* the median for ~24 TD,
        producing a fake plateau, and then the median discontinuously handed
        over to a different year's level — a fake CLIFF of -2.25% between
        td 184 and td 191 that exists in no individual year.
      * meanwhile _seas_lab_window_stats correctly excludes the current year, so
        it reported the honest distribution. The stats were right; the LINE was
        lying.

    Verified: rebasing every year to 0 at td 184 and taking the weighted median
    per TD gives 0.0 -> +0.45 by td 191, i.e. NO drop at all. The drop was 100%
    composition artefact.

    Fix: the current year contributes only its REAL, elapsed portion
    (td <= current_td). Its forward-filled tail is excluded from the median /
    mean / p25 / p75 at every TD beyond today. The year is still returned in
    full in the `years` payload, so the actual-YTD overlay is unaffected — only
    the blend statistics stop being fabricated.
    """
    if not years_dict:
        return {"median": [], "mean": [], "p25": [], "p75": []}
    ys_all = list(years_dict.keys())
    n_td = max(len(years_dict[y]) for y in ys_all)
    med = [0.0] * n_td
    mean = [0.0] * n_td
    p25 = [0.0] * n_td
    p75 = [0.0] * n_td
    for i in range(n_td):
        pts = []
        for ys in ys_all:
            path = years_dict[ys]
            # SEAS-R2: drop the current year's forward-filled tail (see docstring)
            if current_year is not None and current_td is not None:
                try:
                    if int(ys) == int(current_year) and i > int(current_td):
                        continue
                except (TypeError, ValueError):
                    pass
            if i < len(path) and path[i] is not None:
                w = weights.get(ys, 1.0)
                if w > 0:
                    pts.append((path[i], w))
        if not pts:
            med[i] = mean[i] = p25[i] = p75[i] = None
            continue
        # Weighted mean
        sw = sum(w for _, w in pts)
        mean[i] = sum(v * w for v, w in pts) / sw if sw > 0 else None
        # Weighted quantiles: sort by v, accumulate weight, pick 25/50/75
        pts.sort(key=lambda x: x[0])
        cum = 0.0
        q25 = q50 = q75 = None
        for v, w in pts:
            cum += w
            frac = cum / sw
            if q25 is None and frac >= 0.25:
                q25 = v
            if q50 is None and frac >= 0.5:
                q50 = v
            if q75 is None and frac >= 0.75:
                q75 = v
                break
        med[i] = q50 if q50 is not None else pts[len(pts)//2][0]
        p25[i] = q25 if q25 is not None else pts[0][0]
        p75[i] = q75 if q75 is not None else pts[-1][0]
    return {"median": med, "mean": mean, "p25": p25, "p75": p75}

# SEAS-R2 (Issue 2): a window return of exactly zero is neither a long win nor a
# short win. Anything inside +/- this band (in %) is a TIE and is removed from
# BOTH the numerator and the denominator of every win rate.
_SEAS_TIE_EPS = 0.005


def _seas_weighted_rebased(years_dict: dict, weights: dict, anchor_td: int,
                           current_year: int = None,
                           current_td: int = None) -> dict:
    """SEAS-R2 (Issue 2): blend median/mean/p25/p75 of paths REBASED to 0 at
    `anchor_td`, i.e. the honest forward seasonal path from a chosen day.

    Why this exists. The main lab series is a weighted median OF LEVELS
    (cumulative % from Jan 1). That statistic is composition-dependent: the
    year sitting at the median changes from one TD to the next, so the line can
    fall several percent between two days even when the *typical year* rose over
    those same two days. That is exactly the divergence Ben hit — a line that
    dropped while the window stats said up.

    Rebasing first removes the artefact entirely: every year is set to 0 at
    `anchor_td`, then the weighted median is taken of the MOVES. The value at
    `end_td` then equals the window stats' median return by construction, so
    the picture and the number cannot disagree.

    Same current-year pad exclusion as _seas_weighted_stats. Returns lists of
    length n_td with None before `anchor_td`.
    """
    if not years_dict:
        return {"median": [], "mean": [], "p25": [], "p75": [], "anchor_td": anchor_td}
    ys_all = list(years_dict.keys())
    n_td = max(len(years_dict[y]) for y in ys_all)
    anchor_td = max(0, min(n_td - 1, int(anchor_td)))
    # Per-year rebased paths (compounded off the anchor level)
    reb = {}
    for ys in ys_all:
        path = years_dict[ys]
        if anchor_td >= len(path) or path[anchor_td] is None:
            continue
        base = 1.0 + path[anchor_td] / 100.0
        if abs(base) < 1e-9:
            continue
        reb[ys] = [(((1.0 + v / 100.0) / base - 1.0) * 100.0) if v is not None else None
                   for v in path]
    med = [None] * n_td; mean = [None] * n_td
    p25 = [None] * n_td; p75 = [None] * n_td
    for i in range(anchor_td, n_td):
        pts = []
        for ys, path in reb.items():
            if current_year is not None and current_td is not None:
                try:
                    if int(ys) == int(current_year) and i > int(current_td):
                        continue
                except (TypeError, ValueError):
                    pass
            if i < len(path) and path[i] is not None:
                w = weights.get(ys, 1.0)
                if w > 0:
                    pts.append((path[i], w))
        if not pts:
            continue
        sw = sum(w for _, w in pts)
        mean[i] = sum(v * w for v, w in pts) / sw if sw > 0 else None
        pts.sort(key=lambda x: x[0])
        cum = 0.0; q25 = q50 = q75 = None
        for v, w in pts:
            cum += w
            frac = cum / sw if sw > 0 else 1.0
            if q25 is None and frac >= 0.25: q25 = v
            if q50 is None and frac >= 0.5: q50 = v
            if q75 is None and frac >= 0.75:
                q75 = v; break
        med[i] = q50 if q50 is not None else pts[len(pts)//2][0]
        p25[i] = q25 if q25 is not None else pts[0][0]
        p75[i] = q75 if q75 is not None else pts[-1][0]
    return {"median": med, "mean": mean, "p25": p25, "p75": p75,
            "anchor_td": anchor_td}


def _seas_lab_window_stats(years_dict: dict, weights: dict,
                           start_td: int, end_td: int,
                           current_year: int = None,
                           current_td: int = None,
                           wstats: dict = None) -> dict:
    """For a [start_td, end_td] window compute per-year window returns and
    aggregate stats. Returns {window_stats, per_year_windows}.
    Excludes the current year when the window extends past current_td so
    historical stats aren't polluted by an incomplete year.

    ── SEAS-R2 (Issue 2): STATS-LENS UNIFICATION ────────────────────────────
    Ben saw three different numbers describing one selected window. Four
    distinct defects, all fixed here:

    1) ARITHMETIC vs GEOMETRIC. `ret_pct` was `ep - sp`, the arithmetic
       difference of two CUMULATIVE-%-from-Jan-1 levels. That is not the return
       over the window. A year that ran +40% by September and then +2% more
       showed 40 -> 42.8 = "+2.8%" instead of the true 42.8/40 compounded
       +2.0%. The error scales with how far the year had already travelled, so
       high-momentum years were systematically over-weighted in the
       distribution. Now compounded: ((1+ep/100)/(1+sp/100) - 1) * 100.
       max_rise / max_drop are compounded off the entry level the same way.

    2) ZERO-TIE MIS-COUNTING. Long wins required ret > 0 while long losses were
       ret <= 0, so an exactly-flat year was booked as a long LOSS; the short
       mirror had the identical bug the other way (`>= 0` counted as a short
       loss). Flat years therefore depressed BOTH sides' win rate, and
       long_win_rate + short_win_rate did not sum to 1. Ties are now excluded
       from numerator AND denominator on both sides, so the two rates are
       genuine complements and n_up + n_down + n_ties = n_years.

    3) NO SINGLE SOURCE OF TRUTH. The donut, the callout and the table each
       recomputed their own view. There is now ONE `headline` block, on the
       BLEND lens (weighted — the same weights that draw the plotted line), and
       the equal-weight view is preserved beside it as explicit secondary
       `raw_*` fields. The blend lens is primary because it is what the chart
       shows; if the two disagree that is real information, not a bug, and both
       are now on screen.

    4) LINE vs STATS RECONCILIATION. `curve_move_pct` is the rebased
       blend-median move (rebase every year to 0 at start_td, take the weighted
       median of the window return) and equals the headline median exactly.
       `curve_level_move_pct` is the slope of the plotted median-OF-LEVELS line
       between the same two TDs. These two are NOT the same statistic — a
       median of levels can move without any year moving, because the
       identity of the median year changes. Publishing both means a
       line/stats divergence is visible and explainable instead of looking
       like a broken number.
    """
    if not years_dict or start_td >= end_td:
        return {"window_stats": None, "per_year_windows": []}
    rows = []
    for ys, path in years_dict.items():
        # Skip current year when window is not yet complete
        if current_year is not None and int(ys) == current_year:
            if current_td is None or end_td > current_td:
                continue
        if not path or start_td >= len(path) or end_td >= len(path):
            continue
        sp = path[start_td]
        ep = path[end_td]
        if sp is None or ep is None:
            continue
        # SEAS-R2: paths are cumulative % from Jan 1. The window return is the
        # COMPOUNDED move between the two levels, not their difference.
        _base = 1.0 + sp / 100.0
        if abs(_base) < 1e-9:
            continue
        ret_pct = ((1.0 + ep / 100.0) / _base - 1.0) * 100.0
        # Max rise / max drop within window — compounded off the entry level too
        seg = [p for p in path[start_td:end_td+1] if p is not None]
        if len(seg) < 2:
            continue
        max_rise = ((1.0 + max(seg) / 100.0) / _base - 1.0) * 100.0
        max_drop = ((1.0 + min(seg) / 100.0) / _base - 1.0) * 100.0
        w = weights.get(ys, 1.0)
        rows.append({
            "year": int(ys),
            "start_pct": round(sp, 3),
            "end_pct": round(ep, 3),
            "ret_pct": round(ret_pct, 3),
            "max_rise": round(max_rise, 3),
            "max_drop": round(max_drop, 3),
            "weight": round(w, 4),
        })
    if not rows:
        return {"window_stats": None, "per_year_windows": []}
    # Sort newest→oldest
    rows.sort(key=lambda r: -r["year"])
    # Aggregates (weighted)
    sw = sum(r["weight"] for r in rows)
    rets = [r["ret_pct"] for r in rows]
    weighted_rets = [(r["ret_pct"], r["weight"]) for r in rows]
    if sw > 0:
        w_mean = sum(v * w for v, w in weighted_rets) / sw
    else:
        w_mean = sum(rets) / len(rets)
    # Weighted median
    # AUDIT-SCORING: guard the divide. `sw` is only checked before w_mean; if all
    # per-year weights came back zero (halflife/cycle_w combination, or a basket
    # whose years all fell outside the weight dict) this loop raised
    # ZeroDivisionError and took the whole /api/seasonality-lab request down.
    weighted_rets.sort(key=lambda x: x[0])
    cum = 0.0
    w_med = weighted_rets[len(weighted_rets)//2][0]
    if sw > 0:
        for v, w in weighted_rets:
            cum += w
            if cum / sw >= 0.5:
                w_med = v
                break
    # ── SEAS-R2: tie-aware three-way split (up / down / flat) ────────────────
    _eps = _SEAS_TIE_EPS
    gains = [r for r in rows if r["ret_pct"] > _eps]
    losses = [r for r in rows if r["ret_pct"] < -_eps]
    ties = [r for r in rows if abs(r["ret_pct"]) <= _eps]
    w_gain = sum(r["weight"] for r in gains)
    w_loss = sum(r["weight"] for r in losses)
    w_tie = sum(r["weight"] for r in ties)
    # Denominator EXCLUDES ties, so long + short win rates are true complements.
    w_decided = w_gain + w_loss
    win_rate = (w_gain / w_decided) if w_decided > 1e-12 else 0.5
    short_win_rate = (w_loss / w_decided) if w_decided > 1e-12 else 0.5
    avg_gain = sum(r["ret_pct"] * r["weight"] for r in gains) / max(w_gain, 1e-9) if gains else 0.0
    avg_loss = sum(r["ret_pct"] * r["weight"] for r in losses) / max(w_loss, 1e-9) if losses else 0.0
    max_gain = max(rets)
    max_loss = min(rets)
    # Volatility (weighted std)
    if sw > 0 and len(rets) > 1:
        var = sum(w * (v - w_mean) ** 2 for v, w in weighted_rets) / sw
        vol = var ** 0.5
    else:
        vol = 0.0
    # Annualise (window is trading days; 252 = 1 yr)
    td_window = max(1, end_td - start_td)
    ann_factor = 252.0 / td_window
    ann_ret = w_mean * ann_factor
    ann_vol = vol * (ann_factor ** 0.5)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
    # Sortino (downside deviation)
    downside = [(v - w_mean) ** 2 * w for v, w in weighted_rets if v < w_mean]
    if downside and sw > 0:
        d_var = sum(downside) / sw
        d_std = d_var ** 0.5
        d_ann = d_std * (ann_factor ** 0.5)
        sortino = ann_ret / d_ann if d_ann > 1e-9 else 0.0
    else:
        sortino = 0.0
    # ── SHORT MIRROR STATS ────────────────────────────────────────────────────
    # A short trader's P&L = −(price return). Their "win" is a price fall, and
    # their "drawdown" (max adverse excursion) is the max RISE from entry.
    # Seasonax fixes the perspective to long-only — we surface both so the
    # trader picks the side they're on.
    # SEAS-R2: short wins are the strict `losses` bucket and short losses the
    # strict `gains` bucket — ties belong to neither side (see docstring #2).
    short_rets = [-v for v in rets]
    short_gains = losses
    short_losses = gains
    short_avg_gain = (-sum(r["ret_pct"] * r["weight"] for r in short_gains) / max(w_loss, 1e-9)) if short_gains else 0.0
    short_avg_loss = (-sum(r["ret_pct"] * r["weight"] for r in short_losses) / max(w_gain, 1e-9)) if short_losses else 0.0
    short_max_gain = max(short_rets)  # biggest fall = biggest short win
    short_max_loss = min(short_rets)  # biggest rally = biggest short loss
    short_w_mean = -w_mean
    short_w_med = -w_med
    short_ann_ret = -ann_ret
    # Volatility is symmetric; Sortino needs recomputing against short returns' downside
    short_downside = [(v - short_w_mean) ** 2 * w for v, w in [(-v, w) for v, w in weighted_rets] if v < short_w_mean]
    if short_downside and sw > 0:
        sd_var = sum(short_downside) / sw
        sd_std = sd_var ** 0.5
        sd_ann = sd_std * (ann_factor ** 0.5)
        short_sortino = short_ann_ret / sd_ann if sd_ann > 1e-9 else 0.0
    else:
        short_sortino = 0.0
    short_sharpe = short_ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

    # ── SEAS-R2: SECONDARY EQUAL-WEIGHT ("raw") LENS ─────────────────────────
    # Every number above is BLEND-weighted, i.e. computed with the same weights
    # that draw the plotted median line. The plain one-year-one-vote view is
    # still valuable (it is the honest historical base rate, free of cycle and
    # recency opinion) so it ships alongside rather than being discarded.
    _n = len(rows)
    _sorted = sorted(rets)
    raw_median = (_sorted[_n // 2] if _n % 2
                  else 0.5 * (_sorted[_n // 2 - 1] + _sorted[_n // 2]))
    raw_mean = sum(rets) / _n
    raw_up, raw_dn, raw_tie = len(gains), len(losses), len(ties)
    raw_decided = raw_up + raw_dn
    raw_win_rate = (raw_up / raw_decided) if raw_decided else 0.5
    raw_short_win_rate = (raw_dn / raw_decided) if raw_decided else 0.5

    # ── SEAS-R2: LINE vs STATS RECONCILIATION ────────────────────────────────
    # curve_move_pct: rebase every year to 0 at start_td and take the weighted
    #   median of the window move. Identical to the headline median by
    #   construction — this is the number the *shape* of the blend line implies.
    # curve_level_move_pct: the compounded slope of the plotted median-OF-LEVELS
    #   line over the same TDs. Differs whenever the identity of the median year
    #   changes inside the window.
    curve_move_pct = round(w_med, 3)
    curve_level_move_pct = None
    if wstats:
        _medline = wstats.get("median") or []
        if end_td < len(_medline) and start_td < len(_medline):
            _a, _b = _medline[start_td], _medline[end_td]
            if _a is not None and _b is not None and abs(1.0 + _a / 100.0) > 1e-9:
                curve_level_move_pct = round(
                    ((1.0 + _b / 100.0) / (1.0 + _a / 100.0) - 1.0) * 100.0, 3)

    return {
        "window_stats": {
            "win_rate": round(win_rate, 4),
            "w_median": round(w_med, 3),
            "w_mean": round(w_mean, 3),
            "ann_return": round(ann_ret, 3),
            "avg_gain": round(avg_gain, 3),
            "avg_loss": round(avg_loss, 3),
            "max_gain": round(max_gain, 3),
            "max_loss": round(max_loss, 3),
            "volatility": round(ann_vol, 3),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "n_years": len(rows),
            "n_gains": len(gains),
            "n_losses": len(losses),
            "n_ties": len(ties),
            "td_window": td_window,
            "cal_days": round(td_window * 365.0 / 252.0),
            # ── SEAS-R2: THE ONE SOURCE OF TRUTH ────────────────────────────
            # Donut, callout and header must all read `headline`. lens='blend'
            # means these are weighted by the SAME year weights that draw the
            # plotted median line, so the number and the picture cannot diverge.
            "headline": {
                "lens": "blend",
                "win_rate": round(win_rate, 4),
                "short_win_rate": round(short_win_rate, 4),
                "median_pct": round(w_med, 3),
                "mean_pct": round(w_mean, 3),
                "n_up": len(gains),
                "n_down": len(losses),
                "n_ties": len(ties),
                "n_years": len(rows),
                "w_up": round(w_gain, 4),
                "w_down": round(w_loss, 4),
                "w_ties": round(w_tie, 4),
                "tie_eps_pct": _eps,
                "compounded": True,
                "excludes_current_year": True,
            },
            # Secondary equal-weight lens (one year, one vote)
            "raw_win_rate": round(raw_win_rate, 4),
            "raw_short_win_rate": round(raw_short_win_rate, 4),
            "raw_median": round(raw_median, 3),
            "raw_mean": round(raw_mean, 3),
            "raw_n_gains": raw_up,
            "raw_n_losses": raw_dn,
            "raw_n_ties": raw_tie,
            # Line/stats reconciliation
            "curve_move_pct": curve_move_pct,
            "curve_level_move_pct": curve_level_move_pct,
            # ── Short-side mirror (perspective flipped) ──────────────────────
            "short": {
                "win_rate": round(short_win_rate, 4),
                "w_median": round(short_w_med, 3),
                "w_mean": round(short_w_mean, 3),
                "ann_return": round(short_ann_ret, 3),
                "avg_gain": round(short_avg_gain, 3),
                "avg_loss": round(short_avg_loss, 3),
                "max_gain": round(short_max_gain, 3),
                "max_loss": round(short_max_loss, 3),
                "volatility": round(ann_vol, 3),
                "sharpe": round(short_sharpe, 3),
                "sortino": round(short_sortino, 3),
                "n_wins": len(short_gains),
                "n_losses": len(short_losses),
                "n_ties": len(ties),
                "raw_win_rate": round(raw_short_win_rate, 4),
                "raw_median": round(-raw_median, 3),
                "raw_mean": round(-raw_mean, 3),
            },
        },
        "per_year_windows": rows,
    }

@app.get("/api/seasonality-lab")
async def get_seasonality_lab(
    market: str,
    cycle_w: float = None,
    halflife: float = None,
    detrend: bool = False,
    basket: str = None,          # comma-sep year list, e.g. "2020,2016,2012"
    window_start: int = None,    # trading day index (0-251)
    window_end: int = None,
):
    """Seasonality Lab v2. Returns per-year raw paths + meta + months axis,
    PLUS optional weighted average/median/quantiles when cycle_w/halflife given,
    PLUS optional per-year window stats when window_start/window_end given.
    Server-side blend guarantees the same math is used for scoring, charts,
    and Sunday Setup regardless of caller."""
    m = market.upper()
    try:
        await asyncio.get_event_loop().run_in_executor(
            _APP_EXECUTOR, _ensure_market_seas, m)
    except Exception as _e:
        print(f"[seas lab api] ensure {m}: {_e}", flush=True)
    data = _load_seas_data()
    ent = data.get(m)
    if not ent or ent.get("v") != 2 or not ent.get("years"):
        return {"error": f"Seasonality for '{m}' is still building or unavailable"}
    import datetime as _sl_dt
    _today = _sl_dt.date.today()
    _doy = _today.timetuple().tm_yday
    _current_td = max(1, min(252, round((_doy / 365) * 252)))
    _months_axis = []
    for _mo in range(1, 13):
        _md = _sl_dt.date(_today.year, _mo, 1)
        _mtd = max(1, min(252, round((_md.timetuple().tm_yday / 365) * 252)))
        _months_axis.append({"td": _mtd, "label": _md.strftime("%b")})
    years = ent["years"]

    # Optional basket filter (comma-separated year list)
    if basket:
        try:
            bset = set(str(int(y.strip())) for y in basket.split(",") if y.strip())
            if bset:
                years = {ys: p for ys, p in years.items() if ys in bset}
        except Exception:
            pass

    # Optional detrend: subtract per-year linear drift so paths start & end at 0
    if detrend and years:
        det = {}
        for ys, path in years.items():
            if not path:
                det[ys] = path
                continue
            n = len(path)
            end_v = path[-1] if path[-1] is not None else 0.0
            det[ys] = [
                (v - (end_v * i / max(1, n - 1))) if v is not None else None
                for i, v in enumerate(path)
            ]
        years = det

    meta = {}
    for ys in years.keys():
        y = int(ys)
        meta[ys] = {
            "cycle": _cycle_key_for_year(y),
            "even": y % 2 == 0,
            "ret": years[ys][-1] if years[ys] else None,
        }

    # Default blend from asset class
    default_blend = _seas_default_blend(m)
    # Resolve effective blend (query params override default)
    eff_cycle_w = float(cycle_w) if cycle_w is not None else default_blend["cycle_w"]
    eff_halflife = float(halflife) if halflife is not None else default_blend["halflife"]

    # Compute weighted paths + reliability grade
    weights = _seas_year_weights(
        list(years.keys()), eff_cycle_w, eff_halflife,
        _today.year, _cycle_key_for_year(_today.year))
    # SEAS-R2 (Issue 2): pass the current year/TD so the plotted blend line stops
    # being dominated by this year's forward-filled pad past today.
    wstats = _seas_weighted_stats(years, weights,
                                  current_year=_today.year,
                                  current_td=_current_td)

    # Reliability grade: based on # years, signal/noise across full 252, direction consistency
    def _grade():
        n = len(years)
        med = [v for v in wstats["median"] if v is not None]
        p25 = [v for v in wstats["p25"] if v is not None]
        p75 = [v for v in wstats["p75"] if v is not None]
        if n < 8 or not med:
            return {"grade": "D", "score": 0.0, "n": n, "sn": 0.0, "consistency": 0.0}
        # Signal = |median[-1]|; Noise = avg (p75-p25)
        sig = abs(med[-1]) if med else 0.0
        noise = sum((a - b) for a, b in zip(p75, p25) if a is not None and b is not None)
        noise = noise / max(1, len(p25))
        sn = sig / max(noise, 1e-6)
        # Direction consistency at fwd_td+60 (~2 mo out from current_td)
        target_td = min(251, _current_td + 60)
        med_dir = 1 if (med[target_td] if target_td < len(med) else 0) > (med[_current_td] if _current_td < len(med) else 0) else -1
        agree = 0
        total = 0
        for ys, path in years.items():
            if target_td < len(path) and _current_td < len(path) and path[target_td] is not None and path[_current_td] is not None:
                total += 1
                if (path[target_td] > path[_current_td]) == (med_dir > 0):
                    agree += 1
        consistency = agree / max(1, total)
        # Combined score 0-100
        score = min(100.0, 30.0 * min(sn, 3.0) / 3.0 + 40.0 * consistency + 30.0 * min(n, 30) / 30.0)
        # Grade thresholds
        if score >= 75: g = "A"
        elif score >= 60: g = "B"
        elif score >= 45: g = "C"
        else: g = "D"
        return {"grade": g, "score": round(score, 1), "n": n,
                "sn": round(sn, 3), "consistency": round(consistency, 3)}
    reliability = _grade()

    # Optional window stats (only when both bounds given)
    window_result = {"window_stats": None, "per_year_windows": []}
    if window_start is not None and window_end is not None:
        try:
            ws = int(window_start)
            we = int(window_end)
            if 0 <= ws < 252 and 0 <= we < 252 and ws < we:
                window_result = _seas_lab_window_stats(
                    years, weights, ws, we,
                    current_year=_today.year, current_td=_current_td,
                    wstats=wstats)   # SEAS-R2: enables curve_level_move_pct
        except Exception as _we:
            print(f"[seas lab] window exc: {_we}", flush=True)

    # v2.1 consensus — computed on the FULL year set (independent of basket filter)
    try:
        _consensus = _seas_consensus(ent["years"], _current_td, _today.year, m)
    except Exception as _ce:
        print(f"[seas lab] consensus exc: {_ce}", flush=True)
        _consensus = None

    # SEAS-R2 (Issue 2): composition-free forward blend path, rebased to today
    # (or to the selected window start when the user has one). See
    # _seas_weighted_rebased for why the median-of-levels line needs this.
    _reb_anchor = _current_td
    try:
        if window_start is not None and 0 <= int(window_start) < 252:
            _reb_anchor = int(window_start)
    except (TypeError, ValueError):
        pass
    try:
        _wreb = _seas_weighted_rebased(years, weights, _reb_anchor,
                                       current_year=_today.year,
                                       current_td=_current_td)
    except Exception as _re:
        print(f"[seas lab] rebased exc: {_re}", flush=True)
        _wreb = None

    # ══ SEAS-R2 (Issue 1): SWING SETUP PLANNER ON THE LAB ═══════════════════
    # Same engine and same lens as the scoring path, but built on the LAB's
    # EFFECTIVE blend so the entry/exit legs are computed on the exact curve the
    # user is looking at, including any cycle_w / halflife they dialled in. If
    # the plotted line and the plan ever disagreed the plan would be worthless.
    _planner = None
    try:
        _planner = _seas_swing_planner(
            m, ent["years"], _current_td, _today.year, _today,
            cycle_w=eff_cycle_w, halflife=eff_halflife)
    except Exception as _pe:
        print(f"[seas lab] planner exc: {_pe}", flush=True)
        _planner = None

    # ══ SEAS-R2: goldilocks entry timing + horizon reads on the lab ══════════
    # Round-1 queued item. The scores page carried goldilocks / entry-timing but
    # the Lab did not, so the two screens gave the trader different timing
    # advice. Sourced from the SAME _seas_window_stats the headline score uses,
    # so the Lab cannot contradict the score card.
    _gold = None
    try:
        _ws2 = _seas_window_stats(m, _today)
        if _ws2:
            _gold = {
                "score": _ws2.get("score"),
                "raw_score": _ws2.get("raw_score"),
                "near_only_score": _ws2.get("near_only_score"),
                "reliability_grade": _ws2.get("reliability_grade"),
                "seas_shape": _ws2.get("seas_shape"),
                "shape_rotated": _ws2.get("shape_rotated"),
                "imm_score": _ws2.get("imm_score"),
                "imm_median_pct": _ws2.get("imm_median_pct"),
                "imm_hit_rate": _ws2.get("imm_hit_rate"),
                "far_score": _ws2.get("far_score"),
                "far_median_pct": _ws2.get("far_median_pct"),
                "horizon_w_imm": _ws2.get("horizon_w_imm"),
                "days_to_goldilocks": _ws2.get("days_to_goldilocks"),
                "entry_timing": _ws2.get("entry_timing"),
                "entry_note": _ws2.get("entry_note"),
                "goldilocks_dir": _ws2.get("goldilocks_dir"),
                "goldilocks_lean": _ws2.get("goldilocks_lean"),
                "goldilocks_ratio": _ws2.get("goldilocks_ratio"),
                "goldilocks_clipped": _ws2.get("goldilocks_clipped"),
                "goldilocks_curve": _ws2.get("goldilocks_curve"),
                "planner_mode": _ws2.get("planner_mode"),
                "planner_mult": _ws2.get("planner_mult"),
                "planner_capture": _ws2.get("planner_capture"),
            }
    except Exception as _ge:
        print(f"[seas lab] goldilocks exc: {_ge}", flush=True)
        _gold = None

    return _SafeJSONResponse({
        "market": m,
        "years": years,
        "meta": meta,
        "months": _months_axis,
        "current_td": _current_td,
        "current_year_actual": _get_current_year_actual(m),
        "current_year": _today.year,
        "current_cycle": _cycle_key_for_year(_today.year),
        "years_span": ent.get("years_span"),
        # NEW v2 fields
        "asset_class": _seas_asset_class(m),
        "default_blend": default_blend,
        "effective_blend": {"cycle_w": eff_cycle_w, "halflife": eff_halflife, "detrend": detrend},
        "weights": weights,
        "weighted": wstats,
        "reliability": reliability,
        "window": window_result,
        # v2.1: cross-lens consensus conviction — the scoring-grade read
        "consensus": _consensus,
        # SEAS-R2 (Issue 2): blend path REBASED to today — the composition-free
        # forward view. Its endpoint equals the window median by construction,
        # so the plotted shape and the reported stats cannot diverge.
        "weighted_rebased": _wreb,
        # SEAS-R2 (Issue 1): swing setup planner — entry + exit legs, both sides
        "planner": _planner,
        # SEAS-R2 (Issue 3): goldilocks entry timing + horizon reads, from the
        # same engine as the headline score
        "goldilocks": _gold,
    })

# ============================================================
# RELVAL DETAIL ENDPOINT — returns full relval detail incl. chart lines on demand
# ============================================================
_RELVAL_RESULT_CACHE: dict = {}
_RELVAL_RESULT_TTL = 3600  # 1 hour

@app.get("/api/relval")
async def get_relval_detail(market: str):
    """Returns full relval detail (incl. chart lines) for a single market. Cached 1h."""
    m_upper = market.upper()
    mkt = next((x for x in MARKETS if x["id"] == m_upper), None)
    if not mkt:
        return {"error": f"Market '{m_upper}' not found"}
    _rn = time.time()
    _rc = _RELVAL_RESULT_CACHE.get(m_upper)
    if _rc and (_rn - _rc["ts"]) < _RELVAL_RESULT_TTL:
        return _SafeJSONResponse(_rc["data"])
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_APP_EXECUTOR, compute_rel_val_score, m_upper)
    _RELVAL_RESULT_CACHE[m_upper] = {"ts": time.time(), "data": data}
    return _SafeJSONResponse(data)

# ============================================================
# COT HISTORY ENDPOINT — returns weekly net positions for charting
# ============================================================
_COT_HIST_RESULT_CACHE: dict = {}
_COT_HIST_RESULT_TTL = 3600  # 1 hour

@app.get("/api/cot-history")
async def get_cot_history(market: str):
    """Returns weekly COT positions (result-cached 1h)."""
    m_upper = market.upper()
    mkt = next((x for x in MARKETS if x["id"] == m_upper), None)
    if not mkt:
        return {"error": f"Market '{m_upper}' not found"}
    _rn4 = time.time()
    _rc4 = _COT_HIST_RESULT_CACHE.get(m_upper)
    if (_rc4 and (_rn4 - _rc4["ts"]) < _COT_HIST_RESULT_TTL
            and not (mkt.get("ice_code") and _ice_disk_mtime(mkt["ice_code"]) > _rc4["ts"] + 1.0)):
        return _SafeJSONResponse(_rc4["data"])

    # ── FX CROSS PAIR: synthetic 3-category history (commercials / large specs /
    #    small specs), built from the two USD legs the SAME way the live scorer does
    #    (OI-normalised net spread, base − quote). Returns the standard market shape
    #    so the cross renders through the normal 3-category COT view & charts.
    if mkt.get("cross"):
        base_id  = mkt["base_leg"]
        quote_id = mkt["quote_leg"]
        base_mkt  = next((x for x in MARKETS if x["id"] == base_id),  None)
        quote_mkt = next((x for x in MARKETS if x["id"] == quote_id), None)
        if not base_mkt or not quote_mkt:
            return {"error": f"Leg markets {base_id}/{quote_id} not found"}
        df_base  = await fetch_cot_history(base_mkt["cftc_code"],  base_mkt["name"])
        df_quote = await fetch_cot_history(quote_mkt["cftc_code"], quote_mkt["name"])
        if df_base is None or df_base.empty or df_quote is None or df_quote.empty:
            return {"market": m_upper, "dates": [], "cross": True,
                    "comm_net": [], "lspec_net": [], "sspec_net": [],
                    "base_id": base_id, "quote_id": quote_id}

        def rolling_briese_arr(arr, window=520):
            result = []
            for i in range(len(arr)):
                sl = arr[max(0, i - window + 1): i + 1]
                lo, hi = min(sl), max(sl)
                if hi == lo: result.append(50.0)
                else: result.append(round((arr[i] - lo) / (hi - lo) * 100, 1))
            return result

        # Align the two legs by date (nearest weekly), then build the 3-category spread.
        b = df_base[["date", "comm_net", "lspec_net", "sspec_net", "open_interest_all"]].copy()
        q = df_quote[["date", "comm_net", "lspec_net", "sspec_net", "open_interest_all"]].copy()
        b["date"] = pd.to_datetime(b["date"]).dt.tz_localize(None).dt.normalize()
        q["date"] = pd.to_datetime(q["date"]).dt.tz_localize(None).dt.normalize()
        mrg = pd.merge_asof(b.sort_values("date"), q.sort_values("date"), on="date",
                            direction="nearest", tolerance=pd.Timedelta(days=10),
                            suffixes=("_b", "_q")).dropna()
        if len(mrg) < 12:
            return {"market": m_upper, "dates": [], "cross": True,
                    "comm_net": [], "lspec_net": [], "sspec_net": [],
                    "base_id": base_id, "quote_id": quote_id}
        oi_b = mrg["open_interest_all_b"].replace(0, np.nan)
        oi_q = mrg["open_interest_all_q"].replace(0, np.nan)
        ref_oi = (oi_b + oi_q) / 2.0
        cross_cat = {}
        for cat in ("comm_net", "lspec_net", "sspec_net"):
            cross_cat[cat] = ((mrg[f"{cat}_b"] / oi_b - mrg[f"{cat}_q"] / oi_q) * ref_oi)
        comm_net_s  = [int(v) if pd.notna(v) else None for v in cross_cat["comm_net"].round().tolist()]
        lspec_net_s = [int(v) if pd.notna(v) else None for v in cross_cat["lspec_net"].round().tolist()]
        sspec_net_s = [int(v) if pd.notna(v) else None for v in cross_cat["sspec_net"].round().tolist()]
        oi_s        = [int(v) if pd.notna(v) else None for v in ref_oi.round().tolist()]
        dates       = mrg["date"].dt.strftime("%Y-%m-%d").tolist()
        # Per-category rolling Briese indices for the 'index' chart mode
        comm_idx  = rolling_briese_arr([float(v or 0) for v in comm_net_s])
        lspec_idx = rolling_briese_arr([float(v or 0) for v in lspec_net_s])
        sspec_idx = rolling_briese_arr([float(v or 0) for v in sspec_net_s])
        sl = slice(-520, None)   # last ~10yr
        _cot_cross = {
            "market":        m_upper,
            "name":          mkt["name"],
            "cross":         True,
            "base_id":       base_id,
            "quote_id":      quote_id,
            "cross_method":  "3-category net spread (base − quote, OI-normalised)",
            "dates":         dates[sl],
            "comm_net":      comm_net_s[sl],
            "lspec_net":     lspec_net_s[sl],
            "sspec_net":     sspec_net_s[sl],
            "open_interest": oi_s[sl],
            "comm_idx_series":  comm_idx[sl],
            "lspec_idx_series": lspec_idx[sl],
            "sspec_idx_series": sspec_idx[sl],
        }
        _COT_HIST_RESULT_CACHE[m_upper] = {"ts": time.time(), "data": _cot_cross}
        return _SafeJSONResponse(_cot_cross)

    # ── REGULAR MARKET: standard flow ─────────────────────────────────────
    if mkt.get("ice_code"):
        df = await fetch_ice_cot_history(mkt["ice_code"])
    else:
        df = await fetch_cot_history(mkt["cftc_code"], mkt["name"])
    if df is None or df.empty:
        return {"market": m_upper, "dates": [], "comm_net": [], "lspec_net": [], "sspec_net": [], "oi": []}
    # Return last 156 weeks (3 years), or however many we have
    df_out = df.tail(520).copy()  # ~10yr of weekly data
    dates = df_out["date"].dt.strftime("%Y-%m-%d").tolist()
    comm_net  = [int(v) if not pd.isna(v) else None for v in df_out["comm_net"].tolist()]
    lspec_net = [int(v) if not pd.isna(v) else None for v in df_out["lspec_net"].tolist()]
    sspec_net = [int(v) if not pd.isna(v) else None for v in df_out["sspec_net"].tolist()]
    oi        = [int(v) if not pd.isna(v) else None for v in df_out["open_interest_all"].tolist()]
    # Also return Briese index series
    def rolling_briese(arr, window=520):
        result = []
        for i in range(len(arr)):
            sl = arr[max(0,i-window+1):i+1]
            lo, hi = min(sl), max(sl)
            v = arr[i]
            if hi == lo:
                result.append(50.0)
            else:
                result.append(round((v - lo) / (hi - lo) * 100, 1))
        return result
    comm_vals  = [v for v in df_out["comm_net"].tolist()]
    lspec_vals = [v for v in df_out["lspec_net"].tolist()]
    sspec_vals = [v for v in df_out["sspec_net"].tolist()]
    comm_idx_series  = rolling_briese(comm_vals)
    lspec_idx_series = rolling_briese(lspec_vals)
    sspec_idx_series = rolling_briese(sspec_vals)
    _cot_reg_r = {
        "market":           m_upper,
        "name":             mkt["name"],
        "crypto_cot_mode":  bool(mkt.get("crypto_cot_mode", False)),
        "ice_source":       bool(mkt.get("ice_code")),
        "ice_limited_history": bool(mkt.get("ice_limited_history", False)),
        "cot_format":       mkt.get("cot_format", "legacy"),  # 'disagg' or 'tff'
        "dates":            dates,
        "comm_net":         comm_net,
        "lspec_net":        lspec_net,
        "sspec_net":        sspec_net,
        "open_interest":    oi,
        "comm_idx_series":  comm_idx_series,
        "lspec_idx_series": lspec_idx_series,
        "sspec_idx_series": sspec_idx_series,
    }
    _COT_HIST_RESULT_CACHE[m_upper] = {"ts": time.time(), "data": _cot_reg_r}
    return _SafeJSONResponse(_cot_reg_r)

# ============================================================
# SETUP STATS — walk-forward COT-phase evidence engine
# For each market: classify every week since 2008 into its COT phase using
# ONLY data available at the time (rolling Briese indices), then measure what
# price did over the following 4 and 8 weeks. Gives per-phase win rate,
# median move, median adverse excursion and reward:risk — the historical
# evidence behind each COT setup card.
# ============================================================
_SETUP_STATS_CACHE: dict = {}
_SETUP_STATS_TTL = 3600 * 24 * 7   # phases move weekly; recompute weekly

def _fetch_price_weekly_max(yf_ticker: str) -> Optional[pd.DataFrame]:
    """Max-history weekly bars (cached)."""
    cache_key = yf_ticker + "_maxwk"
    now = time.time()
    if cache_key in PRICE_CACHE and (now - PRICE_CACHE.get(cache_key + "_t", 0)) < _SETUP_STATS_TTL:
        return PRICE_CACHE[cache_key]
    try:
        tk = yf.Ticker(yf_ticker)
        df = _yf_with_timeout(tk.history, period="max", interval="1wk", label=yf_ticker + "_maxwk")
        if df is None or df.empty:
            return None
        _price_cache_evict()
        PRICE_CACHE[cache_key] = df
        PRICE_CACHE[cache_key + "_t"] = now
        return df
    except Exception as _e:
        print(f"[fetch_price_weekly_max] {yf_ticker}: {_e}")
        return None


def _compute_setup_stats_sync(market_id: str, px: pd.DataFrame, cot: pd.DataFrame) -> dict:
    """Pure computation: walk-forward phase classification + forward-return stats."""
    MIN_HISTORY = 104   # need 2y of COT before classifying (index stabilisation)
    H_LONG, H_SHORT = 8, 4

    c = cot.copy()
    c["date"] = pd.to_datetime(c["date"]).dt.tz_localize(None).dt.normalize()
    c = c.sort_values("date").reset_index(drop=True)

    def _roll_idx(series):
        s = pd.to_numeric(series, errors="coerce").astype(float)
        lo = s.rolling(520, min_periods=1).min()
        hi = s.rolling(520, min_periods=1).max()
        rng = (hi - lo).replace(0, np.nan)
        return ((s - lo) / rng * 100).fillna(50.0)

    comm_i, lspec_i, sspec_i = _roll_idx(c["comm_net"]), _roll_idx(c["lspec_net"]), _roll_idx(c["sspec_net"])

    pxx = px.copy()
    try:
        pxx.index = pxx.index.tz_localize(None)
    except Exception:
        pass
    pxx = pxx[~pxx.index.duplicated(keep="last")].sort_index()
    closes, highs, lows = pxx["Close"].values, pxx["High"].values, pxx["Low"].values
    pidx = pxx.index

    # Walk the weeks; an 'entry' is the first week a (phase, dir) pair appears.
    prev_key = None
    episodes = []   # (key, cot_row_i)
    phases_seq = []
    for i in range(len(c)):
        ph, pdir, _, _ = _classify_cot_phase(float(comm_i[i]), float(lspec_i[i]), float(sspec_i[i]))
        key = f"{pdir}_p{ph}" if ph > 0 else None
        phases_seq.append(key)
        if i < MIN_HISTORY:
            prev_key = key
            continue
        if key and key != prev_key:
            episodes.append((key, i))
        prev_key = key

    buckets: dict = {}
    first_entry_year = None
    for key, i in episodes:
        d = c["date"].iloc[i]
        # Entry = close of the weekly bar covering the report's release week.
        pos = pidx.searchsorted(d) 
        entry_i = pos - 1 if pos > 0 else 0
        if entry_i < 0 or entry_i + H_LONG >= len(closes):
            continue
        entry = float(closes[entry_i])
        if not np.isfinite(entry) or entry == 0:
            continue
        # Stats are measured in the TRADE direction of the setup, matching how the
        # app surfaces it: P1/P2 = trade with the cycle; P3/P4 (crowded /
        # overstretched) = fade the cycle.
        cyc = 1 if key.startswith("bull") else -1
        ph_num = int(key[-1])
        sign = cyc if ph_num <= 2 else -cyc
        f8 = (float(closes[entry_i + H_LONG]) / entry - 1) * sign
        f4 = (float(closes[entry_i + H_SHORT]) / entry - 1) * sign
        hi_w = highs[entry_i + 1: entry_i + H_LONG + 1]
        lo_w = lows[entry_i + 1: entry_i + H_LONG + 1]
        if sign > 0:
            mfe = float(np.nanmax(hi_w)) / entry - 1
            mae = 1 - float(np.nanmin(lo_w)) / entry
        else:
            mfe = 1 - float(np.nanmin(lo_w)) / entry
            mae = float(np.nanmax(hi_w)) / entry - 1
        b = buckets.setdefault(key, {"f8": [], "f4": [], "mfe": [], "mae": []})
        b["f8"].append(f8); b["f4"].append(f4); b["mfe"].append(max(mfe, 0)); b["mae"].append(max(mae, 0))
        if first_entry_year is None:
            first_entry_year = int(d.year)

    def _trade_dir(key):
        cyc = 1 if key.startswith("bull") else -1
        return "long" if (cyc if int(key[-1]) <= 2 else -cyc) > 0 else "short"

    def _agg(b):
        n = len(b["f8"])
        if n == 0:
            return None
        arr8, arr4 = np.array(b["f8"]), np.array(b["f4"])
        med_mfe = float(np.median(b["mfe"])) * 100
        med_mae = float(np.median(b["mae"])) * 100
        rr = round(med_mfe / med_mae, 2) if med_mae > 0.05 else None
        return {
            "n": n,
            "wr": round(float((arr8 > 0).mean()) * 100, 1),
            "wr4": round(float((arr4 > 0).mean()) * 100, 1),
            "med_ret": round(float(np.median(arr8)) * 100, 2),
            "med_ret4": round(float(np.median(arr4)) * 100, 2),
            "med_mfe": round(med_mfe, 2),
            "med_mae": round(med_mae, 2),
            "rr": rr,
        }

    phases_out = {}
    for k, b in buckets.items():
        v = _agg(b)
        if v:
            v["trade_dir"] = _trade_dir(k)
            phases_out[k] = v

    # Current phase from the latest row (same classification the scorer uses)
    cur_ph, cur_dir, cur_label, _ = _classify_cot_phase(
        float(comm_i.iloc[-1]), float(lspec_i.iloc[-1]), float(sspec_i.iloc[-1]))
    cur_key = f"{cur_dir}_p{cur_ph}" if cur_ph > 0 else None
    return {
        "market": market_id,
        "supported": True,
        "since": first_entry_year,
        "horizon": H_LONG,
        "horizon_short": H_SHORT,
        "current": {"phase": cur_ph, "dir": cur_dir, "label": cur_label, "key": cur_key},
        "current_stats": phases_out.get(cur_key) if cur_key else None,
        "phases": phases_out,
    }


@app.get("/api/setup-stats")
async def get_setup_stats(market: str):
    """Walk-forward historical evidence for the market's COT phases."""
    m_upper = market.upper()
    mkt = next((x for x in MARKETS if x["id"] == m_upper), None)
    if not mkt:
        return {"error": f"Market '{m_upper}' not found"}
    now = time.time()
    rc = _SETUP_STATS_CACHE.get(m_upper)
    if (rc and (now - rc["ts"]) < _SETUP_STATS_TTL
            and not (mkt.get("ice_code") and _ice_disk_mtime(mkt["ice_code"]) > rc["ts"] + 1.0)):
        return _SafeJSONResponse(rc["data"])
    if mkt.get("cross"):
        out = {"market": m_upper, "supported": False, "reason": "cross pair — no native COT phases"}
    elif mkt.get("crypto_cot_mode"):
        out = {"market": m_upper, "supported": False, "reason": "crypto COT mode — phases not applicable"}
    else:
        if mkt.get("ice_code"):
            df = await fetch_ice_cot_history(mkt["ice_code"])
        else:
            df = await fetch_cot_history(mkt.get("cftc_code", ""), mkt.get("name", ""))
        if df is None or len(df) < 150:
            out = {"market": m_upper, "supported": False, "reason": "insufficient COT history"}
        else:
            px = await asyncio.to_thread(_fetch_price_weekly_max, mkt.get("yf", ""))
            if px is None or len(px) < 150:
                out = {"market": m_upper, "supported": False, "reason": "insufficient price history"}
            else:
                try:
                    out = await asyncio.to_thread(_compute_setup_stats_sync, m_upper, px, df)
                except Exception as _e:
                    print(f"[setup-stats] {m_upper}: {_e}", flush=True)
                    out = {"market": m_upper, "supported": False, "reason": "computation error"}
    _SETUP_STATS_CACHE[m_upper] = {"ts": now, "data": out}
    return _SafeJSONResponse(out)


# ============================================================
# PUT/CALL RATIO HISTORY ENDPOINT
# ============================================================
# ── ETF/Crypto PCR history builder (shared by /api/pcr-history and score_pcr) ──
_PCR_ETF_MAP = {"GC": "GLD", "SI": "SLV", "CL": "USO"}

def _build_etf_pcr_series(market_upper: str, refresh: bool = True):
    """
    Build the combined daily PCR series for an ETF-proxied or crypto market:
    direct snapshots (yfinance option-chain OI / Deribit) + scaled ETP proxy
    backfill. Returns (sorted_series [(date_str, pcr), ...], scale_ratio).
    Shared by the /api/pcr-history chart endpoint AND score_pcr so the score
    tile and the chart always read off the same data basis.
    """
    import json as _json
    ticker_cache_file = pathlib.Path(DATA_DIR) / "pcr_ticker_cache.json"
    etp_file = pathlib.Path(DATA_DIR) / "etp_pcr_history.csv"
    proxy_ticker = _PCR_ETF_MAP.get(market_upper)

    def _load_ticker_cache():
        if ticker_cache_file.exists():
            try:
                return _json.loads(ticker_cache_file.read_text())
            except Exception:
                return {}
        return {}

    def _fetch_yf_pcr_today(ticker: str):
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            p_oi = c_oi = 0
            for exp in tk.options:
                try:
                    chain = tk.option_chain(exp)
                    p_oi += float(chain.puts["openInterest"].fillna(0).sum())
                    c_oi += float(chain.calls["openInterest"].fillna(0).sum())
                except Exception:
                    continue
            if c_oi > 0:
                return round(p_oi / c_oi, 4)
        except Exception as e:
            print(f"[PCR-HIST] yfinance {ticker} error: {e}")
        return None

    def _load_etp_proxy():
        if not etp_file.exists():
            return {}
        try:
            df_etp = pd.read_csv(etp_file, index_col=0)
            df_etp.index = pd.to_datetime(df_etp.index)
            col = df_etp.columns[0]
            return {str(dt.date()): float(v) for dt, v in df_etp[col].dropna().items()}
        except Exception as e:
            print(f"[PCR-HIST] ETP proxy load error: {e}")
            return {}

    cache = _load_ticker_cache()
    today_str = str(pd.Timestamp.now().date())
    cache_key = proxy_ticker if proxy_ticker else market_upper
    if cache_key not in cache:
        cache[cache_key] = {}

    if refresh and today_str not in cache[cache_key]:
        if proxy_ticker:
            val = _fetch_yf_pcr_today(proxy_ticker)
        else:
            currency = {"BTC": "BTC", "ETH": "ETH"}.get(market_upper)
            if currency:
                snap = fetch_deribit_pcr(currency)
                val = snap["pcr_oi"] if snap else None
            else:
                val = None
        if val is not None:
            cache[cache_key][today_str] = val
            try:
                ticker_cache_file.write_text(_json.dumps(cache))
            except Exception as e:
                print(f"[PCR-HIST] cache save error: {e}")

    ticker_history = cache.get(cache_key, {})
    etp_proxy = _load_etp_proxy()

    scale_ratio = 1.0
    if ticker_history and etp_proxy:
        common_dates = sorted(set(ticker_history.keys()) & set(etp_proxy.keys()), reverse=True)
        if common_dates:
            latest_common = common_dates[0]
            etp_val = etp_proxy[latest_common]
            tkr_val = ticker_history[latest_common]
            if etp_val and etp_val > 0:
                scale_ratio = tkr_val / etp_val
        else:
            tkr_latest = ticker_history[max(ticker_history.keys())]
            etp_latest = etp_proxy[max(etp_proxy.keys())]
            if etp_latest and etp_latest > 0:
                scale_ratio = tkr_latest / etp_latest

    all_dates = sorted(set(list(etp_proxy.keys()) + list(ticker_history.keys())))
    combined = {}
    for d in all_dates:
        if d in ticker_history:
            combined[d] = ticker_history[d]
        elif d in etp_proxy:
            combined[d] = round(etp_proxy[d] * scale_ratio, 4)

    sorted_series = [(d, v) for d, v in sorted(combined.items()) if v is not None]
    return sorted_series, scale_ratio


@app.get("/api/pcr-history")
async def get_pcr_history(lookback: int = 252, market: str = ""):
    """
    Returns PCR history for charting.
    - Equity markets (ES/NQ/YM/RTY, or market=""): CBOE equity P/C history
    - ETF markets (GC/SI/CL): yfinance daily snapshot cache + ETP proxy backfill
    - Crypto markets (BTC/ETH): Deribit daily snapshot cache + ETP proxy backfill
    lookback: number of trading days to return (default 252 = 1 year)
    market: market ID string (empty or equity ID = equity PCR history)
    """
    import json as _json

    PCR_TICKER_CACHE_FILE = pathlib.Path(DATA_DIR) / "pcr_ticker_cache.json"
    ETP_PCR_FILE = pathlib.Path(DATA_DIR) / "etp_pcr_history.csv"

    ETF_MAP = {"GC": "GLD", "SI": "SLV", "CL": "USO"}

    # ── Helper: load disk cache ───────────────────────────────────────────
    def _load_ticker_cache():
        if PCR_TICKER_CACHE_FILE.exists():
            try:
                return _json.loads(PCR_TICKER_CACHE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_ticker_cache(cache):
        PCR_TICKER_CACHE_FILE.write_text(_json.dumps(cache))

    # ── Helper: fetch today's yfinance PCR snapshot ───────────────────────
    def _fetch_yf_pcr_today(ticker: str) -> float | None:
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            p_oi = c_oi = 0
            for exp in tk.options:
                try:
                    chain = tk.option_chain(exp)
                    p_oi += float(chain.puts["openInterest"].fillna(0).sum())
                    c_oi += float(chain.calls["openInterest"].fillna(0).sum())
                except Exception:
                    continue
            if c_oi > 0:
                return round(p_oi / c_oi, 4)
        except Exception as e:
            print(f"[PCR-HIST] yfinance {ticker} error: {e}")
        return None

    # ── Helper: load ETP proxy CSV ────────────────────────────────────────
    def _load_etp_proxy():
        if not ETP_PCR_FILE.exists():
            return {}
        try:
            df_etp = pd.read_csv(ETP_PCR_FILE, index_col=0)
            df_etp.index = pd.to_datetime(df_etp.index)
            col = df_etp.columns[0]
            return {str(dt.date()): float(v) for dt, v in df_etp[col].dropna().items()}
        except Exception as e:
            print(f"[PCR-HIST] ETP proxy load error: {e}")
            return {}

    # ── Route: equity markets → existing equity P/C series ───────────────
    EQUITY_IDS = {"ES", "NQ", "YM", "RTY", ""}
    if not market or market.upper() in EQUITY_IDS:
        df = fetch_pcr_history()
        if df is None or df.empty:
            return {"error": "Could not fetch P/C ratio data"}

        df_clean = df.dropna(subset=["equity_pc"]).copy()
        if len(df_clean) > lookback:
            df_clean = df_clean.iloc[-lookback:]

        all_ma20 = df.dropna(subset=["pc_ma20"])["pc_ma20"].values

        rows = []
        for idx, row in df_clean.iterrows():
            rows.append({
                "date": str(idx.date()),
                "equity_pc": round(float(row["equity_pc"]), 3),
                "pc_ma5":  round(float(row["pc_ma5"]),  3) if not pd.isna(row.get("pc_ma5",  float("nan"))) else None,
                "pc_ma10": round(float(row["pc_ma10"]), 3) if not pd.isna(row.get("pc_ma10", float("nan"))) else None,
                "pc_ma20": round(float(row["pc_ma20"]), 3) if not pd.isna(row.get("pc_ma20", float("nan"))) else None,
            })

        # Score on 5-day MA (matches scoring logic in score_pcr)
        df_scored5  = df.dropna(subset=["pc_ma5"])
        all_ma5     = df_scored5["pc_ma5"].values
        current_ma5 = float(df_scored5["pc_ma5"].iloc[-1])
        current_pct = float(np.mean(all_ma5 < current_ma5))
        current_score = round(current_pct * 10, 1)

        df_scored20 = df.dropna(subset=["pc_ma20"])
        current_ma20 = float(df_scored20["pc_ma20"].iloc[-1]) if not df_scored20.empty else current_ma5

        return {
            "market": market or "equity",
            "ticker": "CBOE Equity",
            "data": rows,
            "current_ma5":  round(current_ma5,  3),
            "current_ma20": round(current_ma20, 3),
            "current_daily": round(float(df_scored5["equity_pc"].iloc[-1]), 3),
            "current_score": current_score,
            "current_percentile": round(current_pct * 100, 1),
            "scoring_basis": "5-day MA percentile",
            "thresholds": {
                "extreme_greed": round(float(np.percentile(all_ma5, 10)), 3),
                "moderate_greed": round(float(np.percentile(all_ma5, 25)), 3),
                "moderate_fear": round(float(np.percentile(all_ma5, 75)), 3),
                "extreme_fear": round(float(np.percentile(all_ma5, 90)), 3),
            },
            "latest_date": str(df_scored5.index[-1].date()),
        }

    # ── Route: ETF / Crypto markets ───────────────────────────────────────
    market_upper = market.upper()
    proxy_ticker = ETF_MAP.get(market_upper)  # GLD / SLV / USO, or None for crypto

    # Build the combined series via the shared helper (same basis as score_pcr)
    sorted_series, scale_ratio = _build_etf_pcr_series(market_upper, refresh=True)
    if len(sorted_series) > lookback:
        sorted_series = sorted_series[-lookback:]

    if not sorted_series:
        return {"error": f"No PCR history available for {market}"}

    # Compute 20-day MA
    vals_only = [v for _, v in sorted_series]
    ma20_vals = []
    for i in range(len(vals_only)):
        window = vals_only[max(0, i - 19):i + 1]
        ma20_vals.append(round(sum(window) / len(window), 4))

    rows = []
    for i, (d, v) in enumerate(sorted_series):
        rows.append({
            "date": d,
            "pcr": v,
            "pc_ma20": ma20_vals[i],
            # Also populate equity_pc field so frontend chart code works for both paths
            "equity_pc": v,
            "pc_ma10": None,
        })

    # Percentile and thresholds from full series
    all_ma20_vals = [r["pc_ma20"] for r in rows if r["pc_ma20"] is not None]
    all_pcr_vals  = [r["pcr"] for r in rows if r["pcr"] is not None]

    current_ma20 = ma20_vals[-1] if ma20_vals else None
    current_daily = sorted_series[-1][1] if sorted_series else None

    if all_ma20_vals and current_ma20 is not None:
        pct = float(np.mean([v < current_ma20 for v in all_ma20_vals]))
    else:
        pct = 0.5

    # Thresholds from percentiles of the series itself
    if len(all_ma20_vals) >= 10:
        thresholds = {
            "extreme_greed": round(float(np.percentile(all_ma20_vals, 10)), 3),
            "moderate_greed": round(float(np.percentile(all_ma20_vals, 25)), 3),
            "moderate_fear": round(float(np.percentile(all_ma20_vals, 75)), 3),
            "extreme_fear": round(float(np.percentile(all_ma20_vals, 90)), 3),
        }
    else:
        thresholds = {}

    return {
        "market": market_upper,
        "ticker": proxy_ticker or market_upper,
        "source": f"yfinance {proxy_ticker} OI" if proxy_ticker else f"Deribit {market_upper}",
        "scale_ratio": round(scale_ratio, 4),
        "data": rows,
        "current_ma20": round(current_ma20, 3) if current_ma20 else None,
        "current_daily": round(current_daily, 3) if current_daily else None,
        "current_score": round(pct * 10, 1),
        "current_percentile": round(pct * 100, 1),
        "thresholds": thresholds,
        "latest_date": sorted_series[-1][0] if sorted_series else None,
    }


_SEAS_YEARS_BACK = 22
_SEAS_RECENCY = 0.93       # recency decay: age 10 -> 0.48, age 20 -> 0.23
_SEAS_CYCLE_BOOST = 2.5    # boost for years matching current election-cycle position

# AUDIT-SCORING — named swing horizons (were magic numbers scattered inline).
# Ben: "1 to 2 weeks is important, 3 to 4 weeks is important for long-term
# position context, anything beyond six weeks isn't really relevant."
_SEAS_IMM_TD  = 10   # "immediate" 1-2 week read (exposed alongside the near read)
_SEAS_NEAR_TD = 20   # headline 3-4 week swing window
_SEAS_FAR_TD  = 15   # the 3 weeks AFTER the near window (shape / turn detection)
# Goldilocks scan: how far back / forward (trading days) we hunt for the ideal
# seasonal entry offset. ~4 weeks back, ~5 weeks forward — beyond that the
# answer stops being actionable for a 1-4 week swing.
_SEAS_GOLD_BACK = 20
_SEAS_GOLD_FWD  = 25
# How late we're willing to declare an entry. The scan still publishes the full
# -20..+25 curve, but the argmax is capped at -10 so the answer can't pin to the
# back edge of the scan -- 'ideal entry was 20 days ago' is neither actionable
# nor trustworthy; it usually means the real peak is further back still.
_SEAS_GOLD_LATE = 10

# SEAS-R2 (Issue 3) — swing-horizon re-weighting of the HEADLINE seasonal score.
# Round 1 exposed the 10-TD immediate read but did not let it touch the headline.
# Ben: "1-2 week trade positions... maybe 3-4 weeks at a max" — so the headline
# must weight the typical hold, not only the 4-week context window.
# The brief suggested 0.45/0.55. Backtested instead (24,320 zero-lookahead
# samples, 30 markets, holds 10/15/20/28 TD): the optimum sits at 0.25-0.35 on
# BOTH directional hit rate and mean signed return, at every hold, and 0.45 is
# measurably worse than 0.30. Shipping the backtested value.
# Script: /home/user/workspace/r2_imm_near_backtest.py
_SEAS_HORIZON_W_IMM = 0.30
# SEAS-R2 (Issue 4): max fraction of the seasonal lean the swing planner may
# remove. 0.40 keeps the planner a *modifier* — it can move a 7.8 into the 6s
# (what Ben expects for NQ) or a 3.4 toward 4, but it can never cross neutral
# and can never change a factor's sign. See _seas_planner_discount.
_SEAS_PLANNER_DISCOUNT = 0.40


def _seas_wq(vals, ws, q):
    """Weighted quantile of vals with weights ws (0<=q<=1)."""
    pairs = sorted(zip(vals, ws))
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return 0.0
    acc = 0.0
    prev_v = pairs[0][0]
    for v, w in pairs:
        if (acc + w / 2) / tot >= q:
            return v
        acc += w
        prev_v = v
    return prev_v


def _seas_weights(years, asof_year, market_id=None):
    """Per-year weights: election-cycle years matching asof_year's cycle get an
    ASSET-CLASS-SPECIFIC boost (equity indices / major FX / US rates 2.5x,
    FX crosses 1.5x, commodities & crypto 1.0x = pure recency — supply cycles
    dominate politics there); recency decay 0.93^age keeps recent years dominant
    (age 10 -> 0.48, age 20 -> 0.23). Falls back to 2.5x when market unknown."""
    ck = _cycle_key_for_year(asof_year)
    boost = _SEAS_CYCLE_BOOST
    if market_id:
        try:
            boost = float(_seas_default_blend(market_id).get("cycle_w", _SEAS_CYCLE_BOOST))
        except Exception:
            boost = _SEAS_CYCLE_BOOST
    return {y: ((boost if _cycle_key_for_year(y) == ck else 1.0)
                * (_SEAS_RECENCY ** max(0, asof_year - 1 - y))) for y in years}


# AUDIT-SCORING
def _seas_fwd(years_dict, y: int, td_from: int, td_to: int, max_year: int):
    """Forward % return along year `y`'s cumulative seasonal path, from trading
    day `td_from` to `td_to` (both 1-based).

    Handles td_to > 252 by CHAINING into year y+1's path — a window opened in
    November/December legitimately spills into January, and clamping it at 252
    silently shrinks the horizon (a Dec-20 "20-day" window became 8 days, and
    the far window vanished entirely, disabling the whole shape engine for the
    last ~6 weeks of every year).

    `max_year` enforces zero look-ahead: we refuse to chain into year y+1 if
    y+1 is not itself a fully-completed year strictly before the scoring year.
    Returns None whenever the required data isn't available.
    """
    arr = years_dict.get(str(y))
    if not arr or len(arr) < 252:
        return None
    if td_from < 1 or td_to <= td_from:
        return None
    a = arr[min(td_from, 252) - 1]
    if a is None or (1 + a / 100) == 0:
        return None
    if td_to <= 252:
        b = arr[td_to - 1]
        if b is None:
            return None
        return ((1 + b / 100) / (1 + a / 100) - 1) * 100
    # ── Year-wrap: ride year y to its close, then continue into year y+1 ──
    spill = td_to - 252
    if spill < 1 or spill > 252:
        return None
    if y + 1 >= max_year:          # look-ahead guard — y+1 must be complete & prior
        return None
    nxt = years_dict.get(str(y + 1))
    if not nxt or len(nxt) < max(spill, 252):
        return None
    e_y = arr[251]
    s_n = nxt[0]
    e_n = nxt[spill - 1]
    if e_y is None or s_n is None or e_n is None or (1 + s_n / 100) == 0:
        return None
    g1 = (1 + e_y / 100) / (1 + a / 100)
    g2 = (1 + e_n / 100) / (1 + s_n / 100)
    return (g1 * g2 - 1) * 100


def _seas_win_score(rets, ws, min_n: int = 8):
    """Shared 0-10 window score kernel: 5 + 2.5*direction + 2.5*magnitude.
    Returns (score, weighted_hit, weighted_median, iqr_halfwidth) or Nones.
    Single source of truth so the near / immediate / far / goldilocks-scan
    windows can never drift apart."""
    if not rets or len(rets) < min_n:
        return None, None, None, None
    tot_w = sum(ws)
    if tot_w <= 0:
        return None, None, None, None
    hit = sum(w for r, w in zip(rets, ws) if r > 0) / tot_w
    med = _seas_wq(rets, ws, 0.5)
    iqr_half = max(0.5, (_seas_wq(rets, ws, 0.75) - _seas_wq(rets, ws, 0.25)) / 2)
    dir_c = (hit - 0.5) * 2
    mag_c = max(-1.0, min(1.0, med / iqr_half))
    sc = max(0.0, min(10.0, 5 + 2.5 * dir_c + 2.5 * mag_c))
    return sc, hit, med, iqr_half


def _seas_consensus(years_dict, td_a, asof_year, market_id):
    """Cross-lens, multi-horizon seasonal CONSENSUS → conviction (v2.1).

    Lenses: 'all' (every year, recency-weighted only), 'cycle' (only years in
    the current election-cycle position, recency-weighted), 'blend' (asset-class
    default: cycle_w x recency — the Lab's smart blend).
    Horizons: ~1m / ~2m / ~3m forward trading-day windows from today.

    Each of the 9 (lens, horizon) cells votes +1 / 0 / -1 via weighted median
    return AND weighted hit rate. Conviction comes from agreement:
      • cycle & all-years agree across horizons → HIGH (score amplified 1.15x)
      • mixed → MEDIUM (no change) / LOW (0.85x toward neutral)
      • cycle vs all-years directly opposed on 2-3 horizons, and only on a
        genuinely cycle-driven asset → CONFLICT (0.85x / 0.70x)
    """
    try:
        blend = _seas_default_blend(market_id)
    except Exception:
        blend = {"cycle_w": 1.0, "halflife": 15, "tag": None}
    cyc_w = float(blend.get("cycle_w", 1.0))
    hl = float(blend.get("halflife", 15) or 15)
    ck = _cycle_key_for_year(asof_year)
    # Swing-trade horizons: 1-2wk (near), 3wk (mid), 4wk (far).
    # Beyond ~4 weeks isn't relevant to the trading objective — anchor the
    # consensus on the actual holding period, not multi-month drift.
    horizons = [("near", 7), ("mid", 15), ("far", 21)]
    yrs = sorted(int(y) for y in years_dict
                 if int(y) < asof_year and int(y) >= asof_year - _SEAS_YEARS_BACK)

    def _fwd(y, h):
        # AUDIT-SCORING: was clamping b_i at 252, so from ~mid-November every
        # horizon collapsed toward zero length and every cell voted 0 → every
        # market silently dropped to 'low' conviction each December. Now wraps
        # into the following (still completed, still prior) year.
        return _seas_fwd(years_dict, y, td_a, td_a + h, asof_year)

    def _rw(y):
        return 0.5 ** (max(0, asof_year - 1 - y) / hl)

    lenses = [
        ("all",   {y: _rw(y) for y in yrs}),
        ("cycle", {y: _rw(y) for y in yrs if _cycle_key_for_year(y) == ck}),
        ("blend", {y: _rw(y) * (cyc_w if _cycle_key_for_year(y) == ck else 1.0) for y in yrs}),
    ]
    cells = []
    for lname, Wl in lenses:
        for hname, h in horizons:
            pts = [(r, w) for r, w in ((_fwd(y, h), w) for y, w in Wl.items()) if r is not None]
            min_n = 4 if lname == "cycle" else 8
            if len(pts) < min_n:
                cells.append({"lens": lname, "h": hname, "dir": 0,
                              "med": None, "hit": None, "n": len(pts)})
                continue
            rs = [p[0] for p in pts]; wl = [p[1] for p in pts]; tw = sum(wl)
            med = _seas_wq(rs, wl, 0.5)
            hit = (sum(w for r, w in pts if r > 0) / tw) if tw > 0 else 0.5
            dd = 0
            if med > 0.25 and hit > 0.52:
                dd = 1
            elif med < -0.25 and hit < 0.48:
                dd = -1
            cells.append({"lens": lname, "h": hname, "dir": dd,
                          "med": round(med, 2), "hit": round(hit * 100), "n": len(pts)})
    # Cycle-lens relevance scales with the asset's political sensitivity:
    # cycle-heavy (2.5x) → 1.0, mild/fx-cross (1.5x) → ~0.53, supply-cycle
    # commodities (1.0x) → 0.3 (cycle read is context, not signal).
    cyc_rel = max(0.3, min(1.0, 0.3 + (cyc_w - 1.0) / 1.5 * 0.7))
    LW = {"all": 1.0, "cycle": cyc_rel, "blend": 1.25}
    # Front-load the 1-2wk lens — that's the actual trade timing decision.
    HW = {"near": 1.4, "mid": 1.1, "far": 0.8}
    wsum = sum(c["dir"] * LW[c["lens"]] * HW[c["h"]] for c in cells)
    active = [c for c in cells if c["dir"] != 0]
    dom = 1 if wsum > 0 else (-1 if wsum < 0 else 0)
    # Direct cycle-vs-all opposition per horizon — the strongest disagreement
    # signal, but ONLY meaningful for cycle-sensitive assets
    conflicts = 0
    for hname, _h in horizons:
        da = next((c["dir"] for c in cells if c["lens"] == "all" and c["h"] == hname), 0)
        dc = next((c["dir"] for c in cells if c["lens"] == "cycle" and c["h"] == hname), 0)
        if da != 0 and dc != 0 and da != dc:
            conflicts += 1
    # Weighted agreement: each active cell counts by its lens x horizon weight
    tw_active = sum(LW[c["lens"]] * HW[c["h"]] for c in active)
    agree = (round(100 * sum(LW[c["lens"]] * HW[c["h"]] for c in active if c["dir"] == dom)
                   / tw_active) if tw_active > 0 else None)
    # AUDIT-SCORING — conviction multiplier re-scoped to EXTREMES ONLY.
    # Ben's rule ("only really relevant at extremes") applied to signal-vs-signal
    # conflict. Two problems with the old ladder:
    #   1) 0.45–0.65 dampens are large enough to erase a real edge, and they
    #      COMPOUND with the reliability dampen (D=0.40) → 0.18x worst case,
    #      i.e. total annihilation of a merely-mixed read.
    #   2) The conflict gate keyed off cyc_w >= 1.5, so FX crosses took a full
    #      cycle-conflict hit even though cycle_relevance says the cycle lens is
    #      barely meaningful for them (0.53) — internally inconsistent.
    # New ladder gates conflict on cycle_relevance (only genuinely cycle-driven
    # assets can be conflicted BY the cycle lens) and requires all three
    # horizons opposed for the hard dampen. Range is now 0.70–1.15.
    if conflicts >= 3 and cyc_rel >= 0.9:
        level, mult = "conflict", 0.70          # severe: every horizon opposed
    elif conflicts >= 2 and cyc_rel >= 0.9:
        level, mult = "conflict", 0.85
    elif dom == 0 or len(active) < 3:
        level, mult = "low", 0.85
    elif agree is not None and agree >= 85 and len(active) >= 6:
        level, mult = "high", 1.15
    elif agree is not None and agree >= 60:
        level, mult = "medium", 1.0
    else:
        level, mult = "low", 0.85
    return {"cells": cells, "dominant": dom, "agreement_pct": agree,
            "conviction": level, "mult": mult, "n_active": len(active),
            "conflicts": conflicts,
            "horizon_td": {h: n for h, n in horizons},
            "cycle_key": ck, "cycle_w": cyc_w, "cycle_relevance": round(cyc_rel, 2),
            "halflife": hl, "profile": blend.get("tag")}


# ══════════════════════════════════════════════════════════════════════════
# SEAS-R2: SWING SETUP PLANNER  (entry + exit leg engine)
# ──────────────────────────────────────────────────────────────────────────
# Ben (round 2): "we need the tool to be smarter and see that you could get long
# now but you need to be out within 28 days or whatever (the next big seasonal
# dip)".  The round-1 goldilocks scan only answered WHEN TO ENTER. A swing trader
# needs an ENTRY *and* an EXIT.
#
# The planner works on the SAME smart-blend lens as the plotted Lab curve
# (asset-class cycle_w x recency halflife, every available completed year — not
# the 22-year scoring window and not equal weight), so what the eye sees on the
# chart is what the planner trades. It:
#   1. builds the forward blend-weighted median cumulative curve for k = 0..45 TD
#   2. decomposes it into LEGS with a vol-scaled zigzag (amplitude filter, NOT
#      smoothing — Ben: "the devil is in the details in terms of peaks and troughs")
#   3. for each direction emits an actionable plan with a hard exit deadline
#
# Hold is capped at _SEAS_PLAN_MAX_HOLD (20 TD) — Ben: 1-2wk typical, 3-4wk max.
_SEAS_PLAN_FWD      = 45   # how far forward we scan for legs (TD) ~9 weeks
_SEAS_PLAN_MIN_LEG  = 5    # a leg shorter than a trading week isn't tradeable
_SEAS_PLAN_MAX_HOLD = 20   # hard cap on hold (Ben: 3-4 weeks absolute max)
_SEAS_PLAN_WAIT_MAX = 15   # a leg starting >15 TD out is context, not a plan
_SEAS_PLAN_NOW_TD   = 2    # a leg starting inside 2 TD is effectively "now"


def _seas_lens_weights(years_dict, asof_year: int, market_id: str,
                       cycle_w: float = None, halflife: float = None):
    """Year weights for the CHART lens (the smart blend the Lab plots), so the
    planner and the Lab stats cannot diverge from the drawn line.
    Returns (sorted list of completed years, {int_year: weight})."""
    try:
        blend = _seas_default_blend(market_id)
    except Exception:
        blend = {"cycle_w": 1.0, "halflife": 15}
    cw = float(cycle_w) if cycle_w is not None else float(blend.get("cycle_w", 1.0))
    hl = float(halflife) if halflife is not None else float(blend.get("halflife", 15) or 15)
    ck = _cycle_key_for_year(asof_year)
    yrs = sorted(int(y) for y in years_dict if int(y) < asof_year)
    W = {}
    for y in yrs:
        w_rec = (0.5 ** (max(0, asof_year - y) / hl)) if hl > 0 else 1.0
        W[y] = w_rec * (cw if _cycle_key_for_year(y) == ck else 1.0)
    return yrs, W


def _seas_leg_stats(years_dict, yrs, W, td_a: int, k0: int, k1: int,
                    asof_year: int, direction: int):
    """Blend-weighted distribution of the ACTUAL per-year return over the leg
    [td_a+k0, td_a+k1].  Returns (median_pct, win_rate_in_direction, n, n_ties).

    Tie policy (SEAS-R2): an exactly-flat year is excluded from BOTH the
    numerator and the denominator of the win rate. Round 1 counted a 0.0% year
    as a LOSS for long (ret > 0 required) *and* as a loss for short
    (ret < 0 required) — the same year could not win either way, which
    depressed both win rates simultaneously.
    """
    rets, ws = [], []
    for y in yrs:
        v = _seas_fwd(years_dict, y, td_a + k0, td_a + k1, asof_year)
        if v is None:
            continue
        rets.append(v); ws.append(W.get(y, 1.0))
    if len(rets) < 5:
        return None, None, len(rets), 0
    med = _seas_wq(rets, ws, 0.5)
    eps = 1e-9
    w_win = sum(w for r, w in zip(rets, ws) if r * direction > eps)
    w_tie = sum(w for r, w in zip(rets, ws) if abs(r) <= eps)
    w_live = sum(ws) - w_tie
    wr = (w_win / w_live) if w_live > 0 else None
    n_ties = sum(1 for r in rets if abs(r) <= eps)
    return med, wr, len(rets), n_ties


def _seas_zigzag(vals, thresh: float):
    """Amplitude-filtered pivot detection (zigzag). Returns pivot indices.
    NOT a smoother: every value is used verbatim, we only ignore wiggles whose
    amplitude is below `thresh`."""
    n = len(vals)
    if n < 3:
        return list(range(n))
    piv = [0]
    hi = lo = vals[0]
    hi_i = lo_i = 0
    trend = 0
    for i in range(1, n):
        x = vals[i]
        if x > hi:
            hi, hi_i = x, i
        if x < lo:
            lo, lo_i = x, i
        if trend >= 0 and (hi - x) >= thresh and hi_i > piv[-1]:
            piv.append(hi_i); trend = -1
            hi, hi_i = x, i; lo, lo_i = x, i
        elif trend <= 0 and (x - lo) >= thresh and lo_i > piv[-1]:
            piv.append(lo_i); trend = 1
            hi, hi_i = x, i; lo, lo_i = x, i
    tail = hi_i if trend >= 0 else lo_i
    if tail > piv[-1]:
        piv.append(tail)
    if (n - 1) > piv[-1]:
        piv.append(n - 1)
    return piv


def _seas_td_to_date(base_date, td: int):
    """Approximate calendar date `td` TRADING days after base_date
    (252 TD ~ 365 calendar days), snapped off weekends."""
    import datetime as _dt
    if td is None:
        return None
    d = base_date + _dt.timedelta(days=int(round(td * 365.0 / 252.0)))
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return d


def _seas_swing_planner(market_id: str, years_dict: dict, td_a: int,
                        asof_year: int, base_date,
                        cycle_w: float = None, halflife: float = None) -> dict | None:
    """The swing setup planner. See the block comment above.

    Returns {curve, legs, thresh, long: {...}, short: {...}, primary_dir,
             suggested_*} or None when there isn't enough data.
    """
    yrs, W = _seas_lens_weights(years_dict, asof_year, market_id, cycle_w, halflife)
    if len(yrs) < 8:
        return None
    # 1) forward blend-weighted median cumulative curve, rebased to 0 at today
    cum, hits, ns = [0.0], [None], [len(yrs)]
    for k in range(1, _SEAS_PLAN_FWD + 1):
        rets, ws = [], []
        for y in yrs:
            v = _seas_fwd(years_dict, y, td_a, td_a + k, asof_year)
            if v is None:
                continue
            rets.append(v); ws.append(W.get(y, 1.0))
        if len(rets) < 5 or sum(ws) <= 0:
            break
        tw = sum(ws)
        eps = 1e-9
        w_up = sum(w for r, w in zip(rets, ws) if r > eps)
        w_tie = sum(w for r, w in zip(rets, ws) if abs(r) <= eps)
        live = tw - w_tie
        cum.append(_seas_wq(rets, ws, 0.5))
        hits.append((w_up / live) if live > 0 else None)
        ns.append(len(rets))
    if len(cum) < 12:
        return None
    maxk = len(cum) - 1
    # 2) vol-scaled amplitude threshold — self-calibrating from the curve's own
    #    step size, so it means the same thing on ZT (sd ~1%) as on NG (sd ~12%).
    steps = sorted(abs(cum[i + 1] - cum[i]) for i in range(maxk))
    noise = steps[len(steps) // 2] if steps else 0.2
    thresh = max(0.30, min(3.0, 2.2 * noise))
    piv = _seas_zigzag(cum, thresh)
    legs = []
    for j in range(len(piv) - 1):
        i0, i1 = piv[j], piv[j + 1]
        amp = cum[i1] - cum[i0]
        d = 1 if amp > 0 else (-1 if amp < 0 else 0)
        legs.append({
            "start_td": i0, "end_td": i1, "len_td": i1 - i0,
            "amp_pct": round(amp, 2), "dir": d,
            "tradeable": bool((i1 - i0) >= _SEAS_PLAN_MIN_LEG and abs(amp) >= thresh),
        })

    def _plan(direction: int) -> dict:
        out = {
            "dir": direction,
            "plan_action": "no_edge",
            "entry_in_td": None, "exit_in_td": None, "exit_by_date": None,
            "entry_by_date": None, "hold_td": None,
            "expected_move_pct": None, "leg_win_rate": None,
            "next_turn_td": None, "next_turn_type": None,
            "next_entry_in_td": None, "leg_amp_pct": None, "n_years": None,
            "note": None,
        }
        cands = [L for L in legs if L["dir"] == direction and L["tradeable"]]
        chosen = None
        # (a) already inside a favourable leg with >= MIN_LEG TD left to run
        cur = next((L for L in cands if L["start_td"] <= _SEAS_PLAN_NOW_TD), None)
        if cur and (cur["end_td"] - max(0, cur["start_td"])) >= _SEAS_PLAN_MIN_LEG:
            chosen = cur
            entry = max(0, cur["start_td"])
            action = "enter_now"
        else:
            # (b) the next favourable leg that starts inside the actionable window
            nxt = next((L for L in cands
                        if _SEAS_PLAN_NOW_TD < L["start_td"] <= _SEAS_PLAN_WAIT_MAX), None)
            if nxt:
                chosen = nxt
                entry = nxt["start_td"]
                action = "wait"
        if chosen is None:
            far = next((L for L in cands if L["start_td"] > _SEAS_PLAN_WAIT_MAX), None)
            if far:
                out["next_entry_in_td"] = far["start_td"]
                out["note"] = (f"No actionable {'long' if direction > 0 else 'short'} "
                               f"leg inside {_SEAS_PLAN_WAIT_MAX} TD — next one starts "
                               f"{far['start_td']} TD out")
            else:
                out["note"] = (f"No tradeable {'long' if direction > 0 else 'short'} "
                               f"seasonal leg inside the next {maxk} trading days")
            return out
        exit_td = min(chosen["end_td"], entry + _SEAS_PLAN_MAX_HOLD)
        if exit_td - entry < _SEAS_PLAN_MIN_LEG:
            exit_td = min(maxk, entry + _SEAS_PLAN_MIN_LEG)
        med, wr, nyr, nties = _seas_leg_stats(years_dict, yrs, W, td_a, entry,
                                             exit_td, asof_year, direction)
        turn = chosen["end_td"] if chosen["end_td"] < maxk else None
        # the leg that follows the adverse turn, if it points our way again
        after = [L for L in legs if L["dir"] == direction and L["tradeable"]
                 and L["start_td"] >= chosen["end_td"]]
        out.update({
            "plan_action": action,
            "entry_in_td": entry,
            "exit_in_td": exit_td,
            "hold_td": exit_td - entry,
            "entry_by_date": (_seas_td_to_date(base_date, entry).isoformat()
                              if entry is not None else None),
            "exit_by_date": _seas_td_to_date(base_date, exit_td).isoformat(),
            "expected_move_pct": (round(med, 2) if med is not None else None),
            "leg_win_rate": (round(wr, 4) if wr is not None else None),
            "next_turn_td": turn,
            "next_turn_type": (("dip" if direction > 0 else "peak")
                               if turn is not None else None),
            "next_entry_in_td": (after[0]["start_td"] if after else None),
            "leg_amp_pct": chosen["amp_pct"],
            "n_years": nyr,
            "n_ties": nties,
        })
        _side = "long" if direction > 0 else "short"
        _turn = ("" if turn is None else
                 f" — the next seasonal {out['next_turn_type']} lands "
                 f"{turn} TD out")
        if action == "enter_now":
            out["note"] = (f"Enter {_side} now, be out by "
                           f"{out['exit_by_date']} ({exit_td} TD){_turn}")
        else:
            out["note"] = (f"Wait {entry} TD — {_side} entry ~{out['entry_by_date']}, "
                           f"out by {out['exit_by_date']} ({exit_td} TD){_turn}")
        return out

    plans = {"long": _plan(1), "short": _plan(-1)}

    def _quality(p):
        if p["plan_action"] == "no_edge":
            return -1.0
        q = abs(p["expected_move_pct"] or 0.0) * max(0.0, (p["leg_win_rate"] or 0.5) - 0.5) * 2
        if p["plan_action"] == "wait":
            q *= 0.6
        return q
    pl, ps = plans["long"], plans["short"]
    primary = 1 if _quality(pl) >= _quality(ps) else -1
    if _quality(pl) < 0 and _quality(ps) < 0:
        primary = 0
    prim = pl if primary > 0 else (ps if primary < 0 else None)
    return {
        "lens": {"cycle_w": (float(cycle_w) if cycle_w is not None
                             else _seas_default_blend(market_id).get("cycle_w")),
                 "halflife": (float(halflife) if halflife is not None
                              else _seas_default_blend(market_id).get("halflife")),
                 "n_years": len(yrs), "basis": "blend-weighted median (chart lens)"},
        "td_start": td_a,
        "fwd_td": maxk,
        "amp_thresh_pct": round(thresh, 2),
        "curve": [{"k": k, "cum_pct": round(cum[k], 3),
                   "hit": (round(hits[k], 3) if hits[k] is not None else None)}
                  for k in range(len(cum))],
        "legs": legs,
        "long": pl, "short": ps,
        "primary_dir": primary,
        "plan_action": (prim["plan_action"] if prim else "no_edge"),
        "entry_in_td": (prim["entry_in_td"] if prim else None),
        "exit_in_td": (prim["exit_in_td"] if prim else None),
        "exit_by_date": (prim["exit_by_date"] if prim else None),
        "hold_td": (prim["hold_td"] if prim else None),
        "expected_move_pct": (prim["expected_move_pct"] if prim else None),
        "leg_win_rate": (prim["leg_win_rate"] if prim else None),
        "next_turn_td": (prim["next_turn_td"] if prim else None),
        "next_turn_type": (prim["next_turn_type"] if prim else None),
        "note": (prim["note"] if prim else "No tradeable seasonal leg either way"),
        # adaptive default window for the Lab (replaces the hard 60d default)
        "suggested_window": {
            "start_td": (max(0, min(251, td_a + (prim["entry_in_td"] or 0)))
                         if prim else max(0, min(251, td_a))),
            "end_td": (max(1, min(251, td_a + (prim["exit_in_td"] or _SEAS_NEAR_TD)))
                       if prim else max(1, min(251, td_a + _SEAS_NEAR_TD))),
            "hold_td": (prim["hold_td"] if prim else _SEAS_NEAR_TD),
            "dir": primary,
        },
    }


def _seas_planner_discount(score: float, planner: dict, shape_rotated: bool = False):
    """SEAS-R2 (Issue 4) — make the headline score consume the planner.

    Ben: "im also still not convinced that the scoring system takes into account
    all of the seasonality lab analysis since nasdaq seasonal score is still
    reading 7.8 ... the front-loaded leg is mostly spent and there's a dip four
    weeks out". The headline is a 20-TD (_SEAS_NEAR_TD) statistic: it only cares
    where the curve ENDS UP, so a curve that rips for 9 days and then hands the
    move back scores identically to one that grinds up for 20. That is exactly
    the wrong shape for a 1-2 week swing.

    PRINCIPLE — path capture, not window overlap.
    Project the planner's blend-weighted forward curve onto the score's own
    direction, over the score's own window (0 .. NEAR_TD), then ask:

        peak     = best favourable point reached inside the window
        end      = where we are at the END of the window
        capture  = end / peak          (how much of the move SURVIVES)
        off_frac = clip(1 - capture, 0, 1)
        mult     = max(1 - COEF, 1 - COEF * off_frac)

    Read plainly: *what fraction of the seasonal move inside your 4-week window
    is still there when the window closes?*

      - grinds up all window (capture 1.0)  -> off 0    -> 1.00x  ("enter_now
        strong legs stay strong" — ZC, a genuine full-window bull, is untouched)
      - front-loaded then rolls over        -> partial  -> softens (the NQ case)
      - peaks early and gives it ALL back,
        or never goes our way at all        -> off 1.0  -> floor mult
    Purely multiplicative toward 5.0 and floored, so it can never invert the
    factor's sign, and it is clipped downstream to 0-10.

    Deliberately derived from the CURVE, not from the zigzag leg boundaries:
    the leg detector needs an amplitude threshold, and threshold choices should
    not move a headline score. The capture ratio is threshold-free.

    SUPPRESSED when the near/far shape rotation already fired (`shape_rotated`).
    A rotated score is an explicit statement that the P&L lands in the FAR
    window (20-35 TD) and that the near window is the part being traded through
    — re-penalising the near window would double-count and partially undo the
    rotation, which flipped 6A's bear read toward neutral in testing. Same
    precedent as the existing consensus-conflict floor.

    Returns (mult, mode, off_frac, capture).
    """
    lean = score - 5.0
    if abs(lean) < 0.5 or not planner:
        return 1.0, "no_view", None, None
    curve = planner.get("curve") or []
    if not curve:
        return 1.0, "no_curve", None, None
    dirn = 1.0 if lean > 0 else -1.0
    proj = [c["cum_pct"] * dirn for c in curve if c["k"] <= _SEAS_NEAR_TD]
    if len(proj) < 3:
        return 1.0, "no_curve", None, None
    peak = max(proj)
    end = proj[-1]
    # Label the mode from the plan so the UI can explain the number in words.
    p = planner.get("long") if lean > 0 else planner.get("short")
    act = (p or {}).get("plan_action") or "no_edge"
    if act == "no_edge":
        mode = "plan_no_edge"
    elif act == "wait":
        mode = "plan_wait"
    else:
        mode = "plan_enter_now"
    if peak <= 0.05:
        # The curve never meaningfully goes our way anywhere in the window.
        off, capture = 1.0, 0.0
        mode = "window_hostile"
    else:
        capture = end / peak
        off = max(0.0, min(1.0, 1.0 - capture))
        if off > 0.001 and mode == "plan_enter_now":
            mode = "plan_enter_now_late"
        elif off <= 0.001 and mode == "plan_enter_now":
            mode = "plan_enter_now_full"
    if shape_rotated:
        return 1.0, "skipped_shape_rotated", round(off, 3), round(capture, 3)
    mult = max(1.0 - _SEAS_PLANNER_DISCOUNT, 1.0 - _SEAS_PLANNER_DISCOUNT * off)
    return round(mult, 3), mode, round(off, 3), round(capture, 3)


# AUDIT-SCORING-COMPLETE
# Scoring-logic audit finished. Composite-integration agent is clear to start.
# New payload fields available for composite/UI consumption:
#   imm_score / imm_median_pct / imm_hit_rate   — 1-2 week "immediate" horizon
#   days_to_goldilocks (signed int TD) / entry_timing / entry_note
#   goldilocks_lean / goldilocks_ratio / goldilocks_curve / anticipation_nudge
#   n_eff, hit_ci_*_raw, td_end_abs, td_far_end_abs, window_wrapped,
#   shape_rotated, recent_regime_weight
# See /home/user/workspace/seas_audit_scoring.md for the full audit.
def _seas_window_stats(market_id: str, bar_date) -> dict | None:
    """
    Core seasonal statistic engine (v2).

    For the 3-4 week window starting at bar_date, computes the ACTUAL historical
    return over that same trading-day window for each of the last ~22 completed
    years (zero look-ahead: only years strictly before bar_date's year are used).

    Cycle + recency weighted: years matching the scoring year's election-cycle
    position get 2.5x weight, recent years outweigh old (0.93^age).

    Score = 5 + 2.5*direction + 2.5*magnitude where
      direction = (weighted_hit_rate - 0.5) * 2       (how consistently up/down)
      magnitude = clip(weighted_median / IQR_halfwidth) (how big vs typical spread)
    """
    from datetime import date as _date
    seas = _load_seas_data()
    ent = seas.get((market_id or "").upper())
    if not ent or ent.get("v") != 2:
        return None
    years = ent.get("years") or {}
    if hasattr(bar_date, 'date'):
        d = bar_date.date()
    elif isinstance(bar_date, _date):
        d = bar_date
    else:
        try:
            d = pd.to_datetime(bar_date).date()
        except Exception:
            return None
    doy = d.timetuple().tm_yday
    td_a = max(1, min(252, round((doy / 365) * 252)))
    # AUDIT-SCORING: window ends are no longer clamped to 252. Clamping made a
    # Dec-20 "20 trading day" window only 8 days long, and killed the far window
    # (and therefore the entire shape engine) for the last ~6 weeks of every
    # year. _seas_fwd wraps into the next completed year instead.
    td_b = td_a + _SEAS_NEAR_TD
    td_imm = td_a + _SEAS_IMM_TD
    yrs = sorted(int(y) for y in years
                 if int(y) < d.year and int(y) >= d.year - _SEAS_YEARS_BACK)
    rets = []
    used = []
    for y in yrs:
        v = _seas_fwd(years, y, td_a, td_b, d.year)
        if v is None:
            continue
        rets.append(v)
        used.append(y)
    n = len(rets)
    if n < 8:
        return None
    W = _seas_weights(used, d.year, market_id)
    ws = [W[y] for y in used]
    tot_w = sum(ws)
    if tot_w <= 0:
        return None
    whit = sum(w for r, w in zip(rets, ws) if r > 0) / tot_w
    wmed = _seas_wq(rets, ws, 0.5)
    iqr_half = max(0.5, (_seas_wq(rets, ws, 0.75) - _seas_wq(rets, ws, 0.25)) / 2)
    n_pos = sum(1 for r in rets if r > 0)
    dir_c = (whit - 0.5) * 2
    mag_c = max(-1.0, min(1.0, wmed / iqr_half))
    score = round(max(0.0, min(10.0, 5 + 2.5 * dir_c + 2.5 * mag_c)), 1)

    # ── IMMEDIATE HORIZON (AUDIT-SCORING) ──────────────────────────────────
    # Ben: "1 to 2 weeks is important, 3 to 4 weeks is important for long-term
    # position context". The 20-TD near window answers the 4-week question; a
    # 10-TD window answers the 1-2 week question. Both are surfaced so the UI
    # can show them side by side. This does NOT feed the headline score — the
    # headline stays anchored on the 20-TD swing window.
    _imm_rets, _imm_ws = [], []
    for y, w in zip(used, ws):
        v = _seas_fwd(years, y, td_a, td_imm, d.year)
        if v is None:
            continue
        _imm_rets.append(v); _imm_ws.append(w)
    imm_score, _imm_hit, _imm_med, _ = _seas_win_score(_imm_rets, _imm_ws)
    _imm_raw = imm_score
    imm_score = round(imm_score, 1) if imm_score is not None else None
    imm_median = round(_imm_med, 2) if _imm_med is not None else None
    imm_hit_rate = round(_imm_hit * 100) if _imm_hit is not None else None
    # ── Reliability metrics: separate real edge from outlier-driven mirage ──
    # 1) Winsorised median: drop the 2 most extreme years each side, recompute median
    #    (unweighted for interpretability). If the winsorised value collapses toward 0
    #    while raw median is meaningful, the pattern is outlier-driven.
    sorted_rets = sorted(rets)
    trim = 2 if n >= 12 else (1 if n >= 8 else 0)
    trimmed = sorted_rets[trim:len(sorted_rets)-trim] if trim else sorted_rets
    trimmed_med = (sorted(trimmed)[len(trimmed)//2] if len(trimmed) % 2
                   else 0.5*(sorted(trimmed)[len(trimmed)//2-1]+sorted(trimmed)[len(trimmed)//2])) if trimmed else 0.0
    # 2) Cross-year stdev of returns (raw, unweighted) — measures dispersion
    _mean = sum(rets)/n
    _var = sum((r-_mean)**2 for r in rets)/max(1, n-1)
    _sd = _var**0.5
    # 3) Signal-to-noise: median / stdev.  |sn| >= 0.5 is strong, <0.2 is weak.
    sig_noise = (wmed/_sd) if _sd > 0.01 else 0.0
    # 4) Median-absolute-deviation ratio: MAD / |median|. Low = consistent, high = noisy.
    _mad = sorted(abs(r-wmed) for r in rets)
    mad = _mad[n//2] if n % 2 else 0.5*(_mad[n//2-1]+_mad[n//2])
    # AUDIT-SCORING: "meaningful magnitude" must be relative to the market's own
    # dispersion, not an absolute 0.5%. A flat 0.5% floor is vol-blind: it is
    # noise for NG (sd ~12%) but a large, real move for ZT / 6C (sd ~1%), so
    # low-vol markets were being denied reliability points for edges that are
    # genuinely significant *for them*. Scale with sd, keep a small absolute floor.
    _mag_floor = max(0.15, min(0.5, 0.18 * _sd))
    mad_ratio = (mad/abs(wmed)) if abs(wmed) > _mag_floor else None  # undefined for tiny signals
    # 5) Regime-cluster flag: is hit-rate stable across halves of the sample?
    if n >= 12:
        mid = n // 2
        older = rets[:mid]; newer = rets[mid:]
        hr_older = sum(1 for r in older if r > 0)/len(older) if older else 0
        hr_newer = sum(1 for r in newer if r > 0)/len(newer) if newer else 0
        regime_delta = round(abs(hr_newer - hr_older)*100)  # pts diff between halves
    else:
        regime_delta = None
    # 6) Reliability grade — A (real edge) / B (decent) / C (mixed) / D (mirage).
    # Weighted blend of the criteria above.
    _pts = 0
    # Signal-to-noise (0-3 pts)
    _sn_abs = abs(sig_noise)
    if _sn_abs >= 0.6: _pts += 3
    elif _sn_abs >= 0.35: _pts += 2
    elif _sn_abs >= 0.2: _pts += 1
    # Trimmed agrees with raw median (0-2 pts)
    if abs(wmed) > _mag_floor and (trimmed_med * wmed) > 0:  # same sign, meaningful magnitude
        agree = abs(trimmed_med) / abs(wmed)
        if agree >= 0.7: _pts += 2
        elif agree >= 0.4: _pts += 1
    # Hit-rate confidence (0-2 pts) — 90% binomial CI away from 50%.
    # AUDIT-SCORING: the CI was built on the RAW unweighted split (n_pos/n) while
    # the SCORE is built on the weighted hit rate. A market whose recent and
    # cycle-matched years are strongly one-directional but whose raw 22-year
    # split is 12/10 scored strongly yet earned zero confidence points —
    # inconsistent, and it pushed genuinely-graded edges down to C/D.
    # We now test the weighted hit rate against Kish's effective sample size
    # n_eff = (Σw)²/Σw², which is the statistically honest n for weighted data
    # (it PENALISES concentrated weights rather than inventing precision).
    # Raw values are still reported for transparency.
    _n_eff = (tot_w ** 2) / sum(w * w for w in ws) if ws else float(n)
    _n_eff = max(4.0, min(float(n), _n_eff))
    _p_raw = n_pos/n
    _p = whit
    _se = (_p*(1-_p)/_n_eff)**0.5
    _ci_low = _p - 1.645*_se; _ci_high = _p + 1.645*_se
    _se_raw = (_p_raw*(1-_p_raw)/n)**0.5
    _ci_low_raw = _p_raw - 1.645*_se_raw; _ci_high_raw = _p_raw + 1.645*_se_raw
    if _ci_low > 0.55 or _ci_high < 0.45: _pts += 2
    elif abs(_p - 0.5) > 0.15: _pts += 1
    # Regime stability (0-1 pt)
    if regime_delta is not None and regime_delta <= 20: _pts += 1
    # Sample size (0-1 pt)
    if n >= 15: _pts += 1
    # Total: 0-9 pts -> A/B/C/D
    grade = 'A' if _pts >= 7 else ('B' if _pts >= 5 else ('C' if _pts >= 3 else 'D'))

    # ── Reliability-aware score dampening ─────────────────────────
    # A weak/noisy signal (Grade C or D) should NOT be scored as if it's a
    # confident bull/bear read. Pull scores toward neutral proportional to
    # reliability weakness so "Seasonal Bull" doesn't fire on 54% hit rate
    # with CI [42, 76] and s/n = 0.16 (statistically indistinguishable from
    # random). Grade A = full signal; Grade D = 60% dampening toward 5.0.
    raw_score = score
    _dampen_factor = {'A': 1.00, 'B': 0.85, 'C': 0.55, 'D': 0.40}.get(grade, 0.55)
    score = round(5.0 + (raw_score - 5.0) * _dampen_factor, 1)

    # ── Recent-regime override ───────────────────────────────
    # When the last 5 years cluster strongly one-directional (>=4 of 5),
    # the recent regime is telling us something the recency-weighted median
    # is missing due to outlier drag (e.g. one huge +39% year from 2020 or
    # +19% from 2013 masking a run of -8/-8/-5/-1/-5 in recent years).
    # Blend a recent-regime score in so we don't miss regime shifts.
    # AUDIT-SCORING — false-trigger hardening. Two additions:
    #  a) a MAGNITUDE FLOOR relative to the market's own dispersion, so a run of
    #     five ~0.1% drifts in a market that routinely moves 5% no longer
    #     overrides a 22-year read. (The existing median/mean/sum guards catch
    #     sign contradictions but not pure smallness.)
    #  b) the override blend weight now SCALES WITH STRENGTH: a clean 5-of-5 keeps
    #     the original 55%, but a 4-of-5 (one contradicting year) gets 42%. It was
    #     previously flat 55% for both, which let the weaker pattern hit just as
    #     hard as the unanimous one.
    _rr_min_mag = max(0.20, 0.20 * _sd)
    recent5 = [r for r, y in zip(rets, used) if y >= d.year - 5][-5:]
    recent_regime_score = None
    recent_regime_signal = None
    recent_regime_w = None
    if len(recent5) >= 5:
        r5_pos = sum(1 for r in recent5 if r > 0)
        r5_med = sorted(recent5)[len(recent5)//2] if len(recent5) % 2                  else 0.5 * (sorted(recent5)[len(recent5)//2 - 1] + sorted(recent5)[len(recent5)//2])
        # Magnitude guard (r14c): hit-count alone can lie — four tiny +0.5%
        # years next to one -2.3% year is a "4 of 5 up" that's actually bearish
        # in dollar terms. Require the median AND mean of recent5 to agree with
        # the direction the hit-count would suggest before overriding.
        r5_mean = sum(recent5) / 5
        r5_pos_sum = sum(r for r in recent5 if r > 0)
        r5_neg_sum = -sum(r for r in recent5 if r < 0)
        # Only override if clearly one-directional (>=4 of 5) AND the raw
        # score is on the opposite side of neutral AND the magnitudes back it up.
        if r5_pos <= 1:  # 4 or 5 of last 5 down
            # Guard: median AND mean must be negative, and total down > total up
            if r5_med < 0 and r5_mean < 0 and r5_neg_sum > r5_pos_sum and abs(r5_med) >= _rr_min_mag:
                recent_regime_signal = 'bear'
                recent_regime_score = round(max(0.0, 5.0 - abs(r5_med) * 0.15 - 1.5), 1)
                if score > 5.0:
                    recent_regime_w = 0.55 if r5_pos == 0 else 0.42
                    score = round(recent_regime_score * recent_regime_w
                                  + score * (1.0 - recent_regime_w), 1)
        elif r5_pos >= 4:  # 4 or 5 of last 5 up
            # Guard: median AND mean must be positive, and total up > total down
            if r5_med > 0 and r5_mean > 0 and r5_pos_sum > r5_neg_sum and abs(r5_med) >= _rr_min_mag:
                recent_regime_signal = 'bull'
                recent_regime_score = round(min(10.0, 5.0 + abs(r5_med) * 0.15 + 1.5), 1)
                if score < 5.0:
                    recent_regime_w = 0.55 if r5_pos == 5 else 0.42
                    score = round(recent_regime_score * recent_regime_w
                                  + score * (1.0 - recent_regime_w), 1)

    # ── FAR-WINDOW SEASONAL (r15) ──────────────────────────────────
    # The 20-TD near window answers "where does the curve end up in ~4 weeks?"
    # But that misses shape: a seasonal peak inside the window (up then down)
    # can post a positive net return but be a bad long entry. Add a far window
    # (TD_b → TD_b+40, i.e. the 8 weeks AFTER the near window) to detect
    # peak/trough proximity and dampen fading edges.
    # Far-window: 3 weeks (15 TD) beyond the primary 20-TD window.
    # This lets the shape dampener catch a seasonal peak inside a 5-7wk hold
    # without dragging in irrelevant 3-month drift.
    td_c = td_b + _SEAS_FAR_TD
    far_score = None
    far_median = None
    far_hit_rate = None
    seas_shape = 'confirmed'      # confirmed | fading | rising | undefined
    shape_dampen = 1.0            # 1.0 = no dampen; <1.0 = dampen toward 5
    shape_rotated = False         # true when we rotated toward the far side
    if True:
        # AUDIT-SCORING — two bugs fixed here:
        #  1) far weights were taken as ws[:len(far_rets)], a POSITIONAL slice.
        #     Whenever a year was skipped in the far loop but not the near loop
        #     (missing path point, or now: no next-year path to wrap into) every
        #     subsequent weight was silently attached to the wrong year. Weights
        #     are now collected alongside the returns they belong to.
        #  2) the whole block was gated on `td_c > td_b + 5`, which is false once
        #     td_b hits the old 252 clamp → no far window at all from mid-Nov.
        far_rets, _far_ws = [], []
        for y, w in zip(used, ws):
            v = _seas_fwd(years, y, td_b, td_c, d.year)
            if v is None:
                continue
            far_rets.append(v); _far_ws.append(w)
        _fs, _far_hit, _far_med, _ = _seas_win_score(far_rets, _far_ws)
        if _fs is not None:
            far_score = round(_fs, 1)
            far_median = round(_far_med, 2)
            far_hit_rate = round(_far_hit * 100)

            # Classify shape by comparing NEAR vs FAR direction, using the
            # dampened score for near (post-reliability) so a Grade C weak
            # signal doesn't get flagged as a full peak-proximity fade.
            _near_dir_dampened = score - 5.0             # signed − dampened by reliability
            _far_dir_signed = far_score - 5.0
            _near_meaningful = abs(_near_dir_dampened) >= 0.5
            _far_meaningful  = abs(_far_dir_signed) >= 0.5
            if _near_meaningful and _far_meaningful:
                if _near_dir_dampened > 0 and _far_dir_signed < 0:
                    seas_shape = 'fading'   # near bull, far bear → seasonal peak ahead
                    shape_dampen = 0.6
                elif _near_dir_dampened < 0 and _far_dir_signed > 0:
                    seas_shape = 'rising'   # near bear, far bull → seasonal trough ahead
                    shape_dampen = 0.6
                elif (_near_dir_dampened > 0 and _far_dir_signed > 0) or \
                     (_near_dir_dampened < 0 and _far_dir_signed < 0):
                    # Both agree: keep as-is. If far is STRONGER, small reinforcement.
                    if abs(_far_dir_signed) > abs(_near_dir_dampened) * 1.5:
                        seas_shape = 'confirmed (trend building)'
                    else:
                        seas_shape = 'confirmed'
            elif _near_meaningful and not _far_meaningful:
                seas_shape = 'near-only'      # near has a view, far is flat — fine, no adjust
            elif not _near_meaningful and _far_meaningful:
                seas_shape = 'far-only'       # near flat, far has a view — short-term neutral

            # Apply shape adjustment.
            # OLD behaviour: dampen toward 5.0 on fading/rising shapes. That's wrong
            # for a swing horizon — if the seasonal curve peaks inside a 3-4wk hold
            # (near bull, far bear), the position is EXPECTED to give back gains and
            # end lower. That's an actively bearish setup for going long, not a neutral
            # one. Same the other way for a rising trough.
            #
            # NEW behaviour: when shape is fading/rising AND the far signal is
            # meaningfully directional, blend the current (near-dampened) score with
            # the far score. The far side gets 55% weight because the second half of
            # the hold is where P&L actually lands. When far is only mildly directional
            # we fall back toward the old dampen-toward-5 behaviour.
            if seas_shape in ('fading', 'rising') and far_score is not None:
                _far_lean = far_score - 5.0
                if abs(_far_lean) >= 0.5:
                    # blend: 45% current (near-view, already reliability-dampened) + 55% far
                    _blended = score * 0.45 + far_score * 0.55
                    score = round(max(0.0, min(10.0, _blended)), 1)
                    shape_rotated = True
                else:
                    # far isn't strong enough to flip the sign — just dampen as before
                    score = round(5.0 + (score - 5.0) * shape_dampen, 1)
            elif shape_dampen < 1.0:
                score = round(5.0 + (score - 5.0) * shape_dampen, 1)

    # ── SEAS-R2 (Issue 3): SWING-HORIZON RE-WEIGHTING OF THE HEADLINE ───────
    # Round 1 published the 10-TD read but deliberately kept it OUT of the
    # headline, so the number a swing trader stares at was a pure 20-TD (4-week)
    # statistic even though Ben's typical hold is 1-2 weeks:
    #     "typically looking for 1-2 week trade positions... maybe 3-4 weeks at a max"
    # Blend them:  headline = w_imm * imm(10TD) + (1 - w_imm) * near(20TD)
    #
    # w_imm is BACKTESTED, not asserted. The brief suggested 0.45; a 24,320-sample
    # zero-lookahead walk-forward over 30 markets at holds of 10/15/20/28 TD puts
    # the optimum at 0.25-0.35 on BOTH directional hit rate and mean signed
    # return, at every hold and on the horizons-disagree subset, with 0.45
    # measurably worse than 0.30. See _SEAS_HORIZON_W_IMM.
    #
    # Placed AFTER the near/far shape resolution and BEFORE consensus, on purpose:
    #  • the shape engine (fading/rising rotation) is a statement about the 20-40 TD
    #    structure. Blending the 10-TD read in beforehand moved several markets
    #    across the 0.5 "meaningful lean" boundary and silently disabled their
    #    rotation (6A flipped fading → far-only and lost its bear read entirely).
    #  • the immediate read is dampened by the SAME reliability factor before it is
    #    blended, so a grade-D 10-TD mirage cannot out-shout a graded 20-TD edge.
    # _score_seasonality_at routes through this same function, so the historical /
    # backtest path stays automatically consistent with the live weighting.
    near_only_score = score
    horizon_w_imm = 0.0
    imm_score_adj = None
    if _imm_raw is not None:
        imm_score_adj = round(5.0 + (_imm_raw - 5.0) * _dampen_factor, 2)
        horizon_w_imm = _SEAS_HORIZON_W_IMM
        score = round(max(0.0, min(10.0,
                     horizon_w_imm * imm_score_adj
                     + (1.0 - horizon_w_imm) * score)), 1)

    # ── CROSS-LENS CONSENSUS CONVICTION (v2.1) ─────────────────────
    # When cycle-years seasonality AND all-years seasonality point the same
    # way across multiple forward horizons → amplify (high conviction).
    # When lenses/horizons disagree → pull toward neutral. Direct cycle-vs-all
    # opposition on 2+ horizons is the hardest dampen (conflict).
    consensus = None
    conviction_mult = None
    try:
        consensus = _seas_consensus(years, td_a, d.year, market_id)
    except Exception:
        consensus = None
    if consensus:
        conviction_mult = consensus["mult"]
        lean = score - 5.0
        # Never boost a score that leans AGAINST the consensus direction
        if conviction_mult > 1.0 and consensus["dominant"] != 0 and lean * consensus["dominant"] < 0:
            conviction_mult = 1.0
        # When the near/far shape rotation already resolved the horizon disagreement
        # (fading/rising → tilt toward far), don't undo that resolution by applying
        # a "conflict" dampen back to neutral. The shape signal IS the resolution.
        if shape_rotated and conviction_mult < 1.0:
            conviction_mult = max(conviction_mult, 0.85)
        score = round(max(0.0, min(10.0, 5.0 + (score - 5.0) * conviction_mult)), 1)

    # ══ SWING SETUP PLANNER + SCORE WIRING (SEAS-R2, Issues 1 & 4) ══════════
    # Ben: "im also still not convinced that the scoring system takes into account
    # all of the seasonality lab analysis since nasdaq seasonal score is still
    # reading 7.8". It didn't — round 1's goldilocks/leg analysis was published
    # but never consumed. The planner now feeds the headline via a single,
    # explainable, never-inverting multiplicative discount.
    planner = None
    planner_mult = 1.0
    planner_mode = None
    planner_off_frac = None
    try:
        planner = _seas_swing_planner(market_id, years, td_a, d.year, d)
    except Exception as _pe:
        print(f"[seas planner] {market_id}: {_pe}", flush=True)
        planner = None
    planner_capture = None
    if planner:
        (planner_mult, planner_mode, planner_off_frac,
         planner_capture) = _seas_planner_discount(score, planner, shape_rotated)
        if planner_mult != 1.0:
            score = round(max(0.0, min(10.0, 5.0 + (score - 5.0) * planner_mult)), 1)

    # ══ GOLDILOCKS ENTRY TIMING (AUDIT-SCORING) ═════════════════════════════
    # Read A ("am I in the zone NOW?") was already served by the near window.
    # Read B ("am I about to ARRIVE at a seasonal turn?") was not served at all:
    # 21 of 57 markets currently classify as 'far-only' (near flat, far strongly
    # directional) and that shape did literally nothing to the score or payload.
    # 6E, for example, sits at a flat 4.9 while its far window is 1.6 — a hard
    # bearish turn roughly four weeks out, completely invisible to the trader.
    #
    # So: slide the SAME 20-TD swing window across a range of entry offsets and
    # find where its directional lean peaks. That offset is the goldilocks entry.
    #   days_to_goldilocks < 0  → the ideal entry has passed (we're late)
    #   days_to_goldilocks == 0 → ideal entry is NOW
    #   days_to_goldilocks > 0  → ideal entry is that many trading days ahead
    #
    # Bear-side symmetry falls out for free: for a bearish setup we search for the
    # offset whose forward window is most NEGATIVE, and the entry that maximises a
    # coming decline is by definition the LOCAL PEAK before the fall. Same scan,
    # sign flipped by the anchored direction.
    days_to_goldilocks = None
    entry_timing = None
    entry_note = None
    goldilocks_lean = None
    goldilocks_ratio = None
    goldilocks_dir = None
    goldilocks_clipped = False
    anticipation_nudge = 0.0
    goldilocks_curve = []
    try:
        _gold = {}
        for _off in range(-_SEAS_GOLD_BACK, _SEAS_GOLD_FWD + 1):
            _s = td_a + _off
            if _s < 1:
                continue
            _gr, _gw = [], []
            for y, w in zip(used, ws):
                v = _seas_fwd(years, y, _s, _s + _SEAS_NEAR_TD, d.year)
                if v is None:
                    continue
                _gr.append(v); _gw.append(w)
            _gs, _, _, _ = _seas_win_score(_gr, _gw)
            if _gs is None:
                continue
            _gold[_off] = _gs - 5.0            # signed lean, -5 .. +5
        if _gold:
            goldilocks_curve = [{"off": k, "lean": round(v, 2)}
                                for k, v in sorted(_gold.items())]
            # ── Direction anchor ────────────────────────────────────────
            # Follow the headline score when it holds a real view, so the timing
            # read can never contradict the score the trader is staring at. Defer
            # to the FORWARD picture when the score is neutral, or when the coming
            # opportunity is decisively larger and points the other way — which is
            # exactly the fading / rising / far-only case Read B exists to catch.
            _fut = {k: v for k, v in _gold.items() if 0 <= k <= _SEAS_GOLD_FWD}
            _best_fwd = max(_fut.items(), key=lambda kv: abs(kv[1])) if _fut else None
            _lean_now = score - 5.0
            _dir = 0
            if abs(_lean_now) >= 1.0:
                _dir = 1 if _lean_now > 0 else -1
                if (_best_fwd and _best_fwd[1] * _dir < 0
                        and abs(_best_fwd[1]) >= abs(_lean_now) + 1.0):
                    _dir = 1 if _best_fwd[1] > 0 else -1
            elif _best_fwd and abs(_best_fwd[1]) >= 0.5:
                _dir = 1 if _best_fwd[1] > 0 else -1
            if _dir != 0:
                # Best offset in the anchored direction. The argmax is capped at
                # -_SEAS_GOLD_LATE: an answer pinned to the back edge of the scan
                # ('ideal entry was 20 days ago') is a scan artefact rather than a
                # real peak/trough, and it isn't actionable either way. The full
                # curve is still published for the UI. Ties resolve toward NOW — a
                # trader shouldn't be told to wait for a marginally better entry
                # when today is effectively as good.
                _cands = [(k, v) for k, v in _gold.items()
                          if v * _dir > 0 and k >= -_SEAS_GOLD_LATE]
                if _cands:
                    _best_val = max(v * _dir for _, v in _cands)
                    _best_off = min((k for k, v in _cands
                                     if v * _dir >= _best_val - 0.15), key=abs)
                    days_to_goldilocks = int(_best_off)
                    goldilocks_dir = _dir
                    goldilocks_lean = round(_gold[_best_off], 2)
                    goldilocks_clipped = bool(_best_off <= -_SEAS_GOLD_LATE
                                              or _best_off >= _SEAS_GOLD_FWD)
                    _cur = _gold.get(0)
                    if _cur is not None and abs(_best_val) > 1e-9:
                        # How much of the ideal seasonal entry we already have.
                        # Negative = today leans the wrong way entirely.
                        goldilocks_ratio = round(max(-1.0, min(1.0,
                                                (_cur * _dir) / _best_val)), 2)
                    _side = 'long' if _dir > 0 else 'short'
                    _dt = days_to_goldilocks
                    _wks = abs(_dt) / 5.0
                    # Dead zone: ±2 trading days is inside the noise of a 20-day
                    # window statistic. Don't manufacture false precision by
                    # telling a trader they are '1 day late'.
                    if abs(_dt) <= 2:
                        entry_timing = 'now'
                        entry_note = f'Seasonal goldilocks {_side} entry is NOW'
                    elif _dt > 0:
                        entry_timing = ('imminent' if _dt <= 3 else
                                        ('approaching' if _dt <= 10 else 'early'))
                        entry_note = (f'Almost ready seasonally — {_dt} trading days '
                                      f'(~{_wks:.1f}wk) to goldilocks {_side} entry')
                    else:
                        entry_timing = 'just_passed' if _dt >= -5 else 'past'
                        _ago = f'{abs(_dt)}+' if goldilocks_clipped else f'{abs(_dt)}'
                        entry_note = (f'Past the seasonal sweet spot — ideal {_side} '
                                      f'entry was {_ago} trading days ago '
                                      f'(~{_wks:.1f}wk)')
                    # A goldilocks peak that is itself weak isn't worth timing at all.
                    if abs(goldilocks_lean) < 1.0:
                        entry_timing = 'none'
                        entry_note = ('No clear seasonal entry inside the '
                                      '1-4 week swing horizon')
                else:
                    entry_timing = 'none'
                    entry_note = 'No favourable seasonal entry inside the swing horizon'

                # ── ANTICIPATION NUDGE ──────────────────────────────────────
                # Ben's Read B: "if all other fundamentals are strong AND we're
                # close to a good seasonal entry, he still wants a positive-leaning
                # score, but with a clear caveat." So a near-neutral score with a
                # strong turn imminent gets a modest, decaying push toward the
                # coming direction — never enough to masquerade as a live signal,
                # and scaled by reliability so a Grade D mirage barely moves.
                # The caveat itself lives in entry_note / entry_timing.
                if (days_to_goldilocks is not None and 2 < days_to_goldilocks <= 10
                        and entry_timing not in (None, 'none')
                        and goldilocks_lean is not None and abs(goldilocks_lean) >= 1.5
                        and abs(score - 5.0) < 1.0):
                    _prox = (11.0 - days_to_goldilocks) / 10.0     # 1.0 @1d → 0.1 @10d
                    anticipation_nudge = max(-1.2, min(1.2,
                        goldilocks_lean * 0.35 * _prox * _dampen_factor))
                    score = round(max(0.0, min(10.0, score + anticipation_nudge)), 1)
                    anticipation_nudge = round(anticipation_nudge, 2)
    except Exception as _ge:
        print(f"[seas goldilocks] {market_id}: {_ge}", flush=True)

    return {
        "score": score,
        "consensus": consensus,
        "conviction_mult": conviction_mult,
        "raw_score": raw_score,
        "dampen_factor": _dampen_factor,
        "recent5_pos": (sum(1 for r in recent5 if r > 0) if len(recent5) >= 5 else None),
        "recent5_rets": [round(r, 2) for r in recent5] if recent5 else [],
        "recent_regime_signal": recent_regime_signal,
        "recent_regime_score": recent_regime_score,
        "recent_regime_weight": recent_regime_w,
        "n_years": n, "n_pos": n_pos, "n_neg": n - n_pos,
        "n_eff": round(_n_eff, 1),
        "hit_rate": round(whit * 100),
        "raw_hit_rate": round(100 * n_pos / n),
        "median_pct": round(wmed, 2),
        "trimmed_median_pct": round(trimmed_med, 2),
        "mean_pct": round(sum(r * w for r, w in zip(rets, ws)) / tot_w, 2),
        "stdev_pct": round(_sd, 2),
        "signal_noise": round(sig_noise, 2),
        "mad_ratio": (round(mad_ratio, 2) if mad_ratio is not None else None),
        "regime_delta_pts": regime_delta,
        "hit_ci_low": round(_ci_low*100),
        "hit_ci_high": round(_ci_high*100),
        "hit_ci_low_raw": round(_ci_low_raw*100),
        "hit_ci_high_raw": round(_ci_high_raw*100),
        "reliability_grade": grade,
        "reliability_pts": _pts,
        "per_year_rets": [{"y": y, "r": round(r, 2), "w": round(w, 3)}
                          for y, r, w in zip(used, rets, ws)],
        "best_pct": round(max(rets), 2),
        "worst_pct": round(min(rets), 2),
        # Display-clamped bounds (the chart axis is 252 wide); *_abs carry the
        # true unclamped window ends so a Nov/Dec wrap is visible & debuggable.
        "td_start": td_a, "td_end": min(252, td_b),
        "td_end_abs": td_b,
        "window_wrapped": td_b > 252,
        # AUDIT-SCORING: immediate 1-2 week horizon, exposed alongside the near read
        "td_imm_end": min(252, td_imm),
        "imm_td": _SEAS_IMM_TD,
        "near_td": _SEAS_NEAR_TD,
        "imm_score": imm_score,
        "imm_median_pct": imm_median,
        "imm_hit_rate": imm_hit_rate,
        # SEAS-R2: swing-horizon re-weighting of the headline
        "horizon_w_imm": horizon_w_imm,
        "imm_score_adj": imm_score_adj,
        "near_only_score": (round(near_only_score, 1)
                            if near_only_score is not None else None),
        # SEAS-R2: swing setup planner (entry + exit legs) and its score wiring
        "planner": planner,
        "planner_mult": planner_mult,
        "planner_mode": planner_mode,
        "planner_off_frac": planner_off_frac,
        "planner_capture": planner_capture,
        "plan_action": (planner or {}).get("plan_action"),
        "plan_dir": (planner or {}).get("primary_dir"),
        "entry_in_td": (planner or {}).get("entry_in_td"),
        "exit_in_td": (planner or {}).get("exit_in_td"),
        "exit_by_date": (planner or {}).get("exit_by_date"),
        "hold_td": (planner or {}).get("hold_td"),
        "expected_move_pct": (planner or {}).get("expected_move_pct"),
        "leg_win_rate": (planner or {}).get("leg_win_rate"),
        "next_turn_td": (planner or {}).get("next_turn_td"),
        "next_turn_type": (planner or {}).get("next_turn_type"),
        "plan_note": (planner or {}).get("note"),
        # AUDIT-SCORING: goldilocks entry timing (Read B)
        "days_to_goldilocks": days_to_goldilocks,
        "entry_timing": entry_timing,
        "entry_note": entry_note,
        "goldilocks_lean": goldilocks_lean,
        "goldilocks_ratio": goldilocks_ratio,
        "goldilocks_dir": goldilocks_dir,
        "goldilocks_clipped": goldilocks_clipped,
        "anticipation_nudge": anticipation_nudge,
        "goldilocks_curve": goldilocks_curve,
        # r15 far-window & shape
        "td_far_end": min(252, td_c),
        "td_far_end_abs": td_c,
        "far_score": far_score,
        "far_median_pct": far_median,
        "far_hit_rate": far_hit_rate,
        "seas_shape": seas_shape,
        "shape_dampen": shape_dampen,
        "shape_rotated": shape_rotated,
        "years_span": f"{min(used)}\u2013{max(used)}" if used else "",
        "cycle_key": _cycle_key_for_year(d.year),
        "weighting": f"cycle {_seas_default_blend(market_id).get('cycle_w', _SEAS_CYCLE_BOOST)}x + recency 0.93^age",
    }


def score_seasonality(market_id: str) -> dict:
    """
    Public wrapper: compute current seasonality score for a market.
    v2: score is derived from real per-year window statistics (hit rate +
    median return over the next 3-4 week trading-day window) rather than
    point-sampling a noisy averaged curve.
    """
    from datetime import date as _date_cls
    today = _date_cls.today()
    doy = today.timetuple().tm_yday
    current_td = max(1, min(252, round((doy / 365) * 252)))
    cycle_key = _cycle_key_for_year(today.year)

    _ensure_market_seas(market_id)
    stats = _seas_window_stats(market_id.upper(), today)

    if stats is not None:
        score = stats["score"]
    else:
        score = _score_seasonality_at(market_id.upper(), today)

    score = round(float(score), 1)
    if score >= 7.5:
        label = "Strong Seasonal Bull"
    elif score >= 6.0:
        label = "Seasonal Bull"
    elif score >= 4.5:
        label = "Seasonal Neutral"
    elif score >= 2.5:
        label = "Seasonal Bear"
    else:
        label = "Strong Seasonal Bear"

    detail = {
        "score": score,
        "label": label,
        "market_id": market_id.upper(),
        "date": today.isoformat(),
        "current_td": current_td,
        "cycle_key": cycle_key,
        "horizon_td_start": current_td,
        "horizon_td_end": min(252, current_td + 20),
        "source": "stats" if stats is not None else "window",
        # slope_pct kept for backward compat with chart bracket = median window return
        "slope_pct": stats["median_pct"] if stats is not None else None,
    }
    if stats is not None:
        detail.update({
            "n_years": stats["n_years"], "n_pos": stats["n_pos"],
            "n_neg": stats["n_neg"], "hit_rate": stats["hit_rate"],
            "median_pct": stats["median_pct"], "mean_pct": stats["mean_pct"],
            "best_pct": stats["best_pct"], "worst_pct": stats["worst_pct"],
            "years_span": stats["years_span"],
            # Reliability metrics
            "trimmed_median_pct": stats.get("trimmed_median_pct"),
            "stdev_pct": stats.get("stdev_pct"),
            "signal_noise": stats.get("signal_noise"),
            "mad_ratio": stats.get("mad_ratio"),
            "regime_delta_pts": stats.get("regime_delta_pts"),
            "hit_ci_low": stats.get("hit_ci_low"),
            "hit_ci_high": stats.get("hit_ci_high"),
            "reliability_grade": stats.get("reliability_grade"),
            "reliability_pts": stats.get("reliability_pts"),
            "per_year_rets": stats.get("per_year_rets"),
            # r13: dampening + recent regime
            "raw_score": stats.get("raw_score"),
            "dampen_factor": stats.get("dampen_factor"),
            "recent5_pos": stats.get("recent5_pos"),
            "recent5_rets": stats.get("recent5_rets"),
            "recent_regime_signal": stats.get("recent_regime_signal"),
            "recent_regime_score": stats.get("recent_regime_score"),
            # AUDIT-COMPOSITE: these three existed in _seas_window_stats but were
            # dropped on the way out to the API — recent_regime_weight shows how
            # hard the regime override hit, *_abs expose an un-clamped Nov/Dec wrap.
            "recent_regime_weight": stats.get("recent_regime_weight"),
            "td_end_abs": stats.get("td_end_abs"),
            "td_far_end_abs": stats.get("td_far_end_abs"),
            "raw_hit_rate": stats.get("raw_hit_rate"),
            # r15: far-window + shape
            "td_far_end": stats.get("td_far_end"),
            "far_score": stats.get("far_score"),
            "far_median_pct": stats.get("far_median_pct"),
            "far_hit_rate": stats.get("far_hit_rate"),
            "seas_shape": stats.get("seas_shape"),
            "shape_dampen": stats.get("shape_dampen"),
            "shape_rotated": stats.get("shape_rotated"),
            # AUDIT-SCORING: immediate (1-2wk) horizon shown alongside near (3-4wk)
            "imm_score": stats.get("imm_score"),
            # AUDIT-COMPOSITE: explicit alias — the composite payload and the UI
            # both refer to this as `immediate_score`; publish both names so
            # neither side has to know the other's abbreviation.
            "immediate_score": stats.get("imm_score"),
            "imm_median_pct": stats.get("imm_median_pct"),
            "imm_hit_rate": stats.get("imm_hit_rate"),
            # SEAS-R2: swing-horizon re-weighting + swing setup planner
            "horizon_w_imm": stats.get("horizon_w_imm"),
            "imm_score_adj": stats.get("imm_score_adj"),
            "near_only_score": stats.get("near_only_score"),
            "planner": stats.get("planner"),
            "planner_mult": stats.get("planner_mult"),
            "planner_mode": stats.get("planner_mode"),
            "planner_off_frac": stats.get("planner_off_frac"),
            "planner_capture": stats.get("planner_capture"),
            "plan_action": stats.get("plan_action"),
            "plan_dir": stats.get("plan_dir"),
            "entry_in_td": stats.get("entry_in_td"),
            "exit_in_td": stats.get("exit_in_td"),
            "exit_by_date": stats.get("exit_by_date"),
            "hold_td": stats.get("hold_td"),
            "expected_move_pct": stats.get("expected_move_pct"),
            "leg_win_rate": stats.get("leg_win_rate"),
            "next_turn_td": stats.get("next_turn_td"),
            "next_turn_type": stats.get("next_turn_type"),
            "plan_note": stats.get("plan_note"),
            "td_imm_end": stats.get("td_imm_end"),
            "imm_td": stats.get("imm_td"),
            "near_td": stats.get("near_td"),
            # AUDIT-SCORING: goldilocks entry timing (Read B)
            "days_to_goldilocks": stats.get("days_to_goldilocks"),
            "entry_timing": stats.get("entry_timing"),
            "entry_note": stats.get("entry_note"),
            "goldilocks_lean": stats.get("goldilocks_lean"),
            "goldilocks_ratio": stats.get("goldilocks_ratio"),
            "goldilocks_dir": stats.get("goldilocks_dir"),
            "goldilocks_clipped": stats.get("goldilocks_clipped"),
            "anticipation_nudge": stats.get("anticipation_nudge"),
            "goldilocks_curve": stats.get("goldilocks_curve"),
            "n_eff": stats.get("n_eff"),
            "hit_ci_low_raw": stats.get("hit_ci_low_raw"),
            "hit_ci_high_raw": stats.get("hit_ci_high_raw"),
            "window_wrapped": stats.get("window_wrapped"),
            # v2.1: cross-lens consensus conviction
            "consensus": stats.get("consensus"),
            "conviction_mult": stats.get("conviction_mult"),
        })
    return {"score": score, "label": label, "detail": detail}


# ════════════════════════════════════════════════════════════════════════════
# DYNAMIC ROLLING SEASONALITY  (replaces the static 29MB seasonality_all21.json)
# ────────────────────────────────────────────────────────────────────────────
# Curves are computed LIVE from ~22yr of price history per market, prior-years-only
# (zero look-ahead), cycle-weighted (midterm 2.5x) and recency-decayed (0.88^age),
# producing the exact snapshot schema the scorer consumes. Built lazily per market,
# cached in-memory + persisted to a small JSON, and auto-rebuilt weekly so the
# seasonal read is never stale. Markets not yet built fall back to the calendar
# SEASONAL_WINDOWS heuristic in _score_seasonality_at.
_DYN_SEAS_PATH = os.path.join(DATA_DIR, "seasonality_dynamic.json")
_DYN_SEAS_TTL  = 7 * 86400      # rebuild each market weekly
_DYN_SEAS_BUILT: dict = {}      # market_id -> last build ts (in-memory)

_SEAS_BUILDER_VERSION = 5   # v5: ensemble stochastic band (14-framing envelope) replaces 25–75 percentile


def _find_seas_turns(med: list, by_year: dict, prior: list, weights: dict = None) -> list:
    """Detect the significant seasonal peaks/troughs of the median curve via a
    zigzag (reversal threshold = 18% of curve range), then measure how
    consistently prior years actually turned there: for a peak at TD t, the
    (year-weighted) share of years whose path fell over the next 15 TDs
    (rose, for a trough). Only turns where history agrees ≥60% and the median
    forward move matches the turn direction survive; near-duplicates within
    12 TDs collapse to the more consistent one.
    Returns [{td, val, type, consistency, fwd_move}] sorted by td."""
    n = len(med)
    rng = (max(med) - min(med)) or 1.0
    thr = 0.18 * rng
    H = 15                      # forward window (~3 weeks) for consistency
    EDGE = 6                    # ignore year-boundary artefacts
    MIN_SEP = 12                # min TD separation between reported turns

    # ── zigzag extrema ──
    raw = []
    cand_i, cand_v = 0, med[0]
    direction = 0               # +1 tracking a high, -1 tracking a low
    for i in range(1, n):
        v = med[i]
        if direction == 0:
            # bootstrap: pick initial direction once we move thr from start
            if v - med[0] >= thr:
                direction, cand_i, cand_v = 1, i, v
            elif med[0] - v >= thr:
                direction, cand_i, cand_v = -1, i, v
        elif direction > 0:
            if v > cand_v:
                cand_i, cand_v = i, v
            elif cand_v - v >= thr:
                raw.append((cand_i, "peak"))
                direction, cand_i, cand_v = -1, i, v
        else:
            if v < cand_v:
                cand_i, cand_v = i, v
            elif v - cand_v >= thr:
                raw.append((cand_i, "trough"))
                direction, cand_i, cand_v = 1, i, v

    scored = []
    for i, typ in raw:
        if i < EDGE or i > n - 1 - EDGE:
            continue
        w_hit = w_tot = 0.0
        moves = []
        for y in prior:
            p = by_year.get(y)
            if not p:
                continue
            j = min(n - 1, i + H)
            d = p[j] - p[i]
            moves.append(d)
            w = (weights or {}).get(y, 1.0)
            w_tot += w
            if (typ == "peak" and d < 0) or (typ == "trough" and d > 0):
                w_hit += w
        if len(moves) < 6 or w_tot <= 0:
            continue
        cons = round(100.0 * w_hit / w_tot)
        sm = sorted(moves)
        k = len(sm)
        med_move = sm[k // 2] if k % 2 else (sm[k // 2 - 1] + sm[k // 2]) / 2
        # direction sanity: a peak must actually lead down, a trough up
        if typ == "peak" and med_move >= 0:
            continue
        if typ == "trough" and med_move <= 0:
            continue
        scored.append({
            "td": i + 1,
            "val": round(med[i], 2),
            "type": typ,
            "consistency": cons,
            "fwd_move": round(med_move, 2),
        })

    strong = [t for t in scored if t["consistency"] >= 60]
    if len(strong) < 2:   # weak seasonal market: surface the best 2 anyway (≥52%)
        strong = sorted([t for t in scored if t["consistency"] >= 52],
                        key=lambda t: -t["consistency"])[:2]
    strong.sort(key=lambda t: t["td"])

    # collapse near-duplicates: keep the more consistent of any pair within MIN_SEP
    out = []
    for t in strong:
        if out and t["td"] - out[-1]["td"] < MIN_SEP:
            if t["consistency"] > out[-1]["consistency"]:
                out[-1] = t
        else:
            out.append(t)
    return out[:6]


def _build_seasonality_from_closes(closes, current_year: int,
                                   years_back: int = 22, snap_years: int = 11,
                                   market_id: str = None) -> dict:
    """
    Build seasonality v2 structure from a daily Close series.

    {
      "v": 2,
      "years":  {"2004": [252 floats], ...}   raw cumulative %-from-Jan-1 paths,
                                              TD-indexed 1..252, forward-filled,
      "curve":  [[td, val], ...]              cycle-boosted (2.5x matching cycle
                                              years) + recency-weighted (0.93^age)
                                              MEDIAN of prior-year paths, 5-TD smoothed,
      "band":   [[td, p25, p75], ...]         honest interquartile range band,
      "cycles": {"midterm": [[td,val],...], ...}  unweighted median per cycle,
      "n_years", "years_span"
    }

    Median (not mean) so a single outlier year (e.g. PL +124% in 2025) cannot
    bend the whole curve. Zero look-ahead for scoring: the scorer filters
    `years` to those strictly before the bar year.
    """
    closes = closes.dropna()
    try:
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)
    except Exception:
        pass
    by_year: dict = {}
    for yr, grp in closes.groupby(closes.index.year):
        if len(grp) < 30:
            continue
        base = float(grp.iloc[0])
        if base == 0:
            continue
        arr = [None] * 252   # index 0 = TD1
        for ts, px in grp.items():
            doy = ts.timetuple().tm_yday
            td = max(1, min(252, round(doy / 365 * 252)))
            arr[td - 1] = (float(px) / base - 1.0) * 100.0
        last = 0.0
        for i in range(252):
            if arr[i] is None:
                arr[i] = last
            else:
                last = arr[i]
        by_year[int(yr)] = [round(v, 2) for v in arr]

    def _smooth(vals, k=5):
        n = len(vals)
        half = k // 2
        out = []
        for i in range(n):
            lo = max(0, i - half); hi = min(n, i + half + 1)
            seg = vals[lo:hi]
            out.append(sum(seg) / len(seg))
        return out

    prior = sorted(y for y in by_year
                   if y < current_year and y >= current_year - years_back)
    if len(prior) < 8:
        return {}

    W = _seas_weights(prior, current_year, market_id)
    ws = [W[y] for y in prior]
    # Primary median curve (default blend, unsmoothed — the timing signal)
    med_raw = []
    for i in range(252):
        vals = [by_year[y][i] for y in prior]
        med_raw.append(_seas_wq(vals, ws, 0.5))
    med_s = list(med_raw)

    # ── ENSEMBLE BAND ─────────────────────────────────────────────────────
    # Replace the naive 25–75 percentile band (which was just "middle half of
    # this ONE year set" — gives false certainty of a narrow outcome) with a
    # STOCHASTIC ENSEMBLE: compute the weighted median under ~14 alternative
    # reasonable framings (window length, recency decay, cycle blend, parity
    # splits) and take the 15–85% envelope of those medians per TD.
    #
    # Interpretation: "here's the range of plausible seasonal shapes given
    # different reasonable historical framings." Wider where framings disagree
    # (model uncertainty), tighter where they converge. This is honest
    # uncertainty — protects against over-optimising to one specific slice.
    def _cycle_boost(y, boost):
        return boost if _cycle_key_for_year(y) == _cycle_key_for_year(current_year) else 1.0
    def _recency(y, hl):
        return 0.5 ** (max(0, current_year - 1 - y) / hl)
    framings = []
    # (label, year-filter, weight-fn)
    def _make_wf(hl, cycle_boost):
        return lambda y: _recency(y, hl) * _cycle_boost(y, cycle_boost)
    # Vary window length
    for wlen in (10, 15, 20, len(prior)):
        sub = [y for y in prior if y >= current_year - wlen]
        if len(sub) < 6:
            continue
        # Vary recency half-life
        for hl in (10, 15, 25):
            # Vary cycle blend
            for cb in (1.0, 1.5, 2.5):
                framings.append((sub, _make_wf(hl, cb)))
    # Parity splits (US election modulo) — alternative rhythm framings
    even = [y for y in prior if y % 2 == 0]
    odd  = [y for y in prior if y % 2 == 1]
    if len(even) >= 6:
        framings.append((even, _make_wf(15, 1.0)))
    if len(odd) >= 6:
        framings.append((odd, _make_wf(15, 1.0)))
    # Deduplicate identical (subset, weights) approximations by rounded weight tuple
    ensemble_curves = []
    for sub, wf in framings:
        wsub = [wf(y) for y in sub]
        tw = sum(wsub) or 1.0
        wsub = [w / tw for w in wsub]  # normalise — equalises framing scale
        curve_i = []
        for i in range(252):
            vals = [by_year[y][i] for y in sub]
            curve_i.append(_seas_wq(vals, wsub, 0.5))
        ensemble_curves.append(curve_i)
    # Per-TD envelope: 15–85% percentile across ensemble medians
    lo_env, hi_env = [], []
    for i in range(252):
        col = sorted(c[i] for c in ensemble_curves)
        n = len(col)
        if n == 0:
            lo_env.append(med_raw[i]); hi_env.append(med_raw[i]); continue
        # 15/85 percentile via linear interpolation
        def _q(p):
            k = p * (n - 1)
            f = int(k); frac = k - f
            if f + 1 < n:
                return col[f] * (1 - frac) + col[f + 1] * frac
            return col[f]
        lo_env.append(_q(0.15))
        hi_env.append(_q(0.85))
    # Light smoothing (k=5) — envelope is context, not timing; small smooth keeps
    # the band visually calm without hiding genuine disagreement zones.
    lo_s = _smooth(lo_env, k=5)
    hi_s = _smooth(hi_env, k=5)

    curve = [[i + 1, round(med_s[i], 3)] for i in range(252)]
    band = [[i + 1, round(lo_s[i], 3), round(hi_s[i], 3)] for i in range(252)]
    turns = _find_seas_turns(med_s, by_year, prior, W)

    cycles = {}
    for cyc in ("midterm", "pre_election", "post_election", "election"):
        yrs = [y for y in prior if _cycle_key_for_year(y) == cyc]
        if len(yrs) >= 3:
            cm = []
            for i in range(252):
                vals = sorted(by_year[y][i] for y in yrs)
                n = len(vals)
                mid = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
                cm.append(mid)
            # UNSMOOTHED cycle curves — preserve genuine peak/trough dates that
            # a swing trader keys off. Was k=5 smoothed; that shifted turns.
            cycles[cyc] = [[i + 1, round(cm[i], 3)] for i in range(252)]

    return {
        "v": 2,
        "bv": _SEAS_BUILDER_VERSION,
        "years": {str(y): by_year[y] for y in by_year},
        "curve": curve,
        "band": band,
        "turns": turns,
        "cycles": cycles,
        "n_years": len(prior),
        "years_span": f"{min(prior)}\u2013{max(prior)}",
    }

def _load_dyn_seas_file() -> dict:
    try:
        with open(_DYN_SEAS_PATH) as f:
            d = json.load(f)
        for mid, ent in d.items():
            if isinstance(ent, dict) and ent.get("_built"):
                _DYN_SEAS_BUILT[mid] = ent["_built"]
        return d
    except Exception:
        return {}

def _save_dyn_seas_file(data: dict) -> None:
    try:
        with open(_DYN_SEAS_PATH, "w") as f:
            json.dump(data, f)
    except Exception as _e:
        print(f"[seas dyn] save error: {_e}", flush=True)

def _ensure_market_seas(market_id: str) -> None:
    """Ensure dynamic seasonality for one market is built & fresh in the cache."""
    mid = (market_id or "").upper()
    if not mid:
        return
    if _SEASONALITY_CACHE["data"] is None:
        _SEASONALITY_CACHE["data"] = _load_dyn_seas_file()
    data = _SEASONALITY_CACHE["data"]
    now = time.time()
    if (mid in data and data[mid].get("v") == 2 and data[mid].get("curve")
            and data[mid].get("bv") == _SEAS_BUILDER_VERSION
            and (now - _DYN_SEAS_BUILT.get(mid, 0)) < _DYN_SEAS_TTL):
        return
    mkt = next((x for x in MARKETS if x["id"] == mid), None)
    if not mkt or not mkt.get("yf"):
        return
    try:
        from datetime import date as _d
        df = _yf_with_timeout(yf.Ticker(mkt["yf"]).history, period="max", interval="1d",
                              auto_adjust=True, label=f"seas_{mid}")
        if df is None or df.empty or len(df) < 300:
            return
        built = _build_seasonality_from_closes(df["Close"], _d.today().year, market_id=mid)
        if built.get("curve"):
            built["_built"] = now
            data[mid] = built
            _DYN_SEAS_BUILT[mid] = now
            _save_dyn_seas_file(data)
    except Exception as _e:
        print(f"[seas dyn] {mid}: {_e}", flush=True)



def _load_seas_data() -> dict:
    """Return the dynamic seasonality cache (built lazily per market via
    _ensure_market_seas). No longer loads the 29MB static file."""
    if _SEASONALITY_CACHE["data"] is None:
        _SEASONALITY_CACHE["data"] = _load_dyn_seas_file()
    return _SEASONALITY_CACHE["data"] or {}


def _cycle_key_for_year(year: int) -> str:
    """Return presidential cycle position key for a given year."""
    r = year % 4
    if r == 0:
        return "election"
    elif r == 1:
        return "post_election"
    elif r == 2:
        return "midterm"
    else:
        return "pre_election"

def _score_seasonality_at(market_id: str, bar_date) -> float:
    """
    Compute seasonality score for a historical bar date — ZERO lookahead.

    v2: derives the score from real per-year window statistics using only
    years strictly BEFORE bar_date's year (see _seas_window_stats).
    Falls back to the calendar SEASONAL_WINDOWS heuristic when the market
    has no v2 seasonal data yet.
    """
    from datetime import date as _date
    _ensure_market_seas(market_id)
    stats = _seas_window_stats(market_id, bar_date)
    if stats is not None:
        return stats["score"]

    if hasattr(bar_date, 'date'):
        d = bar_date.date()
    elif isinstance(bar_date, _date):
        d = bar_date
    else:
        try:
            d = pd.to_datetime(bar_date).date()
        except Exception:
            return 5.0

    # Window-based fallback
    month = d.month
    windows = SEASONAL_WINDOWS.get(market_id, {})
    in_bull = any(
        (s <= month <= e) if s <= e else (month >= s or month <= e)
        for (s, e) in windows.get("bull", [])
    )
    in_bear = any(
        (s <= month <= e) if s <= e else (month >= s or month <= e)
        for (s, e) in windows.get("bear", [])
    )
    base = 7.5 if in_bull else (2.5 if in_bear else 5.0)
    return round(max(0.0, min(10.0, base)), 1)


# Asset-class momentum normalizers (ema_st_norm%, ema_norm%, roc_norm%)
# Based on typical 4-week trending speed and annual vol by asset class.
# Equity indices: moderate trending (ES/NQ 3/6/8%), FX: slow (1/2/3%),
# Energy NG: fast (8/15/20%), Oil: (4/8/12%), Bonds: slow (0.8/1.5/2%),
# Metals: (2/4/6%), Softs/Grains: (3/6/10%), Livestock: (2/4/7%).
_MOM_NORMALIZERS: dict = {
    "ES":  (3.0, 6.0, 8.0),  "NQ":  (3.0, 6.0, 8.0),
    "YM":  (3.0, 6.0, 8.0),  "RTY": (3.5, 7.0, 10.0),
    "Z":   (2.0, 4.0, 6.0),  "R":   (0.8, 1.5, 2.5),
    "ZB":  (0.8, 1.5, 2.5),  "ZN":  (0.8, 1.5, 2.5),
    "ZF":  (0.6, 1.2, 2.0),  "ZT":  (0.5, 1.0, 1.5),
    "GC":  (2.0, 4.0, 6.0),  "SI":  (3.0, 6.0, 9.0),
    "HG":  (2.5, 5.0, 8.0),  "PA":  (3.0, 6.0, 9.0),
    "PL":  (2.5, 5.0, 7.0),
    "CL":  (4.0, 8.0, 12.0), "B":   (4.0, 8.0, 12.0),
    "HO":  (4.0, 8.0, 12.0), "RB":  (4.0, 8.0, 12.0),
    "NG":  (8.0, 15.0, 20.0),"GO":  (3.5, 7.0, 11.0),
    "GAS": (8.0, 15.0, 20.0),
    "ZC":  (3.0, 6.0, 10.0), "ZS":  (3.0, 6.0, 10.0),
    "ZW":  (3.5, 7.0, 11.0),
    "SB":  (3.0, 6.0, 10.0), "CC":  (3.0, 6.0, 10.0),
    "KC":  (3.0, 6.0, 10.0), "RC":  (3.0, 6.0, 10.0),
    "CT":  (3.0, 6.0, 9.0),
    "LE":  (2.0, 4.0, 7.0),  "HE":  (3.0, 6.0, 9.0),
    "GF":  (2.0, 4.0, 7.0),
    "6E":  (1.0, 2.0, 3.0),  "6B":  (1.0, 2.0, 3.0),
    "6J":  (1.0, 2.0, 3.5),  "6A":  (1.2, 2.5, 4.0),
    "6C":  (0.8, 1.8, 3.0),  "6N":  (1.2, 2.5, 4.0),
    "6S":  (0.8, 1.8, 3.0),  "6M":  (1.5, 3.0, 5.0),
    "DX":  (0.8, 1.8, 3.0),
    "BTC": (15.0, 30.0, 40.0),"ETH": (20.0, 35.0, 50.0),
}
_MOM_NORM_DEFAULT = (3.0, 6.0, 8.0)  # equity-like default


def _score_momentum_at(px_closes: np.ndarray, px_dates_norm, bar_date_norm,
                       market_id: str = "") -> float:
    """
    Compute momentum score using only price data up to and including bar_date.
    Zero lookahead — only uses closes where date <= bar_date.
    market_id: optional, used to select asset-class normalizers.
    """
    mask = px_dates_norm <= bar_date_norm
    closes = px_closes[mask]
    if len(closes) < 20:
        return 5.0

    curr = closes[-1]
    n252 = min(252, len(closes))
    hi52 = closes[-n252:].max()
    lo52 = closes[-n252:].min()
    pct_range  = (curr - lo52) / (hi52 - lo52) if hi52 != lo52 else 0.5
    range_s10  = round(pct_range * 10.0, 1)

    def _ema(arr, n):
        return pd.Series(arr.astype(float)).ewm(span=n, adjust=False).mean().values

    def _sma(arr, n):
        return pd.Series(arr.astype(float)).rolling(n, min_periods=n).mean().values

    ema8   = _ema(closes, 8)[-1]
    ema20  = _ema(closes, 20)[-1]
    ema21  = _ema(closes, 21)[-1]
    ema50  = _ema(closes, 50)[-1]
    sma200_arr = _sma(closes, 200)
    sma200 = sma200_arr[-1] if len(closes) >= 200 and not np.isnan(sma200_arr[-1]) else np.nan

    # Short-term: EMA8 vs EMA21 — catches recent turns fast
    ema_st_slope = (ema8 - ema21) / ema21 * 100 if ema21 else 0
    # Medium-term: EMA20 vs EMA50
    ema_slope    = (ema20 - ema50) / ema50 * 100 if ema50 else 0
    # 4-week ROC (20 days)
    roc4w = (closes[-1] / closes[-20] - 1) * 100 if len(closes) >= 20 else             (closes[-1] / closes[-10] - 1) * 100 if len(closes) >= 10 else 0

    _st_norm, _mt_norm, _roc_norm = _MOM_NORMALIZERS.get(
        market_id.upper() if market_id else "", _MOM_NORM_DEFAULT)
    ema_st_s10 = round(max(0.0, min(10.0, (ema_st_slope / _st_norm) * 5.0 + 5.0)), 1)
    ema_s10    = round(max(0.0, min(10.0, (ema_slope / _mt_norm) * 5.0 + 5.0)), 1)
    roc_s10    = round(max(0.0, min(10.0, (roc4w / _roc_norm) * 5.0 + 5.0)), 1)
    sma200_s10 = round(max(0.0, min(10.0, ((curr - sma200) / sma200 * 100 / 10.0) * 5.0 + 5.0)), 1)                  if not np.isnan(sma200) and sma200 > 0 else 5.0

    # r15: MTF backtest parity — compute ST/MT/LT sub-scores + majority vote,
    # then blend the winning direction to match live scoring philosophy.
    # roc13w and roc26w for MT/LT sub-scores
    roc13w = (closes[-1] / closes[-65] - 1) * 100 if len(closes) >= 65 else 0
    roc26w = (closes[-1] / closes[-130] - 1) * 100 if len(closes) >= 130 else 0
    st_score = round(max(0.0, min(10.0, (ema_st_slope / 2.5 + roc4w / 6.0) * 2.5 + 5.0)), 1)
    mt_score = round(max(0.0, min(10.0, (ema_slope / 4.0 + roc13w / 10.0) * 2.5 + 5.0)), 1)
    if not np.isnan(sma200) and sma200 > 0:
        sma_pct = (curr - sma200) / sma200 * 100
        lt_score = round(max(0.0, min(10.0, (roc26w / 18.0 + sma_pct / 10.0) * 2.5 + 5.0)), 1)
    else:
        lt_score = 5.0
    def _sign(s):
        d = s - 5.0
        if d > 0.5: return 1
        if d < -0.5: return -1
        return 0
    st_sig, mt_sig, lt_sig = _sign(st_score), _sign(mt_score), _sign(lt_score)
    vote_sum = st_sig + mt_sig + lt_sig
    # Majority rule: if 2+ agree, use average of agreeing timeframes.
    # If whipsaw (vote_sum=0 with mixed signs), abstain → 5.0.
    signs = [st_sig, mt_sig, lt_sig]
    scores = [st_score, mt_score, lt_score]
    if vote_sum > 0:
        agreeing = [s for s, sg in zip(scores, signs) if sg > 0]
        mtf_score = sum(agreeing) / len(agreeing) if agreeing else 5.0
    elif vote_sum < 0:
        agreeing = [s for s, sg in zip(scores, signs) if sg < 0]
        mtf_score = sum(agreeing) / len(agreeing) if agreeing else 5.0
    else:
        # Neutral or whipsaw: check if all-zero (neutral) or mixed (whipsaw abstain)
        if any(sg != 0 for sg in signs):
            mtf_score = 5.0  # whipsaw → abstain
        else:
            mtf_score = sum(scores) / 3.0  # all neutral → use average

    return round(max(0.0, min(10.0, mtf_score)), 1)


def _score_relval_at(market_id: str, bar_date_norm,
                     self_series: pd.Series,
                     peer_series_map: dict,
                     periods: list) -> float:
    """
    Compute trend-gated relative-val score at a historical bar date.
    Bernd philosophy: valuation only matters when trend agrees.
      - Cheap + uptrend   → bullish (7.5–8.5)
      - Cheap + downtrend → neutral (5.0) — do not short undervalued
      - Expensive + downtrend → bearish (1.5–3.0)
      - Expensive + uptrend   → neutral (5.0) — do not long overvalued
      - Mid-range → neutral (5.0)
    self_series: pd.Series of self prices (daily), indexed by datetime64[D]
    peer_series_map: {peer_yf: pd.Series} pre-fetched
    All series must already be daily closes.
    Zero lookahead — only uses data up to bar_date_norm.
    """
    if self_series is None or self_series.empty:
        return 5.0

    # Slice self up to bar date
    self_s = self_series[self_series.index <= bar_date_norm]
    if len(self_s) < max(periods, default=13) + 5:
        return 5.0

    all_stochs: list[float] = []

    for peer_yf, peer_s_full in peer_series_map.items():
        peer_s = peer_s_full[peer_s_full.index <= bar_date_norm]
        combined = pd.concat([self_s.rename("s"), peer_s.rename("p")], axis=1).dropna()
        if len(combined) < max(periods, default=13) + 5:
            continue
        ratio = combined["s"] / combined["p"]
        for w in periods:
            if len(ratio) < w:
                continue
            roll_min = ratio.rolling(w).min()
            roll_max = ratio.rolling(w).max()
            denom    = roll_max - roll_min
            stoch    = np.where(denom > 0, (ratio - roll_min) / denom * 100, 50.0)
            last_val = float(stoch[-1]) if not np.isnan(stoch[-1]) else None
            if last_val is not None:
                all_stochs.append(last_val)

    if not all_stochs:
        return 5.0

    avg_stoch = sum(all_stochs) / len(all_stochs)

    # ── Trend gate using SMA200 (or EMA50 fallback) ───────────────────────
    closes_arr = self_s.values.astype(float)
    curr_price = float(closes_arr[-1]) if len(closes_arr) > 0 else None

    if curr_price is not None and len(closes_arr) >= 200:
        sma200_vals = pd.Series(closes_arr).rolling(200, min_periods=200).mean().values
        sma200 = float(sma200_vals[-1]) if not np.isnan(sma200_vals[-1]) else None
    elif curr_price is not None and len(closes_arr) >= 50:
        sma200_vals = pd.Series(closes_arr).ewm(span=50, adjust=False).mean().values
        sma200 = float(sma200_vals[-1])
    else:
        sma200 = None

    if curr_price is not None and sma200 is not None and sma200 > 0:
        pct_vs_200 = (curr_price - sma200) / sma200 * 100
        if pct_vs_200 >= 1.5:
            trend_gate = "bull"
        elif pct_vs_200 <= -1.5:
            trend_gate = "bear"
        else:
            trend_gate = "neutral"
    else:
        trend_gate = "neutral"

    # Market category (for equities exception)
    mkt_obj = next((m for m in MARKETS if m["id"] == market_id), None)
    is_equity = (mkt_obj.get("category", "") == "equity") if mkt_obj else False

    # ── ML-calibrated scoring matrix (per-asset thresholds) ─────────────
    rv_cfg_bt = REL_VAL_CONFIG.get(market_id, {})
    _CT  = rv_cfg_bt.get("cheap_thr", 20)
    _ET  = rv_cfg_bt.get("exp_thr",   80)
    _clo = _CT + (_ET - _CT) * 0.25
    _chi = _CT + (_ET - _CT) * 0.75
    _cde = _CT / 2
    _ede = _ET + (100 - _ET) / 2

    if avg_stoch <= _CT:
        if trend_gate == "bull":
            score = 8.5 if avg_stoch <= _cde else 8.0
        elif trend_gate == "bear":
            score = 7.5 if avg_stoch <= _cde else 7.0  # pullback long
        else:
            score = 6.5 if is_equity else 6.0
    elif avg_stoch <= _clo:
        if trend_gate == "bull":
            score = 7.0
        elif trend_gate == "bear":
            score = 6.5
        else:
            score = 6.0 if is_equity else 5.5
    elif avg_stoch <= _chi:
        score = 5.0
    elif avg_stoch <= _ET:
        if trend_gate == "bear":
            score = 3.5
        elif trend_gate == "bull":
            score = 3.0  # pullback short
        else:
            score = 4.0
    else:
        if trend_gate == "bull":
            score = 2.0 if avg_stoch >= _ede else 2.5
        elif trend_gate == "bear":
            score = 1.5 if avg_stoch >= _ede else 2.0
        else:
            score = 3.0

    return round(score, 1)


def _score_macro_at(market_id: str, bar_ts: float,
                    all_ff_events: list,
                    us_macro_indicator_map: list,
                    parse_ff_value_fn) -> float:
    """
    Compute macro score at a historical timestamp using only FF events
    released on or before bar_ts. Zero lookahead.
    """
    events_up_to = [e for e in all_ff_events if e["ts"] <= bar_ts]
    if not events_up_to:
        return 5.0

    mkt = next((m for m in MARKETS if m["id"] == market_id), None)
    if not mkt:
        return 5.0

    cat = mkt.get("category", "")
    market_id_u = market_id.upper()

    # Build ff_macro snapshot for all currencies at this point in time
    currencies_needed: set = set()
    if cat == "fx" or cat == "fx_cross":
        # Determine which currencies are needed
        fx_currency_map = {"6E":"EUR","6B":"GBP","6A":"AUD","6J":"JPY",
                           "6C":"CAD","6N":"NZD","6S":"CHF","6M":"MXN","DX":"USD"}
        if market_id_u in fx_currency_map:
            currencies_needed.add(fx_currency_map[market_id_u])
        if mkt.get("cross"):
            base_id  = mkt.get("base_leg","")
            quote_id = mkt.get("quote_leg","")
            currencies_needed.add(fx_currency_map.get(base_id, ""))
            currencies_needed.add(fx_currency_map.get(quote_id, ""))
        currencies_needed.add("USD")
    else:
        currencies_needed.add("USD")

    ff_macro_snap: dict = {}
    for curr in currencies_needed:
        if curr and curr != "USD":
            ff_macro_snap[curr] = compute_ff_economy_score(events_up_to, curr)

    # USD: use FF USD events with US_MACRO_INDICATOR_MAP (same logic as compute_macro_all)
    # US_MACRO_INDICATOR_MAP format: {name_substr: (category, higher_is_good)}
    usd_events_up_to = [e for e in events_up_to if e["currency"] == "USD"]
    best: dict = {}
    for evt in sorted(usd_events_up_to, key=lambda x: x["ts"]):
        name_l = evt["name"].lower()
        # Handle both dict format {substr: (category, hig)} and legacy list format
        if isinstance(us_macro_indicator_map, dict):
            _items = [(substr.lower(), substr.upper().replace(' ','_'), hig, cat_key, substr)
                      for substr, (cat_key, hig) in us_macro_indicator_map.items()]
        else:
            _items = us_macro_indicator_map
        for (substr, key, higher_is_good, category, disp_label) in _items:
            if substr in name_l:
                actual_raw   = parse_ff_value_fn(evt["actual"])
                forecast_raw = parse_ff_value_fn(evt["forecast"])
                if actual_raw is not None and forecast_raw is not None:
                    best[key] = {
                        "higher_is_good": higher_is_good,
                        "category":       category,
                        "actual_raw":     actual_raw,
                        "forecast_raw":   forecast_raw,
                    }
                break

    # Build category scores from US components
    # ── FIX 1: Scale by category (not by key name) — key names from substr.upper().replace()
    # do not match the old lowercase US_SCALE dict, causing every K-denominated indicator
    # (NFP, Claims, JOLTS) to use scale=1.0 and saturate to ±2 on any non-zero surprise.
    # Correct scales are calibrated so a typical 1-sigma surprise produces norm ≈ 1.0.
    # _parse_ff_value multiplies K→×1000, M→×1e6, so scales are in raw parsed units.
    US_SCALE_BY_CAT = {
        "JOBS":     40000,   # NFP/ADP: raw persons; ~40K = moderate beat
        "CLAIMS":   15000,   # Claims:  raw persons; ~15K = meaningful week-to-week surprise
        "JOLTS":    200000,  # JOLTS:   raw persons (6.87M → 6870000); ~200K typical
        "UNEMP":    0.1,     # Unemployment rate: % float; 0.1pp typical miss
        "WAGES":    0.1,     # Hourly earnings m/m %: 0.1pp typical
        "GDP":      0.5,     # GDP QoQ %: 0.5pp typical
        "MFG_PMI":  1.0,     # ISM/PMI: index pts; 1pt typical
        "SVC_PMI":  1.0,
        "CPI":      0.1,     # CPI m/m %: 0.1pp typical
        "CORE_CPI": 0.1,
        "PCE":      0.1,
        "PPI":      0.2,
        "RETAIL":   0.3,     # Retail sales MoM %: 0.3pp typical
    }

    # Shared score computation helper (used for both cat scores and components)
    def _norm_score(surprise: float, cat_key: str, higher_is_good: bool) -> int:
        scale = US_SCALE_BY_CAT.get(cat_key, 1.0)
        norm  = surprise / scale if scale else surprise
        if norm > 1.5:    raw = 2
        elif norm > 0.4:  raw = 1
        elif norm < -1.5: raw = -2
        elif norm < -0.4: raw = -1
        else:             raw = 0
        return raw if higher_is_good else -raw

    us_cat_scores: dict = {}
    for key, info in best.items():
        cat_key = info["category"]
        surprise = info["actual_raw"] - info["forecast_raw"]
        sc = _norm_score(surprise, cat_key, info["higher_is_good"])
        if cat_key not in us_cat_scores:
            us_cat_scores[cat_key] = []
        us_cat_scores[cat_key].append(sc)

    # ── FIX 2: Normalise category keys to lowercase so get_macro_score_for_market
    # can read them correctly (it uses .get("jobs"), .get("growth") etc. — all lowercase).
    # Previously usd_cats had uppercase keys ("JOBS", "GDP") causing all non-FX asset
    # macro scores to read as 0 → score always 5.0 (dead neutral) in score_history.
    _CAT_KEY_MAP = {
        "JOBS": "jobs",   "CLAIMS": "jobs",  "JOLTS": "jobs",   "UNEMP": "jobs",
        "WAGES": "jobs",  "GDP": "growth",   "MFG_PMI": "growth", "SVC_PMI": "growth",
        "RETAIL": "growth", "CPI": "inflation", "CORE_CPI": "inflation",
        "PCE": "inflation", "PPI": "inflation", "DGS2": "rates", "YLDCRV": "rates",
    }
    us_cat_scores_normalised: dict = {}  # lowercase merged
    for cat_key, scores in us_cat_scores.items():
        lc_key = _CAT_KEY_MAP.get(cat_key, cat_key.lower())
        if lc_key not in us_cat_scores_normalised:
            us_cat_scores_normalised[lc_key] = []
        us_cat_scores_normalised[lc_key].extend(scores)
    usd_cats: dict = {c: sum(v)/len(v) for c, v in us_cat_scores_normalised.items() if v}

    usd_score = max(-2.0, min(2.0, sum(usd_cats.values()) / len(usd_cats))) if usd_cats else 0.0
    ff_macro_snap["USD"] = {"score": usd_score, "cat_avg": usd_cats, "cat_details": {}, "label": "USD"}

    # Build a full components dict so get_macro_score_for_market can extract
    # per-indicator scores for the non-FX asset formulas (ES uses JOBS/GDP/CPI directly).
    # Components keys must match what get_macro_score_for_market expects.
    _INDICATOR_TO_COMP = {
        # Maps substr-derived uppercase key → standard component key
        "NON-FARM_EMPLOYMENT_CHANGE": "JOBS",
        "ADP_NON-FARM_EMPLOYMENT":    "ADP",
        "UNEMPLOYMENT_CLAIMS":        "CLAIMS",
        "UNEMPLOYMENT_RATE":          "UNEMP",
        "JOLTS_JOB_OPENINGS":         "JOLTS",
        "AVERAGE_HOURLY_EARNINGS":    "WAGES",
        "GDP":                        "GDP",
        "ISM_MANUFACTURING_PMI":      "MFG_PMI",
        "ISM_SERVICES_PMI":           "SVC_PMI",
        "MANUFACTURING_PMI":          "MFG_PMI",
        "SERVICES_PMI":               "SVC_PMI",
        "CORE_RETAIL_SALES":          "RETAIL",
        "RETAIL_SALES":               "RETAIL",
        "INDUSTRIAL_PRODUCTION":      "MFG_PMI",
        "CPI":                        "CPI",
        "CORE_CPI":                   "CPI",
        "PPI":                        "PPI",
        "CORE_PCE_PRICE_INDEX":       "PCE",
        "PCE_PRICE_INDEX":            "PCE",
    }
    components = {}
    for key, info in best.items():
        comp_key = _INDICATOR_TO_COMP.get(key, key)
        surprise = info["actual_raw"] - info["forecast_raw"]
        cat_key  = info["category"]
        sc = _norm_score(surprise, cat_key, info["higher_is_good"])
        components[comp_key] = {"score": sc, "actual": info["actual_raw"],
                                "forecast": info["forecast_raw"]}

    macro_snap = {"category_scores": usd_cats, "components": components}
    result = get_macro_score_for_market(market_id_u, macro_snap, ff_macro=ff_macro_snap)
    # get_macro_score_for_market already returns 0-10 — return directly
    return round(max(0.0, min(10.0, result.get("score", 5.0))), 1)


def _score_pcr_at(market_id: str, bar_date_norm, regime_px: dict) -> float:
    """
    Walk-forward PCR proxy using VIX rolling percentile.
    VIX and equity PCR have r≈0.85 correlation over multi-year periods.
    At each bar, compute VIX percentile vs prior 52-week window.
    High VIX pctile = high fear = bearish (low PCR score).
    For non-equity markets, returns 5.0 (neutral) — PCR is equity-specific.
    """
    # Only meaningful for equity-correlated markets
    _equity_mkts = {"ES", "NQ", "YM", "RTY", "Z", "GC", "SI", "CL", "NG",
                    "KC", "SB", "ZB", "ZN", "6E", "6B", "6J", "6A", "DX"}
    m = market_id.upper()
    if m not in _equity_mkts:
        return 5.0

    vix_series = regime_px.get("VIX")
    if vix_series is None or len(vix_series) < 20:
        return 5.0

    s = vix_series[vix_series.index <= bar_date_norm]
    if len(s) < 20:
        return 5.0

    # Use up to 52 weeks of lookback for percentile
    lookback = min(len(s), 52)
    window = s.values[-lookback:].astype(float)
    current_vix = float(window[-1])

    # Rolling percentile: what fraction of prior readings was BELOW current level
    prior = window[:-1]
    pctile = float(np.sum(prior < current_vix) / len(prior) * 100) if len(prior) > 0 else 50.0

    # Map percentile to score:
    # High VIX pctile = elevated fear = bearish signal (low score)
    # Note: PCR and VIX are inversely related to price, so high = bearish
    if pctile >= 80:
        score = 2.5    # Extreme fear — like a very high PCR
    elif pctile >= 65:
        score = 3.5    # Elevated fear
    elif pctile >= 50:
        score = 4.5    # Mildly elevated
    elif pctile >= 35:
        score = 5.5    # Mildly complacent
    elif pctile >= 20:
        score = 6.5    # Complacent — low put buying
    else:
        score = 7.5    # Extreme complacency

    # Invert logic for inverse markets: if market is a safe haven (GC, ZB, ZN),
    # high VIX = bullish (fear drives buying)
    _safe_havens = {"GC", "SI", "ZB", "ZN"}
    if m in _safe_havens:
        score = 10.0 - score  # Invert: high VIX = bullish for safe havens

    return round(score, 1)


def _score_regime_at(market_id: str, bar_date_norm,
                     regime_px: dict,
                     walcl_full: list = None) -> float:
    """
    Reconstruct regime score at a historical bar date using pre-fetched
    weekly price series for all regime assets. Zero lookahead.
    regime_px: {name: pd.Series of weekly closes, indexed by datetime64[D]}
    """
    # Build returns using only data up to bar_date_norm, looking back ~13 weeks
    bar_dt = pd.Timestamp(str(bar_date_norm))
    cutoff = bar_date_norm
    lookback_start = np.datetime64(str((bar_dt - pd.DateOffset(weeks=14)).date()), 'D')

    returns: dict = {}
    levels:  dict = {}

    for name, series in regime_px.items():
        s = series[series.index <= cutoff]
        if len(s) < 4:
            continue
        close = s.values.astype(float)
        ret_1w = (close[-1] / close[-2] - 1) * 100 if close[-2] != 0 else 0
        ret_1m = (close[-1] / close[max(-4, -len(close))] - 1) * 100  # max() not min(): we want 4 bars back, not all the way to bar[0]
        ret_3m = (close[-1] / close[max(-13, -len(close))] - 1) * 100 if len(close) >= 5 else 0.0
        returns[name] = {"1w": ret_1w, "1m": ret_1m, "3m": ret_3m}
        levels[name]  = float(close[-1])

    if not returns:
        return 5.0

    # ── Shared holistic core (same maths as live compute_risk_regime) ───────
    # Limitations in history: no FRED OAS series → core falls back to the
    # HYG−LQD price spread; no news archive → geo_tension=None → geo pillar 0.
    _core = _regime_core_score(
        returns,
        vix=levels.get("VIX"), vix3m=levels.get("VIX3M"),
        hy_oas_bps=None, hy_delta_4w=None,
        hyg_1m=(returns["HYG"]["1m"] if "HYG" in returns else None),
        lqd_1m=(returns["LQD"]["1m"] if "LQD" in returns else None),
        geo_tension=None,
    )
    regime_score = _core["score"]

    # ── Rate path proxy from IRX (13-week T-bill) ────────────────────────────
    # IRX tracks Fed Funds very closely (correlation >0.97).
    # 6m change in IRX gives us a historical rate_norm without ZQ data.
    # Falling IRX = Fed cutting = rate_norm > 0 (bullish bonds/gold)
    # Rising IRX  = Fed hiking = rate_norm < 0 (bearish bonds/gold)
    hist_rate_norm = 0.0
    if "IRX" in regime_px:
        irx_series = regime_px["IRX"]
        irx_up_to = irx_series[irx_series.index <= cutoff]
        if len(irx_up_to) >= 26:
            irx_now  = float(irx_up_to.values[-1])
            irx_26w  = float(irx_up_to.values[-26])
            irx_6m_chg = irx_now - irx_26w  # positive = rates rising = hiking
            # Map to rate_norm: invert (cutting = positive rate_norm)
            if   irx_6m_chg <= -1.5:  hist_rate_norm =  2.0  # aggressive cut cycle
            elif irx_6m_chg <= -0.75: hist_rate_norm =  1.5
            elif irx_6m_chg <= -0.25: hist_rate_norm =  1.0  # cutting
            elif irx_6m_chg <= -0.08: hist_rate_norm =  0.5  # mild cutting
            elif irx_6m_chg <   0.08: hist_rate_norm =  0.0  # flat
            elif irx_6m_chg <   0.25: hist_rate_norm = -0.5  # mild hiking
            elif irx_6m_chg <   0.75: hist_rate_norm = -1.0  # hiking
            else:                      hist_rate_norm = -2.0  # aggressive hike cycle

    # ── DXY 1m return signal (for copper, grains, gold DXY component) ─────────
    # DXY is already in RISK_ASSETS → available in returns dict
    _hist_dxy_1m = returns.get("DXY", {}).get("1m", 0.0) or 0.0

    # ── TIPS real yield proxy from TIP ETF ────────────────────────────────────
    # TIP price falling  → real yields rising  → bearish gold (_ry_adj negative)
    # TIP price rising   → real yields falling → bullish gold  (_ry_adj positive)
    # TIP modified duration ≋7.5y; TIP 26w % chg / 7.5 ≋ real yield chg in %
    # Normalise to [−2, +2] using same ÷ 1.5 scale as the live DFII10 formula.
    _hist_ry_adj = 0.0
    if "TIP" in regime_px:
        _tip_series = regime_px["TIP"]
        _tip_up_to  = _tip_series[_tip_series.index <= cutoff]
        if len(_tip_up_to) >= 26:
            _tip_now  = float(_tip_up_to.values[-1])
            _tip_26w  = float(_tip_up_to.values[-26])
            if _tip_26w > 0:
                _tip_26w_pct = (_tip_now / _tip_26w - 1) * 100
                # TIP +1% / 7.5 duration ≈ -0.13% real yield change
                # ry_adj = TIP_26w_pct / 7.5 / 1.5 (same normalisation as live signal)
                _hist_ry_adj = max(-2.0, min(2.0, _tip_26w_pct / 7.5 / 1.5))

    # ── WALCL: Fed balance sheet from FRED full history ──────────────────────
    _hist_walcl_sig = 0.0
    if walcl_full:
        # Filter to data up to bar_date
        bar_dt_str = str(bar_date_norm)  # YYYY-MM-DD
        _wrows = [r for r in walcl_full if r.get("date") and r["date"] <= bar_dt_str and r.get("value") is not None]
        if len(_wrows) >= 13:  # need ~3 months of weekly data
            # Use last 13 weeks (quarterly change) — same logic as live WALCL signal
            _w_recent = [r["value"] for r in _wrows[-13:]]
            _w_now  = float(_w_recent[-1])
            _w_3m   = float(_w_recent[0])
            if _w_3m > 0:
                _bs_chg3m = (_w_now / _w_3m - 1.0) * 100.0
                _hist_walcl_sig = max(-1.0, min(1.0, _bs_chg3m / 3.0))

    # Pass raw regime_score (-4..+4) to get_regime_score_for_market.
    # Also pass the DXY/TIPS/WALCL signals through the returns / macro_dashboard
    # sub-keys that get_regime_score_for_market reads from the regime dict.
    regime_dict = {
        "regime": ("Strong Risk-On" if regime_score >= 3.0 else "Risk-On" if regime_score >= 1.8 else "Lean Risk-On" if regime_score >= 0.7 else "Strong Risk-Off" if regime_score <= -3.0 else "Risk-Off" if regime_score <= -1.8 else "Lean Risk-Off" if regime_score <= -0.7 else "Neutral"),
        "score":  regime_score,   # raw -4..+4 scale
        "rate_score": hist_rate_norm,  # IRX-derived historical rate path
        "label":  "",
        "signals": {},
        "levels": levels,
        # Provide DXY 1m return so get_regime_score_for_market can compute _dxy_sig
        "returns": {
            "DXY": {"return_1m": _hist_dxy_1m},
        },
        # Provide TIPS proxy so _ry_adj is non-zero in history
        # Using pre-computed _hist_ry_adj avoids re-deriving inside the function
        # We inject it via a synthetic macro_dashboard that mimics the live format.
        # The live code reads: _ry_val = macro_dashboard["real_yield"]["value"]
        # then computes: ry_adj = -(ry_val - 1.0) / 1.5
        # We set a synthetic ry_val that will reproduce _hist_ry_adj:
        #   _hist_ry_adj = -(syn_ry - 1.0) / 1.5  ⇒  syn_ry = 1.0 - _hist_ry_adj * 1.5
        "macro_dashboard": {
            "real_yield": {
                "value": round(1.0 - _hist_ry_adj * 1.5, 3),  # back-solved synthetic TIPS level
                "label": "hist_proxy",
            },
            "fed_balance": {
                "chg_3m_pct": _hist_walcl_sig * 3.0,  # back-solved from _hist_walcl_sig = chg3m/3.0
            },
        },
    }
    result = get_regime_score_for_market(market_id, regime_dict)
    return round(max(0.0, min(10.0, result.get("score", 5.0))), 1)


_SH_RESULT_CACHE: dict = {}
_SH_RESULT_TTL = 3600 * 2  # 2 hours
# Global concurrency guard: only ONE walk-forward computes at a time. Each walk-forward
# loads full multi-decade FRED/price history and builds large per-week feature arrays;
# two running concurrently was OOM-killing the process. Extra requests get 202 + retry
# (the frontend already polls), so this only adds a short queue, never an error.
_SH_GLOBAL_MAX = 1
_SH_GLOBAL_ACTIVE = 0
_SH_PREFETCH_CACHE: dict = {}
_SH_PREFETCH_TTL = 3600 * 3   # FIX: 12h→3h — prevents pandas yfinance frames accumulating overnight

@app.get("/api/score_history")
async def get_score_history(market: str):
    """Walk-forward composite score history (result-cached 1h)."""
    m_upper = market.upper()
    mkt = next((x for x in MARKETS if x["id"] == m_upper), None)
    if not mkt:
        return {"error": f"Unknown market: {market}", "dates": [], "scores": [], "prices": []}
    # ── Result cache check ────────────────────────────────────────────────────
    _rn = time.time()
    _rc = _SH_RESULT_CACHE.get(m_upper)
    # Invalidate result cache if prefetch cache is pre-FRED (macro was always 5.0)
    _pfc_check = _SH_PREFETCH_CACHE.get(m_upper)
    if _rc and _pfc_check and "fred_us_full" not in _pfc_check:
        _rc = None  # old pre-FRED prefetch; force full recompute
        del _SH_RESULT_CACHE[m_upper]
    if _rc and (_rn - _rc["ts"]) < _SH_RESULT_TTL:
        return _SafeJSONResponse(_rc["data"])
    # Per-market guard — prevents duplicate concurrent prefetches (OOM risk)
    if _SH_MARKET_LOCKS.get(m_upper):
        return JSONResponse({"status": "computing", "message": "Score history computing, retry in 30s"}, status_code=202)
    # Global guard — bound peak memory to a single walk-forward (no await between the
    # check and the increment below, so this is atomic on the event loop).
    global _SH_GLOBAL_ACTIVE
    if _SH_GLOBAL_ACTIVE >= _SH_GLOBAL_MAX:
        return JSONResponse({"status": "computing", "message": "Score history busy, retry in 20s"}, status_code=202)
    _SH_MARKET_LOCKS[m_upper] = True
    _SH_GLOBAL_ACTIVE += 1
    try:
    
        # ── Pre-fetch all historical data (cached per market, 2h TTL) ──────────────
        _now_ts = time.time()
        _cached = _SH_PREFETCH_CACHE.get(m_upper)
        if _cached and (_now_ts - _cached["ts"]) < _SH_PREFETCH_TTL:
            fred_us_full        = _cached["fred_us_full"]
            fred_fx_full        = _cached["fred_fx_full"]
            walcl_full          = _cached["walcl_full"]
            regime_px           = _cached["regime_px"]
            relval_self_series  = _cached["relval_self"]
            relval_peer_map     = _cached["relval_peer_map"]
            relval_periods      = _cached["relval_periods"]
            pcr_s_const         = _cached["pcr_s_const"]
            print(f"score_history[{m_upper}]: using prefetch cache (US:{len(fred_us_full)} FX:{len(fred_fx_full)} WALCL:{len(walcl_full)})")
        else:
            # Run the entire prefetch block in a thread executor — it makes ~60 FF HTTP
            # calls + multiple yfinance calls, all synchronous. Blocking the event loop
            # here would prevent /api/health from responding for minutes.
            def _do_prefetch():
                _pf_ts = time.time()
                # 1. FRED macro: fetch full history for all US series.
                #    ForexFactory is Cloudflare-blocked server-side — replaced with FRED.
                _fred_us_full: dict = {}
                for _key, (_fid, _tr, _hig, _sc, _cat) in _SH_FRED_US_SERIES.items():
                    try:
                        _s = fetch_fred_series_full(_fid)
                        if _s:
                            _fred_us_full[_key] = _s
                    except Exception as _fe:
                        print(f'score_history FRED US prefetch [{_key}]: {_fe}')
                print(f'score_history[{m_upper}]: FRED US series fetched: {list(_fred_us_full.keys())}')

                # WALCL (Fed balance sheet) full history for regime
                _walcl_full: list = []
                try:
                    _ws = fetch_fred_series_full('WALCL')
                    if _ws:
                        _walcl_full = _ws
                except Exception:
                    pass
                print(f'score_history[{m_upper}]: WALCL rows fetched: {len(_walcl_full)}')

                # FX FRED series: fetch for all currencies
                _fred_fx_full: dict = {}  # {ccy: {fred_id: series_list}}
                for _ccy, _ccy_cfg in _SH_FRED_FX_SERIES.items():
                    _ccy_map: dict = {}
                    for (_fid, _tr, _hig, _sc, _cat) in _ccy_cfg:
                        try:
                            _s = fetch_fred_series_full(_fid)
                            if _s:
                                _ccy_map[_fid] = _s
                        except Exception as _fe:
                            print(f'score_history FRED FX prefetch [{_ccy}/{_fid}]: {_fe}')
                    if _ccy_map:
                        _fred_fx_full[_ccy] = _ccy_map
                print(f'score_history[{m_upper}]: FRED FX currencies fetched: {list(_fred_fx_full.keys())}')

                # 2. Regime: fetch max weekly closes for all regime tickers
                _regime_px: dict = {}
                for _rn, _rticker in RISK_ASSETS.items():
                    try:
                        _df = yf.Ticker(_rticker).history(period='max', interval='1wk', auto_adjust=True)
                        if not _df.empty:
                            _s = _df['Close'].copy()
                            _s.index = pd.to_datetime(_s.index).tz_localize(None).normalize()
                            _s.index = _s.index.map(lambda d: np.datetime64(d.date().isoformat(), 'D'))
                            _regime_px[_rn] = _s
                    except Exception:
                        pass

                # 3. Rel-val: fetch max weekly closes for self + all configured peers
                _relval_self: pd.Series = None
                _relval_peer_map: dict = {}
                _relval_periods: list = []
                _rv_cfg = REL_VAL_CONFIG.get(m_upper)
                if _rv_cfg:
                    _relval_periods = _rv_cfg.get('periods', [13, 26])
                    try:
                        _df_s = yf.Ticker(mkt['yf']).history(period='max', interval='1wk', auto_adjust=True)
                        if not _df_s.empty:
                            _ss = _df_s['Close'].copy()
                            _ss.index = pd.to_datetime(_ss.index).tz_localize(None).normalize()
                            _ss.index = _ss.index.map(lambda d: np.datetime64(d.date().isoformat(), 'D'))
                            _relval_self = _ss
                    except Exception:
                        pass
                    for _peer in _rv_cfg.get('peers', []):
                        try:
                            _df_p = yf.Ticker(_peer['yf']).history(period='max', interval='1wk', auto_adjust=True)
                            if not _df_p.empty:
                                _sp = _df_p['Close'].copy()
                                _sp.index = pd.to_datetime(_sp.index).tz_localize(None).normalize()
                                _sp.index = _sp.index.map(lambda d: np.datetime64(d.date().isoformat(), 'D'))
                                _relval_peer_map[_peer['yf']] = _sp
                        except Exception:
                            pass

                # 4. PCR — held constant (live value; walk-forward requires CBOE CSV history)
                _live_pcr = score_pcr(m_upper)
                _pcr_s = _live_pcr.get('score', 5.0) if _live_pcr else 5.0

                # Store in prefetch cache
                _SH_PREFETCH_CACHE[m_upper] = {
                    'fred_us_full':    _fred_us_full,
                    'fred_fx_full':    _fred_fx_full,
                    'walcl_full':      _walcl_full,
                    'regime_px':       _regime_px,
                    'relval_self':     _relval_self,
                    'relval_peer_map': _relval_peer_map,
                    'relval_periods':  _relval_periods,
                    'pcr_s_const':     _pcr_s,
                    'ts':              _pf_ts,
                }
                return _SH_PREFETCH_CACHE[m_upper]
    
            # Run the heavy IO in a thread so the event loop stays responsive
            _pf = await asyncio.get_event_loop().run_in_executor(_SH_EXECUTOR, _do_prefetch)
            fred_us_full       = _pf["fred_us_full"]
            fred_fx_full       = _pf["fred_fx_full"]
            walcl_full         = _pf["walcl_full"]
            regime_px          = _pf["regime_px"]
            relval_self_series = _pf["relval_self"]
            relval_peer_map    = _pf["relval_peer_map"]
            relval_periods     = _pf["relval_periods"]
            pcr_s_const        = _pf["pcr_s_const"]
    
        # ── Determine weights via shared router ──────────────────────────────────
        cat = mkt.get("category", "")
        w_map = _get_weight_map(m_upper)
    
        # ── CROSS PAIR: walk-forward Briese differential ─────────────────────────
        if mkt.get("cross"):
            base_id   = mkt["base_leg"]
            quote_id  = mkt["quote_leg"]
            base_mkt  = next((x for x in MARKETS if x["id"] == base_id),  None)
            quote_mkt = next((x for x in MARKETS if x["id"] == quote_id), None)
            if not base_mkt or not quote_mkt:
                return {"error": "Leg markets not found", "dates": [], "scores": [], "prices": []}
    
            df_base  = await fetch_cot_history(base_mkt["cftc_code"],  base_mkt["name"])
            df_quote = await fetch_cot_history(quote_mkt["cftc_code"], quote_mkt["name"])
            if df_base is None or len(df_base) < 30 or df_quote is None or len(df_quote) < 30:
                return {"error": "Insufficient leg COT data", "dates": [], "scores": [], "prices": []}
    
            n_common = min(len(df_base), len(df_quote))
            df_base  = df_base.tail(n_common).reset_index(drop=True)
            df_quote = df_quote.tail(n_common).reset_index(drop=True)
    
            px_df_cross = fetch_price_data_long(mkt["yf"])
            price_lookup: dict = {}
            if px_df_cross is not None and not px_df_cross.empty:
                for dt, cl in zip(pd.to_datetime(px_df_cross.index).tz_localize(None).normalize(),
                                   px_df_cross["Close"].values):
                    price_lookup[dt] = float(cl)
    
            # Price arrays for momentum
            px_closes_arr  = np.array(list(price_lookup.values()), dtype=float)
            px_dates_arr   = np.array(list(price_lookup.keys()))
            if len(px_dates_arr) > 0:
                sort_idx       = np.argsort(px_dates_arr)
                px_dates_arr   = px_dates_arr[sort_idx]
                px_closes_arr  = px_closes_arr[sort_idx]
                px_dates_norm  = np.array([np.datetime64(pd.Timestamp(d).date().isoformat(), 'D')
                                            for d in px_dates_arr])
            else:
                px_dates_norm = np.array([], dtype="datetime64[D]")
    
            MIN_BARS = 10       # Min bars before starting — 10% Briese fill
            MAX_RETURN = 520   # 10yr history (gated by COT availability)
            COT_FULL_WEIGHT_BARS = 94  # 60% of 156w Briese window
            dates: list = []; scores: list = []; prices: list = []
    
            for i in range(MIN_BARS, n_common):
                sl_base  = df_base.iloc[:i + 1].copy()
                sl_quote = df_quote.iloc[:i + 1].copy()
                window   = min(156, len(sl_base), len(sl_quote))
    
                def _briese(df):
                    arr    = df["comm_net"].values.astype(float)
                    recent = arr[-window:]
                    lo, hi = recent.min(), recent.max()
                    if hi == lo: return 50.0
                    return round((arr[-1] - lo) / (hi - lo) * 100, 1)
    
                diff   = _briese(sl_base) - _briese(sl_quote)
                cot_s  = round(max(0.0, min(10.0, (diff / 100.0) * 5.0 + 5.0)), 1)
                # Continuous COT confidence: ramp from 0 at 10 bars to full at 94 bars
                _cot_conf = min(1.0, max(0.0, (i - MIN_BARS) / max(1, COT_FULL_WEIGHT_BARS - MIN_BARS)))
                cot_s = round(5.0 + (cot_s - 5.0) * _cot_conf, 1)

                if "date" in df_base.columns:
                    bar_date = pd.to_datetime(df_base["date"].iloc[i])
                else:
                    bar_date = pd.to_datetime(df_base.index[i])
                bar_date = bar_date.tz_localize(None) if bar_date.tzinfo else bar_date
                bar_date_norm = np.datetime64(str(bar_date.date()), 'D')
                bar_ts = bar_date.timestamp()
    
                seas_s   = _score_seasonality_at(m_upper, bar_date)
                mom_s    = _score_momentum_at(px_closes_arr, px_dates_norm, bar_date_norm, m_upper) \
                           if len(px_closes_arr) > 20 else 5.0
                macro_s  = _score_macro_at_fred(m_upper, str(bar_date.date()),
                                               fred_us_full, fred_fx_full)
                regime_s = _score_regime_at(m_upper, bar_date_norm, regime_px, walcl_full)
                # Cross pairs: rel-val uses trend-gated scoring (same as regular markets)
                # Trend gate prevents false cheapness signals (e.g. cheap but in downtrend = neutral)
                relval_s = 5.0  # Cross pairs don't have peer-ratio config — neutral by default
    
                composite = round(max(0.0, min(10.0,
                    cot_s    * w_map["cot"]      +
                    seas_s   * w_map["seasonal"] +
                    mom_s    * w_map["momentum"] +
                    macro_s  * w_map["macro"]    +
                    regime_s * w_map["regime"]   +
                    relval_s * w_map["relval"]
                )), 1)
    
                dates.append(str(bar_date.date()))
                scores.append(composite)
    
                price_date = bar_date.normalize()
                close = price_lookup.get(price_date)
                if close is None:
                    # Find nearest date within 5 days, sort keys by proximity to price_date
                    cands = {k: v for k, v in price_lookup.items() if abs((k - price_date).days) <= 5}
                    if cands:
                        nearest_key = min(cands.keys(), key=lambda k: abs((k - price_date).days))
                        close = cands[nearest_key]
                prices.append(round(float(close), 4) if close is not None else None)
    
            dates  = dates[-MAX_RETURN:]
            scores = scores[-MAX_RETURN:]
            prices = prices[-MAX_RETURN:]
    
            return {
                "market": m_upper, "name": mkt["name"],
                "dates": dates, "scores": scores, "prices": prices,
                "note": (
                    f"Full composite walk-forward: COT ({base_id}\u2212{quote_id}), seasonality, momentum, "
                    f"macro, regime, rel-val \u2014 all reconstructed at each bar with zero lookahead. "
                    f"PCR held at today\u2019s reading (no historical snapshots available)."
                ),
            }
    
        # ── REGULAR MARKET ───────────────────────────────────────────────────────
        if mkt.get("ice_code"):
            df_full = await fetch_ice_cot_history(mkt["ice_code"])
        else:
            df_full = await fetch_cot_history(mkt["cftc_code"], mkt["name"])
        if df_full is None or len(df_full) < 20:  # ICE markets have shorter history (57-329w)
            return {"error": "Insufficient COT data", "dates": [], "scores": [], "prices": []}
    
        # Build price arrays
        px_df_long = fetch_price_data_long(mkt["yf"])
        px_closes_all      = np.array([], dtype=float)
        px_dates_norm_all  = np.array([], dtype="datetime64[D]")
        price_lookup_daily: dict = {}
    
        if px_df_long is not None and not px_df_long.empty:
            _px_idx   = pd.to_datetime(px_df_long.index).tz_localize(None).normalize()
            _px_close = px_df_long["Close"].values.astype(float)
            _sort     = np.argsort(_px_idx)
            _px_idx   = np.array(_px_idx)[_sort]
            _px_close = _px_close[_sort]
            px_closes_all     = _px_close
            px_dates_norm_all = np.array([np.datetime64(pd.Timestamp(d).date().isoformat(), 'D')
                                           for d in _px_idx])
            for d, c in zip(_px_idx, _px_close):
                price_lookup_daily[pd.Timestamp(d).normalize()] = float(c)
    
        # Merge price into COT df
        df_merged = df_full.copy()
        if price_lookup_daily:
            try:
                px_idx_s   = pd.to_datetime(list(price_lookup_daily.keys())).normalize().astype("datetime64[us]")
                px_close_s = list(price_lookup_daily.values())
                price_lkp_df = pd.DataFrame({"_cot_date": px_idx_s, "close": px_close_s}).sort_values("_cot_date")
                if "date" in df_merged.columns:
                    cot_idx = pd.to_datetime(df_merged["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[us]")
                else:
                    cot_idx = pd.to_datetime(df_merged.index).tz_localize(None).normalize().astype("datetime64[us]")
                df_merged["_cot_date"] = cot_idx.values
                merged = pd.merge_asof(
                    df_merged.sort_values("_cot_date"), price_lkp_df,
                    on="_cot_date", direction="nearest", tolerance=pd.Timedelta(days=3),
                )
                df_merged = merged.drop(columns=["_cot_date"])
            except Exception as _e:
                print(f"score_history price merge error for {m_upper}: {_e}")
    
        MIN_BARS   = 10       # Min bars before starting — 10% Briese fill
        MAX_RETURN = 520   # 10yr history (gated by COT availability)
        COT_FULL_WEIGHT_BARS = 94  # 60% of 156w Briese window
        dates:  list = []
        scores: list = []
        prices: list = []
    
        n = len(df_merged)
        is_crypto_mkt = (cat == "crypto")
        is_fx_mkt     = (cat in ("fx", "fx_cross"))
    
        # Per-component lists for the frontend to show breakdown
        cot_scores:    list = []
        mom_scores:    list = []
        macro_scores:  list = []
        seas_scores:   list = []
        regime_scores: list = []
        relval_scores: list = []
    
        for i in range(MIN_BARS, n):
            slice_df = df_merged.iloc[:i + 1].copy()
            if is_crypto_mkt:
                cot_result = compute_crypto_cot_score(slice_df, market_id=m_upper)
            else:
                cot_result = compute_cot_score_v2(slice_df, market_id=m_upper)
            cot_s = cot_result["score"]
            # Continuous COT confidence: ramp from 0 at 10 bars to full at 94 bars
            _cot_conf = min(1.0, max(0.0, (i - MIN_BARS) / max(1, COT_FULL_WEIGHT_BARS - MIN_BARS)))
            cot_s = round(5.0 + (cot_s - 5.0) * _cot_conf, 1)

            if "date" in slice_df.columns:
                bar_date = pd.to_datetime(slice_df["date"].iloc[-1])
            else:
                bar_date = pd.to_datetime(slice_df.index[-1])
            bar_date = bar_date.tz_localize(None) if bar_date.tzinfo else bar_date
            bar_date_norm = np.datetime64(str(bar_date.date()), 'D')
            bar_ts = bar_date.timestamp()
    
            seas_s   = _score_seasonality_at(m_upper, bar_date)
            mom_s    = _score_momentum_at(px_closes_all, px_dates_norm_all, bar_date_norm, m_upper) \
                       if len(px_closes_all) > 20 else 5.0
            macro_s  = _score_macro_at_fred(m_upper, str(bar_date.date()),
                                       fred_us_full, fred_fx_full)
            regime_s = _score_regime_at(m_upper, bar_date_norm, regime_px, walcl_full)
    
            # Rel-val now uses trend-gated logic (Bernd philosophy):
            # cheap + uptrend = bullish; cheap + downtrend = neutral (avoids 2022 JPY trap);
            # expensive + downtrend = bearish; expensive + uptrend = neutral.
            # This makes it safe to use for FX as well — trend gate prevents false signals.
            relval_s = _score_relval_at(m_upper, bar_date_norm,
                                         relval_self_series, relval_peer_map, relval_periods) \
                       if relval_self_series is not None else 5.0
    
            pcr_s_wf = _score_pcr_at(m_upper, bar_date_norm, regime_px)
            composite = round(max(0.0, min(10.0,
                cot_s    * w_map["cot"]      +
                seas_s   * w_map["seasonal"] +
                mom_s    * w_map["momentum"] +
                macro_s  * w_map["macro"]    +
                regime_s * w_map["regime"]   +
                relval_s * w_map["relval"]   +
                pcr_s_wf * w_map.get("pcr", 0.0)
            )), 1)
    
            dates.append(str(bar_date.date()))
            scores.append(composite)
            cot_scores.append(cot_s)
            mom_scores.append(round(mom_s, 1))
            macro_scores.append(round(macro_s, 1))
            seas_scores.append(round(seas_s, 1))
            regime_scores.append(round(regime_s, 1))
            relval_scores.append(round(relval_s, 1))
    
            close = slice_df["close"].iloc[-1] if "close" in slice_df.columns else None
            prices.append(round(float(close), 4) if close is not None and not np.isnan(float(close)) else None)
    
        dates         = dates[-MAX_RETURN:]
        scores        = scores[-MAX_RETURN:]
        prices        = prices[-MAX_RETURN:]
        cot_scores    = cot_scores[-MAX_RETURN:]
        mom_scores    = mom_scores[-MAX_RETURN:]
        macro_scores  = macro_scores[-MAX_RETURN:]
        seas_scores   = seas_scores[-MAX_RETURN:]
        regime_scores = regime_scores[-MAX_RETURN:]
        relval_scores = relval_scores[-MAX_RETURN:]
    
        relval_note = (" Rel-val uses trend-gated scoring — valuation only signals when trend confirms."
                       if is_fx_mkt else " Rel-val uses trend-gated scoring — valuation only signals when trend confirms.")
    
        _sh_result = {
            "market": m_upper, "name": mkt["name"],
            "dates": dates, "scores": scores, "prices": prices,
            "cot_scores":    cot_scores,
            "mom_scores":    mom_scores,
            "macro_scores":  macro_scores,
            "seas_scores":   seas_scores,
            "regime_scores": regime_scores,
            "relval_scores": relval_scores,
            "note": (
                "Full composite walk-forward: COT, seasonality, momentum, macro (FF calendar, 5yr), "
                "regime (VIX/yields/credit/DXY, max history), rel-val (max history) \u2014 "
                "all reconstructed at each bar with zero lookahead. "
                "PCR held at today\u2019s reading (no historical snapshots available)."
                + relval_note
            ),
        }
        _SH_RESULT_CACHE[m_upper] = {"ts": time.time(), "data": _sh_result}
        return _SafeJSONResponse(_sh_result)
    except Exception as _sh_exc:
        print(f"[score_history] ERROR for {m_upper}: {type(_sh_exc).__name__}: {_sh_exc}")
        return JSONResponse({"error": "Score history computation failed", "detail": str(_sh_exc)}, status_code=500)
    finally:
        _SH_MARKET_LOCKS.pop(m_upper, None)  # always release — even on exception
        _SH_GLOBAL_ACTIVE = max(0, _SH_GLOBAL_ACTIVE - 1)
        gc.collect()

# ── GLOBAL REGIME HISTORY endpoint ─────────────────────────────────────────
# Returns the raw global regime score (-4..+4) for each weekly bar over the
# last ~52 weeks (12 months), with label, signal breakdown, and colour.
_RH_CACHE: dict = {"data": None, "ts": 0}
_RH_TTL = 3600  # 1h

def _regime_label_from_score(s: float) -> str:
    if s >= 3.0:   return "Strong Risk-On"
    elif s >= 1.8: return "Risk-On"
    elif s >= 0.7: return "Lean Risk-On"
    elif s <= -3.0: return "Strong Risk-Off"
    elif s <= -1.8: return "Risk-Off"
    elif s <= -0.7: return "Lean Risk-Off"
    else:           return "Neutral"

@app.get("/api/regime_history")
async def get_regime_history():
    """Walk-forward global risk regime score for last 52 weekly bars (12 months)."""
    now = time.time()
    if _RH_CACHE["data"] is not None and now - _RH_CACHE["ts"] < _RH_TTL:
        return _SafeJSONResponse(_RH_CACHE["data"])

    def _compute():
        # Fetch all RISK_ASSETS weekly prices (reuse existing prefetch cache if warm)
        # Try to get regime_px from any cached score_history prefetch (ES is most likely warm)
        regime_px: dict = {}
        for _cached in _SH_PREFETCH_CACHE.values():
            if isinstance(_cached, dict) and "regime_px" in _cached and _cached["regime_px"]:
                regime_px = _cached["regime_px"]
                break

        if not regime_px:
            # Fetch fresh — only RISK_ASSETS needed
            for _rn, _rticker in RISK_ASSETS.items():
                try:
                    _df = yf.Ticker(_rticker).history(period="3y", interval="1wk", auto_adjust=True)
                    if not _df.empty:
                        _s = _df["Close"].copy()
                        _s.index = pd.to_datetime(_s.index).tz_localize(None).normalize()
                        _s.index = _s.index.map(lambda d: np.datetime64(d.date().isoformat(), "D"))
                        regime_px[_rn] = _s
                except Exception:
                    pass

        if not regime_px:
            return {"error": "no_data", "dates": [], "scores": [], "labels": []}

        # Build weekly bar dates for the last 52 weeks using SPX as reference
        _spx = regime_px.get("SPX")
        ref_series = _spx if (_spx is not None and len(_spx) > 0) else next(iter(regime_px.values()))
        # All dates available, take last 54 to cover 52 with padding
        all_dates = sorted(ref_series.index.tolist())
        bar_dates = all_dates[-54:]  # ~54 weekly bars

        dates_out, scores_out, labels_out = [], [], []
        signal_rows = []

        for bar_dt in bar_dates:
            # Compute global regime_score at this bar (replicate _score_regime_at internals)
            cutoff = bar_dt
            returns: dict = {}
            levels:  dict = {}

            for name, series in regime_px.items():
                s = series[series.index <= cutoff]
                if len(s) < 4:
                    continue
                close = s.values.astype(float)
                ret_1w = (close[-1] / close[-2] - 1) * 100 if close[-2] != 0 else 0
                ret_1m = (close[-1] / close[max(-4, -len(close))] - 1) * 100
                ret_3m = (close[-1] / close[max(-13, -len(close))] - 1) * 100 if len(close) >= 5 else 0.0
                returns[name] = {"1w": ret_1w, "1m": ret_1m, "3m": ret_3m}
                levels[name]  = float(close[-1])

            if not returns:
                continue

            # Shared holistic core (same maths as live compute_risk_regime).
            # No OAS/news history here — HYG-LQD credit fallback, geo pillar 0.
            vix_level = levels.get("VIX", 20)
            _core = _regime_core_score(
                returns,
                vix=levels.get("VIX"), vix3m=levels.get("VIX3M"),
                hy_oas_bps=None, hy_delta_4w=None,
                hyg_1m=(returns["HYG"]["1m"] if "HYG" in returns else None),
                lqd_1m=(returns["LQD"]["1m"] if "LQD" in returns else None),
                geo_tension=None,
            )
            rsc = round(max(-4.0, min(4.0, _core["score"])), 2)
            label = _regime_label_from_score(rsc)

            # Signal snapshot for tooltip (expanded for richer display)
            _tnx_level = levels.get("TNX", 0)
            _irx_level = levels.get("IRX", 0)
            _term_spread = round(_tnx_level - (_irx_level / 100), 2) if _tnx_level and _irx_level else None
            sig = {
                "spx_1m":     round(returns.get("SPX",    {}).get("1m", 0), 1),
                "rut_1m":     round(returns.get("RUT",    {}).get("1m", 0), 1),
                "vix":        round(vix_level, 1),
                "credit":     round((returns.get("HYG",{}).get("1m",0) - returns.get("LQD",{}).get("1m",0)), 1),
                "usdjpy_1m":  round(returns.get("USDJPY", {}).get("1m", 0), 1),
                "tnx":        round(_tnx_level, 2) if _tnx_level else None,
                "term_spread": _term_spread,
            }
            dates_out.append(str(bar_dt))
            scores_out.append(rsc)
            labels_out.append(label)
            signal_rows.append(sig)

        # Trim to last 52 bars
        dates_out  = dates_out[-52:]
        scores_out = scores_out[-52:]
        labels_out = labels_out[-52:]
        signal_rows = signal_rows[-52:]

        # Convert raw ±4 scores → 0-10 scale (5.0 = neutral) for consistency
        # with all other scores in the system and the arc renderer
        scores_10 = [round((s + 4.0) / 8.0 * 10.0, 1) for s in scores_out]

        return {
            "dates":   dates_out,
            "scores":  scores_10,
            "labels":  labels_out,
            "signals": signal_rows,
            "current_score": scores_10[-1] if scores_10 else None,
            "current_label": labels_out[-1] if labels_out else None,
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_APP_EXECUTOR, _compute)
    _RH_CACHE["data"] = result
    _RH_CACHE["ts"]   = time.time()
    return _SafeJSONResponse(result)


# ── CANDLE / OHLC endpoint ──────────────────────────────────────────────────
_CANDLE_CACHE: dict = {}
_CANDLE_TTL = 3600  # 1h

@app.get("/api/candles")
async def get_candles(market: str, period: str = "6mo", ema_period: int = 50):
    """Return daily OHLC candle data + 20 EMA, 50 EMA, 200 SMA overlays."""
    cache_key = f"{market}:{period}:multi"
    now = time.time()
    if cache_key in _CANDLE_CACHE and now - _CANDLE_CACHE[cache_key]["ts"] < _CANDLE_TTL:
        return _SafeJSONResponse(_CANDLE_CACHE[cache_key]["data"])

    # Find YF ticker
    mkt_obj = next((m for m in MARKETS if m["id"] == market), None)
    if not mkt_obj:
        # Try cross pairs
        CROSS_YF = {
            "EURJPY": "EURJPY=X", "EURGBP": "EURGBP=X", "EURAUD": "EURAUD=X",
            "EURCAD": "EURCAD=X", "EURNZD": "EURNZD=X", "EURCHF": "EURCHF=X",
            "GBPJPY": "GBPJPY=X", "GBPAUD": "GBPAUD=X", "GBPCAD": "GBPCAD=X",
            "GBPNZD": "GBPNZD=X", "GBPCHF": "GBPCHF=X", "AUDJPY": "AUDJPY=X",
            "AUDNZD": "AUDNZD=X", "AUDCAD": "AUDCAD=X", "NZDJPY": "NZDJPY=X",
            "NZDCAD": "NZDCAD=X", "CADJPY": "CADJPY=X", "CHFJPY": "CHFJPY=X",
            "AUDCHF": "AUDCHF=X",
        }
        yf_ticker = CROSS_YF.get(market)
    else:
        yf_ticker = mkt_obj["yf"]

    if not yf_ticker:
        return {"error": "Unknown market", "candles": [], "ema": []}

    try:
        import yfinance as yf
        tk = yf.Ticker(yf_ticker)
        # Always fetch 2y so the 200 SMA has enough history to render
        hist = tk.history(period="2y", interval="1d", auto_adjust=True)
        if hist.empty:
            return {"error": "No data", "candles": [], "ema": []}

        hist = hist.dropna(subset=["Open", "High", "Low", "Close"])

        # Build full candle array (used for MA calculation)
        all_candles = []
        for idx, row in hist.iterrows():
            all_candles.append({
                "t": int(idx.timestamp() * 1000),
                "o": round(float(row["Open"]), 6),
                "h": round(float(row["High"]), 6),
                "l": round(float(row["Low"]), 6),
                "c": round(float(row["Close"]), 6),
            })

        # Display window: last 6 months (~126 trading days)
        display_n = 126
        candles = all_candles[-display_n:] if len(all_candles) > display_n else all_candles
        display_offset = len(all_candles) - len(candles)

        # Multi-MA calculation over full history, then slice to display window
        all_closes = pd.Series([c["c"] for c in all_candles])

        def _calc_ema(series, span):
            vals = series.ewm(span=span, adjust=False).mean().values
            return [round(float(v), 6) if not np.isnan(v) else None for v in vals]

        def _calc_sma(series, window):
            vals = series.rolling(window, min_periods=window).mean().values
            return [round(float(v), 6) if not np.isnan(v) else None for v in vals]

        ema20_all  = _calc_ema(all_closes, 20)
        ema50_all  = _calc_ema(all_closes, 50)
        sma200_all = _calc_sma(all_closes, 200)

        # Slice to display window
        ema20_vals  = ema20_all[display_offset:]
        ema50_vals  = ema50_all[display_offset:]
        sma200_vals = sma200_all[display_offset:]
        result = {
            "market": market,
            "period": period,
            "candles": candles,
            "ema": ema50_vals,       # legacy key — 50 EMA
            "ema20": ema20_vals,
            "ema50": ema50_vals,
            "sma200": sma200_vals,
        }
        _CANDLE_CACHE[cache_key] = {"data": result, "ts": now}
        return _SafeJSONResponse(result)
    except Exception as e:
        return {"error": str(e), "candles": [], "ema": []}


# ── Yield Curve History ───────────────────────────────────────────────────────
_YC_CACHE: dict = {"data": None, "date": None}

FRED_TENORS = [
    ("1M",  "DGS1MO"),
    ("3M",  "DGS3MO"),
    ("6M",  "DGS6MO"),
    ("1Y",  "DGS1"),
    ("2Y",  "DGS2"),
    ("3Y",  "DGS3"),
    ("5Y",  "DGS5"),
    ("7Y",  "DGS7"),
    ("10Y", "DGS10"),
    ("20Y", "DGS20"),
    ("30Y", "DGS30"),
]

# yfinance ticker map for yield curve tenors (best available)
# Tenors without a direct yf ticker are interpolated from neighbours
_YF_TENOR_MAP = {
    "3M":  "^IRX",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}


def _build_yf_curve_by_date() -> dict:
    """
    Build {date_str: {label: value}} for all 11 FRED_TENORS using yfinance.
    Direct tickers: 3M=^IRX, 5Y=^FVX, 10Y=^TNX, 30Y=^TYX.
    Missing tenors interpolated linearly between known anchor points.
    Returns dict keyed by date string YYYY-MM-DD.
    """
    # Fetch all 4 anchor series synchronously (already cached after prewarm)
    anchors = {}
    for label, ticker in _YF_TENOR_MAP.items():
        series = _fetch_yf_yield_series(ticker, 400)
        if series:
            anchors[label] = {x["date"]: x["value"] for x in series}

    if not anchors:
        return {}

    # Collect all dates present in anchors
    all_dates = set()
    for d_map in anchors.values():
        all_dates.update(d_map.keys())

    # Anchor tenor positions on the curve (in years)
    ANCHOR_YEARS = {"3M": 0.25, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}
    # All tenor positions
    ALL_YEARS = {
        "1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0,
        "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0,
        "10Y": 10.0, "20Y": 20.0, "30Y": 30.0
    }

    by_date = {}
    for dt in sorted(all_dates):
        # Get known anchor values for this date
        known = {}
        for lbl, ticker in _YF_TENOR_MAP.items():
            if dt in anchors.get(lbl, {}):
                known[ANCHOR_YEARS[lbl]] = anchors[lbl][dt]
        if len(known) < 2:
            continue

        # Interpolate all 11 tenors
        sorted_anchors = sorted(known.items())  # [(years, value), ...]
        tenors = {}
        for label, yrs in ALL_YEARS.items():
            # Find surrounding anchors
            lo = hi = None
            for ay, av in sorted_anchors:
                if ay <= yrs:
                    lo = (ay, av)
                else:
                    hi = (ay, av)
                    break
            if lo and hi:
                t = (yrs - lo[0]) / (hi[0] - lo[0])
                tenors[label] = round(lo[1] + t * (hi[1] - lo[1]), 3)
            elif lo:
                tenors[label] = round(lo[1], 3)
            elif hi:
                tenors[label] = round(hi[1], 3)
            else:
                tenors[label] = None
        by_date[dt] = tenors
    return by_date


async def _fetch_yield_curve_history_async() -> dict:
    """Fetch FRED daily yield data for all 11 tenors.
    Primary: FRED via httpx (fast when available).
    Fallback: yfinance ^TNX/^IRX/^FVX/^TYX with interpolation.
    Returns snapshots for 'now', '3m', '6m', '12m' keys."""
    today = date.today()
    cached_date = _YC_CACHE.get("date")
    if cached_date == today.isoformat() and _YC_CACHE["data"]:
        return _YC_CACHE["data"]

    HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    async def _fetch_series(client: httpx.AsyncClient, label: str, series_id: str) -> tuple[str, list]:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            r = await client.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            pairs = []
            for line in r.text.strip().split("\n")[1:]:
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                try:
                    d = date.fromisoformat(parts[0].strip())
                    v = float(parts[1].strip())
                    pairs.append((d, v))
                except (ValueError, IndexError):
                    continue
            return label, pairs
        except Exception:
            return label, []

    # ── Try FRED first (10s timeout per series) ───────────────────────────────
    fred_by_date: dict[date, dict] = {}
    fred_ok = False
    try:
        async with httpx.AsyncClient() as client:
            tasks = [_fetch_series(client, label, sid) for label, sid in FRED_TENORS]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=15)
        for label, pairs in results:
            for d, v in pairs:
                fred_by_date.setdefault(d, {})[label] = v
        # Consider FRED OK if we got data for key tenors on recent dates
        recent_dates = sorted(fred_by_date.keys())[-5:] if fred_by_date else []
        fred_ok = any(
            sum(1 for v in fred_by_date[d].values() if v is not None) >= 6
            for d in recent_dates
        )
    except Exception as _fe:
        print(f"[yc_history] FRED fetch failed: {_fe} — falling back to yfinance")

    all_labels = [label for label, _ in FRED_TENORS]

    if fred_ok:
        # Use FRED data
        rows = sorted([
            {"date": d, "tenors": {lbl: fred_by_date[d].get(lbl) for lbl in all_labels}}
            for d in fred_by_date
        ], key=lambda r: r["date"])
    else:
        # ── yfinance fallback ─────────────────────────────────────────────────
        print("[yc_history] Using yfinance fallback for yield curve history")
        loop = asyncio.get_event_loop()
        yf_by_date_str = await loop.run_in_executor(None, _build_yf_curve_by_date)
        rows = sorted([
            {"date": date.fromisoformat(ds), "tenors": tenors}
            for ds, tenors in yf_by_date_str.items()
        ], key=lambda r: r["date"])

    def _snap(target_date: date):
        for row in reversed(rows):
            if row["date"] <= target_date:
                non_null = sum(1 for v in row["tenors"].values() if v is not None)
                if non_null >= 4:
                    return row
        return None

    def _fmt(snap):
        if snap is None:
            return None
        return {"date": snap["date"].isoformat(), "tenors": snap["tenors"]}

    result = {
        "now":  _fmt(_snap(today)),
        "3m":   _fmt(_snap(today - timedelta(days=91))),
        "6m":   _fmt(_snap(today - timedelta(days=182))),
        "12m":  _fmt(_snap(today - timedelta(days=365))),
        "tenor_labels": all_labels,
    }
    _YC_CACHE["data"] = result
    _YC_CACHE["date"] = today.isoformat()
    return result


@app.get("/api/yield-curve-history")
async def yield_curve_history():
    """Full 11-tenor yield curve snapshots: now, -3m, -6m, -12m."""
    result = await _fetch_yield_curve_history_async()
    return _SafeJSONResponse(content=result)


# ── Upcoming Events cache ─────────────────────────────────────────────────────
_UPCOMING_EVENTS_CACHE: dict = {"data": None, "time": 0}
_UPCOMING_EVENTS_TTL: int = 3600  # 1 hour

def _get_future_week_strings(n_weeks: int = 3) -> list:
    """Generate next n_weeks FF week URL strings (starting from current week)."""
    import calendar as _cal
    from datetime import date, timedelta
    today = date.today()
    day_of_week = today.weekday()  # Mon=0, Sun=6
    days_since_sunday = (day_of_week + 1) % 7
    current_sunday = today - timedelta(days=days_since_sunday)
    months_short = ["jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec"]
    week_strings = []
    for i in range(n_weeks):
        sunday = current_sunday + timedelta(weeks=i)
        mon = months_short[sunday.month - 1]
        week_strings.append(f"{mon}{sunday.day}.{sunday.year}")
    return week_strings

FLAG_MAP = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵",
    "AUD": "🇦🇺", "CAD": "🇨🇦", "NZD": "🇳🇿", "CHF": "🇨🇭",
    "CNY": "🇨🇳", "CNH": "🇨🇳",
}

def _parse_ff_datetime(dateline) -> str | None:
    """Parse FF dateline — handles Unix timestamp (int/float) or string formats."""
    import re as _re
    from datetime import datetime, timezone
    if dateline is None:
        return None
    # Handle Unix timestamp (integer or float)
    if isinstance(dateline, (int, float)):
        try:
            dt = datetime.fromtimestamp(dateline, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return None
    dateline = str(dateline).strip()
    if not dateline:
        return None
    try:
        # Unix timestamp as string
        if dateline.isdigit() or (dateline.replace('.','',1).isdigit()):
            dt = datetime.fromtimestamp(float(dateline), tz=timezone.utc)
            return dt.isoformat()
        # Strip timezone name in parens e.g. "(UTC)"
        dateline = _re.sub(r'\s*\(.*?\)', '', dateline).strip()
        # "Wed May 21 2026 08:30:00 GMT+0000"
        m = _re.match(r'\w+ (\w+ \d+ \d+ \d+:\d+:\d+)', dateline)
        if m:
            dt = datetime.strptime(m.group(1), "%b %d %Y %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).isoformat()
        # ISO format
        dt = datetime.fromisoformat(dateline.replace("Z","+00:00"))
        return dt.isoformat()
    except Exception:
        return dateline

@app.get("/api/upcoming-events")
async def upcoming_events(force: bool = False):
    """
    Return high-impact economic events for the next 10 days.

    Multi-source strategy (Render can't reach forexfactory.com HTML due to Cloudflare):
      1. Fair Economy JSON feed (this-week only, works on Render)
      2. FF HTML scrape (works locally / sandbox, blocked on Render — attempted anyway)
      3. FF on-disk event store (populated by the daily inject cron)

    Cached for 1 hour.
    """
    global _UPCOMING_EVENTS_CACHE
    now = time.time()
    if not force and _UPCOMING_EVENTS_CACHE["data"] and (now - _UPCOMING_EVENTS_CACHE["time"]) < _UPCOMING_EVENTS_TTL:
        return _UPCOMING_EVENTS_CACHE["data"]

    from datetime import datetime, timezone, timedelta
    import concurrent.futures as _cf

    all_events: list = []

    # ── Source 1: Fair Economy JSON (Render-safe, this-week only) ────────
    try:
        json_events = fetch_ff_calendar_json(force=force)
        all_events.extend(json_events)
    except Exception as _e:
        print(f"[upcoming-events] Fair Economy JSON fetch failed: {_e}", flush=True)

    # ── Source 2: FF HTML scrape (works from sandbox/local; blocked on Render) ──
    week_strings = _get_future_week_strings(3)  # current + next 2 weeks (covers 10-day window)
    _ex_upev = _cf.ThreadPoolExecutor(max_workers=3)
    try:
        futs = {_ex_upev.submit(_fetch_ff_week_html, ws): ws for ws in week_strings}
        done, pending = _cf.wait(futs, timeout=15)
        for fut in pending:
            fut.cancel()
        for fut in done:
            try:
                all_events.extend(fut.result())
            except Exception:
                pass
    finally:
        _ex_upev.shutdown(wait=False)

    # ── Source 3: FF on-disk event store (populated by daily inject cron) ──
    try:
        store_events = list(_ff_store_load().values())
        for se in store_events:
            all_events.append({
                "name":        se.get("name", "") or se.get("title", ""),
                "actual":      se.get("actual", "") or "",
                "forecast":    se.get("forecast", "") or "",
                "previous":    se.get("previous", "") or "",
                "currency":    se.get("currency", "") or se.get("country", ""),
                "dateline":    se.get("dateline") or se.get("date_ts") or se.get("ts"),
                "impactClass": _ff_impact_norm(se.get("impactClass", "") or se.get("impact", "")),
            })
    except Exception as _e:
        print(f"[upcoming-events] FF store read failed: {_e}", flush=True)

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc + timedelta(days=10)

    filtered = []
    for ev in all_events:
        # High + medium impact
        ic = (ev.get("impactClass") or "").lower()
        title = (ev.get("title") or "").lower()
        is_high   = "high" in ic or "red" in ic
        is_medium = "medium" in ic or "orange" in ic or "mod" in ic
        if not is_high and not is_medium:
            continue
        impact_label = "High" if is_high else "Medium"
        # Parse datetime — dateline may be Unix int or string
        dl = ev.get("dateline")
        dt_str = _parse_ff_datetime(dl)
        # Try to filter by time window
        dt_utc = None
        try:
            from datetime import datetime, timezone
            if dt_str:
                dt_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            dt_utc = None

        if dt_utc:
            if dt_utc < now_utc or dt_utc > cutoff:
                continue

        currency = ev.get("currency", "")
        filtered.append({
            "name":          ev.get("name", ""),
            "currency":      currency,
            "flag":          FLAG_MAP.get(currency, ""),
            "datetime_utc":  dt_str or "",
            "datetime_ts":   dl if isinstance(dl, (int,float)) else None,
            "prior":         ev.get("previous", "") or "",
            "forecast":      ev.get("forecast", "") or "",
            "actual":        ev.get("actual", "") or "",
            "impact":        impact_label,
        })

    # Sort by datetime ascending
    def sort_key(e):
        try:
            from datetime import datetime
            return datetime.fromisoformat(e["datetime_utc"].replace("Z", "+00:00"))
        except Exception:
            return datetime.max.replace(tzinfo=None)

    filtered.sort(key=lambda e: e.get("datetime_utc") or "")

    # Dedupe — the FF calendar HTML embeds each event JSON blob twice, so an
    # identical (name, currency, datetime) tuple appears 2x without this.
    _seen_ev: set = set()
    _deduped: list = []
    for _e in filtered:
        _k = (_e["name"], _e["currency"], _e["datetime_utc"])
        if _k in _seen_ev:
            continue
        _seen_ev.add(_k)
        _deduped.append(_e)
    filtered = _deduped

    result = {"events": filtered, "count": len(filtered), "generated_utc": now_utc.isoformat()}
    _UPCOMING_EVENTS_CACHE["data"] = result
    _UPCOMING_EVENTS_CACHE["time"] = now
    return result

BUILD_ID = "2026-08-13-r15l"
_PROC_START = time.time()

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    try:
        _n_store = len(_ff_store_load())
    except Exception:
        _n_store = -1
    try:
        _g_age = int(time.time() - _FF_GROWTH_CACHE["time"]) if _FF_GROWTH_CACHE.get("data") else None
    except Exception:
        _g_age = None
    return {"status": "ok", "time": datetime.utcnow().isoformat(),
            "build": BUILD_ID, "uptime_s": int(time.time() - _PROC_START),
            "ff_store_n": _n_store, "ff_growth_cache_age_s": _g_age}



@app.api_route("/api/health/ready", methods=["GET", "HEAD"])
async def health_ready():
    """Readiness probe — returns 503 while the cache is warming up.
    Point Render's health-check path at /api/health/ready so it only
    routes traffic once ALL_DATA_CACHE is actually populated."""
    from fastapi.responses import JSONResponse as _JR
    if _WARMING["done"]:
        return {"status": "ready", "time": datetime.utcnow().isoformat()}
    if _WARMING["started"]:
        return _JR({"status": "warming", "message": "Cache warming — not ready"}, status_code=503)
    return _JR({"status": "cold", "message": "Startup not yet begun"}, status_code=503)

@app.get("/api/debug-yc")
async def debug_yc():
    """Debug: test FRED yield curve fetch and show raw response."""
    import traceback
    results = {}
    # Test raw HTTP fetch for T10Y2Y
    try:
        url = FRED_BASE + "T10Y2Y"
        r = requests.get(url, timeout=15)
        raw_text = r.text[:500] if r.text else "(empty)"
        lines = r.text.strip().split("\n") if r.text else []
        parsed = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2:
                parsed.append({"raw": line, "skip": parts[1].strip() == "."})
        results["T10Y2Y_raw"] = {
            "status": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "raw_first500": raw_text,
            "line_count": len(lines),
            "parsed_rows": len(parsed),
            "dot_rows": sum(1 for p in parsed if p["skip"]),
            "last5_lines": lines[-5:] if len(lines) >= 5 else lines
        }
    except Exception as e:
        results["T10Y2Y_raw"] = {"error": str(e), "trace": traceback.format_exc()[-400:]}
    # Test yfinance yield fallback
    for sym in ["^TNX", "^IRX", "^FVX", "^TYX"]:
        try:
            d = _fetch_yf_yield_series(sym, 10)
            results["yf_" + sym] = {"ok": True, "count": len(d) if d else 0, "last": d[-1] if d else None}
        except Exception as e:
            results["yf_" + sym] = {"ok": False, "error": str(e)}
    results["_cache_keys"] = list(FRED_CACHE.keys())
    results["_yf_yield_cache_keys"] = list(_YF_YIELD_CACHE.keys())
    return results


@app.get("/api/clear-regime-cache")
async def clear_regime_cache():
    """Bust the RISK_REGIME_CACHE so the next /api/scores call recomputes fresh."""
    RISK_REGIME_CACHE["data"] = None
    RISK_REGIME_CACHE["time"] = 0
    return {"cleared": True, "message": "Regime cache cleared — next scores call will recompute"}

@app.get("/api/clear-narrative-cache")
async def clear_narrative_cache():
    """Bust all data caches so the next /api/scores call re-fetches everything fresh.
    Use this after major macro releases (PCE, CPI, NFP) to pull in the new figures
    immediately rather than waiting up to 1 hour for the TTL to expire.
    Zeroes: NARR_CACHE, GLOBAL_NARR_CACHE, NEWS_CACHE, ALL_DATA_CACHE,
            US_MACRO_CACHE, FF_MACRO_CACHE, _FF_INFL_CACHE, _FF_LABOUR_CACHE.
    """
    NARR_CACHE["data"] = None
    NARR_CACHE["time"] = 0
    GLOBAL_NARR_CACHE["data"] = None
    GLOBAL_NARR_CACHE["time"] = 0
    NEWS_CACHE["data"] = None
    NEWS_CACHE["time"] = 0
    ALL_DATA_CACHE["data"] = None
    ALL_DATA_CACHE["time"] = 0
    # Macro data caches — ensures new PCE/CPI/NFP figures are picked up immediately
    US_MACRO_CACHE["data"] = None
    US_MACRO_CACHE["time"] = 0
    FF_MACRO_CACHE["data"] = None
    FF_MACRO_CACHE["time"] = 0
    _FF_INFL_CACHE["data"] = None
    _FF_INFL_CACHE["time"] = 0
    _FF_LABOUR_CACHE["data"] = None
    _FF_LABOUR_CACHE["time"] = 0
    _FF_GROWTH_CACHE["data"] = None
    _FF_GROWTH_CACHE["time"] = 0
    RISK_REGIME_CACHE["data"] = None
    RISK_REGIME_CACHE["time"] = 0
    US_MACRO_CACHE["data"] = None
    US_MACRO_CACHE["time"] = 0
    print("[cache] Full cache bust via /api/clear-narrative-cache", flush=True)
    return {
        "cleared": True,
        "caches_zeroed": [
            "NARR_CACHE", "GLOBAL_NARR_CACHE", "NEWS_CACHE", "ALL_DATA_CACHE",
            "US_MACRO_CACHE", "FF_MACRO_CACHE", "_FF_INFL_CACHE", "_FF_LABOUR_CACHE", "_FF_GROWTH_CACHE"
        ],
        "message": "Full cache bust complete — next /api/scores call re-fetches all macro data and regenerates narratives"
    }

@app.get("/api/debug-ice")
async def debug_ice(market: str = "B"):
    """Diagnose ICE COT fetch: tests connectivity and returns row counts per year."""
    import csv as _csv, io as _io, concurrent.futures as _cff
    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.ice.com/report/122",
    }
    current_year = date.today().year
    results = []
    for year in range(current_year - 2, current_year + 1):
        url = f"https://www.ice.com/publicdocs/futures/COTHist{year}.csv"
        try:
            resp = requests.get(url, timeout=20, headers=_headers)
            raw = resp.content.decode("utf-8-sig") if resp.status_code == 200 else ""
            is_html = "<!doctype" in raw[:200].lower()
            if resp.status_code == 200 and not is_html:
                reader = _csv.DictReader(_io.StringIO(raw))
                rows = list(reader)
                matched = [r for r in rows if r.get("CFTC_Commodity_Code","").strip() == market]
                fut = [r for r in matched if r.get("FutOnly_or_Combined","") == "FutOnly"]
                results.append({"year": year, "status": resp.status_code, "total_rows": len(rows),
                                 "market_rows": len(matched), "futonly_rows": len(fut)})
            else:
                results.append({"year": year, "status": resp.status_code, "is_html": is_html, "error": "non-200 or HTML"})
        except Exception as e:
            results.append({"year": year, "error": str(e)})
    mem_cached = market in _ICE_MEM_CACHE and _ICE_MEM_CACHE[market]["df"] is not None
    disk_df = _load_ice_from_disk(market)
    return {
        "market": market,
        "mem_cached": mem_cached,
        "mem_rows": len(_ICE_MEM_CACHE[market]["df"]) if mem_cached else 0,
        "disk_cached": disk_df is not None and not disk_df.empty,
        "disk_rows": len(disk_df) if (disk_df is not None and not disk_df.empty) else 0,
        "annual_cache_years": list(_ICE_ANNUAL_ROW_CACHE.keys()),
        "fetch_test": results,
    }


@app.get("/api/clear-cot-cache")
async def clear_cot_cache(market: str = ""):
    """Bust the per-market COT result caches (cot-history + setup-stats).
    ?market=ES clears one market; no param clears all. Used by the weekly
    COT refresh cron after new CFTC/ICE data lands."""
    if market:
        m = market.upper()
        hit_hist  = _COT_HIST_RESULT_CACHE.pop(m, None) is not None
        hit_stats = _SETUP_STATS_CACHE.pop(m, None) is not None
        return {"ok": True, "market": m, "cleared": {"cot_history": hit_hist, "setup_stats": hit_stats}}
    n_hist, n_stats = len(_COT_HIST_RESULT_CACHE), len(_SETUP_STATS_CACHE)
    _COT_HIST_RESULT_CACHE.clear()
    _SETUP_STATS_CACHE.clear()
    return {"ok": True, "cleared": {"cot_history": n_hist, "setup_stats": n_stats}}


@app.post("/api/inject-ice-cache")
async def inject_ice_cache(payload: dict):
    """Accepts pre-fetched ICE COT CSV rows and injects them into memory + disk cache.
    FIX: pandas work runs in thread executor so event loop stays responsive.
    """
    market = payload.get("market", "")
    rows   = payload.get("rows", [])
    fmt    = payload.get("fmt", "disagg")
    if not market or not rows:
        return {"ok": False, "error": "Missing market or rows"}
    _loop = asyncio.get_event_loop()
    return await _loop.run_in_executor(_APP_EXECUTOR, _inject_ice_sync, market, rows, fmt)

def _inject_ice_sync(market: str, rows: list, fmt: str) -> dict:
    """CPU-bound ICE inject — runs in thread executor, not the async event loop."""
    import time as _time
    try:
        df = pd.DataFrame(rows)

        # ── Date parsing ────────────────────────────────────────────────────
        df["date"] = pd.to_datetime(
            df.get("As_of_Date_Form_MM/DD/YYYY", pd.Series(dtype=str)), errors="coerce"
        )
        mask = df["date"].isna()
        if mask.any():
            df.loc[mask, "date"] = pd.to_datetime(
                df.loc[mask, "As_of_Date_In_Form_YYMMDD"], format="%y%m%d", errors="coerce"
            )
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        df["open_interest_all"] = pd.to_numeric(
            df.get("Open_Interest_All", pd.Series(dtype=float)), errors="coerce"
        )

        if fmt == "tff":
            # TFF: Leveraged Fund = comm_net, Asset Manager = lspec_net
            lf_long  = pd.to_numeric(df.get("Leveraged_Fund_Long_All",  pd.Series(dtype=float)), errors="coerce").fillna(0)
            lf_short = pd.to_numeric(df.get("Leveraged_Fund_Short_All", pd.Series(dtype=float)), errors="coerce").fillna(0)
            am_long  = pd.to_numeric(df.get("Asset_Manager_Long_All",   pd.Series(dtype=float)), errors="coerce").fillna(0)
            am_short = pd.to_numeric(df.get("Asset_Manager_Short_All",  pd.Series(dtype=float)), errors="coerce").fillna(0)
            nr_long  = pd.to_numeric(df.get("NonRept_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce").fillna(0)
            nr_short = pd.to_numeric(df.get("NonRept_Positions_Short_All", pd.Series(dtype=float)), errors="coerce").fillna(0)
            df["comm_positions_long_all"]    = lf_long
            df["comm_positions_short_all"]   = lf_short
            df["noncomm_positions_long_all"] = am_long
            df["noncomm_positions_short_all"]= am_short
            df["nonrept_positions_long_all"] = nr_long
            df["nonrept_positions_short_all"]= nr_short
            df["comm_net"]  = lf_long  - lf_short
            df["lspec_net"] = am_long  - am_short
            df["sspec_net"] = nr_long  - nr_short
        else:
            # Disagg: Prod/Merc + Swap = comm, M_Money = lspec, NonRept = sspec
            pm_long  = pd.to_numeric(df.get("Prod_Merc_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce").fillna(0)
            pm_short = pd.to_numeric(df.get("Prod_Merc_Positions_Short_All", pd.Series(dtype=float)), errors="coerce").fillna(0)
            sw_long  = pd.to_numeric(df.get("Swap_Positions_Long_All",       pd.Series(dtype=float)), errors="coerce").fillna(0)
            sw_short = pd.to_numeric(df.get("Swap_Positions_Short_All",      pd.Series(dtype=float)), errors="coerce").fillna(0)
            mm_long  = pd.to_numeric(df.get("M_Money_Positions_Long_All",    pd.Series(dtype=float)), errors="coerce").fillna(0)
            mm_short = pd.to_numeric(df.get("M_Money_Positions_Short_All",   pd.Series(dtype=float)), errors="coerce").fillna(0)
            nr_long  = pd.to_numeric(df.get("NonRept_Positions_Long_All",    pd.Series(dtype=float)), errors="coerce").fillna(0)
            nr_short = pd.to_numeric(df.get("NonRept_Positions_Short_All",   pd.Series(dtype=float)), errors="coerce").fillna(0)
            df["comm_positions_long_all"]    = pm_long + sw_long
            df["comm_positions_short_all"]   = pm_short + sw_short
            df["noncomm_positions_long_all"] = mm_long
            df["noncomm_positions_short_all"]= mm_short
            df["nonrept_positions_long_all"] = nr_long
            df["nonrept_positions_short_all"]= nr_short
            df["comm_net"]  = (pm_long + sw_long) - (pm_short + sw_short)
            df["lspec_net"] = mm_long - mm_short
            df["sspec_net"] = nr_long - nr_short

        df["lspec_chg"] = df["lspec_net"].diff().fillna(0)

        keep_cols = ["date","comm_net","lspec_net","sspec_net","lspec_chg",
                     "comm_positions_long_all","comm_positions_short_all",
                     "noncomm_positions_long_all","noncomm_positions_short_all",
                     "nonrept_positions_long_all","nonrept_positions_short_all",
                     "open_interest_all"]
        df = df[[c for c in keep_cols if c in df.columns]].dropna(subset=["comm_net"])
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

        n_new = len(df)
        # Merge with existing cached history instead of replacing it — keeps the
        # full multi-year history while adding/refreshing the injected weeks.
        df = _ice_store_merged(market, df)

        # Bust downstream result caches so the fresh data is served immediately
        # (cot-history and setup-stats are result-cached per MARKET ID for 1h).
        try:
            _mkt_id = next((x["id"] for x in MARKETS if x.get("ice_code") == market), None)
            if _mkt_id:
                _COT_HIST_RESULT_CACHE.pop(_mkt_id, None)
                _SETUP_STATS_CACHE.pop(_mkt_id, None)
        except Exception:
            pass

        print(f"[INJECT ICE] {market} ({fmt}): {n_new} rows injected, merged store now "
              f"{len(df)} rows, {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
        return {
            "ok": True,
            "market": market,
            "rows_injected": len(df),
            "date_min": str(df["date"].iloc[0].date()),
            "date_max": str(df["date"].iloc[-1].date()),
        }
    except Exception as e:
        print(f"[INJECT ICE] {market}: error — {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/inject-ff-macro")
async def inject_ff_macro(payload: dict):
    """
    Accepts pre-fetched ForexFactory actual-vs-forecast surprise data for non-USD
    currencies and injects it into _FRED_CCY_CACHE, replacing the FRED trailing-average
    fallback with real consensus-based surprise scores.

    Payload schema:
    {
      "currency": "GBP",
      "score": 6.2,
      "label": "GBP Macro Improving",
      "cats": {"inflation": 0.5, "jobs": 0.8, "growth": -0.2},
      "cat_details": {"inflation": [{"name":"CPI","actual":"0.3%","forecast":"0.2%","score":1}], ...}
    }
    """
    global _FRED_CCY_CACHE, FF_MACRO_CACHE, ALL_DATA_CACHE
    currency = payload.get("currency", "").upper()
    score    = payload.get("score")
    label    = payload.get("label", "")
    cats     = payload.get("cats", {})
    cat_details = payload.get("cat_details", {})

    VALID_CURRENCIES = {"EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
    if currency not in VALID_CURRENCIES:
        return {"ok": False, "error": f"Unknown currency: {currency}"}
    if score is None:
        return {"ok": False, "error": "Missing score"}

    now = time.time()
    injected = {
        "score":       float(score),
        "label":       label,
        "currency":    currency,
        "cats":        cats,
        "cat_details": cat_details,
        "source":      "ff_injected",  # distinguishes from FRED fallback
    }
    # Write into _FRED_CCY_CACHE with source=ff_injected so the TTL check uses 24h
    _FRED_CCY_CACHE[currency] = {"data": injected, "time": now}

    # Bust FF_MACRO_CACHE so next scores request recomputes with new ff_injected data
    FF_MACRO_CACHE["data"] = None
    FF_MACRO_CACHE["time"] = 0

    # Also bust ALL_DATA_CACHE so next /api/scores response includes ff_injected data
    # Without this, stale cache would serve FRED data for up to 60min after injection
    ALL_DATA_CACHE["data"] = None
    ALL_DATA_CACHE["time"] = 0

    print(f"[FF MACRO INJECT] {currency}: score={score:.1f}, label={label}, cats={cats}")
    return {"ok": True, "currency": currency, "score": score, "label": label}


@app.post("/api/inject-ff-labour")
async def inject_ff_labour(payload: dict):
    """
    Accepts pre-fetched ForexFactory USD labour + inflation event data from the sandbox
    (where FF is accessible) and merges it into the on-disk ff_event_store.json.
    This keeps the store populated with NFP/UNEMP/Claims/ADP/Wages/CPI data even
    after the current week rolls over on the faireconomy JSON feed.

    Payload schema:
    {
      "events": [
        {
          "name": "Non-Farm Payrolls",
          "currency": "USD",
          "actual": "57K",
          "forecast": "164K",
          "previous": "139K",
          "dateline": 1751540400,
          "impactClass": "high"
        }, ...
      ]
    }
    Returns: {ok, n_merged, n_store_total}
    """
    global _FF_LABOUR_CACHE, _FF_INFL_CACHE, _FF_GROWTH_CACHE, ALL_DATA_CACHE

    events = payload.get("events", [])
    if not events or not isinstance(events, list):
        return {"ok": False, "error": "Missing or invalid events list"}

    store = _ff_store_load()
    n_merged = 0
    for ev in events:
        if not ev.get("actual") or not ev.get("name"):
            continue
        ts = ev.get("dateline")
        if ts:
            day = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        else:
            day = "na"
        key = f"{ev.get('currency','')}|{ev.get('name','')}|{day}"
        store[key] = ev
        n_merged += 1

    # Prune old events
    cutoff = time.time() - _FF_STORE_MAX_DAYS * 86400
    store = {k: v for k, v in store.items() if (v.get("dateline") or 0) >= cutoff}
    _ff_store_save(store)

    # Bust FF labour, inflation, macro, and regime caches so next scores request re-reads the store
    _FF_LABOUR_CACHE["data"] = None
    _FF_LABOUR_CACHE["time"] = 0
    _FF_INFL_CACHE["data"] = None
    _FF_INFL_CACHE["time"] = 0
    _FF_GROWTH_CACHE["data"] = None
    _FF_GROWTH_CACHE["time"] = 0
    US_MACRO_CACHE["data"] = None
    US_MACRO_CACHE["time"] = 0
    RISK_REGIME_CACHE["data"] = None
    RISK_REGIME_CACHE["time"] = 0
    ALL_DATA_CACHE["data"] = None
    ALL_DATA_CACHE["time"] = 0

    print(f"[FF LABOUR INJECT] Merged {n_merged} events into store ({len(store)} total)")
    return {"ok": True, "n_merged": n_merged, "n_store_total": len(store)}


@app.get("/api/tunnel-url")
async def tunnel_url():
    """Returns the current live Cloudflare tunnel URL.
    The watchdog writes this to /tmp/cloudflare_tunnel_url.txt on every (re)start.
    The frontend fetches this on load so it always uses the right URL.
    """
    url_file = '/tmp/cloudflare_tunnel_url.txt'
    try:
        with open(url_file) as f:
            url = f.read().strip()
        if url:
            return {"url": url}
    except Exception:
        pass
    return {"url": None}

@app.get("/")
async def serve_index():
    idx = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(idx):
        return JSONResponse({"status": "BH Weather System API", "docs": "/docs"})
    return FileResponse(idx, media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate"})

# Startup warmup disabled — caches populate on first request to avoid OOM
# The keepalive cron handles backend health and restarts if needed
_WARMING = {"done": False, "started": False}


async def _startup_load_ice_data():
    """
    Called once at startup. For each ICE market:
      1. Try disk cache (written by the Friday cron injection) — fast, always works
         as long as a deploy hasn’t wiped the disk.
      2. If disk is cold, attempt a live HTTP fetch from ICE.com.
         Render’s shared egress IP may 429 — we swallow all errors silently.

    Runs BEFORE the main scores pre-warm so ICE COT data is available
    immediately for get_all_scores() on the first startup.
    """
    # ICE markets removed from the app (2026-07-12) — nothing to warm up.
    return

    import asyncio as _ice_loop
    import time as _ice_time

    ICE_DISAGG_MARKETS = ["B", "G", "RC"]   # Brent, Gas Oil, Robusta
    ICE_TFF_MARKETS    = ["Z", "R"]          # FTSE 100, Long Gilt

    loaded = []
    cold   = []

    # Step 1: Load from disk cache (near-instant)
    for mkt in ICE_DISAGG_MARKETS + ICE_TFF_MARKETS:
        if mkt in _ICE_MEM_CACHE:
            loaded.append(mkt)
            continue
        disk_df = _load_ice_from_disk(mkt)
        if disk_df is not None and not disk_df.empty:
            _ICE_MEM_CACHE[mkt] = {"df": disk_df, "ts": _ice_time.time()}
            loaded.append(mkt)
            print(f"[startup-ice] {mkt}: loaded {len(disk_df)} rows from disk cache")
        else:
            cold.append(mkt)

    if not cold:
        print(f"[startup-ice] All ICE markets loaded from disk: {loaded}")
        return

    print(f"[startup-ice] Cold markets (attempting live fetch): {cold}")

    # Step 2: For cold markets, try a live fetch in a thread pool
    # _fetch_ice_cot_raw handles HTTP, CSV parsing, mem+disk caching
    loop = _ice_loop.get_event_loop()
    for mkt in cold:
        try:
            df = await loop.run_in_executor(_APP_EXECUTOR, _fetch_ice_cot_raw, mkt)
            if df is not None and not df.empty:
                loaded.append(mkt)
                print(f"[startup-ice] {mkt}: live-fetched {len(df)} rows")
            else:
                print(f"[startup-ice] {mkt}: live fetch returned no data (429 or empty) — will populate via Friday cron")
        except Exception as _e:
            print(f"[startup-ice] {mkt}: live fetch failed ({_e}) — non-fatal")

    print(f"[startup-ice] Done. Loaded: {loaded}, still cold: {[m for m in cold if m not in loaded]}")

@app.get("/api/warmup-status")
async def warmup_status():
    return {"ready": _WARMING["done"], "warming": _WARMING["started"]}

@app.on_event("shutdown")
async def graceful_shutdown():
    """Graceful SIGTERM handler — gives in-flight requests up to 30 seconds to
    complete before the process exits. Prevents dropped requests on Render deploy."""
    import asyncio as _ashutdown
    print("[shutdown] SIGTERM received — starting 30s graceful drain...", flush=True)
    _WARMING["done"] = False  # Stop accepting new scores requests during drain
    # Give in-flight requests time to complete
    await _ashutdown.sleep(30)
    print("[shutdown] 30s drain complete — process exiting", flush=True)


@app.on_event("startup")
async def warmup_cache():
    """Fully pre-warm all caches on startup — calls get_all_scores() directly so
    ALL_DATA_CACHE is fully populated before any user request arrives."""
    import asyncio as _astart
    async def _warm():
        # ── INSTANT cold-start: load last full payload from disk FIRST, before the
        # (slow) full warm-up. This lets /api/scores serve last-known data
        # immediately during the warming window instead of a 202 blank screen.
        try:
            _load_full_scores_snapshot()
        except Exception as _e:
            print(f"[startup] full snapshot load error (non-fatal): {_e}")
        await _astart.sleep(2)
        _WARMING["started"] = True

        # ── Consensus outlook: warm from disk so a redeploy keeps last week's read
        try:
            _load_consensus_from_disk()
        except Exception as _e:
            print(f"[startup] consensus disk warm error (non-fatal): {_e}")

        # ── ICE COT startup fetch ────────────────────────────────────────────
        # Attempt to load ICE data before the main pre-warm so COT scores for
        # ICE markets (B/G/RC/Z/R) are correct from the first request.
        # Strategy: disk cache (from previous cron injection) loads instantly.
        # If disk is cold (first deploy), attempt a live fetch from ICE.com.
        # Render's shared IP may 429 on live fetch — that's fine, the Friday
        # cron will inject within the week.
        await _startup_load_ice_data()

        print("[startup] Pre-warming all caches (full scores run)...")
        try:
            try:
                # Call the full scores endpoint directly (not HTTP) — this populates
                # ALL_DATA_CACHE, MACRO_CACHE, REGIME_CACHE, FF_MACRO_CACHE in one shot.
                # force=True skips the 202 guard so we don't get stuck in warming loop.
                await get_all_scores(force=True)
                print("[startup] Pre-warm complete — ALL_DATA_CACHE populated")
            except Exception as e:
                print(f"[startup] Pre-warm error (non-fatal): {e}")
            # Pre-warm yield curve history cache (separate FRED fetch, not in scores)
            try:
                await _fetch_yield_curve_history_async()
                print("[startup] Yield curve history cache pre-warmed")
            except Exception as e:
                print(f"[startup] Yield curve pre-warm error (non-fatal): {e}")
            _gc_if_heavy("post-startup-warmup")
        finally:
            # CRITICAL: always mark warming done — even if an unhandled exception
            # occurred above, so the backend never gets stuck in permanent 202 state.
            _WARMING["done"] = True
            print("[startup] _WARMING[done]=True set (try/finally guaranteed)")
    _astart.ensure_future(_warm())
    print("[startup] Backend ready — full cache pre-warming in background")


# Suppress /api/health from access logs — keepalives flood the log stream
# Applied at module level so it works whether started via __main__ or Procfile uvicorn
import logging as _logging
class _NoHealthFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Suppress healthy keepalive pings only — still log errors (non-200)
        return not ("/api/health" in msg and "200" in msg)
_logging.getLogger("uvicorn.access").addFilter(_NoHealthFilter())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
