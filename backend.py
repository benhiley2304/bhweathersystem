"""
COT Weather Station — Backend v2
FastAPI server: CFTC COT (all 3 groups), FRED macro (surprise-based),
Yahoo Finance prices, cross-asset regime. Returns structured JSON.
"""

import asyncio
import json
import math
import time
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
import gc, os, pathlib, re
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

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
_APP_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=32, thread_name_prefix="bh-worker")
app.mount("/photos", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")), name="photos")

# GZip: compress responses >1KB — reduces /api/scores from ~216KB to ~29KB over the wire
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    {"id": "6J",  "name": "Yen Futures",   "ticker": "6J1!",  "yf": "JPYUSD=X",    "category": "fx",        "cftc_code": "097741", "cftc_name": "JAPANESE YEN",
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
     "cot_note": "Pure CFTC Arabica COT data (NY, ~196k OI). Arabica and Robusta are structurally different commodities with separate supply chains, participant profiles, and commercial bases — blending their COT data adds noise rather than signal. KC scores are therefore based solely on Arabica positioning. Cross-reference RC (standalone ICE Robusta) for the separate Robusta market view."},
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
     "cot_note": "COT derived: 6E Briese − 6J Briese. Measures EUR positioning advantage over JPY."},
    {"id": "EURGBP", "name": "EUR/GBP", "ticker": "EURGBP", "yf": "EURGBP=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6B",
     "cot_note": "COT derived: 6E Briese − 6B Briese. Measures EUR positioning advantage over GBP."},
    {"id": "EURAUD", "name": "EUR/AUD", "ticker": "EURAUD", "yf": "EURAUD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6A",
     "cot_note": "COT derived: 6E Briese − 6A Briese. Measures EUR positioning advantage over AUD."},
    {"id": "EURCAD", "name": "EUR/CAD", "ticker": "EURCAD", "yf": "EURCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6C",
     "cot_note": "COT derived: 6E Briese − 6C Briese. Measures EUR positioning advantage over CAD."},
    {"id": "EURNZD", "name": "EUR/NZD", "ticker": "EURNZD", "yf": "EURNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6N",
     "cot_note": "COT derived: 6E Briese − 6N Briese. Measures EUR positioning advantage over NZD."},
    {"id": "EURCHF", "name": "EUR/CHF", "ticker": "EURCHF", "yf": "EURCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6E", "quote_leg": "6S",
     "cot_note": "COT derived: 6E Briese − 6S Briese. Measures EUR positioning advantage over CHF."},
    {"id": "GBPJPY", "name": "GBP/JPY", "ticker": "GBPJPY", "yf": "GBPJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6J",
     "cot_note": "COT derived: 6B Briese − 6J Briese. Measures GBP positioning advantage over JPY."},
    {"id": "GBPAUD", "name": "GBP/AUD", "ticker": "GBPAUD", "yf": "GBPAUD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6A",
     "cot_note": "COT derived: 6B Briese − 6A Briese. Measures GBP positioning advantage over AUD."},
    {"id": "GBPCAD", "name": "GBP/CAD", "ticker": "GBPCAD", "yf": "GBPCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6C",
     "cot_note": "COT derived: 6B Briese − 6C Briese. Measures GBP positioning advantage over CAD."},
    {"id": "GBPNZD", "name": "GBP/NZD", "ticker": "GBPNZD", "yf": "GBPNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6N",
     "cot_note": "COT derived: 6B Briese − 6N Briese. Measures GBP positioning advantage over NZD."},
    {"id": "GBPCHF", "name": "GBP/CHF", "ticker": "GBPCHF", "yf": "GBPCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6B", "quote_leg": "6S",
     "cot_note": "COT derived: 6B Briese − 6S Briese. Measures GBP positioning advantage over CHF."},
    {"id": "AUDJPY", "name": "AUD/JPY", "ticker": "AUDJPY", "yf": "AUDJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6J",
     "cot_note": "COT derived: 6A Briese − 6J Briese. Classic risk barometer — bullish = risk-on."},
    {"id": "AUDNZD", "name": "AUD/NZD", "ticker": "AUDNZD", "yf": "AUDNZD=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6N",
     "cot_note": "COT derived: 6A Briese − 6N Briese. Measures AUD positioning advantage over NZD."},
    {"id": "AUDCAD", "name": "AUD/CAD", "ticker": "AUDCAD", "yf": "AUDCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6C",
     "cot_note": "COT derived: 6A Briese − 6C Briese. Both commodity currencies — spread captures relative commodity exposure."},
    {"id": "NZDJPY", "name": "NZD/JPY", "ticker": "NZDJPY", "yf": "NZDJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6N", "quote_leg": "6J",
     "cot_note": "COT derived: 6N Briese − 6J Briese. Risk barometer — bullish = risk-on."},
    {"id": "NZDCAD", "name": "NZD/CAD", "ticker": "NZDCAD", "yf": "NZDCAD=X", "category": "fx_cross", "cross": True, "base_leg": "6N", "quote_leg": "6C",
     "cot_note": "COT derived: 6N Briese − 6C Briese. Commodity currency spread."},
    {"id": "CADJPY", "name": "CAD/JPY", "ticker": "CADJPY", "yf": "CADJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6C", "quote_leg": "6J",
     "cot_note": "COT derived: 6C Briese − 6J Briese. Oil-linked risk barometer."},
    {"id": "CHFJPY", "name": "CHF/JPY", "ticker": "CHFJPY", "yf": "CHFJPY=X", "category": "fx_cross", "cross": True, "base_leg": "6S", "quote_leg": "6J",
     "cot_note": "COT derived: 6S Briese − 6J Briese. Dual safe-haven pair — risk-off = bearish (JPY strengthens more)."},
    {"id": "AUDCHF", "name": "AUD/CHF", "ticker": "AUDCHF", "yf": "AUDCHF=X", "category": "fx_cross", "cross": True, "base_leg": "6A", "quote_leg": "6S",
     "cot_note": "COT derived: 6A Briese − 6S Briese. Risk appetite gauge — AUD vs safe-haven CHF."},

    # ── ICE Europe markets ───────────────────────────────────────────────────────────
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

    # Gas Oil — European diesel/heating oil benchmark (equivalent of CFTC HO)
    # 329 weeks of data. Commercials currently 99th %ile — historically extreme.
    {"id": "GO", "name": "Gas Oil",      "ticker": "QS1!",  "yf": "HO=F",    "category": "commodity",  # HO=F = NYMEX Heating Oil, best YF proxy for Gas Oil
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
        "peers": [
            {"id": "CL",  "yf": "CL=F",           "label": "vs Crude",    "color": "#34d399"},
            {"id": "HO",  "yf": "HO=F",           "label": "vs Heat Oil",  "color": "#fb923c"},
        ],
        "periods": [13, 26],
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
        "peers": [
            {"id": "LE",  "yf": "LE=F",           "label": "vs Live Cattle","color": "#d97706"},
            {"id": "ZC",  "yf": "ZC=F",           "label": "vs Corn",      "color": "#fde68a"},
        ],
        "periods": [13, 26],
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
    _ema_short_tg = _pdtg.Series(closes_arr.astype(float)).ewm(span=10, adjust=False).mean().values
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
# Weights in WEIGHTS_EXTENDED below:
#   TIER-1 (10%): ES, NQ, RTY, YM, GC, SI   — deep markets, strong backtest edge
#   TIER-2 ( 5%): CL                         — moderate liquidity, bidirectional signal
#   TIER-3 ( 3%): BTC, ETH                   — good depth on Deribit, unique norms

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
    "GC":  {"tier": 1, "source": "cboe_etf"},
    "SI":  {"tier": 1, "source": "cboe_etf"},
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
            try:
                resp = requests.get(url, headers=daily_headers, timeout=5)
                if resp.status_code == 200:
                    j = resp.json()
                    equity = next(
                        (float(x["value"]) for x in j.get("ratios", [])
                         if "EQUITY PUT" in x.get("name", "")),
                        None
                    )
                    if equity is not None:
                        return {"DATE": d, "equity_pc": equity}
            except Exception:
                pass
            return None

        with _TPE(max_workers=25) as ex:
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

    # ── EQUITY: use daily CBOE aggregate equity P/C ratio ──────────────────
    if source == "cboe_equity":
        df = fetch_pcr_history()
        if df is None or df.empty:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": "Could not fetch P/C ratio data"}}

        df_clean = df.dropna(subset=["pc_ma20"])
        if df_clean.empty:
            return {"score": 5.0, "label": "No Data", "tier": tier,
                    "detail": {"error": "Insufficient history for MA"}}

        latest        = df_clean.iloc[-1]
        current_daily = float(latest["equity_pc"])
        current_ma20  = float(latest["pc_ma20"])
        latest_date   = str(df_clean.index[-1].date())

        all_ma20   = df_clean["pc_ma20"].values
        percentile = float(np.mean(all_ma20 < current_ma20))
        score      = round(max(0.0, min(10.0, percentile * 10)), 1)

        if percentile >= 0.90:   label = "Extreme Fear"
        elif percentile >= 0.75: label = "High Fear"
        elif percentile >= 0.60: label = "Mild Fear"
        elif percentile >= 0.40: label = "Neutral"
        elif percentile >= 0.25: label = "Mild Greed"
        elif percentile >= 0.10: label = "High Greed"
        else:                    label = "Extreme Greed"

        if percentile >= 0.75:   signal = "Contrarian Bullish"
        elif percentile >= 0.60: signal = "Lean Bullish"
        elif percentile <= 0.25: signal = "Contrarian Bearish"
        elif percentile <= 0.40: signal = "Lean Bearish"
        else:                    signal = "Neutral"

        return {
            "score": score, "label": label, "tier": tier,
            "detail": {
                "current_daily": round(current_daily, 3),
                "ma20": round(current_ma20, 3),
                "percentile": round(percentile * 100, 1),
                "signal": signal, "label": label,
                "latest_date": latest_date,
                "source": "CBOE Aggregate Equity P/C",
                "thresholds": {
                    "extreme_greed": round(float(np.percentile(all_ma20, 10)), 3),
                    "moderate_greed": round(float(np.percentile(all_ma20, 25)), 3),
                    "neutral_low":    round(float(np.percentile(all_ma20, 40)), 3),
                    "neutral_high":   round(float(np.percentile(all_ma20, 60)), 3),
                    "moderate_fear":  round(float(np.percentile(all_ma20, 75)), 3),
                    "extreme_fear":   round(float(np.percentile(all_ma20, 90)), 3),
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

        # Calibrated thresholds (from option market structural analysis + backtest)
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

_ICE_DISK_CACHE_DIR = "/tmp/ice_cot_cache"
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
    cached = _ICE_MEM_CACHE.get(ice_market_code)
    if cached and (now - cached["ts"]) < _ICE_MEM_CACHE_TTL:
        return cached["df"]

    disk_df = _load_ice_from_disk(ice_market_code)
    if disk_df is not None and not disk_df.empty:
        _ICE_MEM_CACHE[ice_market_code] = {"df": disk_df, "ts": now}
        return disk_df

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

        _ICE_MEM_CACHE[ice_market_code] = {"df": df, "ts": now}
        _save_ice_to_disk(ice_market_code, df)
        return df

    except Exception as e:
        print(f"[ICE FIN COT] parse error {ice_market_code}: {e}")
        return None


def _fetch_ice_cot_raw(ice_market_code: str) -> Optional[pd.DataFrame]:
    """
    Fetch ICE Europe COT data for a given market code.
    Uses the official ICE annual CSV files: https://www.ice.com/publicdocs/futures/COTHist{year}.csv
    Each file contains all markets; we filter by CFTC_Commodity_Code == ice_market_code
    and FutOnly_or_Combined == 'FutOnly'.  Data available 2011-present (~15 years).
    Format is disaggregated (Prod/Merc, Swap, M_Money, NonRept) — mapped to standard
    3-group columns (comm/lspec/sspec) to match fetch_cot_history() output.
    Z (FTSE) and R (Long Gilt) are NOT in these files — returns None gracefully.
    """
    import csv as _csv
    import io as _io

    now = time.time()
    cached = _ICE_MEM_CACHE.get(ice_market_code)
    if cached and (now - cached["ts"]) < _ICE_MEM_CACHE_TTL:
        return cached["df"]

    # Try disk cache first
    disk_df = _load_ice_from_disk(ice_market_code)
    if disk_df is not None and not disk_df.empty:
        _ICE_MEM_CACHE[ice_market_code] = {"df": disk_df, "ts": now}
        return disk_df

    # Z (FTSE 100) and R (Long Gilt) use the EUFINCOTHist series (TFF format)
    # instead of the standard COTHist disaggregated files.
    if ice_market_code in ("Z", "R"):
        return _fetch_ice_fin_cot_raw(ice_market_code)

    _headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.ice.com/report/122",
    }

    all_rows = []
    current_year = date.today().year
    # Fetch from 2011 onward for maximum history (~15 years)
    START_YEAR = 2011
    consecutive_failures = 0

    for year in range(START_YEAR, current_year + 1):
        url = f"https://www.ice.com/publicdocs/futures/COTHist{year}.csv"
        try:
            # Small delay to be polite to ICE
            time.sleep(0.4)
            _req = requests.get(url, timeout=30, headers=_headers)
            if _req.status_code == 429:
                # Rate limited — abort if we keep hitting 429 (don’t block the executor)
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    print(f"[ICE COT] {ice_market_code}: rate limited repeatedly, aborting fetch")
                    break
                print(f"[ICE COT] {ice_market_code}: rate limited for {year}, waiting 10s")
                time.sleep(10)
                _req = requests.get(url, timeout=30, headers=_headers)
                if _req.status_code != 200:
                    print(f"[ICE COT] {ice_market_code}: still rate limited for {year}, aborting")
                    break
            if _req.status_code != 200:
                print(f"[ICE COT] {ice_market_code}: HTTP {_req.status_code} for {year}")
                consecutive_failures += 1
                continue
            consecutive_failures = 0
            content = _req.content.decode("utf-8-sig")  # strips BOM
            if "<!doctype" in content[:200].lower():
                print(f"[ICE COT] {ice_market_code}: HTML/rate-limit response for {year}")
                consecutive_failures += 1
                continue

            reader = _csv.DictReader(_io.StringIO(content))
            rows   = list(reader)

            # Filter to target market code, futures-only
            _g  = [r for r in rows if r.get("CFTC_Commodity_Code", "").strip() == ice_market_code]
            _gs = [r for r in _g  if r.get("FutOnly_or_Combined", "") == "FutOnly"]
            if not _gs and _g:
                _gs = _g  # fall back to combined if FutOnly not present

            all_rows.extend(_gs)
        except Exception as e:
            print(f"[ICE COT] {ice_market_code} {year}: {e}")
            continue

    if not all_rows:
        # Rate-limited or no data — fall back to stale disk cache if available
        disk_fallback = _load_ice_from_disk(ice_market_code)
        if disk_fallback is not None and not disk_fallback.empty:
            print(f"[ICE COT] {ice_market_code}: using stale disk cache ({len(disk_fallback)} rows) due to fetch failure")
            _ICE_MEM_CACHE[ice_market_code] = {"df": disk_fallback, "ts": now}
            return disk_fallback
        print(f"[ICE COT] {ice_market_code}: no data found across all years")
        return None

    try:
        df = pd.DataFrame(all_rows)

        # Parse date — ICE format: MM/DD/YYYY in 'As_of_Date_Form_MM/DD/YYYY'
        df["date"] = pd.to_datetime(df.get("As_of_Date_Form_MM/DD/YYYY", pd.Series(dtype=str)), errors="coerce")
        # Fallback: YYMMDD in 'As_of_Date_In_Form_YYMMDD'
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

        # Disaggregated → 3-group mapping:
        #   comm_net   = Prod/Merc + Swap Dealers (hedgers + financial intermediaries)
        #   lspec_net  = Managed Money (CTAs, hedge funds, asset managers)
        #   sspec_net  = Other Reportables (other large reportables — distinct from NonRept retail)
        prod_long  = pd.to_numeric(df.get("Prod_Merc_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce")
        prod_short = pd.to_numeric(df.get("Prod_Merc_Positions_Short_All", pd.Series(dtype=float)), errors="coerce")
        swap_long  = pd.to_numeric(df.get("Swap_Positions_Long_All",       pd.Series(dtype=float)), errors="coerce")
        swap_short = pd.to_numeric(df.get("Swap_Positions_Short_All",      pd.Series(dtype=float)), errors="coerce")
        mm_long    = pd.to_numeric(df.get("M_Money_Positions_Long_All",    pd.Series(dtype=float)), errors="coerce")
        mm_short   = pd.to_numeric(df.get("M_Money_Positions_Short_All",   pd.Series(dtype=float)), errors="coerce")
        # Other Reportables: large-account category not fitting Prod/Swap/MM — e.g. corporates, banks
        or_long    = pd.to_numeric(df.get("Other_Rept_Positions_Long_All",  pd.Series(dtype=float)), errors="coerce")
        or_short   = pd.to_numeric(df.get("Other_Rept_Positions_Short_All", pd.Series(dtype=float)), errors="coerce")

        df["comm_positions_long_all"]    = (prod_long  + swap_long.fillna(0)).fillna(prod_long)
        df["comm_positions_short_all"]   = (prod_short + swap_short.fillna(0)).fillna(prod_short)
        df["noncomm_positions_long_all"] = mm_long
        df["noncomm_positions_short_all"]= mm_short
        df["nonrept_positions_long_all"] = or_long    # Other Reportables (not NonRept retail)
        df["nonrept_positions_short_all"]= or_short

        df["comm_net"]  = df["comm_positions_long_all"]  - df["comm_positions_short_all"]
        df["lspec_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
        df["sspec_net"] = or_long - or_short  # Other Reportables
        df["lspec_chg"] = df["lspec_net"].diff().fillna(0)

        df = df[["date","comm_net","lspec_net","sspec_net","lspec_chg",
                 "comm_positions_long_all","comm_positions_short_all",
                 "noncomm_positions_long_all","noncomm_positions_short_all",
                 "nonrept_positions_long_all","nonrept_positions_short_all",
                 "open_interest_all"]].dropna(subset=["comm_net"])

        df = df.sort_values("date").reset_index(drop=True)

        _ICE_MEM_CACHE[ice_market_code] = {"df": df, "ts": now}
        _save_ice_to_disk(ice_market_code, df)
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


def compute_crypto_cot_score(df: Optional[pd.DataFrame], market_id: str = "") -> dict:
    """
    Crypto-specific COT scoring.
    Primary signal: Large Specs (fund managers) positioning, treated as:
      - Momentum signal in the direction of trend (below 60, rising = bullish)
      - Contrarian signal at extremes (above 80 = bearish, below 20 = bullish)
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

    score = 5.0
    detail_parts = []

    # Momentum: rising managers = bullish crypto (they are the primary signal in crypto)
    n = min(7, len(lspec_net))
    sh = lspec_net[-n:]
    w2 = np.polyfit(np.arange(n), sh, 1)[0] if n >= 3 else 0
    lspec_momentum = round(float(w2), 1)
    if lspec_momentum > 0:
        score += min(1.8, lspec_momentum * 0.0001 * len(lspec_net))
        detail_parts.append(f"Fund mgr momentum +{lspec_momentum:.0f} pts — {'accelerating accumulation' if lspec_momentum > 1000 else 'gradual accumulation'}")
    elif lspec_momentum < 0:
        score += max(-1.8, lspec_momentum * 0.0001 * len(lspec_net))
        detail_parts.append(f"Fund mgr momentum {lspec_momentum:.0f} pts — {'accelerating de-risking' if lspec_momentum < -1000 else 'gradual de-risking'}")

    # Level extremes (contrarian at extremes)
    if lspec_briese >= 95: score -= 1.6; detail_parts.append("Fund Mgr Normalisation BEAR — extreme longs")
    elif lspec_briese >= 80: score -= 0.8; detail_parts.append("Fund Mgr Normalisation BEAR — elevated")
    elif lspec_briese <= 5: score += 1.4; detail_parts.append("Fund Mgr Normalisation BULL — extreme shorts")
    elif lspec_briese <= 20: score += 0.8; detail_parts.append("Fund Mgr Normalisation BULL — depressed")

    # Label
    phase_str = "early ride" if lspec_briese < 40 else "mid ride" if lspec_briese < 65 else "late ride"
    if lspec_briese >= 80:    stance = "heavily long — bearish"
    elif lspec_briese >= 65:  stance = "above neutral — bullish lean"
    elif lspec_briese <= 35:  stance = "below neutral — bearish lean"
    else:                     stance = "heavily short — bearish"

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
            "lspec_momentum": lspec_momentum,
            "cot_phase": 0, "cot_phase_dir": "neutral",
            "cot_phase_label": "Crypto COT", "cot_phase_desc": "",
        },
    }


def compute_cross_cot_score(market_id: str, base_leg: str, quote_leg: str, cot_cache: dict) -> dict:
    """
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


def fetch_price_data(yf_ticker: str) -> Optional[pd.DataFrame]:
    now = time.time()
    if yf_ticker in PRICE_CACHE and (now - PRICE_CACHE.get(yf_ticker + "_t", 0)) < PRICE_CACHE_TTL:
        return PRICE_CACHE[yf_ticker]
    try:
        tk = yf.Ticker(yf_ticker)
        df = tk.history(period="1y", interval="1d")
        PRICE_CACHE[yf_ticker]          = df
        PRICE_CACHE[yf_ticker + "_t"]   = now
        return df
    except Exception:
        return None


def fetch_price_data_long(yf_ticker: str) -> Optional[pd.DataFrame]:
    """Fetch up to 5 years of daily price data."""
    cache_key = yf_ticker + "_long"
    now = time.time()
    if cache_key in PRICE_CACHE and (now - PRICE_CACHE.get(cache_key + "_t", 0)) < PRICE_CACHE_TTL:
        return PRICE_CACHE[cache_key]
    try:
        tk = yf.Ticker(yf_ticker)
        df = tk.history(period="5y", interval="1d")
        PRICE_CACHE[cache_key]          = df
        PRICE_CACHE[cache_key + "_t"]   = now
        return df
    except Exception:
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

    # Sub-scores: each −2 to +2 with tilt toward shorter-term
    sub_scores = {}

    # ST EMA slope (8-EMA vs 5 bars ago) — highest weight
    if ema_st_slope_pct > 5:    sub_scores["ema_st"] = 2
    elif ema_st_slope_pct > 1:  sub_scores["ema_st"] = 1
    elif ema_st_slope_pct < -5: sub_scores["ema_st"] = -2
    elif ema_st_slope_pct < -1: sub_scores["ema_st"] = -1
    else:                        sub_scores["ema_st"] = 0

    # 1w ROC
    if roc1w > 3:    sub_scores["roc1w"] = 2
    elif roc1w > 0.5:sub_scores["roc1w"] = 1
    elif roc1w < -3: sub_scores["roc1w"] = -2
    elif roc1w < -0.5:sub_scores["roc1w"] = -1
    else:             sub_scores["roc1w"] = 0

    # 4w ROC
    if roc4w > 10:   sub_scores["roc4w"] = 2
    elif roc4w > 3:  sub_scores["roc4w"] = 1
    elif roc4w < -10:sub_scores["roc4w"] = -2
    elif roc4w < -3: sub_scores["roc4w"] = -1
    else:             sub_scores["roc4w"] = 0

    # 13w ROC (medium term)
    if roc13w > 13:  sub_scores["roc13w"] = 2
    elif roc13w > 5: sub_scores["roc13w"] = 1
    elif roc13w < -13:sub_scores["roc13w"] = -2
    elif roc13w < -5: sub_scores["roc13w"] = -1
    else:              sub_scores["roc13w"] = 0

    # 200 SMA position (long-term filter)
    if sma200_pct_diff > 10:    sub_scores["sma200"] = 2
    elif sma200_pct_diff > 3:   sub_scores["sma200"] = 1
    elif sma200_pct_diff < -10: sub_scores["sma200"] = -2
    elif sma200_pct_diff < -3:  sub_scores["sma200"] = -1
    else:                        sub_scores["sma200"] = 0

    # Weighted sum: tilt toward shorter-term
    weights = {"ema_st": 0.30, "roc1w": 0.25, "roc4w": 0.20, "roc13w": 0.15, "sma200": 0.10}
    raw = sum(sub_scores.get(k, 0) * w for k, w in weights.items())
    # Map -2..+2 to 0..10
    score = round(max(0.0, min(10.0, raw * 2.5 + 5.0)), 1)

    if score >= 7.5:  label = "Strong Uptrend"
    elif score >= 6.0:label = "Mild Uptrend"
    elif score >= 4.5:label = "Neutral"
    elif score >= 3.0:label = "Mild Downtrend"
    else:              label = "Strong Downtrend"

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
            "sub_scores": sub_scores,
        },
    }


# ============================================================
# FOREX FACTORY — FF-BASED MACRO ENGINE
# ============================================================

FF_CACHE: dict = {"data": None, "time": 0}
FF_CACHE_TTL = 3600 * 3  # 3h

FF_MACRO_CACHE: dict = {"data": None, "time": 0}
FF_MACRO_CACHE_TTL = 3600 * 3

US_MACRO_CACHE: dict = {"data": None, "time": 0}
US_MACRO_TTL = 3600 * 3

_FF_MONTH_CACHE: dict = {}

# FRED fallback caches
FRED_CACHE = {}
FRED_CACHE_TIME_MAP = {}
FRED_CACHE_TTL = 3600 * 6

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
    Returns a list of day-dicts: [{dateline, events: [{ts, currency, name,
    actual, forecast, previous, impactClass}]}]

    FF calendar is rate-limited/blocked in this environment so we fall back
    to an empty list — score_history proceeds using FRED/COT/regime data only.
    """
    try:
        import calendar as _cal
        url = f"https://www.forexfactory.com/calendar?month={year}.{month:02d}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/html, */*",
        }
        r = requests.get(url, timeout=12, headers=headers)
        if r.status_code != 200:
            return []
        # Try JSON first (FF sometimes returns JSON to API clients)
        try:
            data = r.json()
            if isinstance(data, list):
                return data
        except Exception:
            pass
        # FF calendar blocked / HTML returned — return empty
        return []
    except Exception:
        return []


def _fetch_ff_months_parallel(year_month_pairs: list) -> list:
    """
    Fetch multiple months in parallel and return a flat list of event dicts,
    each with: ts, currency, name, actual, forecast, previous, impactClass.
    Uses the shared app executor to avoid thread pool deadlocks.
    """
    flat_events = []
    with _cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_fetch_ff_month, y, m): (y, m) for y, m in year_month_pairs}
        for fut in _cf.as_completed(futs):
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
    return flat_events


def _classify_ff_event(name: str, currency: str):
    """
    Returns (category, higher_is_good) if the event matches a known indicator,
    otherwise None.
    """
    nl = name.lower()
    # USD
    if currency == "USD":
        ind_map = US_MACRO_INDICATOR_MAP
    else:
        ind_map = FF_CURRENCY_INDICATOR_MAP.get(currency, {})

    for substr, (category, hig) in ind_map.items():
        if substr.lower() in nl:
            return category, hig
    return None


# ── FRED series IDs for each non-USD currency ──────────────────────────────
_FRED_CCY_SERIES = {
    # ── EUROZONE ────────────────────────────────────────────────────────────────────
    # DEUCPIALLMINMEI (Germany CPI index, monthly — most current EA CPI proxy)
    # NAEXKP01EZQ661S (EA GDP chain-linked, quarterly)
    # LRHUTTTTEZM156S (EA unemployment rate, monthly)
    "EUR": {
        "cpi":   ("DEUCPIALLMINMEI", "mom", True,  "CPI MoM (Germany proxy)", "inflation"),
        "unemp": ("LRHUTTTTEZM156S", "level", False, "EA Unemployment Rate", "jobs"),
        "gdp":   ("NAEXKP01EZQ661S", "qoq", True,  "GDP (Chain-Linked)", "growth"),
    },
    # ── UK (GBP) ────────────────────────────────────────────────────────────────────
    "GBP": {
        "cpi":   ("GBRCPIALLMINMEI", "mom", True,  "CPI MoM", "inflation"),
        "cpi_q": ("CPHPTT01GBQ659N", "level", True, "CPI YoY (quarterly)", "inflation"),
        "unemp": ("LRHUTTTTGBM156S", "level", False, "Unemployment Rate", "jobs"),
        "gdp":   ("NAEXKP01GBQ661S", "qoq", True,   "GDP (Chain-Linked)", "growth"),
    },
    # ── JAPAN (JPY) ─────────────────────────────────────────────────────────────────
    "JPY": {
        "cpi":   ("JPNCPIALLMINMEI", "mom", True,  "CPI MoM", "inflation"),
        "unemp": ("LRUN64TTJPM156S", "level", False, "Unemployment Rate", "jobs"),
        "indpro":("JPNPROINDMISMEI", "mom", True,  "Industrial Production", "growth"),
    },
    # ── AUSTRALIA (AUD) ─────────────────────────────────────────────────────────────
    "AUD": {
        "cpi":   ("AUSCPIALLQINMEI", "qoq", True,  "CPI (Quarterly)", "inflation"),
        "unemp": ("LRHUTTTTAUM156S", "level", False, "Unemployment Rate", "jobs"),
    },
    # ── CANADA (CAD) ─────────────────────────────────────────────────────────────────
    "CAD": {
        "cpi":   ("CANCPIALLMINMEI", "mom", True,  "CPI MoM", "inflation"),
        "unemp": ("LRHUTTTTCAM156S", "level", False, "Unemployment Rate", "jobs"),
    },
    # ── SWITZERLAND (CHF) ────────────────────────────────────────────────────────────
    "CHF": {
        "cpi":   ("CHECPIALLMINMEI", "mom", True,  "CPI MoM", "inflation"),
    },
    # ── NEW ZEALAND (NZD) ────────────────────────────────────────────────────────────
    "NZD": {
        "cpi":   ("NZLCPIALLQINMEI", "qoq", True,  "CPI (Quarterly)", "inflation"),
        "unemp": ("LRUN64TTNZQ156S", "level", False, "Unemployment Rate", "jobs"),
    },
}

# Cache: {currency: {"data": ..., "time": ...}}
_FRED_CCY_CACHE: dict = {}
_FRED_CCY_TTL = 3600 * 4  # 4h


def compute_fred_economy_score(currency: str) -> dict:
    """
    FRED-based economy score for a non-USD currency.
    Uses trailing-average surprise method: actual vs 3-period trailing average.
    Returns same schema as compute_ff_economy_score for frontend compatibility.
    """
    now = time.time()
    cached = _FRED_CCY_CACHE.get(currency)
    if cached and (now - cached["time"]) < _FRED_CCY_TTL:
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
                if len(vals) < 13:
                    continue
                actual = (vals[-1] / vals[-13] - 1.0) * 100 if vals[-13] != 0 else 0
                yoys   = [(vals[i]/vals[i-12]-1.0)*100 for i in range(max(12,len(vals)-4), len(vals)-1) if len(vals) > i-12 and vals[i-12] != 0]
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
    """Wrapper: always use FRED-based score (FF is blocked in this environment)."""
    return compute_fred_economy_score(currency)


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
    result = {}
    CURRENCIES = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
    for curr in CURRENCIES:
        result[curr] = compute_fred_economy_score(curr)

    # USD from compute_macro_all
    us_macro = compute_macro_all()
    cats = us_macro.get("category_scores", {})
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
    "PCE":      "PCEPI",
    "CFNAI":    "CFNAI",
    "NFP":      "PAYEMS",
    "UNEMP":    "UNRATE",
    "CLAIMS":   "ICSA",
    "DGS2":     "DGS2",
    "DGS10":    "DGS10",
    "YLDCRV":   "T10Y2Y",
    "T10Y3M":   "T10Y3M",
    "DFII10":   "DFII10",
    "FEDFUNDS": "FEDFUNDS",
    "WALCL":    "WALCL",
    # Credit spreads (ICE BofA)
    "HYOAS":    "BAMLH0A0HYM2",   # US HY OAS (bps)
    "IGOAS":    "BAMLC0A0CM",     # US IG OAS (bps)
}


def fetch_fred_series(series_id: str, periods: int = 24) -> Optional[list]:
    resolved_id = FRED_SERIES.get(series_id, series_id)
    cache_key = resolved_id
    now = time.time()
    if cache_key in FRED_CACHE and (now - FRED_CACHE_TIME_MAP.get(cache_key, 0)) < FRED_CACHE_TTL:
        return FRED_CACHE[cache_key]
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
        FRED_CACHE[cache_key]              = recent
        FRED_CACHE_TIME_MAP[cache_key]     = now
        return recent
    except Exception as e:
        print(f"FRED error {series_id} (resolved: {resolved_id}): {e}")
        return None


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
        components["MFG_PMI"] = {**r, "title": "Industrial Production", "category": "growth",
                                 "display": f"{r['actual']:+.2f}" if r['actual'] is not None else "—"}

    cfnai_data = fetch_fred_series("CFNAI", 12)
    if cfnai_data:
        r = compute_macro_surprise(cfnai_data, higher_is_good=True, transform="level", scale=0.2)
        components["SVC_PMI"] = {**r, "title": "Economic Activity (CFNAI)", "category": "growth",
                                 "display": f"{r['actual']:+.2f}" if r['actual'] is not None else "—"}

    retail_data = fetch_fred_series("RSAFS", 12)
    if retail_data:
        r = compute_macro_surprise(retail_data, higher_is_good=True, transform="mom", scale=4000.0)
        if r["actual"] is not None:
            r["display"] = f"{r['actual']:+.0f}M"
        components["RETAIL"] = {**r, "title": "Retail Sales", "category": "growth",
                                "display": r.get("display", "—")}

    # ── INFLATION ─────────────────────────────────────────────────────────────
    cpi_data = fetch_fred_series("CPI", 24)
    if cpi_data:
        r = compute_macro_surprise(cpi_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["CPI"] = {**r, "title": "CPI YoY", "category": "inflation",
                              "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    pce_data = fetch_fred_series("PCE", 24)
    if pce_data:
        r = compute_macro_surprise(pce_data, higher_is_good=True, transform="yoy", scale=0.3)
        components["PCE"] = {**r, "title": "PCE YoY", "category": "inflation",
                              "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    # ── JOBS ──────────────────────────────────────────────────────────────────
    nfp_data = fetch_fred_series("NFP", 12)
    if nfp_data:
        r = compute_macro_surprise(nfp_data, higher_is_good=True, transform="mom", scale=80)
        if r["actual"] is not None:
            r["actual"]   = round(r["actual"])
            r["expected"] = round(r["expected"])
            r["surprise"] = round(r["surprise"])
            r["display"]  = f"{r['actual']:+.0f}K"
        components["JOBS"] = {**r, "title": "Non-Farm Payrolls", "category": "jobs",
                              "display": r.get("display", "—")}

    unemp_data = fetch_fred_series("UNEMP", 12)
    if unemp_data:
        r = compute_macro_surprise(unemp_data, higher_is_good=False, transform="level", scale=0.15)
        components["UNEMP"] = {**r, "title": "Unemployment Rate", "category": "jobs",
                                "display": f"{r['actual']}%" if r['actual'] is not None else "—"}

    claims_data = fetch_fred_series("CLAIMS", 12)
    if claims_data:
        r = compute_macro_surprise(claims_data, higher_is_good=False, transform="level", scale=15000)
        components["CLAIMS"] = {**r, "title": "Initial Claims", "category": "jobs",
                                 "display": f"{int(r['actual']):,}" if r['actual'] is not None else "—"}

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
    growth_scores    = [components[k]["score"] for k in ["GDP", "MFG_PMI", "SVC_PMI", "RETAIL"] if k in components]
    inflation_scores = [components[k]["score"] for k in ["CPI", "PCE"] if k in components]
    jobs_scores      = [components[k]["score"] for k in ["JOBS", "UNEMP", "CLAIMS"] if k in components]
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
    elif m == "HG":
        raw = growth_s2 * 0.50 + jobs_s * 0.25 + infl_avg * 0.10 - dgs2_s * 0.15
        reason = f"Growth: {growth_s2:+.1f}"

    # ── Soft Commodities / Agri ───────────────────────────────────────────
    elif m in ("ZC", "ZS", "ZW", "KC", "SB", "CT", "CC", "RC"):
        raw = (infl_avg * 0.30 + growth_s * 0.20 - dgs2_s * 0.20 + jobs_s * 0.10) / 0.80
        reason = f"Infl: {infl_avg:+.1f}, Growth: {growth_s:+.1f}"

    # ── Livestock ─────────────────────────────────────────────────────────
    elif m in ("LE", "HE", "GF"):
        raw = growth_s * 0.35 + jobs_s * 0.25 + infl_avg * 0.20 - dgs2_s * 0.20
        reason = f"Growth: {growth_s:+.1f}, Jobs: {jobs_s:+.1f}"

    # ── Platinum, Palladium ───────────────────────────────────────────────
    elif m in ("PL", "PA"):
        raw = growth_s2 * 0.40 + infl_avg * 0.20 + jobs_s * 0.20 - dgs2_s * 0.20
        reason = f"Growth: {growth_s2:+.1f}"

    # ── Crypto ────────────────────────────────────────────────────────────
    elif m in ("BTC", "ETH"):
        # Risk-on: growth + jobs bullish, rising rates slightly bearish
        raw = (growth_s * 0.30 + jobs_s * 0.20 - dgs2_s * 0.15 + infl_avg * 0.10) / 0.75
        reason = f"Growth: {growth_s:+.1f}, Rates: {-dgs2_s:+.1f}"

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
    # TIPS ETF: used as real-yield proxy in historical backtest
    # TIP modified duration ~7.5y → price change inversely tracks real yield changes
    "TIP":    "TIP",
    # Treasury yields — needed for yield curve signal in historical regime scoring
    "TNX":    "^TNX",   # 10-year Treasury yield (term spread numerator)
    "IRX":    "^IRX",   # 13-week T-bill (term spread denominator + rate path proxy)
}

RISK_REGIME_CACHE: dict = {"data": None, "time": 0}
RISK_REGIME_CACHE_TTL = 3600 * 2  # 2h



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

def _compute_intl_rates() -> dict:
    """Fetch central bank policy rates from FRED and compute trend signals."""
    now = time.time()
    if _INTL_RATES_CACHE["data"] and (now - _INTL_RATES_CACHE["time"]) < _INTL_RATES_TTL:
        return _INTL_RATES_CACHE["data"]

    from datetime import datetime as _dt
    result = {}
    # Max staleness: only include if data is within 18 months
    STALE_CUTOFF_DAYS = 548  # 18 months

    for cb, fred_id in _CB_RATE_SERIES.items():
        try:
            # Daily series (BOE, ECB) need more periods to capture 6m trend; monthly need fewer
            _DAILY_CBS = {"BOE", "ECB"}  # IUDSOIA and ECBDFR are daily series
            periods = 400 if cb in _DAILY_CBS else 36  # 400 days for daily, 36 months for monthly
            raw = fetch_fred_series(fred_id, periods)
            if not raw or len(raw) < 3:
                continue
            vals = [x["value"] for x in raw if x.get("value") is not None]
            dates = [x["date"]  for x in raw if x.get("value") is not None]
            if len(vals) < 3:
                continue

            # Staleness check: skip if last data point > 18 months old
            try:
                last_date = _dt.strptime(dates[-1], "%Y-%m-%d")
                days_old  = (datetime.utcnow() - last_date).days
                if days_old > STALE_CUTOFF_DAYS:
                    continue  # Too stale to be useful
            except Exception:
                pass

            current = vals[-1]
            # Daily series (BOE, ECB): 3m ≈ 65 obs; 6m ≈ 130 obs
            # Monthly series: 3m ≈ 3 obs; 6m ≈ 6 obs
            if cb in _DAILY_CBS:
                v_3m = vals[-65]  if len(vals) >= 65  else vals[0]
                v_6m = vals[-130] if len(vals) >= 130 else vals[0]
            else:
                v_3m = vals[-3] if len(vals) >= 3 else vals[0]
                v_6m = vals[-6] if len(vals) >= 6 else vals[0]

            trend_3m = round(current - v_3m, 3)
            trend_6m = round(current - v_6m, 3)
            bias = 1 if trend_6m > 0.1 else -1 if trend_6m < -0.1 else 0

            # Label logic:
            # - FEDFUNDS (US) is a reliable policy rate: t3=0 means genuinely on hold.
            #   Use t3-priority: if t3~0 but t6 shows prior trend, label as Paused.
            # - Interbank proxies (IR3TIB01*) reflect market expectations and are
            #   noisy month-to-month. For these, use t6 as the primary signal
            #   (smoother, more reflective of the actual CB cycle), but also
            #   require t3 agreement to confirm an active move vs inherited momentum.
            _IS_POLICY_RATE = fred_id in ("FEDFUNDS", "IUDSOIA", "ECBDFR")
            _t3_flat = abs(trend_3m) < 0.05
            _t6_flat = abs(trend_6m) < 0.1

            if _IS_POLICY_RATE:
                # Direct policy rate: t3=0 means the CB hasn't moved. Paused wins.
                if _t3_flat:
                    if abs(trend_6m) > 0.3:
                        _label = "Paused"  # On hold after a hiking/cutting cycle
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
                # Interbank proxy: use t6 as primary, t3 as confirmation
                # Both must agree to label an active move; otherwise use t6 direction
                # but downgrade the label intensity
                _same_dir = (trend_3m * trend_6m) > 0  # same sign
                if trend_6m > 0.5 and trend_3m > 0.05:
                    _label = "Hiking"
                elif trend_6m > 0.1 and trend_3m > -0.05:
                    _label = "Tightening"
                elif trend_6m < -0.5 and trend_3m < -0.05:
                    _label = "Cutting"
                elif trend_6m < -0.1 and trend_3m < 0.05:
                    _label = "Easing"
                elif _t6_flat:
                    _label = "Flat"
                else:
                    # t6 shows a trend but t3 has reversed → Paused/fading
                    _label = "Paused" if abs(trend_6m) > 0.3 else "Flat"

            result[cb] = {
                "rate":     round(current, 3),
                "trend_3m": trend_3m,
                "trend_6m": trend_6m,
                "bias":     bias,
                "data_date": dates[-1],
                "label":    _label,
            }
        except Exception:
            continue

    _INTL_RATES_CACHE["data"] = result
    _INTL_RATES_CACHE["time"] = now
    return result

def compute_risk_regime() -> dict:
    now = time.time()
    if RISK_REGIME_CACHE["data"] and (now - RISK_REGIME_CACHE["time"]) < RISK_REGIME_CACHE_TTL:
        return RISK_REGIME_CACHE["data"]

    returns: dict = {}
    levels:  dict = {}
    for name, ticker in RISK_ASSETS.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="3mo", interval="1wk")
            if not hist.empty and len(hist) >= 4:
                close = hist["Close"].values.astype(float)
                ret_1w = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else 0
                ret_1m = (close[-1] / close[-4] - 1) * 100 if len(close) >= 4 else 0
                ret_3m = (close[-1] / close[0]  - 1) * 100 if len(close) >= 3 else 0
                returns[name] = {"return_1w": round(ret_1w, 2), "return_1m": round(ret_1m, 2), "return_3m": round(ret_3m, 2)}
                levels[name]  = round(float(close[-1]), 4)
        except Exception as _re:
            returns[name] = {"1w": 0, "return_1m": 0, "3m": 0}

    regime_signals: dict = {}  # structured dict: key -> {signal, value, label}
    regime_score = 0

    # ── SPX / RUT trend ────────────────────────────────────────────────────
    spx_1m = returns.get("SPX", {}).get("return_1m", 0)
    rut_1m = returns.get("RUT", {}).get("return_1m", 0)
    ndx_1m = returns.get("NDX", {}).get("return_1m", 0)
    spx_sig = 1 if spx_1m > 1.0 else -1 if spx_1m < -1.0 else 0
    regime_score += spx_sig
    if rut_1m > 1.0:  regime_score += 0.5
    elif rut_1m < -1.0: regime_score -= 0.5
    # Equity composite signal normalised -2..+2
    eq_raw = spx_sig + (0.5 if rut_1m > 1 else -0.5 if rut_1m < -1 else 0)
    regime_signals["SPX"] = {
        "signal": round(eq_raw / 2, 2),
        "value": f"{spx_1m:+.1f}%",
        "label": "Bullish" if spx_1m > 1.0 else "Bearish" if spx_1m < -1.0 else "Neutral",
    }
    regime_signals["RTY"] = {
        "signal": round((0.5 if rut_1m > 1 else -0.5 if rut_1m < -1 else 0), 2),
        "value": f"{rut_1m:+.1f}%",
        "label": "Bullish" if rut_1m > 1.0 else "Bearish" if rut_1m < -1.0 else "Neutral",
    }

    # ── VIX level + term structure ─────────────────────────────────────────
    # Thresholds re-calibrated: historical median VIX ~17, so <17 should NOT equal risk-on.
    # New neutral band: 17–21.  Extremes: <13 = euphoric (+2), >27 = stressed (−2).
    # Old: <14=+2, <17=+1, >23=−1, >30=−2
    # New: <13=+2, <17=+1, 17–21=0, >21=−1, >27=−2
    vix_level   = levels.get("VIX",  20.0)
    vix3m_level = levels.get("VIX3M", 22.0)
    vix_level_s  = (2 if vix_level < 13 else 1 if vix_level < 17 else
                   -1 if vix_level > 21 else -2 if vix_level > 27 else 0)
    vix_ts_signal = 0
    if vix3m_level > 0 and vix_level > 0:
        ts_ratio = vix3m_level / vix_level
        if ts_ratio > 1.05:    vix_ts_signal = 1  # contango = calm
        elif ts_ratio < 0.95:  vix_ts_signal = -1 # inversion = stress
    regime_score += vix_level_s * 0.6 + vix_ts_signal * 0.3
    ts_label = ("Contango" if vix_ts_signal > 0 else "Inverted" if vix_ts_signal < 0 else "Flat")
    vix_sig_norm = round((vix_level_s * 0.6 + vix_ts_signal * 0.3) / 2, 2)
    regime_signals["VIX"] = {
        "signal": vix_sig_norm,
        "value": f"{vix_level:.1f}",
        "label": f"{ts_label} — {'Low' if vix_level < 16 else 'Elevated' if vix_level > 25 else 'Moderate'} vol",
    }

    # ── Credit spreads (HYG/LQD price signal + FRED BAML OAS) ───────────────
    hyg_1m = returns.get("HYG", {}).get("return_1m", 0)
    lqd_1m = returns.get("LQD", {}).get("return_1m", 0)
    spread_sig = hyg_1m - lqd_1m  # positive = HY outperforming = risk-on
    if spread_sig > 1.5:   regime_score += 1
    elif spread_sig < -1.5: regime_score -= 1
    credit_sig_norm = round(min(1.0, max(-1.0, spread_sig / 3.0)), 2)
    credit_trend = "Tightening" if spread_sig > 0.3 else "Widening" if spread_sig < -0.3 else "Neutral"

    # Fetch actual OAS levels from FRED (BAML indices) — daily, in basis points
    hy_oas_bps = None
    ig_oas_bps = None
    hy_oas_score = 0.0
    hy_delta_4w = None
    hy_delta_3m = None
    hy_ig_ratio = None
    try:
        hy_data = fetch_fred_series("HYOAS", 70)   # ~70 business days = 14 weeks
        if hy_data and len(hy_data) >= 4:
            vals = [row["value"] for row in hy_data if row.get("value") is not None]
            if vals:
                # BAMLH0A0HYM2 is in % (e.g. 2.85 = 285 bps) — convert to bps
                hy_oas_bps = round(vals[-1] * 100, 0)
                # Score: tight=bullish (+1), wide=bearish (-1)
                # Context: <250=tight, 250-350=normal, 350-500=elevated, >500=stress
                hy_oas_score = (1.0 if hy_oas_bps < 250 else
                                0.5 if hy_oas_bps < 300 else
                               -0.5 if hy_oas_bps < 450 else
                               -1.0 if hy_oas_bps >= 450 else 0.0)
                if len(vals) >= 20:
                    hy_delta_4w = round((vals[-1] - vals[-20]) * 100, 0)
                if len(vals) >= 65:
                    hy_delta_3m = round((vals[-1] - vals[-65]) * 100, 0)
                # Boost/dampen regime score based on OAS level
                if hy_oas_bps < 250:   regime_score += 0.5
                elif hy_oas_bps > 500: regime_score -= 0.5
    except Exception: pass
    try:
        ig_data = fetch_fred_series("IGOAS", 70)
        if ig_data and len(ig_data) >= 2:
            ig_vals = [row["value"] for row in ig_data if row.get("value") is not None]
            if ig_vals:
                ig_oas_bps = round(ig_vals[-1] * 100, 0)  # % -> bps
                if hy_oas_bps is not None and ig_oas_bps > 0:
                    hy_ig_ratio = round(hy_oas_bps / ig_oas_bps, 2)
    except Exception: pass

    # Blend HYG/LQD price signal with OAS level signal
    if hy_oas_bps is not None:
        credit_sig_norm = round(credit_sig_norm * 0.5 + hy_oas_score * 0.5, 2)

    regime_signals["Credit"] = {
        "signal": credit_sig_norm,
        "value": f"{spread_sig:+.1f}%",
        "label": f"{credit_trend} (HYG {hyg_1m:+.1f}% / LQD {lqd_1m:+.1f}%)",
    }

    # ── Gold vs equities ──────────────────────────────────────────────────
    gld_1m = returns.get("GLD", {}).get("return_1m", 0)
    copper_1m = returns.get("COPPER", {}).get("return_1m", 0)
    if gld_1m > 3.0 and spx_1m < 0:
        regime_score -= 0.5  # gold surging, equities down = risk-off
    elif gld_1m < -1.0 and spx_1m > 1.0:
        regime_score += 0.5  # gold weak, equities up = risk-on
    gold_sig_norm = round(min(1.0, max(-1.0, -gld_1m / 8.0)), 2)  # inverse: gold up = risk-off
    regime_signals["Gold"] = {
        "signal": gold_sig_norm,
        "value": f"{gld_1m:+.1f}%",
        "label": "Safe-haven bid" if gld_1m > 2 else "Risk-on" if gld_1m < -1 else "Neutral",
    }
    copper_sig_norm = round(min(1.0, max(-1.0, copper_1m / 8.0)), 2)
    regime_signals["Copper"] = {
        "signal": copper_sig_norm,
        "value": f"{copper_1m:+.1f}%",
        "label": "Industrial demand" if copper_1m > 2 else "Demand weakness" if copper_1m < -2 else "Neutral",
    }

    # ── DXY signal ────────────────────────────────────────────────────────
    dxy_1m = returns.get("DXY", {}).get("return_1m", 0)
    dxy_sig_norm = round(min(1.0, max(-1.0, -dxy_1m / 4.0)), 2)  # DXY up = risk-off for risk assets
    regime_signals["DXY"] = {
        "signal": dxy_sig_norm,
        "value": f"{dxy_1m:+.1f}%",
        "label": "Strengthening" if dxy_1m > 0.5 else "Weakening" if dxy_1m < -0.5 else "Flat",
    }

    # ── USD/JPY ───────────────────────────────────────────────────────────
    # USDJPY up = JPY weak = risk-on signal.
    # Previously computed but NOT added to regime_score (was purely decorative).
    # Now wired in at 0.30 weight: adds a meaningful carry/safe-haven dimension
    # orthogonal to the VIX and credit signals already in the composite.
    usdjpy_1m = returns.get("USDJPY", {}).get("return_1m", 0)
    usdjpy_sig = round(min(1.0, max(-1.0, usdjpy_1m / 5.0)), 2)
    regime_score += usdjpy_sig * 0.30
    regime_signals["USD/JPY"] = {
        "signal": usdjpy_sig,
        "value": f"{usdjpy_1m:+.1f}%",
        "label": "JPY weakening (risk-on)" if usdjpy_1m > 1 else "JPY strengthening (risk-off)" if usdjpy_1m < -1 else "Neutral",
    }

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
    vix_ts      = time.time()  # timestamp when VIX was fetched

    # ── MACRO DASHBOARD: FRED enrichment ─────────────────────────────────
    macro_dashboard = {}
    rate_signal = {}
    rate_label = ""
    try:
        # Yield curve: T10Y2Y + T10Y3M (daily series — 130 days covers 6m)
        yc_data   = fetch_fred_series("YLDCRV",  130)
        yc3m_data = fetch_fred_series("T10Y3M",  130)
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
            macro_dashboard["yield_curve"] = {
                "t10y2y":         round(t10y2y, 3),
                "t10y3m":         round(t10y3m, 3) if t10y3m is not None else None,
                "curve_regime":   curve_regime,
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
    except Exception: pass

    try:
        # DGS10 (10Y yield)
        dgs10_data = fetch_fred_series("DGS10", 16)  # may not be in FRED_SERIES, use raw id
        if dgs10_data and len(dgs10_data) >= 2:
            dgs10_now = dgs10_data[-1]["value"]
            dgs10_3m  = dgs10_data[0]["value"] if len(dgs10_data) >= 12 else dgs10_data[0]["value"]
            macro_dashboard["dgs10"] = {
                "level":  round(dgs10_now, 3),
                "chg_3m": round(dgs10_now - dgs10_3m, 3),
            }
    except Exception: pass

    try:
        # Real yield: DFII10
        ry_data = fetch_fred_series("DFII10", 6)
        if ry_data and len(ry_data) >= 1:
            ry_val = ry_data[-1]["value"]
            ry_regime = "Restrictive" if ry_val > 2.0 else "Elevated" if ry_val > 1.0 else "Neutral" if ry_val > 0 else "Accommodative"
            macro_dashboard["real_yield"] = {"value": round(ry_val, 3), "regime": ry_regime}
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
        # Fed balance sheet: WALCL (trillions)
        walcl_data = fetch_fred_series("WALCL", 60)
        if walcl_data and len(walcl_data) >= 4:
            bs_vals = [x["value"] / 1e6 for x in walcl_data if x.get("value") is not None]
            bs_now  = bs_vals[-1]
            # 3m ≈ 13 weekly observations; 6m ≈ 26
            bs_3m   = bs_vals[-13] if len(bs_vals) >= 13 else bs_vals[0]
            bs_6m   = bs_vals[-26] if len(bs_vals) >= 26 else bs_vals[0]
            bs_trend = "Expanding" if bs_now > bs_3m * 1.005 else "QT (Contracting)" if bs_now < bs_3m * 0.995 else "Flat / Stable"
            chg_3m_pct = round((bs_now / bs_3m - 1.0) * 100, 2) if bs_3m else 0
            chg_6m_pct = round((bs_now / bs_6m - 1.0) * 100, 2) if bs_6m else 0
            macro_dashboard["fed_balance"] = {
                "level":      round(bs_now, 2),
                "trend":      bs_trend,
                "chg_3m":     round(bs_now - bs_3m, 2),
                "chg_3m_pct": chg_3m_pct,
                "chg_6m_pct": chg_6m_pct,
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
            "hy_ig_ratio": hy_ig_ratio,
        }
    except Exception: pass

    try:
        # Macro composites: equity + bond + commodity etc.
        eq_comp = round(5 + eq_raw * 1.5, 1)
        macro_dashboard["macro_composites"] = {
            "equity":     {"composite_10": max(0, min(10, eq_comp))},
            "bond":       {"composite_10": max(0, min(10, round(5 - eq_raw * 1.2, 1)))},
            "gold":       {"composite_10": max(0, min(10, round(5 - regime_score * 0.8, 1)))},
            "commodity":  {"composite_10": max(0, min(10, round(5 + regime_score * 0.6, 1)))},
            "fx_foreign": {"composite_10": max(0, min(10, round(5 - dxy_sig_norm * 3, 1)))},
            "crypto":     {"composite_10": max(0, min(10, round(5 + regime_score * 1.0, 1)))},
            "credit":     round(credit_sig_norm, 2),
            "yield_curve": round((macro_dashboard.get("yield_curve", {}).get("t10y2y", 0) or 0) / 2, 2),
        }
    except Exception: pass

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
            rate_signal = {
                "effr":       round(effr_val, 3),
                "effr_spot":  round(_effr_spot, 3),
                "effr_12m":   round(_effr_12m, 3),
                "effr_18m":   round(_effr_18m, 3),
                "cuts_12m":   cuts_12m,
                "cuts_18m":   cuts_18m,
                "rate_norm":  rate_norm_val,
                "source":     "fff",  # fed funds futures
            }
            print(f"[rate_signal] FFF: spot={_effr_spot}% 12m={_effr_12m}% "
                  f"cuts_12m={cuts_12m} cuts_18m={cuts_18m} label={rate_label}")
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

    result = {
        "score":         regime_score,
        "regime":        regime_name,
        "regime_label":  regime_label,
        "raw_score":     regime_score,
        "vix_level":     vix_level,
        "vix3m_level":   vix3m_level,
        "vix_ts":        vix_ts,
        "signals":       regime_signals,
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
    _us_t6   = us_ir.get("trend_6m", 0.0)
    _us_t3   = us_ir.get("trend_3m", 0.0)
    _us_bias = us_ir.get("bias", 0)
    _us_raw  = _us_t6 * 0.65 + _us_t3 * 0.35 + _us_bias * 0.25
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
            _boe_t6 = boe.get("trend_6m", 0.0)
            _boe_t3 = boe.get("trend_3m", 0.0)
            _boe_b  = boe.get("bias", 0)
            _boe_adj = max(-2.0, min(2.0, (_boe_t6 * 0.65 + _boe_t3 * 0.35 + _boe_b * 0.25) / 0.3))
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
        elif raw_score > 0.5:
            label = "Risk-on (Equity Bullish)"
        elif raw_score < -0.5:
            label = "Risk-off (Defensive)"
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
    # ════════════════════════════════════════════════════════════════════
    elif m == "SI":
        score_raw = raw_score * 0.45 + us_rate_adj * (-0.35) + 5.0
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
            t6   = cb_data.get("trend_6m", 0.0)
            t3   = cb_data.get("trend_3m", 0.0)
            bias = cb_data.get("bias", 0)
            raw_cb   = t6 * 0.65 + t3 * 0.35
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
        ("Z",   "FTSE 100 futures (UK equity index, ICE Europe; ~70% revenues international)"),
        # Metals
        ("GC",  "Gold futures"),
        ("SI",  "Silver futures"),
        ("HG",  "Copper futures (global growth proxy)"),
        ("PL",  "Platinum futures"),
        ("PA",  "Palladium futures"),
        # Energy
        ("CL",  "Crude Oil WTI futures"),
        ("B",   "Brent Crude Oil futures (ICE Europe)"),
        ("NG",  "Natural Gas futures (Henry Hub)"),
        ("RB",  "RBOB Gasoline futures"),
        ("HO",  "Heating Oil futures"),
        # Bonds
        ("ZB",  "US 30Y T-Bond futures"),
        ("ZN",  "US 10Y T-Note futures"),
        ("ZF",  "US 5Y T-Note futures"),
        ("ZT",  "US 2Y T-Note futures"),
        ("R",   "UK Long Gilt futures (ICE Europe; BoE policy sensitive)"),
        # FX
        ("6E",  "EUR/USD futures"),
        ("6J",  "Japanese Yen futures (inverse of USD/JPY)"),
        ("6B",  "GBP/USD futures (British Pound)"),
        ("6A",  "AUD/USD futures (Australian Dollar; risk/commodity proxy)"),
        ("6C",  "CAD/USD futures (Canadian Dollar; oil-linked)"),
        ("6N",  "NZD/USD futures (New Zealand Dollar)"),
        ("6S",  "CHF/USD futures (Swiss Franc; safe-haven)"),
        ("DX",  "US Dollar Index futures"),
        # Agriculturals
        ("ZS",  "Soybean futures"),
        ("ZC",  "Corn futures"),
        ("ZW",  "Wheat futures"),
        ("CC",  "Cocoa futures (ICE)"),
        ("KC",  "Coffee futures"),
        ("SB",  "Sugar No.11 futures"),
        ("CT",  "Cotton No.2 futures"),
        # Livestock
        ("LE",  "Live Cattle futures"),
        ("HE",  "Lean Hogs futures"),
        ("GF",  "Feeder Cattle futures"),
        # Crypto
        ("BTC", "Bitcoin (spot/perpetual)"),
        ("ETH", "Ethereum (spot/perpetual)"),
    ]

    asset_str = "\n".join(f"{aid}: {aname}" for aid, aname in ASSET_LIST)
    headline_str = "\n".join(headlines)

    prompt = (
        "You are a professional markets analyst. Below are the latest high/medium-impact financial "
        "news headlines from the last 48 hours.\n\n"
        f"HEADLINES:\n{headline_str}\n\n"
        "For each of the following futures/FX instruments, produce TWO things:\n"
        "1. A SHORT 1-2 sentence analyst comment connecting the current news backdrop to that "
        "instrument's likely price action or sentiment. Be specific: geopolitical escalation → "
        "gold safe-haven bid, crude supply risk, yen flows; rate decisions → bond/FX/equity impact. "
        "Infer the effect even if the asset isn't mentioned. Use direct market language: "
        "'bid', 'offered', 'headwind', 'tailwind', 'under pressure', 'supported'.\n"
        "2. A sentiment SCORE from -1.0 (most bearish) to +1.0 (most bullish), reflecting how the "
        "current news backdrop affects that instrument. 0.0 = neutral/no clear news effect.\n\n"
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
                "max_tokens": 4000,
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

    return {
        "narratives":        narratives_text,    # {assetId: "text string"} — frontend display
        "narrative_scores":  narratives_scores,  # {assetId: 0-10 float} — regime blending
        "news_items":        news_items[:20],
        "global_narrative":  None,
        "price_context":     {},
        "updated_at":        news_ts,
        "ff_event_count":    len(news_items),
    }


# ============================================================
# WEIGHTED SCORE + BIAS LABEL
# ============================================================

WEIGHTS = {
    "cot":      0.30,
    "seasonal": 0.15,
    "momentum": 0.15,
    "macro":    0.15,
    "regime":   0.05,  # Climate score — blends mechanical risk/rate signal + news sentiment
    "relval":   0.15,
    "pcr":      0.00,  # Active for equity markets only (see compute_weighted_bias)
}

# Weights when PCR is active at full tier-1 weight (equities + metals)
WEIGHTS_EQUITY = {
    "cot":      0.25,
    "seasonal": 0.15,
    "momentum": 0.15,
    "macro":    0.15,
    "regime":   0.05,
    "relval":   0.15,
    "pcr":      0.10,  # Tier-1: 10% — deep markets, strong backtest edge
}

# Tier-2: CL oil — 5% weight (moderate liquidity, bull-only signal)
WEIGHTS_PCR_TIER2 = {
    "cot":      0.28,
    "seasonal": 0.15,
    "momentum": 0.15,
    "macro":    0.15,
    "regime":   0.05,
    "relval":   0.17,
    "pcr":      0.05,
}

# Tier-3: BTC/ETH — 3% weight (unique market structure, Deribit depth good but shorter history)
WEIGHTS_PCR_TIER3 = {
    "cot":      0.29,
    "seasonal": 0.15,
    "momentum": 0.15,
    "macro":    0.15,
    "regime":   0.05,
    "relval":   0.18,
    "pcr":      0.03,
}

# ICE Europe thin-data markets (Z=73w, R=57w): COT weight reduced to 12%.
# Research threshold: 156 weeks minimum for reliable Briese index (TradingView, Williams).
# Below threshold, COT is directional only — weight reallocated to momentum + relval.
# Applied to: Z (FTSE 100), R (Long Gilt) — both EUFINCOTHist TFF format, limited history.
WEIGHTS_ICE_THIN = {
    "cot":      0.12,  # Halved vs standard: thin data makes Briese percentile unreliable
    "seasonal": 0.18,  # Slightly boosted: price-derived curves are full history
    "momentum": 0.22,  # Boosted: reliable, market-confirmed signal
    "macro":    0.20,  # Boosted: fundamental signal not data-limited
    "regime":   0.08,  # Slightly boosted: risk regime is critical for equities/bonds
    "relval":   0.20,  # Boosted: relative value vs analogues is robust
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

def compute_weighted_bias(scores: dict, market_id: str = "",
                           cot_detail: dict = None) -> dict:
    """
    Compute weighted composite score (0-10) across all factors.

    Confluence Bonus:
    ─────────────────
    A small bonus (+/-0.35 on the 0-10 scale) is added when the COT
    STORY is genuinely in motion AND the confirming factors agree.

    The COT story must be active — not just a high static snapshot.
    "In motion" means one or more of:
      - divergence signal firing (managers diverging from price)
      - normalise_signal firing (riding an extreme unwind)
      - convergence_signal firing (all three groups aligned)
      - comm_momentum_signal firing (commercials accelerating)
      - flatten_signal firing (managers flat while price at extreme)

    AND at least 2 of the 3 confirmation factors (macro, momentum,
    seasonal) must agree with the COT direction (score >= 6.2 bull
    or <= 3.8 bear).

    Philosophy: COT finds the early thesis with good R/R. Macro,
    momentum, and seasonality confirm that the environment will
    drive follow-through. When ALL of these align around an active
    COT story, it is genuinely a high-conviction setup.
    """
    # Select weight map based on market type and data quality
    # ICE thin-data markets (Z, R): COT history < 156w threshold — down-weight COT
    _ICE_THIN_MARKETS = {"Z", "R"}  # EUFINCOTHist only from Dec 2024 / Mar 2025
    if market_id in _ICE_THIN_MARKETS:
        w_map = WEIGHTS_ICE_THIN
    elif "pcr" in scores and market_id in PCR_ALL_SYMBOLS:
        pcr_tier = PCR_TIERS.get(market_id, {}).get("tier", 0)
        if pcr_tier == 1:
            w_map = WEIGHTS_EQUITY
        elif pcr_tier == 2:
            w_map = WEIGHTS_PCR_TIER2
        elif pcr_tier == 3:
            w_map = WEIGHTS_PCR_TIER3
        else:
            w_map = WEIGHTS
    else:
        w_map = WEIGHTS
    total_w = sum(w_map[k] for k in w_map if k in scores)
    if total_w == 0:
        return {"weighted": 5.0, "bias": "Neutral", "color": "#94a3b8", "confluence_bonus": 0.0}
    weighted = sum(w_map[k] * scores.get(k, 5.0) for k in w_map if k in scores) / total_w

    # ── Confluence Bonus ────────────────────────────────────────────────────
    # Only applies when the COT *story* is active (not just a static level).
    # Bonus: +/-0.35 when COT story + 2/3 confirming factors all agree.
    # Bonus: +/-0.20 when COT story + 1/3 confirming factors agree (weaker).
    confluence_bonus  = 0.0
    confluence_reason = None

    if cot_detail and isinstance(cot_detail, dict):
        cot_score = cot_detail.get("score", 5.0) or 5.0

        # Determine COT direction
        cot_bull = cot_score >= 6.5
        cot_bear = cot_score <= 3.5

        # Check if the COT STORY is genuinely in motion (not just a static extreme)
        story_active = any([
            cot_detail.get("divergence"),         # Layer 2: price/manager divergence
            cot_detail.get("normalise_signal"),   # Layer 8c: riding the unwind
            cot_detail.get("convergence_signal"), # Layer 8b: all three groups aligned
            cot_detail.get("comm_momentum_signal") and  # Layer 6: commercials accelerating
                (cot_detail["comm_momentum_signal"].get("type") == ("bull" if cot_bull else "bear")),
            cot_detail.get("flatten_signal"),     # Layer 8a: manager flattening
            cot_detail.get("exhaustion"),         # Layer 3: manager exhaustion at extreme
        ])

        if story_active and (cot_bull or cot_bear):
            # Count confirming factors (macro, momentum, seasonal)
            # "Confirming" = score >= 6.2 (bull) or <= 3.8 (bear), same direction as COT
            macro_s    = scores.get("macro",    5.0)
            momentum_s = scores.get("momentum", 5.0)
            seasonal_s = scores.get("seasonal", 5.0)

            if cot_bull:
                confirmers = sum([
                    macro_s    >= 6.2,
                    momentum_s >= 6.2,
                    seasonal_s >= 6.2,
                ])
                if confirmers >= 2:
                    confluence_bonus  = +0.35
                    confluence_reason = f"Confluence: COT story active + {confirmers}/3 factors confirming bull"
                elif confirmers == 1:
                    confluence_bonus  = +0.20
                    confluence_reason = f"Confluence: COT story active + 1/3 factors confirming bull"
            elif cot_bear:
                confirmers = sum([
                    macro_s    <= 3.8,
                    momentum_s <= 3.8,
                    seasonal_s <= 3.8,
                ])
                if confirmers >= 2:
                    confluence_bonus  = -0.35
                    confluence_reason = f"Confluence: COT story active + {confirmers}/3 factors confirming bear"
                elif confirmers == 1:
                    confluence_bonus  = -0.20
                    confluence_reason = f"Confluence: COT story active + 1/3 factors confirming bear"

    weighted = round(max(0.0, min(10.0, weighted + confluence_bonus)), 2)

    # Bias labels for 0-10 scale (5.0 = neutral)
    bias = "Neutral"; color = "#94a3b8"
    if   weighted >= 8.0:  bias = "Very Bullish";    color = "#22c55e"
    elif weighted >= 7.0:  bias = "Bullish";          color = "#4ade80"
    elif weighted >= 6.2:  bias = "Mildly Bullish";   color = "#86efac"
    elif weighted >= 5.5:  bias = "Lean Bullish";     color = "#a7f3d0"
    elif weighted >= 4.5:  bias = "Neutral";          color = "#94a3b8"
    elif weighted >= 3.8:  bias = "Lean Bearish";     color = "#fde68a"
    elif weighted >= 3.0:  bias = "Mildly Bearish";   color = "#fca5a5"
    elif weighted >= 2.0:  bias = "Bearish";          color = "#f87171"
    else:                  bias = "Very Bearish";     color = "#ef4444"

    return {
        "weighted":          weighted,
        "bias":              bias,
        "color":             color,
        "confluence_bonus":  confluence_bonus,
        "confluence_reason": confluence_reason,
    }

# ============================================================
# MAIN API ENDPOINTS
# ============================================================

ALL_DATA_CACHE = {"data": None, "time": 0}
ALL_DATA_TTL = 3600  # 60 min — data sources (COT, macro, prices) change at most hourly

@app.get("/api/scores")
async def get_all_scores(force: bool = False):
    now = time.time()
    if force:
        ALL_DATA_CACHE["data"] = None
        FF_CACHE["data"] = None
        FF_MACRO_CACHE["data"] = None
    if not force and ALL_DATA_CACHE["data"] and (now - ALL_DATA_CACHE["time"]) < ALL_DATA_TTL:
        return _SafeJSONResponse(ALL_DATA_CACHE["data"])

    # Run all sync blocking data-fetch functions in thread executors
    # so the async event loop (and /api/health) remain responsive
    _loop = asyncio.get_event_loop()
    macro, regime, ff_macro = await asyncio.gather(
        _loop.run_in_executor(_APP_EXECUTOR, compute_macro_all),
        _loop.run_in_executor(_APP_EXECUTOR, compute_risk_regime),
        _loop.run_in_executor(_APP_EXECUTOR, compute_all_ff_macro),
    )
    # News context — pull from cache if warm; trigger background fetch if cold.
    # compute_news_context() runs the Sonar call in a thread to avoid blocking.
    _news_now = time.time()
    _news_cold = not NEWS_CACHE["data"] or (_news_now - NEWS_CACHE["time"]) >= NEWS_CACHE_TTL
    _narr_cold = not NARR_CACHE["data"] or (_news_now - NARR_CACHE["time"]) >= NARR_CACHE_TTL
    if _news_cold or _narr_cold:
        # Fire-and-forget: populate caches in background without blocking scores
        import asyncio as _anews
        _loop_news = _anews.get_event_loop()
        _loop_news.run_in_executor(_APP_EXECUTOR, compute_news_context)
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
                return (z_blend * sig + mu).round(0).astype(int)

            # Align indices for rolling stats (use primary df aligned to merged dates)
            p_aligned = p[p["date"].isin(merged["date"])].reset_index(drop=True)
            merged = merged.reset_index(drop=True)

            merged["comm_net_blended"]  = zscore_to_contracts(merged["comm_z_blended"],  p_aligned["comm_net"].astype(float))
            merged["lspec_net_blended"] = zscore_to_contracts(merged["lspec_z_blended"], p_aligned["lspec_net"].astype(float))
            merged["sspec_net_blended"] = zscore_to_contracts(merged["sspec_z_blended"], p_aligned["sspec_net"].astype(float))

            # ── Write blended values back into primary df ───────────────────
            out = primary_df.copy()
            out["date"] = pd.to_datetime(out["date"])
            for _, row in merged.iterrows():
                mask = out["date"] == row["date"]
                if mask.any():
                    out.loc[mask, "comm_net"]  = row["comm_net_blended"]
                    out.loc[mask, "lspec_net"] = row["lspec_net_blended"]
                    out.loc[mask, "sspec_net"] = row["sspec_net_blended"]

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
            cot_data = compute_cross_cot_score(
                mid,
                market["base_leg"],
                market["quote_leg"],
                cot_df_cache,
            )
        elif market.get("crypto_cot_mode"):
            cot_df = cot_df_cache.get(mid)
            cot_df = _merge_price_into_cot(cot_df, market["yf"], mid)
            cot_data = compute_crypto_cot_score(cot_df, market_id=mid)
        else:
            cot_df = cot_df_cache.get(mid)
            cot_df = _merge_price_into_cot(cot_df, market["yf"], mid)
            cot_data = compute_cot_score(cot_df, market_id=mid)
        # ────────────────────────────────────────────────────────────────────────

        seasonal_data = score_seasonality(mid)
        momentum_data = score_momentum(market["yf"])
        macro_data    = get_macro_score_for_market(mid, macro, ff_macro=ff_macro)
        _news_sent    = news_ctx.get("narrative_scores", {}).get(mid)
        regime_data   = get_regime_score_for_market(mid, regime, news_sentiment=_news_sent)
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

        bias = compute_weighted_bias(scores, market_id=mid, cot_detail=cot_data)

        scores_out = {
            "cot":      {"score": cot_data["score"],      "label": cot_data["label"],      "detail": cot_data.get("detail", cot_data)},
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

        # Determine the actual weight map used for this market (mirrors compute_weighted_bias routing)
        # This is exposed per-market so the frontend can render the correct weight mini-bars
        _ICE_THIN_MKTS = {"Z", "R"}
        if mid in _ICE_THIN_MKTS:
            mkt_weights = WEIGHTS_ICE_THIN
        elif mid in PCR_ALL_SYMBOLS:
            _tier = PCR_TIERS.get(mid, {}).get("tier", 0)
            if _tier == 1:
                mkt_weights = WEIGHTS_EQUITY
            elif _tier == 2:
                mkt_weights = WEIGHTS_PCR_TIER2
            elif _tier == 3:
                mkt_weights = WEIGHTS_PCR_TIER3
            else:
                mkt_weights = WEIGHTS
        else:
            mkt_weights = WEIGHTS
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
            "scores":            scores_out,
            "weights":           active_weights,  # Per-market weight map (varies by data quality + PCR tier)
        })
      return _results
    # end _compute_all_market_scores

    # Run the loop in a thread — it contains sync yfinance calls (momentum, relval, price merge)
    _loop = asyncio.get_event_loop()
    results = await _loop.run_in_executor(_APP_EXECUTOR, _compute_all_market_scores)

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
            factor_scores = {
                k: mkt["scores"][k]["score"]
                for k in mkt["scores"] if "score" in mkt["scores"][k]
            }
            factor_scores["regime"] = new_regime_score
            cot_detail_for_mid = mkt["scores"].get("cot", {}).get("detail", {})
            new_bias = compute_weighted_bias(factor_scores, market_id=mid, cot_detail=cot_detail_for_mid)

            # Update the result in place
            mkt["scores"]["regime"]["score"]  = new_regime_score
            mkt["scores"]["regime"]["dx_tilt"] = round(raw_tilt, 3)
            mkt["scores"]["regime"]["dx_tilt_source"] = f"DX {dx_score:.1f}/10 → {'bull' if dx_deviation > 0 else 'bear'} dollar feedback"
            mkt["weighted_score"]  = new_bias["weighted"]
            mkt["bias"]            = new_bias["bias"]
            mkt["color"]           = new_bias["color"]
            mkt["confluence_bonus"]= new_bias.get("confluence_bonus", 0.0)

    results.sort(key=lambda x: x["weighted_score"], reverse=True)

    # Strip nulls from cot.detail for every market — saves ~15KB from the payload
    for mkt in results:
        cot_detail = mkt.get("scores", {}).get("cot", {}).get("detail")
        if isinstance(cot_detail, dict):
            mkt["scores"]["cot"]["detail"] = {k: v for k, v in cot_detail.items() if v is not None}

    output = {
        "updated_at":    datetime.utcnow().isoformat() + "Z",
        "regime":        regime,
        "macro_all":     macro,
        "ff_macro":      ff_macro,  # per-currency FF economy scores
        "markets":       results,
        "weights":           WEIGHTS,
        "weights_equity":    WEIGHTS_EQUITY,
        "weights_ice_thin":  WEIGHTS_ICE_THIN,  # Applied to Z (FTSE100) and R (Long Gilt) — thin COT history
        # news_context intentionally excluded — frontend fetches /api/news-context separately
    }
    # Always store with full TTL — narratives have their own endpoint and cache
    ALL_DATA_CACHE["data"] = output
    ALL_DATA_CACHE["time"] = now
    return _SafeJSONResponse(output)

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

# ============================================================
# SEASONALITY ENDPOINT — serves pre-computed curves for all 21 markets
# ============================================================
import os as _os

_SEASONALITY_PATH = _os.path.join(_os.path.dirname(__file__), "seasonality_all21.json")
_SEASONALITY_CACHE = {"data": None}

# Cache for current-year actual price returns (refreshed every 30 min)
_CY_ACTUAL_CACHE: dict = {}
_CY_ACTUAL_TIME: dict = {}
_CY_ACTUAL_TTL = 1800  # 30 min

def _get_current_year_actual(market_id: str) -> list:
    """
    Fetch YTD price data for the given market and return cumulative % return
    from trading day 1 of the current year, as [[td, pct], ...] pairs.
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
        for i, (idx, price) in enumerate(closes.items()):
            # Use same DOY/365*252 formula as the seasonality curve build
            # so the cy_actual TDs align with the historical curve X-axis.
            row_date = idx.date() if hasattr(idx, 'date') else idx
            doy = row_date.timetuple().tm_yday
            td = max(1, min(252, round((doy / 365) * 252)))
            pct = round((float(price) / base - 1.0) * 100.0, 4)
            result.append([td, pct])
        _CY_ACTUAL_CACHE[market_id] = result
        _CY_ACTUAL_TIME[market_id] = now
        return result
    except Exception as e:
        print(f"[cy_actual] {market_id}: {e}")
        return []

@app.get("/api/seasonality")
async def get_seasonality(market: str = None):
    if _SEASONALITY_CACHE["data"] is None:
        try:
            with open(_SEASONALITY_PATH) as f:
                _SEASONALITY_CACHE["data"] = json.load(f)
        except Exception as e:
            return {"error": str(e)}
    data = _SEASONALITY_CACHE["data"]
    if market:
        m = market.upper()
        if m not in data:
            return {"error": f"Market '{m}' not found"}
        mkt_data  = data[m]
        snapshots = mkt_data.get("snapshots", {})
        # Use "current" snapshot for display (all available data, no lookahead needed for live)
        current_snap = snapshots.get("current", {})
        if not current_snap:
            # Legacy flat format fallback
            current_snap = mkt_data
        cy_actual = _get_current_year_actual(m)
        # Compute current_td dynamically (don't use stale value from JSON file)
        import datetime as _seas_dt
        _today = _seas_dt.date.today()
        _doy   = _today.timetuple().tm_yday
        _current_td = max(1, min(252, round((_doy / 365) * 252)))
        return {
            "market": m,
            "all":    current_snap.get("all", []),
            "mt":     current_snap.get("midterm", current_snap.get("mt", [])),
            "post_election": current_snap.get("post_election", []),
            "midterm":       current_snap.get("midterm", []),
            "pre_election":  current_snap.get("pre_election", []),
            "election":      current_snap.get("election", []),
            "months": data.get("months", {}),
            "current_td": _current_td,
            "current_year_actual": cy_actual,
        }
    return _SafeJSONResponse(data)

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
    if _rc4 and (_rn4 - _rc4["ts"]) < _COT_HIST_RESULT_TTL:
        return _SafeJSONResponse(_rc4["data"])

    # ── FX CROSS PAIR: return Briese differential ─────────────────────────────
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
                    "differential": [], "base_briese": [], "quote_briese": [],
                    "base_id": base_id, "quote_id": quote_id}

        window = min(520, len(df_base), len(df_quote))
        # Align on common date range
        n_common = min(len(df_base), len(df_quote))
        df_base  = df_base.tail(n_common).copy()
        df_quote = df_quote.tail(n_common).copy()

        def rolling_briese_arr(arr, window=520):
            result = []
            for i in range(len(arr)):
                sl = arr[max(0, i - window + 1): i + 1]
                lo, hi = min(sl), max(sl)
                if hi == lo: result.append(50.0)
                else: result.append(round((arr[i] - lo) / (hi - lo) * 100, 1))
            return result

        base_comm  = [float(v) for v in df_base["comm_net"].tolist()]
        quote_comm = [float(v) for v in df_quote["comm_net"].tolist()]
        base_briese_series  = rolling_briese_arr(base_comm)
        quote_briese_series = rolling_briese_arr(quote_comm)
        differential_series = [round(b - q, 1) for b, q in zip(base_briese_series, quote_briese_series)]

        # Use base leg dates
        dates = df_base["date"].dt.strftime("%Y-%m-%d").tolist()
        # Tail 520 bars (~10yr)
        dates               = dates[-520:]
        base_briese_series  = base_briese_series[-520:]
        quote_briese_series = quote_briese_series[-520:]
        differential_series = differential_series[-520:]

        _cot_cross = {
            "market":        m_upper,
            "name":          mkt["name"],
            "cross":         True,
            "base_id":       base_id,
            "quote_id":      quote_id,
            "dates":         dates,
            "base_briese":   base_briese_series,
            "quote_briese":  quote_briese_series,
            "differential":  differential_series,
            # Repurpose standard fields for compatibility
            "comm_net":      differential_series,     # differential (0=neutral)
            "lspec_net":     base_briese_series,      # base Briese
            "sspec_net":     quote_briese_series,     # quote Briese
            "open_interest": [],
            "comm_idx_series":  differential_series,
            "lspec_idx_series": base_briese_series,
            "sspec_idx_series": quote_briese_series,
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
# PUT/CALL RATIO HISTORY ENDPOINT
# ============================================================
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
                "pc_ma10": round(float(row["pc_ma10"]), 3) if not pd.isna(row["pc_ma10"]) else None,
                "pc_ma20": round(float(row["pc_ma20"]), 3) if not pd.isna(row["pc_ma20"]) else None,
            })

        df_scored = df.dropna(subset=["pc_ma20"])
        current_ma20 = float(df_scored["pc_ma20"].iloc[-1])
        current_pct = float(np.mean(all_ma20 < current_ma20))
        current_score = round(current_pct * 10, 1)

        return {
            "market": market or "equity",
            "ticker": "CBOE Equity",
            "data": rows,
            "current_ma20": round(current_ma20, 3),
            "current_daily": round(float(df_scored["equity_pc"].iloc[-1]), 3),
            "current_score": current_score,
            "current_percentile": round(current_pct * 100, 1),
            "thresholds": {
                "extreme_greed": round(float(np.percentile(all_ma20, 10)), 3),
                "moderate_greed": round(float(np.percentile(all_ma20, 25)), 3),
                "moderate_fear": round(float(np.percentile(all_ma20, 75)), 3),
                "extreme_fear": round(float(np.percentile(all_ma20, 90)), 3),
            },
            "latest_date": str(df_scored.index[-1].date()),
        }

    # ── Route: ETF / Crypto markets ───────────────────────────────────────
    market_upper = market.upper()
    proxy_ticker = ETF_MAP.get(market_upper)  # GLD / SLV / USO, or None for crypto

    # Load disk cache and maybe refresh today's snapshot
    cache = _load_ticker_cache()
    today_str = str(pd.Timestamp.now().date())

    cache_key = proxy_ticker if proxy_ticker else market_upper  # e.g. "GLD" or "BTC"

    if cache_key not in cache:
        cache[cache_key] = {}

    # Refresh today's snapshot if not already cached
    if today_str not in cache[cache_key]:
        if proxy_ticker:
            val = _fetch_yf_pcr_today(proxy_ticker)
        else:
            # Crypto: use Deribit snapshot
            deribit_map = {"BTC": "BTC", "ETH": "ETH"}
            currency = deribit_map.get(market_upper)
            if currency:
                snap = fetch_deribit_pcr(currency)
                val = snap["pcr_oi"] if snap else None
            else:
                val = None
        if val is not None:
            cache[cache_key][today_str] = val
            try:
                _save_ticker_cache(cache)
            except Exception as e:
                print(f"[PCR-HIST] cache save error: {e}")

    ticker_history = cache.get(cache_key, {})

    # ── Backfill with ETP proxy for dates without direct data ─────────────
    etp_proxy = _load_etp_proxy()

    # Compute scaling ratio (current ticker PCR / current ETP PCR)
    # Use most recent cached value if available
    scale_ratio = 1.0
    if ticker_history and etp_proxy:
        # Find most recent date present in both
        common_dates = sorted(set(ticker_history.keys()) & set(etp_proxy.keys()), reverse=True)
        if common_dates:
            latest_common = common_dates[0]
            etp_val = etp_proxy[latest_common]
            tkr_val = ticker_history[latest_common]
            if etp_val and etp_val > 0:
                scale_ratio = tkr_val / etp_val
        elif ticker_history and etp_proxy:
            # No overlap yet — use latest of each
            tkr_latest = ticker_history[max(ticker_history.keys())]
            etp_latest = etp_proxy[max(etp_proxy.keys())]
            if etp_latest and etp_latest > 0:
                scale_ratio = tkr_latest / etp_latest

    # Build combined series: ETP proxy (scaled) + direct ticker data
    all_dates = sorted(set(list(etp_proxy.keys()) + list(ticker_history.keys())))
    combined = {}
    for d in all_dates:
        if d in ticker_history:
            combined[d] = ticker_history[d]
        elif d in etp_proxy:
            combined[d] = round(etp_proxy[d] * scale_ratio, 4)

    # Convert to sorted list and trim to lookback
    sorted_series = [(d, v) for d, v in sorted(combined.items()) if v is not None]
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


def score_seasonality(market_id: str) -> dict:
    """
    Public wrapper: compute current seasonality score for a market.
    Returns {score, label, detail} matching the pattern of other score_* functions.
    Includes slope_pct, horizon_td_start, horizon_td_end, cycle_key for chart rendering.
    """
    from datetime import date as _date_cls
    today = _date_cls.today()
    doy = today.timetuple().tm_yday
    current_td = max(1, min(252, round((doy / 365) * 252)))
    cycle_key = _cycle_key_for_year(today.year)

    raw = _score_seasonality_at(market_id.upper(), today)
    score = round(float(raw), 1)
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

    # Compute slope for chart horizon shading
    slope_pct = None
    try:
        seas = _load_seas_data()
        m_upper = market_id.upper()
        if m_upper in seas:
            mkt_data = seas[m_upper]
            snapshots = mkt_data.get("snapshots", {})
            snap = snapshots.get(str(today.year)) or snapshots.get("current", {})
            # Always score from the all-years curve (already cycle-weighted internally)
            curve = snap.get("all", [])
            if curve:
                td_map = {p[0]: p[1] for p in curve}
                def _near(t):
                    if t in td_map: return td_map[t]
                    return td_map[min(td_map.keys(), key=lambda k: abs(k - t))]
                val_now = _near(current_td)
                val_3w  = _near(min(252, current_td + 15))
                val_4w  = _near(min(252, current_td + 20))
                slope_pct = round(((val_3w + val_4w) / 2) - val_now, 3)
    except Exception:
        pass

    return {
        "score": score,
        "label": label,
        "detail": {
            "score": score,
            "label": label,
            "market_id": market_id.upper(),
            "date": today.isoformat(),
            "current_td": current_td,
            "cycle_key": cycle_key,
            "slope_pct": slope_pct,
            "horizon_td_start": current_td,
            "horizon_td_end": min(252, current_td + 20),
            "source": "curve" if slope_pct is not None else "window",
        }
    }


# ── Per-asset seasonal slope normaliser ─────────────────────────────────────
# Computed once from the all-years curve: 90th-percentile absolute 20-day slope.
# Used to make scoring relative to each asset's own seasonal volatility.
# e.g. SB p90 ≈ 22pp, 6E p90 ≈ 3.5pp — a +5% slope means very different things.
_SEAS_NORM_CACHE: dict = {}  # {market_id: normaliser_pp}

def _get_seas_normaliser(market_id: str, seas: dict) -> float:
    """Return p90 absolute 20-day slope for market_id, cached after first call."""
    if market_id in _SEAS_NORM_CACHE:
        return _SEAS_NORM_CACHE[market_id]
    try:
        curve = seas.get(market_id, {}).get("snapshots", {}).get("current", {}).get("all", [])
        if not curve:
            return 3.0  # fallback
        td_map = {p[0]: p[1] for p in curve}
        slopes = []
        for td in range(1, 233):
            v0 = td_map.get(td)
            v1 = td_map.get(td + 20)
            if v0 is not None and v1 is not None:
                slopes.append(abs(v1 - v0))
        if not slopes:
            return 3.0
        slopes.sort()
        p90 = slopes[int(len(slopes) * 0.90)]
        # Floor at 1.0 to avoid division by near-zero for very stable assets
        norm = max(1.0, round(p90, 2))
        _SEAS_NORM_CACHE[market_id] = norm
        return norm
    except Exception:
        return 3.0


def _load_seas_data() -> dict:
    """Load seasonality data from disk, using cached value if available."""
    import os as _os2
    if _SEASONALITY_CACHE["data"] is None:
        p = _os2.path.join(_os2.path.dirname(__file__), "seasonality_all21.json")
        try:
            with open(p) as f:
                _SEASONALITY_CACHE["data"] = json.load(f)
        except Exception:
            return {}
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

    Uses the 4-cycle snapshot system:
      - snapshots[str(bar_year)] was built from data in years < bar_year only.
        This means it is the curve that was observable BEFORE bar_year began.
      - Within that snapshot, selects the presidential cycle curve matching bar_year
        (post_election / midterm / pre_election / election), falls back to 'all'.
    """
    from datetime import date as _date
    seas = _load_seas_data()
    if not seas:
        return 5.0

    if hasattr(bar_date, 'date'):
        d = bar_date.date()
    elif isinstance(bar_date, _date):
        d = bar_date
    else:
        try:
            d = pd.to_datetime(bar_date).date()
        except Exception:
            return 5.0

    day_of_year = d.timetuple().tm_yday
    current_td  = max(1, min(252, round((day_of_year / 365) * 252)))

    bar_year  = d.year
    cycle_key = _cycle_key_for_year(bar_year)

    curve_score = None
    if market_id in seas:
        mkt_data  = seas[market_id]
        snapshots = mkt_data.get("snapshots", {})

        # Prefer the year-specific snapshot (zero lookahead).
        # Fall back to "current" if this year's snapshot doesn't exist yet
        # (e.g. bar is before we had enough data to build any curves).
        snap = snapshots.get(str(bar_year)) or snapshots.get("current", {})
        if not snap:
            snap = mkt_data  # legacy flat fallback

        # Always use the all-years curve — it already incorporates cycle weighting
        # (midterm years get 2.5x uplift, recency decay 0.88^age).
        # The cycle-specific curves are for reference only, not scoring.
        curve = snap.get("all", [])

        if curve:
            if isinstance(curve[0], (list, tuple)):
                td_map = {p[0]: p[1] for p in curve}
            else:
                td_map = {round((i / 365) * 252): float(v) for i, v in enumerate(curve) if v is not None}

            def _nearest(td):
                return td_map.get(td, td_map[min(td_map.keys(), key=lambda k: abs(k - td))])

            val_now   = _nearest(current_td)
            val_3w    = _nearest(min(252, current_td + 15))
            val_4w    = _nearest(min(252, current_td + 20))
            slope_pct = ((val_3w + val_4w) / 2) - val_now
            _norm = _get_seas_normaliser(market_id, seas)
            curve_score = round(max(0.0, min(10.0, (slope_pct / _norm) * 5.0 + 5.0)), 1)

    if curve_score is not None:
        return curve_score

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


def _score_momentum_at(px_closes: np.ndarray, px_dates_norm, bar_date_norm) -> float:
    """
    Compute momentum score using only price data up to and including bar_date.
    Zero lookahead — only uses closes where date <= bar_date.
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

    ema_st_s10 = round(max(0.0, min(10.0, (ema_st_slope / 3.0) * 5.0 + 5.0)), 1)
    ema_s10    = round(max(0.0, min(10.0, (ema_slope / 6.0) * 5.0 + 5.0)), 1)
    roc_s10    = round(max(0.0, min(10.0, (roc4w / 8.0) * 5.0 + 5.0)), 1)
    sma200_s10 = round(max(0.0, min(10.0, ((curr - sma200) / sma200 * 100 / 10.0) * 5.0 + 5.0)), 1)                  if not np.isnan(sma200) and sma200 > 0 else 5.0

    return round(max(0.0, min(10.0,
        ema_st_s10 * 0.35 + ema_s10 * 0.20 + roc_s10 * 0.25 + sma200_s10 * 0.20
    )), 1)


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
    US_SCALE = {"nfp": 100, "claims": 10, "gdp": 0.5, "retail": 0.3,
                "cpi": 0.1, "pce": 0.1, "ppi": 0.1, "wages": 0.1,
                "pmi": 1.0, "jolts": 200}
    us_cat_scores: dict = {}
    for key, info in best.items():
        cat_key = info["category"]
        surprise = info["actual_raw"] - info["forecast_raw"]
        scale = US_SCALE.get(key, 1.0)
        norm  = surprise / scale if scale else surprise
        if norm > 1.5:    sc = 2
        elif norm > 0.4:  sc = 1
        elif norm < -1.5: sc = -2
        elif norm < -0.4: sc = -1
        else:             sc = 0
        if not info["higher_is_good"]:
            sc = -sc
        if cat_key not in us_cat_scores:
            us_cat_scores[cat_key] = []
        us_cat_scores[cat_key].append(sc)

    usd_cats: dict = {c: sum(v)/len(v) for c, v in us_cat_scores.items() if v}
    usd_score = max(-2.0, min(2.0, sum(usd_cats.values()) / len(usd_cats))) if usd_cats else 0.0
    ff_macro_snap["USD"] = {"score": usd_score, "cat_avg": usd_cats, "cat_details": {}, "label": "USD"}

    # Build a full components dict so get_macro_score_for_market can extract
    # per-indicator scores (nfp_s, gdp_s, cpi_s etc.) via its s() helper.
    # Without this, non-FX formulas (ES, GC, ZB etc.) get all-zero inputs.
    US_KEY_TO_COMP = {
        "NFP":     "NFP",    "ADP":     "ADP",    "UNEMP":   "UNEMP",
        "CLAIMS":  "CLAIMS", "JOLTS":   "JOLTS",  "WAGES":   "WAGES",
        "GDP":     "GDP",    "MFG_PMI": "MFG_PMI","SVC_PMI": "SVC_PMI",
        "RETAIL":  "RETAIL", "CPI":     "CPI",    "PCE":     "PCE",
        "PPI":     "PPI",    "DGS2":    "DGS2",
    }
    components = {}
    for key, info in best.items():
        comp_key = US_KEY_TO_COMP.get(key, key)
        surprise = info["actual_raw"] - info["forecast_raw"]
        scale = US_SCALE.get(key, 1.0)
        norm = surprise / scale if scale else surprise
        if norm > 1.5:    sc = 2
        elif norm > 0.4:  sc = 1
        elif norm < -1.5: sc = -2
        elif norm < -0.4: sc = -1
        else:             sc = 0
        if not info["higher_is_good"]:
            sc = -sc
        components[comp_key] = {"score": sc, "actual": info["actual_raw"],
                                "forecast": info["forecast_raw"]}

    macro_snap = {"category_scores": usd_cats, "components": components}
    result = get_macro_score_for_market(market_id_u, macro_snap, ff_macro=ff_macro_snap)
    # get_macro_score_for_market already returns 0-10 — return directly
    return round(max(0.0, min(10.0, result.get("score", 5.0))), 1)


def _score_regime_at(market_id: str, bar_date_norm,
                     regime_px: dict) -> float:
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
        returns[name] = {"1w": ret_1w, "1m": ret_1m}
        levels[name]  = float(close[-1])

    if not returns:
        return 5.0

    regime_score = 0.0

    if "SPX" in returns:
        s = 1 if returns["SPX"]["1m"] > 2 else -1 if returns["SPX"]["1m"] < -3 else 0
        regime_score += s * 1.2
    if "RTY" in returns:
        s = 1 if returns["RTY"]["1m"] > 3 else -1 if returns["RTY"]["1m"] < -4 else 0
        regime_score += s * 0.6

    vix_level  = levels.get("VIX", 20)
    vix3m_level = levels.get("VIX3M", 20)
    # VIX thresholds recalibrated to match compute_risk_regime (Phase 1A fix):
    # median VIX ~17 → <17 alone is NOT a risk-on signal; >21 = mild risk-off.
    if vix_level >= 27:   vix_level_s = -2
    elif vix_level >= 21: vix_level_s = -1
    elif vix_level <= 13: vix_level_s = 2
    elif vix_level <= 17: vix_level_s = 1
    else:                 vix_level_s = 0  # 17–21 = neutral

    vix_ts = 0
    if "VIX3M" in levels and "VIX" in levels:
        ts_spread = vix3m_level - vix_level
        if ts_spread > 3:    vix_ts = 1
        elif ts_spread < -2: vix_ts = -2
        elif ts_spread < 0:  vix_ts = -1
    regime_score += (vix_level_s + vix_ts) / 2

    if "HYG" in returns and "LQD" in returns:
        spread = returns["HYG"]["1m"] - returns["LQD"]["1m"]
        cs = 1 if spread > 1.5 else 0.5 if spread > 0.3 else -2 if spread < -2.0 else -1 if spread < -0.5 else 0
        regime_score += cs * 0.8

    tnx = levels.get("TNX")
    irx = levels.get("IRX")
    if tnx and irx:
        term_spread = tnx - (irx / 100)
        tnx_1m = returns.get("TNX", {}).get("1m", 0)
        if term_spread > 1.5 and tnx_1m > 0:   yc_s = 1
        elif term_spread > 0.5:                  yc_s = 0.5
        elif term_spread < -0.5:                 yc_s = -1
        elif term_spread < 0:                    yc_s = -0.5
        else:                                    yc_s = 0
        regime_score += yc_s * 0.7

    if "USDJPY" in returns:
        # USDJPY rising = JPY weakening = risk-on (matches live compute_risk_regime)
        uj_s = max(-1.0, min(1.0, returns["USDJPY"]["1m"] / 5.0))
        regime_score += uj_s * 0.30

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

    # ── WALCL: no historical proxy available via yfinance; default to 0 ───────
    # The WALCL signal is a small 0.15 weight; zero is a safe neutral fallback.
    _hist_walcl_sig = 0.0

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
                "chg_3m_pct": 0.0,  # WALCL not available historically
            },
        },
    }
    result = get_regime_score_for_market(market_id, regime_dict)
    return round(max(0.0, min(10.0, result.get("score", 5.0))), 1)


_SH_RESULT_CACHE: dict = {}
_SH_RESULT_TTL = 3600  # 1 hour
_SH_PREFETCH_CACHE: dict = {}
_SH_PREFETCH_TTL = 3600 * 12  # 12h prefetch cache

@app.get("/api/score_history")
async def get_score_history(market: str):
    """Walk-forward composite score history (result-cached 30 min)."""
    m_upper = market.upper()
    mkt = next((x for x in MARKETS if x["id"] == m_upper), None)
    if not mkt:
        return {"error": f"Unknown market: {market}", "dates": [], "scores": [], "prices": []}
    # ── Result cache check (30 min) ──────────────────────────────────────────
    _rn = time.time()
    _rc = _SH_RESULT_CACHE.get(m_upper)
    if _rc and (_rn - _rc["ts"]) < _SH_RESULT_TTL:
        return _SafeJSONResponse(_rc["data"])

    # ── Pre-fetch all historical data (cached per market, 2h TTL) ──────────────
    _now_ts = time.time()
    _cached = _SH_PREFETCH_CACHE.get(m_upper)
    if _cached and (_now_ts - _cached["ts"]) < _SH_PREFETCH_TTL:
        all_ff_events       = _cached["ff_events"]
        regime_px           = _cached["regime_px"]
        relval_self_series  = _cached["relval_self"]
        relval_peer_map     = _cached["relval_peer_map"]
        relval_periods      = _cached["relval_periods"]
        pcr_s_const         = _cached["pcr_s_const"]
        print(f"score_history[{m_upper}]: using prefetch cache ({len(all_ff_events)} FF events)")
    else:
        # Run the entire prefetch block in a thread executor — it makes ~60 FF HTTP
        # calls + multiple yfinance calls, all synchronous. Blocking the event loop
        # here would prevent /api/health from responding for minutes.
        def _do_prefetch():
            _pf_ts = time.time()
            # 1. FF macro: fetch 5 years of monthly calendar data in parallel
            _months_to_fetch = 60  # ~5 years
            _today_d = date.today()
            _year_month_pairs = []
            for _i in range(_months_to_fetch):
                _m_back = _today_d.month - _i
                _y_back = _today_d.year
                while _m_back <= 0:
                    _m_back += 12
                    _y_back -= 1
                _year_month_pairs.append((_y_back, _m_back))
            _all_ff = _fetch_ff_months_parallel(_year_month_pairs)
            print(f"score_history[{m_upper}]: fetched {len(_all_ff)} FF events over {_months_to_fetch} months")

            # 2. Regime: fetch max weekly closes for all regime tickers
            _regime_px: dict = {}
            for _rn, _rticker in RISK_ASSETS.items():
                try:
                    _df = yf.Ticker(_rticker).history(period="max", interval="1wk", auto_adjust=True)
                    if not _df.empty:
                        _s = _df["Close"].copy()
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
                _relval_periods = _rv_cfg.get("periods", [13, 26])
                try:
                    _df_s = yf.Ticker(mkt["yf"]).history(period="max", interval="1wk", auto_adjust=True)
                    if not _df_s.empty:
                        _ss = _df_s["Close"].copy()
                        _ss.index = pd.to_datetime(_ss.index).tz_localize(None).normalize()
                        _ss.index = _ss.index.map(lambda d: np.datetime64(d.date().isoformat(), 'D'))
                        _relval_self = _ss
                except Exception:
                    pass
                for _peer in _rv_cfg.get("peers", []):
                    try:
                        _df_p = yf.Ticker(_peer["yf"]).history(period="max", interval="1wk", auto_adjust=True)
                        if not _df_p.empty:
                            _sp = _df_p["Close"].copy()
                            _sp.index = pd.to_datetime(_sp.index).tz_localize(None).normalize()
                            _sp.index = _sp.index.map(lambda d: np.datetime64(d.date().isoformat(), 'D'))
                            _relval_peer_map[_peer["yf"]] = _sp
                    except Exception:
                        pass

            # 4. PCR — held constant
            _live_pcr = score_pcr(m_upper)
            _pcr_s = _live_pcr.get("score", 5.0) if _live_pcr else 5.0

            # Store in prefetch cache
            _SH_PREFETCH_CACHE[m_upper] = {
                "ff_events":      _all_ff,
                "regime_px":      _regime_px,
                "relval_self":    _relval_self,
                "relval_peer_map": _relval_peer_map,
                "relval_periods": _relval_periods,
                "pcr_s_const":    _pcr_s,
                "ts":             _pf_ts,
            }
            return _SH_PREFETCH_CACHE[m_upper]

        # Run the heavy IO in a thread so the event loop stays responsive
        _pf = await asyncio.get_event_loop().run_in_executor(_APP_EXECUTOR, _do_prefetch)
        all_ff_events      = _pf["ff_events"]
        regime_px          = _pf["regime_px"]
        relval_self_series = _pf["relval_self"]
        relval_peer_map    = _pf["relval_peer_map"]
        relval_periods     = _pf["relval_periods"]
        pcr_s_const        = _pf["pcr_s_const"]

    # ── Determine weights ────────────────────────────────────────────────────
    cat = mkt.get("category", "")
    if cat == "equity":
        w_map = WEIGHTS_EQUITY
    elif m_upper == "CL":
        w_map = WEIGHTS_PCR_TIER2
    elif cat == "crypto":
        w_map = WEIGHTS_PCR_TIER3
    else:
        w_map = WEIGHTS

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

        MIN_BARS = 26
        MAX_RETURN = 260
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

            if "date" in df_base.columns:
                bar_date = pd.to_datetime(df_base["date"].iloc[i])
            else:
                bar_date = pd.to_datetime(df_base.index[i])
            bar_date = bar_date.tz_localize(None) if bar_date.tzinfo else bar_date
            bar_date_norm = np.datetime64(str(bar_date.date()), 'D')
            bar_ts = bar_date.timestamp()

            seas_s   = _score_seasonality_at(m_upper, bar_date)
            mom_s    = _score_momentum_at(px_closes_arr, px_dates_norm, bar_date_norm) \
                       if len(px_closes_arr) > 20 else 5.0
            macro_s  = _score_macro_at(m_upper, bar_ts, all_ff_events,
                                        US_MACRO_INDICATOR_MAP, _parse_ff_value)
            regime_s = _score_regime_at(m_upper, bar_date_norm, regime_px)
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
                cands = {k: v for k, v in price_lookup.items() if abs((k - price_date).days) <= 5}
                close = next(iter(sorted(cands.values(), key=lambda x: abs(x - list(cands.values())[0]))), None) if cands else None
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
                on="_cot_date", direction="nearest", tolerance=pd.Timedelta(days=7),
            )
            df_merged = merged.drop(columns=["_cot_date"])
        except Exception as _e:
            print(f"score_history price merge error for {m_upper}: {_e}")

    MIN_BARS   = 26
    MAX_RETURN = 260
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
            cot_result = compute_cot_score(slice_df, market_id=m_upper)
        cot_s = cot_result["score"]

        if "date" in slice_df.columns:
            bar_date = pd.to_datetime(slice_df["date"].iloc[-1])
        else:
            bar_date = pd.to_datetime(slice_df.index[-1])
        bar_date = bar_date.tz_localize(None) if bar_date.tzinfo else bar_date
        bar_date_norm = np.datetime64(str(bar_date.date()), 'D')
        bar_ts = bar_date.timestamp()

        seas_s   = _score_seasonality_at(m_upper, bar_date)
        mom_s    = _score_momentum_at(px_closes_all, px_dates_norm_all, bar_date_norm) \
                   if len(px_closes_all) > 20 else 5.0
        macro_s  = _score_macro_at(m_upper, bar_ts, all_ff_events,
                                    US_MACRO_INDICATOR_MAP, _parse_ff_value)
        regime_s = _score_regime_at(m_upper, bar_date_norm, regime_px)

        # Rel-val now uses trend-gated logic (Bernd philosophy):
        # cheap + uptrend = bullish; cheap + downtrend = neutral (avoids 2022 JPY trap);
        # expensive + downtrend = bearish; expensive + uptrend = neutral.
        # This makes it safe to use for FX as well — trend gate prevents false signals.
        relval_s = _score_relval_at(m_upper, bar_date_norm,
                                     relval_self_series, relval_peer_map, relval_periods) \
                   if relval_self_series is not None else 5.0

        composite = round(max(0.0, min(10.0,
            cot_s    * w_map["cot"]      +
            seas_s   * w_map["seasonal"] +
            mom_s    * w_map["momentum"] +
            macro_s  * w_map["macro"]    +
            regime_s * w_map["regime"]   +
            relval_s * w_map["relval"]   +
            pcr_s_const * w_map.get("pcr", 0.0)
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
                returns[name] = {"1w": ret_1w, "1m": ret_1m}
                levels[name]  = float(close[-1])

            if not returns:
                continue

            rsc = 0.0  # regime_score accumulator

            # Equities
            if "SPX" in returns:
                s = 1.2 if returns["SPX"]["1m"] > 3 else -1.2 if returns["SPX"]["1m"] < -4 else 0.5 if returns["SPX"]["1m"] > 0 else -0.5
                rsc += s
            if "RUT" in returns:
                s = 1 if returns["RUT"]["1m"] > 3 else -1 if returns["RUT"]["1m"] < -4 else 0
                rsc += s * 0.6

            # VIX
            vix_level  = levels.get("VIX", 20)
            vix3m_level = levels.get("VIX3M", 20)
            if vix_level >= 27:   vix_level_s = -2
            elif vix_level >= 21: vix_level_s = -1
            elif vix_level <= 13: vix_level_s = 2
            elif vix_level <= 17: vix_level_s = 1
            else:                 vix_level_s = 0
            vix_ts = 0
            if "VIX3M" in levels and "VIX" in levels:
                ts_spread = vix3m_level - vix_level
                if ts_spread > 3:    vix_ts = 1
                elif ts_spread < -2: vix_ts = -2
                elif ts_spread < 0:  vix_ts = -1
            rsc += (vix_level_s + vix_ts) / 2

            # Credit
            if "HYG" in returns and "LQD" in returns:
                spread = returns["HYG"]["1m"] - returns["LQD"]["1m"]
                cs = 1 if spread > 1.5 else 0.5 if spread > 0.3 else -2 if spread < -2.0 else -1 if spread < -0.5 else 0
                rsc += cs * 0.8

            # Yield curve
            tnx = levels.get("TNX")
            irx = levels.get("IRX")
            if tnx and irx:
                term_spread = tnx - (irx / 100)
                tnx_1m = returns.get("TNX", {}).get("1m", 0)
                yc_s = 1 if term_spread > 1.5 and tnx_1m > 0 else 0.5 if term_spread > 0.5 else -1 if term_spread < -0.5 else -0.5 if term_spread < 0 else 0
                rsc += yc_s * 0.7

            # USD/JPY — rising USDJPY = JPY weakening = risk-on (matches live compute_risk_regime)
            if "USDJPY" in returns:
                uj_s = max(-1.0, min(1.0, returns["USDJPY"]["1m"] / 5.0))
                rsc += uj_s * 0.30

            rsc = round(max(-4.0, min(4.0, rsc)), 2)
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

        return {
            "dates":   dates_out,
            "scores":  scores_out,
            "labels":  labels_out,
            "signals": signal_rows,
            "current_score": scores_out[-1] if scores_out else None,
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

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
    return FileResponse(idx, media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate"})

# Startup warmup disabled — caches populate on first request to avoid OOM
# The keepalive cron handles backend health and restarts if needed
@app.on_event("startup")
async def warmup_cache():
    """Startup handler — warmup disabled to prevent OOM on resource-constrained host."""
    print("[startup] Backend ready — caches will populate on first request")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
