"""Crowd phase artifact API.

GET /api/phase?market=NQ  -> the per-market phase file written by research/positioning/phase_export.py
GET /api/phase            -> the index (build stamp, markets, spans)

The files are static research output committed to the repository. They are not refreshed by the
backend, so each response carries its own `built` and `to` stamps and the frontend prints them; the
current point and the recent path come from the live series, never from this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

PHASE_DIR = Path(__file__).resolve().parent / "research" / "positioning" / "phase"
_CACHE: Dict[str, Any] = {}


def load(mid: str) -> Optional[dict]:
    mid = (mid or "").strip().upper()
    if not mid or "/" in mid or "." in mid:
        return None
    if mid in _CACHE:
        return _CACHE[mid]
    p = PHASE_DIR / f"{mid}.json"
    if not p.exists():
        _CACHE[mid] = None
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = None
    _CACHE[mid] = data
    return data


def index() -> Optional[dict]:
    p = PHASE_DIR / "index.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def register(app) -> None:
    from fastapi.responses import JSONResponse

    @app.get("/api/phase")
    async def api_phase(market: Optional[str] = None):
        if not market:
            idx = index()
            if idx is None:
                return JSONResponse({"error": "no phase artifact"}, status_code=404)
            return JSONResponse(idx, headers={"Cache-Control": "public, max-age=3600"})
        data = load(market)
        if data is None:
            return JSONResponse({"error": f"no phase artifact for '{market}'", "market": market.upper()},
                                status_code=404)
        return JSONResponse(data, headers={"Cache-Control": "public, max-age=3600"})
