"""State, ledger and forward-record HTTP surface. Registered from backend.py,
additive only.

  GET  /api/state/status                 which tier is durable, file freshness, last errors
  GET  /api/admin/state/export           every persisted document (admin)
  POST /api/admin/state/import           restore a bundle; newer-only by default (admin)
  GET  /api/admin/ledger                 entries + calibration summary (admin)
  POST /api/admin/ledger                 new plan; the system freezes its read as context (admin)
  PATCH /api/admin/ledger/{id}           update plan levels, status, notes (admin)
  POST /api/admin/ledger/{id}/close      record the outcome (admin)
  GET  /api/admin/ledger/grades          forward grade record, rules v1 and v2 (admin)

Everything under /api/admin/ is behind the x-bhws-admin-token header when
BHWS_ADMIN_TOKEN is set (see bhws_evidence_api.PROTECTED_PREFIXES). The
ledger holds Ben's levels, so reads are protected as well as writes.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from bhws_core import ledger as L

_LOCK = threading.Lock()


def _entries(store) -> Dict[str, Any]:
    doc = store.load("ledger") or {}
    if not isinstance(doc, dict):
        doc = {}
    doc.setdefault("entries", [])
    return doc


def _context_for(market: str, desk_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Freeze what the Desk currently says about the market. Missing Desk
    data is recorded as missing, never guessed."""
    data = (desk_state or {}).get("data") or {}
    row = next((r for r in data.get("rows") or [] if r.get("id") == market), None)
    if not row:
        return {"desk_available": False, "desk_generated_at": data.get("generated_at")}
    su = row.get("setup") or {}
    cf = row.get("confirm") or {}
    return {
        "desk_available": True,
        "desk_generated_at": data.get("generated_at"),
        "ruleset": row.get("ruleset"),
        "grade": (row.get("class") or {}).get("key"),
        "grade_v1": row.get("class_v1"),
        "caps": row.get("caps") or [],
        "desk_side": row.get("side"),
        "archetype": row.get("archetype"),
        "confirm": cf.get("state") if isinstance(cf, dict) else None,
        "confirm_window": cf.get("window") if isinstance(cf, dict) else None,
        "setup_score": su.get("score"),
        "ready": (su.get("ready") or {}).get("key"),
        "timing": (su.get("timing") or {}).get("state"),
        "bias": su.get("bias"),
        "pillars": {k: (v or {}).get("rel") for k, v in (row.get("pillars") or {}).items()},
        "anchors": row.get("primary_edge"),
        "gates": row.get("gates") or [],
    }


def register(app, *, store, desk_state_getter: Callable[[], Optional[Dict[str, Any]]],
             sd_weekly_getter: Optional[Callable[[str], Optional[float]]] = None) -> None:
    from fastapi.responses import JSONResponse

    @app.get("/api/state/status")
    async def state_status():
        return store.status()

    @app.get("/api/admin/state/export")
    async def state_export():
        return store.export()

    @app.post("/api/admin/state/import")
    async def state_import(bundle: dict, only_newer: bool = True):
        with _LOCK:
            out = store.import_bundle(bundle or {}, only_newer=only_newer)
        return {"ok": True, "result": out, "status": store.status()}

    @app.get("/api/admin/ledger")
    async def ledger_list(status: Optional[str] = None, market: Optional[str] = None):
        doc = _entries(store)
        rows = doc["entries"]
        if status:
            rows = [e for e in rows if e.get("status") == status]
        if market:
            rows = [e for e in rows if e.get("market") == market.upper()]
        rows = sorted(rows, key=lambda e: -float(e.get("updated_at") or 0))
        return {"entries": rows, "n": len(rows), "calibration": L.calibration(doc["entries"]),
                "updated_at": doc.get("updated_at")}

    @app.post("/api/admin/ledger")
    async def ledger_create(payload: dict):
        try:
            market = str(payload.get("market") or "").upper()
            sdw = sd_weekly_getter(market) if sd_weekly_getter else None
            entry = L.new_entry(payload, context=_context_for(market, desk_state_getter()), sd_weekly=sdw)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        with _LOCK:
            doc = _entries(store)
            doc["entries"].append(entry)
            doc["updated_at"] = time.time()
            store.save("ledger", doc)
        return {"ok": True, "entry": entry}

    @app.patch("/api/admin/ledger/{entry_id}")
    async def ledger_patch(entry_id: str, patch: dict):
        with _LOCK:
            doc = _entries(store)
            idx = next((i for i, e in enumerate(doc["entries"]) if e.get("id") == entry_id), None)
            if idx is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            try:
                sdw = sd_weekly_getter(doc["entries"][idx]["market"]) if sd_weekly_getter else None
                doc["entries"][idx] = L.update_entry(doc["entries"][idx], patch or {}, sd_weekly=sdw)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            doc["updated_at"] = time.time()
            store.save("ledger", doc)
            return {"ok": True, "entry": doc["entries"][idx]}

    @app.post("/api/admin/ledger/{entry_id}/close")
    async def ledger_close(entry_id: str, outcome: dict):
        with _LOCK:
            doc = _entries(store)
            idx = next((i for i, e in enumerate(doc["entries"]) if e.get("id") == entry_id), None)
            if idx is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            try:
                doc["entries"][idx] = L.close_entry(doc["entries"][idx], outcome or {})
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            doc["updated_at"] = time.time()
            store.save("ledger", doc)
            return {"ok": True, "entry": doc["entries"][idx], "calibration": L.calibration(doc["entries"])}

    @app.get("/api/admin/ledger/grades")
    async def ledger_grades(market: Optional[str] = None, days: int = 400):
        doc = store.load("ledger_grades") or {}
        rows = doc.get("rows") or []
        if market:
            rows = [r for r in rows if r.get("market") == market.upper()]
        cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - max(1, days) * 86400))
        rows = [r for r in rows if str(r.get("day") or "") >= cutoff]
        days_seen = sorted({r.get("day") for r in rows})
        # per-day counts of each grade under both rule sets
        summary = []
        for d in days_seen:
            dr = [r for r in rows if r.get("day") == d]
            c2: Dict[str, int] = {}
            c1: Dict[str, int] = {}
            for r in dr:
                c2[str(r.get("grade"))] = c2.get(str(r.get("grade")), 0) + 1
                c1[str(r.get("grade_v1") or r.get("grade"))] = c1.get(str(r.get("grade_v1") or r.get("grade")), 0) + 1
            summary.append({"day": d, "n": len(dr), "v2": c2, "v1": c1,
                            "capped": sum(1 for r in dr if r.get("caps"))})
        return {"rows": rows, "days": summary, "updated_at": doc.get("updated_at"),
                "note": ("One row per market per Desk build day. grade is the live rule set; grade_v1 the "
                         "original ladder. Outcomes are not attached here; join on close prices later.")}
