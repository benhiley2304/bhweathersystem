"""BHWS payoff API (additive): GET /api/payoff and GET /api/payoff/table.

/api/payoff?market=SI[&entry=&stop=&target=&side=long|short][&arch=&confirm=]
    Payoff profile ("does it pay") for the market's current Desk setup class, with the market's own
    history, the unconditional baseline and, when Ben's levels are supplied, a stress test of those
    levels against the class history. The engine evaluates levels; it never generates them.

/api/payoff/table[?k=1.0]
    Cross-archetype table for the Edge lab.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi.responses import JSONResponse

from bhws_core import payoff as PO
from bhws_core import ledger as L


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except Exception:
        return None


def register(app, *, markets, desk_state_getter: Callable[[], Optional[Dict[str, Any]]],
             sd_weekly_getter: Optional[Callable[[str], Optional[float]]] = None) -> None:
    by_id = {m["id"]: m for m in markets}

    @app.get("/api/payoff")
    def payoff(market: str, entry: Optional[str] = None, stop: Optional[str] = None, target: Optional[str] = None,
               side: Optional[str] = None, arch: Optional[str] = None, confirm: Optional[str] = None):
        mid = (market or "").upper().strip()
        m = by_id.get(mid)
        if not m:
            return JSONResponse({"error": f"unknown market {mid}"}, status_code=404)
        if not PO.available():
            return JSONResponse({"available": False, "reason": "payoff profiles not built"}, status_code=503)
        st = desk_state_getter() or {}
        data = st.get("data") or {}
        row = next((r for r in (data.get("rows") or []) if r.get("id") == mid), None)
        r_side = int(row.get("side") or 0) if row else 0
        r_arch = row.get("archetype") if row else None
        r_conf = ((row.get("confirm") or {}).get("state") if row else None)
        cls_key = ((row.get("class") or {}).get("key") if row else None)
        # Ben's side wins when supplied; archetype/confirm overrides are for the Edge lab explorer only
        p_side = {"long": 1, "short": -1}.get((side or "").lower(), None)
        use_side = p_side if p_side in (1, -1) else r_side
        use_arch = arch or r_arch
        use_conf = confirm or r_conf
        prof = PO.lookup(use_arch, use_side, use_conf, m.get("category"), market=mid)
        prof["setup"] = {"archetype": use_arch, "side": use_side, "confirm": use_conf, "category": m.get("category"),
                         "class": cls_key, "desk_side": r_side, "side_from_plan": p_side is not None and p_side != r_side,
                         "desk_built": st.get("built")}
        sdw = sd_weekly_getter(mid) if sd_weekly_getter else None
        prof["sd_weekly"] = sdw
        plan = {"entry": _f(entry), "stop": _f(stop), "target": _f(target)}
        prof["plan"] = None
        prof["stress"] = None
        if plan["entry"] is not None and plan["stop"] is not None and use_side in (1, -1):
            d = L.derive(plan, use_side, sd_weekly=sdw)
            prof["plan"] = dict(plan, **d)
            if d.get("stop_sd") is not None:
                prof["stress"] = PO.stress(prof, d["stop_sd"], d.get("target_sd"), d.get("r_multiple"))
            elif sdw is None:
                prof["plan"]["warnings"] = list(d.get("warnings") or []) + ["weekly volatility unavailable for this market, so levels cannot be expressed in sigma"]
        # trim meta on the per-market response
        prof["meta"] = {k: v for k, v in (prof.get("meta") or {}).items() if k in ("built", "signals", "markets", "first", "last", "unit", "caveat", "baseline_note")}
        return prof

    @app.get("/api/payoff/table")
    def payoff_table(k: float = 1.0, level_max: int = 2):
        if not PO.available():
            return JSONResponse({"available": False, "reason": "payoff profiles not built"}, status_code=503)
        return PO.table(level_max=level_max, k=k)
