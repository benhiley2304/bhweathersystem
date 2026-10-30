#!/usr/bin/env python3
"""Refresh state/seed_bundle.json from the live backend before a push.

The seed is the public-data part of the state export (COT release stores,
supply events, macro injections, fed pricing, engine grades). The ledger is
never included. Usage:

    BHWS_ADMIN_TOKEN=... python3 tools/refresh_seed.py
    python3 tools/refresh_seed.py --token-file /path/to/token

Writes the bundle only if the export succeeded and contains at least the two
COT stores, so a failed fetch can never blank the seed.
"""
import argparse
import json
import os
import sys
import urllib.request

API = os.environ.get("BHWS_API", "https://bhweathersystem-backend.onrender.com")
EXCLUDE = ("ledger",)
REQUIRED = ("ff_event_store", "supply_event_store")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-file")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "state", "seed_bundle.json"))
    a = ap.parse_args()
    tok = os.environ.get("BHWS_ADMIN_TOKEN", "")
    if a.token_file:
        with open(a.token_file) as f:
            tok = f.read().strip()
    if not tok:
        print("no admin token (BHWS_ADMIN_TOKEN or --token-file)", file=sys.stderr)
        return 2
    req = urllib.request.Request(API + "/api/admin/state/export", headers={"x-bhws-admin-token": tok})
    with urllib.request.urlopen(req, timeout=60) as r:
        bundle = json.load(r)
    items = bundle.get("items") or {}
    missing = [n for n in REQUIRED if not (items.get(n) or {}).get("body")]
    if missing:
        print(f"export missing {missing}; seed not written", file=sys.stderr)
        return 1
    out = {"exported_at": bundle.get("exported_at"), "seed": True,
           "items": {n: v for n, v in items.items() if n not in EXCLUDE}}
    path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    sizes = {n: len(json.dumps(v.get("body"))) for n, v in out["items"].items()}
    print(f"seed written: {path} ({os.path.getsize(path)} bytes) items={sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
