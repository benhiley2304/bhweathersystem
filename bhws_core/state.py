"""Durable state for the things a deploy used to wipe.

One small key -> JSON document store with three tiers, used in this order:

  local    the working copy next to the code (DATA_DIR). Always written.
  postgres when DATABASE_URL is set: table bhws_state(name, body jsonb,
           updated_at). Written through on every save, read at boot.
  github   when BHWS_STATE_REPO and BHWS_STATE_TOKEN are set: one JSON file
           per name in a private repository, via the contents API. Written
           through on every save, read at boot.

At boot `restore_all()` compares the remote copy's updated_at with the local
file's mtime and overlays the newer one, so a redeploy (which resets the
filesystem to the committed files) no longer loses injected feeds, plans or
the ledger. Remote writes run on a background thread so a slow remote never
blocks a request. Failures are logged and never raised into the request path.

A fourth input is the committed seed bundle `state/seed_bundle.json`: the
public-data part of an export (never the ledger), refreshed with
`tools/refresh_seed.py` before a push and applied at boot with the same union
merge as any import, so a deploy without a remote tier still comes up with
the injected histories instead of the stale per-file seeds.

Nothing here invents data: when no remote is configured the behaviour is the
same as before (local files plus the committed seed) and `status()` says so.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

# ── merge semantics ──────────────────────────────────────────────────────
# Keyed stores are unions: a copy that is missing keys the other has (a repo
# seed after a deploy, or a store refreshed from a short live feed before the
# longer injected history arrived) must never erase them. `newer` wins on a
# key both sides hold. Snapshot stores (fed_pricing) are newest-wins.
def _merge_keyed(older, newer):
    if isinstance(older, dict) and isinstance(newer, dict):
        out = dict(older)
        out.update(newer)
        return out
    return newer if newer is not None else older


def _merge_by_time(older, newer, key="time"):
    if isinstance(older, dict) and isinstance(newer, dict):
        out = dict(older)
        for k, v in newer.items():
            o = out.get(k)
            try:
                if o is None or float((v or {}).get(key) or 0) >= float((o or {}).get(key) or 0):
                    out[k] = v
            except Exception:
                out[k] = v
        return out
    return newer if newer is not None else older


def _merge_history(older, newer):
    """fed_pricing_history: list of {date, months}; union by date, newer wins."""
    if isinstance(older, list) and isinstance(newer, list):
        by = {str(x.get("date")): x for x in older if isinstance(x, dict)}
        by.update({str(x.get("date")): x for x in newer if isinstance(x, dict)})
        return [by[k] for k in sorted(by)]
    return newer if newer is not None else older


def _merge_ledger(older, newer):
    """ledger: {entries:[...]} union by id, the later updated_at wins."""
    if isinstance(older, dict) and isinstance(newer, dict):
        by = {}
        for src in (older.get("entries") or [], newer.get("entries") or []):
            for e in src:
                if not isinstance(e, dict) or not e.get("id"):
                    continue
                o = by.get(e["id"])
                if o is None or float(e.get("updated_at") or 0) >= float(o.get("updated_at") or 0):
                    by[e["id"]] = e
        out = dict(older)
        out.update(newer)
        out["entries"] = sorted(by.values(), key=lambda e: float(e.get("created_at") or 0))
        return out
    return newer if newer is not None else older


def _merge_grades(older, newer):
    """ledger_grades: {rows:[...]} union by (day, market), newer wins."""
    if isinstance(older, dict) and isinstance(newer, dict):
        by = {}
        for src in (older.get("rows") or [], newer.get("rows") or []):
            for r in src:
                if isinstance(r, dict) and r.get("day") and r.get("market"):
                    by[(str(r["day"]), str(r["market"]))] = r
        out = dict(older)
        out.update(newer)
        out["rows"] = [by[k] for k in sorted(by)]
        return out
    return newer if newer is not None else older


MERGE = {
    "ff_event_store": _merge_keyed,
    "supply_event_store": _merge_keyed,
    "ff_macro_injected": _merge_by_time,
    "fed_pricing_history": _merge_history,
    "ledger": _merge_ledger,
    "ledger_grades": _merge_grades,
}


def merge(name: str, local, incoming, incoming_is_newer: bool):
    """Combine the local copy with an incoming one. Returns (body, changed)."""
    fn = MERGE.get(name)
    if fn is None:
        return (incoming, incoming != local) if incoming_is_newer else (local, False)
    older, newer = (local, incoming) if incoming_is_newer else (incoming, local)
    out = fn(older, newer)
    return out, out != local


STAMP_FILE = ".bhws_state_stamps.json"
NAMES = (
    "ff_event_store", "supply_event_store", "fed_pricing", "fed_pricing_history",
    "ff_macro_injected", "ledger", "ledger_grades",
)


def _log(msg: str) -> None:
    print(f"[state] {msg}", flush=True)


class _Postgres:
    def __init__(self, url: str):
        self.url = url
        self._ok = None

    def _conn(self):
        import psycopg  # lazy: only installed/needed when DATABASE_URL is set
        return psycopg.connect(self.url, connect_timeout=8)

    def ensure(self) -> bool:
        if self._ok is not None:
            return self._ok
        try:
            with self._conn() as c:
                c.execute("CREATE TABLE IF NOT EXISTS bhws_state (name text PRIMARY KEY, "
                          "body jsonb NOT NULL, updated_at double precision NOT NULL)")
            self._ok = True
        except Exception as e:
            _log(f"postgres unavailable: {e!r}")
            self._ok = False
        return self._ok

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        if not self.ensure():
            return None
        with self._conn() as c:
            row = c.execute("SELECT body, updated_at FROM bhws_state WHERE name=%s", (name,)).fetchone()
        if not row:
            return None
        body = row[0] if not isinstance(row[0], str) else json.loads(row[0])
        return {"body": body, "updated_at": float(row[1])}

    def put(self, name: str, body: Any, updated_at: float) -> None:
        if not self.ensure():
            return
        with self._conn() as c:
            c.execute("INSERT INTO bhws_state(name, body, updated_at) VALUES (%s, %s::jsonb, %s) "
                      "ON CONFLICT (name) DO UPDATE SET body=EXCLUDED.body, updated_at=EXCLUDED.updated_at",
                      (name, json.dumps(body), updated_at))


class _GitHub:
    def __init__(self, repo: str, token: str, branch: str = "main"):
        self.repo, self.token, self.branch = repo, token, branch
        self._sha: Dict[str, str] = {}

    def _hdr(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _url(self, name: str) -> str:
        return f"https://api.github.com/repos/{self.repo}/contents/{name}.json"

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        import base64
        import requests
        r = requests.get(self._url(name), headers=self._hdr(), params={"ref": self.branch}, timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        self._sha[name] = j.get("sha")
        raw = base64.b64decode(j.get("content") or "").decode("utf-8")
        doc = json.loads(raw) if raw else None
        if isinstance(doc, dict) and "__updated_at" in doc:
            return {"body": doc.get("body"), "updated_at": float(doc["__updated_at"])}
        return {"body": doc, "updated_at": 0.0}

    def put(self, name: str, body: Any, updated_at: float) -> None:
        import base64
        import requests
        if name not in self._sha:
            try:
                self.get(name)
            except Exception:
                pass
        doc = {"__updated_at": updated_at, "body": body}
        payload = {"message": f"state: {name} @ {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(updated_at))}",
                   "content": base64.b64encode(json.dumps(doc, separators=(",", ":")).encode()).decode(),
                   "branch": self.branch}
        if self._sha.get(name):
            payload["sha"] = self._sha[name]
        r = requests.put(self._url(name), headers=self._hdr(), json=payload, timeout=30)
        if r.status_code == 409 or r.status_code == 422:
            self._sha.pop(name, None)
            self.get(name)
            if self._sha.get(name):
                payload["sha"] = self._sha[name]
            r = requests.put(self._url(name), headers=self._hdr(), json=payload, timeout=30)
        r.raise_for_status()
        self._sha[name] = (r.json().get("content") or {}).get("sha")


class StateStore:
    def __init__(self, data_dir: str, env: Optional[Dict[str, str]] = None):
        env = env if env is not None else os.environ
        self.data_dir = data_dir
        self.remote = None
        self.remote_kind = "none"
        self._lock = threading.Lock()
        self._pending: Dict[str, float] = {}
        self._errors: List[str] = []
        self.last_restore: Dict[str, str] = {}
        self.last_seed: Dict[str, Any] = {}
        self._on_restore: Dict[str, Callable[[Any], None]] = {}
        url = env.get("DATABASE_URL")
        repo, tok = env.get("BHWS_STATE_REPO"), env.get("BHWS_STATE_TOKEN")
        if url:
            self.remote, self.remote_kind = _Postgres(url), "postgres"
        elif repo and tok:
            self.remote, self.remote_kind = _GitHub(repo, tok, env.get("BHWS_STATE_BRANCH", "main")), "github"

    # ── local tier ───────────────────────────────────────────────────────
    def path(self, name: str) -> str:
        return os.path.join(self.data_dir, f"{name}.json")

    def load(self, name: str, default: Any = None) -> Any:
        try:
            with open(self.path(name)) as f:
                return json.load(f)
        except Exception:
            return default

    # Age of the local copy. Only files this store wrote carry a stamp; a file
    # that came with the repository checkout (a seed) has a fresh filesystem
    # mtime on every deploy but no stamp, so it reports age 0 and any remote or
    # imported copy wins over it. Filesystem mtime is never trusted for this.
    def _stamps(self) -> Dict[str, float]:
        try:
            with open(os.path.join(self.data_dir, STAMP_FILE)) as f:
                d = json.load(f)
            return {k: float(v) for k, v in d.items()} if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _stamp(self, name: str, ts: float) -> None:
        d = self._stamps()
        d[name] = float(ts)
        tmp = os.path.join(self.data_dir, STAMP_FILE + ".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, os.path.join(self.data_dir, STAMP_FILE))
        except Exception as e:
            self._err(f"stamp {name}: {e!r}")

    def local_mtime(self, name: str) -> float:
        if not os.path.exists(self.path(name)):
            return 0.0
        return self._stamps().get(name, 0.0)

    def local_fs_mtime(self, name: str) -> float:
        try:
            return os.path.getmtime(self.path(name))
        except Exception:
            return 0.0

    def save(self, name: str, body: Any, *, mirror: bool = True) -> None:
        """Write the local file now; mirror to the remote tier in the background."""
        tmp = self.path(name) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(body, f)
            os.replace(tmp, self.path(name))
        except Exception as e:
            self._err(f"local save {name}: {e!r}")
            return
        now = time.time()
        self._stamp(name, now)
        if mirror and self.remote is not None:
            self._mirror_async(name, body, now)

    # ── remote tier ──────────────────────────────────────────────────────
    def _mirror_async(self, name: str, body: Any, ts: float) -> None:
        def run():
            try:
                self.remote.put(name, body, ts)
                with self._lock:
                    self._pending.pop(name, None)
            except Exception as e:
                self._err(f"remote put {name}: {e!r}")
        with self._lock:
            self._pending[name] = ts
        threading.Thread(target=run, name=f"state-mirror-{name}", daemon=True).start()

    def mirror_now(self, name: str) -> bool:
        """Synchronous push of the current local file (used by import)."""
        if self.remote is None:
            return False
        body = self.load(name)
        if body is None:
            return False
        try:
            self.remote.put(name, body, time.time())
            return True
        except Exception as e:
            self._err(f"remote put {name}: {e!r}")
            return False

    def on_restore(self, name: str, fn: Callable[[Any], None]) -> None:
        """Register a hook that re-hydrates an in-memory cache after a restore."""
        self._on_restore[name] = fn

    # ── committed seed bundle ────────────────────────────────────────────
    # `state/seed_bundle.json` is an export bundle committed with the code so a
    # deploy without a remote tier comes up with the injected histories instead
    # of the stale per-file seeds. It is applied with the same union merge as
    # any import (newer keys win, nothing erased), so live data always beats it.
    # The ledger (Ben's levels) is never read from the seed: it is private and
    # only lives in the local tier or a private remote.
    SEED_EXCLUDE = ("ledger",)

    def seed_path(self) -> str:
        return os.path.join(self.data_dir, "state", "seed_bundle.json")

    def seed_bundle(self) -> Dict[str, Any]:
        """Export bundle minus the private ledger, for committing as the seed."""
        b = self.export()
        b["items"] = {n: v for n, v in b["items"].items() if n not in self.SEED_EXCLUDE}
        b["seed"] = True
        return b

    def apply_seed(self) -> Dict[str, str]:
        p = self.seed_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                bundle = json.load(f)
            items = bundle.get("items") or {}
            bundle = {"items": {n: v for n, v in items.items() if n not in self.SEED_EXCLUDE}}
            out = self.import_bundle(bundle, only_newer=True)
            self.last_seed = {"applied_at": time.time(), "exported_at": None, "out": out}
            try:
                with open(p) as f:
                    self.last_seed["exported_at"] = json.load(f).get("exported_at")
            except Exception:
                pass
            _log("seed: " + ", ".join(f"{k}={v}" for k, v in out.items()))
            return out
        except Exception as e:
            self._err(f"seed bundle: {e!r}")
            return {}

    def restore_all(self, names=NAMES) -> Dict[str, str]:
        """Apply the committed seed, then overlay local files with newer remote copies. Returns name -> outcome."""
        seed = self.apply_seed()
        out: Dict[str, str] = {}
        for n in names:
            out[n] = "no remote" if self.remote is None else self._restore_one(n)
            if seed.get(n) in ("imported", "merged"):
                out[n] = f"seed {seed[n]}" if self.remote is None else out[n]
            if out[n] in ("restored", "merged", "local newer", "local only") or self.remote is None:
                body = self.load(n)
                fn = self._on_restore.get(n)
                if fn is not None and body is not None:
                    try:
                        fn(body)
                    except Exception as e:
                        self._err(f"restore hook {n}: {e!r}")
        self.last_restore = out
        _log("restore: " + ", ".join(f"{k}={v}" for k, v in out.items()))
        return out

    def _restore_one(self, name: str) -> str:
        try:
            rem = self.remote.get(name)
        except Exception as e:
            self._err(f"remote get {name}: {e!r}")
            return "remote error"
        if rem is None or rem.get("body") is None:
            return "local only"
        lm = self.local_mtime(name)
        local = self.load(name)
        newer = float(rem["updated_at"] or 0) > lm + 1
        body, changed = merge(name, local, rem["body"], incoming_is_newer=newer)
        if not changed:
            return "local newer" if not newer else "restored"
        # a merged body is newer than either side; mirror it back so the remote
        # also carries the union
        self.save(name, body, mirror=(name in MERGE))
        if not (name in MERGE):
            self._stamp(name, rem["updated_at"])
        return "restored" if newer else "merged"

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _err(self, msg: str) -> None:
        _log(msg)
        with self._lock:
            self._errors.append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}")
            del self._errors[:-20]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pend = dict(self._pending)
            errs = list(self._errors)
        files = {}
        for n in NAMES:
            present = os.path.exists(self.path(n))
            mt = self.local_mtime(n)
            files[n] = {"present": present, "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mt)) if mt else None,
                        "seed": present and not mt,   # came with the checkout, not written by this store
                        "bytes": os.path.getsize(self.path(n)) if present else 0}
        sp = self.seed_path()
        seed = {"present": os.path.exists(sp), "bytes": os.path.getsize(sp) if os.path.exists(sp) else 0,
                "exported_at": None, "last_apply": self.last_seed.get("out") or {}}
        if self.last_seed.get("exported_at"):
            seed["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(self.last_seed["exported_at"])))
        return {"remote": self.remote_kind, "durable": self.remote is not None,
                "pending_mirrors": len(pend), "errors": errs[-5:], "last_restore": self.last_restore,
                "seed_bundle": seed, "files": files}

    def export(self) -> Dict[str, Any]:
        return {"exported_at": time.time(),
                "items": {n: {"body": self.load(n), "mtime": self.local_mtime(n)} for n in NAMES
                          if os.path.exists(self.path(n))}}

    def import_bundle(self, bundle: Dict[str, Any], *, only_newer: bool = True) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for n, item in (bundle.get("items") or {}).items():
            if n not in NAMES or not isinstance(item, dict) or item.get("body") is None:
                out[n] = "skipped"
                continue
            mt = float(item.get("mtime") or 0)
            lm = self.local_mtime(n)   # 0 for a seed file or nothing on disk
            newer = (not only_newer) or lm == 0 or mt > lm + 1
            local = self.load(n)
            body, changed = merge(n, local, item["body"], incoming_is_newer=newer)
            if not changed:
                out[n] = "local newer" if not newer else "unchanged"
                continue
            self.save(n, body)
            if mt and n not in MERGE:
                self._stamp(n, mt)
            fn = self._on_restore.get(n)
            if fn is not None:
                try:
                    fn(body)
                except Exception as e:
                    self._err(f"import hook {n}: {e!r}")
            out[n] = "imported" if newer else "merged"
        return out
