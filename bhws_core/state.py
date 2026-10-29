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

Nothing here invents data: when no remote is configured the behaviour is the
same as before (local files only) and `status()` says so.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

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

    def restore_all(self, names=NAMES) -> Dict[str, str]:
        """Overlay local files with newer remote copies. Returns name -> outcome."""
        out: Dict[str, str] = {}
        for n in names:
            out[n] = "no remote" if self.remote is None else self._restore_one(n)
            if out[n] in ("restored", "local newer", "local only") or self.remote is None:
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
        if rem["updated_at"] > self.local_mtime(name) + 1:
            self.save(name, rem["body"], mirror=False)
            self._stamp(name, rem["updated_at"])
            return "restored"
        return "local newer"

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
        return {"remote": self.remote_kind, "durable": self.remote is not None,
                "pending_mirrors": len(pend), "errors": errs[-5:], "last_restore": self.last_restore,
                "files": files}

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
            if only_newer and lm > 0 and mt <= lm + 1:
                out[n] = "local newer"
                continue
            self.save(n, item["body"])
            if mt:
                self._stamp(n, mt)
            fn = self._on_restore.get(n)
            if fn is not None:
                try:
                    fn(item["body"])
                except Exception as e:
                    self._err(f"import hook {n}: {e!r}")
            out[n] = "imported"
        return out
