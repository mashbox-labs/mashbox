"""Add-on modules: sidebar modules that ship outside this package (an app build can carry them next to the wheel).

An add-on is a Python distribution with an entry point in the group "mashbox.app_addons" whose object has:

    ID                                  lowercase letters, 2 to 20 (not a name the app uses itself)
    LABEL                               the sidebar label
    page(reg) -> bytes                  its page at /m/<ID>/, a module shell page
    asset(path) -> (bytes, type)|None   its files (any path holding ui/<ID>/)
    is_server(url, key) -> bool         whether a server added by address belongs to it
    get(handler, path, query) -> bool   GET /<ID>/... and /api/<ID>...; False = not its route (404)
    post(handler, path) -> bool         POST to the same; the add-on reads the body itself
    state(app_state) -> dict            optional: extra fields for its entry in /api/state "addons"
    TPU                                 optional: TPU rows it offers (tpu_rows()): catalog-style rows with "cloud"
    tpu_payload(row_id, tmp) -> dict    with TPU: what a chip needs for that row, {"code": a .tar it wrote inside
                                        tmp, "data": [large files on this machine], "args": [str]} (tpu_payload())

Its routes are answered only after the app's guard and token check (the same as /api). The app loads add-ons ONLY from
the wheels or folders listed in MASHBOX_APP_ADDONS (os.pathsep between them), which the desktop app sets for the add-ons
it carries: each is imported from there as it is, never installed, and a package installed anywhere else is never an
add-on. Fail closed: a listed path that is missing, holds no add-on or more than one distribution, would import its
code from another place (incl. a module of that name already loaded from elsewhere), or holds an add-on that does not
load or has a bad or taken ID is a problem (problems()); `python -m mashbox.ui.app` refuses to start while there is one.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import types
from importlib import metadata

GROUP = "mashbox.app_addons"
ENV = "MASHBOX_APP_ADDONS"
ID = re.compile(r"[a-z]{2,20}")
# Every first path word the app answers itself (its pages, files and /api calls) and every folder of its own files:
# an add-on can never take one. tests/cpu/test_app_addons.py reads the app's routes and fails when one is missing here.
TAKEN = frozenset({"chat", "world", "models", "servers", "agents", "settings", "api", "v1", "ui", "window", "m", "p",
                   "health", "state", "catalog", "check", "gcloud", "open", "release", "tpu", "shell", "app", "index"})
NEEDS = ("ID", "LABEL", "page", "asset", "is_server", "get", "post")

_loaded: list | None = None
_problems: list = []


def _paths() -> list[str]:
    return [p for p in (os.environ.get(ENV) or "").split(os.pathsep) if p]


def _inside(origin, root: str) -> bool:
    """True when origin is under root once every link is resolved (a link out of the listed path is outside)."""
    return bool(origin) and os.path.realpath(origin).startswith(os.path.join(os.path.realpath(root), ""))


def _from_root(m, root: str) -> bool:
    """True when module m was loaded from inside root: its file and every folder of its package (none = not from root)."""
    locs = [x for x in [getattr(m, "__file__", None), *(getattr(m, "__path__", None) or ())] if x]
    return bool(locs) and all(_inside(x, root) for x in locs)


def _links(root: str, top: str):
    """The first link found in the folder add-on's own package (or None): a folder add-on may hold no link at all."""
    base = os.path.join(root, top)
    for d, dirs, files in os.walk(base):
        for n in [d] + [os.path.join(d, x) for x in dirs + files]:
            if os.path.islink(n):
                return n
    return base + ".py" if os.path.islink(base + ".py") else None


def _from(root: str) -> list:
    """The add-on objects of one listed wheel or folder: only distributions found in it, each imported from it."""
    if not os.path.exists(root):
        raise FileNotFoundError("not found")
    dists = list(metadata.distributions(path=[root]))
    if len(dists) > 1:
        # One distribution per listed path: a second metadata folder would bring its own entry points.
        raise LookupError(f"{len(dists)} distributions in it; an add-on path holds exactly one")
    eps = {(e.name, e.value): e for d in dists for e in d.entry_points if e.group == GROUP}
    if not eps:
        raise LookupError(f"no {GROUP} entry point in it")
    if root not in sys.path:
        sys.path.append(root)
    out = []
    for _, e in sorted(eps.items()):
        top = e.value.split(":", 1)[0].strip().split(".", 1)[0]
        spec = importlib.util.find_spec(top)                  # finds, does not run: nothing from elsewhere is executed
        locs = list(spec.submodule_search_locations or ()) if spec else []
        if spec and spec.submodule_search_locations is not None and (spec.origin in (None, "namespace") or len(locs) != 1):
            # A namespace package joins folders from every sys.path entry, so ns.x could come from anywhere.
            raise ImportError(f"{top} is a namespace package ({len(locs)} locations); an add-on must be a regular package"
                              " (with __init__.py) or a module inside its listed path")
        where = spec and spec.origin
        bad = next((w for w in [where] + locs if not _inside(w, root)), None) if where else None
        if not where or bad:
            raise ImportError(f"{top} would be imported from {bad or 'nowhere'}, not from this add-on")
        if os.path.isdir(root) and _links(root, top):
            raise ImportError(f"{top} holds a link ({_links(root, top)}); an add-on folder may hold none")
        mod = e.value.split(":", 1)[0].strip()
        # A module of that name already imported from elsewhere would be handed back by e.load() as it is:
        # the entry point's module and each of its parents must be unloaded or loaded from this path.
        parts = mod.split(".")
        for n in (".".join(parts[:i]) for i in range(1, len(parts) + 1)):
            if n in sys.modules and not _from_root(sys.modules[n], root):
                raise ImportError(f"{n} is already loaded from {getattr(sys.modules[n], '__file__', None)}, not from "
                                  "this add-on")
        a = e.load()
        home = a if isinstance(a, types.ModuleType) else sys.modules.get(getattr(a, "__module__", None) or "")
        if not _from_root(sys.modules.get(mod), root) or home is None or not _from_root(home, root):
            raise ImportError(f"{e.name}: its object comes from {getattr(home, '__file__', None)}, not from this add-on")
        if not all(hasattr(a, k) for k in NEEDS):
            raise TypeError(f"{e.name}: missing " + ", ".join(k for k in NEEDS if not hasattr(a, k)))
        if not isinstance(a.ID, str) or not ID.fullmatch(a.ID) or a.ID in TAKEN:
            raise ValueError(f"{e.name}: bad ID {a.ID!r}")
        out.append(a)
    return out


def load() -> list:
    """The add-on objects of the listed paths, each ID once. Read once per process (the desktop app starts a new runtime
    after every update, so a new add-on is always read fresh); the result is kept for the life of the process."""
    global _loaded
    if _loaded is not None:
        return _loaded
    out, probs = [], []
    for root in _paths():
        try:
            got = _from(root)
        except Exception as e:  # noqa: BLE001  (named in problems(); the app refuses to start on it)
            probs.append(f"{root}: {e}")
            continue
        for a in got:
            if any(x.ID == a.ID for x in out):
                probs.append(f"{root}: add-on ID {a.ID!r} is already taken by another add-on")
            else:
                out.append(a)
    for m in probs:
        sys.stderr.write(f"add-on not loaded: {m}\n")
    _problems[:] = probs
    _loaded = out
    return out


def problems() -> list:
    """What kept a listed add-on from loading (empty when every listed add-on loaded)."""
    load()
    return list(_problems)


def ids() -> tuple:
    return tuple(a.ID for a in load())


def by_id(i):
    return next((a for a in load() if a.ID == i), None)


def for_path(path: str):
    """The add-on whose routes hold this path: /<ID>/..., /api/<ID> or /api/<ID>/..., else None."""
    m = re.match(r"/(?:api/)?([a-z]+)(?:/|$)", path)
    a = by_id(m.group(1)) if m and m.group(1) not in TAKEN else None
    if a and (path.startswith(f"/{a.ID}/") or path == f"/api/{a.ID}" or path.startswith(f"/api/{a.ID}/")):
        return a
    return None


def page_for(path: str):
    """The add-on whose page is at this path (/m/<ID> or /m/<ID>/), else None."""
    m = re.fullmatch(r"/m/([a-z]+)/?", path)
    return by_id(m.group(1)) if m else None


def prefixes() -> tuple:
    """The path prefixes of the add-ons' own routes (the guard asks for the launch token on them)."""
    return tuple(f"/{i}/" for i in ids())


def asset(rel: str):
    for a in load():
        hit = a.asset(rel)
        if hit:
            return hit
    return None


def module_of(url: str, key=None):
    """The ID of the add-on a server at this address belongs to, else None."""
    for a in load():
        try:
            if a.is_server(url, key):
                return a.ID
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"add-on {a.ID}: {e}\n")
    return None


def views(app_state: dict) -> list:
    """The sidebar entries for /api/state: id, label, href, plus what each add-on adds."""
    out = []
    for a in load():
        extra = {}
        if hasattr(a, "state"):
            try:
                extra = dict(a.state(app_state) or {})
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"add-on {a.ID}: {e}\n")
        out.append(dict(extra, id=a.ID, label=a.LABEL, href=f"/m/{a.ID}/"))
    return out


# ------------------------------------------------------------------------------------------------ TPU rows of add-ons
ZONE = re.compile(r"[a-z]+-[a-z]+[0-9]+-[a-z]")                       # as mashbox.tpu.ZONE
TPU_ID = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+){1,8}")


def _tpu_row(a, row) -> dict:
    """One TPU row an add-on offers, checked; a fresh copy with kind "addon" and module = the add-on's ID. ValueError
    when it is not usable: its id must start with "<ID>-" (so it never takes a catalog name), cloud must name the chip,
    runtime, zones, a port from 1024 and a health route, and it may not carry a "run" command or coding-agent settings
    (the chip runs the add-on's own payload, tpu_payload())."""
    if not isinstance(row, dict):
        raise ValueError("a TPU row must be a dict")
    rid, c = row.get("id"), row.get("cloud")
    if not (isinstance(rid, str) and TPU_ID.fullmatch(rid) and rid.startswith(a.ID + "-")):
        raise ValueError(f"TPU row id {rid!r} must start with {a.ID + '-'!r} (lowercase letters, digits, dashes)")
    z, port, disk = (c or {}).get("zones"), (c or {}).get("port"), (c or {}).get("disk_gb")
    ok = (isinstance(c, dict) and isinstance(c.get("accelerator"), str) and isinstance(c.get("runtime"), str)
          and isinstance(z, list) and z and all(isinstance(x, str) and ZONE.fullmatch(x) for x in z)
          and type(port) is int and 1024 <= port <= 65535 and type(disk) is int and disk >= 0
          and isinstance(c.get("health"), str) and c["health"].startswith("/") and "agents" not in c)
    if not ok or "run" in row or not isinstance(row.get("name"), str):
        raise ValueError(f"{rid}: needs a name and cloud accelerator, runtime, zones, port (1024-65535), health (/...) "
                         "and disk_gb; no run command and no agents")
    return dict(json.loads(json.dumps(row)), kind="addon", module=a.ID)


def tpu_rows() -> list:
    """The TPU rows the loaded add-ons offer, each checked by _tpu_row. A bad or repeated row is left out (named on
    stderr); the catalog's own rows always come first wherever a row is looked up by id."""
    out = []
    for a in load():
        rows = getattr(a, "TPU", None) or []
        for row in rows if isinstance(rows, (list, tuple)) else []:
            try:
                r = _tpu_row(a, row)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"add-on {a.ID}: {e}\n")
                continue
            if any(x["id"] == r["id"] for x in out):
                sys.stderr.write(f"add-on {a.ID}: TPU row {r['id']} is listed twice\n")
                continue
            out.append(r)
    return out


def tpu_payload(row: dict, tmp: str) -> dict:
    """What a chip needs for an add-on's TPU row (a row from tpu_rows()), from that add-on: {"code": its .tar, a regular
    file inside tmp; "data": [regular files on this machine, never links]; "args": [str]}. ValueError otherwise."""
    a = by_id(row.get("module") or "")
    if a is None or row.get("kind") != "addon" or not callable(getattr(a, "tpu_payload", None)):
        raise ValueError(f"{row.get('id')}: no add-on of this app gives files for it")
    got = a.tpu_payload(row["id"], tmp)
    if not isinstance(got, dict):
        raise ValueError("tpu_payload must return a dict")
    code, data, args = got.get("code"), got.get("data") or [], got.get("args") or []
    if not (isinstance(code, str) and code.endswith(".tar") and _inside(code, tmp) and os.path.isfile(code)
            and not os.path.islink(code)):
        raise ValueError("its code must be a .tar file written inside the folder it was given")
    if not (isinstance(data, list) and all(isinstance(f, str) and os.path.isfile(f) and not os.path.islink(f)
                                           for f in data)):
        raise ValueError("each data file must be a regular file on this machine (not a link)")
    if not (isinstance(args, list) and all(isinstance(x, str) for x in args)):
        raise ValueError("args must be a list of strings")
    return {"code": code, "data": list(data), "args": list(args)}
