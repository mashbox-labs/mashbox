"""Add-on modules (mashbox.ui.app.addons): the app finds an add-on the desktop app carries, lists it in the sidebar
state, serves its page and files, files a server that belongs to it, and answers its routes only after the guard and
the launch token; a broken add-on is left out and the app runs on."""
import json
import os
import pathlib
import re
import types

import pytest

from mashbox.ui.app import addons

from test_app_security import EVIL, ctx, req, serve  # noqa: F401  (ctx is a fixture)


def demo(seen):
    a = types.SimpleNamespace(ID="demo", LABEL="Demo")
    a.page = lambda reg=None: b"<!doctype html><html><head><title>Demo</title></head><body></body></html>"
    a.asset = lambda path: (b"x=1", "text/javascript; charset=utf-8") if path.endswith("ui/demo/demo.js") else None
    a.is_server = lambda url, key=None: url.endswith(":1")
    a.state = lambda s: {"pick": s.get("demo")}

    def get(h, path, q=None):
        seen.append(("GET", path))
        if path == "/api/demo" or path.startswith("/demo/"):
            h.send(200, {"ok": path})
            return True
        return False

    def post(h, path):
        seen.append(("POST", path))
        h.send(200, {"posted": path})
        return True
    a.get, a.post = get, post
    return a


@pytest.fixture
def actx(ctx, monkeypatch):
    seen = []
    monkeypatch.setattr(addons, "_loaded", [demo(seen)])
    ctx.seen = seen
    return ctx


def test_addon_page_files_state_and_routes(actx):
    st, h, body = req(actx.port, "GET", "/m/demo/")
    assert st == 200 and b"<title>Demo</title>" in body and b'name="mbx-token"' in body
    assert "frame-ancestors 'self'" in h["content-security-policy"]
    assert req(actx.port, "GET", "/ui/demo/demo.js")[0] == 200
    s = json.loads(req(actx.port, "GET", "/api/state", headers=actx.auth)[2])
    assert s["addons"] == [{"pick": None, "id": "demo", "label": "Demo", "href": "/m/demo/"}]
    assert json.loads(req(actx.port, "GET", "/demo/x", headers=actx.auth)[2]) == {"ok": "/demo/x"}
    assert json.loads(req(actx.port, "GET", "/api/demo", headers=actx.auth)[2]) == {"ok": "/api/demo"}
    assert json.loads(req(actx.port, "POST", "/demo/y", "{}", actx.json)[2]) == {"posted": "/demo/y"}
    assert req(actx.port, "GET", "/api/demo/nope", headers=actx.auth)[0] == 404


def test_addon_routes_need_the_guard_and_token(actx):
    assert req(actx.port, "GET", "/demo/x")[0] == 401                                  # no launch token
    assert req(actx.port, "GET", "/api/demo")[0] == 401
    assert req(actx.port, "GET", "/demo/x", headers=dict(actx.auth, Origin=EVIL))[0] == 403
    assert req(actx.port, "POST", "/demo/y", "{}", dict(actx.json, Origin=EVIL))[0] == 403
    assert req(actx.port, "POST", "/demo/y", "{}", {k: v for k, v in actx.json.items() if k != "Content-Type"})[0] in (401, 415)
    assert actx.seen == []


def test_a_server_of_an_addon_is_filed_under_it(actx):
    st, _, body = req(actx.port, "POST", "/api/servers/add", json.dumps({"name": "d", "url": "http://127.0.0.1:1"}), actx.json)
    assert st == 200
    s = json.loads(req(actx.port, "GET", "/api/state", headers=actx.auth)[2])
    assert [x["module"] for x in s["servers"]] == ["demo"] and s["active"] is None


MOD = """
import pathlib, os
if os.environ.get("ADDON_MARK"):
    pathlib.Path(os.environ["ADDON_MARK"], __name__).write_text("ran")
ID, LABEL = {ID!r}, "X"
def page(reg=None): return b"<!doctype html>"
def asset(path): return None
def is_server(url, key=None): return False
def get(h, path, q=None): return False
def post(h, path): return False
{extra}
"""


def folder(tmp, pkg, ID, extra="", ep=None):
    """A folder holding one distribution with one mashbox.app_addons entry point (a folder works like a wheel)."""
    root = tmp / f"{pkg}-root"
    (root / pkg).mkdir(parents=True)
    (root / pkg / "__init__.py").write_text(MOD.format(ID=ID, extra=extra))
    di = root / f"{pkg}-0.1.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {pkg}\nVersion: 0.1.0\n")
    (di / "entry_points.txt").write_text(f"[mashbox.app_addons]\n{ID} = {ep or pkg}\n")
    return root


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(addons, "_loaded", None)
    monkeypatch.setattr(addons, "_problems", [], raising=False)
    monkeypatch.delenv(addons.ENV, raising=False)
    (tmp_path / "marks").mkdir()
    monkeypatch.setenv("ADDON_MARK", str(tmp_path / "marks"))
    return tmp_path


def test_only_the_listed_addons_load_never_an_installed_package(fresh, monkeypatch):
    """An add-on gets a same-origin page with the launch token, so a package installed in the app's Python
    (or anywhere on sys.path) is never one; only the paths the desktop app lists are read."""
    monkeypatch.syspath_prepend(str(folder(fresh, "outsider", "evil")))
    assert addons.load() == [] and addons.problems() == []
    assert not (fresh / "marks" / "outsider").exists()                      # its code never ran
    monkeypatch.setenv(addons.ENV, str(folder(fresh, "listed", "good")))
    monkeypatch.setattr(addons, "_loaded", None)                            # a new process
    assert [a.ID for a in addons.load()] == ["good"] and addons.problems() == []
    assert not (fresh / "marks" / "outsider").exists()


def test_addon_code_that_would_come_from_elsewhere_is_refused_unrun(fresh, monkeypatch):
    other = fresh / "other"
    (other / "shadowed").mkdir(parents=True)
    (other / "shadowed" / "__init__.py").write_text(MOD.format(ID="shadow", extra=""))
    monkeypatch.syspath_prepend(str(other))
    root = folder(fresh, "shadowed", "shadow")                              # same top-level name, earlier on sys.path
    monkeypatch.setenv(addons.ENV, str(root))
    assert addons.load() == []
    assert "would be imported from" in addons.problems()[0]
    assert not (fresh / "marks" / "shadowed").exists()


def test_every_listed_add_on_that_does_not_load_stops_the_start(fresh, monkeypatch, capsys):
    """Fail closed. A listed path that is missing, holds no add-on, breaks on import, or has a bad or taken
    ID is a problem, and the app refuses to start (exit 3, never a run without it)."""
    empty = fresh / "empty"
    empty.mkdir()
    paths = [fresh / "gone.whl", empty, folder(fresh, "broken", "broke", extra="1 / 0"),
             folder(fresh, "badid", "tpu"), folder(fresh, "good", "good"), folder(fresh, "twin", "good")]
    monkeypatch.setenv(addons.ENV, os.pathsep.join(map(str, paths)))
    assert [a.ID for a in addons.load()] == ["good"]
    probs = addons.problems()
    assert len(probs) == 5 and "not found" in probs[0] and "no mashbox.app_addons" in probs[1]
    assert "division by zero" in probs[2] and "bad ID 'tpu'" in probs[3] and "already taken" in probs[4]
    from mashbox.ui.app import main
    assert main(["--port", "0"]) == 3
    assert "cannot start" in capsys.readouterr().err


def test_taken_holds_every_route_word_of_the_app(monkeypatch):
    """An add-on ID can never shadow a page, file folder or /api call of the app itself."""
    src = (pathlib.Path(addons.__file__).parent / "__init__.py").read_text()
    words = set(re.findall(r'"/api/([a-z_]+)', src)) | set(re.findall(r'"/([a-z0-9_]+)/', src))
    words |= set(re.findall(r'"/m/([a-z]+)', src)) | {"shell", "chat", "app"}
    words -= {"opt", "usr"}                                                 # file system paths in the source, not routes
    assert words and words <= addons.TAKEN, sorted(words - addons.TAKEN)
    monkeypatch.setattr(addons, "_loaded", [types.SimpleNamespace(**dict(vars(demo([])), ID="tpu"))])
    assert addons.for_path("/api/tpu/status") is None and addons.for_path("/tpu/x") is None


def test_a_link_out_of_the_listed_folder_is_refused_unrun(fresh, monkeypatch):
    """The package of a listed folder linked to a place outside it is never imported."""
    outside = fresh / "outside"
    (outside / "linked").mkdir(parents=True)
    (outside / "linked" / "__init__.py").write_text(MOD.format(ID="linked", extra=""))
    root = folder(fresh, "linked", "linked")
    for f in (root / "linked").iterdir():
        f.unlink()
    (root / "linked").rmdir()
    (root / "linked").symlink_to(outside / "linked", target_is_directory=True)
    monkeypatch.setenv(addons.ENV, str(root))
    assert addons.load() == [] and "would be imported from" in addons.problems()[0]
    assert not (fresh / "marks" / "linked").exists()
    inner = folder(fresh, "inner", "inner")                                 # a link inside the package, pointing out
    (outside / "x.py").write_text("")
    (inner / "inner" / "x.py").symlink_to(outside / "x.py")
    monkeypatch.setattr(addons, "_loaded", None)
    monkeypatch.setenv(addons.ENV, str(inner))
    assert addons.load() == [] and "holds a link" in addons.problems()[0]
    assert not (fresh / "marks" / "inner").exists()


def test_a_namespace_package_is_refused_unrun(fresh, monkeypatch):
    """A listed folder whose top-level name is a namespace package (no __init__.py) joins every
    folder of that name on sys.path, so ns.plugin could load from outside the listed path; it is refused, unrun."""
    listed, outside = fresh / "listed", fresh / "outside"
    (listed / "ns").mkdir(parents=True)
    (listed / "ns" / "marker.txt").write_text("namespace location")
    di = listed / "listed-0.1.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: listed\nVersion: 0.1.0\n")
    (di / "entry_points.txt").write_text("[mashbox.app_addons]\nevil = ns.plugin\n")
    (outside / "ns").mkdir(parents=True)
    (outside / "ns" / "plugin.py").write_text(MOD.format(ID="evil", extra=""))
    monkeypatch.setattr(addons.sys, "path", [*addons.sys.path, str(outside)])  # outside, after the listed path
    monkeypatch.setenv(addons.ENV, str(listed))
    assert addons.load() == [] and "namespace package" in addons.problems()[0]
    assert not (fresh / "marks" / "ns.plugin").exists()                     # its code never ran
    alone = fresh / "alone"                                                 # a namespace only inside the listed path
    (alone / "solo").mkdir(parents=True)
    (alone / "solo" / "plugin.py").write_text(MOD.format(ID="solo", extra=""))
    di = alone / "solo-0.1.0.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: solo\nVersion: 0.1.0\n")
    (di / "entry_points.txt").write_text("[mashbox.app_addons]\nsolo = solo.plugin\n")
    monkeypatch.setattr(addons, "_loaded", None)
    monkeypatch.setenv(addons.ENV, str(alone))
    assert addons.load() == [] and "namespace package" in addons.problems()[0]
    assert not (fresh / "marks" / "solo.plugin").exists()
    regular = folder(fresh, "regular", "regular")                           # a regular package still loads
    monkeypatch.setattr(addons, "_loaded", None)
    monkeypatch.setenv(addons.ENV, str(regular))
    assert [a.ID for a in addons.load()] == ["regular"] and addons.problems() == []


def _dist(root, name, ep):
    di = root / f"{name}-0.1.0.dist-info"
    di.mkdir(parents=True)
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1.0\n")
    (di / "entry_points.txt").write_text(f"[mashbox.app_addons]\n{ep}\n")


def test_a_second_distribution_in_a_listed_path_is_refused_unrun(fresh, monkeypatch):
    """A listed path with a hidden second metadata folder (its own entry point) is refused."""
    root = folder(fresh, "good", "good", extra="")
    (root / "evilmod.py").write_text(MOD.format(ID="evil", extra=""))
    _dist(root, "hidden", "evil = evilmod")
    monkeypatch.setenv(addons.ENV, str(root))
    assert addons.load() == [] and "2 distributions" in addons.problems()[0]
    assert not (fresh / "marks" / "evilmod").exists() and not (fresh / "marks" / "good").exists()


def test_a_cached_module_from_elsewhere_is_refused(fresh, monkeypatch):
    """Package safe is inside the listed path, but safe.plugin is already loaded from outside it; e.load()
    would hand that copy back, so the add-on is refused (and nothing of it is used)."""
    import importlib.util
    import sys
    listed, outside = fresh / "listed", fresh / "outside"
    (listed / "safe").mkdir(parents=True)
    (listed / "safe" / "__init__.py").write_text("")
    (listed / "safe" / "plugin.py").write_text(MOD.format(ID="good", extra=""))
    _dist(listed, "listed", "evil = safe.plugin")
    outside.mkdir()
    (outside / "plugin.py").write_text(MOD.format(ID="evil", extra=""))
    spec = importlib.util.spec_from_file_location("safe.plugin", outside / "plugin.py")
    cached = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cached)
    monkeypatch.delitem(sys.modules, "safe", raising=False)
    monkeypatch.setitem(sys.modules, "safe.plugin", cached)
    monkeypatch.setenv(addons.ENV, str(listed))
    assert addons.load() == [] and "already loaded from" in addons.problems()[0]
    monkeypatch.delitem(sys.modules, "safe", raising=False)


def test_an_entry_point_object_from_elsewhere_is_refused(fresh, monkeypatch):
    """The entry point's object must be this add-on's own: safe:thing where safe re-exports a module from outside."""
    import sys
    listed = fresh / "listed2"
    (listed / "safe2").mkdir(parents=True)
    (listed / "safe2" / "__init__.py").write_text("import json as thing\n")
    _dist(listed, "listed2", "evil = safe2:thing")
    monkeypatch.setenv(addons.ENV, str(listed))
    assert addons.load() == [] and "not from this add-on" in addons.problems()[0]
    monkeypatch.delitem(sys.modules, "safe2", raising=False)
