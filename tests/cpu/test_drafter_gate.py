"""The drafter add-on's declaration and default rule (INTERFACES-v1 §10.1, runtime/drafter.py, cli.speculation):
speculation turns on without a flag only where the drafter passed its gate on this chip for this deployment's
component_set_sha; --mtp 0 is always off; --mtp K forces it on and says when it is not gated; the draft file is fetched
only when speculation is on; the drafter is part of the deployment key; the engine refuses another drafter ABI."""
import argparse
import copy
import json
import os

import pytest

from mashbox import cli, hub
from mashbox.runtime import drafter as DR
from mashbox.models import qwen3_8_27b as QM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def qwen_row():
    d = json.load(open(os.path.join(ROOT, "src", "mashbox", "models.json")))
    return copy.deepcopy(next(m for m in d["models"] if m["id"] == "qwen3-8-27b"))


def test_models_json_declares_the_mtp_drafter_and_no_unrecorded_pass():
    m = qwen_row()
    dr = DR.declared(m)
    assert dr["kind"] == "mtp" and dr["k_default"] == 2 and dr["k_max"] == 3 and dr["version"] == 1
    assert dr["sha256"] == hub.draft_row(m)["sha256"] and len(dr["sha256"]) == 64 and dr["revision"]
    for chip, g in dr["gate"].items():      # a pass names the record, engine commit and component set it was made on
        assert g["status"] in ("passed", "not_passed"), chip
        if g["status"] == "passed":
            assert g.get("record") and g.get("engine_commit") and g.get("component_set_sha"), chip
    for chip in dr.get("k_table") or {}:   # a measured draft-length table is valid for every chip it names
        DR.k_table(dr, chip)
    root = json.load(open(os.path.join(ROOT, "models.json")))
    assert next(x for x in root["models"] if x["id"] == "qwen3-8-27b")["drafter"] == m["drafter"]


def test_declared_needs_a_draft_file_row():
    m = qwen_row()
    assert DR.declared({k: v for k, v in m.items() if k != "drafter"}) is None
    m["drafter"]["file"] = "missing.bshj"
    with pytest.raises(ValueError):
        DR.declared(m)


def test_key_changes_with_each_drafter_field():
    dr = DR.declared(qwen_row())
    k0 = DR.key(dr)
    assert DR.key(None) is None and DR.key(dict(dr)) == k0
    for f, v in (("sha256", "0" * 64), ("version", 2), ("id", "other"), ("k_default", 1), ("k_max", 2), ("kind", "draft_model"),
                 ("k_table", {"64": 1})):
        assert DR.key(dict(dr, **{f: v})) != k0, f


def gated(dr, chip="v6e-1", css="c" * 64, status="passed"):
    dr = copy.deepcopy(dr)
    dr["gate"] = {chip: dict(status=status, record="release/engine/cert/x", engine_commit="abc", component_set_sha=css)}
    return dr


def test_gate_rule():
    dr = DR.declared(qwen_row())
    assert DR.gate(None, "v6e-1", "c" * 64)[0] is False
    assert DR.gate(dr, "v6e-1", "c" * 64)[0] is False                          # no record for this chip
    assert DR.gate(gated(dr, status="not_passed"), "v6e-1", "c" * 64)[0] is False
    assert DR.gate(gated(dr), None, "c" * 64)[0] is False                      # off TPU
    assert DR.gate(gated(dr), "v6e-4", "c" * 64)[0] is False                   # another chip
    assert DR.gate(gated(dr), "v6e-1", None)[0] is False                       # deployment key not known
    assert DR.gate(gated(dr), "v6e-1", "d" * 64)[0] is False                   # another component set
    assert DR.gate(gated(dr), "v6e-1", "c" * 64)[0] is True


@pytest.fixture
def run(monkeypatch):
    """speculation() with the chip, the download and the MTP weights stubbed; -> (call(m, mtp, tp, css), fetched, logs)"""
    from mashbox.models.qwen3_8_27b import spec as QSP
    fetched, logs = [], []
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    monkeypatch.setattr(cli, "log", lambda s: logs.append(s))
    monkeypatch.setattr(QSP, "load_mtp", lambda path, cfg, log=None: ({"path": path}, {}))

    def fetch(m, path=None, log=None):
        fetched.append(path)
        if m.get("_fail"):
            raise OSError("offline")
        return path or "/draft.bshj"
    monkeypatch.setattr(hub, "draft_file", fetch)

    def call(m, mtp=None, tp=1, css=None, mtp_file=None, model=QM):
        a = argparse.Namespace(mtp=mtp, mtp_file=mtp_file)
        return cli.speculation(a, m, None, tp, component_set_sha=css, model=model)
    return call, fetched, logs


def test_no_flag_off_until_gated(run):
    call, fetched, logs = run
    m = qwen_row()
    assert call(m) == {} and fetched == [] and "not gated on this chip (v6e-1)" in logs[-1]


def test_no_flag_on_when_gated_for_this_component_set(run):
    call, fetched, logs = run
    m = qwen_row()
    dr = DR.declared(m)
    css = lambda key: "c" * 64 if key == DR.key(dr) else "x" * 64
    m["drafter"] = {k: v for k, v in gated(dr).items() if k in m["drafter"]}
    out = call(m, css=css)
    assert out["draft_k"] == 2 and fetched == [None]
    assert json.loads(logs[-1])["gated"] is True
    m2 = copy.deepcopy(m)
    m2["drafter"]["gate"]["v6e-1"]["component_set_sha"] = "d" * 64          # passed for another component set
    assert call(m2, css=css) == {}
    assert call(m, css=None) == {}                                          # deployment key not known: off
    assert call(m, css=css, tp=4) == {}                                     # tensor parallel: off, no error
    m3 = dict(m, _fail=True)
    assert call(m3, css=css) == {} and "could not be fetched" in logs[-1]   # a failed download serves without it


def test_flags(run):
    call, fetched, logs = run
    m = qwen_row()
    assert call(m, mtp=0) == {} and fetched == []
    out = call(m, mtp=3)
    assert out["draft_k"] == 3 and "speculation: not gated on this chip (v6e-1)" in logs
    assert json.loads(logs[-1])["gated"] is False
    with pytest.raises(SystemExit):
        call(m, mtp=4)
    with pytest.raises(SystemExit):
        call(m, mtp=2, tp=4)
    with pytest.raises(SystemExit):
        call(m, mtp_file="/x.bshj")
    with pytest.raises(SystemExit):
        call(dict(m, _fail=True), mtp=2)


def test_k_table_is_this_chips_measured_table():
    dr = DR.declared(qwen_row())
    assert DR.k_table(dr, "v6e-1") == {} and DR.k_table(None, "v6e-1") == {} and DR.k_table(dr, None) == {}
    dr = dict(dr, k_table={"v6e-1": {"1": 2, "8": 0}, "v5e-4": {"1": 1}})
    assert DR.k_table(dr, "v6e-1") == {1: 2, 8: 0} and DR.k_table(dr, "v5e-4") == {1: 1} and DR.k_table(dr, "v6e-4") == {}
    for bad in ({"v6e-1": {"8": 4}}, {"v6e-1": {"8": -1}}, {"v6e-1": {"0": 1}}, {"v6e-1": {"x": 1}}, {"v6e-1": [1]}):
        with pytest.raises(ValueError):
            DR.k_table(dict(dr, k_table=bad), "v6e-1")


def test_measured_k_table_reaches_the_engine(run):
    """Draft length per row bucket comes from the drafter's measured table for this chip (models.json), forced or gated.
    The MTP drafter needs every committed token, so a table with k = 0 at a row bucket never reaches its engine: gated,
    it decodes plain and fetches no draft file; forced (--mtp K), it runs K at every bucket; each says why."""
    call, fetched, logs = run
    m = qwen_row()
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 1}, "v5e-4": {"1": 1}}
    assert call(m, mtp=2)["k_table"] == {1: 2, 8: 1}
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 0}}
    assert call(m, mtp=2)["k_table"] == {} and "runs no row bucket at k = 0" in logs[-2]
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 1}, "v5e-4": {"1": 1}}
    dr = DR.declared(m)
    css = lambda key: "c" * 64 if key == DR.key(dr) else "x" * 64
    m["drafter"] = {k: v for k, v in gated(dr).items() if k in m["drafter"]}
    assert call(m, css=css)["k_table"] == {1: 2, 8: 1}
    n = len(fetched)
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 0}}
    dr = DR.declared(m)
    assert call(m, css=lambda key: "c" * 64) == {} and len(fetched) == n and "runs no row bucket at k = 0" in logs[-1]
    m["drafter"]["k_table"] = {"v5e-4": {"1": 1}}                            # no table for this chip: k everywhere
    assert call(m, mtp=2)["k_table"] == {}
    m["drafter"]["k_table"] = {"v6e-1": {"8": 4}}                            # above k_max
    with pytest.raises(SystemExit):
        call(m, mtp=2)


def test_download_fetches_the_draft_only_when_gated(monkeypatch, capsys):
    m, fetched = qwen_row(), []
    monkeypatch.setattr(cli, "files", lambda a: (m, "/up", "/w.bshj"))
    monkeypatch.setattr(cli, "log", lambda s: None)
    monkeypatch.setattr(hub, "draft_file", lambda m, path=None, log=None: fetched.append(1) or "/draft.bshj")
    a = argparse.Namespace(no_kernels=True)
    cli.cmd_download(a)
    assert fetched == [] and json.loads(capsys.readouterr().out)["draft"] is None
    m["drafter"] = {k: v for k, v in gated(DR.declared(m)).items() if k in m["drafter"]}
    cli.cmd_download(a)
    assert fetched == [1] and json.loads(capsys.readouterr().out)["draft"] == "/draft.bshj"


def test_engine_refuses_another_mtp_head_abi():
    from types import SimpleNamespace
    from mashbox.runtime.engine import Engine
    with pytest.raises(ValueError, match="MTP head ABI None, this engine 1"):
        Engine(SimpleNamespace(), {}, SimpleNamespace(make_forward=lambda *a: None), mtp=(SimpleNamespace(__name__="x"), {}),
               draft_k=2)


def test_engine_refuses_another_drafter_abi(tmp_path):
    """a drafter written for another drafter interface (runtime/drafters DRAFTER_ABI) is refused when the engine is
    built, before any program exists. A real tiny model (tools/gguf_fixtures llama-f32), on which the same drafter at
    this ABI builds, so only the ABI check can refuse it"""
    import importlib.util
    import jax.numpy as jnp
    from mashbox.gguf import open_model
    from mashbox.runtime.engine import Engine
    from mashbox.runtime import drafters as DRS
    from mashbox.runtime.drafters.ngram import NGram
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sp = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(root, "tools", "gguf_fixtures", "fixtures.py"))
    fx = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(fx)
    f, A, c = open_model(fx.build("llama-f32", str(tmp_path)), act=jnp.float32)
    W, _ = A.load(f, c, use_kernels=False)
    kw = dict(CTX=128, S=2, K=4, draft_k=3)

    class Old(NGram):
        abi = DRS.DRAFTER_ABI - 1
    assert NGram(3).abi == DRS.DRAFTER_ABI == 2
    assert Engine(c, W, A, drafter=NGram(3), **kw).drafter.abi == DRS.DRAFTER_ABI
    with pytest.raises(ValueError, match=f"drafter ABI {DRS.DRAFTER_ABI - 1}, this engine {DRS.DRAFTER_ABI}"):
        Engine(c, W, A, drafter=Old(3), **kw)


def test_gate_needs_k_above_zero_at_a_row_bucket(run):
    """a passed gate whose k_table for this chip has k = 0 at every row bucket (measured slower than plain there) is
    not eligible: no flag decodes plain and fetches no draft file; k > 0 at one bucket turns it on with that table"""
    call, fetched, logs = run
    m = qwen_row()
    m["drafter"] = {k: v for k, v in gated(DR.declared(m)).items() if k in m["drafter"]}
    css = lambda key: "c" * 64      # noqa: E731
    m["drafter"]["k_table"] = {"v6e-1": {"1": 0, "8": 0}}
    ok, why = DR.gate(DR.declared(m), "v6e-1", "c" * 64)
    assert not ok and "k = 0 at every row bucket" in why
    assert call(m, css=css) == {} and fetched == [] and "k = 0 at every row bucket" in logs[-1]
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 0}}
    assert DR.gate(DR.declared(m), "v6e-1", "c" * 64)[0]          # eligible, but the MTP drafter runs no bucket at k = 0
    assert call(m, css=css) == {} and fetched == [] and "runs no row bucket at k = 0" in logs[-1]
    m["drafter"]["k_table"] = {"v6e-1": {"1": 2, "8": 1}}
    assert call(m, css=css)["k_table"] == {1: 2, 8: 1} and fetched == [None]


def test_mtp_head_is_the_module_the_model_declares(run):
    """the MTP head comes from the module the model's architecture declares (SPEC_MTP_HEAD), never a fixed import: a
    model that declares none decodes plain without a flag, and --mtp K says why"""
    from types import SimpleNamespace
    call, fetched, logs = run
    m = qwen_row()
    m["drafter"] = {k: v for k, v in gated(DR.declared(m)).items() if k in m["drafter"]}
    css = lambda key: "c" * 64      # noqa: E731
    assert call(m, css=css, model=SimpleNamespace()) == {} and fetched == [] and "SPEC_MTP_HEAD" in logs[-1]
    with pytest.raises(SystemExit, match="SPEC_MTP_HEAD"):
        call(m, mtp=2, css=css, model=SimpleNamespace())
    assert fetched == [] and call(m, css=css)["draft_k"] == 2         # Qwen3.8-27B declares its head


def test_gated_css_is_the_component_set_the_gate_passed_for():
    dr = DR.declared(qwen_row())
    assert DR.gated_css(gated(dr), "v6e-1") == "c" * 64
    assert DR.gated_css(gated(dr, status="not_passed"), "v6e-1") is None
    assert DR.gated_css(gated(dr), "v6e-4") is None                            # another chip
    assert DR.gated_css(gated(dr), None) is None                               # off TPU
    assert DR.gated_css(dr, "v6e-1") is None and DR.gated_css(None, "v6e-1") is None


class StandIn:
    """The engine fields compile_key.for_engine reads, and drop_speculation (as runtime/engine.py's: speculation off)."""
    def __init__(self, DK):
        self.kernels, self.CTX, self.S, self.K, self.chunk, self.tokens_per_call = [], 4096, 8, 8, 512, 4096
        self.attn_bytes, self.budget_busy, self.budget_idle, self.prefix_cache, self.host = 2e9, 1, 1, True, None
        self.kv_block, self.trace, self.DK, self.NR, self.dropped = 256, False, DK, 1, 0

    def drop_speculation(self):
        self.DK, self.NR, self.dropped = 0, 1, self.dropped + 1


def test_load_keeps_speculation_from_the_gate_only_for_its_own_component_set(run):
    """cli.load: speculation(later=True) turns on from the gate's recorded component_set_sha (gate_css) before the engine
    exists; deployment_keys() then keeps it only when the built engine's component_set_sha (drafter on) is that one, else
    turns it off before the warm-up and makes the keys again without the drafter. A forced --mtp K is never dropped."""
    from mashbox.runtime import compile_key as CK
    call, fetched, logs = run
    m = qwen_row()
    dr = DR.declared(m)
    on = CK.for_engine(StandIn(2), m["id"], "cfg", "lay", tp=1, drafter=DR.key(dr))[2]
    plain = CK.for_engine(StandIn(0), m["id"], "cfg", "lay", tp=1, drafter=None)[2]
    assert on != plain
    spec = lambda mm, mtp=None: cli.speculation(argparse.Namespace(mtp=mtp, mtp_file=None), mm, None, 1, later=True, model=QM)
    for css, keep in ((on, True), ("d" * 64, False)):
        mm = copy.deepcopy(m)
        mm["drafter"] = {k: v for k, v in gated(dr, css=css).items() if k in mm["drafter"]}
        out = spec(mm)
        assert out["gate_css"] == css and out["draft_k"] == 2                 # on from the gate, before the engine exists
        eng = StandIn(out["draft_k"])
        cli.deployment_keys(eng, mm, "cfg", "lay", tp=1, gate_css=out["gate_css"])
        if keep:
            assert eng.dropped == 0 and eng.DK == 2 and eng.component_set_sha == on
        else:
            assert eng.dropped == 1 and eng.DK == 0 and eng.component_set_sha == plain
            assert json.loads(logs[-1]) == dict(event="speculation", used=False, reason="gated on v6e-1 for another component set")
    assert spec(m) == {}                                                    # no gate entry: off, no gate_css
    assert "gate_css" not in spec(m, mtp=2)                                 # forced: never from the gate
    eng = StandIn(2)
    cli.deployment_keys(eng, m, "cfg", "lay", tp=1, gate_css=None)
    assert eng.dropped == 0 and eng.component_set_sha == on                 # the drafter is in the key while it runs
    assert spec(m, mtp=0) == {}


def test_mtp_draft_k_takes_1_to_k_max(run):
    """--speculation mtp --draft-k K takes 1 to the head's k_max (docs/speculation.md): 0 and k_max + 1 stop with that
    range (0 is never read as off); 1 and k_max turn the head on with that K"""
    from mashbox import cli
    from mashbox.models import qwen3_8_27b as Q
    m = qwen_row()
    kmax = DR.declared(m)["k_max"]
    ns = lambda k: argparse.Namespace(speculation="mtp", draft_k=k, mtp=None, mtp_file=None)      # noqa: E731
    for k in (0, kmax + 1):
        with pytest.raises(SystemExit, match=f"--draft-k {k}: use 1 to {kmax}"):
            cli.speculation(ns(k), m, None, 1, model=Q)
    for k in (1, kmax):
        assert cli.speculation(ns(k), m, None, 1, model=Q)["draft_k"] == k
