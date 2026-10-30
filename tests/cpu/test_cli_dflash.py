"""`mashbox run --speculation dflash --drafter-dir DIR` (cli.py), CPU, on the tiny random model of the Qwen3.8-27B
architecture from test_spec_in_step.py and the tiny DFlash 2 drafter folder from test_dflash_drafter.py:
  parse    the parser takes --speculation dflash and --drafter-dir DIR
  on       a pick you make: the DFlash drafter from that folder, draft_k 7 (its block of 8 less its first token) unless
           --draft-k 1 to 7 says otherwise; the log says 'speculation: not gated on this chip' and the speculation event
           (method dflash, gated false)
  refused  under tensor parallel, on a model whose forward declares no LAYER_TAPS, on a GGUF file, without --drafter-dir,
           --drafter-dir without --speculation dflash, --draft-k outside 1 to 7: each stops with its own line
  auto     never picks it: with no flag (or auto) nothing changes, and select.py builds no trained drafter in this version
  slots    at --slots' default (8) it runs 1 slot and says so; --slots N runs as given; other kinds keep 8
"""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("ml_dtypes")
jax = pytest.importorskip("jax")
pytest.importorskip("safetensors")

from mashbox import cli  # noqa: E402
from mashbox.models import qwen3_8_27b as Q  # noqa: E402
from mashbox.runtime.drafters import dflash as DF, select as SEL  # noqa: E402
from test_spec_in_step import tiny  # noqa: E402,F401  (the fixture)
from test_dflash_drafter import drafter_dir  # noqa: E402,F401  (the fixture)

M = dict(id="tiny")        # a catalog row with no drafter of its own


def args(**kw):
    a = dict(speculation=None, drafter_dir=None, draft_k=None, mtp=None, mtp_file=None, draft_model=None, gguf=None, slots=8)
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.fixture
def lines(monkeypatch):
    out = []
    monkeypatch.setattr(cli, "log", out.append)
    return out


def test_parser_takes_dflash_and_drafter_dir():
    a = cli.build_parser().parse_args(["run", "qwen3-8-27b", "--speculation", "dflash", "--drafter-dir", "/d/dflash2"])
    assert a.speculation == "dflash" and a.drafter_dir == "/d/dflash2"
    assert cli.build_parser().parse_args(["run", "qwen3-8-27b"]).drafter_dir is None


def test_dflash_on_from_the_folder(tiny, drafter_dir, lines):  # noqa: F811
    cfg, _, _ = tiny["random"]
    got = cli.speculation(args(speculation="dflash", drafter_dir=drafter_dir), M, cfg, 1, model=Q)
    assert set(got) == {"drafter", "draft_k"} and isinstance(got["drafter"], DF.DFlash)
    assert got["draft_k"] == 7 and got["drafter"].k_max == 7
    assert "speculation: not gated on this chip" in lines
    ev = json.loads(lines[-1])
    assert (ev["event"], ev["used"], ev["method"], ev["gated"], ev["draft_k"]) == ("speculation", True, "dflash", False, 7)
    got = cli.speculation(args(speculation="dflash", drafter_dir=drafter_dir, draft_k=3), M, cfg, 1, model=Q)
    assert got["draft_k"] == 3 and got["drafter"].k_max == 3


def test_dflash_refusals(tiny, drafter_dir, lines):  # noqa: F811
    cfg, _, _ = tiny["random"]

    def sp(tp=1, model=Q, **kw):
        return cli.speculation(args(**dict(dict(speculation="dflash", drafter_dir=drafter_dir), **kw)), M, cfg, tp, model=model)
    with pytest.raises(SystemExit, match="one chip"):
        sp(tp=4)

    def make_forward(cfg):          # defined here: this module declares no LAYER_TAPS
        return Q.make_forward(cfg)
    with pytest.raises(SystemExit, match="LAYER_TAPS"):
        sp(model=SimpleNamespace(make_forward=make_forward))
    for k in (0, 8):
        with pytest.raises(SystemExit, match="use 1 to 7"):
            sp(draft_k=k)
    with pytest.raises(SystemExit, match="needs --drafter-dir"):
        sp(drafter_dir=None)
    assert "GGUF" in cli.draft_refusal(args(speculation="dflash", drafter_dir=drafter_dir), gguf=True)
    assert "--drafter-dir" in cli.draft_refusal(args(speculation="ngram", drafter_dir=drafter_dir))
    assert "--drafter-dir" in cli.draft_refusal(args(drafter_dir=drafter_dir))
    assert cli.draft_refusal(args(speculation="dflash", drafter_dir=drafter_dir)) is None
    assert not any("used\": true" in x for x in lines)


@pytest.mark.parametrize("argv", [["run", "x.gguf", "--speculation", "dflash", "--drafter-dir", "/d"],
                                  ["run", "qwen3-8-27b", "--speculation", "dflash"],
                                  ["run", "qwen3-8-27b", "--drafter-dir", "/d"]])
def test_dflash_refused_before_anything_loads(argv, lines):
    assert cli.main(argv) == 2 and len(lines) == 1 and lines[0].startswith("mashbox: ")
    assert ("GGUF" in lines[0]) if argv[1].endswith(".gguf") else ("--drafter-dir" in lines[0])


def test_auto_never_picks_dflash(tiny, lines):  # noqa: F811
    cfg, _, _ = tiny["random"]
    for sv in (None, "auto"):
        assert cli.speculation(args(speculation=sv), M, cfg, 1, model=Q) == {}
    assert "dflash" not in SEL.BUILT, "auto would build the DFlash drafter: docs/speculation.md says it never does"


def test_dflash_slots(lines):
    a = args(speculation="dflash", drafter_dir="/d")
    cli.dflash_slots(a)
    assert a.slots == 1 and json.loads(lines[-1]) == dict(event="dflash_slots", slots=1, asked=8) and "--slots N" in lines[-2]
    for a in (args(speculation="dflash", drafter_dir="/d", slots=2), args(speculation="ngram"), args()):
        s = a.slots
        cli.dflash_slots(a)
        assert a.slots == s
