"""The speculation options as docs/speculation.md states them: --draft-k and --speculation help, --speculation draft
stopping before any file work on both load paths, and a drafter the user picks never reading the auto gate."""
import argparse
import importlib.util
import os
from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mashbox import cli, hub  # noqa: E402
from mashbox.runtime import drafter as DR  # noqa: E402
from mashbox.runtime.drafters import select as SEL  # noqa: E402
from mashbox.runtime.spec_step import SPEC_STEP_VERSION  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(FXT)


def run_help(flag):
    p = cli.build_parser()
    sub = next(x for x in p._actions if isinstance(x, argparse._SubParsersAction)).choices["run"]
    return next(x for x in sub._actions if flag in x.option_strings).help


def test_draft_k_help_gives_each_drafters_range():
    """--draft-k: n-gram 1 to 8, MTP 1 to the head's k_max (Qwen3.8-27B 1 to 3), not one range for every drafter"""
    h = run_help("--draft-k")
    assert "ngram 1 to 8" in h and "mtp 1 to the head's k_max" in h and "Qwen3.8-27B: 1 to 3" in h, h


def test_speculation_help_says_what_auto_reads():
    """--speculation: auto on Qwen3.8-27B reads its MTP head only, so the help never says auto falls back to n-gram there"""
    h = run_help("--speculation")
    assert "else n-gram" not in h and "Qwen3.8-27B reads its MTP head only" in h, h


def test_mtp_0_is_off_and_long_form_0_is_refused(monkeypatch):
    """--mtp 0 turns speculation off; --speculation mtp --draft-k 0 stops with the 1 to k_max range"""
    from mashbox.models import qwen3_8_27b as Q
    monkeypatch.setattr(cli, "log", lambda *a: None)
    m = hub.model_entry("qwen3-8-27b")
    kmax = DR.declared(m)["k_max"]
    a = argparse.Namespace(mtp=0, mtp_file=None)
    assert cli.speculation(a, m, None, 1, model=Q) == {}
    with pytest.raises(SystemExit, match=f"--draft-k 0: use 1 to {kmax}"):
        cli.speculation(argparse.Namespace(speculation="mtp", draft_k=0, mtp=None, mtp_file=None), m, None, 1, model=Q)


class Touched(Exception):
    pass


def sentinel(name, seen):
    def f(*a, **k):
        seen.append(name)
        raise Touched(name)
    return f


def test_draft_stops_before_any_file_on_the_models_json_path(monkeypatch):
    """cli.load with --speculation draft stops with its line before the config, cache, fit check or weights are touched"""
    seen = []
    for name in ("model_entry", "upstream_dir", "weights_file", "compile_cache", "kernel_bundle", "draft_file"):
        monkeypatch.setattr(hub, name, sentinel(name, seen))
    for name in ("setup_cache", "tensor_parallel", "fit_check"):
        monkeypatch.setattr(cli, name, sentinel(name, seen))
    a = cli.build_parser().parse_args(["run", "qwen3-8-27b", "--speculation", "draft", "--prompt", "hi"])
    with pytest.raises(SystemExit, match="--speculation draft: draft models are not in this version"):
        cli.load(a)
    assert seen == []


def test_draft_stops_before_the_gguf_file_is_fetched_or_loaded(monkeypatch, tmp_path, capsys):
    """a GGUF run with --speculation draft stops (exit 2, one line) before the file is resolved or downloaded, and
    load_gguf itself stops before the file is opened"""
    from mashbox import gguf as G
    from mashbox.gguf import resolve as GR
    seen = []
    logs = []
    monkeypatch.setattr(cli, "log", lambda s: logs.append(s))
    monkeypatch.setattr(GR, "resolve", sentinel("resolve", seen))
    monkeypatch.setattr(G, "open_model", sentinel("open_model", seen))
    a = SimpleNamespace(cmd="run", model="x.gguf", weights=None, mtp=None, mtp_file=None, speculation="draft",
                        host="127.0.0.1")
    assert cli.cmd_gguf(a) == 2
    assert seen == [] and "--speculation draft: draft models are not in this version" in logs[-1], logs
    with pytest.raises(G.GGUFError, match="--speculation draft: draft models are not in this version"):
        cli.load_gguf(SimpleNamespace(gguf=dict(path=str(tmp_path / "x.gguf"), repo=None, tag=None, file="x.gguf"),
                                      speculation="draft"))
    assert seen == []


def entry(css, k_table):
    return dict(status="passed", record="release/engine/cert/r", record_sha256="e" * 64, component_set_sha=css,
                spec_step_version=SPEC_STEP_VERSION, k_table=k_table)


@pytest.mark.parametrize("dk", (None, 5))
def test_a_forced_gguf_drafter_never_reads_the_auto_gate(tmp_path, monkeypatch, dk):
    """--speculation ngram on a GGUF file gives the same drafter, K and k_table with no record, a passing record or a
    record for another component set: never gated, no gate_css, so the built-set check can never turn it off. auto with
    the same passing record is gated and takes the record's k_table (the record matters there only)."""
    from mashbox.gguf import open_model
    monkeypatch.setattr(cli, "log", lambda *a: None)
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    _, A, c = open_model(FXT.build("llama-f32", str(tmp_path)), act=jnp.float32)
    outs = []
    for rec in (None, {"ngram": {"v6e-1": entry("c" * 64, {"1": 2})}}, {"ngram": {"v6e-1": entry("d" * 64, {"1": 2})}}):
        out = cli.gguf_speculation(SimpleNamespace(speculation="ngram", draft_k=dk), c, A, records=rec)
        outs.append((out["draft_k"], out["k_table"], "gate_css" in out, type(out["drafter"]).__name__))
    assert outs == [(3 if dk is None else dk, {}, False, "NGram")] * 3, outs
    meta = SEL.meta_of(A, c, "gguf", getattr(c, "nextn", 0))
    rec = {"ngram": {"v6e-1": entry("c" * 64, {"1": 2})}}
    kind, g, why, gcss = SEL.pick(meta, "v6e-1", rec, forced="ngram", later=True)
    assert (kind, g, gcss) == ("ngram", False, None) and why[-1].endswith("speculation: not gated on this chip"), why
    auto = cli.gguf_speculation(SimpleNamespace(speculation=None, draft_k=dk), c, A, records=rec)
    assert auto["k_table"] and auto["gate_css"] == "c" * 64 and auto["draft_k"] == (2 if dk is None else dk), auto


def test_public_run_with_draft_stops_before_the_model_row_keys_and_server(monkeypatch, tmp_path):
    """SPECd-F01: the public `mashbox run <models.json model> --speculation draft` in server form (no --prompt, a
    non-loopback --host) stops with its one line and exit 2 before the model row is read, an API key is made or the
    server front starts; no key file appears"""
    from mashbox.runtime import front
    seen, logs = [], []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MASHBOX_API_KEY", raising=False)
    monkeypatch.setattr(cli, "log", lambda s: logs.append(s))
    monkeypatch.setattr(hub, "model_entry", sentinel("model_entry", seen))
    monkeypatch.setattr(cli, "api_keys", sentinel("api_keys", seen))
    monkeypatch.setattr(front, "serve_cli", sentinel("serve_cli", seen))
    for name in ("load", "load_gguf"):
        monkeypatch.setattr(cli, name, sentinel(name, seen))
    assert cli.main(["run", "qwen3-8-27b", "--speculation", "draft", "--host", "0.0.0.0"]) == 2
    assert seen == []
    assert logs == ["mashbox: --speculation draft: draft models are not in this version; use ngram or mtp"], logs
    assert not (tmp_path / ".config").exists()


def test_cmd_run_with_draft_stops_before_api_keys(monkeypatch):
    """SPECd-F01: cmd_run called directly with --speculation draft stops (exit 2, one line) before api_keys,
    the server front or a load"""
    from mashbox.runtime import front
    seen, logs = [], []
    monkeypatch.setattr(cli, "log", lambda s: logs.append(s))
    monkeypatch.setattr(cli, "api_keys", sentinel("api_keys", seen))
    monkeypatch.setattr(front, "serve_cli", sentinel("serve_cli", seen))
    for name in ("load", "load_gguf"):
        monkeypatch.setattr(cli, name, sentinel(name, seen))
    a = cli.build_parser().parse_args(["run", "qwen3-8-27b", "--speculation", "draft", "--host", "0.0.0.0"])
    assert cli.cmd_run(a, []) == 2
    assert seen == [] and len(logs) == 1 and "--speculation draft" in logs[0], (seen, logs)


# what the --speculation help and docs/speculation.md call each drafter kind (the help's phrase, the page's bold)
HELP_NAME = {"trained": "a trained drafter made for this model", "mtp": "the model's own MTP head",
             "draft_model": "a smaller draft model with the same tokenizer", "ngram": "n-gram over the request's own text"}
PAGE_NAME = {"trained": "A trained drafter made for this model", "mtp": "The model's own MTP head",
             "draft_model": "A smaller model with the same tokenizer", "ngram": "Copy from the conversation"}


def test_speculation_help_names_every_drafter_in_the_order_of_choice():
    """SPECd-F02: the --speculation help lists auto's order exactly as drafters.ORDER and docs/speculation.md do,
    numbered, every kind without a module in this release marked "not in this release", and nothing more"""
    import re
    from mashbox.runtime import drafters as D
    h = run_help("--speculation")
    items = re.findall(r"(\d) ([^;]+?)(?:;|, on only)", h.split("in this order: ", 1)[1])
    assert [int(n) for n, _ in items] == list(range(1, len(D.ORDER) + 1)), (items, h)
    for (n, text), kind in zip(items, D.ORDER):
        assert text.startswith(HELP_NAME[kind]), (kind, text)
        assert ("(not in this release)" in text) == (kind not in D.MODULES), (kind, text)
    page = open(os.path.join(ROOT, "docs", "speculation.md")).read().split("## How auto picks a drafter", 1)[1]
    listed = re.findall(r"^(\d)\. \*\*([^*]+)\*\*", page.split("\n## ", 1)[0], re.M)
    assert [int(n) for n, _ in listed] == list(range(1, len(D.ORDER) + 1)), listed
    for (n, bold), kind in zip(listed, D.ORDER):
        assert bold.startswith(PAGE_NAME[kind].rstrip(".")), (kind, bold)
