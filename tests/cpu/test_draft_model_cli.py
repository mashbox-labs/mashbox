"""The draft-model drafter from the command line (`--draft-model FILE`, `--speculation draft`) on a GGUF target, as
docs/speculation.md states it. Tiny f32 GGUF fixtures (tools/gguf_fixtures), CPU, no network.

  load      cli.load_gguf with --draft-model: the drafter is the draft model, its output equals the plain load's, and the
            deployment's component set holds the draft file's sha256 (another set than the plain load's)
  warm      0 speculation programs compiled after warm(), and no more compiles in all than the plain engine's
  refuse    one line, before any file is fetched: --speculation draft without --draft-model, --draft-model with another
            --speculation kind, a draft model on a models.json model (main, load, cmd_run); a draft file that cannot
            draft (not full attention in every layer) or has another tokenizer stops when it is opened
  select    draft_model applies only when a draft model is given; auto never picks it without one
"""
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import gguf  # noqa: E402,F401
import jax.numpy as jnp  # noqa: E402

from mashbox import cli, hub  # noqa: E402
from mashbox.runtime import drafter as DR  # noqa: E402
from mashbox.runtime.drafters import select as SEL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_fx = importlib.util.spec_from_file_location("gguf_fixtures", os.path.join(ROOT, "tools", "gguf_fixtures", "fixtures.py"))
FXT = importlib.util.module_from_spec(_fx)
_fx.loader.exec_module(FXT)
_COMPILES = [0]
jax.monitoring.register_event_duration_secs_listener(
    lambda ev, d, **k: _COMPILES.__setitem__(0, _COMPILES[0] + (ev == "/jax/core/compile/backend_compile_duration")))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    wd = tmp_path_factory.mktemp("draft_cli")
    target = FXT.build("llama-f32", str(wd))
    draft = str(wd / "qwen3-draft.gguf")
    FXT.write_dense(draft, arch="qwen3", seed=7, layers=2)      # another architecture of the dense module, same tokens
    return dict(target=target, draft=draft, qwen35=FXT.build("qwen35-f32", str(wd)))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MASHBOX_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MASHBOX_GGUF_KERNELS", "off")
    monkeypatch.setattr(DR, "chip_key", lambda: "v6e-1")
    monkeypatch.setattr(hub, "registry", lambda: {})
    lines = []
    monkeypatch.setattr(cli, "log", lambda s: lines.append(s))
    return lines


def args(tmp_path, target, *extra):
    from mashbox.gguf import pick as GR
    a = cli.build_parser().parse_args(["run", target, "--prompt", "hi", "--no-kernels", "--slots", "4", "--ctx", "256",
                                       "--cache-dir", str(tmp_path / "xla"), *extra])
    a.gguf = GR.resolve(target, log=cli.log)
    if getattr(a, "draft_model", None):
        a.draft_gguf = GR.resolve(a.draft_model, log=cli.log)
    return a


def loop_prompts(V, n, seed):
    """prompts that repeat a short phrase, so a draft model that copies a pattern drafts something"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ph = rng.integers(1, V, int(rng.integers(3, 7))).tolist()
        out.append((rng.integers(1, V, int(rng.integers(2, 6))).tolist() + ph * int(rng.integers(2, 4)))[:40])
    return out


def ids_of(res):
    ids = [[int(t) for t in r] for r in res["ids"]]
    assert ids and all(ids), ids
    return ids


def test_load_gguf_with_a_draft_model_equals_plain_and_keys_the_draft_file(files, env, tmp_path):
    from mashbox.runtime import tenancy
    plain = cli.load_gguf(args(tmp_path, files["target"]))[1]
    env.clear()
    spec = cli.load_gguf(args(tmp_path, files["target"], "--draft-model", files["draft"], "--draft-k", "3"))[1]
    ev = [json.loads(x) for x in env if x.startswith("{") and '"speculation"' in x]
    assert ev and ev[-1]["used"] is True and ev[-1]["method"] == "draft_model" and ev[-1]["gated"] is False, ev
    assert "speculation: not gated on this chip" in env
    assert spec.DK == 3 and spec.drafter.kind == "draft_model"
    key = spec.drafter.key()
    assert key["kind"] == "draft_model" and key["weights_sha256"] == tenancy.file_sha256(files["draft"]), key
    assert spec.component_set_sha != plain.component_set_sha
    ps = loop_prompts(plain.c.V, 4, 3)
    assert ids_of(spec.generate(ps, max_new=30)) == ids_of(plain.generate(ps, max_new=30))
    st = spec.spec_stats
    assert st["rounds"] > 0 and not st.get("faults"), st


def test_no_speculation_compiles_after_warm(files, env, tmp_path):
    plain = cli.load_gguf(args(tmp_path, files["target"]))[1]
    spec = cli.load_gguf(args(tmp_path, files["target"], "--draft-model", files["draft"]))[1]
    plain.warm(workers=1)
    spec.warm(workers=1)
    ps = loop_prompts(plain.c.V, 4, 9)        # one call takes at most --slots (4) prompts
    c0 = _COMPILES[0]
    plain.generate(ps, max_new=20)
    plain_n = _COMPILES[0] - c0
    before, c1 = (spec.sstep._cache_size(), spec.drafter.pf._cache_size()), _COMPILES[0]
    spec.generate(ps, max_new=20)
    total = _COMPILES[0] - c1
    print(f"draft-model compiles after warm: speculation engine {total}, plain engine {plain_n}; "
          f"rounds {spec.spec_stats['rounds']}, plain calls {spec.spec_stats['plain_calls']}")
    assert (spec.sstep._cache_size(), spec.drafter.pf._cache_size()) == before
    assert spec.spec_stats["rounds"] > 0
    assert total <= plain_n


def test_draft_without_a_draft_model_stops_before_the_file_is_fetched(monkeypatch, env):
    from mashbox.gguf import pick as GR
    seen = []
    monkeypatch.setattr(GR, "resolve", lambda *a, **k: seen.append("resolve"))
    a = SimpleNamespace(cmd="run", model="x.gguf", weights=None, mtp=None, mtp_file=None, speculation="draft",
                        draft_model=None, host="127.0.0.1")
    assert cli.cmd_gguf(a) == 2 and seen == []
    assert "--speculation draft needs --draft-model FILE" in env[-1], env
    b = SimpleNamespace(cmd="run", model="x.gguf", weights=None, mtp=None, mtp_file=None, speculation="ngram",
                        draft_model="d.gguf", host="127.0.0.1")
    assert cli.cmd_gguf(b) == 2 and seen == []
    assert "--draft-model is the draft model of --speculation draft, not of --speculation ngram" in env[-1], env


def test_a_draft_model_on_a_models_json_model_stops_before_the_model_row(monkeypatch, env):
    seen = []
    monkeypatch.setattr(hub, "model_entry", lambda *a, **k: seen.append("model_entry"))
    for flags in (["--draft-model", "d.gguf"], ["--speculation", "draft"]):
        assert cli.main(["run", "qwen3-8-27b", *flags, "--prompt", "hi"]) == 2 and seen == []
        assert env[-1] == ("mashbox: --speculation draft runs on a GGUF file in this version; for a models.json model "
                           "use ngram or mtp"), env
        a = cli.build_parser().parse_args(["run", "qwen3-8-27b", *flags, "--prompt", "hi"])
        with pytest.raises(SystemExit, match="--speculation draft runs on a GGUF file in this version"):
            cli.load(a)
        assert cli.cmd_run(a, []) == 2 and seen == []


def test_a_draft_file_that_cannot_draft_stops_when_opened(files, env, tmp_path, monkeypatch):
    from mashbox import gguf as G
    with pytest.raises(G.GGUFError, match="--draft-model: a qwen35 file cannot draft"):
        cli.load_gguf(args(tmp_path, files["target"], "--draft-model", files["qwen35"]))
    real = G.open_model

    def other_tokens(path, *a, **k):
        f, A, c = real(path, *a, **k)
        if path == files["draft"]:
            toks = list(f.get("tokenizer.ggml.tokens"))
            toks[5], toks[6] = toks[6], toks[5]
            get = f.get
            f.get = lambda key, default=None: toks if key == "tokenizer.ggml.tokens" else get(key, default)
        return f, A, c
    monkeypatch.setattr(G, "open_model", other_tokens)
    with pytest.raises(G.GGUFError, match="--draft-model: its tokenizer is not the target's"):
        cli.load_gguf(args(tmp_path, files["target"], "--draft-model", files["draft"]))


def test_draft_model_applies_only_when_given(files):
    from mashbox.gguf import open_model
    _, A, c = open_model(files["target"], act=jnp.float32)
    meta = SEL.meta_of(A, c, "gguf")
    ok, w = SEL.applies("draft_model", meta)
    assert ok is False and w == "draft_model: no draft model given (--draft-model)", w
    ok, w = SEL.applies("draft_model", SEL.meta_of(A, c, "gguf", draft_given=True))
    assert ok is True and w == "draft_model: the draft model given with --draft-model", w
    kind, g, why, _ = SEL.pick(meta, "v6e-1", {})
    assert kind is None and "draft_model: no draft model given (--draft-model)" in why, why


def test_draft_model_help():
    import argparse
    p = cli.build_parser()
    sub = next(x for x in p._actions if isinstance(x, argparse._SubParsersAction)).choices["run"]
    h = next(x for x in sub._actions if "--draft-model" in x.option_strings).help
    assert "same tokenizer" in h and "--speculation draft" in h, h


def test_cmd_gguf_with_a_draft_model_resolves_both_files_and_runs(monkeypatch, env):
    """a GGUF run with --draft-model passes the refusal, resolves the model and then the draft file, and reaches cmd_run
    with both (a.gguf, a.draft_gguf); download fetches both and stops"""
    from mashbox.gguf import pick as GR
    got = []
    monkeypatch.setattr(GR, "resolve", lambda m, log=None, download=True: dict(path=m, repo=None, tag=None, file=m))
    monkeypatch.setattr(cli, "cmd_run", lambda a, argv=None: got.append((a.gguf["path"], a.draft_gguf["path"])) or 0)
    for sv in (None, "draft"):
        a = SimpleNamespace(cmd="run", model="t.gguf", weights=None, mtp=None, mtp_file=None, speculation=sv,
                            draft_model="d.gguf", host="127.0.0.1")
        assert cli.cmd_gguf(a) == 0, env
    assert got == [("t.gguf", "d.gguf")] * 2, got
    a = SimpleNamespace(cmd="download", model="t.gguf", weights=None, mtp=None, mtp_file=None, speculation=None,
                        draft_model="d.gguf", host="127.0.0.1")
    assert cli.cmd_gguf(a) == 0 and a.draft_gguf["path"] == "d.gguf" and len(got) == 2
