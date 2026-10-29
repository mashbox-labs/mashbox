"""Checks of the shared core: the registry, its packaged copy, the generated views, imports and the CLI."""
import ast
import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOWED = {"numpy", "jax", "jaxlib", "ml_dtypes", "tokenizers", "jinja2", "huggingface_hub", "mashbox",
           "gguf"}                                  # gguf: llama.cpp's GGUF reader, pinned in pyproject (NOTICE)
IMPORT_NAME = {"pyyaml": "yaml", "imageio-ffmpeg": "imageio_ffmpeg", "pillow": "PIL"}     # pip name -> import name


def extras():
    """[project.optional-dependencies] of pyproject.toml as {extra: [import names]} (each value is a JSON-style list)."""
    out, inside = {}, False
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        if line.startswith("["):
            inside = line.strip() == "[project.optional-dependencies]"
        elif inside and "=" in line:
            k, v = line.split("=", 1)
            pkgs = [re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].lower() for d in json.loads(v)]
            out[k.strip()] = {IMPORT_NAME.get(p, p.replace("-", "_")) for p in pkgs}
    return out


def model_allowed(py, ex):
    """Extra imports for a file inside a model package: the pyproject extra named after the model id, and the top-level
    packages the model loads from upstream source pinned by commit and hash (upstream.json code.keep)."""
    rel = py.relative_to(ROOT / "src" / "mashbox")
    if len(rel.parts) < 3 or rel.parts[0] != "models":
        return set()
    pkg = ROOT / "src" / "mashbox" / "models" / rel.parts[1]
    allowed = set(ex.get(rel.parts[1].replace("_", "-"), ()))
    up = pkg / "upstream.json"
    if up.is_file():
        allowed |= set(json.loads(up.read_text()).get("code", {}).get("keep", ()))
    return allowed


def first_difference(a, b, path="models.json"):
    """The JSON path of the first place two parsed registries differ, or None when they are equal."""
    if type(a) is not type(b):
        return path
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            d = first_difference(a.get(k), b.get(k), f"{path}.{k}") if k in a and k in b else f"{path}.{k}"
            if d:
                return d
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            label = x.get("id", i) if isinstance(x, dict) else i
            d = first_difference(x, y, f"{path}[{label}]")
            if d:
                return d
        return None if len(a) == len(b) else f"{path} (length {len(a)} vs {len(b)})"
    return None if a == b else path


def test_models_json_copy():
    """The wheel ships src/mashbox/models.json. It must be a byte copy of models.json, or the package carries a stale
    licence line and stale pins (it nearly shipped both once)."""
    src, copy = ROOT / "models.json", ROOT / "src" / "mashbox" / "models.json"
    fix = "fix: cp models.json src/mashbox/models.json"
    where = first_difference(json.loads(src.read_text()), json.loads(copy.read_text()))
    assert where is None, f"src/mashbox/models.json differs from models.json first at {where}; {fix}"
    assert src.read_bytes() == copy.read_bytes(), f"src/mashbox/models.json has the same content but other formatting; {fix}"


KNOWN_WITHOUT_DOCSTRING = {"test_hy_nomas.py"}      # hy-nomas lane adds it; this set only shrinks


def test_every_test_module_says_what_it_guards():
    """Every test module opens with a docstring naming the failure it guards (CI.md: every test earns its place)."""
    missing = {p.name for p in (ROOT / "tests").rglob("test_*.py") if not ast.get_docstring(ast.parse(p.read_text()))}
    assert missing <= KNOWN_WITHOUT_DOCSTRING, f"add a module docstring naming the failure each guards: {sorted(missing - KNOWN_WITHOUT_DOCSTRING)}"


def test_every_model_has_its_folders():
    reg = json.loads((ROOT / "models.json").read_text())
    ids = [m["id"] for m in reg["models"]]
    assert len(ids) == len(set(ids))
    for m in reg["models"]:
        assert (ROOT / m["path"] / "__init__.py").is_file(), m["id"]
        assert m["path"] == "src/mashbox/models/" + m["id"].replace("-", "_"), m["id"]
        assert (ROOT / "docs" / "models" / (m["id"] + ".md")).is_file(), m["id"]
        assert any((ROOT / "licenses" / m["id"]).iterdir()), m["id"]


def test_imports_are_public():
    """Package code imports only the core dependencies, its model's declared extra and pinned upstream code."""
    bad, ex = [], extras()
    for py in (ROOT / "src" / "mashbox").rglob("*.py"):
        ok = ALLOWED | model_allowed(py, ex)
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top not in ok and top not in sys.stdlib_module_names:
                    bad.append(f"{py.relative_to(ROOT)}: {n}")
    assert not bad, bad


def test_views_match_models_json():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "hf_github_sync_check.py"), "check", "--repo", str(ROOT), "--offline"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_sync_check_catches_typed_drift(tmp_path):
    """The number guard, the prose checks, the link check and the measured-number guard fail on planted drift. A second NUM_RE once shadowed the guard's regex and
    turned it off without a failure; this keeps it on."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    assert sc.unbacked("99.9 tok/s", ["38.1 tokens/s"]) == ["99.9"]
    assert sc.unbacked("38.1 tok/s", ["38.12 tokens/s"]) == []
    reg = json.loads((ROOT / "models.json").read_text())
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text(
        "One TPU v6e chip is $2.80 an hour (read 1 May 2026).\n\n```bash\npython3 -m uv tool install mashbox\n```\n")
    problems = sc.Problems()
    sc.check_prose(reg, tmp_path, problems)
    text = " ".join(problems)
    assert "$2.80" in text and "read date" in text and "install line" in text, text
    (tmp_path / "docs" / "y.md").write_text("## Top\n\nSee [the old page](models/missing.md), [this](x.md#top), [here](#top), "
                                            "[there](y.md#no-such-heading) and [the other][o].\n\n[o]: models/gone.md\n")
    problems = sc.Problems()
    sc.check_relative_links(tmp_path, problems)
    text = " ".join(problems)
    assert len(problems) == 4 and all(x in text for x in ("models/missing.md", "x.md#top", "y.md#no-such-heading", "models/gone.md")), problems
    (tmp_path / "docs" / "guides" / "tpu").mkdir(parents=True)
    (tmp_path / "docs" / "guides" / "tpu" / "z.md").write_text(
        "Later starts took <!-- r:qwen3-8-27b/3 -->x<!-- /r --> and once 5.2 s, then `5.3 s`, 5.4 sec, 41 tokens/sec, "
        "1.4\u00d7 and 6-7 s; 4 to <!-- r:qwen3-8-27b/3 -->y<!-- /r -->. A GIF sped up 10x. Ours is 10x faster.\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "numbers_allowed.txt").write_text("docs/guides/tpu/z.md | GIF sped up 10x | an edit\n* | 5.2 s | old form\n")
    problems = sc.Problems()
    sc.check_numbers(reg, tmp_path, problems)
    hand = " ".join(p for p in problems if "typed by hand" in p)
    for n in ("'5.2 s'", "'5.3 s'", "'5.4 sec'", "'41 tokens/sec'", "'1.4\u00d7'", "'6-7 s'", "'4 to \\x00'", "faster"):
        assert n in hand, (n, problems)
    assert "line 2" in " ".join(problems) and hand.count("'10x'") == 1, problems
    (tmp_path / "docs" / "w.md").write_text("MASHBOX Labs builds its own models for games.\n")
    problems = sc.Problems()
    sc.check_prose(reg, tmp_path, problems)
    assert any("other words" in p for p in problems), problems
    # what a browser shows or follows besides Markdown: form values, ARIA and meta text, data-*,
    # MDX and SVG pages; unquoted, root-relative, form, refresh, CSS and object targets; version metadata; docstrings
    for f in ("x.md", "y.md", "w.md", "guides/tpu/z.md"):
        (tmp_path / "docs" / f).unlink()
    (tmp_path / "tools" / "numbers_allowed.txt").write_text("")
    (tmp_path / "docs" / "v.md").write_text(
        '---\nrelease: 9.9.9\n---\n<input value="11 tokens/s"> <input type="range" value="3"> <i aria-valuetext="12 tokens/s"></i>\n'
        '<span data-speed="13 tokens/s"></span>\n<meta name="description" content="14 tokens/s">\n'
        '<meta name="mashbox-version" content="9.9.8">\n<span title="MASHBOX Labs is a startup">x</span>\n'
        '<input value="Qwen3.8-27B runs on TPU v5p-8">\n'
        '<a href=gone-1.md>a</a> [b](/docs/gone-2) <form action="gone-3"></form>\n'
        '<meta http-equiv="refresh" content="0; url=gone-4.html">\n<style>.h { background: url(\'gone-5.png\') }</style>\n'
        '<object data="gone-6.svg"></object> <a href=v.md>here</a> [c](/docs/v)\n'
        '<span aria-roledescription="17 tokens/s">x</span>\n<style>@import "gone-7.css";</style>\n')
    (tmp_path / "docs" / "u.mdx").write_text("The server decodes at 15 tokens/s.\n\n<Stat value=\"18 tokens/s\" />\n")
    (tmp_path / "docs" / "t.svg").write_text('<svg><text x="0" y="20">16 tokens/s</text></svg>\n')
    pkg = tmp_path / "src" / "mashbox" / "models" / "qwen3_8_27b"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""Runs on TPU v5p-8. MASHBOX Labs builds chips."""\n')
    problems = sc.Problems()
    sc.check_numbers(reg, tmp_path, problems)
    text = " ".join(problems)
    for n in ("'11 tokens/s'", "'12 tokens/s'", "'13 tokens/s'", "'14 tokens/s'", "'15 tokens/s'", "'16 tokens/s'",
              "'17 tokens/s'", "'18 tokens/s'",
              "runs on v5p-8 ('Qwen3.8-27B", "docstring says Qwen3.8-27B runs on v5p-8"):
        assert n in text, (n, problems)
    assert "'3'" not in text, problems   # a slider's value is not text on the page
    problems = sc.Problems()
    sc.check_relative_links(tmp_path, problems)
    assert sorted(p.split("missing path: ")[-1] for p in problems) == \
        ["docs/gone-2.md", "gone-1.md", "gone-3", "gone-4.html", "gone-5.png", "gone-6.svg", "gone-7.css"], problems
    problems = sc.Problems()
    sc.check_prose(reg, tmp_path, problems)
    text = " ".join(problems)
    assert "declares version 9.9.9" in text and "declares version 9.9.8" in text, problems
    assert sum("other words" in p for p in problems) == 2, problems   # the title attribute and the docstring
    for block, cmd in (("command: mashbox run x --slotz 8", "mashbox run x --slotz 8"),
                       ('  "run": "mashbox tpu up x",', "mashbox tpu up x"), ("- exec: 'mashbox run x'", "mashbox run x")):
        assert sc.KEYED_CMD_RE.sub(r"\2", block) == cmd, block
    for block, cmd in (('command: ["mashbox", "run", "x", "--slotz", "8"]', "mashbox run x --slotz 8"),   # exec form
                       ('ENTRYPOINT ["mashbox", "tpu", "up", "x"]', "mashbox tpu up x"), ("CMD mashbox run x", "mashbox run x")):
        assert sc.code_block(block) == cmd, block
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "run-x").write_text("#!/bin/sh\nexec mashbox run x\n")
    assert [f.name for f in sc.example_files(tmp_path)] == ["run-x"]
    app = tmp_path / "src" / "mashbox" / "ui" / "app"
    app.mkdir(parents=True)
    for src in ('CATALOG_URL = "https://example.com/models.json"\n',
                f'CATALOG_URL = "{sc.catalog_url(reg)}"\nCATALOG_URL = "https://example.com/models.json"\n'):
        (app / "__init__.py").write_text(src)
        problems = sc.Problems()
        sc.check_app(reg, tmp_path, None, None, problems)
        assert len(problems) == 1 and "CATALOG_URL" in problems[0], problems


def test_an_allow_line_covers_every_place_its_words_appear_on_a_line():
    """tools/docs_lint_allow.txt: an allow line covers each hit inside its words wherever on the line they appear (a
    support table row prints a chip's cell once per chip, e.g. "Not tested yet" three times); a hit outside those words,
    or on another page, still fails."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("docs_lint", ROOT / "tools" / "docs_lint.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    key = ("docs/x.md", "Not tested yet")
    row = "| Model | Not tested yet | Not tested yet | Not tested yet |\n"
    used = set()
    assert dl.lint_text("docs/x.md", row, {key: (1, 3)}, used) == []
    assert used == {key}
    outside = dl.lint_text("docs/x.md", "| Not tested yet | Not tested yet | it runs there yet |\n", {key: (1, 2)}, set())
    assert len(outside) == 1 and "[yet]" in outside[0], outside
    assert len(dl.lint_text("docs/y.md", row, {key: (1, 3)}, set())) == 3


@pytest.mark.parametrize("copies", [4, 5, 20, 2, 1])
def test_an_allow_line_covers_exactly_its_count_of_hits(copies):
    """an allow line names how many hits it covers on its page (page | words | hits | why): one more copy of the words
    anywhere on the page (an extra cell on the same row included) or one fewer fails, so an allowance never grows
    silently"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("docs_lint", ROOT / "tools" / "docs_lint.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    key = ("docs/x.md", "Not tested yet")
    page = "| Model | " + " | ".join(["Not tested yet"] * copies) + " |\n"
    probs = dl.lint_text("docs/x.md", page, {key: (7, 3)}, set())
    assert len(probs) == 1 and f"covers {copies} hit" in probs[0] and "says 3" in probs[0] and ":7:" in probs[0], probs
    split = "| A | Not tested yet | Not tested yet |\n| B | Not tested yet |\n| C | Not tested yet |\n"
    assert len(dl.lint_text("docs/x.md", split, {key: (7, 3)}, set())) == 1     # the budget is per page, not per line
    assert dl.lint_text("docs/x.md", split, {key: (7, 4)}, set()) == []


@pytest.mark.parametrize("line,ok", [("docs/x.md | Not tested yet | 3 | a chip cell", True),
                                     ("docs/x.md | Not tested yet | a chip cell", False),
                                     ("docs/x.md | Not tested yet | 0 | a chip cell", False),
                                     ("docs/x.md | Not tested yet | -1 | a chip cell", False),
                                     ("docs/x.md | Not tested yet | 3.0 | a chip cell", False),
                                     ("docs/x.md | Not tested yet | three | a chip cell", False),
                                     ("docs/x.md | Not tested yet | 3 |", False),
                                     ("docs/x.md | Not tested yet | 3", False)])
def test_an_allow_line_states_its_hit_count(tmp_path, line, ok):
    """tools/docs_lint_allow.txt lines are 'page | the words (2 or more) | hits (a whole number, 1 or more) | why'"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("docs_lint", ROOT / "tools" / "docs_lint.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "docs_lint_allow.txt").write_text("# header\n" + line + "\n")
    problems = []
    allowed = dl.load_allow(tmp_path, problems)
    if ok:
        assert problems == [] and allowed == {("docs/x.md", "Not tested yet"): (2, 3)}
    else:
        assert allowed == {} and len(problems) == 1 and "hits" in problems[0], (allowed, problems)


def test_public_text_is_clean():
    """Public pages describe the release as plain facts (tools/docs_lint.py): no dates outside licence text (the
    changelog's version headings carry none either), no run stories, status words, hidden notes, labels or internal
    ids. Each planted kind fails, so the lint stays on."""
    import importlib.util
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "docs_lint.py"), "--repo", str(ROOT)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    spec = importlib.util.spec_from_file_location("docs_lint", ROOT / "tools" / "docs_lint.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    planted = ("Verified on 28 October 2026 in our run: v5e is not supported yet.\n"
               '<!-- TODO source="run record" -->\n'
               "Measured 2026-10-28 03:53 UTC, record RUN-7.\n"
               "Coming soon: a new model. Price: ESTIMATE at list price.\n"
               "Later starts: <!-- r:qwen3-8-27b/3 -->6.6 s<!-- /r -->.\n"
               "It has happened twice. Its owners should switch it (skill example-guide section 4).\n"
               '```text\n{"t": "2026-10-28T09:19:46Z"}\n```\n')
    kinds_of_text = lambda rel, text: {p.split("[", 1)[1].split("]", 1)[0] for p in dl.lint_text(rel, text, {}, set())}
    kinds = kinds_of_text("docs/x.md", planted)
    for k in ("verified", "our-run", "yet", "long-date", "comment", "iso-date", "utc", "record-id", "coming-soon", "label",
              "marker", "story", "owner-note", "skill"):
        assert k in kinds, (k, kinds)
    assert any("code block" in p for p in dl.lint_text("docs/x.md", planted, {}, set()))
    # no dates in public files, the changelog included: a version heading is the version alone
    assert "iso-date" in kinds_of_text("CORE-CHANGES.md", "## 0.1.0 (2026-10-28)\n")
    assert dl.lint_text("CORE-CHANGES.md", "## 0.1.0\n", {}, set()) == []
    assert dl.lint_text("CORE-CHANGES.md", "1. Why: on 2026-10-28 the create failed.\n", {}, set())   # a date in an entry
    assert dl.lint_text("docs/x.md", "Run `mashbox run qwen3-8-27b` and wait.\n\n```text\nnot supported yet\n```\n", {}, set()) == []
    assert dl.lint_text("docs/x.md", "A test set of noisy, partly hidden tracks.\n<!-- resources:start -->\n<!-- resources:end -->\n",
                        {}, set()) == []   # plain meaning and a generated block
    assert dl.lint_text("docs/x.md", "It partly works on v5e.\n", {}, set())
    kinds_of = lambda rel, text: {p.split("[", 1)[1].split("]", 1)[0] for p in dl.lint_text(rel, text, {}, set())}
    # an answer can change with the other requests in its batch: no page promises the same answer alone and batched
    assert "batch-identity" in kinds_of("docs/x.md", "| Greedy | Temperature 0, so the same prompt gives the same answer. |\n")
    assert "batch-identity" in kinds_of("docs/x.md", "A greedy prompt gives identical answers alone or batched.\n")
    assert "batch-identity" in kinds_of("docs/x.md", "Batched rows get the same tokens as a request alone.\n")
    # after a worker fault a request is sent once more while its client has received nothing: not streamed, until its
    # response is written (the worker's tokens never stop it); streamed, until its first output is sent
    assert "retry-rule" in kinds_of("CORE-CHANGES.md", "A request that got no output tokens is sent once more from the start.\n")
    assert "retry-rule" in kinds_of("CORE-CHANGES.md", "A non-streamed request with output, or one whose retry fails too, gets 500.\n")
    assert "retry-rule" in kinds_of("docs/x.md", "A request that has received no output tokens is retried once.\n")
    assert "retry-rule" in kinds_of("docs/x.md", "A request that already has output ends: a stream gets the route's error event "
                                                 "(with the response id), a non-streamed request gets 500.\n")
    assert kinds_of("docs/x.md", "A request whose client has received nothing is sent again (once, from the start): not streamed, "
                                 "until its response is written; streamed, until its first output is sent. A stream that has "
                                 "sent output ends with the route's error event; a request whose retry fails too gets 500.\n") == set()
    assert kinds_of("docs/x.md", "Plain JAX: slower, with the same outputs within rounding.\n") == set()
    # a page template is a public page: a hidden note to the writer fails like any other hidden comment
    assert "comment" in kinds_of("docs/_templates/model.md", "<!--\nCopy to docs/models/<id>.md and keep the headings.\n-->\n# <Name>\n")
    assert "comment" in kinds_of("docs/_templates/guide.md", "## 1. <Step>\n\n<!-- Media slot: one screenshot. -->\n")
    # an MDX page is read like Markdown, with its own hidden-comment form; an SVG's text is read like an app page
    assert "comment" in kinds_of("docs/x.mdx", "# Page\n\n{/* note to the writer */}\n")
    assert "comment" in kinds_of("docs/x.mdx", "<!-- note -->\n") and kinds_of("docs/x.mdx", "Fast on v5e.\n") == set()
    assert {"iso-date", "our-run"} <= kinds_of("docs/assets/x.svg", "<svg><text>Measured 2026-10-28 in our run</text></svg>")
    # an internal id on its own, with no date or status word near it (Markdown, a CLI string, app text)
    assert kinds_of("docs/x.md", "Internal run SYNC-77.\n") == {"internal-id"}
    assert kinds_of("docs/x.md", "Finding Q7-X03 is closed.\n") == {"internal-id"}
    assert dl.lint_text("docs/x.md", "Qwen2.5-VL-7B, SHA-256, UTF-8, ISO-8601, BSD-3-Clause and CC-BY-4.0.\n", {}, set()) == []
    py = 'HELP = "Verified in internal run SYNC-77; partly ready."\nFAST = "Faster decoding; not supported yet on v5e."\n'
    assert {"verified", "internal-id", "staging", "yet"} <= kinds_of("src/mashbox/x.py", py), kinds_of("src/mashbox/x.py", py)
    assert dl.lint_text("src/mashbox/x.py", 'STATE = "pending"\nSTAMP = "verified"\n', {}, set()) == []   # names in code
    assert {"our-run", "internal-id"} <= kinds_of("src/mashbox/ui/x.js", "el.textContent = 'Seen in our run SYNC-77';\n")
    # visible text added to the app page fails the lint; the page as shipped passes
    app = ROOT / "src" / "mashbox" / "ui" / "app" / "index.html"
    lines = app.read_text().splitlines(keepends=True)
    d04 = "".join(lines[:17] + ["<p>Verified on October 28, 2026 in our run SYNC-77; partly ready.</p>\n"] + lines[17:])
    assert {"verified", "us-date", "our-run", "internal-id", "staging"} <= kinds_of("src/mashbox/ui/app/index.html", d04)
    assert dl.lint_text("src/mashbox/ui/app/index.html", app.read_text(), {}, set()) == []
    assert kinds_of("x.html", '<img alt="Measured 2026-10-28"><!-- record RUN-7 --><script>t("our run")</script>') >= {
        "iso-date", "record-id", "our-run"}   # alt text, comment words and script strings are read
    # the lint reads the same app pages as the number guard (hf_github_sync_check.ui_pages)
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    ui = [f.relative_to(ROOT).as_posix() for f in sc.ui_pages(ROOT)]
    assert ui and sorted(ui) == sorted(r for r in dl.public_files(ROOT) if r.endswith(dl.UI_SUFFIXES)), ui
    extra = [("private", re.compile(r"\bexample-host\b"), "no internal host names")]   # maintainers' rules file
    assert dl.lint_text("docs/x.md", "Copied from example-host.\n", {}, set(), extra)
    assert dl.lint_text("docs/x.md", "Copied from example-host.\n", {}, set()) == []


def test_chip_status_keeps_its_condition():
    """A chip status on a page carries the chips row's condition or reason ("Supported: a v5e-4 ...", "Not supported:
    runs only on a v6e-4 ..."), so a page never prints a bare "Supported" for a model that needs a bigger host."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    for m in reg["models"]:
        for chip, row in m["chips"].items():
            want = sc.chip_text(m, chip)
            why = row.get("note") if row.get("status") == "passing" else (row.get("reason") or row.get("note"))
            if sc.chip_note(m, chip):  # a public note replaces "Not supported: <reason>" (test_chip_note_replaces_not_supported)
                assert want == row["note"], (m["id"], chip, want)
            elif why and row.get("status") != "pending":
                assert want.endswith(": " + why), (m["id"], chip, want)
            page = f"Status: <!-- r:{m['id']}/chip-{chip} -->x<!-- /r -->."
            got = sc.fill_numbers(reg, page, "docs/x.md", sc.Problems())
            assert want in got, (m["id"], chip, got)


def test_number_map_puts_markers_back(tmp_path):
    """The private number map: render takes the markers out of a page and records where each number sits; the checks put
    them back exactly, an inline marker wins over the map, and a number whose words changed is reported, not lost."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    page = ("| Later starts | <!-- r:qwen3-8-27b/3 -->6.6 s<!-- /r --> |\n"
            "From 4.0 to <!-- r:qwen3-8-27b/agents-n6 -->4.7 s<!-- /r -->; again 4.0 to <!-- r:cloud/x -->4.7 s<!-- /r -->.\n")
    plain, spots = sc.unmark(page)
    assert "<!--" not in plain and len(spots) == 3
    assert sc.mark_up(plain, spots) == page
    assert sc.mark_up(sc.mark_up(plain, spots[:1]), spots) == page
    problems = sc.Problems()
    sc.mark_up(plain.replace("Later starts", "Later runs"), spots, "docs/x.md", problems)
    assert len(problems) == 1 and "r:qwen3-8-27b/3" in problems[0], problems


def test_cloud_prices_check_catches_drift(tmp_path):
    """docs/data/cloud-prices.json is the one dated source of the TPU vs GPU page: a price read more than max_age_days
    ago fails, models.json prices.per_chip_hour that differs from it fails, and a hand edit inside a cloud block fails."""
    import datetime
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    data = json.loads((ROOT / sc.CLOUD).read_text())
    newest = max(datetime.date.fromisoformat(p["read_utc"][:10]) for p in data["prices"])
    (tmp_path / "docs" / "data").mkdir(parents=True)
    (tmp_path / sc.CLOUD).write_text(json.dumps(data))
    problems = sc.Problems()
    sc.check_cloud(reg, tmp_path, problems, today=newest)
    assert not problems, problems
    problems = sc.Problems()
    sc.check_cloud(reg, tmp_path, problems, today=newest + datetime.timedelta(days=data["max_age_days"] + 1))
    assert any("days ago" in p for p in problems), problems
    reg["prices"]["per_chip_hour"]["v6e"] = "2.80"
    problems = sc.Problems()
    sc.check_cloud(reg, tmp_path, problems, today=newest)
    assert len(problems) == 1 and "2.80" in problems[0], problems
    page = (ROOT / "docs" / "tpu-vs-gpu.md").read_text()
    assert "| $11.06 |" in page
    problems = sc.Problems()
    views = sc.docs_views(json.loads((ROOT / "models.json").read_text()), ROOT, problems)
    assert "docs/tpu-vs-gpu.md" in views, problems
    assert views["docs/tpu-vs-gpu.md"] == page
    assert views["docs/tpu-vs-gpu.md"] != page.replace("| $11.06 |", "| $3.00 |")
    cost = sc.cloud_block("cost", data, json.loads((ROOT / "models.json").read_text()))
    ceil = [line.split("|")[-2].strip() for line in cost.splitlines() if ", ceiling |" in line]
    assert len(ceil) == 3 and ceil[0] == ceil[1], ceil  # one chip and a v6e-4 have the same GB/s per $: same ceiling cost


def test_every_cli_block_is_filled():
    """Every <!-- cli:start CMD --> in a docs page is a block the sync check renders, a model's own command (`run
    track2rig`) included. The Track2Rig page's options block once named a command the block pattern did not match, so
    it stayed empty and the check passed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    for page in sc.doc_pages(ROOT):
        text = page.read_text()
        blocks = sc.CLI_RE.findall(text)
        assert text.count("<!-- cli:start ") == len(blocks), page
        for _, cmd, body, _ in blocks:
            assert body.strip().startswith("| Option |") or body.strip() == "No options.", (page.name, cmd)


def test_number_markers_in_package_pages_are_checked(tmp_path):
    """A number marker in a package page (packages/<name>/README.md) is filled from the named figures like a docs page: a
    stale value there fails the check (the video README once kept an old clip's numbers and passed). The figures live in
    the number map (internal/numbers.json); a tree without the map keeps its markers, so the page is read with them."""
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    nmap = sc.load_numbers(ROOT)
    if nmap is not None:
        sc.merge_figures(reg, nmap)
    else:   # the public tree holds no number map: give one model a figure, the page must be filled from it the same way
        reg["models"][0]["figures"] = {"t": {"text": "7 s", "what": "stand-in figure", "date": "2026-10-29"}}
    m = next(x for x in reg["models"] if x.get("figures"))
    key, fig = next(iter(m["figures"].items()))
    (tmp_path / "packages" / "p").mkdir(parents=True)
    page = tmp_path / "packages" / "p" / "README.md"
    page.write_text(f"Took <!-- r:{m['id']}/{key} -->not the figure<!-- /r -->.\n")
    views = sc.render_all(reg, tmp_path, None, ROOT / "templates" / "hf", sc.Problems())
    assert views.get("packages/p/README.md") == f"Took <!-- r:{m['id']}/{key} -->{fig['text'].split(' (')[0]}<!-- /r -->.\n", views.get("packages/p/README.md")


def test_licence_check_catches_drift(tmp_path):
    """The licence check (report only, never edits) fails on each kind of mismatch with models.json: the package licence,
    an upstream LICENSE that is not the pinned file, and a page that says a licence is not chosen."""
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    problems = sc.Problems()
    sc.check_licences(reg, ROOT, problems)
    assert not problems, problems
    shutil.copy(ROOT / "LICENSE", tmp_path / "LICENSE")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nlicense = "MIT"\nlicense-files = ["LICENSE"]\n')
    mid = next(m["id"] for m in reg["models"] if m["upstream"].get("license_sha256"))
    (tmp_path / "licenses" / mid).mkdir(parents=True)
    (tmp_path / "licenses" / mid / "LICENSE").write_text("edited\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("Our code: licence not chosen yet.\n")
    problems = sc.Problems()
    sc.check_licences(reg, tmp_path, problems)
    text = " ".join(problems)
    assert "pyproject.toml" in text and f"licenses/{mid}/LICENSE" in text and "docs/x.md:1" in text, text


def test_cli_dispatches_only_parsed_commands():
    """mashbox main() reads argv before argparse only for `tpu` and a MODEL_CLI model's run/download, so every other
    command is in build_parser() and so in docs/cli.md (a command handled outside it would be undocumented)."""
    fn = next(n for n in ast.parse((ROOT / "src" / "mashbox" / "cli.py").read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    words = {c.value for n in ast.walk(fn) if isinstance(n, ast.Compare)
             and any(isinstance(x, ast.Name) and x.id == "argv" for x in ast.walk(n))
             for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert words <= {"tpu", "run", "download", ".", "-"}, f"main() dispatches {sorted(words - {'tpu', 'run', 'download', '.', '-'})} before argparse"


def test_cli_model_lists_come_from_the_catalog():
    """The CLI's runnable models and the models with their own command line come from models.json rows, never from ids
    typed into cli.py (ARCHITECTURE-v1 section 5): a row is runnable when the package its path names ships in this version,
    and parses its own `run` / `download` options when that package's __init__ defines main(cmd, argv)."""
    from mashbox import cli
    fake = {"models": [
        {"id": "demo-own-cli", "path": "src/mashbox/models/track2rig"},       # a package with main(cmd, argv)
        {"id": "demo-engine-run", "path": "src/mashbox/models/qwen3_8_27b"},  # a package the engine's own run serves
        {"id": "demo-absent", "path": "src/mashbox/models/not_in_this_version"},
        {"id": "demo-outside", "path": "src/mashbox/runtime"},
        {"id": "demo-no-path", "path": None}]}
    assert cli.catalog_runtimes(fake) == (("demo-own-cli", "demo-engine-run"), {"demo-own-cli": "mashbox.models.track2rig"})
    reg = json.loads((ROOT / "models.json").read_text())
    assert (cli.RUNNABLE, cli.MODEL_CLI) == cli.catalog_runtimes(reg)
    assert set(cli.RUNNABLE) == {m["id"] for m in reg["models"] if m.get("launch") is None}


def test_run_stops_cleanly_for_a_model_without_its_runtime(capsys):
    from mashbox import cli
    reg = json.loads((ROOT / "models.json").read_text())
    for m in reg["models"]:
        if m["id"] not in cli.RUNNABLE:
            assert cli.main(["run", m["id"]]) == 2
            assert cli.main(["download", m["id"]]) == 2
            assert "not in mashbox" in capsys.readouterr().err


KNOWN_WITHOUT_KERNEL_CHECK = {"qwen3-8-27b"}        # Qwen lane adds tests/tpu/qwen3_8_27b_kernel_check.py; this set only shrinks


def test_every_kernel_bundle_has_a_chip_check():
    """A model that ships compiled kernels for a chip needs tests/tpu/<id>_kernel_check.py and its tolerance file, or
    the chip tier (T2) never compares those kernels with plain JAX and a broken bundle ships green."""
    missing = set()
    for m in json.loads((ROOT / "models.json").read_text())["models"]:
        a = m["id"].replace("-", "_")
        if m.get("kernels") and not all((ROOT / "tests" / "tpu" / f).is_file() for f in (f"{a}_kernel_check.py", f"{a}_kernel_tolerances.json")):
            missing.add(m["id"])
    assert missing <= KNOWN_WITHOUT_KERNEL_CHECK, f"kernel bundles with no chip check: {sorted(missing - KNOWN_WITHOUT_KERNEL_CHECK)}"


def test_path_filter_runs_what_a_change_can_break():
    """tools/ci_select.py: a wrong rule silently skips the tests a change can break, so the rules that matter are pinned."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci_select", ROOT / "tools" / "ci_select.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    areas = cs.model_areas()
    every = sorted(set(areas) | ({"core"} if cs.core_has_t1_tests(areas) else set()))
    a = areas[0]
    assert cs.select(["README.md", "docs/faq.md", f"hf/{a}/README.md", "licenses/x"], areas) == []
    assert cs.select([f"src/mashbox/models/{a}/x.py"], areas) == [a]
    assert cs.select([f"tests/cpu/test_{a}_x.py"], areas) == [a]
    assert cs.select(["src/mashbox/cli.py"], areas) == every
    assert cs.select(["some/new/kind/of/file.cfg"], areas) == every
    for p in (ROOT / "tests" / "tpu").glob("*"):                       # a tests/tpu file a CPU test reads runs that test
        readers = {cs.test_area(t.name, areas) for t in (ROOT / "tests" / "cpu").glob("*.py") if p.name in t.read_text()}
        assert readers <= set(cs.select([f"tests/tpu/{p.name}"], areas)), p.name


def test_sync_online_coming_soon_model():
    """The nightly --online check passes a coming-soon model whose HF repo is private (401/403) or absent (404), and
    fails one that is public before its launch or whose state cannot be read (fixtures, no network). The fixture also
    answers the release-wide checks (tag, PyPI's newest version, the org profile's rendered block), so any problem
    left is the coming-soon model's own."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = {"hf": {"org": "mashbox-labs"}, "github": {"org": "mashbox-labs", "repo": "mashbox"}, "version": "0.1.1",
           "models": [{"id": "soon", "name": "Soon", "kind": "world", "in": "text", "out": "video",
                       "upstream": {"maker": "Someone"}, "hf_repo": "mashbox-labs/Soon-TPU", "hf_revision": "",
                       "launch": "coming soon"}]}
    profile = f"# Org\n\n{sc.MARK_START}\n{sc.profile_table(reg)}\n{sc.MARK_END}\n"

    class Fetch:
        def __init__(self, repo_state):
            self.repo_state = repo_state

        def get(self, url):
            if url == "https://huggingface.co/api/models?author=mashbox-labs&limit=1000":
                return 200, "[]"
            if url == "https://huggingface.co/api/models/mashbox-labs/Soon-TPU":
                return self.repo_state
            if url == "https://api.github.com/repos/mashbox-labs/mashbox/git/ref/tags/v0.1.1":
                return 200, "{}"
            if url == "https://pypi.org/pypi/mashbox/json":
                return 200, json.dumps({"info": {"version": "0.1.1"}})
            if url == "https://raw.githubusercontent.com/mashbox-labs/.github/main/profile/README.md":
                return 200, profile
            return 404, ""

    def problems(repo_state):
        p = sc.Problems()
        sc.check_online(reg, {}, Fetch(repo_state), p)
        return [str(x) for x in p]

    assert problems((401, "")) == []
    assert problems((403, "")) == []
    assert problems((404, "")) == []
    assert problems((200, json.dumps({"private": True}))) == []
    assert any("public before its launch" in x for x in problems((200, json.dumps({"private": False}))))
    assert any("cannot read the repo state" in x for x in problems((500, "")))


def test_world_card_licence_lines(tmp_path):
    """The world card says where the upstream LICENSE file comes from and states the licence people chose for our own
    files (the file's row in upstream.json); an own file without one fails the licence check. The LLM cards keep their words."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    reg = json.loads((ROOT / "models.json").read_text())
    m = next(x for x in reg["models"] if x["id"] == "hy-nomas")
    v = sc.card_vars(reg, m, ROOT)
    assert "licence file from [tencent/Hy-Nomas](https://huggingface.co/tencent/Hy-Nomas) on Hugging Face, unchanged" \
        in v["upstream_licence_source"]
    assert v["own_files_licence"] == ("Our own weights in this repo: `decoders/hy-nomas-fast-decoder-v1.safetensors` "
                                      "(the MASHBOX Fast Decoder): Apache-2.0. ")
    assert "${upstream_licence_source}${own_files_licence}" in (ROOT / "templates/hf/card-world.md").read_text()
    for t in ("card-llm-tpu.md", "card-llm-gguf.md", "card-image.md", "card-video.md", "card-voice.md"):
        assert "own_files_licence" not in (ROOT / "templates/hf" / t).read_text(), t
    # a model whose only file is its compile cache still says where its weights come from
    assert sc.files_table(m).startswith("This port loads the original weights unchanged from [tencent/Hy-Nomas]")
    # our own file without a recorded licence: no licence line, and the licence check fails
    up = json.loads((ROOT / m["path"] / "upstream.json").read_text())
    for r in up["repos"].values():
        for f in r["files"]:
            f.pop("license", None)
    (tmp_path / m["path"]).mkdir(parents=True)
    (tmp_path / m["path"] / "upstream.json").write_text(json.dumps(up))
    assert sc.own_files_licence(tmp_path, m) == ""
    problems = sc.Problems()
    sc.check_licences(reg, tmp_path, problems)
    assert any("decoders/hy-nomas-fast-decoder-v1.safetensors: our own file" in p for p in problems), problems


def test_version_has_one_source():
    """The release version is written once, in models.json: mashbox.__version__ reads the packaged copy, pyproject.toml
    takes it from mashbox.__version__ (dynamic), and pyproject.toml writes no version of its own."""
    import tomllib
    import mashbox
    reg = json.loads((ROOT / "models.json").read_text())
    assert mashbox.__version__ == reg["version"] == json.loads((ROOT / "src/mashbox/models.json").read_text())["version"]
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" not in pp["project"] and "version" in pp["project"]["dynamic"], pp["project"]
    assert pp["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "mashbox.__version__"}
    init = (ROOT / "src/mashbox/__init__.py").read_text()
    assert not re.search(r"__version__\s*=\s*[\"']", init), "src/mashbox/__init__.py writes a version literal"


def test_chip_note_replaces_not_supported():
    """A chips row that is not supported can carry a public note (Hy-Nomas on one v6e chip: it fits, at 2.46 s of
    compute per second of video, while this version of mashbox run starts on a v6e-4). Every view prints the note in
    place of "Not supported: <reason>"; the status stays, so mashbox run is unchanged. A result can carry its own
    highlight line for the card."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync_check", ROOT / "tools" / "hf_github_sync_check.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    note = "One chip: it fits on one chip. This version starts on four"
    m = {"id": "x", "name": "X", "chips": {"v6e": {"status": "not supported", "reason": "runs only on four", "note": note}},
         "results": [{"chip": "TPU v6e-1", "metric": "Speed", "value": "2 s", "baseline": "n/a", "highlight": "own words"}],
         "files": []}
    assert sc.chip_cell(m, "v6e") == "Not supported"
    assert sc.chip_text(m, "v6e") == note
    assert "| One chip | it fits on one chip. This version starts on four |" in sc.chips_table(m)
    assert sc.highlights(m) == "- **2 s**: own words"
    reg = json.loads((ROOT / "models.json").read_text())
    hy = next(x for x in reg["models"] if x["id"] == "hy-nomas")
    assert hy["chips"]["v6e"]["status"] == "not supported"
    assert sc.chip_text(hy, "v6e").startswith(
        "One chip: the full pipeline fits on a single TPU v6e chip (32 GB), 2.46 s of compute per second of video.")
