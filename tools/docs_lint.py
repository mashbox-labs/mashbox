#!/usr/bin/env python3
"""Public text lint: the docs, cards, captions, page templates, models.json text, CLI strings and the text of the app
pages (HTML and JavaScript) describe the released version as plain facts. Standard library only.

A public page says what the package does, how to use it, which chips it runs on and the numbers, with the hardware and
the model named. It never carries a log of how it was made: no dated stamps ("verified on <date>", "measured
2026-10-28"), no run stories ("our run", "we saw", "first-time user"), no status chatter ("yet", "pending", "coming
soon", "for now"), no per-page date changelogs, no "read on <date>" source stamps, no notes to the people who
maintain it, no record or run ids ("RUN-7", "ABC-12") and no hidden comments in Markdown, templates included. Dates
belong in licence text only; the changelog (CORE-CHANGES.md) names each release by its version alone.

What is read: Markdown and text by line; JSON by string value; Python by string literal; HTML by its visible text, its
alt/title/aria-label/placeholder words, its comments and the string literals of its scripts; JavaScript by string
literal. A one-word string in code ('pending', 'verified') is a name, so the plain-meaning word rules skip it; any
string with a space is text a person can read and gets every rule.

    python3 tools/docs_lint.py [--repo ROOT]          lint every public file of the repo; exit 1 on a hit
    python3 tools/docs_lint.py --paths FILE...        lint these files (a built site, an exported tree)

A line that must keep a hit (a quoted error message, a licence line) goes in tools/docs_lint_allow.txt as
"page | the words of the line that hold the hit | hits | why", hits being how many hits those words cover on the page;
one hit more or fewer fails, and so does an allow line that matches nothing, so the list only shrinks with the pages. Maintainers can add their own name and path rules in internal/docs_lint_rules.json
({"rules": [[name, regex, what to write instead], ...]}), which the lint reads when the file is there.
"""
import argparse
import ast
import collections
import fnmatch
import html
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW = Path("tools") / "docs_lint_allow.txt"

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
MAINTAINER_RULES = Path("internal") / "docs_lint_rules.json"   # maintainers' extra rules (names, paths); absent in a public tree

# (rule, regex, what to write instead). DATE rules do not apply to licence text.
DATE_RULES = [
    ("iso-date", re.compile(r"(?<![\w/-])20\d\d-[01]\d-[0-3]\d(?:T\d\d:\d\d(?::\d\d)?(?:\.\d+)?Z?)?(?![\w-])"), "no dates on a page: the release version pins the page"),
    ("long-date", re.compile(rf"\b\d{{1,2}} {_MON}[a-z]*\.? 20\d\d\b"), "no dates on a page: the release version pins the page"),
    ("us-date", re.compile(rf"\b{_MON}[a-z]*\.? \d{{1,2}},? 20\d\d\b"), "no dates on a page: the release version pins the page"),
    ("month-year", re.compile(rf"\b{_MONTH} 20\d\d\b"), "no dates on a page: the release version pins the page"),
    ("day-month", re.compile(rf"\b\d{{1,2}} {_MONTH}\b"), "no dates on a page: the release version pins the page"),
    ("utc", re.compile(r"\bUTC\b"), "no measurement times on a page"),
]
RULES = [
    ("verified", re.compile(r"(?i)\bverified\b|\bran as (?:printed|written)\b"), "describe the release; never a verification stamp"),
    ("our-run", re.compile(r"(?i)\bour (?:own )?(?:runs?|tests?|test (?:runs?|sessions?|driver)|sessions?|chips?|new (?:chip|VM|project)|"
                           r"measurements?|first runs|live demo|quantization work|setup|records?)\b"),
     "say what happens, to the reader: \"on a TPU v6e-1 ...\", not \"in our run\""),
    ("we-saw", re.compile(r"(?i)\bwe (?:saw|hit|ran into|ran|measured|tested|got|found|tried|made|added|have not)\b"),
     "say what happens, to the reader; who tested what stays in internal notes"),
    ("for-us", re.compile(r"(?i)\b(?:for|gave|to) us\b"), "say what happens, to the reader"),
    ("first-time-user", re.compile(r"(?i)\bfirst[- ]time[- ]users?\b|\bblind (?:run|test)\b"), "describe the steps, not the test that checked them"),
    ("this-run", re.compile(r"(?i)\b(?:in|from|on) (?:this|that|the same|an earlier|two earlier) (?:run|session|test)\b|"
                            r"\b(?:earlier|first measured|first-time) runs?\b|\bsame (?:day|morning)\b|\bthat morning\b"),
     "describe the result, not the run that produced it"),
    ("yet", re.compile(r"(?i)\byet\b"), "state what the release does or does not do (\"Not supported: <reason>\")"),
    ("pending", re.compile(r"(?i)\bpending\b"), "leave the section out until it has content"),
    ("coming-soon", re.compile(r"(?i)\bcoming soon\b|\bin the works\b|\bcomes? next\b|\blater also\b"), "unreleased things are simply absent"),
    ("for-now", re.compile(r"(?i)\bfor now\b|\bfor the moment\b"), "state what the release does"),
    ("version-chatter", re.compile(r"(?i)\bnot in \d+\.\d+\.\d+\b|\bfrom the release after\b|\bthe release after \d"),
     "describe the released version; version history goes in the changelog"),
    ("staging", re.compile(r"(?i)\bstaging\b|\bnot released\b|\bpre-release\b|\bhalf of (?:it|them)\b"
                           r"|\bpartly\b(?! (?:hidden|covered|occluded|visible|off[- ]screen|cut off|outside)\b)"),   # "partly hidden tracks" is data
     "describe the released version"),
    ("story", re.compile(r"(?i)\bit has happened\b|\bin one (?:session|run|test)\b|\bon another new\b|\bwith two cases\b|"
                         r"\bnot (?:yet )?proven\b|\btest driver\b|\bwhat we (?:ran into|saw|measured)\b"),
     "state the behaviour and the fix, not the session that showed it"),
    ("owner-note", re.compile(r"(?i)\bowners? should\b|\buntil (?:its|their) owners?\b|\bwhoever owns\b|\bnobody had\b|"
                              r"\bnote to (?:self|us)\b|\bfollow[- ]up\b"),
     "no notes to the people who maintain the code: say what changed and how callers keep working"),
    ("skill", re.compile(r"\bskill [a-z][a-z0-9]*-[a-z0-9-]+\b|\bSKILL\.md\b"), "no internal guide names"),
    ("label", re.compile(r"\b(?:MEASURED|ESTIMATE)\b"), "plain words (\"measured\", \"estimated\"), not a label"),
    ("marker", re.compile(r"<!--\s*/?r(?::|\s*-->)|\btest:tpu\b|\bTODO\b|\bFIXME\b|\bXXX\b"),
     "no record markers or notes in the public source"),
    ("record-id", re.compile(r"\b(?:RUN|REC|LEDGER)-\d+\b|\brun id\b|\brecord id\b"), "no run or record ids"),
    # an internal id on its own, whatever the words around it: "SYNC-77", "IDX-14", "R5-F01". Not a model or format name:
    # nothing word-like or hyphenated on either side ("Qwen2.5-VL-7B", "BSD-3-Clause"), no version tail ("CC-BY-4.0") and
    # not a standard's name ("SHA-256", "UTF-8", "ISO-8601", "RFC-9110", "DDR-5")
    ("internal-id", re.compile(r"(?<![\w.-])(?!(?:SHA|UTF|ISO|IEC|RFC|CVE|CWE|PEP|IEEE|HTTP|TLS|GPL|LGPL|AGPL|MPL|BSD|MIT|CC|"
                               r"DDR|LPDDR|HBM|PCIE|USB)-)(?:[A-Z]{2,}(?:-[A-Z]{2,})*-\d+|[A-Z]\d+-[A-Z]{1,2}\d+)(?![\w-]|\.\d)"),
     "no internal ids: name the thing itself"),
    # a request's answer can change with the other requests in its batch (row buckets change the rounding at a near tie)
    ("batch-identity", re.compile(r"(?i)\bsame prompt gives the same answer\b|"
                                  r"\b(?:same|identical) (?:answers?|outputs?|text|tokens)\b[^.;|()]{0,80}"
                                  r"\b(?:batch(?:ed|es)?|alone|together|other (?:requests|users))\b|"
                                  r"\b(?:alone|batched)\b[^.;|()]{0,80}\b(?:same|identical) (?:answers?|outputs?|text|tokens)\b"),
     "an answer can differ when other requests share its batch: never promise the same answer alone and batched"),
    # after a worker fault the front sends a request once more while its client has received nothing: not streamed, until
    # its response is written (tokens the worker made stay in the front, so they never stop the retry); streamed, until
    # its first output is sent. A page never ties the retry to output tokens or gives a non-streamed request with output 500
    ("retry-rule", re.compile(r"(?i)\b(?:got|gets|has|had|have|produced|made|received) no output tokens?\b[^.;]{0,40}"
                              r"\b(?:is|are|gets?|was|were) (?:sent|run|retried|tried)\b|"
                              r"\b(?:non-streamed|not streamed)\b[^.;]{0,40}\b(?:with|has|had|got) (?:output|tokens)\b[^.;]{0,60}\b500\b|"
                              r"\b(?:with|has|had|got) output\b[^.;]{0,80}\b(?:non-streamed|not streamed)\b[^.;]{0,30}\b500\b"),
     "a request whose client has received nothing is sent once more: not streamed, until its response is written; "
     "streamed, until its first output is sent"),
]
DATE_IN_CODE = ("iso-date", "utc")       # read inside code blocks of the docs pages too (sample output keeps no run times)
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.S)
MDX_COMMENT_RE = re.compile(r"\{/\*(.*?)\*/\}", re.S)   # an MDX page's hidden comment: {/* ... */}
FENCE_RE = re.compile(r"^```.*?^```[ \t]*$", re.S | re.M)   # commands and program output: read as they are typed
CODE_RE = re.compile(r"`[^`\n]+`")                          # inline code quotes a literal (an error message, a marker)
# generated-block markers (llama.cpp HELP_START style) and the generated-file header of pages rendered from public code
OK_COMMENT_RE = re.compile(r"^\s*(?:(?:models|cli|api|chips|cloud|charts|resources|catalog|archs):(?:start|end)\b[^>]*|Generated (?:from|by) .*)$", re.S)
LEGAL = {"LICENSE", "LICENSE-KERNELS", "docs/licence.md"}
TEXT_GLOBS = ("*.md", "*.mdx", "llms.txt", "hf/*.txt", "docs/media/*.txt", ".github/ISSUE_TEMPLATE/*.yml", "models.json", "docs/openapi.json")
SKIP_PREFIX = ("tests/", "internal/", "addons/", "licenses/", "src/mashbox/models.json")   # tests; private map; private app add-ons; upstream texts; copy of models.json
PY_ROOT = "src/mashbox/"                       # every string and docstring of the package: messages, help, UI text
# the app pages people read: every .html, .svg and .js file under these folders (the files hf_github_sync_check.ui_pages reads)
UI_ROOTS = ("src/mashbox/", "apps/", "docs/")
UI_SUFFIXES = (".html", ".svg", ".js")
# word rules a one-word string in code trips as a name ('pending', 'verified', 'UTC'); a string with a space is read
# with every rule. Dates, ids, run stories, notes and internal names apply to every string.
CODE_NAME_SKIP = {"verified", "yet", "staging", "this-run", "utc", "pending", "label"}
JSON_DATA_RE = re.compile(r"^(?:20\d\d-[01]\d-[0-3]\d(?:T[\d:]+Z?)?|[0-9a-f]{7,64})$")
# models.json fields that hold data for the tools, never text a page prints (status enums, the launch gate, test dates)
JSON_DATA_KEYS = {"status", "launch", "date", "read", "read_utc", "recorded_utc", "sha256", "license_sha256", "hf_revision", "revision"}


def public_files(repo):
    """Every public text file of the repo (git ls-files, committed and new): the pages, the package's Python modules
    (string literals) and the app pages (.html, .svg and .js under UI_ROOTS)."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):   # an exported tree: every file but tool and build folders
        skip = {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}
        out = [p.relative_to(repo).as_posix() for p in repo.rglob("*")
               if p.is_file() and not skip & set(p.relative_to(repo).parts[:-1])]
    rels = sorted(r for r in set(out) if (repo / r).is_file())
    text = [r for r in rels if any(fnmatch.fnmatchcase(r, g) for g in TEXT_GLOBS) and not r.startswith(SKIP_PREFIX)]
    py = [r for r in rels if r.startswith(PY_ROOT) and r.endswith(".py") and "/__pycache__/" not in r]
    return text + py + ui_files(rels)


def ui_files(rels):
    """The app pages among these repo-relative paths: .html, .svg and .js files under UI_ROOTS."""
    return [r for r in rels if r.startswith(UI_ROOTS) and r.endswith(UI_SUFFIXES)]


SCRIPT_RE = re.compile(r"(?is)<script\b[^>]*>(.*?)</script>")
STYLE_RE = re.compile(r"(?is)<style\b.*?</style>")
TAG_RE = re.compile(r"<[^>]*>")
ATTR_RE = re.compile(r'\b(?:alt|title|aria-label|placeholder)="([^"]*)"')
JS_STR_RE = re.compile(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"|`((?:[^`\\]|\\.)*)`")   # as hf_github_sync_check


def _js_strings(js, first_line):
    """(line, text, True) for each string literal of a script; whole-line // comments are left out, as in Python."""
    js = re.sub(r"(?m)^[ \t]*//.*$", "", js)
    return [(str(first_line + js.count("\n", 0, mt.start())), next((g for g in mt.groups() if g is not None), ""), True)
            for mt in JS_STR_RE.finditer(js)]


def _html_pieces(text):
    """An HTML page as a person reads it, by line: its text, the alt/title/aria-label/placeholder words of its tags,
    the words of its comments, and the string literals of its scripts. Styles and markup are left out."""
    out = []
    for mt in SCRIPT_RE.finditer(text):
        out += _js_strings(mt.group(1), text.count("\n", 0, mt.start(1)) + 1)
    for mt in COMMENT_RE.finditer(text):
        out.append((str(text.count("\n", 0, mt.start()) + 1), mt.group(1), False))
    t = COMMENT_RE.sub(_blank, STYLE_RE.sub(_blank, SCRIPT_RE.sub(_blank, text)))
    t = TAG_RE.sub(lambda mt: " " + " ".join(ATTR_RE.findall(mt.group(0))) + " " + "\n" * mt.group(0).count("\n"), t)
    out += [(str(i), html.unescape(line), False) for i, line in enumerate(t.splitlines(), 1) if line.strip()]
    return out


def _json_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in JSON_DATA_KEYS:
                yield from _json_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str) and not JSON_DATA_RE.match(obj):
        yield path, obj


def lines_of(rel, text):
    """(line label, text, in code) pieces to read: Markdown and text by line, JSON by string value, Python and
    JavaScript by string literal, HTML as a person reads it (_html_pieces). In code: a string literal of a program."""
    if rel.endswith(".json"):
        try:
            return [(w, s, False) for w, s in _json_strings(json.loads(text))]
        except ValueError:
            return [("1", text, False)]
    if rel.endswith(".py"):
        out = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.append((str(node.lineno), node.value, True))
        return out
    if rel.endswith(".js"):
        return _js_strings(text, 1)
    if rel.endswith((".html", ".svg")):   # an SVG's text, title and desc, as on a page
        return _html_pieces(text)
    return [(str(i), line, False) for i, line in enumerate(text.splitlines(), 1)]


def load_allow(repo, problems):
    allowed = {}
    f = repo / ALLOW
    for i, line in enumerate(f.read_text().splitlines() if f.exists() else [], 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) != 4 or not all(parts) or len(parts[1].split()) < 2 or not re.fullmatch(r"[1-9][0-9]*", parts[2]):
            problems.append(f"{ALLOW.as_posix()}:{i}: want 'page | the words that hold the hit (2 or more) | hits (how "
                            "many hits they cover on the page, 1 or more) | why'")
            continue
        allowed[(parts[0], parts[1])] = (i, int(parts[2]))
    return allowed


def _blank(mt):
    return re.sub(r"[^\n]", " ", mt.group(0))


def load_private(repo):
    """The maintainers' extra rules (internal names, paths and hosts) when internal/docs_lint_rules.json is there."""
    f = repo / MAINTAINER_RULES
    if not f.is_file():
        return []
    return [(name, re.compile(rx), fix) for name, rx, fix in json.loads(f.read_text()).get("rules", [])]


def _inside(flat, words, start, end):
    """True when the hit [start, end) lies inside a place where words appears in flat: every place, not only the first
    (a support table row prints a chip's cell once per chip)."""
    i = flat.find(words)
    while i != -1:
        if i <= start and end <= i + len(words):
            return True
        i = flat.find(words, i + 1)
    return False


def lint_text(rel, text, allowed, used, extra=()):
    """Problems in one file. rel is the repo-relative path (decides the legal exemption).
    Code blocks and inline code are not read for words (they quote commands, output and error text as printed), but
    dates and UTC times in the code blocks of docs pages are. The changelog keeps no dates, its version headings included.
    A hidden comment (<!-- -->, and {/* */} in MDX) fails in every public Markdown, MDX or text file, the page templates
    included."""
    probs = []
    hits = collections.Counter()
    rules = list(RULES) + list(extra)
    if rel not in LEGAL:
        rules += DATE_RULES
    if rel.endswith((".md", ".mdx")):
        if rel.startswith("docs/"):
            for mt in FENCE_RE.finditer(text):
                for i, line in enumerate(mt.group(0).splitlines()):
                    for rule, rx, fix in DATE_RULES:
                        if rule in DATE_IN_CODE and rx.search(line):
                            probs.append(f"{rel}:{text.count(chr(10), 0, mt.start()) + 1 + i}: [{rule}] {rx.search(line).group(0)!r} "
                                         f"in a code block: use a placeholder (\"...\") for times and ids in sample output")
        text = CODE_RE.sub(_blank, FENCE_RE.sub(_blank, text))
    if rel.endswith((".md", ".mdx", ".txt")):
        found = [*COMMENT_RE.finditer(text), *(MDX_COMMENT_RE.finditer(text) if rel.endswith(".mdx") else ())]
        for mt in sorted(found, key=lambda m: m.start()):
            if not OK_COMMENT_RE.match(mt.group(1)):
                line = text.count("\n", 0, mt.start()) + 1
                probs.append(f"{rel}:{line}: [comment] hidden comment in a public page: {mt.group(0)[:80]!r}"
                             " (notes stay in internal files; a block marker or a 'Generated from' header is fine)")
    for where, line, in_code in lines_of(rel, text):
        flat = " ".join(line.split())
        name_only = in_code and " " not in flat   # 'pending' in code is a name, not a sentence
        for rule, rx, fix in rules:
            if name_only and rule in CODE_NAME_SKIP:
                continue
            for mt in rx.finditer(flat):
                key = next((k for k in allowed if k[0] == rel and _inside(flat, k[1], mt.start(), mt.end())), None)
                if key:
                    used.add(key)
                    hits[key] += 1
                    continue
                a = max(0, mt.start() - 40)
                probs.append(f"{rel}:{where}: [{rule}] {mt.group(0)!r} in ...{flat[a:mt.end() + 30]}...: {fix}")
    for key, n in hits.items():    # an allow line covers exactly the hits it names: a copy more (or fewer) fails
        i, want = allowed[key]
        if n != want:
            probs.append(f"{ALLOW.as_posix()}:{i}: {key[1]!r} covers {n} hit(s) on {rel}, the allow line says {want}: "
                         "a copy more is a real hit to fix; fewer, lower the count")
    return probs


def lint_repo(repo):
    problems = []
    allowed, used, extra = load_allow(repo, problems), set(), load_private(repo)
    for rel in public_files(repo):
        problems += lint_text(rel, (repo / rel).read_text(errors="replace"), allowed, used, extra)
    for k, (i, _) in sorted(allowed.items(), key=lambda kv: kv[1]):
        if k not in used:
            problems.append(f"{ALLOW.as_posix()}:{i}: {k[1]!r} matches nothing on {k[0]} any more (remove the line)")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--paths", nargs="*", help="lint these files instead of the repo (no allow list, no exemptions by path)")
    a = ap.parse_args(argv)
    if a.paths:
        problems, extra = [], load_private(Path(a.repo).resolve())
        for p in a.paths:
            problems += lint_text(Path(p).name, Path(p).read_text(errors="replace"), {}, set(), extra)
    else:
        problems = lint_repo(Path(a.repo).resolve())
    for p in problems:
        print(p)
    print(f"docs_lint: {len(problems)} problem(s)" if problems else "docs_lint: OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
