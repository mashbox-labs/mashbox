"""Model lists for the public pages, rendered from models.json: the count line, the short list of GGUF files tested on
TPU, the docs catalog by family, and the maker check of every GGUF row.

Standard library only. It reads mashbox/catalog.py (rows, levels, speeds) and tools/gguf_rows.py (MAKER_NAMES), each
loaded by path from this tool's own tree, and changes neither. catalog.py is in the code set of every GGUF chip record,
so the page layout lives here and a new layout never touches a measured record. hf_github_sync_check.py renders the
README, the org profile, llms.txt and the docs catalog with these functions; no page types a list or a count.
"""
import functools
import importlib.util
import re
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent
TOP_GGUF = 8   # GGUF rows the README and the org profile show; the docs catalog and `mashbox models` show every row


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def catalog():
    """mashbox/catalog.py of this tree, loaded by path."""
    return _load("_model_lists_catalog", TREE / "src" / "mashbox" / "catalog.py")


@functools.lru_cache(maxsize=None)
def maker_names():
    """tools/gguf_rows.py's MAKER_NAMES (the one table of makers), loaded by path; never copied."""
    return dict(_load("_model_lists_gguf_rows", TREE / "tools" / "gguf_rows.py").MAKER_NAMES)


# ------------------------------------------------------------------------------------------------ counts

def _n(k, one, many):
    return f"{k} {one if k == 1 else many}"


def counts(reg):
    """{"ports", "own", "files", "gguf_models", "entries"}: MASHBOX ports (models[] rows a view shows that port another
    maker's model), MASHBOX models (format own-model: trained by MASHBOX Labs), GGUF files (gguf[] rows), the models
    those files are of (once per model id) and entries = ports + own + files. Every count a page prints is read from
    here; no page types a count."""
    rows, g = catalog().live(reg), reg.get("gguf") or []
    own = sum((m.get("catalog") or {}).get("format") == "own-model" for m in rows)
    return {"ports": len(rows) - own, "own": own, "files": len(g), "gguf_models": len({r["model"] for r in g}),
            "entries": len(rows) + len(g)}


def count_parts(reg):
    """[(k, words)] of the count line, each part a number the entries total adds up from."""
    c = counts(reg)
    return [(k, w) for k, w in ((c["ports"], _n(c["ports"], "MASHBOX port", "MASHBOX ports")),
                                (c["own"], _n(c["own"], "MASHBOX model", "MASHBOX models")),
                                (c["files"], _n(c["files"], "GGUF file", "GGUF files")
                                 + f" of {_n(c['gguf_models'], 'model', 'models')}")) if k]


def count_line(reg):
    """"17 entries: 2 MASHBOX ports, 1 MASHBOX model and 14 GGUF files of 12 models": the parts add up to the total."""
    words = [w for _, w in count_parts(reg)]
    body = ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else "".join(words)
    return f"{_n(counts(reg)['entries'], 'entry', 'entries')}: {body}" if words else "0 entries"


# ------------------------------------------------------------------------------------------------ makers

def maker_problems(reg):
    """[(row id, problem)] for gguf[] rows whose maker is not their base model's maker: base is the base model's Hugging
    Face repo (org/name), its org is in MAKER_NAMES, maker is that name, and maker is never the model's name."""
    names, out = maker_names(), []
    for g in reg.get("gguf") or []:
        base, maker = g.get("base") or "", g.get("maker") or ""
        org = base.split("/")[0] if re.fullmatch(r"[^/\s]+/[^/\s]+", base) else None
        if org is None:
            out.append((g["id"], f"base {base or None!r} is not the base model's Hugging Face repo (org/name)"))
        elif org not in names:
            out.append((g["id"], f"base org {org!r} has no maker name in tools/gguf_rows.py MAKER_NAMES"))
        elif maker != names[org]:
            out.append((g["id"], f"maker {maker!r} is not {names[org]!r}, the maker of {base}"))
        if maker and maker.casefold() in {str(g.get("model") or "").casefold(), base.split("/")[-1].casefold()}:
            out.append((g["id"], f"maker {maker!r} is the model's name, not its maker"))
    return out


# ------------------------------------------------------------------------------------------------ the short list, families

def gguf_name(r):
    """A gguf row's model name: the base model's repo name, else its family (as catalog.publishers() names it)."""
    return (r["base"] or r["family"]).split("/")[-1]


def tested(reg):
    """gguf[] rows with a passing chip test (level Tuned or Tested), Tuned first, then by maker, model and file: the
    short list the README and the org profile print (the first TOP_GGUF). The rows carry no date, so no view calls them
    new."""
    cat = catalog()
    rows = [r for r in cat.gguf_rows(reg) if r["chips"]]
    return sorted(rows, key=lambda r: (cat.LEVELS.index(r["level"]), r["maker"].casefold(), gguf_name(r).casefold(),
                                       cat._row_key(r)))


FAMILY_RE = re.compile(r"^([A-Za-z]+)[-_ ]?\d+(?:[.\-_ ]|[Bb]\b|$)")   # a version or size follows the word


def family(r):
    """The family a row is listed under: the model name's leading word when a version number follows it (Qwen3.8-27B
    and Qwen2.5-0.5B-Instruct -> Qwen, gemma-3-1b-it -> Gemma, Falcon3-1B -> Falcon, Mistral-7B -> Mistral), else the
    whole name (Hy-Nomas, Track2Rig). A models[] row is named by its catalog family, a gguf[] row by its base model."""
    name = (r.get("family") if r["format"] != "GGUF" else None) or r.get("name") or gguf_name(r)
    m = FAMILY_RE.match(name)
    word = m.group(1) if m else name
    return word[:1].upper() + word[1:]


def families(reg):
    """[{"name", "rows"}]: every row (models[] and gguf[]) under its family, as the docs catalog groups them: families
    with one of our builds first, then by name; in a family our builds first, then by model and file."""
    cat, out = catalog(), {}
    for r in cat.ours(reg) + cat.gguf_rows(reg):
        out.setdefault(family(r), []).append(r)
    key = lambda r: (r["format"] == "GGUF", (r.get("name") or gguf_name(r)).casefold(), cat._row_key(r))  # noqa: E731
    return [{"name": k, "rows": sorted(v, key=key)}
            for k, v in sorted(out.items(), key=lambda kv: (all(r["format"] == "GGUF" for r in kv[1]),
                                                            kv[0].casefold(), kv[0]))]


# ------------------------------------------------------------------------------------------------ the docs catalog

def _cell(s):
    return str(s).replace("|", "\\|")


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows])


BUILDS = "MASHBOX builds"
BUILDS_LINE = ("Models we port or make ourselves. A BSHJ build is our weights file for the model, run on our TPU "
               "kernels. Measured is each one's number against its baseline, on the chip named.")
GGUF = "GGUF files that run in our engine"
GGUF_LINE = ("GGUF files from other publishers, run as they are by mashbox. Level and Runs on say how far each file has "
             "been checked; a speed shows only where a chip run measured it on our kernels.")


def chip_note(chip):
    """The chip after a number, bracketed once: "(TPU v6e-1)", and "(CPU, AMD EPYC 7502P)" for a chip that carries its own
    brackets ("CPU (AMD EPYC 7502P)"), never "(CPU (AMD EPYC 7502P))"."""
    return "(%s)" % (chip.replace(" (", ", ").rstrip(")") if " (" in chip else chip)


def measured(reg, rid):
    """A MASHBOX build's measured lines: its headline numbers marked readme, each against its baseline with its chip
    (the README's Measured cell), else its first two results; [] when it has none."""
    m = next(x for x in reg["models"] if x["id"] == rid)
    hl = [h for h in m.get("headline") or [] if h.get("readme")]
    if hl:
        return [f"**{h['big']}** {h['label']} {chip_note(m['results'][h['results'][0]]['chip'])}" for h in hl]
    return [f"{r['metric'].split(' (', 1)[0]}: **{r['value'].split(' (', 1)[0]}** {chip_note(r['chip'])}"
            for r in (m.get("results") or [])[:2]]


def _gguf_table(rows, speed, run=True):
    """GGUF rows as one table; the Speed column only when a row has a speed (from a run on our kernels)."""
    sp = any(speed(r) for r in rows)
    keep = [i for i in range(len(GGUF_HEAD)) if (sp or GGUF_HEAD[i] != "Speed") and (run or GGUF_HEAD[i] != "Run")]
    return _table([GGUF_HEAD[i] for i in keep], [[c[i] for i in keep] for c in (gguf_row(r, speed) for r in rows)])


def gguf_row(r, speed):
    """One GGUF file as every list prints it: model by maker, file and its publisher, architecture, quant, level with
    its CPU and TPU lines, the chips it runs on, the speed (only from a run on our kernels) and the run line."""
    level = r["level"] + "".join(f"<br>{x}" for x in (r.get("cpu"), r.get("tpu")) if x)
    return [f"**{gguf_name(r)}**" + (" (Experimental)" if r["experimental"] else "") + f"<br>by {r['maker']}",
            f"[{r['file']}]({r['file_url']})<br>from {r['publisher']}, {r['gb']:g} GB", f"`{r['arch']}`", r["tag"],
            level, ", ".join(r["chips"]) or "n/a", speed(r) or "n/a", f"`{r['run']}`"]


GGUF_HEAD = ["Model", "GGUF file", "Architecture", "Quant", "Level", "Runs on", "Speed", "Run"]


def render_markdown(reg, src=None):
    """The docs catalog (the block between <!-- catalog:start --> and <!-- catalog:end -->), one page in two labelled
    tiers: the count line; "MASHBOX builds" (models[]: our ports and models, each measured against its baseline);
    "GGUF files that run in our engine" (gguf[]: every file with maker, architecture, quant, level and chips, one
    section per family, Qwen, Gemma, Llama ..., each row linking the exact file on Hugging Face), or one line saying
    the version lists none; then the GGUF architectures with a passing chip test (catalog.ARCH_LINE's words)."""
    cat = catalog()
    out, speed = [f"**{count_line(reg)}.**", "", f"## {BUILDS}", "", BUILDS_LINE, ""], cat._speed
    rows = []
    for r in sorted(cat.ours(reg), key=lambda r: (r["name"].casefold(), r["id"])):
        fmt = r["format"] + (f": {', '.join(r['quants'])}" if r["quants"] else "")
        rows.append([f"**[{r['name']}](models/{r['id']}.md)**" + (" (Experimental)" if r["experimental"] else "")
                     + f"<br>{r['byline']}",
                     f"{fmt}<br>{r['kind']}: {r['in']} in, {r['out']} out"
                     + (f"<br>[Hugging Face]({r['hf']})" if r["hf"] else ""),
                     r["level"] + "".join(f"<br>{x}" for x in (r.get("cpu"), r.get("tpu")) if x),
                     ", ".join(r["chips"]) or "n/a", "<br>".join(measured(reg, r["id"])) or "n/a", f"`{r['run']}`"])
    out += [_table(["Model", "Format", "Level", "Runs on", "Measured", "Run"], rows), "", f"## {GGUF}", "", GGUF_LINE]
    if not reg.get("gguf"):
        out += ["", "No GGUF file is listed in this version."]
    for f in families(reg):
        g = [r for r in f["rows"] if r["format"] == "GGUF"]
        if g:
            g.sort(key=lambda r: (cat.LEVELS.index(r["level"]), gguf_name(r).casefold(), cat._row_key(r)))
            out += ["", f"### {f['name']}", "", _gguf_table(g, speed)]
    archs = cat.arch_rows(reg, src)
    if archs:
        out += ["", f"{cat.ARCH_LINE[:-1]} (`general.architecture`): "
                + ", ".join(f"`{a['arch']}`" + (f" ({a['models']})" if a["models"] else "") for a in archs) + "."]
    return "\n".join(out)


def gguf_block(reg, catalog_url):
    """The README's and the org profile's GGUF tier: its heading and line, the first TOP_GGUF files with a passing chip
    test (tested), then the catalog link with the count line."""
    speed, top = catalog()._speed, tested(reg)
    out = [f"**{GGUF}**", "", GGUF_LINE, ""]
    if top:
        n = f"{min(len(top), TOP_GGUF)} of {len(top)} " if len(top) > TOP_GGUF else ""
        out += [f"Tested on TPU ({n}files with a passing chip test):" if n else "Tested on TPU:", "",
                _gguf_table(top[:TOP_GGUF], speed, run=False), ""]
    return "\n".join(out + [f"[Full catalog]({catalog_url}): {count_line(reg)}."])


def llms_lines(reg, catalog_url):
    """llms.txt lines for the GGUF tier: each file with a passing chip test, then the catalog link with the count line."""
    out = [f"- {gguf_name(r)} {r['tag']} by {r['maker']} ({r['arch']}, GGUF from {r['publisher']}): {r['level']}, "
           f"runs on {', '.join(r['chips'])}. `{r['run']}`" for r in tested(reg)]
    return out + [f"- [Full catalog]({catalog_url}): {count_line(reg)}."]
