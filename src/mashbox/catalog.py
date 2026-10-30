"""The model catalog: one render of models.json for `mashbox models`, for a docs page (render_markdown) and for the
architecture list in docs/gguf.md, and the same entries as JSON for other pages (`mashbox models --json`).

models.json is the one source:
  models[]          our models (MASHBOX builds); each carries a `catalog` block (source, format, family, params, an
                    experimental flag, number)
  gguf[]            GGUF files from Hugging Face run on a chip with this engine: the repo, file, revision and sha256 that
                    ran, the command, and per chip the run's outcome and speed
  engine_kernels[]  the engine's kernels package, one row per (chip, JAX), fetched and checked on first use
Every view is one tree, sorted the same way whatever the order of the rows: publisher (MASHBOX Labs first, as one
publisher like the others, then by name), then model (by name), then format or file. A gguf[] row names its model id and
publisher (the Hugging Face owner of its repo). The support level is computed here (level()), never stored in a row. Everything shown is read from
those rows; nothing is typed twice. Standard library only: the docs check imports this module where jax and gguf are
not installed."""
import ast
import json
import os

KIND_LABEL = {"llm-gguf": "LLM, GGUF", "llm-tpu": "LLM", "world": "World model", "video": "Video", "image": "Image",
              "voice": "Voice", "rig": "Rigging"}
FORMAT_LABEL = {"bshj-build": "BSHJ build", "port": "TPU port", "own-model": "MASHBOX model"}
LEVELS = ("MASHBOX build", "Tuned", "Tested", "Quick check", "Checked on CPU", "Architecture supported")
# a QUICK chip record (record_kind "quick"): QUICK_PROMPTS zero-flag `mashbox run` greedy runs of QUICK_TOKENS tokens each,
# compared with llama.cpp; its row says only that, with the decode speed for one request it measured
QUICK_KIND, QUICK_PROMPTS, QUICK_TOKENS = "quick", 4, 32
MASHBOX = "MASHBOX Labs"
CHIPS = ("v5e", "v6e", "v5p-8", "v6e-4")
KERNEL_PATHS = ("bundle", "exact")
UNQUANTIZED = ("F32", "F16", "BF16")
HERE = os.path.dirname(os.path.abspath(__file__))


def registry(path=None):
    with open(path or os.path.join(HERE, "models.json"), encoding="utf-8") as f:
        return json.load(f)


def _literal(path, name):
    """the literal value of a top-level `name = ...` in a Python file, read from its source (no import); None when the
    file, the name or a literal value is missing"""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def archs(src=None):
    """The GGUF architectures this package runs: the keys of mashbox.gguf.archs.ARCHS, read from its source (no import, so
    no gguf package needed). [] in a package without the GGUF reader."""
    return list(_literal(src or os.path.join(HERE, "gguf", "archs", "__init__.py"), "ARCHS") or {})


def arch_models(src=None):
    """architecture -> its Models label (docs/gguf.md): the MODELS dict of the module ARCHS maps it to, read from that
    module's source (no import, so no jax at render time). An architecture its own module does not label is left out,
    and its row shows its name only."""
    p = src or os.path.join(HERE, "gguf", "archs", "__init__.py")
    out = {}
    for arch, mod in (_literal(p, "ARCHS") or {}).items():
        models = _literal(os.path.join(os.path.dirname(p), mod.rsplit(".", 1)[-1] + ".py"), "MODELS")
        label = models.get(arch) if isinstance(models, dict) else None
        if isinstance(label, str) and label.strip():
            out[arch] = label
    return out


def live(reg):
    """Rows a view shows: a model with "launch" set is left out of every view."""
    return [m for m in reg["models"] if not m.get("launch")]


def chips_run(row):
    """["CPU", "TPU v6e-1", ...]: CPU when the cpu row passes, then the accelerator of every passing chip row, in CHIPS
    order, each named once."""
    out = ["CPU"] if ((row.get("chips") or {}).get("cpu") or {}).get("status") == "passing" else []
    for c in CHIPS:
        r = (row.get("chips") or {}).get(c) or {}
        if r.get("status") == "passing":
            name = f"TPU {r.get('accelerator') or c}"
            if name not in out:
                out.append(name)
    return out


def byline(m):
    c = m.get("catalog") or {}
    if c.get("format") == "own-model":
        return "Trained by MASHBOX Labs"
    return f"Made by {m['upstream']['maker']}, ported to TPU by MASHBOX Labs"


def quants(m):
    """The quant types of the model's weights files (files[] rows of kind weights), in file order."""
    out = []
    for f in m.get("files") or []:
        q = f.get("quant_type")
        if f.get("kind") == "weights" and q and q not in out:
            out.append(q)
    return out


def number(m):
    """The one number shown for a model ({"big", "label", "chip"}), from catalog.number: a headline or a result by index.
    "chip" is the chip of the result behind it (a headline's first result), so the number is never shown without it."""
    n = (m.get("catalog") or {}).get("number")
    if not n:
        return None
    if "headline" in n:
        h = m["headline"][n["headline"]]
        return {"big": h["big"], "label": h["label"], "chip": m["results"][h["results"][0]]["chip"]}
    r = m["results"][n["result"]]
    return {"big": r["value"].split(" (", 1)[0], "label": r["metric"].split(" (", 1)[0], "chip": r["chip"]}


def _params(p):
    if not p or p.get("total") is None:
        return None
    t = f"{p['total']:g}B"
    return t + (f", {p['active']:g}B active" if p.get("active") is not None else "")


def cpu_checked(row):
    """True for a gguf[] row whose CPU check against llama.cpp passed (cpu_check.status "passing", copied from its
    sealed record; the record guard holds every number to it)."""
    return (row.get("cpu_check") or {}).get("status") == "passing"


def cpu_line(row):
    """The CPU check of a gguf[] row in words, with its numbers: "Runs, checked against llama.cpp on CPU: ..."; None
    when its CPU check did not pass."""
    if not cpu_checked(row):
        return None
    c = row["cpu_check"]
    same, block, g = c["identical_weights"], c["block_path"], c["greedy"]
    return (f"Runs, checked against llama.cpp on CPU: same weights NMSE {same['nmse']:g}, top-1 {same['top1']}; "
            f"llama.cpp's own path top-1 {block['top1']}; greedy {g['prompts']} x {g['tokens']} tokens: {g['full']} "
            f"identical, {g['near_tie']} near tie")


def tpu_line(row):
    """What a gguf[] row says about TPUs: the chips it passed on, else the arch x block-type smoke that covers it with
    that smoke's own number, else "TPU: Not tested yet"."""
    run = chips_run(row)
    if run:
        return "Runs on: " + ", ".join(run)
    for c in CHIPS:
        sm = ((row.get("chips") or {}).get(c) or {}).get("smoke")
        if sm:
            return (f"TPU {sm['accelerator']}: {sm['decode_tok_s_1_user']:g} tok/s, 1 user, measured on {sm['row']} "
                    "(same architecture and block types)")
    return "TPU: Not tested yet"


def quick_only(row):
    """True for a gguf[] row whose passing chip runs are all QUICK records (record_kind "quick")"""
    runs = [r for r in ((row.get("chips") or {}).values()) if (r or {}).get("status") == "passing"]
    return bool(runs) and all(r.get("record_kind") == QUICK_KIND for r in runs)


def quick_line(row):
    """What a gguf[] row's QUICK records say, and nothing more, one clause per passing QUICK chip in CHIPS order:
    "Runs on TPU v6e-1: short check against llama.cpp (4 prompts x 32 tokens), 0.52 tok/s for one request" ("Runs on TPU
    v6e-1 with the engine's TPU kernels: ..." when that run ran on the kernels); None when the row has none."""
    out = []
    for c in CHIPS:
        r = (row.get("chips") or {}).get(c) or {}
        if r.get("status") != "passing" or r.get("record_kind") != QUICK_KIND:
            continue
        on = " with the engine's TPU kernels" if r.get("kernel_path") == "bundle" else ""
        out.append(f"Runs on TPU {r.get('accelerator') or c}{on}: short check against llama.cpp ({QUICK_PROMPTS} prompts x "
                   f"{QUICK_TOKENS} tokens), {r['measured']['decode_tok_s_1_user']:g} tok/s for one request")
    return "; ".join(out) or None


def level(row, ours=False):
    """The support level of a row, computed from what the row records (never stored): every models[] row is a MASHBOX
    build; a gguf[] row whose passing chip runs are all QUICK records is Quick check; a gguf[] row with another passing
    chip run is Tested, and Tuned when it also has a tuned piece; a gguf[] row with no chip run whose CPU check against
    llama.cpp passed is Checked on CPU; any other row claims only its architecture (validate() keeps such a row out of
    models.json)."""
    if ours:
        return LEVELS[0]
    if quick_only(row):
        return LEVELS[3]
    if row.get("tuned") and chips_run(row):
        return LEVELS[1]
    if chips_run(row):
        return LEVELS[2]
    return LEVELS[4] if cpu_checked(row) else LEVELS[5]


def ours(reg):
    rows = []
    for m in live(reg):
        c = m.get("catalog") or {}
        rows.append({
            "id": m["id"], "name": m["name"], "source": c.get("source", "ours"), "byline": byline(m),
            "maker": m["upstream"]["maker"], "kind": KIND_LABEL.get(m["kind"], m["kind"]), "in": m["in"], "out": m["out"],
            "summary": m["summary"], "format": FORMAT_LABEL.get(c.get("format"), c.get("format")), "family": c.get("family"),
            "params": _params(c.get("params_b")), "quants": quants(m), "level": level(m, ours=True),
            "experimental": bool(c.get("experimental")), "number": number(m), "chips": chips_run(m), "run": m["run"],
            "install": m.get("install") or reg["install"],
            "hf": f"https://huggingface.co/{m['hf_repo']}" if m.get("hf_repo") else None,
            "docs": f"https://mashbox.gg/docs/models/{m['id']}",
            "upstream": {"repo": f"https://huggingface.co/{m['upstream']['hf']}", "license": m["upstream"]["license"],
                         "license_link": m["upstream"].get("license_link")},
        })
    return rows


def speed(g):
    """(tok/s for one request, chip) from the first passing chip row of a gguf[] entry that ran on the kernel bundle,
    else (None, None). An exact-path row's number is the reference path's; no catalog view shows it as the file's speed."""
    for c in CHIPS:
        r = (g.get("chips") or {}).get(c) or {}
        if (r.get("status") == "passing" and r.get("kernel_path") == "bundle"
                and (r.get("measured") or {}).get("decode_tok_s_1_user") is not None):
            return r["measured"]["decode_tok_s_1_user"], f"TPU {r.get('accelerator') or c}"
    return None, None


def gguf_rows(reg):
    rows = []
    for g in reg.get("gguf") or []:
        tok, chip = speed(g)
        rows.append({
            "id": g["id"], "model": g["model"], "publisher": g["publisher"], "family": g["family"], "maker": g["maker"],
            "file": g["file"], "tag": g["tag"],
            "repo": g["repo"], "revision": g["revision"], "sha256": g["sha256"], "gb": round(g["bytes"] / 1e9, 2),
            "params": _params(g.get("params_b")), "arch": g["arch"], "moe": g["moe"], "types": g["types"],
            "format": "GGUF", "level": level(g), "experimental": bool(g.get("experimental")), "chips": chips_run(g),
            "tok_s_1_user": tok, "tok_s_chip": chip, "run": g["run"], "cpu": cpu_line(g),
            "tpu": quick_line(g) or (None if chips_run(g) else tpu_line(g)),
            "file_url": f"https://huggingface.co/{g['repo']}/blob/{g['revision']}/{g['file']}",
            "license": g["license"], "license_link": g.get("license_link"), "base": g.get("base"),
        })
    return rows


def moe_arch(arch, label=""):
    """True for a mixture-of-experts architecture: "moe" in its name or "mixture of experts" in its Models label (the
    release check uses the same rule)."""
    return "moe" in arch.lower() or "mixture of experts" in (label or "").lower()


def arch_rows(reg, src=None):
    """One row per architecture in ARCHS that has a passing chip record: the catalog holds a passing chip row of a GGUF
    file of it (the record guard holds that row to its sealed chip record), and for a mixture-of-experts architecture a
    passing chip row of a MoE file of it (no MoE is listed before its routed experts run on the kernel). This is the
    listing (docs/gguf.md and the views) only: an unlisted dense architecture still runs on the exact path, and
    gguf.archs.held runs a MoE file only when a passing row is of that file (its sha256), listed or not."""
    rows, labels = [], arch_models(src)
    for a in archs(src):
        gg = [g for g in reg.get("gguf") or [] if g["arch"] == a]
        moe = moe_arch(a, labels.get(a))
        if not any(chips_run(g) and (g.get("moe") is True or not moe) for g in gg):
            continue
        rows.append({"arch": a, "models": labels.get(a, ""), "catalog_files": [g["id"] for g in gg]})
    return rows


def _row_key(r):
    return (r["format"].casefold(), str(r.get("tag") or "").casefold(), str(r.get("file") or r["id"]).casefold(), r["id"])


def publishers(reg):
    """[{"name", "hf", "models": [{"id", "name", "base", "rows": [...]}]}]: every row under its publisher (MASHBOX Labs
    for models[]; a gguf[] row's publisher, the Hugging Face owner of its repo) and its model (a models[] row is its own
    model; a gguf[] row joins its model id). Sorted, never in file order: MASHBOX Labs first, then publishers by name;
    models by name; rows by format, tag and file."""
    out = {}

    def add(pub, hf, mid, name, base, row):
        p = out.setdefault(pub, {"name": pub, "hf": hf, "models": {}})
        p["models"].setdefault(mid, {"id": mid, "name": name, "base": base, "rows": []})["rows"].append(row)
    org = (reg.get("hf") or {}).get("org")
    for r in ours(reg):
        add(MASHBOX, f"https://huggingface.co/{org}" if org else None, r["id"], r["name"], None, r)
    for r in gguf_rows(reg):
        add(r["publisher"], f"https://huggingface.co/{r['publisher']}", r["model"],
            (r["base"] or r["family"]).split("/")[-1], r["base"], r)
    pubs = sorted(out.values(), key=lambda p: (p["name"] != MASHBOX, p["name"].casefold(), p["name"]))
    return [dict(p, models=[dict(m, rows=sorted(m["rows"], key=_row_key))
                            for m in sorted(p["models"].values(), key=lambda m: (m["name"].casefold(), m["id"]))])
            for p in pubs]


def entries(reg, src=None):
    """Everything a catalog view needs: {"version", "publishers": publishers(reg), "archs": [...]}."""
    return {"version": reg["version"], "publishers": publishers(reg), "archs": arch_rows(reg, src)}


# ------------------------------------------------------------------------------------------------ text (mashbox models)

def _speed(r):
    if r.get("number"):
        return f"{r['number']['big']} on {r['number']['chip']}, {r['number']['label']}"
    if r.get("tok_s_1_user") is not None:
        return f"{r['tok_s_1_user']:g} tok/s for one request on {r['tok_s_chip']}"
    return ""


def render_text(reg, src=None):
    e = entries(reg, src)
    out = [f"Models in mashbox {e['version']}, by publisher"]
    for p in e["publishers"]:
        out += ["", p["name"]]
        for m in p["models"]:
            for r in m["rows"]:
                flag = "  Experimental" if r["experimental"] else ""
                if r["format"] != "GGUF":
                    fmt = r["format"] + (f": {', '.join(r['quants'])}" if r["quants"] else "")
                    out.append(f"  {r['name']}  [{r['id']}]  {r['level']}{flag}")
                    out.append(f"    {r['byline']}. {r['kind']}, {r['in']} in, {r['out']} out. {fmt}.")
                    if r["chips"]:
                        out.append(f"    Runs on: {', '.join(r['chips'])}")
                    if _speed(r):
                        out.append(f"    {_speed(r)}")
                    out.append(f"    {r['run']}")
                    continue
                if r is m["rows"][0]:
                    out.append(f"  {m['name']}" + (f"  (base {m['base']})" if m["base"] else ""))
                out.append(f"    {r['file']}  {r['tag']}  {r['gb']:g} GB" + (f"  {r['params']}" if r["params"] else "")
                           + f"  {r['level']}{flag}" + (f"  {_speed(r)}" if _speed(r) else ""))
                out += [f"      {x}" for x in (r["cpu"], r["tpu"]) if x]
                out.append(f"      {r['run']}")
    if e["archs"]:
        out.append("")
        out.append(f"{ARCH_LINE[:-1]} (general.architecture):")
        out.append("  " + ", ".join(f"{a['arch']}" + (f" ({a['models']})" if a["models"] else "") for a in e["archs"]))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ markdown (docs pages)

def _cell(s):
    return str(s).replace("|", "\\|")


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows])


def render_markdown(reg, src=None):
    """The catalog as Markdown (the block between <!-- catalog:start --> and <!-- catalog:end --> in a docs page): one
    section per publisher, one table row per model format or file; then one line saying so when the catalog lists no
    GGUF file, and (as in render_text) the GGUF architectures with a passing chip test when there are any."""
    e, out = entries(reg, src), []
    for p in e["publishers"]:
        rows = []
        for m in p["models"]:
            for r in m["rows"]:
                flag = " (Experimental)" if r["experimental"] else ""
                if r["format"] != "GGUF":
                    fmt = r["format"] + (f": {', '.join(r['quants'])}" if r["quants"] else "")
                    model = f"**[{r['name']}](models/{r['id']}.md)**{flag}<br>{r['byline']}"
                    what = f"{fmt}<br>{r['kind']}: {r['in']} in, {r['out']} out"
                else:
                    model = f"**{m['name']}**{flag}" + (f"<br>base {m['base']}" if m["base"] else "")
                    what = f"[{r['file']}]({r['file_url']})<br>GGUF {r['tag']}, {r['gb']:g} GB"
                level = r["level"] + "".join(f"<br>{x}" for x in (r.get("cpu"), r.get("tpu")) if x)
                rows.append([model, what, level, ", ".join(r["chips"]), _speed(r), f"`{r['run']}`"])
        head = f"## {p['name']}" + (f" ([Hugging Face]({p['hf']}))" if p["hf"] else "")
        out += ([""] if out else []) + [head, "", _table(["Model", "Format or file", "Level", "Runs on", "Speed", "Run"],
                                                           rows)]
    if not any(r["format"] == "GGUF" for p in e["publishers"] for m in p["models"] for r in m["rows"]):
        out += ["", "No GGUF file is listed in this version."]
    if e["archs"]:
        out += ["", f"{ARCH_LINE[:-1]} (`general.architecture`): "
                + ", ".join(f"`{a['arch']}`" + (f" ({a['models']})" if a["models"] else "") for a in e["archs"]) + "."]
    return "\n".join(out)


ARCH_LINE = "GGUF architectures with a passing chip test:"
NO_ARCHS = "No architecture has a GGUF file with a passing chip test in this version."


def archs_block(reg, src=None):
    """The block between <!-- archs:start --> and <!-- archs:end --> in docs/gguf.md: every architecture arch_rows lists
    (a passing chip record), with the catalog's GGUF files of each; NO_ARCHS when it lists none."""
    rows = arch_rows(reg, src)
    if not rows:
        return NO_ARCHS
    head = ["Architecture (`general.architecture`)", "Models"]
    with_files = any(r["catalog_files"] for r in rows)
    if with_files:
        head.append("Files in the catalog")
    body = []
    for r in rows:
        cells = [f"`{r['arch']}`", r["models"]]
        if with_files:
            cells.append(", ".join(r["catalog_files"]))
        body.append(cells)
    return _table(head, body)
