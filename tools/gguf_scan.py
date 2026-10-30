#!/usr/bin/env python3
"""The GGUF candidate list: the most downloaded and most liked GGUF repos on Hugging Face, each file read from its
header only (an HTTP range read up to the tensor data; nothing else is downloaded), and probed by this engine's own
loader on a sparse copy of that header, so the status is the engine's own answer, never a guess.

  python tools/gguf_scan.py OUT_DIR [--top 60] [--per-author 40] [--authors a,b,...] [--repos r1,r2,...]

Writes OUT_DIR/headers/<repo>__<file>.hdr (the header bytes), OUT_DIR/candidates.json (one entry per repo) and
OUT_DIR/ranking.json (the missing pieces ranked by how many candidates each would unlock). Needs network (HF API only)
and the gguf package; no chip, no model download. HF_TOKEN or ~/.cache/huggingface/token is used for gated repos.
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
AUTHORS = ("bartowski", "unsloth", "ggml-org", "lmstudio-community", "Qwen", "google", "mistralai", "microsoft",
           "ibm-granite", "HuggingFaceTB", "MaziyarPanahi", "mradermacher", "QuantFactory", "TheBloke", "NousResearch",
           "LiquidAI", "nvidia", "openai", "deepseek-ai", "meta-llama", "second-state", "tiiuae", "allenai")
VT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        p = os.path.expanduser("~/.cache/huggingface/token")
        t = open(p).read().strip() if os.path.exists(p) else None
    return t


def get(url, rng=None, tries=4):
    h = {"User-Agent": "mashbox-gguf-scan/1"}
    t = token()
    if t and url.startswith(HF):
        h["Authorization"] = f"Bearer {t}"
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise
            time.sleep(2 ** k)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** k)
    raise RuntimeError(f"GET {url} failed {tries} times")


class More(Exception):
    pass


def parse(buf):
    """GGUF v2/v3 header from bytes -> (meta {key: value or ('array', type, n)}, tensors [(name, type, shape)], data_start).
    Raises More when buf ends before the tensor table does. Arrays of numbers up to 4096 long are kept (per-layer head
    counts); longer arrays and string arrays keep their length only (the tokenizer arrays)."""
    pos = 0

    def rd(fmt):
        nonlocal pos
        n = struct.calcsize(fmt)
        if pos + n > len(buf):
            raise More
        v = struct.unpack_from(fmt, buf, pos)[0]
        pos += n
        return v

    def rstr():
        nonlocal pos
        n = rd("<Q")
        if pos + n > len(buf):
            raise More
        s = bytes(buf[pos:pos + n]).decode("utf-8", "replace")
        pos += n
        return s

    def val(vt):
        nonlocal pos
        if vt == 8:
            return rstr()
        if vt == 9:
            it, n = rd("<I"), rd("<Q")
            if it in VT and n <= 4096:
                return [rd(VT[it]) for _ in range(n)]
            if it in VT:
                pos += n * struct.calcsize(VT[it])
                if pos > len(buf):
                    raise More
            else:
                for _ in range(n):
                    val(it)
            return ("array", it, n)
        return rd(VT[vt])
    if bytes(buf[:4]) != b"GGUF":
        raise ValueError("not a GGUF file")
    pos = 4
    ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    meta = {"__version__": ver}
    for _ in range(nkv):
        k = rstr()
        meta[k] = val(rd("<I"))
    tensors = []
    for _ in range(nt):
        name, nd = rstr(), rd("<I")
        dims = [rd("<Q") for _ in range(nd)]
        tensors.append((name, rd("<I"), tuple(reversed(dims))))
        rd("<Q")
    align = meta.get("general.alignment", 32)
    return meta, tensors, -(-pos // align) * align


def header(url, size):
    """-> (header bytes up to the tensor data, meta, tensors): range reads of 8, 32, 128 then 256 MB"""
    for n in (8 << 20, 32 << 20, 128 << 20, 256 << 20):
        n = min(n, size)
        buf = get(url, (0, n - 1))
        try:
            meta, tensors, start = parse(buf)
            return buf[:start], meta, tensors
        except More:
            if n >= size:
                raise
    raise RuntimeError("header over 256 MB")


def list_repos(authors, per_author, global_lists=True):
    seen = {}
    q = "filter=gguf&sort={s}&direction=-1&limit={n}"
    urls = [f"{HF}/api/models?author={a}&" + q.format(s="downloads", n=per_author) for a in authors]
    if global_lists:
        urls += [f"{HF}/api/models?" + q.format(s=s, n=200) for s in ("downloads", "likes")]
    for u in urls:
        try:
            for m in json.loads(get(u)):
                seen[m["id"]] = dict(repo=m["id"], downloads=m.get("downloads", 0), likes=m.get("likes", 0))
        except Exception as e:                          # an author with no models answers 404/empty
            print(f"list {u}: {e}", file=sys.stderr)
    return list(seen.values())


def select(repos, top):
    by_dl = sorted(repos, key=lambda r: -r["downloads"])
    by_lk = sorted(repos, key=lambda r: -r["likes"])
    out, ids = [], set()
    for r in by_dl[:top] + by_lk[:max(10, top // 4)]:
        if r["repo"] not in ids:
            ids.add(r["repo"])
            out.append(r)
    for i, r in enumerate(by_dl):
        r["rank_downloads"] = i + 1
    for i, r in enumerate(by_lk):
        r["rank_likes"] = i + 1
    return out


def blocked_by(meta, probe):
    """the missing piece, as a kind:name key, from the engine's own refusal (probe) and the header"""
    if probe.get("supported"):
        return None
    why = probe.get("why") or ""
    arch = meta.get("general.architecture")
    from mashbox.gguf import archs as GA, model_keys_missing
    if model_keys_missing(meta.get):
        return "not-a-model"
    if arch not in GA.ARCHS:
        return f"arch:{arch}"
    if "cannot read" in why or "block types" in why:
        return "quant:" + why.split("cannot read: ")[-1].split(".")[0] if "cannot read: " in why else "quant:?"
    if "pre-tokenizer" in why or "tokenizer" in why.lower():
        return f"tokenizer:{meta.get('tokenizer.ggml.model')}/{meta.get('tokenizer.ggml.pre')}"
    if "rope scaling" in why:
        return f"rope:{meta.get(f'{arch}.rope.scaling.type')}"
    if moe(meta):
        return f"moe-kernel:{arch}"
    return "other:" + why[:160]


def moe(meta):
    arch = meta.get("general.architecture")
    return int(meta.get(f"{arch}.expert_count", 0) or 0) > 0


def probe(hdr, size, path_hint):
    """the engine's own answer on a sparse copy of the header (tensor data reads as zeros): open_model + the MoE hold +
    the tokenizer and chat template built from the file -> dict(supported, why, stage)"""
    from mashbox.gguf import open_model, GGUFError, archs as GA, model_keys_missing
    from mashbox.gguf import tokenizer as GT
    meta = parse(hdr)[0]
    arch = meta.get("general.architecture")
    # open_model's refusals in its order: a file that is not a llama.cpp model goes on to open_model for its line; an
    # architecture this version does not run stops here (archs.module's line), without gguf-py's full parse
    if arch not in GA.ARCHS and not model_keys_missing(meta.get):
        return dict(supported=False, stage="load", why=GA.unsupported(arch, path_hint))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, os.path.basename(path_hint))
        with open(p, "wb") as fh:
            fh.write(hdr)
            fh.truncate(size)
        try:
            f, A, c = open_model(p)
            held = GA.held(f.arch, p)
            if held:
                return dict(supported=False, stage="moe_hold", why=held)
        except GGUFError as e:
            return dict(supported=False, stage="load", why=str(e))
        except Exception as e:                          # a crash in our loader is a bug to report, never a pass
            return dict(supported=False, stage="load_crash", why=f"{type(e).__name__}: {e}")
        try:
            fs = GT.files_dict(f, p)
            from tokenizers import Tokenizer
            Tokenizer.from_str(fs["tokenizer.json"])
        except GGUFError as e:
            return dict(supported=False, stage="tokenizer", why=str(e))
        except Exception as e:
            return dict(supported=False, stage="tokenizer_crash", why=f"{type(e).__name__}: {e}")
        return dict(supported=True, stage="ok", why=None, layers=c.L)


def scan_repo(r, out):
    from mashbox.gguf import resolve as GR
    info = json.loads(get(f"{HF}/api/models/{r['repo']}?blobs=true"))
    sib = [s for s in info.get("siblings", []) if s["rfilename"].endswith(".gguf")]
    files = [s["rfilename"] for s in sib]
    card = info.get("cardData") or {}
    e = dict(r, revision=info.get("sha"), gated=info.get("gated"), license=card.get("license"),
             license_name=card.get("license_name"), license_link=card.get("license_link"), base_model=card.get("base_model"),
             n_gguf_files=len(files), quants_in_repo=sorted({q for q in (_tag(p) for p in files) if q}))
    path = GR.pick(files)
    e["file"] = path
    if not path:
        e.update(status="no model file")
        return e
    s = next(x for x in sib if x["rfilename"] == path)
    i, n, _ = GR.split_info(path)
    size = s.get("size") or (s.get("lfs") or {}).get("size")
    e.update(bytes=size, sha256=(s.get("lfs") or {}).get("sha256"), shards=n,
             total_bytes=sum((x.get("size") or 0) for x in sib if x["rfilename"] in GR.shards(path)))
    url = f"{HF}/{r['repo']}/resolve/{info.get('sha')}/{urllib.parse.quote(path)}"
    try:
        hdr, meta, tensors = header(url, size)
    except urllib.error.HTTPError as ex:
        e.update(status=f"header not read: HTTP {ex.code}" + (" (gated)" if info.get("gated") else ""))
        return e
    hp = os.path.join(out, "headers", r["repo"].replace("/", "__") + "__" + os.path.basename(path) + ".hdr")
    open(hp, "wb").write(hdr)
    arch = meta.get("general.architecture")
    from gguf import GGMLQuantizationType as Q
    types = {}
    for _, t, _ in tensors:
        nm = Q(t).name if t in Q._value2member_map_ else f"type{t}"
        types[nm] = types.get(nm, 0) + 1
    e.update(arch=arch, name=meta.get("general.name"), header_bytes=len(hdr), header=os.path.basename(hp),
             tokenizer_model=meta.get("tokenizer.ggml.model"), pre=meta.get("tokenizer.ggml.pre"),
             vocab=(meta.get("tokenizer.ggml.tokens") or ("", 0, 0))[2] if isinstance(meta.get("tokenizer.ggml.tokens"), tuple) else None,
             chat_template="tokenizer.chat_template" in meta, types=types, ctx=meta.get(f"{arch}.context_length"),
             layers=meta.get(f"{arch}.block_count"), moe=moe(meta), experts=meta.get(f"{arch}.expert_count"),
             experts_used=meta.get(f"{arch}.expert_used_count"), rope_scaling=meta.get(f"{arch}.rope.scaling.type"),
             file_type=meta.get("general.file_type"), header_license=meta.get("general.license"),
             params_label=meta.get("general.size_label"), vision=any("mmproj" in p for p in files))
    if n > 1:
        e.update(status="split file: probed on shard 1's header", probe=dict(supported=None, stage="split", why=None))
        meta_only = probe(hdr, size, path)
        e["probe"] = meta_only
    else:
        e["probe"] = probe(hdr, size, path)
    e["blocked_by"] = blocked_by(meta, e["probe"])
    e["status"] = "runs now (probe)" if e["probe"].get("supported") else f"blocked: {e['blocked_by']}"
    return e


def _tag(p):
    import re
    m = re.search(r"(I?Q\d[_A-Z0-9]*|BF16|F16|F32|MXFP4[_A-Z0-9]*|TQ\d_\d)", os.path.basename(p), re.I)
    return m.group(1).upper() if m else None


def rank(entries):
    """missing piece -> candidates it alone would unlock (only blocker) and downloads behind them"""
    out = {}
    for e in entries:
        b = e.get("blocked_by")
        if b:
            x = out.setdefault(b, dict(piece=b, candidates=0, downloads=0, repos=[]))
            x["candidates"] += 1
            x["downloads"] += e.get("downloads", 0)
            x["repos"].append(e["repo"])
    return sorted(out.values(), key=lambda x: (-x["candidates"], -x["downloads"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--per-author", type=int, default=40)
    ap.add_argument("--authors", default=",".join(AUTHORS))
    ap.add_argument("--repos", help="scan these repos only (comma-separated)")
    ap.add_argument("--no-global", action="store_true", help="the named authors only, not Hub-wide top lists")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "headers"), exist_ok=True)
    t0 = time.time()
    if a.repos:
        cands = [dict(repo=x, downloads=0, likes=0) for x in a.repos.split(",")]
    else:
        cands = select(list_repos(a.authors.split(","), a.per_author, not a.no_global), a.top)
    entries = []
    for i, r in enumerate(cands):
        try:
            e = scan_repo(r, a.out)
        except Exception as ex:
            e = dict(r, status=f"scan error: {type(ex).__name__}: {ex}"[:300])
        entries.append(e)
        print(f"[{i + 1}/{len(cands)}] {r['repo']}: {e.get('file')} {e.get('arch')} {e['status']}", flush=True)
    meta = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t0, 1),
                candidates=len(entries), runs_now=sum(1 for e in entries if e.get("probe", {}).get("supported")))
    json.dump(dict(meta=meta, entries=entries), open(os.path.join(a.out, "candidates.json"), "w"), indent=1)
    json.dump(rank(entries), open(os.path.join(a.out, "ranking.json"), "w"), indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
