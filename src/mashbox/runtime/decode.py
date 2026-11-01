"""`--decode auto | fast | long`: one server-wide setting, chosen at start (it decides memory), the same three words on every
model and chip. It is a preset over flags `mashbox run` already has (--ctx, --slots, --speculation, --draft-k,
--drafter-dir, the K/V store); the engine below sees nothing new. A flag given on the command line beats the setting, and
the start line names it.

  auto   today's behaviour, unchanged.
  fast   speculation with the model's block drafter (its weights in int8, made at load; one slot), the longest context
         that fits with the drafter counted (cli.plan_context). A model with no block drafter row, or a context under
         FAST_MIN_CTX with bf16 K/V and with 8-bit K/V, runs as auto, with one line saying why.
  long   one slot, the longest context that fits (8-bit K/V where the model's row records a passkey pass with it on this
         chip, else bf16), plus a small drafter: n-gram where the model's text kinds take it, never a block or DFlash
         drafter.

Speeds printed come only from the row's `chips.<chip>.measured.decode.<setting>` records; none = "not measured on this
chip". Nothing here is tuned per model: models enter through their row and their shapes.
"""
import os

SETTINGS = ("auto", "fast", "long")
FAST_MIN_CTX = 16384


def setting(a):
    v = (getattr(a, "decode", None) or os.environ.get("MASHBOX_DECODE", "") or "auto").strip().lower()
    if v not in SETTINGS:
        raise SystemExit(f"mashbox: --decode {v}: use auto, fast or long")
    return v


def block_drafter(m):
    """the row's block drafter entry (models.json `drafters`, kind trained, method block), or None"""
    return next((d for d in m.get("drafters") or [] if d.get("method") == "block"), None)


def chip_record(m, chip, s):
    """the row's measured record for setting s on this chip family, or None"""
    return (((m.get("chips") or {}).get(chip) or {}).get("measured") or {}).get("decode", {}).get(s)


def given(a, name):
    """True when the person gave the flag (argparse leaves an untouched flag at its default)"""
    v = getattr(a, name, None)
    return {"ctx": v not in (None, "auto"), "slots": v is not None, "speculation": v is not None,
            "draft_k": v is not None, "drafter_dir": bool(v)}.get(name, v is not None)


def resolve(a, m, chip, fetch_drafter=None, log=print, tp=1):
    """Set a's unset flags for the decode setting. -> dict(setting, set={flag: value}, kept=[flags given that win], why).
    fetch_drafter(entry) -> local folder of the block drafter (downloaded and checked). tp > 1 (tensor parallel): 8-bit
    K/V and speculation run on one chip in this version, so long keeps bf16 K/V and decodes plain, and fast runs as auto."""
    s = setting(a)
    out = dict(setting=s, set={}, kept=[], why="")
    if s == "auto":
        return out
    if tp > 1 and s == "fast":
        out.update(setting="auto", why=f"--decode fast: the block drafter runs on one chip in this version ({tp} chips "
                                       f"here); running as auto")
        return out

    def put(name, value):
        if given(a, name):
            out["kept"].append(name)
        else:
            setattr(a, name, value)
            out["set"][name] = value
    if s == "long":
        rec = chip_record(m, chip, "long") or {}
        put("ctx", "max")
        put("slots", 1)
        if tp > 1:
            out["why"] = (f"--decode long: 8-bit K/V and speculation run on one chip in this version ({tp} chips here); "
                          f"bf16 K/V, plain decoding")
            if not given(a, "speculation"):
                put("speculation", "off")
            return out
        if rec.get("kv") == "int8" and not os.environ.get("MASHBOX_KV_CACHE"):
            os.environ["MASHBOX_KV_CACHE"] = "int8"
            out["set"]["kv"] = "int8"
        if not given(a, "speculation") and "ngram" in (m.get("speculation_kinds") or ["ngram"]):
            put("speculation", "ngram")
        elif given(a, "speculation"):
            out["kept"].append("speculation")
        return out
    # fast
    e = block_drafter(m)
    if e is None and not given(a, "drafter_dir"):
        out.update(setting="auto", why=f"--decode fast: {m['id']} has no block drafter in this version; running as auto")
        return out
    if not given(a, "drafter_dir"):
        a.drafter_dir = fetch_drafter(e) if fetch_drafter else None
        out["set"]["drafter_dir"] = a.drafter_dir
    put("speculation", "block")           # --speculation block --drafter-dir (cli.dflash_drafter)
    put("slots", 1)
    put("ctx", "max")
    if e and e.get("k_default"):
        put("draft_k", int(e["k_default"]))
    if not os.environ.get("MASHBOX_DRAFTER_WEIGHTS"):
        os.environ["MASHBOX_DRAFTER_WEIGHTS"] = "int8"
        out["set"]["drafter_weights"] = "int8"
    return out


def fast_kv(fit_bf16, fit_int8):
    """fast's K/V store: bf16 when the plan reaches FAST_MIN_CTX with it, else 8-bit K/V when that reaches it, else None
    (fast does not fit: run as auto). fit_* = the longest one-slot context each plan gives (0 = nothing)."""
    if fit_bf16 >= FAST_MIN_CTX:
        return "bf16"
    if fit_int8 >= FAST_MIN_CTX:
        return "int8"
    return None


def on_weights(rec):
    """a record measured on other weights than the row's own file (same architecture, e.g. a fine-tune) says so"""
    return f" with {rec['weights']}" if rec.get("weights") else ""


def report(m, chip, r, ctx, slots, drafter, k):
    """what `mashbox run` prints for the decode setting: auto (the default, unchanged behaviour) prints nothing, or the
    one line saying why fast or long ran as auto; fast and long print start_line"""
    if r["setting"] == "auto":
        return r["why"] or None
    return start_line(m, chip, r, ctx, slots, drafter, k)


def start_line(m, chip, r, ctx, slots, drafter, k):
    """the one line `mashbox run` prints for the setting: what runs, the measured speed (or "not measured"), and the other
    settings' records"""
    s = r["setting"]
    rec = chip_record(m, chip, s)
    kv_now = os.environ.get("MASHBOX_KV_CACHE") or "bf16"
    on_kv = f" with {rec['kv']} K/V" if rec and rec.get("kv") and rec["kv"] != kv_now else ""
    sp = (f"{rec['tok_s_1_user']} tok/s [measured at {int(rec['ctx']):,}{on_kv}{on_weights(rec)}]"
          if rec and rec.get("tok_s_1_user") else "not measured on this chip")
    dr = f"speculation {drafter}" + (f", k {k}" if k else "") if drafter else "no speculation"
    kv = os.environ.get("MASHBOX_KV_CACHE") or "bf16"
    others = []
    for o in SETTINGS:
        ro = chip_record(m, chip, o)
        if o != s and ro and ro.get("tok_s_1_user"):
            others.append(f"--decode {o}: {int(ro['ctx']):,} tokens, {ro['tok_s_1_user']} tok/s{on_weights(ro)}")
    line = (f"decode {s}: {slots} user{'s' if slots != 1 else ''} x {ctx:,} tokens, {dr}, {kv} K/V, {sp}."
            + (" " + ". ".join(others) + "." if others else ""))
    if r["kept"]:
        line += " Given on the command line, kept: " + ", ".join("--" + k_.replace("_", "-") for k_ in r["kept"]) + "."
    if r["why"]:
        line = r["why"] + ". " + line
    return line


def plan_lines(m, chip):
    """`mashbox plan` / `mashbox inspect`: one header and one line per setting for this chip family, from the row's records
    only (speeds never computed); a setting with nothing recorded says "not measured on this chip"."""
    ch = (m.get("chips") or {}).get(chip) or {}
    me = ch.get("measured") or {}
    out = [f"{m.get('name') or m['id']} on TPU {ch.get('accelerator') or chip}" + ("" if ch.get("status") == "passing"
                                                                                    else f" ({ch.get('status') or 'not in its row'})")]
    a1, a8 = me.get("decode_tok_s_1_user"), me.get("decode_tok_s_8_users")
    out.append("  auto   " + (f"{a1} tok/s alone" + (f", {a8} total at 8 users" if a8 else "") if a1 else "not measured on this chip"))
    for s in ("fast", "long"):
        r = (me.get("decode") or {}).get(s) or {}
        if s == "fast" and not block_drafter(m):
            out.append("  fast   no block drafter for this model: runs as auto")
            continue
        bits = []
        if r.get("ctx"):
            bits.append(f"{r.get('slots', 1)} user x {int(r['ctx']):,} tokens")
        if r.get("kv"):
            bits.append(f"{r['kv']} K/V")
        bits.append(f"{r['tok_s_1_user']} tok/s{on_weights(r)}" if r.get("tok_s_1_user") else "speed not measured on this chip")
        if r.get("passkey"):
            bits.append(f"passkey {r['passkey']}")
        out.append(f"  {s:<6} " + ", ".join(bits))
    return out
