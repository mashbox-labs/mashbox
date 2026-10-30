"""The routed-expert report: its names and its shape, in one place.

The engine reports where the routed experts of a mixture-of-experts model ran: ops.expert_report(), which /health
(routed_experts) and inspect carry. Its producers take every name from this file: ops.expert_report() the report's
keys, ops/bundle.py the routed-expert kernel's id, runtime/openapi.py the /health schema. problems() checks one
report's shape, so a producer and the checks that read its output cannot drift apart.

internal/moe_record.py (the chip-record validator, not in the wheel) takes the kernel's id (KERNEL_ID) from here. What a chip record
must show to prove an MoE row stays there; this file holds only what the engine itself exposes.
"""

# the routed-expert kernel's id (the kernels package that carries it is named in one place: ops.KERNEL_MODULE)
KERNEL_ID = "moe.expert-ffn"
# the report's keys, in ops.expert_report()'s order
PATH, KERNEL_KEY, SITES, WHY = "path", "kernel", "sites", "exact_why"
REPORT = (PATH, KERNEL_KEY, SITES, WHY)
PATHS = ("kernel", "exact", "mixed")   # PATH: every call site on the kernel, every one on the exact path, or both
SITE_PATHS = ("kernel", "exact")       # SITES: the routed-op call sites traced on each path
WHY_MAX = 8                            # WHY: the distinct reasons any site ran exact, at most this many


def _count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _text(v):
    return isinstance(v, str) and bool(v.strip())


def problems(rep):
    """Why one routed-expert report (ops.expert_report() once a routed op was traced; /health routed_experts when it is
    not null) does not have the report's shape. [] = it has. Any JSON value gives a list, never an exception."""
    if not isinstance(rep, dict):
        return [f"the report is a {type(rep).__name__}, not an object"]
    out = []
    if sorted(rep) != sorted(REPORT):
        out.append(f"the report's keys are {sorted(rep)}, not {sorted(REPORT)}")
    path, s, k, why = rep.get(PATH), rep.get(SITES), rep.get(KERNEL_KEY), rep.get(WHY)
    if path not in PATHS:
        out.append(f"{PATH} is {path!r}, not one of {', '.join(PATHS)}")
    if not (isinstance(s, dict) and sorted(s) == sorted(SITE_PATHS) and all(_count(s[p]) for p in SITE_PATHS)):
        out.append(f"{SITES} is {s!r}, not {{{', '.join(SITE_PATHS)}}} as counts (ints >= 0, never bools)")
        s = None
    else:
        on_kernel, exact = (s[p] for p in SITE_PATHS)
        want = None if on_kernel + exact == 0 else PATHS[0] if exact == 0 else PATHS[1] if on_kernel == 0 else PATHS[2]
        if want is None:
            out.append(f"{SITES} counts no call site: a report exists only once a routed op was traced")
        elif path in PATHS and path != want:
            out.append(f"{PATH} is {path!r} but {SITES} {s!r} say {want!r}")
    if not (k is None or (isinstance(k, dict) and all(_text(k.get(f)) for f in ("id", "version", "package")))):
        out.append(f"{KERNEL_KEY} is {k!r}, not null or an object with id, version and package as non-empty strings")
    if not (isinstance(why, list) and all(_text(w) for w in why) and len(set(why)) == len(why) and len(why) <= WHY_MAX):
        out.append(f"{WHY} is {why!r}, not a list of at most {WHY_MAX} distinct non-empty strings")
    elif why and s is not None and s[SITE_PATHS[1]] == 0:
        out.append(f"{WHY} holds {len(why)} reasons but no call site ran exact")
    return out
