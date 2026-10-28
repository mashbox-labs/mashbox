#!/usr/bin/env python3
"""Pass/fail gates for the chip tier (T2, CI.md). The check scripts in tests/tpu measure; this decides.

    python3 tools/ci_tpu_gate.py versions --chip v6e
        jax, jaxlib and libtpu on this machine against the kernel pins of every models.json row for that chip
    python3 tools/ci_tpu_gate.py kernels REPORT.json TOLERANCES.json
        a tests/tpu/<model>_kernel_check.py report against its tolerance file

Kernel gates: every kernel in the bundle checked, no non-finite output, and per op either bit-exact or a relative L2
error vs plain JAX at or under the written tolerance. An op with no rule fails: a new kernel needs a written tolerance.
Prints one line per failure and a summary; exits 1 on any failure.
"""
import argparse
import fnmatch
import importlib.metadata as md
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def versions(chip):
    have = {}
    for pkg in ("jax", "jaxlib", "libtpu"):
        try:
            have[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            have[pkg] = None
    bad = []
    for m in json.loads((ROOT / "models.json").read_text())["models"]:
        for k in m.get("kernels", []):
            if chip not in k.get("chips", []):
                continue
            for pkg in ("jax", "jaxlib", "libtpu"):
                want = k.get(pkg, "")
                if not want[:1].isdigit():
                    print(f"note: {m['id']} {pkg} pin is {want!r}, recorded only (have {have[pkg]})")
                elif have[pkg] is None or not fnmatch.fnmatch(have[pkg], want):
                    bad.append(f"{m['id']}: {pkg} {have[pkg]} does not match the kernel pin {want}")
    for b in bad:
        print("FAIL", b)
    print(json.dumps(dict(gate="versions", chip=chip, have=have, failures=len(bad))))
    return 1 if bad else 0


def rule_for(name, rules):
    keys = [k for k in rules if name == k or name.startswith(k + " ")]
    return rules[max(keys, key=len)] if keys else None


def kernels(report, tolerances):
    rep, tol = json.loads(pathlib.Path(report).read_text()), json.loads(pathlib.Path(tolerances).read_text())
    bad = [f"kernel not checked: {k}" for k in rep.get("kernels_not_checked", ["<report has no kernels_not_checked>"])]
    if not rep.get("results"):
        bad.append("no kernel results")
    worst = {}
    for r in rep.get("results", []):
        rule = rule_for(r["name"], tol["rules"])
        if r["nonfinite"]:
            bad.append(f"{r['name']}: {r['nonfinite']} non-finite values")
        if rule is None:
            bad.append(f"{r['name']}: no tolerance rule in {tolerances}")
        elif rule.get("bit_exact") and not r["bit_exact"]:
            bad.append(f"{r['name']}: not bit-exact (rel_l2 {r['rel_l2']:.3g}, max_abs {r['max_abs']:.3g})")
        elif "rel_l2" in rule and not r["rel_l2"] <= rule["rel_l2"]:
            bad.append(f"{r['name']}: rel_l2 {r['rel_l2']:.3g} > {rule['rel_l2']:.3g}")
        worst[r["name"].split(" ")[0]] = max(worst.get(r["name"].split(" ")[0], 0.0), r["rel_l2"])
    for b in bad:
        print("FAIL", b)
    print(json.dumps(dict(gate="kernels", bundle=rep.get("bundle"), chip=rep.get("chip"), kernels=rep.get("kernels"),
                          checked=rep.get("kernels_checked"), compared=len(rep.get("results", [])),
                          bit_exact=sum(bool(r["bit_exact"]) for r in rep.get("results", [])),
                          worst_rel_l2={k: float(f"{v:.3g}") for k, v in worst.items()}, failures=len(bad))))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("versions")
    v.add_argument("--chip", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("report")
    k.add_argument("tolerances")
    a = ap.parse_args(argv)
    return versions(a.chip) if a.cmd == "versions" else kernels(a.report, a.tolerances)


if __name__ == "__main__":
    sys.exit(main())
