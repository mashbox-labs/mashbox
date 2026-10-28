#!/usr/bin/env python3
"""TPU status badges from real chip runs (public-safe; ships in the main repo as tools/tpu_status.py).

GitHub-hosted runners have no TPU. Our on-demand chip runner calls `record` after `tests/tpu` on a chip;
the daily `tpu-status.yml` job calls `refresh`. A badge is a pure function of its run record, today's
date and the latest release tag, so it cannot stay green on an old, failed or superseded run.

  tpu_status.py record  --status-dir DIR --chip v6e --version 0.1.0 --commit SHA --jax V --jaxlib V --libtpu V
                        --passed N --failed N --skipped N --minutes M [--log PATH]
  tpu_status.py refresh --status-dir DIR --latest-tag v0.1.0 [--max-age-days 14] [--today YYYY-MM-DD]
  tpu_status.py check   --status-dir DIR --latest-tag v0.1.0 [--today YYYY-MM-DD]   (exit 1 if a badge differs)
"""
import argparse, datetime, json, sys
from pathlib import Path

def badge(rec, latest, today, max_age):
    chip = rec["chip"]
    if rec is None or "date" not in rec:
        return {"schemaVersion": 1, "label": f"TPU {chip}", "message": "no run yet", "color": "lightgrey"}
    age = (today - datetime.date.fromisoformat(rec["date"])).days
    if age > max_age or (latest and rec["version"] != latest.lstrip("v")):
        return {"schemaVersion": 1, "label": f"TPU {chip}", "message": f"stale, last {rec['date']}", "color": "lightgrey"}
    ok = rec["failed"] == 0 and rec["passed"] > 0
    return {"schemaVersion": 1, "label": f"TPU {chip}", "message": f"{'passing' if ok else 'failing'} {rec['date']}",
            "color": "brightgreen" if ok else "red"}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["record", "refresh", "check"])
    ap.add_argument("--status-dir", required=True)
    ap.add_argument("--chip", choices=["v6e", "v5e"])
    for k in ("version", "commit", "jax", "jaxlib", "libtpu", "log", "latest-tag", "today"):
        ap.add_argument("--" + k)
    for k in ("passed", "failed", "skipped"):
        ap.add_argument("--" + k, type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--max-age-days", type=int, default=14)
    a = ap.parse_args(argv)
    d = Path(a.status_dir); d.mkdir(parents=True, exist_ok=True)
    today = datetime.date.fromisoformat(a.today) if a.today else datetime.datetime.now(datetime.timezone.utc).date()
    if a.cmd == "record":
        rec = {"chip": a.chip, "version": a.version.lstrip("v"), "commit": a.commit, "jax": a.jax, "jaxlib": a.jaxlib,
               "libtpu": a.libtpu, "passed": a.passed, "failed": a.failed, "skipped": a.skipped, "minutes": a.minutes,
               "date": today.isoformat(), "log": a.log or ""}
        (d / f"tpu-{a.chip}.json").write_text(json.dumps(rec, indent=1) + "\n")
        (d / f"badge-tpu-{a.chip}.json").write_text(json.dumps(badge(rec, None, today, a.max_age_days)) + "\n")
        print("recorded", rec["chip"], "passing" if rec["failed"] == 0 and rec["passed"] > 0 else "failing")
        return 0
    bad = 0
    for chip in ("v6e", "v5e"):
        rp = d / f"tpu-{chip}.json"
        rec = json.loads(rp.read_text()) if rp.exists() else {"chip": chip}
        want = json.dumps(badge(rec, a.latest_tag, today, a.max_age_days)) + "\n"
        bp = d / f"badge-tpu-{chip}.json"
        if a.cmd == "check":
            if not bp.exists() or bp.read_text() != want:
                print(f"badge-tpu-{chip}.json disagrees with its record"); bad = 1
        else:
            bp.write_text(want); print(chip, json.loads(want)["message"])
    return bad

if __name__ == "__main__":
    sys.exit(main())
