#!/bin/bash
# tools/standards_cpu.sh [pytest args...]: the CPU standards tier, CPU only: every test with the marker `standards` and
# not `standards_slow`, in tests/standards and in every other tests/ file that sets the marker (pytest.mark.standards in
# its text: the same rule tests/conftest.py uses for a skip at collection). Only those files are collected, so a module
# skip elsewhere in tests/ (a test that needs an env var the tier does not set) is not counted here.
#
# One pytest process per tiny family (tests/standards/tiny.py families()) in parallel, plus one process for the tests
# that take no family (STD_FAMILY=-). No pytest-xdist. Each process builds its own tiny files in its own folder.
# Load guard: the 1-minute load average is read at the start and at the end; if either is over the core count, the run
# reads VOID (exit 3), never PASS or FAIL: timings and timeouts on an overloaded box say nothing.
# A skip inside the marker is a FAIL (tests/conftest.py, at collection and in a test), and a process whose
# counts show a skip fails here too. Each log keeps the printed lines of passing tests (-rP: ST-B2's random-weights
# record) and every test's setup / call time (--durations=0). Every call of 10 s or more (CI.md: a test over 10 s is
# marked slow with its reason) is listed as an over_10s line and counted in the verdict line: recorded, not gated
# (a time depends on the machine). Exit 0 = PASS, 1 = FAIL, 3 = VOID.
# Env: PYTHON (default python), STD_OUT (logs folder; default a new temp folder), STD_FAMILIES (space-separated subset).
set -u
cd "$(dirname "$0")/.." || exit 2
PY=${PYTHON:-python}
OUT=${STD_OUT:-$(mktemp -d -t standards-cpu.XXXXXX)}
mkdir -p "$OUT"
CORES=$(nproc)
L0=$(cut -d' ' -f1 /proc/loadavg)
export JAX_PLATFORMS=cpu PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
FAMS=${STD_FAMILIES:-$("$PY" -c 'import sys; sys.path[:0] = ["tests/standards"]; import tiny; print(" ".join(tiny.families()))')} || exit 2
# each process gets its own slice of the cores (taskset; XLA sizes its thread pools to the slice; the slices cover
# every core), so the tier itself never loads the box past its core count, and its own cache folder (MASHBOX_CACHE),
# as on a fresh machine
NP=$(( $(echo $FAMS | wc -w) + 1 ))
FILES="tests/standards $(grep -rlE 'mark\.standards' tests --include='test_*.py' | grep -v '^tests/standards/' | sort | tr '\n' ' ')"
echo "standards files: $FILES"
T0=$(date +%s)
pids=(); i=0
for f in $FAMS -; do
  n=$f; [ "$f" = "-" ] && n=common
  lo=$(( i * CORES / NP )); hi=$(( (i + 1) * CORES / NP - 1 )); [ "$hi" -ge "$lo" ] || hi=$lo; i=$((i + 1))
  ( STD_FAMILY=$f STD_TINY_DIR="$OUT/tiny-$n" MASHBOX_CACHE="$OUT/cache-$n" taskset -c "$lo-$hi" "$PY" -m pytest -p no:cacheprovider -q --tb=short -rfEsxXP --durations=0 -m "standards and not standards_slow" \
      $FILES "$@" > "$OUT/$n.log" 2>&1; echo "rc=$?" >> "$OUT/$n.log" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
T1=$(date +%s)
L1=$(cut -d' ' -f1 /proc/loadavg)
fail=0; tot_p=0; tot_f=0; tot_x=0; tot_s=0; tot_r=0
for f in $FAMS -; do
  n=$f; [ "$f" = "-" ] && n=common
  last=$(grep -E "(passed|failed|error|no tests ran|deselected)" "$OUT/$n.log" | tail -1)
  rc=$(grep -E "^rc=" "$OUT/$n.log" | tail -1 | cut -d= -f2)
  num() { echo "$last" | grep -oE "[0-9]+ $1" | grep -oE "^[0-9]+" || echo 0; }
  p=$(num passed); fl=$(num failed); e=$(num error); x=$(num xfailed); s=$(num skipped)
  tot_p=$((tot_p + p)); tot_f=$((tot_f + fl + e)); tot_x=$((tot_x + x)); tot_s=$((tot_s + s))
  th=$(grep -E "^tokenizer_sha256 $f " "$OUT/$n.log" | tail -1 | cut -d' ' -f3 | cut -c1-16)
  rec=$(grep -c '"test": "ST-B2", "family": "[^"]*", "kind": "random"' "$OUT/$n.log"); tot_r=$((tot_r + rec))
  echo "family=$n rc=$rc passed=$p failed=$((fl + e)) xfailed=$x skipped=$s tokenizer_sha256=${th:--}"
  [ "$rc" = 0 ] && [ "$s" = 0 ] || fail=1
done
over=$(grep -hE "^[0-9]+\.[0-9]+s call " "$OUT"/*.log | awk '$1 + 0 >= 10' | sort -rn)
[ -n "$over" ] && echo "$over" | sed 's/^/over_10s /'
n_over=$(printf '%s' "$over" | grep -c . )
verdict=PASS
[ $fail = 1 ] && verdict=FAIL
if awk -v a="$L0" -v b="$L1" -v c="$CORES" 'BEGIN{exit !(a > c || b > c)}'; then verdict=VOID; fi
echo "STANDARDS_CPU verdict=$verdict passed=$tot_p (recorded_not_gated=$tot_r) failed=$tot_f xfailed=$tot_x skipped=$tot_s over_10s=$n_over wall_s=$((T1 - T0)) load_start=$L0 load_end=$L1 cores=$CORES logs=$OUT"
case $verdict in PASS) exit 0 ;; FAIL) exit 1 ;; *) exit 3 ;; esac
