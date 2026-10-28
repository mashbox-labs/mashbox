#!/bin/bash
# Tier T2 on one TPU VM (CI.md). Run from the root of the commit under test:
#   bash tools/ci_tpu_job.sh <bundles dir> <out dir> [chip, default v6e]
# <bundles dir> holds the kernel bundles named as in models.json (the runner fetches and sha256-checks them; the VM
# gets no tokens). Steps, each one a gate:
#   setup    Python 3.12 env with the tpu and dev extras (CPU torch for the reference side)
#   versions jax, jaxlib and libtpu equal the kernel pins in models.json
#   kernels  every tests/tpu/<model>_kernel_check.py vs tests/tpu/<model>_kernel_tolerances.json
#   tiny     the CPU test suite again on the chip, per area: JAX_PLATFORMS=tpu,cpu, f32 matmuls at full precision
# Writes <out dir>/summary.json and the logs; exits 0 only when every step passed.
set -uo pipefail
B=$(realpath "$1"); O=$(realpath -m "$2"); CHIP=${3:-v6e}
mkdir -p "$O"; : > "$O/steps.jsonl"
PY=$PWD/.venv/bin/python
step() {  # step <name> <command...>: run, log, record pass/fail and seconds
  local name=$1; shift; local t=$SECONDS
  echo "=== $name: $*"
  "$@" > >(tee "$O/$name.log") 2>&1; local rc=$?
  printf '{"step": "%s", "ok": %s, "seconds": %d}\n' "$name" "$([ $rc = 0 ] && echo true || echo false)" $((SECONDS - t)) >> "$O/steps.jsonl"
  echo "=== $name: rc=$rc in $((SECONDS - t)) s"; return $rc
}
setup() {
  python3 -m pip install -q --user uv==0.13.6 && UV="python3 -m uv" &&
  $UV venv -q --python 3.12 .venv &&
  $UV pip install -q --python "$PY" -e ".[tpu,dev]" imageio-ffmpeg \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match &&
  "$PY" -c "import jax; print(jax.__version__, jax.devices())"
}
kernels() {
  local rc=0 n=0
  for chk in tests/tpu/*_kernel_check.py; do
    [ -e "$chk" ] || continue
    local m; m=$(basename "$chk" _kernel_check.py); n=$((n + 1))
    local tol=tests/tpu/${m}_kernel_tolerances.json
    local whl; whl=$("$PY" -c "import json,sys;r=[x for x in json.load(open('models.json'))['models'] if x['id'].replace('-','_')==sys.argv[1]];print(next(k['file'].split('/')[-1] for k in r[0]['kernels'] if sys.argv[2] in k['chips']))" "$m" "$CHIP")
    "$PY" "$chk" --bundle "$B/$whl" --out "$O/kernels-$m.json" || { echo "FAIL $chk exited non-zero"; rc=1; continue; }
    [ -f "$tol" ] || { echo "FAIL no tolerance file $tol"; rc=1; continue; }
    "$PY" tools/ci_tpu_gate.py kernels "$O/kernels-$m.json" "$tol" || rc=1
  done
  [ $n -gt 0 ] || { echo "FAIL no kernel checks in tests/tpu"; rc=1; }
  return $rc
}
tiny() {  # one pytest process per area, as in T1: a process that opened the TPU holds it, so a check that starts its
          # own process (the Qwen checks) needs a parent that never touched it. TPU is the default device; CPU stays
          # available for tests that compare against it explicitly.
  local rc=0 a
  for a in $("$PY" -c "import sys; sys.path.insert(0, 'tools'); import ci_select as c; print(' '.join(c.select(['pyproject.toml'], c.model_areas())))"); do
    JAX_PLATFORMS=tpu,cpu JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" -m pytest tests/cpu -p no:cacheprovider \
      -m "$a and not t0 and not quarantine" --durations=10 -rs --junitxml="$O/tiny-$a.xml" || rc=1
  done
  return $rc
}
ok=1
step setup setup || { ok=0; echo "setup failed, nothing else can run"; }
if [ $ok = 1 ]; then
  step versions "$PY" tools/ci_tpu_gate.py versions --chip "$CHIP" || ok=0
  step kernels kernels || ok=0
  step tiny tiny || ok=0
fi
python3 - "$O" "$ok" <<'PY'
import json, sys
o, ok = sys.argv[1], sys.argv[2] == "1"
steps = [json.loads(l) for l in open(f"{o}/steps.jsonl")]
json.dump(dict(ok=ok, steps=steps), open(f"{o}/summary.json", "w"), indent=1)
print(json.dumps(dict(ok=ok, steps=steps)))
PY
[ $ok = 1 ]
