#!/bin/sh
# Installs the `mashbox` command for your user with uv (https://docs.astral.sh/uv/). No sudo. Your system Python and
# your other tools are not changed: uv brings its own Python 3.12, and only uv and mashbox go into ~/.local/bin.
#   On a Cloud TPU VM it installs mashbox[tpu] (JAX for TPU); on any other machine the client (for `mashbox tpu`).
# Settings (environment variables, all optional):
#   MASHBOX_VERSION=0.1.0   that version instead of the newest     MASHBOX_EXTRAS=tpu   extras to install instead
#   MASHBOX_WHEEL=<file or URL>   a wheel file instead of PyPI      MASHBOX_INDEX_URL=<URL>   also search this index
set -eu
say() { printf 'mashbox install: %s\n' "$*"; }
die() { printf 'mashbox install: error: %s\n' "$*" >&2; exit 1; }
UV_VERSION=0.13.6

os=$(uname -s); arch=$(uname -m)
case "$os" in Linux|Darwin) ;; *) die "this installer supports Linux and macOS (found $os); on Windows, use WSL2" ;; esac
case "$arch" in x86_64|amd64|aarch64|arm64) ;; *) die "this installer supports x86_64 and arm64 CPUs (found $arch)" ;; esac
[ "$(id -u)" != 0 ] || [ "${MASHBOX_ALLOW_ROOT:-}" = 1 ] || die "run it as your own user, not as root or with sudo"
[ -d "${HOME:-/nonexistent}" ] || die "HOME is not set to a folder"
if command -v curl >/dev/null 2>&1; then fetch() { curl -fsSL --retry 3 "$1"; }
elif command -v wget >/dev/null 2>&1; then fetch() { wget -qO- "$1"; }
else die "needs curl or wget"; fi

# A Cloud TPU VM: Google hardware, and the metadata server names a TPU type.
tpu=""
if [ "$os" = Linux ] && grep -qs Google /sys/class/dmi/id/product_name && command -v curl >/dev/null 2>&1; then
  tpu=$(curl -fs -m 3 -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || true)
fi
if [ "${MASHBOX_EXTRAS+set}" = set ]; then extras=$MASHBOX_EXTRAS
elif [ -n "$tpu" ]; then extras=tpu
else extras=""; fi
if [ -n "$tpu" ]; then say "machine: Cloud TPU VM ($tpu), extras: ${extras:-none}"
else say "machine: $os $arch (client for mashbox tpu), extras: ${extras:-none}"; fi

if command -v uv >/dev/null 2>&1; then UV=$(command -v uv); say "uv: using yours, $UV ($("$UV" --version))"
elif [ -x "$HOME/.local/bin/uv" ]; then UV=$HOME/.local/bin/uv; say "uv: using $UV ($("$UV" --version))"
else
  say "uv: not found; installing uv $UV_VERSION into ~/.local/bin (shell profiles left as they are)"
  tmp=$(mktemp); fetch "https://astral.sh/uv/$UV_VERSION/install.sh" > "$tmp" || die "could not download the uv installer"
  UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$tmp" >/dev/null 2>&1 || die "the uv installer failed"
  rm -f "$tmp"; UV=$HOME/.local/bin/uv; [ -x "$UV" ] || die "uv is not in ~/.local/bin after its installer ran"
  say "uv: installed, $UV ($("$UV" --version))"
fi

ex=""; [ -z "$extras" ] || ex="[$extras]"
if [ -n "${MASHBOX_WHEEL:-}" ]; then
  case "$MASHBOX_WHEEL" in http://*|https://*|file://*) url=$MASHBOX_WHEEL ;; /*) url="file://$MASHBOX_WHEEL" ;;
    *) url="file://$(pwd)/$MASHBOX_WHEEL" ;; esac
  spec="mashbox$ex @ $url"
elif [ -n "${MASHBOX_VERSION:-}" ]; then spec="mashbox$ex==$MASHBOX_VERSION"
else spec="mashbox$ex"; fi
set -- tool install --python 3.12
[ -z "${MASHBOX_INDEX_URL:-}" ] || set -- "$@" --extra-index-url "$MASHBOX_INDEX_URL"
case ",$extras," in *,hy-nomas,*)     # torch without the CUDA packages a TPU VM has no use for
  [ "$os" != Linux ] || set -- "$@" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match ;; esac
[ -z "${MASHBOX_WHEEL:-}" ] || set -- "$@" --reinstall-package mashbox
say "installing: $spec (uv tool, Python 3.12)"
"$UV" -q "$@" "$spec" || die "uv could not install mashbox (its message is above). If it says an executable already exists, another mashbox is in ~/.local/bin (for example from pip): remove that one yourself, then run this again"

bin=$("$UV" tool dir --bin)
"$bin/mashbox" --version >/dev/null 2>&1 || die "installed, but $bin/mashbox does not start"
say "done: $bin/mashbox ($("$bin/mashbox" --version))"
case ":$PATH:" in
  *":$bin:"*) mb=mashbox ;;
  *) mb="$bin/mashbox"
     say "$bin is not on your PATH. This terminal: export PATH=\"$bin:\$PATH\"  New terminals: add that line to ~/.zshrc or ~/.bashrc, or run: $UV tool update-shell" ;;
esac
if [ -n "$tpu" ]; then say "next: $mb run <model>, for example: $mb run qwen3-8-27b --prompt Hello"; else say "next: $mb tpu up qwen3-8-27b"; fi
