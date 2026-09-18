#!/usr/bin/env bash
# Run the desub pipeline inside the ML container with the validated setup.
# Everything that used to live in chat context is encoded here:
#   - image desub:cu124 (torch/easyocr/opencv/einops/timm stack)
#   - mounts: sandbox lab at /work, this repo at /src
#   - Clash proxy on the host at 127.0.0.1:7897 (model downloads)
#   - PROPAINTER_HOME at <lab>/vendor/ProPainter (weights under weights/)
#
# Usage:
#   scripts/run_container.sh <command...>          # default: bash
#   scripts/run_container.sh remove data/raw/ep4.mp4 -o data/out/ep4.mp4 --propainter --grain
# Env overrides: DESUB_IMAGE, DESUB_LAB, DESUB_REPO, DESUB_PROXY, PROPAINTER_HOME
set -euo pipefail

IMAGE="${DESUB_IMAGE:-desub:cu124}"
LAB="${DESUB_LAB:-W:/QoderCN/desub-lab}"
REPO="${DESUB_REPO:-W:/github.com/aura-bootstrap/fengshen-subtitle-remover}"
PROXY="${DESUB_PROXY:-http://host.docker.internal:7897}"
PP_HOME="${PROPAINTER_HOME:-/work/vendor/ProPainter}"

if [ $# -eq 0 ]; then
  set -- bash
fi
case "${1:-}" in
  remove|rerun)
    sub="$1"; shift
    case " $* " in
      *" --propainter-script "*) set -- /src/bin/desub-linux-amd64 "$sub" "$@" ;;
      *) set -- /src/bin/desub-linux-amd64 "$sub" "$@" --propainter-script /src/scripts/propainter_infer.py ;;
    esac
    ;;
  detect|probe) set -- /src/bin/desub-linux-amd64 "$@" ;;
esac

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${LAB}:/work" \
  -v "${REPO}:/src" \
  -w /work \
  -e HTTP_PROXY="${PROXY}" \
  -e HTTPS_PROXY="${PROXY}" \
  -e PROPAINTER_HOME="${PP_HOME}" \
  "${IMAGE}" "$@"
