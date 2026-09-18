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
# Env overrides: DESUB_IMAGE, DESUB_LAB, DESUB_REPO, DESUB_PROXY, DESUB_BIN, DESUB_GPUS, PROPAINTER_HOME
# NOTE: while a long-running container is executing /src/bin/desub-linux-amd64,
# do not overwrite that file on the host — new execs of the same path crash
# (virtiofs bind mount). Build to another name and point DESUB_BIN at it.
set -euo pipefail

IMAGE="${DESUB_IMAGE:-desub:cu124}"
LAB="${DESUB_LAB:-W:/QoderCN/desub-lab}"
REPO="${DESUB_REPO:-W:/github.com/aura-bootstrap/fengshen-subtitle-remover}"
PROXY="${DESUB_PROXY:-http://host.docker.internal:7897}"
PP_HOME="${PROPAINTER_HOME:-/work/vendor/ProPainter}"
BIN="${DESUB_BIN:-/src/bin/desub-linux-amd64}"
gpu_args=()
if [ -n "${DESUB_GPUS:-}" ]; then gpu_args+=(--gpus "$DESUB_GPUS"); fi

if [ $# -eq 0 ]; then
  set -- bash
fi
case "${1:-}" in
  remove|rerun)
    sub="$1"; shift
    extra=()
    case " $* " in
      *" --propainter-script "*) ;;
      *) extra+=(--propainter-script /src/scripts/propainter_infer.py) ;;
    esac
    case " $* " in
      *" --vlm-qc "*)
        case " $* " in
          *" --vlm-script "*) ;;
          *) extra+=(--vlm-script /src/scripts/vlm_qc.py) ;;
        esac
        ;;
    esac
    set -- "$BIN" "$sub" "$@" "${extra[@]}"
    ;;
  detect|probe) set -- "$BIN" "$@" ;;
esac

wanvace_env=()
for v in WANVACE_OFFLOAD WANVACE_T5_CPU WANVACE_SIZE WANVACE_FRAME_NUM WANVACE_STEPS WANVACE_TASK PYTORCH_CUDA_ALLOC_CONF; do
  if [ -n "${!v:-}" ]; then wanvace_env+=(-e "$v=${!v}"); fi
done

MSYS_NO_PATHCONV=1 docker run --rm \
  "${gpu_args[@]}" \
  -v "${LAB}:/work" \
  -v "${REPO}:/src" \
  -v "${LAB}/docker/paddlex:/root/.paddlex" \
  -w /work \
  -e HTTP_PROXY="${PROXY}" \
  -e HTTPS_PROXY="${PROXY}" \
  -e NO_PROXY="host.docker.internal,127.0.0.1,localhost" \
  -e PROPAINTER_HOME="${PP_HOME}" \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -e WANVACE_HOME="${WANVACE_HOME:-/work/vendor/Wan2.1}" \
  -e WANVACE_CKPT="${WANVACE_CKPT:-/work/vendor/Wan2.1/Wan2.1-VACE-1.3B}" \
  "${wanvace_env[@]}" \
  "${IMAGE}" "$@"
