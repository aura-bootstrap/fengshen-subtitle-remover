#!/usr/bin/env bash
# Wan-VACE bring-up self-check: verifies GPU, checkout, weights and entry
# script before attempting the first sidecar run. Usage:
#   scripts/wanvace_check.sh [WANVACE_HOME]
# Exits 0 when every check passes; prints [OK]/[FAIL] per item.
set -u

HOME_DIR="${1:-${WANVACE_HOME:-/work/vendor/Wan2.1}}"
CKPT="${WANVACE_CKPT:-${HOME_DIR}/Wan2.1-VACE-1.3B}"
TASK="${WANVACE_TASK:-vace-1.3B}"
rc=0

ok()   { echo "[OK]   $1"; }
bad()  { echo "[FAIL] $1"; rc=1; }

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  ok "GPU: $(nvidia-smi -L | head -1)"
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "${mem:-0}" -ge 11000 ]; then ok "VRAM ${mem}MB >= 12GB (1.3B 档)"; else bad "VRAM ${mem}MB 不足 12GB，需 vace-1.3B + WANVACE_OFFLOAD=1 或换卡"; fi
else
  bad "nvidia-smi 不可用：无 N 卡或容器未加 --gpus"
fi

[ -d "$HOME_DIR" ] && ok "WANVACE_HOME=$HOME_DIR" || bad "WANVACE_HOME 不存在: $HOME_DIR"
[ -f "$HOME_DIR/generate.py" ] && ok "generate.py 存在" || bad "缺 $HOME_DIR/generate.py"
[ -d "$CKPT" ] && ok "checkpoint: $CKPT" || bad "缺 checkpoint 目录 $CKPT（huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B）"

if [ -f "$HOME_DIR/generate.py" ]; then
  for arg in --task --size --frame_num --ckpt_dir --src_video --src_mask --prompt --save_file; do
    if grep -q -- "$arg" "$HOME_DIR/generate.py"; then ok "generate.py 支持 $arg"; else bad "generate.py 缺少 $arg（上游可能改名，需同步改 wanvace_infer.py）"; fi
  done
  if grep -q "$TASK" "$HOME_DIR/generate.py"; then ok "generate.py 认识 task $TASK"; else bad "generate.py 未见 task $TASK"; fi
fi

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && ok "torch.cuda 可用" || bad "torch.cuda 不可用（驱动/镜像 CUDA 不匹配）"

if [ "$rc" -eq 0 ]; then echo "ALL GREEN — 可以按 doc/wanvace-bringup.md 跑首个片段"; else echo "存在 FAIL 项，按上面清单补齐"; fi
exit "$rc"
