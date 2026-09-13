#!/usr/bin/env bash
# Otacon Genome Project — one-click GPU voice-trainer install/build
#
# Builds:
#   1) piper-voice-trainer:latest   (base image; CPU torch — build substrate only)
#   2) piper-voice-trainer:gpu      (CUDA 11.8 torch — REQUIRED for Genome train)
#
# Why two tags: the root Dockerfile still installs CPU torch (historical). The
# gpu-build/ layer swaps in cu118. Otacon Executor ONLY launches :gpu with
# `docker run --gpus all` and pipeline.py refuses to train if CUDA is missing.
# NEVER point Genome at :latest for training — that was the CPU hole.
#
# Usage:
#   sudo bash /opt/otacon/voice-trainer/build.sh
#   sudo bash /opt/otacon/voice-trainer/build.sh --gpu-only   # if :latest already exists
#   sudo bash /opt/otacon/voice-trainer/build.sh --verify-only
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GPU_ONLY=0
VERIFY_ONLY=0
SKIP_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --gpu-only) GPU_ONLY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

echo "============================================"
echo "  Otacon Genome — Voice Trainer (GPU)"
echo "============================================"
need docker

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable (try sudo / start docker)." >&2
  exit 1
fi

# nvidia-container-toolkit / --gpus support
if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 \
   && ! docker run --rm --gpus all --entrypoint nvidia-smi piper-voice-trainer:gpu-bench >/dev/null 2>&1 \
   && ! docker run --rm --gpus all --entrypoint nvidia-smi piper-voice-trainer:gpu >/dev/null 2>&1; then
  # Soft warning during verify-only of missing image; hard fail on build path.
  if [[ "$VERIFY_ONLY" -eq 0 ]]; then
    echo "WARN: could not probe nvidia-smi via --gpus all with a stock image."
    echo "      Continuing — final CUDA verify against :gpu will be authoritative."
  fi
fi

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

verify_gpu_image() {
  local img="${1:-piper-voice-trainer:gpu}"
  echo ""
  echo "── Verify $img ──"
  if ! image_exists "$img"; then
    echo "FAIL: image $img not found"
    return 1
  fi
  # Use -c (not stdin heredoc) — docker entrypoint often swallows heredoc stdin.
  docker run --rm --gpus all --entrypoint python3 "$img" -c \
"import sys, torch, numpy
print('torch', torch.__version__)
print('cuda_build', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('numpy', numpy.__version__)
assert torch.cuda.is_available(), 'CUDA not available inside container'
assert torch.version.cuda, 'CPU torch wheel installed — refusing'
maj_min = tuple(int(x) for x in numpy.__version__.split('.')[:2])
if maj_min[0] != 1 or maj_min[1] < 25:
    print('WARN: numpy', numpy.__version__, 'expected ~1.26.4 for ONNX export')
print('device', torch.cuda.get_device_name(0))
print('OK: GPU trainer ready')"
}

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify_gpu_image "piper-voice-trainer:gpu"
  exit $?
fi

# ── Base image (:latest) ────────────────────────────────────────────────
if [[ "$GPU_ONLY" -eq 0 ]]; then
  if image_exists "piper-voice-trainer:latest"; then
    echo "Base piper-voice-trainer:latest already present — skip (use full rebuild by deleting it)."
  elif image_exists "piper-voice-trainer:gpu-bench"; then
    # Bench image was a full CUDA stack build; usable as FROM substrate after
    # gpu-build re-pins torch/numpy. Tag it as :latest so gpu-build/Dockerfile works.
    echo "Tagging existing piper-voice-trainer:gpu-bench → :latest (build substrate)"
    docker tag piper-voice-trainer:gpu-bench piper-voice-trainer:latest
  else
    echo "Building base image piper-voice-trainer:latest (CPU torch substrate — slow)…"
    docker build -t piper-voice-trainer:latest -f Dockerfile .
  fi
else
  if ! image_exists "piper-voice-trainer:latest"; then
    if image_exists "piper-voice-trainer:gpu-bench"; then
      docker tag piper-voice-trainer:gpu-bench piper-voice-trainer:latest
    else
      echo "ERROR: --gpu-only needs piper-voice-trainer:latest (or :gpu-bench to retag)." >&2
      exit 1
    fi
  fi
fi

# ── GPU image (:gpu) ────────────────────────────────────────────────────
echo ""
echo "Building piper-voice-trainer:gpu from gpu-build/Dockerfile…"
docker build -t piper-voice-trainer:gpu -f gpu-build/Dockerfile gpu-build

if [[ "$SKIP_VERIFY" -eq 0 ]]; then
  verify_gpu_image "piper-voice-trainer:gpu"
fi

echo ""
echo "Build complete."
echo ""
echo "Genome expects:  piper-voice-trainer:gpu"
echo "Check status:    curl -s http://127.0.0.1:5757/genome/trainer/status"
echo "Discord/CLI:     #voice-train <name> <youtube_url>"
echo "Portal:          Genome → Upload & Auto-Train"
echo ""
echo "Do NOT train with :latest — that image is CPU torch only."
