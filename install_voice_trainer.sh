#!/usr/bin/env bash
# ==============================================================================
#  OTACONSKEEP // OTACON VOICE TRAINER (Genome)
#  One-click GPU Piper voice-cloning installer
#
#  Designed & Engineered by Antonio G. Garcia
#  "Built for the Keep."
#  https://github.com/Otaconskeep/otacon-voice-trainer
#  Discord: https://discord.gg/cZDeqECzX
# ==============================================================================
#
# Target: Ubuntu / Debian Linux (native or WSL2)
#
# One shot:
#   curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash
#
# What it does (idempotent — safe to re-run):
#   - Detects OS / WSL and guides Windows users into Ubuntu WSL2
#   - Installs Python 3, pip, git, curl, ca-certificates if missing
#   - Installs Docker Engine + NVIDIA Container Toolkit when a GPU is present
#   - Looks up NVIDIA GPU / VRAM and refuses silent CPU training
#   - Clones/updates this repo
#   - Builds piper-voice-trainer:latest (substrate) + :gpu (CUDA train image)
#   - Verifies torch.cuda inside the image
#   - Opens a simple local UI on http://127.0.0.1:8765
#
# Env overrides:
#   OTACON_VT_DIR=~/otacon-voice-trainer
#   OTACON_VT_SKIP_UI=0
#   OTACON_VT_SKIP_BUILD=0
#   OTACON_VT_UI_PORT=8765
# ==============================================================================
set -Eeuo pipefail

BRAND="ANTONIO G. GARCIA // OTACONSKEEP"
PRODUCT="OTACON VOICE TRAINER // GENOME"
TAGLINE="Built for the Keep."
DISCORD_URL="https://discord.gg/cZDeqECzX"
REPO_URL="https://github.com/Otaconskeep/otacon-voice-trainer.git"
RAW_INSTALL_URL="https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh"

INSTALL_DIR="${OTACON_VT_DIR:-$HOME/otacon-voice-trainer}"
SKIP_UI="${OTACON_VT_SKIP_UI:-0}"
SKIP_BUILD="${OTACON_VT_SKIP_BUILD:-0}"
UI_PORT="${OTACON_VT_UI_PORT:-8765}"

log()  { printf '\n\033[1;36m[AGG::VOICE]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[AGG::OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[AGG::WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[AGG::FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

trap 'printf "\n\033[1;31m[AGG::FAIL]\033[0m Installer stopped on line %s.\n" "$LINENO" >&2' ERR

printf '\033[1;35m'
cat <<'VT_ASCII'
  ___ _____  _    ____ ___  _   _
 / _ \_   _|/ \  / ___/ _ \| \ | |
| | | || | / _ \| |  | | | |  \| |
| |_| || |/ ___ \ |__| |_| | |\  |
 \___/ |_/_/   \_\____\___/|_| \_|

 __     _____ ___ ____ _____
 \ \   / / _ \_ _/ ___| ____|
  \ \ / / | | | | |   |  _|
   \ V /| |_| | | |___| |___
    \_/  \___/___\____|_____|

 _____ ____      _    ___ _   _ _____ ____
|_   _|  _ \    / \  |_ _| \ | | ____|  _ \
  | | | |_) |  / _ \  | ||  \| |  _| | |_) |
  | | |  _ <  / ___ \ | || |\  | |___|  _ <
  |_| |_| \_\/_/   \_\___|_| \_|_____|_| \_\

==============================================================================
                     O T A C O N S K E E P
              G E N O M E   V O I C E   T R A I N E R
                     O N E - C L I C K   G P U
==============================================================================
                      ANTONIO G. GARCIA
                      Built for the Keep.
==============================================================================
VT_ASCII
printf '\033[0m\n'
printf '\033[1;36m%s\033[0m\n' "$BRAND"
printf '\033[0;37m%s :: %s\033[0m\n' "$PRODUCT" "$TAGLINE"
printf '\033[0;37mGPU-only Piper fine-tunes. CPU training is forbidden (that hole is closed).\033[0m\n'
printf '\033[0;37mHelp: %s\033[0m\n\n' "$DISCORD_URL"

command_exists() { command -v "$1" >/dev/null 2>&1; }

# Docker helper that works immediately after usermod -aG docker (no re-login).
docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif command_exists sg && sg docker -c "docker info" >/dev/null 2>&1; then
    sg docker -c "docker $(printf '%q ' "$@")"
  elif [[ -n "${SUDO:-}" ]]; then
    $SUDO docker "$@"
  else
    docker "$@"
  fi
}

# ---------- Windows shells → send them to WSL ----------
case "$(uname -s 2>/dev/null || true)" in
  MINGW*|MSYS*|CYGWIN*)
    die "You're in Git Bash / MSYS, not Linux. Double-click install_voice_trainer.bat (sets up WSL2 + Ubuntu), or open the Ubuntu app and re-run: curl -fsSL $RAW_INSTALL_URL | bash"
    ;;
esac

is_debian_family() {
  [[ -f /etc/os-release ]] || return 1
  # shellcheck disable=SC1091
  . /etc/os-release
  [[ "${ID:-}" == "ubuntu" || "${ID:-}" == "debian" || "${ID_LIKE:-}" == *"debian"* ]]
}

is_debian_family || die \
  "Need Ubuntu/Debian (or WSL2 Ubuntu). On Windows: open PowerShell as Admin → wsl --install → reboot → open Ubuntu → re-run this curl|bash. Discord: $DISCORD_URL"

IN_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then
  IN_WSL=1
  ok "WSL2 Linux environment detected"
fi

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if command_exists sudo; then SUDO="sudo"; else die "Need sudo for apt/docker packages."; fi
else
  warn "Running as root — fine for servers; prefer a normal user + sudo on desktops."
fi

# ---------- WSL systemd (needed for docker) ----------
if [[ "$IN_WSL" -eq 1 ]]; then
  WSL_CONF="/etc/wsl.conf"
  if ! grep -qE '^\s*systemd\s*=\s*true' "$WSL_CONF" 2>/dev/null; then
    log "Enabling systemd in WSL (required for Docker)"
    if [[ -f "$WSL_CONF" ]] && grep -q '^\[boot\]' "$WSL_CONF"; then
      $SUDO sed -i '/^\[boot\]/a systemd=true' "$WSL_CONF"
    else
      printf '\n[boot]\nsystemd=true\n' | $SUDO tee -a "$WSL_CONF" >/dev/null
    fi
    warn "systemd just enabled. From Windows PowerShell run:  wsl --shutdown"
    warn "Then open Ubuntu again and re-run:  curl -fsSL $RAW_INSTALL_URL | bash"
    exit 75
  fi
fi

# ---------- apt packages: python, git, curl ----------
log "Installing base packages (Python, git, curl) if missing"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y \
  python3 python3-pip python3-venv \
  git curl ca-certificates gnupg lsb-release \
  pciutils

ok "Python $(python3 --version 2>&1 | awk '{print $2}')"

# ---------- Docker ----------
install_docker() {
  log "Installing Docker Engine"
  if command_exists docker && (docker info >/dev/null 2>&1 || $SUDO docker info >/dev/null 2>&1 || (command_exists sg && sg docker -c "docker info" >/dev/null 2>&1)); then
    ok "Docker already running"
    return 0
  fi
  curl -fsSL https://get.docker.com | $SUDO sh
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    $SUDO usermod -aG docker "$USER" || true
    warn "Added $USER to docker group. This installer will use 'sg docker' / sudo for the rest of this run."
    warn "Open a new login shell later so plain 'docker' works without sg/sudo."
  fi
  $SUDO systemctl enable --now docker 2>/dev/null || $SUDO service docker start || true
  # Prove docker works in THIS session without logout.
  if docker_cmd info >/dev/null 2>&1; then
    ok "Docker installed and usable in this session"
  else
    die "Docker installed but not usable yet. Try: sudo docker info   or: sg docker -c 'docker info'"
  fi
}

install_docker

# If the user was just added to the docker group, re-enter via `sg docker`
# so subsequent build.sh / docker run calls inherit membership without logout.
if ! docker info >/dev/null 2>&1; then
  if [[ "${OTACON_VT_DOCKER_REEXEC:-0}" != "1" ]] && command_exists sg && getent group docker 2>/dev/null | grep -qw "${USER}"; then
    warn "Re-entering installer under 'sg docker' so Docker works without logging out."
    export OTACON_VT_DOCKER_REEXEC=1
    exec sg docker -c "export OTACON_VT_DOCKER_REEXEC=1; exec bash \"$0\" $*"
  fi
fi

# ---------- GPU lookup ----------
log "GPU lookup"
GPU_NAME="none"
GPU_VRAM_MB=0
HAVE_NVIDIA=0
if command_exists nvidia-smi; then
  HAVE_NVIDIA=1
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')"
  GPU_VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  ok "NVIDIA GPU: ${GPU_NAME} (${GPU_VRAM_MB} MiB)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
else
  if lspci 2>/dev/null | grep -qi nvidia; then
    warn "NVIDIA PCI device present but nvidia-smi missing — install proprietary drivers, then re-run."
  fi
  die "No usable NVIDIA GPU (nvidia-smi). Voice training is GPU-only — CPU training was the hole we closed. Install NVIDIA drivers and re-run."
fi

# ---------- NVIDIA Container Toolkit ----------
install_nvidia_ctk() {
  log "Ensuring NVIDIA Container Toolkit (--gpus all)"
  if docker_cmd run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    ok "docker --gpus all already works"
    return 0
  fi
  distribution="$(. /etc/os-release; echo "${ID}${VERSION_ID}")"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y nvidia-container-toolkit
  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker 2>/dev/null || $SUDO service docker restart || true
  sleep 2
  docker_cmd run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi >/dev/null \
    || die "docker --gpus all still failing after toolkit install"
  ok "NVIDIA Container Toolkit ready"
}

install_nvidia_ctk

# ---------- Clone / update repo ----------
log "Syncing trainer sources → $INSTALL_DIR"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" fetch --depth 1 origin main 2>/dev/null || true
  git -C "$INSTALL_DIR" reset --hard origin/main 2>/dev/null \
    || git -C "$INSTALL_DIR" pull --ff-only || true
else
  # If this script is already running from a checkout, prefer it
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
  if [[ -f "$HERE/Dockerfile" && -f "$HERE/gpu-build/Dockerfile" && "$HERE" != "$INSTALL_DIR" ]]; then
    mkdir -p "$(dirname "$INSTALL_DIR")"
    rm -rf "$INSTALL_DIR"
    cp -a "$HERE" "$INSTALL_DIR"
  else
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
  fi
fi
ok "Sources at $INSTALL_DIR"

# ---------- Build images ----------
if [[ "$SKIP_BUILD" != "1" ]]; then
  log "Building Docker images (GPU trainer) — first run can take a while"
  bash "$INSTALL_DIR/build.sh"
else
  warn "Skipping image build (OTACON_VT_SKIP_BUILD=1)"
fi

# ---------- Simple UI ----------
UI_PID=""
start_ui() {
  [[ "$SKIP_UI" == "1" ]] && return 0
  local ui_dir="$INSTALL_DIR/ui"
  [[ -d "$ui_dir" ]] || return 0
  # Write runtime facts the UI can read
  cat > "$ui_dir/status.json" <<EOF
{
  "ok": true,
  "product": "Otacon Voice Trainer",
  "gpu_name": $(python3 -c "import json; print(json.dumps('''$GPU_NAME'''))"),
  "gpu_vram_mb": ${GPU_VRAM_MB:-0},
  "image": "piper-voice-trainer:gpu",
  "install_dir": $(python3 -c "import json; print(json.dumps('''$INSTALL_DIR'''))"),
  "wsl": $IN_WSL,
  "docs": "https://github.com/Otaconskeep/otacon-voice-trainer",
  "site": "https://otaconskeep-site.otaconskeep.workers.dev/otacon/#voice-trainer"
}
EOF
  if command_exists python3; then
    ( cd "$ui_dir" && python3 -m http.server "$UI_PORT" --bind 127.0.0.1 ) >/tmp/otacon-vt-ui.log 2>&1 &
    UI_PID=$!
    sleep 1
    ok "Simple UI → http://127.0.0.1:${UI_PORT}/  (pid $UI_PID)"
    if command_exists xdg-open; then xdg-open "http://127.0.0.1:${UI_PORT}/" >/dev/null 2>&1 || true; fi
  fi
}

start_ui

# ---------- Interactive menu ----------
# Final states: READY(0) / DEGRADED(2) / FAILED(1)
FINAL_STATE=READY
FINAL_RC=0
# Required: docker usable + GPU toolkit path + sources + (unless skipped) GPU image verify
if ! docker info >/dev/null 2>&1 && ! docker_cmd info >/dev/null 2>&1; then
  FINAL_STATE=FAILED
  FINAL_RC=1
fi
if [[ "$HAVE_NVIDIA" != "1" ]]; then
  FINAL_STATE=FAILED
  FINAL_RC=1
fi
if [[ ! -d "$INSTALL_DIR" ]]; then
  FINAL_STATE=FAILED
  FINAL_RC=1
fi

menu() {
  echo ""
  echo "=============================================================================="
  case "$FINAL_STATE" in
    READY) echo "  OTACON VOICE TRAINER — READY" ;;
    DEGRADED) echo "  OTACON VOICE TRAINER — DEGRADED" ;;
    *) echo "  OTACON VOICE TRAINER — FAILED" ;;
  esac
  echo "=============================================================================="
  echo "  GPU     : $GPU_NAME (${GPU_VRAM_MB} MiB)"
  echo "  Image   : piper-voice-trainer:gpu"
  echo "  Sources : $INSTALL_DIR"
  echo "  UI      : http://127.0.0.1:${UI_PORT}/"
  echo "  Status  : $FINAL_STATE (exit $FINAL_RC)"
  echo "=============================================================================="
  echo "  1) Re-verify CUDA inside trainer image"
  echo "  2) Rebuild GPU image"
  echo "  3) Open simple UI in browser"
  echo "  4) Print Otacon Core hook (install_otacon.sh)"
  echo "  5) Exit"
  echo "=============================================================================="
  if [[ ! -t 0 ]]; then
    ok "Non-interactive shell — install finished ($FINAL_STATE). Open http://127.0.0.1:${UI_PORT}/"
    return 0
  fi
  while true; do
    read -r -p "Select [1-5]: " choice || break
    case "$choice" in
      1) bash "$INSTALL_DIR/build.sh" --verify-only ;;
      2) bash "$INSTALL_DIR/build.sh" --gpu-only ;;
      3)
        if command_exists xdg-open; then xdg-open "http://127.0.0.1:${UI_PORT}/" || true
        else echo "Open: http://127.0.0.1:${UI_PORT}/"; fi
        ;;
      4)
        cat <<EOF

Add to Otacon Core install (one line):

  OTACON_INSTALL_VOICE_TRAINER=1 curl -fsSL \\
    https://raw.githubusercontent.com/Otaconskeep/otacons-ai-ecosystem/main/install_otacon.sh | bash

Or after Otacon Core is installed:

  curl -fsSL $RAW_INSTALL_URL | bash

EOF
        ;;
      5|q|Q) break ;;
      *) echo "Pick 1-5" ;;
    esac
  done
}

menu

echo ""
ok "Done ($FINAL_STATE). Train with Genome / #voice-train once Otacon Keep or your trainer launcher points at piper-voice-trainer:gpu"
printf '\033[0;37mWrite-up: https://github.com/Otaconskeep/otacon-voice-trainer#readme\033[0m\n'
printf '\033[0;37mSite:     https://otaconskeep-site.otaconskeep.workers.dev/otacon/#voice-trainer\033[0m\n'
printf '\033[0;37mNote: open a new shell later so plain docker works without sg/sudo if you were just added to the docker group.\033[0m\n'
exit "$FINAL_RC"
