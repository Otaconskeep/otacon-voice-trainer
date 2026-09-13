#!/usr/bin/env bash
# Hook fragment for Otaconskeep/otacons-ai-ecosystem install_otacon.sh
# When OTACON_INSTALL_VOICE_TRAINER=1, run the Genome GPU voice-trainer one-click.
#
# Intended use (near end of install_otacon.sh, after Core succeeds):
#   if [[ "${OTACON_INSTALL_VOICE_TRAINER:-0}" == "1" ]]; then
#     # shellcheck disable=SC1091
#     source /dev/stdin <<<"$(curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/hooks/otacon_install_hook.sh)" \
#       || curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash
#   fi

set -euo pipefail
echo "[AGG::OTACON] Installing Otacon Voice Trainer (Genome / GPU Piper)…"
curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash
