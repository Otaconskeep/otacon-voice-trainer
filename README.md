# Otacon Voice Trainer (Genome)

**GPU-only Piper voice cloning for Otaconskeep.**  
Designed & engineered by **Antonio G. Garcia** · Built for the Keep.

One-click install. No silent CPU training. That hole is closed.

> **Credits:** Otacon Voice Trainer was designed and engineered by Antonio G. Garcia. AI-assisted tooling was used for selected implementation, installer, testing, and hardening tasks. See [`CREDITS.md`](./CREDITS.md).

---

## One command

**Linux / WSL2 Ubuntu**

```bash
curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash
```

**Windows** — double-click [`install_voice_trainer.bat`](./install_voice_trainer.bat)  
(sets up WSL2 + Ubuntu if needed, then runs the Linux installer).

**Together with Otacon** (included by default in the normal Otacon install)

```bash
curl -fsSL \
  https://raw.githubusercontent.com/Otaconskeep/otacons-ai-ecosystem/main/install_otacon.sh | bash
```

Skip Voice Trainer only: `OTACON_INSTALL_VOICE_TRAINER=0` before that command.

After install: open **http://127.0.0.1:8765/** for the simple local UI.

---

## What you get

| Piece | Role |
|-------|------|
| `piper-voice-trainer:latest` | Base image (CPU torch substrate only — **never train with this**) |
| `piper-voice-trainer:gpu` | CUDA 11.8 torch — **required** for training + Whisper |
| `pipeline.py` | YouTube → WAV → Whisper → Piper fine-tune → ONNX; **refuses CPU** |
| Simple UI | Local status panel after install |
| Otacon Core hook | Included by default (skip: `OTACON_INSTALL_VOICE_TRAINER=0`) |

The installer:

1. Detects Ubuntu/Debian or WSL2 (guides Windows users into WSL)
2. Installs Python 3, pip, git, curl
3. Installs Docker Engine
4. Looks up your NVIDIA GPU / VRAM (`nvidia-smi`)
5. Installs NVIDIA Container Toolkit (`docker --gpus all`)
6. Builds `:latest` then `:gpu`
7. Verifies `torch.cuda.is_available()` inside the image
8. Opens the simple UI

---

## Why GPU-only

An earlier Keep path could land Piper fine-tunes on **CPU** (`:latest` CPU torch wheels) and burn days of wall time. The public Keep pipeline now:

- Launches **only** `piper-voice-trainer:gpu` with `docker run --gpus all`
- Hard-fails in `pipeline.py` if CUDA is not visible
- Re-pins `numpy==1.26.4` after the CUDA torch layer (ONNX export + torch 2.0.1 ABI)

Full write-up: [`docs/WRITEUP.md`](./docs/WRITEUP.md)

---

## Requirements

- Ubuntu / Debian (native or **WSL2**)
- NVIDIA GPU + drivers (`nvidia-smi` works)
- ~20GB disk for the Docker image
- `sudo` for apt / Docker packages

---

## Repo layout

```
install_voice_trainer.sh   # one-click (ASCII banner + full bootstrap)
install_voice_trainer.bat  # Windows → WSL bridge
build.sh                   # docker build :latest + :gpu + verify
Dockerfile                 # base substrate
gpu-build/Dockerfile       # CUDA swap layer
pipeline.py                # training pipeline (GPU enforced)
patches/                   # Lightning checkpoint retention
ui/                        # simple local status UI
hooks/                     # Otacon Core install hook
docs/WRITEUP.md            # long-form write-up
```

---

## Links

- Site: [Otacon · Voice Trainer](https://otaconskeep-site.otaconskeep.workers.dev/otacon/#voice-trainer)
- Otacon Core: [otacons-ai-ecosystem](https://github.com/Otaconskeep/otacons-ai-ecosystem)
- Discord: https://discord.gg/cZDeqECzX

---

## License

MIT — see [LICENSE](./LICENSE).
