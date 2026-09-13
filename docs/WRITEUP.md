# Otacon Voice Trainer — Write-up

**Antonio G. Garcia // Otaconskeep**  
Genome Piper fine-tunes · GPU-only · one-click public installer

## Problem

Voice cloning for Keep agents (Piper fine-tunes from YouTube / uploads) is a
multi-hour GPU job. A packaging hole let the launcher use
`piper-voice-trainer:latest`, which ships **CPU** PyTorch wheels. Training
“worked” but ran on host CPUs for days, while the RTX sat idle. Operators
thought the pipeline was healthy because containers exited 0.

A second footgun: installing CUDA torch over the base image could pull
**numpy 2.x**, which breaks torch 2.0.1 checkpoint load / ONNX export —
surfacing as a confusing “CUDA forbidden” path even when the GPU was fine.

## Fix (Keep + public package)

1. **Image contract** — Otacon Executor / Genome launchers only pull
   `piper-voice-trainer:gpu` and always pass `docker run --gpus all`.
2. **Pipeline guard** — `pipeline.py` checks `torch.cuda.is_available()` and
   **raises** if false. No CPU accelerator args.
3. **Layered Docker** — `Dockerfile` builds the substrate; `gpu-build/Dockerfile`
   swaps in `torch==2.0.1+cu118` and re-pins `numpy==1.26.4`.
4. **One-click installer** — `install_voice_trainer.sh` installs Python, Docker,
   NVIDIA Container Toolkit, builds both tags, verifies CUDA inside the image,
   and opens a tiny local UI on `:8765`.
5. **Windows** — `install_voice_trainer.bat` routes through WSL2 Ubuntu (same
   pattern as Otacon Core’s Windows installer).
6. **Otacon Core hook** — `OTACON_INSTALL_VOICE_TRAINER=1` on
   `install_otacon.sh` chains this installer after Core.

## One-click commands

```bash
# Voice trainer alone
curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash

# Otacon Core + voice trainer
OTACON_INSTALL_VOICE_TRAINER=1 curl -fsSL \
  https://raw.githubusercontent.com/Otaconskeep/otacons-ai-ecosystem/main/install_otacon.sh | bash
```

## Verify

```bash
bash build.sh --verify-only
# expect: cuda_available True, device = your NVIDIA GPU, numpy ~1.26.4
```

## What this is / isn’t

| Is | Isn’t |
|----|-------|
| Public GPU Piper fine-tune image + installer | Full private Keep Genome portal |
| Local status UI on :8765 | Cloud training SaaS |
| Hook into Otacon Core bootstrap | A substitute for NVIDIA drivers |

## Credits

Designed and engineered by Antonio G. Garcia for Otaconskeep.  
Discord: https://discord.gg/cZDeqECzX
