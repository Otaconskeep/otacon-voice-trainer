FROM piper-voice-trainer:latest
# The base trainer ships torch 2.0.1+CPU, so piper_train runs on CPU (days for
# a fine-tune). Swap in CUDA 11.8 torch so training uses whichever GPU this
# host's Docker daemon exposes via `--gpus all` (currently the RTX 3090 on
# .219 -- there is no device-index selection, so despite the "2060" in
# earlier comments/messages elsewhere in the codebase, training has always
# landed on whatever GPU is local to this host). pytorch-lightning 1.6.5
# stays compatible.
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
# 2026-09-11 fix: installing the cu118 wheel above silently pulled numpy up
# to 2.2.6, breaking the base image's deliberate numpy pin ("numpy BEFORE
# torch -- avoids silent version conflicts", see ../Dockerfile). A
# torch-2.0.1-compiled extension trying to deserialize a cached checkpoint
# under numpy 2.x hits an ABI mismatch (_ARRAY_API not found) partway through
# torch.load()'s tensor rebuild, which pipeline.py's GPU-vs-CPU guard then
# reports as "training accelerator: GPU (cuda) -- CPU forbidden" -- a
# misleading symptom of a numpy version problem, not an actual GPU/CUDA
# failure (CUDA itself works fine; verified live with `--gpus all`).
#
# Re-pin to 1.26.4 (not 1.24.4): onnx 1.22 / ml_dtypes 0.6 need
# numpy.exceptions (>=1.25). Pinning 1.24.4 made train succeed then ONNX
# export fail with AttributeError: numpy has no attribute 'exceptions'
# (sniper_wolf_v1, 2026-09-11). Stay <2 for torch 2.0.1 ABI.
RUN pip install --no-cache-dir numpy==1.26.4
