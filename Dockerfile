FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Denver \
    PYTHONUNBUFFERED=1

# System deps: ffmpeg, espeak-ng (phonemizer), build tools for piper-phonemize
RUN apt-get update && apt-get install -y \
    ffmpeg git wget curl \
    espeak-ng espeak-ng-data \
    libsndfile1 libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN wget -qO /usr/local/bin/yt-dlp \
    https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

# 1) Pin pip FIRST (piper_train requires exactly 23.3.1)
RUN pip install pip==23.3.1

# 2) numpy BEFORE torch (avoids silent version conflicts).
#    Must be >=1.25 so onnx/ml_dtypes can import numpy.exceptions, and <2 so
#    torch 2.0.1 extensions keep a compatible NumPy ABI (numpy 2.x breaks
#    torch.load on this stack — see gpu-build/Dockerfile). 1.26.4 is the pin.
#    Failure mode if left at 1.24.4: training finishes, then ONNX export dies with
#    AttributeError: module 'numpy' has no attribute 'exceptions' (seen 2026-09-11
#    on sniper_wolf_v1 after epoch 3899).
RUN pip install numpy==1.26.4

# 3) PyTorch 2.0.1 CPU-only build (no CUDA runtime needed)
RUN pip install \
    torch==2.0.1 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 4) torchmetrics pinned
RUN pip install torchmetrics==0.11.4

# 5) pytorch-lightning pinned to 1.6.5 — 1.7.x added strict LR scheduler validation
#    that breaks piper_train's custom VITS ExponentialLR (MisconfigurationException)
RUN pip install pytorch-lightning==1.6.5

# 5b) Patch PL 1.6.5 to accept torch 2.0's renamed LRScheduler.
#     torch 2.0 made _LRScheduler a *subclass* of the new LRScheduler base class,
#     so ExponentialLR (inheriting from LRScheduler) fails PL's isinstance(_LRScheduler) check.
#     Add LRScheduler to LRSchedulerTypeTuple so the validation passes.
RUN python3 -c "import pytorch_lightning,os; f=os.path.join(os.path.dirname(pytorch_lightning.__file__),'utilities','types.py'); src=open(f).read(); old='LRSchedulerTypeTuple = (torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)'; new=old+\" + ((torch.optim.lr_scheduler.LRScheduler,) if hasattr(torch.optim.lr_scheduler,'LRScheduler') else ())\"; assert old in src; open(f,'w').write(src.replace(old,new)); print('Patched',f)"

# 6) piper_train dependencies (cython needed to compile monotonic_align extension)
#    Pin onnx/ml_dtypes so pip does not pull ml_dtypes 0.6 (requires numpy>=2),
#    which fights the torch 2.0.1 ABI pin above. onnx 1.16.x + ml_dtypes 0.4.x
#    work with numpy 1.26.4 and still support piper_train.export_onnx.
RUN pip install \
    cython==0.29.37 \
    piper-phonemize \
    onnxruntime==1.17.3 \
    "onnx==1.16.2" \
    "ml_dtypes==0.4.1" \
    phonemizer \
    librosa \
    requests \
    "numpy==1.26.4"


# 7) Install piper_train from GitHub source (not on PyPI)
#    --no-deps: all deps are already pinned above; prevents pip from
#    re-resolving and upgrading pytorch-lightning to 1.7.x which breaks
#    piper_train's custom VITS ExponentialLR scheduler.
#
#    monotonic_align fix: piper_train/__init__.py imports from
#    .monotonic_align.core (nested subpackage). Build from package root,
#    then move .so into the monotonic_align/ subdir where Python expects it.
RUN git clone --depth 1 https://github.com/rhasspy/piper.git /tmp/piper && \
    cd /tmp/piper/src/python && \
    pip install -e . --no-deps && \
    python piper_train/vits/monotonic_align/setup.py build_ext --inplace && \
    mkdir -p piper_train/vits/monotonic_align/monotonic_align && \
    touch piper_train/vits/monotonic_align/monotonic_align/__init__.py && \
    mv piper_train/vits/monotonic_align/core*.so \
       piper_train/vits/monotonic_align/monotonic_align/ && \
    rm -rf /tmp/piper/.git

# 7b) 2026-09-11: piper_train's VitsModel already accepts num_workers as a
#     constructor kwarg (vits/lightning.py, default=1, wired straight into
#     its train/val/test DataLoaders) but never exposes it on the CLI --
#     add_model_specific_args() simply omits it, so every run is stuck at
#     num_workers=1 regardless of how many cores the host has. Diagnosed
#     live on sniper_wolf_v1 (2026-09-11): with a 239-clip dataset and
#     num_workers=1, two host CPU cores pegged near 100% (single dataloader
#     worker, GIL-bound audio/mel preprocessing) while the RTX 3090 sat at
#     ~48% utilization waiting on batches -- a CPU-side bottleneck, not a
#     GPU capability limit. This just adds the missing --num-workers flag,
#     default=1 (IDENTICAL to current hardcoded behavior -- no existing job
#     changes), so it can actually be benchmarked and tuned. `dict_args`
#     (== vars(args)) is passed straight into VitsModel(**dict_args) in
#     __main__.py, so nothing else needs to change for this to take effect.
RUN python3 -c "\
import os; \
f = '/tmp/piper/src/python/piper_train/vits/lightning.py'; \
src = open(f).read(); \
old = 'parser.add_argument(\"--n-heads\", type=int, default=2)'; \
new = old + '\n        parser.add_argument(\"--num-workers\", type=int, default=1, help=\"DataLoader worker processes (default: 1, matches prior hardcoded behavior)\")'; \
assert old in src, 'anchor line not found -- piper_train source has changed, patch needs updating'; \
assert src.count(old) == 1, 'anchor line not unique'; \
open(f, 'w').write(src.replace(old, new)); \
print('Patched', f, '-- --num-workers now configurable, default=1')"

# demucs — vocal/background separation (strips music, SFX, ambient noise before training)
# Runs on CPU; slower than GPU but no CUDA runtime required
RUN pip install demucs

# Install openai-whisper last (pulls in its own torch dep — already satisfied)
RUN pip install openai-whisper

WORKDIR /workspace
COPY pipeline.py /workspace/pipeline.py

ENTRYPOINT ["python3", "/workspace/pipeline.py"]
