#!/usr/bin/env python3
"""
Otacon Genome Project — Voice Cloning Pipeline
Automated: YouTube URL(s) → WAV → Whisper → piper_train CPU → ONNX export

Crash-safe: each stage writes a completion marker so restarts skip done stages.
CPU training resumes from last checkpoint automatically.

Resilience rules (2026-03-21):
  - Never hard-fail due to dataset size alone
  - Small datasets (< MINIMAL_DATASET_THRESHOLD clips) activate Minimal Dataset Mode
  - Audio normalization is automatic before preprocess
  - Minimal Dataset Mode pads samples + uses batch_size=1
  - All failures are logged with cause; 'unknown' is valid when cause is absent

Dataset Intelligence (2026-03-21):
  - compute_dataset_metrics()     — 8 quality metrics from wav files + metadata.csv
  - compute_dataset_health_score() — 0–100 score (documented formula)
  - classify_dataset()             — STRONG/USABLE/WEAK/CRITICAL + training mode + quality prediction
  - Training modes: NORMAL, DEGRADED, MINIMAL, SURVIVAL
  - Expected output quality: HIGH, MEDIUM, LOW, VERY_LOW
  - All metrics persisted in pipeline_state.json for executor awareness integration
"""

import argparse
import collections
import glob
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

import requests
import whisper

CHECKPOINT_URL = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
)
CHECKPOINT_FILENAME = "en_US-lessac-medium-epoch2164.ckpt"

# Resilience thresholds (used for proactive padding in preprocess_dataset)
MINIMAL_DATASET_THRESHOLD = 50   # pad to target when valid_clips < this
MINIMAL_DATASET_TARGET    = 60   # pad to this count (above threshold, avoids edge cases)
MIN_PREPROCESS_VALID_CLIPS = 2
MIN_PREPROCESS_UTTERANCES = 2
MIN_PREPROCESS_TOTAL_AUDIO_SECONDS = 10.0

# Intelligence thresholds (used for training mode classification)
MINIMAL_MODE_THRESHOLD    = 8    # training_mode = MINIMAL when valid_clips < this (5–7 range)
# (below MINIMAL_MODE_THRESHOLD → proactive pad is always used;
#  below MINIMAL_DATASET_THRESHOLD but above MINIMAL_MODE_THRESHOLD → DEGRADED mode,
#  pad only triggered reactively if preprocess fails)

# Dataset health classification thresholds
HEALTH_STRONG   = 80
HEALTH_USABLE   = 50
HEALTH_WEAK     = 20
# < HEALTH_WEAK → CRITICAL

# Adaptive mode constants
PRUNE_PCT_DEFAULT         = 30    # default bottom % to prune
PRUNE_PCT_MAX             = 50    # safety cap: never prune more than 50%
ADAPTIVE_RETRAIN_DELTA    = 10    # min score improvement to trigger retrain
ADAPTIVE_RETRAIN_DUP_DROP = 0.15  # min dup ratio decrease to trigger retrain

# Training mode constants
TRAIN_MODE_NORMAL   = 'NORMAL'
TRAIN_MODE_DEGRADED = 'DEGRADED'
TRAIN_MODE_MINIMAL  = 'MINIMAL'
TRAIN_MODE_SURVIVAL = 'SURVIVAL'

# ---------------------------------------------------------------------------
# GPU training engine defaults (2026-09-11, flipped to measured winner 2026-09-12)
# ---------------------------------------------------------------------------
# Centralized here so a future benchmark-driven change is a ONE-LINE edit
# instead of hunting down scattered magic numbers. All of these are still
# per-job overridable (see train()'s num_workers/precision/batch_size
# params) -- changing a value here only changes what happens when a caller
# does NOT specify one explicitly.
#
# 2026-09-12: promoted to production defaults after a real 7-config
# benchmark sweep on sniper_wolf_v1 (same dataset/checkpoint as production),
# then validated end-to-end -- the resumed run using these exact values
# actually trained, converged, exported, and deployed successfully.
# Full comparison: /opt/otacon/voice-trainer/bench_winner_sniper_wolf_v1.json
#   baseline (workers=1, batch=4):  3.8 epochs/min,  ~38% GPU util
#   winner   (workers=1, batch=32): 12.4 epochs/min, ~76% GPU util, stall 1.7%
# Worker count (1/4/8) barely moved throughput at batch=4 (3.6-3.7 range) --
# the original "CPU dataloader bottleneck" hypothesis was only half right;
# batch size was the dominant lever. FP16/AMP twin of the winning config
# was measured too and did NOT beat FP32 batch=32, so precision stays 32.
#
# DEFAULT_ADDITIONAL_EPOCHS: replaces the old practice of hardcoding "4000"
# both here and (separately, redundantly) at pipeline.py's own callers.
# Piper's own upstream TRAINING.md: "2000 epochs is usually good for models
# trained from scratch, and an additional 1000 epochs when fine-tuning."
# Validated live: sniper_wolf_v1 was allotted 1000 additional epochs and
# the plateau detector correctly stopped it early at +400, well inside budget.
DEFAULT_NUM_WORKERS       = 1      # measured winner -- more workers didn't help at any batch size tested
DEFAULT_PRECISION         = "32"   # measured winner -- FP16/AMP twin of the best FP32 config did not beat it
DEFAULT_BATCH_SIZE_NORMAL = 32     # measured winner -- was 4; batch size, not worker count, was the real GPU-utilization lever
DEFAULT_BATCH_SIZE_SMALL  = 1      # unchanged: MINIMAL/SURVIVAL training_mode
DEFAULT_CHECKPOINT_EPOCHS = 100    # unchanged
DEFAULT_ADDITIONAL_EPOCHS = 1000   # upstream fine-tuning guidance; was a blind 4000

# Evidence-based stopping: convergence is evaluated (and always logged to
# pipeline_state.json's convergence_history) every time a checkpoint lands,
# using the same signal Piper's own docs describe as the "done" criterion:
# "the model is done when loss_disc_all levels off." A run is flagged
# plateaued when the mean of the most recent PLATEAU_WINDOW_EPOCHS-worth of
# loss_disc_all steps is within PLATEAU_THRESHOLD_PCT of the mean from the
# PREVIOUS window of the same size -- i.e. it stopped meaningfully improving.
# 2026-09-12: flipped ON, then flipped back OFF the same day -- sniper_wolf_v1
# is exactly the regression this comment warned about. The auto-stop fired
# cleanly (checkpoint/export/deploy all completed normally) at +400 of 1000
# allotted epochs, and the resulting voice was confirmed by actual human
# listening (through the real production path, serving-settings mismatch
# separately ruled out) to sound worse than before -- while the automated
# proxy metrics (Whisper WER, mel-spectrogram L2 vs reference clips) said
# it was fine. Loss-plateau is not a reliable proxy for perceptual quality
# on this dataset/model combo; back to OFF until that gap is closed with a
# real perceptual metric, not just loss curves.
PLATEAU_WINDOW_EPOCHS   = 200
PLATEAU_THRESHOLD_PCT   = 5.0
PLATEAU_MIN_EPOCHS_SEEN = 300   # don't evaluate plateau before this many epochs of THIS run have accumulated
AUTO_STOP_ON_PLATEAU_DEFAULT = False

# Expected output quality constants
QUALITY_HIGH     = 'HIGH'
QUALITY_MEDIUM   = 'MEDIUM'
QUALITY_LOW      = 'LOW'
QUALITY_VERY_LOW = 'VERY_LOW'


# ---------------------------------------------------------------------------
# Stage state tracking — crash-safe pipeline
# ---------------------------------------------------------------------------

def state_path(work_dir):
    return os.path.join(work_dir, "pipeline_state.json")


def load_state(work_dir):
    p = state_path(work_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_stages": []}


def mark_stage_done(work_dir, stage):
    state = load_state(work_dir)
    if stage not in state["completed_stages"]:
        state["completed_stages"].append(stage)
    _write_state(work_dir, state)
    print(f"[state] Stage '{stage}' marked complete")


def stage_done(work_dir, stage):
    return stage in load_state(work_dir).get("completed_stages", [])


def _write_state(work_dir, state):
    """Atomic state write via tmp + replace."""
    p = state_path(work_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, p)


def _update_dataset_health(work_dir, health: dict):
    """Write dataset_health into pipeline_state.json without touching other keys."""
    state = load_state(work_dir)
    state["dataset_health"] = health
    _write_state(work_dir, state)


def _force_preprocess_failure_requested(work_dir: str) -> bool:
    """Controlled failure probe hook for live runtime verification only."""
    markers = (
        os.path.join(work_dir, "force_preprocess_failure"),
        os.path.join(work_dir, ".force_preprocess_failure"),
    )
    return any(os.path.exists(marker) for marker in markers)


def _clear_runtime_failure(work_dir: str):
    state = load_state(work_dir)
    if "runtime_failure" in state:
        state.pop("runtime_failure", None)
        _write_state(work_dir, state)


def _record_runtime_failure(work_dir: str, exc: Exception):
    import traceback

    detail = traceback.format_exc().strip()
    summary = str(exc).strip() or exc.__class__.__name__
    state = load_state(work_dir)
    state["runtime_failure"] = {
        "summary": summary[:300],
        "detail": detail[-4000:] if detail else summary[:300],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_state(work_dir, state)


# ---------------------------------------------------------------------------
# Discord helpers
# ---------------------------------------------------------------------------

def discord_post(webhook: str, message: str):
    if not webhook:
        print(f"[discord] {message}")
        return
    try:
        # Discord 403s default bot UAs (python-requests / urllib); send the same
        # DiscordBot UA the executor uses so trainer progress posts actually land.
        requests.post(webhook, json={"content": message}, timeout=10,
                      headers={"User-Agent": "DiscordBot (https://github.com/otacon, 1.0)"})
    except Exception as exc:
        print(f"[discord error] {exc}")


# ---------------------------------------------------------------------------
# Dataset Intelligence — metrics, scoring, classification
# ---------------------------------------------------------------------------

def compute_dataset_metrics(dataset_dir: str) -> dict:
    """Compute 8 quality metrics from wav files and metadata.csv.

    Metrics:
        total_clips              — wav files found in dataset/wavs/
        valid_clips              — wavs that ffprobe can read + duration >= 0.5s
        metadata_rows            — non-blank lines with '|' in metadata.csv
        avg_clip_duration        — mean duration in seconds (valid clips only)
        duration_stddev          — std deviation of clip durations
        transcript_length_avg    — mean chars per transcript line
        total_audio_duration      — total duration in seconds across valid clips
        transcript_validity_ratio — (alpha + space chars) / total chars across all transcripts
        duplicate_ratio          — 1 - (unique transcripts / total transcripts)

    All values are safe: division-by-zero guarded, returns 0.0 on probe failure.
    """
    wavs_dir      = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")

    wav_files  = glob.glob(os.path.join(wavs_dir, "*.wav"))
    total_clips = len(wav_files)

    valid_clips = 0
    durations: list[float] = []

    for wav in wav_files:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", wav],
                capture_output=True, text=True, timeout=10
            )
            dur = float(json.loads(probe.stdout)["format"]["duration"])
            if dur >= 0.5:
                valid_clips += 1
                durations.append(dur)
        except Exception:
            pass  # probe failed — clip doesn't count as valid

    avg_clip_duration = sum(durations) / len(durations) if durations else 0.0
    total_audio_duration = sum(durations) if durations else 0.0
    if len(durations) > 1:
        variance = sum((d - avg_clip_duration) ** 2 for d in durations) / len(durations)
        duration_stddev = math.sqrt(variance)
    else:
        duration_stddev = 0.0

    transcripts: list[str] = []
    metadata_rows = 0
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2 and parts[1].strip():
                        transcripts.append(parts[1].strip())
                        metadata_rows += 1
        except Exception:
            pass

    if transcripts:
        all_text         = " ".join(transcripts)
        total_chars      = len(all_text)
        letter_chars     = sum(1 for c in all_text if c.isalpha() or c.isspace())
        transcript_validity_ratio = letter_chars / max(total_chars, 1)
        transcript_length_avg     = sum(len(t) for t in transcripts) / len(transcripts)

        unique_lower  = set(t.lower().strip() for t in transcripts)
        duplicate_ratio = 1.0 - (len(unique_lower) / len(transcripts))
    else:
        transcript_validity_ratio = 0.0
        transcript_length_avg     = 0.0
        duplicate_ratio           = 0.0

    return {
        "total_clips":               total_clips,
        "valid_clips":               valid_clips,
        "metadata_rows":             metadata_rows,
        "total_audio_duration":      round(total_audio_duration, 2),
        "avg_clip_duration":         round(avg_clip_duration, 2),
        "duration_stddev":           round(duration_stddev, 2),
        "transcript_length_avg":     round(transcript_length_avg, 1),
        "transcript_validity_ratio": round(transcript_validity_ratio, 3),
        "duplicate_ratio":           round(duplicate_ratio, 3),
    }


def compute_dataset_health_score(metrics: dict) -> float:
    """Compute 0–100 health score.

    Formula (weights sum to 1.0):
        valid_ratio    * 0.20  valid clips / total clips
        coverage       * 0.15  metadata rows / valid clips (capped at 1.0)
        validity       * 0.20  letter+space chars / all chars
        diversity      * 0.15  1 − duplicate_ratio
        duration_score * 0.15  ideal 5–15s clips; penalise extremes
        size_score     * 0.15  absolute clip count factor

    Duration scoring:
        < 0.5s  → 0.0   (likely silent/broken)
        0.5–3s  → 0.3
        3–5s    → 0.7
        5–15s   → 1.0   (optimal)
        15–25s  → 0.7
        > 25s   → 0.4

    Size scoring (absolute clip count — small datasets penalised even if per-clip quality is high):
        >= 50 clips → 1.0
        20–49       → 0.7
        10–19       → 0.4
        5–9         → 0.2
        < 5 clips   → 0.0
    """
    total_clips    = max(metrics.get("total_clips", 0), 1)
    valid_clips    = metrics.get("valid_clips", 0)
    metadata_rows  = metrics.get("metadata_rows", 0)
    validity       = metrics.get("transcript_validity_ratio", 0.0)
    dup_ratio      = metrics.get("duplicate_ratio", 0.0)
    avg_dur        = metrics.get("avg_clip_duration", 0.0)

    valid_ratio    = valid_clips / total_clips
    coverage       = min(metadata_rows / max(valid_clips, 1), 1.0)
    diversity      = max(0.0, 1.0 - dup_ratio)

    if avg_dur < 0.5:
        duration_score = 0.0
    elif avg_dur < 3.0:
        duration_score = 0.3
    elif avg_dur < 5.0:
        duration_score = 0.7
    elif avg_dur <= 15.0:
        duration_score = 1.0
    elif avg_dur <= 25.0:
        duration_score = 0.7
    else:
        duration_score = 0.4

    vc = valid_clips
    if vc >= 50:
        size_score = 1.0
    elif vc >= 20:
        size_score = 0.7
    elif vc >= 10:
        size_score = 0.4
    elif vc >= 5:
        size_score = 0.2
    else:
        size_score = 0.0

    score = (
        valid_ratio    * 0.20 +
        coverage       * 0.15 +
        validity       * 0.20 +
        diversity      * 0.15 +
        duration_score * 0.15 +
        size_score     * 0.15
    ) * 100

    return round(min(max(score, 0.0), 100.0), 1)


def classify_dataset(score: float, metrics: dict) -> dict:
    """Classify dataset health and determine training mode + expected output quality.

    Health classes:
        STRONG   80–100  → high quality, diverse, well-transcribed
        USABLE   50–79   → acceptable, may have minor issues
        WEAK     20–49   → significant issues, expect degraded output
        CRITICAL  0–19   → severe problems, training may not converge

    Training modes (size-first, then quality):
        NORMAL   → STRONG score AND valid_clips >= MINIMAL_MODE_THRESHOLD (20), batch_size=4
        DEGRADED → USABLE/WEAK score OR clips in [20, ∞) but not STRONG, batch_size=4
        MINIMAL  → valid_clips in [5, MINIMAL_MODE_THRESHOLD), padding will be used, batch_size=1
        SURVIVAL → valid_clips < 5 OR duplicate_ratio > 0.75, batch_size=1

    Expected output quality — score-based, then capped by training mode:
        HIGH     → score >= 70 and dup < 0.35 → NORMAL mode only
        MEDIUM   → score >= 45 and dup < 0.50 → NORMAL or DEGRADED (capped to MEDIUM for DEGRADED)
        LOW      → score >= 25 → MINIMAL mode (capped here)
        VERY_LOW → score < 25 or dup > 0.50 → SURVIVAL mode (always here)
    """
    valid_clips   = metrics.get("valid_clips", 0)
    dup_ratio     = metrics.get("duplicate_ratio", 0.0)

    # Health class
    if score >= HEALTH_STRONG:
        health_class = "STRONG"
    elif score >= HEALTH_USABLE:
        health_class = "USABLE"
    elif score >= HEALTH_WEAK:
        health_class = "WEAK"
    else:
        health_class = "CRITICAL"

    # Training mode
    if valid_clips < 5 or dup_ratio > 0.75:
        training_mode = TRAIN_MODE_SURVIVAL
    elif valid_clips < MINIMAL_MODE_THRESHOLD:
        training_mode = TRAIN_MODE_MINIMAL
    elif health_class == "STRONG":
        training_mode = TRAIN_MODE_NORMAL
    else:
        training_mode = TRAIN_MODE_DEGRADED

    # Expected output quality — score-based first
    if score < 25 or dup_ratio > 0.50:
        expected_quality = QUALITY_VERY_LOW
    elif score < 45 or dup_ratio > 0.35:
        expected_quality = QUALITY_LOW
    elif score < 70:
        expected_quality = QUALITY_MEDIUM
    else:
        expected_quality = QUALITY_HIGH

    # Mode-based quality caps (modes with padded/sparse data cannot achieve top quality):
    #   SURVIVAL → always VERY_LOW
    #   MINIMAL  → cap at LOW (real unique patterns too few even after padding)
    #   DEGRADED → cap at MEDIUM (real data present but quality concerns)
    if training_mode == TRAIN_MODE_SURVIVAL:
        expected_quality = QUALITY_VERY_LOW
    elif training_mode == TRAIN_MODE_MINIMAL and expected_quality in (QUALITY_HIGH, QUALITY_MEDIUM):
        expected_quality = QUALITY_LOW
    elif training_mode == TRAIN_MODE_DEGRADED and expected_quality == QUALITY_HIGH:
        expected_quality = QUALITY_MEDIUM

    # Human-readable notes
    notes: list[str] = []
    if valid_clips < 5:
        notes.append(f"critically small dataset ({valid_clips} valid clips)")
    elif valid_clips < 15:
        notes.append(f"very few samples ({valid_clips} valid clips)")
    elif valid_clips < MINIMAL_DATASET_THRESHOLD:
        notes.append(f"small dataset ({valid_clips} clips, padding required)")
    if dup_ratio > 0.75:
        notes.append(f"extreme duplication ({dup_ratio:.0%}) — training will overfit")
    elif dup_ratio > 0.50:
        notes.append(f"high duplication ({dup_ratio:.0%}) — expect low diversity")
    elif dup_ratio > 0.20:
        notes.append(f"moderate duplication ({dup_ratio:.0%})")
    tvr = metrics.get("transcript_validity_ratio", 1.0)
    if tvr < 0.60:
        notes.append("transcript quality poor (low letter ratio — possible noise/garbled text)")
    elif tvr < 0.80:
        notes.append("transcript quality concerns (some noise in transcriptions)")
    avg_dur = metrics.get("avg_clip_duration", 10.0)
    if avg_dur < 1.0:
        notes.append("clips very short — may degrade phoneme coverage")
    elif avg_dur > 20.0:
        notes.append("clips very long — may affect training stability")
    if not notes:
        notes.append("no major issues detected")

    return {
        "health_class":          health_class,
        "training_mode":         training_mode,
        "expected_output_quality": expected_quality,
        "notes":                 notes,
    }


# ---------------------------------------------------------------------------
# Adaptive Genome — per-clip scoring, pruning, rebalancing, retrain decision
# ---------------------------------------------------------------------------

def score_clips(dataset_dir: str, work_dir: str) -> dict:
    """Score every wav in dataset/wavs/ across 5 dimensions.

    Scores:
        audio_score       — ffprobe readable + non-zero size. 100=mono, 70=stereo, 0=unreadable/empty (<1KB)
        duration_score    — <0.3s→0, 0.3-1s→20, 1-2s→50, 2-10s→100, 10-15s→80, 15-20s→50, >20s→20
        transcript_score  — alpha_ratio*0.70 + len_factor*0.30 (0 if no transcript)
        alignment_score   — word_count vs expected (duration*2.5 words/sec). 0 if no transcript/duration.
        duplicate_score   — first-50-chars lowercase fingerprint; 1 copy→100, 2→60, 3-4→30, 5+→0
        clip_score        — weighted composite: audio*0.20 + duration*0.25 + transcript*0.30 + alignment*0.15 + duplicate*0.10

    Writes {work_dir}/adaptive/clip_scores.json (atomic).
    Returns dict: {stem: {score, audio_score, duration_score, transcript_score, alignment_score,
                          duplicate_score, duration, transcript_preview, fingerprint, pruned, prune_reason}}
    """
    wavs_dir      = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    adaptive_dir  = os.path.join(work_dir, "adaptive")
    os.makedirs(adaptive_dir, exist_ok=True)

    # Load transcripts keyed by stem
    transcripts: dict[str, str] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "|" not in line:
                        continue
                    parts = line.split("|", 2)
                    stem = parts[0].strip()
                    text = parts[1].strip() if len(parts) >= 2 else ""
                    transcripts[stem] = text
        except Exception:
            pass

    wav_files = glob.glob(os.path.join(wavs_dir, "*.wav"))
    # Build fingerprint counts first (for duplicate_score)
    fp_counts: dict[str, int] = collections.Counter()
    stem_fp: dict[str, str] = {}
    for wav in wav_files:
        stem = os.path.splitext(os.path.basename(wav))[0]
        txt  = transcripts.get(stem, "")
        fp   = txt.lower().strip()[:50] if txt else ""
        stem_fp[stem] = fp
        if fp:
            fp_counts[fp] += 1

    clip_scores: dict[str, dict] = {}

    for wav in wav_files:
        stem = os.path.splitext(os.path.basename(wav))[0]
        size = os.path.getsize(wav) if os.path.exists(wav) else 0

        # ── audio_score ────────────────────────────────────────────────────
        audio_score = 0
        duration    = 0.0
        is_stereo   = False
        if size < 1024:
            audio_score = 0
        else:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_format", "-show_streams", wav],
                    capture_output=True, text=True, timeout=10
                )
                pd = json.loads(probe.stdout)
                duration = float(pd.get("format", {}).get("duration", 0))
                channels = int(pd.get("streams", [{}])[0].get("channels", 1))
                if duration > 0:
                    is_stereo   = channels > 1
                    audio_score = 70 if is_stereo else 100
                else:
                    audio_score = 0
            except Exception:
                audio_score = 0

        # ── duration_score ─────────────────────────────────────────────────
        if duration <= 0:
            duration_score = 0
        elif duration < 0.3:
            duration_score = 0
        elif duration < 1.0:
            duration_score = 20
        elif duration < 2.0:
            duration_score = 50
        elif duration <= 10.0:
            duration_score = 100
        elif duration <= 15.0:
            duration_score = 80
        elif duration <= 20.0:
            duration_score = 50
        else:
            duration_score = 20

        # ── transcript_score ───────────────────────────────────────────────
        txt = transcripts.get(stem, "")
        if not txt:
            transcript_score = 0
        else:
            alpha_chars = sum(1 for c in txt if c.isalpha() or c.isspace())
            alpha_ratio = alpha_chars / max(len(txt), 1)
            len_factor  = min(len(txt) / 15.0, 1.0)
            transcript_score = alpha_ratio * 70.0 + len_factor * 30.0

        # ── alignment_score ────────────────────────────────────────────────
        if not txt or duration <= 0:
            alignment_score = 0
        else:
            word_count    = len(txt.split())
            expected_words = duration * 2.5
            ratio         = word_count / max(expected_words, 0.01)
            if 0.4 <= ratio <= 2.5:
                alignment_score = 100
            elif (0.2 <= ratio < 0.4) or (2.5 < ratio <= 4.0):
                alignment_score = 60
            else:
                alignment_score = 20

        # ── duplicate_score ────────────────────────────────────────────────
        fp = stem_fp.get(stem, "")
        if not fp:
            duplicate_score = 50  # no transcript → neutral
        else:
            count = fp_counts.get(fp, 1)
            if count == 1:
                duplicate_score = 100
            elif count == 2:
                duplicate_score = 60
            elif count <= 4:
                duplicate_score = 30
            else:
                duplicate_score = 0

        # ── composite clip_score ───────────────────────────────────────────
        clip_score = (
            audio_score      * 0.20 +
            duration_score   * 0.25 +
            transcript_score * 0.30 +
            alignment_score  * 0.15 +
            duplicate_score  * 0.10
        )

        clip_scores[stem] = {
            "score":              round(clip_score, 2),
            "audio_score":        audio_score,
            "duration_score":     duration_score,
            "transcript_score":   round(transcript_score, 2),
            "alignment_score":    alignment_score,
            "duplicate_score":    duplicate_score,
            "duration":           round(duration, 3),
            "transcript_preview": txt[:80] if txt else "",
            "fingerprint":        fp,
            "pruned":             False,
            "prune_reason":       "",
        }

    # Atomic write
    out_path = os.path.join(adaptive_dir, "clip_scores.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clip_scores, f)
    os.replace(tmp_path, out_path)

    print(f"[adaptive] scored {len(clip_scores)} clips → {out_path}")
    return clip_scores


def auto_prune_dataset(dataset_dir: str, work_dir: str, clip_scores: dict,
                       prune_pct: float = PRUNE_PCT_DEFAULT,
                       min_clip_score: float = 15.0) -> dict:
    """Move low-quality clips out of dataset to adaptive/rejected/.

    SAFETY: NEVER deletes files. Only moves via shutil.move().
    - Backs up metadata.csv to metadata.csv.adaptive_bak (only if bak doesn't exist yet)
    - Caps prune at min(prune_pct, PRUNE_PCT_MAX)
    - Never prunes below 5 remaining clips
    - Prunes clips where: score < min_clip_score OR audio_score==0 OR duplicate_score==0
    - Then fills remaining quota from bottom-scored clips (up to the cap)

    Returns: {clips_removed, clips_kept, reasons: {stem: reason}, metadata_backed_up}
    """
    wavs_dir      = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    adaptive_dir  = os.path.join(work_dir, "adaptive")
    rejected_dir  = os.path.join(adaptive_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    total = len(clip_scores)
    if total == 0:
        return {"clips_removed": 0, "clips_kept": 0, "reasons": {}, "metadata_backed_up": False}

    # Backup metadata
    backed_up = False
    bak_path  = metadata_path + ".adaptive_bak"
    if os.path.exists(metadata_path) and not os.path.exists(bak_path):
        shutil.copy2(metadata_path, bak_path)
        backed_up = True
        print(f"[adaptive] backed up metadata.csv → metadata.csv.adaptive_bak")

    # Cap prune percentage
    effective_pct = min(prune_pct, PRUNE_PCT_MAX)
    max_to_prune  = int(total * effective_pct / 100.0)
    # Never go below 5 remaining
    max_to_prune  = min(max_to_prune, max(0, total - 5))

    # Phase 1: mandatory prunes (bad audio, silent, exact dupes ×5+)
    reasons: dict[str, str] = {}
    mandatory: list[str] = []
    for stem, info in clip_scores.items():
        if info.get("audio_score", 100) == 0:
            reasons[stem] = "unreadable_audio"
            mandatory.append(stem)
        elif info.get("duplicate_score", 100) == 0:
            reasons[stem] = "extreme_duplicate"
            mandatory.append(stem)
        elif info.get("score", 100) < min_clip_score:
            reasons[stem] = f"low_score_{info['score']:.1f}"
            mandatory.append(stem)

    # Phase 2: fill remaining quota from worst scored (not already in mandatory)
    already_flagged = set(mandatory)
    sorted_by_score = sorted(
        [(s, info["score"]) for s, info in clip_scores.items() if s not in already_flagged],
        key=lambda x: x[1]
    )
    remaining_quota = max(0, max_to_prune - len(mandatory))
    optional: list[str] = []
    for stem, score in sorted_by_score[:remaining_quota]:
        optional.append(stem)
        reasons[stem] = f"bottom_pct_score_{score:.1f}"

    to_prune = mandatory[:max_to_prune] + optional
    # Enforce max_to_prune hard cap
    to_prune = to_prune[:max_to_prune]

    # Move wavs to rejected dir
    moved = 0
    for stem in to_prune:
        wav_src = os.path.join(wavs_dir, stem + ".wav")
        wav_dst = os.path.join(rejected_dir, stem + ".wav")
        if os.path.exists(wav_src):
            shutil.move(wav_src, wav_dst)
            moved += 1
        clip_scores[stem]["pruned"]       = True
        clip_scores[stem]["prune_reason"] = reasons.get(stem, "unknown")

    # Rewrite metadata.csv excluding pruned stems
    pruned_set = set(to_prune)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            kept_lines = []
            for line in lines:
                if "|" not in line:
                    kept_lines.append(line)
                    continue
                stem = line.split("|", 1)[0].strip()
                if stem not in pruned_set:
                    kept_lines.append(line)
            tmp_meta = metadata_path + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            os.replace(tmp_meta, metadata_path)
        except Exception as exc:
            print(f"[adaptive] warning: failed to rewrite metadata.csv: {exc}")

    # Update clip_scores.json
    out_path = os.path.join(adaptive_dir, "clip_scores.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clip_scores, f)
    os.replace(tmp_path, out_path)

    kept = total - moved
    print(f"[adaptive] pruned {moved} clips, kept {kept} — max_to_prune was {max_to_prune}")
    return {
        "clips_removed":     moved,
        "clips_kept":        kept,
        "reasons":           reasons,
        "metadata_backed_up": backed_up,
    }


def rebalance_dataset(dataset_dir: str, clip_scores: dict, work_dir: str) -> dict:
    """Enforce transcript diversity — cap duplicate fingerprints at 2 per group.

    Skips if ≤10 clips remain (don't over-prune tiny datasets).
    Keeps the best 2 clips per fingerprint group (by clip score).
    Never removes below 5 remaining clips.

    Returns: {removed, reason}
    """
    wavs_dir      = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    adaptive_dir  = os.path.join(work_dir, "adaptive")
    rejected_dir  = os.path.join(adaptive_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    # Only consider non-pruned clips
    active = {s: info for s, info in clip_scores.items() if not info.get("pruned", False)}
    if len(active) <= 10:
        print(f"[adaptive] rebalance skipped: only {len(active)} active clips (≤10)")
        return {"removed": 0, "reason": "too_few_clips"}

    # Group by fingerprint
    fp_groups: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for stem, info in active.items():
        fp = info.get("fingerprint", "")
        if fp:
            fp_groups[fp].append((stem, info.get("score", 0.0)))

    to_remove: list[str] = []
    for fp, items in fp_groups.items():
        if len(items) <= 2:
            continue
        # Sort best first, keep top 2, remove the rest
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        for stem, _ in items_sorted[2:]:
            to_remove.append(stem)

    # Never remove below 5 remaining
    cap = max(0, len(active) - 5)
    to_remove = to_remove[:cap]

    moved = 0
    for stem in to_remove:
        wav_src = os.path.join(wavs_dir, stem + ".wav")
        wav_dst = os.path.join(rejected_dir, stem + ".wav")
        if os.path.exists(wav_src):
            shutil.move(wav_src, wav_dst)
            moved += 1
        clip_scores[stem]["pruned"]       = True
        clip_scores[stem]["prune_reason"] = "rebalance_duplicate"

    # Rewrite metadata.csv
    remove_set = set(to_remove)
    if to_remove and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            kept_lines = []
            for line in lines:
                if "|" not in line:
                    kept_lines.append(line)
                    continue
                stem = line.split("|", 1)[0].strip()
                if stem not in remove_set:
                    kept_lines.append(line)
            tmp_meta = metadata_path + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            os.replace(tmp_meta, metadata_path)
        except Exception as exc:
            print(f"[adaptive] rebalance warning: failed to rewrite metadata.csv: {exc}")

    # Update clip_scores.json
    out_path = os.path.join(adaptive_dir, "clip_scores.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clip_scores, f)
    os.replace(tmp_path, out_path)

    print(f"[adaptive] rebalance removed {moved} excess duplicates")
    return {"removed": moved, "reason": "duplicate_cap_2_per_fingerprint"}


def should_retrain(before_health: dict, after_health: dict) -> tuple:
    """Compute retrain decision after adaptive pruning.

    Returns (decision, reason) where decision is one of:
        FORCE_RETRAIN  — major improvement, training strongly recommended
        RETRAIN        — worthwhile improvement detected
        SKIP           — improvement too small or dataset still broken

    Logic:
        - Still CRITICAL → SKIP
        - score_delta >= 25 AND after_class in STRONG/USABLE → FORCE_RETRAIN
        - score_delta >= ADAPTIVE_RETRAIN_DELTA → RETRAIN
        - dup_delta >= ADAPTIVE_RETRAIN_DUP_DROP → RETRAIN
        - class improved (CRITICAL→WEAK→USABLE→STRONG) → RETRAIN
        - score_delta < 3 → SKIP
        - else → SKIP
    """
    CLASS_RANK = {"CRITICAL": 0, "WEAK": 1, "USABLE": 2, "STRONG": 3}
    before_score = before_health.get("score", 0.0)
    after_score  = after_health.get("score", 0.0)
    score_delta  = after_score - before_score

    before_dup   = before_health.get("duplicate_ratio", 0.0)
    after_dup    = after_health.get("duplicate_ratio", 0.0)
    dup_delta    = before_dup - after_dup  # positive = improvement

    before_class = before_health.get("health_class", "CRITICAL")
    after_class  = after_health.get("health_class", "CRITICAL")

    # Still CRITICAL → don't bother training
    if after_class == "CRITICAL":
        return ("SKIP", "Dataset still CRITICAL")

    # Major score jump
    if score_delta >= 25 and after_class in ("STRONG", "USABLE"):
        return ("FORCE_RETRAIN", f"Major improvement: score {before_score:.0f}→{after_score:.0f} (+{score_delta:.0f}), class {after_class}")

    # Meaningful score improvement
    if score_delta >= ADAPTIVE_RETRAIN_DELTA:
        return ("RETRAIN", f"Score improved {before_score:.0f}→{after_score:.0f} (+{score_delta:.0f})")

    # Dup ratio improved significantly
    if dup_delta >= ADAPTIVE_RETRAIN_DUP_DROP:
        return ("RETRAIN", f"Duplicate ratio reduced {before_dup:.0%}→{after_dup:.0%} (delta {dup_delta:.0%})")

    # Class upgraded
    before_rank = CLASS_RANK.get(before_class, 0)
    after_rank  = CLASS_RANK.get(after_class, 0)
    if after_rank > before_rank:
        return ("RETRAIN", f"Dataset class improved: {before_class} → {after_class}")

    # Negligible improvement
    if score_delta < 3:
        return ("SKIP", "improvement too small")

    return ("SKIP", "marginal improvement")


def _update_adaptive_state(work_dir: str, summary: dict) -> None:
    """Atomically write adaptive summary to pipeline_state.json."""
    state = load_state(work_dir)
    state["adaptive"] = summary
    _write_state(work_dir, state)


def adaptive_improve_loop(dataset_dir: str, work_dir: str, webhook: str,
                          prune_pct: float = PRUNE_PCT_DEFAULT) -> tuple:
    """Run ASSESS → CORRECT → REASSESS → DECIDE loop before preprocessing.

    Returns (training_mode, retrain_decision, adaptive_summary).
    """
    print("[adaptive] Starting adaptive improvement loop")
    discord_post(webhook, f"🔬 **Adaptive Genome** — starting dataset assessment for `{os.path.basename(work_dir)}`")

    # ── ASSESS ────────────────────────────────────────────────────────────────
    clip_scores    = score_clips(dataset_dir, work_dir)
    before_health  = validate_dataset(dataset_dir, work_dir)
    before_mode    = before_health.get("training_mode", TRAIN_MODE_NORMAL)
    before_score   = before_health.get("score", 0.0)
    before_class   = before_health.get("health_class", "CRITICAL")

    clips_total = len(clip_scores)
    print(f"[adaptive] before: score={before_score}, class={before_class}, clips={clips_total}")

    # Guard: too few clips to prune safely
    if clips_total <= 5:
        summary = {
            "enabled": True,
            "skipped": True,
            "skip_reason": "too_few_clips",
            "clips_scored": clips_total,
            "clips_removed": 0,
            "clips_kept": clips_total,
            "before_score": before_score,
            "before_class": before_class,
            "after_score": before_score,
            "after_class": before_class,
            "improvement_delta": 0,
            "retrain_decision": "SKIP",
            "retrain_reason": "too few clips to prune",
            "prune_reasons": {},
            "training_mode": before_mode,
        }
        _update_adaptive_state(work_dir, summary)
        return (before_mode, "SKIP", summary)

    # ── CORRECT ───────────────────────────────────────────────────────────────
    prune_result  = auto_prune_dataset(dataset_dir, work_dir, clip_scores, prune_pct=prune_pct)
    rebal_result  = rebalance_dataset(dataset_dir, clip_scores, work_dir)

    total_removed = prune_result["clips_removed"] + rebal_result.get("removed", 0)

    # If nothing was removed, skip reassess
    if total_removed == 0:
        summary = {
            "enabled": True,
            "skipped": False,
            "clips_scored": clips_total,
            "clips_removed": 0,
            "clips_kept": clips_total,
            "before_score": before_score,
            "before_class": before_class,
            "after_score": before_score,
            "after_class": before_class,
            "improvement_delta": 0,
            "retrain_decision": "SKIP",
            "retrain_reason": "nothing pruned",
            "prune_reasons": prune_result.get("reasons", {}),
            "training_mode": before_mode,
        }
        _update_adaptive_state(work_dir, summary)
        discord_post(webhook, f"🔬 **Adaptive Genome** — no clips pruned, dataset already clean. Proceeding as-is.")
        return (before_mode, "SKIP", summary)

    # ── REASSESS ──────────────────────────────────────────────────────────────
    after_health  = validate_dataset(dataset_dir, work_dir)
    after_mode    = after_health.get("training_mode", TRAIN_MODE_NORMAL)
    after_score   = after_health.get("score", 0.0)
    after_class   = after_health.get("health_class", "CRITICAL")
    delta         = after_score - before_score

    # ── DECIDE ────────────────────────────────────────────────────────────────
    decision, retrain_reason = should_retrain(before_health, after_health)

    clips_kept = prune_result["clips_kept"] - rebal_result.get("removed", 0)

    summary = {
        "enabled": True,
        "skipped": False,
        "clips_scored": clips_total,
        "clips_removed": total_removed,
        "clips_kept": clips_kept,
        "before_score": round(before_score, 2),
        "before_class": before_class,
        "after_score": round(after_score, 2),
        "after_class": after_class,
        "improvement_delta": round(delta, 2),
        "retrain_decision": decision,
        "retrain_reason": retrain_reason,
        "prune_reasons": prune_result.get("reasons", {}),
        "training_mode": after_mode,
    }
    _update_adaptive_state(work_dir, summary)

    discord_post(
        webhook,
        f"🔬 **Adaptive Genome** — assessment complete for `{os.path.basename(work_dir)}`\n"
        f"Score: **{before_score:.0f}** → **{after_score:.0f}** (+{delta:.0f}) | "
        f"Class: {before_class} → {after_class}\n"
        f"Removed: {total_removed} clips | Kept: {clips_kept} | "
        f"Decision: **{decision}** — {retrain_reason}"
    )
    print(f"[adaptive] decision={decision} | score {before_score:.0f}→{after_score:.0f} | removed {total_removed}")
    return (after_mode, decision, summary)


# ---------------------------------------------------------------------------
# Phase 1: Dataset validation — called before preprocess
# ---------------------------------------------------------------------------

def validate_dataset(dataset_dir: str, work_dir: str) -> dict:
    """Compute full dataset intelligence and write to pipeline_state.json.

    Calls compute_dataset_metrics() → compute_dataset_health_score() → classify_dataset().

    Returns health dict:
        score                  — 0–100 float
        health_class           — STRONG | USABLE | WEAK | CRITICAL
        training_mode          — NORMAL | DEGRADED | MINIMAL | SURVIVAL
        expected_output_quality — HIGH | MEDIUM | LOW | VERY_LOW
        total_clips            — wavs in dataset/wavs/
        valid_clips            — readable wavs >= 0.5s
        metadata_rows          — lines with '|' in metadata.csv
        total_audio_duration   — total valid wav duration (s)
        avg_clip_duration      — mean duration (s)
        duration_stddev
        transcript_validity_ratio
        duplicate_ratio
        notes                  — list of human-readable observations
        minimal_mode           — True if training_mode in MINIMAL/SURVIVAL (legacy compat)
        metadata_exists        — bool
        padded                 — False (set to True by _pad_dataset_to_minimum)
        timestamp              — ISO-8601
    """
    metadata_path   = os.path.join(dataset_dir, "metadata.csv")
    metadata_exists = os.path.exists(metadata_path)

    # Full metrics (includes valid_clip probing — may take a few seconds for large datasets)
    metrics = compute_dataset_metrics(dataset_dir)
    score   = compute_dataset_health_score(metrics)
    cls     = classify_dataset(score, metrics)

    training_mode = cls["training_mode"]
    minimal_mode  = training_mode in (TRAIN_MODE_MINIMAL, TRAIN_MODE_SURVIVAL)

    health: dict = {
        "score":                   score,
        "health_class":            cls["health_class"],
        "training_mode":           training_mode,
        "expected_output_quality": cls["expected_output_quality"],
        "notes":                   cls["notes"],
        # raw metrics
        "total_clips":               metrics["total_clips"],
        "valid_clips":               metrics["valid_clips"],
        "metadata_rows":             metrics["metadata_rows"],
        "total_audio_duration":      metrics["total_audio_duration"],
        "avg_clip_duration":         metrics["avg_clip_duration"],
        "duration_stddev":           metrics["duration_stddev"],
        "transcript_length_avg":     metrics["transcript_length_avg"],
        "transcript_validity_ratio": metrics["transcript_validity_ratio"],
        "duplicate_ratio":           metrics["duplicate_ratio"],
        # legacy / control flags
        "minimal_mode":    minimal_mode,
        "metadata_exists": metadata_exists,
        "padded":          False,
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    _update_dataset_health(work_dir, health)

    print(
        f"[dataset] score={score} ({cls['health_class']}) | "
        f"{metrics['valid_clips']}/{metrics['total_clips']} valid clips | "
        f"dup={metrics['duplicate_ratio']:.0%} | "
        f"mode={training_mode} | quality={cls['expected_output_quality']}"
    )
    if cls["notes"]:
        for note in cls["notes"]:
            print(f"[dataset]   note: {note}")

    return health


def enforce_preprocess_requirements(health: dict) -> list[str]:
    """Return human-readable rejection reasons for datasets that cannot safely preprocess."""
    reasons: list[str] = []
    valid_clips = int(health.get("valid_clips") or 0)
    utterances = int(health.get("metadata_rows") or 0)
    total_audio = float(health.get("total_audio_duration") or 0.0)

    if valid_clips < MIN_PREPROCESS_VALID_CLIPS:
        reasons.append(
            f"need at least {MIN_PREPROCESS_VALID_CLIPS} valid clips; found {valid_clips}"
        )
    if utterances < MIN_PREPROCESS_UTTERANCES:
        reasons.append(
            f"need at least {MIN_PREPROCESS_UTTERANCES} utterances/transcripts; found {utterances}"
        )
    if total_audio < MIN_PREPROCESS_TOTAL_AUDIO_SECONDS:
        reasons.append(
            f"need at least {MIN_PREPROCESS_TOTAL_AUDIO_SECONDS:.0f}s total valid audio; found {total_audio:.2f}s"
        )
    return reasons


# ---------------------------------------------------------------------------
# Phase 4: Audio normalization — mono, 22050 Hz, PCM before preprocess
# ---------------------------------------------------------------------------

def normalize_wavs_for_preprocess(dataset_dir: str):
    """Ensure all wavs in dataset/wavs/ are mono, 22050 Hz, pcm_s16le.

    Normalizes in-place via ffmpeg. Skips files that are already correct.
    A .bak file is NOT kept — the original is replaced atomically.
    Logs per-file results at warning level only.
    """
    wavs_dir  = os.path.join(dataset_dir, "wavs")
    wav_files = glob.glob(os.path.join(wavs_dir, "*.wav"))

    if not wav_files:
        print("[normalize] no wavs found — skipping normalization")
        return

    already_ok = 0
    fixed      = 0
    failed     = 0

    for wav_path in wav_files:
        # Probe current format
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", wav_path],
                capture_output=True, text=True, timeout=10
            )
            streams = json.loads(probe.stdout).get("streams", [])
            audio   = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if audio:
                sr    = int(audio.get("sample_rate", 0))
                ch    = int(audio.get("channels", 0))
                codec = audio.get("codec_name", "")
                if sr == 22050 and ch == 1 and codec == "pcm_s16le":
                    already_ok += 1
                    continue
        except Exception:
            pass  # probe failed — attempt normalization anyway

        tmp_path = wav_path + ".norm.tmp"
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le",
                 tmp_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and os.path.exists(tmp_path):
                os.replace(tmp_path, wav_path)
                fixed += 1
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                print(f"[normalize] ⚠️  failed to normalize {os.path.basename(wav_path)}: "
                      f"{result.stderr[:120]}")
                failed += 1
        except Exception as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            print(f"[normalize] ⚠️  exception normalizing {os.path.basename(wav_path)}: {exc}")
            failed += 1

    print(f"[normalize] {already_ok} already ok | {fixed} normalized | {failed} failed "
          f"(total: {len(wav_files)})")


# ---------------------------------------------------------------------------
# Phase 3: Pad dataset to minimum — for Minimal Dataset Mode
# ---------------------------------------------------------------------------

def _pad_dataset_to_minimum(dataset_dir: str, target_count: int) -> int:
    """Duplicate existing samples until dataset reaches target_count.

    Backs up metadata.csv before rewriting it.
    Returns the new wav count.
    """
    wavs_dir      = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")

    wav_files = sorted(glob.glob(os.path.join(wavs_dir, "*.wav")))
    if not wav_files:
        print("[pad] no source wavs — cannot pad")
        return 0

    # Build entry map from existing metadata
    entry_map: dict[str, str] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "|" not in line:
                        continue
                    name, _, text = line.partition("|")
                    entry_map[name.strip()] = text.strip()
        except Exception:
            pass

    current_count = len(wav_files)
    if current_count >= target_count:
        return current_count

    # Backup metadata before modifying
    if os.path.exists(metadata_path):
        shutil.copy2(metadata_path, metadata_path + ".bak")
        print(f"[pad] backed up metadata.csv → metadata.csv.bak")

    cycle     = list(wav_files)
    pad_index = 0

    # Seed new_entries from backup (preserves original order)
    bak_path    = metadata_path + ".bak"
    new_entries: list[str] = []
    if os.path.exists(bak_path):
        try:
            with open(bak_path, "r", encoding="utf-8") as f:
                new_entries = [line.rstrip("\n") for line in f if "|" in line]
        except Exception:
            pass
    if not new_entries:
        # Fallback: reconstruct from in-memory map (no guaranteed order, but functional)
        new_entries = [f"{k}|{v}" for k, v in entry_map.items()]

    while current_count < target_count:
        src      = cycle[pad_index % len(cycle)]
        pad_index += 1
        src_name = os.path.splitext(os.path.basename(src))[0]
        new_name = f"pad_{pad_index:05d}"
        dest     = os.path.join(wavs_dir, f"{new_name}.wav")

        try:
            shutil.copy2(src, dest)
        except Exception as exc:
            print(f"[pad] ⚠️  could not copy {src} → {dest}: {exc}")
            continue

        text = entry_map.get(src_name, "training sample")
        new_entries.append(f"{new_name}|{text}")
        current_count += 1

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_entries) + "\n")
    except Exception as exc:
        print(f"[pad] ⚠️  failed to rewrite metadata.csv: {exc}")

    print(f"[pad] dataset padded: {len(wav_files)} → {current_count} samples "
          f"(target was {target_count})")
    return current_count


# ---------------------------------------------------------------------------
# Step 1: Download audio from YouTube
# ---------------------------------------------------------------------------

def download_audio(urls: list, raw_dir: str, work_dir: str, webhook: str):
    if stage_done(work_dir, "download"):
        print("[skip] download already complete")
        return

    if not urls:
        print("[skip] no URLs provided — skipping download stage")
        mark_stage_done(work_dir, "download")
        return

    os.makedirs(raw_dir, exist_ok=True)

    for i, url in enumerate(urls, 1):
        discord_post(webhook, f"⬇️ Downloading source {i}/{len(urls)}: {url}")
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--retries", "5",
            "--fragment-retries", "5",
            "-o", os.path.join(raw_dir, "%(id)s.%(ext)s"),
            url,
        ]

        for attempt in range(3):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            print(f"[warn] yt-dlp attempt {attempt + 1} failed: {result.stderr[:200]}")
            time.sleep(5)
        else:
            raise RuntimeError(f"yt-dlp failed after 3 attempts for {url}:\n{result.stderr}")

        print(result.stdout)

    mark_stage_done(work_dir, "download")


# ---------------------------------------------------------------------------
# Step 2: Demucs vocal separation
# ---------------------------------------------------------------------------

def separate_vocals(raw_dir: str, clean_dir: str, work_dir: str, webhook: str):
    if stage_done(work_dir, "vocals"):
        print("[skip] vocal separation already complete")
        return

    os.makedirs(clean_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*.wav"))
    if not raw_files:
        raise RuntimeError(f"No WAV files found in {raw_dir}")

    discord_post(
        webhook,
        f"🎵 **Vocal separation** — isolating voice from {len(raw_files)} file(s) with demucs. (~5-20 min)"
    )

    succeeded, failed = 0, 0
    for raw_path in raw_files:
        base = os.path.splitext(os.path.basename(raw_path))[0]
        dest = os.path.join(clean_dir, os.path.basename(raw_path))

        if os.path.exists(dest):
            print(f"[skip] {base} already separated")
            succeeded += 1
            continue

        cmd = [
            "python3", "-m", "demucs",
            "--two-stems=vocals",
            "--device", "cuda",
            "--out", clean_dir,
            raw_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        vocals_path = os.path.join(clean_dir, "htdemucs", base, "vocals.wav")

        if result.returncode == 0 and os.path.exists(vocals_path):
            shutil.move(vocals_path, dest)
            succeeded += 1
        else:
            print(f"[warn] demucs failed for {base}, using raw audio")
            shutil.copy2(raw_path, dest)
            failed += 1

    htdemucs_dir = os.path.join(clean_dir, "htdemucs")
    if os.path.exists(htdemucs_dir):
        shutil.rmtree(htdemucs_dir, ignore_errors=True)

    discord_post(webhook, f"🎵 Vocal separation done — {succeeded} cleaned, {failed} raw fallback.")
    mark_stage_done(work_dir, "vocals")


# ---------------------------------------------------------------------------
# Step 3: Convert + segment audio into clips
# ---------------------------------------------------------------------------

def process_audio(raw_dir: str, clips_dir: str, work_dir: str, webhook: str):
    if stage_done(work_dir, "segment"):
        total_clips = len(glob.glob(os.path.join(clips_dir, "*.wav")))
        print(f"[skip] segmentation already complete ({total_clips} clips)")
        return total_clips

    os.makedirs(clips_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "*.wav"))
    if not raw_files:
        raise RuntimeError(f"No WAV files found in {raw_dir}")

    discord_post(webhook, f"🔊 Processing {len(raw_files)} file(s): resampling, silence removal, segmenting...")

    clip_index = 0
    for raw_path in raw_files:
        base = os.path.splitext(os.path.basename(raw_path))[0]
        converted = os.path.join(raw_dir, f"{base}_converted.wav")

        if not os.path.exists(converted):
            cmd_convert = [
                "ffmpeg", "-y", "-i", raw_path,
                "-af",
                (
                    "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-50dB"
                    ":stop_periods=-1:stop_silence=0.3:stop_threshold=-50dB,"
                    "aresample=22050"
                ),
                "-ac", "1", "-ar", "22050",
                converted,
            ]
            result = subprocess.run(cmd_convert, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[warn] ffmpeg convert failed for {raw_path}: {result.stderr[:200]}")
                continue

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", converted],
            capture_output=True, text=True
        )
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            continue

        # 8s clips train faster, use far less GPU memory, and align better than
        # the old blind 15s cut (which ran 500-700 phonemes/clip). Auto-applied to
        # every future upload — no manual tuning needed.
        segment_len = 8.0
        num_segments = math.ceil(duration / segment_len)

        for seg in range(num_segments):
            start = seg * segment_len
            out_clip = os.path.join(clips_dir, f"clip_{clip_index:05d}.wav")

            if not os.path.exists(out_clip):
                cmd_seg = [
                    "ffmpeg", "-y",
                    "-ss", str(start), "-t", str(segment_len),
                    "-i", converted,
                    "-ar", "22050", "-ac", "1",
                    out_clip,
                ]
                subprocess.run(cmd_seg, capture_output=True, text=True)

            try:
                probe2 = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", out_clip],
                    capture_output=True, text=True
                )
                clip_dur = float(json.loads(probe2.stdout)["format"]["duration"])
                if clip_dur < 1.0:
                    os.remove(out_clip)
                    continue
            except Exception:
                pass

            clip_index += 1

    total_clips = len(glob.glob(os.path.join(clips_dir, "*.wav")))
    if total_clips == 0:
        raise RuntimeError("No usable audio clips after processing. Check source audio.")

    discord_post(webhook, f"📁 {total_clips} audio clips ready.")
    mark_stage_done(work_dir, "segment")
    return total_clips


# ---------------------------------------------------------------------------
# Step 4: Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_clips(clips_dir: str, dataset_dir: str, voice_name: str, work_dir: str, webhook: str):
    if stage_done(work_dir, "transcribe"):
        print("[skip] transcription already complete")
        return os.path.join(dataset_dir, "metadata.csv")

    wavs_dir = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(clips_dir, "*.wav")))
    discord_post(webhook, f"📝 Transcribing {len(clips)} clips with Whisper on GPU...")

    whisper_cache = "/workspace/whisper_cache"
    os.makedirs(whisper_cache, exist_ok=True)

    try:
        import torch as _wtorch
        if not _wtorch.cuda.is_available():
            raise RuntimeError("CUDA not available — Whisper must run on GPU (never CPU)")
    except Exception as _exc:
        if "CUDA not available" in str(_exc):
            raise
        raise RuntimeError(f"CUDA check failed for Whisper: {_exc}") from _exc

    for attempt in range(2):
        try:
            model = whisper.load_model("base", device="cuda", download_root=whisper_cache)
            break
        except Exception as exc:
            if attempt == 0 and "checksum" in str(exc).lower():
                print("[warn] Whisper model checksum failed — clearing cache and retrying")
                for f in glob.glob(os.path.join(whisper_cache, "*.pt")):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            else:
                raise

    metadata_path = os.path.join(dataset_dir, "metadata.csv")

    existing = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                existing.add(line.split("|")[0])

    written = len(existing)
    with open(metadata_path, "a", encoding="utf-8") as meta_f:
        for i, clip_path in enumerate(clips):
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            if clip_name in existing:
                continue

            dest_wav = os.path.join(wavs_dir, f"{clip_name}.wav")
            if not os.path.exists(dest_wav):
                shutil.copy2(clip_path, dest_wav)

            try:
                result = model.transcribe(clip_path, language="en")
                text = result["text"].strip()
                if not text:
                    continue
                meta_f.write(f"{clip_name}|{text}\n")
                meta_f.flush()
                written += 1
            except Exception as exc:
                print(f"[warn] Whisper failed on {clip_path}: {exc}")

            if (i + 1) % 50 == 0:
                print(f"  Transcribed {i + 1}/{len(clips)} clips...")

    if written == 0:
        raise RuntimeError("Whisper produced no transcriptions. Check audio quality.")

    discord_post(webhook, f"📝 Transcription complete — {written} clips.")
    mark_stage_done(work_dir, "transcribe")
    return metadata_path


# ---------------------------------------------------------------------------
# Step 5: piper_train.preprocess  (Phase 1 + 2 + 3 + 4 resilience)
# ---------------------------------------------------------------------------

def preprocess_dataset(dataset_dir: str, preprocessed_dir: str, work_dir: str, webhook: str) -> str:
    """Run piper_train.preprocess with full resilience.

    Returns training_mode string (NORMAL|DEGRADED|MINIMAL|SURVIVAL).

    Resilience layers:
      Phase 4 — normalize all wavs to mono 22050 Hz PCM before preprocess
      Phase 1 — full dataset intelligence: metrics + score + classification
      Phase 2 — capture stderr; on ValueError('n must be at least one') or size trigger:
                   pad dataset to MINIMAL_DATASET_TARGET, retry with --max-workers 1
      Phase 3 — training mode carried forward to train() for batch_size selection
    """
    if stage_done(work_dir, "preprocess"):
        print("[skip] preprocessing already complete")
        # Read existing training_mode from state if available
        existing = load_state(work_dir).get("dataset_health", {})
        return existing.get("training_mode", TRAIN_MODE_NORMAL)

    os.makedirs(preprocessed_dir, exist_ok=True)

    # Phase 4: normalize wavs BEFORE anything else
    normalize_wavs_for_preprocess(dataset_dir)

    # Phase 1: full dataset intelligence
    health        = validate_dataset(dataset_dir, work_dir)
    training_mode = health["training_mode"]
    needs_padding = training_mode in (TRAIN_MODE_MINIMAL, TRAIN_MODE_SURVIVAL)
    validation_errors = enforce_preprocess_requirements(health)

    total_clips = health["total_clips"]
    mode_label  = f"⚠️ {training_mode}" if training_mode != TRAIN_MODE_NORMAL else "NORMAL"
    quality_str = health["expected_output_quality"]

    discord_post(
        webhook,
        f"🔧 Preprocessing dataset — {total_clips} clips | score={health['score']} ({health['health_class']}) | "
        f"mode={mode_label} | expected={quality_str}"
    )

    if total_clips == 0:
        raise RuntimeError(
            "preprocess aborted: dataset/wavs/ is empty. "
            "No wavs found to process — check that data preparation completed successfully."
        )

    if validation_errors:
        health["validation_ok"] = False
        health["validation_errors"] = validation_errors
        _update_dataset_health(work_dir, health)
        reason_text = "; ".join(validation_errors)
        discord_post(
            webhook,
            f"❌ Dataset rejected before preprocess for `{os.path.basename(work_dir)}`: {reason_text}"
        )
        raise RuntimeError(f"dataset validation failed before preprocess: {reason_text}")

    health["validation_ok"] = True
    health["validation_errors"] = []
    _update_dataset_health(work_dir, health)

    # If SURVIVAL/MINIMAL → pre-pad now (before first preprocess attempt)
    if needs_padding:
        pre_pad_count = _pad_dataset_to_minimum(dataset_dir, MINIMAL_DATASET_TARGET)
        health["total_clips"] = pre_pad_count
        health["padded"] = True
        _update_dataset_health(work_dir, health)
        discord_post(
            webhook,
            f"⚠️ {training_mode} mode — padded dataset to {pre_pad_count} samples, using batch_size=1."
        )

    # Compute worker count. phonemizer/espeak deadlocks with many workers inside
    # Docker (workers sit on futex forever, dataset.jsonl stays 0 bytes). Always 1.
    max_workers = 1

    if _force_preprocess_failure_requested(work_dir):
        msg = (
            "controlled preprocess failure probe triggered after dataset validation "
            "and before piper_train.preprocess invocation"
        )
        print(f"[failprobe] {msg}", file=sys.stderr)
        discord_post(webhook, f"🧪 Controlled failure probe: {msg}.")
        raise RuntimeError(msg)

    def _run_preprocess(workers: int) -> subprocess.CompletedProcess:
        cmd = [
            "python3", "-m", "piper_train.preprocess",
            "--language", "en-us",
            "--input-dir", dataset_dir,
            "--output-dir", preprocessed_dir,
            "--dataset-format", "ljspeech",
            "--single-speaker",
            "--sample-rate", "22050",
            "--max-workers", str(workers),
        ]
        return subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    # First attempt
    result = _run_preprocess(max_workers)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Phase 2: preprocess guard — catch ValueError + emergency minimal mode recovery
    if result.returncode != 0:
        err_text = result.stderr or ""
        size_error = (
            "n must be at least one" in err_text
            or ("sample" in err_text.lower() and "empty" in err_text.lower())
        )

        if size_error or needs_padding:
            print(
                "[preprocess] ⚠️  Preprocess failed — "
                "activating emergency padding and retry with max-workers 1"
            )
            discord_post(
                webhook,
                f"⚠️ Preprocess failed (dataset too small). "
                f"Padding to {MINIMAL_DATASET_TARGET} samples and retrying."
            )

            new_count = _pad_dataset_to_minimum(dataset_dir, MINIMAL_DATASET_TARGET)
            health["total_clips"] = new_count
            health["padded"]      = True
            if training_mode == TRAIN_MODE_NORMAL:
                health["training_mode"] = TRAIN_MODE_MINIMAL
                training_mode = TRAIN_MODE_MINIMAL
            _update_dataset_health(work_dir, health)

            # Clear preprocessed_dir before retry
            for item in os.listdir(preprocessed_dir):
                item_path = os.path.join(preprocessed_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as exc:
                    print(f"[preprocess] could not clear {item_path}: {exc}")

            print(f"[preprocess] Retrying with {new_count} samples, --max-workers 1")
            result = _run_preprocess(1)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

            if result.returncode != 0:
                raise RuntimeError(
                    f"preprocess failed even after emergency padding to {new_count} samples.\n"
                    f"stderr tail:\n{result.stderr[-600:]}"
                )

            discord_post(
                webhook,
                f"✅ Preprocessing recovered — {new_count} samples (padded), mode={training_mode}."
            )
        else:
            raise RuntimeError(
                f"preprocess failed.\nstderr tail:\n{err_text[-600:]}"
            )

    discord_post(webhook, "🔧 Preprocessing complete.")
    mark_stage_done(work_dir, "preprocess")
    return training_mode


# ---------------------------------------------------------------------------
# Step 6: Download lessac checkpoint
# ---------------------------------------------------------------------------

def ensure_checkpoint(checkpoint_dir: str, webhook: str, base_voice: str = None) -> str:
    """
    Return the base checkpoint path to start training from.
    If base_voice is set and a matching .ckpt exists in /checkpoints, use it.
    Otherwise fall back to the default lessac checkpoint (downloading if needed).
    """
    # Try user-specified base voice checkpoint
    if base_voice:
        for pattern in [
            os.path.join(checkpoint_dir, f"{base_voice}*.ckpt"),
            os.path.join(checkpoint_dir, f"{base_voice}", "**", "*.ckpt"),
            os.path.join("/workspace/voices", f"{base_voice}", "**", "*.ckpt"),
            os.path.join("/workspace/voices", f"{base_voice}_*", "**", "*.ckpt"),
        ]:
            matches = glob.glob(pattern, recursive=True)
            matches = [m for m in matches if "lessac" not in os.path.basename(m).lower()]
            if matches:
                matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                print(f"[checkpoint] Using base voice '{base_voice}': {matches[0]}")
                discord_post(webhook, f"🎯 Using base voice `{base_voice}` checkpoint as training starting point.")
                return matches[0]
        print(f"[checkpoint] Base voice '{base_voice}' not found — falling back to lessac")

    ckpt_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    if os.path.exists(ckpt_path):
        print(f"[checkpoint] Using cached: {ckpt_path}")
        return ckpt_path

    discord_post(webhook, "📥 Downloading lessac checkpoint (~400MB)...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with requests.get(CHECKPOINT_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(ckpt_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (50 * 1024 * 1024) < 8192:
                    pct = int(downloaded / total * 100)
                    print(f"  {pct}% ({downloaded // (1024 * 1024)}MB / {total // (1024 * 1024)}MB)")

    discord_post(webhook, "📥 Checkpoint downloaded.")
    return ckpt_path


# ---------------------------------------------------------------------------
# Step 7: GPU training — crash-safe, resumes from last.ckpt
# ---------------------------------------------------------------------------

def find_resume_checkpoint(train_dir: str):
    """Find the most recent saved checkpoint to resume from."""
    for pattern in ["**/last.ckpt", "**/*.ckpt"]:
        ckpts = glob.glob(os.path.join(train_dir, pattern), recursive=True)
        ckpts = [c for c in ckpts if "lessac" not in os.path.basename(c).lower()]
        if ckpts:
            ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return ckpts[0]
    return None


def evaluate_convergence(train_dir: str, window_epochs: int = PLATEAU_WINDOW_EPOCHS,
                          threshold_pct: float = PLATEAU_THRESHOLD_PCT,
                          min_epochs_seen: int = PLATEAU_MIN_EPOCHS_SEEN) -> dict:
    """Evidence-based convergence check using Piper's own documented signal:
    'the model is done when loss_disc_all levels off' (upstream TRAINING.md).

    Compares the mean of the most recent `window_epochs`-worth of
    loss_disc_all steps against the mean of the window immediately before
    it. Returns a data dict -- this is advisory evidence, not a decision by
    itself; callers decide whether/how to act on `plateaued`.

    Read-only: only inspects the tfevents file piper_train is already
    writing. Safe to call while training is running.
    """
    result = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plateaued": False, "confidence": "insufficient_data",
        "loss_disc_all": None, "loss_gen_all": None,
    }
    try:
        versions = sorted(glob.glob(os.path.join(train_dir, "lightning_logs", "version_*")))
        if not versions:
            return result
        ev_files = glob.glob(os.path.join(versions[-1], "events.out.tfevents.*"))
        if not ev_files:
            return result
        from tensorboard.backend.event_processing import event_file_loader
        import numpy as np
        loader = event_file_loader.EventFileLoader(ev_files[0])
        series = {}
        for event in loader.Load():
            if event.HasField("summary"):
                for v in event.summary.value:
                    if v.tag in ("loss_disc_all", "loss_gen_all", "epoch") and v.HasField("tensor"):
                        t = v.tensor
                        arr = (np.frombuffer(t.tensor_content, dtype=np.float32)
                               if t.tensor_content else np.array(t.float_val or t.double_val))
                        if len(arr):
                            series.setdefault(v.tag, []).append((event.step, float(arr[0])))

        epoch_series = series.get("epoch", [])
        if not epoch_series:
            return result
        epoch_span = epoch_series[-1][1] - epoch_series[0][1]
        result["epochs_of_data"] = epoch_span

        for tag in ("loss_disc_all", "loss_gen_all"):
            vals = [v for _, v in series.get(tag, [])]
            if len(vals) < 20:
                result[tag] = {"note": "too few samples"}
                continue
            n_window = max(10, len(vals) * window_epochs // max(1, int(epoch_span) or 1))
            n_window = min(n_window, len(vals) // 2)
            recent = vals[-n_window:]
            prior = vals[-2 * n_window:-n_window] if len(vals) >= 2 * n_window else None
            entry = {"recent_mean": statistics.mean(recent), "recent_stdev": statistics.stdev(recent) if len(recent) > 1 else 0.0}
            if prior:
                prior_mean = statistics.mean(prior)
                pct_change = abs(entry["recent_mean"] - prior_mean) / abs(prior_mean) * 100 if prior_mean else None
                entry["prior_mean"] = prior_mean
                entry["pct_change"] = pct_change
            result[tag] = entry

        if epoch_span < min_epochs_seen:
            result["confidence"] = "insufficient_epochs"
        else:
            disc = result.get("loss_disc_all", {})
            if "pct_change" in disc and disc["pct_change"] is not None:
                result["plateaued"] = disc["pct_change"] < threshold_pct
                result["confidence"] = "based_on_loss_disc_all_trend"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def train(preprocessed_dir: str, checkpoint_path: str, train_dir: str, webhook: str,
          epochs: int = DEFAULT_ADDITIONAL_EPOCHS, training_mode: str = TRAIN_MODE_NORMAL,
          num_workers: int = DEFAULT_NUM_WORKERS, precision: str = DEFAULT_PRECISION,
          batch_size_override: int | None = None,
          checkpoint_epochs: int = DEFAULT_CHECKPOINT_EPOCHS,
          auto_stop_on_plateau: bool = AUTO_STOP_ON_PLATEAU_DEFAULT):
    """Run piper_train on GPU only (CUDA required).

    training_mode: NORMAL|DEGRADED → batch_size=DEFAULT_BATCH_SIZE_NORMAL (4)
                   MINIMAL|SURVIVAL → batch_size=DEFAULT_BATCH_SIZE_SMALL (1)
    batch_size_override: if set, wins over the training_mode-derived value
        (still logged so it's visible which one actually applied).

    num_workers / precision are new (2026-09-11) -- previously hardcoded to
    1 and FP32 respectively with no way to override. Defaults above
    preserve prior behavior exactly; nothing changes unless a caller passes
    a different value explicitly.

    auto_stop_on_plateau: when True, training requests a clean stop (same
    path as a SIGTERM -- current checkpoint is preserved) once
    evaluate_convergence() reports plateaued=True with real confidence.
    Defaults OFF platform-wide (see AUTO_STOP_ON_PLATEAU_DEFAULT) --
    convergence is always evaluated and logged to pipeline_state.json
    regardless of this flag, so the evidence is there even when nothing
    acts on it automatically.
    """
    work_dir = os.path.dirname(train_dir)

    if stage_done(work_dir, "train"):
        print("[skip] training already complete")
        return

    os.makedirs(train_dir, exist_ok=True)

    small_mode = training_mode in (TRAIN_MODE_MINIMAL, TRAIN_MODE_SURVIVAL)
    if batch_size_override is not None:
        batch_size = batch_size_override
        print(f"[train] batch_size overridden to {batch_size}")
    else:
        batch_size = DEFAULT_BATCH_SIZE_SMALL if small_mode else DEFAULT_BATCH_SIZE_NORMAL
    if small_mode:
        print(f"[train] {training_mode} mode — using batch_size={batch_size}")
    print(f"[train] engine config: num_workers={num_workers} precision={precision} "
          f"batch_size={batch_size} checkpoint_epochs={checkpoint_epochs} "
          f"auto_stop_on_plateau={auto_stop_on_plateau}")

    resume_ckpt = find_resume_checkpoint(train_dir)
    if resume_ckpt:
        discord_post(
            webhook,
            f"♻️ **Resuming GPU training** from checkpoint: `{os.path.basename(resume_ckpt)}`\n"
            f"({epochs:,} total additional epochs on CUDA)"
        )
        resume_arg = resume_ckpt
    else:
        mode_note = f" [{training_mode} mode — batch_size={batch_size}]" if training_mode != TRAIN_MODE_NORMAL else ""
        discord_post(
            webhook,
            f"🚀 **GPU training started** ({epochs:,} additional epochs){mode_note}. "
            "Progress updates every 500 epochs. CUDA only — never CPU."
        )
        resume_arg = checkpoint_path

    ckpt_epoch = 0
    try:
        import torch as _torch
        _ckpt = _torch.load(resume_arg, map_location="cpu")
        ckpt_epoch = int(_ckpt.get("epoch", 0))
    except Exception:
        pass

    max_epochs = ckpt_epoch + epochs

    # GPU-ONLY: launcher must pass --gpus all with piper-voice-trainer:gpu.
    # Never fall back to CPU — that silently burns days and violates the ops rule.
    try:
        import torch as _tgpu
        _use_gpu = _tgpu.cuda.is_available()
    except Exception as _exc:
        raise RuntimeError(f"CUDA check failed — refusing CPU training: {_exc}") from _exc
    if not _use_gpu:
        raise RuntimeError(
            "CUDA not visible inside trainer container — refusing CPU training. "
            "Use piper-voice-trainer:gpu with docker --gpus all."
        )
    _accel_args = ["--accelerator", "gpu", "--devices", "1"]
    print("[pipeline] training accelerator: GPU (cuda) — CPU forbidden", flush=True)

    cmd = [
        "python3", "-m", "piper_train",
        "--dataset-dir", preprocessed_dir,
        *_accel_args,
        "--batch-size", str(batch_size),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(max_epochs),
        "--resume_from_checkpoint", resume_arg,
        "--checkpoint-epochs", str(checkpoint_epochs),
        "--default_root_dir", train_dir,
        # 15s segmented clips run 500-700 phonemes; a 400 cap silently skipped
        # ~all of them → empty dataset → no .onnx. 800 keeps long clips trainable.
        "--max-phoneme-ids", "800",
        "--log_every_n_steps", "1",
        "--precision", str(precision),
        "--num-workers", str(num_workers),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    plateau_stop_requested = {"flag": False}

    def handle_signal(sig, frame):
        print(f"[signal] Received {sig} — training will save checkpoint and exit cleanly")
        proc.terminate()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    last_notify = time.time()
    last_epoch = 0
    last_convergence_check = time.time()
    CONVERGENCE_CHECK_INTERVAL_SEC = 300  # every 5 min of wall clock, cheap read-only tfevents parse

    for line in proc.stdout:
        print(line, end="")

        if "Epoch" in line and "/" in line:
            try:
                for part in line.split():
                    if "/" in part:
                        epoch = int(part.split("/")[0])
                        if epoch != last_epoch and epoch % 500 == 0 and time.time() - last_notify > 300:
                            discord_post(webhook, f"⚙️ GPU training: epoch {epoch}")
                            last_notify = time.time()
                            last_epoch = epoch
                        break
            except Exception:
                pass

        # Evidence-based convergence check -- always evaluated & logged
        # (see evaluate_convergence docstring). Read-only against the same
        # tfevents file piper_train is writing; never touches the subprocess.
        if time.time() - last_convergence_check > CONVERGENCE_CHECK_INTERVAL_SEC:
            last_convergence_check = time.time()
            try:
                conv = evaluate_convergence(train_dir)
                state = load_state(work_dir)
                state.setdefault("convergence_history", []).append(conv)
                state["convergence_history"] = state["convergence_history"][-100:]
                _write_state(work_dir, state)
                if conv.get("plateaued"):
                    print(f"[convergence] loss_disc_all plateaued (pct_change="
                          f"{conv.get('loss_disc_all', {}).get('pct_change')}, "
                          f"confidence={conv.get('confidence')})")
                    if auto_stop_on_plateau and not plateau_stop_requested["flag"]:
                        plateau_stop_requested["flag"] = True
                        print("[convergence] auto_stop_on_plateau is ON -- requesting clean stop "
                              "(current checkpoint is preserved; same path as a manual SIGTERM)")
                        discord_post(webhook, "📉 **Convergence plateau detected** (`loss_disc_all` leveled off) "
                                               "— stopping cleanly per auto_stop_on_plateau. Latest checkpoint preserved.")
                        proc.terminate()
            except Exception as exc:
                print(f"[convergence] evaluation failed (non-fatal): {exc}")

    proc.wait()

    if proc.returncode not in (0, -15, 1):
        raise RuntimeError(f"piper_train exited with code {proc.returncode}")

    if proc.returncode == 0:
        discord_post(webhook, "🏁 Training complete!")
        mark_stage_done(work_dir, "train")
        state = load_state(work_dir)
        state["training_outcome"] = {
            "status": "train_complete",
            "stop_reason": "max_epochs_reached",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _write_state(work_dir, state)
    elif plateau_stop_requested["flag"]:
        # Clean convergence stop — NOT a failure. Checkpoint preserved; ONNX
        # export is deliberately deferred so an operator can perceptual-check
        # before promoting. Queue sync must map this to converged_pending_export
        # (never "failed / no .onnx").
        latest_ckpt = None
        try:
            latest_ckpt = find_best_checkpoint(train_dir)
        except Exception:
            latest_ckpt = None
        state = load_state(work_dir)
        state["training_outcome"] = {
            "status": "converged_pending_export",
            "stop_reason": "plateau_converged",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": os.path.basename(latest_ckpt) if latest_ckpt else None,
            "checkpoint_path": latest_ckpt,
            "auto_stop_on_plateau": True,
            "note": "Clean convergence stop; ONNX export deferred pending review. Not a training failure.",
        }
        _write_state(work_dir, state)
        discord_post(webhook, "📉 Training stopped at convergence plateau — checkpoint preserved "
                               "(status=`converged_pending_export`). Ready to export or extend after review.")
        print("[convergence] wrote training_outcome=converged_pending_export to pipeline_state.json")
    else:
        state = load_state(work_dir)
        # Preserve an explicit pause marker if one was already set (e.g. benchmark);
        # otherwise record a generic clean pause so queue sync does not call it failed.
        existing = (state.get("training_outcome") or {}).get("status")
        if existing not in ("paused_for_benchmark", "converged_pending_export"):
            latest_ckpt = None
            try:
                latest_ckpt = find_best_checkpoint(train_dir)
            except Exception:
                latest_ckpt = None
            state["training_outcome"] = {
                "status": "paused_pending_resume",
                "stop_reason": "signal_or_external_stop",
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "checkpoint": os.path.basename(latest_ckpt) if latest_ckpt else None,
                "checkpoint_path": latest_ckpt,
                "note": "Clean pause; checkpoint preserved. Not a training failure.",
            }
            _write_state(work_dir, state)
        discord_post(webhook, "⏸️ Training paused — will resume from last checkpoint on restart.")


# ---------------------------------------------------------------------------
# Step 8: Find best checkpoint
# ---------------------------------------------------------------------------

def find_best_checkpoint(train_dir: str) -> str:
    pattern = os.path.join(train_dir, "**", "*.ckpt")
    ckpts = glob.glob(pattern, recursive=True)
    ckpts = [c for c in ckpts if "lessac" not in os.path.basename(c).lower()]

    if not ckpts:
        raise RuntimeError(f"No trained checkpoint found in {train_dir}")

    last = [c for c in ckpts if "last" in os.path.basename(c).lower()]
    if last:
        return last[0]

    ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return ckpts[0]


# ---------------------------------------------------------------------------
# Step 9: Export ONNX
# ---------------------------------------------------------------------------

def export_onnx(checkpoint_path: str, output_dir: str, model_name: str, work_dir: str, webhook: str):
    if stage_done(work_dir, "export"):
        print("[skip] ONNX export already complete")
        return (
            os.path.join(output_dir, f"{model_name}.onnx"),
            os.path.join(output_dir, f"{model_name}.onnx.json"),
        )

    os.makedirs(output_dir, exist_ok=True)
    discord_post(webhook, "🔧 Exporting ONNX model...")

    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    json_path = onnx_path + ".json"

    # onnx is installed in the image; direct invocation avoids shell/hang issues
    result = subprocess.run(
        ["python3", "-m", "piper_train.export_onnx", checkpoint_path, onnx_path],
        capture_output=True, text=True
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"ONNX export failed (exit {result.returncode}):\n{result.stderr[-500:]}")

    if not os.path.exists(onnx_path):
        raise RuntimeError(f"ONNX export claimed success but file missing: {onnx_path}")

    if os.path.getsize(onnx_path) < 5 * 1024 * 1024:
        raise RuntimeError(f"ONNX file suspiciously small ({os.path.getsize(onnx_path)} bytes) — likely corrupt")

    if not os.path.exists(json_path):
        src_config = os.path.join(work_dir, "preprocessed", "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, json_path)
            print(f"[export] Copied fallback config: {src_config} -> {json_path}")
        else:
            raise RuntimeError(
                f"ONNX exported but no JSON config found. Expected at: {src_config}"
            )

    discord_post(webhook, f"✅ ONNX export complete: `{model_name}.onnx` + `{model_name}.onnx.json`")
    mark_stage_done(work_dir, "export")
    return onnx_path, json_path


# ---------------------------------------------------------------------------
# Step 10: Deploy voice to runtime workspace
# ---------------------------------------------------------------------------

def deploy_voice(onnx_path: str, json_path: str, job_name: str, runtime_name: str,
                 work_dir: str, webhook: str):
    """Copy versioned ONNX to /workspace/{runtime_name}.onnx for Piper auto-discovery."""
    if stage_done(work_dir, "deploy"):
        print("[skip] deploy already complete")
        return

    runtime_root = "/workspace"
    dest_onnx = os.path.join(runtime_root, f"{runtime_name}.onnx")
    dest_json = os.path.join(runtime_root, f"{runtime_name}.onnx.json")

    shutil.copy2(onnx_path, dest_onnx)
    shutil.copy2(json_path, dest_json)
    print(f"[deploy] {job_name} → {dest_onnx}")

    discord_post(webhook, f"🚀 Voice deployed: `{runtime_name}` ready for selection in Piper TTS.")
    mark_stage_done(work_dir, "deploy")


# ---------------------------------------------------------------------------
# Step 11: Post-training artifacts — sample WAVs + voice_info.json
# ---------------------------------------------------------------------------

SAMPLE_SENTENCES = [
    "Hello, I am Otacon. All systems are operational.",
    "Snake, it's me. The mission is a go.",
    "Voice cloning complete. Ready for deployment.",
]

def generate_artifacts(onnx_path: str, output_dir: str, voice_name: str,
                       work_dir: str, webhook: str, base_voice: str = None):
    """Generate sample WAVs and voice_info.json for post-training review."""
    if stage_done(work_dir, "artifacts"):
        print("[skip] artifacts already complete")
        return

    artifacts_dir = os.path.join(output_dir, "samples")
    os.makedirs(artifacts_dir, exist_ok=True)
    generated = []

    try:
        for i, sentence in enumerate(SAMPLE_SENTENCES):
            wav_path = os.path.join(artifacts_dir, f"sample_{i+1:02d}.wav")
            result = subprocess.run(
                ["python3", "-m", "piper", "--model", onnx_path, "--output_file", wav_path],
                input=sentence.encode(),
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0 and os.path.exists(wav_path):
                size = os.path.getsize(wav_path)
                generated.append({"file": f"sample_{i+1:02d}.wav", "text": sentence, "size": size})
                print(f"[artifacts] sample {i+1}: {wav_path} ({size} bytes)")
            else:
                print(f"[artifacts] sample {i+1} failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"[artifacts] WAV generation error: {e}")

    # Write voice_info.json — includes dataset health for UI feedback (Phase 8)
    onnx_size = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
    state     = load_state(work_dir)
    info = {
        "voice_name":     voice_name,
        "onnx_path":      onnx_path,
        "onnx_size_mb":   round(onnx_size / (1024 * 1024), 1),
        "base_voice":     base_voice or "lessac",
        "samples":        generated,
        "dataset_health": state.get("dataset_health", {}),
        "created_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    info_path = os.path.join(output_dir, "voice_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[artifacts] voice_info.json written: {info_path}")

    if generated:
        discord_post(
            webhook,
            f"🎤 **Post-training samples generated** for `{voice_name}`\n"
            f"{len(generated)} sample WAVs in `{artifacts_dir}`\n"
            f"ONNX size: {info['onnx_size_mb']}MB"
        )
    mark_stage_done(work_dir, "artifacts")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Piper TTS voice cloning pipeline")
    parser.add_argument("--name", required=True)
    # nargs='*' so dataset-mode jobs (which pre-mark download as done) can pass
    # placeholder URLs without --urls being required=True at the OS level.
    parser.add_argument("--urls", nargs="*", default=[])
    parser.add_argument("--discord-webhook", default="")
    parser.add_argument("--checkpoint-dir", default="/checkpoints")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=DEFAULT_ADDITIONAL_EPOCHS,
                        help=f"Additional epochs beyond the resume checkpoint (default: {DEFAULT_ADDITIONAL_EPOCHS}, "
                             f"per Piper's own fine-tuning guidance -- see DEFAULT_ADDITIONAL_EPOCHS)")
    parser.add_argument("--base-voice", default="",
                        help="Base voice checkpoint to start training from (e.g. mei_ling)")
    parser.add_argument("--adaptive", action="store_true", default=False,
                        help="Run adaptive ASSESS→CORRECT→REASSESS→DECIDE loop before preprocessing")
    parser.add_argument("--prune-pct", type=float, default=PRUNE_PCT_DEFAULT,
                        help="Percentage of bottom-scored clips to prune in adaptive mode")
    # 2026-09-11: engine controls that were previously hardcoded (num_workers=1,
    # FP32, batch_size derived only from training_mode). Defaults below are
    # IDENTICAL to prior behavior -- nothing changes for existing callers that
    # don't pass these. See DEFAULT_NUM_WORKERS/DEFAULT_PRECISION docstring
    # block near the top of this file for why, and benchmark_configs.py for
    # the measurement harness that should inform ever changing these defaults.
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"DataLoader worker processes (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--train-precision", default=DEFAULT_PRECISION,
                        help=f"piper_train --precision passthrough: 32/16/bf16/64 (default: {DEFAULT_PRECISION})")
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="Override the training_mode-derived batch size entirely")
    parser.add_argument("--checkpoint-epochs", type=int, default=DEFAULT_CHECKPOINT_EPOCHS,
                        help=f"Save a checkpoint every N epochs (default: {DEFAULT_CHECKPOINT_EPOCHS})")
    parser.add_argument("--auto-stop-on-plateau", action="store_true", default=AUTO_STOP_ON_PLATEAU_DEFAULT,
                        help="Stop cleanly (checkpoint preserved) once loss_disc_all plateaus. "
                             "Convergence is always evaluated/logged regardless of this flag; "
                             "this only controls whether anything acts on it automatically. Default: off.")
    args = parser.parse_args()

    name    = args.name
    webhook = args.discord_webhook
    work_dir = f"/workspace/jobs/{name}"

    raw_dir        = os.path.join(work_dir, "raw")
    clean_dir      = os.path.join(work_dir, "clean")
    clips_dir      = os.path.join(work_dir, "clips")
    dataset_dir    = os.path.join(work_dir, "dataset")
    preprocessed_dir = os.path.join(work_dir, "preprocessed")
    train_dir      = os.path.join(work_dir, "training")

    os.makedirs(work_dir, exist_ok=True)
    _clear_runtime_failure(work_dir)

    state     = load_state(work_dir)
    is_resume = bool(state.get("completed_stages"))

    # Detect dataset mode: ingest stages pre-marked by executor
    ingest_stages = {"download", "vocals", "segment", "transcribe"}
    pre_marked     = ingest_stages.issubset(set(state.get("completed_stages", [])))
    train_mode     = "existing_dataset" if pre_marked else "youtube"

    try:
        if is_resume:
            discord_post(
                webhook,
                f"♻️ **Otacon Genome Project** — Resuming `{name}` pipeline\n"
                f"Completed stages: {', '.join(state['completed_stages'])}"
            )
        else:
            if train_mode == "existing_dataset":
                discord_post(
                    webhook,
                    f"🧬 **Otacon Genome Project** — Dataset retrain started for `{name}`\n"
                    f"Mode: existing curated dataset (skipping ingest)\n"
                    f"Pipeline: preprocess → train → export"
                )
            else:
                discord_post(
                    webhook,
                    f"🧬 **Otacon Genome Project** — Voice cloning started for `{name}`\n"
                    f"Sources: {len(args.urls)} YouTube URL(s)\n"
                    f"Pipeline: download → denoise → segment → transcribe → train → export"
                )

        download_audio(args.urls, raw_dir, work_dir, webhook)
        separate_vocals(raw_dir, clean_dir, work_dir, webhook)
        process_audio(clean_dir, clips_dir, work_dir, webhook)
        transcribe_clips(clips_dir, dataset_dir, name, work_dir, webhook)

        # Adaptive mode: ASSESS → CORRECT → REASSESS → DECIDE before preprocessing
        if args.adaptive and not stage_done(work_dir, 'preprocess'):
            existing_adaptive = load_state(work_dir).get('adaptive', {})
            if existing_adaptive.get('enabled'):
                print('[adaptive] already ran — using existing results')
                _adaptive_training_mode = existing_adaptive.get('training_mode', TRAIN_MODE_NORMAL)
                _adaptive_decision      = existing_adaptive.get('retrain_decision', 'SKIP')
            else:
                _adaptive_training_mode, _adaptive_decision, _adaptive_summary = adaptive_improve_loop(
                    dataset_dir, work_dir, webhook, prune_pct=args.prune_pct
                )
                # Store training_mode in adaptive state for resume
                _state = load_state(work_dir)
                if _state.get('adaptive'):
                    _state['adaptive']['training_mode'] = _adaptive_training_mode
                    _write_state(work_dir, _state)

        # preprocess_dataset returns training_mode string
        training_mode = preprocess_dataset(dataset_dir, preprocessed_dir, work_dir, webhook)

        # Log training mode selection with dataset context
        state  = load_state(work_dir)
        health = state.get("dataset_health", {})
        if health:
            quality = health.get("expected_output_quality", "unknown")
            score   = health.get("score", "?")
            hclass  = health.get("health_class", "?")
            dup     = health.get("duplicate_ratio", 0.0)
            print(
                f"[train] Starting in mode={training_mode} | "
                f"dataset score={score} ({hclass}) | "
                f"expected_quality={quality} | "
                f"dup={dup:.0%}"
            )
            if training_mode in (TRAIN_MODE_SURVIVAL, TRAIN_MODE_MINIMAL):
                discord_post(
                    webhook,
                    f"⚠️ **Training mode: {training_mode}** — "
                    f"dataset score {score}/100 ({hclass}), expected output quality: **{quality}**\n"
                    f"Duplicate ratio: {dup:.0%} | {health.get('valid_clips','?')} valid clips"
                )

        base_voice = (args.base_voice or '').strip() or None
        checkpoint_path = ensure_checkpoint(args.checkpoint_dir, webhook, base_voice=base_voice)
        train(preprocessed_dir, checkpoint_path, train_dir, webhook, args.epochs,
              training_mode=training_mode,
              num_workers=args.num_workers, precision=args.train_precision,
              batch_size_override=args.train_batch_size,
              checkpoint_epochs=args.checkpoint_epochs,
              auto_stop_on_plateau=args.auto_stop_on_plateau)

        if stage_done(work_dir, "train"):
            best_ckpt = find_best_checkpoint(train_dir)
            final_output_dir = args.output_dir.strip() if args.output_dir.strip() else work_dir
            onnx_path, json_path = export_onnx(best_ckpt, final_output_dir, name, work_dir, webhook)

            # Deploy to /workspace/{runtime_name} for auto-discovery by Piper
            # runtime_name strips version suffix: mei_ling_v4 → mei_ling
            runtime_name = re.sub(r'_v\d+$', '', name)
            deploy_voice(onnx_path, json_path, name, runtime_name, work_dir, webhook)

            # P7: Generate sample WAVs + voice_info.json for review
            generate_artifacts(onnx_path, final_output_dir, name, work_dir, webhook, base_voice=base_voice)

            discord_post(
                webhook,
                f"🎉 **Otacon Genome Project — `{name}` voice is ready!**\n"
                f"Versioned: `{final_output_dir}/{name}.onnx`\n"
                f"Runtime alias: `/workspace/{runtime_name}.onnx` (select as `{runtime_name}` in Piper)"
            )
        else:
            outcome = (load_state(work_dir).get("training_outcome") or {})
            status = outcome.get("status") or "paused_pending_resume"
            ckpt_name = outcome.get("checkpoint") or "latest checkpoint"
            if status == "converged_pending_export":
                discord_post(
                    webhook,
                    f"📉 **`{name}` converged** — status=`converged_pending_export`.\n"
                    f"Checkpoint preserved (`{ckpt_name}`). Export/deploy deferred pending perceptual review."
                )
            elif status == "paused_for_benchmark":
                discord_post(
                    webhook,
                    f"🧪 **`{name}` paused for benchmark** — checkpoint preserved (`{ckpt_name}`)."
                )
            else:
                discord_post(
                    webhook,
                    f"⏸️ **`{name}` training paused** — restart the container to continue from last checkpoint."
                )

    except Exception as exc:
        import traceback
        _record_runtime_failure(work_dir, exc)
        traceback.print_exc()
        discord_post(webhook, f"❌ **Otacon Genome Project failed** for `{name}`:\n```{exc}```")
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
