#!/usr/bin/env python3
"""GPU training-config benchmark harness -- Piper/piper_train on the RTX 3090.

DO NOT RUN THIS WHILE ANY OTHER GPU JOB (training, ComfyUI, etc.) IS ACTIVE.
Each benchmark launches a real `docker run --gpus all` piper_train process
and will fully compete for the 3090. Written 2026-09-11 while sniper_wolf_v1
was actively training -- deliberately NOT executed until that job exits and
`nvidia-smi --query-compute-apps` shows the GPU genuinely idle.

Why this exists: sniper_wolf_v1 measured ~48% GPU utilization with the
platform's current hardcoded config (workers=1, batch=4, FP32). Diagnosis
pointed at a CPU-side dataloader bottleneck (num_workers hardcoded to 1,
two host cores pegged near 100%) plus a very conservative batch size
(piper's own TRAINING.md says batch-size 32 is the documented reference
for a 24GB 3090/4090; this platform uses 4). Before changing any
production default, gather real short-run measurements across candidate
configs and compare -- no speculative production changes until this has run.

Sweep (per explicit user request, 2026-09-11):
    1. workers=1 / batch=4  / FP32   (current production baseline)
    2. workers=4 / batch=4  / FP32
    3. workers=8 / batch=4  / FP32
    4. workers=8 / batch=16 / FP32
    5. workers=8 / batch=32 / FP32   -- ONLY if a VRAM safety check after
       config 4 projects it will fit with headroom; skipped otherwise
    6. best-performing FP32 config above, re-run with FP16/AMP           -- chosen
       dynamically from the actual measurements, not guessed in advance

Each config runs for a fixed WALL-CLOCK window (not a fixed epoch count),
because epoch duration itself varies with batch size -- wall-clock-windowed
step counting is the fair way to compare throughput across configs. A
short warm-up period (model load, CUDA context init, first-batch JIT/cuDNN
autotune) is excluded from the measurement window.

Requires the num-workers-configurable image built as
'piper-voice-trainer:gpu-bench' (Dockerfile patch added 2026-09-11) --
--num-workers did not exist as a CLI flag before that patch.

Usage (after training completion, GPU confirmed idle):
    python3 benchmark_configs.py --dataset-dir /path/to/preprocessed \\
        --checkpoint /path/to/starting.ckpt --output /path/to/results.json \\
        --dataset-clip-count 239

Safe by construction:
  - Never touches the production 'piper-voice-trainer:gpu' image/tag.
  - Every launched container is uniquely named and force-removed after
    each benchmark window, even on failure (try/finally).
  - Refuses to start if the GPU is not idle (checks nvidia-smi
    --query-compute-apps first) -- will not silently contend with a real job.
  - Never writes to a production job's own workspace/output dirs; each
    benchmark gets its own throwaway --default_root_dir under a temp dir.
  - Read-only mounts for dataset + checkpoint -- benchmark runs cannot
    mutate the real job's data.
"""
import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

BENCH_IMAGE = "piper-voice-trainer:gpu-bench"
WARMUP_SECONDS = 45     # exclude model-load / cuDNN autotune from the measurement window
WINDOW_SECONDS = 120    # steady-state measurement window per config
SAMPLE_INTERVAL = 2     # nvidia-smi / cpu sampling cadence during the window
VRAM_SAFE_CEILING_MB = 21000  # stay well clear of the 3090's 24576MB -- leave real headroom
EPOCH_PROJECTION_TARGETS = [500, 1000, 4000]

FP32_SWEEP = [
    {"name": "baseline_w1_fp32_b4",  "num_workers": 1, "precision": "32", "batch_size": 4},
    {"name": "w4_fp32_b4",           "num_workers": 4, "precision": "32", "batch_size": 4},
    {"name": "w8_fp32_b4",           "num_workers": 8, "precision": "32", "batch_size": 4},
    {"name": "w8_fp32_b16",          "num_workers": 8, "precision": "32", "batch_size": 16},
    {"name": "w8_fp32_b32",          "num_workers": 8, "precision": "32", "batch_size": 32},  # gated by VRAM check
]


def sh(cmd, timeout=30, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def gpu_idle() -> bool:
    # ollama.service is a normal, always-resident background process (agent
    # chat model) -- it typically holds ~3GB of VRAM even when not actively
    # generating. That's not a competing GPU job, just idle residency, so
    # the threshold sits above its typical footprint. Anything genuinely
    # competing (ComfyUI rendering, another training job) uses far more.
    r = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    apps = [l for l in r.stdout.strip().splitlines() if l.strip()]
    real = [l for l in apps if int(l.split(",")[-1].strip().split()[0]) > 4096]
    return len(real) == 0


def gpu_sample() -> dict:
    r = sh(["nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
    return {
        "gpu_util_pct": float(parts[0]), "mem_util_pct": float(parts[1]),
        "vram_used_mb": float(parts[2]), "sm_clock_mhz": float(parts[3]),
        "power_w": float(parts[4]), "temp_c": float(parts[5]),
    }


def cpu_sample(prev) -> tuple:
    with open("/proc/stat") as f:
        lines = [l for l in f if l.startswith("cpu")]
    cur = {l.split()[0]: list(map(int, l.split()[1:])) for l in lines}
    if prev is None:
        return None, cur
    results = {}
    for name, vals in cur.items():
        pvals = prev.get(name)
        if not pvals:
            continue
        total_d = sum(vals) - sum(pvals)
        idle_d = vals[3] - pvals[3]
        results[name] = 100.0 * (1 - idle_d / total_d) if total_d > 0 else 0.0
    per_core = {k: v for k, v in results.items() if k != "cpu"}
    return per_core, cur


def read_tfevents_series(logdir) -> dict:
    """Full step->{tag:val} series from the most recent lightning_logs version dir
    (needed for loss-stability, not just an endpoint value).

    IMPORTANT: `event.step` is NOT a safe proxy for "how much happened in
    this window" here. Every benchmark config resumes from the SAME
    pretrained checkpoint (en_US-lessac-medium-epoch2164.ckpt), and
    PyTorch Lightning restores its *global_step* counter from that
    checkpoint on resume -- the original Lessac pretraining ran on a much
    bigger dataset, so that restored global_step is already in the
    hundreds of thousands. The very first step logged after resume jumps
    straight from "no events yet" (step effectively 0 in our reading) to
    that huge number, which previously produced nonsense steps/sec
    (a ~670,000-step "delta" inside a 2-minute window). Every timestamp
    (`event.wall_time`) is real wall-clock time regardless of resume
    semantics, so that -- not the step number -- is what run_one_config
    now uses to count how many steps actually happened inside the
    measurement window."""
    versions = sorted(glob.glob(os.path.join(logdir, "lightning_logs", "version_*")))
    if not versions:
        return {"series": {}}
    latest = versions[-1]
    ev_files = glob.glob(os.path.join(latest, "events.out.tfevents.*"))
    if not ev_files:
        return {"series": {}}
    from tensorboard.backend.event_processing import event_file_loader
    import numpy as np
    loader = event_file_loader.EventFileLoader(ev_files[0])
    series = {}  # tag -> list of (step, val, wall_time)
    for event in loader.Load():
        if event.HasField("summary"):
            for v in event.summary.value:
                if v.HasField("tensor"):
                    t = v.tensor
                    arr = (np.frombuffer(t.tensor_content, dtype=np.float32)
                           if t.tensor_content else np.array(t.float_val or t.double_val))
                    if len(arr):
                        series.setdefault(v.tag, []).append((event.step, float(arr[0]), event.wall_time))
    return {"series": series}


def run_one_config(cfg, dataset_dir, checkpoint, work_root) -> dict:
    name = cfg["name"]
    container_name = f"piper-bench-{name}-{int(time.time())}"
    run_dir = os.path.join(work_root, name)
    os.makedirs(run_dir, exist_ok=True)

    # Piper dataset.jsonl stores absolute container paths like
    # /workspace/jobs/<voice>/preprocessed/cache/... — mounting only the
    # preprocessed dir at a different path makes every sample FileNotFound.
    dataset_dir = os.path.abspath(dataset_dir)
    voice_ws = os.path.dirname(dataset_dir.rstrip("/"))  # .../<voice>
    voice_name = os.path.basename(voice_ws)
    container_dataset = f"/workspace/jobs/{voice_name}/preprocessed"
    cmd = [
        "docker", "run", "--rm", "-d", "--gpus", "all", "--name", container_name,
        "--entrypoint", "python3",
        "--shm-size=512m",
        "-v", f"{voice_ws}:/workspace/jobs/{voice_name}:ro",
        "-v", f"{checkpoint}:/bench/checkpoint.ckpt:ro",
        "-v", f"{run_dir}:/bench/out",
        BENCH_IMAGE, "-m", "piper_train",
        "--dataset-dir", container_dataset,
        "--accelerator", "gpu", "--devices", "1",
        "--batch-size", str(cfg["batch_size"]),
        "--validation-split", "0.0", "--num-test-examples", "0",
        "--max_epochs", "999999",
        "--resume_from_checkpoint", "/bench/checkpoint.ckpt",
        "--checkpoint-epochs", "999999",
        "--default_root_dir", "/bench/out",
        "--max-phoneme-ids", "800",
        "--log_every_n_steps", "1",
        "--precision", str(cfg["precision"]),
        "--num-workers", str(cfg["num_workers"]),
    ]

    result = {"config": cfg, "started_at": datetime.now(timezone.utc).isoformat()}
    samples = []
    cpu_prev = None
    try:
        launch = sh(cmd, timeout=30)
        if launch.returncode != 0:
            result["error"] = f"launch failed: {launch.stderr.strip()[:500]}"
            return result

        deadline_warmup = time.time() + WARMUP_SECONDS
        while time.time() < deadline_warmup:
            running = sh(["docker", "ps", "-q", "--filter", f"name={container_name}"]).stdout.strip()
            if not running:
                logs = sh(["docker", "logs", container_name], timeout=15).stdout
                result["error"] = f"container exited during warmup -- last log lines: {logs[-800:]}"
                return result
            g = gpu_sample()
            if g["vram_used_mb"] > VRAM_SAFE_CEILING_MB:
                sh(["docker", "kill", container_name], timeout=15)
                result["error"] = f"aborted during warmup -- VRAM {g['vram_used_mb']}MB exceeded safety ceiling {VRAM_SAFE_CEILING_MB}MB"
                return result
            time.sleep(SAMPLE_INTERVAL)

        window_start = time.time()
        deadline_window = window_start + WINDOW_SECONDS
        while time.time() < deadline_window:
            running = sh(["docker", "ps", "-q", "--filter", f"name={container_name}"]).stdout.strip()
            if not running:
                result["error"] = "container exited mid-measurement-window"
                break
            g = gpu_sample()
            per_core, cpu_prev = cpu_sample(cpu_prev)
            saturated_cores = sum(1 for v in (per_core or {}).values() if v > 90)
            samples.append({**g, "t": time.time() - window_start, "saturated_cores": saturated_cores})
            time.sleep(SAMPLE_INTERVAL)
        window_end = time.time()
        window_elapsed = window_end - window_start
        progress_end = read_tfevents_series(run_dir)

        # Count real training steps by WALL-CLOCK time inside [window_start,
        # window_end], not by step-number delta -- see read_tfevents_series
        # docstring for why step numbers are unsafe here (resumed
        # global_step). loss_gen_all is logged once per training step, so
        # its count in-window IS the step count.
        in_window = [(step, val) for step, val, wt in progress_end["series"].get("loss_gen_all", [])
                     if window_start <= wt <= window_end]
        step_delta = len(in_window)

        # Loss stability over the measured window: std dev + simple trend
        # (last-quarter mean minus first-quarter mean) for each loss tag,
        # restricted to samples whose wall_time falls inside this window.
        loss_stability = {}
        for tag in ("loss_gen_all", "loss_disc_all"):
            vals = [val for _, val, wt in progress_end["series"].get(tag, [])
                    if window_start <= wt <= window_end]
            if len(vals) >= 4:
                q = len(vals) // 4
                loss_stability[tag] = {
                    "n_samples": len(vals),
                    "mean": statistics.mean(vals),
                    "stdev": statistics.stdev(vals),
                    "first_quarter_mean": statistics.mean(vals[:q]),
                    "last_quarter_mean": statistics.mean(vals[-q:]),
                    "trend": statistics.mean(vals[-q:]) - statistics.mean(vals[:q]),
                }
            else:
                loss_stability[tag] = {"n_samples": len(vals), "note": "too few samples in window"}

        near_zero_util_samples = sum(1 for s in samples if s["gpu_util_pct"] < 5)

        result.update({
            "window_seconds": window_elapsed,
            "steps_completed": step_delta,
            "steps_per_sec": step_delta / window_elapsed if window_elapsed > 0 else None,
            "gpu_util_avg": statistics.mean(s["gpu_util_pct"] for s in samples) if samples else None,
            "gpu_util_min": min((s["gpu_util_pct"] for s in samples), default=None),
            "gpu_util_max": max((s["gpu_util_pct"] for s in samples), default=None),
            "vram_used_mb_avg": statistics.mean(s["vram_used_mb"] for s in samples) if samples else None,
            "vram_used_mb_max": max((s["vram_used_mb"] for s in samples), default=None),
            "power_w_avg": statistics.mean(s["power_w"] for s in samples) if samples else None,
            "sm_clock_mhz_avg": statistics.mean(s["sm_clock_mhz"] for s in samples) if samples else None,
            "avg_saturated_cpu_cores": statistics.mean(s["saturated_cores"] for s in samples) if samples else None,
            "max_saturated_cpu_cores": max((s["saturated_cores"] for s in samples), default=None),
            "stall_samples_gpu_util_under_5pct": near_zero_util_samples,
            "stall_fraction": near_zero_util_samples / len(samples) if samples else None,
            "loss_stability": loss_stability,
            "raw_samples": samples,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sh(["docker", "rm", "-f", container_name], timeout=30)
    return result


def derive_metrics(res, clip_count):
    """steps/epoch, sec/epoch, epochs/min, and wall-clock projections."""
    if res.get("error") or not res.get("steps_per_sec"):
        return res
    batch = res["config"]["batch_size"]
    steps_per_epoch = math.ceil(clip_count / batch)
    sec_per_step = 1.0 / res["steps_per_sec"]
    sec_per_epoch = steps_per_epoch * sec_per_step
    epochs_per_min = 60.0 / sec_per_epoch if sec_per_epoch > 0 else None
    res["derived"] = {
        "steps_per_epoch": steps_per_epoch,
        "sec_per_step": round(sec_per_step, 4),
        "sec_per_epoch": round(sec_per_epoch, 2),
        "epochs_per_min": round(epochs_per_min, 3) if epochs_per_min else None,
        "projected_wallclock_hours": {
            str(n): round(n * sec_per_epoch / 3600, 2) for n in EPOCH_PROJECTION_TARGETS
        },
    }
    return res


def print_comparison_table(all_results):
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON")
    print("=" * 100)
    header = f"{'config':22} {'sec/step':>9} {'sec/epoch':>10} {'epochs/min':>11} {'gpu%avg':>8} {'vram_max':>9} {'watts':>7} {'cpu_sat':>8} {'stall%':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if r.get("error"):
            print(f"{r['config']['name']:22} ERROR: {r['error'][:80]}")
            continue
        d = r.get("derived", {})
        print(f"{r['config']['name']:22} "
              f"{d.get('sec_per_step', float('nan')):9.3f} "
              f"{d.get('sec_per_epoch', float('nan')):10.1f} "
              f"{d.get('epochs_per_min', float('nan')):11.3f} "
              f"{r.get('gpu_util_avg', float('nan')):8.1f} "
              f"{r.get('vram_used_mb_max', float('nan')):9.0f} "
              f"{r.get('power_w_avg', float('nan')):7.0f} "
              f"{r.get('avg_saturated_cpu_cores', float('nan')):8.2f} "
              f"{(r.get('stall_fraction') or 0)*100:7.1f}")
    print("=" * 100)
    print("\nWall-clock projections (hours):")
    for r in all_results:
        if r.get("error"):
            continue
        proj = r.get("derived", {}).get("projected_wallclock_hours", {})
        print(f"  {r['config']['name']:22} " + "  ".join(f"{n}ep={h}h" for n, h in proj.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="/tmp/piper_bench_results.json")
    ap.add_argument("--dataset-clip-count", type=int, required=True,
                     help="Accepted clip count for this dataset (steps/epoch = ceil(count/batch))")
    ap.add_argument("--configs", default=None, help="JSON file overriding FP32_SWEEP")
    args = ap.parse_args()

    if not gpu_idle():
        print("REFUSING TO RUN: GPU is not idle (a real job appears to be using it). "
              "Check `nvidia-smi --query-compute-apps` and re-run once clear.", file=sys.stderr)
        sys.exit(1)

    fp32_configs = FP32_SWEEP
    if args.configs:
        with open(args.configs) as f:
            fp32_configs = json.load(f)

    work_root = tempfile.mkdtemp(prefix="piper_bench_")
    print(f"Benchmark scratch dir: {work_root}")
    all_results = []

    def save():
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    b16_result = None
    for cfg in fp32_configs:
        if cfg["batch_size"] == 32:
            # Safety gate: only attempt batch=32 if batch=16 fit with real headroom.
            if b16_result is None or b16_result.get("error"):
                print(f"SKIPPING {cfg['name']}: no successful batch=16 result to extrapolate from.")
                continue
            projected = (b16_result.get("vram_used_mb_max") or 0) * 2 - 500  # rough model-weight-shared estimate
            if projected > VRAM_SAFE_CEILING_MB:
                print(f"SKIPPING {cfg['name']}: batch=16 used {b16_result.get('vram_used_mb_max')}MB, "
                      f"projected batch=32 (~{projected:.0f}MB) exceeds safety ceiling {VRAM_SAFE_CEILING_MB}MB. "
                      f"Recorded as skipped, not run.")
                all_results.append({"config": cfg, "error": f"skipped -- projected VRAM {projected:.0f}MB unsafe"})
                save()
                continue

        print(f"--- {cfg['name']} ({WARMUP_SECONDS}s warmup + {WINDOW_SECONDS}s measured window) ---")
        # CUDA context teardown after --rm can leave VRAM occupied for a few
        # seconds. Wait for genuine idle instead of aborting the whole sweep.
        idle_ok = False
        for _wait in range(30):
            if gpu_idle():
                idle_ok = True
                break
            time.sleep(2)
        if not idle_ok:
            print("GPU no longer idle before this config -- stopping sweep early.", file=sys.stderr)
            break
        res = run_one_config(cfg, args.dataset_dir, args.checkpoint, work_root)
        res = derive_metrics(res, args.dataset_clip_count)
        all_results.append(res)
        save()
        if cfg["batch_size"] == 16 and cfg["precision"] == "32":
            b16_result = res
        print(json.dumps({k: res.get(k) for k in ("steps_per_sec", "gpu_util_avg", "vram_used_mb_max",
                                                     "power_w_avg", "avg_saturated_cpu_cores", "error")}, indent=2))

    # Pick best successful FP32 config by steps_per_sec, run its FP16 twin.
    successful_fp32 = [r for r in all_results if not r.get("error") and r.get("steps_per_sec")]
    if successful_fp32:
        best = max(successful_fp32, key=lambda r: r["steps_per_sec"])
        fp16_cfg = dict(best["config"])
        fp16_cfg["precision"] = "16"
        fp16_cfg["name"] = fp16_cfg["name"].replace("fp32", "fp16") + "_bestfp32twin"
        print(f"\n--- Best FP32 config was {best['config']['name']} "
              f"({best['steps_per_sec']:.3f} steps/sec) -- running its FP16/AMP twin: {fp16_cfg['name']} ---")
        idle_ok = False
        for _wait in range(30):
            if gpu_idle():
                idle_ok = True
                break
            time.sleep(2)
        if idle_ok:
            fp16_res = run_one_config(fp16_cfg, args.dataset_dir, args.checkpoint, work_root)
            fp16_res = derive_metrics(fp16_res, args.dataset_clip_count)
            fp16_res["fp32_twin_of"] = best["config"]["name"]
            all_results.append(fp16_res)
            save()
            print(json.dumps({k: fp16_res.get(k) for k in ("steps_per_sec", "gpu_util_avg", "vram_used_mb_max",
                                                              "power_w_avg", "loss_stability", "error")}, indent=2))
    else:
        print("No successful FP32 config to base an FP16 comparison on -- skipping FP16 run.", file=sys.stderr)

    print_comparison_table(all_results)
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"\nDone. Full results: {args.output}")


if __name__ == "__main__":
    main()
