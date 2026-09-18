#!/usr/bin/env python3
"""Write Voice Trainer ui/status.json from actual machine state.

No hardcoded usernames, install paths, GPU models, or VRAM.
Install dir is resolved from OTACON_VT_DIR, --install-dir, or $HOME/otacon-voice-trainer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def resolve_install_dir(explicit: str | None = None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit).expanduser().resolve()
    env = (os.environ.get('OTACON_VT_DIR') or '').strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / 'otacon-voice-trainer').resolve()


def _nvidia_smi_bin() -> str | None:
    # Prefer platform resolver when Expansion/Core is on PYTHONPATH.
    try:
        from core.platform import _resolve_nvidia_smi
        found = _resolve_nvidia_smi()
        if found:
            return found
    except Exception:
        pass
    wsl = Path('/usr/lib/wsl/lib/nvidia-smi')
    if wsl.is_file() and os.access(wsl, os.X_OK):
        return str(wsl)
    return shutil.which('nvidia-smi')


def _nvidia_env() -> dict[str, str]:
    env = dict(os.environ)
    try:
        from core.platform import _nvidia_smi_env
        env.update(_nvidia_smi_env())
    except Exception:
        lib = '/usr/lib/wsl/lib'
        if Path(lib).is_dir():
            prev = env.get('PATH', '')
            if lib not in prev.split(':'):
                env['PATH'] = f'{lib}:{prev}' if prev else lib
    return env


def probe_gpu() -> dict[str, Any]:
    """Return gpu fields distinguishing detected / cpu_only / probe_failed."""
    smi = _nvidia_smi_bin()
    if not smi:
        return {
            'gpu_state': 'cpu_only',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': None,
        }
    try:
        r = subprocess.run(
            [
                smi,
                '--query-gpu=name,memory.total',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=_nvidia_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            'gpu_state': 'probe_failed',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': str(exc),
        }
    if r.returncode != 0:
        err = (r.stderr or r.stdout or f'nvidia-smi exit {r.returncode}').strip()[:500]
        return {
            'gpu_state': 'probe_failed',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': err or 'nvidia-smi failed',
        }
    line = (r.stdout or '').strip().splitlines()
    if not line:
        return {
            'gpu_state': 'probe_failed',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': 'nvidia-smi returned empty GPU list',
        }
    parts = [p.strip() for p in line[0].split(',')]
    name = parts[0] if parts else ''
    vram: int | None = None
    if len(parts) > 1:
        try:
            vram = int(float(parts[1]))
        except ValueError:
            vram = None
    if not name:
        return {
            'gpu_state': 'probe_failed',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': 'nvidia-smi returned blank GPU name',
        }
    return {
        'gpu_state': 'gpu_detected',
        'gpu_name': name,
        'gpu_vram_mb': vram,
        'gpu_probe_error': None,
    }


def detect_image() -> str:
    docker = shutil.which('docker')
    if not docker:
        return ''
    for tag in ('piper-voice-trainer:gpu', 'piper-voice-trainer:latest'):
        try:
            r = subprocess.run(
                [docker, 'image', 'inspect', tag],
                capture_output=True,
                timeout=8,
                check=False,
            )
            if r.returncode == 0:
                return tag
        except (OSError, subprocess.TimeoutExpired):
            continue
    return ''


def build_status(
    install_dir: Path,
    *,
    force_ok: bool | None = None,
) -> dict[str, Any]:
    gpu = probe_gpu()
    image = detect_image()
    ui_dir = install_dir / 'ui'
    installed = install_dir.is_dir() and ui_dir.is_dir()
    # ok means install layout is present and status is writable facts — not "training ready".
    if force_ok is not None:
        ok = bool(force_ok)
    else:
        ok = bool(installed)
    payload: dict[str, Any] = {
        'ok': ok,
        'product': 'Otacon Voice Trainer',
        'gpu_name': gpu['gpu_name'],
        'gpu_vram_mb': gpu['gpu_vram_mb'],
        'gpu_state': gpu['gpu_state'],
        'image': image or None,
        'install_dir': str(install_dir),
        'wsl': Path('/proc/sys/fs/binfmt_misc/WSLInterop').exists()
        or 'microsoft' in Path('/proc/version').read_text(encoding='utf-8', errors='ignore').lower()
        if Path('/proc/version').is_file()
        else False,
        'docs': 'https://github.com/Otaconskeep/otacon-voice-trainer',
        'site': 'https://otaconskeep-site.otaconskeep.workers.dev/otacon/#voice-trainer',
    }
    if gpu.get('gpu_probe_error'):
        payload['gpu_probe_error'] = gpu['gpu_probe_error']
    return payload


def write_status_json(
    install_dir: Path | None = None,
    *,
    force_ok: bool | None = None,
) -> dict[str, Any]:
    root = resolve_install_dir(str(install_dir) if install_dir else None)
    ui_dir = root / 'ui'
    ui_dir.mkdir(parents=True, exist_ok=True)
    payload = build_status(root, force_ok=force_ok)
    path = ui_dir / 'status.json'
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return {'path': str(path), 'status': payload}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Write Voice Trainer ui/status.json')
    p.add_argument('--install-dir', default='', help='Voice Trainer install root (no username hardcode)')
    p.add_argument('--print', action='store_true', help='Print JSON to stdout')
    args = p.parse_args(argv)
    out = write_status_json(Path(args.install_dir) if args.install_dir else None)
    if args.print:
        print(json.dumps(out['status'], indent=2))
    else:
        print(out['path'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
