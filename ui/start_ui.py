#!/usr/bin/env python3
"""Start / repair Voice Trainer UI on :8765 from the install ui/ directory.

- Always regenerates status.json (actual GPU / install path)
- Serves from $INSTALL_DIR/ui (never arbitrary cwd)
- Verifies GET / and GET /status.json before reporting READY
- Does not kill foreign processes on the port
- Tracks ownership via PID file under the ui/ directory
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Same directory as this script when shipped in the repo.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from write_status import resolve_install_dir, write_status_json  # noqa: E402

DEFAULT_PORT = int(os.environ.get('OTACON_VT_UI_PORT') or '8765')
PID_NAME = '.otacon-vt-ui.pid'
LOG_PATH = Path('/tmp/otacon-vt-ui.log')


def port_listening(port: int, host: str = '127.0.0.1') -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _pid_path(ui_dir: Path) -> Path:
    return ui_dir / PID_NAME


def _read_pid(ui_dir: Path) -> int | None:
    path = _pid_path(ui_dir)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding='utf-8').strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cmdline(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'replace')
    except OSError:
        return ''


def _cwd(pid: int) -> str:
    try:
        return os.readlink(f'/proc/{pid}/cwd')
    except OSError:
        return ''


def _is_our_http_server(pid: int, ui_dir: Path) -> bool:
    cmd = _cmdline(pid)
    if 'http.server' not in cmd and 'start_ui.py' not in cmd and 'genome' not in cmd.lower():
        return False
    cwd = _cwd(pid)
    if cwd and Path(cwd).resolve() == ui_dir.resolve():
        return True
    # PID file in ui_dir is authoritative when we wrote it.
    owned = _read_pid(ui_dir)
    return owned == pid


def verify_ui(port: int, timeout: float = 3.0) -> dict:
    base = f'http://127.0.0.1:{port}'
    out = {
        'ok': False,
        'index_ok': False,
        'status_ok': False,
        'status': None,
        'error': '',
    }
    deadline = time.time() + timeout
    last_err = ''
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{base}/', timeout=1.5) as resp:
                out['index_ok'] = resp.status == 200
            with urllib.request.urlopen(f'{base}/status.json', timeout=1.5) as resp:
                raw = resp.read()
                data = json.loads(raw.decode('utf-8'))
                out['status'] = data
                out['status_ok'] = resp.status == 200 and isinstance(data, dict) and bool(data.get('ok'))
            if out['index_ok'] and out['status_ok']:
                out['ok'] = True
                return out
            last_err = 'index or status.json check failed'
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_err = str(exc)
        time.sleep(0.25)
    out['error'] = last_err or 'verify timeout'
    return out


def _start_http_server(ui_dir: Path, port: int) -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logf = open(LOG_PATH, 'ab')
    proc = subprocess.Popen(
        [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
        cwd=str(ui_dir),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _pid_path(ui_dir).write_text(str(proc.pid) + '\n', encoding='utf-8')
    return proc.pid


def ensure_ui(
    *,
    install_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    repair: bool = True,
) -> dict:
    root = resolve_install_dir(str(install_dir) if install_dir else None)
    ui_dir = root / 'ui'
    if not ui_dir.is_dir():
        return {
            'ok': False,
            'action': 'missing_ui',
            'install_dir': str(root),
            'error': f'UI directory missing: {ui_dir}',
        }

    written = write_status_json(root)
    status_path = written['path']

    # Already healthy?
    if port_listening(port):
        check = verify_ui(port)
        if check['ok']:
            return {
                'ok': True,
                'action': 'already_ready',
                'url': f'http://127.0.0.1:{port}/',
                'install_dir': str(root),
                'status_json': status_path,
                'verify': check,
            }
        # Port up but status.json missing/bad — repair file first (no kill).
        if repair:
            write_status_json(root)
            check2 = verify_ui(port, timeout=2.0)
            if check2['ok']:
                return {
                    'ok': True,
                    'action': 'repaired_status_json',
                    'url': f'http://127.0.0.1:{port}/',
                    'install_dir': str(root),
                    'status_json': status_path,
                    'verify': check2,
                }
            # Our owned server serving wrong cwd → restart ours only.
            owned = _read_pid(ui_dir)
            if owned and _pid_alive(owned) and _is_our_http_server(owned, ui_dir):
                try:
                    os.kill(owned, signal.SIGTERM)
                except OSError:
                    pass
                time.sleep(0.4)
            elif port_listening(port):
                # Foreign occupant — do not kill.
                return {
                    'ok': False,
                    'action': 'port_conflict',
                    'install_dir': str(root),
                    'error': (
                        f'Port {port} is in use by another process and does not serve a valid '
                        f'status.json for this install. Refusing to kill it.'
                    ),
                    'hint': (
                        f'Stop the foreign listener on :{port}, then re-run: '
                        f'python3 {Path(__file__).name} --install-dir {root}'
                    ),
                    'verify': check2,
                }

    if port_listening(port):
        return {
            'ok': False,
            'action': 'port_conflict',
            'error': f'Port {port} still occupied after repair attempt',
            'install_dir': str(root),
        }

    pid = _start_http_server(ui_dir, port)
    time.sleep(0.5)
    check = verify_ui(port)
    if not check['ok']:
        return {
            'ok': False,
            'action': 'verify_failed',
            'pid': pid,
            'install_dir': str(root),
            'status_json': status_path,
            'error': check.get('error') or 'UI started but / or status.json failed',
            'verify': check,
            'log': str(LOG_PATH),
        }
    return {
        'ok': True,
        'action': 'started',
        'pid': pid,
        'url': f'http://127.0.0.1:{port}/',
        'install_dir': str(root),
        'status_json': status_path,
        'verify': check,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Start/repair Voice Trainer UI from install ui/')
    p.add_argument('--install-dir', default='', help='Detected install root (not a hardcoded username)')
    p.add_argument('--port', type=int, default=DEFAULT_PORT)
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    result = ensure_ui(
        install_dir=Path(args.install_dir) if args.install_dir else None,
        port=args.port,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result.get('action'), result.get('url') or result.get('error') or '')
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
