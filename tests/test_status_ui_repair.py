#!/usr/bin/env python3
"""Regression: Voice Trainer status.json + UI serve/repair (Cases A–F)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'ui'
sys.path.insert(0, str(UI))

import write_status as ws  # noqa: E402
import start_ui as su  # noqa: E402


class VoiceTrainerStatusUiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / 'otacon-voice-trainer'
        self.ui = self.home / 'ui'
        self.ui.mkdir(parents=True)
        # Minimal index so http.server can serve /
        (self.ui / 'index.html').write_text('<html>vt</html>', encoding='utf-8')
        # Copy writers into temp ui so start_ui imports work when cwd differs
        for name in ('write_status.py', 'start_ui.py'):
            (self.ui / name).write_text((UI / name).read_text(encoding='utf-8'), encoding='utf-8')

    def tearDown(self):
        # Stop any server we started via PID file
        pid_path = self.ui / su.PID_NAME
        if pid_path.is_file():
            try:
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 15)
            except (OSError, ValueError):
                pass
        self.tmp.cleanup()

    def _free_port(self) -> int:
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_A_fresh_gpu_status_json(self):
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'gpu_detected',
            'gpu_name': 'NVIDIA GeForce RTX 3080',
            'gpu_vram_mb': 10240,
            'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value='piper-voice-trainer:gpu'):
            out = ws.write_status_json(self.home)
        data = out['status']
        self.assertTrue(data['ok'])
        self.assertEqual(data['gpu_name'], 'NVIDIA GeForce RTX 3080')
        self.assertEqual(data['gpu_vram_mb'], 10240)
        self.assertEqual(data['install_dir'], str(self.home.resolve()))
        self.assertNotIn('3070', json.dumps(data))
        self.assertNotIn('/home/crist/', json.dumps(data))
        # Serve + verify
        port = self._free_port()
        with mock.patch.object(su, 'write_status_json', return_value=out):
            result = su.ensure_ui(install_dir=self.home, port=port)
        self.assertTrue(result['ok'], result)
        self.assertTrue(result['verify']['ok'])

    def test_B_cpu_only_no_fake_gpu(self):
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'cpu_only',
            'gpu_name': None,
            'gpu_vram_mb': None,
            'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value=''):
            out = ws.write_status_json(self.home)
        data = out['status']
        self.assertIsNone(data['gpu_name'])
        self.assertIsNone(data['gpu_vram_mb'])
        self.assertEqual(data['gpu_state'], 'cpu_only')
        self.assertNotEqual(data['gpu_vram_mb'], 8192)

    def test_C_repair_missing_status_json(self):
        port = self._free_port()
        # Start server from ui/ WITHOUT status.json
        log = open('/tmp/otacon-vt-ui-test.log', 'ab')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
            cwd=str(self.ui),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (self.ui / su.PID_NAME).write_text(str(proc.pid) + '\n', encoding='utf-8')
        time.sleep(0.4)
        self.assertFalse((self.ui / 'status.json').is_file())
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'gpu_detected',
            'gpu_name': 'Test GPU',
            'gpu_vram_mb': 4096,
            'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value='piper-voice-trainer:gpu'):
            result = su.ensure_ui(install_dir=self.home, port=port)
        self.assertTrue(result['ok'], result)
        self.assertIn(result['action'], ('repaired_status_json', 'already_ready'))
        self.assertTrue((self.ui / 'status.json').is_file())
        data = json.loads((self.ui / 'status.json').read_text(encoding='utf-8'))
        self.assertTrue(data['ok'])

    def test_D_wrong_cwd_owned_server_repaired(self):
        port = self._free_port()
        wrong = Path(self.tmp.name) / 'wrong_cwd'
        wrong.mkdir()
        (wrong / 'index.html').write_text('wrong', encoding='utf-8')
        log = open('/tmp/otacon-vt-ui-test2.log', 'ab')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
            cwd=str(wrong),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Claim ownership so repair may restart
        (self.ui / su.PID_NAME).write_text(str(proc.pid) + '\n', encoding='utf-8')
        time.sleep(0.4)
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'cpu_only', 'gpu_name': None, 'gpu_vram_mb': None, 'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value=''), \
             mock.patch.object(su, '_is_our_http_server', return_value=True):
            result = su.ensure_ui(install_dir=self.home, port=port)
        # Either repaired by restart or reported conflict if kill race — must not claim READY without status
        if result.get('ok'):
            self.assertTrue(result['verify']['ok'])
            self.assertTrue((self.ui / 'status.json').is_file())
        else:
            self.assertIn(result.get('action'), ('port_conflict', 'verify_failed'))

    def test_E_foreign_port_not_killed(self):
        port = self._free_port()
        # Unrelated server (no status.json, not our pid)
        wrong = Path(self.tmp.name) / 'foreign'
        wrong.mkdir()
        (wrong / 'index.html').write_text('foreign', encoding='utf-8')
        log = open('/tmp/otacon-vt-ui-test3.log', 'ab')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1'],
            cwd=str(wrong),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(0.4)
        before = proc.poll()
        self.assertIsNone(before)
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'cpu_only', 'gpu_name': None, 'gpu_vram_mb': None, 'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value=''):
            result = su.ensure_ui(install_dir=self.home, port=port)
        self.assertFalse(result['ok'])
        self.assertEqual(result['action'], 'port_conflict')
        self.assertIsNone(proc.poll(), 'foreign process must not be killed')
        try:
            os.kill(proc.pid, 15)
        except OSError:
            pass

    def test_F_repeated_repair_no_duplicate(self):
        port = self._free_port()
        with mock.patch.object(ws, 'probe_gpu', return_value={
            'gpu_state': 'gpu_detected', 'gpu_name': 'GPU', 'gpu_vram_mb': 8, 'gpu_probe_error': None,
        }), mock.patch.object(ws, 'detect_image', return_value='piper-voice-trainer:gpu'):
            a = su.ensure_ui(install_dir=self.home, port=port)
            b = su.ensure_ui(install_dir=self.home, port=port)
        self.assertTrue(a['ok'] and b['ok'])
        self.assertEqual(b['action'], 'already_ready')
        # Still one listener
        self.assertTrue(su.port_listening(port))

    def test_install_dir_not_hardcoded_crist(self):
        fake_home = Path(self.tmp.name) / 'home' / 'block'
        fake_home.mkdir(parents=True)
        with mock.patch.object(ws, 'Path') as P:
            pass
        with mock.patch.dict(os.environ, {'OTACON_VT_DIR': str(self.home), 'HOME': str(fake_home)}):
            d = ws.resolve_install_dir()
        self.assertEqual(d, self.home.resolve())
        self.assertNotIn('crist', str(d))

    def test_probe_failed_not_cpu_only(self):
        with mock.patch.object(ws, '_nvidia_smi_bin', return_value='/usr/bin/nvidia-smi'):
            with mock.patch('subprocess.run', side_effect=OSError('boom')):
                g = ws.probe_gpu()
        self.assertEqual(g['gpu_state'], 'probe_failed')
        self.assertIsNone(g['gpu_name'])
        self.assertIsNotNone(g['gpu_probe_error'])


if __name__ == '__main__':
    unittest.main()
