#!/usr/bin/env python3
"""Voice Trainer must never hang on sudo password with no TTY."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SH = (ROOT / "install_voice_trainer.sh").read_text(encoding="utf-8")


def must(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)
        print(f"FAIL  {msg}")
    else:
        print(f"OK    {msg}")


def main() -> int:
    fails: list[str] = []
    must("sudo -n true" in SH, "probes non-interactive sudo before using sudo", fails)
    must("[[ -t 0 ]]" in SH and "[[ -t 1 ]]" in SH, "TTY check before interactive sudo", fails)
    must("no TTY to type a password" in SH or "non-interactive sudo" in SH, "fail-fast message when no TTY/sudo -n", fails)
    must(
        'if command_exists sudo; then SUDO="sudo"; else die' not in SH,
        "old blind SUDO=sudo assignment removed",
        fails,
    )
    print("=" * 60)
    if fails:
        print(f"{len(fails)} voice-trainer sudo regression(s) failed")
        return 1
    print("Voice Trainer noninteractive sudo regressions: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
