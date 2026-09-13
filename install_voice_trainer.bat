@echo off
REM Otaconskeep // Otacon Voice Trainer — Windows one-click
REM Sets up WSL2 + Ubuntu if needed, then runs install_voice_trainer.sh inside Linux.
setlocal enabledelayedexpansion
title Otacon Voice Trainer Installer (Windows)

echo ============================================================
echo  OTACONSKEEP // OTACON VOICE TRAINER
echo  Genome GPU Piper installer — Antonio G. Garcia
echo ============================================================
echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
  echo WSL is not installed. Requesting Administrator to run: wsl --install
  net session >nul 2>&1
  if errorlevel 1 (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
  )
  wsl.exe --install -d Ubuntu
  echo.
  echo After Windows finishes / reboots, open Ubuntu once to create a user,
  echo then double-click this .bat again.
  pause
  exit /b
)

echo Launching installer inside WSL Ubuntu...
wsl.exe -e bash -lc "curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash"
set ERR=!ERRORLEVEL!
if not "!ERR!"=="0" (
  if "!ERR!"=="75" (
    echo.
    echo systemd was just enabled in WSL. Shutting WSL down so it restarts cleanly...
    wsl.exe --shutdown
    timeout /t 3 >nul
    echo Re-running installer...
    wsl.exe -e bash -lc "curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh | bash"
  ) else (
    echo Installer exited with code !ERR!
    pause
  )
)
echo.
echo If CUDA verify passed, open http://127.0.0.1:8765/ from Windows.
pause
