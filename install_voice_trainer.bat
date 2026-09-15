@echo off
REM Otaconskeep // Otacon Voice Trainer - Windows one-click
REM Selects a supported Ubuntu WSL distro (same logic as Otacon Core), then
REM downloads and runs install_voice_trainer.sh inside that distro.
setlocal enabledelayedexpansion
title Otacon Voice Trainer Installer (Windows)

set "SCRIPT_DIR=%~dp0"
set "FIND_UBUNTU_PS1=%SCRIPT_DIR%deploy\find-ubuntu.ps1"
if not exist "%FIND_UBUNTU_PS1%" set "FIND_UBUNTU_PS1=%SCRIPT_DIR%find-ubuntu.ps1"

echo ============================================================
echo  OTACONSKEEP // OTACON VOICE TRAINER
echo  Genome GPU Piper installer - Antonio G. Garcia
echo ============================================================
echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
  echo WSL is not installed. Requesting Administrator to run: wsl --install
  net session >nul 2>&1
  if errorlevel 1 (
    REM Never append %~f0 after powershell -Command (path becomes script text).
    set "OTACON_VT_SELF=%~f0"
    powershell -NoProfile -Command "Start-Process -LiteralPath $env:OTACON_VT_SELF -Verb RunAs"
    exit /b
  )
  wsl.exe --install -d Ubuntu
  echo.
  echo After Windows finishes / reboots, open Ubuntu once to create a user,
  echo then double-click this .bat again.
  pause
  exit /b 1
)

call :FIND_UBUNTU
if not defined UBUNTU_NAME (
  echo No Ubuntu WSL distro found. Installing Ubuntu...
  net session >nul 2>&1
  if errorlevel 1 (
    set "OTACON_VT_SELF=%~f0"
    powershell -NoProfile -Command "Start-Process -LiteralPath $env:OTACON_VT_SELF -Verb RunAs"
    exit /b
  )
  wsl.exe --install -d Ubuntu
  echo Open Ubuntu once to finish setup, then re-run this installer.
  pause
  exit /b 1
)

echo Selected WSL distro: %UBUNTU_NAME%
echo Launching installer inside that Ubuntu environment...
call :RUN_VT_INSTALLER
set ERR=!ERRORLEVEL!
if not "!ERR!"=="0" (
  if "!ERR!"=="75" (
    echo.
    echo systemd was just enabled. Terminating ONLY %UBUNTU_NAME% so it restarts cleanly...
    wsl.exe --terminate "%UBUNTU_NAME%"
    timeout /t 3 >nul
    echo Re-running installer...
    call :RUN_VT_INSTALLER
    set ERR=!ERRORLEVEL!
  )
)
if not "!ERR!"=="0" (
  if "!ERR!"=="2" (
    echo Voice Trainer finished DEGRADED ^(exit 2^).
    pause
    exit /b 2
  )
  echo Installer exited with code !ERR!
  pause
  exit /b !ERR!
)
echo.
echo If CUDA verify passed, open http://127.0.0.1:8765/ from Windows.
echo Distro used: %UBUNTU_NAME%
pause
exit /b 0

:RUN_VT_INSTALLER
wsl.exe -d "%UBUNTU_NAME%" -- bash -lc "set -euo pipefail; TMP=$(mktemp /tmp/otacon-vt-install.XXXXXX.sh); trap 'rm -f \"$TMP\"' EXIT; curl -fsSL https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/install_voice_trainer.sh -o \"$TMP\"; test -s \"$TMP\" || { echo 'Download failed or empty installer' >&2; exit 1; }; env OTACON_VT_DIR='%OTACON_VT_DIR%' OTACON_VT_SKIP_UI='%OTACON_VT_SKIP_UI%' OTACON_VT_SKIP_BUILD='%OTACON_VT_SKIP_BUILD%' OTACON_VT_UI_PORT='%OTACON_VT_UI_PORT%' bash \"$TMP\""
exit /b %errorlevel%

:FIND_UBUNTU
REM Prefer bundled deploy\find-ubuntu.ps1 (-File). Never put PowerShell ) inside IF (...).
REM Standalone .bat download: fetch helper to TEMP via top-level -Command, then -File.
set "UBUNTU_NAME="
if exist "%FIND_UBUNTU_PS1%" goto :FIND_UBUNTU_RUN
set "FIND_UBUNTU_PS1=%TEMP%\otacon-vt-find-ubuntu.ps1"
if exist "%FIND_UBUNTU_PS1%" goto :FIND_UBUNTU_RUN
set "FIND_UBUNTU_URL=https://raw.githubusercontent.com/Otaconskeep/otacon-voice-trainer/main/deploy/find-ubuntu.ps1"
REM Top-level -Command only (Otacon Setup rule) - path via env, not trailing args.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try{[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12}catch{}; Invoke-WebRequest -Uri $env:FIND_UBUNTU_URL -OutFile $env:FIND_UBUNTU_PS1 -UseBasicParsing"

:FIND_UBUNTU_RUN
if exist "%FIND_UBUNTU_PS1%" for /f "delims=" %%D in ('powershell -NoProfile -ExecutionPolicy Bypass -File "%FIND_UBUNTU_PS1%"') do set "UBUNTU_NAME=%%D"
exit /b

