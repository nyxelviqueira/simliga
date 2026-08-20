@echo off
REM ---------------------------------------------------------------------------
REM  Doble clic aqui: actualiza la web publica de GitHub Pages.
REM
REM  Requiere guardar una vez el token de GitHub en SIMLIGA_GITHUB_TOKEN.
REM  No guardes el token dentro del proyecto ni lo subas al repositorio.
REM ---------------------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"

set TEMPORADA=2026-27
set SIMS=20000

if "%~1" NEQ "" set TEMPORADA=%~1
if "%~2" NEQ "" set SIMS=%~2

set SCRIPT=scripts\lanzar_publicacion_github.ps1

if not exist "%SCRIPT%" (
    echo.
    echo  No encuentro %SCRIPT%.
    echo.
    pause
    exit /b 1
)

echo.
echo  SimLiga - actualizar web publica
echo  =================================
echo.
echo  Temporada: %TEMPORADA%
echo  Simulaciones: %SIMS%
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -Temporada "%TEMPORADA%" -Sims "%SIMS%" -Esperar

if errorlevel 1 (
    echo.
    echo  No se ha podido lanzar o completar la publicacion.
    echo.
    pause
    exit /b 1
)

echo.
pause
