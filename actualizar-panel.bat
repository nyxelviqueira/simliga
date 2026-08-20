@echo off
REM ---------------------------------------------------------------------------
REM  Doble clic aqui: actualiza los datos, simula y abre el panel.
REM
REM  %~dp0 es la carpeta donde vive este .bat, asi que funciona aunque se
REM  ejecute desde un acceso directo en el escritorio o desde otra ruta.
REM ---------------------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"

set TEMPORADA=2026-27
set PYTHON=.venv\Scripts\python.exe

if not exist "%PYTHON%" (
    echo.
    echo  No encuentro el entorno virtual en %CD%\.venv
    echo  Crealo una vez con:
    echo.
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo  Actualizando SimLiga - temporada %TEMPORADA%
echo  ============================================
echo.

"%PYTHON%" -m simliga actualizar --temporada %TEMPORADA% --sims 20000 --panel

if errorlevel 1 (
    echo.
    echo  Algo ha fallado. El panel anterior sigue siendo valido.
    echo.
    pause
    exit /b 1
)

echo.
echo  Listo. Abriendo el panel...
start "" "out\panel.html"
