@echo off
REM ---------------------------------------------------------------------------
REM  Doble clic aqui: abre el panel con el boton de regenerar.
REM  Deja esta ventana abierta mientras uses el panel; Ctrl+C para cerrarlo.
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
echo  SimLiga - panel interactivo (%TEMPORADA%)
echo  =========================================
echo.
echo  Se abrira el navegador solo. Deja esta ventana abierta.
echo  Se usara un puerto libre para no mezclarse con paneles antiguos.
echo.

"%PYTHON%" -m simliga servidor --temporada %TEMPORADA% --puerto 0

if errorlevel 1 pause
