@echo off
REM ---------------------------------------------------------------------------
REM  Doble clic aqui: abre el panel accesible desde el movil.
REM
REM  El PC y el movil tienen que estar en la misma wifi. La ventana muestra la
REM  direccion que hay que escribir en el navegador del telefono.
REM
REM  OJO: mientras esto corra, cualquiera en esa red puede abrir el panel y
REM  tocar los escenarios. No lo dejes abierto en una red que no controles.
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
echo  SimLiga - panel accesible desde el movil (%TEMPORADA%)
echo  ======================================================
echo.
echo  Deja esta ventana abierta mientras lo uses. Ctrl+C para cerrarlo.
echo.

"%PYTHON%" -m simliga servidor --temporada %TEMPORADA% --en-red --sin-abrir

if errorlevel 1 pause
