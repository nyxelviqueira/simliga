#!/bin/sh
# Arranque del contenedor.
#
# Dos cosas antes de servir: sembrar el volumen la primera vez y negarse a
# arrancar sin contrasena. Lo segundo es a proposito: el panel escribe en la
# base de datos sin preguntar quien llama, y una URL publica sin token seria
# editable por cualquiera que diese con ella.
set -e

if [ -z "$SIMLIGA_TOKEN" ]; then
    echo "ERROR: falta SIMLIGA_TOKEN."
    echo "El panel expone endpoints de escritura; publicarlo sin contrasena"
    echo "significa que cualquiera que dé con la URL puede tocar los datos."
    echo
    echo "  fly secrets set SIMLIGA_TOKEN=\$(openssl rand -hex 16)"
    exit 1
fi

mkdir -p "$SIMLIGA_DATA"
if [ ! -f "$SIMLIGA_DATA/simliga.sqlite" ]; then
    echo "Volumen vacío: copiando la base de datos inicial."
    cp /app/semilla/simliga.sqlite "$SIMLIGA_DATA/simliga.sqlite"
fi

exec python -m simliga servidor \
    --temporada "${SIMLIGA_TEMPORADA:-2026-27}" \
    --puerto "${PORT:-8080}" \
    --en-red \
    --sin-abrir
