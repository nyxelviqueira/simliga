"""Ingesta del calendario desde openfootball/espana.

Datos abiertos CC0 en GitHub, sin API key y sin cuota. Es la unica fuente
verificada que publica el calendario de una temporada **antes** de jugarse:
football-data.co.uk solo publica partidos ya disputados, y el plan gratuito de
API-Football esta limitado a las temporadas 2022-2024 (comprobado con la clave
del proyecto celta-dashboard: devuelve `Free plans do not have access to this
season`).

Se usa solo para el calendario. El historico de resultados sigue viniendo de
football-data.co.uk, que ademas trae cuotas de mercado para el benchmark.

Formato del fichero (texto plano, una temporada por fichero):

    ▪ Matchday 1
      Sun Aug 16 2026
        17:00  Club Atlético de Madrid v Málaga CF
               Real Racing Club de Santander v Villarreal CF

Los partidos ya jugados aparecen con el marcador en vez de la `v`:

        19:00   Girona FC  1-3 (0-3)  Rayo Vallecano
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata

import pandas as pd
import requests

from ..config import COMP_LALIGA, COMP_SEGUNDA, RAW_DIR
from ..db import get_or_create_team, register_alias, resolve_alias, upsert_match

SOURCE = "openfootball/espana"
BASE_URL = "https://raw.githubusercontent.com/openfootball/espana/master"
FILES = {COMP_LALIGA: "1-liga.txt", COMP_SEGUNDA: "2-liga2.txt"}

MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

RE_JORNADA = re.compile(r"^\s*[▪*]\s*(?:Matchday|Regular Season\s*-)\s*(\d+)", re.IGNORECASE)
RE_FECHA = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
RE_JUGADO = re.compile(
    r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s+(\d+)\s*-\s*(\d+)(?:\s*\(\d+-\d+\))?\s+(.+?)\s*$")
RE_PENDIENTE = re.compile(r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s+v\s+(.+?)\s*$")

# Nombres largos de openfootball -> nombre canonico del proyecto.
NOMBRES = {
    "Club Atlético de Madrid": "Atletico de Madrid",
    "Club Atletico de Madrid": "Atletico de Madrid",
    "Atlético Madrid": "Atletico de Madrid",
    "Atletico Madrid": "Atletico de Madrid",
    "Málaga CF": "Malaga CF",
    "Malaga CF": "Malaga CF",
    "Real Racing Club de Santander": "Racing de Santander",
    "Racing Santander": "Racing de Santander",
    "Villarreal CF": "Villarreal CF",
    "Villarreal": "Villarreal CF",
    "FC Barcelona": "FC Barcelona",
    "Barcelona": "FC Barcelona",
    "Athletic Club": "Athletic Club",
    "Real Madrid CF": "Real Madrid",
    "Real Madrid": "Real Madrid",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Sociedad": "Real Sociedad",
    "RCD Espanyol de Barcelona": "RCD Espanyol",
    "RCD Espanyol": "RCD Espanyol",
    "Espanyol": "RCD Espanyol",
    "Levante UD": "Levante UD",
    "Sevilla FC": "Sevilla FC",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Rayo Vallecano": "Rayo Vallecano",
    "Deportivo Alavés": "Deportivo Alaves",
    "Deportivo Alaves": "Deportivo Alaves",
    "Getafe CF": "Getafe CF",
    "RC Deportivo La Coruña": "Deportivo de La Coruna",
    "RC Deportivo La Coruna": "Deportivo de La Coruna",
    "Deportivo La Coruña": "Deportivo de La Coruna",
    "Deportivo La Coruna": "Deportivo de La Coruna",
    "Elche CF": "Elche CF",
    "RC Celta de Vigo": "Celta de Vigo",
    "Celta de Vigo": "Celta de Vigo",
    "Celta Vigo": "Celta de Vigo",
    "CA Osasuna": "CA Osasuna",
    "Valencia CF": "Valencia CF",
    "Real Betis Balompié": "Real Betis",
    "Real Betis": "Real Betis",
    "RCD Mallorca": "RCD Mallorca",
    "Girona FC": "Girona FC",
    "Real Oviedo": "Real Oviedo",
    "UD Las Palmas": "UD Las Palmas",
    "CD Leganés": "CD Leganes",
    "Real Valladolid CF": "Real Valladolid",
    "UD Almería": "UD Almeria",
    "UD Almeria": "UD Almeria",
    "Cádiz CF": "Cadiz CF",
    "Cadiz CF": "Cadiz CF",
    "Granada CF": "Granada CF",
}


def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto)
                   if unicodedata.category(c) != "Mn")


def descargar(season: str, competition: str = COMP_LALIGA, force: bool = False):
    """Descarga (con cache) el fichero de una temporada. `season` = '2026-27'."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"openfootball_{competition}_{season}.txt"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    resp = requests.get(f"{BASE_URL}/{season}/{FILES[competition]}", timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def parse(texto: str, season: str) -> pd.DataFrame:
    """Convierte el fichero de texto en un DataFrame de partidos."""
    año_inicio = int(season.split("-")[0])
    filas = []
    jornada = None
    fecha = None

    for linea in texto.splitlines():
        if not linea.strip() or linea.lstrip().startswith(("#", "=")):
            continue

        m = RE_JORNADA.match(linea)
        if m:
            jornada = int(m.group(1))
            continue

        m = RE_FECHA.match(linea)
        if m:
            mes_txt, dia, año = m.group(1), int(m.group(2)), m.group(3)
            mes = MESES.get(mes_txt)
            if mes is None:
                continue
            # El fichero solo repite el año al cambiar; si falta, se deduce:
            # de agosto a diciembre es el primer año de la temporada.
            if año is None:
                año = año_inicio if mes >= 7 else año_inicio + 1
            fecha = pd.Timestamp(int(año), mes, dia)
            continue

        if fecha is None or jornada is None:
            continue
        if "(" in linea and ")" in linea and not re.search(r"\d+\s*-\s*\d+", linea):
            continue  # linea de goleadores

        m = RE_JUGADO.match(linea)
        if m and " v " not in linea:
            local, hg, ag, visitante = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            filas.append((jornada, fecha, local.strip(), visitante.strip(), hg, ag))
            continue

        m = RE_PENDIENTE.match(linea)
        if m:
            filas.append((jornada, fecha, m.group(1).strip(), m.group(2).strip(), None, None))

    df = pd.DataFrame(filas, columns=["matchday", "match_date", "home", "away",
                                      "home_goals", "away_goals"])
    return df


def _resolver_equipo(conn: sqlite3.Connection, nombre: str) -> int:
    """Traduce un nombre de openfootball al team_id del proyecto.

    Falla ruidosamente ante un nombre desconocido: un equipo mal mapeado
    duplicaria la ficha y falsearia la simulacion en silencio.
    """
    existente = resolve_alias(conn, nombre, SOURCE)
    if existente:
        return existente

    canonico = NOMBRES.get(nombre)
    if canonico is None:
        # Ultimo intento: comparar sin acentos contra los equipos ya conocidos.
        objetivo = _sin_acentos(nombre).lower()
        for (tid, conocido) in conn.execute("SELECT team_id, name FROM teams").fetchall():
            if _sin_acentos(conocido).lower() == objetivo:
                register_alias(conn, nombre, SOURCE, tid)
                return tid
        raise KeyError(
            f"Nombre de equipo no reconocido en openfootball: {nombre!r}. "
            f"Anadelo a NOMBRES en simliga/ingest/openfootball.py"
        )

    team_id = get_or_create_team(conn, canonico)
    register_alias(conn, nombre, SOURCE, team_id)
    return team_id


def ingest_season(
    conn: sqlite3.Connection,
    season: str,
    competition: str = COMP_LALIGA,
    force_download: bool = False,
    esperados: int | None = 380,
) -> int:
    """Carga el calendario de una temporada. Devuelve el numero de partidos."""
    texto = descargar(season, competition, force_download).read_text(encoding="utf-8")
    df = parse(texto, season)

    if esperados is not None and len(df) != esperados:
        raise ValueError(
            f"Se esperaban {esperados} partidos en {season} y se han parseado {len(df)}. "
            f"El formato del fichero puede haber cambiado."
        )

    n = 0
    for fila in df.itertuples(index=False):
        jugado = fila.home_goals is not None and not pd.isna(fila.home_goals)
        upsert_match(
            conn,
            competition=competition, season=season, stage="league",
            match_date=fila.match_date.strftime("%Y-%m-%d"), matchday=int(fila.matchday),
            home_team_id=_resolver_equipo(conn, fila.home),
            away_team_id=_resolver_equipo(conn, fila.away),
            home_goals=int(fila.home_goals) if jugado else None,
            away_goals=int(fila.away_goals) if jugado else None,
            status="played" if jugado else "scheduled",
            source=SOURCE,
        )
        n += 1

    conn.commit()
    return n
