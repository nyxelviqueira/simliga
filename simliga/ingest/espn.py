"""Fechas, horarios y resultados recientes desde la API publica de ESPN.

openfootball publica el emparejamiento de toda la temporada, pero con fechas de
marcador de posicion: los diez partidos de cada jornada el mismo domingo. LaLiga
confirma dia y hora unas semanas antes, y ESPN lo refleja.

    https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard?dates=...

Sin clave y con un solo aviso importante: **ESPN tambien usa marcadores de
posicion** para las jornadas lejanas (los diez partidos el mismo dia a las
18:00). No se puede dar por confirmada una fecha solo porque venga de aqui; eso
lo decide `provisional_dates`, que mira si la jornada esta repartida en varios
dias y horas.

Este modulo solo toca partidos que ya existen. openfootball crea el calendario
y football-data.co.uk sigue siendo la fuente mas rica de resultados, xG y
cuotas, pero ESPN suele marcar antes un partido como finalizado. Cuando ESPN
trae un evento completado, guardamos ese marcador como resultado provisional;
si football-data lo publica despues, su ingesta puede corregirlo.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import requests

from ..config import COMP_LALIGA, RAW_DIR
from .club_names import ResolverEquipos

SOURCE = "espn"
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
LIGAS = {COMP_LALIGA: "esp.1"}


def descargar(season: str, competition: str = COMP_LALIGA, force: bool = False) -> dict:
    """Marcador de toda la temporada en una sola peticion."""
    import json

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"espn_{competition}_{season}.json"
    if path.exists() and path.stat().st_size > 0 and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    inicio = int(season.split("-")[0])
    rango = f"{inicio}0701-{inicio + 1}0701"
    url = f"{BASE_URL}/{LIGAS[competition]}/scoreboard?dates={rango}&limit=1000"
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.json()


def _score(equipo: dict) -> int | None:
    valor = equipo.get("score")
    if valor is None or valor == "":
        return None
    try:
        return int(valor)
    except ValueError:
        return None


def parse(datos: dict) -> pd.DataFrame:
    """Extrae calendario y marcadores finales de la respuesta de ESPN."""
    filas = []
    for evento in datos.get("events", []):
        competicion = evento.get("competitions", [{}])[0]
        equipos = {c.get("homeAway"): c for c in competicion.get("competitors", [])}
        if not equipos.get("home") or not equipos.get("away"):
            continue

        estado = competicion.get("status", {}).get("type", {})
        completado = bool(estado.get("completed"))
        gl = _score(equipos["home"]) if completado else None
        gv = _score(equipos["away"]) if completado else None
        if completado and (gl is None or gv is None):
            completado = False
            gl = gv = None

        marca = pd.Timestamp(evento["date"])          # viene en UTC
        filas.append({
            "match_date": marca.strftime("%Y-%m-%d"),
            "kickoff_utc": marca.strftime("%H:%M"),
            "home": equipos["home"].get("team", {}).get("displayName"),
            "away": equipos["away"].get("team", {}).get("displayName"),
            "home_goals": gl,
            "away_goals": gv,
            "status": "played" if completado else "scheduled",
        })
    return pd.DataFrame(filas)


def update_schedule(
    conn: sqlite3.Connection,
    season: str,
    competition: str = COMP_LALIGA,
    force_download: bool = False,
) -> dict:
    """Actualiza calendario y resultados finalizados. Devuelve un recuento."""
    df = parse(descargar(season, competition, force_download))
    if df.empty:
        return {"actualizados": 0, "resultados": 0, "sin_encontrar": 0, "total": 0}

    resolver = ResolverEquipos(conn)
    actualizados = resultados = sin_encontrar = 0

    desconocidos: set[str] = set()
    for fila in df.itertuples(index=False):
        local = resolver.resolver_existente(fila.home)
        visitante = resolver.resolver_existente(fila.away)
        if local is None or visitante is None:
            # No se crea nada: este modulo solo actualiza filas que ya existen.
            desconocidos.update(n for n, t in ((fila.home, local), (fila.away, visitante))
                                if t is None)
            sin_encontrar += 1
            continue

        if pd.notna(fila.home_goals) and pd.notna(fila.away_goals):
            cur = conn.execute(
                """UPDATE matches
                   SET match_date = ?, kickoff_utc = ?, home_goals = ?,
                       away_goals = ?, status = 'played', source = ?
                   WHERE season = ? AND competition = ? AND stage = 'league'
                     AND home_team_id = ? AND away_team_id = ?""",
                (fila.match_date, fila.kickoff_utc, int(fila.home_goals),
                 int(fila.away_goals), SOURCE, season, competition, local, visitante),
            )
            if cur.rowcount:
                actualizados += cur.rowcount
                resultados += cur.rowcount
            else:
                sin_encontrar += 1
            continue

        cur = conn.execute(
            """UPDATE matches SET match_date = ?, kickoff_utc = ?
               WHERE season = ? AND competition = ? AND stage = 'league'
                 AND home_team_id = ? AND away_team_id = ?
                 AND home_goals IS NULL""",
            (fila.match_date, fila.kickoff_utc, season, competition, local, visitante),
        )
        if cur.rowcount:
            actualizados += cur.rowcount
            continue

        # Ya jugado: se le pone la hora, pero no se le toca la fecha. Esa la
        # manda el resultado, que es el que sabe cuando se disputo de verdad.
        cur = conn.execute(
            """UPDATE matches SET kickoff_utc = ?
               WHERE season = ? AND competition = ? AND stage = 'league'
                 AND home_team_id = ? AND away_team_id = ?""",
            (fila.kickoff_utc, season, competition, local, visitante),
        )
        if cur.rowcount:
            actualizados += cur.rowcount
        else:
            sin_encontrar += 1

    conn.commit()
    return {"actualizados": actualizados, "sin_encontrar": sin_encontrar,
            "resultados": resultados, "total": len(df),
            "nombres_desconocidos": sorted(desconocidos)}
