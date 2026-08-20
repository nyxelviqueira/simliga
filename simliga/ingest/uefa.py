"""Ingesta de las competiciones UEFA desde openfootball/champions-league.

El repositorio publica un fichero por competicion y temporada con el formato
nuevo (liguilla unica de 36 equipos desde 2024-25) y las eliminatorias:

    ▪ League, Matchday 1
      Tue Sep 16 2025
        18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)

    ▪ Playoffs, Matchday 1
    ▪ Finals, Round of 16
    ▪ Finals, Quarterfinals / Semifinals / Final

El codigo de pais entre parentesis identifica a los equipos espanoles sin
depender del nombre, y los nombres se cruzan con el resto de fuentes mediante
`club_names.ResolverEquipos`.

Las rondas previas (`clq.txt`, `elq.txt`, `confq.txt`) no se ingieren: los
equipos espanoles nunca las juegan y solo anadirian ruido al pool de Elo.
"""
from __future__ import annotations

import re
import sqlite3

import pandas as pd
import requests

from ..config import COMP_UCL, COMP_UECL, COMP_UEL, RAW_DIR
from ..db import upsert_match
from .club_names import ResolverEquipos

SOURCE = "openfootball/champions-league"
BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"
FICHEROS = {COMP_UCL: "cl.txt", COMP_UEL: "el.txt", COMP_UECL: "conf.txt"}

MESES = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# Fases tal y como las nombra el fichero -> etiqueta interna.
FASES = {
    "round of 16": "R16",
    "quarterfinals": "QF",
    "semifinals": "SF",
    "final": "F",
}

RE_SECCION = re.compile(r"^\s*[▪*]\s*(.+?)\s*$")
RE_JORNADA = re.compile(r"(?:League|Playoffs)\s*,\s*Matchday\s*(\d+)", re.IGNORECASE)
RE_FECHA = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
RE_PARTIDO = re.compile(
    r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?"
    r"(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)"
    r"(?:\s+(\d+)\s*-\s*(\d+))?"
)


def descargar(season: str, competition: str, force: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"uefa_{competition}_{season}.txt"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    resp = requests.get(f"{BASE_URL}/{season}/{FICHEROS[competition]}", timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _fase_de_seccion(seccion: str) -> tuple[str, int | None]:
    """Traduce el titulo de una seccion a (fase, jornada)."""
    m = RE_JORNADA.search(seccion)
    if m:
        jornada = int(m.group(1))
        fase = "playoff" if "playoff" in seccion.lower() else "league_phase"
        return fase, jornada

    resto = seccion.lower().replace("finals,", "").strip()
    if "playoff" in resto:
        # Segun la temporada el fichero escribe "Playoffs" o "Playoffs, Matchday N".
        return "playoff", None
    return FASES.get(resto, resto.replace(" ", "_")), None


def parse(texto: str, season: str) -> pd.DataFrame:
    """Convierte el fichero en un DataFrame de partidos con fase y fecha."""
    año_inicio = int(season.split("-")[0])
    filas = []
    fase, jornada, fecha = None, None, None

    for linea in texto.splitlines():
        if not linea.strip() or linea.lstrip().startswith(("#", "=")):
            continue

        m = RE_SECCION.match(linea)
        if m and not RE_FECHA.match(linea):
            fase, jornada = _fase_de_seccion(m.group(1))
            continue

        m = RE_FECHA.match(linea)
        if m:
            mes = MESES.get(m.group(1))
            if mes is None:
                continue
            año = m.group(3) or (año_inicio if mes >= 7 else año_inicio + 1)
            fecha = pd.Timestamp(int(año), mes, int(m.group(2)))
            continue

        if fase is None or fecha is None:
            continue

        m = RE_PARTIDO.match(linea)
        if m:
            local, pais_l, visitante, pais_v, hg, ag = m.groups()
            filas.append({
                "stage": fase, "matchday": jornada, "match_date": fecha,
                "home": local.strip(), "home_country": pais_l,
                "away": visitante.strip(), "away_country": pais_v,
                "home_goals": int(hg) if hg is not None else None,
                "away_goals": int(ag) if ag is not None else None,
            })

    return pd.DataFrame(filas)


def ingest_season(
    conn: sqlite3.Connection,
    season: str,
    competition: str = COMP_UCL,
    force_download: bool = False,
    resolver: ResolverEquipos | None = None,
) -> int:
    """Carga una competicion UEFA de una temporada. Devuelve el numero de partidos."""
    texto = descargar(season, competition, force_download).read_text(
        encoding="utf-8", errors="replace")
    df = parse(texto, season)
    if df.empty:
        return 0

    resolver = resolver or ResolverEquipos(conn)
    n = 0
    for fila in df.itertuples(index=False):
        # pandas convierte los None de una columna numerica en NaN, asi que
        # `is not None` daria verdadero para un partido aun sin jugar: sin este
        # `notna` un sorteo recien publicado entraria entero como disputado.
        jugado = pd.notna(fila.home_goals)
        # Las dos manos de una eliminatoria comparten fase pero invierten la
        # localia, asi que la clave natural (fase, local, visitante) no choca.
        upsert_match(
            conn,
            competition=competition, season=season, stage=fila.stage,
            match_date=fila.match_date.strftime("%Y-%m-%d"),
            matchday=fila.matchday,
            home_team_id=resolver.resolver(fila.home, fila.home_country),
            away_team_id=resolver.resolver(fila.away, fila.away_country),
            home_goals=int(fila.home_goals) if jugado else None,
            away_goals=int(fila.away_goals) if jugado else None,
            status="played" if jugado else "scheduled",
            source=SOURCE,
        )
        n += 1

    conn.commit()
    return n


def ingest_range(
    conn: sqlite3.Connection,
    seasons: tuple[str, ...],
    competitions: tuple[str, ...] = (COMP_UCL, COMP_UEL, COMP_UECL),
    force_download: bool = False,
) -> dict[str, int]:
    """Ingesta masiva. Una temporada-competicion ausente no interrumpe el resto."""
    resolver = ResolverEquipos(conn)
    counts: dict[str, int] = {}
    for season in seasons:
        for comp in competitions:
            clave = f"{comp} {season}"
            try:
                counts[clave] = ingest_season(conn, season, comp, force_download, resolver)
            except requests.HTTPError:
                counts[clave] = 0
    return counts
