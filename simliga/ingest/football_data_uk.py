"""Ingesta de resultados historicos desde football-data.co.uk.

Fuente libre, sin API key: un CSV por division y temporada con resultados
completos y cuotas de mercado. Cubre Primera (SP1) y Segunda (SP2); la Segunda
importa porque da Elo de arranque a los equipos recien ascendidos, que si no
entrarian en LaLiga sin ningun historico.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import requests

from ..config import COMP_LALIGA, COMP_SEGUNDA, RAW_DIR
from ..db import get_or_create_team, register_alias, season_label, upsert_match

SOURCE = "football-data.co.uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
DIVISIONS = {
    "SP1": COMP_LALIGA, "SP2": COMP_SEGUNDA,
    # Primeras divisiones europeas: dan Elo a los rivales continentales.
    "E0": "ENG1", "D1": "GER1", "I1": "ITA1", "F1": "FRA1", "N1": "NED1",
    "P1": "POR1", "B1": "BEL1", "T1": "TUR1", "G1": "GRE1", "SC0": "SCO1",
}
EUROPEAN_DIVISIONS = ("E0", "D1", "I1", "F1", "N1", "P1", "B1", "T1", "G1", "SC0")

# Pais de cada division, para poder casar equipos con los codigos de openfootball.
DIVISION_COUNTRY = {
    "SP1": "ESP", "SP2": "ESP", "E0": "ENG", "D1": "GER", "I1": "ITA", "F1": "FRA",
    "N1": "NED", "P1": "POR", "B1": "BEL", "T1": "TUR", "G1": "GRE", "SC0": "SCO",
}

# Nombre abreviado de la fuente -> nombre canonico que veran el JSON y el frontend.
CANONICAL_NAMES = {
    "Alaves": "Deportivo Alaves", "Albacete": "Albacete Balompié",
    "Alcorcon": "AD Alcorcon", "Alcoyano": "CD Alcoyano", "Almeria": "UD Almeria",
    "Amorebieta": "SD Amorebieta", "Andorra": "FC Andorra",
    "Ath Bilbao": "Athletic Club", "Ath Bilbao B": "Bilbao Athletic",
    "Ath Madrid": "Atletico de Madrid", "Barcelona": "FC Barcelona",
    "Barcelona B": "Barcelona Atletic", "Betis": "Real Betis", "Burgos": "Burgos CF",
    "Cadiz": "Cadiz CF", "Cartagena": "FC Cartagena", "Castellon": "CD Castellon",
    "Celta": "Celta de Vigo", "Ceuta": "AD Ceuta", "Cordoba": "Cordoba CF",
    "Cultural Leonesa": "Cultural Leonesa", "Leonesa": "Cultural Leonesa",
    "Eibar": "SD Eibar", "Elche": "Elche CF", "Eldense": "CD Eldense",
    "Espanol": "RCD Espanyol", "Extremadura UD": "Extremadura UD",
    "Ferrol": "Racing de Ferrol", "Fuenlabrada": "CF Fuenlabrada", "Getafe": "Getafe CF",
    "Gimnastic": "Gimnastic de Tarragona", "Girona": "Girona FC", "Granada": "Granada CF",
    "Guadalajara": "CD Guadalajara", "Hercules": "Hercules CF", "Huesca": "SD Huesca",
    "Ibiza": "UD Ibiza", "Jaen": "Real Jaen", "La Coruna": "Deportivo de La Coruna",
    "Dep. A Coruna": "Deportivo de La Coruna",   # grafia usada desde 2026-27
    "Celta B": "Celta de Vigo B",
    "Las Palmas": "UD Las Palmas", "Leganes": "CD Leganes", "Levante": "Levante UD",
    "Llagostera": "CF Llagostera", "Logrones": "UD Logrones", "Lorca": "Lorca FC",
    "Lugo": "CD Lugo", "Malaga": "Malaga CF", "Mallorca": "RCD Mallorca",
    "Mirandes": "CD Mirandes", "Murcia": "Real Murcia", "Numancia": "CD Numancia",
    "Osasuna": "CA Osasuna", "Oviedo": "Real Oviedo", "Ponferradina": "SD Ponferradina",
    "Rayo Majadahonda": "Rayo Majadahonda", "Real Madrid": "Real Madrid",
    "Real Madrid B": "Real Madrid Castilla", "Recreativo": "Recreativo de Huelva",
    "Reus Deportiu": "CF Reus Deportiu", "Sabadell": "CE Sabadell",
    "Salamanca": "UD Salamanca", "Santander": "Racing de Santander", "Sevilla": "Sevilla FC",
    "Sevilla B": "Sevilla Atletico", "Sociedad": "Real Sociedad",
    "Sociedad B": "Real Sociedad B", "Sp Gijon": "Sporting de Gijon",
    "Tenerife": "CD Tenerife", "UCAM Murcia": "UCAM Murcia", "Valencia": "Valencia CF",
    "Valladolid": "Real Valladolid", "Vallecano": "Rayo Vallecano",
    "Villarreal": "Villarreal CF", "Villarreal B": "Villarreal B", "Xerez": "Xerez CD",
    "Zaragoza": "Real Zaragoza",
}

# Preferencia de columnas de cuotas: cierre promedio > apertura promedio > Bet365.
ODDS_COLUMNS = [
    ("avg_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("avg_opening", ("AvgH", "AvgD", "AvgA")),
    ("b365", ("B365H", "B365D", "B365A")),
]


def season_code(start_year: int) -> str:
    """2024 -> 2425 (formato de la URL de football-data)."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def download_season(start_year: int, division: str, force: bool = False) -> Path:
    """Descarga (con cache en disco) el CSV de una division y temporada."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{division}_{season_code(start_year)}.csv"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    url = f"{BASE_URL}/{season_code(start_year)}/{division}.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _parse_dates(raw: pd.Series) -> pd.Series:
    """La fuente mezcla formatos de fecha de 2 y 4 digitos segun la temporada."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(raw, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(fallback)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["date"] = _parse_dates(df["Date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def canonical(name: str, division: str = "SP1") -> str:
    """Nombre canonico. Solo hay mapa curado para las divisiones espanolas.

    Para el resto de ligas se usa el nombre de la fuente tal cual: solo se
    necesitan como rivales con un rating Elo, no se muestran en la tabla, y
    curar 400 nombres mas no aportaria nada.

    Un nombre espanol desconocido se avisa por consola: la fuente cambia de
    grafia de vez en cuando (en 2026-27 el Deportivo paso de `La Coruna` a
    `Dep. A Coruna`) y, sin aviso, se crea una ficha duplicada que parte en dos
    el historial del equipo y mete un equipo de mas en la tabla.
    """
    name = name.strip()
    if division not in ("SP1", "SP2"):
        return name
    if name not in CANONICAL_NAMES:
        _AVISOS.add(name)
        print(f"  aviso: nombre espanol no catalogado: {name!r} "
              f"(anadelo a CANONICAL_NAMES si es un equipo ya conocido)")
    return CANONICAL_NAMES.get(name, name)


_AVISOS: set[str] = set()


def ingest_season(
    conn: sqlite3.Connection, start_year: int, division: str, force_download: bool = False
) -> int:
    """Carga una temporada-division en la base de datos. Devuelve el nº de partidos."""
    competition = DIVISIONS[division]
    season = season_label(start_year)
    df = load_csv(download_season(start_year, division, force=force_download))

    n = 0
    for _, row in df.iterrows():
        ids = {}
        for side in ("Home", "Away"):
            raw_name = str(row[f"{side}Team"]).strip()
            team_id = get_or_create_team(conn, canonical(raw_name, division),
                                         DIVISION_COUNTRY.get(division, "ESP"))
            register_alias(conn, raw_name, SOURCE, team_id)
            ids[side] = team_id

        match_id = upsert_match(
            conn,
            competition=competition,
            season=season,
            stage="league",
            match_date=row["date"].strftime("%Y-%m-%d"),
            matchday=None,
            home_team_id=ids["Home"],
            away_team_id=ids["Away"],
            home_goals=int(row["FTHG"]),
            away_goals=int(row["FTAG"]),
            status="played",
            source=SOURCE,
        )
        _store_odds(conn, match_id, row)
        n += 1

    conn.commit()
    return n


def _store_odds(conn: sqlite3.Connection, match_id: int, row: pd.Series) -> None:
    for book, cols in ODDS_COLUMNS:
        if not all(c in row.index for c in cols):
            continue
        values = [row[c] for c in cols]
        if any(pd.isna(v) or float(v) <= 1.0 for v in values):
            continue
        conn.execute(
            """INSERT OR REPLACE INTO match_odds (match_id, book, odds_h, odds_d, odds_a)
               VALUES (?, ?, ?, ?, ?)""",
            (match_id, book, *[float(v) for v in values]),
        )
        return  # solo guardamos la mejor fuente disponible por partido


def ingest_range(
    conn: sqlite3.Connection,
    first_year: int,
    last_year: int,
    divisions: tuple[str, ...] = ("SP1", "SP2"),
    force_download: bool = False,
) -> dict[str, int]:
    """Ingesta masiva por rango de años de inicio de temporada."""
    counts: dict[str, int] = {}
    for year in range(first_year, last_year + 1):
        for div in divisions:
            try:
                counts[f"{div} {season_label(year)}"] = ingest_season(
                    conn, year, div, force_download
                )
            except (requests.HTTPError, FileNotFoundError) as exc:
                counts[f"{div} {season_label(year)}"] = 0
                print(f"  aviso: {div} {season_label(year)} no disponible ({exc})")
    return counts
