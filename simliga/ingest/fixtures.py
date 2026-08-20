"""Generacion de un calendario sintetico de liga.

football-data.co.uk solo publica partidos ya jugados, asi que para simular una
temporada que aun no ha empezado hace falta el calendario por otra via. Sin API
key disponible, esto genera un doble round-robin valido (metodo del circulo)
con una jornada por semana.

Sirve para tener el motor operativo hoy sobre 2026-27: la distribucion de
posiciones apenas depende del orden de los partidos cuando se simula desde la
jornada 0. En cuanto haya calendario real (API-Football / football-data.org)
debe sustituirse, porque el orden si importa para simular a mitad de temporada
y, sobre todo, para el efecto de fatiga por competicion europea entre semana.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from ..config import COMP_LALIGA
from ..db import upsert_match

SOURCE = "synthetic-round-robin"


def round_robin(team_ids: list[int]) -> list[list[tuple[int, int]]]:
    """Doble vuelta por el metodo del circulo: cada equipo juega a todos ida y vuelta."""
    teams = list(team_ids)
    if len(teams) % 2:
        teams.append(-1)  # equipo fantasma: quien le toca, descansa
    n = len(teams)

    first_half = []
    for _ in range(n - 1):
        pairs = []
        for i in range(n // 2):
            home, away = teams[i], teams[n - 1 - i]
            if -1 in (home, away):
                continue
            # Alterna la localia para que ningun equipo acumule locales seguidos.
            pairs.append((home, away) if len(first_half) % 2 == 0 else (away, home))
        first_half.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]

    second_half = [[(a, h) for h, a in day] for day in first_half]
    return first_half + second_half


def generate_league_fixtures(
    conn: sqlite3.Connection,
    team_ids: list[int],
    season: str,
    start_date: str,
    competition: str = COMP_LALIGA,
    days_between_rounds: int = 7,
) -> int:
    """Escribe en la base de datos un calendario completo sin resultados."""
    schedule = round_robin(sorted(team_ids))
    date = pd.Timestamp(start_date)
    n = 0
    for matchday, pairs in enumerate(schedule, start=1):
        for home, away in pairs:
            upsert_match(
                conn,
                competition=competition, season=season, stage="league",
                match_date=date.strftime("%Y-%m-%d"), matchday=matchday,
                home_team_id=home, away_team_id=away,
                home_goals=None, away_goals=None,
                status="scheduled", source=SOURCE,
            )
            n += 1
        date += pd.Timedelta(days=days_between_rounds)
    conn.commit()
    return n
