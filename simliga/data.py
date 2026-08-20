"""Acceso de lectura a la base de datos en forma de DataFrames."""
from __future__ import annotations

import sqlite3

import pandas as pd

from .config import COMP_LALIGA


def load_matches(
    conn: sqlite3.Connection,
    competitions: tuple[str, ...] = (COMP_LALIGA,),
    seasons: tuple[str, ...] | None = None,
    only_played: bool = True,
) -> pd.DataFrame:
    """Partidos en orden cronologico, con ids y nombres de equipo."""
    where = ["m.competition IN (%s)" % ",".join("?" * len(competitions))]
    params: list = list(competitions)
    if seasons:
        where.append("m.season IN (%s)" % ",".join("?" * len(seasons)))
        params += list(seasons)
    if only_played:
        where.append("m.status = 'played'")

    query = f"""
        SELECT m.match_id, m.competition, m.season, m.stage, m.match_date,
               m.kickoff_utc, m.matchday,
               m.home_team_id, m.away_team_id, m.home_goals, m.away_goals, m.status,
               m.live_home_goals, m.live_away_goals, m.live_detail,
               th.name AS home_team, ta.name AS away_team
        FROM matches m
        JOIN teams th ON th.team_id = m.home_team_id
        JOIN teams ta ON ta.team_id = m.away_team_id
        WHERE {' AND '.join(where)}
        ORDER BY m.match_date, m.match_id
    """
    df = pd.read_sql_query(query, conn, params=params)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df


def load_odds(conn: sqlite3.Connection, book: str | None = None) -> pd.DataFrame:
    """Cuotas 1X2 por partido (benchmark de calibracion, no entra en el modelo)."""
    query = "SELECT match_id, book, odds_h, odds_d, odds_a FROM match_odds"
    params: list = []
    if book:
        query += " WHERE book = ?"
        params.append(book)
    return pd.read_sql_query(query, conn, params=params)


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Clasificacion real a partir de partidos jugados (para validar el backtest).

    Aplica los desempates de LaLiga en su orden real: puntos, enfrentamiento
    directo entre los empatados (puntos y luego diferencia de goles en esa
    mini-liga), diferencia de goles general y goles a favor. El enfrentamiento
    directo NO es un detalle menor: en 2025-26 Levante y Mallorca acabaron
    empatados a 42 puntos y bajo el Mallorca pese a tener mejor diferencia
    general, porque el Levante le gano el head-to-head.
    """
    played = matches[matches["home_goals"].notna()]
    rows = []
    for _, m in played.iterrows():
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        hp, ap = (3, 0) if hg > ag else (0, 3) if ag > hg else (1, 1)
        rows.append((m["home_team"], hp, hg, ag))
        rows.append((m["away_team"], ap, ag, hg))

    df = pd.DataFrame(rows, columns=["team", "points", "gf", "ga"])
    table = df.groupby("team").agg(
        played=("points", "size"), points=("points", "sum"),
        gf=("gf", "sum"), ga=("ga", "sum"),
    )
    table["gd"] = table["gf"] - table["ga"]
    table = table.reset_index()

    stats = table.set_index("team")
    order = []
    for puntos in sorted(table["points"].unique(), reverse=True):
        empatados = table[table["points"] == puntos]["team"].tolist()
        order.extend(_break_tie(empatados, played, stats))

    table = table.set_index("team").loc[order].reset_index()
    table["position"] = range(1, len(table) + 1)
    return table


def _break_tie(teams: list[str], played: pd.DataFrame, stats: pd.DataFrame) -> list[str]:
    """Ordena un grupo de equipos empatados a puntos segun los criterios de LaLiga."""
    if len(teams) == 1:
        return teams

    entre_ellos = played[
        played["home_team"].isin(teams) & played["away_team"].isin(teams)
    ]
    h2h_pts = {t: 0 for t in teams}
    h2h_gd = {t: 0 for t in teams}
    for _, m in entre_ellos.iterrows():
        h, a = m["home_team"], m["away_team"]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        h2h_pts[h] += 3 if hg > ag else (1 if hg == ag else 0)
        h2h_pts[a] += 3 if ag > hg else (1 if hg == ag else 0)
        h2h_gd[h] += hg - ag
        h2h_gd[a] += ag - hg

    return sorted(
        teams,
        key=lambda t: (-h2h_pts[t], -h2h_gd[t], -stats.loc[t, "gd"], -stats.loc[t, "gf"]),
    )


def season_teams(conn: sqlite3.Connection, season: str, competition: str = COMP_LALIGA) -> list[str]:
    rows = conn.execute(
        """SELECT DISTINCT t.name FROM matches m
           JOIN teams t ON t.team_id = m.home_team_id
           WHERE m.competition = ? AND m.season = ? ORDER BY t.name""",
        (competition, season),
    ).fetchall()
    return [r[0] for r in rows]


def check_season_integrity(
    conn: sqlite3.Connection,
    season: str,
    competition: str = COMP_LALIGA,
    complete: bool = False,
) -> list[str]:
    """Comprueba que una temporada tiene la forma que debe tener.

    Existe porque un cambio de grafia en la fuente creo una ficha duplicada del
    Deportivo y metio un equipo 21 en LaLiga 2026-27 sin que nada fallase. Un
    recuento mal no debe llegar nunca hasta la simulacion: alli ya solo se ve
    como un resultado raro.

    Solo se denuncia lo que es inequivocamente un error. Que falten partidos no
    lo es: a mitad de temporada, o en una division cuyo calendario no se ha
    cargado, es lo normal. Con `complete=True` (temporada terminada) tambien se
    exige el recuento completo.
    """
    problemas = []
    total = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE competition = ? AND season = ?",
        (competition, season),
    ).fetchone()[0]
    if total == 0:
        return [f"{competition} {season}: no hay partidos"]

    equipos = conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT home_team_id AS t FROM matches WHERE competition = ? AND season = ?
               UNION SELECT away_team_id FROM matches WHERE competition = ? AND season = ?)""",
        (competition, season, competition, season),
    ).fetchone()[0]

    esperados = 20 if competition == COMP_LALIGA else 22
    partidos_esperados = esperados * (esperados - 1)

    if equipos > esperados:
        problemas.append(
            f"{competition} {season}: {equipos} equipos, no puede haber mas de {esperados} "
            f"(sintoma tipico de una ficha duplicada por un cambio de grafia)")
    if total > partidos_esperados:
        problemas.append(
            f"{competition} {season}: {total} partidos, no puede haber mas de "
            f"{partidos_esperados}")

    duplicados = conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT home_team_id, away_team_id FROM matches
               WHERE competition = ? AND season = ?
               GROUP BY home_team_id, away_team_id HAVING COUNT(*) > 1)""",
        (competition, season),
    ).fetchone()[0]
    if duplicados:
        problemas.append(f"{competition} {season}: {duplicados} emparejamientos repetidos")

    if complete:
        if equipos != esperados:
            problemas.append(f"{competition} {season}: {equipos} equipos, se esperaban {esperados}")
        if total != partidos_esperados:
            problemas.append(
                f"{competition} {season}: {total} partidos, se esperaban {partidos_esperados}")

    return problemas


def promoted_into(
    matches: pd.DataFrame, season: str, competition: str = COMP_LALIGA
) -> set[int]:
    """Equipos que juegan `competition` en `season` y no la jugaron la anterior.

    Se calcula de los partidos y no del estado interno del Elo. Deducirlo del
    rating parecia mas comodo y fallaba en un caso concreto: un equipo que bajo
    hace dos temporadas y vuelve a subir tenia registrada Primera como origen y
    se quedaba sin correccion.
    """
    inicio = int(season.split("-")[0])
    anterior = f"{inicio - 1}-{str(inicio)[-2:]}"

    def participantes(etiqueta: str) -> set[int]:
        sub = matches[(matches["season"] == etiqueta)
                      & (matches["competition"] == competition)]
        return set(sub["home_team_id"]) | set(sub["away_team_id"])

    return participantes(season) - participantes(anterior)
