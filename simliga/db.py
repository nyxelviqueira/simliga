"""Capa de persistencia (SQLite).

Un unico esquema sirve para LaLiga, Segunda y (fase 2) las competiciones UEFA:
`matches` es la tabla central y `competition` + `stage` distinguen el torneo.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS teams (
    team_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,      -- nombre canonico
    country   TEXT NOT NULL DEFAULT 'ESP'
);

-- Mapea los nombres de cada fuente al nombre canonico.
CREATE TABLE IF NOT EXISTS team_aliases (
    alias   TEXT NOT NULL,
    source  TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(team_id),
    PRIMARY KEY (alias, source)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    competition  TEXT NOT NULL,          -- ESP1 | ESP2 | UCL | UEL | UECL
    season       TEXT NOT NULL,          -- '2024-25'
    stage        TEXT NOT NULL DEFAULT 'league',
    match_date   TEXT NOT NULL,          -- ISO 'YYYY-MM-DD'
    kickoff_utc  TEXT,                   -- 'HH:MM' en UTC; NULL si no se sabe
    matchday     INTEGER,
    home_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals   INTEGER,
    away_goals   INTEGER,
    status       TEXT NOT NULL DEFAULT 'scheduled',   -- scheduled | played
    source       TEXT,
    UNIQUE (competition, season, stage, home_team_id, away_team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(competition, season);

-- Cuotas de casa de apuestas: no alimentan el modelo, son el benchmark
-- de calibracion contra el que se mide si el simulador aporta algo.
CREATE TABLE IF NOT EXISTS match_odds (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    book     TEXT NOT NULL,
    odds_h   REAL, odds_d REAL, odds_a REAL,
    PRIMARY KEY (match_id, book)
);

-- Serie temporal del Elo: una fila por equipo y partido.
CREATE TABLE IF NOT EXISTS elo_ratings (
    match_id      INTEGER NOT NULL REFERENCES matches(match_id),
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    match_date    TEXT NOT NULL,
    rating_before REAL NOT NULL,
    rating_after  REAL NOT NULL,
    PRIMARY KEY (match_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_elo_team_date ON elo_ratings(team_id, match_date);

-- Campeones de copa. Importa porque el ganador de la Copa del Rey se lleva una
-- plaza de Europa League, y eso desplaza el reparto que sale de la liga. No hay
-- fuente gratuita de la Copa, asi que se introduce a mano.
CREATE TABLE IF NOT EXISTS cup_winners (
    season      TEXT NOT NULL,
    competition TEXT NOT NULL,      -- 'CDR' = Copa del Rey
    team_id     INTEGER NOT NULL REFERENCES teams(team_id),
    PRIMARY KEY (season, competition)
);

-- Resultados hipoteticos para probar escenarios ("y si el Betis gana estos
-- cuatro?"). Van en su propia tabla, nunca en `matches`: un resultado inventado
-- no debe poder confundirse con uno real ni colarse en el ajuste del modelo.
-- Solo entran en la clasificacion de partida de la simulacion.
CREATE TABLE IF NOT EXISTS scenario_results (
    match_id   INTEGER PRIMARY KEY REFERENCES matches(match_id),
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ajustes cualitativos por evento: cambio de entrenador, lesiones, fichajes.
-- No hay fuente gratuita fiable para esto, asi que se rellena a mano
-- (`simliga ajuste`) o con un scraper futuro.
CREATE TABLE IF NOT EXISTS team_adjustments (
    adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    season     TEXT NOT NULL,
    kind       TEXT NOT NULL,      -- coach_change | injuries | transfers | squad_depth
    elo_delta  REAL NOT NULL,      -- positivo refuerza, negativo debilita
    valid_from TEXT,               -- ISO; NULL = desde siempre
    valid_to   TEXT,               -- ISO; NULL = hasta el final
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_adjustments_team ON team_adjustments(team_id, season);

-- xG por equipo y partido (fase 3, Understat/FBref).
CREATE TABLE IF NOT EXISTS match_xg (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    xg       REAL,
    source   TEXT,
    PRIMARY KEY (match_id, team_id)
);
"""


# Columnas anadidas despues de que existieran bases de datos en uso. `CREATE
# TABLE IF NOT EXISTS` no las agrega a una tabla ya creada, asi que se aplican
# aparte al abrir.
MIGRACIONES = (
    ("matches", "kickoff_utc", "TEXT"),
)


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Abre (creando si hace falta) la base de datos y aplica el esquema."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _aplicar_migraciones(conn)
    return conn


def _aplicar_migraciones(conn: sqlite3.Connection) -> None:
    for tabla, columna, tipo in MIGRACIONES:
        existentes = {fila["name"] for fila in conn.execute(f"PRAGMA table_info({tabla})")}
        if columna not in existentes:
            conn.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {tipo}")
    conn.commit()


def get_or_create_team(conn: sqlite3.Connection, name: str, country: str = "ESP") -> int:
    row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,)).fetchone()
    if row:
        return row["team_id"]
    cur = conn.execute(
        "INSERT INTO teams (name, country) VALUES (?, ?)", (name, country)
    )
    return int(cur.lastrowid)


def register_alias(conn: sqlite3.Connection, alias: str, source: str, team_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
        (alias, source, team_id),
    )


def resolve_alias(conn: sqlite3.Connection, alias: str, source: str) -> int | None:
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ? AND source = ?", (alias, source)
    ).fetchone()
    return row["team_id"] if row else None


def upsert_match(conn: sqlite3.Connection, **kw) -> int:
    """Inserta o actualiza un partido; devuelve su match_id.

    La clave natural es (competition, season, stage, home, away), asi que
    re-ingerir la misma temporada actualiza resultados en vez de duplicar.
    """
    conn.execute(
        """
        INSERT INTO matches (competition, season, stage, match_date, matchday,
                             home_team_id, away_team_id, home_goals, away_goals,
                             status, source)
        VALUES (:competition, :season, :stage, :match_date, :matchday,
                :home_team_id, :away_team_id, :home_goals, :away_goals,
                :status, :source)
        ON CONFLICT (competition, season, stage, home_team_id, away_team_id)
        DO UPDATE SET
            -- Un resultado ya guardado no se borra nunca. Sin esto, refrescar el
            -- calendario (que para una temporada en curso no trae marcadores)
            -- dejaba en blanco todos los partidos ya jugados.
            home_goals = COALESCE(excluded.home_goals, matches.home_goals),
            away_goals = COALESCE(excluded.away_goals, matches.away_goals),
            status     = CASE
                             WHEN excluded.home_goals IS NOT NULL THEN 'played'
                             WHEN matches.home_goals  IS NOT NULL THEN 'played'
                             ELSE excluded.status
                         END,
            -- La fecha y la fuente solo las pisa quien aporte resultado, o
            -- cualquiera si el partido aun no se ha jugado: para un partido ya
            -- disputado manda la fecha real, no la provisional del calendario.
            match_date = CASE
                             WHEN excluded.home_goals IS NOT NULL
                               OR matches.home_goals IS NULL THEN excluded.match_date
                             ELSE matches.match_date
                         END,
            source     = CASE
                             WHEN excluded.home_goals IS NOT NULL
                               OR matches.home_goals IS NULL THEN excluded.source
                             ELSE matches.source
                         END,
            matchday   = COALESCE(excluded.matchday, matches.matchday)
        """,
        kw,
    )
    row = conn.execute(
        """SELECT match_id FROM matches
           WHERE competition = :competition AND season = :season AND stage = :stage
             AND home_team_id = :home_team_id AND away_team_id = :away_team_id""",
        kw,
    ).fetchone()
    return int(row["match_id"])


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_start_year(label: str) -> int:
    return int(label.split("-")[0])


def previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    inicio = int(season.split("-")[0])
    return f"{inicio - 1}-{str(inicio)[-2:]}"


def merge_teams(conn: sqlite3.Connection, from_id: int, into_id: int) -> dict:
    """Funde dos fichas del mismo club en una sola.

    Cuando una fuente cambia la grafia de un equipo (`La Coruna` ->
    `Dep. A Coruna`, `Celta B` -> `Celta de Vigo B`) se crea una ficha nueva y
    el historial del club queda partido en dos, con partidos duplicados. Esta
    funcion repunta todo a la ficha buena y descarta lo que quede repetido.

    Devuelve un recuento de lo que ha hecho, para poder revisarlo.
    """
    if from_id == into_id:
        return {"repuntados": 0, "duplicados_borrados": 0}

    # Un partido de la ficha vieja cuyo equivalente ya existe con la ficha buena
    # es un duplicado: se borra en vez de repuntarlo, o violaria la clave natural.
    duplicados = conn.execute(
        """SELECT viejo.match_id AS viejo_id, bueno.match_id AS bueno_id
           FROM matches viejo
           JOIN matches bueno
             ON bueno.competition = viejo.competition
            AND bueno.season = viejo.season
            AND bueno.stage = viejo.stage
            AND bueno.home_team_id = CASE WHEN viejo.home_team_id = :viejo
                                          THEN :bueno ELSE viejo.home_team_id END
            AND bueno.away_team_id = CASE WHEN viejo.away_team_id = :viejo
                                          THEN :bueno ELSE viejo.away_team_id END
            AND bueno.match_id <> viejo.match_id
           WHERE viejo.home_team_id = :viejo OR viejo.away_team_id = :viejo""",
        {"viejo": from_id, "bueno": into_id},
    ).fetchall()
    ids = [fila["viejo_id"] for fila in duplicados]
    if ids:
        for fila in duplicados:
            viejo = conn.execute("SELECT * FROM matches WHERE match_id = ?",
                                 (fila["viejo_id"],)).fetchone()
            conn.execute(
                """UPDATE matches
                   SET home_goals = COALESCE(?, home_goals),
                       away_goals = COALESCE(?, away_goals),
                       status = CASE WHEN ? IS NOT NULL THEN 'played' ELSE status END,
                       match_date = CASE WHEN ? IS NOT NULL THEN ? ELSE match_date END,
                       source = CASE WHEN ? IS NOT NULL THEN ? ELSE source END,
                       matchday = COALESCE(matchday, ?),
                       kickoff_utc = COALESCE(kickoff_utc, ?)
                   WHERE match_id = ?""",
                (
                    viejo["home_goals"], viejo["away_goals"],
                    viejo["home_goals"],
                    viejo["home_goals"], viejo["match_date"],
                    viejo["home_goals"], viejo["source"],
                    viejo["matchday"], viejo["kickoff_utc"],
                    fila["bueno_id"],
                ),
            )
            conn.execute("UPDATE OR IGNORE match_odds SET match_id = ? WHERE match_id = ?",
                         (fila["bueno_id"], fila["viejo_id"]))
            conn.execute("UPDATE OR IGNORE scenario_results SET match_id = ? WHERE match_id = ?",
                         (fila["bueno_id"], fila["viejo_id"]))
            conn.execute("UPDATE OR IGNORE match_xg SET match_id = ? WHERE match_id = ?",
                         (fila["bueno_id"], fila["viejo_id"]))
        marcas = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM match_odds WHERE match_id IN ({marcas})", ids)
        conn.execute(f"DELETE FROM scenario_results WHERE match_id IN ({marcas})", ids)
        conn.execute(f"DELETE FROM match_xg WHERE match_id IN ({marcas})", ids)
        conn.execute(f"DELETE FROM elo_ratings WHERE match_id IN ({marcas})", ids)
        conn.execute(f"DELETE FROM matches WHERE match_id IN ({marcas})", ids)

    cur_h = conn.execute("UPDATE matches SET home_team_id = ? WHERE home_team_id = ?",
                         (into_id, from_id))
    repuntados = cur_h.rowcount
    cur_a = conn.execute("UPDATE matches SET away_team_id = ? WHERE away_team_id = ?",
                         (into_id, from_id))
    repuntados += cur_a.rowcount

    conn.execute("UPDATE OR IGNORE team_aliases SET team_id = ? WHERE team_id = ?",
                 (into_id, from_id))
    conn.execute("DELETE FROM team_aliases WHERE team_id = ?", (from_id,))
    conn.execute("UPDATE OR IGNORE elo_ratings SET team_id = ? WHERE team_id = ?",
                 (into_id, from_id))
    conn.execute("DELETE FROM elo_ratings WHERE team_id = ?", (from_id,))
    conn.execute("DELETE FROM teams WHERE team_id = ?", (from_id,))
    conn.commit()

    return {"repuntados": repuntados, "duplicados_borrados": len(ids)}


# --------------------------------------------------------------- escenarios
def set_scenario_result(
    conn: sqlite3.Connection, match_id: int, home_goals: int, away_goals: int
) -> None:
    """Fija un resultado hipotetico para un partido aun sin jugar.

    Se niega a tocar un partido ya disputado: lo que paso, paso, y dejar que un
    escenario lo sobrescriba haria que la clasificacion "real" del panel dejase
    de serlo sin que nada lo indicase.
    """
    fila = conn.execute(
        "SELECT status, home_goals FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    if fila is None:
        raise KeyError(f"No existe el partido {match_id}")
    if fila["status"] == "played" or fila["home_goals"] is not None:
        raise ValueError("Ese partido ya se jugo: su resultado no se puede cambiar")
    if home_goals < 0 or away_goals < 0:
        raise ValueError("Los goles no pueden ser negativos")

    conn.execute(
        """INSERT INTO scenario_results (match_id, home_goals, away_goals)
           VALUES (?, ?, ?)
           ON CONFLICT (match_id) DO UPDATE SET
               home_goals = excluded.home_goals,
               away_goals = excluded.away_goals,
               created_at = datetime('now')""",
        (match_id, int(home_goals), int(away_goals)),
    )
    conn.commit()


def clear_scenario_result(conn: sqlite3.Connection, match_id: int) -> bool:
    cur = conn.execute("DELETE FROM scenario_results WHERE match_id = ?", (match_id,))
    conn.commit()
    return cur.rowcount > 0


def clear_all_scenarios(conn: sqlite3.Connection, season: str | None = None) -> int:
    if season is None:
        cur = conn.execute("DELETE FROM scenario_results")
    else:
        cur = conn.execute(
            """DELETE FROM scenario_results WHERE match_id IN
               (SELECT match_id FROM matches WHERE season = ?)""",
            (season,),
        )
    conn.commit()
    return cur.rowcount


def load_scenario_results(conn: sqlite3.Connection, season: str | None = None) -> dict:
    """{match_id: (goles_local, goles_visitante)} de los escenarios guardados.

    Descarta los de partidos que entretanto se hayan jugado de verdad: el
    resultado real siempre gana.
    """
    query = """SELECT s.match_id, s.home_goals, s.away_goals
               FROM scenario_results s JOIN matches m ON m.match_id = s.match_id
               WHERE m.home_goals IS NULL"""
    params: list = []
    if season:
        query += " AND m.season = ?"
        params.append(season)
    return {r["match_id"]: (r["home_goals"], r["away_goals"])
            for r in conn.execute(query, params).fetchall()}


# ------------------------------------------------------------ campeon de copa
def set_cup_winner(
    conn: sqlite3.Connection, season: str, team_id: int, competition: str = "CDR"
) -> None:
    """Registra quien gano la copa esa temporada."""
    conn.execute(
        """INSERT INTO cup_winners (season, competition, team_id) VALUES (?, ?, ?)
           ON CONFLICT (season, competition) DO UPDATE SET team_id = excluded.team_id""",
        (season, competition, team_id),
    )
    conn.commit()


def get_cup_winner(
    conn: sqlite3.Connection, season: str, competition: str = "CDR"
) -> str | None:
    """Nombre del campeon de copa, o None si no consta."""
    fila = conn.execute(
        """SELECT t.name FROM cup_winners c JOIN teams t ON t.team_id = c.team_id
           WHERE c.season = ? AND c.competition = ?""",
        (season, competition),
    ).fetchone()
    return fila["name"] if fila else None
