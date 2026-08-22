"""Construccion del JSON de salida.

Este JSON es el contrato con el frontend: su esquema esta documentado en
`docs/data-contract.md` y versionado en `SCHEMA_VERSION`. La estructura ya
contempla las competiciones europeas (fase 2) para que quien construya el
dashboard no tenga que rehacerlo cuando se anadan.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data import league_table
from ..model.dixon_coles import DixonColesFit
from ..sim.league import LeagueSimResult
from ..sim.uefa import UefaSimResult
from .team_identity import team_identity

SCHEMA_VERSION = "1.12.0"
ENGINE_VERSION = "0.7.0"

COMPETITION_NAMES = {
    "ESP1": "LaLiga EA Sports",
    "ESP2": "LaLiga Hypermotion",
    "UCL": "UEFA Champions League",
    "UEL": "UEFA Europa League",
    "UECL": "UEFA Conference League",
}

def laliga_qualification_note(ucl_slots: int) -> str:
    if ucl_slots >= 5:
        return (
            "Reparto europeo provisional: se incluye una 5a plaza Champions por "
            "coeficiente UEFA (EPS). Europa League y Conference tambien pueden "
            "moverse por la Copa del Rey."
        )
    return (
        "Reparto europeo provisional: LaLiga tiene 4 plazas fijas de Champions; "
        "la 5a depende de que Espana obtenga una plaza extra por coeficiente UEFA "
        "(EPS). Europa League y Conference tambien pueden moverse por la Copa del Rey."
    )

PERCENTILES = {"p05": 5, "p25": 25, "median": 50, "p75": 75, "p95": 95}


def _distribution(values: np.ndarray) -> dict:
    """Resumen de una distribucion simulada: media, desviacion y percentiles."""
    out = {"mean": round(float(values.mean()), 2), "sd": round(float(values.std()), 2)}
    for name, q in PERCENTILES.items():
        out[name] = float(np.percentile(values, q))
    return out


def _current_standings(
    played: pd.DataFrame,
    team_ids: list[int],
    real_ids: set | None = None,
) -> dict[int, dict]:
    """Clasificacion de partida de la simulacion.

    Incluye los resultados hipoteticos, porque son los puntos con los que la
    simulacion arranca de verdad; `played_real` y `scenario_matches` permiten
    saber cuantos de esos puntos se ganaron de verdad y cuantos son supuestos.
    Sin esa distincion, un panel con escenarios activos ensenaria una tabla que
    parece real y no lo es.
    """
    state = {t: {"played": 0, "played_real": 0, "scenario_matches": 0,
                 "points": 0, "goals_for": 0, "goals_against": 0} for t in team_ids}
    for m in played.itertuples(index=False):
        if pd.isna(m.home_goals):
            continue
        es_real = real_ids is None or int(m.match_id) in real_ids
        hg, ag = int(m.home_goals), int(m.away_goals)
        h, a = int(m.home_team_id), int(m.away_team_id)
        for tid, gf, ga in ((h, hg, ag), (a, ag, hg)):
            if tid not in state:
                continue
            state[tid]["played"] += 1
            state[tid]["played_real" if es_real else "scenario_matches"] += 1
            state[tid]["goals_for"] += gf
            state[tid]["goals_against"] += ga
            state[tid]["points"] += 3 if gf > ga else (1 if gf == ga else 0)

    ordered = sorted(
        team_ids,
        key=lambda t: (
            -state[t]["points"],
            -(state[t]["goals_for"] - state[t]["goals_against"]),
            -state[t]["goals_for"],
        ),
    )
    for pos, tid in enumerate(ordered, start=1):
        state[tid]["goal_difference"] = state[tid]["goals_for"] - state[tid]["goals_against"]
        state[tid]["position"] = pos if state[tid]["played"] else None
    return state


def build_league_block(
    result: LeagueSimResult,
    fit: DixonColesFit,
    names: dict[int, str],
    elo: dict[int, float],
    played: pd.DataFrame,
    pending: pd.DataFrame,
    cfg: Config,
    competition: str = "ESP1",
    real_played: pd.DataFrame | None = None,
) -> dict:
    """Bloque de una competicion de liga (LaLiga) para el JSON de salida."""
    n = len(result.team_ids)
    real_ids = (set(real_played["match_id"].astype(int))
                if real_played is not None else None)
    standings = _current_standings(played, result.team_ids, real_ids)
    ucl, uel, uecl = cfg.sim.ucl_slots, cfg.sim.uel_slots, cfg.sim.uecl_slots
    releg_from = n - cfg.sim.relegation_slots + 1
    # La franja de en medio: ni Europa ni descenso. No es el complemento de
    # "desciende", que incluiria tambien a los europeos; es el "temporada
    # tranquila", que para la mayoria de los equipos es el desenlace mas
    # probable y hasta ahora no aparecia por ningun lado.
    mid_from, mid_to = ucl + uel + uecl + 1, releg_from - 1

    p_pos = result.position_probabilities()
    teams = []
    for i, tid in enumerate(result.team_ids):
        pos = result.positions[:, i]
        name = names.get(tid, str(tid))
        teams.append({
            "team_id": int(tid),
            "name": name,
            **team_identity(name),
            "current": standings[tid],
            "ratings": {
                "elo": round(float(elo.get(tid, float("nan"))), 1),
                "attack": round(float(fit.attack[fit.index_of(tid)]), 4),
                "defence": round(float(fit.defence[fit.index_of(tid)]), 4),
            },
            "projection": {
                "points": _distribution(result.points[:, i]),
                "goal_difference": _distribution(result.goal_diff[:, i]),
                "position": {
                    "mean": round(float(pos.mean()), 2),
                    "mode": int(np.bincount(pos).argmax()),
                    "p05": float(np.percentile(pos, 5)),
                    "p95": float(np.percentile(pos, 95)),
                },
                "position_probabilities": [round(float(x), 5) for x in p_pos[i]],
            },
            "outcomes": {
                "title": round(float((pos == 1).mean()), 5),
                "ucl": round(float(((pos >= 1) & (pos <= ucl)).mean()), 5),
                "uel": round(float(((pos > ucl) & (pos <= ucl + uel)).mean()), 5),
                "uecl": round(float(((pos > ucl + uel) & (pos <= ucl + uel + uecl)).mean()), 5),
                "european_qualification": round(float((pos <= ucl + uel + uecl).mean()), 5),
                "mid_table": round(float(((pos >= mid_from) & (pos <= mid_to)).mean()), 5)
                             if mid_to >= mid_from else 0.0,
                "relegation": round(float((pos >= releg_from).mean()), 5),
            },
        })

    teams.sort(key=lambda t: t["projection"]["position"]["mean"])
    return {
        "code": competition,
        "type": "league",
        "name": COMPETITION_NAMES.get(competition, competition),
        "n_teams": n,
        "qualification_rules": {
            "ucl": [1, ucl],
            "uel": [ucl + 1, ucl + uel],
            "uecl": [ucl + uel + 1, ucl + uel + uecl],
            "mid_table": [mid_from, mid_to],
            "relegation": [releg_from, n],
        },
        "qualification_note": laliga_qualification_note(ucl),
        "matches_played": int(len(played)),
        "matches_played_real": int(len(real_played)) if real_played is not None
                               else int(len(played)),
        "matches_remaining": int(len(pending)),
        "teams": teams,
    }


# Nombres de ronda en el JSON, mas explicitos que las etiquetas internas.
STAGE_LABELS = {"R16": "round_of_16", "QF": "quarter_finals",
                "SF": "semi_finals", "F": "final", "winner": "winner"}


def build_european_block(
    result: UefaSimResult,
    names: dict[int, str],
    countries: dict[int, str],
    elo: dict[int, float],
    played: pd.DataFrame,
    pending: pd.DataFrame,
    competition: str,
    direct_slots: int = 8,
    playoff_slots: int = 16,
    provisional: dict | None = None,
) -> dict:
    """Bloque de una competicion UEFA con formato de liguilla unica.

    `provisional` se rellena cuando el sorteo aun no esta publicado y la
    proyeccion se ha hecho sobre un campo estimado. En ese caso las cifras
    valen para hacerse una idea, no son la simulacion del torneo real, y el
    frontend debe decirlo.
    """
    n = len(result.team_ids)
    corte = direct_slots + playoff_slots
    liguilla = result.league_phase_outcomes(direct_slots, playoff_slots)
    fases = result.stage_probabilities()
    standings = _current_standings(played, result.team_ids)

    equipos = []
    for i, tid in enumerate(result.team_ids):
        pos = result.league_positions[:, i]
        probs_pos = np.bincount(pos, minlength=n + 1)[1:] / result.n_sims
        name = names.get(tid, str(tid))
        equipos.append({
            "team_id": int(tid),
            "name": name,
            **team_identity(name),
            "country": countries.get(tid, "?"),
            "current": standings[tid],
            "ratings": {"elo": round(float(elo.get(tid, float("nan"))), 1)},
            "league_phase": {
                "expected_points": round(float(result.league_points[:, i].mean()), 2),
                "expected_position": round(float(pos.mean()), 2),
                "position_probabilities": [round(float(x), 5) for x in probs_pos],
                "p_direct_to_r16": round(float(liguilla["direct_to_r16"][i]), 5),
                "p_playoff": round(float(liguilla["playoff"][i]), 5),
                "p_eliminated": round(float(liguilla["eliminated_in_league_phase"][i]), 5),
            },
            "stage_probabilities": {
                etiqueta: round(float(fases[fase][i]), 5)
                for fase, etiqueta in STAGE_LABELS.items() if fase in fases
            },
        })

    equipos.sort(key=lambda e: e["league_phase"]["expected_position"])
    return {
        "code": competition,
        "type": "uefa_league_phase",
        "name": COMPETITION_NAMES.get(competition, competition),
        "n_teams": n,
        "spanish_teams_only": False,
        "provisional": provisional,
        "qualification_rules": {
            "direct_to_r16": [1, direct_slots],
            "playoff": [direct_slots + 1, corte],
            "eliminated": [corte + 1, n],
        },
        "matches_played": int(len(played)),
        "matches_remaining": int(len(pending)),
        "teams": equipos,
    }


def provisional_dates(pending: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Marca los partidos cuya fecha aun no es la definitiva.

    LaLiga confirma dia y hora unas semanas antes. Hasta entonces las fuentes
    ponen un marcador de posicion, y todas lo hacen igual: la jornada entera el
    mismo dia a la misma hora. Tres indicios:

    1. La fecha ya paso y el partido sigue sin jugarse.
    2. Tres o mas partidos de la jornada comparten dia **y hora** exactos.
    3. Sin hora conocida, tres o mas comparten dia.

    Limitacion conocida: en las dos ultimas jornadas LaLiga si juega los diez
    partidos a la vez, asi que ahi el criterio no puede distinguir un horario de
    verdad de uno provisional. Faltando nueve meses, siempre es provisional.
    """
    if pending.empty:
        return pd.Series(dtype=bool)

    caducada = pending["match_date"] < pd.Timestamp(as_of)

    amontonada = pd.Series(False, index=pending.index)
    if "matchday" in pending.columns:
        con_jornada = pending.dropna(subset=["matchday"])
        tiene_hora = "kickoff_utc" in pending.columns
        for _, grupo in con_jornada.groupby("matchday"):
            claves = (grupo["match_date"].astype(str) + " " + grupo["kickoff_utc"].astype(str)
                      if tiene_hora else grupo["match_date"].astype(str))
            repetidas = claves.value_counts()
            sospechosas = repetidas[repetidas >= 3].index
            amontonada.loc[grupo.index[claves.isin(sospechosas)]] = True

    return caducada | amontonada


def build_fixtures_block(
    fit: DixonColesFit,
    pending: pd.DataFrame,
    names: dict[int, str],
    limit: int | None = 20,
    as_of: pd.Timestamp | None = None,
) -> list[dict]:
    """Probabilidades 1X2 y goles esperados de los proximos partidos."""
    upcoming = pending.sort_values(["match_date", "match_id"])
    provisional = provisional_dates(pending, as_of or pending["match_date"].min())
    if limit:
        upcoming = upcoming.head(limit)

    out = []
    for m in upcoming.itertuples():
        h, a = int(m.home_team_id), int(m.away_team_id)
        home_name = names.get(h, str(h))
        away_name = names.get(a, str(a))
        p_home, p_draw, p_away = fit.probs_1x2(h, a)
        lam_h, lam_a = fit.rates(h, a)
        matchday = getattr(m, "matchday", None)
        out.append({
            "match_id": int(m.match_id),
            "competition": m.competition,
            "matchday": int(matchday) if pd.notna(matchday) else None,
            "date": pd.Timestamp(m.match_date).strftime("%Y-%m-%d"),
            "kickoff_utc": _hora_iso(m),
            # La fecha aun no es definitiva: LaLiga no ha confirmado el horario.
            "date_provisional": bool(provisional.get(m.Index, False)),
            "status": getattr(m, "status", "scheduled"),
            "live_home_goals": (
                None if pd.isna(getattr(m, "live_home_goals", None))
                else int(getattr(m, "live_home_goals"))
            ),
            "live_away_goals": (
                None if pd.isna(getattr(m, "live_away_goals", None))
                else int(getattr(m, "live_away_goals"))
            ),
            "live_detail": (
                None if pd.isna(getattr(m, "live_detail", None))
                else str(getattr(m, "live_detail"))
            ),
            "home_team": {"team_id": h, "name": home_name, **team_identity(home_name)},
            "away_team": {"team_id": a, "name": away_name, **team_identity(away_name)},
            "probabilities": {"home": round(p_home, 4), "draw": round(p_draw, 4),
                              "away": round(p_away, 4)},
            "expected_goals": {"home": round(float(lam_h), 3), "away": round(float(lam_a), 3)},
        })
    return out


def _hora_iso(fila) -> str | None:
    """Instante del saque inicial en ISO UTC, o None si no se conoce.

    Se emite en UTC y no en hora espanola a proposito: el cambio de hora cae a
    mitad de temporada, y dejar la conversion al frontend (que sabe de husos)
    evita tener que acertar con el cambio de octubre en el servidor.
    """
    hora = getattr(fila, "kickoff_utc", None)
    if hora is None or (isinstance(hora, float) and pd.isna(hora)):
        return None
    fecha = pd.Timestamp(fila.match_date).strftime("%Y-%m-%d")
    return f"{fecha}T{hora}:00Z"


def _texto(valor) -> str | None:
    """Un campo de texto que puede venir vacio de la base o como NaN de pandas."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip()
    return texto or None


def build_standings_block(
    played: pd.DataFrame,
    all_teams: list[str],
    cfg: Config,
) -> dict:
    """Clasificacion real: solo resultados de verdad, con los desempates de LaLiga.

    Deliberadamente **no** incluye los resultados hipoteticos. Una tabla titulada
    "clasificacion real" que colase supuestos dejaria de merecer el nombre; para
    ver el efecto de un escenario esta la proyeccion, que si los cuenta.
    """
    n_teams = len(all_teams)

    # Partes que la clasificacion necesita y `league_table` no calcula:
    # ganados/empatados/perdidos y la forma reciente.
    detalle: dict[str, dict] = {}
    # Antes de la primera jornada no hay nada que recorrer, y ordenar un
    # DataFrame vacio por columnas que no existen revienta.
    en_orden = (played.sort_values(["match_date", "match_id"]) if len(played)
                else played)
    for m in en_orden.itertuples(index=False):
        if pd.isna(m.home_goals):
            continue
        hg, ag = int(m.home_goals), int(m.away_goals)
        for equipo, propios, contrarios in ((m.home_team, hg, ag), (m.away_team, ag, hg)):
            d = detalle.setdefault(equipo, {"won": 0, "drawn": 0, "lost": 0, "form": []})
            if propios > contrarios:
                d["won"] += 1
                d["form"].append("G")
            elif propios == contrarios:
                d["drawn"] += 1
                d["form"].append("E")
            else:
                d["lost"] += 1
                d["form"].append("P")

    ucl, uel, uecl = cfg.sim.ucl_slots, cfg.sim.uel_slots, cfg.sim.uecl_slots
    releg_from = n_teams - cfg.sim.relegation_slots + 1

    # La tabla incluye a los veinte equipos, tambien a los que aun no han
    # jugado: una clasificacion a la que le faltan filas no es una clasificacion.
    marcadores = {}
    if not played.empty:
        for fila in league_table(played).itertuples(index=False):
            marcadores[fila.team] = fila

    orden = sorted(
        all_teams,
        key=lambda equipo: (
            marcadores[equipo].position if equipo in marcadores else n_teams + 1,
            equipo,
        ),
    )

    filas = []
    for posicion, equipo in enumerate(orden, start=1):
        d = detalle.get(equipo, {"won": 0, "drawn": 0, "lost": 0, "form": []})
        m = marcadores.get(equipo)
        filas.append({
            "position": posicion,
            "team": equipo,
            **team_identity(equipo),
            "played": int(m.played) if m is not None else 0,
            "won": d["won"], "drawn": d["drawn"], "lost": d["lost"],
            "goals_for": int(m.gf) if m is not None else 0,
            "goals_against": int(m.ga) if m is not None else 0,
            "goal_difference": int(m.gd) if m is not None else 0,
            "points": int(m.points) if m is not None else 0,
            # Los cinco ultimos, del mas reciente al mas antiguo.
            "form": list(reversed(d["form"][-5:])),
        })

    return {
        "rows": filas,
        "n_teams": n_teams,
        "matches_played": int(len(played)),
        "complete": len(played) == n_teams * (n_teams - 1),
        "qualification_rules": {
            "ucl": [1, ucl],
            "uel": [ucl + 1, ucl + uel],
            "uecl": [ucl + uel + 1, ucl + uel + uecl],
            "relegation": [releg_from, n_teams],
        },
    }


def build_european_qualification(
    previous_matches: pd.DataFrame,
    previous_season: str,
    cfg: Config,
    cup_winner: str | None = None,
) -> dict | None:
    """Quien juega Europa esta temporada, segun la liga del año pasado.

    Se calcula con la clasificacion FINAL de la temporada anterior, que es lo
    que reparte las plazas de verdad. Usar la clasificacion en curso daria una
    lista sin ningun sentido (a la jornada 1, el lider provisional apareceria
    como clasificado para la Champions).

    El campeon de copa se lleva una plaza de Europa League. Si no se habia
    clasificado por liga, ocupa una de las dos y **todo el reparto de abajo se
    desplaza**: el sexto conserva la otra plaza de Europa League y el septimo
    cae a la Conference. Si ya estaba clasificado por liga, su plaza revierte y
    el reparto es el normal.
    """
    if previous_matches.empty:
        return None

    tabla = league_table(previous_matches)
    if len(tabla) < 8:
        return None

    # Aqui se reparte lo que dio la temporada PASADA, que es un hecho conocido,
    # no un supuesto sobre la que se esta jugando: por eso las plazas de
    # Champions son las de aquel año y no las fijas.
    ucl = cfg.sim.ucl_slots_temporada_anterior
    uel, uecl = cfg.sim.uel_slots, cfg.sim.uecl_slots
    por_liga = list(tabla.itertuples(index=False))
    plazas_base = ucl + uel + uecl
    campeon_ya_clasificado = cup_winner in {f.team for f in por_liga[:plazas_base]}
    entra_por_copa = bool(cup_winner) and not campeon_ya_clasificado
    revierte_a_liga = bool(cup_winner) and campeon_ya_clasificado

    plazas = ([("UCL", "Champions League")] * ucl
              + [("UEL", "Europa League")] * (uel + (1 if revierte_a_liga else 0))
              + [("UECL", "Conference League")] * uecl)

    equipos = []
    for fila, (codigo, nombre) in zip(por_liga, plazas):
        equipos.append({
            "team": fila.team,
            **team_identity(fila.team),
            "previous_position": int(fila.position),
            "previous_points": int(fila.points),
            "competition": codigo,
            "competition_name": nombre,
            "via": "liga",
        })

    if entra_por_copa:
        fila = next((f for f in por_liga if f.team == cup_winner), None)
        equipos.append({
            "team": cup_winner,
            **team_identity(cup_winner),
            "previous_position": int(fila.position) if fila is not None else None,
            "previous_points": int(fila.points) if fila is not None else None,
            "competition": "UEL",
            "competition_name": "Europa League",
            "via": "copa",
        })
        equipos.sort(key=lambda e: ("UCL", "UEL", "UECL").index(e["competition"]))

    if cup_winner:
        aviso = (f"{cup_winner} entra por la Copa del Rey, asi que el reparto de "
                 f"la liga se desplaza un puesto."
                 if entra_por_copa else
                 f"{cup_winner} gano la Copa del Rey pero ya estaba clasificado "
                 f"por liga, asi que su plaza revierte a la clasificacion.")
    else:
        aviso = ("Falta el campeon de Copa del Rey, que tambien entra en Europa "
                 "League y desplazaria el reparto. Se registra con "
                 "`simliga copa --temporada ... --campeon ...`.")

    return {
        "source_season": previous_season,
        "cup_winner": cup_winner,
        "teams": equipos,
        "caveat": aviso,
    }


def _empezada(bloque: dict, hoy: str) -> bool:
    """Si a esta jornada ya le ha llegado su turno.

    Basta con que su primer partido este fechado hoy o antes. Un aplazamiento
    mueve partidos hacia adelante, nunca hacia atras, asi que esa primera fecha
    no se le escapa a la jornada aunque le queden partidos sueltos meses
    despues.
    """
    return (min(p["date"] for p in bloque["matches"]) <= hoy
            or any(p["status"] == "played" for p in bloque["matches"]))


def _entera(bloque: dict) -> bool:
    return all(p["status"] == "played" for p in bloque["matches"])


def current_matchday(bloques: list[dict], as_of: pd.Timestamp) -> int | None:
    """La jornada que toca mirar hoy, para abrir el calendario por ella.

    No sirve "la primera que no este jugada entera", que es lo que hacia el
    panel: un partido aplazado deja su jornada incompleta durante meses y la
    vista se quedaba anclada en agosto mientras se jugaba la jornada 14. El
    caso no es teorico: el 22 de agosto de 2026 la jornada 1 tenia cuatro
    partidos movidos al fin de semana siguiente mientras se jugaba la 2.

    Asi que es la ultima jornada que ya ha empezado, y la siguiente en cuanto
    esa esta entera (sin esperar a que llegue su fin de semana). La busqueda se
    para en la primera que no ha empezado en vez de recorrerlas todas: un
    partido adelantado —que los hay— haria que una jornada de septiembre
    pareciera empezada en agosto y se saltaria las de en medio.
    """
    if not bloques:
        return None

    hoy = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    i = 0
    for k, bloque in enumerate(bloques):
        if not _empezada(bloque, hoy):
            break
        i = k

    while i + 1 < len(bloques) and _entera(bloques[i]):
        i += 1
    return int(bloques[i]["matchday"])


def build_calendar_block(
    fit: DixonColesFit,
    season_matches: pd.DataFrame,
    names: dict[int, str],
    scenario: dict,
    as_of: pd.Timestamp,
) -> dict:
    """Calendario completo por jornadas, con el estado de cada partido.

    Cuatro estados posibles, y la diferencia importa:

    - `played`: se jugo de verdad. Su marcador no se puede tocar.
    - `live`: se esta jugando. Su marcador parcial se muestra, pero no cuenta
      como resultado final ni se puede editar como escenario.
    - `scenario`: resultado hipotetico puesto a mano. Cuenta en la clasificacion
      de partida de la simulacion, pero no altera la fuerza de los equipos.
    - `pending`: sin jugar y sin hipotesis. Lo decide la simulacion.
    """
    ordenados = season_matches.sort_values(["matchday", "match_date", "match_id"])
    provisional = provisional_dates(
        ordenados[ordenados["home_goals"].isna()], as_of)

    jornadas: dict[int, dict] = {}
    for m in ordenados.itertuples():
        jugado = pd.notna(m.home_goals)
        en_juego = (not jugado) and getattr(m, "status", None) == "live"
        hipotetico = (not jugado) and (not en_juego) and int(m.match_id) in scenario
        estado = "played" if jugado else (
            "live" if en_juego else ("scenario" if hipotetico else "pending"))
        home_name = names.get(int(m.home_team_id), str(m.home_team_id))
        away_name = names.get(int(m.away_team_id), str(m.away_team_id))

        partido = {
            "match_id": int(m.match_id),
            "date": pd.Timestamp(m.match_date).strftime("%Y-%m-%d"),
            "kickoff_utc": _hora_iso(m),
            "date_provisional": bool(provisional.get(m.Index, False)),
            # El id del evento en ESPN. Con el, el panel puede seguir el
            # partido en directo sin volver a casar nombres de equipo, que es
            # de donde salen las fichas duplicadas.
            "espn_event_id": _texto(getattr(m, "espn_event_id", None)),
            "status": estado,
            "home_team": {
                "team_id": int(m.home_team_id),
                "name": home_name,
                **team_identity(home_name),
            },
            "away_team": {
                "team_id": int(m.away_team_id),
                "name": away_name,
                **team_identity(away_name),
            },
            "home_goals": None,
            "away_goals": None,
            "live_home_goals": None,
            "live_away_goals": None,
            "live_detail": None,
        }
        if jugado:
            partido["home_goals"] = int(m.home_goals)
            partido["away_goals"] = int(m.away_goals)
        elif en_juego:
            gl = getattr(m, "live_home_goals", None)
            gv = getattr(m, "live_away_goals", None)
            partido["live_home_goals"] = None if pd.isna(gl) else int(gl)
            partido["live_away_goals"] = None if pd.isna(gv) else int(gv)
            detalle = getattr(m, "live_detail", None)
            partido["live_detail"] = None if pd.isna(detalle) else str(detalle)
        elif hipotetico:
            partido["home_goals"], partido["away_goals"] = scenario[int(m.match_id)]

        if not jugado and not en_juego:
            # La prediccion del modelo, para poder contrastar la hipotesis con
            # lo que el simulador considera probable.
            p_home, p_draw, p_away = fit.probs_1x2(int(m.home_team_id), int(m.away_team_id))
            partido["probabilities"] = {"home": round(p_home, 4), "draw": round(p_draw, 4),
                                        "away": round(p_away, 4)}

        jornada = int(m.matchday) if pd.notna(m.matchday) else 0
        jornadas.setdefault(jornada, {"matchday": jornada, "matches": []})
        jornadas[jornada]["matches"].append(partido)

    lista = []
    for numero in sorted(jornadas):
        bloque = jornadas[numero]
        estados = [p["status"] for p in bloque["matches"]]
        bloque["played"] = estados.count("played")
        bloque["live"] = estados.count("live")
        bloque["scenario"] = estados.count("scenario")
        bloque["pending"] = estados.count("pending")
        bloque["date_from"] = min(p["date"] for p in bloque["matches"])
        bloque["date_to"] = max(p["date"] for p in bloque["matches"])
        lista.append(bloque)

    return {
        "matchdays": lista,
        "current_matchday": current_matchday(lista, as_of),
        "scenario_count": sum(b["scenario"] for b in lista),
        "live_count": sum(b["live"] for b in lista),
        "editable": True,
    }


def build_output(
    season: str,
    league_block: dict,
    fixtures: list[dict],
    fit: DixonColesFit,
    cfg: Config,
    as_of: pd.Timestamp,
    data_sources: list[str],
    europe: dict | None = None,
    modifiers: list[str] | None = None,
    validation: dict | None = None,
    calendar: dict | None = None,
    standings: dict | None = None,
    european_qualification: dict | None = None,
) -> dict:
    """Ensambla el documento completo que consumira el frontend."""
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "as_of": {
            "date": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            "matches_played": league_block["matches_played"],
            "matches_remaining": league_block["matches_remaining"],
        },
        "simulation": {
            "n_sims": cfg.sim.n_sims,
            "seed": cfg.sim.seed,
            "modifiers_enabled": modifiers or [],
        },
        "model": {
            "type": "dixon-coles-poisson",
            "elo": {
                "k_factor": cfg.elo.k_factor,
                "home_advantage_elo": cfg.elo.home_advantage,
                "season_regression": cfg.elo.season_regression,
            },
            "dixon_coles": {
                "half_life_days": cfg.dixon_coles.half_life_days,
                "elo_prior_weight": cfg.dixon_coles.elo_prior_weight,
                "mu": round(fit.mu, 4),
                "home_advantage": round(fit.home_advantage, 4),
                "rho": round(fit.rho, 4),
                "fitted_on_matches": fit.n_matches,
                "effective_sample_size": round(fit.effective_n, 1),
            },
        },
        "competitions": {league_block["code"]: league_block, **(europe or {})},
        "standings": standings,
        "european_qualification": european_qualification,
        "calendar": calendar,
        "fixtures": fixtures,
        "validation": validation,
        "meta": {"data_sources": data_sources},
    }


def write_json(document: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
