"""Ajustes cualitativos sobre la fuerza de un equipo.

Todos funcionan igual: dado un partido, cada modificador devuelve cuantos puntos
Elo hay que sumar o restar a cada equipo *solo para ese partido*. La conversion
de puntos Elo a goles esperados no es arbitraria: usa los coeficientes
`kappa_attack` y `kappa_defence` que el propio ajuste Dixon-Coles estimo, o sea
la relacion entre Elo y goles que se observa en los datos.

    delta_ataque  = kappa_attack  * (delta_elo / 400)
    delta_defensa = kappa_defence * (delta_elo / 400)

Hay dos familias:

- **Calculables**: fatiga y motivacion salen del calendario y de la clasificacion,
  sin necesidad de ninguna fuente externa.
- **De evento**: cambio de entrenador, lesiones y fichajes necesitan datos que
  ninguna fuente gratuita da de forma fiable. Entran por la tabla
  `team_adjustments`, que se rellena a mano o con un scraper futuro.

## Por que vienen desactivados por defecto

Porque se midieron y no se sostienen. Sobre 6.074 observaciones de ocho
temporadas, comparando los goles reales con los que el modelo esperaba:

| Contraste                                        | Efecto  | t     |
|--------------------------------------------------|---------|-------|
| Descanso <=3 dias frente a >=6                    | -0,055  | -1,38 |
| Venia de Europa con <=4 dias, frente a >=6        | -0,074  | -1,19 |
| 2+ dias mas de descanso que el rival, que 2+ menos| +0,001  | +0,02 |
| Sin nada en juego en las ultimas jornadas         | +0,023  | +0,33 |

Ninguno llega a significacion. El tercero es el mas revelador, porque es el
planteamiento correcto (lo que importa es el descanso *relativo* al rival, no el
absoluto) y da practicamente cero exacto.

La lectura razonable no es que la fatiga no exista, sino que **el modelo ya la
absorbe**: los equipos que juegan entre semana son los buenos, y el Elo ya sabe
que son buenos. Activar un modificador con una magnitud inventada empeoraria las
predicciones en lugar de mejorarlas.

`scripts/analisis_modificadores.py` reproduce la medicion y
`scripts/comparar_modificadores.py` mide el RPS con y sin ellos.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import ELO_TO_STRENGTH, ModifierConfig

UEFA = ("UCL", "UEL", "UECL")


@dataclass
class MatchAdjustments:
    """Puntos Elo a sumar a cada equipo, partido a partido."""

    home: np.ndarray
    away: np.ndarray
    detail: pd.DataFrame          # desglose por modificador, para poder auditarlo

    def is_empty(self) -> bool:
        return not (np.any(self.home) or np.any(self.away))


def compute_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """Dias desde el partido anterior de cada equipo, en todas las competiciones.

    Aqui es donde sirve el calendario combinado: si un equipo jugo en Europa el
    miercoles, su partido de liga del sabado aparece con tres dias de descanso.
    """
    largo = pd.concat([
        matches[["match_date", "competition", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}),
        matches[["match_date", "competition", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}),
    ]).sort_values("match_date")

    largo["rest_days"] = largo.groupby("team_id")["match_date"].diff().dt.days
    largo["previous_competition"] = largo.groupby("team_id")["competition"].shift(1)
    return largo.groupby(["team_id", "match_date"])[
        ["rest_days", "previous_competition"]].first()


def fatigue_delta(
    rest_days: np.ndarray, previous_was_european: np.ndarray, cfg: ModifierConfig
) -> np.ndarray:
    """Penalizacion por poco descanso, en puntos Elo (negativa).

    Se penaliza por cada dia que falte hasta `fatigue_reference_days`, con un
    extra si el partido anterior fue de competicion europea (viaje incluido).
    """
    if not cfg.fatigue_enabled:
        return np.zeros(len(rest_days))

    dias = np.nan_to_num(rest_days, nan=cfg.fatigue_reference_days)
    deficit = np.clip(cfg.fatigue_reference_days - dias, 0, cfg.fatigue_max_deficit_days)
    delta = -cfg.fatigue_elo_per_day * deficit
    delta -= np.where(previous_was_european & (deficit > 0), cfg.fatigue_european_extra, 0.0)
    return delta


def motivation_delta(
    points: np.ndarray,
    matchday: int,
    n_matchdays: int,
    european_threshold: float,
    relegation_threshold: float,
    cfg: ModifierConfig,
) -> np.ndarray:
    """Penalizacion a los equipos sin nada en juego, en puntos Elo (negativa).

    Un equipo esta "libre" si, con los puntos que quedan por repartir, ya no
    puede alcanzar Europa ni caer en descenso.

    Limitacion conocida: se evalua con la clasificacion de la fecha de corte, no
    con la de cada simulacion. Sirve para simular desde un punto avanzado de la
    temporada, pero no para una proyeccion de pretemporada, donde a la jornada
    35 cada simulacion tiene su propia clasificacion. Hacerlo por simulacion
    exige recalcular las tasas dentro del bucle y solo compensaria si alguna vez
    se demuestra que el efecto existe; hoy la medicion da +0,02 goles (t = 0,33).
    """
    if not cfg.motivation_enabled or matchday < n_matchdays - cfg.motivation_last_matchdays:
        return np.zeros(len(points))

    restantes = max(n_matchdays - matchday, 0)
    techo = points + 3 * restantes
    libre = (techo < european_threshold) & (points > relegation_threshold)
    return np.where(libre, -cfg.motivation_elo_penalty, 0.0)


def load_team_adjustments(
    conn: sqlite3.Connection, season: str | None = None
) -> pd.DataFrame:
    """Ajustes manuales guardados: entrenador, lesiones, fichajes."""
    query = """SELECT adjustment_id, team_id, season, kind, elo_delta,
                      valid_from, valid_to, note
               FROM team_adjustments"""
    params: list = []
    if season:
        query += " WHERE season = ?"
        params.append(season)
    df = pd.read_sql_query(query, conn, params=params)
    for col in ("valid_from", "valid_to"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def event_delta(
    team_ids: np.ndarray,
    dates: pd.Series,
    adjustments: pd.DataFrame,
    cfg: ModifierConfig,
) -> np.ndarray:
    """Suma los ajustes manuales vigentes para cada equipo en la fecha del partido."""
    delta = np.zeros(len(team_ids))
    if adjustments.empty:
        return delta

    activos = adjustments[adjustments["kind"].isin(cfg.enabled_event_kinds())]
    if activos.empty:
        return delta

    fechas = pd.to_datetime(pd.Series(dates).to_numpy()).to_numpy()
    for fila in activos.itertuples(index=False):
        vigente = np.asarray(team_ids) == fila.team_id
        if pd.notna(fila.valid_from):
            vigente &= fechas >= np.datetime64(fila.valid_from)
        if pd.notna(fila.valid_to):
            vigente &= fechas <= np.datetime64(fila.valid_to)
        delta += vigente * fila.elo_delta
    return delta


def build_match_adjustments(
    fixtures: pd.DataFrame,
    cfg: ModifierConfig,
    rest: pd.DataFrame | None = None,
    adjustments: pd.DataFrame | None = None,
    standings: dict[int, float] | None = None,
    matchday_info: tuple[int, int] | None = None,
    thresholds: tuple[float, float] | None = None,
) -> MatchAdjustments:
    """Reune todos los modificadores activos en un ajuste por partido y lado."""
    n = len(fixtures)
    home_total = np.zeros(n)
    away_total = np.zeros(n)
    detalle = {"match_id": fixtures["match_id"].to_numpy()}

    if cfg.fatigue_enabled and rest is not None:
        for lado, acumulador in (("home", home_total), ("away", away_total)):
            idx = pd.MultiIndex.from_arrays(
                [fixtures[f"{lado}_team_id"], fixtures["match_date"]])
            sub = rest.reindex(idx)
            delta = fatigue_delta(
                sub["rest_days"].to_numpy(dtype=float),
                sub["previous_competition"].isin(UEFA).to_numpy(),
                cfg,
            )
            detalle[f"fatiga_{lado}"] = delta
            acumulador += delta

    if cfg.motivation_enabled and standings and matchday_info and thresholds:
        matchday, n_matchdays = matchday_info
        europa, descenso = thresholds
        for lado, acumulador in (("home", home_total), ("away", away_total)):
            puntos = fixtures[f"{lado}_team_id"].map(standings).fillna(0.0).to_numpy()
            delta = motivation_delta(puntos, matchday, n_matchdays, europa, descenso, cfg)
            detalle[f"motivacion_{lado}"] = delta
            acumulador += delta

    if adjustments is not None and not adjustments.empty:
        for lado, acumulador in (("home", home_total), ("away", away_total)):
            delta = event_delta(fixtures[f"{lado}_team_id"].to_numpy(),
                                fixtures["match_date"], adjustments, cfg)
            detalle[f"eventos_{lado}"] = delta
            acumulador += delta

    detalle["total_home"] = home_total
    detalle["total_away"] = away_total
    return MatchAdjustments(home_total, away_total, pd.DataFrame(detalle))


def elo_delta_to_rate_shift(
    elo_delta_attacking: np.ndarray,
    elo_delta_defending: np.ndarray,
    kappa_attack: float,
    kappa_defence: float,
) -> np.ndarray:
    """Traduce puntos Elo a un desplazamiento del logaritmo de los goles esperados.

    Los goles que marca un equipo suben con su propio ataque y bajan con la
    defensa del rival, asi que el desplazamiento combina los dos deltas.
    """
    return (kappa_attack * elo_delta_attacking * ELO_TO_STRENGTH
            + kappa_defence * elo_delta_defending * ELO_TO_STRENGTH)
