"""Validacion de las competiciones europeas contra lo que realmente paso.

Se simula el torneo desde antes del primer partido de la liguilla (con el
sorteo ya conocido, que es como se hace en la practica) y se compara la
probabilidad que dio el modelo a cada equipo de alcanzar cada ronda con las
rondas que alcanzo de verdad.

La referencia contra la que medir es el bombo: repartir la misma probabilidad
entre los 36 participantes. Un modelo que no bata eso no esta aportando nada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..pipeline import ModelContext, build_context, simulate_european
from . import metrics

# Cuantos equipos alcanzan cada ronda, para poder construir la referencia ciega.
PLAZAS = {"direct_to_r16": 8, "R16": 16, "QF": 8, "SF": 4, "F": 2, "winner": 1}
RONDAS = ["direct_to_r16", "R16", "QF", "SF", "F", "winner"]


def actual_stages(ctx: ModelContext, season: str, competition: str) -> dict[str, set[int]]:
    """Equipos que realmente alcanzaron cada ronda, leidos de los resultados."""
    sub = ctx.matches[(ctx.matches["competition"] == competition)
                      & (ctx.matches["season"] == season)]
    alcanzo: dict[str, set[int]] = {}
    for fase in ("playoff", "R16", "QF", "SF", "F"):
        partidos = sub[sub["stage"] == fase]
        alcanzo[fase] = set(partidos["home_team_id"]) | set(partidos["away_team_id"])

    final = sub[(sub["stage"] == "F") & sub["home_goals"].notna()]
    campeon: set[int] = set()
    if len(final):
        fila = final.iloc[0]
        if fila["home_goals"] != fila["away_goals"]:
            ganador = (fila["home_team_id"] if fila["home_goals"] > fila["away_goals"]
                       else fila["away_team_id"])
            campeon = {int(ganador)}
    alcanzo["winner"] = campeon

    # Pasar directo a octavos = estar en octavos sin haber jugado el playoff.
    alcanzo["direct_to_r16"] = alcanzo["R16"] - alcanzo["playoff"]
    return alcanzo


@dataclass
class UefaBacktestResult:
    por_equipo: pd.DataFrame
    por_ronda: pd.DataFrame
    resumen: dict


def backtest_uefa(
    season: str,
    competition: str,
    config: Config | None = None,
    conn=None,
) -> UefaBacktestResult:
    """Simula el torneo desde su inicio y mide el acierto ronda a ronda."""
    cfg = config or load_config()
    sub_ctx = build_context(conn, cfg, as_of=None)
    liguilla = sub_ctx.matches[(sub_ctx.matches["competition"] == competition)
                               & (sub_ctx.matches["season"] == season)
                               & (sub_ctx.matches["stage"] == "league_phase")]
    if liguilla.empty:
        raise ValueError(f"No hay liguilla de {competition} {season}")

    inicio = liguilla["match_date"].min()
    ctx = build_context(conn, cfg, as_of=inicio)
    resultado, _ = simulate_european(ctx, season, competition)
    reales = actual_stages(ctx, season, competition)

    probs = dict(resultado.stage_probabilities())
    probs.update(resultado.league_phase_outcomes())
    filas = []
    for i, tid in enumerate(resultado.team_ids):
        fila = {"season": season, "competition": competition,
                "team": ctx.names.get(tid, str(tid)),
                "country": ctx.countries.get(tid, "?"),
                "exp_points": float(resultado.league_points[:, i].mean()),
                "exp_position": float(resultado.league_positions[:, i].mean())}
        for ronda in RONDAS:
            fila[f"p_{ronda}"] = float(probs[ronda][i])
            fila[f"real_{ronda}"] = int(tid in reales.get(ronda, set()))
        filas.append(fila)

    por_equipo = pd.DataFrame(filas)
    n = len(por_equipo)

    filas_ronda = []
    for ronda in RONDAS:
        p = por_equipo[f"p_{ronda}"].to_numpy()
        real = por_equipo[f"real_{ronda}"].to_numpy(dtype=float)
        if real.sum() == 0:
            continue
        ciega = np.full(n, PLAZAS[ronda] / n)      # el bombo: todos iguales
        filas_ronda.append({
            "ronda": ronda,
            "clasificados": int(real.sum()),
            "brier_modelo": float(metrics.binary_brier(p, real).mean()),
            "brier_bombo": float(metrics.binary_brier(ciega, real).mean()),
            "logloss_modelo": float(-np.mean(real * np.log(np.clip(p, 1e-9, 1))
                                             + (1 - real) * np.log(np.clip(1 - p, 1e-9, 1)))),
            "prob_media_acertados": float(p[real == 1].mean()),
            "prob_media_resto": float(p[real == 0].mean()),
        })

    por_ronda = pd.DataFrame(filas_ronda)
    resumen = {
        "season": season, "competition": competition, "equipos": n,
        "brier_modelo": float(por_ronda["brier_modelo"].mean()),
        "brier_bombo": float(por_ronda["brier_bombo"].mean()),
        "mejora_sobre_bombo": float(
            1 - por_ronda["brier_modelo"].mean() / por_ronda["brier_bombo"].mean()),
    }
    return UefaBacktestResult(por_equipo, por_ronda, resumen)
