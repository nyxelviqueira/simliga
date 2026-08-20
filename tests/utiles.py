"""Piezas compartidas por varias pruebas.

Sobre todo un documento de salida completo y pequeño: hace falta en cuanto una
prueba quiere trabajar con el JSON tal y como lo consume el panel, y montarlo a
mano en cada fichero acaba en veinte copias que se desincronizan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simliga.config import DixonColesConfig, load_config
from simliga.model.dixon_coles import DixonColesFit
from simliga.output import contract
from simliga.sim.league import simulate_league

EQUIPOS = list(range(1, 21))
NOMBRES = {t: f"Equipo {t:02d}" for t in EQUIPOS}
INICIO = pd.Timestamp("2026-08-15")


def ajuste_de_prueba() -> DixonColesFit:
    """Un ajuste con equipos claramente escalonados, de mejor a peor."""
    fuerza = np.linspace(0.4, -0.4, len(EQUIPOS))
    return DixonColesFit(
        team_ids=EQUIPOS, attack=fuerza.copy(), defence=fuerza * 0.7,
        mu=0.1, home_advantage=0.3, rho=0.02, kappa_attack=0.5, kappa_defence=0.35,
        n_matches=380, effective_n=250.0, log_likelihood=-100.0, converged=True,
        config=DixonColesConfig(max_goals=10),
    )


def calendario_de_prueba(jugadas: int = 2) -> pd.DataFrame:
    """Temporada entera de 38 jornadas, con las primeras `jugadas` disputadas.

    El emparejamiento es el metodo del circulo, el mismo que usa una liga de
    verdad: cada equipo juega una vez por jornada, ida y vuelta.
    """
    n = len(EQUIPOS)
    rng = np.random.default_rng(7)
    filas = []
    rueda = EQUIPOS[:]
    for jornada in range(1, 2 * (n - 1) + 1):
        vuelta = jornada > n - 1
        for i in range(n // 2):
            local, visitante = rueda[i], rueda[n - 1 - i]
            if (i % 2 == 0) != vuelta:
                local, visitante = visitante, local
            fecha = INICIO + pd.Timedelta(days=7 * (jornada - 1))
            jugado = jornada <= jugadas
            filas.append({
                "match_id": len(filas) + 1,
                "competition": "ESP1", "season": "2026-27", "stage": "league",
                "matchday": jornada,
                "match_date": fecha,
                "kickoff_utc": "19:00",
                "home_team_id": local, "away_team_id": visitante,
                "home_goals": int(rng.poisson(1.5)) if jugado else None,
                "away_goals": int(rng.poisson(1.1)) if jugado else None,
                "status": "played" if jugado else "scheduled",
                "home_team": NOMBRES[local], "away_team": NOMBRES[visitante],
            })
        if jornada == n - 1:
            rueda = EQUIPOS[:]                     # la vuelta repite el sorteo
        else:
            rueda = [rueda[0]] + [rueda[-1]] + rueda[1:-1]
    return pd.DataFrame(filas)


def documento_de_prueba(n_sims: int = 40_000, jugadas: int = 2) -> dict:
    """Documento de salida completo, con calendario, listo para el panel."""
    fit = ajuste_de_prueba()
    partidos = calendario_de_prueba(jugadas)
    jugados = partidos[partidos["home_goals"].notna()]
    pendientes = partidos[partidos["home_goals"].isna()]

    cfg = load_config()
    cfg.sim.n_sims = n_sims
    resultado = simulate_league(fit, pendientes, played=jugados, teams=EQUIPOS,
                                config=cfg.sim)

    elo = {t: 1500 + 300 * float(fit.attack[fit.index_of(t)]) for t in EQUIPOS}
    bloque = contract.build_league_block(resultado, fit, NOMBRES, elo, jugados,
                                         pendientes, cfg, real_played=jugados)
    return contract.build_output(
        season="2026-27", league_block=bloque,
        fixtures=contract.build_fixtures_block(fit, pendientes, NOMBRES, limit=5,
                                               as_of=INICIO),
        fit=fit, cfg=cfg, as_of=INICIO, data_sources=["test"],
        calendar=contract.build_calendar_block(fit, partidos, NOMBRES, {}, INICIO),
        standings=contract.build_standings_block(
            jugados, [NOMBRES[t] for t in EQUIPOS], cfg),
    )
