"""Tests del motor Monte Carlo: contabilidad de la tabla y desempates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simliga.config import DixonColesConfig, SimConfig
from simliga.model.dixon_coles import DixonColesFit
from simliga.sim.league import _rank, simulate_league
from simliga.ingest.fixtures import round_robin


def _fit(team_ids, attack=None, defence=None, rho=0.0):
    n = len(team_ids)
    return DixonColesFit(
        team_ids=list(team_ids),
        attack=np.zeros(n) if attack is None else np.array(attack, dtype=float),
        defence=np.zeros(n) if defence is None else np.array(defence, dtype=float),
        mu=0.1, home_advantage=0.3, rho=rho, kappa_attack=0.0, kappa_defence=0.0,
        n_matches=0, effective_n=0.0, log_likelihood=0.0, converged=True,
        config=DixonColesConfig(max_goals=8),
    )


def _fixtures(pairs, season="2024-25"):
    return pd.DataFrame([
        {"match_id": i, "competition": "ESP1", "season": season, "stage": "league",
         "match_date": pd.Timestamp("2024-09-01") + pd.Timedelta(days=i),
         "home_team_id": h, "away_team_id": a, "home_goals": None, "away_goals": None,
         "status": "scheduled", "home_team": str(h), "away_team": str(a)}
        for i, (h, a) in enumerate(pairs)
    ])


def _played(rows, season="2024-25"):
    return pd.DataFrame([
        {"match_id": 1000 + i, "competition": "ESP1", "season": season, "stage": "league",
         "match_date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=i),
         "home_team_id": h, "away_team_id": a, "home_goals": hg, "away_goals": ag,
         "status": "played", "home_team": str(h), "away_team": str(a)}
        for i, (h, a, hg, ag) in enumerate(rows)
    ])


def test_cada_simulacion_reparte_los_puntos_que_toca():
    """Por partido se reparten 3 puntos (victoria) o 2 (empate), nunca otra cosa."""
    teams = [1, 2, 3, 4]
    fixtures = _fixtures([(1, 2), (3, 4), (2, 3), (4, 1)])
    res = simulate_league(_fit(teams), fixtures, teams=teams, config=SimConfig(n_sims=500))
    total = res.points.sum(axis=1)
    assert total.min() >= 2 * len(fixtures)
    assert total.max() <= 3 * len(fixtures)


def test_los_goles_cuadran_entre_a_favor_y_en_contra():
    teams = [1, 2, 3, 4]
    res = simulate_league(_fit(teams), _fixtures([(1, 2), (3, 4), (2, 1), (4, 3)]),
                          teams=teams, config=SimConfig(n_sims=300))
    # La diferencia de goles de toda la liga tiene que ser cero en cada simulacion.
    assert (res.goal_diff.sum(axis=1) == 0).all()


def test_las_posiciones_son_una_permutacion():
    teams = [1, 2, 3, 4, 5, 6]
    fixtures = _fixtures([(h, a) for h in teams for a in teams if h != a])
    res = simulate_league(_fit(teams), fixtures, teams=teams, config=SimConfig(n_sims=200))
    esperado = np.arange(1, len(teams) + 1)
    assert all(np.array_equal(np.sort(fila), esperado) for fila in res.positions)


def test_los_partidos_jugados_entran_sin_aleatoriedad():
    teams = [1, 2, 3, 4]
    played = _played([(1, 2, 3, 0), (3, 4, 1, 1)])
    res = simulate_league(_fit(teams), _fixtures([]), played=played, teams=teams,
                          config=SimConfig(n_sims=50))
    puntos = res.points[:, [0, 1, 2, 3]]
    assert (puntos[:, 0] == 3).all()   # equipo 1 gano
    assert (puntos[:, 1] == 0).all()   # equipo 2 perdio
    assert (puntos[:, 2] == 1).all() and (puntos[:, 3] == 1).all()


def test_el_equipo_mas_fuerte_gana_mas_a_menudo():
    teams = [1, 2]
    fixtures = _fixtures([(1, 2), (2, 1)] * 10)
    res = simulate_league(_fit(teams, attack=[0.6, -0.6], defence=[0.4, -0.4]),
                          fixtures, teams=teams, config=SimConfig(n_sims=2000))
    assert res.points[:, 0].mean() > res.points[:, 1].mean() + 15


def test_las_probabilidades_de_posicion_suman_uno():
    teams = [1, 2, 3, 4, 5]
    fixtures = _fixtures([(h, a) for h in teams for a in teams if h != a])
    res = simulate_league(_fit(teams), fixtures, teams=teams, config=SimConfig(n_sims=400))
    probs = res.position_probabilities()
    assert np.allclose(probs.sum(axis=1), 1.0)     # cada equipo acaba en alguna posicion
    assert np.allclose(probs.sum(axis=0), 1.0)     # cada posicion la ocupa alguien


def test_la_misma_semilla_da_el_mismo_resultado():
    teams = [1, 2, 3, 4]
    fixtures = _fixtures([(1, 2), (3, 4), (2, 3), (4, 1)])
    cfg = SimConfig(n_sims=200, seed=99)
    a = simulate_league(_fit(teams), fixtures, teams=teams, config=cfg)
    b = simulate_league(_fit(teams), fixtures, teams=teams, config=cfg)
    assert np.array_equal(a.positions, b.positions)


# ------------------------------------------------------------------- desempates
def test_desempata_por_diferencia_de_goles():
    puntos = np.array([[10, 10]], dtype=np.int16)
    gd = np.array([[2, 5]], dtype=np.int16)
    gf = np.array([[10, 10]], dtype=np.int16)
    pos = _rank(puntos, gd, gf, None, None)
    assert pos[0, 1] == 1 and pos[0, 0] == 2


def test_el_enfrentamiento_directo_manda_sobre_la_diferencia_general():
    """Regla de LaLiga: entre dos empatados a puntos decide el head-to-head."""
    puntos = np.array([[10, 10]], dtype=np.int16)
    gd = np.array([[8, 1]], dtype=np.int16)       # el equipo 0 tiene mejor diferencia
    gf = np.array([[20, 12]], dtype=np.int16)
    h2h_pts = np.zeros((1, 2, 2), dtype=np.int8)
    h2h_gd = np.zeros((1, 2, 2), dtype=np.int8)
    h2h_pts[0, 1, 0] = 4                           # pero el 1 le gano los dos partidos
    h2h_pts[0, 0, 1] = 0
    h2h_gd[0, 1, 0], h2h_gd[0, 0, 1] = 3, -3

    pos = _rank(puntos, gd, gf, h2h_pts, h2h_gd)
    assert pos[0, 1] == 1, "el head-to-head debe imponerse a la diferencia general"


def test_sin_empate_a_puntos_el_head_to_head_no_altera_nada():
    puntos = np.array([[12, 10]], dtype=np.int16)
    gd = np.array([[0, 9]], dtype=np.int16)
    gf = np.array([[9, 20]], dtype=np.int16)
    h2h_pts = np.zeros((1, 2, 2), dtype=np.int8)
    h2h_gd = np.zeros((1, 2, 2), dtype=np.int8)
    h2h_pts[0, 1, 0] = 6
    pos = _rank(puntos, gd, gf, h2h_pts, h2h_gd)
    assert pos[0, 0] == 1


def test_empate_a_tres_sin_h2h_decisivo_usa_la_diferencia_general():
    """Si la mini-liga entre los tres queda igualada, decide el criterio general."""
    puntos = np.array([[10, 10, 10]], dtype=np.int16)
    gd = np.array([[1, 5, 3]], dtype=np.int16)
    gf = np.array([[10, 10, 10]], dtype=np.int16)
    h2h_pts = np.zeros((1, 3, 3), dtype=np.int8)
    h2h_gd = np.zeros((1, 3, 3), dtype=np.int8)
    pos = _rank(puntos, gd, gf, h2h_pts, h2h_gd)
    assert list(pos[0]) == [3, 1, 2]


def test_la_mini_liga_resuelve_un_empate_a_tres():
    """Caso de 2025-26: tres equipos a 42 puntos y el head-to-head decidio el descenso."""
    puntos = np.array([[42, 42, 42]], dtype=np.int16)
    gd = np.array([[-14, -6, -10]], dtype=np.int16)     # el 2 tiene la mejor general
    gf = np.array([[47, 40, 47]], dtype=np.int16)
    h2h_pts = np.zeros((1, 3, 3), dtype=np.int8)
    h2h_gd = np.zeros((1, 3, 3), dtype=np.int8)
    # El equipo 0 domina la mini-liga pese a tener la peor diferencia general.
    h2h_pts[0, 0, 1] = 4; h2h_pts[0, 1, 0] = 1
    h2h_pts[0, 0, 2] = 4; h2h_pts[0, 2, 0] = 1
    h2h_pts[0, 1, 2] = 3; h2h_pts[0, 2, 1] = 3
    h2h_gd[0, 0, 1], h2h_gd[0, 1, 0] = 2, -2
    h2h_gd[0, 0, 2], h2h_gd[0, 2, 0] = 3, -3

    pos = _rank(puntos, gd, gf, h2h_pts, h2h_gd)
    assert pos[0, 0] == 1, "la mini-liga debe imponerse a la diferencia general"
    assert pos[0, 2] == 3, "quien pierde la mini-liga cae al ultimo puesto del grupo"


def test_un_grupo_de_cuatro_tambien_se_resuelve_por_mini_liga():
    puntos = np.array([[30, 30, 30, 30]], dtype=np.int16)
    gd = np.array([[0, 0, 0, 0]], dtype=np.int16)
    gf = np.array([[30, 30, 30, 30]], dtype=np.int16)
    h2h_pts = np.zeros((1, 4, 4), dtype=np.int8)
    h2h_gd = np.zeros((1, 4, 4), dtype=np.int8)
    for i in range(4):                       # el equipo i suma 3*(3-i) en la mini-liga
        for j in range(4):
            if i < j:
                h2h_pts[0, i, j] = 3
    pos = _rank(puntos, gd, gf, h2h_pts, h2h_gd)
    assert list(pos[0]) == [1, 2, 3, 4]


# -------------------------------------------------------------------- calendario
def test_el_round_robin_es_completo_y_equilibrado():
    equipos = list(range(1, 21))
    calendario = round_robin(equipos)
    partidos = [p for jornada in calendario for p in jornada]

    assert len(calendario) == 38
    assert len(partidos) == 380
    assert len(set(partidos)) == 380              # ningun emparejamiento repetido
    for t in equipos:
        locales = sum(1 for h, _ in partidos if h == t)
        visitantes = sum(1 for _, a in partidos if a == t)
        assert locales == visitantes == 19


def test_cada_jornada_juega_cada_equipo_una_vez():
    equipos = list(range(1, 21))
    for jornada in round_robin(equipos):
        implicados = [t for pareja in jornada for t in pareja]
        assert sorted(implicados) == equipos
