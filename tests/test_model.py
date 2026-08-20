"""Tests unitarios del Elo y del ajuste Dixon-Coles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from simliga.config import DixonColesConfig, EloConfig
from simliga.model.dixon_coles import (
    _objective, fit_dixon_coles, score_matrix_from_rates, time_weights,
)
from simliga.model.elo import EloEngine, expected_score, goal_difference_multiplier, run_elo


# --------------------------------------------------------------------------- Elo
def test_expected_score_es_simetrica():
    assert expected_score(1500, 1500, 0) == pytest.approx(0.5)
    a = expected_score(1600, 1400, 0)
    b = expected_score(1400, 1600, 0)
    assert a + b == pytest.approx(1.0)


def test_ventaja_local_sube_la_expectativa():
    assert expected_score(1500, 1500, 65) > expected_score(1500, 1500, 0)


def test_elo_es_suma_cero():
    """Lo que gana un equipo lo pierde el otro: la media del sistema no se mueve."""
    engine = EloEngine(EloConfig())
    engine.update(1, 2, 3, 0, "ESP1", "2024-25", pd.Timestamp("2024-09-01"))
    assert engine.state.ratings[1] + engine.state.ratings[2] == pytest.approx(3000.0)


def test_ganar_sube_el_rating_y_perder_lo_baja():
    engine = EloEngine(EloConfig())
    engine.update(1, 2, 2, 0, "ESP1", "2024-25", pd.Timestamp("2024-09-01"))
    assert engine.state.ratings[1] > 1500 > engine.state.ratings[2]


def test_multiplicador_por_diferencia_de_goles_es_creciente():
    valores = [goal_difference_multiplier(g) for g in range(6)]
    assert valores[0] == valores[1] == 1.0
    assert all(b >= a for a, b in zip(valores, valores[1:]))


def test_goleada_mueve_mas_que_victoria_ajustada():
    def delta(hg, ag):
        e = EloEngine(EloConfig())
        e.update(1, 2, hg, ag, "ESP1", "2024-25", pd.Timestamp("2024-09-01"))
        return e.state.ratings[1] - 1500

    assert delta(5, 0) > delta(1, 0) > 0


def test_regresion_entre_temporadas_encoge_hacia_la_media():
    cfg = EloConfig(season_regression=0.5)
    engine = EloEngine(cfg)
    engine.state.ratings = {1: 1700.0, 2: 1300.0}
    engine._current_season = "2023-24"
    engine.update(1, 2, 1, 1, "ESP1", "2024-25", pd.Timestamp("2024-09-01"))
    # Tras regresar a la media (1500) los ratings parten de 1600 y 1400.
    assert engine.history[0]["home_elo_before"] == pytest.approx(1600.0)
    assert engine.history[0]["away_elo_before"] == pytest.approx(1400.0)


def test_equipo_de_segunda_arranca_mas_bajo():
    engine = EloEngine(EloConfig())
    engine.update(1, 2, 1, 0, "ESP2", "2024-25", pd.Timestamp("2024-09-01"))
    assert engine.history[0]["home_elo_before"] == 1380.0


# ------------------------------------------------------------------ Dixon-Coles
def test_matriz_de_marcadores_suma_uno():
    mat = score_matrix_from_rates(1.6, 1.1, 0.05, 12)
    assert mat.sum() == pytest.approx(1.0)
    assert (mat >= 0).all()


def test_rho_solo_toca_los_marcadores_bajos():
    base = score_matrix_from_rates(1.5, 1.2, 0.0, 10)
    ajustada = score_matrix_from_rates(1.5, 1.2, 0.12, 10)
    cambiadas = ~np.isclose(base, ajustada, rtol=1e-6)
    # Fuera del cuadrante 2x2 solo cambia la renormalizacion, no la forma.
    assert cambiadas[:2, :2].all()
    ratio = (ajustada[3:, 3:] / base[3:, 3:])
    assert np.allclose(ratio, ratio.flat[0])


def test_gradiente_analitico_coincide_con_el_numerico():
    rng = np.random.default_rng(11)
    n, m = 6, 300
    hi = rng.integers(0, n, m)
    ai = (hi + 1 + rng.integers(0, n - 1, m)) % n
    args = (n, hi, ai, rng.poisson(1.5, m).astype(float), rng.poisson(1.1, m).astype(float),
            rng.uniform(0.3, 1.0, m), rng.normal(0, 0.3, n), 12.0, 1.0)
    x = np.concatenate([rng.normal(0, .2, n), rng.normal(0, .2, n), [.2, .25, .08, .15, .12]])

    numerico = approx_fprime(x, lambda p: _objective(p, *args)[0], 1e-6)
    analitico = _objective(x, *args)[1]
    assert np.allclose(numerico, analitico, rtol=1e-3, atol=1e-4)


def test_pesos_temporales_decaen_por_media_vida():
    fechas = pd.Series(pd.to_datetime(["2024-01-01", "2023-07-04", "2022-07-05"]))
    w = time_weights(fechas, pd.Timestamp("2024-01-01"), half_life_days=180)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5, abs=0.02)
    assert w[2] == pytest.approx(0.125, abs=0.02)


def _liga_sintetica(rng, fuerzas, n_vueltas=6):
    """Genera una liga en la que la fuerza real de cada equipo es conocida."""
    filas, mid = [], 0
    fecha = pd.Timestamp("2023-08-01")
    for vuelta in range(n_vueltas):
        for h in fuerzas:
            for a in fuerzas:
                if h == a:
                    continue
                lam_h = np.exp(0.1 + fuerzas[h] - fuerzas[a] + 0.3)
                lam_a = np.exp(0.1 + fuerzas[a] - fuerzas[h])
                mid += 1
                filas.append({
                    "match_id": mid, "competition": "ESP1", "season": "2023-24",
                    "stage": "league", "match_date": fecha + pd.Timedelta(days=mid // 5),
                    "home_team_id": h, "away_team_id": a,
                    "home_goals": rng.poisson(lam_h), "away_goals": rng.poisson(lam_a),
                    "status": "played", "home_team": str(h), "away_team": str(a),
                })
    return pd.DataFrame(filas)


def test_el_ajuste_recupera_las_fuerzas_reales():
    """Con datos generados por el propio modelo, el ajuste debe reencontrarlos.

    Se promedia sobre varias semillas a proposito: con una liga de 5 equipos la
    ventaja local estimada tiene una desviacion tipica de ~0.04 incluso con 2000
    partidos, asi que exigirle precision a una sola muestra mediria la suerte de
    la semilla. Promediando se comprueba lo que importa: que el estimador no
    esta sesgado.
    """
    fuerzas = {1: 0.45, 2: 0.20, 3: 0.0, 4: -0.20, 5: -0.45}
    reales = np.array([fuerzas[t] for t in sorted(fuerzas)])
    cfg = DixonColesConfig(half_life_days=10_000, elo_prior_weight=0.0, l2_weight=0.01)

    ventajas, mus = [], []
    for semilla in range(5):
        partidos = _liga_sintetica(np.random.default_rng(semilla), fuerzas, n_vueltas=60)
        fit = fit_dixon_coles(
            partidos, {}, cutoff=partidos["match_date"].max() + pd.Timedelta(days=1), config=cfg
        )
        assert fit.converged
        assert fit.team_ids == sorted(fuerzas)
        estimadas = fit.attack - fit.attack.mean()
        assert np.corrcoef(reales, estimadas)[0, 1] > 0.98
        ventajas.append(fit.home_advantage)
        mus.append(fit.mu)

    assert np.mean(ventajas) == pytest.approx(0.3, abs=0.06)
    assert np.mean(mus) == pytest.approx(0.1, abs=0.06)


def test_el_prior_elo_da_fuerza_a_un_equipo_sin_partidos():
    """Un recien ascendido sin historico en Primera no puede quedar en el limbo."""
    rng = np.random.default_rng(9)
    partidos = _liga_sintetica(rng, {1: 0.3, 2: 0.1, 3: -0.1, 4: -0.3}, n_vueltas=4)
    cutoff = partidos["match_date"].max() + pd.Timedelta(days=1)

    elo = {1: 1700, 2: 1600, 3: 1500, 4: 1400, 99: 1300}
    fit = fit_dixon_coles(partidos, elo, cutoff=cutoff, teams=[1, 2, 3, 4, 99],
                          config=DixonColesConfig(elo_prior_weight=12.0))

    nuevo = fit.index_of(99)
    assert fit.attack[nuevo] < fit.attack[fit.index_of(1)]
    lam_h, lam_a = fit.rates(1, 99)
    assert lam_h > lam_a  # el mejor equipo, en casa, es favorito claro


def test_probabilidades_1x2_suman_uno():
    rng = np.random.default_rng(3)
    partidos = _liga_sintetica(rng, {1: 0.3, 2: 0.0, 3: -0.3}, n_vueltas=8)
    fit = fit_dixon_coles(partidos, {}, cutoff=partidos["match_date"].max() + pd.Timedelta(days=1))
    assert sum(fit.probs_1x2(1, 3)) == pytest.approx(1.0)


def test_run_elo_procesa_todos_los_partidos():
    rng = np.random.default_rng(5)
    partidos = _liga_sintetica(rng, {1: 0.2, 2: 0.0, 3: -0.2}, n_vueltas=2)
    engine = run_elo(partidos)
    assert len(engine.history) == len(partidos)
    assert set(engine.state.ratings) == {1, 2, 3}


def test_el_recentrado_no_cambia_las_predicciones():
    """Fijar el gauge de ataque/defensa debe dejar intactos los goles esperados."""
    rng = np.random.default_rng(2)
    partidos = _liga_sintetica(rng, {1: 0.4, 2: 0.1, 3: -0.1, 4: -0.4}, n_vueltas=20)
    cutoff = partidos["match_date"].max() + pd.Timedelta(days=1)
    fit = fit_dixon_coles(partidos, {}, cutoff=cutoff)

    assert fit.attack.mean() == pytest.approx(0.0, abs=1e-9)
    assert fit.defence.mean() == pytest.approx(0.0, abs=1e-9)

    # Las tasas siguen reproduciendo la media de goles observada.
    lam_h = [fit.rates(h, a)[0] for h in fit.team_ids for a in fit.team_ids if h != a]
    lam_a = [fit.rates(h, a)[1] for h in fit.team_ids for a in fit.team_ids if h != a]
    assert np.mean(lam_h) == pytest.approx(partidos["home_goals"].mean(), rel=0.1)
    assert np.mean(lam_a) == pytest.approx(partidos["away_goals"].mean(), rel=0.1)


def test_la_penalizacion_por_ascenso_se_aplica_a_quien_se_indica():
    """Primera y Segunda comparten pool de Elo pero sus escalas derivan.

    Sin esta correccion el modelo sobrestimaba a los recien ascendidos en +5,1
    puntos por temporada (45 casos sobre 15 temporadas).
    """
    engine = EloEngine(EloConfig(promotion_penalty=50.0))
    fecha = pd.Timestamp("2025-03-01")
    engine.update(1, 2, 1, 0, "ESP1", "2024-25", fecha)
    engine.update(3, 4, 1, 0, "ESP2", "2024-25", fecha)

    crudos = engine.ratings_at_cutoff()
    ajustados = engine.ratings_at_cutoff(promoted_teams={3, 4})

    assert ajustados[1] == crudos[1]              # ya estaba en Primera
    assert ajustados[3] == crudos[3] - 50.0       # asciende
    assert ajustados[4] == crudos[4] - 50.0


def test_sin_lista_de_ascendidos_no_se_penaliza_a_nadie():
    engine = EloEngine(EloConfig(promotion_penalty=50.0))
    engine.update(3, 4, 1, 0, "ESP2", "2024-25", pd.Timestamp("2025-03-01"))
    assert engine.ratings_at_cutoff() == engine.state.ratings


def test_la_penalizacion_no_se_apaga_al_jugar_en_la_nueva_division():
    """Regresion: se deducia del ultimo partido, y bastaba una jornada.

    El desfase de escala entre Segunda y Primera no desaparece porque el equipo
    juegue noventa minutos. Medido sobre 864 partidos de ascendidos, mantener la
    correccion toda la temporada predice mejor que apagarla, y la mejora viene
    sobre todo de la segunda vuelta.
    """
    engine = EloEngine(EloConfig(promotion_penalty=50.0))
    engine.update(3, 1, 1, 1, "ESP1", "2025-26", pd.Timestamp("2025-08-16"))
    ajustados = engine.ratings_at_cutoff(promoted_teams={3})
    assert ajustados[3] == engine.state.ratings[3] - 50.0


def test_promoted_into_identifica_a_quien_cambia_de_division():
    from simliga.data import promoted_into

    partidos = pd.DataFrame([
        {"season": "2025-26", "competition": "ESP1", "home_team_id": 1, "away_team_id": 2},
        {"season": "2026-27", "competition": "ESP1", "home_team_id": 1, "away_team_id": 3},
    ])
    assert promoted_into(partidos, "2026-27", "ESP1") == {3}


def test_reconoce_a_quien_ya_estuvo_en_primera_hace_dos_temporadas():
    """Un equipo yoyo tambien viene de Segunda, aunque en 2024-25 estuviera arriba.

    Deducir el origen del estado del Elo fallaba justo aqui: tenia registrada
    Primera y se quedaba sin correccion.
    """
    from simliga.data import promoted_into

    partidos = pd.DataFrame([
        {"season": "2024-25", "competition": "ESP1", "home_team_id": 7, "away_team_id": 1},
        {"season": "2025-26", "competition": "ESP2", "home_team_id": 7, "away_team_id": 9},
        {"season": "2025-26", "competition": "ESP1", "home_team_id": 1, "away_team_id": 2},
        {"season": "2026-27", "competition": "ESP1", "home_team_id": 7, "away_team_id": 1},
    ])
    assert 7 in promoted_into(partidos, "2026-27", "ESP1")


def test_el_recentrado_no_cambia_las_predicciones():
    """Fijar el gauge de ataque/defensa debe dejar intactos los goles esperados."""
    rng = np.random.default_rng(2)
    partidos = _liga_sintetica(rng, {1: 0.4, 2: 0.1, 3: -0.1, 4: -0.4}, n_vueltas=20)
    cutoff = partidos["match_date"].max() + pd.Timedelta(days=1)
    fit = fit_dixon_coles(partidos, {}, cutoff=cutoff)

    assert fit.attack.mean() == pytest.approx(0.0, abs=1e-9)
    assert fit.defence.mean() == pytest.approx(0.0, abs=1e-9)

    # Las tasas siguen reproduciendo la media de goles observada.
    lam_h = [fit.rates(h, a)[0] for h in fit.team_ids for a in fit.team_ids if h != a]
    lam_a = [fit.rates(h, a)[1] for h in fit.team_ids for a in fit.team_ids if h != a]
    assert np.mean(lam_h) == pytest.approx(partidos["home_goals"].mean(), rel=0.1)
    assert np.mean(lam_a) == pytest.approx(partidos["away_goals"].mean(), rel=0.1)
