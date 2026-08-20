"""Tests de los ajustes cualitativos (fase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simliga.config import DixonColesConfig, ModifierConfig, SimConfig, load_config
from simliga.model.dixon_coles import DixonColesFit
from simliga.model.modifiers import (build_match_adjustments, compute_rest_days,
                                     elo_delta_to_rate_shift, event_delta,
                                     fatigue_delta, motivation_delta)
from simliga.sim.league import simulate_league


def _partidos(filas):
    return pd.DataFrame([
        {"match_id": i, "competition": comp, "season": "2026-27", "stage": "league",
         "match_date": pd.Timestamp(fecha), "home_team_id": h, "away_team_id": a,
         "home_goals": None, "away_goals": None, "status": "scheduled",
         "home_team": str(h), "away_team": str(a)}
        for i, (fecha, comp, h, a) in enumerate(filas)
    ])


# ------------------------------------------------------- configuracion por defecto
def test_todos_los_modificadores_vienen_desactivados():
    """Se midieron y ninguno mejora la prediccion; activarlos es opt-in explicito."""
    cfg = load_config()
    assert cfg.modifiers.enabled_names() == []
    assert not cfg.modifiers.any_enabled()


def test_los_nombres_activos_se_reflejan_para_el_json():
    cfg = ModifierConfig(fatigue_enabled=True, injuries_enabled=True)
    assert cfg.enabled_names() == ["fatigue", "injuries"]
    assert cfg.enabled_event_kinds() == ("injuries",)


# --------------------------------------------------------------------- descanso
def test_calcula_los_dias_de_descanso_del_calendario_combinado():
    """El punto del calendario combinado: un miercoles europeo acorta el descanso."""
    partidos = _partidos([
        ("2026-09-12", "ESP1", 1, 2),
        ("2026-09-16", "UCL", 1, 3),      # entre semana
        ("2026-09-19", "ESP1", 4, 1),     # solo 3 dias despues del europeo
    ])
    rest = compute_rest_days(partidos)
    assert np.isnan(rest.loc[(1, pd.Timestamp("2026-09-12")), "rest_days"])
    assert rest.loc[(1, pd.Timestamp("2026-09-16")), "rest_days"] == 4
    fila = rest.loc[(1, pd.Timestamp("2026-09-19"))]
    assert fila["rest_days"] == 3
    assert fila["previous_competition"] == "UCL"


def test_sin_partido_anterior_no_hay_penalizacion():
    cfg = ModifierConfig(fatigue_enabled=True)
    delta = fatigue_delta(np.array([np.nan]), np.array([False]), cfg)
    assert delta[0] == 0.0


# ----------------------------------------------------------------------- fatiga
def test_la_fatiga_penaliza_solo_por_debajo_del_descanso_de_referencia():
    cfg = ModifierConfig(fatigue_enabled=True, fatigue_reference_days=6,
                         fatigue_elo_per_day=8, fatigue_european_extra=0)
    dias = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 10.0])
    delta = fatigue_delta(dias, np.zeros(6, dtype=bool), cfg)
    assert list(delta) == [-24.0, -16.0, -8.0, 0.0, 0.0, 0.0]


def test_la_fatiga_tiene_tope():
    cfg = ModifierConfig(fatigue_enabled=True, fatigue_reference_days=6,
                         fatigue_elo_per_day=10, fatigue_max_deficit_days=3,
                         fatigue_european_extra=0)
    delta = fatigue_delta(np.array([1.0]), np.array([False]), cfg)
    assert delta[0] == -30.0, "un dia de descanso no debe penalizar mas que tres de deficit"


def test_venir_de_europa_penaliza_algo_mas():
    cfg = ModifierConfig(fatigue_enabled=True, fatigue_european_extra=5.0)
    domestico = fatigue_delta(np.array([3.0]), np.array([False]), cfg)[0]
    europeo = fatigue_delta(np.array([3.0]), np.array([True]), cfg)[0]
    assert europeo == domestico - 5.0


def test_desactivada_no_hace_nada():
    cfg = ModifierConfig(fatigue_enabled=False)
    assert fatigue_delta(np.array([1.0, 2.0]), np.array([True, True]), cfg).tolist() == [0.0, 0.0]


# -------------------------------------------------------------------- motivacion
def test_la_motivacion_solo_actua_en_las_ultimas_jornadas():
    cfg = ModifierConfig(motivation_enabled=True, motivation_last_matchdays=5)
    puntos = np.array([45.0])
    pronto = motivation_delta(puntos, 20, 38, 60.0, 35.0, cfg)
    tarde = motivation_delta(puntos, 35, 38, 60.0, 35.0, cfg)
    assert pronto[0] == 0.0
    assert tarde[0] < 0.0


def test_penaliza_a_quien_no_puede_alcanzar_europa_ni_bajar():
    cfg = ModifierConfig(motivation_enabled=True, motivation_elo_penalty=25.0)
    # A falta de 3 jornadas (9 puntos), con 45 no llega a 60 ni baja de 35.
    puntos = np.array([45.0, 58.0, 30.0])
    delta = motivation_delta(puntos, 35, 38, 60.0, 35.0, cfg)
    assert delta[0] == -25.0                      # sin nada en juego
    assert delta[1] == 0.0, "aun puede alcanzar Europa"
    assert delta[2] == 0.0, "aun puede descender"


# ------------------------------------------------------------ ajustes de evento
def _ajustes(filas):
    return pd.DataFrame(filas, columns=["adjustment_id", "team_id", "season", "kind",
                                        "elo_delta", "valid_from", "valid_to", "note"])


def test_un_ajuste_solo_se_aplica_dentro_de_su_vigencia():
    ajustes = _ajustes([(1, 7, "2026-27", "injuries", -30.0,
                         pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-15"), None)])
    cfg = ModifierConfig(injuries_enabled=True)
    fechas = pd.Series(pd.to_datetime(["2026-08-20", "2026-09-20", "2026-11-01"]))
    delta = event_delta(np.array([7, 7, 7]), fechas, ajustes, cfg)
    assert list(delta) == [0.0, -30.0, 0.0]


def test_un_ajuste_de_tipo_desactivado_se_ignora():
    ajustes = _ajustes([(1, 7, "2026-27", "injuries", -30.0, None, None, None)])
    cfg = ModifierConfig(injuries_enabled=False, transfers_enabled=True)
    delta = event_delta(np.array([7]), pd.Series([pd.Timestamp("2026-09-20")]), ajustes, cfg)
    assert delta[0] == 0.0


def test_los_ajustes_de_un_mismo_equipo_se_suman():
    ajustes = _ajustes([
        (1, 7, "2026-27", "injuries", -30.0, None, None, None),
        (2, 7, "2026-27", "transfers", 12.0, None, None, None),
    ])
    cfg = ModifierConfig(injuries_enabled=True, transfers_enabled=True)
    delta = event_delta(np.array([7]), pd.Series([pd.Timestamp("2026-09-20")]), ajustes, cfg)
    assert delta[0] == pytest.approx(-18.0)


def test_un_ajuste_no_alcanza_a_otro_equipo():
    ajustes = _ajustes([(1, 7, "2026-27", "injuries", -30.0, None, None, None)])
    cfg = ModifierConfig(injuries_enabled=True)
    delta = event_delta(np.array([7, 8]), pd.Series([pd.Timestamp("2026-09-20")] * 2),
                        ajustes, cfg)
    assert list(delta) == [-30.0, 0.0]


# ------------------------------------------------------------ conversion a goles
def test_la_conversion_de_elo_a_goles_usa_los_coeficientes_ajustados():
    """100 puntos Elo con kappa 0.8 equivalen a 0.8 * 100/400 = 0.2 en log-goles."""
    shift = elo_delta_to_rate_shift(np.array([100.0]), np.array([0.0]), 0.8, 0.4)
    assert shift[0] == pytest.approx(0.2)


def test_debilitar_al_rival_sube_los_goles_propios():
    propio = elo_delta_to_rate_shift(np.array([0.0]), np.array([50.0]), 0.8, 0.4)[0]
    assert propio > 0


# -------------------------------------------------------- integracion en el motor
def _fit(team_ids):
    n = len(team_ids)
    return DixonColesFit(
        team_ids=list(team_ids), attack=np.zeros(n), defence=np.zeros(n),
        mu=0.2, home_advantage=0.3, rho=0.0, kappa_attack=0.8, kappa_defence=0.4,
        n_matches=100, effective_n=80.0, log_likelihood=-1.0, converged=True,
        config=DixonColesConfig(max_goals=8),
    )


def test_un_ajuste_negativo_reduce_los_puntos_esperados():
    equipos = [1, 2]
    fixtures = _partidos([("2026-09-01", "ESP1", 1, 2), ("2026-09-08", "ESP1", 2, 1)] * 8)
    fit = _fit(equipos)
    cfg = SimConfig(n_sims=4000, seed=3)

    sin = simulate_league(fit, fixtures, teams=equipos, config=cfg)
    penalizacion = np.full(len(fixtures), -0.25)      # castiga al local de cada partido
    con = simulate_league(fit, fixtures, teams=equipos, config=cfg,
                          rate_shift_home=penalizacion, rate_shift_away=np.zeros(len(fixtures)))
    assert con.goals_for[:, 0].mean() < sin.goals_for[:, 0].mean()


def test_sin_modificadores_el_resultado_es_identico_al_de_antes():
    """Pasar desplazamientos nulos no puede cambiar nada."""
    equipos = [1, 2, 3, 4]
    fixtures = _partidos([("2026-09-01", "ESP1", h, a)
                          for h in equipos for a in equipos if h != a])
    fit = _fit(equipos)
    cfg = SimConfig(n_sims=500, seed=11)
    a = simulate_league(fit, fixtures, teams=equipos, config=cfg)
    b = simulate_league(fit, fixtures, teams=equipos, config=cfg,
                        rate_shift_home=np.zeros(len(fixtures)),
                        rate_shift_away=np.zeros(len(fixtures)))
    assert np.array_equal(a.points, b.points)


def test_build_match_adjustments_sin_nada_activo_no_ajusta():
    fixtures = _partidos([("2026-09-01", "ESP1", 1, 2)])
    ajustes = build_match_adjustments(fixtures, ModifierConfig())
    assert ajustes.is_empty()


def test_build_match_adjustments_combina_fatiga_y_eventos():
    partidos = _partidos([
        ("2026-09-12", "ESP1", 1, 2),
        ("2026-09-16", "UCL", 1, 3),
        ("2026-09-19", "ESP1", 1, 4),
    ])
    rest = compute_rest_days(partidos)
    cfg = ModifierConfig(fatigue_enabled=True, injuries_enabled=True,
                         fatigue_reference_days=6, fatigue_elo_per_day=8,
                         fatigue_european_extra=5)
    ajustes = _ajustes([(1, 1, "2026-27", "injuries", -20.0, None, None, None)])

    resultado = build_match_adjustments(
        partidos[partidos["competition"] == "ESP1"].iloc[[1]], cfg,
        rest=rest, adjustments=ajustes)
    # 3 dias de descanso (deficit 3) tras un partido europeo, mas la lesion.
    assert resultado.home[0] == pytest.approx(-8 * 3 - 5 - 20)
    assert resultado.away[0] == pytest.approx(0.0)
    assert "fatiga_home" in resultado.detail.columns
