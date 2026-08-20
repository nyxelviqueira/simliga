"""Tests de las metricas y de la validacion contra temporadas reales.

Los tests marcados `historico` necesitan la base de datos ya ingerida
(`python -m simliga ingest`) y son lentos porque reajustan el modelo fecha a
fecha. Para saltarlos:

    pytest -m "not historico"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simliga.config import DB_PATH, load_config
from simliga.data import (check_season_integrity, league_table, load_matches,
                          load_odds)
from simliga.db import connect
from simliga.validation import metrics
from simliga.validation.backtest import assign_matchday, backtest_matches, backtest_season

SEASONS = ("2022-23", "2023-24", "2024-25")


# ----------------------------------------------------------------- metricas puras
def test_rps_es_cero_con_prediccion_perfecta():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert metrics.ranked_probability_score(probs, [0, 2]) == pytest.approx([0.0, 0.0])


def test_rps_penaliza_mas_equivocarse_lejos():
    """Dar la victoria al visitante cuando gana el local cuesta mas que dar empate."""
    cerca = metrics.ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), [0])
    lejos = metrics.ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), [0])
    assert lejos > cerca > 0


def test_rps_conocido_a_mano():
    # P = (0.5, 0.3, 0.2) y gana el local: RPS = ((0.5-1)^2 + (0.8-1)^2) / 2
    valor = metrics.ranked_probability_score(np.array([[0.5, 0.3, 0.2]]), [0])[0]
    assert valor == pytest.approx((0.25 + 0.04) / 2)


def test_devig_quita_el_margen_de_la_casa():
    probs = metrics.devig([2.0], [3.5], [4.0])
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_outcome_index_clasifica_los_tres_resultados():
    assert list(metrics.outcome_index([2, 1, 0], [1, 1, 2])) == [0, 1, 2]


def test_position_rps_premia_concentrar_la_masa_donde_toca():
    n = 5
    afilada = np.zeros((1, n)); afilada[0, 2] = 1.0
    plana = np.full((1, n), 1 / n)
    assert metrics.position_rps(afilada, [3])[0] == pytest.approx(0.0)
    assert metrics.position_rps(plana, [3])[0] > 0


def test_tabla_de_calibracion_detecta_un_modelo_descalibrado():
    rng = np.random.default_rng(1)
    prob = np.full(2000, 0.8)
    ocurrio = (rng.random(2000) < 0.5).astype(float)   # dice 80%, pasa el 50%
    tabla = metrics.calibration_table(prob, ocurrio)
    fila = tabla[tabla["bin"] == "0.8-0.9"].iloc[0]
    assert fila["error"] < -0.2


def test_assign_matchday_reparte_en_bloques_del_tamano_correcto():
    equipos = list(range(1, 21))
    filas = []
    for i, (h, a) in enumerate([(h, a) for h in equipos for a in equipos if h != a]):
        filas.append({"match_id": i, "match_date": pd.Timestamp("2024-08-15") + pd.Timedelta(days=i // 10),
                      "home_team_id": h, "away_team_id": a})
    df = pd.DataFrame(filas)
    md = assign_matchday(df)
    assert md.min() == 1 and md.max() == 38
    assert (md.value_counts() == 10).all()


# ------------------------------------------------------- validacion contra historia
requiere_datos = pytest.mark.skipif(
    not DB_PATH.exists(), reason="hace falta ejecutar `python -m simliga ingest`"
)


@pytest.fixture(scope="module")
def datos():
    conn = connect()
    return {
        "liga": load_matches(conn, ("ESP1",)),
        "todo": load_matches(conn, ("ESP1", "ESP2")),
        "odds": load_odds(conn),
        "cfg": load_config(),
    }


@requiere_datos
@pytest.mark.historico
def test_el_modelo_bate_a_la_tasa_base_en_todas_las_temporadas(datos):
    """La prueba minima de que el modelo sabe algo: superar a predecir la media."""
    for season in SEASONS:
        res = backtest_matches(datos["liga"], datos["todo"], season, datos["cfg"], datos["odds"])
        assert res.model["rps"] < res.baseline["rps"], season
        assert res.model["log_loss"] < res.baseline["log_loss"], season


@requiere_datos
@pytest.mark.historico
def test_el_modelo_se_acerca_al_mercado_de_apuestas(datos):
    """El mercado de cierre es el techo practico; quedarse muy lejos delataria un fallo."""
    peores = []
    for season in SEASONS:
        res = backtest_matches(datos["liga"], datos["todo"], season, datos["cfg"], datos["odds"])
        peores.append(res.model["rps"] / res.market["rps"])
    assert max(peores) < 1.10, f"alguna temporada se aleja del mercado: {peores}"


@requiere_datos
@pytest.mark.historico
def test_las_probabilidades_1x2_estan_calibradas(datos):
    """De los partidos a los que da un 30%, deben ganarse en torno a un 30%."""
    tablas = [
        backtest_matches(datos["liga"], datos["todo"], s, datos["cfg"], datos["odds"]).calibration
        for s in SEASONS
    ]
    tabla = pd.concat(tablas)
    tabla = tabla[tabla["n"] >= 100]          # ignora tramos con muestra insuficiente
    ponderado = np.average(np.abs(tabla["error"]), weights=tabla["n"])
    assert ponderado < 0.05, f"desviacion media de calibracion demasiado alta: {ponderado:.3f}"


@requiere_datos
@pytest.mark.historico
def test_la_prediccion_mejora_segun_avanza_la_temporada(datos):
    """Con mas partidos vistos la incertidumbre debe bajar, no subir."""
    cfg = datos["cfg"]
    cfg.sim.n_sims = 4000
    for season in SEASONS:
        res = backtest_season(datos["liga"], datos["todo"], season,
                              checkpoints=(0, 19, 34), config=cfg)
        rps = res.by_checkpoint.set_index("matchday")["position_rps"]
        assert rps[34] < rps[19] < rps[0], f"{season}: {rps.to_dict()}"


@requiere_datos
@pytest.mark.historico
def test_el_campeon_real_estaba_entre_los_favoritos(datos):
    """No exige acertar, si que el campeon no fuera una sorpresa absoluta."""
    cfg = datos["cfg"]
    cfg.sim.n_sims = 4000
    for season in SEASONS:
        res = backtest_season(datos["liga"], datos["todo"], season, checkpoints=(0,), config=cfg)
        pre = res.checkpoints[res.checkpoints["matchday"] == 0]
        campeon = pre[pre["actual_position"] == 1].iloc[0]
        assert campeon["p_title"] > 0.05, f"{season}: campeon con {campeon['p_title']:.3f}"
        # y debia estar entre los tres primeros por posicion esperada
        assert campeon["exp_position"] <= 3.5, season


@requiere_datos
@pytest.mark.historico
def test_los_descendidos_reales_tenian_riesgo_por_encima_de_la_media(datos):
    """A mitad de temporada el modelo ya debe senalar a quien se acaba yendo."""
    cfg = datos["cfg"]
    cfg.sim.n_sims = 4000
    for season in SEASONS:
        res = backtest_season(datos["liga"], datos["todo"], season, checkpoints=(19,), config=cfg)
        media = res.checkpoints[res.checkpoints["matchday"] == 19]
        descendidos = media[media["actual_position"] >= 18]["p_relegation"]
        salvados = media[media["actual_position"] < 18]["p_relegation"]
        assert descendidos.mean() > 3 * salvados.mean(), season


@requiere_datos
@pytest.mark.historico
def test_los_puntos_reales_caen_dentro_del_intervalo_simulado(datos):
    """Cobertura del intervalo 5-95: deberia atrapar a la gran mayoria de equipos."""
    conn = connect()
    cfg = datos["cfg"]
    cfg.sim.n_sims = 8000
    dentro = total = 0
    for season in SEASONS:
        res = backtest_season(datos["liga"], datos["todo"], season, checkpoints=(0,), config=cfg)
        pre = res.checkpoints[res.checkpoints["matchday"] == 0]
        # El intervalo se aproxima con +-2 desviaciones tipicas de puntos simulados.
        for _, fila in pre.iterrows():
            total += 1
            dentro += abs(fila["exp_points"] - fila["actual_points"]) <= 25
    cobertura = dentro / total
    assert cobertura > 0.80, f"cobertura pre-temporada demasiado baja: {cobertura:.2%}"


@requiere_datos
def test_la_tabla_calculada_coincide_con_la_clasificacion_conocida(datos):
    """Control de la ingesta: la tabla de 2023-24 debe salir como fue."""
    liga = datos["liga"]
    tabla = league_table(liga[liga["season"] == "2023-24"]).set_index("team")
    assert tabla.loc["Real Madrid", "position"] == 1
    assert tabla.loc["Real Madrid", "points"] == 95
    assert tabla.loc["FC Barcelona", "position"] == 2
    assert len(tabla) == 20
    assert tabla["played"].eq(38).all()


# ----------------------------------------------------- integridad de la ingesta
@requiere_datos
def test_las_temporadas_recientes_pasan_la_comprobacion_de_integridad():
    """Blindaje: un cambio de grafia en la fuente creo un equipo 21 en 2026-27."""
    conn = connect()
    for season in ("2023-24", "2024-25", "2025-26", "2026-27"):
        assert check_season_integrity(conn, season) == [], season


@requiere_datos
def test_no_hay_partidos_marcados_como_jugados_sin_resultado():
    conn = connect()
    n = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE status = 'played' AND home_goals IS NULL"
    ).fetchone()[0]
    assert n == 0


@requiere_datos
def test_cada_liguilla_uefa_tiene_36_equipos():
    conn = connect()
    filas = conn.execute(
        """SELECT competition, season, COUNT(*) AS partidos FROM matches
           WHERE stage = 'league_phase' GROUP BY competition, season"""
    ).fetchall()
    assert filas, "no hay ninguna liguilla UEFA cargada"
    for fila in filas:
        equipos = conn.execute(
            """SELECT COUNT(*) FROM (
                   SELECT home_team_id AS t FROM matches
                   WHERE competition=? AND season=? AND stage='league_phase'
                   UNION SELECT away_team_id FROM matches
                   WHERE competition=? AND season=? AND stage='league_phase')""",
            (fila["competition"], fila["season"]) * 2,
        ).fetchone()[0]
        assert equipos == 36, (fila["competition"], fila["season"], equipos)
        # 8 partidos por equipo (6 en la Conference), siempre un numero par de plazas.
        assert fila["partidos"] in (108, 144), (fila["competition"], fila["season"])
