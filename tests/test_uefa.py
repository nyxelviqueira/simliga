"""Tests del motor UEFA: cuadro, eliminatorias y coherencia de las fases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simliga.config import DixonColesConfig, SimConfig
from simliga.ingest.club_names import normalizar
from simliga.ingest.uefa import _fase_de_seccion, parse
from simliga.model.dixon_coles import DixonColesFit
from simliga.model.league_strength import apply_offsets, estimate_league_offsets
from simliga.sim.uefa import (PLAYOFF_PAIRINGS, R16_PAIRINGS, build_all_cdfs,
                              simulate_tie, simulate_uefa)

FICHERO = """= UEFA Champions League 2025/26

# Teams      36


▪ League, Matchday 1
  Tue Sep 16 2025
    18:45  Athletic Club (ESP)     v Arsenal FC (ENG)         0-2 (0-0)
           PSV (NED)               v Royale Union Saint-Gilloise (BEL)  1-3 (0-2)

▪ Playoffs, Matchday 2
  Tue Feb 24
    21:00  Real Madrid CF (ESP)    v Sport Lisboa e Benfica (POR)  2-1 (1-1)
           Juventus FC (ITA)       v Galatasaray SK (TUR)     3-2 a.e.t. (3-0, 1-0)

▪ Finals, Quarterfinals
  Tue Apr 7
    21:00  FC Barcelona (ESP)      v Club Atlético de Madrid (ESP)  0-2 (0-1)

▪ Finals, Final
  Sat May 30 2026
    21:00  Arsenal FC (ENG)        v FC Barcelona (ESP)
"""


# --------------------------------------------------------------------- parser
def test_parsea_fases_jornadas_y_paises():
    df = parse(FICHERO, "2025-26")
    assert len(df) == 6
    assert list(df["stage"]) == ["league_phase", "league_phase", "playoff",
                                 "playoff", "QF", "F"]
    assert df.iloc[0]["home_country"] == "ESP" and df.iloc[0]["away_country"] == "ENG"
    assert df.iloc[0]["matchday"] == 1


def test_parsea_marcadores_incluida_la_proroga():
    df = parse(FICHERO, "2025-26")
    juve = df[df["home"] == "Juventus FC"].iloc[0]
    assert (juve["home_goals"], juve["away_goals"]) == (3, 2)


def test_un_partido_sin_jugar_queda_sin_marcador():
    df = parse(FICHERO, "2025-26")
    final = df[df["stage"] == "F"].iloc[0]
    assert pd.isna(final["home_goals"])
    assert final["home"] == "Arsenal FC" and final["away"] == "FC Barcelona"


def test_el_ano_de_los_meses_de_invierno_es_el_siguiente():
    df = parse(FICHERO, "2025-26")
    assert df[df["stage"] == "QF"].iloc[0]["match_date"] == pd.Timestamp("2026-04-07")
    assert df.iloc[0]["match_date"] == pd.Timestamp("2025-09-16")


@pytest.mark.parametrize("seccion,esperado", [
    ("League, Matchday 5", ("league_phase", 5)),
    ("Playoffs, Matchday 1", ("playoff", 1)),
    ("Playoffs", ("playoff", None)),
    ("Finals, Round of 16", ("R16", None)),
    ("Finals, Quarterfinals", ("QF", None)),
    ("Finals, Semifinals", ("SF", None)),
    ("Finals, Final", ("F", None)),
])
def test_traduce_las_cabeceras_de_seccion(seccion, esperado):
    assert _fase_de_seccion(seccion) == esperado


# ------------------------------------------------------- normalizacion de nombres
@pytest.mark.parametrize("a,b", [
    ("FC Bayern München", "FC Bayern Munchen"),
    ("Club Atlético de Madrid", "Club Atletico de Madrid"),
    ("FK Bodø/Glimt", "FK Bodo Glimt"),
    ("Qarabag Agdam FK", "qarabag agdam fk"),
])
def test_la_normalizacion_ignora_acentos_y_puntuacion(a, b):
    assert normalizar(a) == normalizar(b)


# ------------------------------------------------------------------ cuadro
def test_el_cuadro_cubre_las_posiciones_9_a_24_sin_repetir():
    posiciones = [p for alto, bajo in PLAYOFF_PAIRINGS for p in alto + bajo]
    assert sorted(posiciones) == list(range(9, 25))


def test_las_cabezas_de_serie_cubren_las_posiciones_1_a_8():
    posiciones = [p for par, _ in R16_PAIRINGS for p in par]
    assert sorted(posiciones) == list(range(1, 9))


def test_cada_playoff_alimenta_una_sola_llave_de_octavos():
    indices = [idx for _, idx in R16_PAIRINGS]
    assert sorted(indices) == [0, 1, 2, 3]


# ------------------------------------------------------------- eliminatorias
def _fit(n=36, fuerza=None):
    fuerza = np.zeros(n) if fuerza is None else np.asarray(fuerza, dtype=float)
    return DixonColesFit(
        team_ids=list(range(n)), attack=fuerza.copy(), defence=fuerza * 0.7,
        mu=0.2, home_advantage=0.3, rho=0.03, kappa_attack=0.5, kappa_defence=0.35,
        n_matches=500, effective_n=300.0, log_likelihood=-1.0, converged=True,
        config=DixonColesConfig(max_goals=8),
    )


def test_la_eliminatoria_siempre_produce_un_ganador_de_los_dos():
    fit = _fit(4)
    cdfs, n_cols = build_all_cdfs(fit)
    rng = np.random.default_rng(0)
    a = np.zeros(2000, dtype=int)
    b = np.ones(2000, dtype=int)
    ganador = simulate_tie(cdfs, n_cols, fit, a, b, rng)
    assert set(np.unique(ganador)) <= {0, 1}
    assert len(ganador) == 2000


def test_entre_iguales_la_eliminatoria_es_una_moneda():
    """Sin diferencia de fuerza, cada equipo debe pasar la mitad de las veces."""
    fit = _fit(4)
    cdfs, n_cols = build_all_cdfs(fit)
    rng = np.random.default_rng(3)
    n = 40000
    ganador = simulate_tie(cdfs, n_cols, fit, np.zeros(n, dtype=int),
                           np.ones(n, dtype=int), rng)
    assert (ganador == 0).mean() == pytest.approx(0.5, abs=0.02)


def test_el_equipo_mas_fuerte_pasa_mas_veces():
    fit = _fit(4, fuerza=[0.6, -0.6, 0.0, 0.0])
    cdfs, n_cols = build_all_cdfs(fit)
    rng = np.random.default_rng(5)
    n = 20000
    ganador = simulate_tie(cdfs, n_cols, fit, np.zeros(n, dtype=int),
                           np.ones(n, dtype=int), rng)
    assert (ganador == 0).mean() > 0.75


def test_las_cdfs_precalculadas_son_distribuciones_validas():
    fit = _fit(5)
    cdfs, n_cols = build_all_cdfs(fit)
    assert cdfs.shape == (5, 5, n_cols * n_cols)
    assert np.allclose(cdfs[:, :, -1], 1.0)
    assert (np.diff(cdfs, axis=2) >= -1e-12).all()      # monotona no decreciente


# ------------------------------------------------------- competicion completa
def _liguilla(n_teams=36, partidos_por_equipo=8):
    """Calendario de liguilla: cada equipo con rivales distintos y mitad en casa."""
    filas, mid = [], 0
    for salto in range(1, partidos_por_equipo // 2 + 1):
        for i in range(n_teams):
            j = (i + salto) % n_teams
            local, visitante = (i, j) if (i + salto) % 2 == 0 else (j, i)
            filas.append({
                "match_id": mid, "competition": "UCL", "season": "2025-26",
                "stage": "league_phase",
                "match_date": pd.Timestamp("2025-09-16") + pd.Timedelta(days=mid // 18),
                "home_team_id": local, "away_team_id": visitante,
                "home_goals": None, "away_goals": None, "status": "scheduled",
                "home_team": str(local), "away_team": str(visitante),
            })
            mid += 1
    return pd.DataFrame(filas)


@pytest.fixture(scope="module")
def resultado_uefa():
    fit = _fit(36, fuerza=np.linspace(0.5, -0.5, 36))
    return simulate_uefa(fit, _liguilla(), played=None, teams=list(range(36)),
                         config=SimConfig(n_sims=4000, seed=1))


def test_cada_ronda_tiene_exactamente_las_plazas_que_toca(resultado_uefa):
    esperado = {"playoff": 16, "R16": 16, "QF": 8, "SF": 4, "F": 2, "winner": 1}
    for fase, plazas in esperado.items():
        por_simulacion = resultado_uefa.reached[fase].sum(axis=1)
        assert (por_simulacion == plazas).all(), fase


def test_las_probabilidades_de_ronda_son_decrecientes(resultado_uefa):
    """Nadie puede llegar a semifinales mas veces que a cuartos."""
    probs = resultado_uefa.stage_probabilities()
    for antes, despues in (("R16", "QF"), ("QF", "SF"), ("SF", "F"), ("F", "winner")):
        assert (probs[despues] <= probs[antes] + 1e-9).all(), (antes, despues)


def test_los_desenlaces_de_la_liguilla_suman_uno(resultado_uefa):
    liguilla = resultado_uefa.league_phase_outcomes()
    total = (liguilla["direct_to_r16"] + liguilla["playoff"]
             + liguilla["eliminated_in_league_phase"])
    assert np.allclose(total, 1.0)


def test_pasar_directo_implica_estar_en_octavos(resultado_uefa):
    liguilla = resultado_uefa.league_phase_outcomes()
    probs = resultado_uefa.stage_probabilities()
    assert (probs["R16"] >= liguilla["direct_to_r16"] - 1e-9).all()


def test_el_favorito_gana_mas_que_el_ultimo_cabeza_de_serie(resultado_uefa):
    probs = resultado_uefa.stage_probabilities()
    assert probs["winner"][0] > probs["winner"][-1]
    assert probs["winner"].sum() == pytest.approx(1.0, abs=1e-6)


def test_la_suma_de_probabilidades_de_final_es_dos(resultado_uefa):
    probs = resultado_uefa.stage_probabilities()
    assert probs["F"].sum() == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------- desplazamiento por liga
def test_el_desplazamiento_detecta_la_liga_mas_fuerte():
    """Si una liga gana siempre en Europa con el mismo Elo, debe salir por encima."""
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "home_league": ["FUERTE"] * n + ["DEBIL"] * n,
        "away_league": ["DEBIL"] * n + ["FUERTE"] * n,
        "home_elo": 1600.0, "away_elo": 1600.0,
        "home_goals": np.concatenate([np.full(n, 3), np.full(n, 0)]),
        "away_goals": np.concatenate([np.full(n, 0), np.full(n, 3)]),
    })
    off = estimate_league_offsets(df, home_advantage=0.0)
    assert off["FUERTE"] > off["DEBIL"]
    assert sum(off.values()) == pytest.approx(0.0, abs=1e-6)


def test_una_liga_sin_partidos_europeos_no_recibe_desplazamiento():
    df = pd.DataFrame({
        "home_league": ["A"] * 30 + ["B"] * 30,
        "away_league": ["B"] * 30 + ["A"] * 30,
        "home_elo": 1500.0, "away_elo": 1500.0,
        "home_goals": 1, "away_goals": 1,
    })
    off = estimate_league_offsets(df, home_advantage=70.0)
    assert "SEGUNDA" not in off


def test_apply_offsets_solo_toca_a_quien_tiene_liga_conocida():
    ratings = {1: 1500.0, 2: 1600.0, 3: 1400.0}
    ajustados = apply_offsets(ratings, {1: "A", 2: "B"}, {"A": 50.0, "B": -50.0})
    assert ajustados == {1: 1550.0, 2: 1550.0, 3: 1400.0}


# ---------------------------------------------------- sorteo de la liguilla
from simliga.sim.uefa import draw_league_phase, draw_to_fixtures      # noqa: E402


@pytest.fixture(params=range(6))
def sorteo(request):
    equipos = list(range(36))
    ratings = {t: 1900 - 10 * t for t in equipos}
    rng = np.random.default_rng(request.param)
    return equipos, draw_league_phase(equipos, ratings, rng)


def test_el_sorteo_produce_los_144_partidos_de_la_liguilla(sorteo):
    _, partidos = sorteo
    assert len(partidos) == 144


def test_cada_equipo_juega_cuatro_en_casa_y_cuatro_fuera(sorteo):
    equipos, partidos = sorteo
    for equipo in equipos:
        assert sum(1 for h, _ in partidos if h == equipo) == 4
        assert sum(1 for _, a in partidos if a == equipo) == 4


def test_los_ocho_rivales_de_cada_equipo_son_distintos(sorteo):
    """El formato lo exige, y sin cuidado salen repetidos: un cruce de ida y
    vuelta contra el mismo rival deja al equipo con seis rivales en vez de ocho."""
    equipos, partidos = sorteo
    rivales = {t: [] for t in equipos}
    for h, a in partidos:
        rivales[h].append(a)
        rivales[a].append(h)
    for equipo, lista in rivales.items():
        assert len(lista) == 8, equipo
        assert len(set(lista)) == 8, f"{equipo} repite rival"


def test_nadie_se_enfrenta_a_si_mismo(sorteo):
    _, partidos = sorteo
    assert all(h != a for h, a in partidos)


def test_cada_equipo_cruza_dos_veces_con_cada_bombo(sorteo):
    """Dos rivales de cada uno de los cuatro bombos, que es la regla del formato."""
    equipos, partidos = sorteo
    bombo = {t: t // 9 for t in equipos}      # ya vienen ordenados por rating
    for equipo in equipos:
        rivales = [a for h, a in partidos if h == equipo]
        rivales += [h for h, a in partidos if a == equipo]
        cuenta = [0, 0, 0, 0]
        for r in rivales:
            cuenta[bombo[r]] += 1
        assert cuenta == [2, 2, 2, 2], (equipo, cuenta)


def test_el_sorteo_se_convierte_en_partidos_utilizables():
    equipos = list(range(36))
    rng = np.random.default_rng(3)
    partidos = draw_league_phase(equipos, {t: 1500 for t in equipos}, rng)
    df = draw_to_fixtures(partidos, "2026-27", "UCL", pd.Timestamp("2026-09-15"))
    assert len(df) == 144
    assert set(df.columns) >= {"match_id", "competition", "season", "stage",
                               "match_date", "home_team_id", "away_team_id"}
    assert df["home_goals"].isna().all()
    assert (df["stage"] == "league_phase").all()


def test_dos_sorteos_con_semillas_distintas_no_coinciden():
    equipos = list(range(36))
    ratings = {t: 1900 - 10 * t for t in equipos}
    a = draw_league_phase(equipos, ratings, np.random.default_rng(1))
    b = draw_league_phase(equipos, ratings, np.random.default_rng(2))
    assert set(a) != set(b), "sin variedad de sorteos no se mide la suerte del bombo"
