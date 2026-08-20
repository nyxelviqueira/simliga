"""Tests de la capa de persistencia: el upsert y la fusion de fichas."""
from __future__ import annotations

import pytest

from simliga.db import connect, get_or_create_team, merge_teams, upsert_match


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "prueba.sqlite")


def _partido(conn, **kw):
    base = dict(competition="ESP1", season="2026-27", stage="league",
                match_date="2026-08-16", matchday=1, home_team_id=1, away_team_id=2,
                home_goals=None, away_goals=None, status="scheduled", source="calendario")
    base.update(kw)
    return upsert_match(conn, **base)


def test_el_upsert_no_duplica_el_mismo_partido(conn):
    get_or_create_team(conn, "A"); get_or_create_team(conn, "B")
    primero = _partido(conn)
    segundo = _partido(conn)
    assert primero == segundo
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_refrescar_el_calendario_no_borra_un_resultado(conn):
    """Regresion: refrescar el calendario dejaba en blanco los partidos jugados.

    openfootball publica el calendario de la temporada en curso sin marcadores;
    al reingerirlo, el upsert escribia NULL encima de los resultados que ya
    estaban. El simulador pasaba a creer que no se habia jugado nada.
    """
    get_or_create_team(conn, "A"); get_or_create_team(conn, "B")
    _partido(conn)
    _partido(conn, home_goals=2, away_goals=0, status="played",
             source="football-data.co.uk", match_date="2026-08-19")

    _partido(conn)                       # el calendario vuelve a pasar, sin marcador

    fila = conn.execute("SELECT * FROM matches").fetchone()
    assert (fila["home_goals"], fila["away_goals"]) == (2, 0)
    assert fila["status"] == "played"
    assert fila["match_date"] == "2026-08-19", "la fecha real manda sobre la provisional"
    assert fila["source"] == "football-data.co.uk"


def test_una_fuente_con_resultado_si_puede_corregir_a_otra(conn):
    """Cuando la fuente oficial publica, debe poder pisar una entrada manual."""
    get_or_create_team(conn, "A"); get_or_create_team(conn, "B")
    _partido(conn, home_goals=2, away_goals=0, status="played", source="manual")
    _partido(conn, home_goals=2, away_goals=1, status="played",
             source="football-data.co.uk")

    fila = conn.execute("SELECT * FROM matches").fetchone()
    assert (fila["home_goals"], fila["away_goals"]) == (2, 1)
    assert fila["source"] == "football-data.co.uk"


def test_un_partido_sin_jugar_si_admite_cambio_de_fecha(conn):
    get_or_create_team(conn, "A"); get_or_create_team(conn, "B")
    _partido(conn)
    _partido(conn, match_date="2026-09-01")
    assert conn.execute("SELECT match_date FROM matches").fetchone()[0] == "2026-09-01"


def test_los_dos_partidos_de_una_eliminatoria_conviven(conn):
    """Ida y vuelta comparten fase pero invierten la localia: son filas distintas."""
    for nombre in ("A", "B"):
        get_or_create_team(conn, nombre)
    _partido(conn, competition="UCL", stage="R16", home_team_id=1, away_team_id=2)
    _partido(conn, competition="UCL", stage="R16", home_team_id=2, away_team_id=1)
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 2


# ------------------------------------------------------------------ fusion de fichas
def test_fusionar_repunta_los_partidos_de_la_ficha_vieja(conn):
    for nombre in ("Deportivo", "Dep. A Coruna", "Elche"):
        get_or_create_team(conn, nombre)
    bueno, malo, otro = 1, 2, 3
    _partido(conn, home_team_id=malo, away_team_id=otro, match_date="2026-08-17",
             home_goals=1, away_goals=1, status="played")

    resultado = merge_teams(conn, malo, bueno)
    assert resultado["repuntados"] == 1
    fila = conn.execute("SELECT home_team_id FROM matches").fetchone()
    assert fila["home_team_id"] == bueno
    assert conn.execute("SELECT COUNT(*) FROM teams WHERE team_id = ?", (malo,)).fetchone()[0] == 0


def test_fusionar_descarta_el_partido_duplicado(conn):
    """El mismo partido bajo las dos fichas: repuntarlo violaria la clave natural."""
    for nombre in ("Deportivo", "Dep. A Coruna", "Elche"):
        get_or_create_team(conn, nombre)
    bueno, malo, otro = 1, 2, 3
    _partido(conn, home_team_id=bueno, away_team_id=otro, home_goals=1, away_goals=1,
             status="played")
    _partido(conn, home_team_id=malo, away_team_id=otro, home_goals=1, away_goals=1,
             status="played")

    resultado = merge_teams(conn, malo, bueno)
    assert resultado["duplicados_borrados"] == 1
    assert conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1


def test_fusionar_un_duplicado_conserva_el_resultado_real(conn):
    for nombre in ("Deportivo Alaves", "Deportivo Alavés", "Getafe"):
        get_or_create_team(conn, nombre)
    bueno, malo, otro = 1, 2, 3
    _partido(conn, home_team_id=bueno, away_team_id=otro, match_date="2026-08-15",
             matchday=1, status="scheduled", source="openfootball/espana")
    _partido(conn, home_team_id=malo, away_team_id=otro, match_date="2026-08-16",
             matchday=None, home_goals=3, away_goals=0, status="played",
             source="football-data.co.uk")

    resultado = merge_teams(conn, malo, bueno)

    fila = conn.execute("SELECT * FROM matches").fetchone()
    assert resultado["duplicados_borrados"] == 1
    assert (fila["home_team_id"], fila["away_team_id"]) == (bueno, otro)
    assert (fila["home_goals"], fila["away_goals"]) == (3, 0)
    assert fila["status"] == "played"
    assert fila["match_date"] == "2026-08-16"
    assert fila["matchday"] == 1
    assert fila["source"] == "football-data.co.uk"


def test_fusionar_una_ficha_consigo_misma_no_hace_nada(conn):
    get_or_create_team(conn, "A")
    assert merge_teams(conn, 1, 1) == {"repuntados": 0, "duplicados_borrados": 0}
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 1


# ------------------------------------------------------------------ escenarios
def _con_partidos(conn):
    for nombre in ("A", "B"):
        get_or_create_team(conn, nombre)
    pendiente = _partido(conn, match_date="2026-09-20")
    jugado = _partido(conn, stage="league2", home_goals=2, away_goals=0, status="played")
    return pendiente, jugado


def test_se_puede_fijar_un_resultado_hipotetico(conn):
    from simliga.db import load_scenario_results, set_scenario_result

    pendiente, _ = _con_partidos(conn)
    set_scenario_result(conn, pendiente, 3, 1)
    assert load_scenario_results(conn) == {pendiente: (3, 1)}


def test_un_partido_ya_jugado_no_admite_hipotesis(conn):
    """Lo que paso, paso: dejarlo sobrescribir haria que la tabla real mintiera."""
    from simliga.db import set_scenario_result

    _, jugado = _con_partidos(conn)
    with pytest.raises(ValueError, match="no admite escenarios"):
        set_scenario_result(conn, jugado, 0, 5)


def test_un_partido_en_juego_no_admite_hipotesis(conn):
    """El directo se ensena, pero no se pisa con una hipotesis manual."""
    from simliga.db import set_scenario_result

    pendiente, _ = _con_partidos(conn)
    conn.execute(
        """UPDATE matches
           SET status = 'live', live_home_goals = 1, live_away_goals = 0,
               live_detail = ?
           WHERE match_id = ?""",
        ("63'", pendiente),
    )
    conn.commit()

    with pytest.raises(ValueError, match="no admite escenarios"):
        set_scenario_result(conn, pendiente, 0, 5)


def test_un_partido_inexistente_falla_claro(conn):
    from simliga.db import set_scenario_result

    _con_partidos(conn)
    with pytest.raises(KeyError):
        set_scenario_result(conn, 99999, 1, 0)


def test_no_se_admiten_goles_negativos(conn):
    from simliga.db import set_scenario_result

    pendiente, _ = _con_partidos(conn)
    with pytest.raises(ValueError):
        set_scenario_result(conn, pendiente, -1, 0)


def test_volver_a_fijar_el_mismo_partido_lo_sustituye(conn):
    from simliga.db import load_scenario_results, set_scenario_result

    pendiente, _ = _con_partidos(conn)
    set_scenario_result(conn, pendiente, 1, 0)
    set_scenario_result(conn, pendiente, 0, 4)
    assert load_scenario_results(conn) == {pendiente: (0, 4)}


def test_borrar_una_hipotesis(conn):
    from simliga.db import clear_scenario_result, load_scenario_results, set_scenario_result

    pendiente, _ = _con_partidos(conn)
    set_scenario_result(conn, pendiente, 1, 1)
    assert clear_scenario_result(conn, pendiente) is True
    assert load_scenario_results(conn) == {}
    assert clear_scenario_result(conn, pendiente) is False


def test_si_el_partido_se_juega_de_verdad_la_hipotesis_deja_de_contar(conn):
    """El resultado real siempre gana sobre el supuesto, sin avisar de nada."""
    from simliga.db import load_scenario_results, set_scenario_result

    pendiente, _ = _con_partidos(conn)
    set_scenario_result(conn, pendiente, 3, 0)
    conn.execute(
        "UPDATE matches SET home_goals=0, away_goals=1, status='played' WHERE match_id=?",
        (pendiente,))
    conn.commit()
    assert load_scenario_results(conn) == {}


def test_limpiar_borra_todas_las_hipotesis(conn):
    from simliga.db import clear_all_scenarios, load_scenario_results, set_scenario_result

    pendiente, _ = _con_partidos(conn)
    set_scenario_result(conn, pendiente, 1, 1)
    assert clear_all_scenarios(conn) == 1
    assert load_scenario_results(conn) == {}


def test_se_registra_y_se_lee_el_campeon_de_copa(conn):
    from simliga.db import get_cup_winner, set_cup_winner

    get_or_create_team(conn, "Real Sociedad")
    assert get_cup_winner(conn, "2025-26") is None
    set_cup_winner(conn, "2025-26", 1)
    assert get_cup_winner(conn, "2025-26") == "Real Sociedad"


def test_registrar_de_nuevo_sustituye_al_campeon(conn):
    from simliga.db import get_cup_winner, set_cup_winner

    get_or_create_team(conn, "Real Sociedad")
    get_or_create_team(conn, "FC Barcelona")
    set_cup_winner(conn, "2025-26", 1)
    set_cup_winner(conn, "2025-26", 2)
    assert get_cup_winner(conn, "2025-26") == "FC Barcelona"
