"""Tests del calendario con fechas y horarios reales (ESPN)."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from simliga.db import connect, get_or_create_team, upsert_match
from simliga.config import load_config
from simliga.ingest import espn
from simliga.ingest.espn import parse
from simliga.output import contract

RESPUESTA = {
    "events": [
        {
            "date": "2026-08-27T18:30Z",
            "competitions": [{
                "status": {"type": {"completed": False}},
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Celta Vigo"}, "score": "0"},
                    {"homeAway": "away", "team": {"displayName": "Osasuna"}, "score": "0"},
                ]
            }],
        },
        {
            "date": "2026-08-22T15:00Z",
            "competitions": [{
                "status": {"type": {"completed": False}},
                "competitors": [
                    {"homeAway": "away", "team": {"displayName": "Sevilla"}, "score": "0"},
                    {"homeAway": "home", "team": {"displayName": "Athletic Club"}, "score": "0"},
                ]
            }],
        },
    ]
}

RESPUESTA_CON_RESULTADO = {
    "events": [{
        "date": "2026-08-19T19:00Z",
        "competitions": [{
            "status": {"type": {"completed": True}},
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Atlético Madrid"}, "score": "2"},
                {"homeAway": "away", "team": {"displayName": "Málaga"}, "score": "0"},
            ],
        }],
    }]
}

RESPUESTA_EN_JUEGO = {
    "events": [{
        "date": "2026-08-19T19:00Z",
        "competitions": [{
            "status": {
                "displayClock": "63'",
                "type": {
                    "completed": False,
                    "state": "in",
                    "name": "STATUS_IN_PROGRESS",
                    "shortDetail": "63'",
                },
            },
            "competitors": [
                {"homeAway": "home", "team": {"displayName": "Atlético Madrid"}, "score": "1"},
                {"homeAway": "away", "team": {"displayName": "Málaga"}, "score": "0"},
            ],
        }],
    }]
}


def test_separa_fecha_y_hora_del_instante_utc():
    df = parse(RESPUESTA)
    assert len(df) == 2
    primero = df.iloc[0]
    assert primero["match_date"] == "2026-08-27"
    assert primero["kickoff_utc"] == "18:30"


def test_respeta_quien_es_local_aunque_venga_en_otro_orden():
    """ESPN no garantiza el orden de `competitors`: manda el campo homeAway."""
    df = parse(RESPUESTA)
    segundo = df.iloc[1]
    assert segundo["home"] == "Athletic Club"
    assert segundo["away"] == "Sevilla"


def test_los_partidos_finalizados_traen_marcador():
    df = parse(RESPUESTA_CON_RESULTADO)
    fila = df.iloc[0]
    assert fila["status"] == "played"
    assert fila["home_goals"] == 2
    assert fila["away_goals"] == 0


def test_los_ceros_de_un_partido_programado_no_son_resultado():
    df = parse(RESPUESTA)
    fila = df.iloc[0]
    assert fila["status"] == "scheduled"
    assert pd.isna(fila["home_goals"])
    assert pd.isna(fila["away_goals"])


def test_los_partidos_en_juego_traen_marcador_parcial_sin_ser_finales():
    df = parse(RESPUESTA_EN_JUEGO)
    fila = df.iloc[0]
    assert fila["status"] == "live"
    assert pd.isna(fila["home_goals"])
    assert pd.isna(fila["away_goals"])
    assert fila["live_home_goals"] == 1
    assert fila["live_away_goals"] == 0
    assert fila["live_detail"] == "63'"


def test_los_estados_de_futbol_de_espn_tambien_cuentan_como_en_juego():
    respuesta = copy.deepcopy(RESPUESTA_EN_JUEGO)
    tipo = respuesta["events"][0]["competitions"][0]["status"]["type"]
    tipo.pop("state")
    tipo["name"] = "STATUS_FIRST_HALF"

    fila = parse(respuesta).iloc[0]
    assert fila["status"] == "live"
    assert fila["live_home_goals"] == 1
    assert pd.isna(fila["home_goals"])


def test_un_evento_incompleto_se_descarta_sin_romper():
    incompleto = {"events": [{"date": "2026-09-01T18:00Z", "competitions": [{"competitors": []}]}]}
    assert parse(incompleto).empty


def test_espn_marca_como_jugado_un_resultado_final(tmp_path, monkeypatch):
    conn = connect(tmp_path / "simliga.sqlite")
    local = get_or_create_team(conn, "Atletico de Madrid")
    visitante = get_or_create_team(conn, "Malaga CF")
    conn.execute(
        "INSERT INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
        ("Atlético Madrid", "espn", local),
    )
    conn.execute(
        "INSERT INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
        ("Málaga", "espn", visitante),
    )
    upsert_match(
        conn, competition="ESP1", season="2026-27", stage="league",
        match_date="2026-08-19", matchday=1, home_team_id=local,
        away_team_id=visitante, home_goals=None, away_goals=None,
        status="scheduled", source="openfootball/espana",
    )
    monkeypatch.setattr(espn, "descargar", lambda *a, **k: RESPUESTA_CON_RESULTADO)

    resumen = espn.update_schedule(conn, "2026-27", force_download=True)

    fila = conn.execute("SELECT * FROM matches").fetchone()
    assert resumen["resultados"] == 1
    assert (fila["home_goals"], fila["away_goals"]) == (2, 0)
    assert fila["status"] == "played"
    assert fila["source"] == "espn"


def test_espn_marca_en_juego_sin_guardarlo_como_resultado_final(tmp_path, monkeypatch):
    conn = connect(tmp_path / "simliga.sqlite")
    local = get_or_create_team(conn, "Atletico de Madrid")
    visitante = get_or_create_team(conn, "Malaga CF")
    conn.execute(
        "INSERT INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
        ("Atlético Madrid", "espn", local),
    )
    conn.execute(
        "INSERT INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
        ("Málaga", "espn", visitante),
    )
    upsert_match(
        conn, competition="ESP1", season="2026-27", stage="league",
        match_date="2026-08-19", matchday=1, home_team_id=local,
        away_team_id=visitante, home_goals=None, away_goals=None,
        status="scheduled", source="openfootball/espana",
    )
    monkeypatch.setattr(espn, "descargar", lambda *a, **k: RESPUESTA_EN_JUEGO)

    resumen = espn.update_schedule(conn, "2026-27", force_download=True)

    fila = conn.execute("SELECT * FROM matches").fetchone()
    assert resumen["en_juego"] == 1
    assert fila["status"] == "live"
    assert fila["home_goals"] is None
    assert fila["away_goals"] is None
    assert (fila["live_home_goals"], fila["live_away_goals"]) == (1, 0)
    assert fila["live_detail"] == "63'"


# ------------------------------------------- deteccion de fecha provisional
def _pendientes(filas):
    return pd.DataFrame(
        [{"match_date": pd.Timestamp(f), "kickoff_utc": h, "matchday": j}
         for f, h, j in filas])


def test_una_jornada_con_dias_y_horas_variados_esta_confirmada():
    """Con horario real, LaLiga reparte la jornada; ese es el indicio bueno."""
    pendientes = _pendientes([
        ("2026-08-22", "15:00", 2), ("2026-08-22", "17:30", 2), ("2026-08-22", "19:30", 2),
        ("2026-08-23", "15:00", 2), ("2026-08-23", "17:30", 2), ("2026-08-23", "19:30", 2),
        ("2026-08-24", "17:30", 2), ("2026-08-24", "19:30", 2),
    ])
    marcas = contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20"))
    assert not marcas.any()


def test_una_jornada_entera_a_la_misma_hora_es_provisional():
    """El marcador de posicion de las fuentes: los diez el mismo dia y hora."""
    pendientes = _pendientes([("2027-05-09", "18:00", 36)] * 10)
    marcas = contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20"))
    assert marcas.all()


def test_tres_partidos_a_la_vez_el_mismo_dia_delatan_el_marcador():
    pendientes = _pendientes([
        ("2026-10-04", "18:00", 8), ("2026-10-04", "18:00", 8), ("2026-10-04", "18:00", 8),
        ("2026-10-03", "20:00", 8),
    ])
    marcas = contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20"))
    assert list(marcas) == [True, True, True, False]


def test_sin_hora_conocida_se_cae_al_criterio_de_solo_fechas():
    pendientes = pd.DataFrame([
        {"match_date": pd.Timestamp("2027-05-09"), "matchday": 36} for _ in range(10)
    ])
    assert contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20")).all()


def test_una_fecha_pasada_sigue_siendo_provisional_aunque_tenga_hora():
    pendientes = _pendientes([("2026-08-16", "17:00", 1)])
    assert contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20")).all()


# --------------------------------------------------------- hora en el JSON
def test_la_hora_se_emite_en_iso_utc(documento_con_hora):
    partido = documento_con_hora
    assert partido["kickoff_utc"] == "2026-08-27T18:30:00Z"


@pytest.fixture
def documento_con_hora():
    class Fila:
        match_date = "2026-08-27"
        kickoff_utc = "18:30"

    return {"kickoff_utc": contract._hora_iso(Fila())}


def test_sin_hora_el_campo_va_a_nulo():
    class Fila:
        match_date = "2026-08-27"
        kickoff_utc = None

    assert contract._hora_iso(Fila()) is None


# ------------------------------------------------- sondeo barato de un dia
def test_preguntar_por_un_dia_pide_solo_ese_dia(monkeypatch):
    """La respuesta de la temporada entera pesa 2,7 MB; la de un dia, 41 KB.

    Sondeando cada tres cuartos de minuto la diferencia deja de ser un detalle.
    """
    pedidas = []

    class Respuesta:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"events": []}

    monkeypatch.setattr(espn.requests, "get",
                        lambda url, **k: (pedidas.append(url), Respuesta())[1])

    espn.descargar_dias(["2026-08-22"])
    assert pedidas[-1].endswith("dates=20260822")

    espn.descargar_dias(["2026-08-23", "2026-08-22"])
    assert pedidas[-1].endswith("dates=20260822-20260823"), "de la primera fecha a la ultima"


def test_un_dia_suelto_no_se_guarda_en_disco(monkeypatch, tmp_path):
    """El marcador que se viene a buscar caduca en segundos: cachearlo estorba."""
    class Respuesta:
        content = b'{"events": []}'
        def raise_for_status(self): pass
        def json(self): return {"events": []}

    monkeypatch.setattr(espn, "RAW_DIR", tmp_path)
    monkeypatch.setattr(espn.requests, "get", lambda url, **k: Respuesta())
    espn.descargar_dias(["2026-08-22"])
    assert list(tmp_path.iterdir()) == []


def test_el_id_del_evento_se_guarda_para_poder_seguirlo(tmp_path, monkeypatch):
    """Sin el, el panel tendria que casar los partidos por nombre de equipo.

    Que es justo de donde salen las fichas duplicadas cuando una fuente cambia
    de grafia, y ademas obligaria a repetir esa logica en el navegador.
    """
    conn = connect(tmp_path / "simliga.sqlite")
    local = get_or_create_team(conn, "Atletico de Madrid")
    visitante = get_or_create_team(conn, "Malaga CF")
    for alias, tid in (("Atlético Madrid", local), ("Málaga", visitante)):
        conn.execute("INSERT INTO team_aliases (alias, source, team_id) VALUES (?, ?, ?)",
                     (alias, "espn", tid))
    upsert_match(
        conn, competition="ESP1", season="2026-27", stage="league",
        match_date="2026-08-19", matchday=1, home_team_id=local,
        away_team_id=visitante, home_goals=None, away_goals=None,
        status="scheduled", source="openfootball/espana",
    )

    con_id = copy.deepcopy(RESPUESTA_EN_JUEGO)
    con_id["events"][0]["id"] = "401882912"
    monkeypatch.setattr(espn, "descargar", lambda *a, **k: con_id)

    espn.update_schedule(conn, "2026-27")
    assert conn.execute("SELECT espn_event_id FROM matches").fetchone()[0] == "401882912"
