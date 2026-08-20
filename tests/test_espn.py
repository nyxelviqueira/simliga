"""Tests del calendario con fechas y horarios reales (ESPN)."""
from __future__ import annotations

import pandas as pd
import pytest

from simliga.config import load_config
from simliga.ingest.espn import parse
from simliga.output import contract

RESPUESTA = {
    "events": [
        {
            "date": "2026-08-27T18:30Z",
            "competitions": [{
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "Celta Vigo"}},
                    {"homeAway": "away", "team": {"displayName": "Osasuna"}},
                ]
            }],
        },
        {
            "date": "2026-08-22T15:00Z",
            "competitions": [{
                "competitors": [
                    {"homeAway": "away", "team": {"displayName": "Sevilla"}},
                    {"homeAway": "home", "team": {"displayName": "Athletic Club"}},
                ]
            }],
        },
    ]
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


def test_un_evento_incompleto_se_descarta_sin_romper():
    incompleto = {"events": [{"date": "2026-09-01T18:00Z", "competitions": [{"competitors": []}]}]}
    assert parse(incompleto).empty


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
