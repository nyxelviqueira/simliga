"""Tests de la espera activa al resultado de un partido recien terminado.

El cron no puede afinar por debajo de los cinco minutos, y ademas GitHub se
salta ejecuciones programadas cuando va cargado. Lo que arregla eso es esperar
*dentro* de una ejecucion: sondear cada dos minutos y medio hasta que la fuente
publique el resultado.

Lo que se comprueba aqui es lo que hace util a esa espera y lo que la hace
segura: que vuelve en cuanto entra el dato (no despues), que no se queda
esperando a un partido que no acaba de terminar, y que siempre tiene tope.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

import simliga.cli as cli
import simliga.ingest.espn as espn
from simliga.db import connect

CADA = 150
LIMITE = 2100


class Reloj:
    """Reloj de mentira: `sleep` no duerme, solo adelanta la hora."""

    def __init__(self) -> None:
        self.ahora = 0.0
        self.dormido = 0.0

    def monotonic(self) -> float:
        return self.ahora

    def sleep(self, segundos: float) -> None:
        self.ahora += segundos
        self.dormido += segundos


@pytest.fixture
def reloj(monkeypatch):
    r = Reloj()
    monkeypatch.setattr(time, "monotonic", r.monotonic)
    monkeypatch.setattr(time, "sleep", r.sleep)
    return r


def _base(tmp_path, minutos_desde_el_comienzo: int):
    """Una base con un solo partido sin resultado, empezado hace N minutos."""
    conn = connect(tmp_path / "prueba.sqlite")
    conn.executemany("INSERT INTO teams (team_id, name) VALUES (?, ?)",
                     [(1, "Rayo Vallecano"), (2, "Deportivo Alaves")])
    comienzo = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
        minutes=minutos_desde_el_comienzo)
    conn.execute(
        """INSERT INTO matches
           (match_id, competition, season, stage, match_date, kickoff_utc,
            home_team_id, away_team_id, home_goals, away_goals, status)
           VALUES (10, 'ESP1', '2026-27', 'league', ?, ?, 1, 2, NULL, NULL, 'scheduled')""",
        (comienzo.strftime("%Y-%m-%d"), comienzo.strftime("%H:%M")),
    )
    conn.commit()
    return conn


def _args(**extra):
    campos = {"temporada": "2026-27", "cada": CADA, "limite": LIMITE,
              "frescura": cli.FRESCURA_MIN, "db": None, "config": None}
    campos.update(extra)
    return type("args", (), campos)()


@pytest.fixture
def espn_falso(monkeypatch):
    """Sustituye a ESPN. Guarda el resultado en el sondeo numero `entra_en`."""
    estado = {"llamadas": 0, "entra_en": None, "conn": None}

    def update_schedule(conn, season, *a, **k):
        estado["llamadas"] += 1
        if estado["entra_en"] is not None and estado["llamadas"] >= estado["entra_en"]:
            conn.execute("UPDATE matches SET home_goals = 1, away_goals = 1, "
                         "status = 'played' WHERE match_id = 10")
            conn.commit()
        return {}

    monkeypatch.setattr(espn, "update_schedule", update_schedule)
    return estado


def _correr(conn, monkeypatch, **extra):
    monkeypatch.setattr(cli, "_conn_and_names", lambda args: (conn, {}))
    return cli.cmd_esperar_resultados(_args(**extra))


def test_vuelve_en_el_acto_si_no_hay_nada_reciente(tmp_path, monkeypatch, reloj, espn_falso):
    """Sin partido recien terminado no se sondea ni una vez.

    Es lo que evita que la espera se coma cada ejecucion: en la inmensa mayoria
    de los ticks no ha acabado ningun partido.
    """
    conn = _base(tmp_path, minutos_desde_el_comienzo=30)   # todavia jugandose
    assert _correr(conn, monkeypatch) == 0
    assert espn_falso["llamadas"] == 0
    assert reloj.dormido == 0


def test_no_espera_a_un_partido_viejo(tmp_path, monkeypatch, reloj, espn_falso):
    """Un aplazado sigue contando como pendiente tres dias, pero no se le espera.

    La puerta lo da por pendiente para reintentar la publicacion; quedarse media
    hora parada por el en cada tick seria otra cosa muy distinta.
    """
    conn = _base(tmp_path, minutos_desde_el_comienzo=60 * 30)
    assert _correr(conn, monkeypatch) == 0
    assert espn_falso["llamadas"] == 0


def test_vuelve_en_cuanto_entra_el_resultado(tmp_path, monkeypatch, reloj, espn_falso):
    """Ni antes ni despues: en el tercer sondeo, tras dos esperas."""
    conn = _base(tmp_path, minutos_desde_el_comienzo=110)
    espn_falso["entra_en"] = 3

    assert _correr(conn, monkeypatch) == 0
    assert espn_falso["llamadas"] == 3
    assert reloj.dormido == 2 * CADA


def test_el_primer_sondeo_no_espera(tmp_path, monkeypatch, reloj, espn_falso):
    """Si la fuente ya tiene el dato, se publica sin dormir nada.

    Es el caso de la ejecucion que llega tarde, que con el cron de GitHub es de
    lo mas comun.
    """
    conn = _base(tmp_path, minutos_desde_el_comienzo=110)
    espn_falso["entra_en"] = 1

    assert _correr(conn, monkeypatch) == 0
    assert espn_falso["llamadas"] == 1
    assert reloj.dormido == 0


def test_la_espera_tiene_tope(tmp_path, monkeypatch, reloj, espn_falso):
    """Si el resultado no llega nunca, se vuelve igual para publicar lo que haya.

    Devolver un fallo cerraria la publicacion de ese ciclo, y el panel se
    quedaria viejo por un partido que quiza ni se jugo.
    """
    conn = _base(tmp_path, minutos_desde_el_comienzo=110)
    espn_falso["entra_en"] = None                  # nunca entra

    assert _correr(conn, monkeypatch) == 0
    assert reloj.dormido <= LIMITE
    assert reloj.dormido > LIMITE - CADA


def test_un_fallo_de_red_no_cancela_la_espera(tmp_path, monkeypatch, reloj):
    """ESPN se cae un momento y vuelve: el resultado tiene que salir igual."""
    estado = {"llamadas": 0}

    def update_schedule(conn, season, *a, **k):
        estado["llamadas"] += 1
        if estado["llamadas"] <= 2:
            raise RuntimeError("connection reset")
        conn.execute("UPDATE matches SET home_goals = 1, away_goals = 1, "
                     "status = 'played' WHERE match_id = 10")
        conn.commit()
        return {}

    monkeypatch.setattr(espn, "update_schedule", update_schedule)
    conn = _base(tmp_path, minutos_desde_el_comienzo=110)

    assert _correr(conn, monkeypatch) == 0
    assert estado["llamadas"] == 3
