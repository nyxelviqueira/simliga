"""Tests de la puerta que decide si hay que republicar el panel.

Existe para que la publicacion automatica no vaya a ciegas. Sin ella solo caben
dos malas opciones: republicar cada pocos minutos por si acaso, o hacerlo a
horas fijas y llegar tarde.

Lo que se comprueba aqui es que abre cuando debe y, sobre todo, que **se cierra**
cuando no hay nada: una puerta que se queda abierta convierte el ahorro en la
misma tormenta de ejecuciones que habia antes.
"""
from __future__ import annotations

import pandas as pd
import pytest

from simliga.data import partidos_por_actualizar
from simliga.db import connect

INICIO = "2026-08-22 19:00"


@pytest.fixture
def base(tmp_path):
    conn = connect(tmp_path / "prueba.sqlite")
    conn.executemany("INSERT INTO teams (team_id, name) VALUES (?, ?)",
                     [(1, "Celta de Vigo"), (2, "CA Osasuna"), (3, "Getafe CF")])
    conn.executemany(
        """INSERT INTO matches
           (match_id, competition, season, stage, match_date, kickoff_utc,
            home_team_id, away_team_id, home_goals, away_goals, status)
           VALUES (?, 'ESP1', '2026-27', 'league', ?, ?, ?, ?, ?, ?, ?)""",
        [
            # Sin resultado, empieza a las 19:00.
            (10, "2026-08-22", "19:00", 1, 2, None, None, "scheduled"),
            # Ya jugado: nunca debe aparecer.
            (11, "2026-08-22", "19:00", 2, 3, 1, 0, "played"),
            # Sin hora conocida: no se puede saber si termino.
            (12, "2026-08-22", None, 3, 1, None, None, "scheduled"),
        ],
    )
    conn.commit()
    return conn


def _ids(conn, ahora):
    return set(partidos_por_actualizar(conn, "2026-27", ahora=ahora)["match_id"])


def test_durante_el_partido_no_abre(base):
    """A los 45 minutos no ha terminado: publicar ahora seria publicar a medias."""
    assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(minutes=45)) == set()


def test_antes_de_empezar_no_abre(base):
    assert _ids(base, pd.Timestamp(INICIO) - pd.Timedelta(minutes=1)) == set()


def test_al_terminar_abre(base):
    """105 minutos: 90 mas descuento y descanso."""
    assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(minutes=110)) == {10}


def test_un_partido_ya_guardado_no_abre(base):
    """La clave de todo: se pregunta por lo que falta, no por lo que se jugo.

    Preguntando por "que se ha jugado hace poco" se republicaria una y otra vez
    aunque el resultado llevase horas guardado.
    """
    base.execute("UPDATE matches SET home_goals = 2, away_goals = 1, status = 'played' "
                 "WHERE match_id = 10")
    base.commit()
    assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(minutes=110)) == set()


def test_sin_hora_de_comienzo_se_deja_fuera(base):
    """No se le supone una hora a un partido que no la tiene."""
    tarde = pd.Timestamp(INICIO) + pd.Timedelta(hours=6)
    assert 12 not in _ids(base, tarde)


def test_un_partido_aplazado_deja_de_contar(base):
    """Si no, la puerta se queda abierta para siempre.

    Un partido aplazado conserva la fecha vieja y nunca recibe resultado. Sin
    tope de antiguedad se publicaria en cada ciclo, indefinidamente, que es
    justo lo que se queria evitar.
    """
    assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(days=2)) == {10}
    assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(days=4)) == set()


def test_reintenta_mientras_la_fuente_tarde(base):
    """Entre el final y el tope, sigue abierta: las fuentes no son inmediatas."""
    for horas in (2, 6, 24, 48):
        assert _ids(base, pd.Timestamp(INICIO) + pd.Timedelta(hours=horas)) == {10}, horas


def test_otra_temporada_no_cuenta(base):
    assert partidos_por_actualizar(
        base, "2025-26", ahora=pd.Timestamp(INICIO) + pd.Timedelta(hours=3)).empty


# ---------------------------------------------------------------- el comando

def test_el_comando_imprime_lo_que_espera_actions(base, monkeypatch, capsys):
    """`--para-actions` tiene que emitir exactamente `publicar=true|false`."""
    import simliga.cli as cli

    monkeypatch.setattr(cli, "_conn_and_names", lambda args: (base, {}))
    args = type("args", (), {"temporada": "2026-27", "para_actions": True})()

    monkeypatch.setattr(cli, "partidos_por_actualizar",
                        lambda *a, **k: pd.DataFrame({"match_id": [10]}))
    assert cli.cmd_novedades(args) == 0
    assert capsys.readouterr().out.strip() == "publicar=true"

    monkeypatch.setattr(cli, "partidos_por_actualizar",
                        lambda *a, **k: pd.DataFrame({"match_id": []}))
    assert cli.cmd_novedades(args) == 0
    assert capsys.readouterr().out.strip() == "publicar=false"


def test_si_la_base_falla_se_publica_igual(base, monkeypatch, capsys):
    """Ante la duda, publicar. Quedarse quieto dejaria el panel viejo sin avisar."""
    import simliga.cli as cli

    def revienta(*a, **k):
        raise RuntimeError("base corrupta")

    monkeypatch.setattr(cli, "_conn_and_names", lambda args: (base, {}))
    monkeypatch.setattr(cli, "partidos_por_actualizar", revienta)
    args = type("args", (), {"temporada": "2026-27", "para_actions": True})()

    assert cli.cmd_novedades(args) == 0
    assert capsys.readouterr().out.strip() == "publicar=true"
