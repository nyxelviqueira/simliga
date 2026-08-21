"""Tests del parser de openfootball y del desempate de la tabla real."""
from __future__ import annotations

import pandas as pd
import pytest

from simliga.data import league_table
from simliga.ingest.openfootball import _resolver_equipo, parse

CALENDARIO = """= Spain Primera Division 2026/27

# Teams      20
# Matches    380


\u25aa Matchday 1
  Sun Aug 16 2026
    17:00  Club Atletico de Madrid v Malaga CF
           Real Racing Club de Santander v Villarreal CF

\u25aa Matchday 2
  Sun Aug 23
    17:00  Malaga CF               v Real Madrid CF
"""

JUGADO = """= Spain | Primera Division 2025/26

\u25aa Regular Season - 1
Fri Aug 15 2025
  19:00   Girona FC  1-3 (0-3)  Rayo Vallecano
                  (Joel Roca 57';
                   Jorge DE FRUTOS 18')
Sat Aug 16
  21:30   Deportivo Alaves  2-1 (1-0)  Levante UD
"""


def test_parsea_partidos_pendientes_con_jornada_y_fecha():
    df = parse(CALENDARIO, "2026-27")
    assert len(df) == 3
    assert list(df["matchday"]) == [1, 1, 2]
    assert df.iloc[0]["home"] == "Club Atletico de Madrid"
    assert df.iloc[0]["away"] == "Malaga CF"
    assert df["home_goals"].isna().all()


def test_arrastra_el_ano_cuando_la_fecha_no_lo_repite():
    """El fichero solo escribe el ano al cambiar; agosto pertenece al primero."""
    df = parse(CALENDARIO, "2026-27")
    assert df.iloc[0]["match_date"] == pd.Timestamp("2026-08-16")
    assert df.iloc[2]["match_date"] == pd.Timestamp("2026-08-23")


def test_deduce_el_ano_siguiente_para_los_meses_de_invierno():
    texto = "\u25aa Matchday 20\n  Sun Jan 10\n    17:00  Getafe CF v Sevilla FC\n"
    df = parse(texto, "2026-27")
    assert df.iloc[0]["match_date"] == pd.Timestamp("2027-01-10")


def test_parsea_marcadores_e_ignora_las_lineas_de_goleadores():
    df = parse(JUGADO, "2025-26")
    assert len(df) == 2, "las lineas de goleadores no deben contarse como partidos"
    assert df.iloc[0]["home"] == "Girona FC"
    assert (df.iloc[0]["home_goals"], df.iloc[0]["away_goals"]) == (1, 3)
    assert (df.iloc[1]["home_goals"], df.iloc[1]["away_goals"]) == (2, 1)


def test_acepta_las_dos_formas_de_cabecera_de_jornada():
    assert parse(CALENDARIO, "2026-27")["matchday"].tolist() == [1, 1, 2]
    assert parse(JUGADO, "2025-26")["matchday"].unique().tolist() == [1]


def test_openfootball_no_crea_ficha_nueva_por_tildes(tmp_path):
    from simliga.db import connect, get_or_create_team

    conn = connect(tmp_path / "test.sqlite")
    alaves = get_or_create_team(conn, "Deportivo Alaves")
    depor = get_or_create_team(conn, "Deportivo de La Coruna")
    atleti = get_or_create_team(conn, "Atletico de Madrid")
    malaga = get_or_create_team(conn, "Malaga CF")

    assert _resolver_equipo(conn, "Deportivo Alavés") == alaves
    assert _resolver_equipo(conn, "RC Deportivo La Coruña") == depor
    assert _resolver_equipo(conn, "Club Atlético de Madrid") == atleti
    assert _resolver_equipo(conn, "Málaga CF") == malaga
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 4


def test_football_data_no_usa_nombres_de_presentacion_como_identidad():
    from simliga.ingest.football_data_uk import canonical

    assert canonical("Alaves", "SP1") == "Deportivo Alaves"
    assert canonical("Dep. A Coruna", "SP1") == "Deportivo de La Coruna"
    assert canonical("La Coruna", "SP1") == "Deportivo de La Coruna"
    # Grafias que la fuente estreno a mitad de la primera jornada de 2026-27.
    assert canonical("Atl. Madrid", "SP1") == "Atletico de Madrid"
    assert canonical("Rayo Vallecano", "SP1") == "Rayo Vallecano"


def test_football_data_no_abre_ficha_nueva_por_un_cambio_de_grafia(monkeypatch, tmp_path):
    """El caso que rompio la publicacion: `Ath Madrid` paso a `Atl. Madrid`.

    Sin catalogar, la ficha nueva metia un equipo 21 en LaLiga y un partido 381,
    porque el mismo Atletico-Malaga entraba dos veces con dos identidades.
    """
    from simliga.db import connect, get_or_create_team
    from simliga.ingest import club_names
    from simliga.ingest.football_data_uk import _CatalogoPerezoso, _team_id

    monkeypatch.setattr(club_names, "cargar_alias", lambda paises=None, force=False: {
        "atletico de madrid": "Atlético Madrid",
        "atl madrid": "Atlético Madrid",
    })

    conn = connect(tmp_path / "test.sqlite")
    atleti = get_or_create_team(conn, "Atletico de Madrid")

    # Nombre sin catalogar y que tampoco casa por acentos: solo lo salva el catalogo.
    assert _team_id(conn, "Atl. Madrid", "SP1", _CatalogoPerezoso(conn)) == atleti
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 1


def test_football_data_si_abre_ficha_a_un_club_de_verdad_nuevo(monkeypatch, tmp_path):
    """En Segunda suben cada año clubes que nunca han aparecido: esos si son nuevos."""
    from simliga.db import connect, get_or_create_team
    from simliga.ingest import club_names
    from simliga.ingest.football_data_uk import _CatalogoPerezoso, _team_id

    monkeypatch.setattr(club_names, "cargar_alias", lambda paises=None, force=False: {})

    conn = connect(tmp_path / "test.sqlite")
    get_or_create_team(conn, "Atletico de Madrid")

    nuevo = _team_id(conn, "Club Recien Ascendido", "SP2", _CatalogoPerezoso(conn))
    assert conn.execute("SELECT name FROM teams WHERE team_id = ?",
                        (nuevo,)).fetchone()[0] == "Club Recien Ascendido"


# ------------------------------------------------------- desempate de la tabla real
def _partidos(filas):
    return pd.DataFrame([
        {"match_id": i, "match_date": pd.Timestamp("2025-09-01") + pd.Timedelta(days=i),
         "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag}
        for i, (h, a, hg, ag) in enumerate(filas)
    ])


def test_la_tabla_desempata_por_enfrentamiento_directo():
    """Caso real: Levante quedo por delante del Mallorca pese a peor diferencia."""
    partidos = _partidos([
        ("A", "B", 1, 1),      # h2h: empate...
        ("B", "A", 2, 0),      # ...y victoria de B -> B 4 puntos, A 1
        ("A", "C", 5, 0),      # A gana a C dos veces e infla su diferencia
        ("C", "A", 0, 3),
        ("B", "C", 1, 0),      # B reparte con C: gana una y pierde otra
        ("C", "B", 1, 0),
    ])
    # A y B acaban con 7 puntos; A con +6 de diferencia y B con +2.
    tabla = league_table(partidos).set_index("team")
    assert tabla.loc["A", "points"] == tabla.loc["B", "points"]
    assert tabla.loc["A", "gd"] > tabla.loc["B", "gd"]
    assert tabla.loc["B", "position"] < tabla.loc["A", "position"], (
        "el enfrentamiento directo debe pesar mas que la diferencia general"
    )


def test_sin_empate_a_puntos_manda_la_clasificacion_normal():
    partidos = _partidos([("A", "B", 3, 0), ("B", "A", 0, 1)])
    tabla = league_table(partidos).set_index("team")
    assert tabla.loc["A", "position"] == 1


def test_el_empate_a_tres_usa_la_mini_liga_entre_los_empatados():
    partidos = _partidos([
        ("A", "B", 1, 0), ("B", "C", 1, 0), ("C", "A", 1, 0),   # ciclo: 3 puntos cada uno
        ("A", "D", 0, 3), ("B", "D", 0, 3), ("C", "D", 0, 3),
        ("D", "A", 3, 0), ("D", "B", 3, 0), ("D", "C", 3, 0),
        ("B", "A", 0, 0), ("C", "B", 0, 0), ("A", "C", 0, 0),
    ])
    tabla = league_table(partidos)
    assert tabla.iloc[0]["team"] == "D"
    # Los tres empatados quedan agrupados en las posiciones 2, 3 y 4.
    assert set(tabla.iloc[1:]["team"]) == {"A", "B", "C"}
    assert tabla.iloc[1:]["points"].nunique() == 1
