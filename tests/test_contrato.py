"""Tests del JSON de salida.

Cada asercion se corresponde con una garantia escrita en `docs/data-contract.md`.
Si algo aqui falla, el documento ha dejado de ser cierto: hay que arreglar el
codigo o actualizar el contrato y subir `schema_version`.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from simliga.config import load_config
from simliga.model.dixon_coles import DixonColesFit
from simliga.config import DixonColesConfig
from simliga.output import contract
from simliga.sim.league import simulate_league

EQUIPOS = list(range(1, 21))
NOMBRES = {t: f"Equipo {t:02d}" for t in EQUIPOS}


@pytest.fixture(scope="module")
def documento():
    rng = np.random.default_rng(0)
    fuerza = np.linspace(0.4, -0.4, len(EQUIPOS))
    fit = DixonColesFit(
        team_ids=EQUIPOS, attack=fuerza.copy(), defence=fuerza * 0.7,
        mu=0.1, home_advantage=0.3, rho=0.02, kappa_attack=0.5, kappa_defence=0.35,
        n_matches=380, effective_n=250.0, log_likelihood=-100.0, converged=True,
        config=DixonColesConfig(max_goals=10),
    )
    pares = [(h, a) for h in EQUIPOS for a in EQUIPOS if h != a]
    pendientes = pd.DataFrame([
        {"match_id": i, "competition": "ESP1", "season": "2026-27", "stage": "league",
         "match_date": pd.Timestamp("2026-08-15") + pd.Timedelta(days=i // 10),
         "home_team_id": h, "away_team_id": a, "home_goals": None, "away_goals": None,
         "status": "scheduled", "home_team": NOMBRES[h], "away_team": NOMBRES[a]}
        for i, (h, a) in enumerate(pares)
    ])
    jugados = pendientes.iloc[:0]

    cfg = load_config()
    cfg.sim.n_sims = 2000
    res = simulate_league(fit, pendientes, played=jugados, teams=EQUIPOS, config=cfg.sim)
    elo = {t: 1500 + 300 * f for t, f in zip(EQUIPOS, fuerza)}

    bloque = contract.build_league_block(res, fit, NOMBRES, elo, jugados, pendientes, cfg)
    return contract.build_output(
        season="2026-27", league_block=bloque,
        fixtures=contract.build_fixtures_block(fit, pendientes, NOMBRES, limit=5),
        fit=fit, cfg=cfg, as_of=pd.Timestamp("2026-08-15"),
        data_sources=["test"],
    )


def test_tiene_las_claves_de_primer_nivel_documentadas(documento):
    esperadas = {"schema_version", "engine_version", "generated_at", "season", "as_of",
                 "simulation", "model", "competitions", "standings", "calendar",
                 "european_qualification",
                 "fixtures", "validation", "meta"}
    assert esperadas <= set(documento)
    assert documento["schema_version"] == "1.11.0"


def test_es_serializable_a_json(documento):
    texto = json.dumps(documento, ensure_ascii=False)
    assert json.loads(texto)["season"] == "2026-27"


def test_las_probabilidades_de_posicion_suman_uno_por_equipo(documento):
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        total = sum(equipo["projection"]["position_probabilities"])
        assert total == pytest.approx(1.0, abs=1e-4), equipo["name"]


def test_la_matriz_de_posiciones_es_doblemente_estocastica(documento):
    """Garantia del contrato: cada posicion la ocupa exactamente un equipo."""
    matriz = np.array([e["projection"]["position_probabilities"]
                       for e in documento["competitions"]["ESP1"]["teams"]])
    assert matriz.shape == (20, 20)
    assert np.allclose(matriz.sum(axis=0), 1.0, atol=1e-4)
    assert np.allclose(matriz.sum(axis=1), 1.0, atol=1e-4)


def test_la_probabilidad_de_titulo_coincide_con_la_de_ser_primero(documento):
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        assert equipo["outcomes"]["title"] == pytest.approx(
            equipo["projection"]["position_probabilities"][0], abs=1e-4)


def test_las_plazas_europeas_son_disjuntas_y_suman_la_clasificacion_total(documento):
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        o = equipo["outcomes"]
        assert o["ucl"] + o["uel"] + o["uecl"] == pytest.approx(
            o["european_qualification"], abs=1e-4), equipo["name"]


def test_todas_las_probabilidades_estan_entre_cero_y_uno(documento):
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        for clave, valor in equipo["outcomes"].items():
            assert 0.0 <= valor <= 1.0, (equipo["name"], clave)
        assert all(0.0 <= p <= 1.0 for p in equipo["projection"]["position_probabilities"])


def test_los_equipos_vienen_ordenados_por_posicion_esperada(documento):
    medias = [e["projection"]["position"]["mean"]
              for e in documento["competitions"]["ESP1"]["teams"]]
    assert medias == sorted(medias)


def test_las_reglas_de_clasificacion_cubren_rangos_validos(documento):
    reglas = documento["competitions"]["ESP1"]["qualification_rules"]
    n = documento["competitions"]["ESP1"]["n_teams"]
    for nombre, (desde, hasta) in reglas.items():
        assert 1 <= desde <= hasta <= n, nombre
    assert reglas["ucl"][1] + 1 == reglas["uel"][0]      # rangos contiguos
    assert reglas["uel"][1] + 1 == reglas["uecl"][0]
    # Espana tiene la 5a plaza de Champions por coeficiente UEFA (EPS): en
    # 2025-26 el quinto, el Betis, entro en Champions. Las plazas que cuentan
    # aqui son las que reparte LA LIGA; la Europa League del campeon de Copa va
    # aparte, porque puede tocarle a alguien de mitad de tabla.
    assert reglas["ucl"] == [1, 5], "cinco plazas Champions con el EPS"
    assert reglas["uel"] == [6, 6], "la otra UEL es la del campeon de Copa"
    assert reglas["uecl"] == [7, 7]
    assert "qualification_note" in documento["competitions"]["ESP1"]
    assert "coeficiente" in documento["competitions"]["ESP1"]["qualification_note"]


def test_los_percentiles_estan_ordenados(documento):
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        p = equipo["projection"]["points"]
        assert p["p05"] <= p["p25"] <= p["median"] <= p["p75"] <= p["p95"]
        assert p["p05"] <= p["mean"] <= p["p95"]


def test_el_recuento_de_partidos_cuadra(documento):
    liga = documento["competitions"]["ESP1"]
    assert liga["matches_played"] + liga["matches_remaining"] == 380
    assert documento["as_of"]["matches_played"] == liga["matches_played"]


def test_los_fixtures_traen_probabilidades_que_suman_uno(documento):
    assert len(documento["fixtures"]) == 5
    for partido in documento["fixtures"]:
        p = partido["probabilities"]
        assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-3)
        assert partido["expected_goals"]["home"] > 0
        assert set(partido) >= {"match_id", "competition", "date", "home_team",
                                "away_team", "probabilities", "expected_goals"}


def test_los_proximos_reales_no_descartan_escenarios():
    from simliga.pipeline import real_upcoming_matches, season_split

    class Ctx:
        as_of = pd.Timestamp("2026-08-20")
        scenario = {2: (1, 0)}
        matches = pd.DataFrame([
            {"match_id": 1, "competition": "ESP1", "season": "2026-27",
             "match_date": pd.Timestamp("2026-08-15"), "home_goals": 2,
             "away_goals": 0},
            {"match_id": 2, "competition": "ESP1", "season": "2026-27",
             "match_date": pd.Timestamp("2026-08-25"), "home_goals": None,
             "away_goals": None},
            {"match_id": 3, "competition": "ESP1", "season": "2026-27",
             "match_date": pd.Timestamp("2026-09-01"), "home_goals": None,
             "away_goals": None},
        ])

    _, pendientes_para_simular, _ = season_split(Ctx, "2026-27")
    proximos_reales = real_upcoming_matches(Ctx, "2026-27")

    assert set(pendientes_para_simular["match_id"]) == {3}
    assert set(proximos_reales["match_id"]) == {2, 3}


def test_los_equipos_traen_identidad_visual(documento):
    equipo = documento["competitions"]["ESP1"]["teams"][0]
    assert set(equipo) >= {"team_id", "name", "display_name", "logo"}
    assert equipo["display_name"] == equipo["name"]
    assert "display_name" in documento["fixtures"][0]["home_team"]
    assert "logo" in documento["fixtures"][0]["home_team"]


def test_la_identidad_visual_cura_nombres_y_escudos():
    from simliga.output.team_identity import display_name, logo_source_url, logo_url

    assert display_name("Deportivo de La Coruna") == "Deportivo de A Coruña"
    assert display_name("Deportivo Alaves") == "Deportivo Alavés"
    assert display_name("Atletico de Madrid") == "Atlético de Madrid"
    assert logo_url("Real Betis") == "assets/escudos/real-betis.png"
    assert logo_source_url("Real Betis") == "https://a.espncdn.com/i/teamlogos/soccer/500/244.png"
    assert logo_url("Equipo inventado") is None


def test_el_equipo_mas_fuerte_tiene_mas_opciones_de_titulo(documento):
    equipos = documento["competitions"]["ESP1"]["teams"]
    assert equipos[0]["outcomes"]["title"] > equipos[-1]["outcomes"]["title"]
    assert equipos[-1]["outcomes"]["relegation"] > equipos[0]["outcomes"]["relegation"]


def test_el_bloque_de_modelo_documenta_los_parametros_del_ajuste(documento):
    dc = documento["model"]["dixon_coles"]
    assert set(dc) >= {"half_life_days", "elo_prior_weight", "mu", "home_advantage",
                       "rho", "fitted_on_matches", "effective_sample_size"}
    assert documento["simulation"]["n_sims"] == 2000
    assert documento["simulation"]["modifiers_enabled"] == []


# ------------------------------------------------------- bloque de competicion UEFA
@pytest.fixture(scope="module")
def bloque_europeo():
    from simliga.sim.uefa import simulate_uefa

    n = 36
    fuerza = np.linspace(0.5, -0.5, n)
    fit = DixonColesFit(
        team_ids=list(range(n)), attack=fuerza.copy(), defence=fuerza * 0.7,
        mu=0.2, home_advantage=0.3, rho=0.03, kappa_attack=0.5, kappa_defence=0.35,
        n_matches=500, effective_n=300.0, log_likelihood=-1.0, converged=True,
        config=DixonColesConfig(max_goals=8),
    )
    filas, mid = [], 0
    for salto in range(1, 5):
        for i in range(n):
            j = (i + salto) % n
            local, visitante = (i, j) if (i + salto) % 2 == 0 else (j, i)
            filas.append({
                "match_id": mid, "competition": "UCL", "season": "2026-27",
                "stage": "league_phase",
                "match_date": pd.Timestamp("2026-09-15") + pd.Timedelta(days=mid // 18),
                "home_team_id": local, "away_team_id": visitante,
                "home_goals": None, "away_goals": None, "status": "scheduled",
                "home_team": str(local), "away_team": str(visitante),
            })
            mid += 1
    pendientes = pd.DataFrame(filas)

    cfg = load_config()
    cfg.sim.n_sims = 3000
    res = simulate_uefa(fit, pendientes, played=None, teams=list(range(n)), config=cfg.sim)
    return contract.build_european_block(
        res, {i: f"Club {i:02d}" for i in range(n)},
        {i: ("ESP" if i < 5 else "ENG") for i in range(n)},
        {i: 1500.0 + 10 * i for i in range(n)},
        pendientes.iloc[:0], pendientes, "UCL",
    )


def test_el_bloque_europeo_declara_su_tipo_y_sus_reglas(bloque_europeo):
    assert bloque_europeo["type"] == "uefa_league_phase"
    assert bloque_europeo["n_teams"] == 36
    reglas = bloque_europeo["qualification_rules"]
    assert reglas["direct_to_r16"] == [1, 8]
    assert reglas["playoff"] == [9, 24]
    assert reglas["eliminated"] == [25, 36]


def test_los_desenlaces_de_liguilla_suman_uno_por_equipo(bloque_europeo):
    for equipo in bloque_europeo["teams"]:
        lp = equipo["league_phase"]
        total = lp["p_direct_to_r16"] + lp["p_playoff"] + lp["p_eliminated"]
        assert total == pytest.approx(1.0, abs=1e-4), equipo["name"]


def test_las_probabilidades_de_ronda_son_monotonas(bloque_europeo):
    """Garantia del contrato: el embudo nunca puede ensancharse."""
    orden = ["round_of_16", "quarter_finals", "semi_finals", "final", "winner"]
    for equipo in bloque_europeo["teams"]:
        valores = [equipo["stage_probabilities"][k] for k in orden]
        assert valores == sorted(valores, reverse=True), equipo["name"]


def test_las_plazas_de_cada_ronda_cuadran_en_el_agregado(bloque_europeo):
    equipos = bloque_europeo["teams"]
    esperado = {"round_of_16": 16, "quarter_finals": 8, "semi_finals": 4,
                "final": 2, "winner": 1}
    for ronda, plazas in esperado.items():
        total = sum(e["stage_probabilities"][ronda] for e in equipos)
        assert total == pytest.approx(plazas, abs=0.02), ronda


def test_las_posiciones_de_liguilla_forman_una_matriz_valida(bloque_europeo):
    matriz = np.array([e["league_phase"]["position_probabilities"]
                       for e in bloque_europeo["teams"]])
    assert matriz.shape == (36, 36)
    assert np.allclose(matriz.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(matriz.sum(axis=0), 1.0, atol=1e-4)


def test_cada_equipo_europeo_declara_su_pais(bloque_europeo):
    paises = {e["country"] for e in bloque_europeo["teams"]}
    assert paises == {"ESP", "ENG"}
    assert sum(1 for e in bloque_europeo["teams"] if e["country"] == "ESP") == 5


# ------------------------------------------------------------------- panel HTML
def test_el_panel_incrusta_los_datos_y_no_deja_la_marca(documento):
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard(documento)
    assert "DATOS_JSON" not in html, "la marca de la plantilla debe quedar sustituida"
    assert '<script id="datos" type="application/json">' in html
    assert "2026-27" in html


def test_el_panel_escapa_las_etiquetas_de_cierre(documento):
    """Un `</script>` dentro del JSON cerraria la etiqueta y romperia la pagina."""
    from simliga.output.dashboard import render_dashboard

    envenenado = dict(documento)
    envenenado["meta"] = {"data_sources": ["</script><script>alert(1)</script>"]}
    html = render_dashboard(envenenado)
    bloque = html.split('<script id="datos" type="application/json">')[1].split("</script>")[0]
    assert "alert(1)" in bloque, "el contenido sigue estando, solo escapado"
    assert "</script>" not in bloque


def test_el_panel_se_escribe_a_disco(documento, tmp_path):
    from simliga.output.dashboard import write_dashboard

    destino = write_dashboard(documento, tmp_path / "sub" / "panel.html")
    assert destino.exists()
    assert (destino.parent / "assets" / "escudos" / "real-betis.png").exists()
    assert (destino.parent / "manifest.webmanifest").exists()
    assert (destino.parent / "service-worker.js").exists()
    assert (destino.parent / "icono.png").exists()
    texto = destino.read_text(encoding="utf-8")
    assert texto.lstrip().startswith("<!--") or texto.lstrip().startswith("<meta")
    assert len(texto) > 10_000


# --------------------------------------------------- fechas provisionales
def test_una_fecha_ya_pasada_de_un_partido_sin_jugar_es_provisional():
    """Si la fecha quedo atras y el partido sigue pendiente, no es la real."""
    pendientes = pd.DataFrame({
        "match_date": pd.to_datetime(["2026-08-16", "2026-09-20"]),
        "matchday": [1, 5],
    })
    marcas = contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20"))
    assert list(marcas) == [True, False]


def test_una_jornada_entera_en_la_misma_fecha_es_provisional():
    """El marcador de posicion de la fuente: los diez partidos el mismo domingo."""
    pendientes = pd.DataFrame({
        "match_date": pd.to_datetime(["2026-09-20"] * 10 + ["2026-09-26", "2026-09-27"]),
        "matchday": [5] * 10 + [6, 6],
    })
    marcas = contract.provisional_dates(pendientes, pd.Timestamp("2026-08-20"))
    assert marcas.iloc[:10].all()
    assert not marcas.iloc[10:].any(), "una jornada repartida en varios dias si es real"


def test_sin_partidos_pendientes_no_falla():
    vacio = pd.DataFrame({"match_date": pd.to_datetime([]), "matchday": []})
    assert contract.provisional_dates(vacio, pd.Timestamp("2026-08-20")).empty


def test_los_fixtures_declaran_jornada_y_si_la_fecha_es_firme(documento):
    for partido in documento["fixtures"]:
        assert "date_provisional" in partido
        assert isinstance(partido["date_provisional"], bool)
        assert "matchday" in partido


def test_el_panel_estatico_se_marca_como_tal(documento):
    """Un panel sin servidor detras no debe ofrecer botones que no funcionan."""
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard(documento)
    assert 'name="simliga-modo" content="estatico"' in html
    assert "MODO_PANEL" not in html


def test_el_panel_servido_se_marca_como_servidor(documento):
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard(documento, servido=True)
    assert 'name="simliga-modo" content="servidor"' in html


def test_el_modo_no_se_deduce_del_protocolo(documento):
    """Una copia estatica subida a un hosting tambien se sirve por http.

    Si el panel dedujese el modo del protocolo, ahi activaria los botones y
    fallarian contra una API inexistente con un error incomprensible.
    """
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard(documento)
    assert "location.protocol" not in html, (
        "el modo debe venir marcado por quien genera la pagina, no del protocolo")


def test_el_panel_estatico_explica_donde_se_editan_los_resultados(documento):
    """Sin esto, quien quiera probar un escenario no encuentra por donde."""
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard(documento)
    assert 'id="llamada-servidor"' in html
    assert "panel.bat" in html
    assert "simliga servidor" in html


def test_las_casillas_de_marcador_nunca_se_vuelven_invisibles():
    """Regresion: ocultarlas en solo lectura dejaba la fila con un guion suelto.

    Sin la casilla no hay ninguna pista de que ese partido lleva un marcador ni
    de que en el panel interactivo se puede escribir ahi.
    """
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert "border-color: transparent; background: transparent;" not in plantilla
    assert ".marcador input.vacia-bloqueada" in plantilla, (
        "las casillas bloqueadas deben distinguirse, pero seguir viendose")


def test_el_panel_dirige_las_hipotesis_a_la_clasificacion_simulada():
    """La pista debe llevar a la tabla que se actualiza al instante."""
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert "Clasificación simulada" in plantilla
    assert "Ya cuenta en «Clasificación simulada»" in plantilla
    assert "pestaña de clasificación simulada" in plantilla
    assert "para eso está la pestaña de proyección" not in plantilla
    assert "Pulsa «Simular con estos resultados» para que cuenten" not in plantilla


def test_los_puntos_proyectados_se_pintan_como_enteros():
    """La media puede ser decimal, pero en el panel los puntos son enteros."""
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert "const puntosProyectados = (x) => String(Math.round(Number(x)));" in plantilla
    assert "${puntosProyectados(p.mean)}" in plantilla
    assert "${puntosProyectados(p.p05)}–${puntosProyectados(p.p95)}" in plantilla
    assert '["Puntos al final", puntosProyectados(p.mean),' in plantilla
    assert "entre ${puntosProyectados(p.p05)} y ${puntosProyectados(p.p95)}" in plantilla


def test_los_proximos_partidos_y_la_ficha_no_ignoran_escenarios():
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    # Se pintan por tandas desde `doc.fixtures`, que es la lista que ya tiene
    # los escenarios aplicados. La tanda cambio; la fuente no puede cambiar.
    assert "doc.fixtures.slice(pintados, pintados + cuantos)" in plantilla
    assert "local.append(nodoEquipo(m.home_team));" in plantilla
    assert "local.append(nodoEquipo(m.home_team, { escudoDespues: true }));" in plantilla
    assert "partidosDelEquipo(equipo, 5)" in plantilla
    assert "m.status !== \"played\"" in plantilla
    assert "Siguientes 5 partidos" in plantilla


def test_el_panel_muestra_los_partidos_en_juego_sin_hacerlos_editables():
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert 'm.status === "live"' in plantilla
    assert "insignia-live" in plantilla
    assert "marcador-live" in plantilla
    assert "se muestran con marcador parcial" in plantilla
    assert 'if (m.status === "played" || m.status === "live") continue;' in plantilla


def test_la_ficha_compara_sin_y_con_escenario():
    """Con escenarios activos, la ficha debe separar la vista real de la simulada."""
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert "Sin escenario aplicado" in plantilla
    assert "Con escenario aplicado" in plantilla
    assert "copiaParaFicha(modo)" in plantilla
    assert "escenarioYaIntegradoEnLiga()" in plantilla
    assert ", ${hipoteticos} hipotético" not in plantilla


def test_la_ficha_usa_puesto_de_tabla_y_muestra_todas_las_plazas_europeas():
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert '["Puesto esperado", `${puestoEnTabla}º`' in plantilla
    assert "ligaFicha.teams.findIndex" in plantilla
    assert '["Europa League", pct(o.uel), ""]' in plantilla
    assert '["Conference", pct(o.uecl), ""]' in plantilla
    assert "liga.qualification_note" in plantilla
    assert 'id="ficha-supuesto"' in plantilla
    assert '"Reparto europeo: " + fraseSupuestoPlazas()' in plantilla


def test_los_checks_de_plazas_recalculan_la_clasificacion():
    plantilla = (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")
    assert 'id="plaza-extra-europa"' in plantilla
    assert 'id="plaza-extra-copa"' in plantilla
    assert "const reglasBasePlazas" in plantilla
    assert "function reglasClasificacion()" in plantilla
    assert "outcomesConPlazas" in plantilla
    assert "plazasExtra.europa" in plantilla
    assert "plazasExtra.copa" in plantilla
    assert "const base = doc.competitions.ESP1.qualification_rules;" not in plantilla
    assert "sin ninguna plaza extra" in plantilla
    assert "con plaza extra europea y de Copa" in plantilla


# ------------------------------------------------------- clasificacion real
EQUIPOS_REALES = ["Alfa", "Bravo", "Charlie", "Delta"]


def _jugados(filas):
    return pd.DataFrame([
        {"match_id": i, "match_date": pd.Timestamp("2026-08-15") + pd.Timedelta(days=i),
         "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag}
        for i, (h, a, hg, ag) in enumerate(filas)
    ])


def _clasificacion(filas, equipos=None):
    return contract.build_standings_block(
        _jugados(filas), equipos or EQUIPOS_REALES, load_config())


def test_la_clasificacion_incluye_a_los_equipos_que_aun_no_han_jugado():
    """Una tabla a la que le faltan filas no es una clasificacion."""
    tabla = _clasificacion([("Alfa", "Bravo", 2, 0)])
    assert len(tabla["rows"]) == 4
    assert tabla["n_teams"] == 4
    sin_jugar = [f for f in tabla["rows"] if f["played"] == 0]
    assert {f["team"] for f in sin_jugar} == {"Charlie", "Delta"}
    assert all(f["points"] == 0 for f in sin_jugar)


def test_las_posiciones_son_consecutivas_desde_uno():
    tabla = _clasificacion([("Alfa", "Bravo", 2, 0), ("Charlie", "Delta", 1, 1)])
    assert [f["position"] for f in tabla["rows"]] == [1, 2, 3, 4]


def test_cuenta_bien_ganados_empatados_y_perdidos():
    tabla = _clasificacion([
        ("Alfa", "Bravo", 2, 0), ("Alfa", "Charlie", 1, 1), ("Delta", "Alfa", 3, 0),
    ])
    alfa = next(f for f in tabla["rows"] if f["team"] == "Alfa")
    assert (alfa["played"], alfa["won"], alfa["drawn"], alfa["lost"]) == (3, 1, 1, 1)
    assert alfa["points"] == 4
    assert (alfa["goals_for"], alfa["goals_against"]) == (3, 4)
    assert alfa["goal_difference"] == -1


def test_la_forma_va_del_partido_mas_reciente_al_mas_antiguo():
    tabla = _clasificacion([
        ("Alfa", "Bravo", 2, 0),      # ganado, el mas antiguo
        ("Charlie", "Alfa", 1, 1),    # empatado
        ("Alfa", "Delta", 0, 3),      # perdido, el mas reciente
    ])
    alfa = next(f for f in tabla["rows"] if f["team"] == "Alfa")
    assert alfa["form"] == ["P", "E", "G"]


def test_la_forma_se_queda_en_los_cinco_ultimos():
    filas = [("Alfa", "Bravo", 1, 0)] * 8
    tabla = contract.build_standings_block(_jugados(filas), EQUIPOS_REALES, load_config())
    alfa = next(f for f in tabla["rows"] if f["team"] == "Alfa")
    assert len(alfa["form"]) == 5


def test_la_clasificacion_aplica_el_enfrentamiento_directo():
    """Mismo criterio que LaLiga: el head-to-head manda sobre la diferencia."""
    tabla = _clasificacion([
        ("Alfa", "Bravo", 1, 1),
        ("Bravo", "Alfa", 2, 0),      # Bravo gana el head-to-head
        ("Alfa", "Charlie", 5, 0),    # Alfa infla su diferencia general
        ("Charlie", "Alfa", 0, 3),
        ("Bravo", "Charlie", 1, 0),
        ("Charlie", "Bravo", 1, 0),
    ])
    posiciones = {f["team"]: f["position"] for f in tabla["rows"]}
    puntos = {f["team"]: f["points"] for f in tabla["rows"]}
    assert puntos["Alfa"] == puntos["Bravo"] == 7
    assert posiciones["Bravo"] < posiciones["Alfa"]


def test_sin_partidos_jugados_la_tabla_esta_a_cero_pero_completa():
    vacio = pd.DataFrame(columns=["match_id", "match_date", "home_team", "away_team",
                                  "home_goals", "away_goals"])
    tabla = contract.build_standings_block(vacio, EQUIPOS_REALES, load_config())
    assert len(tabla["rows"]) == 4
    assert tabla["matches_played"] == 0
    assert all(f["points"] == 0 and f["form"] == [] for f in tabla["rows"])


def test_marca_cuando_la_temporada_esta_completa():
    incompleta = _clasificacion([("Alfa", "Bravo", 1, 0)])
    assert incompleta["complete"] is False

    todos = [(h, a, 1, 0) for h in EQUIPOS_REALES for a in EQUIPOS_REALES if h != a]
    assert _clasificacion(todos)["complete"] is True


def test_la_clasificacion_real_ignora_los_resultados_hipoteticos(documento):
    """Se llama «real»: colar supuestos le quitaria el sentido al nombre."""
    liga = documento["competitions"]["ESP1"]
    assert documento["standings"] is None or (
        documento["standings"]["matches_played"] == liga["matches_played_real"])


# --------------------------------------------- clasificados a competicion europea
def test_las_plazas_europeas_salen_de_la_temporada_anterior():
    """Regresion: se calculaban con la clasificacion EN CURSO.

    A la jornada 1 eso daba una lista disparatada (el lider provisional
    figuraba como clasificado para la Champions) y encima etiquetada como
    "por su puesto del año pasado", que era afirmar algo falso.
    """
    filas = []
    equipos = [f"Equipo {i:02d}" for i in range(1, 11)]
    # Liga completa donde gana el 01, luego el 02, y asi sucesivamente.
    for i, local in enumerate(equipos):
        for j, visitante in enumerate(equipos):
            if i == j:
                continue
            filas.append((local, visitante, 3 if i < j else 0, 0 if i < j else 3))
    anterior = _jugados(filas)

    bloque = contract.build_european_qualification(anterior, "2025-26", load_config())
    assert bloque["source_season"] == "2025-26"
    orden = [t["team"] for t in bloque["teams"]]
    assert orden[:5] == equipos[:5], "las cinco primeras plazas son de Champions"
    assert [t["competition"] for t in bloque["teams"][:5]] == ["UCL"] * 5
    assert bloque["teams"][5]["competition"] == "UEL"
    assert bloque["teams"][6]["competition"] == "UECL"


def test_avisa_de_que_falta_el_campeon_de_copa():
    filas = [(f"E{i}", f"E{j}", 1, 0) for i in range(1, 11) for j in range(1, 11) if i != j]
    bloque = contract.build_european_qualification(_jugados(filas), "2025-26", load_config())
    assert "Copa del Rey" in bloque["caveat"]


def test_sin_temporada_anterior_no_se_inventa_nada():
    vacio = pd.DataFrame(columns=["match_id", "match_date", "home_team", "away_team",
                                  "home_goals", "away_goals"])
    assert contract.build_european_qualification(vacio, "2025-26", load_config()) is None


# ------------------------------------------------------------ ficha de equipo
def _plantilla() -> str:
    return (pathlib.Path(contract.__file__).parent / "dashboard.html").read_text(
        encoding="utf-8")


def test_la_ficha_de_equipo_existe_y_es_accesible():
    plantilla = _plantilla()
    assert '<dialog class="ficha" id="ficha"' in plantilla
    assert 'aria-labelledby="ficha-titulo"' in plantilla
    assert 'role", "button"' in plantilla, "las filas clicables deben anunciarse como boton"
    assert 'e.key === "Enter"' in plantilla, "debe abrirse tambien con teclado"


def test_el_css_de_la_distribucion_llego_a_la_plantilla():
    """Regresion: el bloque se inserto con un `replace` cuya ancla no existia.

    El resultado era que las barras no se dibujaban (la fila caia a `display:
    block`), pero la pagina no daba ningun error: solo se veian los numeros.
    """
    plantilla = _plantilla()
    assert ".dist-fila {" in plantilla
    assert "grid-template-columns: 34px 1fr 96px" in plantilla
    assert ".dist-barra i {" in plantilla


def test_la_distribucion_usa_una_sola_serie_de_color():
    """Una serie, un color: la banda va como franja lateral, no pintando la barra.

    Colorear la barra por banda ademas juntaba verde y ambar, que bajo
    protanopia quedan a una distancia de solo 7 (el minimo aceptable es 8).
    """
    plantilla = _plantilla()
    inicio = plantilla.index(".dist-barra i {")
    bloque = plantilla[inicio:inicio + 220]
    assert "var(--accent)" in bloque
    for banda in ("var(--ucl)", "var(--uel)", "var(--uecl)", "var(--drop)"):
        assert banda not in bloque, "la barra no debe llevar el color de la banda"


def test_las_veces_por_puesto_se_pueden_reconstruir_del_json(documento):
    """La ficha muestra recuentos: han de salir de las probabilidades y n_sims."""
    n = documento["simulation"]["n_sims"]
    for equipo in documento["competitions"]["ESP1"]["teams"]:
        veces = [round(p * n) for p in equipo["projection"]["position_probabilities"]]
        assert all(v >= 0 for v in veces)
        assert abs(sum(veces) - n) <= len(veces), equipo["name"]


# ------------------------------------------------- pestanas de las dos tablas
def test_las_dos_tablas_comparten_sitio_en_pestanas():
    """Apiladas se confunden: son parecidas y es facil mirar una creyendo otra."""
    plantilla = _plantilla()
    assert 'role="tablist"' in plantilla
    assert 'id="pestana-real"' in plantilla and 'id="pestana-proy"' in plantilla
    assert 'id="panel-real"' in plantilla and 'id="panel-proy"' in plantilla
    assert "Clasificación actual" in plantilla
    assert "Proyección final" in plantilla


def test_las_pestanas_son_accesibles():
    plantilla = _plantilla()
    for marca in ('role="tab"', 'role="tabpanel"', 'aria-selected', 'aria-controls',
                  'aria-labelledby'):
        assert marca in plantilla, marca
    assert 'e.key !== "ArrowRight"' in plantilla, "deben moverse con flechas"


def test_la_pestana_de_clasificacion_se_esconde_si_no_hay_partidos():
    """En pretemporada una tabla de ceros no dice nada: se abre en proyeccion."""
    plantilla = _plantilla()
    assert 'document.getElementById("pestana-real").hidden = !hayClasificacion' in plantilla
    assert 'activarPestana("proy")' in plantilla


def test_solo_hay_una_tabla_de_cada_clase():
    """Regresion: al mover las tablas es facil dejar la vieja duplicada."""
    plantilla = _plantilla()
    assert plantilla.count('id="tabla-real"') == 1
    assert plantilla.count('id="tabla"') == 1
    assert plantilla.count('id="seccion-real"') == 0, "la seccion suelta ya no existe"


# --------------------------------------------------- campeon de copa y plazas
def _liga_ordenada(n=10):
    """Liga donde gana el 01, luego el 02, y asi: posicion = numero de equipo."""
    equipos = [f"Equipo {i:02d}" for i in range(1, n + 1)]
    filas = [(local, visitante, 3 if i < j else 0, 0 if i < j else 3)
             for i, local in enumerate(equipos)
             for j, visitante in enumerate(equipos) if i != j]
    return equipos, _jugados(filas)


def test_el_campeon_de_copa_desplaza_el_reparto_de_la_liga():
    """Si el campeon de Copa no va por liga, ocupa una plaza de Europa League."""
    equipos, partidos = _liga_ordenada()
    bloque = contract.build_european_qualification(
        partidos, "2025-26", load_config(), cup_winner="Equipo 10")

    plazas = {t["team"]: t["competition"] for t in bloque["teams"]}
    assert [t for t, c in plazas.items() if c == "UCL"] == equipos[:5]
    assert set(t for t, c in plazas.items() if c == "UEL") == {"Equipo 06", "Equipo 10"}
    assert [t for t, c in plazas.items() if c == "UECL"] == ["Equipo 07"], (
        "si el campeon de Copa entra aparte, el septimo se queda en Conference")
    assert "Equipo 08" not in plazas


def test_el_campeon_entra_marcado_como_tal():
    _, partidos = _liga_ordenada()
    bloque = contract.build_european_qualification(
        partidos, "2025-26", load_config(), cup_winner="Equipo 10")
    campeon = next(t for t in bloque["teams"] if t["team"] == "Equipo 10")
    assert campeon["via"] == "copa"
    assert all(t["via"] == "liga" for t in bloque["teams"] if t["team"] != "Equipo 10")


def test_si_el_campeon_ya_estaba_clasificado_el_reparto_no_cambia():
    """Su plaza revierte a la liga: entonces el 7º si entra en Conference."""
    equipos, partidos = _liga_ordenada()
    bloque = contract.build_european_qualification(
        partidos, "2025-26", load_config(), cup_winner="Equipo 02")

    plazas = {t["team"]: t["competition"] for t in bloque["teams"]}
    assert set(t for t, c in plazas.items() if c == "UEL") == {"Equipo 06", "Equipo 07"}
    assert [t for t, c in plazas.items() if c == "UECL"] == ["Equipo 08"]
    assert "revierte" in bloque["caveat"]


def test_sin_campeon_registrado_se_avisa_de_que_falta():
    _, partidos = _liga_ordenada()
    bloque = contract.build_european_qualification(partidos, "2025-26", load_config())
    assert bloque["cup_winner"] is None
    assert "Copa del Rey" in bloque["caveat"]
    assert "simliga copa" in bloque["caveat"]


def test_las_casillas_de_marcador_centran_el_numero():
    """Regresion: las flechas del control numerico descentraban el marcador.

    `text-align: center` no basta en un `input type="number"`: los botones de
    subir y bajar ocupan sitio a la derecha y empujan el texto a la izquierda.
    """
    plantilla = _plantilla()
    inicio = plantilla.index(".marcador input {")
    bloque = plantilla[inicio:inicio + 700]
    assert "text-align: center" in bloque
    assert "appearance: textfield" in bloque
    assert "::-webkit-inner-spin-button" in plantilla
    assert "-webkit-appearance: none" in plantilla


def test_la_tabla_europea_explica_sobre_cuantos_equipos_van_los_porcentajes():
    """Ver 5 filas de 36 invita a pensar que deberian sumar 100 entre ellas."""
    plantilla = _plantilla()
    assert "participantes. Entre los ${comp.n_teams}" in plantilla
    assert "las de octavos suman 16 plazas" in plantilla
    assert 'const uno = filas.length === 1;' in plantilla, "singular y plural bien escritos"


def test_las_probabilidades_de_ronda_suman_las_plazas_de_cada_ronda(bloque_europeo):
    """La comprobacion de fondo: no suman 100% por equipo, suman plazas por ronda."""
    equipos = bloque_europeo["teams"]
    esperado = {"round_of_16": 16, "quarter_finals": 8, "semi_finals": 4,
                "final": 2, "winner": 1}
    for ronda, plazas in esperado.items():
        total = sum(e["stage_probabilities"][ronda] for e in equipos)
        assert total == pytest.approx(plazas, abs=0.02), ronda

    directo = sum(e["league_phase"]["p_direct_to_r16"] for e in equipos)
    playoff = sum(e["league_phase"]["p_playoff"] for e in equipos)
    fuera = sum(e["league_phase"]["p_eliminated"] for e in equipos)
    assert directo == pytest.approx(8, abs=0.02)
    assert playoff == pytest.approx(16, abs=0.02)
    assert fuera == pytest.approx(12, abs=0.02)


# ----------------------------------------------------------------- movil
def test_el_panel_se_puede_anadir_a_la_pantalla_de_inicio():
    plantilla = _plantilla()
    assert 'name="apple-mobile-web-app-capable"' in plantilla
    assert 'name="apple-mobile-web-app-title"' in plantilla
    assert 'name="theme-color"' in plantilla
    assert 'rel="manifest"' in plantilla
    assert "serviceWorker" in plantilla
    assert "prefers-color-scheme: dark" in plantilla.split("<style>")[0], (
        "el color de la barra debe seguir al tema del movil")


def test_hay_workflow_para_publicar_en_github_pages():
    workflow = pathlib.Path(".github/workflows/publish-panel.yml").read_text(
        encoding="utf-8")
    assert "deploy-pages" in workflow
    assert "workflow_dispatch" in workflow
    assert "schedule:" in workflow
    assert "python -m simliga actualizar" in workflow
    assert "cp out/panel.html out/index.html" in workflow


def test_en_modo_estatico_los_mandos_siguen_sirviendo():
    """Sin servidor los mandos se quedan, porque ahora funcionan.

    Antes se escondian: no habia nada detras que pudiera simular y unos botones
    muertos solo estorban en un movil. Con el motor incrustado en la pagina si
    lo hay, asi que esconderlos seria quitar lo unico que da acceso a cambiar el
    numero de simulaciones sin conexion.
    """
    plantilla = _plantilla()
    assert 'document.getElementById("mandos").hidden = true;' not in plantilla
    assert "motor.simular(doc, n, avance)" in plantilla


def test_solo_el_panel_servido_puede_descargar_datos():
    """«Actualizar datos» necesita Python y red: en una copia suelta se esconde.

    Dejarlo visible daria un boton que falla contra una API que no existe, que
    es peor que no tenerlo: parece que el panel esta roto en vez de que esa
    funcion pide el servidor.
    """
    plantilla = _plantilla()
    assert 'document.getElementById("actualizar").hidden = !servido;' in plantilla


# ---------------------------------------------------- clasificacion simulada
#
# La tercera pestaña: la tabla que sale de lo ya jugado mas los resultados
# hipoteticos. No es una proyeccion, es una suma de puntos.

def test_el_panel_tiene_la_pestana_de_clasificacion_simulada():
    plantilla = _plantilla()
    assert 'id="pestana-sim"' in plantilla
    assert 'id="panel-sim"' in plantilla
    assert "function pintarSimulada()" in plantilla


def test_la_pestana_simulada_nace_oculta():
    """Sin hipotesis seria una copia exacta de la clasificacion actual."""
    plantilla = _plantilla()
    marca = plantilla.split('id="pestana-sim"')[1].split(">")[0]
    assert "hidden" in marca, "la pestaña debe venir oculta del servidor"
    assert 'pestana.hidden = !filas;' in plantilla


def test_al_quedarse_sin_hipotesis_no_deja_la_vista_vacia():
    """Si borras la ultima hipotesis mirando esa pestaña, hay que sacarte.

    Una pestaña activa y oculta a la vez deja la seccion en blanco, y desde
    fuera parece que el panel se ha roto.
    """
    plantilla = _plantilla()
    assert 'if (pestana.getAttribute("aria-selected") === "true") activarPestana("proy");' \
        in plantilla


def test_las_flechas_se_saltan_las_pestanas_ocultas():
    """Navegar con el teclado no puede llevar a una pestaña invisible."""
    plantilla = _plantilla()
    assert "const visibles = ORDEN_PESTANAS.filter(" in plantilla
    assert 'c => !document.getElementById("pestana-" + c).hidden);' in plantilla


def test_la_tabla_simulada_usa_el_desempate_de_laliga():
    """Ordenar por diferencia general daria una tabla distinta de la real.

    Se reutiliza `motor.ordenar`, que es el mismo desempate que aplica la
    simulacion y que esta contrastado contra la version de Python.
    """
    plantilla = _plantilla()
    assert "motor.ordenar(new Int32Array(n), e.puntos, diferencia, e.favor," in plantilla


def test_la_tabla_simulada_se_rehace_al_editar_un_marcador():
    """Es lo unico del panel que responde sin volver a simular."""
    plantilla = _plantilla()
    guardar = plantilla.split("async function guardarEscenario")[1].split("\n  }")[0]
    assert 'seguro("clasificacion simulada", pintarSimulada);' in guardar


# ------------------------------------------------- la jornada en curso
def _jornada(numero, fechas, jugados=0):
    """Bloque de jornada minimo: solo lo que mira `current_matchday`."""
    return {
        "matchday": numero,
        "matches": [{"date": f, "status": "played" if i < jugados else "pending"}
                    for i, f in enumerate(sorted(fechas))],
    }


def _liga(jugadas_enteras=0):
    """Cinco jornadas de cuatro partidos, una por fin de semana desde el 15/08."""
    bloques = []
    for j in range(1, 6):
        sabado = pd.Timestamp("2026-08-15") + pd.Timedelta(days=7 * (j - 1))
        fechas = [sabado.strftime("%Y-%m-%d")] * 3 + [
            (sabado + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
        bloques.append(_jornada(j, fechas, 4 if j <= jugadas_enteras else 0))
    return bloques


def test_la_jornada_en_curso_es_la_de_esta_semana():
    # Domingo de la jornada 3, con la 1 y la 2 ya jugadas.
    assert contract.current_matchday(
        _liga(jugadas_enteras=2), pd.Timestamp("2026-08-30")) == 3


def test_al_completarse_la_jornada_pasa_a_la_siguiente():
    """Terminada la jornada 3, lo que interesa es la 4, sin esperar a su semana."""
    assert contract.current_matchday(
        _liga(jugadas_enteras=3), pd.Timestamp("2026-08-30")) == 4


def test_un_partido_aplazado_no_ancla_la_vista_en_su_jornada():
    """El caso real del 22 de agosto de 2026, que es de donde sale todo esto.

    La jornada 1 se jugo del 15 al 19 salvo cuatro partidos, movidos al fin de
    semana siguiente. Mirar "la primera sin acabar" abria por la 1 mientras se
    estaba jugando la 2.
    """
    jornadas = [
        _jornada(1, ["2026-08-15", "2026-08-15", "2026-08-16", "2026-08-16",
                     "2026-08-17", "2026-08-19",           # jugados
                     "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-27"],
                  jugados=6),
        _jornada(2, ["2026-08-20", "2026-08-21",           # jugados
                     "2026-08-22", "2026-08-22", "2026-08-22", "2026-08-23",
                     "2026-08-23", "2026-08-23", "2026-08-24", "2026-08-24"],
                  jugados=2),
        _jornada(3, ["2026-08-28"] + ["2026-08-29"] * 3 + ["2026-08-30"] * 4
                 + ["2026-08-31"] * 2),
    ]
    assert contract.current_matchday(jornadas, pd.Timestamp("2026-08-22")) == 2


def test_un_partido_adelantado_no_salta_por_encima_de_las_jornadas_de_en_medio():
    """Al reves que el aplazado: un partido que se juega antes de tiempo.

    La jornada 5 tiene un partido adelantado a septiembre. Sin parar la
    busqueda en la primera jornada sin empezar, esa fecha suelta la haria
    parecer en curso y se saltaria la 3 y la 4.
    """
    jornadas = _liga(jugadas_enteras=2)
    jornadas[4]["matches"][0]["date"] = "2026-08-26"
    jornadas[4]["matches"].sort(key=lambda p: p["date"])
    assert contract.current_matchday(jornadas, pd.Timestamp("2026-08-30")) == 3


def test_antes_de_empezar_la_temporada_la_jornada_en_curso_es_la_primera():
    assert contract.current_matchday(_liga(), pd.Timestamp("2026-07-01")) == 1


def test_acabada_la_temporada_se_queda_en_la_ultima_jornada():
    assert contract.current_matchday(
        _liga(jugadas_enteras=5), pd.Timestamp("2027-06-01")) == 5


def test_sin_calendario_no_hay_jornada_en_curso():
    assert contract.current_matchday([], pd.Timestamp("2026-08-30")) is None


def test_el_calendario_publica_la_jornada_en_curso():
    """Sobre un documento completo de verdad, con calendario dentro."""
    from tests.utiles import documento_de_prueba

    cal = documento_de_prueba(n_sims=200, jugadas=2)["calendar"]
    assert cal["current_matchday"] in [b["matchday"] for b in cal["matchdays"]]
    assert cal["current_matchday"] == 3, "jugadas la 1 y la 2, toca abrir por la 3"


def test_el_panel_abre_el_calendario_por_la_jornada_en_curso():
    """Quien decide es el contrato; el panel no vuelve a inventar el criterio."""
    plantilla = _plantilla()
    assert "jornadaVista = existe ? cal.current_matchday : cal.matchdays[0].matchday;" in plantilla


def test_una_jornada_entera_se_marca_en_verde():
    """Verde es el color de lo ya conseguido en el resto del panel (`--ucl`)."""
    plantilla = _plantilla()
    assert '.jornada-btn[data-estado="played"] { box-shadow: inset 0 -3px 0 var(--ucl); }' in plantilla


# ------------------------------------------------- seguimiento en directo
def test_el_calendario_publica_el_id_del_evento_en_espn():
    """Es la identidad con la que el panel sigue un partido sin casar nombres."""
    from tests.utiles import documento_de_prueba

    cal = documento_de_prueba(n_sims=200, jugadas=2)["calendar"]
    for bloque in cal["matchdays"]:
        for partido in bloque["matches"]:
            assert "espn_event_id" in partido, "la clave no puede faltar segun el caso"


def test_el_panel_sigue_el_directo_contra_espn_y_no_contra_el_workflow():
    """Refrescar la pagina publicada costaria un despliegue entero de Pages.

    ESPN permite CORS y sirve con `Cache-Control: max-age=10`: preguntarle
    desde el navegador sale gratis y llega antes.
    """
    plantilla = _plantilla()
    assert "site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard" in plantilla
    assert "function seguirEnDirecto()" in plantilla
    assert "seguirEnDirecto();" in plantilla


def test_el_directo_casa_los_partidos_por_id_y_no_por_nombre():
    plantilla = _plantilla()
    assert "porId.get(String(m.espn_event_id))" in plantilla


def test_el_directo_no_toca_la_proyeccion():
    """Un resultado en vivo no cuenta hasta el pitido final, como en el motor.

    El bucle solo escribe en los campos `live_*` y en `status`; si algun dia
    tocara `home_goals`, la tabla ensenaria puntos que no estan ganados.
    """
    plantilla = _plantilla()
    cuerpo = plantilla.split("async function refrescarDirecto()")[1].split("\n  }")[0]
    for prohibido in ("m.home_goals =", "m.away_goals =", "doc.competitions"):
        assert prohibido not in cuerpo, f"el directo no debe escribir en {prohibido}"


def test_el_service_worker_no_cachea_lo_que_no_es_suyo():
    """El marcador en directo caduca en segundos: guardarlo llena la cache."""
    import inspect

    from simliga.output import dashboard

    fuente = inspect.getsource(dashboard.write_mobile_assets)
    assert "!== self.location.origin) return;" in fuente


def test_un_partido_acabado_que_el_modelo_aun_no_tiene_no_dice_en_juego():
    """ESPN da «FT» antes de que la publicacion recoja el resultado.

    Ensenarlo como «en juego · FT» es contradictorio, y ensenarlo como jugado
    seria mentira: todavia no cuenta para la proyeccion.
    """
    plantilla = _plantilla()
    assert "m.live_final = marcador.completado;" in plantilla
    assert 'else if (enJuego && m.live_final) {' in plantilla
    assert 'el("span", "insignia insignia-live", "final")' in plantilla


# ------------------------------------------------- compactado del panel
def test_cada_pestana_apunta_a_secciones_que_existen():
    """Un id mal escrito no falla: la pestana sale y no ensena nada."""
    import re

    plantilla = _plantilla()
    bloque = plantilla.split("const SECCIONES = [")[1].split("];")[0]
    ids = re.findall(r'"(resumen|seccion-[a-z]+)"', bloque)
    assert len(ids) >= 5, "las cinco secciones tienen que estar en la barra"
    for destino in ids:
        assert f'id="{destino}"' in plantilla, f"la pestana apunta a #{destino}, que no existe"


def test_la_barra_de_secciones_se_queda_arriba_al_bajar():
    plantilla = _plantilla()
    barra = plantilla.split(".navegacion {")[1].split("}")[0]
    assert "position: sticky" in barra and "top: 0" in barra


def test_solo_se_ve_una_seccion_a_la_vez():
    """Es lo que convierte doce pantallas de scroll en una vista por pestana."""
    plantilla = _plantilla()
    assert 'seccion.classList.toggle("fuera-de-vista", !dentro);' in plantilla
    assert ".fuera-de-vista { display: none !important; }" in plantilla


def test_una_seccion_sin_datos_no_saca_pestana():
    """Sin sorteo europeo o sin calendario, la pestana sobraria."""
    plantilla = _plantilla()
    assert "const pestanaTieneAlgo = (p) => seccionesDe(p).some(s => !s.hidden);" in plantilla
    assert "SECCIONES.filter(pestanaTieneAlgo)" in plantilla


def test_ocultar_por_pestana_no_pisa_el_ocultar_por_falta_de_datos():
    """Son dos cosas distintas: una la decide el usuario y la otra los datos.

    Con `hidden` para las dos, volver de una pestana resucitaba una seccion que
    no tenia nada que ensenar.
    """
    plantilla = _plantilla()
    cuerpo = plantilla.split("function activarSeccion(")[1].split("function montarPestanas")[0]
    assert 'classList.toggle("fuera-de-vista"' in cuerpo
    assert "seccion.hidden =" not in cuerpo


def test_se_recuerda_la_seccion_abierta():
    plantilla = _plantilla()
    assert 'clave: "simliga:seccion"' in plantilla
    assert "activarSeccion(memoriaPestana.leer() || SECCIONES[0].clave);" in plantilla


def test_la_tabla_de_proyeccion_no_se_arrastra_en_movil():
    """980 px de tabla en 375 de pantalla: habia que recomponerla, no encogerla."""
    plantilla = _plantilla()
    movil = plantilla.split("@media (max-width: 640px) {")[1]
    assert ".tabla-liga { display: block; min-width: 0; }" in movil
    assert ".tabla-liga colgroup, .tabla-liga thead { display: none; }" in movil
    assert "content: attr(data-etiqueta);" in movil


def test_al_recomponer_la_tabla_no_se_pierde_ninguna_cifra():
    """Apilar si; ocultar columnas no. Cada celda lleva el nombre de la suya."""
    plantilla = _plantilla()
    for etiqueta in ("J", "Pts", "Puntos al final", "Título", "Champions",
                     "Europa L.", "Conference", "Descenso"):
        assert f'"{etiqueta}"' in plantilla, f"falta la etiqueta {etiqueta}"
    assert 'celdaProb(t.outcomes.title, "Título")' in plantilla
    assert 'celdaProyeccion.dataset.etiqueta = "Puntos al final";' in plantilla


def test_el_calendario_no_se_arrastra_en_movil():
    """Pedia 760 px por fila; en movil se parte en dos alturas."""
    plantilla = _plantilla()
    movil = plantilla.split("@media (max-width: 640px) {")[1]
    assert ".encuentros { min-width: 0; }" in movil
    assert '"cuando cuando nota"' in movil
    assert '"local marcador visita"' in movil


def test_los_proximos_partidos_empiezan_en_seis_y_el_resto_van_a_peticion():
    """Veinte tarjetas eran casi cuatro pantallas de movil, y repiten calendario.

    Los veinte siguen en el JSON: la copia sin conexion no pierde ninguno.
    """
    plantilla = _plantilla()
    assert "const DE_ENTRADA = 6;" in plantilla
    assert "pintarTanda(DE_ENTRADA);" in plantilla
    assert "`Ver los ${quedan} siguientes`" in plantilla


def test_las_competiciones_europeas_van_en_pestanas():
    """Tres tablas apiladas eran tres pantallas; en pestanas, una."""
    plantilla = _plantilla()
    cuerpo = plantilla.split("function pintarEuropa(")[1].split("\n  function ")[0]
    assert 'const barra = el("div", "pestanas");' in cuerpo
    assert "otro.bloque.hidden = j !== i;" in cuerpo
