"""El motor JavaScript del panel debe dar lo mismo que el de Python.

Hay dos implementaciones del mismo modelo (`simliga/sim/league.py` y
`simliga/output/motor.js`) porque el navegador no puede ejecutar la de Python,
y dos implementaciones separadas se van despegando en cuanto alguien toca una
sin acordarse de la otra. Estas pruebas son lo que lo impide.

Necesitan Node instalado. Donde no lo haya se saltan en vez de fallar: el
proyecto sigue siendo utilizable sin el, solo que sin esta comprobacion.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

MOTOR = Path(__file__).resolve().parents[1] / "simliga" / "output" / "motor.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="Node no esta instalado")


def ejecutar(script: str, doc: dict | None = None) -> dict:
    """Corre un fragmento de JavaScript con el motor cargado y devuelve su JSON.

    El guion y el documento van por fichero y no por `node -e`: un documento de
    salida completo pasa del medio mega y Windows corta la linea de ordenes en
    32 KB, asi que por argumento fallaria con un error que no dice nada del
    problema real.

    Dentro del guion hay tres nombres disponibles: `motor`, `doc` y `copia()`,
    que devuelve un duplicado del documento (el motor lo modifica al simular).
    """
    with tempfile.TemporaryDirectory() as carpeta:
        base = Path(carpeta)
        (base / "doc.json").write_text(json.dumps(doc or {}), encoding="utf-8")
        cabecera = textwrap.dedent("""
            const fs = require("fs");
            const motor = require(RUTA_MOTOR);
            const doc = JSON.parse(fs.readFileSync(RUTA_DOC, "utf8"));
            const copia = () => JSON.parse(JSON.stringify(doc));
            (async () => {
                const salida = await (async () => { CUERPO })();
                fs.writeFileSync(RUTA_SALIDA, JSON.stringify(salida));
            })().catch(e => { console.error(e); process.exit(1); });
        """)
        guion = (cabecera
                 .replace("RUTA_MOTOR", json.dumps(str(MOTOR)))
                 .replace("RUTA_DOC", json.dumps(str(base / "doc.json")))
                 .replace("RUTA_SALIDA", json.dumps(str(base / "salida.json")))
                 .replace("CUERPO", script))
        (base / "guion.js").write_text(guion, encoding="utf-8")

        proceso = subprocess.run(["node", str(base / "guion.js")],
                                 capture_output=True, text=True, timeout=300)
        if proceso.returncode != 0:
            raise AssertionError(f"Node fallo:\n{proceso.stderr}")
        return json.loads((base / "salida.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# La distribucion de marcadores

def test_cdf_de_marcador_coincide_con_la_de_python():
    """La rejilla de marcadores es la misma en los dos motores.

    Es la pieza donde un error pasa mas desapercibido: un signo cambiado en la
    correccion de Dixon-Coles altera los empates a cero un 3% y no rompe nada,
    solo hace que el panel mienta un poco.
    """
    from simliga.model.dixon_coles import score_matrix_from_rates

    lam_local, lam_visita, rho = 1.62, 1.09, 0.0242
    esperada = score_matrix_from_rates(lam_local, lam_visita, rho, max_goals=10)

    acumulada = np.array(ejecutar(
        f"return Array.from(motor.cdfMarcador({lam_local}, {lam_visita}, {rho}));"))
    # El motor JS lleva la acumulada; se deshace para comparar celda a celda.
    obtenida = np.diff(np.concatenate([[0.0], acumulada])).reshape(11, 11)

    assert obtenida.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(obtenida - esperada / esperada.sum()).max() < 1e-12


@pytest.mark.parametrize("rho", [0.0, 0.05, -0.05])
def test_la_cdf_siempre_es_una_distribucion(rho):
    """Con cualquier rho la rejilla sigue sumando uno y no decrece."""
    acumulada = np.array(ejecutar(
        f"return Array.from(motor.cdfMarcador(2.1, 0.8, {rho}));"))
    assert acumulada[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(acumulada) >= 0)


# --------------------------------------------------------------------------
# La simulacion completa

@pytest.fixture(scope="module")
def documento(tmp_path_factory) -> dict:
    """Un documento pequeño pero real, generado por el motor de Python."""
    from tests.utiles import documento_de_prueba          # type: ignore

    return documento_de_prueba()


def test_las_dos_simulaciones_dan_lo_mismo(documento):
    """Con 40.000 tiradas los dos motores tienen que coincidir dentro del ruido.

    No se comparan numero a numero porque cada uno usa su propio generador de
    aleatorios; lo que se comprueba es que la diferencia no supera lo que
    explica el muestreo. Un error de verdad en el modelo (una ventaja de campo
    mal aplicada, un desempate distinto) desplaza las cifras muy por encima de
    esa banda.
    """
    n_sims = 40_000
    salida = ejecutar(f"return await motor.simular(doc, {n_sims});", documento)

    por_python = {e["name"]: e for e in documento["competitions"]["ESP1"]["teams"]}
    # Tres errores tipicos de una proporcion a 40.000 tiradas, mas el error de
    # la propia corrida de Python que sirve de referencia.
    banda = 4 * (0.5 / n_sims ** 0.5)

    for equipo in salida["teams"]:
        ref = por_python[equipo["name"]]
        for clave in ("title", "ucl", "relegation", "european_qualification"):
            assert abs(equipo["outcomes"][clave] - ref["outcomes"][clave]) < banda, (
                f"{equipo['name']} / {clave}: JS {equipo['outcomes'][clave]:.4f} "
                f"vs Python {ref['outcomes'][clave]:.4f}")
        assert abs(equipo["projection"]["position"]["mean"]
                   - ref["projection"]["position"]["mean"]) < 0.25
        assert abs(equipo["projection"]["points"]["mean"]
                   - ref["projection"]["points"]["mean"]) < 1.5


def test_cada_equipo_ocupa_un_puesto_en_cada_simulacion(documento):
    """Las probabilidades de puesto de cada equipo suman uno."""
    salida = ejecutar("return await motor.simular(doc, 2000);", documento)
    for equipo in salida["teams"]:
        total = sum(equipo["projection"]["position_probabilities"])
        assert total == pytest.approx(1.0, abs=2e-3), equipo["name"]


def test_cada_puesto_lo_ocupa_un_equipo_en_cada_simulacion(documento):
    """Y por columnas tambien: en cada puesto acaba exactamente un equipo.

    Esta es la que caza un error de ordenacion. Sumar por filas da uno aunque
    el desempate reparta mal los puestos; sumar por columnas no.
    """
    salida = ejecutar("return await motor.simular(doc, 2000);", documento)
    columnas = np.array([e["projection"]["position_probabilities"]
                         for e in salida["teams"]]).sum(axis=0)
    assert np.allclose(columnas, 1.0, atol=2e-3)


def test_un_resultado_hipotetico_suma_sus_puntos(documento):
    """Poner un marcador a mano tiene que verse en la tabla de partida.

    Es lo que hace el calendario editable, y el sitio donde se rompe sin avisar:
    si el motor ignorase los partidos marcados como `scenario`, el panel
    ensenaria los mismos numeros de siempre y pareceria que no pasa nada.
    """
    doc = json.loads(json.dumps(documento))
    jornada = next(b for b in doc["calendar"]["matchdays"]
                   if any(m["status"] == "pending" for m in b["matches"]))
    partido = next(m for m in jornada["matches"] if m["status"] == "pending")
    partido["status"] = "scenario"
    partido["home_goals"], partido["away_goals"] = 5, 0

    local = partido["home_team"]["name"]
    antes = next(e for e in doc["competitions"]["ESP1"]["teams"] if e["name"] == local)
    salida = ejecutar("return await motor.simular(doc, 2000);", doc)
    despues = next(e for e in salida["teams"] if e["name"] == local)

    assert despues["current"]["points"] == antes["current"]["points"] + 3
    assert despues["current"]["goals_for"] == antes["current"]["goals_for"] + 5
    assert despues["current"]["scenario_matches"] == 1
    assert despues["current"]["played_real"] == antes["current"]["played"]


def test_el_motor_conserva_la_identidad_visual_del_equipo(documento):
    """Las vistas recalculadas en navegador deben seguir mostrando escudo y nombre."""
    salida = ejecutar("return (await motor.simular(doc, 200)).teams[0];", documento)
    ref = documento["competitions"]["ESP1"]["teams"][0]

    assert salida["display_name"] == ref["display_name"]
    assert "logo" in salida


def test_estado_inicial_cuenta_resultados_y_racha_para_la_tabla_simulada(documento):
    """La clasificación simulada usa el mismo contador que la simulación.

    Si G/E/P o la racha se calculasen en otro sitio, el panel podria enseñar
    una tabla coherente en puntos pero incoherente en detalle.
    """
    doc = json.loads(json.dumps(documento))
    equipos = doc["competitions"]["ESP1"]["teams"]
    indice = {t["team_id"]: i for i, t in enumerate(equipos)}

    jornada = next(b for b in doc["calendar"]["matchdays"]
                   if any(m["status"] == "pending" for m in b["matches"]))
    partido = next(m for m in jornada["matches"] if m["status"] == "pending")
    partido["status"] = "scenario"
    partido["home_goals"], partido["away_goals"] = 0, 2

    guion = """
        const equipos = doc.competitions.ESP1.teams;
        const indice = new Map(equipos.map((t, i) => [t.team_id, i]));
        const estado = motor.estadoInicial(doc.calendar, indice, equipos.length);
        const local = indice.get(PARTIDO.home_team.team_id);
        const visitante = indice.get(PARTIDO.away_team.team_id);
        return {
            local: {
                puntos: estado.puntos[local],
                ganados: estado.ganados[local],
                empatados: estado.empatados[local],
                perdidos: estado.perdidos[local],
                hipoteticos: estado.hipoteticos[local],
                forma: estado.forma[local].slice(-1)[0],
            },
            visitante: {
                puntos: estado.puntos[visitante],
                ganados: estado.ganados[visitante],
                empatados: estado.empatados[visitante],
                perdidos: estado.perdidos[visitante],
                hipoteticos: estado.hipoteticos[visitante],
                forma: estado.forma[visitante].slice(-1)[0],
            },
        };
    """.replace("PARTIDO", json.dumps(partido))
    salida = ejecutar(guion, doc)

    local = indice[partido["home_team"]["team_id"]]
    visitante = indice[partido["away_team"]["team_id"]]
    assert salida["local"]["puntos"] == equipos[local]["current"]["points"]
    assert salida["local"]["perdidos"] >= 1
    assert salida["local"]["hipoteticos"] == 1
    assert salida["local"]["forma"] == {"r": "P", "hipotesis": True}
    assert salida["visitante"]["puntos"] == equipos[visitante]["current"]["points"] + 3
    assert salida["visitante"]["ganados"] >= 1
    assert salida["visitante"]["hipoteticos"] == 1
    assert salida["visitante"]["forma"] == {"r": "G", "hipotesis": True}


def test_la_misma_semilla_da_el_mismo_resultado(documento):
    """Dos tiradas con la misma semilla coinciden exactamente.

    Sin esto no se puede comparar un escenario con otro: cualquier diferencia
    podria ser del azar del muestreo en vez de los resultados puestos a mano.
    """
    guion = """
        const a = await motor.simular(copia(), 1500);
        const b = await motor.simular(copia(), 1500);
        return {
            a: a.teams.map(t => [t.name, t.outcomes.title, t.outcomes.relegation]),
            b: b.teams.map(t => [t.name, t.outcomes.title, t.outcomes.relegation]),
        };
    """
    salida = ejecutar(guion, documento)
    assert salida["a"] == salida["b"]


def test_semillas_distintas_dan_resultados_distintos(documento):
    """Y con semillas distintas no: si no, la semilla no se estaria usando."""
    guion = """
        const uno = copia(); uno.simulation.seed = 1;
        const dos = copia(); dos.simulation.seed = 424242;
        const a = await motor.simular(uno, 1500);
        const b = await motor.simular(dos, 1500);
        return {
            a: a.teams.map(t => t.outcomes.title),
            b: b.teams.map(t => t.outcomes.title),
        };
    """
    salida = ejecutar(guion, documento)
    assert salida["a"] != salida["b"]


# --------------------------------------------------------------------------
# Los desempates

def test_el_desempate_usa_el_enfrentamiento_directo():
    """Empatados a puntos, manda la mini-liga y no la diferencia general.

    El caso es el de Levante y Mallorca en 2025-26: 42 puntos los dos, mejor
    diferencia general el Mallorca, y bajo el Mallorca porque el Levante gano
    el enfrentamiento directo. Ordenar por diferencia general da la respuesta
    contraria, y durante un tiempo el simulador la daba.
    """
    guion = """
        const n = 3;
        const puntos = Int16Array.from([42, 42, 50]);
        const diferencia = Int16Array.from([-5, 0, 10]);   // el 1 es mejor en general
        const favor = Int16Array.from([40, 40, 55]);
        const h2hPuntos = new Int16Array(n * n);
        const h2hGoles = new Int16Array(n * n);
        // El equipo 0 gano los dos duelos al equipo 1.
        h2hPuntos[0 * n + 1] = 6;
        h2hGoles[0 * n + 1] = 3;
        h2hGoles[1 * n + 0] = -3;
        const orden = new Int32Array(n);
        motor.ordenar(orden, puntos, diferencia, favor, h2hPuntos, h2hGoles, n);
        return Array.from(orden);
    """
    # Primero el que tiene 50 puntos; despues el 0, que pierde en diferencia
    # general pero gana el enfrentamiento directo.
    assert ejecutar(guion) == [2, 0, 1]


def test_sin_enfrentamiento_directo_decide_la_diferencia_general():
    """Si la mini-liga tambien empata, se cae al criterio siguiente."""
    guion = """
        const n = 2;
        const orden = new Int32Array(n);
        motor.ordenar(orden, Int16Array.from([42, 42]), Int16Array.from([-5, 3]),
                      Int16Array.from([40, 44]), new Int16Array(4), new Int16Array(4), n);
        return Array.from(orden);
    """
    assert ejecutar(guion) == [1, 0]


def test_el_motor_va_incrustado_en_el_panel():
    """El panel generado tiene que llevar el motor dentro, no una referencia.

    Es un fichero suelto que se abre con doble clic o se manda por WhatsApp: si
    el motor viviera en otro fichero, la copia llegaria sin el y los botones
    dejarian de funcionar sin decir por que.
    """
    from simliga.output.dashboard import render_dashboard

    html = render_dashboard({"season": "2026-27"}, servido=False)
    assert "MotorSimLiga" in html
    assert "MOTOR_JS" not in html
    assert 'src="motor.js"' not in html


# --------------------------------------------------------------------------
# El recuento que alimenta la pestaña de clasificacion simulada
#
# `estadoInicial` cuenta el calendario una sola vez y de ahi salen dos cosas:
# el punto de partida de la simulacion y la tabla que se ensena. Tenerlo en un
# solo sitio es lo que impide que las dos acaben sin cuadrar.

CALENDARIO = {
    "matchdays": [{
        "matchday": 1,
        "matches": [
            # Jugado de verdad: 1 gana a 2.
            {"match_id": 1, "status": "played", "home_goals": 3, "away_goals": 0,
             "home_team": {"team_id": 10}, "away_team": {"team_id": 20}},
            # Hipotesis: 3 empata con 4.
            {"match_id": 2, "status": "scenario", "home_goals": 1, "away_goals": 1,
             "home_team": {"team_id": 30}, "away_team": {"team_id": 40}},
            # Sin tocar: no cuenta para nada.
            {"match_id": 3, "status": "pending", "home_goals": None, "away_goals": None,
             "home_team": {"team_id": 10}, "away_team": {"team_id": 30}},
        ],
    }],
}


def _estado(calendario=CALENDARIO):
    guion = """
        const indice = new Map([[10, 0], [20, 1], [30, 2], [40, 3]]);
        const e = motor.estadoInicial(doc, indice, 4);
        const lista = (a) => Array.from(a);
        return {
            puntos: lista(e.puntos), jugados: lista(e.jugados),
            reales: lista(e.jugadosReales), hipoteticos: lista(e.hipoteticos),
            ganados: lista(e.ganados), empatados: lista(e.empatados),
            perdidos: lista(e.perdidos), forma: e.forma,
            favor: lista(e.favor), contra: lista(e.contra),
        };
    """
    return ejecutar(guion, calendario)


def test_cuenta_ganados_empatados_y_perdidos():
    e = _estado()
    assert e["ganados"] == [1, 0, 0, 0]
    assert e["empatados"] == [0, 0, 1, 1]
    assert e["perdidos"] == [0, 1, 0, 0]
    assert e["puntos"] == [3, 0, 1, 1]


def test_los_partidos_pendientes_no_cuentan():
    """Es lo que separa esta tabla de la proyeccion: aqui no se simula nada."""
    e = _estado()
    assert e["jugados"] == [1, 1, 1, 1]        # el pendiente del 10 y el 30 no suma
    assert e["favor"] == [3, 0, 1, 1]
    assert e["contra"] == [0, 3, 1, 1]


def test_los_partidos_en_juego_no_cuentan_como_resultado():
    """El parcial se muestra, pero la tabla real espera al final del partido."""
    calendario = {"matchdays": [{
        "matchday": 1,
        "matches": [
            {"match_id": 1, "status": "live", "home_goals": None, "away_goals": None,
             "live_home_goals": 1, "live_away_goals": 0, "live_detail": "63'",
             "home_team": {"team_id": 10}, "away_team": {"team_id": 20}},
        ],
    }]}

    e = _estado(calendario)
    assert e["jugados"] == [0, 0, 0, 0]
    assert e["puntos"] == [0, 0, 0, 0]
    assert e["favor"] == [0, 0, 0, 0]


def test_distingue_lo_jugado_de_lo_hipotetico():
    """La tabla tiene que poder decir cuantos de esos puntos te los inventaste."""
    e = _estado()
    assert e["reales"] == [1, 1, 0, 0]
    assert e["hipoteticos"] == [0, 0, 1, 1]


def test_la_racha_marca_los_resultados_puestos_a_mano():
    e = _estado()
    assert e["forma"][0] == [{"r": "G", "hipotesis": False}]
    assert e["forma"][1] == [{"r": "P", "hipotesis": False}]
    assert e["forma"][2] == [{"r": "E", "hipotesis": True}]


def test_sin_hipotesis_el_recuento_es_el_de_la_clasificacion_real():
    """Sin resultados puestos, la tabla simulada debe ser la de verdad.

    Si difirieran, una de las dos estaria contando mal, y la pestaña nueva
    seria una segunda version de la clasificacion que se contradice con la
    primera.
    """
    solo_jugados = {"matchdays": [{
        "matchday": 1,
        "matches": [m for m in CALENDARIO["matchdays"][0]["matches"]
                    if m["status"] == "played"],
    }]}
    e = _estado(solo_jugados)
    assert e["hipoteticos"] == [0, 0, 0, 0]
    assert e["jugados"] == e["reales"]
    assert e["puntos"] == [3, 0, 0, 0]
