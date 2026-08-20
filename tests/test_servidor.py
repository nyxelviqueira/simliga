"""Tests del servidor local del panel."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from simliga.server import MAX_SIMS, MIN_SIMS, SimLigaHTTPServer, crear_handler


class MotorFalso:
    """Sustituye al motor real: los tests prueban el servidor, no el modelo."""

    def __init__(self):
        self.cfg = type("cfg", (), {"sim": type("s", (), {"n_sims": 1000})()})()
        self.documento = {"season": "2026-27", "simulation": {"n_sims": 1000},
                          "competitions": {}, "as_of": {"date": "2026-08-20"}}
        self.llamadas = []
        self.fallo: Exception | None = None
        self.desfasado = False

    def escenario_desfasado(self):
        return self.desfasado

    def recargar_escenario(self):
        pass

    def simular(self, n_sims, refrescar=False):
        if self.fallo:
            raise self.fallo
        n = max(MIN_SIMS, min(int(n_sims), MAX_SIMS))
        self.llamadas.append((n, refrescar))
        self.documento = {**self.documento, "simulation": {"n_sims": n}}
        return self.documento


@pytest.fixture
def servidor():
    motor = MotorFalso()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), crear_handler(motor))
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", motor
    httpd.shutdown()
    httpd.server_close()


# `Connection: close` en todas las peticiones: el servidor habla HTTP/1.1 con
# conexiones persistentes, y urllib puede reutilizar una que quedo abierta
# contra un servidor de prueba anterior ya cerrado. Eso hacia fallar un test al
# azar de vez en cuando.
CABECERAS = {"Content-Type": "application/json", "Connection": "close"}


def _post(base, ruta, datos):
    peticion = urllib.request.Request(
        base + ruta, data=json.dumps(datos).encode(),
        headers=CABECERAS, method="POST")
    try:
        with urllib.request.urlopen(peticion, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_simular_devuelve_el_documento_nuevo(servidor):
    base, motor = servidor
    codigo, cuerpo = _post(base, "/api/simular", {"n_sims": 5000})
    assert codigo == 200
    assert cuerpo["simulation"]["n_sims"] == 5000
    assert motor.llamadas == [(5000, False)]


def test_el_numero_de_simulaciones_se_acota(servidor):
    """Un cero o un millon no deben tumbar el servidor ni congelar el navegador."""
    base, _ = servidor
    assert _post(base, "/api/simular", {"n_sims": 10_000_000})[1]["simulation"]["n_sims"] == MAX_SIMS
    assert _post(base, "/api/simular", {"n_sims": 0})[1]["simulation"]["n_sims"] == MIN_SIMS


def test_se_puede_pedir_refrescar_los_datos(servidor):
    base, motor = servidor
    _post(base, "/api/simular", {"n_sims": 2000, "refrescar": True})
    assert motor.llamadas == [(2000, True)]


def test_un_problema_de_integridad_devuelve_409_y_lo_explica(servidor):
    base, motor = servidor
    motor.fallo = ValueError("ESP1 2026-27: 21 equipos")
    codigo, cuerpo = _post(base, "/api/simular", {"n_sims": 1000})
    assert codigo == 409
    assert "21 equipos" in cuerpo["error"]


def test_un_fallo_inesperado_devuelve_500_sin_tumbar_el_servidor(servidor):
    base, motor = servidor
    motor.fallo = RuntimeError("algo se rompio")
    assert _post(base, "/api/simular", {"n_sims": 1000})[0] == 500
    motor.fallo = None
    assert _post(base, "/api/simular", {"n_sims": 1000})[0] == 200, "sigue en pie"


def test_una_peticion_mal_formada_devuelve_400(servidor):
    base, _ = servidor
    peticion = urllib.request.Request(
        base + "/api/simular", data=b"{esto no es json",
        headers=CABECERAS, method="POST")
    try:
        urllib.request.urlopen(peticion, timeout=10)
        raise AssertionError("deberia haber fallado")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_una_ruta_desconocida_devuelve_404(servidor):
    base, _ = servidor
    assert _post(base, "/api/otra-cosa", {})[0] == 404


def test_el_servidor_atiende_peticiones_seguidas(servidor):
    """Varias peticiones sobre el mismo servidor no deben pisarse entre si."""
    base, _ = servidor
    for _ in range(5):
        assert _post(base, "/api/simular", {"n_sims": 500})[0] == 200


def test_la_pagina_principal_sirve_el_panel(servidor):
    base, _ = servidor
    with urllib.request.urlopen(
            urllib.request.Request(base + "/", headers={"Connection": "close"}),
            timeout=10) as r:
        html = r.read().decode("utf-8")
    assert r.status == 200
    assert "DATOS_JSON" not in html
    assert 'id="regenerar"' in html


def test_el_servidor_sirve_los_escudos_locales(servidor):
    base, _ = servidor
    with urllib.request.urlopen(
            urllib.request.Request(
                base + "/assets/escudos/real-betis.png",
                headers={"Connection": "close"}),
            timeout=10) as r:
        contenido = r.read()
    assert r.status == 200
    assert r.headers["Content-Type"] == "image/png"
    assert contenido.startswith(b"\x89PNG")


def test_la_pagina_se_rehace_si_el_escenario_cambio_por_fuera(servidor):
    """Recargar debe mostrar el estado real, no uno de hace dos cambios."""
    base, motor = servidor
    with urllib.request.urlopen(
            urllib.request.Request(base + "/", headers={"Connection": "close"}), timeout=10):
        pass
    antes = len(motor.llamadas)

    motor.desfasado = True
    with urllib.request.urlopen(
            urllib.request.Request(base + "/", headers={"Connection": "close"}), timeout=10):
        pass
    assert len(motor.llamadas) == antes + 1


def test_poner_y_quitar_un_resultado_hipotetico(servidor, monkeypatch):
    base, _ = servidor
    guardados = {}
    monkeypatch.setattr("simliga.server.set_scenario_result",
                        lambda conn, m, h, a: guardados.__setitem__(m, (h, a)))
    monkeypatch.setattr("simliga.server.clear_scenario_result",
                        lambda conn, m: guardados.pop(m, None) is not None)
    monkeypatch.setattr("simliga.server.connect", lambda *a, **k: None)

    codigo, cuerpo = _post(base, "/api/escenario",
                           {"match_id": 7, "home_goals": 2, "away_goals": 1})
    assert codigo == 200 and guardados == {7: (2, 1)}

    codigo, cuerpo = _post(base, "/api/escenario", {"match_id": 7})
    assert codigo == 200 and cuerpo["borrado"] is True and guardados == {}


def test_no_deja_tocar_un_partido_ya_jugado(servidor, monkeypatch):
    base, _ = servidor

    def negarse(conn, m, h, a):
        raise ValueError("Ese partido ya se jugo: su resultado no se puede cambiar")

    monkeypatch.setattr("simliga.server.set_scenario_result", negarse)
    monkeypatch.setattr("simliga.server.connect", lambda *a, **k: None)

    codigo, cuerpo = _post(base, "/api/escenario",
                           {"match_id": 7, "home_goals": 9, "away_goals": 9})
    assert codigo == 409
    assert "ya se jugo" in cuerpo["error"]


def test_un_partido_inexistente_devuelve_404(servidor, monkeypatch):
    base, _ = servidor

    def no_existe(conn, m, h, a):
        raise KeyError("No existe el partido 999")

    monkeypatch.setattr("simliga.server.set_scenario_result", no_existe)
    monkeypatch.setattr("simliga.server.connect", lambda *a, **k: None)
    assert _post(base, "/api/escenario",
                 {"match_id": 999, "home_goals": 1, "away_goals": 0})[0] == 404


def test_sin_identificador_de_partido_devuelve_400(servidor):
    base, _ = servidor
    assert _post(base, "/api/escenario", {"home_goals": 1})[0] == 400


def test_una_ruta_desconocida_no_deja_la_conexion_corrupta(servidor):
    """Regresion: el 404 respondia sin leer el cuerpo de la peticion.

    Con conexiones persistentes, ese cuerpo sin consumir se quedaba en el socket
    y rompia la peticion siguiente. El sintoma era un corte aleatorio en otra
    llamada, sin relacion aparente con la que lo causaba.
    """
    base, _ = servidor
    assert _post(base, "/api/desconocida", {"relleno": "x" * 500})[0] == 404
    assert _post(base, "/api/simular", {"n_sims": 500})[0] == 200
    assert _post(base, "/api/otra", {"relleno": "y" * 500})[0] == 404
    assert _post(base, "/api/simular", {"n_sims": 500})[0] == 200


# ------------------------------------------------------------ acceso en red
def test_por_defecto_solo_escucha_en_local():
    """Abrirlo a la red tiene que ser una decision explicita, nunca el defecto.

    El panel deja escribir escenarios y no pide contrasena a nadie.
    """
    import inspect

    from simliga.server import servir

    assert inspect.signature(servir).parameters["host"].default == "127.0.0.1"


def test_la_bandera_de_red_es_la_que_abre_el_servidor():
    from simliga.cli import build_parser

    local = build_parser().parse_args(["servidor", "--temporada", "2026-27"])
    en_red = build_parser().parse_args(["servidor", "--temporada", "2026-27", "--en-red"])
    assert local.en_red is False
    assert en_red.en_red is True


def test_detecta_la_direccion_de_red_del_equipo():
    """Sin ella habria que buscar la IP a mano para abrirlo desde el movil."""
    from simliga.server import _direcciones_locales

    direcciones = _direcciones_locales()
    assert isinstance(direcciones, list)
    for d in direcciones:
        partes = d.split(".")
        assert len(partes) == 4 and all(p.isdigit() for p in partes)
        assert d != "127.0.0.1", "la de bucle local no sirve para el movil"


# ---------------------------------------------------------------------------
# Control de acceso
#
# El token no existe para el uso local: en 127.0.0.1 no llega nadie mas. Existe
# para el momento en que el panel sale de la maquina (`--en-red`, o un hosting),
# porque `/api/escenario` y `/api/simular` escriben en la base de datos sin
# preguntar quien llama.

@pytest.fixture
def servidor_con_token(monkeypatch):
    """Un servidor que exige token, con el modulo parcheado antes de arrancar."""
    import simliga.server as servidor_mod

    monkeypatch.setattr(servidor_mod, "TOKEN", "clave-de-prueba")
    motor = MotorFalso()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), servidor_mod.crear_handler(motor))
    hilo = threading.Thread(target=httpd.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", motor
    httpd.shutdown()
    httpd.server_close()


def _get(base, ruta, cabeceras=None):
    peticion = urllib.request.Request(
        base + ruta, headers={**CABECERAS, **(cabeceras or {})})
    try:
        with urllib.request.urlopen(peticion, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_sin_token_no_se_puede_leer(servidor_con_token):
    base, _ = servidor_con_token
    codigo, _, _ = _get(base, "/")
    assert codigo == 401


def test_sin_token_no_se_puede_escribir(servidor_con_token):
    """La que de verdad importa: sin token no se puede tocar nada."""
    base, motor = servidor_con_token
    codigo, _ = _post(base, "/api/simular", {"n_sims": 5000})
    assert codigo == 401
    assert motor.llamadas == [], "no debe haber llegado a simular"


def test_un_token_equivocado_tampoco_vale(servidor_con_token):
    base, _ = servidor_con_token
    codigo, _ = _post(base, "/api/simular?t=otra-cosa", {"n_sims": 5000})
    assert codigo == 401


def test_con_el_token_en_el_enlace_se_entra_y_queda_en_cookie(servidor_con_token):
    """El enlace largo se pega una vez; despues basta la direccion."""
    base, _ = servidor_con_token
    codigo, _, cabeceras = _get(base, "/?t=clave-de-prueba")
    assert codigo == 200

    galleta = cabeceras.get("Set-Cookie", "")
    assert "simliga_token=clave-de-prueba" in galleta
    # Sin HttpOnly cualquier script de la pagina podria leerlo, y no lo necesita.
    assert "HttpOnly" in galleta
    assert "SameSite=Strict" in galleta

    codigo, _, _ = _get(base, "/", {"Cookie": "simliga_token=clave-de-prueba"})
    assert codigo == 200


def test_con_cookie_valida_se_puede_escribir(servidor_con_token):
    base, motor = servidor_con_token
    peticion = urllib.request.Request(
        base + "/api/simular", data=json.dumps({"n_sims": 5000}).encode(),
        headers={**CABECERAS, "Cookie": "simliga_token=clave-de-prueba"},
        method="POST")
    with urllib.request.urlopen(peticion, timeout=10) as r:
        assert r.status == 200
    assert motor.llamadas == [(5000, False)]


def test_sin_token_configurado_no_se_pide_nada(servidor):
    """El uso local no cambia: sin `SIMLIGA_TOKEN`, el panel abre y punto."""
    base, _ = servidor
    codigo, _, cabeceras = _get(base, "/")
    assert codigo == 200
    assert "Set-Cookie" not in cabeceras


def test_el_token_se_compara_en_tiempo_constante():
    """Comparar con `==` filtra el token carácter a carácter por el tiempo."""
    import inspect

    import simliga.server as servidor_mod

    fuente = inspect.getsource(servidor_mod.token_valido)
    assert "compare_digest" in fuente


def test_el_servidor_local_sigue_escuchando_solo_en_local():
    """`--en-red` tiene que seguir siendo explicito, nunca el valor por defecto."""
    import inspect

    from simliga.server import servir

    assert inspect.signature(servir).parameters["host"].default == "127.0.0.1"


def test_no_reutiliza_puerto_para_no_mezclar_servidores_viejos():
    """Si dos servidores comparten puerto, el navegador puede caer en el viejo."""
    assert SimLigaHTTPServer.allow_reuse_address is False
