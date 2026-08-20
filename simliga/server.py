"""Servidor local del panel.

El panel suelto ya simula por su cuenta: lleva dentro `motor.js`. Lo que no
puede hacer solo es **descargar datos nuevos**, que necesita Python y red. Ese
es el trabajo de este servidor, y lo que activa el boton "Actualizar datos".

    python -m simliga servidor --temporada 2026-27

Por defecto escucha solo en 127.0.0.1 y no pide contrasena: en local no hace
falta, porque a esa direccion no llega nadie mas. En cuanto sale de la maquina
(`--en-red`, o desplegado en un hosting) la cosa cambia, porque sus endpoints
escriben en la base de datos sin preguntar quien llama. Para eso esta
`SIMLIGA_TOKEN`: si la variable de entorno tiene valor, se exige en todas las
peticiones.

Lo caro es construir el contexto del modelo (recorrer el Elo sobre 50.000
partidos), no simular. Por eso el contexto se cachea y solo se reconstruye
cuando llegan datos nuevos: asi el boton responde en un par de segundos en vez
de en veinte.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .config import COMP_LALIGA, Config, load_config
from .data import check_season_integrity
from .db import (clear_all_scenarios, clear_scenario_result, connect,
                 set_scenario_result)
from .output.contract import (build_calendar_block, build_european_block,
                              build_fixtures_block, build_league_block,
                              build_european_qualification, build_output,
                              build_standings_block)
from .output.dashboard import render_dashboard
from .output.team_identity import ASSET_DIR
from .pipeline import (build_context, clasificados_a_europa, european_blocks,
                       real_upcoming_matches, season_split, simulate_laliga)

MAX_SIMS = 200_000
MIN_SIMS = 100

# Contrasena de acceso, si la hay. Vacia = sin comprobacion, que es lo correcto
# en local (127.0.0.1 solo lo alcanza esta maquina). Puesta, se exige en todas
# las peticiones: hace falta en cuanto el panel sale de la maquina, porque sus
# endpoints escriben en la base de datos sin preguntar quien llama.
TOKEN = os.environ.get("SIMLIGA_TOKEN", "").strip()
COOKIE = "simliga_token"


class SimLigaHTTPServer(ThreadingHTTPServer):
    """Servidor sin reutilizacion de puerto.

    En Windows, `SO_REUSEADDR` puede dejar varios procesos escuchando en el
    mismo puerto. Entonces el navegador cae unas veces en el servidor nuevo y
    otras en uno viejo, que fue exactamente lo que hizo desaparecer escudos.
    """

    allow_reuse_address = False


def token_valido(recibido: str | None) -> bool:
    """Compara en tiempo constante, para no filtrar el token carácter a carácter."""
    if not TOKEN:
        return True
    return bool(recibido) and hmac.compare_digest(recibido, TOKEN)


class Motor:
    """Estado compartido: contexto del modelo y ultimo documento generado."""

    def __init__(self, season: str, cfg: Config, data_sources: list[str]):
        self.season = season
        self.cfg = cfg
        self.data_sources = data_sources
        self.lock = threading.Lock()
        self.ctx = None
        self.documento: dict | None = None

    def refrescar_datos(self) -> list[str]:
        """Descarga datos nuevos de todas las fuentes. Devuelve avisos de integridad.

        Es la misma ingesta que `simliga actualizar`, incluida ESPN: sin ella se
        actualizarian los resultados pero no los dias y horas que LaLiga acaba
        de confirmar, y el calendario seguiria ensenando fechas de relleno.
        """
        from .ingest.espn import update_schedule
        from .ingest.football_data_uk import ingest_range
        from .ingest.openfootball import ingest_season as ingest_calendario
        from .ingest.uefa import ingest_range as ingest_uefa

        conn = connect()
        inicio = int(self.season.split("-")[0])
        for nombre, tarea in (
            ("resultados", lambda: ingest_range(conn, inicio, inicio, force_download=True)),
            ("calendario", lambda: ingest_calendario(conn, self.season, force_download=True)),
            ("horarios", lambda: update_schedule(conn, self.season, force_download=True)),
            ("competiciones UEFA", lambda: ingest_uefa(conn, (self.season,),
                                                       force_download=True)),
        ):
            try:
                tarea()
            except Exception as exc:                      # noqa: BLE001
                # Una fuente caida no debe tumbar el servidor: se sigue con las
                # otras y con lo que ya hubiera en la base de datos.
                print(f"  aviso al refrescar {nombre}: {exc}")

        self.ctx = None                                   # obliga a rehacer el Elo
        return check_season_integrity(conn, self.season)

    def recargar_escenario(self) -> None:
        """Relee los resultados hipoteticos manteniendo el contexto cacheado.

        Cambiar una hipotesis no cambia la fuerza de ningun equipo, asi que no
        hay que rehacer el Elo: eso es lo que permite que el panel responda al
        instante al editar un marcador.
        """
        from .db import load_scenario_results

        if self.ctx is not None:
            self.ctx.scenario = load_scenario_results(connect())

    def escenario_desfasado(self) -> bool:
        """¿El escenario guardado difiere del que refleja el ultimo documento?"""
        from .db import load_scenario_results

        if self.documento is None:
            return True
        guardado = load_scenario_results(connect(), self.season)
        calendario = self.documento.get("calendar") or {}
        reflejado = {
            m["match_id"]: (m["home_goals"], m["away_goals"])
            for bloque in calendario.get("matchdays", [])
            for m in bloque["matches"] if m["status"] == "scenario"
        }
        return guardado != reflejado

    def simular(self, n_sims: int, refrescar: bool = False) -> dict:
        with self.lock:
            avisos = self.refrescar_datos() if refrescar else []
            if avisos:
                raise ValueError("; ".join(avisos))

            cfg = load_config()
            cfg.sim.n_sims = max(MIN_SIMS, min(int(n_sims), MAX_SIMS))
            cfg.modifiers = self.cfg.modifiers

            if self.ctx is None:
                self.ctx = build_context(cfg=cfg, season=self.season)
            else:
                self.recargar_escenario()
            ctx = self.ctx
            ctx.cfg = cfg

            resultado, fit = simulate_laliga(ctx, self.season)
            liga = ctx.matches[(ctx.matches["competition"] == COMP_LALIGA)
                               & (ctx.matches["season"] == self.season)]
            jugados, pendientes, reales = season_split(ctx, self.season)
            proximos_reales = real_upcoming_matches(ctx, self.season)

            bloque = build_league_block(resultado, fit, ctx.names, ctx.elo_raw,
                                        jugados, pendientes, cfg, real_played=reales)
            europa = european_blocks(ctx, self.season)

            documento = build_output(
                season=self.season, league_block=bloque,
                fixtures=build_fixtures_block(fit, proximos_reales, ctx.names, limit=20,
                                              as_of=ctx.as_of),
                fit=fit, cfg=cfg, as_of=ctx.as_of, data_sources=self.data_sources,
                europe=europa, modifiers=cfg.modifiers.enabled_names(),
                calendar=build_calendar_block(fit, liga, ctx.names, ctx.scenario, ctx.as_of),
                standings=build_standings_block(
                    reales, [t['name'] for t in bloque['teams']], cfg),
                european_qualification=clasificados_a_europa(ctx, self.season, cfg),
            )
            documento["model"]["league_offsets"] = {
                k: round(v, 1) for k, v in ctx.league_offsets.items()}
            self.documento = documento
            return documento


def _direcciones_locales() -> list[str]:
    """Direcciones IP de este equipo en la red local, para abrirlo desde el movil."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))            # no envia nada, solo elige ruta
            return [s.getsockname()[0]]
    except OSError:
        return []


def crear_handler(motor: Motor):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, formato, *args):            # noqa: A002
            pass                                          # sin ruido en consola

        def _responder(self, codigo: int, cuerpo: bytes, tipo: str,
                       con_cookie: bool = False) -> None:
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            if con_cookie:
                self._guardar_token_en_cookie()
            self.end_headers()
            self.wfile.write(cuerpo)

        def _json(self, codigo: int, datos: dict) -> None:
            self._responder(codigo, json.dumps(datos, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8")

        # ------------------------------------------------------------- acceso
        def _token_recibido(self) -> str | None:
            """El token del enlace (`?t=...`) o el de la cookie que dejo ese enlace."""
            del_enlace = parse_qs(urlparse(self.path).query).get("t")
            if del_enlace:
                return del_enlace[0]
            galletas = SimpleCookie(self.headers.get("Cookie", ""))
            return galletas[COOKIE].value if COOKIE in galletas else None

        def _autorizado(self) -> bool:
            """Corta la peticion si no trae el token. Devuelve si puede seguir."""
            if token_valido(self._token_recibido()):
                return True
            self._json(401, {"error": "Falta el token de acceso"})
            return False

        def _guardar_token_en_cookie(self) -> None:
            """Deja el token en una cookie para no arrastrarlo en cada enlace.

            Asi el enlace largo se pega una vez en el movil y las siguientes
            visitas funcionan solas. `HttpOnly` porque el JavaScript de la
            pagina no lo necesita, y `SameSite=Strict` para que no viaje desde
            otro sitio.
            """
            if not TOKEN:
                return
            self.send_header(
                "Set-Cookie",
                f"{COOKIE}={TOKEN}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict")

        def do_GET(self):                                 # noqa: N802
            if not self._autorizado():
                return
            ruta = urlparse(self.path).path
            if ruta in ("/", "/index.html"):
                # Si el escenario cambio desde la ultima simulacion (por la API,
                # o por la linea de comandos en otra ventana), se rehace: una
                # recarga debe mostrar el estado real, no uno de hace dos
                # cambios sin avisar de que esta viejo.
                if motor.documento is None or motor.escenario_desfasado():
                    motor.simular(motor.documento["simulation"]["n_sims"]
                                  if motor.documento else motor.cfg.sim.n_sims)
                html = render_dashboard(motor.documento, servido=True).encode("utf-8")
                self._responder(200, html, "text/html; charset=utf-8", con_cookie=True)
            elif ruta.startswith("/assets/escudos/"):
                asset = ASSET_DIR / ruta.rsplit("/", 1)[-1]
                if asset.exists() and asset.is_file():
                    self._responder(200, asset.read_bytes(), "image/png")
                else:
                    self._json(404, {"error": "No existe ese escudo"})
            elif ruta == "/api/datos":
                self._json(200, motor.documento or {})
            else:
                self._json(404, {"error": "No existe esa ruta"})

        def _leer_json(self) -> dict:
            longitud = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(longitud) or b"{}")

        def do_POST(self):                                # noqa: N802
            # El cuerpo se lee SIEMPRE, incluso para responder un 404. Con
            # conexiones persistentes, dejarlo sin consumir en el socket
            # corrompe la siguiente peticion de esa misma conexion: el fallo se
            # veia como un corte aleatorio en una peticion posterior y sin
            # relacion aparente con esta.
            try:
                peticion = self._leer_json()
            except json.JSONDecodeError:
                self._json(400, {"error": "Peticion mal formada"})
                return
            if not self._autorizado():
                return

            ruta = urlparse(self.path).path
            if ruta not in ("/api/simular", "/api/escenario", "/api/escenario/limpiar"):
                self._json(404, {"error": "No existe esa ruta"})
                return

            if ruta == "/api/escenario":
                self._escenario(peticion)
                return
            if ruta == "/api/escenario/limpiar":
                n = clear_all_scenarios(connect(), motor.season)
                motor.recargar_escenario()
                self._json(200, {"borrados": n})
                return

            n_sims = peticion.get("n_sims", motor.cfg.sim.n_sims)
            refrescar = bool(peticion.get("refrescar"))
            try:
                documento = motor.simular(n_sims, refrescar)
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
                return
            except Exception as exc:                      # noqa: BLE001
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, documento)

        def _escenario(self, peticion: dict) -> None:
            """Fija o borra el resultado hipotetico de un partido."""
            try:
                match_id = int(peticion["match_id"])
            except (KeyError, TypeError, ValueError):
                self._json(400, {"error": "Falta el identificador del partido"})
                return

            conn = connect()
            local, visitante = peticion.get("home_goals"), peticion.get("away_goals")
            if local is None or visitante is None:
                borrado = clear_scenario_result(conn, match_id)
                motor.recargar_escenario()
                self._json(200, {"match_id": match_id, "borrado": borrado})
                return

            try:
                set_scenario_result(conn, match_id, int(local), int(visitante))
            except ValueError as exc:
                self._json(409, {"error": str(exc)})
                return
            except KeyError as exc:
                self._json(404, {"error": str(exc)})
                return
            except (TypeError, ValueError):
                self._json(400, {"error": "Marcador no valido"})
                return

            motor.recargar_escenario()
            self._json(200, {"match_id": match_id,
                             "home_goals": int(local), "away_goals": int(visitante)})

    return Handler


def servir(season: str, puerto: int = 8000, abrir: bool = True,
           cfg: Config | None = None, data_sources: list[str] | None = None,
           host: str = "127.0.0.1") -> None:
    cfg = cfg or load_config()
    motor = Motor(season, cfg, data_sources or ["football-data.co.uk", "openfootball"])

    print(f"Preparando el modelo para {season}...")
    motor.simular(cfg.sim.n_sims)
    print(f"Listo. {len(motor.documento['competitions'])} competicion(es), "
          f"corte {motor.documento['as_of']['date']}.")

    servidor = SimLigaHTTPServer((host, puerto), crear_handler(motor))
    puerto_real = servidor.server_address[1]
    url = f"http://127.0.0.1:{puerto_real}/"
    print(f"\nPanel en {url}   (Ctrl+C para parar)")
    if host != "127.0.0.1":
        # Escuchando en toda la red: se dan las direcciones utiles y se avisa si
        # ademas deja escribir escenarios sin pedirle contrasena a nadie.
        sufijo = f"?t={TOKEN}" if TOKEN else ""
        for direccion in _direcciones_locales():
            print(f"  desde el movil:  http://{direccion}:{puerto_real}/{sufijo}")
        if TOKEN:
            print("\n  Con token: hay que abrir el enlace entero la primera vez.")
            print("  Despues queda en una cookie y basta con la direccion.")
        else:
            print("\n  AVISO: accesible para cualquiera en esta red, sin contrasena.")
            print("  No lo dejes abierto en una red que no controles. Para exigir")
            print("  una, arranca con la variable SIMLIGA_TOKEN puesta.")
        print("  Si el movil no conecta es el cortafuegos de Windows; permite")
        print("  Python en redes privadas o, como administrador, ejecuta:")
        print(f'    netsh advfirewall firewall add rule name="SimLiga"'
              f' dir=in action=allow protocol=TCP localport={puerto_real}')
    if abrir:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        servidor.server_close()
