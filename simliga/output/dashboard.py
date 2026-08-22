"""Generacion del panel HTML.

El panel lleva el CSS, JavaScript y JSON incrustados. Los assets visuales que no
conviene meter en base64 (escudos) se copian al lado del HTML generado. Se abre
con doble clic, sin servidor y sin conexion.

Va incrustado y no con `fetch` a proposito: un `fetch` desde un fichero local
lo bloquea el navegador por politica de origen, asi que habria que levantar un
servidor solo para ver una pagina estatica. Por lo mismo se incrusta tambien
`motor.js`, el simulador en JavaScript, que es lo que permite que una copia
suelta vuelva a simular por su cuenta.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .team_identity import ASSET_DIR

PLANTILLA = Path(__file__).parent / "dashboard.html"
MOTOR = Path(__file__).parent / "motor.js"
ICONO = Path(__file__).parent / "icono.png"
MARCA = "DATOS_JSON"
MARCA_MODO = "MODO_PANEL"
MARCA_MOTOR = "MOTOR_JS"


def render_dashboard(document: dict, servido: bool = False) -> str:
    """HTML del panel con los datos incrustados.

    `servido` marca la pagina como servida por `simliga servidor`, que es lo
    unico que habilita los botones de regenerar y editar escenarios. No vale
    mirar el protocolo desde JavaScript: una copia estatica subida a cualquier
    sitio tambien se sirve por http, y entonces los botones se activarian para
    llamar a una API que no existe.
    """
    plantilla = PLANTILLA.read_text(encoding="utf-8")
    for marca in (MARCA, MARCA_MODO, MARCA_MOTOR):
        if marca not in plantilla:
            raise ValueError(f"La plantilla no contiene la marca {marca}")
    plantilla = plantilla.replace(MARCA_MODO, "servidor" if servido else "estatico")
    plantilla = plantilla.replace(MARCA_MOTOR, MOTOR.read_text(encoding="utf-8"))

    # `</script>` dentro del JSON cerraria la etiqueta que lo contiene y
    # rompería la pagina; escaparlo es la unica precaucion necesaria porque el
    # bloque va como application/json, no como codigo.
    datos = json.dumps(document, ensure_ascii=False).replace("</", "<\\/")
    return plantilla.replace(MARCA, datos)


def write_dashboard(document: dict, path: str | Path, servido: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(document, servido), encoding="utf-8")
    copy_dashboard_assets(path.parent)
    write_mobile_assets(path.parent)
    return path


def copy_dashboard_assets(output_dir: Path) -> None:
    """Copia los assets locales que referencia el HTML generado."""
    if not ASSET_DIR.exists():
        return
    target_dir = output_dir / "assets" / "escudos"
    target_dir.mkdir(parents=True, exist_ok=True)
    for asset in ASSET_DIR.glob("*"):
        if asset.is_file():
            shutil.copy2(asset, target_dir / asset.name)


def write_mobile_assets(output_dir: Path) -> None:
    """Ficheros auxiliares para instalar el panel como app movil."""
    if ICONO.exists():
        shutil.copy2(ICONO, output_dir / "icono.png")

    manifest = {
        "name": "SimLiga",
        "short_name": "SimLiga",
        "description": "Panel de simulacion de LaLiga",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": "#F1F3F5",
        "theme_color": "#8E2437",
        "icons": [
            {
                "src": "icono.png",
                "sizes": "180x180",
                "type": "image/png",
                "purpose": "any maskable",
            }
        ],
    }
    (output_dir / "manifest.webmanifest").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    service_worker = """\
const CACHE = "simliga-panel-v1";
const ASSETS = [
  "./",
  "./index.html",
  "./panel.html",
  "./manifest.webmanifest",
  "./icono.png"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE)
      .then(cache => cache.addAll(ASSETS))
      .catch(() => undefined)
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== CACHE).map(key => caches.delete(key))
    ))
  );
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  // Solo se guarda lo propio. El marcador en directo se le pide a ESPN cada
  // tres cuartos de minuto y caduca en segundos: cachearlo no sirve de nada
  // sin conexion y llena la cache de respuestas viejas.
  if (new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const copy = response.clone();
        caches.open(CACHE).then(cache => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request).then(cached => cached || caches.match("./index.html")))
  );
});
"""
    (output_dir / "service-worker.js").write_text(service_worker, encoding="utf-8")
