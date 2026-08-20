"""Identidad visual de equipos para el panel.

El modelo usa `team_id` y el nombre que llega de las fuentes como identidad
estable. Esta capa es solo de presentacion: corrige tildes y variantes visibles,
y aporta escudos cuando hay una fuente fiable.
"""
from __future__ import annotations

from pathlib import Path

from ..ingest.club_names import normalizar


ASSET_PREFIX = "assets/escudos"
ASSET_DIR = Path(__file__).parent / "assets" / "escudos"
ESPN_LOGO_BASE_URL = "https://a.espncdn.com/i/teamlogos/soccer/500"


DISPLAY_NAMES = {
    "atletico de madrid": "Atlético de Madrid",
    "deportivo alaves": "Deportivo Alavés",
    "deportivo de la coruna": "Deportivo de A Coruña",
    "malaga cf": "Málaga CF",
    "cadiz cf": "Cádiz CF",
    "cordoba cf": "Córdoba CF",
    "gimnastic de tarragona": "Gimnàstic de Tarragona",
    "ud almeria": "UD Almería",
    "ud logrones": "UD Logroñés",
}


ESPN_LOGO_IDS = {
    "athletic club": 93,
    "atletico de madrid": 1068,
    "ca osasuna": 97,
    "celta de vigo": 85,
    "deportivo alaves": 96,
    "deportivo de la coruna": 90,
    "elche cf": 3751,
    "fc barcelona": 83,
    "getafe cf": 2922,
    "levante ud": 1538,
    "malaga cf": 99,
    "racing de santander": 87,
    "rayo vallecano": 101,
    "rcd espanyol": 88,
    "real betis": 244,
    "real madrid": 86,
    "real sociedad": 89,
    "sevilla fc": 243,
    "valencia cf": 94,
    "villarreal cf": 102,
}


def logo_filename(name: str) -> str:
    """Nombre estable del fichero local de escudo."""
    return f"{normalizar(name).replace(' ', '-')}.png"


def display_name(name: str) -> str:
    """Nombre curado para ensenar en el panel."""
    return DISPLAY_NAMES.get(normalizar(name), name)


def logo_source_url(name: str) -> str | None:
    """URL de origen del escudo, usada solo para descargar el asset local."""
    espn_id = ESPN_LOGO_IDS.get(normalizar(name))
    if espn_id is None:
        return None
    return f"{ESPN_LOGO_BASE_URL}/{espn_id}.png"


def logo_url(name: str) -> str | None:
    """Ruta local del escudo, si el equipo esta mapeado."""
    if normalizar(name) not in ESPN_LOGO_IDS:
        return None
    return f"{ASSET_PREFIX}/{logo_filename(name)}"


def team_identity(name: str) -> dict:
    """Campos de presentacion comunes para cualquier objeto de equipo."""
    return {
        "display_name": display_name(name),
        "logo": logo_url(name),
    }
