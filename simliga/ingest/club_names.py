"""Resolucion de nombres de club entre fuentes distintas.

El mismo equipo se llama `Bayern Munich` en football-data.co.uk y
`FC Bayern München` en openfootball. Sin un puente entre ambos, el pool de Elo
europeo se parte en dos fichas por equipo y todo el modelo continental queda
inservible.

El puente lo da `openfootball/clubs`: un repositorio CC0 con el nombre canonico
de cada club y sus alias, incluidas las variantes inglesas marcadas `[en]`, que
son justo las que usa football-data.co.uk. Formato:

    Bayern München, 1900,    @ Allianz Arena,   München
      | Bayern | Bayern Mün.
      | FC Bayern | FC Bayern München
      | Bayern Munich [en] | FC Bayern Munich [en]

Los filiales (lineas que empiezan por `ii)`) se ignoran a proposito: no juegan
competicion europea y sus alias chocarian con los del primer equipo.
"""
from __future__ import annotations

import re
import unicodedata
import zipfile
from pathlib import Path

import requests

from ..config import RAW_DIR

CLUBS_URL = "https://github.com/openfootball/clubs/archive/refs/heads/master.zip"
CLUBS_ZIP = RAW_DIR / "openfootball_clubs.zip"

# Codigo de pais UEFA -> carpeta del repositorio de clubes.
PAIS_A_CARPETA = {
    "ESP": "spain", "ENG": "england", "GER": "germany", "ITA": "italy",
    "FRA": "france", "NED": "netherlands", "POR": "portugal", "BEL": "belgium",
    "TUR": "turkey", "GRE": "greece", "SCO": "scotland", "AUT": "austria",
    "SUI": "switzerland", "CZE": "czech-republic", "DEN": "denmark",
    "NOR": "norway", "SWE": "sweden", "UKR": "ukraine", "CRO": "croatia",
    "SRB": "serbia", "POL": "poland", "CYP": "cyprus", "AZE": "azerbaijan",
    "SVK": "slovakia", "SVN": "slovenia", "HUN": "hungary", "ROU": "romania",
    "BUL": "bulgaria", "ISL": "iceland", "MDA": "moldova", "ARM": "armenia",
    "GEO": "georgia", "IRL": "ireland", "NIR": "northern-ireland",
    "WAL": "wales", "LUX": "luxembourg", "MLT": "malta", "FRO": "faroe-islands",
    "FIN": "finland", "EST": "estonia", "BLR": "belarus", "ALB": "albania",
    "BIH": "bosnia-n-herzegovina", "KOS": "kosovo", "MNE": "montenegro",
    "MKD": "macedonia", "GIB": "gibraltar", "AND": "andorra", "LVA": "latvija",
    "LTU": "lithuania", "MCO": "monaco", "LIE": "liechtenstein",
    "SMR": "san-marino", "RUS": "russia",
}

RE_CABECERA = re.compile(r"^([^|=\s][^,|]*?)\s*(?:,.*)?$")


def normalizar(nombre: str) -> str:
    """Clave de comparacion: sin acentos, sin puntuacion, en minusculas.

    `FC Bayern München`, `Bayern Munchen` y `bayern munchen` colapsan a la misma
    clave, que es lo que permite casar fuentes que escriben distinto.
    """
    sin_acentos = "".join(
        c for c in unicodedata.normalize("NFD", nombre)
        if unicodedata.category(c) != "Mn"
    )
    sin_acentos = sin_acentos.replace("ø", "o").replace("Ø", "O")
    sin_acentos = sin_acentos.replace("ł", "l").replace("Ł", "L")
    limpio = re.sub(r"\[[a-z]{2}\]", "", sin_acentos)          # marcas de idioma
    limpio = re.sub(r"[^\w\s]", " ", limpio)
    return " ".join(limpio.lower().split())


def descargar_clubes(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CLUBS_ZIP.exists() and CLUBS_ZIP.stat().st_size > 0 and not force:
        return CLUBS_ZIP
    resp = requests.get(CLUBS_URL, timeout=120)
    resp.raise_for_status()
    CLUBS_ZIP.write_bytes(resp.content)
    return CLUBS_ZIP


def _parsear_pais(texto: str) -> dict[str, str]:
    """Devuelve {alias normalizado: nombre canonico} para un fichero de pais."""
    alias: dict[str, str] = {}
    canonico = None
    for linea in texto.splitlines():
        if not linea.strip() or linea.lstrip().startswith(("=", "#")):
            continue

        if linea.startswith(("ii)", "  ii)")) or linea.lstrip().startswith("ii)"):
            canonico = None          # filial: se ignora hasta el siguiente club
            continue

        if linea.lstrip().startswith("|"):
            if canonico is None:
                continue
            for parte in linea.lstrip().lstrip("|").split("|"):
                clave = normalizar(parte)
                if clave:
                    alias.setdefault(clave, canonico)
            continue

        if linea[0].isspace():
            continue                 # direccion del estadio u otra continuacion

        m = RE_CABECERA.match(linea.split("###")[0].strip())
        if m:
            canonico = m.group(1).strip()
            alias.setdefault(normalizar(canonico), canonico)
    return alias


def cargar_alias(paises: tuple[str, ...] | None = None, force: bool = False) -> dict[str, str]:
    """Construye el diccionario {alias normalizado -> nombre canonico}.

    Con `paises` (codigos UEFA) se limita a esos; por defecto carga toda Europa.
    """
    zip_path = descargar_clubes(force)
    carpetas = ({PAIS_A_CARPETA[p] for p in paises if p in PAIS_A_CARPETA}
                if paises else set(PAIS_A_CARPETA.values()))

    alias: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        for nombre in z.namelist():
            if not nombre.endswith(".clubs.txt") or "/europe/" not in nombre:
                continue
            carpeta = nombre.split("/europe/")[1].split("/")[0]
            if carpeta not in carpetas:
                continue
            texto = z.read(nombre).decode("utf-8", errors="replace")
            for clave, canon in _parsear_pais(texto).items():
                alias.setdefault(clave, canon)
    return alias


class ResolverClubes:
    """Traduce nombres de cualquier fuente al nombre canonico del proyecto."""

    def __init__(self, paises: tuple[str, ...] | None = None):
        self.alias = cargar_alias(paises)
        self.sin_resolver: set[str] = set()

    def resolver(self, nombre: str) -> str:
        """Nombre canonico, o el original si la fuente no esta en el catalogo.

        No falla ante un desconocido: en el pool europeo hay clubes de ligas
        menores que ningun catalogo cubre del todo, y perder uno es preferible
        a no poder simular. Los fallos quedan en `sin_resolver` para revision.
        """
        clave = normalizar(nombre)
        canonico = self.alias.get(clave)
        if canonico is None:
            self.sin_resolver.add(nombre)
            return nombre.strip()
        return canonico


CLAVE_FUENTE = "club-key"


class ResolverEquipos:
    """Traduce un nombre de club, venga de donde venga, al `team_id` del proyecto.

    Se apoya en `ResolverClubes` para obtener una clave comun entre fuentes y la
    guarda en `team_aliases`, de modo que la segunda vez que aparezca un nombre
    ya no haga falta recalcular nada.

    Los equipos espanoles conservan el nombre curado del proyecto (`Atletico de
    Madrid`, no `Atletico Madrid`): son los que se muestran en la tabla y en el
    JSON, y renombrarlos romperia el contrato con el frontend. La clave comun
    solo sirve de puente entre fuentes, nunca pisa el nombre visible.
    """

    def __init__(self, conn, paises: tuple[str, ...] | None = None):
        from ..db import get_or_create_team, register_alias, resolve_alias

        self.conn = conn
        self._get_or_create = get_or_create_team
        self._register = register_alias
        self._resolve = resolve_alias
        self.clubes = ResolverClubes(paises)
        self.colisiones: dict[str, list[str]] = {}
        self._indexar_existentes()

    def _indexar_existentes(self) -> None:
        """Registra la clave comun de cada equipo que ya esta en la base de datos."""
        vistos: dict[str, str] = {}
        for team_id, nombre in self.conn.execute("SELECT team_id, name FROM teams").fetchall():
            clave = self.clubes.resolver(nombre)
            if clave in vistos and vistos[clave] != nombre:
                self.colisiones.setdefault(clave, [vistos[clave]]).append(nombre)
                continue
            vistos[clave] = nombre
            self._register(self.conn, clave, CLAVE_FUENTE, team_id)

    def resolver(self, nombre: str, pais: str = "EUR") -> int:
        team_id = self.resolver_existente(nombre)
        if team_id is not None:
            return team_id

        clave = self.clubes.resolver(nombre)
        team_id = self._get_or_create(self.conn, clave, pais)
        self._register(self.conn, clave, CLAVE_FUENTE, team_id)
        return team_id

    def resolver_existente(self, nombre: str) -> int | None:
        """Igual que `resolver`, pero sin crear nada si no lo encuentra.

        Lo usa quien solo viene a actualizar filas existentes, como el modulo de
        horarios: un modulo que solo cambia fechas no tiene por que poder crear
        equipos, y si lo hace acaba partiendo en dos la ficha de alguien por una
        diferencia de grafia (`Racing Santander` frente a `Racing de Santander`).
        """
        clave = self.clubes.resolver(nombre)
        existente = self._resolve(self.conn, clave, CLAVE_FUENTE)
        if existente:
            return existente
        # Segunda oportunidad: comparar la forma normalizada del nombre con la
        # de los equipos ya registrados, por si el catalogo canoniza distinto.
        objetivo = normalizar(nombre)
        for team_id, guardado in self.conn.execute(
                "SELECT team_id, name FROM teams").fetchall():
            if normalizar(guardado) == objetivo:
                self._register(self.conn, clave, CLAVE_FUENTE, team_id)
                return team_id
        return None
