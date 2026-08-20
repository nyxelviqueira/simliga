"""Configuracion central del simulador.

Todo parametro ajustable vive aqui o en un JSON de overrides, nunca hardcodeado
en la logica. `load_config()` mezcla los defaults con un fichero opcional.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# `SIMLIGA_DATA` reubica la carpeta de datos. Solo hace falta al desplegarlo en
# un hosting, donde el disco que sobrevive a un reinicio esta montado en otro
# sitio: sin esto, cada reinicio volveria a la base de datos de la imagen y se
# perderia todo lo descargado desde entonces.
DATA_DIR = Path(os.environ.get("SIMLIGA_DATA") or ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "out"
DB_PATH = DATA_DIR / "simliga.sqlite"

# Competiciones soportadas por la capa de datos.
COMP_LALIGA = "ESP1"
COMP_SEGUNDA = "ESP2"

# Competiciones UEFA.
COMP_UCL = "UCL"
COMP_UEL = "UEL"
COMP_UECL = "UECL"
UEFA_COMPETITIONS = (COMP_UCL, COMP_UEL, COMP_UECL)

# Ligas domesticas europeas: alimentan un unico pool de Elo. A diferencia de
# Primera y Segunda, estas si se cruzan entre si en competicion continental,
# asi que el pool se ancla solo y no necesita correccion de escala.
EUROPEAN_LEAGUES = (
    "ENG1", "GER1", "ITA1", "FRA1", "NED1", "POR1", "BEL1", "TUR1", "GRE1", "SCO1",
)


@dataclass
class EloConfig:
    """Elo dinamico tipo ClubElo, actualizado partido a partido."""

    k_factor: float = 10.0              # elegido por rejilla sobre 13.000 partidos
    home_advantage: float = 70.0        # en puntos Elo, estimado desde datos
    initial_rating: float = 1500.0
    # Rating de arranque para equipos vistos por primera vez, por division.
    initial_by_competition: dict = field(
        default_factory=lambda: {COMP_LALIGA: 1500.0, COMP_SEGUNDA: 1380.0}
    )
    # Regresion parcial a la media entre temporadas: R' = mean + phi*(R - mean)
    season_regression: float = 0.85
    # Multiplicador por diferencia de goles (estilo World Football Elo).
    use_goal_difference: bool = True
    # Puntos Elo que se restan a un equipo que cambia de division. Primera y
    # Segunda forman un unico pool de rating pero casi no juegan entre si, asi
    # que sus escalas derivan. Sin esta correccion el modelo sobrestimaba a los
    # recien ascendidos en +5,1 puntos por temporada (45 casos, 15 temporadas).
    # Se mantiene durante toda la temporada del ascenso y desaparece a la
    # siguiente, cuando el equipo ya tiene un año de Primera en su rating.
    promotion_penalty: float = 50.0


@dataclass
class DixonColesConfig:
    """Ajuste Poisson bivariante con correccion de resultados bajos."""

    # Vida media (en dias) del peso temporal: w = 0.5 ** (edad_dias / half_life)
    half_life_days: float = 365.0
    # Ventana maxima de historico usada en el ajuste.
    max_history_days: int = 1100
    # Fuerza del encogimiento de ataque/defensa hacia el prior derivado de Elo.
    elo_prior_weight: float = 12.0
    # Regularizacion L2 general sobre ataque/defensa (evita divergencias).
    l2_weight: float = 1.0
    # Cota inferior/superior de rho (correccion Dixon-Coles).
    rho_bounds: tuple = (-0.25, 0.25)
    max_goals: int = 12                 # rejilla de goles para el muestreo exacto


# 400 puntos Elo equivalen a una unidad de fuerza en el modelo de goles.
ELO_TO_STRENGTH = 1.0 / 400.0


@dataclass
class ModifierConfig:
    """Ajustes cualitativos. Todos desactivados por defecto, y no por prudencia:

    se midieron sobre 6.074 observaciones de ocho temporadas y ninguno alcanza
    significacion estadistica (ver `simliga/model/modifiers.py`). Las magnitudes
    que hay aqui son un punto de partida razonable para experimentar, no valores
    estimados de los datos.
    """

    # --- fatiga por calendario ---
    fatigue_enabled: bool = False
    fatigue_reference_days: float = 6.0      # descanso a partir del cual no penaliza
    fatigue_elo_per_day: float = 8.0         # penalizacion por cada dia que falte
    fatigue_max_deficit_days: float = 3.0    # tope, para que no se dispare
    fatigue_european_extra: float = 5.0      # extra si el partido anterior fue en Europa

    # --- motivacion situacional ---
    motivation_enabled: bool = False
    motivation_last_matchdays: int = 5
    motivation_elo_penalty: float = 25.0

    # --- ajustes de evento (tabla `team_adjustments`) ---
    coach_change_enabled: bool = False
    injuries_enabled: bool = False
    transfers_enabled: bool = False
    squad_depth_enabled: bool = False

    def enabled_event_kinds(self) -> tuple[str, ...]:
        activos = []
        if self.coach_change_enabled:
            activos.append("coach_change")
        if self.injuries_enabled:
            activos.append("injuries")
        if self.transfers_enabled:
            activos.append("transfers")
        if self.squad_depth_enabled:
            activos.append("squad_depth")
        return tuple(activos)

    def enabled_names(self) -> list[str]:
        """Identificadores de los modificadores activos, para el JSON de salida."""
        nombres = list(self.enabled_event_kinds())
        if self.fatigue_enabled:
            nombres.insert(0, "fatigue")
        if self.motivation_enabled:
            nombres.append("motivation")
        return nombres

    def any_enabled(self) -> bool:
        return bool(self.enabled_names())


@dataclass
class SimConfig:
    n_sims: int = 20000
    seed: int = 20262027
    # Reglas de clasificacion de LaLiga.
    points_win: int = 3
    points_draw: int = 1
    # Plazas europeas que reparte LA LIGA. La de Europa League del campeon de
    # Copa NO se cuenta aqui: va aparte, porque puede tocarle a un equipo que no
    # esta entre los de arriba (la Real Sociedad acabo 10a en 2025-26 y entro).
    #
    # La quinta de Champions es la que la UEFA da por rendimiento continental
    # (European Performance Spot) y no esta garantizada de un año para otro.
    # Espana la tuvo en 2025-26: el quinto, el Betis con 60 puntos, entro en
    # Champions, y todo el reparto de abajo bajo un escalon. Si algun año no se
    # consigue, hay que volver a poner 4 aqui.
    ucl_slots: int = 5
    uel_slots: int = 1
    uecl_slots: int = 1
    relegation_slots: int = 3


@dataclass
class Config:
    elo: EloConfig = field(default_factory=EloConfig)
    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    modifiers: ModifierConfig = field(default_factory=ModifierConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    """Carga la configuracion por defecto, opcionalmente pisada por un JSON."""
    cfg = Config()
    if path is None:
        return cfg
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    for section, values in overrides.items():
        target = getattr(cfg, section, None)
        if target is None:
            raise KeyError(f"Seccion de configuracion desconocida: {section}")
        for key, value in values.items():
            if not hasattr(target, key):
                raise KeyError(f"Parametro desconocido: {section}.{key}")
            setattr(target, key, value)
    return cfg
