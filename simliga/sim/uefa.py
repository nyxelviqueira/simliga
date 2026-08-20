"""Motor Monte Carlo de las competiciones UEFA (formato desde 2024-25).

Estructura de la competicion:

1. **Liguilla unica**: 36 equipos en una sola clasificacion. Cada uno juega 8
   partidos contra rivales distintos (6 en la Conference). Del 1 al 8 pasan
   directos a octavos; del 9 al 24, a un playoff a doble partido; del 25 al 36
   quedan eliminados.
2. **Playoff y eliminatorias**: a doble partido, con prorroga y penaltis si la
   eliminatoria acaba empatada (la regla del gol fuera ya no existe).

La liguilla reutiliza el motor de liga (`sim.league`), que ya sabe simular una
clasificacion cualquiera; lo unico que cambia es el desempate, porque la UEFA no
usa el enfrentamiento directo sino diferencia de goles y goles marcados.

Las eliminatorias se simulan por simulacion: cada una de las `n_sims` temporadas
tiene su propia clasificacion y, por tanto, su propio cuadro. Para que siga
siendo rapido se precalculan las distribuciones de marcador de los 36x36
emparejamientos posibles (unas 219.000 celdas, nada) y despues cada simulacion
solo hace una busqueda en su fila.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..config import SimConfig
from ..model.dixon_coles import DixonColesFit
from .league import simulate_league

# Cuadro de la fase final. Cada entrada del playoff enfrenta a un par de cabezas
# de serie con un par de no clasificados directamente; dentro de cada par decide
# el sorteo, que aqui se modela como aleatorio.
PLAYOFF_PAIRINGS = [
    ((9, 10), (23, 24)),
    ((11, 12), (21, 22)),
    ((13, 14), (19, 20)),
    ((15, 16), (17, 18)),
]

# Octavos: cada cabeza de serie (par 1-2, 3-4, ...) recibe al ganador de un
# playoff concreto. El cuadro esta fijado de antemano, de modo que el 1 y el 2
# solo pueden cruzarse en la final.
R16_PAIRINGS = [
    ((1, 2), 3),      # el par de cabezas 1-2 recibe al ganador del playoff 4 (15/16 v 17/18)
    ((7, 8), 0),
    ((3, 4), 2),
    ((5, 6), 1),
]

STAGE_ORDER = ["playoff", "R16", "QF", "SF", "F"]


@dataclass
class UefaSimResult:
    """Probabilidades por equipo de alcanzar cada fase."""

    team_ids: list[int]
    league_positions: np.ndarray        # (n_sims, n_equipos)
    league_points: np.ndarray
    reached: dict[str, np.ndarray]      # fase -> (n_sims, n_equipos) booleano
    n_sims: int

    def stage_probabilities(self) -> dict[str, np.ndarray]:
        return {fase: alcanzo.mean(axis=0) for fase, alcanzo in self.reached.items()}

    def league_phase_outcomes(self, direct_slots: int = 8, playoff_slots: int = 16) -> dict:
        """Desenlaces de la liguilla, que son los que miden exito de verdad.

        `playoff` no sirve para eso: caer en el playoff es quedar entre el 9 y el
        24, asi que un equipo fuerte tiene poca probabilidad de "llegar" ahi
        precisamente porque aspira a pasar directo. Lo que hay que mirar es
        `direct_to_r16`.
        """
        pos = self.league_positions
        corte = direct_slots + playoff_slots
        return {
            "direct_to_r16": (pos <= direct_slots).mean(axis=0),
            "playoff": ((pos > direct_slots) & (pos <= corte)).mean(axis=0),
            "eliminated_in_league_phase": (pos > corte).mean(axis=0),
        }

    def summary(self, names: dict[int, str]) -> pd.DataFrame:
        datos = {"team": [names.get(t, str(t)) for t in self.team_ids],
                 "exp_points": self.league_points.mean(axis=0),
                 "exp_position": self.league_positions.mean(axis=0)}
        datos.update(self.league_phase_outcomes())
        datos.update({k: v for k, v in self.stage_probabilities().items() if k != "playoff"})
        return pd.DataFrame(datos).sort_values("exp_position").reset_index(drop=True)


def build_all_cdfs(fit: DixonColesFit) -> tuple[np.ndarray, int]:
    """CDF del marcador para los N x N emparejamientos posibles.

    Precalcularlas permite que las eliminatorias, donde cada simulacion tiene un
    cruce distinto, se muestreen con una sola indexacion en vez de reconstruir
    la distribucion partido a partido.
    """
    n = len(fit.team_ids)
    max_goals = fit.config.max_goals
    goles = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, max_goals + 1))]))

    idx = np.arange(n)
    hi, ai = np.meshgrid(idx, idx, indexing="ij")
    lam_h, lam_a = fit.rates_batch(hi.ravel(), ai.ravel())

    p_h = np.exp(goles[None, :] * np.log(lam_h)[:, None] - lam_h[:, None] - log_fact[None, :])
    p_a = np.exp(goles[None, :] * np.log(lam_a)[:, None] - lam_a[:, None] - log_fact[None, :])
    joint = p_h[:, :, None] * p_a[:, None, :]

    rho = fit.rho
    joint[:, 0, 0] *= 1.0 - lam_h * lam_a * rho
    joint[:, 0, 1] *= 1.0 + lam_h * rho
    joint[:, 1, 0] *= 1.0 + lam_a * rho
    joint[:, 1, 1] *= 1.0 - rho
    joint = np.clip(joint, 1e-15, None)
    joint /= joint.sum(axis=(1, 2), keepdims=True)

    plano = joint.reshape(n * n, -1)
    return np.cumsum(plano, axis=1).reshape(n, n, -1), max_goals + 1


def sample_matches(
    cdfs: np.ndarray, n_cols: int, home: np.ndarray, away: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Muestrea un marcador por simulacion, con cruces distintos en cada una."""
    u = rng.random(len(home))
    filas = cdfs[home, away]                       # (n_sims, celdas)
    celda = (filas < u[:, None]).sum(axis=1)
    np.clip(celda, 0, n_cols * n_cols - 1, out=celda)
    return celda // n_cols, celda % n_cols


def simulate_tie(
    cdfs: np.ndarray,
    n_cols: int,
    fit: DixonColesFit,
    team_a: np.ndarray,
    team_b: np.ndarray,
    rng: np.random.Generator,
    single_leg: bool = False,
) -> np.ndarray:
    """Resuelve una eliminatoria. Devuelve el equipo que pasa.

    `team_a` juega la ida en casa. Si el global queda empatado hay prorroga
    (media hora, modelada como un tercio de los goles esperados de un partido) y,
    si persiste, penaltis, que se resuelven a cara o cruz: por debajo de los
    datos que maneja el modelo, la tanda es esencialmente azar.

    Con `single_leg` se resuelve a partido unico, como la final, que se juega en
    campo neutral: se simulan las dos localias y se promedian, que es la forma
    barata de quitar la ventaja de campo sin reajustar el modelo.
    """
    if single_leg:
        goles_a1, goles_b1 = sample_matches(cdfs, n_cols, team_a, team_b, rng)
        goles_b2, goles_a2 = sample_matches(cdfs, n_cols, team_b, team_a, rng)
        neutral = rng.random(len(team_a)) < 0.5
        global_a = np.where(neutral, goles_a1, goles_a2)
        global_b = np.where(neutral, goles_b1, goles_b2)
    else:
        ida_h, ida_a = sample_matches(cdfs, n_cols, team_a, team_b, rng)
        vuelta_h, vuelta_a = sample_matches(cdfs, n_cols, team_b, team_a, rng)
        global_a = ida_h + vuelta_a
        global_b = ida_a + vuelta_h

    empate = global_a == global_b
    if empate.any():
        # Prorroga: 30 minutos sobre 90, sin ventaja de campo (se juega en el
        # estadio del equipo B, que ya la tuvo en el partido de vuelta).
        lam_a_ext, lam_b_ext = fit.rates_batch(team_b[empate], team_a[empate])
        goles_b = rng.poisson(lam_a_ext / 3.0)
        goles_a = rng.poisson(lam_b_ext / 3.0)
        global_a = global_a.copy()
        global_b = global_b.copy()
        global_a[empate] += goles_a
        global_b[empate] += goles_b

        penaltis = empate & (global_a == global_b)
        if penaltis.any():
            gana_a = rng.random(penaltis.sum()) < 0.5
            global_a[penaltis] += gana_a
            global_b[penaltis] += ~gana_a

    return np.where(global_a > global_b, team_a, team_b)


def _shuffle_pair(first: np.ndarray, second: np.ndarray, rng: np.random.Generator):
    """Sortea cual de los dos equipos de un par ocupa cada lado del cuadro."""
    intercambia = rng.random(len(first)) < 0.5
    a = np.where(intercambia, second, first)
    b = np.where(intercambia, first, second)
    return a, b


def simulate_uefa(
    fit: DixonColesFit,
    fixtures: pd.DataFrame,
    played: pd.DataFrame | None,
    teams: list[int],
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
) -> UefaSimResult:
    """Simula una competicion UEFA completa: liguilla, playoff y eliminatorias."""
    cfg = config or SimConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    n_sims = cfg.n_sims
    cdfs, n_cols = build_all_cdfs(fit)

    # --- 1. Liguilla: sin enfrentamiento directo, la UEFA desempata por goles ---
    liga = simulate_league(fit, fixtures, played=played, teams=teams, config=cfg,
                           rng=rng, exact_h2h_tiebreak=False)
    posiciones = liga.positions
    n_teams = len(teams)

    # `seed[s, k]` = indice del equipo que acabo en la posicion k+1 de la simulacion s.
    # argsort sobre las posiciones invierte el mapa equipo->puesto.
    seed = np.argsort(posiciones, axis=1, kind="stable")

    reached = {fase: np.zeros((n_sims, n_teams), dtype=bool) for fase in STAGE_ORDER}
    filas = np.arange(n_sims)

    def marcar(fase: str, equipos: np.ndarray) -> None:
        reached[fase][filas, equipos] = True

    # --- 2. Playoff: del 9 al 24, emparejados por pares de posiciones ---
    ganadores_playoff = []
    for (alto, bajo) in PLAYOFF_PAIRINGS:
        fuerte_a, fuerte_b = _shuffle_pair(seed[:, alto[0] - 1], seed[:, alto[1] - 1], rng)
        debil_a, debil_b = _shuffle_pair(seed[:, bajo[0] - 1], seed[:, bajo[1] - 1], rng)
        for cabeza, rival in ((fuerte_a, debil_a), (fuerte_b, debil_b)):
            marcar("playoff", cabeza)
            marcar("playoff", rival)
            # El peor clasificado juega la ida en casa y la vuelta fuera.
            ganadores_playoff.append(
                simulate_tie(cdfs, n_cols, fit, rival, cabeza, rng))

    # --- 3. Octavos: los ocho primeros esperan al ganador de su playoff ---
    ties_r16 = []
    for (par, indice_playoff) in R16_PAIRINGS:
        cabeza_a, cabeza_b = _shuffle_pair(seed[:, par[0] - 1], seed[:, par[1] - 1], rng)
        rival_a = ganadores_playoff[indice_playoff * 2]
        rival_b = ganadores_playoff[indice_playoff * 2 + 1]
        ties_r16.append((cabeza_a, rival_a))
        ties_r16.append((cabeza_b, rival_b))

    for cabeza, rival in ties_r16:
        marcar("R16", cabeza)
        marcar("R16", rival)

    ronda = [simulate_tie(cdfs, n_cols, fit, rival, cabeza, rng)
             for cabeza, rival in ties_r16]

    # --- 4. Cuartos, semifinales y final ---
    for fase in ("QF", "SF"):
        for equipo in ronda:
            marcar(fase, equipo)
        ronda = [
            simulate_tie(cdfs, n_cols, fit, ronda[i], ronda[i + 1], rng)
            for i in range(0, len(ronda), 2)
        ]

    for equipo in ronda:
        marcar("F", equipo)
    campeon = simulate_tie(cdfs, n_cols, fit, ronda[0], ronda[1], rng, single_leg=True)
    reached["winner"] = np.zeros((n_sims, n_teams), dtype=bool)
    reached["winner"][filas, campeon] = True

    return UefaSimResult(
        team_ids=teams, league_positions=posiciones, league_points=liga.points,
        reached=reached, n_sims=n_sims,
    )


# ---------------------------------------------------------------- sorteo
def draw_league_phase(
    team_ids: list[int],
    ratings: dict[int, float],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sortea una liguilla completa: 8 rivales DISTINTOS por equipo.

    Reproduce la estructura del formato: los 36 se reparten en cuatro bombos de
    nueve por rating, y cada equipo juega dos partidos contra equipos de cada
    bombo, uno en casa y otro fuera.

    - Contra su propio bombo, una permutacion sin puntos fijos ni ciclos de dos:
      cada equipo recibe a uno y visita a otro, y esos dos son distintos entre
      si. Sin excluir los ciclos de dos, A recibiria a B y B a A, y ambos se
      quedarian con un rival repetido en lugar de dos.
    - Contra cada otro bombo, dos emparejamientos que no coinciden en ninguna
      posicion, uno por sentido de la localia, para no repetir cruce.

    No se aplica la restriccion de no cruzar equipos del mismo pais: afecta poco
    a la probabilidad de avanzar y mucho a la complejidad del sorteo.
    """
    ordenados = sorted(team_ids, key=lambda t: -ratings.get(t, 1500.0))
    n_bombos = 4
    tam = len(ordenados) // n_bombos
    bombos = [ordenados[i * tam:(i + 1) * tam] for i in range(n_bombos)]

    partidos: list[tuple[int, int]] = []
    for i, bombo in enumerate(bombos):
        recibe = _permutacion_sin_ciclos_cortos(bombo, rng)
        partidos += list(zip(bombo, recibe))

        for otro in bombos[i + 1:]:
            primero = _mezclado(otro, rng)
            segundo = _mezclado_distinto(otro, primero, rng)
            partidos += list(zip(bombo, primero))            # local el bombo i
            partidos += list(zip(segundo, bombo))            # local el otro bombo

    return partidos


def _mezclado(elementos: list[int], rng: np.random.Generator) -> list[int]:
    mezclado = list(elementos)
    rng.shuffle(mezclado)
    return mezclado


def _mezclado_distinto(
    elementos: list[int], evitar: list[int], rng: np.random.Generator
) -> list[int]:
    """Otra ordenacion que no coincide con `evitar` en ninguna posicion."""
    while True:
        candidato = _mezclado(elementos, rng)
        if all(a != b for a, b in zip(candidato, evitar)):
            return candidato


def _permutacion_sin_ciclos_cortos(
    elementos: list[int], rng: np.random.Generator
) -> list[int]:
    """Permutacion sin puntos fijos ni ciclos de longitud dos."""
    posicion = {equipo: i for i, equipo in enumerate(elementos)}
    while True:
        candidato = _mezclado(elementos, rng)
        if any(a == b for a, b in zip(elementos, candidato)):
            continue
        if all(candidato[posicion[destino]] != origen
               for origen, destino in zip(elementos, candidato)):
            return candidato


def draw_to_fixtures(
    partidos: list[tuple[int, int]], season: str, competition: str, start: pd.Timestamp
) -> pd.DataFrame:
    """Convierte un sorteo en el DataFrame de partidos que espera el motor."""
    return pd.DataFrame([
        {"match_id": -(i + 1), "competition": competition, "season": season,
         "stage": "league_phase",
         "match_date": start + pd.Timedelta(days=7 * (i // (len(partidos) // 8 or 1))),
         "home_team_id": h, "away_team_id": a,
         "home_goals": None, "away_goals": None, "status": "scheduled",
         "home_team": str(h), "away_team": str(a)}
        for i, (h, a) in enumerate(partidos)
    ])


def simulate_uefa_over_draws(
    fit: DixonColesFit,
    teams: list[int],
    ratings: dict[int, float],
    season: str,
    competition: str,
    start: pd.Timestamp,
    config: SimConfig,
    n_draws: int = 25,
) -> UefaSimResult:
    """Simula el torneo promediando sobre varios sorteos distintos.

    Con el sorteo sin publicar, la suerte del bombo es una fuente de
    incertidumbre real: un mismo equipo puede quedar con un grupo asequible o
    con tres cocos. Simular un unico sorteo daria una respuesta que depende de
    cual salio. Se reparten las simulaciones entre `n_draws` sorteos y se
    juntan los resultados.
    """
    rng = np.random.default_rng(config.seed)
    por_sorteo = max(config.n_sims // n_draws, 1)

    acumulado: dict[str, list] = {}
    posiciones, puntos = [], []
    for _ in range(n_draws):
        fixtures = draw_to_fixtures(
            draw_league_phase(teams, ratings, rng), season, competition, start)
        cfg_parcial = replace(config, n_sims=por_sorteo)
        parcial = simulate_uefa(fit, fixtures, played=None, teams=teams,
                                config=cfg_parcial, rng=rng)
        posiciones.append(parcial.league_positions)
        puntos.append(parcial.league_points)
        for fase, alcanzo in parcial.reached.items():
            acumulado.setdefault(fase, []).append(alcanzo)

    return UefaSimResult(
        team_ids=teams,
        league_positions=np.vstack(posiciones),
        league_points=np.vstack(puntos),
        reached={fase: np.vstack(trozos) for fase, trozos in acumulado.items()},
        n_sims=por_sorteo * n_draws,
    )
