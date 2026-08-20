"""Motor Monte Carlo de la liga regular.

Cada simulacion juega los partidos pendientes muestreando el marcador de la
distribucion conjunta de Dixon-Coles (muestreo exacto por CDF, no Poisson
independiente), acumula la clasificacion y aplica los criterios de desempate
de LaLiga. Todo esta vectorizado sobre las `n_sims` simulaciones: el bucle en
Python recorre partidos (380 como mucho), nunca simulaciones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SimConfig
from ..model.dixon_coles import DixonColesFit


@dataclass
class LeagueSimResult:
    """Resultado crudo de la simulacion, en arrays (n_sims x n_equipos)."""

    team_ids: list[int]
    positions: np.ndarray      # posicion final 1..N
    points: np.ndarray
    goal_diff: np.ndarray
    goals_for: np.ndarray
    n_sims: int

    def position_probabilities(self) -> np.ndarray:
        """Matriz (n_equipos x n_equipos): P(equipo i acabe en la posicion j+1)."""
        n = len(self.team_ids)
        counts = np.zeros((n, n))
        for j in range(n):
            counts[:, j] = (self.positions == j + 1).sum(axis=0)
        return counts / self.n_sims

    def prob_position_range(self, lo: int, hi: int) -> np.ndarray:
        """P(posicion final entre lo y hi, ambos incluidos)."""
        return ((self.positions >= lo) & (self.positions <= hi)).mean(axis=0)

    def summary(self, names: dict[int, str], cfg: SimConfig) -> pd.DataFrame:
        """Vista compacta y legible del resultado (una fila por equipo)."""
        n = len(self.team_ids)
        return pd.DataFrame({
            "team": [names.get(t, str(t)) for t in self.team_ids],
            "exp_points": self.points.mean(axis=0),
            "exp_position": self.positions.mean(axis=0),
            "p_title": (self.positions == 1).mean(axis=0),
            "p_ucl": self.prob_position_range(1, cfg.ucl_slots),
            "p_relegation": self.prob_position_range(n - cfg.relegation_slots + 1, n),
        }).sort_values("exp_position").reset_index(drop=True)


def build_score_cdfs(
    fit: DixonColesFit,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    rate_shift_home: np.ndarray | None = None,
    rate_shift_away: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """CDF acumulada del marcador para cada partido pendiente.

    Devuelve una matriz (n_partidos x (G+1)^2) donde cada fila es la CDF de la
    distribucion conjunta aplanada, y G = max_goals.

    `rate_shift_*` son desplazamientos del logaritmo de los goles esperados, tal
    y como los producen los modificadores cualitativos. Van aqui y no en el
    ajuste porque dependen del partido concreto (cuanto descanso hubo, quien
    estaba lesionado ese dia), no del equipo en abstracto.
    """
    max_goals = fit.config.max_goals
    goals = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, max_goals + 1))]))

    lam_h, lam_a = fit.rates_batch(home_idx, away_idx)
    if rate_shift_home is not None:
        lam_h = lam_h * np.exp(rate_shift_home)
    if rate_shift_away is not None:
        lam_a = lam_a * np.exp(rate_shift_away)
    p_home = np.exp(goals[None, :] * np.log(lam_h)[:, None] - lam_h[:, None] - log_fact[None, :])
    p_away = np.exp(goals[None, :] * np.log(lam_a)[:, None] - lam_a[:, None] - log_fact[None, :])
    joint = p_home[:, :, None] * p_away[:, None, :]

    rho = fit.rho
    joint[:, 0, 0] *= 1.0 - lam_h * lam_a * rho
    joint[:, 0, 1] *= 1.0 + lam_h * rho
    joint[:, 1, 0] *= 1.0 + lam_a * rho
    joint[:, 1, 1] *= 1.0 - rho
    joint = np.clip(joint, 1e-15, None)
    joint /= joint.sum(axis=(1, 2), keepdims=True)

    flat = joint.reshape(len(lam_h), -1)
    return np.cumsum(flat, axis=1), max_goals + 1


def simulate_league(
    fit: DixonColesFit,
    fixtures: pd.DataFrame,
    played: pd.DataFrame | None = None,
    teams: list[int] | None = None,
    config: SimConfig | None = None,
    rng: np.random.Generator | None = None,
    exact_h2h_tiebreak: bool = True,
    rate_shift_home: np.ndarray | None = None,
    rate_shift_away: np.ndarray | None = None,
) -> LeagueSimResult:
    """Simula `n_sims` veces los partidos de `fixtures`.

    `played` son los partidos ya disputados de la temporada: entran en la tabla
    con su resultado real, sin aleatoriedad. Pasando `played=None` se simula la
    temporada entera desde cero (modo pre-temporada).
    """
    cfg = config or SimConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    n_sims = cfg.n_sims

    team_ids = teams or sorted(set(fixtures["home_team_id"]) | set(fixtures["away_team_id"]))
    idx = {t: i for i, t in enumerate(team_ids)}
    n_teams = len(team_ids)

    points = np.zeros((n_sims, n_teams), dtype=np.int16)
    gf = np.zeros((n_sims, n_teams), dtype=np.int16)
    ga = np.zeros((n_sims, n_teams), dtype=np.int16)
    h2h_pts = h2h_gd = None
    if exact_h2h_tiebreak:
        h2h_pts = np.zeros((n_sims, n_teams, n_teams), dtype=np.int8)
        h2h_gd = np.zeros((n_sims, n_teams, n_teams), dtype=np.int8)

    def account(h: int, a: int, hg, ag) -> None:
        """Suma un partido (resultado escalar o array por simulacion) a la tabla."""
        home_win = hg > ag
        away_win = ag > hg
        draw = ~(home_win | away_win)
        hp = 3 * home_win + draw
        ap = 3 * away_win + draw
        points[:, h] += hp
        points[:, a] += ap
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg
        if h2h_pts is not None:
            h2h_pts[:, h, a] += hp
            h2h_pts[:, a, h] += ap
            h2h_gd[:, h, a] += hg - ag
            h2h_gd[:, a, h] += ag - hg

    # --- partidos ya jugados: resultado fijo en todas las simulaciones ---
    if played is not None and len(played):
        for m in played.itertuples(index=False):
            if pd.isna(m.home_goals):
                continue
            account(idx[int(m.home_team_id)], idx[int(m.away_team_id)],
                    np.int16(m.home_goals), np.int16(m.away_goals))

    # --- partidos pendientes: marcador muestreado ---
    if len(fixtures):
        home_idx = fixtures["home_team_id"].map(lambda t: fit.index_of(t)).to_numpy()
        away_idx = fixtures["away_team_id"].map(lambda t: fit.index_of(t)).to_numpy()
        cdfs, n_cols = build_score_cdfs(fit, home_idx, away_idx,
                                        rate_shift_home, rate_shift_away)

        table_home = fixtures["home_team_id"].map(idx).to_numpy()
        table_away = fixtures["away_team_id"].map(idx).to_numpy()

        draws = rng.random((len(fixtures), n_sims))
        for m in range(len(fixtures)):
            cell = np.searchsorted(cdfs[m], draws[m])
            np.clip(cell, 0, n_cols * n_cols - 1, out=cell)
            hg = (cell // n_cols).astype(np.int16)
            ag = (cell % n_cols).astype(np.int16)
            account(int(table_home[m]), int(table_away[m]), hg, ag)

    goal_diff = gf - ga
    positions = _rank(points, goal_diff, gf, h2h_pts, h2h_gd)
    return LeagueSimResult(
        team_ids=team_ids, positions=positions, points=points,
        goal_diff=goal_diff, goals_for=gf, n_sims=n_sims,
    )


def _rank(
    points: np.ndarray,
    goal_diff: np.ndarray,
    goals_for: np.ndarray,
    h2h_pts: np.ndarray | None,
    h2h_gd: np.ndarray | None,
    max_group: int = 5,
) -> np.ndarray:
    """Posiciones finales aplicando los desempates de LaLiga.

    Orden real: puntos > mini-liga entre los empatados (puntos y luego
    diferencia de goles solo de esos partidos) > diferencia de goles general >
    goles a favor.

    La mini-liga se resuelve de forma exacta para grupos de hasta `max_group`
    equipos, que cubre cualquier empate realista. No es un detalle cosmetico:
    en 2025-26 Levante, Osasuna y Mallorca acabaron los tres con 42 puntos y el
    criterio decidio quien descendia, no un puesto intermedio sin consecuencias.
    """
    n_sims, n_teams = points.shape
    # Clave compuesta: puntos, luego diferencia de goles, luego goles a favor.
    key = (points.astype(np.int64) * 1_000_000
           + (goal_diff.astype(np.int64) + 500) * 1_000
           + goals_for.astype(np.int64))
    order = np.argsort(-key, axis=1, kind="stable")

    if h2h_pts is not None:
        order = _apply_h2h(order, points, goal_diff, goals_for, h2h_pts, h2h_gd, max_group)

    positions = np.empty((n_sims, n_teams), dtype=np.int16)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(1, n_teams + 1)[None, :]
    return positions


def _apply_h2h(order, points, goal_diff, goals_for, h2h_pts, h2h_gd, max_group) -> np.ndarray:
    """Reordena cada grupo de equipos empatados a puntos por su mini-liga."""
    n_sims, n_teams = points.shape
    order = order.copy()
    puntos_ordenados = np.take_along_axis(points, order, axis=1)

    for inicio in range(n_teams - 1):
        for tam in range(min(max_group, n_teams - inicio), 1, -1):
            fin = inicio + tam
            # Grupo aislado: los `tam` empatan entre si y difieren de los vecinos.
            empatados = np.ones(n_sims, dtype=bool)
            for j in range(1, tam):
                empatados &= puntos_ordenados[:, inicio] == puntos_ordenados[:, inicio + j]
            if inicio > 0:
                empatados &= puntos_ordenados[:, inicio - 1] != puntos_ordenados[:, inicio]
            if fin < n_teams:
                empatados &= puntos_ordenados[:, fin] != puntos_ordenados[:, inicio]
            if not empatados.any():
                continue

            idx = order[empatados, inicio:fin]                 # (m, tam)
            filas = np.nonzero(empatados)[0][:, None, None]
            # Mini-liga: solo cuentan los partidos entre los propios empatados.
            # La diagonal de h2h vale 0, asi que no contamina la suma.
            mini_pts = h2h_pts[filas, idx[:, :, None], idx[:, None, :]].sum(axis=2)
            mini_gd = h2h_gd[filas, idx[:, :, None], idx[:, None, :]].sum(axis=2)

            filas_planas = np.nonzero(empatados)[0][:, None]
            clave = (mini_pts.astype(np.int64) * 10 ** 12
                     + (mini_gd.astype(np.int64) + 500) * 10 ** 9
                     + (goal_diff[filas_planas, idx].astype(np.int64) + 500) * 10 ** 5
                     + goals_for[filas_planas, idx].astype(np.int64))
            nuevo = np.argsort(-clave, axis=1, kind="stable")
            order[empatados, inicio:fin] = np.take_along_axis(idx, nuevo, axis=1)

    return order


def split_season(matches: pd.DataFrame, cutoff: pd.Timestamp | None):
    """Divide los partidos de una temporada en (jugados, pendientes) segun el corte."""
    if cutoff is None:
        return matches.iloc[0:0], matches
    played = matches[(matches["match_date"] < cutoff) & matches["home_goals"].notna()]
    pending = matches[~matches.index.isin(played.index)]
    return played, pending
