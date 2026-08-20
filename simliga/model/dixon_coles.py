"""Modelo Poisson bivariante de Dixon-Coles (1997).

Para un partido entre local h y visitante a:

    log(lambda_local)     = mu + ataque[h] - defensa[a] + gamma
    log(lambda_visitante) = mu + ataque[a] - defensa[h]

Los goles son Poisson independientes salvo por la correccion `tau`, que
reajusta los cuatro marcadores bajos (0-0, 1-0, 0-1, 1-1), donde el Poisson
puro se queda corto respecto a lo que se observa.

Dos anadidos sobre el paper original:

1. Peso temporal exponencial: los partidos viejos pesan menos (media vida
   configurable), asi el ajuste sigue la forma reciente.
2. Prior derivado del Elo: ataque y defensa se encogen hacia el valor implicado
   por el rating Elo del equipo. Esto es lo que permite dar una fuerza sensata a
   un recien ascendido sin partidos en Primera, y estabiliza las primeras
   jornadas, cuando hay muy poca muestra de la temporada en curso.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import DixonColesConfig

# Escala del Elo a "unidades de fuerza": 400 puntos Elo = 1 unidad.
ELO_SCALE = 400.0


def tau(home_goals, away_goals, lam_home, lam_away, rho):
    """Correccion de Dixon-Coles para marcadores bajos (vectorizada)."""
    out = np.ones_like(lam_home, dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    out[m00] = 1.0 - lam_home[m00] * lam_away[m00] * rho
    out[m01] = 1.0 + lam_home[m01] * rho
    out[m10] = 1.0 + lam_away[m10] * rho
    out[m11] = 1.0 - rho
    return out


@dataclass
class DixonColesFit:
    """Resultado del ajuste: parametros y utilidades de prediccion."""

    team_ids: list[int]
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    home_advantage: float
    rho: float
    kappa_attack: float
    kappa_defence: float
    n_matches: int
    effective_n: float
    log_likelihood: float
    converged: bool
    config: DixonColesConfig

    def index_of(self, team_id: int) -> int:
        return self.team_ids.index(team_id)

    def rates(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Goles esperados (local, visitante)."""
        h, a = self.index_of(home_id), self.index_of(away_id)
        lam_home = np.exp(self.mu + self.attack[h] - self.defence[a] + self.home_advantage)
        lam_away = np.exp(self.mu + self.attack[a] - self.defence[h])
        return float(lam_home), float(lam_away)

    def rates_batch(self, home_ids: np.ndarray, away_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Version vectorizada de `rates` sobre indices internos ya resueltos."""
        lam_home = np.exp(self.mu + self.attack[home_ids] - self.defence[away_ids] + self.home_advantage)
        lam_away = np.exp(self.mu + self.attack[away_ids] - self.defence[home_ids])
        return lam_home, lam_away

    def score_matrix(self, home_id: int, away_id: int, max_goals: int | None = None) -> np.ndarray:
        """Matriz de probabilidad conjunta de marcadores, ya normalizada."""
        lam_home, lam_away = self.rates(home_id, away_id)
        return score_matrix_from_rates(lam_home, lam_away, self.rho,
                                       max_goals or self.config.max_goals)

    def probs_1x2(self, home_id: int, away_id: int) -> tuple[float, float, float]:
        mat = self.score_matrix(home_id, away_id)
        home = float(np.tril(mat, -1).sum())
        draw = float(np.trace(mat))
        away = float(np.triu(mat, 1).sum())
        return home, draw, away

    def subset(self, team_ids: list[int]) -> "DixonColesFit":
        """Recorta el ajuste a un subconjunto de equipos, sin reajustar nada.

        Sirve para ajustar los parametros globales (tasa base de goles, ventaja
        de campo, rho) con todo el historico disponible y simular despues solo
        con los equipos que participan, sin arrastrar matrices enormes.
        """
        indices = [self.index_of(t) for t in team_ids]
        return DixonColesFit(
            team_ids=list(team_ids),
            attack=self.attack[indices], defence=self.defence[indices],
            mu=self.mu, home_advantage=self.home_advantage, rho=self.rho,
            kappa_attack=self.kappa_attack, kappa_defence=self.kappa_defence,
            n_matches=self.n_matches, effective_n=self.effective_n,
            log_likelihood=self.log_likelihood, converged=self.converged,
            config=self.config,
        )

    def strength_table(self, names: dict[int, str] | None = None) -> pd.DataFrame:
        df = pd.DataFrame({
            "team_id": self.team_ids,
            "attack": self.attack,
            "defence": self.defence,
        })
        if names:
            df["team"] = df["team_id"].map(names)
        df["net"] = df["attack"] + df["defence"]
        return df.sort_values("net", ascending=False).reset_index(drop=True)


def score_matrix_from_rates(lam_home: float, lam_away: float, rho: float, max_goals: int) -> np.ndarray:
    """Distribucion conjunta de marcadores para unas tasas dadas."""
    goals = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, max_goals + 1))]))
    p_home = np.exp(goals * np.log(lam_home) - lam_home - log_fact)
    p_away = np.exp(goals * np.log(lam_away) - lam_away - log_fact)
    mat = np.outer(p_home, p_away)

    mat[0, 0] *= 1.0 - lam_home * lam_away * rho
    mat[0, 1] *= 1.0 + lam_home * rho
    mat[1, 0] *= 1.0 + lam_away * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 1e-15, None)
    return mat / mat.sum()


def time_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Peso exponencial: 1 el dia del corte, 0.5 tras una media vida."""
    age = (cutoff - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (np.clip(age, 0.0, None) / half_life_days)


def _unpack(params: np.ndarray, n_teams: int):
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    mu, home_adv, rho, k_att, k_def = params[2 * n_teams:2 * n_teams + 5]
    return attack, defence, mu, home_adv, rho, k_att, k_def


def _objective(params, n_teams, hi, ai, hg, ag, w, elo_std, prior_w, l2_w):
    """Log-verosimilitud negativa penalizada y su gradiente analitico."""
    attack, defence, mu, home_adv, rho, k_att, k_def = _unpack(params, n_teams)

    lam_h = np.exp(mu + attack[hi] - defence[ai] + home_adv)
    lam_a = np.exp(mu + attack[ai] - defence[hi])

    t = tau(hg, ag, lam_h, lam_a, rho)
    t = np.clip(t, 1e-10, None)

    ll = w * (np.log(t) + hg * np.log(lam_h) - lam_h + ag * np.log(lam_a) - lam_a)
    nll = -float(ll.sum())

    # --- derivadas de tau ---
    dt_dlh = np.zeros_like(lam_h)
    dt_dla = np.zeros_like(lam_a)
    dt_drho = np.zeros_like(lam_h)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    dt_dlh[m00] = -lam_a[m00] * rho
    dt_dla[m00] = -lam_h[m00] * rho
    dt_drho[m00] = -lam_h[m00] * lam_a[m00]
    dt_dlh[m01] = rho
    dt_drho[m01] = lam_h[m01]
    dt_dla[m10] = rho
    dt_drho[m10] = lam_a[m10]
    dt_drho[m11] = -1.0

    # A y B: derivada de la log-verosimilitud respecto a log(lambda).
    A = w * (lam_h * dt_dlh / t + hg - lam_h)
    B = w * (lam_a * dt_dla / t + ag - lam_a)

    grad_attack = np.zeros(n_teams)
    grad_defence = np.zeros(n_teams)
    np.add.at(grad_attack, hi, A)
    np.add.at(grad_attack, ai, B)
    np.add.at(grad_defence, ai, -A)
    np.add.at(grad_defence, hi, -B)

    grad_mu = float((A + B).sum())
    grad_home = float(A.sum())
    grad_rho = float((w * dt_drho / t).sum())

    grad = np.concatenate([
        -grad_attack, -grad_defence,
        [-grad_mu, -grad_home, -grad_rho, 0.0, 0.0],
    ])

    # --- penalizaciones: prior Elo + L2 ---
    res_att = attack - k_att * elo_std
    res_def = defence - k_def * elo_std
    nll += prior_w * float((res_att ** 2).sum() + (res_def ** 2).sum())
    nll += l2_w * float((attack ** 2).sum() + (defence ** 2).sum())

    grad[:n_teams] += 2 * prior_w * res_att + 2 * l2_w * attack
    grad[n_teams:2 * n_teams] += 2 * prior_w * res_def + 2 * l2_w * defence
    grad[2 * n_teams + 3] = -2 * prior_w * float((res_att * elo_std).sum())
    grad[2 * n_teams + 4] = -2 * prior_w * float((res_def * elo_std).sum())

    return nll, grad


def fit_dixon_coles(
    matches: pd.DataFrame,
    elo_ratings: dict[int, float],
    cutoff: pd.Timestamp | None = None,
    teams: list[int] | None = None,
    config: DixonColesConfig | None = None,
) -> DixonColesFit:
    """Ajusta el modelo con los partidos anteriores al corte.

    `elo_ratings` debe contener el rating de cada equipo a la fecha de corte;
    los equipos que no aparecen en `matches` (p. ej. un recien ascendido)
    obtienen ataque y defensa unicamente del prior Elo.
    """
    cfg = config or DixonColesConfig()
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else matches["match_date"].max()

    hist = matches[
        (matches["match_date"] < cutoff)
        & (matches["match_date"] >= cutoff - pd.Timedelta(days=cfg.max_history_days))
        & matches["home_goals"].notna()
    ].copy()

    team_ids = sorted(set(teams) if teams is not None
                      else set(hist["home_team_id"]) | set(hist["away_team_id"]))
    index = {tid: i for i, tid in enumerate(team_ids)}
    n_teams = len(team_ids)

    hist = hist[hist["home_team_id"].isin(index) & hist["away_team_id"].isin(index)]
    hi = hist["home_team_id"].map(index).to_numpy(dtype=int)
    ai = hist["away_team_id"].map(index).to_numpy(dtype=int)
    hg = hist["home_goals"].to_numpy(dtype=float)
    ag = hist["away_goals"].to_numpy(dtype=float)
    w = time_weights(hist["match_date"], cutoff, cfg.half_life_days)

    # Prior: Elo centrado en la media de los equipos modelados y escalado.
    elo = np.array([elo_ratings.get(tid, np.nan) for tid in team_ids], dtype=float)
    if np.isnan(elo).all():
        elo = np.zeros(n_teams)
    elo = np.where(np.isnan(elo), np.nanmean(elo), elo)
    elo_std = (elo - elo.mean()) / ELO_SCALE

    x0 = np.concatenate([
        0.15 * elo_std, 0.15 * elo_std,
        [np.log(max(hg.mean(), 0.5)) if len(hg) else 0.0, 0.25, 0.0, 0.15, 0.15],
    ])
    bounds = ([(-2.0, 2.0)] * (2 * n_teams)
              + [(-2.0, 2.0), (-0.5, 1.0), cfg.rho_bounds, (0.0, 1.5), (0.0, 1.5)])

    res = minimize(
        _objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        args=(n_teams, hi, ai, hg, ag, w, elo_std,
              cfg.elo_prior_weight, cfg.l2_weight),
        options={"maxiter": 500, "ftol": 1e-10},
    )

    attack, defence, mu, home_adv, rho, k_att, k_def = _unpack(res.x, n_teams)
    # Identificacion: sumar la misma constante a ataque y defensa de todos los
    # equipos no cambia ninguna prediccion, solo desplaza mu. Fijamos el gauge
    # centrando ambos vectores en cero y absorbiendo el desplazamiento en mu,
    # que asi pasa a significar el log de la tasa base de goles.
    a_bar, d_bar = attack.mean(), defence.mean()
    attack = attack - a_bar
    defence = defence - d_bar
    mu = mu + a_bar - d_bar

    return DixonColesFit(
        team_ids=team_ids, attack=attack, defence=defence, mu=float(mu),
        home_advantage=float(home_adv), rho=float(rho),
        kappa_attack=float(k_att), kappa_defence=float(k_def),
        n_matches=len(hist), effective_n=float(w.sum()),
        log_likelihood=-float(res.fun), converged=bool(res.success), config=cfg,
    )
