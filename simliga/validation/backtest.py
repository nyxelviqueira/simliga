"""Backtest historico: que habria dicho el modelo sabiendo solo el pasado.

Dos modos, tal y como se pidieron:

- `backtest_matches`: recorre la temporada en orden cronologico reajustando el
  modelo antes de cada fecha y compara las probabilidades 1X2 partido a partido
  contra lo que ocurrio, y contra el mercado de apuestas.
- `backtest_season`: en una serie de jornadas de corte, re-simula la temporada
  entera con lo jugado hasta ese punto y compara la distribucion de posiciones
  con la clasificacion final real.

En ambos casos el modelo nunca ve un partido anterior a la fecha que predice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import COMP_LALIGA, Config, load_config
from ..data import league_table, promoted_into
from ..model.dixon_coles import DixonColesFit, fit_dixon_coles
from ..model.dixon_coles import score_matrix_from_rates
from ..model.elo import EloEngine
from ..model.modifiers import (build_match_adjustments, compute_rest_days,
                               elo_delta_to_rate_shift)
from ..sim.league import simulate_league
from . import metrics


def assign_matchday(season_matches: pd.DataFrame) -> pd.Series:
    """Numero de jornada aproximado a partir del orden cronologico.

    football-data.co.uk no publica la jornada. Con N equipos hay N/2 partidos
    por jornada, asi que numeramos por bloques en orden de fecha. Los partidos
    aplazados pueden caer en un bloque contiguo al real; para elegir puntos de
    corte del backtest la aproximacion es inocua.
    """
    n_teams = len(set(season_matches["home_team_id"]) | set(season_matches["away_team_id"]))
    per_round = max(n_teams // 2, 1)
    ordered = season_matches.sort_values(["match_date", "match_id"])
    return pd.Series(np.arange(len(ordered)) // per_round + 1, index=ordered.index)


@dataclass
class MatchBacktestResult:
    predictions: pd.DataFrame
    model: dict
    market: dict | None
    baseline: dict
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


def backtest_matches(
    league_matches: pd.DataFrame,
    all_matches: pd.DataFrame,
    season: str,
    config: Config | None = None,
    odds: pd.DataFrame | None = None,
) -> MatchBacktestResult:
    """Prediccion 1X2 partido a partido, con reajuste antes de cada fecha.

    `league_matches` son los partidos de LaLiga (historico + temporada a validar)
    y `all_matches` incluye ademas Segunda, que alimenta el Elo de los ascendidos.
    """
    cfg = config or load_config()
    season_matches = league_matches[league_matches["season"] == season].sort_values(
        ["match_date", "match_id"]
    )
    if season_matches.empty:
        raise ValueError(f"No hay partidos de la temporada {season}")

    # Descanso de cada equipo, del calendario combinado que se le pase.
    rest = (compute_rest_days(all_matches)
            if cfg.modifiers.fatigue_enabled else None)

    start = season_matches["match_date"].min()
    teams = sorted(set(season_matches["home_team_id"]) | set(season_matches["away_team_id"]))

    ascendidos = promoted_into(all_matches, season, COMP_LALIGA)

    # Elo con todo lo anterior al inicio de temporada; se ira actualizando fecha a fecha.
    engine = EloEngine(cfg.elo)
    engine.run(all_matches[all_matches["match_date"] < start])
    pending_elo = all_matches[
        (all_matches["match_date"] >= start)
        & (all_matches["match_date"] <= season_matches["match_date"].max())
    ].sort_values(["match_date", "match_id"])

    rows = []
    for date, day_matches in season_matches.groupby("match_date", sort=True):
        # Elo al dia: incorpora todo lo jugado (Primera y Segunda) antes de esta fecha.
        due = pending_elo[pending_elo["match_date"] < date]
        pending_elo = pending_elo[pending_elo["match_date"] >= date]
        engine.run(due)

        fit = fit_dixon_coles(
            league_matches, engine.ratings_at_cutoff(promoted_teams=ascendidos),
            cutoff=date, teams=teams, config=cfg.dixon_coles,
        )
        ajustes = build_match_adjustments(day_matches, cfg.modifiers, rest=rest)
        desplaza = not ajustes.is_empty()

        for pos, m in enumerate(day_matches.itertuples(index=False)):
            lam_h, lam_a = fit.rates(int(m.home_team_id), int(m.away_team_id))
            if desplaza:
                sh = elo_delta_to_rate_shift(
                    np.array([ajustes.home[pos]]), np.array([-ajustes.away[pos]]),
                    fit.kappa_attack, fit.kappa_defence)[0]
                sa = elo_delta_to_rate_shift(
                    np.array([ajustes.away[pos]]), np.array([-ajustes.home[pos]]),
                    fit.kappa_attack, fit.kappa_defence)[0]
                lam_h, lam_a = lam_h * np.exp(sh), lam_a * np.exp(sa)

            mat = score_matrix_from_rates(lam_h, lam_a, fit.rho, fit.config.max_goals)
            p_home = float(np.tril(mat, -1).sum())
            p_draw = float(np.trace(mat))
            p_away = float(np.triu(mat, 1).sum())
            rows.append({
                "match_id": int(m.match_id), "match_date": date, "season": season,
                "home_team": m.home_team, "away_team": m.away_team,
                "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                "exp_goals_home": lam_h, "exp_goals_away": lam_a,
                "home_goals": int(m.home_goals), "away_goals": int(m.away_goals),
            })

    preds = pd.DataFrame(rows)
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = metrics.outcome_index(preds["home_goals"], preds["away_goals"])

    model = metrics.summarize_match_metrics(probs, outcomes, "modelo")
    base_rate = np.bincount(outcomes, minlength=3) / len(outcomes)
    baseline = metrics.summarize_match_metrics(
        np.tile(base_rate, (len(outcomes), 1)), outcomes, "tasa base de la temporada"
    )

    market = None
    if odds is not None and len(odds):
        merged = preds.merge(odds, on="match_id", how="inner")
        if len(merged):
            market_probs = metrics.devig(merged["odds_h"], merged["odds_d"], merged["odds_a"])
            market_outcomes = metrics.outcome_index(merged["home_goals"], merged["away_goals"])
            market = metrics.summarize_match_metrics(market_probs, market_outcomes, "mercado")
            # Comparacion justa: mismas filas para modelo y mercado.
            model = metrics.summarize_match_metrics(
                merged[["p_home", "p_draw", "p_away"]].to_numpy(), market_outcomes, "modelo"
            )

    calib = metrics.calibration_table(
        np.concatenate([probs[:, 0], probs[:, 1], probs[:, 2]]),
        np.concatenate([(outcomes == 0), (outcomes == 1), (outcomes == 2)]).astype(float),
    )
    return MatchBacktestResult(preds, model, market, baseline, calib)


@dataclass
class SeasonBacktestResult:
    checkpoints: pd.DataFrame       # una fila por (jornada de corte, equipo)
    by_checkpoint: pd.DataFrame     # metricas agregadas por jornada de corte
    final_table: pd.DataFrame


def backtest_season(
    league_matches: pd.DataFrame,
    all_matches: pd.DataFrame,
    season: str,
    checkpoints: tuple[int, ...] = (0, 5, 10, 19, 28, 34),
    config: Config | None = None,
) -> SeasonBacktestResult:
    """Re-simula la temporada completa en varios puntos de corte.

    `checkpoints` son jornadas ya disputadas: 0 = pre-temporada (sin ningun
    partido visto), 19 = ecuador de liga, etc.
    """
    cfg = config or load_config()
    season_matches = league_matches[league_matches["season"] == season].copy()
    if season_matches.empty:
        raise ValueError(f"No hay partidos de la temporada {season}")
    season_matches["matchday"] = assign_matchday(season_matches)
    ascendidos = promoted_into(all_matches, season, COMP_LALIGA)

    teams = sorted(set(season_matches["home_team_id"]) | set(season_matches["away_team_id"]))
    names = dict(zip(season_matches["home_team_id"], season_matches["home_team"]))
    final = league_table(season_matches)
    actual_position = dict(zip(final["team"], final["position"]))
    actual_points = dict(zip(final["team"], final["points"]))

    rows, agg = [], []
    for md in checkpoints:
        played = season_matches[season_matches["matchday"] <= md]
        pending = season_matches[season_matches["matchday"] > md]
        cutoff = (pending["match_date"].min() if len(pending)
                  else season_matches["match_date"].max() + pd.Timedelta(days=1))

        engine = EloEngine(cfg.elo)
        engine.run(all_matches[all_matches["match_date"] < cutoff])
        fit = fit_dixon_coles(
            league_matches, engine.ratings_at_cutoff(promoted_teams=ascendidos),
            cutoff=cutoff, teams=teams, config=cfg.dixon_coles,
        )
        res = simulate_league(
            fit, pending, played=played, teams=teams, config=cfg.sim,
            rng=np.random.default_rng(cfg.sim.seed + md),
        )

        pos_probs = res.position_probabilities()
        actual = np.array([actual_position[names[t]] for t in res.team_ids])
        rps = metrics.position_rps(pos_probs, actual)
        n = len(res.team_ids)

        for i, tid in enumerate(res.team_ids):
            team = names[tid]
            rows.append({
                "season": season, "matchday": md, "team": team,
                "exp_points": float(res.points[:, i].mean()),
                "exp_position": float(res.positions[:, i].mean()),
                "p_title": float((res.positions[:, i] == 1).mean()),
                "p_ucl": float(res.prob_position_range(1, cfg.sim.ucl_slots)[i]),
                "p_relegation": float(res.prob_position_range(n - cfg.sim.relegation_slots + 1, n)[i]),
                "position_rps": float(rps[i]),
                "actual_position": int(actual_position[team]),
                "actual_points": int(actual_points[team]),
            })

        sub = pd.DataFrame(rows[-len(res.team_ids):])
        agg.append({
            "season": season, "matchday": md,
            "position_rps": float(rps.mean()),
            "mae_points": float((sub["exp_points"] - sub["actual_points"]).abs().mean()),
            "spearman_position": float(sub["exp_position"].corr(
                sub["actual_position"], method="spearman")),
            "brier_title": float(metrics.binary_brier(
                sub["p_title"], (sub["actual_position"] == 1).astype(float)).mean()),
            "brier_ucl": float(metrics.binary_brier(
                sub["p_ucl"], (sub["actual_position"] <= cfg.sim.ucl_slots).astype(float)).mean()),
            "brier_relegation": float(metrics.binary_brier(
                sub["p_relegation"],
                (sub["actual_position"] > n - cfg.sim.relegation_slots).astype(float)).mean()),
        })

    return SeasonBacktestResult(pd.DataFrame(rows), pd.DataFrame(agg), final)
