"""Rating Elo dinamico, actualizado partido a partido.

Sigue el espiritu de ClubElo: factor local en puntos de rating, multiplicador
por diferencia de goles y regresion parcial a la media entre temporadas. Se
calcula en casa (no se descarga) para que sea reproducible, auditable y
extensible a Segunda, cuyos equipos ascendidos necesitan un rating de partida.

El Elo NO produce por si solo las probabilidades de resultado: alimenta como
prior el ajuste Dixon-Coles, que es quien modela goles y empates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import EloConfig


def expected_score(rating_home: float, rating_away: float, home_advantage: float) -> float:
    """Probabilidad esperada (en puntos Elo) del local: victoria=1, empate=0.5."""
    diff = rating_home + home_advantage - rating_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def goal_difference_multiplier(goal_diff: int) -> float:
    """Amplifica el ajuste segun la contundencia del resultado."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


@dataclass
class EloState:
    """Ratings vigentes y metadatos por equipo."""

    ratings: dict[int, float] = field(default_factory=dict)
    last_season: dict[int, str] = field(default_factory=dict)
    last_competition: dict[int, str] = field(default_factory=dict)
    matches_played: dict[int, int] = field(default_factory=dict)

    def get(self, team_id: int, default: float) -> float:
        return self.ratings.get(team_id, default)


class EloEngine:
    """Recorre los partidos en orden cronologico manteniendo el rating de cada equipo."""

    def __init__(self, config: EloConfig | None = None):
        self.cfg = config or EloConfig()
        self.state = EloState()
        self.history: list[dict] = []
        self._current_season: str | None = None

    # -- ciclo de vida de la temporada -------------------------------------------------
    def _initial_rating(self, competition: str) -> float:
        return self.cfg.initial_by_competition.get(competition, self.cfg.initial_rating)

    def _apply_season_regression(self) -> None:
        """R' = media + phi*(R - media) al cambiar de temporada.

        Encoge la dispersion: lo que un equipo demostro el año pasado sigue
        informando, pero menos que si la temporada continuase.

        """
        if not self.state.ratings:
            return
        phi = self.cfg.season_regression
        mean = float(np.mean(list(self.state.ratings.values())))
        for team_id, rating in self.state.ratings.items():
            self.state.ratings[team_id] = mean + phi * (rating - mean)

    # -- actualizacion partido a partido ----------------------------------------------
    def update(
        self,
        home_id: int,
        away_id: int,
        home_goals: int,
        away_goals: int,
        competition: str,
        season: str,
        match_date,
        match_id: int | None = None,
    ) -> tuple[float, float]:
        """Procesa un partido y devuelve (rating_local_antes, rating_visitante_antes)."""
        if self._current_season is not None and season != self._current_season:
            self._apply_season_regression()
        self._current_season = season

        default = self._initial_rating(competition)
        r_home = self.state.get(home_id, default)
        r_away = self.state.get(away_id, default)

        exp_home = expected_score(r_home, r_away, self.cfg.home_advantage)
        if home_goals > away_goals:
            actual = 1.0
        elif home_goals < away_goals:
            actual = 0.0
        else:
            actual = 0.5

        mult = (
            goal_difference_multiplier(home_goals - away_goals)
            if self.cfg.use_goal_difference
            else 1.0
        )
        delta = self.cfg.k_factor * mult * (actual - exp_home)

        self.state.ratings[home_id] = r_home + delta
        self.state.ratings[away_id] = r_away - delta
        for tid in (home_id, away_id):
            self.state.last_season[tid] = season
            self.state.last_competition[tid] = competition
            self.state.matches_played[tid] = self.state.matches_played.get(tid, 0) + 1

        self.history.append(
            {
                "match_id": match_id, "match_date": match_date, "season": season,
                "competition": competition,
                "home_team_id": home_id, "away_team_id": away_id,
                "home_elo_before": r_home, "away_elo_before": r_away,
                "home_elo_after": self.state.ratings[home_id],
                "away_elo_after": self.state.ratings[away_id],
                "expected_home": exp_home,
            }
        )
        return r_home, r_away

    # -- API de alto nivel -------------------------------------------------------------
    def run(self, matches: pd.DataFrame) -> "EloEngine":
        """Procesa un DataFrame de partidos jugados, ya ordenado cronologicamente."""
        for m in matches.itertuples(index=False):
            if pd.isna(m.home_goals) or pd.isna(m.away_goals):
                continue
            self.update(
                home_id=int(m.home_team_id), away_id=int(m.away_team_id),
                home_goals=int(m.home_goals), away_goals=int(m.away_goals),
                competition=m.competition, season=m.season,
                match_date=m.match_date, match_id=int(m.match_id),
            )
        return self

    def ratings_at_cutoff(
        self,
        apply_season_regression: bool = False,
        promoted_teams: set[int] | None = None,
    ) -> dict[int, float]:
        """Ratings actuales; opcionalmente ya regresados para la temporada siguiente.

        `promoted_teams` son los equipos que este año cambian de division. A
        ellos se les resta `promotion_penalty`, porque el Elo de Segunda y el de
        Primera estan en escalas distintas: los dos pools apenas se cruzan, asi
        que un ascendido llega con el rating inflado.

        La lista se pasa desde fuera, calculada de los partidos, en lugar de
        deducirla aqui del ultimo partido de cada equipo. Deducirla fallaba de
        dos maneras: se apagaba en cuanto el ascendido jugaba una jornada, y no
        reconocia a un equipo que ya habia estado en Primera hace dos años.

        La correccion se mantiene toda la temporada. Medido sobre 864 partidos
        de ascendidos en ocho temporadas, cuanto mas persiste mejor predice, y
        la mejora viene sobre todo de la segunda vuelta: el Elo no se
        autocorrige tan rapido como cabria suponer, porque el factor K es bajo.
        """
        ratings = dict(self.state.ratings)
        if apply_season_regression and ratings:
            phi = self.cfg.season_regression
            mean = float(np.mean(list(ratings.values())))
            ratings = {t: mean + phi * (r - mean) for t, r in ratings.items()}

        if promoted_teams and self.cfg.promotion_penalty:
            ratings = {t: r - self.cfg.promotion_penalty if t in promoted_teams else r
                       for t, r in ratings.items()}
        return ratings

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def run_elo(matches: pd.DataFrame, config: EloConfig | None = None) -> EloEngine:
    """Atajo: crea el motor y procesa todos los partidos."""
    return EloEngine(config).run(matches)


def calibrate(
    matches: pd.DataFrame,
    k_grid: tuple[float, ...] = (10, 15, 20, 25, 30, 35),
    hfa_grid: tuple[float, ...] = (40, 50, 60, 70, 80, 90),
    regression_grid: tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 1.0),
    burn_in: int = 1000,
) -> tuple[EloConfig, pd.DataFrame]:
    """Busca en rejilla K, ventaja local y regresion minimizando el error de prediccion.

    Metrica: error cuadratico medio entre la puntuacion esperada del local y la
    real (1 / 0.5 / 0), descartando los primeros `burn_in` partidos, en los que
    los ratings aun se estan asentando.
    """
    results = []
    for k in k_grid:
        for hfa in hfa_grid:
            for phi in regression_grid:
                cfg = EloConfig(k_factor=k, home_advantage=hfa, season_regression=phi)
                engine = run_elo(matches, cfg)
                hist = engine.history_frame()
                if len(hist) <= burn_in:
                    continue
                sub = hist.iloc[burn_in:]
                goals = matches.set_index("match_id").loc[sub["match_id"]]
                actual = np.where(
                    goals["home_goals"].values > goals["away_goals"].values, 1.0,
                    np.where(goals["home_goals"].values < goals["away_goals"].values, 0.0, 0.5),
                )
                mse = float(np.mean((sub["expected_home"].values - actual) ** 2))
                results.append({"k_factor": k, "home_advantage": hfa,
                                "season_regression": phi, "mse": mse})

    grid = pd.DataFrame(results).sort_values("mse").reset_index(drop=True)
    best = grid.iloc[0]
    return (
        EloConfig(
            k_factor=float(best["k_factor"]),
            home_advantage=float(best["home_advantage"]),
            season_regression=float(best["season_regression"]),
        ),
        grid,
    )
