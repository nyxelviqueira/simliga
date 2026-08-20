"""Calibracion de la fuerza relativa de cada liga.

El Elo es de suma cero dentro de su pool: si los equipos de una liga apenas
juegan contra los de otra, la media de cada liga se queda clavada donde arranco
y los ratings dejan de ser comparables entre paises. Con los datos del proyecto
eso se ve a simple vista: sin corregir, el Olympiakos aparece por delante del
Liverpool, porque domina la liga griega y el Elo no tiene forma de saber que la
liga griega es mas debil.

La informacion que si permite compararlas son los partidos de competicion
europea, los unicos que cruzan ligas. Este modulo estima un desplazamiento por
liga ajustandolo a esos resultados: cuanto hay que sumar o restar al Elo de los
equipos de cada pais para que sus resultados continentales dejen de sorprender.

Es el mismo problema que la penalizacion por ascenso entre Primera y Segunda,
solo que con once pools en vez de dos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _expected(diff: np.ndarray) -> np.ndarray:
    # Se acota el exponente: con diferencias absurdas la potencia desborda y
    # ensucia la busqueda con avisos, aunque el resultado ya sea 0 o 1.
    return 1.0 / (1.0 + 10.0 ** np.clip(-diff / 400.0, -30.0, 30.0))


def estimate_league_offsets(
    uefa_matches: pd.DataFrame,
    home_advantage: float = 70.0,
    reference: str | None = None,
    min_matches: int = 20,
) -> dict[str, float]:
    """Ajusta un desplazamiento de Elo por liga con los partidos europeos.

    `uefa_matches` necesita las columnas `home_league`, `away_league` y el
    resultado; `elo_before` aporta el rating de cada equipo antes del partido.
    El desplazamiento de la liga de referencia se fija a 0 porque solo importan
    las diferencias entre ligas (si se sumase lo mismo a todas, nada cambiaria).

    Una liga con menos de `min_matches` partidos europeos no tiene su
    desplazamiento identificado y se deja en 0. Es el caso de la Segunda
    espanola, que no juega competicion continental: sin este filtro el
    optimizador le asignaba un valor arbitrario (+53 en una prueba, por encima
    de la Ligue 1) que ademas se contradecia con la penalizacion por ascenso,
    que es el mecanismo correcto para esa diferencia.
    """
    df = uefa_matches.dropna(subset=["home_goals", "away_goals"]).copy()
    if df.empty:
        return {}

    apariciones = pd.concat([df["home_league"], df["away_league"]]).value_counts()
    ligas = sorted(apariciones[apariciones >= min_matches].index)
    if len(ligas) < 2:
        return {}
    df = df[df["home_league"].isin(ligas) & df["away_league"].isin(ligas)]
    reference = reference or max(
        ligas, key=lambda l: ((df["home_league"] == l) | (df["away_league"] == l)).sum()
    )
    libres = [l for l in ligas if l != reference]
    index = {l: i for i, l in enumerate(libres)}

    hi = df["home_league"].map(lambda l: index.get(l, -1)).to_numpy()
    ai = df["away_league"].map(lambda l: index.get(l, -1)).to_numpy()
    base = (df["home_elo"].to_numpy() - df["away_elo"].to_numpy() + home_advantage)
    actual = np.where(df["home_goals"] > df["away_goals"], 1.0,
                      np.where(df["home_goals"] < df["away_goals"], 0.0, 0.5))

    def error(offsets: np.ndarray) -> float:
        ext = np.concatenate([offsets, [0.0]])          # la referencia va al final
        diff = base + ext[hi] - ext[ai]
        return float(np.mean((_expected(diff) - actual) ** 2))

    res = minimize(error, np.zeros(len(libres)), method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 20000})

    offsets = {liga: float(res.x[i]) for liga, i in index.items()}
    offsets[reference] = 0.0
    # Recentrado: media cero, para que los ratings ajustados sigan en la escala
    # habitual de Elo y se puedan leer junto a los no ajustados.
    media = float(np.mean(list(offsets.values())))
    return {liga: valor - media for liga, valor in offsets.items()}


def build_uefa_training_frame(
    uefa_matches: pd.DataFrame,
    elo_history: pd.DataFrame,
    team_league: dict[int, str],
) -> pd.DataFrame:
    """Prepara los partidos europeos con el Elo previo y la liga de cada equipo."""
    hist = elo_history.set_index("match_id")
    df = uefa_matches[uefa_matches["match_id"].isin(hist.index)].copy()
    df["home_elo"] = hist.loc[df["match_id"], "home_elo_before"].to_numpy()
    df["away_elo"] = hist.loc[df["match_id"], "away_elo_before"].to_numpy()
    df["home_league"] = df["home_team_id"].map(team_league)
    df["away_league"] = df["away_team_id"].map(team_league)
    return df.dropna(subset=["home_league", "away_league"])


def apply_offsets(
    ratings: dict[int, float], team_league: dict[int, str], offsets: dict[str, float]
) -> dict[int, float]:
    """Devuelve los ratings ya comparables entre ligas."""
    return {t: r + offsets.get(team_league.get(t, ""), 0.0) for t, r in ratings.items()}


def team_league_map(matches: pd.DataFrame, domestic_only: bool = True) -> dict[int, str]:
    """Liga domestica de cada equipo: aquella en la que jugo mas recientemente.

    Un equipo puede cambiar de division (ascensos, descensos); se toma la ultima
    porque es la que describe su contexto competitivo actual.
    """
    df = matches
    if domestic_only:
        df = df[~df["competition"].isin(("UCL", "UEL", "UECL"))]

    largo = pd.concat([
        df[["match_date", "competition", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}),
        df[["match_date", "competition", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}),
    ])
    ultimo = largo.sort_values("match_date").groupby("team_id").tail(1)
    return dict(zip(ultimo["team_id"], ultimo["competition"]))
