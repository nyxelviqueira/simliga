"""Metricas para medir si el modelo predice bien.

Todas asumen probabilidades ya normalizadas. La referencia obligada es el
mercado: un modelo de futbol que no se acerca a las cuotas de cierre no esta
aportando informacion, solo ruido con forma de tabla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """RPS por observacion para resultados ordinales.

    `probs` es (n, k) con las categorias en orden natural (1, X, 2) y `outcomes`
    el indice de la categoria observada. Penaliza mas equivocarse "lejos": dar
    probabilidad a la victoria visitante cuando gana el local cuesta mas que
    habersela dado al empate.
    """
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    obs = np.zeros_like(probs)
    obs[np.arange(n), np.asarray(outcomes, dtype=int)] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(obs, axis=1)[:, :-1]
    return ((cum_p - cum_o) ** 2).sum(axis=1) / (k - 1)


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    picked = probs[np.arange(len(probs)), np.asarray(outcomes, dtype=int)]
    return -np.log(np.clip(picked, eps, 1.0))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Brier multiclase (suma de cuadrados sobre todas las categorias)."""
    probs = np.asarray(probs, dtype=float)
    obs = np.zeros_like(probs)
    obs[np.arange(len(probs)), np.asarray(outcomes, dtype=int)] = 1.0
    return ((probs - obs) ** 2).sum(axis=1)


def binary_brier(prob: np.ndarray, happened: np.ndarray) -> np.ndarray:
    return (np.asarray(prob, dtype=float) - np.asarray(happened, dtype=float)) ** 2


def devig(odds_h, odds_d, odds_a) -> np.ndarray:
    """Convierte cuotas decimales en probabilidades quitando el margen.

    Normalizacion proporcional: suficiente para usar el mercado como referencia
    (metodos como Shin cambian los resultados en la tercera decimal).
    """
    raw = np.column_stack([1.0 / np.asarray(odds_h, dtype=float),
                           1.0 / np.asarray(odds_d, dtype=float),
                           1.0 / np.asarray(odds_a, dtype=float)])
    return raw / raw.sum(axis=1, keepdims=True)


def outcome_index(home_goals, away_goals) -> np.ndarray:
    """0 = gana el local, 1 = empate, 2 = gana el visitante."""
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def position_rps(position_probs: np.ndarray, actual_positions: np.ndarray) -> np.ndarray:
    """RPS de la distribucion de posicion final, por equipo.

    `position_probs` es (n_equipos, n_posiciones); `actual_positions` la posicion
    real (base 1). Mide si la distribucion simulada concentra masa cerca de donde
    el equipo acabo de verdad.
    """
    return ranked_probability_score(position_probs, np.asarray(actual_positions, dtype=int) - 1)


def calibration_table(prob: np.ndarray, happened: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Frecuencia observada frente a probabilidad predicha, por tramos.

    Un modelo calibrado cumple que, de los sucesos a los que asigno un 30%,
    ocurre aproximadamente el 30%.
    """
    prob = np.asarray(prob, dtype=float)
    happened = np.asarray(happened, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": int(mask.sum()),
            "predicted": float(prob[mask].mean()),
            "observed": float(happened[mask].mean()),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["error"] = df["observed"] - df["predicted"]
    return df


def summarize_match_metrics(
    probs: np.ndarray, outcomes: np.ndarray, label: str = "modelo"
) -> dict:
    return {
        "fuente": label,
        "n": int(len(probs)),
        "rps": float(ranked_probability_score(probs, outcomes).mean()),
        "log_loss": float(log_loss(probs, outcomes).mean()),
        "brier": float(brier_score(probs, outcomes).mean()),
        "accuracy": float((probs.argmax(axis=1) == np.asarray(outcomes)).mean()),
    }
