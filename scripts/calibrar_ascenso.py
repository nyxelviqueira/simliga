"""Calibra la correccion por ascenso de division.

Dos preguntas separadas:

1. **Cuanto** hay que restar al Elo de un recien ascendido (`promotion_penalty`).
2. **Durante cuanto** se mantiene esa resta (`promotion_penalty_fade`), medido en
   partidos jugados en la nueva division.

La segunda existe porque cada partido en la categoria nueva reajusta el Elo
contra rivales de la escala correcta: el desfase se corrige solo, y mantener la
resta entera en la jornada 30 seria contarla dos veces.

Se mide con el RPS partido a partido, que es la unica prueba que importa:
predecir mejor partidos que el modelo no ha visto.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from simliga.config import COMP_LALIGA, COMP_SEGUNDA, load_config
from simliga.data import load_matches, load_odds
from simliga.db import connect
from simliga.validation.backtest import backtest_matches

SEASONS = tuple(f"{a}-{str(a + 1)[-2:]}" for a in range(2018, 2026))


def rps(liga, todo, odds, penalty: float, fade: float) -> dict[str, float]:
    cfg = load_config()
    cfg.elo.promotion_penalty = penalty
    cfg.elo.promotion_penalty_fade = fade
    return {s: backtest_matches(liga, todo, s, cfg, odds).model["rps"] for s in SEASONS}


def main() -> None:
    conn = connect()
    liga = load_matches(conn, (COMP_LALIGA,))
    todo = load_matches(conn, (COMP_LALIGA, COMP_SEGUNDA))
    odds = load_odds(conn)

    print("Efecto de la correccion por ascenso sobre el RPS (8 temporadas)")
    print("=" * 66)
    resultados = {}
    combinaciones = [
        (0.0, 0.0, "sin correccion"),
        (50.0, 1.0, "solo el primer partido (el fallo que habia)"),
        (50.0, 10.0, "se desvanece en 10 partidos"),
        (50.0, 19.0, "se desvanece en media temporada"),
        (50.0, 38.0, "se desvanece en toda la temporada"),
        (50.0, 0.0, "no se desvanece nunca"),
        (35.0, 19.0, "mas suave, media temporada"),
        (70.0, 19.0, "mas dura, media temporada"),
    ]
    for penalty, fade, etiqueta in combinaciones:
        por_temporada = rps(liga, todo, odds, penalty, fade)
        media = float(np.mean(list(por_temporada.values())))
        resultados[(penalty, fade)] = por_temporada
        print(f"  {etiqueta:<44} RPS {media:.5f}", flush=True)

    base = np.mean(list(resultados[(0.0, 0.0)].values()))
    print()
    print("Diferencia frente a no corregir (negativo = mejora):")
    for (penalty, fade), valores in resultados.items():
        if (penalty, fade) == (0.0, 0.0):
            continue
        diferencias = [valores[s] - resultados[(0.0, 0.0)][s] for s in SEASONS]
        media = float(np.mean(diferencias))
        error = float(np.std(diferencias) / np.sqrt(len(diferencias)))
        veredicto = "mejora" if media < -2 * error else ("empeora" if media > 2 * error
                                                         else "indistinguible")
        print(f"  penalty={penalty:>5} fade={fade:>5}  {media:+.5f} (ee {error:.5f})  {veredicto}")

    mejor = min(resultados, key=lambda k: np.mean(list(resultados[k].values())))
    print()
    print(f"Mejor combinacion por RPS: penalty={mejor[0]}, fade={mejor[1]}")
    print(f"  (base sin corregir: {base:.5f})")


if __name__ == "__main__":
    main()
