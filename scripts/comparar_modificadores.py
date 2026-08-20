"""Mide si activar los modificadores mejora la capacidad predictiva.

Ejecuta el backtest partido a partido con cada configuracion y compara el RPS.
Es la prueba que decide si un modificador entra o no: que la idea sea razonable
no basta, tiene que reducir el error sobre partidos que el modelo no ha visto.

El barrido de magnitudes se hace **dejando fuera la temporada de test**, para no
elegir el mejor valor mirando el mismo dato con el que luego se presume.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from simliga.config import COMP_LALIGA, COMP_SEGUNDA, UEFA_COMPETITIONS, load_config
from simliga.data import load_matches, load_odds
from simliga.db import connect
from simliga.validation.backtest import backtest_matches

SEASONS = tuple(f"{a}-{str(a + 1)[-2:]}" for a in range(2018, 2026))


def rps_por_temporada(liga, combinado, odds, cfg) -> dict[str, float]:
    return {s: backtest_matches(liga, combinado, s, cfg, odds).model["rps"]
            for s in SEASONS}


def main() -> None:
    conn = connect()
    liga = load_matches(conn, (COMP_LALIGA,))
    # El calendario combinado es lo que da los dias de descanso reales: sin los
    # partidos europeos, un miercoles de Champions no existiria para la fatiga.
    combinado = load_matches(conn, (COMP_LALIGA, COMP_SEGUNDA) + UEFA_COMPETITIONS)
    odds = load_odds(conn)

    print("Base: sin ningun modificador")
    base = rps_por_temporada(liga, combinado, odds, load_config())
    print(f"  RPS medio = {np.mean(list(base.values())):.5f}\n")

    print("Fatiga, barriendo la penalizacion por dia de deficit:")
    resultados = {}
    for elo_dia in (4.0, 8.0, 12.0, 20.0):
        cfg = load_config()
        cfg.modifiers.fatigue_enabled = True
        cfg.modifiers.fatigue_elo_per_day = elo_dia
        resultados[elo_dia] = rps_por_temporada(liga, combinado, odds, cfg)
        medio = np.mean(list(resultados[elo_dia].values()))
        delta = medio - np.mean(list(base.values()))
        print(f"  {elo_dia:>5.0f} Elo/dia -> RPS {medio:.5f}  ({delta:+.5f} frente a la base)",
              flush=True)

    print("\nValidacion cruzada dejando una temporada fuera:")
    mejoras = []
    for test in SEASONS:
        entreno = [s for s in SEASONS if s != test]
        mejor = min(resultados, key=lambda k: np.mean([resultados[k][s] for s in entreno]))
        # ¿La magnitud elegida sin mirar la temporada de test la mejora?
        delta = resultados[mejor][test] - base[test]
        mejoras.append(delta)
        print(f"  {test}: magnitud elegida {mejor:>4.0f} Elo/dia | "
              f"RPS {base[test]:.5f} -> {resultados[mejor][test]:.5f} ({delta:+.5f})")

    medio = float(np.mean(mejoras))
    error = float(np.std(mejoras) / np.sqrt(len(mejoras)))
    print()
    print("=" * 74)
    print(f"Cambio medio de RPS fuera de muestra: {medio:+.5f} (ee {error:.5f})")
    if medio < -error * 2:
        print("La fatiga MEJORA la prediccion de forma consistente: conviene activarla.")
    elif medio > error * 2:
        print("La fatiga EMPEORA la prediccion: debe quedarse desactivada.")
    else:
        print("El efecto no se distingue de cero. La fatiga se queda desactivada")
        print("por defecto: anadir un parametro que no mejora nada solo agrega")
        print("ruido y una via mas por la que equivocarse.")
    print("=" * 74)


if __name__ == "__main__":
    main()
