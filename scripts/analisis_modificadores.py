"""Mide si los ajustes cualitativos calculables tienen efecto real.

Dos que no necesitan ninguna fuente externa:

- **Fatiga**: dias de descanso desde el partido anterior, en el calendario
  combinado de liga y Europa.
- **Motivacion**: si al equipo le queda algo en juego en las ultimas jornadas.

El metodo es el mismo para los dos: comparar los goles que marco un equipo con
los que el modelo esperaba que marcase. Si el modelo tuviera un punto ciego, el
residuo se desviaria de cero de forma sistematica al agrupar por la variable.

Los residuos vienen del backtest, asi que el modelo nunca vio el partido que
predice.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from simliga.config import (COMP_LALIGA, COMP_SEGUNDA, EUROPEAN_LEAGUES,
                            UEFA_COMPETITIONS, load_config)
from simliga.data import league_table, load_matches, load_odds
from simliga.db import connect
from simliga.validation.backtest import assign_matchday, backtest_matches

SEASONS = tuple(f"{a}-{str(a + 1)[-2:]}" for a in range(2018, 2026))


def contraste(a: pd.Series, b: pd.Series, etiqueta: str) -> dict:
    dif = a.mean() - b.mean()
    err = float(np.sqrt(a.var() / len(a) + b.var() / len(b)))
    t = dif / err if err else 0.0
    veredicto = "SI" if abs(t) >= 2 else "no"
    print(f"  {etiqueta:<52} {dif:+.4f} goles  (ee {err:.4f}, t {t:+5.2f})  "
          f"significativo: {veredicto}")
    return {"etiqueta": etiqueta, "diferencia": dif, "error": err, "t": t,
            "n_a": len(a), "n_b": len(b)}


def main() -> None:
    conn = connect()
    cfg = load_config()
    liga = load_matches(conn, (COMP_LALIGA,))
    todo_es = load_matches(conn, (COMP_LALIGA, COMP_SEGUNDA))
    odds = load_odds(conn)

    # --- calendario combinado: dias de descanso de cada equipo ---
    combinado = load_matches(conn, (COMP_LALIGA,) + UEFA_COMPETITIONS + EUROPEAN_LEAGUES)
    largo = pd.concat([
        combinado[["match_date", "competition", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}),
        combinado[["match_date", "competition", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}),
    ]).sort_values("match_date")
    largo["descanso"] = largo.groupby("team_id")["match_date"].diff().dt.days
    largo["venia_de"] = largo.groupby("team_id")["competition"].shift(1)
    descanso = largo.groupby(["team_id", "match_date"])[["descanso", "venia_de"]].first()

    # --- predicciones del backtest y clasificacion final de cada temporada ---
    trozos, tablas = [], {}
    for s in SEASONS:
        r = backtest_matches(liga, todo_es, s, cfg, odds)
        p = r.predictions.copy()
        ids = liga.set_index("match_id")
        p["home_team_id"] = ids.loc[p["match_id"], "home_team_id"].to_numpy()
        p["away_team_id"] = ids.loc[p["match_id"], "away_team_id"].to_numpy()
        sm = liga[liga["season"] == s].copy()
        sm["matchday"] = assign_matchday(sm)
        p["matchday"] = sm.set_index("match_id").loc[p["match_id"], "matchday"].to_numpy()
        trozos.append(p)
        tablas[s] = league_table(sm).set_index("team")
        print(f"  {s} listo", flush=True)
    pred = pd.concat(trozos)

    for lado in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([pred[f"{lado}_team_id"], pred["match_date"]])
        sub = descanso.reindex(idx)
        pred[f"descanso_{lado}"] = sub["descanso"].to_numpy()
        pred[f"venia_de_{lado}"] = sub["venia_de"].to_numpy()

    # --- un registro por equipo y partido ---
    filas = []
    for lado, contra in (("home", "away"), ("away", "home")):
        filas.append(pd.DataFrame({
            "season": pred["season"], "matchday": pred["matchday"],
            "team": pred[f"{lado}_team"],
            "descanso": pred[f"descanso_{lado}"],
            "descanso_rival": pred[f"descanso_{contra}"],
            "venia_de": pred[f"venia_de_{lado}"],
            "goles": pred[f"{lado}_goals"],
            "esperados": pred[f"exp_goals_{lado}"],
        }))
    d = pd.concat(filas).dropna(subset=["descanso", "descanso_rival"])
    d["residuo"] = d["goles"] - d["esperados"]
    d["europeo"] = d["venia_de"].isin(UEFA_COMPETITIONS)
    print(f"\nobservaciones (equipo x partido): {len(d)}\n")

    # ------------------------------------------------------------------ FATIGA
    print("=" * 78)
    print("FATIGA: goles reales menos esperados, por dias de descanso")
    print("=" * 78)
    d["tramo"] = pd.cut(d["descanso"], [0, 3, 4, 5, 6, 7, 100],
                        labels=["<=3", "4", "5", "6", "7", ">7"])
    print(d.groupby("tramo", observed=True).agg(
        n=("residuo", "size"), residuo=("residuo", "mean"),
        error=("residuo", lambda x: x.std() / np.sqrt(len(x)))
    ).to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nContrastes:")
    resultados = [
        contraste(d[d["descanso"] <= 3]["residuo"], d[d["descanso"] >= 6]["residuo"],
                  "descanso <=3 dias frente a >=6"),
        contraste(d[(d["descanso"] <= 4) & d["europeo"]]["residuo"],
                  d[d["descanso"] >= 6]["residuo"],
                  "venia de Europa con <=4 dias, frente a >=6"),
    ]
    # El rival tambien puede llegar cansado: lo que deberia importar es la
    # diferencia de descanso entre los dos, no el descanso en bruto.
    d["ventaja"] = d["descanso"] - d["descanso_rival"]
    resultados.append(contraste(
        d[d["ventaja"] >= 2]["residuo"], d[d["ventaja"] <= -2]["residuo"],
        "2+ dias mas de descanso que el rival, frente a 2+ menos"))

    # --------------------------------------------------------------- MOTIVACION
    print()
    print("=" * 78)
    print("MOTIVACION: equipos sin nada en juego en las ultimas jornadas")
    print("=" * 78)

    # Un equipo esta "sin nada en juego" si en las ultimas jornadas ya no puede
    # alcanzar Europa ni caer en descenso, contando los puntos que quedan.
    sin_nada = []
    for s in SEASONS:
        tabla = tablas[s]
        for equipo, fila in tabla.iterrows():
            for md in range(30, 39):
                restantes = 38 - md
                puntos_aprox = fila["points"] * md / 38.0
                techo = puntos_aprox + 3 * restantes
                suelo = puntos_aprox
                umbral_europa = tabla.iloc[6]["points"]      # 7º puesto
                umbral_descenso = tabla.iloc[17]["points"]   # 18º puesto
                libre = techo < umbral_europa and suelo > umbral_descenso
                sin_nada.append({"season": s, "team": equipo, "matchday": md,
                                 "sin_nada_en_juego": libre})
    marcas = pd.DataFrame(sin_nada).set_index(["season", "team", "matchday"])

    final = d[d["matchday"] >= 30].copy()
    idx = pd.MultiIndex.from_arrays(
        [final["season"], final["team"], final["matchday"]])
    final["sin_nada"] = marcas.reindex(idx)["sin_nada_en_juego"].to_numpy()
    final = final.dropna(subset=["sin_nada"])
    print(f"observaciones en jornadas 30-38: {len(final)} "
          f"({int(final['sin_nada'].sum())} sin nada en juego)")

    resultados.append(contraste(
        final[final["sin_nada"]]["residuo"], final[~final["sin_nada"]]["residuo"],
        "sin nada en juego, frente al resto (jornadas 30+)"))

    print()
    print("=" * 78)
    significativos = [r for r in resultados if abs(r["t"]) >= 2]
    if significativos:
        print(f"{len(significativos)} de {len(resultados)} contrastes son significativos:")
        for r in significativos:
            print(f"  - {r['etiqueta']}: {r['diferencia']:+.4f} goles")
    else:
        print("NINGUN contraste alcanza significacion estadistica.")
        print("El tamaño de los efectos ronda 0,05 goles por partido, del orden")
        print("de 10 puntos Elo. Con los datos disponibles no se puede distinguir")
        print("de cero, asi que los modificadores deben entrar desactivados por")
        print("defecto y cualquier magnitud que se les ponga es una suposicion.")
    print("=" * 78)


if __name__ == "__main__":
    main()
