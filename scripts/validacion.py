"""Validacion final con los parametros por defecto; escribe out/validacion.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, time
import numpy as np, pandas as pd
from simliga.db import connect
from simliga.data import load_matches, load_odds
from simliga.config import load_config
from simliga.validation.backtest import backtest_matches, backtest_season

SEASONS = ("2022-23", "2023-24", "2024-25")
conn = connect()
liga = load_matches(conn, ("ESP1",)); allm = load_matches(conn, ("ESP1","ESP2")); odds = load_odds(conn)
cfg = load_config(); cfg.sim.n_sims = 20000

report = {"seasons": list(SEASONS), "config": cfg.to_dict(), "matches": {}, "season": {}}
calib_all = []
for s in SEASONS:
    r = backtest_matches(liga, allm, s, cfg, odds)
    report["matches"][s] = {"modelo": r.model, "mercado": r.market, "tasa_base": r.baseline}
    calib_all.append(r.calibration)
    print(f"{s}: modelo RPS {r.model['rps']:.4f} | mercado {r.market['rps']:.4f} | "
          f"base {r.baseline['rps']:.4f}", flush=True)

m = [report["matches"][s]["modelo"]["rps"] for s in SEASONS]
k = [report["matches"][s]["mercado"]["rps"] for s in SEASONS]
b = [report["matches"][s]["tasa_base"]["rps"] for s in SEASONS]
report["matches"]["global"] = {"rps_modelo": np.mean(m), "rps_mercado": np.mean(k),
    "rps_tasa_base": np.mean(b),
    "pct_del_camino_base_a_mercado": (np.mean(b)-np.mean(m))/(np.mean(b)-np.mean(k))}
print(f"\nGLOBAL modelo {np.mean(m):.4f} | mercado {np.mean(k):.4f} | base {np.mean(b):.4f}")
print(f"El modelo cubre el {report['matches']['global']['pct_del_camino_base_a_mercado']:.1%} "
      f"del camino entre la tasa base y el mercado de cierre.")

aggs = []
for s in SEASONS:
    r = backtest_season(liga, allm, s, config=cfg)
    aggs.append(r.by_checkpoint)
agg = pd.concat(aggs)
report["season"]["por_temporada"] = json.loads(agg.to_json(orient="records"))
media = agg.groupby("matchday").mean(numeric_only=True).reset_index()
report["season"]["media_por_jornada"] = json.loads(media.to_json(orient="records"))
print("\n" + media.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

calib = pd.concat(calib_all).groupby("bin").apply(
    lambda g: pd.Series({"n": g["n"].sum(),
                         "predicha": np.average(g["predicted"], weights=g["n"]),
                         "observada": np.average(g["observed"], weights=g["n"])}),
    include_groups=False).reset_index()
report["calibration"] = json.loads(calib.to_json(orient="records"))
print("\nCalibracion 1X2 agregada (3 temporadas, 3.420 probabilidades):")
print(calib.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

with open("out/validacion.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=float)
print("\nInforme escrito en out/validacion.json")


# ---------------------------------------------------------------- competiciones UEFA
from simliga.validation.uefa_backtest import backtest_uefa      # noqa: E402

print()
print("=== COMPETICIONES EUROPEAS ===")
casos = [("2024-25", "UCL"), ("2025-26", "UCL"), ("2024-25", "UEL"), ("2024-25", "UECL")]
resumenes, rondas = [], []
for temporada, comp in casos:
    r = backtest_uefa(temporada, comp, cfg)
    resumenes.append(r.resumen)
    rondas.append(r.por_ronda.assign(season=temporada, competition=comp))
    print(f"  {comp} {temporada}: Brier {r.resumen['brier_modelo']:.4f} "
          f"vs bombo {r.resumen['brier_bombo']:.4f} "
          f"({r.resumen['mejora_sobre_bombo']:+.1%})", flush=True)

eu = pd.DataFrame(resumenes)
report["europe"] = {
    "por_torneo": json.loads(eu.to_json(orient="records")),
    "por_ronda": json.loads(pd.concat(rondas).to_json(orient="records")),
    "mejora_media_sobre_bombo": float(eu["mejora_sobre_bombo"].mean()),
}
print()
print(f"Mejora media sobre el bombo: {eu['mejora_sobre_bombo'].mean():.1%}")
print(pd.concat(rondas).groupby("ronda", sort=False)[
    ["brier_modelo", "brier_bombo", "prob_media_acertados", "prob_media_resto"]
].mean().to_string(float_format=lambda v: f"{v:.4f}"))

with open("out/validacion.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=float)
print()
print("Informe actualizado en out/validacion.json")
