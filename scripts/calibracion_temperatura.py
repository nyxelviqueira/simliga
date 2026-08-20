"""Calibracion por temperatura, elegida dejando fuera la temporada de test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np, pandas as pd
from simliga.db import connect
from simliga.data import load_matches, load_odds
from simliga.config import load_config
from simliga.validation.backtest import backtest_matches
from simliga.validation.metrics import outcome_index, ranked_probability_score, log_loss

SEASONS = ("2022-23", "2023-24", "2024-25")
conn = connect()
liga = load_matches(conn, ("ESP1",)); allm = load_matches(conn, ("ESP1","ESP2")); odds = load_odds(conn)
cfg = load_config()

preds = {}
for s in SEASONS:
    r = backtest_matches(liga, allm, s, cfg, odds)
    p = r.predictions
    preds[s] = (p[["p_home","p_draw","p_away"]].to_numpy(),
                outcome_index(p["home_goals"], p["away_goals"]))

def sharpen(p, t):
    q = np.power(p, t)
    return q / q.sum(axis=1, keepdims=True)

grid = np.arange(0.8, 1.61, 0.02)
print("temp  RPS(las 3 temporadas juntas)")
allp = np.vstack([preds[s][0] for s in SEASONS]); allo = np.concatenate([preds[s][1] for s in SEASONS])
curva = [(t, ranked_probability_score(sharpen(allp,t), allo).mean()) for t in grid]
mejor_insample = min(curva, key=lambda x: x[1])
for t, v in curva[::5]: print(f"{t:.2f}  {v:.5f}")
print(f"optimo in-sample: t={mejor_insample[0]:.2f} RPS={mejor_insample[1]:.5f}")

print("\n--- validacion cruzada dejando una temporada fuera ---")
base_r, cal_r, base_l, cal_l = [], [], [], []
for test in SEASONS:
    train = [s for s in SEASONS if s != test]
    tp = np.vstack([preds[s][0] for s in train]); to = np.concatenate([preds[s][1] for s in train])
    t_opt = min(grid, key=lambda t: ranked_probability_score(sharpen(tp,t), to).mean())
    p, o = preds[test]
    r0 = ranked_probability_score(p, o).mean(); r1 = ranked_probability_score(sharpen(p,t_opt), o).mean()
    l0 = log_loss(p, o).mean(); l1 = log_loss(sharpen(p,t_opt), o).mean()
    base_r.append(r0); cal_r.append(r1); base_l.append(l0); cal_l.append(l1)
    print(f"{test}: t elegido en las otras dos = {t_opt:.2f} | RPS {r0:.5f} -> {r1:.5f} "
          f"({(r0-r1)/r0:+.2%}) | logloss {l0:.4f} -> {l1:.4f}")
print(f"\nMEDIA fuera de muestra: RPS {np.mean(base_r):.5f} -> {np.mean(cal_r):.5f} "
      f"({(np.mean(base_r)-np.mean(cal_r))/np.mean(base_r):+.2%}) | "
      f"logloss {np.mean(base_l):.4f} -> {np.mean(cal_l):.4f}")
