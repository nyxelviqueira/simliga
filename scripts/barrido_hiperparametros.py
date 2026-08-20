"""Barrido de hiperparametros contra el RPS partido a partido de 3 temporadas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import itertools, time
import pandas as pd
from simliga.db import connect
from simliga.data import load_matches, load_odds
from simliga.config import load_config
from simliga.validation.backtest import backtest_matches

SEASONS = ("2022-23", "2023-24", "2024-25")
conn = connect()
liga = load_matches(conn, ("ESP1",))
allm = load_matches(conn, ("ESP1", "ESP2"))
odds = load_odds(conn)

grid = list(itertools.product([120, 180, 240, 365], [4.0, 12.0, 30.0], [10.0, 20.0], [0.85]))
rows = []
for hl, pw, k, phi in grid:
    cfg = load_config()
    cfg.dixon_coles.half_life_days = hl
    cfg.dixon_coles.elo_prior_weight = pw
    cfg.elo.k_factor = k
    cfg.elo.season_regression = phi
    rps = []; ll = []; mkt = []
    for s in SEASONS:
        r = backtest_matches(liga, allm, s, cfg, odds)
        rps.append(r.model["rps"]); ll.append(r.model["log_loss"])
        if r.market: mkt.append(r.market["rps"])
    rows.append({"half_life": hl, "prior_w": pw, "k": k, "phi": phi,
                 "rps": sum(rps)/len(rps), "log_loss": sum(ll)/len(ll),
                 "rps_mercado": sum(mkt)/len(mkt) if mkt else None})
    print(f"hl={hl:>3} pw={pw:>4} k={k:>4} -> RPS {rows[-1]['rps']:.5f}", flush=True)

df = pd.DataFrame(rows).sort_values("rps")
df.to_csv("out_sweep.csv", index=False)
print("\n=== MEJORES ===")
print(df.head(8).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
