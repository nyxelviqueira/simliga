"""Interfaz de linea de comandos del simulador.

    python -m simliga ingest --desde 2010 --hasta 2025
    python -m simliga calibrar-elo
    python -m simliga backtest --temporadas 2022-23 2023-24 2024-25
    python -m simliga simular --temporada 2024-25 --sims 20000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .config import COMP_LALIGA, COMP_SEGUNDA, OUT_DIR, load_config
from .data import check_season_integrity, load_matches, load_odds
from .db import connect
from .model.dixon_coles import fit_dixon_coles
from .model.elo import EloEngine, calibrate, run_elo
from .output.contract import (build_calendar_block, build_fixtures_block,
                              build_league_block, build_output,
                              build_european_qualification,
                              build_standings_block, write_json)
from .output.dashboard import write_dashboard
from .output.team_identity import display_name
from .sim.league import simulate_league

DATA_SOURCES = [
    "football-data.co.uk (resultados historicos y cuotas de mercado)",
    "openfootball/espana (calendario, CC0)",
]


def _conn_and_names(args):
    conn = connect(args.db) if args.db else connect()
    names = dict(conn.execute("SELECT team_id, name FROM teams").fetchall())
    return conn, names


# --------------------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    from .ingest.football_data_uk import ingest_range

    conn, _ = _conn_and_names(args)
    counts = ingest_range(conn, args.desde, args.hasta, force_download=args.forzar_descarga)
    print(f"Partidos ingeridos: {sum(counts.values())}")
    print(f"Equipos en base de datos: {conn.execute('SELECT COUNT(*) FROM teams').fetchone()[0]}")
    return 0


# ---------------------------------------------------------------------- calibrar-elo
def cmd_calibrate_elo(args) -> int:
    conn, _ = _conn_and_names(args)
    matches = load_matches(conn, (COMP_LALIGA, COMP_SEGUNDA))
    best, grid = calibrate(matches)
    print("Mejor configuracion Elo por error cuadratico medio:")
    print(f"  k_factor          = {best.k_factor}")
    print(f"  home_advantage    = {best.home_advantage}")
    print(f"  season_regression = {best.season_regression}")
    print("\nTop 10 de la rejilla:")
    print(grid.head(10).to_string(index=False))
    print("\nAviso: la superficie es muy plana; conviene confirmar la eleccion")
    print("con el RPS del backtest partido a partido antes de fijar los valores.")
    return 0


# -------------------------------------------------------------------------- backtest
def cmd_backtest(args) -> int:
    from .validation.backtest import backtest_matches, backtest_season

    conn, _ = _conn_and_names(args)
    cfg = load_config(args.config)
    if args.sims:
        cfg.sim.n_sims = args.sims

    liga = load_matches(conn, (COMP_LALIGA,))
    todo = load_matches(conn, (COMP_LALIGA, COMP_SEGUNDA))
    odds = load_odds(conn)

    if args.modo in ("partidos", "ambos"):
        print("=== Prediccion partido a partido ===")
        rows = []
        for season in args.temporadas:
            res = backtest_matches(liga, todo, season, cfg, odds)
            rows.append({"temporada": season, **res.model})
            if res.market:
                rows.append({"temporada": season, **res.market})
            rows.append({"temporada": season, **res.baseline})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.modo in ("temporada", "ambos"):
        print("\n=== Re-simulacion de temporada por jornada de corte ===")
        aggs = []
        for season in args.temporadas:
            aggs.append(backtest_season(liga, todo, season, config=cfg).by_checkpoint)
        agg = pd.concat(aggs)
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nMedia por jornada de corte:")
        print(agg.groupby("matchday").mean(numeric_only=True)
              .to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


# ------------------------------------------------------------------------ copa
def cmd_copa(args) -> int:
    """Registra el campeon de Copa del Rey de una temporada.

    Se introduce a mano porque no hay fuente gratuita de la Copa, y hace falta:
    el campeon se lleva una plaza de Europa League y eso desplaza el reparto que
    sale de la liga.
    """
    from .db import get_cup_winner, set_cup_winner

    conn, _ = _conn_and_names(args)
    if args.ver:
        campeon = get_cup_winner(conn, args.temporada)
        print(f"Copa del Rey {args.temporada}: {campeon or 'sin registrar'}")
        return 0

    fila = conn.execute("SELECT team_id FROM teams WHERE name = ?", (args.campeon,)).fetchone()
    if fila is None:
        print(f"Equipo desconocido: {args.campeon!r}", file=sys.stderr)
        return 1

    set_cup_winner(conn, args.temporada, fila["team_id"])
    print(f"Campeon de Copa {args.temporada}: {args.campeon}")
    print("Vuelve a simular para que se aplique al reparto de plazas europeas.")
    return 0


# ------------------------------------------------------------------- escenario
def cmd_escenario(args) -> int:
    """Fija, quita o lista resultados hipoteticos desde la linea de comandos."""
    from .db import (clear_all_scenarios, clear_scenario_result,
                     load_scenario_results, set_scenario_result)

    conn, names = _conn_and_names(args)

    if args.limpiar:
        print(f"{clear_all_scenarios(conn, args.temporada)} hipotesis borradas.")
        return 0

    if args.listar or not (args.local and args.visitante):
        guardados = load_scenario_results(conn, args.temporada)
        if not guardados:
            print("No hay ninguna hipotesis guardada.")
            return 0
        print(f"{len(guardados)} hipotesis en {args.temporada}:")
        for match_id, (local, visitante) in guardados.items():
            fila = conn.execute(
                """SELECT m.matchday, th.name h, ta.name a FROM matches m
                   JOIN teams th ON th.team_id = m.home_team_id
                   JOIN teams ta ON ta.team_id = m.away_team_id
                   WHERE m.match_id = ?""", (match_id,)).fetchone()
            print(f"  J{fila['matchday']:<3} {fila['h']} {local}-{visitante} {fila['a']}")
        return 0

    fila = conn.execute(
        """SELECT m.match_id FROM matches m
           JOIN teams th ON th.team_id = m.home_team_id
           JOIN teams ta ON ta.team_id = m.away_team_id
           WHERE m.season = ? AND m.competition = ? AND th.name = ? AND ta.name = ?""",
        (args.temporada, COMP_LALIGA, args.local, args.visitante)).fetchone()
    if fila is None:
        print(f"No hay ningun {args.local} - {args.visitante} en {args.temporada}. "
              f"Comprueba la localia.", file=sys.stderr)
        return 1

    if args.quitar:
        borrado = clear_scenario_result(conn, fila["match_id"])
        print("Hipotesis quitada." if borrado else "Ese partido no tenia hipotesis.")
        return 0

    try:
        local, visitante = (int(x) for x in args.marcador.split("-"))
        set_scenario_result(conn, fila["match_id"], local, visitante)
    except ValueError as exc:
        print(f"No se pudo guardar: {exc}", file=sys.stderr)
        return 1
    print(f"Hipotesis guardada: {args.local} {args.marcador} {args.visitante}")
    print("Vuelve a simular para que cuente.")
    return 0


# -------------------------------------------------------------------- servidor
def cmd_servidor(args) -> int:
    """Arranca el panel con servidor local, para poder regenerar desde el boton."""
    from .server import servir

    cfg = load_config(args.config)
    if args.sims:
        cfg.sim.n_sims = args.sims
    servir(args.temporada, puerto=args.puerto, abrir=not args.sin_abrir,
           cfg=cfg, data_sources=DATA_SOURCES,
           host="0.0.0.0" if args.en_red else "127.0.0.1")
    return 0


# --------------------------------------------------------------------------- ajuste
def cmd_ajuste(args) -> int:
    """Registra un ajuste cualitativo: entrenador, lesion, fichaje.

    No hay fuente gratuita fiable de estos eventos, asi que se introducen a
    mano. El signo es el del efecto sobre la fuerza del equipo: negativo para
    una baja importante, positivo para un refuerzo.
    """
    conn, _ = _conn_and_names(args)
    fila = conn.execute("SELECT team_id FROM teams WHERE name = ?", (args.equipo,)).fetchone()
    if fila is None:
        print(f"Equipo desconocido: {args.equipo!r}", file=sys.stderr)
        return 1

    conn.execute(
        """INSERT INTO team_adjustments
               (team_id, season, kind, elo_delta, valid_from, valid_to, note)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (fila["team_id"], args.temporada, args.tipo, args.elo,
         args.desde, args.hasta, args.nota),
    )
    conn.commit()
    print(f"Ajuste registrado: {args.equipo} {args.elo:+.0f} Elo ({args.tipo})")

    cfg = load_config(args.config)
    if args.tipo not in cfg.modifiers.enabled_event_kinds():
        print(f"Aviso: el modificador '{args.tipo}' esta desactivado, asi que este")
        print("ajuste no afectara a la simulacion. Actívalo con un JSON de config:")
        print(f'  {{"modifiers": {{"{args.tipo}_enabled": true}}}}')
    return 0


def cmd_ajustes_listar(args) -> int:
    conn, names = _conn_and_names(args)
    filas = conn.execute(
        """SELECT a.adjustment_id, t.name, a.season, a.kind, a.elo_delta,
                  a.valid_from, a.valid_to, a.note
           FROM team_adjustments a JOIN teams t ON t.team_id = a.team_id
           ORDER BY a.season, t.name"""
    ).fetchall()
    if not filas:
        print("No hay ningun ajuste registrado.")
        return 0
    print(f"{'id':>4}  {'equipo':<24} {'temporada':<10} {'tipo':<14} {'elo':>6}  vigencia")
    for f in filas:
        vig = f"{f['valid_from'] or '...'} -> {f['valid_to'] or '...'}"
        print(f"{f['adjustment_id']:>4}  {f['name']:<24} {f['season']:<10} "
              f"{f['kind']:<14} {f['elo_delta']:>+6.0f}  {vig}  {f['note'] or ''}")
    return 0


# --------------------------------------------------------------------------- simular
def cmd_simulate(args) -> int:
    from .pipeline import (build_context, clasificados_a_europa, european_blocks,
                           real_upcoming_matches, season_split, simulate_laliga)

    conn, _ = _conn_and_names(args)
    cfg = load_config(args.config)
    if args.sims:
        cfg.sim.n_sims = args.sims

    ctx = build_context(conn, cfg, as_of=args.hasta, season=args.temporada)
    try:
        resultado, fit = simulate_laliga(ctx, args.temporada)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    liga = ctx.matches[(ctx.matches["competition"] == COMP_LALIGA)
                       & (ctx.matches["season"] == args.temporada)]
    jugados, pendientes, reales = season_split(ctx, args.temporada)
    proximos_reales = real_upcoming_matches(ctx, args.temporada)

    bloque = build_league_block(resultado, fit, ctx.names, ctx.elo_raw,
                                jugados, pendientes, cfg, real_played=reales)

    # --- competiciones europeas: sorteo real si lo hay, proyeccion si no ---
    europa = {} if args.sin_europa else european_blocks(ctx, args.temporada)

    document = build_output(
        season=args.temporada, league_block=bloque,
        fixtures=build_fixtures_block(fit, proximos_reales, ctx.names,
                                      limit=args.max_fixtures, as_of=ctx.as_of),
        fit=fit, cfg=cfg, as_of=ctx.as_of, data_sources=DATA_SOURCES, europe=europa,
        modifiers=cfg.modifiers.enabled_names(),
        calendar=build_calendar_block(fit, liga, ctx.names, ctx.scenario, ctx.as_of),
        standings=build_standings_block(
            reales, [t['name'] for t in bloque['teams']], cfg),
        european_qualification=clasificados_a_europa(ctx, args.temporada, cfg),
    )
    document["model"]["league_offsets"] = {k: round(v, 1)
                                           for k, v in ctx.league_offsets.items()}
    out_path = args.salida or (OUT_DIR / f"simulacion_{args.temporada}.json")
    write_json(document, out_path)

    panel = getattr(args, "panel", None)
    if panel:
        write_dashboard(document, panel)

    resumen = resultado.summary(ctx.names, cfg.sim)
    resumen["team"] = resumen["team"].map(display_name)
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for comp, bloque_eu in europa.items():
        espanoles = [e for e in bloque_eu["teams"] if e["country"] == "ESP"]
        if not espanoles:
            continue
        print()
        print(f"{bloque_eu['name']} (equipos espanoles):")
        for e in espanoles:
            fases = e["stage_probabilities"]
            nombre = e.get("display_name", e["name"])
            print(f"  {nombre:<22} pos. esperada {e['league_phase']['expected_position']:>5.1f}"
                  f" | directo a octavos {e['league_phase']['p_direct_to_r16']:.0%}"
                  f" | octavos {fases['round_of_16']:.0%}"
                  f" | semis {fases['semi_finals']:.0%}"
                  f" | titulo {fases['winner']:.1%}")

    print()
    print(f"JSON escrito en {out_path}")
    if panel:
        print(f"Panel escrito en {panel} (abrelo con doble clic)")
    return 0


# ------------------------------------------------------------------------ actualizar
def cmd_actualizar(args) -> int:
    """Refresca los datos de la temporada en curso y regenera el JSON.

    Es el comando pensado para ejecutarse en bucle (cron, tarea programada):
    vuelve a bajar lo que puede haber cambiado, simula y deja el JSON listo
    para que lo sirva un frontend.
    """
    from .ingest.football_data_uk import ingest_range
    from .ingest.openfootball import ingest_season as ingest_calendario
    from .ingest.uefa import ingest_range as ingest_uefa

    conn, _ = _conn_and_names(args)
    inicio = int(args.temporada.split("-")[0])

    print(f"[1/5] Resultados de {args.temporada} (football-data.co.uk)")
    try:
        n = sum(ingest_range(conn, inicio, inicio, force_download=True).values())
        print(f"      {n} partidos")
    except Exception as exc:                       # noqa: BLE001
        print(f"      aviso: no disponible ({exc})")

    print(f"[2/5] Calendario de {args.temporada} (openfootball)")
    try:
        print(f"      {ingest_calendario(conn, args.temporada, force_download=True)} partidos")
    except Exception as exc:                       # noqa: BLE001
        print(f"      aviso: no disponible ({exc})")

    print(f"[3/5] Fechas, horarios y resultados recientes (ESPN)")
    try:
        from .ingest.espn import update_schedule

        r = update_schedule(conn, args.temporada, force_download=True)
        print(f"      {r['actualizados']} partidos con fecha y hora")
        if r.get("resultados"):
            print(f"      {r['resultados']} resultado(s) finalizado(s)")
        if r.get("nombres_desconocidos"):
            print(f"      aviso: nombres sin casar: {', '.join(r['nombres_desconocidos'])}")
    except Exception as exc:                       # noqa: BLE001
        print(f"      aviso: no disponible ({exc})")

    print(f"[4/5] Competiciones UEFA de {args.temporada}")
    for clave, n in ingest_uefa(conn, (args.temporada,), force_download=True).items():
        print(f"      {clave}: {n if n else 'sorteo no publicado todavia'}")

    problemas = check_season_integrity(conn, args.temporada)
    if problemas:
        print("  ERROR de integridad tras la ingesta:")
        for problema in problemas:
            print(f"    - {problema}")
        print("  Se aborta antes de simular: un recuento mal solo se veria")
        print("  despues como un resultado raro, no como un fallo.")
        return 1

    print("[5/5] Simulando")
    args.hasta = None
    args.sin_europa = False
    return cmd_simulate(args)


# ------------------------------------------------------------------------ resultado
def cmd_resultado(args) -> int:
    """Registra a mano el resultado de un partido ya disputado.

    football-data.co.uk publica con uno o dos dias de retraso, asi que un
    partido recien terminado no esta en ninguna fuente todavia. Esto permite
    incorporarlo sin esperar; cuando la fuente lo publique, la ingesta normal
    sobrescribira la fila con el mismo resultado.
    """
    conn, _ = _conn_and_names(args)

    try:
        goles_local, goles_visitante = (int(x) for x in args.marcador.split("-"))
    except ValueError:
        print(f"Marcador no valido: {args.marcador!r}. Formato esperado: 2-0",
              file=sys.stderr)
        return 1

    def buscar(nombre: str) -> int | None:
        fila = conn.execute("SELECT team_id FROM teams WHERE name = ?", (nombre,)).fetchone()
        if fila:
            return fila["team_id"]
        # Segunda oportunidad: cualquier alias registrado de cualquier fuente.
        fila = conn.execute(
            "SELECT team_id FROM team_aliases WHERE alias = ? LIMIT 1", (nombre,)).fetchone()
        return fila["team_id"] if fila else None

    local, visitante = buscar(args.local), buscar(args.visitante)
    for nombre, tid in ((args.local, local), (args.visitante, visitante)):
        if tid is None:
            print(f"Equipo desconocido: {nombre!r}", file=sys.stderr)
            parecidos = conn.execute(
                "SELECT name FROM teams WHERE name LIKE ? LIMIT 5",
                (f"%{nombre.split()[0]}%",)).fetchall()
            if parecidos:
                print("  quiza: " + ", ".join(r["name"] for r in parecidos), file=sys.stderr)
            return 1

    partido = conn.execute(
        """SELECT match_id, match_date, status, home_goals, away_goals FROM matches
           WHERE season = ? AND competition = ? AND home_team_id = ? AND away_team_id = ?""",
        (args.temporada, args.competicion, local, visitante),
    ).fetchone()
    if partido is None:
        print(f"No hay ningun {args.local} - {args.visitante} en {args.competicion} "
              f"{args.temporada}. Comprueba la localia.", file=sys.stderr)
        return 1

    if partido["status"] == "played" and not args.forzar:
        print(f"Ese partido ya consta jugado ({partido['home_goals']}-"
              f"{partido['away_goals']}). Usa --forzar para sobrescribirlo.",
              file=sys.stderr)
        return 1

    fecha = args.fecha or partido["match_date"]
    conn.execute(
        """UPDATE matches SET home_goals = ?, away_goals = ?, status = 'played',
                              match_date = ?, source = 'manual'
           WHERE match_id = ?""",
        (goles_local, goles_visitante, fecha, partido["match_id"]),
    )
    conn.commit()
    print(f"Registrado: {args.local} {goles_local}-{goles_visitante} {args.visitante} "
          f"({fecha}, {args.competicion} {args.temporada})")

    jugados = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE season = ? AND competition = ? AND status = 'played'",
        (args.temporada, args.competicion),
    ).fetchone()[0]
    print(f"La temporada tiene ya {jugados} partidos con resultado.")
    return 0


# ----------------------------------------------------------------------- calendario
def cmd_calendario(args) -> int:
    from .ingest.openfootball import ingest_season as ingest_calendario

    conn, _ = _conn_and_names(args)
    n = ingest_calendario(conn, args.temporada, force_download=args.forzar_descarga)
    jugados = conn.execute(
        """SELECT COUNT(*) FROM matches
           WHERE season = ? AND competition = ? AND status = 'played'""",
        (args.temporada, COMP_LALIGA),
    ).fetchone()[0]
    print(f"Calendario de {args.temporada}: {n} partidos ({jugados} ya jugados).")
    return 0


# ------------------------------------------------------------------ calendario-uefa
def cmd_calendario_uefa(args) -> int:
    from .ingest.uefa import ingest_range

    conn, _ = _conn_and_names(args)
    counts = ingest_range(conn, tuple(args.temporadas), force_download=args.forzar_descarga)
    for clave, n in counts.items():
        estado = f"{n} partidos" if n else "no publicado todavia"
        print(f"  {clave:<14} {estado}")
    return 0


# ---------------------------------------------------------------- generar-calendario
def cmd_generate_fixtures(args) -> int:
    from .ingest.fixtures import generate_league_fixtures

    conn, names = _conn_and_names(args)
    ids = []
    for name in args.equipos:
        row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,)).fetchone()
        if row is None:
            print(f"Equipo desconocido: {name}", file=sys.stderr)
            return 1
        ids.append(row[0])

    n = generate_league_fixtures(conn, ids, args.temporada, args.inicio)
    print(f"Calendario sintetico de {args.temporada}: {n} partidos, {len(ids)} equipos.")
    print("Aviso: orden de jornadas aproximado; sustituir por calendario real cuando lo haya.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simliga", description="Simulador Monte Carlo de LaLiga")
    p.add_argument("--db", help="Ruta alternativa de la base de datos SQLite")
    p.add_argument("--config", help="JSON con overrides de configuracion")
    sub = p.add_subparsers(dest="comando", required=True)

    ing = sub.add_parser("ingest", help="Descarga e ingiere resultados historicos")
    ing.add_argument("--desde", type=int, default=2010)
    ing.add_argument("--hasta", type=int, default=2025)
    ing.add_argument("--forzar-descarga", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    cal = sub.add_parser("calibrar-elo", help="Busca en rejilla los parametros del Elo")
    cal.set_defaults(func=cmd_calibrate_elo)

    bt = sub.add_parser("backtest", help="Valida el modelo contra temporadas pasadas")
    bt.add_argument("--temporadas", nargs="+", default=["2022-23", "2023-24", "2024-25"])
    bt.add_argument("--modo", choices=["partidos", "temporada", "ambos"], default="ambos")
    bt.add_argument("--sims", type=int)
    bt.set_defaults(func=cmd_backtest)

    sim = sub.add_parser("simular", help="Simula una temporada y escribe el JSON")
    sim.add_argument("--temporada", required=True)
    sim.add_argument("--hasta", help="Fecha de corte YYYY-MM-DD (por defecto, pre-temporada)")
    sim.add_argument("--sims", type=int)
    sim.add_argument("--salida")
    sim.add_argument("--max-fixtures", type=int, default=20)
    sim.add_argument("--sin-europa", action="store_true",
                     help="Simula solo LaLiga, sin las competiciones UEFA")
    sim.add_argument("--panel", nargs="?", const=str(OUT_DIR / "panel.html"),
                     help="Genera tambien el panel HTML (por defecto out/panel.html)")
    sim.set_defaults(func=cmd_simulate)

    cal_uefa = sub.add_parser(
        "calendario-uefa", help="Descarga el sorteo de las competiciones UEFA")
    cal_uefa.add_argument("--temporadas", nargs="+", required=True)
    cal_uefa.add_argument("--forzar-descarga", action="store_true")
    cal_uefa.set_defaults(func=cmd_calendario_uefa)

    cal_real = sub.add_parser(
        "calendario", help="Descarga el calendario real desde openfootball (sin API key)")
    cal_real.add_argument("--temporada", required=True, help="Por ejemplo 2026-27")
    cal_real.add_argument("--forzar-descarga", action="store_true")
    cal_real.set_defaults(func=cmd_calendario)

    copa = sub.add_parser(
        "copa", help="Registra el campeon de Copa del Rey (afecta a las plazas europeas)")
    copa.add_argument("--temporada", required=True, help="La temporada de la Copa, p. ej. 2025-26")
    copa.add_argument("--campeon")
    copa.add_argument("--ver", action="store_true")
    copa.set_defaults(func=cmd_copa)

    esc = sub.add_parser(
        "escenario", help="Resultados hipoteticos para probar situaciones")
    esc.add_argument("--temporada", required=True)
    esc.add_argument("--local")
    esc.add_argument("--visitante")
    esc.add_argument("--marcador", help="Por ejemplo 2-0")
    esc.add_argument("--quitar", action="store_true", help="Borra la hipotesis de ese partido")
    esc.add_argument("--limpiar", action="store_true", help="Borra todas las de la temporada")
    esc.add_argument("--listar", action="store_true")
    esc.set_defaults(func=cmd_escenario)

    srv = sub.add_parser(
        "servidor", help="Abre el panel con boton para regenerar la simulacion")
    srv.add_argument("--temporada", required=True)
    srv.add_argument("--puerto", type=int, default=8000)
    srv.add_argument("--sims", type=int)
    srv.add_argument("--sin-abrir", action="store_true",
                     help="No abrir el navegador automaticamente")
    srv.add_argument("--en-red", action="store_true",
                     help="Accesible desde otros dispositivos de la red (movil, tablet)")
    srv.set_defaults(func=cmd_servidor)

    aj = sub.add_parser("ajuste", help="Registra un ajuste cualitativo de un equipo")
    aj.add_argument("--equipo", required=True)
    aj.add_argument("--temporada", required=True)
    aj.add_argument("--tipo", required=True,
                    choices=["coach_change", "injuries", "transfers", "squad_depth"])
    aj.add_argument("--elo", type=float, required=True,
                    help="Puntos Elo: negativo debilita, positivo refuerza")
    aj.add_argument("--desde", help="YYYY-MM-DD")
    aj.add_argument("--hasta", help="YYYY-MM-DD")
    aj.add_argument("--nota")
    aj.set_defaults(func=cmd_ajuste)

    ajl = sub.add_parser("ajustes", help="Lista los ajustes registrados")
    ajl.set_defaults(func=cmd_ajustes_listar)

    res = sub.add_parser(
        "resultado", help="Registra a mano un resultado que la fuente aun no publica")
    res.add_argument("--temporada", required=True)
    res.add_argument("--local", required=True)
    res.add_argument("--visitante", required=True)
    res.add_argument("--marcador", required=True, help="Por ejemplo 2-0")
    res.add_argument("--fecha", help="YYYY-MM-DD; por defecto, la del calendario")
    res.add_argument("--competicion", default=COMP_LALIGA)
    res.add_argument("--forzar", action="store_true")
    res.set_defaults(func=cmd_resultado)

    act = sub.add_parser(
        "actualizar",
        help="Refresca datos de la temporada en curso y regenera el JSON (para cron)")
    act.add_argument("--temporada", required=True)
    act.add_argument("--sims", type=int)
    act.add_argument("--salida")
    act.add_argument("--max-fixtures", type=int, default=20)
    act.add_argument("--panel", nargs="?", const=str(OUT_DIR / "panel.html"),
                     help="Genera tambien el panel HTML (por defecto out/panel.html)")
    act.set_defaults(func=cmd_actualizar)

    gen = sub.add_parser("generar-calendario",
                         help="Calendario sintetico (solo si openfootball no cubre la temporada)")
    gen.add_argument("--temporada", required=True)
    gen.add_argument("--inicio", required=True, help="Fecha de la jornada 1 (YYYY-MM-DD)")
    gen.add_argument("--equipos", nargs="+", required=True)
    gen.set_defaults(func=cmd_generate_fixtures)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
