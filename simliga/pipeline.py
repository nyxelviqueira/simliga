"""Ensamblaje del modelo: de la base de datos a una simulacion.

Este modulo es el punto de entrada unico. Todo lo que quiera usar el simulador
(la linea de comandos, los tests, un servidor web) construye primero un
`ModelContext` con los datos hasta una fecha y despues pide simulaciones.

El orden importa y es siempre el mismo:

    datos -> Elo (pool multiliga) -> desplazamiento por liga -> Dixon-Coles -> Monte Carlo

El desplazamiento por liga se calcula aqui y no dentro del Elo porque necesita
los partidos europeos ya procesados: es una correccion sobre el rating, no parte
de su actualizacion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (COMP_LALIGA, COMP_SEGUNDA, EUROPEAN_LEAGUES, UEFA_COMPETITIONS,
                     Config, DixonColesConfig, load_config)
from .data import load_matches, promoted_into
from .db import connect, get_cup_winner, load_scenario_results, previous_season
from .model.dixon_coles import DixonColesFit, fit_dixon_coles
from .output.contract import build_european_qualification
from .model.elo import EloEngine
from .model.modifiers import (build_match_adjustments, compute_rest_days,
                              elo_delta_to_rate_shift, load_team_adjustments)
from .model.league_strength import (apply_offsets, build_uefa_training_frame,
                                    estimate_league_offsets, team_league_map)
from .sim.league import LeagueSimResult, simulate_league
from .sim.uefa import UefaSimResult, simulate_uefa, simulate_uefa_over_draws

ALL_COMPETITIONS = (COMP_LALIGA, COMP_SEGUNDA) + EUROPEAN_LEAGUES + UEFA_COMPETITIONS

# El ajuste europeo necesita un prior mas fuerte que el domestico: un equipo
# juega 8 partidos de Champions al año frente a 38 de liga, asi que su ataque y
# su defensa se estiman casi enteramente desde el Elo.
UEFA_DC_CONFIG = DixonColesConfig(
    half_life_days=365.0, elo_prior_weight=40.0, max_history_days=1100,
)


@dataclass
class ModelContext:
    """Estado del modelo a una fecha: ratings, desplazamientos y datos crudos."""

    cfg: Config
    as_of: pd.Timestamp
    matches: pd.DataFrame               # todo, incluidos partidos sin jugar
    names: dict[int, str]
    countries: dict[int, str]
    elo: dict[int, float]               # ya ajustado por fuerza de liga
    elo_raw: dict[int, float]
    league_offsets: dict[str, float]
    team_league: dict[int, str]
    promoted_teams: set = field(default_factory=set)
    team_adjustments: pd.DataFrame | None = None
    # {match_id: (goles_local, goles_visitante)} de los partidos hipoteticos.
    scenario: dict = field(default_factory=dict)
    _fits: dict = field(default_factory=dict)

    def teams_of(self, season: str, competition: str, stage: str | None = None) -> list[int]:
        sub = self.matches[(self.matches["season"] == season)
                           & (self.matches["competition"] == competition)]
        if stage:
            sub = sub[sub["stage"] == stage]
        return sorted(set(sub["home_team_id"]) | set(sub["away_team_id"]))

    def spanish(self, team_ids: list[int]) -> list[int]:
        return [t for t in team_ids if self.countries.get(t) == "ESP"]


def build_context(
    conn=None,
    cfg: Config | None = None,
    as_of: str | pd.Timestamp | None = None,
    season: str | None = None,
) -> ModelContext:
    """Construye el estado del modelo usando solo lo ocurrido antes de `as_of`."""
    conn = conn or connect()
    cfg = cfg or load_config()

    matches = load_matches(conn, ALL_COMPETITIONS, only_played=False)
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
    else:
        # El dia siguiente al ultimo partido con resultado. Tomar el maximo del
        # calendario fecharia el corte en mayo del año que viene: los resultados
        # contarian igual, pero el JSON anunciaria una fecha que no es.
        jugados = matches[matches["home_goals"].notna()]["match_date"]
        as_of = (jugados.max() + pd.Timedelta(days=1) if len(jugados)
                 else matches["match_date"].min())

    historia = matches[(matches["match_date"] < as_of) & matches["home_goals"].notna()]
    engine = EloEngine(cfg.elo).run(historia)

    team_league = team_league_map(historia)
    uefa = historia[historia["competition"].isin(UEFA_COMPETITIONS)]
    entrenamiento = build_uefa_training_frame(uefa, engine.history_frame(), team_league)
    offsets = estimate_league_offsets(entrenamiento, home_advantage=cfg.elo.home_advantage)

    # Los recien ascendidos se identifican con los partidos, no con el estado
    # del Elo: es un dato de calendario, no de rating.
    temporada = season or matches[matches["competition"] == COMP_LALIGA]["season"].max()
    ascendidos = promoted_into(matches, temporada, COMP_LALIGA)
    elo_raw = engine.ratings_at_cutoff(promoted_teams=ascendidos)
    return ModelContext(
        cfg=cfg, as_of=as_of, matches=matches,
        names=dict(conn.execute("SELECT team_id, name FROM teams").fetchall()),
        countries=dict(conn.execute("SELECT team_id, country FROM teams").fetchall()),
        elo=apply_offsets(elo_raw, team_league, offsets),
        elo_raw=elo_raw,
        league_offsets=offsets,
        team_league=team_league,
        promoted_teams=ascendidos,
        team_adjustments=load_team_adjustments(conn),
        scenario=load_scenario_results(conn),
    )


def _split(ctx: ModelContext, sub: pd.DataFrame):
    """Separa una competicion en (jugados, pendientes), aplicando el escenario.

    Un partido con resultado hipotetico pasa al lado de los jugados con ese
    marcador: la simulacion lo da por hecho y sus puntos ya cuentan, que es
    justo lo que se quiere al probar un "y si...".

    Lo que NO hace es tocar la fuerza de los equipos. El ajuste Dixon-Coles y el
    Elo se calculan sobre `ctx.matches`, que sale de la base de datos sin
    escenarios, asi que ganar cuatro partidos imaginarios da puntos pero no
    convierte a nadie en mejor equipo. Es la lectura correcta de la pregunta:
    "si pasa esto, como queda la tabla", no "aprende de esto".
    """
    reales = sub[(sub["match_date"] < ctx.as_of) & sub["home_goals"].notna()]
    pendientes = sub[~sub.index.isin(reales.index)]
    if not ctx.scenario:
        return reales, pendientes

    marcados = pendientes["match_id"].isin(ctx.scenario)
    if not marcados.any():
        return reales, pendientes

    hipoteticos = pendientes[marcados].copy()
    hipoteticos["home_goals"] = hipoteticos["match_id"].map(
        lambda m: ctx.scenario[m][0])
    hipoteticos["away_goals"] = hipoteticos["match_id"].map(
        lambda m: ctx.scenario[m][1])

    return pd.concat([reales, hipoteticos]), pendientes[~marcados]


def season_split(ctx: ModelContext, season: str, competition: str = COMP_LALIGA):
    """(jugados incluyendo escenario, pendientes, jugados de verdad).

    Los tres hacen falta: la simulacion arranca de los primeros, el JSON tiene
    que poder distinguir cuantos puntos son reales y cuantos hipoteticos.
    """
    sub = ctx.matches[(ctx.matches["competition"] == competition)
                      & (ctx.matches["season"] == season)]
    jugados, pendientes = _split(ctx, sub)
    reales = sub[(sub["match_date"] < ctx.as_of) & sub["home_goals"].notna()]
    return jugados, pendientes, reales


def real_upcoming_matches(
    ctx: ModelContext,
    season: str,
    competition: str = COMP_LALIGA,
) -> pd.DataFrame:
    """Partidos no jugados de verdad, aunque tengan un escenario encima.

    Para simular, un escenario se retira de pendientes porque ya cuenta como
    fijo. Para agenda y fichas no: el partido real sigue siendo el siguiente
    partido del equipo hasta que tenga marcador de verdad en la base.
    """
    sub = ctx.matches[(ctx.matches["competition"] == competition)
                      & (ctx.matches["season"] == season)]
    reales = sub[(sub["match_date"] < ctx.as_of) & sub["home_goals"].notna()]
    return sub[~sub.index.isin(reales.index)]


def laliga_fit(ctx: ModelContext, season: str) -> DixonColesFit:
    """Ajuste Dixon-Coles para LaLiga, con el Elo domestico como prior."""
    clave = ("ESP1", season)
    if clave in ctx._fits:
        return ctx._fits[clave]

    liga = ctx.matches[ctx.matches["competition"] == COMP_LALIGA]
    equipos = ctx.teams_of(season, COMP_LALIGA)
    fit = fit_dixon_coles(
        liga[liga["home_goals"].notna()], ctx.elo_raw, cutoff=ctx.as_of,
        teams=equipos, config=ctx.cfg.dixon_coles,
    )
    ctx._fits[clave] = fit
    return fit


def uefa_fit(ctx: ModelContext, teams: list[int]) -> DixonColesFit:
    """Ajuste Dixon-Coles europeo, con el Elo ya comparable entre ligas.

    Se ajusta con todos los equipos que tengan historia europea reciente, no
    solo con los participantes: los parametros globales (tasa base de goles,
    ventaja de campo, rho) se estiman mucho mejor con 1.300 partidos que con
    los 200 que se juegan entre los 36 de este año.
    """
    clave = ("UEFA", tuple(teams))
    if clave in ctx._fits:
        return ctx._fits[clave]

    historico = ctx.matches[
        ctx.matches["competition"].isin(UEFA_COMPETITIONS)
        & (ctx.matches["match_date"] < ctx.as_of)
        & ctx.matches["home_goals"].notna()
    ]
    con_historia = sorted(
        set(historico["home_team_id"]) | set(historico["away_team_id"]) | set(teams)
    )
    completo = fit_dixon_coles(historico, ctx.elo, cutoff=ctx.as_of,
                               teams=con_historia, config=UEFA_DC_CONFIG)
    fit = completo.subset(teams)
    ctx._fits[clave] = fit
    return fit


def match_rate_shifts(ctx: ModelContext, fixtures: pd.DataFrame, season: str):
    """Desplazamiento de goles esperados por partido segun los modificadores.

    Devuelve (shift_local, shift_visitante, desglose) o (None, None, None) si no
    hay ningun modificador activo, que es el caso por defecto.
    """
    cfg = ctx.cfg.modifiers
    if not cfg.any_enabled() or fixtures.empty:
        return None, None, None

    rest = compute_rest_days(ctx.matches) if cfg.fatigue_enabled else None
    eventos = ctx.team_adjustments
    if eventos is not None and not eventos.empty:
        eventos = eventos[eventos["season"] == season]

    ajustes = build_match_adjustments(fixtures, cfg, rest=rest, adjustments=eventos)
    if ajustes.is_empty():
        return None, None, ajustes.detail

    fit = laliga_fit(ctx, season)
    # Los goles del local suben con su Elo ajustado y bajan con el del visitante.
    shift_home = elo_delta_to_rate_shift(ajustes.home, -ajustes.away,
                                         fit.kappa_attack, fit.kappa_defence)
    shift_away = elo_delta_to_rate_shift(ajustes.away, -ajustes.home,
                                         fit.kappa_attack, fit.kappa_defence)
    return shift_home, shift_away, ajustes.detail


def simulate_laliga(ctx: ModelContext, season: str) -> tuple[LeagueSimResult, DixonColesFit]:
    sub = ctx.matches[(ctx.matches["competition"] == COMP_LALIGA)
                      & (ctx.matches["season"] == season)]
    if sub.empty:
        raise ValueError(f"No hay partidos de LaLiga {season}")
    jugados, pendientes = _split(ctx, sub)
    fit = laliga_fit(ctx, season)
    shift_home, shift_away, _ = match_rate_shifts(ctx, pendientes, season)
    resultado = simulate_league(fit, pendientes, played=jugados,
                                teams=ctx.teams_of(season, COMP_LALIGA), config=ctx.cfg.sim,
                                rate_shift_home=shift_home, rate_shift_away=shift_away)
    return resultado, fit


def simulate_european(
    ctx: ModelContext, season: str, competition: str
) -> tuple[UefaSimResult, DixonColesFit]:
    """Simula una competicion UEFA completa desde la liguilla."""
    sub = ctx.matches[(ctx.matches["competition"] == competition)
                      & (ctx.matches["season"] == season)
                      & (ctx.matches["stage"] == "league_phase")]
    if sub.empty:
        raise ValueError(
            f"No hay liguilla de {competition} {season}. El sorteo puede no estar "
            f"publicado todavia: `simliga calendario-uefa --temporada {season}`."
        )

    equipos = sorted(set(sub["home_team_id"]) | set(sub["away_team_id"]))
    jugados, pendientes = _split(ctx, sub)
    fit = uefa_fit(ctx, equipos)
    resultado = simulate_uefa(fit, pendientes, played=jugados, teams=equipos,
                              config=ctx.cfg.sim,
                              rng=np.random.default_rng(ctx.cfg.sim.seed + 7))
    return resultado, fit


def available_european(ctx: ModelContext, season: str) -> list[str]:
    """Competiciones UEFA con liguilla cargada para esa temporada."""
    sub = ctx.matches[(ctx.matches["season"] == season)
                      & (ctx.matches["stage"] == "league_phase")]
    return sorted(set(sub["competition"]))


def clasificados_a_europa(ctx, season: str, cfg: Config) -> dict | None:
    """Equipos espanoles en Europa, segun la liga de la temporada anterior."""
    anterior = previous_season(season)
    previos = ctx.matches[(ctx.matches["competition"] == COMP_LALIGA)
                          & (ctx.matches["season"] == anterior)
                          & ctx.matches["home_goals"].notna()]
    return build_european_qualification(
        previos, anterior, cfg, cup_winner=get_cup_winner(connect(), anterior))


# ------------------------------------------------- proyeccion europea sin sorteo
COMPETICION_POR_CODIGO = {"UCL": "UCL", "UEL": "UEL", "UECL": "UECL"}


def european_field(ctx: ModelContext, season: str, competition: str) -> list[int] | None:
    """Campo estimado de una competicion UEFA cuando aun no hay sorteo.

    Se parte de los participantes REALES de la temporada pasada y se sustituye
    el contingente espanol por el de este año. El resto del campo cambia poco de
    un año a otro, y lo que se quiere medir -hasta donde llega un equipo
    espanol- depende sobre todo de su propia fuerza frente a un campo tipico.

    Es una estimacion, no el campo definitivo, y se etiqueta como tal.
    """
    # Se busca hacia atras la ultima temporada con liguilla de esa competicion:
    # openfootball no publico la de Europa League y Conference de 2025-26, y
    # quedarse solo con la temporada inmediatamente anterior dejaria a esas dos
    # sin proyeccion por un hueco de la fuente.
    liguillas = ctx.matches[(ctx.matches["competition"] == competition)
                            & (ctx.matches["stage"] == "league_phase")
                            & (ctx.matches["season"] < season)]
    if liguillas.empty:
        return None
    anterior = liguillas["season"].max()
    previos = liguillas[liguillas["season"] == anterior]

    campo = sorted(set(previos["home_team_id"]) | set(previos["away_team_id"]))
    clasificados = clasificados_a_europa(ctx, season, ctx.cfg)
    if not clasificados:
        return None

    nombre_a_id = {nombre: tid for tid, nombre in ctx.names.items()}
    espanoles_nuevos = [nombre_a_id[t["team"]] for t in clasificados["teams"]
                        if t["competition"] == competition and t["team"] in nombre_a_id]
    espanoles_viejos = [t for t in campo if ctx.countries.get(t) == "ESP"]

    campo = [t for t in campo if t not in espanoles_viejos]
    campo += [t for t in espanoles_nuevos if t not in campo]

    # El formato exige 36 exactos: se recorta o se completa con los equipos de
    # mas rating que jugaron Europa el año pasado y no estan ya dentro.
    if len(campo) > 36:
        campo = sorted(campo, key=lambda t: -ctx.elo.get(t, 0))[:36]
    elif len(campo) < 36:
        europeos = ctx.matches[
            ctx.matches["competition"].isin(UEFA_COMPETITIONS)
            & (ctx.matches["season"] == anterior)]
        candidatos = sorted(
            (set(europeos["home_team_id"]) | set(europeos["away_team_id"])) - set(campo),
            key=lambda t: -ctx.elo.get(t, 0))
        campo += candidatos[:36 - len(campo)]

    return sorted(campo) if len(campo) == 36 else None


def simulate_european_provisional(
    ctx: ModelContext, season: str, competition: str, n_draws: int = 25
):
    """Proyeccion de una competicion UEFA sin sorteo publicado.

    Devuelve (resultado, ajuste, campo, temporada_del_campo) o None.
    Promedia sobre varios sorteos porque, sin bombos hechos, la suerte del
    cruce es una fuente de incertidumbre tan real como el propio juego.
    """
    campo = european_field(ctx, season, competition)
    if campo is None:
        return None

    liguillas = ctx.matches[(ctx.matches["competition"] == competition)
                            & (ctx.matches["stage"] == "league_phase")
                            & (ctx.matches["season"] < season)]
    base = liguillas["season"].max()

    fit = uefa_fit(ctx, campo)
    inicio = pd.Timestamp(f"{season.split('-')[0]}-09-15")
    resultado = simulate_uefa_over_draws(
        fit, campo, ctx.elo, season, competition, inicio, ctx.cfg.sim, n_draws=n_draws)
    return resultado, fit, campo, base


def european_blocks(ctx: ModelContext, season: str) -> dict:
    """Bloques de las tres competiciones UEFA para el JSON.

    Usa el sorteo real si esta publicado y, si no, la proyeccion sobre un campo
    estimado, marcada como provisional. Vive aqui y no en la linea de comandos
    para que el panel servido y el generado en fichero digan exactamente lo
    mismo.
    """
    from .output.contract import build_european_block

    bloques = {}
    con_sorteo = set(available_european(ctx, season))

    for comp in UEFA_COMPETITIONS:
        if comp in con_sorteo:
            resultado, _ = simulate_european(ctx, season, comp)
            sub = ctx.matches[(ctx.matches["competition"] == comp)
                              & (ctx.matches["season"] == season)
                              & (ctx.matches["stage"] == "league_phase")]
            jugados = sub[(sub["match_date"] < ctx.as_of) & sub["home_goals"].notna()]
            bloques[comp] = build_european_block(
                resultado, ctx.names, ctx.countries, ctx.elo, jugados,
                sub[~sub.index.isin(jugados.index)], comp)
            continue

        proyeccion = simulate_european_provisional(ctx, season, comp)
        if proyeccion is None:
            continue
        resultado, _, campo, base = proyeccion
        vacio = ctx.matches.iloc[:0]
        bloques[comp] = build_european_block(
            resultado, ctx.names, ctx.countries, ctx.elo, vacio, vacio, comp,
            provisional={
                "field_based_on": base,
                "draws_simulated": 25,
                "reason": "El sorteo de la fase de liga aun no esta publicado.",
                "method": ("Campo estimado: los participantes reales de " + base +
                           " con el contingente espanol sustituido por el de esta "
                           "temporada. El emparejamiento se sortea y se promedia "
                           "sobre 25 sorteos distintos."),
                # Las probabilidades arrancan en la fase de liga. Quien tenga que
                # jugar una ronda previa (el caso habitual del representante en
                # Conference) tiene ademas que superarla, y eso no se simula.
                "assumes_league_phase": True,
                "caveat": ("Las cifras parten de la fase de liga: si al equipo le "
                           "toca ronda previa, el modelo no la simula y sus "
                           "opciones reales son menores."),
            })
    return bloques
