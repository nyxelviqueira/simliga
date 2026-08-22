# Contrato de datos — salida del simulador

**Versión de esquema:** `1.11.0` · **Formato:** un único fichero JSON, UTF-8

Este documento describe el JSON que produce el motor de simulación. Está pensado
para que se pueda construir un dashboard **sin leer el código del motor**. Si
algo no está descrito aquí, no forma parte del contrato y puede cambiar.

Se genera con:

```bash
.venv/Scripts/python.exe -m simliga simular --temporada 2024-25 --sims 20000 --salida out/simulacion.json
```

---

## 1. Reglas generales

| Regla | Detalle |
|---|---|
| Codificación | UTF-8, nombres de equipo en texto plano |
| Fechas | `YYYY-MM-DD`; marcas de tiempo en ISO 8601 UTC (`...Z`) |
| Probabilidades | Float en `[0, 1]`, redondeado a 5 decimales. **Nunca porcentajes** |
| Nulos | `null` explícito; una clave nunca desaparece según el caso |
| Identidad de equipo | `team_id` (entero estable) es la clave; `name` es el nombre de fuente; `display_name` es para mostrar |
| Ordenación | `competitions.*.teams` viene ordenado por posición esperada ascendente |
| Versionado | `schema_version` sigue semver: un cambio menor añade claves, uno mayor las quita o cambia su significado |

**Garantías numéricas** (se pueden usar como aserciones en el frontend):

- Para cada equipo, `position_probabilities` suma 1,0 (±1e-4).
- Para cada posición, la suma sobre todos los equipos de la probabilidad de esa
  posición es 1,0 (±1e-4). Es decir, la matriz es doblemente estocástica.
- `outcomes.title` == `position_probabilities[0]`.
- Las probabilidades vienen de un Monte Carlo: con `n_sims = 20000` el error
  típico de una probabilidad cercana a 0,5 es de ±0,004. **No mostrar más de
  un decimal en porcentaje**, y tratar como cero cualquier valor por debajo de
  `1 / n_sims`.

---

## 2. Estructura de alto nivel

```jsonc
{
  "schema_version": "1.11.0",
  "engine_version": "0.7.0",
  "generated_at": "2026-08-19T21:14:05Z",
  "season": "2024-25",
  "as_of": { ... },          // hasta dónde ha visto resultados reales
  "simulation": { ... },     // parámetros de la tirada Monte Carlo
  "model": { ... },          // parámetros del modelo estadístico
  "competitions": { ... },   // una entrada por competición, clave = código
  "standings": { ... },      // clasificación real, solo resultados de verdad
  "european_qualification": { ... },  // quién juega Europa y en qué competición
  "calendar": { ... },       // calendario por jornadas y escenarios
  "fixtures": [ ... ],       // próximos partidos con sus probabilidades
  "validation": null,        // informe de backtest, o null
  "meta": { ... }
}
```

### `as_of`

Punto temporal de la foto. Con `matches_played: 0` el documento es una
proyección de pretemporada; si no, es una proyección con temporada en curso.

```jsonc
"as_of": {
  "date": "2024-08-15",
  "matches_played": 0,        // partidos de liga ya incorporados con su resultado real
  "matches_remaining": 380    // partidos que se han simulado
}
```

### `simulation`

```jsonc
"simulation": {
  "n_sims": 20000,
  "seed": 20262027,           // misma semilla + mismos datos = mismo resultado
  "modifiers_enabled": []     // ajustes cualitativos activos (ver §6)
}
```

### `model`

Metadatos del ajuste. Útiles para una pestaña de "cómo se ha calculado"; el
dashboard no debería depender de ellos para nada funcional.

```jsonc
"model": {
  "type": "dixon-coles-poisson",
  "elo": { "k_factor": 10.0, "home_advantage_elo": 70.0, "season_regression": 0.85 },
  "dixon_coles": {
    "half_life_days": 365.0,
    "elo_prior_weight": 12.0,
    "mu": 0.0831,               // log de la tasa base de goles
    "home_advantage": 0.3124,   // en log-goles: exp(0.31) = x1.37 al equipo local
    "rho": 0.0103,              // corrección de marcadores bajos
    "fitted_on_matches": 906,
    "effective_sample_size": 344.2   // partidos equivalentes tras el peso temporal
  }
}
```

---

## 3. `competitions`

Diccionario cuya clave es el código de competición. En la fase 1 solo existe
`ESP1`. Cada entrada declara su `type`, y **el frontend debe ramificar por ese
campo**, no por el código:

| `type` | Significado | Estado |
|---|---|---|
| `league` | Liga a doble vuelta con clasificación única | implementado |
| `uefa_league_phase` | Liguilla única UEFA (36 equipos) + playoff y eliminatorias | implementado |

Un documento puede traer solo `ESP1`, o `ESP1` más una o varias competiciones
europeas, según haya sorteo cargado para esa temporada. **El frontend debe
iterar sobre las claves de `competitions`, no asumir cuáles hay.**

Códigos: `ESP1` (LaLiga), `ESP2` (Hypermotion), `UCL`, `UEL`, `UECL`.

### 3.1 Bloque de tipo `league`

```jsonc
{
  "code": "ESP1",
  "type": "league",
  "name": "LaLiga EA Sports",
  "n_teams": 20,
  "qualification_rules": {      // rangos de posición [desde, hasta], ambos incluidos
    "ucl":        [1, 5],       // 5 con la plaza extra por coeficiente UEFA (EPS)
    "uel":        [6, 6],       // la otra UEL es la del campeón de Copa, que va aparte
    "uecl":       [7, 7],
    "relegation": [18, 20]
  },
  "qualification_note": "Reparto europeo provisional: ...",
  "matches_played": 0,
  "matches_remaining": 380,
  "teams": [ /* ver 3.2 */ ]
}
```

`qualification_rules` es la **fuente de verdad base para pintar las bandas de
color** de la tabla. No se deben cablear las posiciones en el frontend: la quinta
plaza de Champions depende del EPS por coeficiente UEFA y la segunda plaza de
Europa League puede venir por Copa del Rey. El panel permite activar esos dos
supuestos con checks y recalcula las probabilidades europeas desde
`position_probabilities`.

### 3.2 Objeto de equipo

```jsonc
{
  "team_id": 12,
  "name": "Real Madrid",
  "display_name": "Real Madrid",
  "logo": "assets/escudos/real-madrid.png",

  "current": {                  // situación real a fecha `as_of`, sin simular
    "played": 0,
    "points": 0,
    "goals_for": 0,
    "goals_against": 0,
    "goal_difference": 0,
    "position": null            // null si aún no se ha jugado nada
  },

  "ratings": {
    "elo": 1759.3,              // fuerza dinámica; ~1500 es la media histórica
    "attack": 0.5378,           // log-multiplicador de goles marcados
    "defence": 0.3226           // log-multiplicador de goles evitados (más alto = mejor)
  },

  "projection": {
    "points":          { "mean": 81.81, "sd": 7.04, "p05": 70, "p25": 77, "median": 82, "p75": 87, "p95": 93 },
    "goal_difference": { "mean": 47.44, "sd": 10.53, "p05": 30, "p25": 40, "median": 47, "p75": 54, "p95": 65 },
    "position":        { "mean": 1.53, "mode": 1, "p05": 1, "p95": 3 },
    "position_probabilities": [0.644, 0.2404, /* ... */ 0.0]
  },

  "outcomes": {
    "title": 0.644,
    "ucl": 0.9968,
    "uel": 0.003,
    "uecl": 0.0002,
    "european_qualification": 1.0,
    "relegation": 0.0
  }
}
```

**`position_probabilities`** es un array de longitud `n_teams`. El índice `i`
(base 0) es la probabilidad de terminar en la posición `i + 1`. Es el dato
central para el gráfico habitual de "mapa de calor de la tabla".

**`display_name` y `logo`** son campos de presentación. `logo` es una ruta a un
asset local que se copia junto al panel estático y se sirve desde
`simliga servidor`; no debe depender de una URL externa en tiempo de uso. La
identidad estable sigue siendo `team_id`; `name` conserva el nombre usado
internamente para casar fuentes y no debe usarse cuando se quiera enseñar el
equipo al usuario.

**`ratings.attack` / `ratings.defence`** están centrados en 0 y en escala
logarítmica: un ataque de `+0.54` significa que ese equipo marca `exp(0.54) ≈ 1,7`
veces lo que marcaría un equipo medio contra el mismo rival. Sirven para una
vista de "fuerza del equipo", no para ordenar la tabla.

**`outcomes`** son mutuamente excluyentes entre `ucl`, `uel` y `uecl` (cada uno
es un rango disjunto de posiciones); `european_qualification` es la suma de los
tres. Todos se derivan de `position_probabilities` y de `qualification_rules`,
así que se pueden recalcular en el frontend si hiciera falta otra agrupación.

---

## 4. Competiciones UEFA (`type: "uefa_league_phase"`)

```jsonc
{
  "code": "UCL",
  "type": "uefa_league_phase",
  "name": "UEFA Champions League",
  "n_teams": 36,
  "spanish_teams_only": false,     // se detallan los 36, no solo los españoles
  "qualification_rules": {
    "direct_to_r16": [1, 8],       // la liguilla clasifica directo del 1º al 8º
    "playoff":       [9, 24],
    "eliminated":    [25, 36]
  },
  "matches_played": 0,
  "matches_remaining": 144,
  "teams": [ /* ver 4.1 */ ]
}
```

La Conference League usa el mismo bloque pero con 6 partidos de liguilla en vez
de 8; `matches_remaining` lo refleja y los rangos de clasificación no cambian.

### 4.1 Objeto de equipo europeo

```jsonc
{
  "team_id": 12,
  "name": "Real Madrid",
  "display_name": "Real Madrid",
  "logo": "assets/escudos/real-madrid.png",
  "country": "ESP",               // código UEFA de 3 letras
  "current": { "played": 0, "points": 0, "goals_for": 0, "goals_against": 0,
               "goal_difference": 0, "position": null },
  "ratings": { "elo": 1835.6 },   // ya comparable entre ligas (ver §4.2)

  "league_phase": {
    "expected_points": 15.66,
    "expected_position": 8.45,
    "position_probabilities": [ /* 36 valores, índice 0 = 1ª posición */ ],
    "p_direct_to_r16": 0.41,
    "p_playoff": 0.52,
    "p_eliminated": 0.07
  },

  "stage_probabilities": {        // probabilidad de ALCANZAR cada ronda
    "round_of_16": 0.915,
    "quarter_finals": 0.645,
    "semi_finals": 0.411,
    "final": 0.212,
    "winner": 0.146
  }
}
```

**Garantías** (verificadas por tests, se pueden usar como aserciones):

- `p_direct_to_r16 + p_playoff + p_eliminated` = 1,0 por equipo.
- `stage_probabilities` es **monótona decreciente**: nadie llega a semifinales
  más veces que a cuartos. El frontend puede dibujar un embudo sin comprobarlo.
- Sumando sobre los 36 equipos, cada ronda da su número de plazas: octavos 16,
  cuartos 8, semifinales 4, final 2, campeón 1.
- `position_probabilities` es doblemente estocástica, igual que en liga.

Ojo con `p_playoff`: **no mide éxito**. Caer en el playoff es acabar entre el 9º
y el 24º, así que un equipo fuerte tiene *poca* probabilidad de "llegar" ahí
precisamente porque aspira a pasar directo. Para medir rendimiento en la
liguilla hay que usar `p_direct_to_r16`.

### 4.2 Sobre `ratings.elo` en competición europea

El Elo que aparece en un bloque europeo **no es el mismo número** que en el
bloque de LaLiga: lleva aplicado el desplazamiento por fuerza de liga
(§ `model.league_offsets`), sin el cual los ratings de países distintos no son
comparables. Un mismo equipo puede aparecer con 1835 en `UCL` y 1750 en `ESP1`.
No los mezcles en un mismo gráfico sin decir cuál es cuál.

---


### 4.3 Proyeccion provisional (`provisional`)

Mientras el sorteo de la fase de liga no este publicado, el bloque trae
`provisional` relleno en vez de `null`:

```jsonc
"provisional": {
  "field_based_on": "2025-26",
  "draws_simulated": 25,
  "reason": "El sorteo de la fase de liga aun no esta publicado.",
  "method": "Campo estimado: los participantes reales de 2025-26 con el contingente español sustituido por el de esta temporada. El emparejamiento se sortea y se promedia sobre 25 sorteos distintos.",
  "assumes_league_phase": true,
  "caveat": "Las cifras parten de la fase de liga: si al equipo le toca ronda previa, el modelo no la simula y sus opciones reales son menores."
}
```

**`assumes_league_phase`** avisa de algo que conviene no pasar por alto: las
probabilidades arrancan con el equipo ya en la liguilla. El modelo **no simula
ninguna ronda previa**, asi que a quien tenga que jugarla (el caso habitual del
representante en Conference) hay que leerle las cifras como condicionadas a
haberla superado.

**Cuando este campo no es `null`, las cifras son orientativas y el dashboard
debe decirlo.** El campo rival es una estimacion, no el definitivo. Lo que si es
real es la fuerza de cada equipo, que es lo que mas pesa en cuanto se llega a
las eliminatorias.

Dos consecuencias practicas:

- Los equipos **no espanoles** del bloque son relleno plausible. Conviene
  mostrar solo los espanoles, que son los que se han sustituido por los de
  verdad.
- `league_phase.expected_position` sigue siendo util (a que altura de la tabla
  acaba), pero el orden del array `teams` no representa una clasificacion real.

Se promedia sobre varios sorteos a proposito: sin bombos hechos, la suerte del
cruce es una fuente de incertidumbre tan real como el propio juego, y simular
uno solo daria una respuesta que depende de cual salio.

---

## 5. `fixtures`

Próximos partidos reales de LaLiga, en orden cronológico. Longitud controlada
por `--max-fixtures` (20 por defecto); array vacío si no queda nada por jugar.
Un resultado hipotético de `calendar` **no** saca el partido de esta lista: el
escenario cuenta para la simulación, pero el partido sigue siendo un próximo
partido real hasta que tenga marcador verdadero en la base.

```jsonc
{
  "match_id": 11789,
  "competition": "ESP1",
  "matchday": 2,
  "date": "2026-08-23",
  "kickoff_utc": "2026-08-23T17:30:00Z",   // null si no se conoce
  "date_provisional": false,    // ver abajo
  "status": "scheduled",        // scheduled | live
  "live_home_goals": null,      // marcador parcial si status = live
  "live_away_goals": null,
  "live_detail": null,          // minuto/estado textual si status = live
  "home_team": {
    "team_id": 19,
    "name": "Atletico de Madrid",
    "display_name": "Atlético de Madrid",
    "logo": "assets/escudos/atletico-de-madrid.png"
  },
  "away_team": {
    "team_id": 88,
    "name": "Malaga CF",
    "display_name": "Málaga CF",
    "logo": "assets/escudos/malaga-cf.png"
  },
  "probabilities": { "home": 0.6519, "draw": 0.1998, "away": 0.1484 },
  "expected_goals": { "home": 1.86, "away": 0.74 }
}
```

`probabilities` suma 1,0. `expected_goals` son las lambdas del modelo Poisson,
no un xG observado.

Cuando `status` es `live`, los campos `live_home_goals`, `live_away_goals` y
`live_detail` reflejan el marcador parcial de ESPN. Ese parcial se muestra, pero
no se incorpora a la clasificación ni sustituye a un resultado real hasta que la
fuente marque el partido como finalizado.

**`kickoff_utc`** es el instante del saque inicial en ISO UTC
(`"2026-08-27T18:30:00Z"`), o `null` si no se conoce. Va en UTC a proposito: el
cambio de hora cae a mitad de temporada, asi que la conversion se deja al
frontend, que sabe de husos. En JavaScript basta con
`new Intl.DateTimeFormat("es-ES", { timeZone: "Europe/Madrid", hour: "2-digit",
minute: "2-digit" }).format(new Date(kickoff_utc))`.

**`date_provisional`** merece atencion porque afecta a lo que se puede mostrar.
LaLiga confirma dia y hora de cada jornada unas semanas antes; hasta entonces
las fuentes ponen un marcador de posicion, y todas lo hacen igual: la jornada
entera el mismo dia a la misma hora. Cuando este campo es `true`, ni la fecha ni
la hora son definitivas y **el dashboard no debe presentarlas como tales**.

Se marca por tres indicios: que la fecha ya haya pasado con el partido sin
jugar; que tres o mas partidos de la jornada compartan dia **y hora** exactos; o,
sin hora conocida, que tres o mas compartan dia.

Limitacion conocida: en las dos ultimas jornadas LaLiga si juega los diez
partidos a la vez, asi que ahi el criterio no distingue un horario real de uno
provisional. Faltando nueve meses, siempre es provisional.

---

## 6. `standings` — clasificacion real

La tabla tal y como esta, calculada solo con resultados de verdad.

```jsonc
"standings": {
  "n_teams": 20,
  "matches_played": 6,
  "complete": false,            // true cuando la temporada ha terminado
  "qualification_rules": { ... },   // igual que en el bloque de liga
  "rows": [
    {
      "position": 1,
      "team": "Deportivo Alaves",
      "display_name": "Deportivo Alavés",
      "logo": "assets/escudos/deportivo-alaves.png",
      "played": 1, "won": 1, "drawn": 0, "lost": 0,
      "goals_for": 3, "goals_against": 0, "goal_difference": 3,
      "points": 3,
      "form": ["G"]             // hasta 5, del mas reciente al mas antiguo
    }
  ]
}
```

`form` usa `G` ganado, `E` empatado, `P` perdido.

**Incluye siempre a los `n_teams` equipos**, tambien a los que aun no han
jugado: aparecen a cero al final. Una tabla a la que le faltan filas no es una
clasificacion.

**No incluye los resultados hipoteticos.** Se llama clasificacion real y colar
supuestos le quitaria el sentido al nombre; el efecto de un escenario se ve en
la proyeccion (`competitions.ESP1`), que si los cuenta. La forma de saber si hay
supuestos en juego es `calendar.scenario_count`.

El orden aplica los desempates de LaLiga: puntos, mini-liga entre los empatados
(puntos y luego diferencia de goles solo de esos partidos), diferencia general y
goles a favor. No es un detalle: en 2025-26 decidio un descenso.

`standings` es `null` cuando no se ha generado.

---

## 7. `european_qualification` — quien juega Europa

Equipos espanoles en competicion europea esta temporada, con la plaza que les
corresponde.

```jsonc
"european_qualification": {
  "source_season": "2025-26",
  "cup_winner": "Real Sociedad",      // null si no consta
  "teams": [
    { "team": "FC Barcelona", "previous_position": 1, "previous_points": 94,
      "competition": "UCL", "competition_name": "Champions League", "via": "liga" },
    { "team": "Real Sociedad", "previous_position": 10, "previous_points": 49,
      "competition": "UEL", "competition_name": "Europa League", "via": "copa" }
  ],
  "caveat": "Real Sociedad entra por la Copa del Rey, asi que el reparto..."
}
```

Sale de la clasificacion **final de la temporada anterior**, que es la que
reparte las plazas. Es `null` si no hay temporada anterior en la base de datos.

**`via`** distingue quien entra por liga y quien por copa. El campeon de Copa del
Rey se lleva una plaza de Europa League, y si no se habia clasificado por liga
**todo el reparto de abajo se desplaza un puesto**: el sexto conserva la otra
plaza de Europa League y el septimo cae a la Conference. Si ya estaba
clasificado, su plaza revierte y el reparto es el normal.

No hay fuente gratuita de la Copa, asi que el campeon se introduce a mano con
`simliga copa --temporada 2025-26 --campeon "Real Sociedad"`. **Muestra siempre
`caveat`**: cuando no consta, la lista puede tener un equipo de mas o de menos, y
ocultarlo la haria pasar por completa sin serlo.

---

## 8. `calendar` — calendario y escenarios

Calendario completo por jornadas. Es lo que permite montar una vista editable:
cada partido declara su estado, y solo los pendientes admiten hipotesis.

```jsonc
"calendar": {
  "current_matchday": 2,        // la jornada en curso: por la que abrir la vista
  "scenario_count": 4,          // cuantos resultados hipoteticos hay activos
  "live_count": 1,              // cuantos partidos estan en juego
  "editable": true,
  "matchdays": [
    {
      "matchday": 1,
      "date_from": "2026-08-15",
      "date_to": "2026-08-19",
      "played": 6, "live": 1, "scenario": 0, "pending": 3,
      "matches": [
        {
          "match_id": 13478,
          "date": "2026-08-15",
          "kickoff_utc": "2026-08-15T17:30:00Z",
          "date_provisional": false,
          "espn_event_id": "401882916", // id del evento en ESPN, o null
          "status": "played",           // played | live | scenario | pending
          "home_team": {
            "team_id": 4,
            "name": "Sevilla FC",
            "display_name": "Sevilla FC",
            "logo": "assets/escudos/sevilla-fc.png"
          },
          "away_team": {
            "team_id": 41,
            "name": "Rayo Vallecano",
            "display_name": "Rayo Vallecano",
            "logo": "assets/escudos/rayo-vallecano.png"
          },
          "home_goals": 2,
          "away_goals": 1,
          "live_home_goals": null,
          "live_away_goals": null,
          "live_detail": null
          // sin `probabilities`: ya se jugo, no hay nada que predecir
        }
      ]
    }
  ]
}
```

Los cuatro estados y lo que significan:

| `status` | Marcador | Editable | Cuenta en la simulacion |
|---|---|---|---|
| `played` | El real | **No** | Si, como hecho |
| `live` | El parcial de ESPN | **No** | No: espera al final |
| `scenario` | El que puso el usuario | Si | Si, como hecho |
| `pending` | `null` | Si | No: lo decide la simulacion |

Los partidos pendientes y de escenario traen ademas `probabilities` con la
prediccion del modelo (1X2), util para ensenarla al lado del campo de entrada.

**`espn_event_id` es para seguir el partido en directo.** El panel publicado
es una foto y refrescarla cuesta un despliegue entero de Pages, asi que el
directo lo sigue el navegador: le pide a ESPN el marcador del dia
(`site.api.espn.com/.../scoreboard?dates=YYYYMMDD`, que permite CORS y sirve
con `Cache-Control: max-age=10`) y casa los eventos por este id, nunca por
nombre de equipo. Es `null` mientras ESPN no haya dado ese partido.

Lo que se refresca asi son **solo** `live_home_goals`, `live_away_goals`,
`live_detail` y `status`. La proyeccion no se toca, y es lo correcto: un
resultado en vivo no cuenta hasta el pitido final, que es la misma regla que
aplica el motor. Un partido ya terminado que la publicacion aun no ha recogido
se ensena como `live` con «FT» de etiqueta: se ve el marcador, pero sigue sin
contar.

**`current_matchday` es por donde debe abrirse la vista de calendario.** No es
lo mismo que "la primera jornada sin acabar": un partido aplazado deja su
jornada incompleta durante meses, y abrir por ella dejaria la vista anclada en
agosto mientras se juega la jornada 14. Es la ultima jornada que ya ha
empezado (su primer partido esta fechado hoy o antes, o ya tiene algun
resultado), y la siguiente en cuanto esa se jugo entera, sin esperar a que
llegue su fin de semana. Es `null` solo si no hay calendario.

**Un `scenario` no es un resultado real y el frontend no debe presentarlo como
tal.** En el bloque de liga, `current.played` incluye los hipoteticos porque es
la clasificacion desde la que arranca la simulacion, pero `current.played_real`
y `current.scenario_matches` permiten separarlos, y `matches_played_real` hace
lo mismo a nivel de competicion.

Los escenarios **no alteran los ratings**: el Elo y el ajuste Dixon-Coles se
calculan sobre resultados reales. Por eso `ratings.elo` no se mueve al anadir
hipotesis, y no es un fallo.

---

## 9. `validation` y `meta`

`validation` es `null` en una simulación normal. Cuando se rellena, contiene el
resumen del backtest con la misma forma que `out/validacion.json`, que es donde
lo escribe `simliga backtest`. Las métricas clave:

- `rps`: Ranked Probability Score, más bajo es mejor. Referencias sobre LaLiga:
  ~0,226 predecir siempre la tasa base, ~0,191 el mercado de apuestas de cierre,
  y el modelo debería quedar entre ambos.
- `position_rps`: lo mismo aplicado a la distribución de posición final.
- `brier_*`: error cuadrático de los sucesos binarios (título, Champions, descenso).

`meta.data_sources` es la lista de fuentes usadas, para mostrar atribución.

---

## 10. Ajustes cualitativos

`simulation.modifiers_enabled` lista los modificadores activos en la tirada:

```jsonc
"simulation": {
  "n_sims": 20000,
  "seed": 20262027,
  "modifiers_enabled": []      // vacío: ninguno activo, que es el defecto
}
```

Valores posibles: `"fatigue"`, `"motivation"`, `"coach_change"`, `"injuries"`,
`"transfers"`, `"squad_depth"`.

**Vienen desactivados y conviene dejarlos así.** No es prudencia: se midieron
sobre 6.074 observaciones de ocho temporadas y ninguno alcanza significación
estadística; activando la fatiga, el RPS fuera de muestra *empeora* de forma
consistente. El detalle está en el [README](../README.md#fase-3-los-ajustes-cualitativos-no-mejoran-el-modelo).

Para el frontend esto significa que **con la configuración por defecto el campo
siempre es `[]`**. Si algún día se activa alguno, las probabilidades cambian pero
la estructura del documento no: no aparece ninguna clave nueva. Un dashboard que
ignore este campo seguirá funcionando; uno que quiera avisar de que la tirada
lleva ajustes puede mostrarlo tal cual.

---

## 11. Ejemplo mínimo para empezar

```js
const doc = await (await fetch("simulacion.json")).json();
const liga = doc.competitions.ESP1;
const [ucl_desde, ucl_hasta] = liga.qualification_rules.ucl;

for (const equipo of liga.teams) {          // ya vienen ordenados
  const pct = (x) => (100 * x).toFixed(1) + "%";
  console.log(
    equipo.name,
    "pts:", equipo.projection.points.mean.toFixed(1),
    `(${equipo.projection.points.p05}-${equipo.projection.points.p95})`,
    "| título:", pct(equipo.outcomes.title),
    "| Champions:", pct(equipo.outcomes.ucl),
    "| descenso:", pct(equipo.outcomes.relegation),
  );
}
```
