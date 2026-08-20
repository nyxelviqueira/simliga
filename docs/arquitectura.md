# Cómo funciona el simulador y cómo ponerle una interfaz

Dos preguntas: qué hace el motor por dentro, y cómo montar encima algo que se
mire en pantalla con datos frescos.

---

## 1. Qué hace el motor

El recorrido completo, de datos crudos a probabilidades:

```
   football-data.co.uk        openfootball
   (resultados + cuotas)   (calendario + UEFA)
            │                      │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │  SQLite (50.000      │   una tabla `matches` para todo:
            │  partidos, 12 ligas) │   liga, Segunda y competiciones UEFA
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
   PASO 1   │  Elo dinámico        │   un rating por equipo, actualizado
            │  (pool multiliga)    │   partido a partido
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
   PASO 2   │  Ajuste por liga     │   +143 Premier … −173 Grecia
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
   PASO 3   │  Dixon-Coles         │   Elo → goles esperados por partido
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
   PASO 4   │  Monte Carlo         │   20.000 temporadas completas
            └──────────┬───────────┘
                       ▼
                  JSON de salida
```

### Paso 1: el Elo dice *cuánta fuerza* tiene cada equipo

Un solo número por equipo, que sube cuando gana y baja cuando pierde, más si el
resultado es contundente y más aún si el rival era mejor. Se actualiza partido a
partido sobre 50.000 partidos de 12 competiciones.

Dos correcciones que no son opcionales, porque el Elo es de suma cero *dentro de
su grupo*: si dos ligas apenas se cruzan, sus ratings no son comparables.

- **Entre temporadas** se regresa parcialmente a la media (85%): lo del año
  pasado sigue informando, pero menos.
- **Al cambiar de división** se restan 50 puntos. Sin esto el modelo
  sobrestimaba a los recién ascendidos en +5,1 puntos por temporada.
- **Entre países** se aplica un desplazamiento por liga, estimado con los
  partidos europeos, que son los únicos que las cruzan. Sin esto el Olympiakos
  aparecía por delante del Liverpool.

### Paso 2 y 3: Dixon-Coles convierte fuerza en goles

El Elo no sabe de goles ni de empates. Dixon-Coles sí: para cada partido calcula
cuántos goles se espera que marque cada equipo.

```
goles esperados del local     = exp(μ + ataque[local] − defensa[visitante] + γ)
goles esperados del visitante = exp(μ + ataque[visitante] − defensa[local])
```

`ataque` y `defensa` se estiman de los resultados recientes (con los partidos
viejos pesando menos) pero **se encogen hacia lo que dice el Elo**. Eso es lo que
permite dar una fuerza sensata a un recién ascendido sin partidos en Primera, y
lo que evita que en la jornada 3 el modelo se crea que el líder sorpresa es el
mejor equipo de la liga.

A partir de ahí, la probabilidad de cada marcador posible (0-0, 1-0, 2-1…) sale
de una distribución de Poisson, con una corrección para los marcadores bajos,
que el Poisson puro predice mal.

### Paso 4: Monte Carlo

Se juega la temporada entera 20.000 veces. En cada una, cada partido pendiente
saca un marcador al azar según sus probabilidades, se suman los puntos, se
ordena la tabla aplicando los desempates reales de LaLiga y se anota dónde acabó
cada equipo.

Si el Barça acaba primero en 13.860 de las 20.000, su probabilidad de título es
69,3%. **Eso es todo lo que significa un porcentaje en la salida.**

Los partidos ya jugados entran con su resultado real, sin azar. Por eso las
probabilidades se afilan según avanza la temporada.

En Europa el proceso es el mismo, más el cuadro: la liguilla de 36 decide quién
pasa directo a octavos (1-8), quién juega el playoff (9-24) y quién se va
(25-36); después las eliminatorias a doble partido, con prórroga y penaltis.
Cada una de las 20.000 simulaciones tiene su propio cuadro.

### Qué NO hace, y por qué

**No sabe de lesiones, entrenadores ni fichajes.** Un equipo con media plantilla
lesionada tiene exactamente la misma fuerza que la semana anterior. El marco para
introducir esos eventos existe (`simliga ajuste`), pero hay que meterlos a mano:
ninguna fuente gratuita los da de forma fiable.

**Sí sabe de fatiga, pero está desactivada a propósito.** El calendario combinado
permite calcular cuántos días de descanso llevaba cada equipo, y el modificador
está implementado. Se midió y el efecto no existe en los datos: comparando a los
equipos con 2+ días más de descanso que su rival contra los que tenían 2+ menos,
la diferencia es de +0,001 goles. Activarla *empeora* el RPS de forma consistente.

La lectura no es que la fatiga no importe, sino que el modelo ya la absorbe: los
equipos que juegan entre semana son los buenos, y el Elo ya sabe que lo son.

---

## 2. Cómo ponerle una interfaz

**El motor no es un servidor y no debería serlo.** Es un proceso que tarda unos
segundos, produce un JSON y termina. La interfaz no llama al simulador: lee su
JSON.

```
   cron / tarea programada          servidor web o hosting estático
   ┌──────────────────────┐         ┌──────────────────────────────┐
   │ simliga actualizar   │ ──────► │  simulacion_2026-27.json     │ ──► navegador
   │ (1-2 min)            │  escribe│  (unos 50 KB)                │ lee
   └──────────────────────┘         └──────────────────────────────┘
```

Esa separación es deliberada: si el frontend llamase al motor, cada visita
pagaría 20.000 simulaciones. Así, mil visitas leen el mismo fichero.

### El comando que lo hace todo

```bash
.venv/Scripts/python.exe -m simliga actualizar --temporada 2026-27 --sims 20000 --salida web/datos.json
```

En orden: baja los resultados nuevos, refresca el calendario, mira si ya hay
sorteo de las competiciones UEFA, **comprueba la integridad de los datos** y, si
todo cuadra, simula y escribe el JSON. Si algo no cuadra (por ejemplo, aparece un
equipo 21 porque la fuente cambió una grafía) aborta antes de simular en vez de
producir un JSON con datos silenciosamente mal.

Devuelve código de salida 0 si fue bien, 1 si falló, así que se puede encadenar.

### Resultados en directo

football-data.co.uk publica con uno o dos días de retraso. Para que el panel
refleje un partido que acaba de terminar:

```bash
python -m simliga resultado --temporada 2026-27 --local "Atletico de Madrid" --visitante "Malaga CF" --marcador 2-0
```

El resultado entra marcado como `manual` y la simulación lo incorpora de
inmediato. Cuando la fuente oficial lo publique, la ingesta normal lo sustituye
por su versión. Al revés no pasa nunca: **una fuente sin marcador no puede
borrar un marcador ya guardado**. Eso importa porque el calendario de la
temporada en curso viene sin resultados, y al refrescarlo borraba todo lo
jugado; era un fallo real y hay tests de regresión que lo vigilan.

### Cada cuánto ejecutarlo

Los resultados de LaLiga en football-data.co.uk se publican con un día de
retraso. No tiene sentido refrescar más de una vez al día:

| Momento | Qué cambia |
|---|---|
| Lunes y jueves por la mañana | Resultados de la jornada de liga |
| Miércoles y viernes | Resultados de competición europea |
| Diario a las 6:00 | Opción simple que cubre todo |

En Windows, con el Programador de tareas; en Linux, con cron:

```bash
0 6 * * * cd /ruta/SimLiga && .venv/bin/python -m simliga actualizar --temporada 2026-27 --salida web/datos.json
```

### En el movil

El mismo panel `out/panel.html` vale: trae los datos, CSS y JavaScript
incrustados, y usa la carpeta `out/assets/escudos` para los escudos. Te mandas
ambas cosas por correo o lo subes a cualquier hosting estatico y funciona sin
conexion, en solo lectura.

Para la version completa (regenerar y editar escenarios desde el telefono) hace
falta que el PC sirva el panel a la red local: `simliga servidor --en-red`, o
doble clic en `panel-movil.bat`. La ventana imprime la direccion del movil.
Requiere misma wifi, PC encendido, y conviene recordar que no pide contrasena a
nadie de esa red.

### Tres formas de montar la interfaz, de menos a más trabajo

**a) El panel que ya viene hecho.** `--panel` escribe `out/panel.html` y copia
los assets visuales a `out/assets`. Se abre con doble clic. Es la opción por
defecto y para un panel personal sobra con ella.

```bash
python -m simliga actualizar --temporada 2026-27 --panel
```

Los datos van incrustados en la página en lugar de leerse con `fetch`, porque
el navegador bloquea las peticiones desde un fichero local: con `fetch` habría
que levantar un servidor solo para ver una página estática.

Si prefieres otro diseño, el JSON está documentado campo a campo en
[`data-contract.md`](data-contract.md) precisamente para que se pueda construir
sin leer el código del motor. Sirve igual en GitHub Pages o Netlify.

**b) Servidor pequeño.** Si quieres que la interfaz pueda *disparar* una
simulación (un botón «recalcular», o simular escenarios del tipo «¿y si el
Barça gana los tres próximos?»), hace falta un proceso que ejecute el motor.
El punto de entrada es `simliga/pipeline.py`:

```python
from simliga.pipeline import build_context, simulate_laliga, simulate_european

ctx = build_context(as_of="2027-01-15")      # el modelo con datos hasta esa fecha
resultado, fit = simulate_laliga(ctx, "2026-27")
probabilidades = resultado.position_probabilities()   # matriz 20x20
```

`build_context` es la parte cara (Elo sobre 50.000 partidos, unos segundos);
conviene construirlo una vez y reutilizarlo. Cada simulación posterior es de
alrededor de un segundo.

**c) Base de datos como API.** El SQLite ya está ahí y cualquier cosa lo lee. Útil
si la interfaz necesita datos que el JSON no lleva (resultados históricos,
series de Elo, calendario completo). Solo lectura, para no pelearse con el motor.

### Lo que debe saber quien construya el frontend

- **Todo lo que necesita está en `docs/data-contract.md`**, con las garantías
  numéricas que puede dar por ciertas (las probabilidades de posición suman 1
  por filas y por columnas, el embudo europeo es monótono decreciente, etc.).
  Hay tests que fallan si alguna deja de cumplirse.
- **No cablees posiciones.** El número de plazas de Champions cambia según el
  coeficiente UEFA; viene resuelto en `qualification_rules`.
- **Itera sobre `competitions`**, no asumas qué competiciones hay: dependen de
  si ya se ha sorteado la fase de liga europea.
- **No muestres más de un decimal** en los porcentajes. Con 20.000 simulaciones,
  el error típico de una probabilidad cercana al 50% es de ±0,4 puntos.
- **Un 30% no es un fallo si no ocurre.** El modelo está calibrado: de las cosas
  a las que da un 30%, ocurre un 30%. La forma correcta de juzgarlo es a lo
  largo de muchas predicciones, que es lo que hace `scripts/validacion.py`.
