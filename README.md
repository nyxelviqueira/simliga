# SimLiga

Simulador Monte Carlo de LaLiga: modelo Dixon-Coles con ratings Elo dinámicos,
validado contra tres temporadas históricas. Produce un JSON con la distribución
de probabilidad de posición final de cada equipo.

**Estado: las tres fases, completas.** LaLiga, las tres competiciones UEFA con
su formato de liguilla única, y el marco de ajustes cualitativos — que, medido,
resulta que no mejora el modelo y viene desactivado
([por qué](#fase-3-los-ajustes-cualitativos-no-mejoran-el-modelo)).

---

## Resultado de la validación

Backtest sobre 2022-23, 2023-24 y 2024-25 (1.140 partidos), reajustando el
modelo antes de cada fecha, sin que vea nunca un partido posterior al que
predice.

### Predicción partido a partido (RPS, más bajo es mejor)

| Temporada | Modelo | Mercado de cierre | Tasa base |
|---|---|---|---|
| 2022-23 | 0,2069 | 0,2018 | 0,2271 |
| 2023-24 | 0,1864 | 0,1826 | 0,2237 |
| 2024-25 | 0,1957 | 0,1875 | 0,2285 |
| **Media** | **0,1959** | **0,1906** | **0,2264** |

El mercado de apuestas es el techo práctico: incorpora alineaciones, lesiones y
dinero real. **El modelo cubre el 85,3% de la distancia entre no saber nada
(tasa base) y el mercado de cierre**, usando solo resultados históricos.

### Re-simulación de la temporada por jornada de corte

Media de las tres temporadas. `position_rps` mide si la distribución de posición
final concentra masa donde el equipo acabó de verdad.

| Jornada vista | position RPS | MAE de puntos | Spearman posición | Brier descenso |
|---|---|---|---|---|
| 0 (pretemporada) | 0,115 | 7,97 | 0,680 | 0,103 |
| 5 | 0,089 | 6,38 | 0,809 | 0,086 |
| 10 | 0,087 | 5,82 | 0,824 | 0,072 |
| 19 (ecuador) | 0,080 | 4,79 | 0,859 | 0,075 |
| 28 | 0,049 | 3,24 | 0,940 | 0,024 |
| 34 | 0,032 | 2,16 | 0,974 | 0,010 |

La mejora es monótona, que es lo que debe pasar: cada jornada vista reduce la
incertidumbre. Un MAE de 8,0 puntos en pretemporada está en el rango de lo
razonable para predecir una liga a 38 jornadas.

### Calibración

De los partidos a los que el modelo asigna un 26%, gana el local el 25,5%. La
desviación media ponderada es de 1,7 puntos porcentuales. Hay un patrón leve de
**infraconfianza** en los extremos (donde dice 75%, ocurre el 82%), pero al
comprobarlo con validación cruzada dejando una temporada fuera, corregirlo con
un parámetro de temperatura aporta un +0,10% de RPS y empeora una de las tres
temporadas: **no es señal estable con tres temporadas**, así que no se ha
añadido. Merece revisarse con 8-10 temporadas
(`scripts/calibracion_temperatura.py` reproduce el análisis).

### Competiciones europeas

Simulando cada torneo desde antes de la primera jornada de la liguilla (con el
sorteo ya conocido) y comparando con las rondas que cada equipo alcanzó de
verdad. La referencia es **el bombo**: repartir la misma probabilidad entre los
36 participantes.

| Torneo | Brier del modelo | Brier del bombo | Mejora |
|---|---|---|---|
| Champions 2024-25 | 0,1047 | 0,1285 | +18,5% |
| Champions 2025-26 | 0,0849 | 0,1285 | +33,9% |
| Europa League 2024-25 | 0,0977 | 0,1285 | +24,0% |
| Conference 2024-25 | 0,1100 | 0,1285 | +14,4% |
| **Media** | | | **+22,7%** |

La separación por rondas es clara: los equipos que llegaron a octavos tenían una
probabilidad media asignada de 0,57 frente a 0,34 los que no; en cuartos, 0,41
frente a 0,17.

### Fase 3: los ajustes cualitativos no mejoran el modelo

Se implementó el marco completo (fatiga por calendario, motivación situacional,
y cambios de entrenador, lesiones y fichajes vía eventos manuales) y después se
midió si sirven. La respuesta es que no.

**Medición del efecto** — goles reales menos los que el modelo esperaba, sobre
6.074 observaciones de ocho temporadas:

| Contraste | Efecto | t | ¿Significativo? |
|---|---|---|---|
| Descanso ≤3 días frente a ≥6 | −0,055 goles | −1,38 | no |
| Venía de Europa con ≤4 días, frente a ≥6 | −0,074 goles | −1,19 | no |
| **2+ días más de descanso que el rival, frente a 2+ menos** | **+0,001 goles** | **+0,02** | **no** |
| Sin nada en juego en las últimas jornadas | +0,023 goles | +0,33 | no |

El tercero es el más revelador porque es el planteamiento correcto: lo que
debería importar no es el descanso absoluto sino el que se tiene *respecto al
rival*. Da prácticamente cero exacto.

**Medición del efecto sobre la predicción** — RPS con la fatiga activada,
eligiendo la magnitud sin mirar la temporada de test:

| Penalización | RPS medio | Frente a la base |
|---|---|---|
| desactivada | 0,19830 | — |
| 4 Elo/día | 0,19845 | +0,00015 |
| 8 Elo/día | 0,19863 | +0,00033 |
| 20 Elo/día | 0,19967 | +0,00137 |

Monótono: cuanta más penalización, peor. Fuera de muestra empeora en 7 de las 8
temporadas (+0,00015 de media, error estándar 0,00005).

**La lectura no es que la fatiga no exista**, sino que el modelo ya la absorbe:
los equipos que juegan entre semana son los buenos, y el Elo ya sabe que son
buenos. Añadir un parámetro que no mejora nada solo agrega ruido y una vía más
por la que equivocarse.

Por eso el marco existe, es configurable y está **desactivado por defecto**.
`scripts/analisis_modificadores.py` y `scripts/comparar_modificadores.py`
reproducen las dos mediciones.

---

## Cómo funciona

```
Elo dinámico  ──prior──►  Dixon-Coles  ──►  Monte Carlo  ──►  JSON
(fuerza base)             (goles y empates)  (20.000 temporadas)
```

**1. Elo** ([`simliga/model/elo.py`](simliga/model/elo.py)). Rating actualizado
partido a partido con multiplicador por diferencia de goles y ventaja local en
puntos de rating. Entre temporadas se aplica regresión parcial a la media
(`R' = media + 0,85·(R − media)`). Se calcula en casa en vez de descargarlo de
ClubElo: es reproducible, auditable, y **cubre también la Segunda División**,
que es de donde sale la fuerza inicial de los recién ascendidos.

**2. Dixon-Coles** ([`simliga/model/dixon_coles.py`](simliga/model/dixon_coles.py)).
Poisson bivariante con corrección de marcadores bajos:

```
log λ_local     = μ + ataque[local] − defensa[visitante] + γ
log λ_visitante = μ + ataque[visitante] − defensa[local]
```

Dos añadidos sobre el paper original:

- **Peso temporal exponencial** (media vida de 365 días) para seguir la forma reciente.
- **Prior derivado del Elo**: ataque y defensa se encogen hacia el valor implicado
  por el rating. Es lo que evita que un recién ascendido sin partidos en Primera
  quede sin fuerza estimada, y lo que estabiliza las primeras jornadas.

Ajuste por máxima verosimilitud penalizada con gradiente analítico: **0,03 s por
ajuste**, lo que hace viable reajustar antes de cada fecha en el backtest.

**3. Monte Carlo** ([`simliga/sim/league.py`](simliga/sim/league.py)). Muestreo
exacto de la distribución conjunta de marcadores por CDF invertida (no Poisson
independientes: la corrección ρ se respeta). Vectorizado sobre las simulaciones
— el bucle recorre partidos, nunca simulaciones. **20.000 temporadas completas
en 1,1 s.** Aplica los desempates de LaLiga, con el enfrentamiento directo
resuelto de forma exacta para los empates entre dos equipos.

---

## Instalación y uso

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 1. Ingesta de datos históricos

```bash
.venv/Scripts/python.exe -m simliga ingest --desde 2010 --hasta 2025
```

Descarga Primera y Segunda desde football-data.co.uk a `data/simliga.sqlite`
(13.472 partidos, 75 equipos, cuotas de mercado en 13.446). No hace falta API key.

### 2. Validación

```bash
.venv/Scripts/python.exe -m simliga backtest --temporadas 2022-23 2023-24 2024-25
```

O el informe completo con calibración, que escribe `out/validacion.json`:

```bash
.venv/Scripts/python.exe scripts/validacion.py
```

### 3. Simulación

```bash
.venv/Scripts/python.exe -m simliga simular --temporada 2024-25 --sims 20000 --salida out/simulacion.json
```

Con `--hasta 2025-01-15` simula desde esa fecha, incorporando como hechos los
resultados anteriores. Sin `--hasta`, simula la temporada entera desde cero.

El esquema del JSON está documentado en
[`docs/data-contract.md`](docs/data-contract.md).

### 4. Mantenerlo al día

Un solo comando refresca todo y regenera el JSON:

```bash
.venv/Scripts/python.exe -m simliga actualizar --temporada 2026-27 --sims 20000 --salida web/datos.json
```

Baja resultados, refresca el calendario, mira si ya hay sorteo europeo,
**comprueba la integridad de los datos** y solo entonces simula. Pensado para
lanzarlo desde el Programador de tareas o cron.

**Resultados que la fuente aún no publica.** football-data.co.uk va uno o dos
días por detrás, así que un partido recién terminado no está en ningún sitio.
Se puede meter a mano:

```bash
.venv/Scripts/python.exe -m simliga resultado --temporada 2026-27 --local "Atletico de Madrid" --visitante "Malaga CF" --marcador 2-0 --fecha 2026-08-19
```

Cuando la fuente lo publique, sobrescribirá la entrada manual con su versión.
Lo que **nunca** ocurre al revés: un refresco de calendario no borra un
resultado ya guardado (lo hacía, y era un fallo serio; hay tests que lo vigilan).

### 5. El panel

Hay dos formas, segun si quieres poder simular desde el propio panel.

**Interactivo (con boton de regenerar): doble clic en `panel.bat`.**
Arranca un servidor local y abre el navegador. Desde ahi puedes elegir cuantas
simulaciones lanzar y volver a simular sin tocar el terminal, o marcar
"descargar datos nuevos antes" para que baje resultados frescos primero.
Deja la ventana negra abierta mientras lo uses.

```powershell
.\.venv\Scripts\python.exe -m simliga servidor --temporada 2026-27
```

**De una sola vez: doble clic en `actualizar-panel.bat`.** Actualiza los datos,
simula y escribe `out/panel.html`, un fichero autocontenido que se abre con
doble clic sin servidor ni conexion. Es el adecuado para una tarea programada.

Ese fichero **tambien simula por su cuenta**: lleva dentro el motor Monte Carlo
en JavaScript (`simliga/output/motor.js`), asi que puedes cambiar el numero de
simulaciones y poner resultados en el calendario sin nada detras. Lo unico que
le falta respecto al panel servido es descargar datos nuevos, que necesita
Python y red; el boton "Actualizar datos" solo aparece cuando hay servidor.

```powershell
.\.venv\Scripts\python.exe -m simliga actualizar --temporada 2026-27 --panel
```

No hace falta "activar" el entorno virtual: llamando directamente a
`.venv\Scripts\python.exe` ya se usan sus dependencias.

El panel abre en pestanas que comparten sitio: **Clasificacion actual** (como
esta la liga ahora, con la forma reciente) y **Proyeccion final** (como se espera
que acabe, con bandas y rango de puntos). Debajo, el mapa de calor de posicion
final, el calendario por jornadas con dia y hora reales, los proximos partidos y
las competiciones europeas. Se adapta al tema claro u oscuro.

En cuanto pones algun resultado a mano aparece una tercera, **Clasificacion
simulada**: la tabla que sale de sumar lo ya jugado y tus hipotesis. No es una
proyeccion y ahi esta la diferencia: no se simula ningun partido, solo se suman
los puntos que ya estan puestos, asi que los equipos no llevan todos los mismos
partidos. Se actualiza al instante al editar un marcador, sin volver a simular,
y una flecha al lado de cada equipo dice cuanto se mueve respecto a la
clasificacion real. Los partidos y las rachas que salen de una hipotesis van
marcados, para que no se confunda con una tabla de verdad.

**Pulsando en cualquier equipo** se abre su ficha: cuantas veces acabo en cada
puesto de las 20.000 simulaciones, con recuento y porcentaje, sus puntos
esperados y su recorrido europeo si juega alguna competicion.

**Para que se actualice solo**, programa `actualizar-panel.bat` en el Programador
de tareas de Windows (`taskschd.msc` -> Crear tarea basica -> diaria a las 6:00
-> Iniciar un programa -> la ruta del `.bat`). Quitale la ultima linea
(`start ...`) si no quieres que abra el navegador cada manana.

Si prefieres servirlo en red, `out/simulacion_*.json` es el contrato documentado
en [`docs/data-contract.md`](docs/data-contract.md) y cualquier frontend puede
leerlo.

### 6. Probar escenarios

En el panel interactivo hay un calendario por jornadas donde se puede poner el
resultado que quieras en cualquier partido **aun sin jugar**. Esos puntos pasan
a contar como seguros en la simulacion.

Los partidos ya disputados salen bloqueados con su marcador real: lo que paso,
paso. Si se pudieran sobrescribir, la clasificacion "real" del panel dejaria de
serlo sin que nada lo indicase.

Cada hipotesis se ve en dos sitios, y conviene no confundirlos. En
**Clasificacion simulada** aparece al momento, porque solo hay que sumar puntos.
En **Proyeccion final** no aparece hasta que pulsas «Simular con estos
resultados»: ahi hay que volver a jugar los partidos que quedan veinte mil
veces.

Lo mismo desde el terminal:

```powershell
.\.venv\Scripts\python.exe -m simliga escenario --temporada 2026-27 --local "Celta de Vigo" --visitante "CA Osasuna" --marcador 2-0
.\.venv\Scripts\python.exe -m simliga escenario --temporada 2026-27 --listar
.\.venv\Scripts\python.exe -m simliga escenario --temporada 2026-27 --limpiar
```

**Dos cosas que conviene entender del resultado.**

Un resultado hipotetico **no cambia la fuerza de ningun equipo**. Los ratings
Elo y el ajuste Dixon-Coles se calculan sobre la base de datos, que nunca ve los
escenarios (viven en su propia tabla). Ganar cuatro partidos imaginarios da doce
puntos, pero no convierte a nadie en mejor equipo. Es la lectura correcta de la
pregunta que se esta haciendo: *si pasa esto, como queda la tabla*, no
*aprende de esto*.

Y los puntos que suman son **menos de los que parece**. Dar cuatro victorias al
Celta no le anade doce puntos a su proyeccion sino unos seis: el simulador ya le
daba de media unos seis puntos en esos cuatro partidos. Lo que suma la hipotesis
es la diferencia entre lo seguro y lo que ya se esperaba.

En la tabla, las columnas J y Pts salen en ambar cuando incluyen hipotesis, para
que no se confunda con la clasificacion real.

### 7. El campeon de Copa del Rey

El ganador de la Copa se lleva una plaza de Europa League, y eso **desplaza el
reparto que sale de la liga**: si no se habia clasificado por liga, el sexto
conserva la otra plaza de Europa League y el septimo cae a la Conference.

No hay fuente gratuita de la Copa del Rey (football-data.org la tiene solo en
plan de pago, openfootball no la publica y el plan gratuito de API-Football esta
capado a 2022-2024), asi que se introduce a mano:

```powershell
.\.venv\Scripts\python.exe -m simliga copa --temporada 2025-26 --campeon "Real Sociedad"
```

Mientras no conste, el panel avisa de que a la lista de clasificados le puede
faltar o sobrar un equipo, en lugar de presentarla como completa.

### 8. Simular 2026-27

El calendario real de 2026-27 viene de `openfootball/espana` (datos abiertos
CC0 en GitHub, sin API key y sin cuota). Es la única fuente verificada que
publica el calendario **antes** de jugarse:

```bash
.venv/Scripts/python.exe -m simliga calendario --temporada 2026-27
.venv/Scripts/python.exe -m simliga simular --temporada 2026-27 --sims 20000
```

380 partidos, 38 jornadas y los 20 equipos reales, con fechas. El comando
`generar-calendario` (round-robin sintético) queda como respaldo por si alguna
temporada no estuviera cubierta.

Proyección de pretemporada 2026-27 (extracto):

| Equipo | Puntos esperados | Título | Champions | Descenso |
|---|---|---|---|---|
| FC Barcelona | 84,4 | 70,9% | 100% | 0,0% |
| Real Madrid | 78,0 | 26,3% | 99,1% | 0,0% |
| Atlético de Madrid | 64,2 | 1,4% | 75,9% | 0,2% |
| Villarreal CF | 63,7 | 1,2% | 73,6% | 0,1% |
| … | | | | |
| Racing de Santander | 46,3 | 0,0% | 5,3% | 19,4% |
| Málaga CF | 43,0 | 0,0% | 2,1% | 33,4% |
| Deportivo de La Coruña | 43,0 | 0,0% | 2,1% | 33,1% |

### 9. Llevarlo al movil

Tres formas, de menos a mas trabajo.

La opcion recomendada para usarlo a diario sin PC encendido es **GitHub Pages +
GitHub Actions**. Ya hay un workflow preparado en
`.github/workflows/publish-panel.yml`: actualiza datos, simula, publica `out/`
y deja una URL fija para el movil. El paso a paso completo esta en
[`docs/movil-github-pages.md`](docs/movil-github-pages.md).

**a) Mandarte el fichero.** `out/panel.html` es autocontenido: unos 600 KB, sin
dependencias externas salvo las tipografias. Te lo envias por correo o lo dejas
en Drive, Dropbox o iCloud y lo abres desde el movil. Funciona **sin conexion** y
sin el PC encendido.

No es solo lectura: el motor va incrustado, asi que **simula en el propio
telefono**. Cambiar el numero de simulaciones, poner resultados en el calendario
y ver como cambia la tabla funciona igual que en el PC. Las hipotesis se guardan
en el navegador del movil y siguen ahi al volver a abrirlo. Lo unico que no
puede es traer datos nuevos.

Un orden de magnitud, medido en este proyecto: 20.000 temporadas tardan menos de
un segundo, 50.000 unos tres. El tope sin servidor son 100.000 a proposito;
mas alla la espera deja de compensar en un movil.

En el movil, "Compartir -> Anadir a pantalla de inicio" le pone icono propio: la
pagina trae los `meta` y el icono incrustados para que se vea como una app, con
la barra de estado en el color del tema.

**b) Publicarlo con GitHub Pages.** GitHub Actions publica el panel **poco
despues de cada partido**: se despierta cada 20 minutos en horario de futbol,
pregunta si hay algun encuentro terminado sin resultado y solo entonces
actualiza y publica; si no hay nada, termina en medio minuto sin tocar la
pagina. Tambien puedes lanzarlo a mano desde el movil con **Actions -> Publicar
panel movil -> Run workflow**. Misma pagina y
mismas capacidades que (a), pero con URL fija y sin reenviarte el fichero. Como
no hay endpoints de escritura, tampoco hay nada que proteger.

Con el usuario `nyxelviqueira` y un repositorio llamado `simliga`, la URL sera:
`https://nyxelviqueira.github.io/simliga/`.

Desde el PC se puede forzar esa misma actualizacion con `actualizar-github.bat`.
La primera vez pide haber guardado un token de GitHub en
`SIMLIGA_GITHUB_TOKEN`; el detalle esta en
[`docs/movil-github-pages.md`](docs/movil-github-pages.md).

**c) Servirlo desde el PC, con todo funcionando.** Doble clic en
`panel-movil.bat` (o `simliga servidor --en-red`). La ventana muestra la
direccion que hay que escribir en el movil:

```
Panel en http://127.0.0.1:8000/
  desde el movil:  http://192.168.1.79:8000/
```

Asi tienes lo unico que (a) y (b) no dan: el boton **"Actualizar datos"**, que
descarga resultados, calendario, horarios y competiciones europeas y vuelve a
simular, sin tocar el terminal. Requiere estar en la misma wifi y el PC
encendido.

**Dos avisos sobre (c).** Mientras corra, cualquiera en esa red puede abrir el
panel y tocar los escenarios: no pide contrasena. No lo dejes abierto en una red
que no controles. Y si el movil no conecta, casi siempre es el cortafuegos de
Windows: permite Python en redes privadas, o ejecuta como administrador el
comando que la propia ventana te imprime.

Por eso `--en-red` es explicito y nunca el comportamiento por defecto: sin esa
bandera el servidor solo escucha en local.

**d) Desplegarlo en un hosting, para no depender del PC.** Es lo unico que da
datos que se actualizan solos estando el PC apagado. Hay `Dockerfile`,
`arranque.sh` y `fly.toml` preparados; falta crear la cuenta, que es gratis.

```bash
fly auth signup
fly secrets set SIMLIGA_TOKEN=$(openssl rand -hex 16)
fly volumes create datos --size 1 --region mad
fly deploy
```

El token **no es opcional**: `arranque.sh` se niega a arrancar sin el. La razon
es que el panel escribe en la base de datos (escenarios, datos descargados) sin
preguntar quien llama, asi que una URL publica sin contrasena seria editable por
cualquiera que diese con ella. Se pega una vez en el movil como
`https://tu-app.fly.dev/?t=EL-TOKEN` y a partir de ahi queda en una cookie.

Tres cosas que conviene saber antes:

- La maquina **se apaga sola** cuando no la usas (`auto_stop_machines`). Es lo
  que lo mantiene en el plan gratuito; el precio es que la primera visita
  despues de un rato tarda unos veinte segundos, porque tiene que rehacer el
  Elo sobre 50.000 partidos.
- El volumen es imprescindible. Sin el, cada reinicio vuelve a la base de datos
  de la imagen y se pierde todo lo descargado desde el despliegue.
- Para que se actualice de verdad **solo**, sin entrar tu, hace falta ademas una
  tarea programada que llame al endpoint. En Fly, `fly machine run` con un cron;
  o desde cualquier sitio, un `POST /api/simular` con `{"refrescar": true}` y la
  cookie del token.

**La imagen no esta probada**: Docker Desktop no estaba arrancado al escribirla.
Lo que si esta comprobado es lo que hay debajo: que `SIMLIGA_DATA` reubica la
base de datos, que `arranque.sh` se niega sin token, y que el servidor real
devuelve 401 sin token y 200 con el.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests -q          # 323 tests, ~70 s
.venv/Scripts/python.exe -m pytest tests -q -m "not historico"   # solo unitarios, ~1 s
```

Cubren tres niveles:

- **Unitarios**: propiedades del Elo (suma cero, monotonía), del ajuste
  (gradiente analítico contra el numérico, recuperación de parámetros conocidos
  sobre datos sintéticos) y del motor (contabilidad de puntos y goles,
  desempates, reproducibilidad por semilla).
- **Métricas**: valores del RPS calculados a mano, detección de modelos
  descalibrados.
- **Contrato**: cada garantía escrita en `docs/data-contract.md` tiene su
  aserción (matriz de posiciones doblemente estocástica, plazas europeas
  disjuntas, percentiles ordenados, recuento de partidos). Si el esquema se
  rompe, falla un test aquí, no el dashboard.
- **Históricos** (marcados `historico`): que el modelo bata a la tasa base en
  las tres temporadas, que no se aleje más de un 10% del mercado, que la
  precisión mejore según avanza la temporada, que el campeón real estuviera
  entre los favoritos y que los descendidos reales tuvieran a mitad de temporada
  un riesgo al menos tres veces superior a la media.
- **Los dos motores** (`test_motor_js.py`, necesita Node): que el Monte Carlo en
  JavaScript dé lo mismo que el de Python. Son dos implementaciones del mismo
  modelo, y dos implementaciones se separan en cuanto alguien toca una sin
  acordarse de la otra. Se comparan la rejilla de marcadores (exacta, a 1e-12) y
  las probabilidades finales (dentro de la banda que explica el muestreo). Donde
  no haya Node se saltan en lugar de fallar.

---

## Decisiones tomadas y sus motivos

**El Monte Carlo, dos veces: en Python y en JavaScript.** Duplicar una
implementacion es normalmente mala idea, y aqui se hace a sabiendas. El motivo
es que el ajuste y la simulacion tienen costes muy distintos: ajustar el modelo
recorre 50.000 partidos historicos y necesita numpy y scipy; simular con el
modelo ya ajustado solo necesita veintitantos numeros por equipo. Esa asimetria
es lo que permite que el panel funcione entero en un movil sin nada detras,
porque lo caro viaja ya hecho dentro del JSON.

La alternativa era compilar Python a WebAssembly (Pyodide): unos 10 MB de
descarga para ejecutar 200 lineas de bucle, en una pagina que pesa 600 KB. El
riesgo real de duplicar es que las dos versiones se separen, y contra eso estan
los tests de `test_motor_js.py`, que comparan las dos salidas.

**Elo propio en lugar de ClubElo.** ClubElo no responde desde este entorno
(timeout en todas las peticiones, probado cuatro veces con hasta 60 s). Además
el Elo propio es reproducible, ajustable y cubre la Segunda División, que es de
donde sale la fuerza inicial de los recién ascendidos.

**Corrección por ascenso de división (−50 puntos Elo).** Primera y Segunda
comparten un único pool de Elo pero casi no juegan entre sí, así que sus escalas
derivan. Medido sobre 45 ascensos en 15 temporadas, el modelo **sobrestimaba a
los recién ascendidos en +5,1 puntos de media** (error estándar 1,47, sesgo
positivo en 12 de 15 temporadas) mientras el resto de equipos no tenía sesgo
(−0,7 ± 0,6). Restar 50 puntos Elo lo deja en +0,1 ± 1,5 y baja el MAE global de
7,62 a 7,43. Es configurable (`elo.promotion_penalty`).

Quién recibe la corrección se calcula **de los partidos**, no del estado interno
del Elo. Deducirlo del rating parecía más cómodo y fallaba de tres maneras a la
vez: se apagaba en cuanto el ascendido jugaba una sola jornada, no reconocía a un
equipo que ya hubiera estado en Primera dos años antes, y —lo más grave— acababa
penalizando a **465 de los 485 equipos** del pool, es decir a toda Europa menos a
los veinte de LaLiga, lo que inflaba las opciones continentales españolas.
Corregido, el modelo pasó de 84,2% a 85,3% del camino al mercado y de +22,1% a
+22,7% sobre el bombo en Europa. La corrección se mantiene toda la temporada:
medido sobre 864 partidos de ascendidos, cuanto más persiste mejor predice, y la
mejora viene sobre todo de la segunda vuelta.

**El desempate por enfrentamiento directo no es un detalle.** La tabla real
usaba diferencia de goles para desempatar y eso da posiciones falsas: en 2025-26
Levante y Mallorca acabaron a 42 puntos y bajó el Mallorca *pese a tener mejor
diferencia general*, porque el Levante le ganó el head-to-head. Como esa tabla
es la verdad contra la que se valida, el error contaminaba las métricas.
Corregido con la mini-liga entre empatados, que es la regla real de LaLiga.

**El ajuste de Dixon-Coles usa solo partidos de Primera.** Los ratings de ataque
y defensa no son comparables entre divisiones; el puente entre ellas lo hace el
Elo, que sí se calcula sobre las dos. Un recién ascendido entra con ataque y
defensa derivados enteramente de su prior Elo.

**Los parámetros no se sobreajustaron.** El barrido
(`scripts/barrido_hiperparametros.py`, 24 configuraciones × 3 temporadas) da una
superficie plana: entre media vida de 240 y 1.095 días el RPS varía 0,0001,
frente a un error estándar de ±0,004 sobre 1.140 partidos. **El barrido no
discrimina**, así que se eligieron valores interpretables (365 días = una
temporada) en lugar de fingir que el óptimo numérico significa algo.

**Desempate exacto solo para pares en la simulación.** Con tres o más equipos
empatados a puntos el motor cae al criterio de diferencia general (la tabla
real sí resuelve el caso completo). Los empates triples son raros y el sesgo
sobre las probabilidades finales es de décimas de punto porcentual.

**La identidad de un equipo es su `team_id`, nunca su nombre.** Los partidos
guardan dos identificadores, y el nombre solo se usa para pintar. El problema
no es ese, sino la puerta: football-data.co.uk no publica identificadores, solo
un nombre corto, y lo cambia sin avisar (en 2026-27, a mitad de la primera
jornada, `Ath Madrid` pasó a `Atl. Madrid` y `Vallecano` a `Rayo Vallecano`).
Un nombre que no estaba en el mapa curado abría ficha nueva, y con ella un
equipo 21 y un partido 381, porque el mismo Atlético–Málaga entraba dos veces
con dos identidades distintas.

Ahora hay tres defensas, de fuera adentro: el mapa curado `CANONICAL_NAMES`;
el catálogo de clubes de openfootball como red de seguridad, que ya sabía que
`Atl. Madrid` es el Atlético y devuelve la ficha que ya existía; y el chequeo
de integridad, que aborta antes de simular si el recuento no cuadra. Solo se
crea ficha si ninguna de las dos primeras reconoce el nombre, que es lo que
debe pasar con un club recién ascendido de verdad.

---

## Hoja de ruta

**Las tres fases están hechas.** Lo que queda son mejoras concretas, no fases:

- **Datos de lesiones y entrenadores.** El marco de la fase 3 los acepta por la
  tabla `team_adjustments` (`simliga ajuste`), pero hay que meterlos a mano: no
  hay fuente gratuita fiable. Un scraper de Transfermarkt los automatizaría. Que
  el efecto medido de la fatiga sea nulo no dice nada sobre estos: una baja
  clave sí debería notarse, simplemente no hay datos para comprobarlo.
- **Motivación por simulación.** Hoy se evalúa con la clasificación de la fecha
  de corte, no con la de cada simulación. Para una proyección de pretemporada
  eso es una aproximación pobre. Solo compensa arreglarlo si alguna vez se
  demuestra que el efecto existe.
- **xG como señal adicional.** Understat responde correctamente. Sustituir goles
  por xG en el ajuste, o mezclarlos, es la vía más prometedora que queda para
  mejorar de verdad, porque ataca la estimación de fuerza y no un modificador
  marginal.
- **Europa League y Conference 2025-26** no están en openfootball (solo las
  rondas previas).
- **El sorteo europeo de 2026-27** no está publicado. Mientras tanto el panel da
  una proyección provisional sobre un campo estimado (los participantes reales
  del año pasado con los españoles sustituidos por los de este), promediando
  sobre 25 sorteos. `simliga calendario-uefa --temporadas 2026-27` recogerá el
  sorteo real en cuanto salga y la proyección pasará a ser la de verdad, sin
  tocar código.
- **El cuadro de eliminatorias** modela el sorteo como aleatorio dentro de cada
  par de cabezas de serie, sin las restricciones reales por país.

Fuera de alcance por ahora: la Copa del Rey.

---

## Estructura

```
simliga/
  config.py            parámetros ajustables (nada hardcodeado en la lógica)
  db.py                esquema SQLite; una tabla `matches` para liga y UEFA
  data.py              lectura a DataFrames, clasificación, integridad
  pipeline.py          ensamblaje: datos -> Elo -> ajuste -> simulación
  ingest/
    football_data_uk.py  resultados y cuotas (12 ligas europeas)
    espn.py              fechas y horarios confirmados de LaLiga
    openfootball.py      calendario de temporadas futuras
    uefa.py              sorteo y resultados de Champions/Europa/Conference
    club_names.py        puente de nombres entre fuentes (8.286 alias)
    fixtures.py          round-robin sintético de respaldo
  model/
    elo.py               rating dinámico
    league_strength.py   desplazamiento por fuerza de liga
    dixon_coles.py       modelo de goles
    modifiers.py         ajustes cualitativos (desactivados por defecto)
  sim/
    league.py            motor Monte Carlo de liga
    uefa.py              liguilla, playoff y cuadro de eliminatorias
  validation/            metrics.py, backtest.py, uefa_backtest.py
  output/
    contract.py          construcción del JSON
    dashboard.py/.html   panel HTML autocontenido
    motor.js             el mismo Monte Carlo, en JavaScript, para el navegador
    icono.png            icono de la pantalla de inicio (va incrustado)
  cli.py                 interfaz de línea de comandos
docs/
  data-contract.md     esquema del JSON, para construir el dashboard
  arquitectura.md      cómo funciona y cómo ponerle una interfaz
Dockerfile             imagen para desplegarlo en un hosting
arranque.sh            arranque del contenedor (exige SIMLIGA_TOKEN)
fly.toml               configuración de Fly.io
scripts/               validación completa, barrido, análisis de calibración
tests/                 323 tests
```

## Fuentes de datos

Estado comprobado con sondas reales (agosto 2026):

| Fuente | Uso | Estado |
|---|---|---|
| football-data.co.uk | Resultados y cuotas de 12 ligas europeas | ✅ en uso, sin clave |
| ESPN (API pública) | Fechas y horarios confirmados de LaLiga | ✅ en uso, sin clave |
| openfootball/espana | Calendario de temporadas futuras (CC0) | ✅ en uso, sin clave |
| openfootball/champions-league | Sorteo y resultados UEFA (CC0) | ✅ en uso, sin clave |
| openfootball/clubs | Alias de nombres entre fuentes (CC0) | ✅ en uso, sin clave |
| Understat | xG y estadísticas avanzadas | ✅ responde; pendiente para fase 3 |
| API-Football | Calendario, resultados, lesiones | ⚠️ clave válida, plan gratuito solo cubre 2022-2024 |
| football-data.org | Calendario y clasificación | ⚠️ requiere token, no configurado |
| ClubElo | Ratings europeos | ❌ no accesible (timeout); sustituido por Elo propio |
| FBref | Estadísticas avanzadas | ❌ 403 desde este entorno |
| Transfermarkt | Plantillas, fichajes, lesiones | fase 3, solo señal secundaria |

La clave de API-Football del proyecto `celta-dashboard` es válida (plan Free
activo hasta agosto de 2027, 100 peticiones/día) pero devuelve
`Free plans do not have access to this season, try from 2022 to 2024` para
LaLiga 2025-26 y 2026-27, así que **no sirve para la temporada actual**. Por eso
el calendario viene de openfootball.
