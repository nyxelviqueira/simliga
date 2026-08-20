# SimLiga en el movil con GitHub Pages

Esta es la forma recomendada para tener una URL fija sin depender del ordenador.
GitHub ejecuta la actualizacion, genera `out/panel.html`, lo copia como
`out/index.html` y publica la carpeta `out/` en GitHub Pages.

## 1. Crear el repositorio

En GitHub, crea un repositorio nuevo, por ejemplo `simliga`.

En esta carpeta:

```powershell
git init
git add .
git commit -m "Publicar SimLiga en GitHub Pages"
git branch -M main
git remote add origin https://github.com/nyxelviqueira/simliga.git
git push -u origin main
```

La base local `data/simliga.sqlite`, `data/raw/` y `out/` no se suben: estan en
`.gitignore`. El workflow los reconstruye y los cachea dentro de GitHub Actions.

## 2. Activar GitHub Pages

En el repositorio de GitHub:

1. Abre **Settings**.
2. Entra en **Pages**.
3. En **Build and deployment**, elige **Source: GitHub Actions**.

## 3. Primera publicacion

1. Abre la pestana **Actions**.
2. Entra en **Publicar panel movil**.
3. Pulsa **Run workflow**.
4. Deja `temporada = 2026-27` y `sims = 20000`.
5. Pulsa **Run workflow**.

La primera vez puede tardar mas porque descarga el historico desde 2010 y crea
la base de datos. Las siguientes reutilizan cache.

Cuando termine, el workflow ensena la URL publicada. Normalmente sera:

```text
https://nyxelviqueira.github.io/simliga/
```

## 4. Usarlo como app

En el movil, abre la URL publicada.

En iPhone:

1. Compartir.
2. **Anadir a pantalla de inicio**.

En Android/Chrome:

1. Menu de tres puntos.
2. **Anadir a pantalla de inicio** o **Instalar app**.

El panel incluye manifest, icono y service worker. Si el movil se queda sin
conexion, deberia abrir la ultima version cacheada.

## 5. Como se actualiza

El workflow se despierta cada 20 minutos entre las 11:00 y las 23:00 UTC (de la
una del mediodia a la una de la madrugada, hora espanola de verano), ademas de
cuando subes cambios a `main` o lo lanzas a mano desde **Actions > Publicar
panel movil > Run workflow**.

Despertarse no es publicar. Antes de hacer nada caro, el paso **Hay algo nuevo
que publicar** pregunta a la base si hay algun partido que ya deberia haber
terminado y del que aun no tenemos resultado. Si no lo hay, se salta el resto y
la ejecucion termina en medio minuto sin tocar la pagina.

En la practica eso significa que el panel se actualiza **poco despues de cada
partido** y no a intervalos fijos. Un partido se da por terminado a los 105
minutos de su hora de comienzo (90 mas descuento y descanso), asi que la
publicacion cae entre 5 y 25 minutos despues del pitido final, segun donde
caiga el ciclo.

Dos detalles del funcionamiento:

- **Se pregunta por lo que falta, no por lo que acaba de jugarse.** Asi la
  puerta se cierra sola en cuanto el resultado entra en la base, y sigue
  reintentando mientras la fuente tarde en publicarlo.
- **Un partido aplazado deja de contar a los tres dias.** Conserva su fecha
  vieja y nunca recibe resultado; sin ese tope mantendria la puerta abierta
  indefinidamente y volveriamos a publicar en cada ciclo.

Lanzandolo a mano se publica siempre, haya novedades o no: si lo pulsas tu es
porque lo quieres ahora.

En un repositorio publico, los runners estandar de GitHub Actions no consumen
minutos de pago; aun asi, GitHub puede retrasar una ejecucion programada si la
plataforma tiene carga. Y desactiva los cron de los repositorios sin actividad
durante 60 dias: si dejas de tocarlo mucho tiempo, hay que reactivarlo desde la
pestana Actions.

Desde el PC tambien puedes hacerlo sin entrar en GitHub con
`actualizar-github.bat`. Para eso hay que guardar una vez un token:

1. En GitHub, abre **Settings** de tu usuario.
2. Entra en **Developer settings > Personal access tokens > Fine-grained tokens**.
3. Crea un token para el repositorio `nyxelviqueira/simliga`.
4. Dale permiso **Actions: Read and write**.
5. Copia el token y guardalo en Windows:

```powershell
setx SIMLIGA_GITHUB_TOKEN "github_pat_TU_TOKEN"
```

Cierra esa terminal y abre una nueva. A partir de ahi, doble clic en
`actualizar-github.bat` lanza la publicacion remota con temporada `2026-27` y
`20000` simulaciones.

## 6. Limitaciones de esta opcion

La pagina publicada es estatica. Eso significa:

- los escenarios funcionan y se guardan en el navegador del movil;
- puedes cambiar simulaciones desde el movil;
- no hace falta tener el PC encendido;
- el boton **Actualizar datos** no aparece, porque no hay servidor Python detras.

Para actualizar datos se usa GitHub Actions. Si algun dia quieres que el boton
**Actualizar datos** funcione dentro de la web publica, entonces la alternativa
es desplegar el servidor con Docker/Fly.io y token privado.
