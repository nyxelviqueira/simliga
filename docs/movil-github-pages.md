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

El workflow se ejecuta:

- automaticamente cada 5 minutos;
- cada vez que subes cambios a `main`;
- manualmente desde **Actions > Publicar panel movil > Run workflow**.

Desde el movil puedes abrir GitHub, entrar en esa accion y pulsar **Run
workflow** cuando quieras forzar una actualizacion de resultados/calendario.

En un repositorio publico, los runners estandar de GitHub Actions no consumen
minutos de pago. El intervalo de 5 minutos es el minimo que permite GitHub para
workflow programados; aun asi, GitHub puede retrasar alguna ejecucion si la
plataforma tiene mucha carga.

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
