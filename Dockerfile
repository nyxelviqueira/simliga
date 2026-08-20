# Imagen del panel para desplegarlo en un hosting (Fly.io, Render, Railway...).
#
# Solo hace falta si quieres el boton "Actualizar datos" desde el movil sin
# tener el PC encendido. Para todo lo demas basta con `out/panel.html`, que
# simula por su cuenta y no necesita servidor de ninguna clase.
#
# Ver la seccion "Llevarlo al movil" del README para el paso a paso.

FROM python:3.12-slim

# Las dependencias primero, en su propia capa: cambian mucho menos que el
# codigo, asi que un despliegue normal no vuelve a compilar numpy y scipy.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simliga/ ./simliga/
COPY data/simliga.sqlite ./semilla/simliga.sqlite

# La base de datos de trabajo vive en el volumen, no en la imagen. El arranque
# la copia desde la semilla solo si el volumen esta vacio (primer despliegue):
# si no, se sobrescribirian los datos descargados desde el ultimo reinicio.
ENV SIMLIGA_DATA=/datos \
    PYTHONUNBUFFERED=1
VOLUME /datos

COPY arranque.sh .
RUN chmod +x arranque.sh

EXPOSE 8080
CMD ["./arranque.sh"]
