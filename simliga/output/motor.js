// Motor Monte Carlo en el navegador.
//
// Es el mismo modelo que `simliga/sim/league.py`, reescrito en JavaScript para
// que el panel pueda volver a simular sin ningun proceso detras: en el movil,
// sin conexion y con el PC apagado.
//
// Lo que NO hace es reajustar el modelo. El ajuste Dixon-Coles y el Elo se
// calculan en Python sobre 50.000 partidos y llegan ya hechos dentro del JSON
// (mu, ventaja de campo, rho, y el ataque y la defensa de cada equipo). Aqui
// solo se muestrean partidos con esas tasas, que es la parte barata.
//
// Esa division es lo que hace viable el movil: reajustar exigiria el historico
// entero; muestrear solo necesita veintitantos numeros por equipo.
//
// El fichero se incrusta tal cual dentro del panel y ademas se puede cargar
// desde Node, que es como `tests/test_motor_js.py` comprueba que da los mismos
// numeros que la version de Python.

(function (raiz) {
  "use strict";

  var MAX_GOLES = 10;                    // rejilla de marcadores: 0..10 por equipo
  var LADO = MAX_GOLES + 1;
  var CELDAS = LADO * LADO;

  // Logaritmos de factoriales, para la densidad de Poisson.
  var LOG_FACT = new Float64Array(LADO);
  for (var k = 1; k < LADO; k++) LOG_FACT[k] = LOG_FACT[k - 1] + Math.log(k);

  /**
   * Distribucion conjunta de marcadores de un partido, acumulada.
   *
   * Poisson independientes salvo por la correccion de Dixon-Coles, que reajusta
   * los cuatro marcadores bajos (0-0, 1-0, 0-1, 1-1), donde el Poisson puro se
   * queda corto respecto a lo que se observa de verdad.
   */
  function cdfMarcador(lamLocal, lamVisita, rho) {
    var pLocal = new Float64Array(LADO);
    var pVisita = new Float64Array(LADO);
    for (var g = 0; g < LADO; g++) {
      pLocal[g] = Math.exp(g * Math.log(lamLocal) - lamLocal - LOG_FACT[g]);
      pVisita[g] = Math.exp(g * Math.log(lamVisita) - lamVisita - LOG_FACT[g]);
    }

    var cdf = new Float64Array(CELDAS);
    var total = 0;
    for (var h = 0; h < LADO; h++) {
      for (var a = 0; a < LADO; a++) {
        var p = pLocal[h] * pVisita[a];
        if (h === 0 && a === 0) p *= 1 - lamLocal * lamVisita * rho;
        else if (h === 0 && a === 1) p *= 1 + lamLocal * rho;
        else if (h === 1 && a === 0) p *= 1 + lamVisita * rho;
        else if (h === 1 && a === 1) p *= 1 - rho;
        if (p < 1e-15) p = 1e-15;        // un rho grande puede dar negativos
        total += p;
        cdf[h * LADO + a] = total;
      }
    }
    for (var i = 0; i < CELDAS; i++) cdf[i] /= total;
    return cdf;
  }

  /** Celda sorteada: busqueda binaria sobre la acumulada. */
  function sortearCelda(cdf, u) {
    var bajo = 0;
    var alto = CELDAS - 1;
    while (bajo < alto) {
      var medio = (bajo + alto) >> 1;
      if (cdf[medio] < u) bajo = medio + 1;
      else alto = medio;
    }
    return bajo;
  }

  /**
   * Generador reproducible (mulberry32).
   *
   * Con la misma semilla salen los mismos numeros. Eso permite comparar dos
   * escenarios sabiendo que la diferencia viene de los resultados que has
   * puesto y no del azar del muestreo, que es justo para lo que sirve el
   * calendario editable.
   */
  function generador(semilla) {
    var a = semilla >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Estado de partida: puntos y goles que cada equipo ya tiene.
   *
   * Cuentan tanto los partidos jugados de verdad como los hipoteticos, porque
   * para la simulacion ambos son hechos consumados. Lo que los distingue se
   * lleva aparte (`jugadosReales` / `hipoteticos`) para que la tabla pueda
   * decir cuantos de esos puntos son reales.
   *
   * Ganados, empatados, perdidos y la racha no los necesita la simulacion: los
   * usa la pestana de clasificacion simulada. Se calculan aqui de todas formas
   * porque es la misma pasada sobre el calendario, y tener dos sitios que
   * cuentan lo mismo es como se acaba con dos tablas que no cuadran.
   */
  function estadoInicial(calendario, indicePorId, n) {
    var e = {
      puntos: new Int16Array(n), favor: new Int16Array(n), contra: new Int16Array(n),
      jugados: new Int16Array(n), jugadosReales: new Int16Array(n),
      hipoteticos: new Int16Array(n),
      ganados: new Int16Array(n), empatados: new Int16Array(n), perdidos: new Int16Array(n),
      h2hPuntos: new Int16Array(n * n), h2hGoles: new Int16Array(n * n),
      forma: [],
    };
    for (var t = 0; t < n; t++) e.forma.push([]);

    var jornadas = (calendario && calendario.matchdays) || [];
    for (var j = 0; j < jornadas.length; j++) {
      var partidos = jornadas[j].matches;
      for (var m = 0; m < partidos.length; m++) {
        var p = partidos[m];
        if (p.status === "pending" || p.home_goals === null) continue;
        var l = indicePorId.get(p.home_team.team_id);
        var v = indicePorId.get(p.away_team.team_id);
        if (l === undefined || v === undefined) continue;

        var gl = p.home_goals, gv = p.away_goals;
        var ptsL = gl > gv ? 3 : (gl === gv ? 1 : 0);
        var ptsV = gv > gl ? 3 : (gl === gv ? 1 : 0);

        e.puntos[l] += ptsL; e.puntos[v] += ptsV;
        e.favor[l] += gl; e.contra[l] += gv;
        e.favor[v] += gv; e.contra[v] += gl;
        e.jugados[l] += 1; e.jugados[v] += 1;
        var hipotesis = p.status === "scenario";
        if (hipotesis) { e.hipoteticos[l] += 1; e.hipoteticos[v] += 1; }
        else { e.jugadosReales[l] += 1; e.jugadosReales[v] += 1; }

        if (gl > gv) { e.ganados[l] += 1; e.perdidos[v] += 1; }
        else if (gl < gv) { e.ganados[v] += 1; e.perdidos[l] += 1; }
        else { e.empatados[l] += 1; e.empatados[v] += 1; }
        e.forma[l].push({ r: ptsL === 3 ? "G" : ptsL === 1 ? "E" : "P", hipotesis: hipotesis });
        e.forma[v].push({ r: ptsV === 3 ? "G" : ptsV === 1 ? "E" : "P", hipotesis: hipotesis });

        e.h2hPuntos[l * n + v] += ptsL; e.h2hPuntos[v * n + l] += ptsV;
        e.h2hGoles[l * n + v] += gl - gv; e.h2hGoles[v * n + l] += gv - gl;
      }
    }
    return e;
  }

  /** Partidos que quedan por simular, con sus tasas de gol ya convertidas en CDF. */
  function partidosPendientes(doc, equipos, indicePorId) {
    var dc = doc.model.dixon_coles;
    var mu = dc.mu, gamma = dc.home_advantage, rho = dc.rho;
    var ataque = equipos.map(function (t) { return t.ratings.attack; });
    var defensa = equipos.map(function (t) { return t.ratings.defence; });

    var lista = [];
    var jornadas = (doc.calendar && doc.calendar.matchdays) || [];
    for (var j = 0; j < jornadas.length; j++) {
      var partidos = jornadas[j].matches;
      for (var m = 0; m < partidos.length; m++) {
        var p = partidos[m];
        if (p.status !== "pending" && p.status !== "live") continue;
        var l = indicePorId.get(p.home_team.team_id);
        var v = indicePorId.get(p.away_team.team_id);
        if (l === undefined || v === undefined) continue;

        var lamLocal = Math.exp(mu + ataque[l] - defensa[v] + gamma);
        var lamVisita = Math.exp(mu + ataque[v] - defensa[l]);
        lista.push({ local: l, visitante: v, cdf: cdfMarcador(lamLocal, lamVisita, rho) });
      }
    }
    return lista;
  }

  /**
   * Ordena una simulacion con los desempates de LaLiga.
   *
   * Primero puntos. Entre los empatados, una mini-liga con solo los partidos
   * entre ellos: puntos y luego diferencia de goles de esos partidos. Si sigue
   * el empate, diferencia general y goles a favor.
   *
   * No es lo mismo que ordenar por diferencia general, y la distincion decide
   * descensos de verdad: en 2025-26 Levante y Mallorca acabaron a 42 puntos y
   * bajo el Mallorca por el enfrentamiento directo.
   */
  function ordenar(orden, puntos, diferencia, favor, h2hPuntos, h2hGoles, n) {
    for (var i = 0; i < n; i++) orden[i] = i;
    orden.sort(function (a, b) {
      return (puntos[b] - puntos[a])
          || (diferencia[b] - diferencia[a])
          || (favor[b] - favor[a]);
    });

    var ini = 0;
    while (ini < n) {
      var fin = ini + 1;
      while (fin < n && puntos[orden[fin]] === puntos[orden[ini]]) fin++;
      if (fin - ini > 1) {
        var grupo = Array.prototype.slice.call(orden, ini, fin);
        grupo.sort(function (a, b) {
          var ptsA = 0, ptsB = 0, gdA = 0, gdB = 0;
          for (var g = 0; g < grupo.length; g++) {
            var otro = grupo[g];
            if (otro !== a) { ptsA += h2hPuntos[a * n + otro]; gdA += h2hGoles[a * n + otro]; }
            if (otro !== b) { ptsB += h2hPuntos[b * n + otro]; gdB += h2hGoles[b * n + otro]; }
          }
          return (ptsB - ptsA) || (gdB - gdA)
              || (diferencia[b] - diferencia[a])
              || (favor[b] - favor[a]);
        });
        for (var k = ini; k < fin; k++) orden[k] = grupo[k - ini];
      }
      ini = fin;
    }
    return orden;
  }

  /** Percentil sobre un array YA ordenado, con interpolacion lineal como numpy. */
  function percentil(ordenados, q) {
    var pos = (ordenados.length - 1) * q;
    var bajo = Math.floor(pos);
    var alto = Math.ceil(pos);
    if (bajo === alto) return ordenados[bajo];
    return ordenados[bajo] + (ordenados[alto] - ordenados[bajo]) * (pos - bajo);
  }

  function redondear(x, decimales) {
    var f = Math.pow(10, decimales);
    return Math.round(x * f) / f;
  }

  /**
   * Simula la temporada `nSims` veces y devuelve el bloque de LaLiga rehecho.
   *
   * Se trocea por bloques y se cede el hilo entre ellos: en un movil una tirada
   * larga pasa del segundo, y una pagina que no responde parece rota aunque
   * este trabajando. `alProgresar` recibe la fraccion completada.
   */
  async function simular(doc, nSims, alProgresar) {
    var liga = doc.competitions.ESP1;
    var equipos = liga.teams;
    var n = equipos.length;
    var indicePorId = new Map(equipos.map(function (t, i) { return [t.team_id, i]; }));

    var inicial = estadoInicial(doc.calendar, indicePorId, n);
    var pendientes = partidosPendientes(doc, equipos, indicePorId);
    var azar = generador((doc.simulation && doc.simulation.seed) || 1);

    var puntos = new Int16Array(n);
    var favor = new Int16Array(n);
    var contra = new Int16Array(n);
    var diferencia = new Int16Array(n);
    var h2hPuntos = new Int16Array(n * n);
    var h2hGoles = new Int16Array(n * n);
    var orden = new Int32Array(n);

    // Acumuladores del resultado. Puntos y diferencia se guardan por simulacion
    // porque hacen falta enteros sueltos para sacar percentiles.
    var cuentaPuestos = new Int32Array(n * n);
    var sumaPuestos = new Float64Array(n);
    var puntosPorSim = new Int16Array(nSims * n);
    var difPorSim = new Int16Array(nSims * n);

    var BLOQUE = 400;
    for (var desde = 0; desde < nSims; desde += BLOQUE) {
      var hasta = Math.min(desde + BLOQUE, nSims);
      for (var s = desde; s < hasta; s++) {
        puntos.set(inicial.puntos);
        favor.set(inicial.favor);
        contra.set(inicial.contra);
        h2hPuntos.set(inicial.h2hPuntos);
        h2hGoles.set(inicial.h2hGoles);

        for (var m = 0; m < pendientes.length; m++) {
          var p = pendientes[m];
          var celda = sortearCelda(p.cdf, azar());
          var gl = (celda / LADO) | 0;
          var gv = celda % LADO;
          var ptsL = gl > gv ? 3 : (gl === gv ? 1 : 0);
          var ptsV = gv > gl ? 3 : (gl === gv ? 1 : 0);
          var l = p.local, v = p.visitante;

          puntos[l] += ptsL; puntos[v] += ptsV;
          favor[l] += gl; contra[l] += gv;
          favor[v] += gv; contra[v] += gl;
          h2hPuntos[l * n + v] += ptsL; h2hPuntos[v * n + l] += ptsV;
          h2hGoles[l * n + v] += gl - gv; h2hGoles[v * n + l] += gv - gl;
        }

        var base = s * n;
        for (var t = 0; t < n; t++) {
          diferencia[t] = favor[t] - contra[t];
          puntosPorSim[base + t] = puntos[t];
          difPorSim[base + t] = diferencia[t];
        }
        ordenar(orden, puntos, diferencia, favor, h2hPuntos, h2hGoles, n);
        for (var puesto = 0; puesto < n; puesto++) {
          cuentaPuestos[orden[puesto] * n + puesto] += 1;
          sumaPuestos[orden[puesto]] += puesto + 1;
        }
      }

      if (alProgresar) alProgresar(hasta / nSims);
      // Ceder el hilo: sin esto el navegador se congela durante toda la tirada.
      if (typeof setTimeout === "function") {
        await new Promise(function (listo) { setTimeout(listo, 0); });
      }
    }

    return construirBloque(liga, equipos, n, nSims, inicial, {
      cuentaPuestos: cuentaPuestos, sumaPuestos: sumaPuestos,
      puntosPorSim: puntosPorSim, difPorSim: difPorSim,
    });
  }

  /** Rehace el bloque ESP1 con las cifras nuevas, en el formato que emite Python. */
  function construirBloque(liga, equipos, n, nSims, inicial, acc) {
    var reglas = liga.qualification_rules;

    function resumen(valores) {
      var suma = 0;
      for (var i = 0; i < nSims; i++) suma += valores[i];
      var media = suma / nSims;
      var varianza = 0;
      for (var j = 0; j < nSims; j++) varianza += (valores[j] - media) * (valores[j] - media);
      var ordenados = Array.prototype.slice.call(valores).sort(function (a, b) { return a - b; });
      return {
        mean: redondear(media, 2), sd: redondear(Math.sqrt(varianza / nSims), 2),
        p05: percentil(ordenados, 0.05), p25: percentil(ordenados, 0.25),
        median: percentil(ordenados, 0.50), p75: percentil(ordenados, 0.75),
        p95: percentil(ordenados, 0.95),
      };
    }

    var nuevos = equipos.map(function (equipo, i) {
      var probs = [];
      for (var p = 0; p < n; p++) probs.push(acc.cuentaPuestos[i * n + p] / nSims);

      var suma = function (rango) {
        if (!rango) return 0;
        var total = 0;
        for (var q = rango[0]; q <= rango[1]; q++) total += probs[q - 1];
        return redondear(total, 5);
      };

      var misPuntos = new Int16Array(nSims);
      var misDif = new Int16Array(nSims);
      for (var s = 0; s < nSims; s++) {
        misPuntos[s] = acc.puntosPorSim[s * n + i];
        misDif[s] = acc.difPorSim[s * n + i];
      }

      // Los puestos se reconstruyen desde el recuento en vez de guardarse por
      // simulacion: sale la misma lista ordenada, sin otro array de nSims x n.
      var puestos = new Int16Array(nSims);
      var escrito = 0;
      var modo = 0;
      for (var pos = 0; pos < n; pos++) {
        var veces = acc.cuentaPuestos[i * n + pos];
        if (veces > acc.cuentaPuestos[i * n + modo]) modo = pos;
        for (var w = 0; w < veces; w++) puestos[escrito++] = pos + 1;
      }

      var ucl = suma(reglas.ucl), uel = suma(reglas.uel), uecl = suma(reglas.uecl);
      return {
        team_id: equipo.team_id,
        name: equipo.name,
        display_name: equipo.display_name,
        logo: equipo.logo,
        current: {
          played: inicial.jugados[i],
          played_real: inicial.jugadosReales[i],
          scenario_matches: inicial.hipoteticos[i],
          points: inicial.puntos[i],
          goals_for: inicial.favor[i],
          goals_against: inicial.contra[i],
          goal_difference: inicial.favor[i] - inicial.contra[i],
          position: null,
        },
        ratings: equipo.ratings,
        projection: {
          points: resumen(misPuntos),
          goal_difference: resumen(misDif),
          position: {
            mean: redondear(acc.sumaPuestos[i] / nSims, 2),
            mode: modo + 1,
            p05: percentil(puestos, 0.05),
            p95: percentil(puestos, 0.95),
          },
          position_probabilities: probs.map(function (x) { return redondear(x, 5); }),
        },
        outcomes: {
          title: redondear(probs[0], 5),
          ucl: ucl, uel: uel, uecl: uecl,
          european_qualification: redondear(ucl + uel + uecl, 5),
          // Ni Europa ni descenso. `suma` devuelve 0 si el rango no viene, asi
          // que un documento de un esquema anterior no rompe nada.
          mid_table: suma(reglas.mid_table),
          relegation: suma(reglas.relegation),
        },
      };
    });

    nuevos.sort(function (a, b) {
      return a.projection.position.mean - b.projection.position.mean;
    });

    // La posicion de partida se recalcula: con escenarios activos, la que traia
    // el documento es de antes de meterlos.
    var porTabla = nuevos.slice().sort(function (a, b) {
      return (b.current.points - a.current.points)
          || (b.current.goal_difference - a.current.goal_difference)
          || (b.current.goals_for - a.current.goals_for);
    });
    porTabla.forEach(function (equipo, i) {
      equipo.current.position = equipo.current.played ? i + 1 : null;
    });

    var jugados = 0, reales = 0;
    nuevos.forEach(function (e) {
      jugados += e.current.played;
      reales += e.current.played_real;
    });
    return Object.assign({}, liga, {
      qualification_note: liga.qualification_note,
      teams: nuevos,
      matches_played: jugados / 2,
      matches_played_real: reales / 2,
      matches_remaining: (n * (n - 1)) - jugados / 2,
    });
  }

  var api = {
    simular: simular,
    cdfMarcador: cdfMarcador,
    ordenar: ordenar,
    generador: generador,
    estadoInicial: estadoInicial,
    MAX_GOLES: MAX_GOLES,
  };

  raiz.MotorSimLiga = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
