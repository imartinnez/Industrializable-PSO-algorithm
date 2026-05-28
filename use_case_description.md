# Caso de uso: refrigeración de un data center con PSO

Este documento explica el caso de uso aplicado del proyecto. La idea es coger el PSO que ya tenemos montado y aplicarlo a un problema con cierto sentido en ingeniería: ajustar cómo se refrigera un data center para gastar menos energía sin que los racks pasen de cierta temperatura.

**No es una simulación física real**. Es un modelo simplificado a propósito para que el problema sea defendible como ejercicio de optimización, no un dimensionamiento real de instalación. Las limitaciones están al final del documento.

---

## Por qué este caso

Los data centers consumen mucha energía y una parte se va en refrigeración. Bajar el setpoint del aire frío evita problemas térmicos, pero dispara el consumo del chiller. Subirlo ahorra energía pero arriesga sobrecalentar racks. Lo mismo pasa con los ventiladores y el reparto de caudal por zonas: cuanto más rápido vayan, más enfrían, pero más consumen. Hay un trade-off claro entre energía y seguridad térmica, y un PSO es un buen candidato para encontrar un punto razonable.

## Qué se optimiza

Cada partícula del PSO representa una configuración completa del data center:

```
x = [T_set,
     fan_speed_1, ..., fan_speed_4,
     airflow_zone_1, ..., airflow_zone_4]
```

- `T_set`: temperatura del aire frío del chiller, entre 18 y 26 °C.
- `fan_speed_j`: velocidad del ventilador j (4 ventiladores en las esquinas), entre 0.3 y 1.0.
- `airflow_zone_z`: caudal relativo en la zona z (los 4 cuadrantes del centro), entre 0.2 y 1.0.

Total: 9 variables continuas.

### Detalle de implementación: normalización

El PSO de este proyecto acepta un único par de bounds `(low, high)` para todas las dimensiones, pero aquí los bounds son distintos (18-26, 0.3-1.0, 0.2-1.0). Para no tocar el núcleo, trabajo internamente con variables normalizadas en `[0, 1]` y las decodifico a unidades físicas dentro de la propia función objetivo. El PSO ve un problema en `[0, 1]^9`, y la función objetivo se encarga de pasar a unidades reales antes de calcular nada.

## Layout

20 racks distribuidos en una rejilla de 4 filas × 5 columnas. 4 ventiladores en las cuatro esquinas. 4 zonas de caudal cubriendo los cuadrantes del grid.

Las cargas térmicas de cada rack (`L_r`) y unos pequeños sesgos térmicos fijos (`eta_r`) se generan una vez con una seed para que la escena sea reproducible.

## Modelo de refrigeración por rack

Para cada rack `r` calculo una refrigeración efectiva:

```
F_r(x) = F_min + sum_j A[r, j] * s_j + sum_z B[r, z] * q_z
```

- `A` es una matriz de influencia ventilador → rack. La construyo con un decaimiento gaussiano en distancia (los ventiladores enfrían más a los racks cercanos) y normalizo cada columna para que el total de cooling que entrega cada ventilador sume 1.
- `B` asigna cada rack a una zona (matriz one-hot 20×4).
- `F_min` es un valor mínimo para evitar divisiones por cero más abajo.

## Modelo de temperatura

La temperatura simulada en cada rack es:

```
T_r(x) = T_set
       + alpha * L_r / (F_r(x) + epsilon)
       + beta * sum_k C[r, k] * L_k / (F_k(x) + epsilon)
       + eta_r
```

- El primer término es el aire frío que entrega el chiller.
- El segundo es el calor propio del rack disipado por su propia refrigeración. Si tiene poco cooling, sube más.
- El tercero es el acoplamiento térmico con racks vecinos. Si los vecinos están calientes y poco refrigerados, te contagian un poco. La matriz `C` también decae gaussianamente con la distancia, y tiene diagonal cero para no contarse a sí mismo.
- `eta_r` es un sesgo fijo por rack para introducir algo de heterogeneidad.

Esto no es CFD ni nada parecido. No hay turbulencia, ni mezcla real de aire, ni dinámica temporal. Es una aproximación estática suficiente para que el problema sea un problema de optimización con sentido.

## Consumo energético

```
E(x) = E_chiller(T_set) + E_fans(s) + E_airflow(q)
```

Cada uno:

- `E_chiller(T_set) = P_chiller_ref * exp(gamma * (T_ref - T_set))`. Bajar el setpoint cuesta exponencialmente más, como pasa en chillers reales.
- `E_fans(s) = P_fan_max * sum_j s_j^3`. La famosa ley cúbica de los ventiladores: doblar velocidad multiplica por 8 el consumo.
- `E_airflow(q) = P_flow_max * sum_z q_z^3`. La misma idea para los caudales por zona.

Con esto ya tenemos el trade-off completo: bajar T_set baja temperaturas pero dispara el chiller, subirla ahorra chiller pero exige más cooling por ventiladores y aire, y eso a su vez cuesta cubo de la velocidad.

## Función objetivo

```
fitness(x) = E(x) / E_baseline
           + lambda_safe * P_safe(x)
           + lambda_hot * P_hot(x)
           + lambda_over * P_over(x)
           + lambda_balance * P_balance(x)
           + lambda_change * P_change(x)
```

El primer término es la energía normalizada por la del baseline. Si fuera 1.0 significa que la solución consume lo mismo que la operativa actual. Si baja a 0.85 significa 15 % de ahorro, por ejemplo.

Después vienen las cinco penalizaciones:

- `P_safe`: media de cuánto exceden los racks la temperatura segura (T_safe = 27 °C). Es la penalización principal, peso 50.
- `P_hot`: castiga el rack más caliente respecto al umbral de hotspot (T_hot = 30 °C). Más agresiva, peso 100. Solo dispara si hay alguien muy caliente.
- `P_over`: castiga sobreenfriamiento por debajo de T_min = 18 °C, que sería tirar energía. Peso 5.
- `P_balance`: la varianza térmica entre racks. Premia distribuciones equilibradas. Peso 2.
- `P_change`: distancia respecto al baseline en unidades normalizadas. Evita soluciones que se alejan radicalmente de la configuración operativa actual. Peso 1.

Los pesos son los que dimos en el enunciado. La idea es que las dos primeras dominen cuando algo se calienta, y las otras tres moldeen la solución sin imponerse.

## Baseline

```
x_baseline = [22 °C, 0.75, 0.75, 0.75, 0.75, 0.70, 0.70, 0.70, 0.70]
```

Una configuración intermedia razonable. Se evalúa primero y todo se compara contra ella. El consumo del baseline se usa además para normalizar la parte energética del fitness.

## Cómo se ejecuta

```bash
python -m pso.run_scripts.run_datacenter_case
```

El script:

1. Construye el escenario con seed fija 42.
2. Evalúa el baseline.
3. Lanza el PSO (40 partículas, 500 iteraciones, modo secuencial, estrategia clamp).
4. Evalúa la solución encontrada.
5. Imprime una comparación en consola.
6. Guarda CSVs, JSON y figuras en `pso/results/datacenter_<timestamp>/`.

## Qué outputs genera

En la carpeta de la ejecución vas a encontrar:

- `summary.csv`: una fila por baseline y otra por PSO con las métricas principales.
- `convergence.csv`: curva de mejor fitness por iteración.
- `config.json`: configuración del escenario, del PSO y resumen de resultados.
- `run.log`: log de la ejecución.
- `heatmap_baseline.png` y `heatmap_pso.png`: temperaturas de los 20 racks en formato 4×5.
- `energy_comparison.png`: barra simple de energía total baseline vs PSO.
- `energy_breakdown.png`: desglose de energía (chiller, fans, airflow).
- `convergence.png`: la curva de convergencia del PSO.
- `variable_choice.png`: comparativa de las variables óptimas vs baseline.

## Cuaderno de análisis

El cuaderno `pso/analysis/datacenter_analysis.ipynb` carga la ejecución más reciente y reconstruye el escenario a partir de la seed para reevaluar baseline y PSO usando los módulos del paquete `use_case/`. Produce las mismas figuras que el script pero inline, con bloques de markdown explicando qué muestra cada una y cómo leer los resultados.

```bash
jupyter notebook pso/analysis/datacenter_analysis.ipynb
```

Es la vista pensada para revisar los resultados con calma: tablas comparativas, curva de convergencia, desglose energético, mapas térmicos y variables óptimas. Cada sección incluye un párrafo de discusión.

## Dashboard interactivo (`app.py`)

Para cuando hay que enseñarle los resultados al cliente, además del cuaderno hay un dashboard hecho con Streamlit que vive en la raíz del proyecto (`app.py`). Lee la carpeta `pso/results/` y muestra una vista web pensada como entregable comercial:

- **KPIs grandes** arriba: ahorro energético, energía total, temperatura máxima, número de hotspots y mejora de fitness.
- **Comparativa energética** con gráficos interactivos (Plotly): energía total y desglose por componente (chiller, ventiladores, caudal).
- **Mapas térmicos** lado a lado (baseline y PSO) con escala de color común para que la comparación sea visual.
- **Curva de convergencia** con la línea del baseline como referencia.
- **Variables de control** optimizadas: setpoint, velocidades de los ventiladores y caudales por zona, todas comparadas con sus valores baseline.
- **Resumen ejecutivo** en forma de tabla con todas las métricas y la diferencia absoluta y relativa.
- **Sidebar** con la configuración del escenario, los parámetros del PSO usados, y un selector que permite cambiar entre las distintas ejecuciones que tengas guardadas en `pso/results/datacenter_*`.

Para abrirlo:

```bash
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`. No guarda nada nuevo: lee los resultados ya generados por `run_datacenter_case`.

La paleta es corporativa (navy + emerald) y el chrome de Streamlit está ocultado con un bloque de CSS para que el resultado se vea más como una página web propia que como un script de prototipado. La idea es que si alguien tiene que enseñar esto en una reunión con cliente, no quede ridículo.

## Cómo interpretar los resultados

Si el PSO funciona, esperamos ver:

- Un `fitness` menor que el del baseline.
- Un consumo total más bajo, con porcentaje de ahorro positivo.
- Ningún rack en peligro de sobrecalentamiento por encima del límite duro (T_hot).
- Una temperatura máxima cerca o por debajo de T_safe.
- Una std térmica baja, señal de un data center equilibrado.

La curva de convergencia debería bajar de forma monótona (el motor garantiza que el mejor global nunca empeora).

Un caveat: el PSO está buscando el trade-off entre energía y temperatura, así que puede acercarse al límite de seguridad. Si quieres que se quede más conservador, sube `lambda_safe`. Si quieres apurar más el ahorro, baja `lambda_change` para que no se aferre al baseline.

## Limitaciones

Lo digo claro porque importa: este caso de uso **no es realista al 100 %**. Lo simplifiqué a propósito para que sea defendible como ejercicio académico.

Lo que NO modela:

- Convección, turbulencia, mezcla real de aire. La temperatura es una función estática de la configuración, no resultado de una simulación física.
- Dinámica temporal. No hay evolución en el tiempo, ni inercia térmica, ni transitorios.
- Acoplamientos no lineales realistas entre setpoint, caudales y temperatura.
- Restricciones operativas concretas (humedad, ruido, fallos parciales, redundancia, capacidades máximas reales del chiller).
- Variabilidad de carga. Las cargas térmicas se fijan al inicio y no cambian.
- Eficiencia real de un chiller, que depende de muchísimas más variables.
- Efectos típicos de pasillo caliente / pasillo frío.

Lo que SÍ modela razonablemente:

- El trade-off central entre energía y temperatura.
- La ley cúbica de los ventiladores.
- El crecimiento exponencial del coste del chiller al bajar el setpoint.
- Una influencia local de los ventiladores (gaussiana en distancia).
- Un acoplamiento débil entre racks vecinos.
- Restricciones de seguridad como penalizaciones suaves.

En resumen: es un problema de optimización continua con la estructura típica de un caso real (trade-off claro, restricciones blandas, dimensión moderada, bounds heterogéneos). Sirve para mostrar que el PSO encuentra configuraciones mejores que la operativa por defecto, pero no debe interpretarse como un dimensionamiento real de una instalación.

## Reproducibilidad

Todo está fijado por la seed 42:

- Generación del escenario (cargas y sesgos).
- Inicialización del PSO.
- Cada iteración del PSO genera su propia subseed de forma determinista para que el resultado sea exactamente reproducible.

Ejecutar el script dos veces debe dar exactamente los mismos números.