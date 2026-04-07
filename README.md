# PSO Algorithm — Particle Swarm Optimization

Implementación modular de Particle Swarm Optimization (PSO) en Python, diseñada como banco de pruebas para comparar distintas estrategias de evaluación paralela y concurrente aplicadas al algoritmo. El proyecto incluye una suite de benchmarks, grid search de hiperparámetros, visualización 2D/3D, persistencia estructurada de resultados y tests unitarios.

**Autor:** Íñigo Martínez Jiménez

---

## Tabla de contenidos

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Arquitectura y decisiones de diseño](#arquitectura-y-decisiones-de-diseño)
3. [Instalación](#instalación)
4. [Uso rápido](#uso-rápido)
5. [Comandos y scripts](#comandos-y-scripts)
6. [El algoritmo PSO](#el-algoritmo-pso)
7. [Funciones benchmark](#funciones-benchmark)
8. [Estrategias de paralelismo](#estrategias-de-paralelismo)
9. [Grid search de hiperparámetros](#grid-search-de-hiperparámetros)
10. [Persistencia y resultados](#persistencia-y-resultados)
11. [Visualización](#visualización)
12. [Tests](#tests)
13. [Reproducibilidad](#reproducibilidad)
14. [Dependencias](#dependencias)

---

## Estructura del proyecto

```
PSO_Algorithm/
├── README.md
├── .gitignore
├── app.py
└── pso/
    ├── __init__.py
    ├── core/                        # Motor del PSO
    │   ├── pso_engine.py            #   Bucle principal del optimizador
    │   ├── swarm.py                 #   Estado del enjambre (posiciones, velocidades, bests)
    │   └── result.py                #   Contenedor de resultados (@dataclass)
    ├── objectives/                  # Funciones objetivo
    │   ├── functions.py             #   Sphere, Rosenbrock, Rastrigin, Ackley
    │   └── registry.py              #   Registro con bounds, óptimos y metadatos
    ├── parallel/                    # Estrategias de evaluación
    │   ├── evaluator.py             #   Factory que selecciona el evaluador
    │   ├── V0_sequential.py         #   Evaluación secuencial (baseline)
    │   ├── V1_threading.py          #   Evaluación con hilos (ThreadPoolExecutor)
    │   ├── V2_multiprocessing.py    #   Evaluación con procesos (ProcessPoolExecutor)
    │   ├── V3_async.py              #   Evaluación asíncrona (asyncio) [en desarrollo]
    │   └── V4_vectorized.py         #   Evaluación vectorizada (NumPy) [en desarrollo]
    ├── experiments/                 # Orquestación de experimentos
    │   ├── benchmarks.py            #   Instance, make_instances, run_suite
    │   └── pyswarm_reference.py     #   Wrapper del baseline PySwarm externo
    ├── io/                          # Entrada/salida y utilidades
    │   ├── logging.py               #   Configuración del logger (consola + fichero)
    │   ├── paths.py                 #   Gestión de directorios timestamped
    │   └── save_results.py          #   Exportación a CSV y JSON
    ├── viz/                         # Visualización
    │   ├── animator.py              #   Animaciones 2D y 3D con matplotlib
    │   └── make_viz.py              #   Script para generar las animaciones
    ├── run_scripts/                 # Puntos de entrada principales
    │   ├── run.py                   #   Ejecución individual + comparación
    │   ├── run_benchmarks.py        #   Suite completa de benchmarks
    │   └── run_grid_search.py       #   Búsqueda de hiperparámetros
    ├── tests/                       # Tests unitarios
    │   └── tests_pso.py             #   (reproducibilidad, bounds, monotonía, etc.)
    └── results/                     # Resultados generados (no se sube a Git)
```

---

## Arquitectura y decisiones de diseño

### Diagrama de dependencias entre módulos

```
┌─────────────────────────────────────────────────────────────────────┐
│                         run_scripts/                                │
│             run.py  ·  run_benchmarks.py  ·  run_grid_search.py     │
└──────┬──────────────────┬───────────────────────┬───────────────────┘
       │                  │                       │
       ▼                  ▼                       ▼
┌─────────────┐   ┌──────────────┐        ┌─────────────┐
│ experiments/ │   │     viz/     │        │     io/     │
│ benchmarks   │   │  animator    │        │  logging    │
│ pyswarm_ref  │   │  make_viz    │        │  paths      │
└──────┬───────┘   └──────┬───────┘        │  save_results│
       │                  │                └──────┬──────┘
       ▼                  │                       │
┌─────────────┐           │                       │
│   core/     │◄──────────┘                       │
│  pso_engine │                                   │
│  swarm      │           (io/ es utilizado por   │
│  result     │            run_scripts/ y         │
└──────┬──────┘            experiments/)          │
       │                                          │
       ▼                                          │
┌─────────────┐   ┌──────────────┐                │
│ objectives/ │   │  parallel/   │                │
│  functions  │   │  evaluator   │◄───────────────┘
│  registry   │   │  V0..V4      │
└─────────────┘   └──────────────┘
```

### Decisiones principales

**1. Representación del enjambre con matrices NumPy, no con objetos Particle.**

Aunque lo habitual es modelar cada partícula como un objeto independiente, en este proyecto el enjambre se almacena como matrices NumPy dentro de la clase `Swarm` (posiciones, velocidades, personal bests, global best). El motivo no es simplemente rendimiento: el enunciado exige un core común del algoritmo donde solo cambie la estrategia de evaluación. Con una representación matricial, el estado global del enjambre es homogéneo y cualquier evaluador puede recibir el bloque completo de posiciones sin tocar el motor principal. Además, facilita la integración con persistencia, visualización y benchmarking.

Esto no implica que la implementación ya sea la versión vectorizada V4: una cosa es almacenar el estado en arrays y otra distinta es que toda la evaluación y actualización se haga sin bucles Python por partícula.

**2. Evaluadores intercambiables (Strategy Pattern).**

El módulo `parallel/` implementa un patrón factory (`choose_evaluator`) que devuelve el evaluador adecuado según el modo seleccionado (`sequential`, `threading`, `multiprocessing`, etc.). Todos los evaluadores comparten la misma interfaz:

```python
class Evaluator:
    def evaluate(self, positions: np.ndarray) -> np.ndarray: ...
    def shutdown(self) -> None: ...
```

De esta forma, el `PSO` engine no sabe ni necesita saber qué estrategia se está usando. Se puede añadir un nuevo evaluador sin modificar `pso_engine.py`.

**3. Fitness policies separadas de boundary strategies.**

Las estrategias de frontera (`clamp`, `reflect`) actúan sobre las posiciones tras cada actualización. La policy de fitness (`plain`, `penalty`) actúa sobre los valores evaluados. Son dos ejes ortogonales: se pueden combinar libremente (excepto `penalty` + `clamp`/`reflect`, que no tiene sentido porque la penalización ya absorbe las violaciones de frontera, y por tanto el motor lo detecta y desactiva la strategy automáticamente).

**4. Instancias de experimento como dataclass.**

Cada ejecución queda definida por una `Instance` (dataclass) que encapsula todos los parámetros. Esto permite generar combinaciones de forma programática con `make_instances()` y ejecutar suites completas con `run_suite()`, manteniendo la configuración siempre explícita y trazable.

**5. Persistencia timestamped.**

Cada ejecución crea su propio directorio con formato `{tipo}_{YYYYMMDD_HHMMSS}` dentro de `pso/results/`. Esto evita sobreescribir resultados anteriores y permite comparar ejecuciones a lo largo del tiempo. Se guardan los resultados en CSV (fácil de cargar con pandas y de inspeccionar manualmente) y la configuración en JSON (legible y serializable). Se eligió CSV sobre formatos binarios porque el volumen de datos es manejable y la inspección manual es importante durante el desarrollo.

---

## Instalación

### Requisitos previos

- Python 3.10 o superior (se usan `match/case` y union types con `|`)
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/imartinnez/Industrializable-PSO-algorithm
cd PSO_Algorithm

# 2. Recomendación crear un entorno virtual
python -m venv .venv

# En Windows:
.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate

# 3. Instalar dependencias
pip install numpy pandas matplotlib pyswarm
```

### Lista de dependencias

| Paquete      | Uso                                               |
| ------------ | ------------------------------------------------- |
| `numpy`      | Operaciones numéricas, representación del enjambre |
| `pandas`     | Gestión de resultados tabulares, exportación CSV   |
| `matplotlib` | Visualización, animaciones 2D/3D                   |
| `pyswarm`    | Implementación PSO externa usada como referencia   |

---

## Uso rápido

### Ejecución individual

Compara dos configuraciones del PSO propio (secuencial y multiprocessing) contra el baseline PySwarm sobre la función Sphere:

```bash
python -m pso.run_scripts.run
```

Salida de ejemplo:

```
sphere_d10
  best value   1.234567e-08
  iterations   347 / 1000
  total time   0.152 s
  eval time    0.098 s  (64.5%)
  best pos     [0.00012  -0.00003  0.00001]
```

### Suite de benchmarks

Ejecuta el PSO sobre las 4 funciones objetivo en dimensiones 2, 10 y 30, con 5 seeds y los modos secuencial, threading y PySwarm:

```bash
python -m pso.run_scripts.run_benchmarks
```

Genera automáticamente en `pso/results/benchmarks_{timestamp}/`:
- `benchmark_results.csv` — resultados completos de cada ejecución
- `benchmark_summary.csv` — media y desviación estándar agrupadas
- `config.json` — parámetros + info de hardware
- `run_benchmarks.log` — log de ejecución

### Grid search

Explora combinaciones de hiperparámetros (w, c1, c2, n_particles) buscando las mejores configuraciones:

```bash
python -m pso.run_scripts.run_grid_search
```

Genera en `pso/results/grid_search_{timestamp}/`:
- `grid_search_full.csv` — todas las combinaciones evaluadas
- `grid_search_summary.csv` — agregados por objetivo/dimensión/modo
- `grid_search_best.csv` — mejor configuración por cada grupo
- `config.json` — rejilla de parámetros usada

### Visualización

Genera animaciones GIF del enjambre moviéndose sobre las funciones objetivo en 2D y 3D:

```bash
python -m pso.viz.make_viz
```

Genera en `pso/results/viz_{timestamp}/`:
- `sphere_2d.gif`, `sphere_3d.gif`
- `rosenbrock_2d.gif`, `rosenbrock_3d.gif`
- `rastrigin_2d.gif`, `rastrigin_3d.gif`
- `ackley_2d.gif`, `ackley_3d.gif`

### Tests

```bash
python -m pytest pso/tests/tests_pso.py -v
```

---

## Comandos y scripts

| Comando | Descripción | Salida |
| ------- | ----------- | ------ |
| `python -m pso.run_scripts.run` | Ejecución individual comparativa | Consola + log |
| `python -m pso.run_scripts.run_benchmarks` | Suite de benchmarks (4 obj × 3 dims × 5 seeds × 3 modos) | CSV + JSON + log |
| `python -m pso.run_scripts.run_grid_search` | Grid search de hiperparámetros | CSV + JSON + log |
| `python -m pso.viz.make_viz` | Generación de animaciones 2D/3D | GIF |
| `python -m pytest pso/tests/ -v` | Tests unitarios | Consola |

---

## El algoritmo PSO

### Descripción general

El PSO canónico implementado minimiza funciones continuas en R^d. En cada iteración:

1. Se evalúa el fitness de todas las partículas (delegando al evaluador seleccionado).
2. Se aplica la fitness policy (plain o penalty).
3. Se actualizan los personal bests de cada partícula.
4. Se actualiza el global best del enjambre.
5. Se actualizan las velocidades con la regla estándar del PSO:

```
v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)
```

6. Se actualizan las posiciones: `x_i = x_i + v_i`
7. Se aplica la estrategia de frontera (clamp, reflect o ninguna).
8. Se comprueban los criterios de parada.

### Parámetros configurables

| Parámetro | Descripción | Valor típico |
| --------- | ----------- | ------------ |
| `n_particles` | Tamaño del enjambre | 30–50 |
| `dim` | Dimensión del espacio de búsqueda | 2, 10, 30 |
| `w` | Coeficiente de inercia | 0.5–0.9 |
| `c1` | Coeficiente cognitivo (atracción al pbest) | 1.0–2.0 |
| `c2` | Coeficiente social (atracción al gbest) | 1.0–2.0 |
| `max_iter` | Máximo de iteraciones | 500–2000 |
| `patience` | Iteraciones sin mejora antes de parar | 50–200 |
| `imp_min` | Mejora mínima para resetear el contador de paciencia | 1e-8 |
| `tol` | Tolerancia respecto al óptimo conocido para parada temprana | 1e-12 |
| `strategy` | Estrategia de frontera: `clamp`, `reflect`, `None` | `clamp` |
| `fitness_policy` | Política de fitness: `plain`, `penalty` | `plain` |
| `topology` | Topología del enjambre: `global` | `global` |

### Estrategias de frontera (boundary handling)

- **Clamp**: recorta cada coordenada al rango `[low, high]` con `np.clip`. Es la opción por defecto; simple y efectiva.
- **Reflect**: cuando una partícula cruza un límite, se refleja simétricamente y se invierte su velocidad en esa dimensión. Después se aplica un `clip` de seguridad.
- **None**: no se aplica ninguna corrección. Útil cuando se usa la penalty policy, que ya penaliza las violaciones de frontera en el fitness.

La penalty policy aplica un término `lambda * sum(violation^2)` al fitness de cada partícula, donde `violation` es cuánto excede los bounds. Si se selecciona `penalty` junto con `clamp` o `reflect`, el motor lo detecta y desactiva la strategy automáticamente (ya que no tiene sentido corregir posiciones y además penalizarlas).

### Criterios de parada

El bucle se detiene cuando se cumple cualquiera de estas condiciones:
1. Se alcanza `max_iter`.
2. El global best no mejora más de `imp_min` durante `patience` iteraciones consecutivas (stagnation).
3. El global best está a menos de `tol` del óptimo conocido (si se proporciona).

---

## Funciones benchmark

Se incluyen 4 funciones estándar de optimización, registradas en `objectives/registry.py` con sus bounds, valor óptimo y punto óptimo para cada dimensión:

| Función | Fórmula | Bounds | Óptimo | Punto óptimo | Tipo |
| ------- | ------- | ------ | ------ | ------------- | ---- |
| **Sphere** | f(x) = sum(x_i^2) | [-5.12, 5.12] | 0.0 | (0, ..., 0) | Unimodal, convexa |
| **Rosenbrock** | f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2) | [-5.0, 10.0] | 0.0 | (1, ..., 1) | Valle estrecho |
| **Rastrigin** | f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i)) | [-5.12, 5.12] | 0.0 | (0, ..., 0) | Altamente multimodal |
| **Ackley** | f(x) = -20*exp(...) - exp(...) + 20 + e | [-32.768, 32.768] | 0.0 | (0, ..., 0) | Multimodal con meseta |

Todas las funciones aceptan un `np.ndarray` como entrada y devuelven un `float`. Se evalúan en dimensiones d=2, d=10 y d=30 en los benchmarks.

Para añadir una nueva función basta con definirla en `functions.py` y registrarla en el diccionario `OBJECTIVES` de `registry.py`.

---

## Estrategias de paralelismo

El objetivo principal del proyecto es mantener el mismo core del PSO y cambiar únicamente la estrategia de evaluación del fitness. Todos los evaluadores implementan la misma interfaz (`evaluate` + `shutdown`) y se seleccionan a través del factory `choose_evaluator()`.

### V0 — Secuencial (baseline)

```python
# V0_sequential.py
return np.array([self.fitness_f(position) for position in positions])
```

Evaluación partícula a partícula en un bucle Python. Sirve como referencia para calcular speedups. Sin overhead de creación de pools ni serialización.

### V1 — Threading (concurrencia con hilos)

```python
# V1_threading.py
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    values = list(executor.map(self.fitness_f, positions))
```

Usa `concurrent.futures.ThreadPoolExecutor` para distribuir la evaluación entre hilos. El número de workers se puede configurar (por defecto lo elige Python).

**Consideraciones sobre el GIL**: En CPython, el GIL impide que dos hilos ejecuten bytecode Python puro simultáneamente. Sin embargo, las funciones benchmark usan operaciones NumPy internamente, que liberan el GIL durante los cálculos numéricos. Esto significa que la mejora real depende de cuánto tiempo pasa la función dentro de código C de NumPy vs bytecode Python. Para funciones cortas el overhead del pool puede superar la ganancia; para funciones más costosas o con componente I/O, threading puede aportar mejora.

### V2 — Multiprocessing (paralelismo con procesos)

```python
# V2_multiprocessing.py
self._executor = ProcessPoolExecutor(max_workers=max_workers)
values = list(self._executor.map(self.fitness_f, positions, chunksize=self.chunksize))
```

Usa `concurrent.futures.ProcessPoolExecutor` para ejecutar evaluaciones en procesos independientes, cada uno con su propio intérprete Python y sin el cuello de botella del GIL.

**Pool persistente**: el pool de procesos se crea una sola vez en `__init__` y se reutiliza en cada iteración, evitando el coste de crear y destruir procesos en cada llamada a `evaluate`. Se libera en `shutdown()`.

**Batching (chunksize)**: en vez de enviar una partícula por tarea, se envían bloques de `chunksize` partículas por worker. Esto reduce el overhead de comunicación inter-proceso (IPC/pickling), que es significativo cuando el coste de evaluación de una partícula individual es bajo.

**Serialización**: las posiciones (arrays NumPy) y la función fitness deben ser serializables con `pickle` para transferirse entre procesos. Las funciones definidas a nivel de módulo (como las de `functions.py`) son serializables sin problema; las lambdas o closures no lo son.

### V3 — Asyncio (en desarrollo)

Evaluación asíncrona con `asyncio`. La idea es diseñar un caso donde el fitness tenga latencia variable (simulando un servicio externo o I/O asimétrico) para que `asyncio.gather` aporte una mejora real frente a la evaluación secuencial.

### V4 — Vectorizada (en desarrollo)

Evaluación completamente vectorizada con NumPy, donde tanto la evaluación de fitness como la actualización de velocidades y posiciones se realizan sin bucles Python por partícula. Se aprovecha el "paralelismo implícito" de las operaciones BLAS/LAPACK que NumPy delega al hardware.

### Comparativa de estrategias

| Versión | Tecnología | Bypasses GIL | Overhead principal | Mejor para |
| ------- | ---------- | ------------ | ------------------ | ---------- |
| V0 | Bucle Python | — | Ninguno | Baseline, funciones baratas |
| V1 | ThreadPoolExecutor | Parcial (NumPy sí) | Creación de pool | I/O-bound, funciones con NumPy |
| V2 | ProcessPoolExecutor | Sí (procesos separados) | IPC + pickling | CPU-bound, funciones costosas |
| V3 | asyncio | N/A | Event loop | I/O asíncrono, latencias variables |
| V4 | NumPy vectorizado | Sí (C extensions) | Ninguno significativo | Funciones vectorizables |

---

## Grid search de hiperparámetros

El módulo `run_grid_search.py` implementa una búsqueda exhaustiva sobre combinaciones de hiperparámetros del PSO utilizando `itertools.product`.

### Parámetros de la rejilla

La configuración por defecto explora:

| Parámetro | Valores |
| --------- | ------- |
| `n_particles` | [30, 50] |
| `w` | [0.5, 0.7] |
| `c1` | [1.0, 1.5] |
| `c2` | [1.0, 1.5] |
| `max_iter` | [500] |

Se cruza con objetivos, dimensiones, seeds y modos de evaluación. Cada combinación se ejecuta con control de seed para reproducibilidad.

### Métricas calculadas

| Métrica | Descripción |
| ------- | ----------- |
| `best_value` | Mejor valor de fitness al final de la ejecución |
| `gap_to_optimum` | Distancia absoluta al óptimo conocido |
| `auc_fitness` | Media de la curva de convergencia (área bajo la curva simplificada) |
| `auc_gap` | Media del gap al óptimo a lo largo de las iteraciones |
| `total_time` | Tiempo total de ejecución |
| `fitness_eval_time_total` | Tiempo acumulado en evaluación de fitness |
| `iterations` | Iteraciones ejecutadas (puede ser menor que `max_iter` por early stopping) |

### Salida

Los resultados se agregan por `(objective, dim, mode, n_particles, w, c1, c2, max_iter)` promediando sobre las seeds, y se selecciona la mejor configuración por cada grupo `(objective, dim, mode)`.

---

## Persistencia y resultados

### Formato y estructura

Cada ejecución crea un directorio con timestamp dentro de `pso/results/`:

```
pso/results/
├── benchmarks_20260407_132716/
│   ├── benchmark_results.csv      # Resultados completos (una fila por ejecución)
│   ├── benchmark_summary.csv      # Agregados por (objetivo, dim, modo)
│   ├── config.json                # Parámetros + info de hardware
│   └── run_benchmarks.log         # Log de ejecución
├── grid_search_20260407_144218/
│   ├── grid_search_full.csv
│   ├── grid_search_summary.csv
│   ├── grid_search_best.csv
│   ├── config.json
│   └── run_grid_search.log
└── viz_20260407_183126/
    ├── sphere_2d.gif
    ├── sphere_3d.gif
    └── ...
```

### Elección de formatos

- **CSV** para resultados tabulares: es el formato más cómodo para cargar con `pandas`, inspeccionar manualmente en cualquier editor, y compartir. El volumen de datos no justifica formatos binarios como Parquet o HDF5.
- **JSON** para configuración: legible, serializable, y soporta estructuras anidadas (como la info de hardware). Se consideró YAML pero JSON tiene soporte nativo en Python sin dependencias extra.
- **GIF** para animaciones: formato universal que se puede abrir en cualquier navegador o visor de imágenes. Se soporta también MP4 si se tiene ffmpeg instalado.

### Contenido del config.json

```json
{
  "objectives": ["sphere", "rosenbrock", "rastrigin", "ackley"],
  "dims": [2, 10, 30],
  "seeds": [1, 2, 3, 4, 5],
  "modes": ["sequential", "threading", "pyswarm"],
  "max_iter": 2000,
  "n_particles": 50,
  "strategy": "clamp",
  "fitness_policy": "plain",
  "topology": "global",
  "w": 0.7, "c1": 1.5, "c2": 1.5,
  "patience": 100,
  "imp_min": 1e-08,
  "tol": 1e-12,
  "hardware": {
    "platform": "Windows",
    "platform_version": "10.0.26100",
    "processor": "...",
    "cpu_count_logical": 8,
    "python_version": "3.12.x"
  }
}
```

### Trayectorias

Las trayectorias de las partículas (posiciones en cada iteración) se almacenan solo para dimensiones d <= 3, ya que para dimensiones mayores el volumen crece rápidamente y la visualización no es factible. Para d=2 y d=3 se guardan completas porque se necesitan para generar las animaciones.

---

## Visualización

El módulo `viz/` genera animaciones que muestran la evolución del enjambre durante la optimización.

### Animación 2D (`animate_2d`)

Cada frame muestra dos paneles:
- **Panel izquierdo**: mapa de contorno de la función objetivo (escala logarítmica) con las posiciones de las partículas (puntos cyan) y el global best (estrella roja).
- **Panel derecho**: curva de convergencia del mejor fitness a lo largo de los frames.

### Animación 3D (`animate_3d`)

Misma estructura de dos paneles, pero el izquierdo muestra la superficie 3D de la función con las partículas flotando sobre ella. La cámara rota suavemente para dar perspectiva.

### Detalles técnicos

- Resolución del grid de fondo: 250×250 puntos (2D), 60×60 (3D).
- En 3D los valores de fitness de las partículas se precalculan antes de la animación para evitar recalcularlos frame a frame.
- FPS por defecto: 15 (2D), 12 (3D).
- Blitting activado solo en 2D (en 3D no está soportado por matplotlib).
- Cada animación incluye un cuadro de texto con el frame actual y el mejor fitness.

---

## Tests

Los tests se encuentran en `pso/tests/tests_pso.py` y se ejecutan con pytest. Cubren los aspectos principales exigidos:

### Reproducibilidad

| Test | Qué verifica |
| ---- | ------------ |
| `test_same_seed_same_result` | Misma seed produce mismo `b_value` y `b_position` |
| `test_different_seeds_different_results` | Seeds distintas producen posiciones distintas |

### Manejo de fronteras

| Test | Qué verifica |
| ---- | ------------ |
| `test_best_position_within_bounds` | La mejor posición está dentro de los bounds |
| `test_bounds_respected_across_dims` | Se respetan los bounds en dimensiones 2, 5 y 10 |

### Monotonía del global best

| Test | Qué verifica |
| ---- | ------------ |
| `test_global_best_never_worsens` | La curva de convergencia es no creciente (con tolerancia numérica 1e-12) |

### Correctitud en Sphere

| Test | Qué verifica |
| ---- | ------------ |
| `test_sphere_converges_to_zero` | Con 1000 iteraciones y 50 partículas en d=3, converge a < 1e-4 |
| `test_sphere_best_value_non_negative` | El fitness de Sphere nunca es negativo |

### Consistencia del resultado

| Test | Qué verifica |
| ---- | ------------ |
| `test_iterations_within_limit` | El número de iteraciones está entre 1 y `max_iter` |
| `test_result_fields_consistent` | La longitud de `best_fitness_by_iter` coincide con las iteraciones; `eval_time <= total_time` |

### Ejecución

```bash
# Todos los tests
python -m pytest pso/tests/tests_pso.py -v

# Un test específico
python -m pytest pso/tests/tests_pso.py::test_sphere_converges_to_zero -v
```

---

## Reproducibilidad

Todas las ejecuciones del PSO aceptan una `seed` que controla la generación de números aleatorios:

1. Se fija `np.random.seed(seed)` al inicio de `pso.run()`.
2. Se genera un array de seeds derivadas (una por iteración) usando `np.random.randint`.
3. Antes de cada iteración se aplica la seed correspondiente con `np.random.seed(seeds[i])`.

Esto garantiza que con la misma seed y los mismos parámetros se obtiene exactamente el mismo resultado, independientemente de cuántas veces se ejecute.

En los benchmarks y grid search se utilizan siempre múltiples seeds (por defecto 5) para promediar resultados y evaluar la variabilidad. Las seeds usadas quedan registradas en el `config.json` de cada ejecución.

**Nota sobre multiprocessing:** la reproducibilidad está garantizada para V0 y V1. En V2, el orden de finalización de los procesos puede variar, pero como cada proceso evalúa una partícula de forma independiente y el resultado se recoge en orden (la API de `executor.map` preserva el orden), los resultados finales son deterministas para la misma seed.

---

## Dependencias

El proyecto usa exclusivamente la librería estándar de Python y las siguientes dependencias externas:

| Paquete | Versión mínima | Propósito |
| ------- | -------------- | --------- |
| `numpy` | >= 1.24 | Operaciones matriciales, generación aleatoria |
| `pandas` | >= 2.0 | DataFrames para resultados y exportación CSV |
| `matplotlib` | >= 3.7 | Visualización, animaciones FuncAnimation |
| `pyswarm` | >= 0.6 | Implementación PSO externa de referencia |

Python 3.10+ es necesario por el uso de `match/case` (PEP 634) y union types `X | Y` (PEP 604).
