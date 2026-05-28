# @author: Íñigo Martínez Jiménez
# Interactive dashboard for the cooling optimization use case of
# Data centers with PSO. Reads the pso/results/datacenter_* folder and produces a
# executive view designed to present to the client.


from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pso.use_case.scenario import create_default_datacenter_scenario
from pso.use_case.encoding import phys_to_norm
from pso.use_case.objective import evaluate_solution


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Page setup

st.set_page_config(
    page_title="Data Center Cooling | PSO",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    /* Hide default Streamlit chrome for a cleaner client-facing look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Hero */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0B5394;
        margin: 0.4rem 0 0.1rem 0;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #6C757D;
        margin-bottom: 1.2rem;
        max-width: 880px;
    }
    .hero-divider {
        height: 3px;
        background: linear-gradient(90deg, #0B5394 0%, #0F9D58 100%);
        border: 0;
        margin: 0 0 1.6rem 0;
    }

    /* Section headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #212529;
        margin-top: 1.6rem;
        margin-bottom: 0.2rem;
        border-left: 4px solid #0B5394;
        padding-left: 12px;
    }
    .section-help {
        font-size: 0.92rem;
        color: #6C757D;
        margin-bottom: 1rem;
        padding-left: 16px;
        max-width: 880px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #6C757D;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        color: #0B5394;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.82rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F8F9FA;
        border-right: 1px solid #E5E7EB;
    }
    section[data-testid="stSidebar"] h3 {
        color: #0B5394;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Data loading

RESULTS_ROOT = PROJECT_ROOT / "pso" / "results"


@st.cache_data(show_spinner=False)
def list_datacenter_runs() -> list[Path]:
    """Return all datacenter_* runs that have both config.json and convergence.csv."""
    if not RESULTS_ROOT.exists():
        return []
    runs = sorted(
        [
            p for p in RESULTS_ROOT.glob("datacenter_*")
            if (p / "config.json").exists() and (p / "convergence.csv").exists()
        ],
        reverse=True,
    )
    return runs


@st.cache_data(show_spinner=False)
def load_run(run_dir_str: str):
    """Load config, convergence and reevaluate baseline and PSO solutions."""
    run_dir = Path(run_dir_str)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    convergence = pd.read_csv(run_dir / "convergence.csv")

    scenario = create_default_datacenter_scenario(seed=config["seed"])

    x_baseline_norm = phys_to_norm(scenario.baseline_x_phys, scenario)
    baseline = evaluate_solution(x_baseline_norm, scenario)

    pso_sol = config["pso_solution"]
    x_pso_phys = np.concatenate(
        [[pso_sol["t_set"]], pso_sol["fan_speeds"], pso_sol["zone_airflows"]]
    )
    x_pso_norm = phys_to_norm(x_pso_phys, scenario)
    pso = evaluate_solution(x_pso_norm, scenario)

    return config, convergence, scenario, baseline, pso


# Header

st.markdown('<div class="hero-title">Data Center Cooling Optimization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Particle Swarm Optimization aplicado al ajuste de '
    'refrigeración del data center. Reducimos el consumo energético manteniendo las '
    'temperaturas dentro de los límites de seguridad operativa.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="hero-divider"/>', unsafe_allow_html=True)


# Run selector

runs = list_datacenter_runs()
if not runs:
    st.error(
        "No se han encontrado ejecuciones en `pso/results/datacenter_*`. "
        "Lanza primero `python -m pso.run_scripts.run_datacenter_case`."
    )
    st.stop()

with st.sidebar:
    st.markdown("### Ejecución")
    selected_name = st.selectbox(
        "Selecciona una corrida",
        options=[r.name for r in runs],
        index=0,
        label_visibility="collapsed",
    )
    selected_run = next(r for r in runs if r.name == selected_name)

    config, convergence, scenario, baseline, pso = load_run(str(selected_run))

    st.markdown("---")
    st.markdown("### Escenario")
    st.markdown(
        f"- **Racks:** {scenario.n_racks} en rejilla {scenario.grid_shape[0]}×{scenario.grid_shape[1]}\n"
        f"- **Ventiladores:** {scenario.n_fans}\n"
        f"- **Zonas de caudal:** {scenario.n_zones}\n"
        f"- **Seed:** {config['seed']}"
    )

    st.markdown("---")
    st.markdown("### Configuración del PSO")
    st.markdown(
        f"- **Partículas:** {config['pso_config']['n_particles']}\n"
        f"- **Iteraciones máx:** {config['pso_config']['max_iter']}\n"
        f"- **w:** {config['pso_config']['w']}\n"
        f"- **c1 / c2:** {config['pso_config']['c1']} / {config['pso_config']['c2']}"
    )

    st.markdown("---")
    st.caption(
        f"PSO ejecutado en **{config['pso_runtime_seconds']:.3f} s** "
        f"a lo largo de **{config['pso_iterations']} iteraciones**."
    )


# KPI row

energy_savings = (
    (baseline["total_energy"] - pso["total_energy"]) / baseline["total_energy"] * 100
)
fitness_improvement = (
    (baseline["fitness"] - pso["fitness"]) / baseline["fitness"] * 100
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric(
        "Ahorro energético",
        f"{energy_savings:+.2f}%",
        delta=f"{baseline['total_energy'] - pso['total_energy']:.2f} unidades",
    )
with k2:
    st.metric(
        "Energía total PSO",
        f"{pso['total_energy']:.2f}",
        delta=f"baseline {baseline['total_energy']:.2f}",
        delta_color="off",
    )
with k3:
    st.metric(
        "Temperatura máxima",
        f"{pso['max_temperature']:.2f} °C",
        delta=f"{pso['max_temperature'] - baseline['max_temperature']:+.2f} °C",
        delta_color="inverse",
    )
with k4:
    delta_hot = pso["hotspots"] - baseline["hotspots"]
    st.metric(
        "Hotspots (>T_hot)",
        f"{pso['hotspots']}",
        delta=(f"{delta_hot:+d} vs baseline" if delta_hot != 0 else "sin cambios"),
        delta_color="inverse",
    )
with k5:
    st.metric(
        "Mejora de fitness",
        f"{fitness_improvement:+.2f}%",
        delta=f"baseline {baseline['fitness']:.3f}",
        delta_color="off",
    )


# Energy section

st.markdown('<div class="section-title">Comparativa energética</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-help">El PSO redistribuye el presupuesto energético entre '
    'chiller, ventiladores y caudales. La ley cúbica de los ventiladores es la mayor '
    'fuente de ahorro: pequeñas reducciones de velocidad se traducen en grandes '
    'reducciones de consumo.</div>',
    unsafe_allow_html=True,
)

col_total, col_break = st.columns([1, 2])

with col_total:
    fig = go.Figure()
    fig.add_bar(
        x=["Baseline", "PSO"],
        y=[baseline["total_energy"], pso["total_energy"]],
        marker_color=["#9CA3AF", "#0F9D58"],
        text=[f"{baseline['total_energy']:.2f}", f"{pso['total_energy']:.2f}"],
        textposition="outside",
    )
    fig.update_layout(
        title="Energía total",
        yaxis_title="energía (u.a.)",
        showlegend=False,
        height=380,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig, use_container_width=True)

with col_break:
    components = ["chiller_energy", "fan_energy", "airflow_energy"]
    labels = ["Chiller", "Ventiladores", "Caudal por zona"]
    fig = go.Figure()
    fig.add_bar(
        name="Baseline",
        x=labels,
        y=[baseline[c] for c in components],
        marker_color="#9CA3AF",
    )
    fig.add_bar(
        name="PSO",
        x=labels,
        y=[pso[c] for c in components],
        marker_color="#0F9D58",
    )
    fig.update_layout(
        title="Desglose por componente",
        barmode="group",
        yaxis_title="energía (u.a.)",
        height=380,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig, use_container_width=True)


# Thermal section

st.markdown('<div class="section-title">Distribución térmica de los racks</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-help">Temperatura simulada en cada rack sobre la rejilla '
    f'{scenario.grid_shape[0]}×{scenario.grid_shape[1]}. La escala de color es común '
    f'a ambos mapas para facilitar la comparación visual. T_safe = {scenario.t_safe} °C, '
    f'T_hot = {scenario.t_hot} °C.</div>',
    unsafe_allow_html=True,
)


def heatmap_grid(temperatures: np.ndarray, scenario) -> np.ndarray:
    """Reshape per-rack temperatures into a 2D grid using the rack positions."""
    rows, cols = scenario.grid_shape
    grid = np.zeros((rows, cols))
    for r, (i, j) in enumerate(scenario.rack_positions.astype(int)):
        grid[i, j] = temperatures[r]
    return grid


vmin = float(min(baseline["temperatures"].min(), pso["temperatures"].min()))
vmax = float(max(baseline["temperatures"].max(), pso["temperatures"].max()))

h1, h2 = st.columns(2)
for col, (name, sol) in zip(
    [h1, h2], [("Baseline", baseline), ("PSO", pso)]
):
    grid = heatmap_grid(sol["temperatures"], scenario)
    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale="Inferno",
            zmin=vmin,
            zmax=vmax,
            text=np.round(grid, 1),
            texttemplate="%{text}",
            textfont={"color": "white", "size": 11},
            colorbar=dict(title="T (°C)"),
        )
    )
    fig.update_layout(
        title=(
            f"{name} — máx {sol['max_temperature']:.2f} °C · "
            f"media {sol['mean_temperature']:.2f} °C · "
            f"hotspots {sol['hotspots']}"
        ),
        xaxis_title="columna",
        yaxis_title="fila",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(autorange="reversed")
    col.plotly_chart(fig, use_container_width=True)


# Convergence

st.markdown('<div class="section-title">Convergencia del optimizador</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-help">Mejor fitness encontrado por el enjambre en cada '
    'iteración. La línea discontinua marca el fitness de la configuración baseline '
    'como referencia.</div>',
    unsafe_allow_html=True,
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=convergence["iteration"],
        y=convergence["best_fitness"],
        mode="lines",
        line=dict(color="#0B5394", width=2.5),
        name="PSO",
        fill="tozeroy",
        fillcolor="rgba(11, 83, 148, 0.06)",
    )
)
fig.add_hline(
    y=baseline["fitness"],
    line_dash="dash",
    line_color="#9CA3AF",
    annotation_text=f"baseline = {baseline['fitness']:.3f}",
    annotation_position="top right",
)
fig.update_layout(
    xaxis_title="iteración",
    yaxis_title="mejor fitness",
    height=380,
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=40),
    showlegend=False,
)
fig.update_yaxes(gridcolor="#E5E7EB")
fig.update_xaxes(gridcolor="#E5E7EB")
st.plotly_chart(fig, use_container_width=True)


# Optimized variables

st.markdown('<div class="section-title">Variables de control optimizadas</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-help">Las 9 variables que el PSO ajusta: temperatura de '
    'consigna del chiller, velocidad de cada ventilador y caudal por zona.</div>',
    unsafe_allow_html=True,
)

baseline_fans = scenario.baseline_x_phys[1:1 + scenario.n_fans]
baseline_zones = scenario.baseline_x_phys[1 + scenario.n_fans:]

v1, v2, v3 = st.columns(3)

with v1:
    fig = go.Figure()
    fig.add_bar(
        x=["Baseline", "PSO"],
        y=[scenario.baseline_x_phys[0], pso["t_set"]],
        marker_color=["#9CA3AF", "#0F9D58"],
        text=[f"{scenario.baseline_x_phys[0]:.2f}", f"{pso['t_set']:.2f}"],
        textposition="outside",
    )
    fig.add_hline(
        y=scenario.t_safe,
        line_dash="dot",
        line_color="#E0A458",
        annotation_text=f"T_safe = {scenario.t_safe}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Temperatura de consigna (°C)",
        height=340,
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(gridcolor="#E5E7EB")
    st.plotly_chart(fig, use_container_width=True)

with v2:
    fig = go.Figure()
    fans_labels = [f"Fan {i}" for i in range(scenario.n_fans)]
    fig.add_bar(name="Baseline", x=fans_labels, y=baseline_fans, marker_color="#9CA3AF")
    fig.add_bar(name="PSO", x=fans_labels, y=pso["fan_speeds"], marker_color="#0F9D58")
    fig.update_layout(
        title="Velocidad de ventiladores",
        barmode="group",
        height=340,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor="#E5E7EB", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with v3:
    fig = go.Figure()
    zones_labels = [f"Zona {i}" for i in range(scenario.n_zones)]
    fig.add_bar(name="Baseline", x=zones_labels, y=baseline_zones, marker_color="#9CA3AF")
    fig.add_bar(name="PSO", x=zones_labels, y=pso["zone_airflows"], marker_color="#0F9D58")
    fig.update_layout(
        title="Caudal por zona",
        barmode="group",
        height=340,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(gridcolor="#E5E7EB", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


# Executive summary table

st.markdown('<div class="section-title">Resumen ejecutivo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-help">Comparativa completa de todas las métricas relevantes '
    'entre la operativa baseline y la solución encontrada por el PSO.</div>',
    unsafe_allow_html=True,
)

metric_rows = [
    ("Fitness", "fitness", ""),
    ("Energía total", "total_energy", " u.a."),
    ("Energía chiller", "chiller_energy", " u.a."),
    ("Energía ventiladores", "fan_energy", " u.a."),
    ("Energía caudal", "airflow_energy", " u.a."),
    ("Temperatura media", "mean_temperature", " °C"),
    ("Temperatura máxima", "max_temperature", " °C"),
    ("Temperatura mínima", "min_temperature", " °C"),
    ("Desviación térmica", "std_temperature", " °C"),
    ("Racks por encima de T_safe", "unsafe_racks", ""),
    ("Hotspots por encima de T_hot", "hotspots", ""),
    ("Racks sobreenfriados", "overcooled_racks", ""),
]

summary_rows = []
for label, key, unit in metric_rows:
    b_val = baseline[key]
    p_val = pso[key]
    if isinstance(b_val, (int, np.integer)) and not isinstance(b_val, bool):
        b_str = f"{int(b_val)}{unit}"
        p_str = f"{int(p_val)}{unit}"
        diff = int(p_val) - int(b_val)
        diff_str = f"{diff:+d}" if diff != 0 else "—"
    else:
        b_str = f"{b_val:.3f}{unit}"
        p_str = f"{p_val:.3f}{unit}"
        diff = p_val - b_val
        diff_str = f"{diff:+.3f}{unit}"
    summary_rows.append(
        {"Métrica": label, "Baseline": b_str, "PSO": p_str, "Diferencia": diff_str}
    )

df_summary = pd.DataFrame(summary_rows)
st.dataframe(df_summary, hide_index=True, use_container_width=True)


# Technical details

with st.expander("Configuración técnica completa de la ejecución"):
    st.json(config)


# Footer

st.markdown("---")
st.caption(
    "Dashboard generado a partir de la última ejecución de "
    "`python -m pso.run_scripts.run_datacenter_case`. El modelo térmico es una "
    "aproximación estática pensada como ejercicio académico; no sustituye un "
    "análisis CFD ni un dimensionamiento real de instalación."
)
