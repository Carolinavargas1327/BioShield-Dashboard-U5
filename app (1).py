
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="BioShield Health & Security",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ BioShield Health & Security")
st.subheader("Dashboard analítico de ciberseguridad y medicina")
st.caption("Unidad 5 | Ingeniería en Ciencia de Datos")

# Cargar archivos
if not Path("ueba_telemetry.csv").exists() or not Path("biomed_metrics.csv").exists():
    st.error("No se encontraron los archivos CSV. Colócalos junto a app.py.")
    st.stop()

ueba = pd.read_csv("ueba_telemetry.csv")
biomed = pd.read_csv("biomed_metrics.csv")

# Clasificar niveles de riesgo
def nivel_riesgo(valor):
    if valor < 40:
        return "Verde"
    elif valor < 70:
        return "Amarillo"
    return "Rojo"

ueba["nivel_alerta"] = ueba["risk_score"].apply(nivel_riesgo)
ueba["hora_num"] = pd.to_numeric(
    ueba["hora_acceso"].str.split(":").str[0]
)

# Menú lateral
st.sidebar.title("Filtros del dashboard")

roles = ["Todos"] + sorted(ueba["rol"].unique().tolist())
rol_seleccionado = st.sidebar.selectbox(
    "Seleccionar rol", roles
)

hora_min, hora_max = st.sidebar.slider(
    "Ventana horaria",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

datos = ueba[
    (ueba["hora_num"] >= hora_min) &
    (ueba["hora_num"] <= hora_max)
].copy()

if rol_seleccionado != "Todos":
    datos = datos[datos["rol"] == rol_seleccionado]

# Indicadores generales
c1, c2, c3 = st.columns(3)

c1.metric("Usuarios analizados", len(datos))
c2.metric(
    "Alertas rojas",
    int((datos["nivel_alerta"] == "Rojo").sum())
)
c3.metric(
    "Riesgo promedio",
    f"{datos['risk_score'].mean():.2f}" if len(datos) else "Sin datos"
)

st.divider()

# MÓDULO 1: UEBA
st.header("1. Monitoreo de riesgo de identidad (UEBA)")

st.write(
    "Relación entre el volumen de descarga y el índice de riesgo "
    "de los usuarios."
)

fig1 = px.scatter(
    datos,
    x="volumen_descarga_mb",
    y="risk_score",
    color="nivel_alerta",
    color_discrete_map={
        "Verde": "green",
        "Amarillo": "orange",
        "Rojo": "red"
    },
    hover_data=[
        "user_id",
        "rol",
        "hora_acceso",
        "ubicacion_ip"
    ],
    labels={
        "volumen_descarga_mb": "Descarga (MB)",
        "risk_score": "Índice de riesgo",
        "nivel_alerta": "Nivel de alerta"
    },
    title="Volumen de descarga vs. riesgo"
)

st.plotly_chart(fig1, use_container_width=True)

st.divider()

# MÓDULO 2: BIOMÉDICO
st.header("2. Desempeño biomédico: U-Net vs. MedSAM")

promedios = biomed.groupby("tipo_organo")[
    ["dice_unet", "dice_medsam"]
].mean().reset_index()

promedios_largos = promedios.melt(
    id_vars="tipo_organo",
    var_name="Modelo",
    value_name="DSC promedio"
)

promedios_largos["Modelo"] = (
    promedios_largos["Modelo"].replace({
        "dice_unet": "U-Net",
        "dice_medsam": "MedSAM"
    })
)

fig2 = px.bar(
    promedios_largos,
    x="tipo_organo",
    y="DSC promedio",
    color="Modelo",
    barmode="group",
    title="Comparación del coeficiente Dice por órgano",
    labels={
        "tipo_organo": "Órgano",
        "DSC promedio": "Coeficiente Dice"
    }
)

st.plotly_chart(fig2, use_container_width=True)

# Distribución pLDDT
st.subheader("Confianza estructural de complejos proteína-ligando")

fig3 = px.histogram(
    biomed,
    x="plddt_score",
    nbins=10,
    title="Distribución de puntuaciones pLDDT",
    labels={"plddt_score": "Puntuación pLDDT"}
)

st.plotly_chart(fig3, use_container_width=True)

# Candidatos de alta confianza
alta_confianza = biomed[
    biomed["plddt_score"] > 90
]

st.write("Candidatos con alta confianza estructural (pLDDT > 90)")
st.dataframe(alta_confianza, use_container_width=True)

# Tiempo de inferencia
st.metric(
    "Tiempo promedio de inferencia",
    f"{biomed['tiempo_inferencia_seg'].mean():.2f} segundos"
)

st.divider()

# MÓDULO 3: ALERTAS Y XAI
st.header("3. Panel de alertas tempranas y explicabilidad (XAI)")

def explicar_alerta(fila):
    razones = []

    hora = int(fila["hora_num"])

    if hora < 7 or hora >= 22:
        razones.append("acceso fuera del horario habitual")

    if fila["volumen_descarga_mb"] >= 500:
        razones.append("volumen de descarga elevado")

    if fila["ubicacion_ip"] == "Externa":
        razones.append("conexión desde ubicación externa")

    if fila["isolation_forest_score"] < 0:
        razones.append("anomalía detectada por Isolation Forest")

    if fila["z_score_temporal"] >= 3:
        razones.append("desviación temporal elevada")

    if not razones:
        return "Sin anomalías destacadas en las variables evaluadas."

    return "Alerta por: " + "; ".join(razones) + "."

# Priorizar los cinco usuarios con mayor riesgo
top5 = ueba.sort_values(
    by=["risk_score", "z_score_temporal"],
    ascending=False
).head(5).copy()

top5["justificacion_XAI"] = top5.apply(
    explicar_alerta, axis=1
)

columnas = [
    "user_id",
    "rol",
    "hora_acceso",
    "volumen_descarga_mb",
    "risk_score",
    "z_score_temporal",
    "nivel_alerta",
    "justificacion_XAI"
]

st.subheader("Cinco usuarios con mayor prioridad de revisión")
st.dataframe(
    top5[columnas],
    use_container_width=True
)

st.info(
    "Las explicaciones XAI muestran las variables que contribuyeron "
    "a generar cada alerta. Una alerta requiere revisión humana y "
    "no demuestra por sí sola una amenaza."
)

st.caption(
    "Nota: los conjuntos de datos y las métricas de esta demostración "
    "son simulados con fines académicos. Los valores pLDDT no son "
    "resultados reales de AlphaFold 3."
)

st.success("Dashboard BioShield cargado correctamente.")
