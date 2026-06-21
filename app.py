import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

st.set_page_config(
    page_title="Predictor Diabetes RNA",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    modelo = keras.models.load_model("modelo_rna.keras")
    scaler = joblib.load("scaler.pkl")
    return modelo, scaler

modelo, scaler = cargar_modelo()

st.title("🩺 Predictor de Diabetes — Red Neuronal Artificial")
st.write("Aplicación desarrollada con Python, Keras y Streamlit.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    edad = st.number_input("Edad", min_value=16, max_value=90, value=45)
    genero = st.selectbox("Género", ["Masculino (1)", "Femenino (2)"])
    poliuria = st.checkbox("Poliuria")
    polidipsia = st.checkbox("Polidipsia")
    perdida = st.checkbox("Pérdida repentina de peso")
    debilidad = st.checkbox("Debilidad")

with col2:
    polifagia = st.checkbox("Polifagia")
    candidiasis = st.checkbox("Candidiasis genital")
    vision = st.checkbox("Visión borrosa")
    picazon = st.checkbox("Picazón")
    irritabilidad = st.checkbox("Irritabilidad")
    cicatrizacion = st.checkbox("Cicatrización lenta")

with col3:
    paresia = st.checkbox("Paresia parcial")
    rigidez = st.checkbox("Rigidez muscular")
    alopecia = st.checkbox("Alopecia")
    obesidad = st.checkbox("Obesidad")

st.divider()

if st.button("🔍 Predecir Riesgo", use_container_width=True):

    genero_valor = 1 if "Masculino" in genero else 2

    datos = np.array([[
        edad,
        genero_valor,
        int(poliuria),
        int(polidipsia),
        int(perdida),
        int(debilidad),
        int(polifagia),
        int(candidiasis),
        int(vision),
        int(picazon),
        int(irritabilidad),
        int(cicatrizacion),
        int(paresia),
        int(rigidez),
        int(alopecia),
        int(obesidad)
    ]])

    datos_escalados = scaler.transform(datos)

    prob = float(modelo.predict(datos_escalados)[0][0])

    if prob >= 0.5:
        st.error(f"⚠️ RIESGO ALTO de Diabetes — {prob * 100:.1f}%")
    else:
        st.success(f"✅ RIESGO BAJO de Diabetes — {prob * 100:.1f}%")

    st.progress(prob)

    st.caption("⚠️ Resultado orientativo. Consulte siempre a un profesional médico.")
