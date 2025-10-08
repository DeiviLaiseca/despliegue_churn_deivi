# ============================================================
#        TELCO CUSTOMER CHURN PREDICTOR - STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("model_pipeline.pkl")
    return model

model_pipeline = load_model()

st.set_page_config(page_title="Churn Prediction App", page_icon="📊", layout="centered")

st.title("📊 Telco Customer Churn Prediction")
st.write("Esta aplicación predice si un cliente **abandonará el servicio (Churn)** usando un modelo de Machine Learning entrenado con datos reales de telecomunicaciones.")

st.sidebar.header("📋 Ingrese la información del cliente:")

gender = st.sidebar.selectbox("Género", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox("¿Es ciudadano senior?", [0, 1])
Partner = st.sidebar.selectbox("¿Tiene pareja?", ["Yes", "No"])
Dependents = st.sidebar.selectbox("¿Tiene dependientes?", ["Yes", "No"])
tenure = st.sidebar.number_input("Meses de antigüedad (tenure)", min_value=0, max_value=100, value=12)
PhoneService = st.sidebar.selectbox("¿Tiene servicio telefónico?", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("¿Líneas múltiples?", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Tipo de Internet", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Seguridad en línea", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Backup en línea", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Protección de dispositivo", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Soporte técnico", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("¿Streaming TV?", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("¿Streaming Movies?", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Facturación electrónica", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Método de pago", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("Cargo mensual", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.sidebar.number_input("Cargos totales", min_value=0.0, max_value=10000.0, value=2500.0)

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

st.write("### 🧾 Datos ingresados:")
st.dataframe(input_data, use_container_width=True)

if st.button("🚀 Predecir Churn"):
    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0][1]

    st.subheader("📈 Resultado de la predicción:")
    if prediction == 1:
        st.error(f"⚠️ El cliente probablemente **ABANDONARÁ** el servicio.\n\nProbabilidad de churn: **{probability:.2%}**")
    else:
        st.success(f"✅ El cliente **PERMANECERÁ** con la empresa.\n\nProbabilidad de churn: **{probability:.2%}**")

    st.progress(int(probability * 100))

st.markdown("---")
st.caption("Desarrollado por Deivi Laiseca · Proyecto de Análisis Predictivo de Churn · 2025")
