# 📊 Telco Customer Churn Prediction (2025)

Aplicación de **Machine Learning** que predice si un cliente de telecomunicaciones **abandonará el servicio (Churn)** usando datos históricos reales y modelos entrenados con técnicas de aprendizaje supervisado.

El proyecto integra todo el flujo de ciencia de datos: limpieza, preprocesamiento, ajuste de hiperparámetros, evaluación y despliegue en **Streamlit Cloud**.

---

## 🧠 Objetivo del proyecto

Desarrollar un modelo predictivo y una aplicación web interactiva para anticipar el abandono de clientes (*churn*) en una empresa de telecomunicaciones, aplicando un proceso de **Machine Learning de extremo a extremo**:

1. Limpieza y preparación de los datos  
2. Codificación, imputación y escalamiento mediante un **Pipeline de Scikit-learn**  
3. Entrenamiento y comparación de modelos (**Logistic Regression** y **XGBoost**)  
4. Optimización de hiperparámetros con **RandomizedSearchCV**  
5. Despliegue de una aplicación web en **Streamlit Cloud**

---

## 🧩 Tecnologías utilizadas

| Componente | Librería / Versión |
|-------------|--------------------|
| Lenguaje base | **Python 3.13** |
| Manipulación de datos | `pandas 2.2.3`, `numpy 2.1.2` |
| Modelado y ML | `scikit-learn 1.5.2`, `xgboost 2.1.1` |
| Persistencia de modelos | `joblib 1.4.2` |
| Visualización | `matplotlib 3.9.2`, `seaborn 0.13.2` |
| Aplicación web | `streamlit 1.38.0` |

---

## ⚙️ Instalación y ejecución local

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/<TU_USUARIO>/telco-churn-app.git
   cd telco-churn-app
