# diabetes-rna-streamlit
# Predictor de Diabetes: Red Neuronal Artificial

Este proyecto implementa una aplicación web desarrollada con **Python, Keras y Streamlit** para predecir de manera orientativa el riesgo de diabetes en pacientes, utilizando un modelo de Red Neuronal Artificial entrenado con el dataset `diabetes_data_es_GOLD.csv`.

## Descripción del proyecto

El objetivo del proyecto es construir una Red Neuronal Artificial feedforward para clasificar si un paciente presenta riesgo positivo o negativo de diabetes a partir de 16 variables clínicas. Además, el modelo de red neuronal se compara con un modelo de **Gradient Boosted Trees**, utilizando métricas como Accuracy, F1-score y AUC.

## Tecnologías utilizadas

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Joblib

## Archivos principales

* `app.py`: aplicación web desarrollada en Streamlit.
* `modelo_rna.keras`: modelo de Red Neuronal Artificial entrenado.
* `scaler.pkl`: objeto de escalamiento usado para normalizar los datos de entrada.
* `requirements.txt`: librerías necesarias para ejecutar la aplicación.
* `01_entrenamiento.ipynb`: notebook utilizado para el entrenamiento y evaluación del modelo.

## Variables de entrada

La aplicación utiliza las siguientes variables:

* Edad
* Género
* Poliuria
* Polidipsia
* Pérdida repentina de peso
* Debilidad
* Polifagia
* Candidiasis genital
* Visión borrosa
* Picazón
* Irritabilidad
* Cicatrización lenta
* Paresia parcial
* Rigidez muscular
* Alopecia
* Obesidad

## Ejecución local

Para ejecutar la aplicación localmente, instalar las dependencias:

```bash
pip install -r requirements.txt
```

Luego ejecutar:

```bash
streamlit run app.py
```

## Resultado

La aplicación muestra una predicción orientativa del riesgo de diabetes:

* Riesgo alto si la probabilidad es mayor o igual a 0.5.
* Riesgo bajo si la probabilidad es menor a 0.5.

## Advertencia

El resultado generado por esta aplicación es únicamente orientativo y no reemplaza la evaluación de un profesional médico.
