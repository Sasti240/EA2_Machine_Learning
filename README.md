# Modelos de Predicción y Clasificación

## Integrantes

- Santiago Jimenez Otalvaro
- Santiago Ramírez Vásquez

## Descripción

Este proyecto aplica técnicas de Machine Learning a un conjunto de datos de viviendas de King County, Washington, Estados Unidos.

Se implementan dos modelos: uno de regresión lineal para predecir el valor de las viviendas y otro de clasificación logística para clasificarlas en dos categorías según su precio.

Además, el proyecto genera automáticamente un informe con la información del dataset, el preprocesamiento realizado, los parámetros de los modelos y sus métricas de evaluación.

## Dataset

- **Nombre:** House Sales (King County).
- **Fuente:** Hugging Face Datasets.
- **Repositorio:** [https://huggingface.co/datasets/inria-soda/tabular-benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark)
- **Archivo:** `reg_cat/house_sales.csv`.
- **Registros:** 21.613.
- **Columnas:** 18.
- **Variable objetivo:** `price`.

Los datos incluyen características de las viviendas, como habitaciones, baños, áreas, ubicación y precio. El dataset se descarga desde Hugging Face al ejecutar el programa.

## Herramientas

- Python
- Pandas
- NumPy
- Hugging Face Datasets
- Scikit-learn
- Matplotlib
- Git y GitHub
- GitHub Actions

## Instalación

1. Clonar el repositorio.
2. Crear un entorno virtual:

   ```bash
   python -m venv .venv
   ```

3. Activar el entorno virtual en Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Instalar las dependencias:

   ```bash
   python -m pip install -r requirements.txt
   ```

Se recomienda utilizar Python 3.11 para mantener la compatibilidad con la ejecución automática de GitHub Actions.

## Ejecución

Para ejecutar el análisis completo, entrenar los modelos y generar los archivos de resultados, utiliza:

```bash
python analisis_datos.py
```

El programa realiza las siguientes acciones:

- Carga y revisa el dataset.
- Comprueba los valores faltantes y los registros duplicados.
- Divide los datos en conjuntos de entrenamiento y prueba.
- Estandariza las variables predictoras.
- Entrena y evalúa los modelos de regresión lineal y clasificación logística.
- Genera la matriz de confusión.
- Crea automáticamente el informe de resultados.

## Archivos generados

- `resumen.txt`: contiene la información del dataset, el preprocesamiento, los parámetros, las métricas, la interpretación de los resultados y las conclusiones.
- `matriz_confusion.png`: muestra gráficamente los resultados de la clasificación.

## Resultados de referencia

En la ejecución local se obtuvieron los siguientes resultados:

**Regresión lineal**
- MAE: 0.2024.
- RMSE: 0.2629.
- R²: 0.7575.

**Clasificación logística**
- Accuracy: 0.8450.
- Precision: 0.8548.
- Recall: 0.8409.
- F1-score: 0.8478.

Estos valores corresponden a la ejecución de referencia y pueden variar si cambian los datos o la configuración del proyecto.

## Automatización con GitHub Actions

El proyecto cuenta con un flujo de trabajo ubicado en `.github/workflows/machine-learning.yml`.

Este flujo se ejecuta al subir cambios a la rama `main`, al crear o actualizar solicitudes de integración dirigidas a esa rama y cuando se inicia manualmente desde GitHub.

El flujo instala las dependencias, ejecuta el análisis y publica los archivos `resumen.txt` y `matriz_confusion.png` como un artefacto descargable cuando la ejecución finaliza correctamente.

## Estado del proyecto

Proyecto desarrollado con los modelos de regresión y clasificación implementados, generación automática del informe y configuración de ejecución mediante GitHub Actions.