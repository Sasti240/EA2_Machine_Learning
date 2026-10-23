import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Cargar el dataset desde Hugging Face
print("Cargando el dataset...")

dataset = load_dataset(
    "inria-soda/tabular-benchmark",
    data_files="reg_cat/house_sales.csv",
    split="train"
)

df = dataset.to_pandas()

print(f"Registros cargados: {df.shape[0]}")
print(f"Columnas cargadas: {df.shape[1]}")

# 2. Revisar la calidad de los datos
print("\n--- REVISIÓN DE LOS DATOS ---")
print(f"Valores faltantes: {df.isnull().sum().sum()}")
print(f"Registros duplicados: {df.duplicated().sum()}")

# 3. Separar variables predictoras y objetivo
X = df.drop(columns=["price"])
y = df["price"]

print("\n--- VARIABLES DEL MODELO ---")
print(f"Variables predictoras: {X.shape[1]}")
print("Variable objetivo: price")

# 4. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- DIVISIÓN DE LOS DATOS ---")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")

# 5. Escalar las variables predictoras
# El escalador se ajusta únicamente con los datos de entrenamiento.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocesamiento completado correctamente.")

# 6. MODELO DE REGRESIÓN LINEAL
print("\n" + "=" * 45)
print("MODELO 1: REGRESIÓN LINEAL")
print("=" * 45)

regresion = LinearRegression()
regresion.fit(X_train_scaled, y_train)

y_pred_reg = regresion.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
r2 = r2_score(y_test, y_pred_reg)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\nIntercepto:", round(regresion.intercept_, 4))

# Mostrar los cinco coeficientes con mayor valor absoluto.
coeficientes = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": regresion.coef_
})

coeficientes["Magnitud"] = coeficientes["Coeficiente"].abs()
coeficientes = coeficientes.sort_values(
    by="Magnitud",
    ascending=False
)

print("\nCinco coeficientes de mayor magnitud:")
print(
    coeficientes[["Variable", "Coeficiente"]]
    .head(5)
    .to_string(index=False)
)

# 7. MODELO DE CLASIFICACIÓN
# Se clasifica cada vivienda como precio alto o bajo.
# El umbral se calcula solamente con los datos de entrenamiento.
print("\n" + "=" * 45)
print("MODELO 2: CLASIFICACIÓN LOGÍSTICA")
print("=" * 45)

umbral = y_train.median()

y_train_cls = (y_train >= umbral).astype(int)
y_test_cls = (y_test >= umbral).astype(int)

print(f"Umbral de clasificación: {umbral:.4f}")
print("Clase 0: precio por debajo del umbral")
print("Clase 1: precio igual o superior al umbral")

clasificador = LogisticRegression(max_iter=1000)
clasificador.fit(X_train_scaled, y_train_cls)

y_pred_cls = clasificador.predict(X_test_scaled)

accuracy = accuracy_score(y_test_cls, y_pred_cls)
precision = precision_score(
    y_test_cls, y_pred_cls, zero_division=0
)
recall = recall_score(
    y_test_cls, y_pred_cls, zero_division=0
)
f1 = f1_score(
    y_test_cls, y_pred_cls, zero_division=0
)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 8. Matriz de confusión
matriz = confusion_matrix(
    y_test_cls,
    y_pred_cls,
    labels=[0, 1]
)

print("\nMatriz de confusión:")
print(matriz)
print("\nFilas: valores reales (0, 1)")
print("Columnas: valores predichos (0, 1)")

# Guardar una imagen de la matriz para el informe.
fig, ax = plt.subplots(figsize=(6, 5))
imagen = ax.imshow(matriz, cmap="Blues")

ax.set_title("Matriz de confusión - Clasificación logística")
ax.set_xlabel("Clase predicha")
ax.set_ylabel("Clase real")
ax.set_xticks([0, 1], labels=["Precio bajo", "Precio alto"])
ax.set_yticks([0, 1], labels=["Precio bajo", "Precio alto"])

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(matriz[i, j]),
            ha="center", va="center", color="black"
        )

fig.colorbar(imagen, ax=ax)
fig.tight_layout()
fig.savefig("matriz_confusion.png", dpi=150)
plt.close(fig)


print("\nImagen guardada: matriz_confusion.png")

# 9. Generar informe automático
fecha_ejecucion = datetime.now().astimezone().strftime(
    "%Y-%m-%d %H:%M:%S %Z"
)

# Registrar las estadísticas antes y después del escalado.
escalado = pd.DataFrame({
    "Variable": X.columns,
    "Media antes": X_train.mean().values,
    "Desviación antes": X_train.std(ddof=0).values,
    "Media después": X_train_scaled.mean(axis=0),
    "Desviación después": X_train_scaled.std(axis=0)
})

# Guardar los coeficientes de todas las variables.
coeficientes_informe = coeficientes[
    ["Variable", "Coeficiente"]
].to_string(index=False)

# Interpretación básica de los resultados.
if r2 >= 0.70:
    interpretacion_r2 = (
        "El modelo explica una proporción considerable de la "
        "variabilidad de la variable objetivo."
    )
elif r2 >= 0.40:
    interpretacion_r2 = (
        "El modelo explica una proporción intermedia de la "
        "variabilidad de la variable objetivo."
    )
else:
    interpretacion_r2 = (
        "El modelo explica una proporción limitada de la "
        "variabilidad de la variable objetivo."
    )

informe = f"""
INFORME DE MACHINE LEARNING
===========================

Fecha de ejecución: {fecha_ejecucion}

1. INFORMACIÓN DEL DATASET
--------------------------
Repositorio: inria-soda/tabular-benchmark
Archivo: reg_cat/house_sales.csv
Fuente: https://huggingface.co/datasets/inria-soda/tabular-benchmark

Descripción:
El dataset contiene información de viviendas de King County,
Washington, Estados Unidos. La variable objetivo es price.
Sus valores parecen estar transformados, por lo que no se
interpretan directamente como dólares.

Número de registros: {df.shape[0]}
Número de columnas: {df.shape[1]}

Columnas:
{", ".join(df.columns)}

Tipos de datos:
{df.dtypes.to_string()}

Estadísticas descriptivas:
{df.describe().round(4).to_string()}


2. REVISIÓN Y PREPROCESAMIENTO
------------------------------
Valores faltantes: {df.isnull().sum().sum()}
Registros duplicados: {df.duplicated().sum()}

No se encontraron valores faltantes ni duplicados.
Por este motivo, no fue necesario imputar datos ni eliminar
registros duplicados. No se eliminaron posibles valores atípicos
automáticamente.

Variable objetivo: price
Número de variables predictoras: {X.shape[1]}

División de los datos:
- Entrenamiento: {len(X_train)} registros.
- Prueba: {len(X_test)} registros.
- Proporción: 80 % entrenamiento y 20 % prueba.
- random_state: 42.

Transformación aplicada:
Se utilizó StandardScaler para estandarizar las variables
predictoras. El escalador se ajustó únicamente con los datos
de entrenamiento y luego se aplicó al conjunto de prueba.

Estadísticas antes y después del escalado en entrenamiento:
{escalado.round(4).to_string(index=False)}

La media de las variables escaladas queda aproximadamente en
cero y su desviación estándar en uno. Esto facilita que los
modelos trabajen con variables de diferentes escalas.


3. MODELO DE REGRESIÓN LINEAL
-----------------------------
Modelo: LinearRegression
Parámetros:
- fit_intercept: {regresion.fit_intercept}
- Variables predictoras: {X.shape[1]}
- Datos utilizados para evaluar: {len(X_test)}

Métricas:
- MAE: {mae:.4f}
- RMSE: {rmse:.4f}
- R²: {r2:.4f}

Interpretación:
El MAE representa el error absoluto medio y el RMSE penaliza
más los errores grandes. Ambos están expresados en la escala
de la variable price disponible en el dataset.

El R² fue de {r2:.4f}. {interpretacion_r2}

Intercepto: {regresion.intercept_:.6f}

Coeficientes:
{coeficientes_informe}

Los coeficientes corresponden a variables estandarizadas.
Representan el cambio estimado en price asociado con un aumento
de una desviación estándar en cada predictor, manteniendo
constantes las demás variables. No implican causalidad.


4. MODELO DE CLASIFICACIÓN LOGÍSTICA
------------------------------------
Modelo: LogisticRegression
Parámetros:
- max_iter: {clasificador.max_iter}
- Variables predictoras: {X.shape[1]}
- Umbral de clasificación: {umbral:.4f}

Definición de las clases:
- Clase 0: price inferior al umbral.
- Clase 1: price igual o superior al umbral.

El umbral se calculó utilizando únicamente los precios del
conjunto de entrenamiento.

Métricas:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-score: {f1:.4f}

Matriz de confusión:
Filas: clase real. Columnas: clase predicha.

                 Predicha 0    Predicha 1
Real 0              {matriz[0, 0]}           {matriz[0, 1]}
Real 1              {matriz[1, 0]}           {matriz[1, 1]}

Interpretación:
Accuracy mide la proporción total de predicciones correctas.
Precision indica qué proporción de las predicciones de clase 1
fueron correctas. Recall mide cuántos casos reales de clase 1
fueron identificados. F1-score combina precision y recall.


5. LIMITACIONES
---------------
- La variable price parece estar transformada; es necesario
  verificar la transformación original antes de convertir
  las métricas a unidades monetarias.
- La regresión lineal puede no representar todas las relaciones
  entre las características de las viviendas y su precio.
- La clasificación agrupa los valores en dos clases y pierde
  información sobre las diferencias exactas de precio.
- Los resultados corresponden a una única división de los datos.
- Las predicciones no demuestran relaciones de causa y efecto.


6. CONCLUSIONES
---------------
Se entrenaron y evaluaron dos modelos utilizando el mismo
dataset y una división de entrenamiento y prueba del 80/20.

La regresión lineal obtuvo un MAE de {mae:.4f}, un RMSE de
{rmse:.4f} y un R² de {r2:.4f}.

La clasificación logística obtuvo un accuracy de
{accuracy:.4f} y un F1-score de {f1:.4f} para diferenciar
viviendas por debajo o por encima del umbral definido.

Estos resultados permiten realizar una primera evaluación
de ambos modelos. En futuras pruebas sería útil compararlos
con otros algoritmos y utilizar diferentes particiones para
analizar su capacidad de generalización.

Archivos generados:
- resumen.txt
- matriz_confusion.png
"""

with open("resumen.txt", "w", encoding="utf-8") as archivo:
    archivo.write(informe.strip() + "\n")

print("Informe generado: resumen.txt")
print("\nEntrenamiento y evaluación finalizados.")

