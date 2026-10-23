import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
print("\nEntrenamiento y evaluación finalizados.")
