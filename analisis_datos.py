
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# 3. Separar las variables predictoras y el objetivo
# El objetivo será predecir la variable price.
X = df.drop(columns=["price"])
y = df["price"]

print("\n--- VARIABLES DEL MODELO ---")
print(f"Variables predictoras: {X.shape[1]}")
print("Variable objetivo: price")

# 4. Dividir los datos en entrenamiento y prueba
# 80 % para entrenar y 20 % para evaluar.
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
# El escalador aprende únicamente de los datos de entrenamiento.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- PREPROCESAMIENTO ---")
print(f"Forma de X_train escalado: {X_train_scaled.shape}")
print(f"Forma de X_test escalado: {X_test_scaled.shape}")

print("\nPreprocesamiento completado correctamente.")
