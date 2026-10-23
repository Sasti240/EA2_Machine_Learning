
from datasets import load_dataset

# Cargar el conjunto de datos desde Hugging Face
dataset = load_dataset(
    "inria-soda/tabular-benchmark",
    data_files="reg_cat/house_sales.csv",
    split="train"
)

# Convertir los datos a un DataFrame de Pandas
df = dataset.to_pandas()

# Mostrar información general
print("PRIMERAS FILAS DEL DATASET")
print(df.head())

print("\nDIMENSIONES DEL DATASET")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

print("\nNOMBRES DE LAS COLUMNAS")
print(df.columns.tolist())

print("\nTIPOS DE DATOS Y VALORES NO NULOS")
df.info()

print("\nVALORES FALTANTES POR COLUMNA")
print(df.isnull().sum())

print("\nRESUMEN ESTADÍSTICO")
print(df.describe().round(2))

# Revisar registros duplicados
print("\nREGISTROS DUPLICADOS")
print(f"Cantidad: {df.duplicated().sum()}")

# Revisar la distribución del precio
print("\nDISTRIBUCIÓN DE PRICE")
print(df["price"].describe().round(3))

# Revisar los valores de date_year
print("\nVALORES DE DATE_YEAR")
print(df["date_year"].value_counts().sort_index())

# Revisar los valores de habitaciones
print("\nVALORES DE BEDROOMS")
print(df["bedrooms"].value_counts().sort_index())
