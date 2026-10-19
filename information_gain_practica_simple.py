import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv('/home/Nick/Documents/Machine Learning/Datasets/Automobile_data.xls')

# Conversion y limpieza de variable objetivo
objective_variable = df['price'] = pd.to_numeric(df['price'].replace('?', np.nan))
objective_variable = df.dropna(subset=['price'], inplace=True)

# Conversion y limpieza de variables independientes
clean_df = df[['make', 'fuel-type', 'aspiration', 'num-of-doors',
              'body-style', 'drive-wheels', 'engine-location',
              'wheel-base', 'length', 'width', 'height',
              'curb-weight', 'engine-type', 'num-of-cylinders',
              'engine-size', 'fuel-system', 'compression-ratio',
              'city-mpg', 'highway-mpg']].copy()

clean_df = clean_df.replace('?', np.nan)
clean_df.dropna(inplace=True)

for col in clean_df:
    if not is_numeric_dtype(clean_df[col]):
        clean_df[col] = pd.factorize(clean_df[col])[0] # como .factorize devuelve una tupla(codigo numerico, indice original), me quedo solo con el codigo numerico.

# Al borrar valores nulos, los indices entre clean_df y objetive_variable ya no coinciden.
# Se deben de extraer los indices de los valores/filas que sigan existiendo.

# 1. Alinear la variable objetivo (price) con el DataFrame limpio
y = df.loc[clean_df.index, 'price']

# 2. Calcular la Ganancia de Información
feature_importance = pd.Series(mutual_info_regression(clean_df, y, random_state=42), index=clean_df.columns)

# NOTA: Se pasa a una serie ya que "mutual_info_regression devuelve un numpy Array, por lo que la relacion
# entre valores y columnas se pierde.

print("\n--- Ranking de Importancia (Information Gain) ordenado de mayor a menor informacion respecto a y---")
print(feature_importance.sort_values(ascending=False))
