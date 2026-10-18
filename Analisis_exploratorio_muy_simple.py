import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

edades_clientes = np.random.uniform(low=18, high=65, size=1000)

ingresos_clientes = np.random.lognormal(mean=7.82, sigma=0.60, size=1000) # sigma = mediana

df_clientes = pd.DataFrame(
    {
        "EDAD": np.round(edades_clientes).astype(int),
        "INGRESOS": np.round(ingresos_clientes, 2),
    }
)

print('Media, mediana y desviacion estandar de columna EDAD:')
print(df_clientes['EDAD'].mean())
print(df_clientes['EDAD'].median())
print(df_clientes['EDAD'].std())

print('Media, mediana y desviacion estandar de columna INGRESOS:')
print(df_clientes['INGRESOS'].mean())
print(df_clientes['INGRESOS'].median())
print(df_clientes['INGRESOS'].std())

print('Asimetria del dataframe:')
print(df_clientes.skew())

print('Quartiles:')
q1, q2, q3 = np.percentile(df_clientes['EDAD'], [25, 50, 75])
print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")

print('Matriz de correlacion de Pearson:')
matriz_correlacion = df_clientes.corr(method='pearson')
print(matriz_correlacion)

plt.figure(figsize=(10,5))
sns.heatmap(matriz_correlacion, annot=True)
plt.show()
