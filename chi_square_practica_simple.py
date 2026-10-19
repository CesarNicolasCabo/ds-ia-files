import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv('/home/Nick/Documents/Machine Learning/Datasets/Automobile_data.xls')

# Crear tabla de frecuencias (https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)
table = pd.crosstab(df['make'], df['fuel-system'])

# 2. Ejecutar el test de Chi-cuadrado
chi2, p_valor, degrees_of_freedom, esperados = chi2_contingency(table)

if p_valor < 0.05:
    print("Relación encontrada")
    print(p_valor, chi2, degrees_of_freedom, esperados)

"""
p_valor: Responde a la pregunta: ¿Qué tan probable es que esta relación sea puro azar?
Si es menor a 0.05, se rechaza la idea de que son independientes. Es decir, hay una relación significativa.
Cuanto mas cerca de 0 es el valor, menos "azarosa" es la relacion.
Si es mayor a 0.05: No se puede asegurar que haya una relación; la diferencia se debe probablemente a la casualidad.

chi2: Número que mide qué tan diferentes son los datos reales de lo que se esperaría si no hubiera ninguna relación.
En estadística, cuanto más alejado de cero esté este número, más "distantes" están los  datos reales de una distribución
aleatoria.

degrees_of_freedom: Indica cuánta "libertad" tienen los datos para variar. Es un valor informativo que Python usa
internamente para calcular el p-valor.
A mayor grado de libertad, más datos necesitas para que el test sea fiable.

esperados: Es una matriz (un array de NumPy) que sirve para comparar la tabla original (pd.crosstab) con la de esperados.
En este punto se observa dónde están las diferencias más grandes.

NOTA:
chi_2, dof y esperados se suelen guardar por si se necesita hacer un reporte estadístico más profundo,
o para validar que el test se hizo sobre una muestra lo suficientemente grande.
"""