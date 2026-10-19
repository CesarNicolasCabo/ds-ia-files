"""
El algoritmo kNN calcula la distancia entre un punto de datos de muestra y 
todos los demás puntos de datos de entrenamiento; dicha distancia puede ser
euclidiana o de Manhattan.
Luego, selecciona los k puntos de datos más cercanos, donde k puede ser
cualquier número entero.
Finalmente, asigna el punto de datos de muestra a la clase a la que pertenece
la mayoría de los k puntos de datos.
"""
# IMPORTS

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# LECTURA DE DATASET
url = "https://docs.google.com/spreadsheets/d/1zpIvlcGlGgqHIcDjzGMPp2r4T93QjuTUmhqO1kujxqc/edit?gid=0#gid=0"
fixed_url = url.replace('/edit?gid=', '/export?format=csv&amp;gid=')

df = pd.read_csv(fixed_url)

# FEATURE ENGINEERING
df = df.rename(columns={'anio de nacimiento':'Año de nacimiento'})

# VISUALIZAR AÑO VS GENERACION
sns.histplot(df, x='Año de nacimiento', hue='Generacion')
plt.show()

# DIVISION DE DATOS
X = df[['Año de nacimiento']]
y = df.iloc[:,1] # En el dataframe, esta columna representa la "Generacion".
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# CREAR Y ENTRENAR MODELO
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# CREAR VARIABLE PREDICTIVA
y_pred = knn.predict(X_test)

# EVALUAR MODELO
print(np.mean(y_pred == y_test))

"""
Segun las ejecuciones realizadas, se espera que para este modelo, 
el acierto sea de un 0.9722222222222222%.
"""
# PROBAR NUEVO DATO
new_X = [[2008]]
new_predict = knn.predict(new_X)

"""
Para este caso, "new_predict" contiene el valor: "Generacion_Z",
el cual es correcto tomando encuenta el Año ingresado en "new_X"
"""
