#%% IMPORTS
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
#%% CARGA DE DATOS
df = pd.read_csv('path_to_the_csv_file')

#%% PREPARACION DE DATOS
df.drop(columns='house_id', inplace=True)
processed_df = pd.get_dummies(df, drop_first=True)

#%% DIVISION DE DATOS EN VARIABLES
X = processed_df.drop(columns='price_usd') 
y = processed_df['price_usd']

#%% DIVISION DE DATOS PARA ENTRENAMIENTO Y EVALUACION

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% CREACION DE LOS MODELOS
regression_tree_absolute_error = DecisionTreeRegressor(criterion='absolute_error')
regression_tree_squared_error = DecisionTreeRegressor(criterion='squared_error')
regression_tree_poisson = DecisionTreeRegressor(criterion='poisson')
#%% ENTRENAMIENTO DE LOS MODELOS
regression_tree_squared_error.fit(X_train, y_train)
regression_tree_absolute_error.fit(X_train, y_train)
regression_tree_poisson.fit(X_train, y_train)
#%% PREDICCIONES
prediction_model_1 = regression_tree_squared_error.predict(X_test)
prediction_model_2 = regression_tree_absolute_error.predict(X_test)
prediction_model_3 = regression_tree_poisson.predict(X_test)
#%% EVALUACIONES
accuracy_model_1 = regression_tree_squared_error.score(X_test, y_test)
accuracy_model_2 = regression_tree_absolute_error.score(X_test, y_test)
accuracy_model_3 = regression_tree_poisson.score(X_test, y_test)

#%%
"""
Se comprueba que para el dataset utilizado en este desarrollo, el criterio de
error cuadratico obtiene una puntuacion mayor que los otros dos criterios:
"""
print(accuracy_model_1)
print(accuracy_model_2)
print(accuracy_model_3)
