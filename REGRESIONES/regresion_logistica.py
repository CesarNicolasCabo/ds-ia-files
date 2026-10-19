# IMPORTS

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

# DATASET

df = pd.read_csv(r'C:/Users/cesar/Documents/machine_learning/Datasets/Social_Network_Ads.csv')

X = df.iloc[:,1:4]
y = df.iloc[:,-1]

# TRATAMIENTO DE VARIABLE CATEGORICA

column_transformer = ColumnTransformer(transformers=[(
    'encoder',
    OneHotEncoder(),
    [0])],
    remainder='passthrough')

X = np.array(column_transformer.fit_transform(X))

# DIVISION DE DATA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# AJUSTAR DATOS DE ENTRENAMINETO

standar_scaler = StandardScaler()
X_train = standar_scaler.fit_transform(X_train) 
X_test = standar_scaler.fit(X_test)

# CREACION DEL MODELO

log_regressor = LogisticRegression(random_state=13, max_iter=101)
log_regressor.fit(X_train, y_train)

# REALIZAR PREDICCION

y_pred = log_regressor.predict(X_test)

print(y_pred)
print("----------")
print(y_test)

# RENDIMIENTO DEL MODELO

print(mean_squared_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print(r2_score(y_test, y_pred))

# MATRIZ DE CONFUSION

matriz_confusion = confusion_matrix(y_test, y_pred)
print(matriz_confusion)
