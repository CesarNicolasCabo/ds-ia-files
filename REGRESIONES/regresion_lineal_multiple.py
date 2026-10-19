import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:/Users/cesar/Desktop/50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# FEATURE ENGINERING

column_transformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough')

X = np.array(column_transformer.fit_transform(X))

# DIVISION DE DATOS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

# ENTRENAMIENTO DE MODELO

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# COMPARATIVA CON DATOS DE TEST

y_pred = regressor.predict(X_test)
print(y_pred)
print("------")
print(y_test)

# EVALUACION DE DESEMPENIO DEL MODELO

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Squared Error Multioutput:", mean_squared_error(y_test, y_pred, multioutput='raw_values'))
print("R-squared:", r2_score(y_test, y_pred))
