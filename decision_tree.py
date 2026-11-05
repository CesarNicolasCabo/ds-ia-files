#%% IMPORTS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#%% CARGA DE DATOS

df = pd.read_csv('path_to_dataset.csv')

#%% PREPARACION Y DIVISION DE DATOS

df = df.drop(columns='customer_id')

procesed_df = pd.get_dummies(df, drop_first=True)

X = procesed_df.drop(columns='churn')
y = procesed_df['churn']

#%% SEPARACION DE DATOS PARA ENTRENAMIENTO

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#%% INSTANCIACION DEL MODELO

decision_tree = DecisionTreeClassifier(criterion="gini")
decision_tree.fit(X_train, y_train)

#%% EVALUACION DEL MODELO
accuracy = decision_tree.score(X, y)