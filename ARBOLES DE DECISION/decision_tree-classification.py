#%% IMPORTS

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#%% CARGA DE DATOS

df = pd.read_csv('path_to_the_csv_file')

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

#%% AGREGAR NUEVA VARIABLE A PREDECIR

new_x = pd.DataFrame([{
    'age': 42,
    'tenure_months': 6,
    'monthly_charges': 85.50,
    'total_charges': 513.00,
    'num_support_tickets': 2,
    'contract_type_One year': True,
    'contract_type_Two year': False,
    'payment_method_Credit card': True,
    'payment_method_Electronic check': False,
    'payment_method_Mailed check': False,
    'internet_service_Fiber optic': False,
    'internet_service_No': True,
    'tech_support_Yes': True
}])

prediccion = decision_tree.predict(new_x)
print(f"Predicción de Churn (0=Permanece, 1=Abandona): {prediccion[0]}")

probabilidad_de_prediccion = decision_tree.predict_proba(new_x)
print(f"Probabilidad de prediccion: {probabilidad_de_prediccion}")
