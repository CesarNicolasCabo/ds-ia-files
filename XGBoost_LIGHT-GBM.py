#%%
import lightgbm as lgb
import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import pandas as pd
#%%
# Cargar DataFrame
df = pd.read_csv("customer_churn_dataset.csv")

# Separar las variables independientes (X) y la variable objetivo (y)
feature_columns = [
"tenure_months",
"monthly_charges",
"total_spend",
"support_tickets",
"login_frequency",
]

X = df[feature_columns]
y = df["churn"]
#%%
# Separacion de datos de entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y) # stratify asegura que se respeten los porcetajes de división del datasets
#%%
# Estructurar modelo: XGBoost
scale_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

xgboost_model = xgb.XGBClassifier(
n_estimators=100,
learning_rate=0.05,
max_depth=4, # Limite maximo de niveles de arboles
scale_pos_weight=scale_weight,
random_state=42
)
#%%
# Entrenamiento del modelo
xgboost_model.fit(X_train, y_train)
xgb_preds = xgboost_model.predict(X_test)
#%%
# Estructurar modelo: LIGHTBM
lightgbm_model = lgb.LGBMClassifier(
n_estimators=100,
learning_rate=0.05,
max_depth=4,
class_weight="balanced",
random_state=42
)
#%%
# Entrenamiento del modelo
lightgbm_model.fit(X_train, y_train)
lgb_preds = lightgbm_model.predict(X_test)
#%%
# Mostrar resultados de evaluación de modelos
print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_preds))

print("\n=== LightGBM Classification Report ===")
print(classification_report(y_test, lgb_preds))
#%%