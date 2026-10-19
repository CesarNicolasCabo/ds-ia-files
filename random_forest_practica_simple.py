from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = load_iris()
X, y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(
        n_estimators=100,
        bootstrap=True,      # Activa el muestreo aleatorio con reemplazo
        oob_score=True,      # Activa la validación interna con los datos no usados
        random_state=42,
        n_jobs=-1            # Usa todos los procesadores de tu PC para ir más rápido
    )

random_forest.fit(X_train, y_train)

predictions = random_forest.predict(X_test)
precision = accuracy_score(y_test, predictions)

print(f"y_test: {y_test}")
print(f"Predictions: {predictions}")
print(f"Precision: {precision}")

