import pandas as pd

dataset_izquierda = pd.DataFrame({
    'planetas': ['Marte', 'Venus', 'Saturno', 'Tierra'],
    'diametro':[6779, 12104, 120536, 12742]
    })

dataset_derecha = pd.DataFrame({
    'planetas': ['Jupiter', 'Marte', 'Saturno', 'Mercurio'],
    'diametro':[142984, 6779, 120536, 4880]
    })

print("///////////INNER////////////////")
print(dataset_izquierda.merge(dataset_derecha, on='planetas', how='inner'))

print("///////////LEFT/////////////////")
print(dataset_izquierda.merge(dataset_derecha, on='planetas', how='left'))

print("///////////RIGHT////////////////")
print(dataset_izquierda.merge(dataset_derecha, on='planetas', how='right'))

print("///////////OUTER////////////////")
print(dataset_izquierda.merge(dataset_derecha, on='planetas', how='outer'))
