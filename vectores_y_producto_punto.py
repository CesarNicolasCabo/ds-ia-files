import numpy as np

vectorA = np.array([1, 2, 3, 4, 5]) # Vector unidimensional.
vectorB = np.array([6, 7, 8, 9, 10]) # Vector unidimensional.

# Producto punto: Devuelve un número escalar que resulta de sumar las multiplicaciones de sus elementos correspondientes.

print(np.dot(vectorA, vectorB)) # Forma 1
print(vectorA@vectorB) # Forma 2

producto_punto = 0
for a,b in zip(vectorA, vectorB):
	producto_punto += a*b
	
print(producto_punto)
