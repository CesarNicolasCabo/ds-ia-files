"""
La red va a reconocer digitos escritos a mano encontrados en el dataset MNIST.
Cada imagen tendra un tamaño de 28*28 pixeles, que da un total de 784 pixeles.
Cada pixel tendra un valor que va desde 0 = negro a 255 = blanco.

Representacion matematica:
        [[x¹   .   .   .]]
    X = [[x²   .   .   .]]
        [[x³   .   .   .]]
        Filas = Ejemplos
        Columnas = 784 pixeles
        
Para este caso, X sera una matriz en donde cada fila (xⁿ) sera un ejemplo y,
cada uno de esos ejemplos (filas) tendra un total de 784 columnas de longitud, 
que son la totalidad de pixeles para cada imagen.

Luego, la matriz sera transpolada, para generar la matriz traspuesta, ya que
sera de utilidad al momento de realizar la retro-propagacion (backpropagation).
Ahora, cada columna es un ejemplo y tendra un total de 784 filas de longitud.

                         (t)
        [[x¹,   x²,  x³]]
        [[  .    .    .]]
    X = [[  .    .    .]]
        [[  .    .    .]]
        Filas = 784 pixeles
        Columnas = Ejemplos
        
Representacion de la red neuronal:
    
Input layer   hidden layer    output layer
    o               o               o
    o               o               o
    o               o               o        y^
    .               .               .
    .               .               .
  (784)           (10)            (10)   
   
La capa 0 (Input layer) tendra un total de 784 neuronas (una para cada pixel).
Las capas 1 y 2 (hidden y output layer) tendran un total de 10 neuronas (una 
para cada valor numerico posible)

Pasos a realizar:

1) Forward propagation:
Los datos de entrada se transmiten a través de las capas de la red, calculando
y almacenando las salidas de cada capa secuencialmente, hasta llegar a la capa
de salida.
    
- A⁰ = X                               (input layer)

- Z¹ =    W¹   *   A⁰  +   b¹          (hidden layer)
  10*m  10*784   784*m    10*1 => 10*m
  
- A¹ = g(z¹) = ReLU(z¹)                (activation function)
Nota: A¹ = Funcion ReLU aplicada a cada valor en Z¹. (x>0 -> x \ x<=0 -> 0)

- Z² =    W²   *   A¹  +  b²           (output layer)
  10*m   10*10   10*m    10*1 => 10*m

- A² = softmax(z²)                  (activation function)
Nota: A² = Funcion softmax aplicada a cada valor de z²

2) Back propagation:
Se comienza desde la prediccion y se vuelve hacia atras para encontrar cuánto
se desvió la predicción con respecto a la etiqueta real. Este desvio se repre-
senta como un error y, la idea es encontrar que pesos (W) y sesgos (b) contri-
buyeron mas a ese error, con el fin de ajustarlos.

--Para este caso se comienza desde la capa 2 (output layer)--

- dZ² = A² - y
Nota: dZ² representa un error en la capa 2 (output layer) e indica el grado
de error de la capa de salida.

Se toman las predicciones, se les resta las etiquetas reales y se codifica la
etiqueta correcta.
Por ejemplo: si y = 4, no se resta 4, sino que codificamos "y=4" en un array
similar al siguiente:

[0.1
 0.2
 0.05
 0.03
 0.90
 .
 .]

en donde el cuarto índice, que representa la cuarta clase, es uno y lo demás
es cero.

- dW² = 1/m * dZ² * A(¹*t) 
Nota: dW² es la derivada de la función de pérdida con respecto a los pesos
en la capa 2

- db² = 1/m Σ dZ²
Nota: db² es un promedio del error absoluto, indica cuánto se desvió la
salida.

--Para la capa 1 (hidden layer)--

dz¹ = W(²*t) * dz² * g'(Z¹)
Nota: En este paso se toma el error transpuesto de la segunda capa y se le
aplican los pesos en sentido inverso, para llegar a los errores de la primer
capa.
g prima (g') es la derivada de la función de activación (porque se tiene que
deshacer la función de activación para obtener el error apropiado para la
primera capa)

dW¹ = 1/m * dZ¹ * X^[t]
db¹ = 1/m Σ dZ¹
Nota: Se calcula cuánto contribuyeron W1 y b1 al error en la primera capa.

--Actualizacion de parametros--

Una vez que se encuentra cuanto contribuyeron los pesos y sesgos al error,
se deben de ajustar los parametros:
    
W¹ := W¹ - α * dW¹
b¹ := b¹ - α * db¹

W² := W² - α * dW²
b² := b² - α * db²

Nota: El operador morsa (:=) permite asignar un valor a una variable dentro de
una expresión, combinando asignación y uso en un solo paso.

Nota 2: Alfa (α) es una taza de aprendizaje; es un hiperparámetro que no es
entrenado por el modelo, sino que cuando se ejecuta el descenso de gradiente,
es establecido puramente por la persona, por lo tanto: NO es establecido por
el descenso de gradiente.

-- Sobre el dataset a utilizar --
Los archivos de datos train.csv y test.csv contienen imágenes en escala de
grises de dígitos dibujados a mano, del cero al nueve.
Cada imagen tiene 28 píxeles de alto y 28 píxeles de ancho, lo que suma un
total de 784 píxeles. Cada píxel a su vez, tiene un valor numerico asociado,
que va desde 0 hasta 255 e indica su nivel de claridad u oscuridad.

El conjunto de datos de entrenamiento (train.csv) tiene 785 columnas.
La primera columna, llamada "etiqueta", corresponde al dígito dibujado por el
usuario. El resto de las columnas contienen los valores de los píxeles de la
imagen asociada.

Cada columna de píxeles del conjunto de entrenamiento tiene un nombre como
"pixelx", donde "x" es un número entero entre 0 y 783, ambos inclusive.

Ejemplo:
"píxel31" indica el píxel que se encuentra en la cuarta columna (desde la iz-
quierda y la segunda fila desde arriba:

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 

El conjunto de datos de prueba (test.csv) es el mismo que el conjunto de entre-
namiento, excepto que no contiene la columna "etiqueta".
"""
#%% IMPORTS

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# """"""""""""""""""""""""""" PREPARACION DE DATOS """""""""""""""""""""""""""""""""""

dataframe = pd.read_csv(r'C:/Users/cesar/Documents/machine_learning/Datasets/digit-recognizer/train.csv')

matriz = np.array(dataframe)
filas, columnas = matriz.shape            # Nota: filas = m / columnas = n = features.
matriz_mezclada = np.random.shuffle(matriz)

matriz_traspuesta = matriz_mezclada[0:1000].T  # Primeros 1000 datos, traspuestos.
X = matriz_traspuesta[1:columnas] / 255.0      # Todas las columnas, menos la de labels.
y = matriz_traspuesta[0].astype(int)           # Columna de labels.

train = matriz_mezclada[1000:filas].T     # Resto de datos de la matriz, traspuestos.
X_train = train[1:columnas] / 255.0       # Todas las columnas, menos la de labels.
y_train = train[0].astype(int)            # Columna de labels.

#  "/255.0" para normalizar los valores entre 0 y 1.

# """"""""""""""""""""""""""" FIN PREPARACION DE DATOS """""""""""""""""""""""""""




# """"""""""""""""""""""""""" INICIALIZAR PESOS Y SESGOS """""""""""""""""""""""""""

def inicializar_pesos_y_sesgos():
    W1 = np.random.rand(10 ,784) - 0.5
    b1 = np.random.rand(10, 1)   - 0.5
    W2 = np.random.rand(10, 10)  - 0.5
    b2 = np.random.rand(10, 1)   - 0.5
    return W1, b1, W2, b2

def actualizar_pesos_y_sesgos(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# """"""""""""""""""""""""""" FIN INICIALIZACION DE PESOS Y SESGOS """""""""""""""""""""""




# """"""""""""""""""""""""""" DEFINICION DE FUNCIONES DE ACTIVACION """""""""""""""""""""""

def funcion_activacion_ReLU(Z): # Retorna "Z" (si z>0) o retorna "0" (si Z<=0)
    return np.maximum(Z, 0)

def ReLU_derivada(Z):
    return Z > 0

def funcion_activacion_softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# """"""""""""""""""""""""""""""" FIN FUNCIONES DE ACTIVACION """"""""""""""""""""""""""""""




# """""""""""""""""""""""""""""""""""""" ENCODING """"""""""""""""""""""""""""""""""""""

def one_hot_encoding(Y): # Convertir las etiquetas en array. Ver:"dZ² = A² - y"
    one_hot_encoding_Y = np.zeros((Y.size, Y.max() + 1)) # Matriz de ceros(())
    one_hot_encoding_Y[np.arange(Y.size), Y] = 1
    Y_encodeado_traspuesto = one_hot_encoding_Y.T
    return Y_encodeado_traspuesto

# """""""""""""""""""""""""""""""""" FIN ENCODING """"""""""""""""""""""""""""""""""




# """ RECORRIDO DE LA RED """

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1                  # hidden layer
    A1 = funcion_activacion_ReLU(Z1)
    Z2 = W2.dot(A1) + b2                 # output layer
    A2 = funcion_activacion_softmax(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    Y_resultante = one_hot_encoding(Y)

    # -- output layer --
    dZ2 = A2 - Y_resultante
    dW2 = 1 / filas * dZ2.dot(A1.T)
    db2 = 1 / filas * np.sum(dZ2)

    # -- hidden layer --
    dZ1 = W2.T.dot(dZ2) * ReLU_derivada(Z1)
    dW1 = 1 / filas * dZ1.dot(X.T)
    db1 = 1 / filas * np.sum(dZ1)

    return dW1, db1, dW2, db2

# """ FIN RECORRIDO """

def exactitud_de_modelo(predicciones, Y):
    predicciones_correctas = predicciones == Y
    exactitud = np.sum(predicciones_correctas) / Y.size
    return exactitud

def funcion_de_perdida(Y, A2):
    features = Y.size
    log_likelihood = np.log(A2[Y, range(features)])
    perdida = 1 / features * np.sum(log_likelihood)
    return perdida

def predicciones(A2):
    predicciones = np.argmax(A2, 0)
    return predicciones

def descenso_de_gradiente(X, y, alpha, iterations):
    W1, b1, W2, b2 = inicializar_pesos_y_sesgos()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, y)
        W1, b1, W2, b2 = actualizar_pesos_y_sesgos(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteracion: ", i)
            predictions = predicciones(A2)
            print("Exactitud: ", exactitud_de_modelo(predictions, y))
    return W1, b1, W2, b2

# PROBAR EL GRADIENTE DESCENDENTE:
W1, b1, W2, b2 = descenso_de_gradiente(X_train, y_train, 0.10, 150)

# EJECUCION
def realizar_prediccion(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    prediction_result = predicciones(A2)
    return prediction_result

def evaluar_predicciones(index, W1, b1, W2, b2):
    imagen = X_train[:, index, None]
    prediccion = realizar_prediccion(X_train[:, index, None], W1, b1, W2, b2)
    etiqueta = y_train[index]
    print("Prediccion: ", prediccion)
    print("Etiqueta: ", etiqueta)
    
    imagen = imagen.reshape((28, 28)) * 255
    plt.imshow(imagen, cmap='gray')
    plt.show()
    
# PRUEBA DE MODELO
evaluar_predicciones(0, W1, b1, W2, b2)
evaluar_predicciones(1, W1, b1, W2, b2) # Fallo
evaluar_predicciones(2, W1, b1, W2, b2)
evaluar_predicciones(3, W1, b1, W2, b2)
evaluar_predicciones(160, W1, b1, W2, b2)
evaluar_predicciones(20, W1, b1, W2, b2)
evaluar_predicciones(21, W1, b1, W2, b2)
evaluar_predicciones(22, W1, b1, W2, b2) # Fallo
evaluar_predicciones(23, W1, b1, W2, b2)
evaluar_predicciones(30, W1, b1, W2, b2) # Fallo


"""
Variables y forma

Forward prop

    A[0]=X: 784 x m
    Z[1]∼A[1]: 10 x m
    W[1]: 10 x 784 (as W[1]A[0]∼Z[1])
    B[1]: 10 x 1
    Z[2]∼A[2]: 10 x m
    W[1]: 10 x 10 (as W[2]A[1]∼Z[2])
    B[2]: 10 x 1

Backprop

    dZ[2]: 10 x m ( A[2])
    dW[2]: 10 x 10
    dB[2]: 10 x 1
    dZ[1]: 10 x m ( A[1])
    dW[1]: 10 x 10
    dB[1]: 10 x 1
"""
