"""
-- Acerca de la red --
Red neuronal convolucional (CNN) simple para clasificar imágenes CIFAR.
Utilizacion de la API secuencial de Keras.

-- Acerca del dataset --
Nombre:
    CIFAR10
Datos:
    60.000 imágenes en color en 10 clases, 6000 imágenes en cada clase.
Division:
    50.000 imágenes de entrenamiento.
    10.000 imágenes de prueba.

Las clases son mutuamente excluyentes y no hay superposición entre ellas.
"""
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#%% Carga del dataset

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # load_data() retorna una tupla de 4 arrays: 2 de entrenamiento y 2 de test.

# Normalizar los valores de los pixeles entre 0 y 1

train_images = train_images / 255.0 
test_images = test_images / 255.0

#%% Crear arquitectura de la red neuronal

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#%% Compilar y entrenar el modelo

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, 
                    train_labels, 
                    epochs=10, 
                    validation_data=(test_images, test_labels))

#%% Evaluar el modelo

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
