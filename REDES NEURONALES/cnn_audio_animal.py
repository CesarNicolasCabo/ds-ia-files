#%%
import numpy as np
import random
import tensorflow as tf
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model # <- Importar Model
# Importaciones para Transfer Learning
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D # <- Importado GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

#%%

# -------------------------------------------------------------
# 1. Definición de Rutas (SIN CAMBIOS)
# -------------------------------------------------------------

CARPETA_RAIZ = Path('C:/Users/cesar/Documents/machine_learning/Datasets/cats_dogs_audio/audios')
CARPETA_SALIDA = CARPETA_RAIZ / 'espectrogramas'

# -------------------------------------------------------------
# 2. Generacion de espectrograma (SIN CAMBIOS)
# -------------------------------------------------------------

def generar_espectrograma(audio_path, output_path):

    try:
        # Cargar el audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generar el espectrograma
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Configurar la figura sin bordes ni ejes
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        librosa.display.specshow(spectrogram_db, ax=ax, y_axis='mel', x_axis='time')
        
        # Guardar la imagen en la ruta de salida
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Liberar memoria
        
        print(f"Espectrograma guardado: {output_path.name}")
        return True
            
    except Exception as e:
        print(f"Error al procesar {audio_path.name}: {e}")
        return False

# -------------------------------------------------------------
# 3. Creación de Carpetas y Recorrido (SIN CAMBIOS)
# -------------------------------------------------------------

# Asegurar que el directorio de salida principal exista
CARPETA_SALIDA.mkdir(exist_ok=True)

# Iterar sobre las subcarpetas del directorio de entrada
for subcarpeta in CARPETA_RAIZ.iterdir():

    if subcarpeta.name in ['espectrogramas']:
        continue # Corta el recorrido y salta la carpeta.

    # Crear el subdirectorio de salida para esta clase
    # El nombre de la subcarpeta es la etiqueta de la clase
    class_name = subcarpeta.name
    class_output_dir = CARPETA_SALIDA / class_name
    class_output_dir.mkdir(exist_ok=True)
    
    # Recorrer los archivos .wav dentro de la subcarpeta de audio
    for audio_file in subcarpeta.glob('*.wav'):
        
        # Crear la ruta de salida de la imagen
        image_filename = audio_file.stem + '.png'
        image_path = class_output_dir / image_filename
        
        # Generar y guardar el espectrograma
        generar_espectrograma(audio_file, image_path)

print("\nProceso de conversión de audios a espectrogramas completado.")
#%% FIJAR SEMILLA ALEATORIA PARA REPRODUCIBILIDAD (SIN CAMBIOS)

seed_value = 42 # Nota: El valor de la semilla puede ser cualquiera que se elija.
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
#%% CARGA Y NORMALIZACION DE IMAGENES (SIN CAMBIOS)

# ======= Generador de Entrenamiento (CON Aumento de Datos) ==========
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2, # Esto para separar los archivos
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ========= Generador de Validación (SIN Aumento de Datos) ===========
validation_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2 # Debe de ser el mismo split que el train_datagen.
)
#%%
directorio_entrenamiento = 'C:/Users/cesar/Documents/machine_learning/Datasets/cats_dogs_audio/audios/espectrogramas'

# === Crear los generadores usando sus respectivas configuraciones ===
train_generator = train_datagen.flow_from_directory(
    directorio_entrenamiento,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=32,
    seed=seed_value, # Semilla pasada directamente para disminuir aleatoriedad.
    class_mode='categorical',
    subset='training' # Usa el 80% para entrenamiento
)

validation_generator = validation_datagen.flow_from_directory(
    directorio_entrenamiento,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=32,
    seed=seed_value, # Semilla pasada directamente para disminuir aleatoriedad.
    class_mode='categorical',
    subset='validation' # Usa el 20% para validación
)

# Imprimir las etiquetas que Keras ha detectado
print(train_generator.class_indices)

# Obtener el número de clases
num_classes = len(train_generator.class_indices) 

#%% CONSTRUCCION DE LA RED CON TRANSFER LEARNING

# 1. CARGA DEL MODELO BASE VGG16 PRE-ENTRENADO EN IMAGENET
VGG16_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(128, 128, 3))

#%% FASE 1: CONGELAMIENTO PARCIAL (AJUSTE FINO)

# 1. Asegurar que TODAS las capas de la base VGG16 sean entrenables
for layer in VGG16_model.layers:
    layer.trainable = True

# 2. Congelar las primeras capas (las de bajo nivel)
# VGG16 tiene 19 capas. Se congelan las primeras 13 capas (Bloques 1 al 4) para proteger la detección de bordes y texturas básicas.
print("Congelando las primeras 13 capas de VGG16...")
for layer in VGG16_model.layers[:13]: 
    layer.trainable = False

# Las últimas 6 capas convolucionales de VGG16 (capas del bloque 5) y el clasificador están ahora disponibles para el ajuste fino.
# El número de parámetros entrenables aumentará.

# 3. CONSTRUCCIÓN DEL CLASIFICADOR (TOP LAYERS)
x = VGG16_model.output
x = GlobalAveragePooling2D()(x)

# Capa Densa (Clasificador)
x = Dense(512, activation='relu')(x)
 
# Desactiva el 50% de las neuronas al azar
x = Dropout(0.5)(x) 
 
# Capa de Salida (2 clases:gatos y perros)
predictions = Dense(num_classes, activation='softmax')(x)
 
# 4. DEFINIR EL MODELO FINAL
model = Model(inputs=VGG16_model.input, outputs=predictions)

#%% COMPILACIÓN Y ENTRENAMIENTO

# Nota: Es crucial usar una tasa de aprendizaje MUY baja para el TL (5e-5 o incluso 1e-5 son comunes).

# --- 1. Compilacion ---
modified_optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=modified_optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("\nResumen del modelo con capas parcialmente descongeladas:")
model.summary()

# --- 2. Definición de EarlyStopping ---
early_stopping = EarlyStopping(monitor='val_accuracy',  # Vigilar la precisión.
                               mode='max',              # Buscar el valor máximo.
                               patience=20,
                               restore_best_weights=True,
                               verbose=1)
# --- 3. Entrenamiento ---
print("\nIniciando entrenamiento...")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

model.save('modelo_clasificacion_audio_TRANSFER_LEARNING.h5')

print("\nEvaluación final con los mejores pesos restaurados:")
val_loss, val_accuracy = model.evaluate(validation_generator)

print("\nFin entrenamiento")