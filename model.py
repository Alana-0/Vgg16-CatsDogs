import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Caminho para salvar o modelo
model_path = 'model.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    # Diretórios do conjunto de dados
    train_dir = 'dataset/train'
    val_dir = 'dataset/valid'

    # Data augmentation e pre-processamento
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # Carregar o modelo VGG16 pré-treinado sem as camadas de classificação no topo
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congelar as camadas da base VGG16
    for layer in base_model.layers:
        layer.trainable = False

    # Construir o modelo
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Duas classes, saída binária
    ])

    # Compilar o modelo
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Avaliar o modelo
    loss, accuracy = model.evaluate(val_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Salvar o modelo treinado
    model.save(model_path)
