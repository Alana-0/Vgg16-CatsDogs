import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Carregar o modelo salvo
model = load_model('model.h5')

def predict_image(image_path):
    try:
        # Carregar a imagem e redimensioná-la para o tamanho esperado pelo modelo
        img = image.load_img(image_path, target_size=(224, 224))
    except FileNotFoundError:
        print(f"Erro: A imagem {image_path} não foi encontrada.")
        return None

    # Converter a imagem para um array numpy e adicionar uma dimensão extra para representar o batch size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Pré-processar a imagem da mesma forma que foi feita durante o treinamento
    img_array /= 255.0

    # Fazer a previsão
    prediction = model.predict(img_array)

    # Interpretar a previsão
    if prediction[0] > 0.5:
        return "Classe: 1" #cachorro
    else:
        return "Classe: 0" #gato

# Lista de caminhos das imagens que você deseja testar
image_paths = [
    r'C:\Users\Lana\Desktop\Lana\python\imagem_de_teste.jpeg',
    r'C:\Users\Lana\Desktop\Lana\python\imagem_de_teste2.jpeg',
    r'C:\Users\Lana\Desktop\Lana\python\imagem_de_teste3.jpeg',
    r'C:\Users\Lana\Desktop\Lana\python\imagem_de_teste4.jpeg',
 
]

# Loop através de cada imagem, fazer a previsão e exibir resultados
for image_path in image_paths:
    print(f"Imagem: {image_path}")
    prediction_result = predict_image(image_path)
    if prediction_result:
        print(f"Predição: {prediction_result}")

        # Exibir a imagem
        img = image.load_img(image_path, target_size=(224, 224))
        plt.imshow(img)
        plt.title(f'Predição: {prediction_result}')
        plt.show()
    print("\n")
