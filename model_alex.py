import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import pandas as pd
import google.generativeai as genai

class Prompt:
    def __init__(self, nutrition_file_path):
        self.nutrition_file_path = nutrition_file_path
        self.combined_nutrition_data = self.load_combined_nutrition_data()

    def load_combined_nutrition_data(self):
        # Baca data dari file CSV
        combined_df = pd.read_csv(self.nutrition_file_path, index_col=0)
        # Ubah kunci dalam kamus ke huruf kecil
        combined_nutrition_data_lower = {key.lower(): value for key, value in combined_df.to_dict(orient='index').items()}
        return combined_nutrition_data_lower

    def get_nutrition_info(self, list_makanan):
        nutrisi_dict = {}
        for predicted in list_makanan:
            predicted_lower = predicted.lower()
            if predicted_lower in self.combined_nutrition_data:
                nutrition_info = self.combined_nutrition_data[predicted_lower]
                nutrisi_dict[predicted.capitalize()] = nutrition_info
            else:
                nutrisi_dict[predicted.capitalize()] = "Data nutrisi tidak ditemukan."
        return nutrisi_dict

    def generate_content(self, prompt):
        # Konfigurasi API
        genai.configure(api_key='AIzaSyDk9BblI-kPgGHGZxaWGOBzeYG9KOzu4UU')
        model_gem = genai.GenerativeModel('gemini-pro')
        # Menggunakan model atau API untuk menghasilkan konten berdasarkan prompt
        response = model_gem.generate_content(prompt)
        return response.text

class Predict:
    def __init__(self, model_path, item_names):
        self.model = tf.keras.models.load_model(model_path)
        self.item_names = item_names

    def predict_image(self, img_path, target_size):
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        class_pred, _ = self.model.predict(img)
        class_id = np.argmax(class_pred)
        predicted_class = self.item_names[class_id]
        return predicted_class

def main():
    # List of class names
    label = [
        "Kentang",       # 0
        "Tomat",         # 1
        "Timun jepang",  # 2
        "Bawang bombay", # 3
        "Lobak",         # 4
        "Terong",        # 5
        "Daun bawang",   # 6
        "Kol ungu",      # 7
        "Wortel",        # 8
        "Telur",         # 9
        "Daging sapi",   # 10
        "Pork",          # 11
        "Daging ayam"    # 12
    ]

    # Path to the image
    img_path = '/content/WhatsApp Image 2024-06-07 at 15.11.27_feb3e7c6.jpg'
    target_size = (224, 224)

    # Initialize the Predict object
    predictor = Predict('/content/mobilenet_model.h5', label)

    # Predict class and bounding box
    predicted_class = predictor.predict_image(img_path, target_size)

    # Initialize the Prompt object
    prompt = Prompt('combined_nutrition_data.csv')

    # Get nutrition info
    list_makanan = ["ayam", "Abon", "Bakwan", "Bawang Merah"]
    list_makanan.append(predicted_class.lower())
    nutrisi_dict = prompt.get_nutrition_info(list_makanan)

    # Generate recipe prompt
    prompt_text = f"Saya punya bahan utama makanan berikut: {', '.join(list_makanan)}. Tolong rekomendasikan resep makanan untuk ibu hamil dan tampilan nutrisi bahan utama {nutrisi_dict}, dan berikan nutrisi juga untuk bahan makanan pelengkap (jangan gunakan tabel)"
    
    # Generate content using prompt
    konten = prompt.generate_content(prompt_text)
    print(konten)

if __name__ == "__main__":
    main()
