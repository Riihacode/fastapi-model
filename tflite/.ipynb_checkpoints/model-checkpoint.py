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
        combined_df = pd.read_csv(self.nutrition_file_path, index_col=0)
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
        genai.configure(api_key='AIzaSyDk9BblI-kPgGHGZxaWGOBzeYG9KOzu4UU')
        model_gem = genai.GenerativeModel('gemini-pro')
        response = model_gem.generate_content(prompt)
        return response.text

class Predict:
    def __init__(self, model_path, item_names):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.item_names = item_names

    def predict_image(self, img_path, target_size):
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        class_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_id = np.argmax(class_pred)
        predicted_class = self.item_names[class_id]
        return predicted_class
