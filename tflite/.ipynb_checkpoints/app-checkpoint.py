from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model import Predict, Prompt
import shutil
import os

app = FastAPI()

# Initialize model and classes
label = [
    "Kentang", "Tomat", "Timun jepang", "Bawang bombay", "Lobak", "Terong", 
    "Daun bawang", "Kol ungu", "Wortel", "Telur", "Daging sapi", "Pork", "Daging ayam"
]
predictor = Predict(r'E:\CapstoneProject\ModelML\model_alex\model_alex\tflite\mobilenet_model.tflite', label)
prompt = Prompt(r'E:\CapstoneProject\ModelML\model_alex\model_alex\combined_nutrition_data.csv')

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan dengan origin yang sesuai dengan aplikasi Android Anda
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Save the uploaded file
    img_path = f"temp_{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Predict class
        predicted_class = predictor.predict_image(img_path, (224, 224))
        
        # Add predicted class to the list of ingredients
        list_makanan = ["ayam", "Abon", "Bakwan", "Bawang Merah"]
        list_makanan.append(predicted_class.lower())

        # Get nutrition info
        nutrisi_dict = prompt.get_nutrition_info(list_makanan)

        # Generate content
        prompt_text = (
            f"Saya punya bahan utama makanan berikut: {', '.join(list_makanan)}. "
            f"Tolong rekomendasikan resep makanan untuk ibu hamil dan tampilan nutrisi bahan utama {nutrisi_dict}, "
            f"dan berikan nutrisi juga untuk bahan makanan pelengkap (jangan gunakan tabel)"
        )
        konten = prompt.generate_content(prompt_text)

        return JSONResponse(content={"predicted_class": predicted_class, "nutrition_info": nutrisi_dict, "content": konten})
    finally:
        # Clean up the saved file
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
