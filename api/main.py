from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from .model import predict_image

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_path)
    os.remove(temp_path)
    return JSONResponse(content=result)

