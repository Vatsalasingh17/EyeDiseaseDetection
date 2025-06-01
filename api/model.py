import numpy as np
import tensorflow as tf
from .utils import preprocess_image

model = tf.keras.models.load_model("models/eye_disease_model.h5")
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def predict_image(file_path):
    img_array = preprocess_image(file_path)
    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    return {
        "prediction": class_names[class_idx],
        "confidence": float(preds[class_idx])
    }

