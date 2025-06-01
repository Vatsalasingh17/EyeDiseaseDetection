from PIL import Image
import numpy as np
import cv2

def preprocess_image(image_path, size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # normalize
    return np.expand_dims(image, axis=0)

