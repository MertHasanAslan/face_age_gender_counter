import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from norfair import Detection, Tracker


interpreter = tf.lite.Interpreter(model_path="reid_model.tflite")
interpreter.allocate_tensors() # seperate memory for model's input and output
input_details = interpreter.get_input_details #get input details (name, shape, dtype)
output_details = interpreter.get_output_details #get output details

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(image):
    image = cv2.resize(image, (128, 256)) #resize according to model's input size
    image = image / 255.0 #normalize 0-1 (normally it was 0-255)
    image = np.expand_dims(image.astype(np.float32), axis=0) #tfLite input is generally float32 (this create a tensor (1, 256, 128, 3))
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke() #run the model
    return interpreter.get_tensor(output_details[0]['index'])[0]



    