import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from norfair import Detection, Tracker

interpreter = tf.lite.Interpreter(model_path="...")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(crop):
    resized = cv2.resize(crop, (128, 256))
    