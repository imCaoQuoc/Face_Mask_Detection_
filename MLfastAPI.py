from fastapi import FastAPI
import cv2
import numpy as np
import tensorflow

app = FastAPI()
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = tensorflow.keras.models.load_model("MobileNet.h5", compile=False)
labels = {0: 'Mask', 1: 'NoMask'}

@app.get("/")
def home():
    return {"message": "Welcome to Face Mask Detection system"}

@app.get("/detect")
def detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    