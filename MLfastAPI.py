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
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was captured successfully, ret will be True
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the captured frame
        cv2.imshow("Camera Feed", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()