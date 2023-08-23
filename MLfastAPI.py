from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import tensorflow

app = FastAPI()
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = tensorflow.keras.models.load_model("MobileNet.h5", compile=False)
labels = {0: 'Mask', 1: 'NoMask'}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)

@app.get("/detect")
def detect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        #frm = frame.to_array(format="bgr24")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)

        for x,y,w,h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            face = np.reshape(face, [1, 224, 224, 3])/255.0
            predict = model.predict(face)
            prediction_index = np.argmax(predict, axis=-1)[0]
            prediction_label = labels[prediction_index]	
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the captured frame
        cv2.imshow("Camera Feed", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()