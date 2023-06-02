# import cv2
# import streamlit as st
# import numpy as np
# import tensorflow
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# model = tensorflow.keras.models.load_model("D:\Face_Mask_Detection_\MobileNet.h5", compile=False)
# face_cascade = cv2.CascadeClassifier("D:\Face_Mask_Detection_\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml") # đường dẫn đến tệp XML của Haar Cascade
# video_capture = cv2.VideoCapture(0) # mở camera
# labels = {0: 'Mask', 1: 'NoMask'}
# if not video_capture.isOpened():
#     print ("Could not open cam")
#     exit()
# while True:
#     # Read a frame from the video
#     ret, frame = video_capture.read()
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (224, 224))
#         face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
#         face = np.reshape(face, [1, 224, 224, 3])/255.0
#         predict = model.predict(face)
#         prediction_index = np.argmax(predict, axis=-1)[0]
#         prediction_label = labels[prediction_index]
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#         #st.image(frame, channels="BGR", use_column_width=True)
#     cv2.imshow("face",frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # giải phóng bộ nhớ và đóng tất cả các cửa sổ
# video_capture.release()
# cv2.destroyAllWindows()


import cv2
import av
import tempfile
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# Đường dẫn tới mô hình và file cascade
MODEL_PATH = "D:\Face_Mask_Detection_\MobileNet.h5"
CASCADE_PATH = "D:\Face_Mask_Detection_\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml"
labels = {0: 'Mask', 1: 'NoMask'}

# Load mô hình và file cascade
model = tf.keras.models.load_model(MODEL_PATH, compile= False)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Use this line to capture video from the webcam
cap = cv2.VideoCapture(0)

# Set the title for the Streamlit app
st.title("Video Capture with OpenCV")

frame_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        st.write("The video capture has ended.")
        break

    # You can process the frame here if needed
    # e.g., apply filters, transformations, or object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # for (x, y, w, h) in faces:
    #     face = gray[y:y+h, x:x+w]
    #     face = cv2.resize(face, (224, 224))
    #     face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    #     face = np.reshape(face, [1, 224, 224, 3])/255.0
    #     predict = model.predict(face)
    #     prediction_index = np.argmax(predict, axis=-1)[0]
    #     prediction_label = labels[prediction_index]
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #     # Convert the frame from BGR to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Streamlit's st.image
    frame_placeholder.image(frame, channels="RGB")

    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
        break

cap.release()
cv2.destroyAllWindows()