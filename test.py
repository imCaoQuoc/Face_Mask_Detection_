import cv2
from mtcnn import MTCNN
import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model("D:\Face_Mask_Detection_\VGG19.h5", compile=False)

face_cascade = cv2.CascadeClassifier("D:\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml") # đường dẫn đến tệp XML của Haar Cascade
#detector = MTCNN()
video_capture = cv2.VideoCapture(0) # mở camera
labels = {0: 'Mask', 1: 'NoMask'}
if not video_capture.isOpened():
    print ("Could not open cam")
    exit()
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (128, 128))
    img_color = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
    img_reshape = np.reshape(img_color, (1, 128, 128, 3))
    predict = model.predict(img_reshape)
    print(predict)
    
    prediction_index = np.argmax(predict, axis=-1)[0]
    prediction_label = labels[prediction_index]
    #   results = detector.detect_faces(frame)
    print(prediction_label)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # for result in results:
    #     x, y, w, h = result['box']
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("face",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# giải phóng bộ nhớ và đóng tất cả các cửa sổ
video_capture.release()
cv2.destroyAllWindows()