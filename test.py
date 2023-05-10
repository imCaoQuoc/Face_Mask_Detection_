import cv2
import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model("D:\Face_Mask_Detection_\model_face_mask_detection.h5", compile=False)

face_cascade = cv2.CascadeClassifier("D:\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml") # đường dẫn đến tệp XML của Haar Cascade
video_capture = cv2.VideoCapture(0) # mở camera
labels = {0: 'Không đeo khẩu trang', 1: 'Đeo khẩu trang'}
if not video_capture.isOpened():
    print ("Could not open cam")
    exit()
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (150, 150))
    img_color = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
    img_reshape = np.reshape(img_color, (1, 150, 150, 3))
    print(gray.shape)
    print(gray_resized.shape)
    print(img_color.shape)
    print(img_reshape.shape)
    predict = model.predict(img_reshape)
    print(predict)
    faces = face_cascade.detectMultiScale(gray_resized)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 200), 2)
    cv2.imshow("face",frame)            
        #     # Add a batch dimension to the face image
        #     face = np.expand_dims(face, axis=0)
            
        #     # Predict the label of the face image
        #     pred = model.predict(face)[0]
            
        #     # Get the predicted label index and corresponding label text
        #     label_idx = np.argmax(pred)
        #     label_text = labels[label_idx]
            
        #     # Draw a rectangle around the face
        #     color = (0, 255, 0) if label_idx == 1 else (0, 0, 255)
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
        #     # Add label text above the rectangle
        #     cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the resulting frame
        #cv2.imshow('frame', frame)

        # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# giải phóng bộ nhớ và đóng tất cả các cửa sổ
video_capture.release()
cv2.destroyAllWindows()