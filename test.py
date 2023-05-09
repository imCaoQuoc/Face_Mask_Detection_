import cv2
import numpy as np
import tensorflow

model = tensorflow.keras.models.load_model("D:\Face_Mask_Detection_\model.h5", compile=False)

face_cascade = cv2.CascadeClassifier("D:\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml") # đường dẫn đến tệp XML của Haar Cascade
video_capture = cv2.VideoCapture(0) # mở camera
labels = {0: 'Không đeo khẩu trang', 1: 'Đeo khẩu trang'}

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using haarcascade
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the face region
            face = gray[y:y+h, x:x+w]
            
            # Resize the face image to the input size of the model
            face = cv2.resize(face, (224, 224))
            
            # Convert the face image to float32 and normalize it
            face = face.astype(np.float32) / 255.0
            
            # Add a batch dimension to the face image
            face = np.expand_dims(face, axis=0)
            
            # Predict the label of the face image
            pred = model.predict(face)[0]
            
            # Get the predicted label index and corresponding label text
            label_idx = np.argmax(pred)
            label_text = labels[label_idx]
            
            # Draw a rectangle around the face
            color = (0, 255, 0) if label_idx == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add label text above the rectangle
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# giải phóng bộ nhớ và đóng tất cả các cửa sổ
video_capture.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from tensorflow

# # Load pre-trained model
# model = tensorflow.keras.models.load_model('path/to/model.h5')

# # Load haarcascade classifier
# face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# # Define labels
# labels = {0: 'Không đeo khẩu trang', 1: 'Đeo khẩu trang'}

# # Open video file
# cap = cv2.VideoCapture('path/to/video.mp4')

# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()

#     if ret:
#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces using haarcascade
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             # Extract the face region
#             face = gray[y:y+h, x:x+w]
            
#             # Resize the face image to the input size of the model
#             face = cv2.resize(face, (224, 224))
            
#             # Convert the face image to float32 and normalize it
#             face = face.astype(np.float32) / 255.0
            
#             # Add a batch dimension to the face image
#             face = np.expand_dims(face, axis=0)
            
#             # Predict the label of the face image
#             pred = model.predict(face)[0]
            
#             # Get the predicted label index and corresponding label text
#             label_idx = np.argmax(pred)
#             label_text = labels[label_idx]
            
#             # Draw a rectangle around the face
#             color = (0, 255, 0) if label_idx == 1 else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
#             # Add label text above the rectangle
#             cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#         # Display the resulting frame
#         cv2.imshow('frame', frame)

#         # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # giải phóng bộ nhớ và đóng tất cả các cửa sổ
# video_capture.release()
# cv2.destroyAllWindows()
