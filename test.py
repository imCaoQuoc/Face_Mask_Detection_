import cv2
from mtcnn import MTCNN
import numpy as np
import tensorflow
from scipy.spatial import distance

model = tensorflow.keras.models.load_model("D:\Face_Mask_Detection_\VGG19.h5", compile=False)
face_cascade = cv2.CascadeClassifier("D:\Face_Mask_Detection_\haarcascades\haarcascade_frontalface_default.xml") # đường dẫn đến tệp XML của Haar Cascade
MIN_DISTANCE = 130
video_capture = cv2.VideoCapture(0) # mở camera

mask_label = {0:'MASK',1:'NO MASK'}
dist_label = {0:(0,255,0),1:(255,0,0)}

if not video_capture.isOpened():
    print ("Could not open cam")
    exit()

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_resized = cv2.resize(gray, (128, 128))
    # img_color = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
    # img_reshape = np.reshape(img_color, (1, 128, 128, 3))/255.0

    # predict = model.predict(img_reshape)
    # print(predict)
    
    # prediction_index = np.argmax(predict, axis=-1)[0]
    # prediction_label = labels[prediction_index]
    # #   results = detector.detect_faces(frame)
    # print(prediction_label)

    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
    new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        crop = new_img[y:y+h,x:x+w]
        crop = cv2.resize(crop,(128,128))
        crop = np.reshape(crop,[1,128,128,3])/255.0
        mask_result = model.predict(crop)
        cv2.rectangle(frame,(x,y),(x+w,y+h),dist_label[label[i]],2)
        cv2.putText(frame,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,dist_label[label[i]],2)
    cv2.imshow("face",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# giải phóng bộ nhớ và đóng tất cả các cửa sổ
video_capture.release()
cv2.destroyAllWindows()