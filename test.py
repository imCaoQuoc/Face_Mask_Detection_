from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import os
import cv2
import numpy as np
import tensorflow

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)
token = client.tokens.create()

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = tensorflow.keras.models.load_model("MobileNet.h5", compile=False)
labels = {0: 'Mask', 1: 'NoMask'}

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")
		gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
		faces = cascade.detectMultiScale(gray, 1.1, 5)

		for x,y,w,h in faces:
			face = gray[y:y+h, x:x+w]
			face = cv2.resize(face, (224, 224))
			face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
			face = np.reshape(face, [1, 224, 224, 3])/255.0
			predict = model.predict(face)
			prediction_index = np.argmax(predict, axis=-1)[0]
			prediction_label = labels[prediction_index]	
			cv2.rectangle(frm, (x,y), (x+w, y+h), (0, 255, 0), 2)
			cv2.putText(frm, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')

# webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
# 				rtc_configuration=RTCConfiguration(
# 					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# 					)
# 	)

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": token.ice_servers}
					)
	)