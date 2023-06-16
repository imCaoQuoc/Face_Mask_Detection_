# Face Mask Classification using Haar Cascade and MobileNet V2

This project aims to develop a face mask classification system using Haar Cascade and MobileNet V2 deep learning models. The system can determine whether a person is wearing a face mask or not, based on the input image or video.

 ---

### **INTRODUCTION**
In the wake of the COVID-19 pandemic, face masks have become an essential part of our daily lives to prevent the spread of the virus. Automating the process of face mask detection can be crucial in ensuring compliance with safety guidelines in various settings such as public places, workplaces, and transportation.

This project leverages two powerful deep learning models: Haar Cascade and MobileNet V2. Haar Cascade is a machine learning-based approach that can detect faces in real-time by analyzing the features present in the image. MobileNet V2, on the other hand, is a lightweight convolutional neural network architecture known for its efficiency and accuracy in image classification tasks.

Technologies I used:
  - [Streamlit](https://streamlit.io/) to create a simple web demo.
  - [Streamlit_webrtc](https://pypi.org/project/streamlit-webrtc/) to load use real-time camera on Streamlit.
  - [Tensorflow](https://www.tensorflow.org/) to build a deep learning model.
  - [OpenCV](https://opencv.org/) doing tasks relate to Computer Vision.
  - [Sci-kit Learn](https://scikit-learn.org/stable/) to process data.
  - [Streamlit documentation](https://www.youtube.com/playlist?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE) to learn the basic of streamlit.

---

### **INSTALLATION**
I highly recommend you using Google Colab to run the TrafficSign.ipynb file because it already has backages and libraries I use. But if you want to run on your local machine, following the instruction below.
  - Install essential libraries and packages:
  
  ```
  pip install -r requirements.txt
  ```
  
  - Run demo:
  
  ```
  streamlit run TrafficApp.py
  ```

---

### **DATA INFORMATION** 

This data 43 classes, which stays for 43 types of traffic sign: 

Data has 51869 labeled images, which splitted into 39239 images for training and 12630 images for testing.

---

### **CONVOLUTIONAL NEURAL NETWORK**

Convolutional Neural Networks (CNNs) are a type of deep learning algorithm that have proven to be highly effective in image recognition, classification, and other computer vision tasks. They are inspired by the structure and function of the human visual system, and use a series of convolutional layers to automatically learn and extract features from input images. These features are then processed through a series of fully connected layers, which make predictions about the class of the input image. CNNs have achieved state-of-the-art results in a wide range of applications, including object recognition, facial recognition, and self-driving cars.

In this repository, I provide an example of how to build a CNNs model using TensorFlow library in Python. The model is training on an [image dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?ref=morioh.com&utm_source=morioh.com) and using to classify traffic signs.

---

### **DEMO**

#### Uploading image

You need to upload your image, it should be in PNG format. 

---

### **RESULT**

A demo will return a traffic sign's class.

![alt text](https://github.com/imCaoQuoc/TrafficSign_Classification/blob/main/Screenshot%202023-05-02%20171839.png)
![alt text](dataset/mask.png)
![alt text](dataset/NoMask.png)
