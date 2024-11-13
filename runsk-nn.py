import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

model_path = os.path.abspath('knn_mango_ripeness_model.pkl')
print(f'Model saved at: {model_path}')
# Load the trained model
knn = joblib.load('knn_mango_ripeness_model.pkl')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match training data
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)  # Reshape for prediction

#new_image_path = "D:/NSC/test/over-ripe/over-ripe.01.jpg"
new_image_path = "/home/nsc/Documents/NSC_model_Inw/test/images/D1.jpg"
new_image_features = preprocess_image(new_image_path)
# Make prediction
prediction = knn.predict(new_image_features)

predicted_label = 'ripe' if prediction[0] == 1 else 'over-ripe'
print(f'The predicted ripeness of the mango is: {predicted_label}')
