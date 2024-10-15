import serial
import time
from detectsent import *
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Use video file or webcam
use_webcam = False  # Set to True to use the webcam
video_path = "C:/Users/ASUS TUF FX505DT/Pictures/Camera Roll/WIN_20240726_14_01_53_Pro.mp4"  # Update this with the path to your video file
weights= "D:/NSC/NSC_model_Inw/yolov5/runs/train/exp2/weights/best.pt"
data = "D:/NSC/NSC_model_Inw/yolov5/data/coco.yaml"
knn = joblib.load("D:/NSC/NSC_model_Inw/yolov5/knn_mango_ripeness_model.pkl")

# Define the reference lines (x-coordinates relative to the ROI)
reference_lines = [550]  # Update these values based on your setup

# Define the region of interest (ROI)
top_left = (400, 175)
bottom_right = (1200, 825)
cooldown_frames = 30
# Object counter
count = 0

# Replace 'COM3' with the port your Arduino is connected to
arduino_port = 'COM5'
baud_rate = 115200 # Match this with the baud rate in your Arduino code

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match training data
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)  # Reshape for prediction

def predict_ripeness(image_path):
    if os.path.isfile(image_path):  # Check if it is a file
        new_image_features = preprocess_image(image_path)
        # Make prediction
        prediction = knn.predict(new_image_features)
        predicted_label = 'ripe' if prediction[0] == 1 else 'over-ripe'
        ser.write(f'{predicted_label}'.encode('utf-8'))
        #ser.write(f"{x}\n".encode('utf-8'))
    else:
        print(f'{image_path} is not a valid file path.')

# Directory where images will be saved
save_directory = 'D:/NSC/captured_images/'  # Update this with your desired directory path
# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)

# Define the HSV color range for detection
lower_hsv = np.array([0, 116, 82])  # Update with your desired lower HSV range
upper_hsv = np.array([179, 255, 255])  # Update with your desired upper HSV range

# Dictionary to store the objects' last known positions
object_positions = {}
object_cooldowns = {}



try:
    # Initialize serial communication
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2)  # Wait for the connection to establish

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the ROI
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Apply the HSV mask
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the ROI for drawing the reference lines
        roi_with_lines = roi.copy()

        # Draw the reference lines on the copied ROI
        for line_x in reference_lines:
            cv2.line(roi_with_lines, (line_x, 0), (line_x, roi.shape[0]), (255, 0, 0), 2)

        current_positions = {}
        for contour in contours:
            if cv2.contourArea(contour) < 7000:  # Filter out small contours
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            object_center_x = x + w / 2
            object_right_edge_x = x + w

            # Draw bounding box on the ROI with the reference lines
            cv2.rectangle(roi_with_lines, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Track the object
            object_id = len(current_positions) + 1  # Simple object ID based on count
            current_positions[object_id] = object_center_x

            # Check if the object completely crosses the third reference line
            third_line_x = reference_lines[0]
            if object_id in object_positions and object_id not in object_cooldowns:
                if (object_positions[object_id] < third_line_x) and (object_right_edge_x > third_line_x):
                    # Object has completely crossed the third reference line
                    count += 1
                    # Capture the frame with the object from the original ROI (without the reference lines)
                    object_img = roi
                    # Define the file path for saving
                    file_path = os.path.join(save_directory, f'object_{count}.png')
                    cv2.imwrite(file_path, object_img)
                    source = f'D:/NSC/captured_images/object_{count}.png' 
                    #run(weights, source, data)
                    predict_ripeness(source)                    
                    object_cooldowns[object_id] = cooldown_frames
            # Update the position of the object
            object_positions[object_id] = object_center_x

        # Update the last positions of the objects
        object_positions = current_positions
        for object_id in list(object_cooldowns.keys()):
            object_cooldowns[object_id] -= 1
            if object_cooldowns[object_id] <= 0:
                del object_cooldowns[object_id]

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ser.closes()
            break


except Exception as e:
    print("Error:", e)


