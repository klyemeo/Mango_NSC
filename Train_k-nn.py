import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib




def load_dataset(base_path):
    data = []
    labels = []

    for label in ["ripe", "over-ripe"]:
        folder_path = os.path.join(base_path, label)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))  # Resize for consistency
            data.append(image)
            labels.append(0 if label == "over-ripe" else 1)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def extract_features(images):
    features = []
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    return np.array(features)

base_path = "D:/NSC/test"
data, labels = load_dataset(base_path)
features = extract_features(data)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(f'date :{data}',f'labels :{labels}\n')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

def plot_predictions(images, true_labels, pred_labels):
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'True: {"Ripe" if true_labels[i] == 1 else "over-ripe"}\nPred: {"Ripe" if pred_labels[i] == 1 else "over-ripe"}')
        plt.axis('off')
    plt.show()



joblib.dump(knn, 'knn_mango_ripeness_model.pkl')


