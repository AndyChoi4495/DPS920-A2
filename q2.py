import glob
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore' , category=UserWarning)

# === DATA ===
features = []
labels = []


for i, address in enumerate(glob.glob("train\\*\\*.*")):
    img = cv2.imread(address)
    img = cv2.resize(img, (32, 32))
    img = img / 255
    img = img.flatten()

    features.append(img)
    labels.append(address.split('\\')[-2])  # [-2] is Dog or Cat

    if i % 200 == 0:
        print(f'[INFO] {i} images processed!')

features = np.array(features)
labels = np.array(labels)
print(labels)

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,  random_state=42)

# === MODEL ===

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, "catdog_knn_model.z")

# === Logistic Regression MODEL ===
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
joblib.dump(log_model, "catdog_logreg_model.z")


# === EVALUATE ===
knn_preds = knn_model.predict(X_test)
log_preds = log_model.predict(X_test)

knn_acc = accuracy_score(y_test, knn_preds)
log_acc = accuracy_score(y_test, log_preds)

print(f"KNN Accuracy:{knn_acc}")
print(f"Logistic Regression Accuracy:{log_acc}")

# === TEST ===
image = cv2.imread("test/Cat/Cat (5).jpg")  
image = cv2.resize(image, (32, 32))
image = image / 255
image = image.flatten()
print("KNN Prediction:             ", knn_model.predict([image])[0])
print("Logistic Regression Prediction:", log_model.predict([image])[0])
