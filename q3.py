import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# === DATA ===
df = pd.read_csv("mnist_train.csv")
#print(df.columns)

y = df.iloc[:, 0].values # Frist column = label
X = df.iloc[:, 1:].values # pixels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 0.2 test , 0.8 train. 
# if test size is too high - not enough data to train.  if test size is too low - not reliable. 
# ramdom - get the same split every time you run the code.


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === LOGISTIC REGRESSION ===
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_preds)
print(f"Logistic Regression Accuracy: {log_acc}")

# === K-NEAREST NEIGHBORS ===
knn_model = KNeighborsClassifier() 
knn_model.fit(X_train, y_train)  
knn_preds = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_preds)
print(f"KNN Accuracy: {knn_acc}")
