import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("house_price.csv")


X = df[['size', 'bedroom']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# === Linear Regression  ===
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Linear Regression:")
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)

# === SGD Regressor  ===
sgd_model = SGDRegressor(max_iter=1000)
sgd_model.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_model.predict(X_test_scaled)

mae_sgd = mean_absolute_error(y_test, y_pred_sgd)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
mape_sgd = mean_absolute_percentage_error(y_test, y_pred_sgd)

print("\nSGD Regressor:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
print("MAE:", mae_sgd)
print("MSE:", mse_sgd)
print("RMSE:", rmse_sgd)
print("MAPE:", mape_sgd)
