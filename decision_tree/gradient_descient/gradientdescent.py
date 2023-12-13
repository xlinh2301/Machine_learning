import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def compute_cost(X, Y, w, b):
    m = len(Y)
    total_cost = 0
    for i in range(m):
        prediction = sum(w[j] * X[i][j] for j in range(len(w))) + b
        total_cost += (prediction - Y[i]) ** 2
    return total_cost / (2 * m)

def gradient_descent(X, Y, w, b, alpha, iterations):
    m = len(Y)
    num_features = len(w)
    cost_history = []  

    for _ in range(iterations):
        delta_w = [0] * num_features
        delta_b = 0
        for i in range(m):
            prediction = sum(w[j] * X[i][j] for j in range(len(w))) + b
            delta_b += (prediction - Y[i])
            for j in range(num_features):
                delta_w[j] += (prediction - Y[i]) * X[i][j]

        b -= delta_b * alpha / m
        for j in range(num_features):
            w[j] = w[j] - (delta_w[j] * alpha / m)

        cost = compute_cost(X, Y, w, b)
        cost_history.append(cost)

    return w, b, cost_history

df = pd.read_csv(r'C:\Users\Admin\Downloads\archive\Real estate.csv')

cut = ['No','X1 transaction date']
predict = 'Y house price of unit area'

X = df.drop(cut, axis=1).values
Y = df[predict].values

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y.reshape(-1, 1))


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1)

# Using scikit-learn's LinearRegression
linear = linear_model.LinearRegression()
linear.fit(X_train, Y_train)

alpha = 0.1  
iterations = 1000 

initial_w = np.zeros(X_train.shape[1])
initial_b = 0

w, b, cost_history = gradient_descent(X_train, Y_train, initial_w, initial_b, alpha, iterations)
print("w =", w)
print("b =", b)
print("Cost after training:", compute_cost(X_train, Y_train, w, b))


plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Decrease Over Iterations")

# Hiển thị các điểm dữ liệu
plt.scatter(range(len(Y_train)), Y_train, c='r', label='Actual')
plt.scatter(range(len(Y_train)), linear.predict(X_train), c='g', label='Predicted (Scikit-Learn)')
plt.scatter(range(len(Y_train)), [sum(w[j] * x[j] for j in range(len(w))) + b for x in X_train], c='b', label='Predicted (Gradient Descent)')

plt.legend()    
plt.show()

# Dự đoán trên dữ liệu kiểm tra
predictions = linear.predict(X_test)  # Dự đoán bằng mô hình Scikit-Learn
# Dự đoán bằng mô hình Gradient Descent
predictions_gradient_descent = [sum(w[j] * x[j] for j in range(len(w))) + b for x in X_test]

# Tính toán sai số
from sklearn.metrics import mean_squared_error, r2_score

mse_scikit_learn = mean_squared_error(Y_test, predictions)
mse_gradient_descent = mean_squared_error(Y_test, predictions_gradient_descent)

r2_scikit_learn = r2_score(Y_test, predictions)
r2_gradient_descent = r2_score(Y_test, predictions_gradient_descent)

print("Mean Squared Error (Scikit-Learn):", mse_scikit_learn)
print("Mean Squared Error (Gradient Descent):", mse_gradient_descent)

print("R-squared (Scikit-Learn):", r2_scikit_learn)
print("R-squared (Gradient Descent):", r2_gradient_descent)

# Hiển thị biểu đồ so sánh
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(Y_test, predictions, c='b')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scikit-Learn Linear Regression")

plt.subplot(1, 2, 2)
plt.scatter(Y_test, predictions_gradient_descent, c='r')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Gradient Descent Linear Regression")

plt.show()
