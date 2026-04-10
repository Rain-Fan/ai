#1.单变量

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. Generate dataset
# =========================
np.random.seed(42)
n = 200

area = np.random.uniform(40, 200, n)  # Area in m2
age = np.random.uniform(0, 30, n)  # Age in years
bedrooms = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
floor = np.random.randint(1, 30, n)  # Floor number

# True relationship (with some noise)
price = 2.1 * area - 3.5 * age + 15 * bedrooms + 0.8 * floor + np.random.normal(0, 20, n)

df = pd.DataFrame({
    'area': np.round(area, 1),
    'age': np.round(age, 1),
    'bedrooms': bedrooms,
    'floor': floor,
    'price': np.round(price, 1)
})

print(f"Dataset: {df.shape[0]} houses, {df.shape[1]} features")
print(df.head())

# =========================
# 2. Visualize area vs price
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(df['area'], df['price'], alpha=0.5, edgecolors='white', linewidth=0.5)
plt.xlabel('Area (m2)')
plt.ylabel('Price (10k yuan)')
plt.title('House Prices vs Area')
plt.show()

# =========================
# 3. Extract single feature and target
# =========================
x = df['area'].values
y = df['price'].values

print(f"x (area):  first 5 values = {x[:5]}")
print(f"y (price): first 5 values = {y[:5]}")

# =========================
# 4. Linear Regression from scratch using Gradient Descent
# =========================

# Step 1: Normalize the data
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std

# Step 2: Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
n_iterations = 100
n_samples = len(x)

# Step 3: Gradient descent loop
losses = []
for iteration in range(n_iterations):
    # Forward pass
    y_pred = w * x_norm + b

    # Loss (MSE)
    loss = np.mean((y_pred - y_norm) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2 / n_samples) * np.sum((y_pred - y_norm) * x_norm)
    db = (2 / n_samples) * np.sum(y_pred - y_norm)

    # Update
    w -= learning_rate * dw
    b -= learning_rate * db

    if iteration % 20 == 0:
        print(f"Iteration {iteration:3d}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.6f}")

print(f"\nFinal (normalized): w = {w:.4f}, b = {b:.4f}")

# Step 4: Convert back to original scale
w_original = w * y_std / x_std
b_original = y_mean + w * y_std * (-x_mean / x_std) + b * y_std

print(f"\nOur model: price = {w_original:.3f} * area + {b_original:.3f}")
print(f"Interpretation: each additional m2 adds {w_original:.1f} (x10k yuan) to the price")

# =========================
# 5. Plot fitted line
# =========================
y_pred_all = w_original * x + b_original

plt.figure(figsize=(9, 6))
plt.scatter(x, y, alpha=0.4, label='Actual data', edgecolors='white', linewidth=0.5)
plt.plot(sorted(x), w_original * np.sort(x) + b_original, 'r-', linewidth=2,
         label=f'GD model: y = {w_original:.2f}x + {b_original:.2f}')
plt.xlabel('Area (m2)')
plt.ylabel('Price (10k yuan)')
plt.title('Linear Regression via Gradient Descent')
plt.legend()
plt.show()

# =========================
# 6. Evaluate model
# =========================
mse = np.mean((y_pred_all - y) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y - y_pred_all) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f} (average error in 10k yuan)")
print(f"R2:   {r2:.4f} (model explains {r2 * 100:.1f}% of the variance)")

# =========================
# 7. Predict a new house
# =========================
new_area = 100
predicted_price = w_original * new_area + b_original
print(f"A {new_area}m2 house → predicted price: {predicted_price:.1f} (10k yuan)")
print(f"That's about {predicted_price / 100:.2f} million yuan")

# =========================
# 8. Verify with scikit-learn
# =========================
X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

print(f"Our GD w:      {w_original:.6f}")
print(f"Sklearn w:     {model.coef_[0]:.6f}")
print(f"\nOur GD b:      {b_original:.6f}")
print(f"Sklearn b:     {model.intercept_:.6f}")
print(f"\nClose enough: {np.allclose(w_original, model.coef_[0], atol=0.01)}")
print(f"\nGradient descent found (almost) the same answer as the exact solution!")

# =========================
# 9. Plot loss curve
# =========================
plt.figure(figsize=(8, 5))
plt.plot(losses, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curve')
plt.show()












#1.精简版
import numpy as np
import pandas as pd

# data
np.random.seed(42)
n = 200
area = np.random.uniform(40, 200, n)
age = np.random.uniform(0, 30, n)
bedrooms = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
floor = np.random.randint(1, 30, n)
price = 2.1 * area - 3.5 * age + 15 * bedrooms + 0.8 * floor + np.random.normal(0, 20, n)

df = pd.DataFrame({
    'area': np.round(area, 1),
    'price': np.round(price, 1)
})

x = df['area'].values
y = df['price'].values

# normalize
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
x_norm = (x - x_mean) / x_std
y_norm = (y - y_mean) / y_std

# initialize
w, b = 0.0, 0.0
learning_rate = 0.1
n_iterations = 100
n_samples = len(x)

# gradient descent
for iteration in range(n_iterations):
    y_pred = w * x_norm + b
    dw = (2 / n_samples) * np.sum((y_pred - y_norm) * x_norm)
    db = (2 / n_samples) * np.sum(y_pred - y_norm)
    w -= learning_rate * dw
    b -= learning_rate * db

# convert back
w_original = w * y_std / x_std
b_original = y_mean + w * y_std * (-x_mean / x_std) + b * y_std

print(f"price = {w_original:.3f} * area + {b_original:.3f}")












#2.多元线性回归
from sklearn.linear_model import LinearRegression
import numpy as np

# Use all 4 features
X_multi = df[['area', 'age', 'bedrooms', 'floor']].values
y = df['price'].values

# Fit model
multi_model = LinearRegression()
multi_model.fit(X_multi, y)

# Print coefficients
feature_names = ['area', 'age', 'bedrooms', 'floor']
print("Coefficients (how much each feature affects price):")
for name, coef in zip(feature_names, multi_model.coef_):
    direction = 'increases' if coef > 0 else 'decreases'
    print(f"  {name:>10s}: {coef:+.3f}  (each unit {direction} price by {abs(coef):.1f})")

print(f"  {'intercept':>10s}: {multi_model.intercept_:.3f}")

# Evaluate
y_pred_multi = multi_model.predict(X_multi)
r2_multi = 1 - np.sum((y - y_pred_multi) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"\nR2 with 4 features (all): {r2_multi:.4f}")