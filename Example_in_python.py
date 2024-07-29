import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
height = np.array([60, 62, 65, 70, 72]).reshape(-1, 1)
weight = np.array([115, 120, 125, 140, 150])

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(height, weight)

# Predict weights based on heights
weight_pred = model.predict(height)

# Plot the data and the regression line
plt.scatter(height, weight, color='blue')
plt.plot(height, weight_pred, color='red')
plt.title('Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Slope: {model.coef_[0]}')
