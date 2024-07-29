import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
height = np.array([60, 62, 65, 70, 72]).reshape(-1, 1)
weight = np.array([115, 120, 125, 140, 150])

# Fit the model
model = LinearRegression()
model.fit(height, weight)
predicted_weight = model.predict(height)

# Calculate residuals
residuals = weight - predicted_weight

# Residual plot
plt.scatter(predicted_weight, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Weight')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()