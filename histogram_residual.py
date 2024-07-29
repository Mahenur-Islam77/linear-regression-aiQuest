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
# Histogram of residuals
plt.hist(residuals, bins=5)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
