import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Sample data
X = np.arange(100).reshape((100, 1))
y = 2 * X.flatten() + 3 + np.random.randn(100) * 10  # Linear relationship with noise

# Linear regression model
model = LinearRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

print("Cross-validation scores (negative MSE):", scores)
print("Mean cross-validation score:", np.mean(scores))
