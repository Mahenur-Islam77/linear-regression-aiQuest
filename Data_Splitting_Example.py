import numpy as np
from sklearn.model_selection import train_test_split

# Sample data
X = np.arange(10).reshape((10, 1))
y = np.arange(10)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set:")
print(X_train)
print(y_train)

print("\nTest set:")
print(X_test)
print(y_test)
