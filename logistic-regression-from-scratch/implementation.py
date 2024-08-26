import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionFromScratch

# Usage example:
# Generate synthetic data with one feature
X, y = datasets.make_classification(n_samples=600, n_features=5, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# Create the Logistic Regression model
model = LogisticRegressionFromScratch(learning_rate=0.1, iterations=1000)

# Fit the model on the data
model.fit(X_train, y_train)

# Output final parameters
print(f"Final parameters: m = {model.m}, c = {model.c}")

# Predict and classify the data
predictions = model.classify(X_test)

print("Accuracy: ",model.accuracy(y_test, predictions))