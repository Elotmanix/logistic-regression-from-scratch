import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split



class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = None
        self.c = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return self.sigmoid(np.dot(X, self.m) + self.c)

    def LossFunction(self, X, y):
        N = len(y)
        predictions = self.predict(X)
        loss = (-1/N) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def gradient_descent(self, X, y):
        N = len(y)
        self.m = np.zeros(X.shape[1])
        self.c = 0

        for i in range(self.iterations):
            predictions = self.predict(X)
            dm = (1/N) * np.dot(X.T, predictions - y)
            dc = (1/N) * np.sum(predictions - y)
            self.m = self.m - self.learning_rate * dm
            self.c = self.c - self.learning_rate * dc
            
            if i % 100 == 0:
                cost = self.LossFunction(X, y)
                print(f"Iteration {i}: Cost {cost}")

    def fit(self, X, y):
        self.gradient_descent(X, y)

    def classify(self, X):
        predictions = self.predict(X)
        return [1 if p >= 0.5 else 0 for p in predictions]
    
    def accuracy(self,y_true,y_pred):
        accuracy = np.sum(y_true == y_pred)/len(y_true)
        return accuracy
    

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



