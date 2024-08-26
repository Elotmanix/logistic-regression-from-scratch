# Logistic Regression from Scratch

This project demonstrates the implementation of a Logistic Regression model from scratch using Python and NumPy. The goal is to provide a deeper understanding of how Logistic Regression works by manually coding the algorithm rather than relying on machine learning libraries.

## Project Structure

- **logistic_regression.py**: Contains the `LogisticRegressionFromScratch` class, which implements the logistic regression model, including methods for fitting the model, predicting, and evaluating accuracy.

- **implementation.py**: Demonstrates how to use the `LogisticRegressionFromScratch` class. It includes generating synthetic data, training the model, and evaluating its performance.

## LogisticRegressionFromScratch Class

### Methods:
- **`__init__(learning_rate=0.01, iterations=1000)`**: Initializes the model with the specified learning rate and number of iterations.
- **`sigmoid(z)`**: Computes the sigmoid function.
- **`predict(X)`**: Predicts the probability of the positive class.
- **`LossFunction(X, y)`**: Computes the binary cross-entropy loss.
- **`gradient_descent(X, y)`**: Optimizes the model parameters using gradient descent.
- **`fit(X, y)`**: Fits the model to the training data.
- **`classify(X)`**: Classifies the input data based on the learned parameters.
- **`accuracy(y_true, y_pred)`**: Calculates the accuracy of the model.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Elotmanix/logistic-regression-from-scratch.git
   cd logistic-regression-from-scratch
2. **Run the Implementation:**
   ```bash
   python implementation.py
3.**Expected Output:**
- The model will be trained on synthetic data.
- The final model parameters (weights and bias) will be printed.
- The accuracy of the model on the test set will be displayed.
## Dependencies
- Python 3.7
- NumPy
- scikit-learn

Install the dependencies using pip:
```bash
pip install numpy scikit-learn

