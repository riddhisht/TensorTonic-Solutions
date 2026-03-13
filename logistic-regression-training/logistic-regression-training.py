import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    W = np.random.randn(X.shape[1])
    bias = 0
    n = len(X)
    for i in range(steps):
        y_pred = _sigmoid(X @ W + bias)
        gradW = (1/n)* X.T @ (y_pred-y)
        gradB = np.mean(y_pred-y)
        W -= lr * gradW
        bias -=lr*gradB

    return W, bias