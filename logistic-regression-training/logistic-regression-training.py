import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):

    W = np.random.randn(X.shape[1])
    print(W)
    n = len(X)
    bias = 0
    b=0
    
    for i in range(steps):
        y_pred = _sigmoid(X @ W + bias)
        grad_W = (1/n)* X.T @ (y_pred - y)
        grad_b = np.mean(y_pred - y)
        W -= lr * grad_W
        bias -= lr * grad_b

    return W, bias
        