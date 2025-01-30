import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of the tanh function."""
    return 1 - np.tanh(x) ** 2

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of the Leaky ReLU function."""
    return np.where(x > 0, 1, alpha)

def softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def softmax_derivative(x):
    """Derivative of the softmax function.
    Note: The softmax derivative is more complex in practice, requiring the Jacobian matrix."""
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

def linear(x):
    """Linear activation function."""
    return x

def linear_derivative(x):
    """Derivative of the linear function."""
    return np.ones_like(x)

def swish(x):
    """Swish activation function."""
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivative of the swish function."""
    s = sigmoid(x)
    return s + x * s * (1 - s)

def elu(x, alpha=1.0):
    """ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivative of the ELU function."""
    return np.where(x > 0, 1, alpha * np.exp(x))

def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x):
    """Derivative of the GELU function."""
    tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
    return 0.5 * (1 + tanh_term) + 0.5 * x * (1 - tanh_term**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * np.power(x, 2))

# Example usage:
if __name__ == "__main__":
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Softmax:", softmax(x))

