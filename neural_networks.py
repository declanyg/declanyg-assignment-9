import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        self.activation_fn = activation
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda z: 1 - np.tanh(z) ** 2
        elif activation == 'relu':
            self.activation = lambda z: np.maximum(0, z)
            self.activation_derivative = lambda z: (z > 0).astype(float)
        elif activation == "sigmoid":
            self.activation = lambda z: 1 / (1 + np.exp(-z))
            self.activation_derivative = lambda z: self.activation(z) * (1 - self.activation(z))
        else:
            raise ValueError("Unsupported activation function. Use 'tanh' or 'relu'.")

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.activation(self.Z2)
        # TODO: store activations for visualization
        out = self.A2
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = X.shape[0]
        dZ2 = (self.A2 - y) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # TODO: update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        gradients = [dW1, db1, dW2, db2]
        return gradients

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        gradients = mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.A1
    hidden_features = mlp.A1
    if hidden_features.shape[1] == 2:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Features (2D)")
        ax_hidden.set_xlabel("Hidden Feature 1")
        ax_hidden.set_ylabel("Hidden Feature 2")
    elif hidden_features.shape[1] == 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Features (3D)")
        ax_hidden.set_xlabel("Hidden Feature 1")
        ax_hidden.set_ylabel("Hidden Feature 2")
        ax_hidden.set_zlabel("Hidden Feature 3")
    # TODO: Hyperplane visualization in the hidden space
    if hidden_features.shape[1] == 3:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        zz = (-mlp.W2[0, 0] * xx - mlp.W2[1, 0] * yy - mlp.b2[0, 0]) / mlp.W2[2, 0]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.5, color='gray')
    # TODO: Distorted input space transformed by the hidden layer
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title("Distorted Input Space")
    ax_input.set_xlabel("Input Feature 1")
    ax_input.set_ylabel("Input Feature 2")

    # TODO: Plot input layer decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).ravel()
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=50, cmap='bwr', alpha=0.3)

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    magnitudes = [np.linalg.norm(g) for g in gradients]
    ax_gradient.set_title("Gradient Magnitudes")
    ax_gradient.set_xlabel("Layer")
    ax_gradient.set_ylabel("Magnitude")
    ax_gradient.bar(range(len(magnitudes)), magnitudes, color='blue')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)