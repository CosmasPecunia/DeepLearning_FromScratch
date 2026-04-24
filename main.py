import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class Model:
    def __init__(self):
        self.w1 = np.random.randn(2, 16) *0.1
        self.b1 = np.zeros(16)

        self.w2 = np.random.randn(16, 4) *0.1
        self.b2 = np.zeros(4)

        self.w3 = np.random.randn(4, 1) *0.1
        self.b3 = np.zeros(1)

        self.v_w1, self.v_b1 = np.zeros_like(self.w1), np.zeros_like(self.b1)
        self.v_w2, self.v_b2 = np.zeros_like(self.w2), np.zeros_like(self.b2)
        self.v_w3, self.v_b3 = np.zeros_like(self.w3), np.zeros_like(self.b3)

    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, y_pred, y): 
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def feed_forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = self.sigmoide(self.z3)
        return self.a3

    def backward(self, x, y):
        m = x.shape[0]
        lr = 0.1
        beta = 0.9
     
        dz3 = self.a3 - y
        dw3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0) / m
       
        da2 = dz3 @ self.w3.T
        dz2 = da2 * (self.z2 > 0)
        dw2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        dw1 = (x.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        self.v_w1 = beta * self.v_w1 + (1 - beta) * dw1
        self.v_b1 = beta * self.v_b1 + (1 - beta) * db1
        self.w1 -= lr * self.v_w1
        self.b1 -= lr * self.v_b1

        self.v_w2 = beta * self.v_w2 + (1 - beta) * dw2
        self.v_b2 = beta * self.v_b2 + (1 - beta) * db2
        self.w2 -= lr * self.v_w2
        self.b2 -= lr * self.v_b2

        self.v_w3 = beta * self.v_w3 + (1 - beta) * dw3
        self.v_b3 = beta * self.v_b3 + (1 - beta) * db3
        self.w3 -= lr * self.v_w3
        self.b3 -= lr * self.v_b3

def generate_circle_data(n_samples=1000):

    r1 = np.random.randn(n_samples//2, 1) * 0.2
    theta1 = np.random.rand(n_samples//2, 1) * 2 * np.pi
    X1 = np.hstack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.zeros((n_samples//2, 1))

    r2 = np.random.randn(n_samples//2, 1) * 0.2 + 1.0
    theta2 = np.random.rand(n_samples//2, 1) * 2 * np.pi
    X2 = np.hstack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.ones((n_samples//2, 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

def visualisation(X, y):

    plt.figure(figsize=(8, 8)) 
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='Spectral', edgecolors='k', alpha=0.7)
    plt.title("Visualisation du Dataset")
    plt.xlabel("Caractéristique X1 (Standardisée)")
    plt.ylabel("Caractéristique X2 (Standardisée)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.show()

def visualisation_after_training(model, X, y):
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.05  # Pas de la grille
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.feed_forward(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    plt.title("Surface de Décision de l'IA")
    plt.show()

X, y = generate_circle_data()


visualisation(X, y)
X = (X - X.mean(axis=0)) / X.std(axis=0)


visualisation(X, y)

model= Model()
batch_size = 32
n_samples = X.shape[0]

for epoch in range(1001):
    epoch_loss = 0
    indices = np.random.permutation(n_samples)
    x_shuffle = X[indices]
    y_shuffle = y[indices]

    for i in range(0, n_samples, batch_size):
        x_batch = x_shuffle[i: i+batch_size]
        y_batch = y_shuffle[i: i+batch_size]

        y_pred = model.feed_forward(x_batch)
        epoch_loss += model.loss(y_pred, y_batch)
        model.backward(x_batch, y_batch)

    if epoch % 100 == 0:
        print(f"Époque {epoch} | Loss moyenne: {epoch_loss / (n_samples/batch_size):.4f}")

visualisation_after_training(model, X, y)
