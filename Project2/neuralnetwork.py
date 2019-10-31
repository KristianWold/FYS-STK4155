import numpy as np
import numba as nb
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2


class Tanh():
    def __call__(self, x):
        return np.tanh(x)

    def deriv(self, x):
        return 1 + np.tanh(x)**2


class Relu():
    def __call__(self, x):
        return np.tanh(x)

    def deriv(self, x):
        return 1 + np.tanh(x)**2


class SoftMax():
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def deriv(self, x):
        return self(x) * (1 - self(x))


class Pass():
    def __call__(self, x):
        return x

    def deriv(self, x):
        return 1


class SquareLoss():
    def __call__(self, y_pred, y):
        return 0.5 * sum((y_pred - y)**2)

    def deriv(self, y_pred, y):
        return (y_pred - y)


class CrossEntropy():
    def __call__(self, y_pred, y):
        return -sum(y * np.log(y_pred) + (1 - y) * log(1 - y_pred))

    def deriv(self, y_pred, y):
        return (y_pred - y) / (y_pred * (1 - y_pred))


class NeuralNetwork():

    def __init__(self, dim, acf, cost):
        self.dim = dim
        self.acf = np.array(acf)
        self.cost = cost

        self.W = np.empty(len(dim) - 1, dtype=np.ndarray)  # weight matricies
        self.b = np.empty(len(dim) - 1, dtype=np.ndarray)  # biases
        self.z = np.empty(len(dim), dtype=np.ndarray)
        self.a = np.empty(len(dim), dtype=np.ndarray)
        self.grad = np.empty(len(dim) - 1, dtype=np.ndarray)
        self.delta = np.empty(len(dim) - 1, dtype=np.ndarray)

        for i in range(len(dim) - 1):
            m = dim[i + 1]
            n = dim[i]
            self.W[i] = np.random.normal(0, 1, (m, n))
            self.b[i] = 0.01 * np.ones(m)

    def forward(self, x):
        self.z[0] = x.T

        self.a[0] = x.T
        for i in range(len(self.W)):
            self.z[i + 1] = self.W[i]@self.a[i] + self.b[i][:, np.newaxis]
            self.a[i + 1] = self.acf[i](self.z[i + 1])

    def backward(self, x, y):
        self.forward(x)

        self.grad[-1] = self.acf[-1].deriv(self.z[-1]) * \
            self.cost.deriv(self.a[-1], y)

        for i in range(len(self.W) - 1, 0, -1):
            self.grad[i - 1] = self.W[i].T @ self.grad[i] * \
                self.acf[i].deriv(self.z[i])

    def train(self, X, y, mu):

        self.backward(X, y)

        for i in range(len(self.grad)):
            self.delta[i] = self.grad[i]@self.a[i].T

        self.W -= mu * self.delta
        for i in range(len(self.grad)):
            self.b[i] -= mu * np.sum(self.grad[i], axis=1)


tanh = Tanh()
sig = Sigmoid()
softMax = SoftMax()
crossEntropy = CrossEntropy()
squareLoss = SquareLoss()

data = load_digits()
enc = OneHotEncoder(categories='auto')

N = 100

y = enc.fit_transform(np.array(data.target[:2 * N]).reshape(-1, 1)).toarray()
x = np.array(data.data[:2 * N])

np.random.seed(42)
nn = NeuralNetwork((64, 30, 10), [tanh, softMax], squareLoss)

nn.forward(x)


for i in range(6000):
    nn.train(x[:N], y[:N].T, 0.002)
    if i % (3000 / 100) == 0:
        print(i * (100 / 3000))

success = 0

nn.forward(x)


for i in range(N, 2 * N):
    success += np.array_equal(np.round((nn.a)[-1][:, i]), y[i])

print(success)
