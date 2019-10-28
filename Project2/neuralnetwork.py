import numpy as np
import numba as nb


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


class SoftMax():
    def __call__(self, x):
        return np.exp(x) / sum(np.exp(x))

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


"""
dense1 = linear(50, 100)
out = dense1(in)
dense2 = linear(100, 1000)
out = dense2(out)
"""


def forward(self, x):
    self.z[0] = x.T
    self.a[0] = x.T

    for i in range(len(self.W)):
        self.z[i + 1] = self.W[i]@self.a[i] + self.b[i]
        self.a[i + 1] = self.acf[i](self.z[i + 1])


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
            self.z[i + 1] = self.W[i]@self.a[i] + self.b[i]
            self.a[i + 1] = self.acf[i](self.z[i + 1])

    def backward(self, x, y):
        self.forward(x)

        self.grad[-1] = self.acf[-1].deriv(self.z[-1]) * \
            self.cost.deriv(self.a[-1], y)

        for i in range(len(self.W) - 1, 0, -1):
            self.grad[i - 1] = self.W[i].T @ self.grad[i] * \
                self.acf[i].deriv(self.z[i])

    def train(self, X, y):

        self.backward(X[0], y[0])

        for i in range(len(self.grad)):
            self.delta[i] = np.outer(self.grad[i], self.a[i])

        for i in range(1, len(y)):
            self.backward(X[i], y[i])
            for j in range(len(self.grad)):
                self.delta[j] += np.outer(self.grad[j], self.a[j])

        self.W -= 0.01 * self.delta
        self.b -= 0.01 * self.grad


tanh = Tanh()
sig = Sigmoid()
softMax = SoftMax()
crossEntropy = CrossEntropy()

np.random.seed(42)

X = np.random.uniform(0, 1, (100, 40 * 40))

y = np.round((X[:, 0] + X[:, 1]) / 2)

nn = NeuralNetwork((40 * 40, 20 * 20, 10 * 20, 1),
                   [tanh, tanh, sig], crossEntropy)

for i in range(10):
    nn.train(X[:10], y[:10])

for i in range(10):
    nn.forward(X[i])
    print(nn.a[-1])
    print(y[i])

"""
y_train = np.round(nn.forward(X[:50])[-1][-1])
y_test = np.round(nn.forward(X[50:])[-1][-1])
print(np.mean(y_train == y[:50]))
print(np.mean(y_test == y[50:]))
"""
