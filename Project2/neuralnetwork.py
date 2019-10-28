import numpy as np


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

        for i in range(len(dim) - 1):
            m = dim[i + 1]
            n = dim[i]
            self.W[i] = np.random.normal(0, 1, (m, n))
            self.b[i] = 0.01 * np.ones(m)

    def forward(self, x):
        self.z[0] = x.T
        self.a[0] = x.T

        for i in range(len(self.W)):
            print(self.W[i].shape)
            print(self.a[i].shape)
            self.z[i + 1] = self.W[i]@self.a[i]
            self.a[i + 1] = self.acf[i](self.z[i + 1])

    def backward(self, x, y):
        self.forward(x)

        self.grad[-1] = self.acf[-1].deriv(self.z[-1]) * \
            self.cost.deriv(self.a[-1], y)

        for i in range(len(self.W) - 1, 0, -1):
            self.grad[i] = self.W[i].T @ self.grad[i + 1] * \
                self.acf[i].deriv(z[i])

    def train(self, X, y):

        delta = []
        grad, a = self.backward(X[0], y[0])

        for i in range(len(grad)):
            delta.append(np.outer(grad[i], a[i]))

        for i in range(1, len(y)):
            grad, a = self.backward(X[i], y[i])
            for j in range(len(grad)):
                delta[j] += np.outer(grad[j], a[j])

        for i in range(len(delta)):
            self.W[i] -= 0.01 * delta[i]


tanh = Tanh()
sig = Sigmoid()
softMax = SoftMax()
crossEntropy = CrossEntropy()

np.random.seed(42)

X = np.random.uniform(0, 1, (100, 2))

y = np.round((X[:, 0] + X[:, 1]) / 2)

nn = NeuralNetwork((2, 4, 1), [tanh, sig], crossEntropy)

nn.forward(X[0])
print(nn.z)

# grad, a = nn.backward(X[0], y[0])

# print(grad)
# a = np.array([np.ones((3, 3)), np.ones((2, 2))])
# print(a+a)
"""
y_train = np.round(nn.forward(X[:50])[-1][-1])
y_test = np.round(nn.forward(X[50:])[-1][-1])
print(np.mean(y_train == y[:50]))
print(np.mean(y_test == y[50:]))
"""
