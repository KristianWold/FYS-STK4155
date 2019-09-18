from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random as rd

def frankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

class LinearModel:

    def designMatrix(self, x, p, intercept = True):
        n = x.shape[0]
        P = int(((p+2)*(p+1))/2) - !intercept
        X = np.zeros((n, P))

        idx = 0
        for i in range(!interact, p+1):
            X[:,idx] = x[:,0]**i
            idx += 1

        for i in range(p + 1):
            for j in range(1, p - i + 1):
                X[:,idx] = (x[:,0]**j)*(x[:,1]**i)
                idx += 1

        return X, P

    def ols(self, x, y, poly_deg):
        self.intercept = True
        self.centering = False
        self.standardize = False

        self.poly_deg = poly_deg
        X, self.params = designMatrix(x, poly_deg)
        self.b = np.linalg.inv(X.T @ X) @ X.T @ y #matrix inversion

    def ridge(self, x, y, poly_deg, lamb):
        self.intercept = False
        self.center_data = True
        self.standardize_data = True

        self.x_ave = np.mean(x, axis = 1)
        self.y_ave = np.mean(y)
        self.x_std = np.std(x, axis = 1)

        x = (x - self.x_ave)/self.standardize

        X, self.params = designMatrix(x, poly_deg, self.intercept)

        self.params += 1
        b = np.zeros(self.params)
        b[0] = self.y_center
        b[1:] = np.linalg.inv(X.T @ X + lamb np.identity(self.params)) \
                @ X.T @ (y - self.y_ave)


    def predict(self, x):
        if self.center_data 
        X, = designMatrix(x, self.poly_deg, self.intercept)
        pred = X @ self.b
        return pred

    def mse(self, x,y):
        n = y.size
        _mse = 1/n * np.sum((y - self.predict(x))**2)
        return _mse

    def r2(self, y, y_pred):
        n = y.size
        y_ave = np.mean(y)
        _r2 = 1 - np.sum((y - self.predict(x))**2)/np.sum((y - y_ave)**2)
        return _r2


def split_data(n, p = 0.25):
    test_n = int(p*n)
    idx = list(range(n))
    rd.shuffle(idx)
    test_idx = [idx.pop() for i in range(test_n)]
    train_idx = idx
    return test_idx, train_idx


def kfold(n, k = 5):
    idx = np.array(list(range(n)))
    np.random.shuffle(idx)
    idx = np.array_split(idx, k)

    def folds(i):
        test_idx = idx[i]
        train_idx = np.concatenate((idx[:i], idx[i+1:]), axis = None)
        train_idx = train_idx.astype("int16")
        return train_idx, test_idx

    return folds


if __name__ == "__main__":


    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)

    print(x)


    z = frankeFunction(x, y)

    # Plot the surface.

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
