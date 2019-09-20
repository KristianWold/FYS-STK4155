from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random as rd
from scipy import stats

def frankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


class LinearModel:
    def design_matrix(self, x, poly_deg, intercept = True):
        n = x.shape[0]
        params = int(((poly_deg+2)*(poly_deg+1))/2) - (not intercept)
        X = np.zeros((n, params))

        idx = 0
        for i in range((not intercept), poly_deg+1):
            X[:,idx] = x[:,0]**i
            idx += 1

        for i in range(1,poly_deg + 1):
            for j in range(poly_deg - i + 1):
                X[:,idx] = (x[:,0]**j)*(x[:,1]**i)
                idx += 1

        return X, params


    def ols(self, x, y, poly_deg):
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.N = x.shape[0]

        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg)
        self.inv_cov_matrix = np.linalg.inv(X.T @ X)
        self.b = self.inv_cov_matrix @ X.T @ y

        self.eff_params = self.params
        self.b_var = np.diag(self.inv_cov_matrix)*\
                     self.N/(self.N-self.eff_params)*self.mse(x,y)


    def ridge(self, x, y, poly_deg, lamb):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        self.y_mean = np.mean(y)
        self.N = x.shape[0]

        self.poly_deg = poly_deg
        x = (x-self.x_mean)/self.x_std

        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg, intercept = False)
        self.inv_cov_matrix = np.linalg.inv(X.T @ X +\
                              lamb * np.identity(self.params))

        self.params += 1
        self.eff_params = np.trace(X @ self.inv_cov_matrix @ X.T) + 1
        self.b = np.zeros(self.params)
        self.b[0] = self.y_mean
        self.b[1:] = self.inv_cov_matrix @ X.T @ (y - self.y_mean)

        self.b_var = np.zeros(self.params)
        self.b_var[0] = 1/self.N
        self.b_var[1:] = np.diag(self.inv_cov_matrix @ X.T @ X @ self.inv_cov_matrix)
        self.b_var *= self.N/(self.N - self.eff_params) * self.mse(x,y)


    def predict(self, x):
        x = (x-self.x_mean)/self.x_std
        X, P = self.design_matrix(x, self.poly_deg)
        pred = X @ self.b
        return pred


    def confidence_interval(p):
        t = stats.t(df = N-self.eff_params).ppf(p)
        self.cinterval = [[self.b[i] - self.b_var[i]*t, b[i] + b_var[i]*t] for \
                         i in range(P)]


    def mse(self, x, y):
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
