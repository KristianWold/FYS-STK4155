from franke import frankeFunction
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


def designMatrix(x, y, p):
    n, = x.shape
    P = int(((p+2)*(p+1))/2)
    X = np.zeros((n, P))
    idx = 0
    for i in range(p+1):
        for j in range(i+1):
            X[:,idx] = x**i*y**j
            idx += 1
    return X

#generate data
np.random.seed(1)
N = 1000000           #Number of data points
sigma2 = 0.1          #Irreducable noise
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = frankeFunction(x, y) + np.random.normal(0, sigma2, N)

X = designMatrix(x, y, 5)

b = np.linalg.inv(X.T@X)@X.T@z

M = 40
x_lin = np.linspace(0, 1, M)
y_lin = np.linspace(0, 1, M)

x_grid, y_grid = np.meshgrid(x_lin, y_lin)
x_lin, y_lin = np.ravel(x_grid), np.ravel(y_grid)


X_lin = designMatrix(x_lin, y_lin, 5)
z_lin = X_lin@b

print(len(z_lin))

z_grid = np.reshape(z_lin,(M,M))


fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
