import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#1)
n = 1000

x = np.random.rand(n,1)

y = 5*x*x + 0.1*np.random.randn(n,1)

X = x**2

b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(b)
x_lin = np.linspace(0, 1, 100)

plt.plot(x, y, "o")
plt.plot(x_lin, b[0]*x_lin**2)
plt.show()


#2)

X_lin = x_lin[:, np.newaxis]**2
linreg = LinearRegression(fit_intercept=False)
coeff = linreg.fit(X,y)

print(linreg.intercept_)
print(linreg.coef_)

ypredict = linreg.predict(X_lin)

plt.plot(x, y, "o")
plt.plot(x_lin, ypredict)
plt.show()
