from neuralnetwork import *
import numpy as np
import pandas as pd
import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
url_main = "https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/"
data_file_name = "Ising2DFM_reSample_L40_T=All.pkl"
label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"

labels = pickle.load(urlopen(url_main + label_file_name))

data = pickle.load(urlopen(url_main + data_file_name))
data = np.unpackbits(data).reshape(-1, 1600)
data = data.astype('int')

np.save("labels", labels)
np.save("spin_data", data)
"""

y = np.load("labels.npy")
X = np.load("spin_data.npy")

tanh = Tanh()
sig = Sigmoid()
relu = Relu()
crossEntropy = CrossEntropy()

np.random.seed(42)

nn = NeuralNetwork([1600, 400, 100, 25, 1], [sig, sig, sig, sig], crossEntropy)

idx = np.arange(len(y))
np.random.shuffle(idx)

idx_train = idx[:10000]
idx_test = idx[10000:11000]

y_train = y[idx_train]
X_train = X[idx_train]
y_test = y[idx_test]
X_test = X[idx_test]

nn.train(X_train, y_train, 0.0003, 100, 100)


y_pred = np.round(nn.predict(X_test)[:, 0]).astype(int)
print(y_pred[:10])
print(y_test[:10])

success = np.sum(y_pred == y_test)
print(success / len(y_test))
