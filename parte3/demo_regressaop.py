import scipy.io as scipy
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.loadmat('data_preg.mat')
array_data = np.array(mat['data'])

x = array_data[:, 0]
y = array_data[:, 1]

r = np.polyfit(x, y, 1)
r = r[::-1]

plt.scatter(x, y)
plt.plot(x, r)
plt.ylabel("y")
plt.xlabel("x")
plt.show()
