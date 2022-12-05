import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)


plt.plot(z, sigma_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
# Using LaTex symbols for labeling
plt.ylabel(r'$\sigma (z)$')
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.savefig('./Results/3_Logistic_Sigmoid_Function.png')
plt.close()
