# Custom Learning Algorithm
from Perceptron import Perceptron
from Adaline import Adaline
from AdalineSGD import AdalineSGD

# Standard Libraries
from operator import eq

# Computation Libraries
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set()

# Importing Dataset
df = pd.read_csv('Iris.csv', index_col=0)

# Seperating features and classes
y = df.iloc[0:100, 4].values
y = np.where(eq(y, 'Iris-setosa'), 0, 1)

X = df.iloc[0:100, [0, 2]].values

# First Visualization
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
            marker='s', label='Versicolor')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc='upper left')
plt.savefig("./Results/image_1.png")
plt.close()


# Perceptron training graph
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X=X, y=y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.savefig("./Results/image_2.png")
plt.close()


# Decision Region Graph Function
def plot_decision_regions(X: np.ndarray, y: np.ndarray, classifier: Perceptron | Adaline | AdalineSGD, resolution: float = 0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    lab = classifier.predict(np.transpose(
        np.array([xx1.ravel(), xx2.ravel()])))
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'class {cl}',
                    edgecolor='black')


# Decision region graph
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc='upper left')
plt.savefig("./Results/image_3.png")
plt.close()


# Adaline training graph
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# First Adaline model training
ada1 = Adaline(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean Squared Error)')
ax[0].set_title('Adaline - Learning Rate 0.1')

# Second Adaline model training
ada2 = Adaline(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Mean Squared Error')
ax[1].set_title('Adaline - Learning Rate 0.0001')
plt.savefig('./Results/image_4.png')
plt.close()


# Feature scaled Adaline Model
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_std = Adaline(n_iter=20, eta=0.5)
ada_std.fit(X_std, y)

# Decision Boundary Graph
plot_decision_regions(X_std, y, classifier=ada_std)
plt.title('Adaline - Gradient Descent (Standardized)')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Petal Length (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("./Results/image_5.png")
plt.close()

# Mean Squared Error Graph
plt.plot(range(1, len(ada_std.losses_) + 1), ada_std.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.tight_layout()
plt.savefig("./Results/image_6.png")
plt.close()


# Adaline using Stochastic Gradient Descent (Standardized)
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

# Decision Boundary Graph
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent (Standardized)')
plt.xlabel('Sepal Length (Standardized)')
plt.ylabel('Petal Length (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("./Results/image_7.png")
plt.close()

# Loss Graph
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.tight_layout()
plt.savefig("./Results/image_8.png")
plt.close()
