"""
This script can be run to produce all results without
running each of the cells in the notebook separately.
"""

from Helpers import plot_decision_regions
from Algorithms import LogisticRegressionGD

from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('Iris_Dataset.csv', index_col=0)

X = df.iloc[:, [2, 3]]
y = df.iloc[:, [4]].values.ravel()


# Stratify ensures that each class is assigned equal number of samples
# for both test and train splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)


# Standardizing the dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


# Training the Perceptron model
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# Acquiring predictions from the trained model
y_pred = ppn.predict(X_test_std)


# Accuracy of the trained Perceptron model
accuracy = accuracy_score(y_test, y_pred)
output = f'Accuracy of the model is: {accuracy:.3f}'

with open('./Results/1_Perceptron_Accuracy.txt', 'w') as file:
    file.write(output)


# Plot of decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined_std = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined_std,
                      classifier=ppn,
                      test_idx=range(105, 150)
                      )
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./Results/2_Perceptron_Decision_Regions_Graph.png')
plt.close()


# Logistic Regression using Gradient Descent
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(
    eta=0.3,
    n_iter=1000,
    random_state=1
)

lrgd.fit(
    X_train_01_subset,
    y_train_01_subset
)

plot_decision_regions(
    X=X_train_01_subset,
    y=y_train_01_subset,
    classifier=lrgd)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./Results/4_LogisticRegressionGD_Graph.png')
plt.close()

# Logistic Regression using sklearn
# Training a model using sklearn's LogisticRegression class (ovr)
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

# Plotting the regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(
    X_combined_std,
    y_combined,
    classifier=lr,
    test_idx=range(105, 150)
)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./Results/5_LogisticRegression_sklearn_ovr.png')
plt.close()

# Training a model using sklearn's LogisticRegression class (multinomial)
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_std, y_train)

# Plotting the regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(
    X_combined_std,
    y_combined,
    classifier=lr,
    test_idx=range(105, 150)
)
plt.xlabel('Petal Length (Standardized)')
plt.ylabel('Petal Width (Standardized)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./Results/5_LogisticRegression_sklearn_multinomial.png')
plt.close()

# The term C is inversely proportional to the Regularization parameter.
# Thus decreasing the value of inverse regularization parameter C,
# means that we are increasing the regularization strength,
# which we can visualize by plotting L2 regularization path
# for the two weight coefficients

weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C=np.power(10.0, c), multi_class="ovr")
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(np.power(10.0, c))

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='Petal Length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal Width')
plt.ylabel('Weight Coefficients')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.savefig('./Results/6_Regularization_Vs_C.png')
plt.close()
