from Helpers import plot_decision_regions
from Algorithms import LogisticRegressionGD

from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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


# Logistic Regressing using Gradient Descent
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
