from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Processing a sklearn dataset and saving it as a csv file
iris = load_iris()

df_features = pd.DataFrame(iris.data[:])
df_classes = pd.DataFrame(iris.target)

df = pd.concat([df_features, df_classes], axis=1, ignore_index=True)

df.columns = ['Sepal Length (cm)', 'Sepal Width (cm)',
              'Petal Length (cm)', 'Petal Width (cm)', 'Class']

df.to_csv('Iris_Dataset.csv')
