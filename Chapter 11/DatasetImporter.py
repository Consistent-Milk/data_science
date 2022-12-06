# Script to download and save MNIST dataset in CSV form
import os
import pandas as pd
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
df = pd.concat([X, y], axis=1)

# Search if any folder named 'Data' exists in the root of the script
# if not make a folder named 'Data'
if 'Data' not in os.listdir():
    os.mkdir('Data')

df.to_csv('./Data/MNIST.csv')
