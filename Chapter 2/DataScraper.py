import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, header=None, encoding='utf-8')

df.columns = [
    'Sepal_Length(cm)', 'Sepal_Width(cm)', 'Petal_Length(cm)', 'Petal_Width(cm)', 'Class']

df.to_csv('Iris.csv')
