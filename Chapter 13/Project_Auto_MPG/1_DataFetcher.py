import os
import pandas as pd

if 'data' not in os.listdir():
    os.mkdir('data')

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

column_names = ['MPG', 'Cylinders', 'Displacement',
                'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

df: pd.DataFrame = pd.read_csv(url,
                               names=column_names,
                               na_values="?",
                               comment='\t',
                               sep=' ',
                               skipinitialspace=True
                               )

save_path = './data/auto_mpg.csv'

df.to_csv(save_path)
