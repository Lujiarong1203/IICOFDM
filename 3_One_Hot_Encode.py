import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Read the data
data=pd.read_csv('data/data_2.csv')
print(data.shape)

print(data.isnull().sum(), '\n', data.dtypes)

type_dummy = pd.get_dummies(data['type'])
data_3= pd.concat([data, type_dummy], axis=1)

# Generate the latest dataset and view the data types for each feature
data_3.drop('type', axis=1, inplace=True)
print(data_3.dtypes, '\n', data_3.shape)

data_3.to_csv(path_or_buf=r'data/data_3.csv', index=None)