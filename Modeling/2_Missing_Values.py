import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data= pd.read_csv("data/data_1.csv")
print(data.shape)

print(Counter(data['fraud']))

print(data.isnull().sum())

na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio)

fig,axes=plt.subplots(1,1,figsize=(12,6))
sns.set(font_scale=1.4, font='Times New Roman')
# axes.grid(color='#909090',linestyle=':',linewidth=2)
plt.xticks(rotation=90)
sns.barplot(x=na_ratio.index,y=na_ratio,palette='coolwarm_r')
plt.title('Missing Value Ratio',color=('#000000'),y=1.03)
plt.tight_layout();
plt.show()

sns.set(style="ticks")
msno.matrix(data)
plt.show()

# handle with the missing values
# Fill rating_count with the mode
data['rating_count'].fillna(float(data['rating_count'].mode()), inplace=True)

# Fill in the following features with the average
data['rating_finance']=data['rating_finance'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_finance']=data['rating_finance'].fillna(data['rating_finance'].astype(float).mean())

data['rating_marketing']=data['rating_marketing'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_marketing']=data['rating_marketing'].fillna(data['rating_marketing'].astype(float).mean())

data['rating_overall']=data['rating_overall'].apply(lambda x: np.NaN if x=='NoEntry' else x)
data['rating_overall']=data['rating_overall'].fillna(data['rating_overall'].astype(float).mean())

# Fill in the missing value of type with the previous line
data['type'].fillna(method='pad', axis=0, inplace=True)

print('After filling in the missing values：', '\n', data.isnull().sum())

# Converts partial feature types
columns=['rating_finance', 'rating_marketing', 'rating_overall']
for k in columns:
    data[k]=data[k].apply(lambda x: float(x))
print('The converted data type：', '\n', data.dtypes)

data.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_2.csv', index=None)