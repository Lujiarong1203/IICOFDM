import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import scikitplot as skplt
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_3.csv")
print(data.shape)

print(Counter(data['fraud']))

# Numerical features distribution
nem_col=['rating_count','rating_finance', 'rating_marketing', 'rating_overall']
print(nem_col)

dist_cols = 2
dist_rows = len(nem_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(2, 2, i)
    ax = sns.kdeplot(data=data[data.fraud==0][col], bw=0.5, label="Not fraud", color="Red", shade=True)
    ax = sns.kdeplot(data=data[data.fraud==1][col], bw=0.5, label="Fraud", color="Blue", shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    i += 1
plt.show()

num_data=data[['rating_count', 'rating_finance', 'rating_marketing', 'rating_overall']]
print(num_data.shape)
print(num_data.describe().T)

# Distribution of character features 1
str_col= ['KYC', 'Registration', 'medium', 'Premium']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))
i = 1
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="fraud", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left')

    i += 1
    plt.tight_layout();

plt.show()

# Distribution of character features 2
str_col= ['linkedin', 'bitcointalk', 'Whitepaper', 'github']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="fraud", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left')

    i += 1
    plt.tight_layout();
plt.show()

# Distribution of character features 3
str_col= ['About', 'reddit', 'facebook', 'ICO']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="fraud", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if i == 4:
        plt.legend(loc='upper right')
    else:
        plt.legend(loc='upper left')
    i += 1
    plt.tight_layout();
plt.show()

# Distribution of character features 4
str_col= ['Agent', 'Airdrop', 'Bounty', 'ICO']
print(str_col)

dist_cols = 2
dist_rows = len(str_col)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="fraud", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right')
    i += 1
    plt.tight_layout();
plt.show()

# Analysis of correlation
# Analyze the correlation between each feature and fraud
corrs=data.drop('fraud',axis=1).corrwith(data['fraud']).sort_values(ascending=False)
print(corrs)

fig,axes=plt.subplots(1,1,figsize=(8,6))
axes.axhline(corrs[corrs>0].mean(), ls=':',color='black',linewidth=2)
axes.text(10,corrs[corrs>0].mean()+.015, "Average = {:.3f}".format(corrs[corrs>0].mean()),color='black',size=15)
axes.axhline(corrs[corrs<0].mean(), ls=':',color='black',linewidth=2)
axes.text(10,corrs[corrs<0].mean()+.015, "Average = {:.3f}".format(corrs[corrs<0].mean()),color='black',size=15)
sns.barplot(y=corrs,x=corrs.index,palette='Spectral')
# plt.title('Correlation of fraud to other Features',size=20,color='black',y=1.03)
plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
for p in axes.patches:
            value = p.get_height()
            if value <=.5:
                continue
            x = p.get_x() + p.get_width()-.9
            y = p.get_y() + p.get_height()+(.02*value)
            axes.text(x, y, str(value)[1:5], ha="left",fontsize=12,color='#000000')
plt.tight_layout();
plt.show()

# 相关性热力图
plt.rcParams['axes.unicode_minus']=False
data_corr=data[['github','facebook','twitter','telegram','youtube',
                 'bitcointalk', 'linkedin','medium','reddit',
                 'About','rating_count','rating_overall','KYC',
                 'Registration','Premium', 'fraud']]
print(data_corr.shape)
corr=data_corr.corr()
print(corr)
mask=np.triu(np.ones_like(corr, dtype=np.bool))
fig=plt.figure(figsize=(10, 12))
ax=sns.heatmap(corr, mask=mask, fmt=".2f", cmap='gist_heat', cbar_kws={"shrink": .8},
            annot=True, linewidths=1, annot_kws={"fontsize":15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize=15, rotation=30)
plt.yticks(fontsize=15, rotation=30)
plt.show()

# Remove 'Cryptocurrency' with extremely low correlation coefficient
data_1=data.drop('Cryptocurrency', axis=1)
print(data_1.shape)
y=data_1['fraud']
x=data_1.drop('fraud', axis=1)
print('X and y dimensions after Cryptocurrency is removed:', x.shape, y.shape)
#
# standardized
mm=MinMaxScaler()
X=pd.DataFrame(mm.fit_transform(x))
X.columns=x.columns
data_2=pd.concat([X, y], axis=1)
print(data_2.head(5))
#
# # Random forest feature selection
RF=RandomForestClassifier(random_state=42)
RF.fit(X, y)
FI=RF.feature_importances_
print(FI)
FI_RF = pd.DataFrame({"Feature Importance":FI}, index=X.columns)
print(FI_RF.sort_values("Feature Importance",ascending=True).round(5), FI_RF.shape)

print('num of features:', len(FI_RF["Feature Importance"]>0.0001))
plt.figure(figsize=(12, 8))
FI_RF[FI_RF["Feature Importance"] > - 0.1].sort_values("Feature Importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0,fontsize=15)
plt.yticks(rotation=0,fontsize=15)
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature',fontsize=15)
plt.legend(fontsize=15, loc=4)
plt.show()

#
FI_RF_C=FI_RF[FI_RF["Feature Importance"] >0.0001 ].sort_values("Feature Importance", ascending=True)
print(FI_RF_C, FI_RF_C.shape)

choose_cols = FI_RF_C.index.tolist()
print(choose_cols)

choose_cols.append('fraud')
print(choose_cols)
choose_data = data_2[choose_cols].copy()
print(choose_data.shape, choose_data.head(5))

choose_data.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_clean.csv', index=None)
