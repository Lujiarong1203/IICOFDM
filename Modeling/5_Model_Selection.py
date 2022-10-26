import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix,f1_score, recall_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, RandomizedSearchCV
from collections import Counter
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_clean.csv")
print(data.shape, '\n', data.head(5))

y=data['fraud']
x=data.drop('fraud', axis=1)
print(x.shape, y.shape)
print(Counter(y))

# kernelPCA 2D projection
plot_2Dprojection_and_cardinality(x, y)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.legend(loc='upper right')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# View the label distribution in the training and test sets
print('train_set:', Counter(y_train), '\n', 'test_set:', Counter(y_test))

# Save the test set data and training set data for later parameter adjustment and evaluation
data_train=pd.concat([x_train, y_train], axis=1)
data_test=pd.concat([x_test, y_test], axis=1)
print(data_train.shape, data_test.shape)
data_train.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_train.csv', index=None)
data_test.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/ICO_fraud_detection/data/data_test.csv', index=None)

# stratified K-fold
stra_kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
cnt=1
for train_index, test_index in stra_kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# Model selection
# LR
score_data1=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(linear_model.LogisticRegression(penalty='l2', C=1, max_iter=100, class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data1 = score_data1.append(pd.DataFrame({'LR': [score.mean()]}), ignore_index=True)
# print(score_data1)

# KNN
score_data2=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data2 = score_data2.append(pd.DataFrame({'KNN': [score.mean()]}), ignore_index=True)
# print(score_data2)

# MLP
score_data3=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(MLPClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data3 = score_data3.append(pd.DataFrame({'MLP': [score.mean()]}), ignore_index=True)
# print(score_data3)

# DT
score_data4=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(DecisionTreeClassifier(max_depth=4, class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data4 = score_data4.append(pd.DataFrame({'DT': [score.mean()]}), ignore_index=True)
# print(score_data4)

# # SVC
score_data5=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(SVC(class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data5 = score_data5.append(pd.DataFrame({'SVC': [score.mean()]}), ignore_index=True)
# print(score_data5)
#
# ## Random Forest
score_data6=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(RandomForestClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data6 = score_data6.append(pd.DataFrame({'RF': [score.mean()]}), ignore_index=True)
# print(score_data6)
#
## adaboost
score_data7=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(AdaBoostClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data7 = score_data7.append(pd.DataFrame({'Ada': [score.mean()]}), ignore_index=True)
# print(score_data7)

## LightGBM
score_data8=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(LGBMClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data8 =score_data8.append(pd.DataFrame({'LGBM': [score.mean()]}), ignore_index=True)
# print(score_data8)

# GB_C
score_data9=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(GradientBoostingClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data9 =score_data9.append(pd.DataFrame({'GBC': [score.mean()]}), ignore_index=True)
# print(score_data9)

## xgboost
score_data10=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(XGBClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data10 = score_data10.append(pd.DataFrame({'XG': [score.mean()]}), ignore_index=True)
# print(score_data10)

## PassiveAggressiveClassifier
score_data11=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(PassiveAggressiveClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data11 = score_data11.append(pd.DataFrame({'PAC': [score.mean()]}), ignore_index=True)
# print(score_data11)

## SGDClassifier
score_data12=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(SGDClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data12 = score_data12.append(pd.DataFrame({'SGDC': [score.mean()]}), ignore_index=True)
# print(score_data12)

## ExtraTreesClassifier
score_data13=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(ExtraTreesClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
    score_data13 = score_data13.append(pd.DataFrame({'ETC': [score.mean()]}), ignore_index=True)
# print(score_data14)

# The evaluation indexes of all models were summarized
score_data=pd.concat([score_data1, score_data2, score_data3, score_data4,
                     score_data5, score_data6, score_data7, score_data8, score_data9,
                      score_data10, score_data11, score_data12, score_data13], axis=1)
score_Data=score_data.rename(index={0:'accuracy', 1:'precision', 2:'recall', 3:'f1', 4:'roc_auc'}).T
print(score_Data)

score_Data.to_csv(path_or_buf=r'C:\Users\Gealen\PycharmProjects\pythonProject\ICO_fraud_detection\data\data_score.csv', index=True)