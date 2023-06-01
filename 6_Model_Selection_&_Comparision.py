import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, accuracy_score,f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
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


data_train = pd.read_csv('data/data_train.csv')
data_test = pd.read_csv('data/data_test.csv')
print('train_set:', data_train.shape, data_train.head(5), 'test_set:', data_test.shape)

# na_ratio=data_train.isnull().sum()[data_train.isnull().sum()>=0].sort_values(ascending=False)/len(data_train)
# na_sum=data_train.isnull().sum().sort_values(ascending=False)
# print(na_ratio)
#
# # x_train=data_train.drop('fraud', axis=1)
# # y_train=data_train.fraud
# # x_test=data_test.drop('fraud', axis=1)
# # y_test=data_test.fraud
#
#
#
# # XG=XGBClassifier(random_state=1234)
# # XG.fit(x_train, y_train)
# # y_pred=XG.predict(x_test)
# # acc=accuracy_score(y_test, y_pred)
# # print(acc)


# # 删除特征，试试
# # data_train = data_train.drop(['github', 'twitter', 'instagram', 'reddit', 'telegram', 'youtube','linkedin', 'bitcointalk', 'facebook', 'medium'], axis=1)
# # data_test = data_test.drop(['github', 'twitter', 'instagram', 'reddit', 'telegram', 'youtube','linkedin', 'bitcointalk', 'facebook', 'medium'], axis=1)
# data_train = data_train.drop(['Registration', 'rating_count', 'About'], axis=1)
# data_test = data_test.drop(['Registration', 'rating_count', 'About'], axis=1)
# print('train_set:', data_train.shape, 'test_set:', data_test.shape)

# # stratified K-fold
# stra_kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
# cnt=1
# for train_index, test_index in stra_kf.split(x_train, y_train):
#     print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
#     cnt += 1
#
# # Model selection
# # LR
# score_data1=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(linear_model.LogisticRegression(penalty='l2', C=1, max_iter=100, class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data1 = score_data1.append(pd.DataFrame({'LR': [score.mean()]}), ignore_index=True)
# # print(score_data1)
#
# # KNN
# score_data2=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(KNeighborsClassifier(), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data2 = score_data2.append(pd.DataFrame({'KNN': [score.mean()]}), ignore_index=True)
# # print(score_data2)
#
# # MLP
# score_data3=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(MLPClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data3 = score_data3.append(pd.DataFrame({'MLP': [score.mean()]}), ignore_index=True)
# # print(score_data3)
#
# # DT
# score_data4=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(DecisionTreeClassifier(max_depth=4, class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data4 = score_data4.append(pd.DataFrame({'DT': [score.mean()]}), ignore_index=True)
# # print(score_data4)
#
# # # SVC
# score_data5=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(SVC(class_weight={0:0.1,1:0.9}, random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data5 = score_data5.append(pd.DataFrame({'SVC': [score.mean()]}), ignore_index=True)
# # print(score_data5)
# #
# # ## Random Forest
# score_data6=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(RandomForestClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data6 = score_data6.append(pd.DataFrame({'RF': [score.mean()]}), ignore_index=True)
# # print(score_data6)
# #
# ## adaboost
# score_data7=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(AdaBoostClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data7 = score_data7.append(pd.DataFrame({'Ada': [score.mean()]}), ignore_index=True)
# # print(score_data7)
#
# ## LightGBM
# score_data8=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(LGBMClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data8 =score_data8.append(pd.DataFrame({'LGBM': [score.mean()]}), ignore_index=True)
# # print(score_data8)
#
# # GB_C
# score_data9=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(GradientBoostingClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data9 =score_data9.append(pd.DataFrame({'GBC': [score.mean()]}), ignore_index=True)
# # print(score_data9)
#
# ## xgboost
# score_data10=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(XGBClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data10 = score_data10.append(pd.DataFrame({'XG': [score.mean()]}), ignore_index=True)
# # print(score_data10)
#
# ## PassiveAggressiveClassifier
# score_data11=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(PassiveAggressiveClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data11 = score_data11.append(pd.DataFrame({'PAC': [score.mean()]}), ignore_index=True)
# # print(score_data11)
#
# ## SGDClassifier
# score_data12=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(SGDClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data12 = score_data12.append(pd.DataFrame({'SGDC': [score.mean()]}), ignore_index=True)
# # print(score_data12)
#
# ## ExtraTreesClassifier
# score_data13=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(ExtraTreesClassifier(random_state=1234), x_train, y_train, cv=stra_kf, scoring=sco)
#     score_data13 = score_data13.append(pd.DataFrame({'ETC': [score.mean()]}), ignore_index=True)
# # print(score_data14)
#
# # The evaluation indexes of all models were summarized
# score_data=pd.concat([score_data1, score_data2, score_data3, score_data4,
#                      score_data5, score_data6, score_data7, score_data8, score_data9,
#                       score_data10, score_data11, score_data12, score_data13], axis=1)
# score_Data=score_data.rename(index={0:'accuracy', 1:'precision', 2:'recall', 3:'f1', 4:'roc_auc'}).T
# print(score_Data)
#
# score_Data.to_csv(path_or_buf=r'data\data_score.csv', index=True)

def model_comparison(train_set, test_set, estimator):
    x_train = train_set.drop('fraud', axis=1)
    y_train = train_set['fraud']
    x_test = test_set.drop('fraud', axis=1)
    y_test = test_set['fraud']
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 归一化
    mm = MinMaxScaler()
    x_train_std = pd.DataFrame(mm.fit_transform(x_train))
    x_train_std.columns=x_train.columns
    x_test_std=pd.DataFrame(mm.fit_transform(x_test))
    x_test_std.columns=x_test.columns

    # # estimator
    est = estimator
    est.fit(x_train_std, y_train)
    y_pred = est.predict(x_test_std)
    score_data = []
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test, y_pred)
        score_data.append(score)
    print(score_data)

random_seed = 1234

# 与主流机器学习模型的对比
LR=linear_model.LogisticRegression(random_state=random_seed)
KNN=KNeighborsClassifier()
MLP=MLPClassifier(random_state=random_seed)
SVM=SVC(random_state=random_seed)
DT=DecisionTreeClassifier(random_state=random_seed)
PA=PassiveAggressiveClassifier(random_state=random_seed)
ET=ExtraTreesClassifier(random_state=random_seed)
SGD=SGDClassifier(random_state=random_seed)
RF=RandomForestClassifier(random_state=random_seed)
GBDT=GradientBoostingClassifier(random_state=random_seed)
Ada=AdaBoostClassifier(random_state=random_seed)
LGBM=LGBMClassifier(random_state=random_seed)
XG=XGBClassifier(random_state=random_seed)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
GNB=GaussianNB()

#
# 比较模型
model_comparison(data_train, data_test, LR)
print('LR')
model_comparison(data_train, data_test, KNN)
print('KNN')
model_comparison(data_train, data_test, MLP)
print('MLP')
model_comparison(data_train, data_test, SVM)
print('SVM')
model_comparison(data_train, data_test, DT)
print('DT')
model_comparison(data_train, data_test, PA)
print('PA')
model_comparison(data_train, data_test, GNB)
print('GNB')
model_comparison(data_train, data_test, ET)
print('ET')
model_comparison(data_train, data_test, SGD)
print('SGD')
model_comparison(data_train, data_test, XG)
print('XG')
model_comparison(data_train, data_test, GBDT)
print('GBDT')
model_comparison(data_train, data_test, Ada)
print('Ada')
model_comparison(data_train, data_test, LGBM)
print('LGBM')
model_comparison(data_train, data_test, RF)
print('RF')