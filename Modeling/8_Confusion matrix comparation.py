import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
import xgboost
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve

import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Read data
data_train = pd.read_csv('data/data_train.csv')
data_test = pd.read_csv('data/data_test.csv')

x_train=data_train.drop('fraud', axis=1)
y_train=data_train['fraud']

x_test=data_test.drop('fraud', axis=1)
y_test=data_test['fraud']
print('Test set label ratio：', Counter(y_test))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Comparison of each model
LR=LogisticRegression(penalty='l2', C=1, max_iter=100,
                      class_weight={0:0.1,1:0.9}, random_state=1234)
LR.fit(x_train, y_train)
y_pred_LR=LR.predict(x_test)
y_proba_LR=LR.predict_proba(x_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)

# LightGBM
lgbm=LGBMClassifier(n_estimators=300, max_depth=6, random_state=1234)
lgbm.fit(x_train, y_train)
y_pred_lgbm=lgbm.predict(x_test)
y_proba_lgbm=lgbm.predict_proba(x_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
#
# XGboost
xg=xgboost.XGBClassifier(n_estimators=300, max_depth=6, random_state=1234)
xg.fit(x_train, y_train)
y_pred_xg=xg.predict(x_test)
y_proba_xg=xg.predict_proba(x_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)

# KNN
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_knn=knn.predict(x_test)
y_proba_knn=knn.predict_proba(x_test)
cm_knn=confusion_matrix(y_test, y_pred_knn)

# SVM
SVM=SVC(class_weight={0:0.1,1:0.9}, random_state=1234, probability=True)
SVM.fit(x_train, y_train)
y_pred_SVM=SVM.predict(x_test)
y_proba_SVM=SVM.predict_proba(x_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# # PA
# PA=PassiveAggressiveClassifier(random_state=1234)
# PA.fit(x_train, y_train)
# y_pred_PA=PA.predict(x_test)
# y_proba_PA=PA.predict_proba(x_test)
# cm_PA=confusion_matrix(y_test, y_pred_PA)

# ET
ETC=ExtraTreesClassifier(random_state=1234)
ETC.fit(x_train, y_train)
y_pred_ETC=ETC.predict(x_test)
y_proba_ETC=ETC.predict_proba(x_test)
cm_ETC=confusion_matrix(y_test, y_pred_ETC)

# Ada
Ada=AdaBoostClassifier(random_state=1234)
Ada.fit(x_train, y_train)
y_pred_Ada=Ada.predict(x_test)
y_proba_Ada=Ada.predict_proba(x_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

#SGD
SGD=SGDClassifier(random_state=1234, loss="log")
SGD.fit(x_train, y_train)
y_pred_SGD=SGD.predict(x_test)
y_proba_SGD=SGD.predict_proba(x_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)

# GBDT
GBDT=GradientBoostingClassifier(random_state=1234)
GBDT.fit(x_train, y_train)
y_pred_GBDT=GBDT.predict(x_test)
y_proba_GBDT=GBDT.predict_proba(x_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# DT
DT=DecisionTreeClassifier(max_depth=4, class_weight={0:0.1,1:0.9}, random_state=1234)
DT.fit(x_train, y_train)
y_pred_DT=DT.predict(x_test)
y_proba_DT=DT.predict_proba(x_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# MLP
MLP=MLPClassifier(random_state=1234)
MLP.fit(x_train, y_train)
y_pred_MLP=MLP.predict(x_test)
y_proba_MLP=MLP.predict_proba(x_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# IICOFDM
RF=RandomForestClassifier(n_estimators=25, max_depth=4, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=1234)
RF.fit(x_train, y_train)
y_pred_RF = RF.predict(x_test)
y_proba_RF = RF.predict_proba(x_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# confusion matrix heat map
# LR
skplt.metrics.plot_confusion_matrix(y_test, y_pred_LR, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.show()

# KNN
skplt.metrics.plot_confusion_matrix(y_test, y_pred_knn, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(b) KNN', y=-0.2, fontsize=15)
plt.show()

# DT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(c) DT', y=-0.2, fontsize=15)
plt.show()

# XGboost
skplt.metrics.plot_confusion_matrix(y_test, y_pred_xg, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(d) XGboost', y=-0.2, fontsize=15)
plt.show()

# Ada
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(e) Adaboost', y=-0.2, fontsize=15)
plt.show()

# IICOFDM
skplt.metrics.plot_confusion_matrix(y_test, y_pred_RF, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(f) IICOFDM', y=-0.2, fontsize=15)
plt.show()

# # ROC curve
# # KNN
# skplt.metrics.plot_roc(y_test, y_proba_knn, cmap='Set3_r', text_fontsize=15)
# plt.title('(a) KNN', y=-0.2, fontsize=15)
# plt.show()
#
# # Adaboost
# skplt.metrics.plot_roc(y_test, y_proba_Ada, cmap='Set3_r', text_fontsize=15)
# plt.title('(b) Adaboost', y=-0.2, fontsize=15)
# plt.show()
#
# # IICOFDM
# skplt.metrics.plot_roc(y_test, y_proba_RF, title='IICOFDM', cmap='Set3_r', text_fontsize=15)
# plt.title('(c) IICOFDM', y=-0.2, fontsize=15)
# plt.show()
#
# # KS curve
# # LR
# skplt.metrics.plot_ks_statistic(y_test, y_proba_LR, text_fontsize=15)
# plt.title('(a) LR', y=-0.2, fontsize=15)
# plt.legend(fontsize=14, loc=1)
# plt.show()
#
# # XGboost
# skplt.metrics.plot_ks_statistic(y_test, y_proba_xg, text_fontsize=15)
# plt.title('(a) XGboost', y=-0.2, fontsize=15)
# plt.legend(fontsize=15, loc=0)
# plt.show()
#
# # IICOFDM
# skplt.metrics.plot_ks_statistic(y_test, y_proba_RF, text_fontsize=15)
# plt.title('(a) IICOFDM', y=-0.2, fontsize=15)
# plt.legend(fontsize=15, loc=0)
# plt.show()
#
# # cumulative_gain curve
# # LR
# skplt.metrics.plot_cumulative_gain(y_test, y_proba_LR, title='LR', text_fontsize=15)
# plt.show()
#
# # LightGBM
# skplt.metrics.plot_cumulative_gain(y_test, y_proba_lgbm, title='LightGBM', text_fontsize=15)
# plt.show()
#
# # IICOFDM
# skplt.metrics.plot_cumulative_gain(y_test, y_proba_RF, title='IICOFDM', text_fontsize=15)
# plt.show()












