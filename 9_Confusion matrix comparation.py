import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
mm=MinMaxScaler()
X_train=pd.DataFrame(mm.fit_transform(x_train, y_train), columns=x_train.columns)

x_test=data_test.drop('fraud', axis=1)
y_test=data_test['fraud']
X_test=pd.DataFrame(mm.fit_transform(x_test, y_test), columns=x_test.columns)
print('Test set label ratio：', Counter(y_test))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

random_seed=1234
# Comparison of each model
LR=LogisticRegression(random_state=random_seed)
LR.fit(X_train, y_train)
y_pred_LR=LR.predict(X_test)
y_proba_LR=LR.predict_proba(X_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)

# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
lgbm.fit(X_train, y_train)
y_pred_lgbm=lgbm.predict(X_test)
y_proba_lgbm=lgbm.predict_proba(X_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
#
# XGboost
xg=xgboost.XGBClassifier(random_state=random_seed)
xg.fit(X_train, y_train)
y_pred_xg=xg.predict(X_test)
y_proba_xg=xg.predict_proba(X_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)

# KNN
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn=knn.predict(X_test)
y_proba_knn=knn.predict_proba(X_test)
cm_knn=confusion_matrix(y_test, y_pred_knn)

# SVM
SVM=SVC(random_state=random_seed, probability=True)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
y_proba_SVM=SVM.predict_proba(X_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# PA
# PA=PassiveAggressiveClassifier(random_state=random_seed)
# PA.fit(x_train, y_train)
# y_pred_PA=PA.predict(x_test)
# y_proba_PA=PA.predict_proba(x_test)
# cm_PA=confusion_matrix(y_test, y_pred_PA)

# ET
ETC=ExtraTreesClassifier(random_state=random_seed)
ETC.fit(X_train, y_train)
y_pred_ETC=ETC.predict(X_test)
y_proba_ETC=ETC.predict_proba(X_test)
cm_ETC=confusion_matrix(y_test, y_pred_ETC)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

#SGD
SGD=SGDClassifier(random_state=random_seed, loss="log")
SGD.fit(X_train, y_train)
y_pred_SGD=SGD.predict(X_test)
y_proba_SGD=SGD.predict_proba(X_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(X_train, y_train)
y_pred_MLP=MLP.predict(X_test)
y_proba_MLP=MLP.predict_proba(X_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# IICOFDM
RF=RandomForestClassifier(n_estimators=120, max_depth=13, max_features=4, min_samples_leaf=1, min_samples_split=2, random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_proba_RF = RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# confusion matrix heat map

LR
skplt.metrics.plot_confusion_matrix(y_test, y_pred_LR, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.savefig('Fig.7(a).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# KNN
skplt.metrics.plot_confusion_matrix(y_test, y_pred_knn, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(b) KNN', y=-0.2, fontsize=15)
plt.savefig('Fig.7(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# DT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(c) DT', y=-0.2, fontsize=15)
plt.savefig('Fig.7(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# XGboost
skplt.metrics.plot_confusion_matrix(y_test, y_pred_xg, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(d) XGboost', y=-0.2, fontsize=15)
plt.savefig('Fig.7(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# Ada
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(e) Adaboost', y=-0.2, fontsize=15)
plt.savefig('Fig.7(e).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# IICOFDM
skplt.metrics.plot_confusion_matrix(y_test, y_pred_RF, title=None, cmap='Set3_r', text_fontsize=15)
plt.title('(f) IICOFDM', y=-0.2, fontsize=15)
plt.savefig('Fig.7(f).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()


# # KS curve

# IICOFDM
skplt.metrics.plot_ks_statistic(y_test, y_proba_RF, title=None, text_fontsize=15, figsize=(6, 6))
# plt.title('(a) IICOFDM', y=-0.2, fontsize=15)
plt.legend(fontsize=15, loc=0)
plt.savefig('Fig.10(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# # cumulative_gain curve
# IICOFDM
skplt.metrics.plot_cumulative_gain(y_test, y_proba_RF, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='best', fontsize=15)
plt.savefig('Fig.10(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# Lift curve
skplt.metrics.plot_lift_curve(y_test, y_proba_RF, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='best', fontsize=15)
plt.savefig('Fig.10(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# 多个模型的ROC曲线对比
# import matplotlib.pylab as plt
plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif']=['SimHei']
fpr1, tpr1, thres1 = roc_curve(y_test, y_proba_knn[:, 1])
fpr2, tpr2, thres2 = roc_curve(y_test, y_proba_DT[:, 1])
fpr3, tpr3, thres3 = roc_curve(y_test, y_proba_Ada[:,1])
fpr4, tpr4, thres4 = roc_curve(y_test, y_proba_xg[:, 1])
fpr5, tpr5, thres5 = roc_curve(y_test, y_proba_RF[:, 1])

plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(fpr1, tpr1, 'b', label='KNN ', color='black',lw=1.5,ls=':')
plt.plot(fpr2, tpr2, 'b', label='DT ', color='green',lw=1.5,ls='-.')
plt.plot(fpr3, tpr3, 'b', label='Adaboost ', color='RoyalBlue',lw=1.5,ls=':')
plt.plot(fpr4, tpr4, 'b', label='XGboost ', color='violet',lw=1.5, ls='--')
plt.plot(fpr5, tpr5, 'b', ms=1,label='IICOFDM ', lw=3,color='red',marker='*')

plt.plot([0, 1], [0, 1], 'darkgrey')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=15)
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=15)
plt.savefig('Fig.10(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0)
plt.show()